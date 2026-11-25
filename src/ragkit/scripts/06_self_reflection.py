"""Self-reflection RAG with self-grading on retrieval and generation.

```mermaid
flowchart TB
    Start([Start]) --> R[retrieve]
    R --> GD[grade_documents]
    GD -->|relevant / out of retries| G[generate]
    GD -->|not relevant| W[rewrite query]
    W --> R
    G --> GG[grade_generation]
    GG -->|grounded + answers| End([End])
    GG -->|hallucinated| G
    GG -->|doesn't answer| W
    GG -->|max retries| End
```

Flow:
  1. Retrieve docs for the current *search query*
  2. Grade docs - if not relevant, rewrite the search query and retry
  3. Generate an answer to the user's *original* question
  4. Grade for hallucinations -> if hallucinated, regenerate with a stricter prompt
  5. Grade for answer quality -> if it doesn't address the question, rewrite and retry
  6. End when grounded + answers question, or after MAX_RETRIES
     (rewrites and regenerations are capped independently, so the graph
     always terminates)

The rewriter only changes the search query. Generation and answer grading always
use the original question, so a rewrite can't make the graph answer (and grade
itself against) a different question than the user asked.

Graders run as their own nodes (not inside edge functions), so every grade is a
visible step in LangSmith traces and is stored in the state.
"""

import argparse
import sys
from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from ragkit.corpus import default_retriever
from ragkit.pipeline import RAGResult
from ragkit.utils import format_docs, get_llm

MAX_RETRIES = 2
RETRY_TEMPERATURE = 0.7  # temperature 0 would regenerate the exact same hallucination


# ============================================================
# Graders (structured output via function calling)
# ============================================================
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="'yes' if relevant, 'no' if not")


class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="'yes' if grounded, 'no' if hallucinated")


class GradeAnswer(BaseModel):
    binary_score: str = Field(description="'yes' if addresses question, 'no' otherwise")


def build_doc_grader():
    prompt = ChatPromptTemplate.from_template(
        """You are grading retrieved documents for relevance to a question.

Documents:
{documents}

Question: {question}

Respond with 'yes' if ANY document is relevant, 'no' if NONE are relevant.
Output ONLY 'yes' or 'no'."""
    )
    return prompt | get_llm().with_structured_output(GradeDocuments)


def build_hallucination_grader():
    prompt = ChatPromptTemplate.from_template(
        """You are grading an answer for hallucinations.

Documents (ground truth):
{documents}

Answer:
{generation}

Respond with 'yes' if the answer is fully grounded in the documents (no claims
outside the documents), 'no' if it contains unsupported claims or made-up facts.
Output ONLY 'yes' or 'no'."""
    )
    return prompt | get_llm().with_structured_output(GradeHallucinations)


def build_answer_grader():
    prompt = ChatPromptTemplate.from_template(
        """You are grading an answer for whether it addresses the question.

Question: {question}
Answer:
{generation}

Respond with 'yes' if the answer directly addresses the question, 'no' otherwise.
Output ONLY 'yes' or 'no'."""
    )
    return prompt | get_llm().with_structured_output(GradeAnswer)


def build_rewriter():
    prompt = ChatPromptTemplate.from_template(
        """You are rephrasing a search query for better document retrieval.

User question: {question}
Previous search query: {query}

Write a clearer, more specific search query that will retrieve more relevant
documents for the user question. Output ONLY the new query, no preamble."""
    )
    return prompt | get_llm() | StrOutputParser()


RAG_TEMPLATE = """Answer the question based ONLY on the following context.
If the context is insufficient, say "I don't know."
{extra}
Context:
{context}

Question: {question}
Answer:"""

STRICT_NOTE = (
    "\nA previous answer contained claims that are not in the context. Only state "
    "facts that appear in the context; leave out anything you cannot support.\n"
)


def build_rag_chain(temperature: float = 0):
    prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    return prompt | get_llm(temperature=temperature) | StrOutputParser()


# ============================================================
# Graph state
# ============================================================
class GraphState(TypedDict, total=False):
    original_question: str  # what the user asked; never rewritten
    question: str  # current search query (rewritten on retry)
    generation: str
    documents: list
    docs_relevant: bool
    grounded: bool
    answers: bool
    retries: int  # number of query rewrites so far
    generations: int  # number of answers generated so far


def is_yes(score) -> bool:
    """Graders return free-form strings; accept 'yes', 'Yes', ' yes.' etc."""
    return score.binary_score.strip().lower().startswith("yes")


def _asked(state) -> str:
    return state.get("original_question") or state["question"]


# ============================================================
# Nodes (do work, return a state update)
# ============================================================
def make_retrieve(retriever):
    def retrieve(state):
        print(f"  [retrieve] Getting docs for: {state['question'][:80]}", file=sys.stderr)
        return {"documents": retriever.invoke(state["question"])}

    return retrieve


def make_grade_documents(doc_grader):
    def grade_documents(state):
        score = doc_grader.invoke(
            {"question": _asked(state), "documents": format_docs(state["documents"])}
        )
        return {"docs_relevant": is_yes(score)}

    return grade_documents


def make_generate(rag_chain, retry_chain):
    def generate(state):
        # Regenerating after a failed groundedness check: be stricter and sample,
        # otherwise a temperature-0 model reproduces the same answer.
        retrying = state.get("grounded") is False
        suffix = " (strict retry)" if retrying else ""
        print(f"  [generate] Producing answer{suffix}...", file=sys.stderr)
        chain = retry_chain if retrying else rag_chain
        generation = chain.invoke(
            {
                "context": format_docs(state["documents"]),
                "question": _asked(state),
                "extra": STRICT_NOTE if retrying else "",
            }
        )
        return {"generation": generation, "generations": state.get("generations", 0) + 1}

    return generate


def make_grade_generation(hallucination_grader, answer_grader):
    def grade_generation(state):
        h_score = hallucination_grader.invoke(
            {"documents": format_docs(state["documents"]), "generation": state["generation"]}
        )
        if not is_yes(h_score):
            return {"grounded": False, "answers": False}
        a_score = answer_grader.invoke(
            {"question": _asked(state), "generation": state["generation"]}
        )
        return {"grounded": True, "answers": is_yes(a_score)}

    return grade_generation


def make_rewrite(rewriter):
    def rewrite(state):
        print("  [rewrite] Rephrasing search query...", file=sys.stderr)
        new_q = rewriter.invoke({"question": _asked(state), "query": state["question"]})
        print(f"    → {new_q[:100]}", file=sys.stderr)
        # Reset the generation verdict so the next answer starts from the normal prompt.
        return {"question": new_q, "retries": state.get("retries", 0) + 1, "grounded": None}

    return rewrite


# ============================================================
# Routing (read the grades, no LLM calls, no state changes)
# ============================================================
def route_after_grade_documents(state):
    if state["docs_relevant"]:
        print("  [route] docs relevant → generate", file=sys.stderr)
        return "generate"
    if state.get("retries", 0) >= MAX_RETRIES:
        print("  [route] docs not relevant, no retries left → generate", file=sys.stderr)
        return "generate"
    print("  [route] docs not relevant → rewrite", file=sys.stderr)
    return "rewrite"


def route_after_grade_generation(state):
    retries = state.get("retries", 0)
    regenerations = state.get("generations", 1) - 1
    if not state["grounded"]:
        print(f"  [route] hallucinated (regeneration {regenerations})", file=sys.stderr)
        return "generate" if regenerations < MAX_RETRIES else "end"
    if not state["answers"]:
        print(f"  [route] doesn't answer (retry {retries})", file=sys.stderr)
        return "rewrite" if retries < MAX_RETRIES else "end"
    print("  [route] grounded + answers → end", file=sys.stderr)
    return "end"


# ============================================================
# Build the graph
# ============================================================
def build_graph(retriever=None):
    retriever = retriever or default_retriever()

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", make_retrieve(retriever))
    workflow.add_node("grade_documents", make_grade_documents(build_doc_grader()))
    workflow.add_node(
        "generate", make_generate(build_rag_chain(), build_rag_chain(RETRY_TEMPERATURE))
    )
    workflow.add_node(
        "grade_generation",
        make_grade_generation(build_hallucination_grader(), build_answer_grader()),
    )
    workflow.add_node("rewrite", make_rewrite(build_rewriter()))

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grade_documents,
        {"generate": "generate", "rewrite": "rewrite"},
    )
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", "grade_generation")
    workflow.add_conditional_edges(
        "grade_generation",
        route_after_grade_generation,
        {"generate": "generate", "rewrite": "rewrite", "end": END},
    )
    return workflow.compile()


def initial_state(question: str) -> GraphState:
    return {"original_question": question, "question": question, "retries": 0, "generations": 0}


@lru_cache(maxsize=1)
def _app():
    return build_graph()


def run(question: str) -> RAGResult:
    result = _app().invoke(initial_state(question))
    return RAGResult(
        answer=result.get("generation", ""),
        contexts=[d.page_content for d in result.get("documents", [])],
        metadata={
            "rewrites": result.get("retries", 0),
            "generations": result.get("generations", 0),
            "final_query": result.get("question"),
            "grounded": result.get("grounded"),
            "answers": result.get("answers"),
        },
    )


# ============================================================
# CLI entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Self-reflection RAG with self-grading")
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    print(f"\n[run] Question: {args.question}\n", file=sys.stderr)
    result = run(args.question)
    print("=" * 60)
    print(result.answer or "(No generation produced)")
    print("=" * 60)
    print(f"[trace] {result.metadata}", file=sys.stderr)


if __name__ == "__main__":
    main()
