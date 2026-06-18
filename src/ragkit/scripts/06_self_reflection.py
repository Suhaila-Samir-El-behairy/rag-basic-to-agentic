"""Self-reflection RAG with self-grading on retrieval and generation.

```mermaid
flowchart TB
    Start([Start]) --> R[retrieve]
    R -->|docs relevant| G[generate]
    R -->|docs not relevant| W[rewrite]
    W --> R
    G -->|grounded + answers| End([End])
    G -->|hallucinated| G
    G -->|doesn't answer| W
    G -->|max retries| End

Flow:
  1. Retrieve docs
  2. Grade docs - if not relevant, rewrite question and retry
  3. Generate answer from docs
  4. Grade for hallucinations → if hallucinated, regenerate
  5. Grade for answer quality → if not addressing question, rewrite and retry
  6. End when grounded + answers question, or after MAX_RETRIES

Built with LangGraph for explicit stateful control flow.
"""

import argparse
import pickle
import sys
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from ragkit.utils import (
    format_docs,
    get_llm,
    get_vectorstore,
    load_web,
    split_docs,
)

URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"
MAX_RETRIES = 2
BM25_CACHE = Path("./bm25_cache_06.pkl")


# ============================================================
# Graders (structured output via Groq function calling)
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
        """You are rephrasing a question for better document retrieval.

Original question: {question}

Write a clearer, more specific version that will retrieve more relevant documents.
Output ONLY the rewritten question, no preamble."""
    )
    return prompt | get_llm() | StrOutputParser()


def build_rag_chain():
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based ONLY on the following context.
If the context is insufficient, say "I don't have enough information."

Context:
{context}

Question: {question}
Answer:"""
    )
    return prompt | get_llm() | StrOutputParser()


# ============================================================
# Graph state
# ============================================================
class GraphState(TypedDict):
    question: str
    generation: str
    documents: list
    retries: int


# ============================================================
# Node functions (do work, return state update)
# ============================================================
def make_retrieve(retriever):
    def retrieve(state):
        print(f"  [retrieve] Getting docs for: {state['question'][:80]}", file=sys.stderr)
        documents = retriever.invoke(state["question"])
        return {"documents": documents}

    return retrieve


def make_generate(rag_chain):
    def generate(state):
        print("  [generate] Producing answer...", file=sys.stderr)
        generation = rag_chain.invoke(
            {
                "context": format_docs(state["documents"]),
                "question": state["question"],
            }
        )
        return {"generation": generation}

    return generate


def make_rewrite(rewriter):
    def rewrite(state):
        print("  [rewrite] Rephrasing question...", file=sys.stderr)
        new_q = rewriter.invoke({"question": state["question"]})
        print(f"    → {new_q[:100]}", file=sys.stderr)
        return {
            "question": new_q,
            "retries": state.get("retries", 0) + 1,
        }

    return rewrite


# ============================================================
# Routing functions (decide next node, no state changes)
# ============================================================
def make_route_after_retrieve(doc_grader):
    def route_after_retrieve(state):
        score = doc_grader.invoke(
            {
                "question": state["question"],
                "documents": state["documents"],
            }
        )
        if score.binary_score == "yes":
            print("  [route] docs relevant → generate", file=sys.stderr)
            return "generate"
        print("  [route] docs not relevant → rewrite", file=sys.stderr)
        return "rewrite"

    return route_after_retrieve


def make_route_after_generate(hallucination_grader, answer_grader):
    def route_after_generate(state):
        retries = state.get("retries", 0)

        # Check hallucination first
        h_score = hallucination_grader.invoke(
            {
                "documents": state["documents"],
                "generation": state["generation"],
            }
        )
        if h_score.binary_score == "no":
            print(f"  [route] hallucinated (retry {retries})", file=sys.stderr)
            return "generate" if retries < MAX_RETRIES else "end"

        # Then check answer quality
        a_score = answer_grader.invoke(
            {
                "question": state["question"],
                "generation": state["generation"],
            }
        )
        if a_score.binary_score == "no":
            print(f"  [route] doesn't answer (retry {retries})", file=sys.stderr)
            return "rewrite" if retries < MAX_RETRIES else "end"

        print("  [route] grounded + answers → end", file=sys.stderr)
        return "end"

    return route_after_generate


# ============================================================
# Build the graph
# ============================================================
def build_graph():
    print(f"\n[setup] Loading + indexing {URL}...", file=sys.stderr)
    if BM25_CACHE.exists():
        splits = pickle.load(open(BM25_CACHE, "rb"))
        print(f"  Loaded {len(splits)} cached splits", file=sys.stderr)
    else:
        docs = load_web(URL)
        splits = split_docs(docs)
        pickle.dump(splits, open(BM25_CACHE, "wb"))
        print(f"  Indexed {len(splits)} chunks", file=sys.stderr)

    vectorstore = get_vectorstore(splits, name="06_self_reflection")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    print("[setup] Building graders + chains...", file=sys.stderr)
    doc_grader = build_doc_grader()
    hallucination_grader = build_hallucination_grader()
    answer_grader = build_answer_grader()
    rewriter = build_rewriter()
    rag_chain = build_rag_chain()

    # Nodes
    retrieve = make_retrieve(retriever)
    generate = make_generate(rag_chain)
    rewrite = make_rewrite(rewriter)

    # Routing functions
    route_after_retrieve = make_route_after_retrieve(doc_grader)
    route_after_generate = make_route_after_generate(hallucination_grader, answer_grader)

    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_node("rewrite", rewrite)

    workflow.set_entry_point("retrieve")

    workflow.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"generate": "generate", "rewrite": "rewrite"},
    )

    workflow.add_edge("rewrite", "retrieve")

    workflow.add_conditional_edges(
        "generate",
        route_after_generate,
        {"generate": "generate", "rewrite": "rewrite", "end": END},
    )

    return workflow.compile()


# ============================================================
# CLI entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Self-reflection RAG with self-grading (Groq + LangGraph)"
    )
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    print("\n[graph] Compiling self-reflection graph...", file=sys.stderr)
    app = build_graph()

    print(f"\n[run] Question: {args.question}\n", file=sys.stderr)
    print("=" * 60)
    result = app.invoke({"question": args.question, "retries": 0})

    if result.get("generation"):
        print(result["generation"])
    else:
        print("(No generation produced)")
    print("=" * 60)


if __name__ == "__main__":
    main()
