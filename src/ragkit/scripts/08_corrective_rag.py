"""Corrective RAG (CRAG) - paper-faithful implementation.

Based on: Yan et al. (2024) "Corrective Retrieval Augmented Generation"
Paper: https://arxiv.org/abs/2401.15884

The key idea: not all retrieved documents are equally useful. Before
generating, grade each retrieved doc and decide:
  - CORRECT:     high avg score -> use retrieved docs as-is
  - INCORRECT:   low avg score  -> fall back to web search
  - AMBIGUOUS:   in between     -> refine query + combine both sources

This is the "original" CRAG (Corrective), distinct from
cache-augmented CRAG (faster/cheaper) and from script 06's
self-reflection (multi-turn retry).
"""
import sys
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ragkit.utils import (
    get_llm,
    get_embeddings,
    load_web,
    split_docs,
    get_vectorstore,
    format_docs,
    web_search,
)

URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"
BM25_CACHE = Path("./bm25_cache_08.pkl")
UPPER_THRESHOLD = 0.7
LOWER_THRESHOLD = 0.3

# ---------- Document grader ----------
class DocumentGrade(BaseModel):
    score: float = Field(description="Relevance score 0.0 to 1.0")
    reasoning: str = Field(description="One-sentence explanation")

def build_doc_grader():
    prompt = ChatPromptTemplate.from_template(
        """Grade the relevance of this document chunk to the question.

Score 0.0 (irrelevant) to 1.0 (highly relevant).
- 0.0-0.3:  not relevant
- 0.4-0.6:  tangentially related
- 0.7-1.0:  directly addresses the question

Question: {question}
Document: {document}

Respond with JSON: {{"score": 0.X, "reasoning": "..."}}"""
    )
    return prompt | get_llm().with_structured_output(DocumentGrade)

# ---------- Query refiner ----------
def build_query_refiner():
    prompt = ChatPromptTemplate.from_template(
        """Refine the following question to be more specific and answerable.
The retrieved context was partially relevant; rewrite the question to
target the missing information better.

Original question: {question}
Partial context: {context}

Output ONLY the refined question, no preamble."""
    )
    return prompt | get_llm() | StrOutputParser()

# ---------- Answer generator ----------
def build_answer_chain():
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based ONLY on the context below.
If the context is insufficient, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""
    )
    return prompt | get_llm() | StrOutputParser()

# ---------- Setup ----------
def setup_vectorstore():
    if BM25_CACHE.exists():
        splits = pickle.load(open(BM25_CACHE, "rb"))
        print(f"  Loaded {len(splits)} cached splits", file=sys.stderr)
    else:
        docs = load_web(URL)
        splits = split_docs(docs)
        pickle.dump(splits, open(BM25_CACHE, "wb"))
        print(f"  Indexed {len(splits)} chunks", file=sys.stderr)
    vectorstore = get_vectorstore(splits, name="08_corrective")
    return vectorstore.as_retriever(search_kwargs={"k": 4})

# ---------- T-correct algorithm ----------
def t_correct_decision(scores, upper=UPPER_THRESHOLD, lower=LOWER_THRESHOLD):
    """Apply the paper's T-correct decision rule."""
    avg = sum(scores) / len(scores)
    if avg >= upper:
        return "CORRECT", avg
    if avg <= lower:
        return "INCORRECT", avg
    return "AMBIGUOUS", avg

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(
        description="Corrective RAG (CRAG) - paper-faithful (Groq-powered)"
    )
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--upper",
        type=float,
        default=UPPER_THRESHOLD,
        help="Upper threshold for CORRECT decision (default 0.7)",
    )
    parser.add_argument(
        "--lower",
        type=float,
        default=LOWER_THRESHOLD,
        help="Lower threshold for INCORRECT decision (default 0.3)",
    )
    args = parser.parse_args()

    # Setup
    print(f"\n[setup] Building/loading vector store...", file=sys.stderr)
    retriever = setup_vectorstore()
    grader = build_doc_grader()
    refiner = build_query_refiner()
    answer_chain = build_answer_chain()

    # Step 1: retrieve
    print(f"\n[retrieve] Question: {args.question}", file=sys.stderr)
    docs = retriever.invoke(args.question)
    print(f"  Retrieved {len(docs)} docs", file=sys.stderr)

    # Step 2: grade each doc
    print(f"\n[grade] Scoring retrieved docs...", file=sys.stderr)
    grades = []
    for i, doc in enumerate(docs):
        g = grader.invoke({
            "question": args.question,
            "document": doc.page_content[:2000],
        })
        grades.append((doc, g.score, g.reasoning))
        preview = g.reasoning[:80].replace("\n", " ")
        print(f"  Doc {i+1}: score={g.score:.2f} - {preview}", file=sys.stderr)

    # Step 3: T-correct decision
    scores = [g[1] for g in grades]
    decision, avg_score = t_correct_decision(scores, args.upper, args.lower)
    print(f"\n[t-correct] Avg score: {avg_score:.2f} -> {decision}", file=sys.stderr)

    # Step 4: gather context based on decision
    if decision == "CORRECT":
        filtered = [d[0] for d in grades if d[1] >= 0.5]
        context = format_docs(filtered) if filtered else format_docs(docs)
        print(f"  Using {len(filtered) if filtered else len(docs)} retrieved docs as-is", file=sys.stderr)

    elif decision == "INCORRECT":
        print(f"  Falling back to web search...", file=sys.stderr)
        context = web_search(args.question, max_results=4)

    else:  # AMBIGUOUS
        print(f"  Refining query...", file=sys.stderr)
        refined = refiner.invoke({
            "question": args.question,
            "context": format_docs(docs),
        })
        print(f"  Refined: {refined}", file=sys.stderr)
        filtered = [d[0] for d in grades if d[1] >= 0.5]
        retrieved_ctx = format_docs(filtered) if filtered else ""
        web_ctx = web_search(refined, max_results=3)
        context = (
            f"Retrieved context (filtered):\n{retrieved_ctx}\n\n"
            f"Web search results (refined query):\n{web_ctx}"
        )

    # Step 5: generate answer
    print(f"\n[generate] Producing final answer...\n", file=sys.stderr)
    print("=" * 60)
    answer = answer_chain.invoke({"context": context, "question": args.question})
    print(answer)
    print("=" * 60)

    print(f"\n[trace] Decision: {decision}, avg_score: {avg_score:.2f}", file=sys.stderr)

if __name__ == "__main__":
    main()