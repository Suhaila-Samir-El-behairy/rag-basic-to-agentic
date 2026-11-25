"""Query transformation techniques for RAG.

Implements four methods to improve retrieval recall:
  - multi-query: LLM generates query variants, retrieve for each, union results
  - rag-fusion: like multi-query, but rank with Reciprocal Rank Fusion (RRF)
  - hyde: generate hypothetical answer, retrieve based on it
  - step-back: generate more general question, retrieve based on that

All methods return a final generated answer (not just retrieved docs).
"""

import argparse
import sys

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ragkit.corpus import default_retriever
from ragkit.pipeline import RAGResult
from ragkit.utils import generate_answer, get_llm, reciprocal_rank_fusion


def _lines(text: str) -> list[str]:
    return [q.strip() for q in text.split("\n") if q.strip()]


# ============================================================
# Method 1: Multi-Query Retriever (LangChain built-in)
# ============================================================
def multi_query_method(retriever, question):
    """LLM generates multiple query variants; union of all retrieved docs."""
    mqr = MultiQueryRetriever.from_llm(retriever=retriever, llm=get_llm())
    docs = mqr.invoke(question)
    print(f"  Retrieved {len(docs)} unique docs (union across variants)", file=sys.stderr)
    return docs


# ============================================================
# Method 2: RAG-Fusion (custom, with RRF ranking)
# ============================================================
def rag_fusion_method(retriever, question, num_queries: int = 4, top_n: int = 4):
    """Generate multiple queries, retrieve for each, fuse with Reciprocal Rank Fusion."""
    generate_prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Generate {num} different search queries
that would help answer the user's question. Each query should approach the
topic from a different angle (e.g., definition, example, mechanism, trade-off).

Original question: {question}

Output one query per line. No numbering, no bullets, no preamble."""
    )
    query_chain = generate_prompt | get_llm() | StrOutputParser() | _lines
    # Keep the original question too, so a bad rewrite can't lose the obvious hits.
    queries = [question, *query_chain.invoke({"num": num_queries, "question": question})]

    print(f"  Generated {len(queries) - 1} query variants:", file=sys.stderr)
    for q in queries[1:]:
        print(f"    • {q}", file=sys.stderr)

    fused = reciprocal_rank_fusion(retriever.batch(queries))
    docs = fused[:top_n]
    print(f"  Fused to top {len(docs)} docs via RRF", file=sys.stderr)
    return docs


# ============================================================
# Method 3: HyDE (Hypothetical Document Embeddings)
# ============================================================
def hyde_method(retriever, question):
    """Generate a hypothetical answer, then retrieve docs similar to it."""
    hyde_prompt = ChatPromptTemplate.from_template(
        """You are an expert. Write a detailed, technical passage that would
answer the question below, as if it were excerpted from a relevant document.
Do NOT say "I don't know" — make your best informed guess.

Question: {question}

Hypothetical passage:"""
    )
    hypothetical = (hyde_prompt | get_llm() | StrOutputParser()).invoke({"question": question})

    print("  Hypothetical passage (first 200 chars):", file=sys.stderr)
    print(f"    {hypothetical[:200]}...", file=sys.stderr)

    docs = retriever.invoke(hypothetical)
    print(f"  Retrieved {len(docs)} docs similar to the hypothetical answer", file=sys.stderr)
    return docs


# ============================================================
# Method 4: Step-Back Prompting
# ============================================================
def step_back_method(retriever, question):
    """Retrieve for both the original and a more general question, answer the original."""
    sb_prompt = ChatPromptTemplate.from_template(
        """You are an expert. Generate a more general, abstract version of the
question that captures the underlying concept or principle. This should be
answerable even if the original question is too specific.

Original question: {question}

Output ONLY the step-back question."""
    )
    general_q = (sb_prompt | get_llm() | StrOutputParser()).invoke({"question": question})
    print(f"  Step-back question: {general_q}", file=sys.stderr)

    # As in Zheng et al. (2023), ground the answer in both the specific and the
    # step-back context, not the step-back context alone.
    docs = reciprocal_rank_fusion(retriever.batch([question, general_q]))[:4]
    print(f"  Retrieved {len(docs)} docs (original + step-back, fused)", file=sys.stderr)
    return docs


# ============================================================
# Registry
# ============================================================
METHODS = {
    "multi-query": multi_query_method,
    "rag-fusion": rag_fusion_method,
    "hyde": hyde_method,
    "step-back": step_back_method,
}


def run(question: str, method: str = "rag-fusion") -> RAGResult:
    docs = METHODS[method](default_retriever(), question)
    return RAGResult(
        answer=generate_answer(question, docs),
        contexts=[d.page_content for d in docs],
        metadata={"method": method},
    )


# ============================================================
# CLI entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Query transformation techniques for RAG")
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--method",
        choices=list(METHODS.keys()),
        default="rag-fusion",
        help="Query transformation method (default: rag-fusion)",
    )
    args = parser.parse_args()

    print(f"\n[retrieve] Method: {args.method}", file=sys.stderr)
    print(f"[retrieve] Question: {args.question}", file=sys.stderr)
    result = run(args.question, method=args.method)

    print("=" * 60)
    print(result.answer)
    print("=" * 60)


if __name__ == "__main__":
    main()
