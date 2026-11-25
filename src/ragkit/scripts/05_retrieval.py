"""Retrieval mechanisms for RAG.

Methods:
  - basic:      vanilla vector similarity search
  - rerank:     retrieve top-k candidates, then rerank with FlashRank (cross-encoder)
  - rag-fusion: generate multiple queries, retrieve for each, fuse with RRF
  - hybrid:     combine BM25 (keyword) + dense (semantic) via weighted RRF
"""

import argparse
import importlib
import sys
from functools import lru_cache

from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnableLambda

from ragkit import corpus
from ragkit.pipeline import RAGResult
from ragkit.utils import generate_answer

METHODS = ["basic", "rerank", "rag-fusion", "hybrid"]


def build_basic(vectorstore, k: int = 4):
    return vectorstore.as_retriever(search_kwargs={"k": k})


def build_rerank(vectorstore, candidates: int = 10, top_n: int = 4):
    """Retrieve top-`candidates` by embedding, rerank to `top_n` with FlashRank."""
    return ContextualCompressionRetriever(
        base_compressor=FlashrankRerank(top_n=top_n),
        base_retriever=vectorstore.as_retriever(search_kwargs={"k": candidates}),
    )


def build_rag_fusion(vectorstore):
    """Reuse script 02's RAG-Fusion so both scripts measure the same thing."""
    script02 = importlib.import_module("ragkit.scripts.02_query_transformations")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return RunnableLambda(lambda q: script02.rag_fusion_method(retriever, q))


def build_hybrid(splits, vectorstore, k: int = 4, weights=(0.4, 0.6)):
    """Ensemble: BM25 (keyword) + dense (semantic), fused with weighted RRF."""
    bm25 = BM25Retriever.from_documents(splits)
    bm25.k = k
    return EnsembleRetriever(
        retrievers=[bm25, vectorstore.as_retriever(search_kwargs={"k": k})],
        weights=list(weights),
    )


@lru_cache(maxsize=len(METHODS))
def get_retriever(method: str):
    vectorstore = corpus.default_vectorstore()
    if method == "basic":
        return build_basic(vectorstore)
    if method == "rerank":
        print("  Note: FlashRank downloads its model once, then caches.", file=sys.stderr)
        return build_rerank(vectorstore)
    if method == "rag-fusion":
        return build_rag_fusion(vectorstore)
    if method == "hybrid":
        return build_hybrid(corpus.load_splits(), vectorstore)
    raise ValueError(f"Unknown method: {method}")


def run(question: str, method: str = "rerank") -> RAGResult:
    docs = get_retriever(method).invoke(question)
    return RAGResult(
        answer=generate_answer(question, docs),
        contexts=[d.page_content for d in docs],
        metadata={"method": method},
    )


def main():
    parser = argparse.ArgumentParser(description="Compare retrieval mechanisms")
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--method",
        choices=METHODS,
        default="rerank",
        help="Retrieval method (default: rerank)",
    )
    args = parser.parse_args()

    print(f"\n[retrieve] Method: {args.method}", file=sys.stderr)
    result = run(args.question, method=args.method)
    print(f"  Retrieved {len(result.contexts)} docs", file=sys.stderr)

    print("=" * 60)
    print(result.answer)
    print("=" * 60)


if __name__ == "__main__":
    main()
