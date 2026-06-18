"""Retrieval mechanisms for RAG.

Methods:
  - basic:     vanilla vector similarity search
  - rerank:    retrieve top-k candidates, then rerank with FlashRank
  - rag-fusion: generate multiple queries, retrieve for each, fuse with RRF
  - hybrid:    combine BM25 (keyword) + dense (semantic) via ensemble

All methods are run on the same query for easy comparison.
"""

import argparse
import pickle
import sys
from pathlib import Path

from langchain_classic.load import dumps, loads
from langchain_classic.retrievers import (
    BM25Retriever,
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_classic.retrievers.document_compressors import FlashrankRerank
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ragkit.utils import (
    format_docs,
    get_llm,
    get_vectorstore,
    load_web,
    split_docs,
)

URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"
BM25_CACHE = Path("./bm25_cache.pkl")


# ============================================================
# Method 1: Basic dense retrieval
# ============================================================
def build_basic(vectorstore):
    return vectorstore.as_retriever(search_kwargs={"k": 4})


# ============================================================
# Method 2: Reranking with FlashRank
# ============================================================
def build_rerank(vectorstore):
    """Retrieve top-10 candidates, rerank to top-4 with FlashRank."""
    base = vectorstore.as_retriever(search_kwargs={"k": 10})
    compressor = FlashrankRerank(top_n=4)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base,
    )


# ============================================================
# Method 3: RAG-Fusion
# ============================================================
def build_rag_fusion(vectorstore, num_queries: int = 4):
    """Generate multiple queries, retrieve for each, fuse with RRF."""
    generate_prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Generate {num} different search queries
that would help answer the user's question. Each query should approach the
topic from a different angle.

Original question: {question}

Output one query per line. No numbering, no bullets, no preamble."""
    )

    def retrieve(question: str):
        chain = (
            generate_prompt
            | get_llm()
            | StrOutputParser()
            | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
        )
        queries = chain.invoke({"num": num_queries, "question": question})
        all_results = vectorstore.as_retriever(search_kwargs={"k": 4}).batch(queries)

        # Reciprocal Rank Fusion
        fused = {}
        for docs in all_results:
            for rank, doc in enumerate(docs):
                key = dumps(doc)
                fused[key] = fused.get(key, 0) + 1 / (rank + 60)
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return [loads(key) for key, _ in ranked[:4]]

    # Wrap as a callable retriever-like object
    class RAGFusionRetriever:
        def invoke(self, question):
            return retrieve(question)

        def batch(self, questions):
            return [retrieve(q) for q in questions]

    return RAGFusionRetriever()


# ============================================================
# Method 4: Hybrid (BM25 + dense)
# ============================================================
def build_hybrid(splits, vectorstore):
    """Ensemble: BM25 (keyword) + dense (semantic) with weighted scoring."""
    bm25 = BM25Retriever.from_documents(splits)
    bm25.k = 4
    dense = vectorstore.as_retriever(search_kwargs={"k": 4})
    return EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[0.4, 0.6],  # favor semantic slightly
    )


# ============================================================
# Final answer generation
# ============================================================
def generate_answer(question: str, docs) -> str:
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based ONLY on the context below.

Context:
{context}

Question: {question}

Answer:"""
    )
    chain = prompt | get_llm() | StrOutputParser()
    return chain.invoke({"context": format_docs(docs), "question": question})


# ============================================================
# CLI entry point
# ============================================================
METHODS = ["basic", "rerank", "rag-fusion", "hybrid"]


def main():
    parser = argparse.ArgumentParser(
        description="Compare retrieval mechanisms (Groq + HuggingFace)"
    )
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--method",
        choices=METHODS,
        default="rerank",
        help="Retrieval method (default: rerank)",
    )
    args = parser.parse_args()

    print(f"\n[setup] Loading + indexing {URL}...", file=sys.stderr)
    docs = load_web(URL)

    if BM25_CACHE.exists():
        print(f"  Loading cached splits from {BM25_CACHE}", file=sys.stderr)
        splits = pickle.load(open(BM25_CACHE, "rb"))
    else:
        splits = split_docs(docs)
        pickle.dump(splits, open(BM25_CACHE, "wb"))
        print(f"  Cached {len(splits)} splits to {BM25_CACHE}", file=sys.stderr)

    vectorstore = get_vectorstore(splits, name="05_retrieval")

    print(f"\n[retrieve] Method: {args.method}", file=sys.stderr)
    print(f"[retrieve] Question: {args.question}", file=sys.stderr)

    if args.method == "basic":
        retriever = build_basic(vectorstore)
        docs = retriever.invoke(args.question)
    elif args.method == "rerank":
        print("  Note: FlashRank downloads ~100MB on first run, then caches.", file=sys.stderr)
        retriever = build_rerank(vectorstore)
        docs = retriever.invoke(args.question)
    elif args.method == "rag-fusion":
        retriever = build_rag_fusion(vectorstore)
        docs = retriever.invoke(args.question)
    elif args.method == "hybrid":
        retriever = build_hybrid(splits, vectorstore)
        docs = retriever.invoke(args.question)

    print(f"  Retrieved {len(docs)} docs", file=sys.stderr)

    print("\n[generate] Producing final answer...\n", file=sys.stderr)
    print("=" * 60)
    print(generate_answer(args.question, docs))
    print("=" * 60)


if __name__ == "__main__":
    main()
