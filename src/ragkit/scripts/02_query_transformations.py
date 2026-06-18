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

from langchain_classic.load import dumps, loads
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ragkit.utils import format_docs, get_llm, get_vectorstore, load_web, split_docs

URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"

# ============================================================
# Shared answer-generation prompt
# ============================================================
ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question based ONLY on the context below.
If the context doesn't contain the answer, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""
)


def generate_answer(question: str, docs) -> str:
    """Generate a final answer from retrieved documents."""
    chain = ANSWER_PROMPT | get_llm() | StrOutputParser()
    return chain.invoke({"context": format_docs(docs), "question": question})


# ============================================================
# Reciprocal Rank Fusion (used by rag-fusion)
# ============================================================
def reciprocal_rank_fusion(results, k: int = 60):
    """Fuse ranked lists from multiple queries into a single ranked list."""
    fused = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            key = dumps(doc)
            fused[key] = fused.get(key, 0) + 1 / (rank + k)
    return [loads(key) for key, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)]


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
def rag_fusion_method(retriever, question, num_queries: int = 4):
    """Generate multiple queries, retrieve for each, fuse with Reciprocal Rank Fusion."""
    generate_prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Generate {num} different search queries
that would help answer the user's question. Each query should approach the
topic from a different angle (e.g., definition, example, mechanism, trade-off).

Original question: {question}

Output one query per line. No numbering, no bullets, no preamble."""
    )
    query_chain = (
        generate_prompt
        | get_llm()
        | StrOutputParser()
        | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
    )
    queries = query_chain.invoke({"num": num_queries, "question": question})

    print(f"  Generated {len(queries)} query variants:", file=sys.stderr)
    for q in queries:
        print(f"    • {q}", file=sys.stderr)

    # Retrieve for each query in parallel
    all_results = retriever.batch(queries)
    # Fuse with RRF and take top 4
    fused = reciprocal_rank_fusion(all_results)
    docs = fused[:4]
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
    hyde_chain = hyde_prompt | get_llm() | StrOutputParser()
    hypothetical = hyde_chain.invoke({"question": question})

    print("  Hypothetical passage (first 200 chars):", file=sys.stderr)
    print(f"    {hypothetical[:200]}...", file=sys.stderr)

    docs = retriever.invoke(hypothetical)
    print(f"  Retrieved {len(docs)} docs similar to the hypothetical answer", file=sys.stderr)
    return docs


# ============================================================
# Method 4: Step-Back Prompting
# ============================================================
def step_back_method(retriever, question):
    """Generate a more general question, retrieve based on that, answer original."""
    sb_prompt = ChatPromptTemplate.from_template(
        """You are an expert. Generate a more general, abstract version of the
question that captures the underlying concept or principle. This should be
answerable even if the original question is too specific.

Original question: {question}

Step-back question:"""
    )
    sb_chain = sb_prompt | get_llm() | StrOutputParser()
    general_q = sb_chain.invoke({"question": question})

    print(f"  Step-back question: {general_q}", file=sys.stderr)

    docs = retriever.invoke(general_q)
    print(f"  Retrieved {len(docs)} docs based on the step-back question", file=sys.stderr)
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


def setup_index():
    """Build (or load) the vectorstore once. Persists to disk via Chroma."""
    print(f"  Loading + indexing {URL} ...", file=sys.stderr)
    docs = load_web(URL)
    splits = split_docs(docs)
    vectorstore = get_vectorstore(splits, name="02_queries")
    print(f"  Indexed {len(splits)} chunks", file=sys.stderr)
    return vectorstore


# ============================================================
# CLI entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Query transformation techniques for RAG (Groq-powered)"
    )
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--method",
        choices=list(METHODS.keys()),
        default="rag-fusion",
        help="Query transformation method (default: rag-fusion)",
    )
    args = parser.parse_args()

    print("\n[setup] Building/loading vector store...", file=sys.stderr)
    vectorstore = setup_index()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    print(f"\n[retrieve] Method: {args.method}", file=sys.stderr)
    print(f"[retrieve] Question: {args.question}", file=sys.stderr)

    method_fn = METHODS[args.method]
    docs = method_fn(retriever, args.question)

    print("\n[generate] Producing final answer...\n", file=sys.stderr)
    print("=" * 60)
    answer = generate_answer(args.question, docs)
    print(answer)
    print("=" * 60)


if __name__ == "__main__":
    main()
