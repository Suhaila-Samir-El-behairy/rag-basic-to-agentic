"""Indexing strategies for vector stores.

Strategies:
  - recursive: default, splits on a hierarchy of separators
  - character: simple character-based splitting
  - token:     splits on token count (LLM-aware boundaries)
  - semantic:  splits based on embedding similarity between chunks
  - multi-rep: stores summaries, returns full original docs on retrieval

The script indexes once per strategy, then lets you query it.
"""

import argparse
import sys
import warnings

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from ragkit.config import CHROMA_DIR
from ragkit.utils import (
    format_docs,
    get_embeddings,
    get_llm,
    load_web,
)

# Suppress the noisy semantic chunker warning
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_experimental")

URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"


# ============================================================
# Chunking strategies
# ============================================================
def chunk_by_strategy(docs, strategy: str, embeddings):
    """Split documents using the chosen strategy."""
    if strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents(docs), "Recursive character splitting"

    if strategy == "character":
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separator="\n")
        return splitter.split_documents(docs), "Character splitting (single separator)"

    if strategy == "token":
        splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)
        return splitter.split_documents(docs), "Token-based splitting (512 tokens)"

    if strategy == "semantic":
        from langchain_experimental.text_splitter import SemanticChunker

        splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
        )
        return splitter.split_documents(docs), "Semantic chunking (embedding similarity)"

    raise ValueError(f"Unknown strategy: {strategy}")


def basic_strategy_index(splits, strategy: str, embeddings):
    """Build a simple vector store from chunks."""
    persist_dir = CHROMA_DIR / f"04_{strategy}"
    if (persist_dir / "chroma.sqlite3").exists():
        return Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)
    return Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )


# ============================================================
# Multi-representation indexing
# ============================================================
def multi_representation_index(docs, question: str, embeddings) -> str:
    """Store summaries, retrieve full docs that match the summary."""
    persist_dir = CHROMA_DIR / "04_multi_rep"

    # Step 1: generate summaries
    print("  Generating summaries for each doc (one LLM call per doc)...", file=sys.stderr)
    summary_prompt = ChatPromptTemplate.from_template(
        """Summarize the following document in 2-3 sentences.
        Focus on the main topic and key technical concepts.

        Document:
        {doc}

        Summary:"""
    )
    summary_chain = summary_prompt | get_llm() | StrOutputParser()

    summaries = summary_chain.batch(
        [{"doc": d.page_content} for d in docs],
        {"max_concurrency": 3},
    )
    print(f"  Generated {len(summaries)} summaries", file=sys.stderr)

    # Step 2: build (or load) summary vector store
    if (persist_dir / "chroma.sqlite3").exists():
        summary_vs = Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)
    else:
        summary_vs = Chroma.from_texts(
            texts=summaries,
            embedding=embeddings,
            metadatas=[{"doc_id": i} for i in range(len(docs))],
            persist_directory=str(persist_dir),
        )

    # Step 3: retrieve summaries, return full docs
    retrieved_summaries = summary_vs.similarity_search(question, k=3)
    full_docs = [docs[int(r.metadata["doc_id"])] for r in retrieved_summaries]

    print(
        f"  Matched {len(retrieved_summaries)} summaries → returning {len(full_docs)} full docs",
        file=sys.stderr,
    )
    return full_docs


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
STRATEGIES = ["recursive", "character", "token", "semantic", "multi-rep"]


def main():
    parser = argparse.ArgumentParser(description="Compare indexing strategies (Groq + HuggingFace)")
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default="recursive",
        help="Indexing strategy (default: recursive)",
    )
    args = parser.parse_args()

    print(f"\n[setup] Loading {URL}...", file=sys.stderr)
    docs = load_web(URL)
    embeddings = get_embeddings()

    print(f"\n[index] Strategy: {args.strategy}", file=sys.stderr)

    if args.strategy == "multi-rep":
        full_docs = multi_representation_index(docs, args.question, embeddings)
        print("\n[generate] Producing final answer...\n", file=sys.stderr)
        print("=" * 60)
        print(generate_answer(args.question, full_docs))
        print("=" * 60)
        return

    # For the 4 chunking strategies
    splits, description = chunk_by_strategy(docs, args.strategy, embeddings)
    print(f"  Description: {description}", file=sys.stderr)
    print(f"  Produced {len(splits)} chunks", file=sys.stderr)

    print(
        f"  Building vector store (persisted to chroma_db/04_{args.strategy}/)...", file=sys.stderr
    )
    vectorstore = basic_strategy_index(splits, args.strategy, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    print(f"\n[retrieve] Fetching top-4 chunks for: {args.question}", file=sys.stderr)
    retrieved = retriever.invoke(args.question)

    # Show chunk size distribution
    sizes = [len(d.page_content) for d in retrieved]
    print(
        f"  Chunk sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes) // len(sizes)}",
        file=sys.stderr,
    )

    print("\n[generate] Producing final answer...\n", file=sys.stderr)
    print("=" * 60)
    print(generate_answer(args.question, retrieved))
    print("=" * 60)


if __name__ == "__main__":
    main()
