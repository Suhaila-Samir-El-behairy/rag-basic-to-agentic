"""Indexing strategies for vector stores.

Strategies:
  - recursive: default, splits on a hierarchy of separators
  - character: simple character-based splitting
  - token:     splits on token count (LLM-aware boundaries)
  - semantic:  splits based on embedding similarity between sentences
  - multi-rep: stores summaries, returns full original docs on retrieval

The script indexes once per strategy (persisted), then lets you query it.
"""

import argparse
import sys
import warnings

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from ragkit import corpus
from ragkit.pipeline import RAGResult
from ragkit.utils import generate_answer, get_embeddings, get_llm, get_vectorstore, split_docs

# Suppress the noisy semantic chunker warning
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_experimental")

STRATEGIES = ["recursive", "character", "token", "semantic", "multi-rep"]


# ============================================================
# Chunking strategies
# ============================================================
def chunk_by_strategy(docs, strategy: str, embeddings):
    """Split documents using the chosen strategy."""
    if strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents(docs), "Recursive character splitting"

    if strategy == "character":
        char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separator="\n")
        return char_splitter.split_documents(docs), "Character splitting (single separator)"

    if strategy == "token":
        token_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)
        return token_splitter.split_documents(docs), "Token-based splitting (512 tokens)"

    if strategy == "semantic":
        from langchain_experimental.text_splitter import SemanticChunker

        print(
            "  Note: semantic chunking embeds every sentence; on the free HF tier this "
            "can take a few minutes and is retried with backoff on rate limits.",
            file=sys.stderr,
        )
        semantic_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
        return semantic_splitter.split_documents(docs), "Semantic chunking (embedding similarity)"

    raise ValueError(f"Unknown strategy: {strategy}")


def basic_strategy_index(docs, strategy: str, embeddings):
    """Build (or load) a vector store for one chunking strategy.

    Chunking only runs when the store is empty: the semantic splitter costs one
    embedding call per sentence, which must not be paid on every query.
    """
    store = get_vectorstore([], name=f"04_{strategy}", embeddings=embeddings)
    if not store.get(limit=1)["ids"]:
        splits, description = chunk_by_strategy(docs, strategy, embeddings)
        print(f"  {description}: {len(splits)} chunks", file=sys.stderr)
        store.add_documents(splits)
    return store


# ============================================================
# Multi-representation indexing
# ============================================================
def multi_representation_index(docs, question: str, embeddings, k: int = 3) -> list[Document]:
    """Store summaries, retrieve full docs that match the summary."""
    # The web loader returns the whole post as a single document: split it into
    # large "parent" chunks so each summary call fits in the LLM's token limits.
    docs = split_docs(docs, chunk_size=4000, chunk_overlap=0)

    summary_vs = get_vectorstore([], name="04_multi_rep", embeddings=embeddings)
    if not summary_vs.get(limit=1)["ids"]:
        summary_vs.add_documents(_summarize(docs))

    # Retrieve summaries, return the full parent chunks they point to
    retrieved_summaries = summary_vs.similarity_search(question, k=k)
    full_docs = [docs[int(r.metadata["doc_id"])] for r in retrieved_summaries]

    print(
        f"  Matched {len(retrieved_summaries)} summaries → returning {len(full_docs)} full docs",
        file=sys.stderr,
    )
    return full_docs


def _summarize(docs) -> list[Document]:
    """One LLM summary per doc, tagged with the index of the doc it summarizes."""
    print(f"  Generating summaries for {len(docs)} docs (one LLM call each)...", file=sys.stderr)
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
    return [Document(page_content=s, metadata={"doc_id": i}) for i, s in enumerate(summaries)]


def size_stats(docs) -> str:
    sizes = [len(d.page_content) for d in docs]
    if not sizes:
        return "no chunks retrieved (is the index empty?)"
    return f"min={min(sizes)}, max={max(sizes)}, avg={sum(sizes) // len(sizes)}"


def run(question: str, strategy: str = "recursive", k: int = 4) -> RAGResult:
    docs = corpus.load_docs()
    embeddings = get_embeddings()

    if strategy == "multi-rep":
        retrieved = multi_representation_index(docs, question, embeddings)
    else:
        store = basic_strategy_index(docs, strategy, embeddings)
        retrieved = store.as_retriever(search_kwargs={"k": k}).invoke(question)

    return RAGResult(
        answer=generate_answer(question, retrieved),
        contexts=[d.page_content for d in retrieved],
        metadata={"strategy": strategy, "chunk_sizes": size_stats(retrieved)},
    )


# ============================================================
# CLI entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Compare indexing strategies")
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default="recursive",
        help="Indexing strategy (default: recursive)",
    )
    args = parser.parse_args()

    print(f"\n[index] Strategy: {args.strategy}", file=sys.stderr)
    result = run(args.question, strategy=args.strategy)
    print(f"  Retrieved chunk sizes: {result.metadata['chunk_sizes']}", file=sys.stderr)

    print("=" * 60)
    print(result.answer)
    print("=" * 60)


if __name__ == "__main__":
    main()
