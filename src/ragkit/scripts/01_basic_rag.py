"""Basic RAG pipeline: load -> split -> embed -> retrieve -> generate."""

import argparse

from ragkit.corpus import default_retriever
from ragkit.pipeline import RAGResult
from ragkit.utils import generate_answer


def run(question: str, k: int = 4) -> RAGResult:
    docs = default_retriever(k).invoke(question)
    return RAGResult(
        answer=generate_answer(question, docs),
        contexts=[d.page_content for d in docs],
    )


def main():
    parser = argparse.ArgumentParser(description="Basic RAG")
    parser.add_argument("--question", required=True)
    parser.add_argument("--k", type=int, default=4, help="Chunks to retrieve (default 4)")
    args = parser.parse_args()

    print(run(args.question, k=args.k).answer)


if __name__ == "__main__":
    main()
