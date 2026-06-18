"""Route questions to the best data source.

Sources:
  - vectorstore: Lilian Weng's LLM agent blog post
  - wiki:       DuckDuckGo (general knowledge, no key needed)
  - arxiv:      arXiv (academic papers, new Client API)

The LLM acts as a classifier via structured output, then we retrieve
from the chosen source and generate a final answer.
"""

import argparse
import sys
from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ragkit.utils import format_docs, get_llm, get_vectorstore, load_web, split_docs, web_search

URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"


# ============================================================
# Router: structured-output classifier
# ============================================================
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "wiki", "arxiv"] = Field(
        description="The best data source to answer the question."
    )


def build_router():
    prompt = ChatPromptTemplate.from_template(
        """You are routing a question to the best data source.

Choose exactly ONE source:

- vectorstore: Questions specifically about LLM agents, task decomposition,
  ReAct, chain-of-thought, planning, memory, tool use, or anything covered
  in Lilian Weng's blog post on LLM-powered autonomous agents.

- wiki: General knowledge questions (history, science, geography, people,
  everyday concepts, definitions). Backed by DuckDuckGo web search.

- arxiv: Academic or research questions, paper lookups, technical concepts
  in machine learning, AI, statistics, or physics.

Question: {question}

Return the source name."""
    )
    return prompt | get_llm().with_structured_output(RouteQuery)


# ============================================================
# Source-specific retrievers
# ============================================================
def get_vector_retriever():
    docs = load_web(URL)
    splits = split_docs(docs)
    vectorstore = get_vectorstore(splits, name="03_routing")
    return vectorstore.as_retriever(search_kwargs={"k": 4})


# ============================================================
# Retrieve from the chosen source
# ============================================================
def retrieve_from_source(datasource: str, question: str) -> str:
    """Returns a single context string ready for the answer LLM."""
    if datasource == "vectorstore":
        retriever = get_vector_retriever()
        docs = retriever.invoke(question)
        return format_docs(docs[:4])

    if datasource == "wiki":
        # DuckDuckGo (no key, more reliable than Wikipedia API)
        return web_search(question, max_results=4)

    if datasource == "arxiv":
        # New arXiv library API (arxiv.Client + .results())
        import arxiv

        client = arxiv.Client(page_size=3, delay_seconds=2, num_retries=3)
        search = arxiv.Search(
            query=question,
            max_results=3,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = []
        for paper in client.results(search):
            results.append(
                f"Title: {paper.title}\n"
                f"Authors: {', '.join(a.name for a in paper.authors)}\n"
                f"Published: {paper.published.strftime('%Y-%m-%d')}\n"
                f"Summary: {paper.summary}"
            )
        return "\n\n---\n\n".join(results) if results else "No papers found."

    raise ValueError(f"Unknown datasource: {datasource}")


# ============================================================
# Final answer generation
# ============================================================
def generate_answer(question: str, context: str) -> str:
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based ONLY on the context below.
If the context doesn't contain the answer, say "I don't know based on this source."

Context:
{context}

Question: {question}

Answer:"""
    )
    chain = prompt | get_llm() | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


# ============================================================
# CLI entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Route questions to vectorstore / wiki / arxiv (Groq-powered)"
    )
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    # Step 1: route
    print("\n[router] Classifying question...", file=sys.stderr)
    router = build_router()
    route = router.invoke({"question": args.question})
    print(f"[router] → {route.datasource}\n", file=sys.stderr)

    # Step 2: retrieve
    print(f"[retrieve] Fetching from {route.datasource}...", file=sys.stderr)
    context = retrieve_from_source(route.datasource, args.question)
    preview = context[:200].replace("\n", " ")
    print(f"[retrieve] Context preview: {preview}...\n", file=sys.stderr)

    # Step 3: generate
    print("[generate] Producing final answer...\n", file=sys.stderr)
    print("=" * 60)
    answer = generate_answer(args.question, context)
    print(answer)
    print("=" * 60)


if __name__ == "__main__":
    main()
