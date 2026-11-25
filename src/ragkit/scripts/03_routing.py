"""Route questions to the best data source.

Sources:
  - vectorstore: Lilian Weng's LLM agent blog post
  - web:         public web search (DuckDuckGo, or Tavily if a key is set)
  - arxiv:       arXiv (academic papers)

The LLM acts as a classifier via structured output, then we retrieve
from the chosen source and generate a final answer.
"""

import argparse
import sys
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ragkit.corpus import default_retriever
from ragkit.pipeline import RAGResult
from ragkit.utils import generate_answer, get_llm, web_search

ROUTES = ("vectorstore", "web", "arxiv")
FALLBACK_ROUTE = "vectorstore"


# ============================================================
# Router: structured-output classifier
# ============================================================
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "web", "arxiv"] = Field(
        description="The best data source to answer the question."
    )


def build_router():
    prompt = ChatPromptTemplate.from_template(
        """You are routing a question to the best data source.

Choose exactly ONE source:

- vectorstore: Questions specifically about LLM agents, task decomposition,
  ReAct, chain-of-thought, planning, memory, tool use, or anything covered
  in Lilian Weng's blog post on LLM-powered autonomous agents.

- web: General knowledge or current-events questions (history, science,
  geography, people, products, everyday concepts, definitions).

- arxiv: Questions asking for specific academic papers or recent research
  results in machine learning, AI, statistics, or physics.

Question: {question}

Return the source name."""
    )
    return prompt | get_llm().with_structured_output(RouteQuery)


def route(question: str) -> tuple[str, str | None]:
    """Return (datasource, error). Falls back to the vector store if the router fails."""
    try:
        return build_router().invoke({"question": question}).datasource, None
    except Exception as e:  # noqa: BLE001 - malformed structured output is common on free models
        return FALLBACK_ROUTE, f"{type(e).__name__}: {e}"


# ============================================================
# Retrieve from the chosen source
# ============================================================
def search_arxiv(query: str, max_results: int = 3, summary_chars: int | None = None) -> str:
    import arxiv

    client = arxiv.Client(page_size=max_results, delay_seconds=2, num_retries=3)
    search = arxiv.Search(
        query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
    )
    results = [
        f"Title: {paper.title}\n"
        f"Authors: {', '.join(a.name for a in paper.authors)}\n"
        f"Published: {paper.published.strftime('%Y-%m-%d')}\n"
        f"Summary: {paper.summary[:summary_chars] if summary_chars else paper.summary}"
        for paper in client.results(search)
    ]
    return "\n\n---\n\n".join(results) if results else "No papers found."


def retrieve_from_source(datasource: str, question: str) -> list[str]:
    """Return the retrieved contexts for the chosen source."""
    if datasource == "vectorstore":
        return [d.page_content for d in default_retriever().invoke(question)]
    if datasource == "web":
        return [web_search(question, max_results=4)]
    if datasource == "arxiv":
        return [search_arxiv(question)]
    raise ValueError(f"Unknown datasource: {datasource}")


def run(question: str) -> RAGResult:
    datasource, router_error = route(question)
    contexts = retrieve_from_source(datasource, question)
    return RAGResult(
        answer=generate_answer(question, "\n\n".join(contexts)),
        contexts=contexts,
        tool_calls=[datasource],
        metadata={"router_error": router_error} if router_error else {},
    )


# ============================================================
# CLI entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Route questions to vectorstore / web / arxiv")
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    print("\n[router] Classifying question...", file=sys.stderr)
    result = run(args.question)
    print(f"[router] → {result.tool_calls[0]}", file=sys.stderr)
    if result.metadata.get("router_error"):
        print(f"[router] router failed, used fallback: {result.metadata}", file=sys.stderr)
    preview = "\n".join(result.contexts)[:200].replace("\n", " ")
    print(f"[retrieve] Context preview: {preview}...\n", file=sys.stderr)

    print("=" * 60)
    print(result.answer)
    print("=" * 60)


if __name__ == "__main__":
    main()
