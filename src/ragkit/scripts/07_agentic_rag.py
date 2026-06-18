"""Agentic RAG: a ReAct agent that picks tools dynamically.

Tools:
  - vector_search: search the local vector store
  - web_search:    search the public web (DuckDuckGo)
  - arxiv_search:  search arXiv academic papers

The agent decides which tool(s) to use based on the question.
"""

import argparse
import pickle
import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage

from ragkit.utils import (
    format_docs,
    get_llm,
    get_vectorstore,
    load_web,
    split_docs,
    web_search,
)

URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"
BM25_CACHE = Path("./bm25_cache_07.pkl")
_vectorstore = None  # global, set in setup()


# ============================================================
# Tools
# ============================================================
@tool
def vector_search(query: str) -> str:
    """Search the local vector store of Lilian Weng's blog post on LLM agents.
    Use for questions about LLM agents, task decomposition, ReAct, planning,
    memory, tool use, or chain-of-thought."""
    docs = _vectorstore.similarity_search(query, k=4)
    if not docs:
        return "No relevant documents found in the vector store."
    return format_docs(docs)


@tool
def web_search_tool(query: str) -> str:
    """Search the public web for current information or general knowledge.
    Use for questions about current events, general topics, definitions,
    or anything not in the local vector store."""
    return web_search(query, max_results=3)


@tool
def arxiv_search(query: str) -> str:
    """Search arXiv for academic papers on a topic.
    Use for research/academic questions in ML, AI, statistics, or physics."""
    import arxiv

    client = arxiv.Client(page_size=3, delay_seconds=2, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=3,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results = []
    for paper in client.results(search):
        results.append(
            f"Title: {paper.title}\n"
            f"Authors: {', '.join(a.name for a in paper.authors)}\n"
            f"Summary: {paper.summary[:500]}"
        )
    return "\n\n---\n\n".join(results) if results else "No papers found."


# ============================================================
# Setup
# ============================================================
def setup_vectorstore():
    global _vectorstore
    if BM25_CACHE.exists():
        splits = pickle.load(open(BM25_CACHE, "rb"))
        print(f"  Loaded {len(splits)} cached splits", file=sys.stderr)
    else:
        docs = load_web(URL)
        splits = split_docs(docs)
        pickle.dump(splits, open(BM25_CACHE, "wb"))
        print(f"  Indexed {len(splits)} chunks", file=sys.stderr)
    _vectorstore = get_vectorstore(splits, name="07_agentic")


# ============================================================
# Build the agent
# ============================================================
def build_agent():
    setup_vectorstore()

    return create_agent(
        get_llm(),
        tools=[vector_search, web_search_tool, arxiv_search],
        system_prompt=(
            "You are a research assistant with three tools:\n"
            "1. vector_search: Lilian Weng's blog post on LLM agents (LLM agents, "
            "task decomposition, ReAct, planning, memory, tool use, chain-of-thought)\n"
            "2. web_search: public web via DuckDuckGo (current events, general "
            "knowledge, definitions, anything not in the local vector store)\n"
            "3. arxiv_search: arXiv academic papers (research questions in ML, AI, "
            "statistics, physics)\n\n"
            "Rules:\n"
            "- If you can answer directly from general knowledge without a tool, "
            "do so. Only use a tool when needed.\n"
            "- Never call the same tool twice with essentially the same query. "
            "If a tool returned nothing useful, try a different tool or answer "
            "directly.\n"
            "- Pick ONE tool per question unless you genuinely need multiple sources.\n"
            "- For LLM-agent topics (task decomposition, ReAct, planning, memory), "
            "ALWAYS prefer vector_search first.\n"
            "- Cite which tool(s) you used in your final answer."
        ),
    )


# ============================================================
# CLI entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Agentic RAG: agent picks tools dynamically (Groq-powered)"
    )
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show all tool calls and agent reasoning",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=15,
        help="Max agent steps (recursion limit, default 15)",
    )
    args = parser.parse_args()

    print("\n[setup] Building agent...", file=sys.stderr)
    agent = build_agent()

    print(f"\n[agent] Question: {args.question}", file=sys.stderr)
    print(f"[agent] Max steps: {args.max_steps}\n", file=sys.stderr)
    print("=" * 60)

    # Run the agent
    result = agent.invoke(
        {"messages": [HumanMessage(content=args.question)]},
        {"recursion_limit": args.max_steps},
    )

    if args.verbose:
        # Print every step
        for i, msg in enumerate(result["messages"]):
            msg_type = msg.type
            content = msg.content if hasattr(msg, "content") else str(msg)
            if msg_type == "human":
                print(f"\n[step {i}] USER:\n{content}")
            elif msg_type == "ai":
                tool_calls = getattr(msg, "tool_calls", [])
                if tool_calls:
                    print(f"\n[step {i}] AGENT (calling tools):")
                    for tc in tool_calls:
                        print(f"  → {tc['name']}({tc['args']})")
                else:
                    print(f"\n[step {i}] AGENT:\n{content}")
            elif msg_type == "tool":
                preview = content[:200].replace("\n", " ")
                print(f"  ← result: {preview}...")
    else:
        # Just the final answer
        print(result["messages"][-1].content)

    print("\n" + "=" * 60)
    n_tool_calls = sum(1 for m in result["messages"] if m.type == "tool")
    print(f"[trace] Steps: {len(result['messages'])}, tool calls: {n_tool_calls}", file=sys.stderr)


# ============================================================
# Note on tuning agent behavior with Llama 3.3
# ============================================================
#
# Llama 3.3 70B on Groq is solid at tool selection but can occasionally:
#   1. Call the same tool twice with slightly different queries (harmless but slow)
#   2. Pick the wrong tool for ambiguous questions (system_prompt helps)
#   3. Over-use tools when a direct answer would work
#
# To control these, edit the system_prompt above:
#   - Be very specific about WHEN to use each tool
#   - Add rules like "don't call the same tool twice"
#   - Tell it when NOT to use tools ("if you can answer directly, do so")
#
# To limit agent loops, pass recursion_limit at invoke time:
#   result = agent.invoke(
#       {"messages": [HumanMessage(content=q)]},
#       {"recursion_limit": 10},  # default is 25
#   )

if __name__ == "__main__":
    main()
