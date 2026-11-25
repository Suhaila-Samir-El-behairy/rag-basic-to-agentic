"""Agentic RAG: a ReAct-style tool-calling agent that picks tools dynamically.

Tools:
  - vector_search: search the local vector store
  - web_search:    search the public web (DuckDuckGo, or Tavily if a key is set)
  - arxiv_search:  search arXiv academic papers

The agent decides which tool(s) to use based on the question. If it runs out of
steps (``--max-steps``), the last answer it produced is returned and the result
is flagged, instead of crashing with a GraphRecursionError.
"""

import argparse
import importlib
import sys
from functools import lru_cache

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from ragkit.corpus import default_vectorstore
from ragkit.pipeline import RAGResult
from ragkit.utils import format_docs, get_llm, web_search

TOOL_NAMES = ("vector_search", "web_search", "arxiv_search")
DEFAULT_MAX_STEPS = 15

SYSTEM_PROMPT = (
    "You are a research assistant with three tools:\n"
    "1. vector_search: Lilian Weng's blog post on LLM agents (LLM agents, "
    "task decomposition, ReAct, planning, memory, tool use, chain-of-thought)\n"
    "2. web_search: the public web (current events, general knowledge, "
    "definitions, anything not in the local vector store)\n"
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
)


# ============================================================
# Tools (built from a factory so the vector store is injected, not global)
# ============================================================
def make_tools(vectorstore):
    @tool
    def vector_search(query: str) -> str:
        """Search the local vector store of Lilian Weng's blog post on LLM agents.
        Use for questions about LLM agents, task decomposition, ReAct, planning,
        memory, tool use, or chain-of-thought."""
        docs = vectorstore.similarity_search(query, k=4)
        if not docs:
            return "No relevant documents found in the vector store."
        return format_docs(docs)

    @tool("web_search")
    def web_search_tool(query: str) -> str:
        """Search the public web for current information or general knowledge.
        Use for questions about current events, general topics, definitions,
        or anything not in the local vector store."""
        return web_search(query, max_results=3)

    @tool
    def arxiv_search(query: str) -> str:
        """Search arXiv for academic papers on a topic.
        Use for research/academic questions in ML, AI, statistics, or physics."""
        routing = importlib.import_module("ragkit.scripts.03_routing")
        return routing.search_arxiv(query, max_results=3, summary_chars=500)

    return [vector_search, web_search_tool, arxiv_search]


def build_agent(vectorstore=None, llm=None):
    tools = make_tools(vectorstore if vectorstore is not None else default_vectorstore())
    return create_agent(llm or get_llm(), tools=tools, system_prompt=SYSTEM_PROMPT)


@lru_cache(maxsize=1)
def _agent():
    return build_agent()


# ============================================================
# Running with a step budget
# ============================================================
def invoke_with_budget(agent, question: str, max_steps: int) -> tuple[list, bool]:
    """Run the agent, keeping the latest state so a step-limit hit isn't fatal.

    Returns (messages, hit_step_limit).
    """
    messages: list = [HumanMessage(content=question)]
    try:
        for state in agent.stream(
            {"messages": messages}, {"recursion_limit": max_steps}, stream_mode="values"
        ):
            messages = state["messages"]
    except GraphRecursionError:
        return messages, True
    return messages, False


def final_answer(messages) -> str:
    """The last AI message that is an answer (not just a tool call)."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def run(question: str, max_steps: int = DEFAULT_MAX_STEPS, agent=None) -> RAGResult:
    messages, hit_limit = invoke_with_budget(agent or _agent(), question, max_steps)
    tool_calls = [tc["name"] for m in messages if isinstance(m, AIMessage) for tc in m.tool_calls]
    contexts = [str(m.content) for m in messages if m.type == "tool"]
    return RAGResult(
        answer=final_answer(messages),
        contexts=contexts,
        tool_calls=tool_calls,
        metadata={"hit_step_limit": hit_limit, "messages": len(messages)},
    )


# ============================================================
# CLI entry point
# ============================================================
def print_trace(messages):
    for i, msg in enumerate(messages):
        content = msg.content if hasattr(msg, "content") else str(msg)
        if msg.type == "human":
            print(f"\n[step {i}] USER:\n{content}")
        elif msg.type == "ai":
            if msg.tool_calls:
                print(f"\n[step {i}] AGENT (calling tools):")
                for tc in msg.tool_calls:
                    print(f"  → {tc['name']}({tc['args']})")
            else:
                print(f"\n[step {i}] AGENT:\n{content}")
        elif msg.type == "tool":
            preview = str(content)[:200].replace("\n", " ")
            print(f"  ← result: {preview}...")


def main():
    parser = argparse.ArgumentParser(description="Agentic RAG: agent picks tools dynamically")
    parser.add_argument("--question", required=True)
    parser.add_argument("--verbose", action="store_true", help="Show all tool calls")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Max agent steps (recursion limit, default {DEFAULT_MAX_STEPS})",
    )
    args = parser.parse_args()

    print(f"\n[agent] Question: {args.question}", file=sys.stderr)
    messages, hit_limit = invoke_with_budget(_agent(), args.question, args.max_steps)
    print("=" * 60)
    if args.verbose:
        print_trace(messages)
    else:
        print(final_answer(messages) or "(no final answer)")
    print("\n" + "=" * 60)

    if hit_limit:
        print(f"[agent] Stopped at the step limit ({args.max_steps}).", file=sys.stderr)
    n_tool_calls = sum(1 for m in messages if m.type == "tool")
    print(f"[trace] Messages: {len(messages)}, tool calls: {n_tool_calls}", file=sys.stderr)


if __name__ == "__main__":
    main()
