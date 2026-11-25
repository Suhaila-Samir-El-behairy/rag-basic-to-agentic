"""A common interface over all scripts, so they can be benchmarked side by side.

Every script exposes ``run(question, **opts) -> RAGResult``. ``PIPELINES`` names
each (script, options) combination, e.g. ``"05:hybrid"``; ``run_pipeline`` adds
timing and LLM-call/token accounting.
"""

import importlib
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from ragkit.tracking import track_usage


@dataclass
class RAGResult:
    answer: str
    contexts: list[str] = field(default_factory=list)
    # Tools the agent called, or the route the router chose (script 03).
    tool_calls: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    llm_calls: int = 0
    total_tokens: int = 0
    latency_s: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# name -> (script module, run() kwargs)
PIPELINES: dict[str, tuple[str, dict[str, Any]]] = {
    "01:basic": ("01_basic_rag", {}),
    "02:multi-query": ("02_query_transformations", {"method": "multi-query"}),
    "02:rag-fusion": ("02_query_transformations", {"method": "rag-fusion"}),
    "02:hyde": ("02_query_transformations", {"method": "hyde"}),
    "02:step-back": ("02_query_transformations", {"method": "step-back"}),
    "03:routing": ("03_routing", {}),
    "04:recursive": ("04_indexing", {"strategy": "recursive"}),
    "04:token": ("04_indexing", {"strategy": "token"}),
    "04:semantic": ("04_indexing", {"strategy": "semantic"}),
    "04:multi-rep": ("04_indexing", {"strategy": "multi-rep"}),
    "05:rerank": ("05_retrieval", {"method": "rerank"}),
    "05:hybrid": ("05_retrieval", {"method": "hybrid"}),
    "06:self-reflection": ("06_self_reflection", {}),
    "07:agent": ("07_agentic_rag", {}),
    "08:crag": ("08_corrective_rag", {}),
    "08:crag-avg-rule": ("08_corrective_rag", {"rule": "avg"}),
    "09:graph": ("09_graph_rag", {}),
}

# A cheap default set that covers every family of techniques once.
DEFAULT_SUITE = [
    "01:basic",
    "02:rag-fusion",
    "02:hyde",
    "03:routing",
    "05:rerank",
    "05:hybrid",
    "06:self-reflection",
    "07:agent",
    "08:crag",
    "09:graph",
]


def get_pipeline(name: str) -> Callable[[str], RAGResult]:
    if name not in PIPELINES:
        raise KeyError(f"Unknown pipeline {name!r}. Choose from: {', '.join(PIPELINES)}")
    module_name, kwargs = PIPELINES[name]
    module = importlib.import_module(f"ragkit.scripts.{module_name}")
    return lambda question: module.run(question, **kwargs)


def run_pipeline(fn: Callable[[str], RAGResult], question: str) -> RAGResult:
    """Run one question, recording latency and LLM usage; errors become results."""
    start = time.perf_counter()
    with track_usage() as usage:
        try:
            result = fn(question)
        except Exception as e:  # noqa: BLE001 - a benchmark must survive one bad question
            result = RAGResult(answer="", error=f"{type(e).__name__}: {e}")
    result.latency_s = round(time.perf_counter() - start, 3)
    result.llm_calls = usage.llm_calls
    result.total_tokens = usage.total_tokens
    return result
