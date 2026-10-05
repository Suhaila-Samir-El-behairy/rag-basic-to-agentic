"""Offline tests for the pure logic in scripts 04, 07, 08 and 09 (no API keys needed)."""

import importlib

import networkx as nx
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ragkit import utils
from ragkit.cli import SCRIPTS


@pytest.mark.parametrize("module", SCRIPTS.values())
def test_every_script_imports_and_has_main(module):
    assert callable(importlib.import_module(module).main)


# ---------- 04: multi-representation indexing ----------
def test_multi_rep_chunks_large_docs_and_caches_summaries(monkeypatch, tmp_path):
    mod = importlib.import_module("ragkit.scripts.04_indexing")
    monkeypatch.setattr(utils, "CHROMA_DIR", tmp_path)
    calls = []

    class CountingLLM(FakeListChatModel):
        def _call(self, *args, **kwargs):
            calls.append(1)
            return "summary"

    monkeypatch.setattr(mod, "get_llm", lambda **_: CountingLLM(responses=["unused"]))
    one_big_doc = [Document(page_content=("word " * 2000 + "\n\n") * 6)]
    emb = DeterministicFakeEmbedding(size=16)

    full_docs = mod.multi_representation_index(one_big_doc, "q", emb)
    assert len(calls) > 1, "the single web doc should be split before summarizing"
    assert all(len(d.page_content) <= 4000 for d in full_docs)

    calls.clear()
    mod.multi_representation_index(one_big_doc, "q", emb)
    assert calls == [], "summaries should be reused from the persisted store"


# ---------- 07: tool names must match the system prompt ----------
def test_agent_tool_names_match_system_prompt():
    mod = importlib.import_module("ragkit.scripts.07_agentic_rag")
    names = [t.name for t in (mod.vector_search, mod.web_search_tool, mod.arxiv_search)]
    assert names == ["vector_search", "web_search", "arxiv_search"]


# ---------- 08: T-correct decision rule ----------
@pytest.fixture
def crag():
    return importlib.import_module("ragkit.scripts.08_corrective_rag")


@pytest.mark.parametrize(
    ("scores", "decision"),
    [([0.9, 0.8], "CORRECT"), ([0.1, 0.2], "INCORRECT"), ([0.5], "AMBIGUOUS"), ([], "INCORRECT")],
)
def test_t_correct_decision(crag, scores, decision):
    assert crag.t_correct_decision(scores)[0] == decision


def test_t_correct_clamps_out_of_range_scores(crag):
    assert crag.t_correct_decision([1.5, -1.0]) == ("AMBIGUOUS", 0.5)


# ---------- 09: subgraph matching ----------
def test_graph_matching_ignores_stopwords():
    mod = importlib.import_module("ragkit.scripts.09_graph_rag")
    g = nx.MultiDiGraph()
    g.add_edge("ReAct", "Chain-of-Thought", relation="extends")
    g.add_node("Somewhat Unrelated Tool")

    subgraph, matched = mod.find_relevant_subgraph(g, "What does ReAct do?")
    assert matched == ["ReAct"]
    assert set(subgraph.nodes) == {"ReAct", "Chain-of-Thought"}
