"""Offline tests for the logic in scripts 02-09 (no API keys needed)."""

import importlib

import networkx as nx
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from ragkit import utils
from ragkit.cli import SCRIPTS
from ragkit.pipeline import PIPELINES

from ..conftest import fake_retriever


def _load(name):
    return importlib.import_module(f"ragkit.scripts.{name}")


@pytest.mark.parametrize("module", SCRIPTS.values())
def test_every_script_imports_and_has_main_and_run(module):
    mod = importlib.import_module(module)
    assert callable(mod.main)
    assert callable(mod.run)


@pytest.mark.parametrize(("name", "spec"), PIPELINES.items())
def test_pipeline_registry_points_at_real_options(name, spec):
    module_name, kwargs = spec
    mod = _load(module_name)
    import inspect

    params = inspect.signature(mod.run).parameters
    assert set(kwargs) <= set(params), f"{name}: run() lacks {set(kwargs) - set(params)}"


# ---------- 02: query transformations ----------
def test_rag_fusion_keeps_original_query_and_fuses(fake_llm):
    mod = _load("02_query_transformations")
    fake_llm(mod, responses=["variant one\nvariant two"])
    seen = []

    def retrieve(q):
        seen.append(q)
        return [Document(page_content=f"doc for {q}"), Document(page_content="shared")]

    docs = mod.rag_fusion_method(RunnableLambda(retrieve), "original", top_n=2)
    assert seen[0] == "original"
    assert docs[0].page_content == "shared", "a doc found by every query should rank first"


def test_step_back_retrieves_for_both_questions(fake_llm):
    mod = _load("02_query_transformations")
    fake_llm(mod, responses=["general question"])
    seen = []
    retriever = RunnableLambda(lambda q: seen.append(q) or [Document(page_content=q)])
    mod.step_back_method(retriever, "specific question")
    assert sorted(seen) == ["general question", "specific question"]


# ---------- 03: routing ----------
def test_router_failure_falls_back_to_vectorstore(fake_llm):
    mod = _load("03_routing")

    def boom(_):
        raise ValueError("malformed tool call")

    fake_llm(mod, structured={"RouteQuery": boom})
    route, error = mod.route("q")
    assert route == mod.FALLBACK_ROUTE
    assert "malformed" in error


def test_routing_run_reports_the_route(monkeypatch, fake_llm):
    mod = _load("03_routing")
    fake_llm(mod, structured={"RouteQuery": mod.RouteQuery(datasource="web")})
    fake_llm(utils)
    monkeypatch.setattr(mod, "web_search", lambda q, max_results: "web text")
    result = mod.run("q")
    assert result.tool_calls == ["web"]
    assert result.contexts == ["web text"]


# ---------- 04: indexing ----------
def test_multi_rep_chunks_large_docs_and_caches_summaries(monkeypatch):
    mod = _load("04_indexing")
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


def test_strategy_index_chunks_only_once(monkeypatch):
    mod = _load("04_indexing")
    chunked = []
    real = mod.chunk_by_strategy

    def counting(docs, strategy, emb):
        chunked.append(strategy)
        return real(docs, strategy, emb)

    monkeypatch.setattr(mod, "chunk_by_strategy", counting)
    docs = [Document(page_content="alpha beta gamma. " * 200)]
    emb = DeterministicFakeEmbedding(size=16)
    mod.basic_strategy_index(docs, "recursive", emb)
    mod.basic_strategy_index(docs, "recursive", emb)
    assert chunked == ["recursive"]


def test_size_stats_handles_empty_retrieval():
    assert "no chunks" in _load("04_indexing").size_stats([])


# ---------- 07: agent ----------
def test_agent_tool_names_match_system_prompt():
    mod = _load("07_agentic_rag")
    names = [t.name for t in mod.make_tools(vectorstore=None)]
    assert tuple(names) == mod.TOOL_NAMES
    assert all(n in mod.SYSTEM_PROMPT for n in names)


def test_vector_search_tool_uses_injected_store():
    mod = _load("07_agentic_rag")

    class Store:
        def similarity_search(self, query, k):
            return [Document(page_content=f"hit for {query}")]

    vector_search = mod.make_tools(Store())[0]
    assert vector_search.invoke({"query": "react"}) == "hit for react"


def test_agent_step_limit_returns_partial_answer_instead_of_crashing():
    mod = _load("07_agentic_rag")
    from langgraph.errors import GraphRecursionError

    class LoopingAgent:
        def stream(self, inputs, config, stream_mode):
            msgs = inputs["messages"]
            yield {"messages": [*msgs, AIMessage(content="partial answer")]}
            raise GraphRecursionError("limit")

    result = mod.run("q", max_steps=3, agent=LoopingAgent())
    assert result.metadata["hit_step_limit"] is True
    assert result.answer == "partial answer"


def test_agent_run_collects_tool_calls():
    mod = _load("07_agentic_rag")
    from langchain_core.messages import ToolMessage

    class OneToolAgent:
        def stream(self, inputs, config, stream_mode):
            call = {"name": "vector_search", "args": {"query": "q"}, "id": "1"}
            yield {
                "messages": [
                    *inputs["messages"],
                    AIMessage(content="", tool_calls=[call]),
                    ToolMessage(content="ctx", tool_call_id="1"),
                    AIMessage(content="final"),
                ]
            }

    result = mod.run("q", agent=OneToolAgent())
    assert result.tool_calls == ["vector_search"]
    assert result.contexts == ["ctx"]
    assert result.answer == "final"
    assert result.metadata["hit_step_limit"] is False


# ---------- 08: CRAG ----------
@pytest.fixture
def crag():
    return _load("08_corrective_rag")


@pytest.mark.parametrize(
    ("scores", "decision"),
    [
        ([0.9, 0.1, 0.1], "CORRECT"),  # one strong doc is enough (paper rule)
        ([0.1, 0.2], "INCORRECT"),
        ([0.5, 0.2], "AMBIGUOUS"),
        ([0.31, 0.0], "AMBIGUOUS"),  # not *all* below the lower threshold
        ([], "INCORRECT"),
    ],
)
def test_paper_decision_rule(crag, scores, decision):
    assert crag.t_correct_decision(scores)[0] == decision


def test_avg_rule_differs_from_paper_rule(crag):
    scores = [0.9, 0.1, 0.1]
    assert crag.t_correct_decision(scores, rule="paper")[0] == "CORRECT"
    assert crag.t_correct_decision(scores, rule="avg")[0] == "AMBIGUOUS"


def test_decision_clamps_out_of_range_scores(crag):
    assert crag.t_correct_decision([1.5, -1.0]) == ("CORRECT", 1.0)
    assert crag.t_correct_decision([1.5, -1.0], rule="avg") == ("AMBIGUOUS", 0.5)


def test_grader_errors_count_as_irrelevant(crag):
    def grade(x):
        if "bad" in x["document"]:
            raise ValueError("no JSON")
        return crag.DocumentGrade(score=0.8, reasoning="ok")

    docs = [Document(page_content="good"), Document(page_content="bad")]
    grades = crag.grade_documents(RunnableLambda(grade), "q", docs)
    assert [g[0] for g in grades] == [0.8, 0.0]


def test_decompose_splits_into_short_strips(crag):
    strips = crag.decompose("One. Two. Three. Four. Five.", sentences_per_strip=2)
    assert strips == ["One. Two.", "Three. Four.", "Five."]


def test_refinement_keeps_only_selected_strips_in_order(crag):
    selector = RunnableLambda(lambda _: crag.StripSelection(relevant=[2, 0, 99]))
    out = crag.refine_knowledge(selector, "q", ["A. B. C. D. E. F."])
    assert out == "A. B.\nE. F."


def test_crag_correct_path_uses_refined_internal_docs(monkeypatch, crag, fake_llm):
    fake_llm(
        crag,
        structured={
            "DocumentGrade": crag.DocumentGrade(score=0.9, reasoning="relevant"),
            "StripSelection": crag.StripSelection(relevant=[0]),
        },
    )
    fake_llm(utils)
    monkeypatch.setattr(crag, "web_search", lambda *a, **k: pytest.fail("no web on CORRECT"))
    result = crag.run("q", retriever=fake_retriever(["Keep this. Too. Drop this. Also."]))
    assert result.metadata["decision"] == "CORRECT"
    assert result.contexts == ["Keep this. Too."]
    assert result.tool_calls == []


def test_crag_incorrect_path_searches_the_web(monkeypatch, crag, fake_llm):
    fake_llm(
        crag,
        responses=["rewritten keywords"],
        structured={"DocumentGrade": crag.DocumentGrade(score=0.0, reasoning="no")},
    )
    fake_llm(utils)
    queries = []
    monkeypatch.setattr(crag, "web_search", lambda q, max_results: queries.append(q) or "web")
    result = crag.run("q", refine=False, retriever=fake_retriever())
    assert result.metadata["decision"] == "INCORRECT"
    assert queries == ["rewritten keywords"]
    assert result.contexts == ["web"]


# ---------- 09: graph RAG ----------
def test_graph_matching_ignores_stopwords():
    mod = _load("09_graph_rag")
    g = nx.MultiDiGraph()
    g.add_edge("ReAct", "Chain-of-Thought", relation="extends")
    g.add_node("Somewhat Unrelated Tool")

    subgraph, matched = mod.find_relevant_subgraph(g, "What does ReAct do?")
    assert matched == ["ReAct"]
    assert set(subgraph.nodes) == {"ReAct", "Chain-of-Thought"}


def test_k_hop_follows_incoming_edges():
    mod = _load("09_graph_rag")
    g = nx.MultiDiGraph()
    g.add_edge("Reflexion", "ReAct", relation="builds on")
    g.add_edge("ReAct", "HotpotQA", relation="evaluated on")

    subgraph, _ = mod.find_relevant_subgraph(g, "Tell me about HotpotQA", k_hops=2)
    assert set(subgraph.nodes) == {"HotpotQA", "ReAct", "Reflexion"}


def test_extraction_merges_case_variants_and_keeps_dangling_edges():
    mod = _load("09_graph_rag")
    g = nx.MultiDiGraph(chunks=["c0", "c1"])
    mod.add_extraction(
        g,
        mod.ExtractionResult(
            entities=[mod.Entity(name="ReAct", type="METHOD")],
            relationships=[mod.Relationship(source="react ", target="CoT", relation="uses")],
        ),
        chunk_id=0,
    )
    mod.add_extraction(
        g,
        mod.ExtractionResult(entities=[mod.Entity(name="REACT", type="METHOD")], relationships=[]),
        1,
    )
    assert set(g.nodes) == {"react", "cot"}
    assert g.nodes["react"]["name"] == "ReAct"
    assert g.nodes["react"]["chunks"] == {0, 1}
    assert g.number_of_edges() == 1


def test_graph_run_includes_source_chunks(fake_llm):
    mod = _load("09_graph_rag")
    fake_llm(utils)
    g = nx.MultiDiGraph(chunks=["ReAct paper text", "unrelated text"])
    mod.add_extraction(
        g,
        mod.ExtractionResult(entities=[mod.Entity(name="ReAct", type="METHOD")], relationships=[]),
        0,
    )
    result = mod.run("What is ReAct?", graph=g)
    assert "ReAct paper text" in result.contexts
    assert result.metadata["matched_entities"] == ["ReAct"]
