"""Offline tests for script 06: the self-reflection graph must always terminate."""

import importlib

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

mod = importlib.import_module("ragkit.scripts.06_self_reflection")


class _FakeVectorStore:
    def as_retriever(self, **_):
        return RunnableLambda(lambda _: [Document(page_content="some chunk")])


def _build_app(monkeypatch, tmp_path, doc="yes", grounded="yes", answers="yes"):
    """Compile the graph with a fake LLM whose graders return fixed scores."""
    scores = {
        "GradeDocuments": mod.GradeDocuments(binary_score=doc),
        "GradeHallucinations": mod.GradeHallucinations(binary_score=grounded),
        "GradeAnswer": mod.GradeAnswer(binary_score=answers),
    }

    class FakeLLM(FakeListChatModel):
        def with_structured_output(self, schema, **_):
            return RunnableLambda(lambda _: scores[schema.__name__])

    monkeypatch.setattr(mod, "get_llm", lambda **_: FakeLLM(responses=["fake"]))
    monkeypatch.setattr(mod, "get_vectorstore", lambda *_, **__: _FakeVectorStore())
    monkeypatch.setattr(mod, "load_web", lambda _: [])
    monkeypatch.setattr(mod, "BM25_CACHE", tmp_path / "cache.pkl")
    return mod.build_graph()


def _run(app):
    # A low recursion limit turns any unbounded loop into a test failure.
    return app.invoke({"question": "q", "retries": 0, "generations": 0}, {"recursion_limit": 20})


def test_happy_path_generates_once(monkeypatch, tmp_path):
    result = _run(_build_app(monkeypatch, tmp_path))
    assert result["generation"] == "fake"
    assert result["generations"] == 1
    assert result["retries"] == 0


def test_irrelevant_docs_stop_after_max_retries(monkeypatch, tmp_path):
    result = _run(_build_app(monkeypatch, tmp_path, doc="no", answers="no"))
    assert result["retries"] == mod.MAX_RETRIES
    assert result["generation"] == "fake"


def test_hallucinations_stop_after_max_regenerations(monkeypatch, tmp_path):
    result = _run(_build_app(monkeypatch, tmp_path, grounded="no"))
    assert result["generations"] == mod.MAX_RETRIES + 1


@pytest.mark.parametrize("raw", ["yes", "Yes", " YES.", "yes, relevant"])
def test_is_yes_tolerates_formatting(raw):
    assert mod.is_yes(mod.GradeDocuments(binary_score=raw))


@pytest.mark.parametrize("raw", ["no", "No", "", "maybe"])
def test_is_yes_rejects_non_yes(raw):
    assert not mod.is_yes(mod.GradeDocuments(binary_score=raw))
