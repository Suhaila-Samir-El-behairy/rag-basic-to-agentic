"""Offline tests for script 06: termination, original question, strict retry."""

import importlib

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from ..conftest import fake_retriever

mod = importlib.import_module("ragkit.scripts.06_self_reflection")


def _build_app(monkeypatch, doc="yes", grounded="yes", answers="yes", log=None):
    """Compile the graph with a fake LLM whose graders return fixed scores.

    ``log`` collects (prompt kind, prompt input) so tests can inspect what each
    chain was asked.
    """
    log = log if log is not None else []
    scores = {
        "GradeDocuments": mod.GradeDocuments(binary_score=doc),
        "GradeHallucinations": mod.GradeHallucinations(binary_score=grounded),
        "GradeAnswer": mod.GradeAnswer(binary_score=answers),
    }

    class FakeLLM(FakeListChatModel):
        temperature: float = 0

        def with_structured_output(self, schema, **_):
            def grade(x):
                log.append((schema.__name__, x))
                return scores[schema.__name__]

            return RunnableLambda(grade)

    def get_llm(temperature=0, **_):
        return FakeLLM(responses=[f"answer@{temperature}"], temperature=temperature)

    monkeypatch.setattr(mod, "get_llm", get_llm)
    return mod.build_graph(retriever=fake_retriever())


def _run(app, question="q"):
    # A low recursion limit turns any unbounded loop into a test failure.
    return app.invoke(mod.initial_state(question), {"recursion_limit": 40})


def test_happy_path_generates_once(monkeypatch):
    result = _run(_build_app(monkeypatch))
    assert result["generation"] == "answer@0"
    assert result["generations"] == 1
    assert result["retries"] == 0


def test_irrelevant_docs_stop_after_max_retries(monkeypatch):
    result = _run(_build_app(monkeypatch, doc="no", answers="no"))
    assert result["retries"] == mod.MAX_RETRIES
    assert result["generation"].startswith("answer@")


def test_hallucinations_stop_after_max_regenerations(monkeypatch):
    result = _run(_build_app(monkeypatch, grounded="no"))
    assert result["generations"] == mod.MAX_RETRIES + 1


def test_regeneration_after_hallucination_uses_retry_temperature(monkeypatch):
    result = _run(_build_app(monkeypatch, grounded="no"))
    assert result["generation"] == f"answer@{mod.RETRY_TEMPERATURE}"


def test_rewrites_never_replace_the_users_question(monkeypatch):
    log = []
    result = _run(_build_app(monkeypatch, doc="no", answers="no", log=log), question="original?")
    assert result["original_question"] == "original?"
    assert result["question"] != "original?", "the search query should have been rewritten"
    prompts = [x.to_string() for kind, x in log if kind in ("GradeDocuments", "GradeAnswer")]
    assert prompts
    assert all("Question: original?" in p for p in prompts)


@pytest.mark.parametrize("raw", ["yes", "Yes", " YES.", "yes, relevant"])
def test_is_yes_tolerates_formatting(raw):
    assert mod.is_yes(mod.GradeDocuments(binary_score=raw))


@pytest.mark.parametrize("raw", ["no", "No", "", "maybe"])
def test_is_yes_rejects_non_yes(raw):
    assert not mod.is_yes(mod.GradeDocuments(binary_score=raw))
