"""Offline tests for script 01 and the shared corpus cache."""

import importlib

from langchain_core.documents import Document

from ragkit import corpus

from ..conftest import fake_retriever

mod = importlib.import_module("ragkit.scripts.01_basic_rag")


def test_run_returns_answer_and_contexts(monkeypatch, fake_llm):
    from ragkit import utils

    fake_llm(utils)
    monkeypatch.setattr(mod, "default_retriever", lambda k: fake_retriever(["c1", "c2"]))
    result = mod.run("q")
    assert result.answer == "fake answer"
    assert result.contexts == ["c1", "c2"]


def test_corpus_splits_are_cached_and_numbered(monkeypatch):
    calls = []

    def fake_load(url):
        calls.append(url)
        return [Document(page_content="word " * 1000)]

    monkeypatch.setattr(corpus.utils, "load_web", fake_load)
    first = corpus.load_splits()
    second = corpus.load_splits()
    assert len(calls) == 1, "the web page should be downloaded once"
    assert [s.metadata["chunk_id"] for s in first] == list(range(len(first)))
    assert [s.page_content for s in second] == [s.page_content for s in first]
