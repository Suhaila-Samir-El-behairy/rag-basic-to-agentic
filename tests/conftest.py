"""Shared fixtures: an isolated cache dir and fakes for LLMs and retrieval."""

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from ragkit import config, corpus


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    """Every test gets its own cache dir; module-level caches start empty."""
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    corpus.default_vectorstore.cache_clear()
    yield tmp_path / "chroma"
    corpus.default_vectorstore.cache_clear()


class StructuredFakeLLM(FakeListChatModel):
    """Fake chat model whose with_structured_output returns canned objects per schema.

    ``structured`` maps a schema class name to either an instance or a callable
    taking the prompt input and returning one.
    """

    structured: dict = {}

    def with_structured_output(self, schema, **_):
        value = self.structured[schema.__name__]
        return RunnableLambda(lambda x: value(x) if callable(value) else value)


@pytest.fixture
def fake_llm(monkeypatch):
    """Factory: patch ``module.get_llm`` to return a StructuredFakeLLM."""

    def install(module, responses=("fake answer",), structured=None):
        llm = StructuredFakeLLM(responses=list(responses), structured=structured or {})
        monkeypatch.setattr(module, "get_llm", lambda **_: llm)
        return llm

    return install


def fake_retriever(texts=("some chunk",)):
    return RunnableLambda(lambda _: [Document(page_content=t) for t in texts])
