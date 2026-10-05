"""Tests for utils module."""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from ragkit import utils
from ragkit.utils import format_docs, get_vectorstore, split_docs


def test_format_docs_joins_content():
    docs = [
        Document(page_content="Hello", metadata={}),
        Document(page_content="World", metadata={}),
    ]
    assert format_docs(docs) == "Hello\n\nWorld"


def test_format_docs_empty_list():
    assert format_docs([]) == ""


def test_split_docs_creates_chunks():
    docs = [Document(page_content="A " * 1000, metadata={})]
    splits = split_docs(docs, chunk_size=100, chunk_overlap=10)
    assert len(splits) > 1
    assert all(isinstance(s, Document) for s in splits)


@pytest.fixture
def chroma_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(utils, "CHROMA_DIR", tmp_path)
    return tmp_path


def test_get_vectorstore_reuses_existing_store(chroma_dir):
    emb = DeterministicFakeEmbedding(size=16)
    get_vectorstore([Document(page_content="a"), Document(page_content="b")], "s", emb)
    reloaded = get_vectorstore([Document(page_content="ignored")], "s", emb)
    assert sorted(reloaded.get()["documents"]) == ["a", "b"]


def test_get_vectorstore_rebuilds_after_failed_build(chroma_dir):
    class FailingEmbedding(DeterministicFakeEmbedding):
        def embed_documents(self, texts):
            raise RuntimeError("rate limited")

    with pytest.raises(RuntimeError):
        get_vectorstore([Document(page_content="a")], "s", FailingEmbedding(size=16))
    assert (chroma_dir / "s" / "chroma.sqlite3").exists()  # empty store left behind

    store = get_vectorstore([Document(page_content="a")], "s", DeterministicFakeEmbedding(size=16))
    assert store.get()["documents"] == ["a"]


def test_get_llm_raises_without_groq_key(monkeypatch):
    # utils imports the key by value, so patch it where it is read
    monkeypatch.setattr(utils, "GROQ_API_KEY", None)
    monkeypatch.setattr(utils, "LLM_PROVIDER", "groq")
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        utils.get_llm()


def test_get_llm_rejects_unknown_provider(monkeypatch):
    monkeypatch.setattr(utils, "LLM_PROVIDER", "nope")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        utils.get_llm()
