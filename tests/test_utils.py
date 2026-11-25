"""Tests for utils module."""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from ragkit import utils
from ragkit.utils import format_docs, get_vectorstore, reciprocal_rank_fusion, split_docs


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
def chroma_dir(isolated_cache):
    return isolated_cache


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


def _d(text):
    return Document(page_content=text)


def test_rrf_rewards_docs_ranked_high_in_many_lists():
    fused = reciprocal_rank_fusion([[_d("a"), _d("b")], [_d("b"), _d("c")], [_d("b")]])
    assert [d.page_content for d in fused] == ["b", "a", "c"]


def test_rrf_handles_unhashable_metadata():
    doc = Document(page_content="x", metadata={"tags": ["a", "b"]})
    assert reciprocal_rank_fusion([[doc], [doc]]) == [doc]


def test_generate_answer_accepts_docs_or_string():
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    llm = FakeListChatModel(responses=["ok", "ok"])
    assert utils.generate_answer("q", [_d("ctx")], llm=llm) == "ok"
    assert utils.generate_answer("q", "ctx", llm=llm) == "ok"


def test_retrying_embeddings_retries_transient_errors():
    class Flaky(DeterministicFakeEmbedding):
        calls: int = 0

        def embed_query(self, text):
            self.calls += 1
            if self.calls < 2:
                raise RuntimeError("429")
            return super().embed_query(text)

    inner = Flaky(size=4)
    from tenacity import wait_none

    wrapped = utils.RetryingEmbeddings(inner, attempts=3, wait=wait_none())
    assert len(wrapped.embed_query("x")) == 4
    assert inner.calls == 2


def test_rate_limit_is_attached_to_new_llms(monkeypatch):
    monkeypatch.setattr(utils, "GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(utils, "LLM_PROVIDER", "groq")
    utils.set_rate_limit(60)
    try:
        assert utils.get_llm().rate_limiter is not None
    finally:
        utils.set_rate_limit(None)
    assert utils.get_llm().rate_limiter is None
