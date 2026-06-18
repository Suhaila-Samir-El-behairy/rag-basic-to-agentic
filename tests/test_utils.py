"""Tests for utils module."""


from langchain_core.documents import Document

from src.ragkit.utils import format_docs, split_docs


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
