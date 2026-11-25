"""The shared corpus every script retrieves from, with on-disk caching.

All scripts used to re-download the blog post and re-split it on every run, each
with its own copy of the caching code. This module does it once:

  - ``load_docs()``         raw documents (cached as a pickle next to the vector stores)
  - ``load_splits()``       chunked documents for given chunk params (cached)
  - ``default_vectorstore()`` the Chroma index over the default splits, shared by the
                            scripts that don't study indexing themselves (01-03, 05-08)
"""

import logging
import pickle
from functools import lru_cache

from langchain_core.documents import Document

from ragkit import config, utils

logger = logging.getLogger(__name__)

CORPUS_URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"
CORPUS_NAME = "lilianweng_agents"
CORPUS_DESCRIPTION = (
    "Lilian Weng's blog post 'LLM Powered Autonomous Agents' (planning, task "
    "decomposition, chain-of-thought, tree of thoughts, ReAct, Reflexion, memory, "
    "maximum inner product search, tool use, MRKL, Toolformer, HuggingGPT, case "
    "studies such as ChemCrow, Generative Agents and AutoGPT)"
)


def _cache_path(stem: str):
    return config.CHROMA_DIR / f"{CORPUS_NAME}_{stem}.pkl"


def _cached(stem: str, build):
    path = _cache_path(stem)
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)  # noqa: S301 - local cache written by this module
    value = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(value, f)
    return value


def load_docs() -> list[Document]:
    """The raw corpus documents (downloaded once, then cached)."""
    return _cached("docs", lambda: utils.load_web(CORPUS_URL))


def load_splits(
    chunk_size: int = config.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Chunked corpus; each chunk gets a stable ``chunk_id`` in its metadata."""

    def build():
        splits = utils.split_docs(load_docs(), chunk_size, chunk_overlap)
        for i, s in enumerate(splits):
            s.metadata["chunk_id"] = i
        return splits

    return _cached(f"splits_{chunk_size}_{chunk_overlap}", build)


@lru_cache(maxsize=1)
def default_vectorstore():
    """Chroma index over the default splits (built once, shared across scripts)."""
    return utils.get_vectorstore(load_splits(), name=f"{CORPUS_NAME}_default")


def default_retriever(k: int = config.DEFAULT_K):
    return default_vectorstore().as_retriever(search_kwargs={"k": k})
