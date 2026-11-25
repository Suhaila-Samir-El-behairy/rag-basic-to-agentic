"""Shared utilities — LLM/embedding factories, loaders, vector store, fusion, answering."""

import logging
from functools import lru_cache

# Import config first: it sets USER_AGENT before langchain_community's web loader reads it.
from ragkit import config  # isort: skip
from ragkit.config import (  # isort: skip
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    EMBED_MODEL,
    GOOGLE_API_KEY,
    GROQ_API_KEY,
    HUGGINGFACEHUB_API_TOKEN,
    LLM_PROVIDER,
    TAVILY_API_KEY,
    get_default_llm_model,
)

from langchain_chroma import Chroma  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402
from pydantic import SecretStr  # noqa: E402
from tenacity import retry, stop_after_attempt, wait_exponential  # noqa: E402

logger = logging.getLogger(__name__)

# Shared across every LLM created by get_llm(), so a whole benchmark run stays under
# the provider's free-tier requests-per-minute limit. Off unless set_rate_limit() is called.
_rate_limiter = None


def set_rate_limit(requests_per_minute: float | None) -> None:
    """Throttle all LLMs created after this call (None disables throttling)."""
    global _rate_limiter
    if not requests_per_minute:
        _rate_limiter = None
        return
    from langchain_core.rate_limiters import InMemoryRateLimiter

    _rate_limiter = InMemoryRateLimiter(
        requests_per_second=requests_per_minute / 60, check_every_n_seconds=0.1, max_bucket_size=1
    )


def get_llm(temperature: float = 0, model: str | None = None, **kwargs):
    """Free LLM via Groq (default) or Gemini."""
    chosen_model = model or get_default_llm_model()
    if _rate_limiter is not None:
        kwargs.setdefault("rate_limiter", _rate_limiter)

    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY missing. Get a free key at https://console.groq.com")
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=chosen_model,
            temperature=temperature,
            api_key=SecretStr(GROQ_API_KEY),
            max_retries=2,
            **kwargs,
        )

    if LLM_PROVIDER == "gemini":
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY missing. Get one free at https://aistudio.google.com")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=chosen_model,
            temperature=temperature,
            google_api_key=GOOGLE_API_KEY,
            **kwargs,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}. Use 'groq' or 'gemini'.")


class RetryingEmbeddings(Embeddings):
    """Wrap an embedding model with exponential backoff.

    The free HF Inference API rate-limits bursts (semantic chunking embeds every
    sentence), so transient failures are retried instead of aborting the build.
    """

    def __init__(self, inner: Embeddings, attempts: int = 4, wait=None):
        self.inner = inner
        self._retry = retry(
            stop=stop_after_attempt(attempts),
            wait=wait if wait is not None else wait_exponential(multiplier=2, max=30),
            reraise=True,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._retry(self.inner.embed_documents)(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._retry(self.inner.embed_query)(text)


@lru_cache(maxsize=4)
def get_embeddings(model: str = EMBED_MODEL) -> Embeddings:
    """Free embeddings via HuggingFace Inference API (with retry/backoff)."""
    if not HUGGINGFACEHUB_API_TOKEN:
        raise ValueError(
            "HUGGINGFACEHUB_API_TOKEN missing. Get a free token at "
            "https://huggingface.co/settings/tokens"
        )
    from langchain_huggingface import HuggingFaceEndpointEmbeddings

    logger.info("Loading embedding model: %s", model)
    return RetryingEmbeddings(
        HuggingFaceEndpointEmbeddings(
            model=model,
            task="feature-extraction",
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        )
    )


def load_web(url: str) -> list[Document]:
    from langchain_community.document_loaders import WebBaseLoader

    return WebBaseLoader(url).load()


def load_pdf(path: str) -> list[Document]:
    from langchain_community.document_loaders import PyPDFLoader

    return PyPDFLoader(path).load()


def split_docs(
    docs, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> list[Document]:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    ).split_documents(docs)


def get_vectorstore(splits, name: str, embeddings=None) -> Chroma:
    """Build or load a persistent Chroma vectorstore."""
    persist_dir = config.CHROMA_DIR / name
    persist_dir.mkdir(parents=True, exist_ok=True)
    embeddings = embeddings or get_embeddings()

    vectorstore = Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)
    # A previous run can leave an empty store behind if embedding failed midway
    # (e.g. HF rate limit), so check for content rather than for the file.
    if vectorstore.get(limit=1)["ids"]:
        logger.info("Loading existing vector store: %s", name)
        return vectorstore

    if splits:
        logger.info("Building new vector store: %s (%d splits)", name, len(splits))
        vectorstore.add_documents(splits)
    return vectorstore


def web_search(query: str, max_results: int = 3) -> str:
    """Free web search: Tavily if key set, else DuckDuckGo."""
    if TAVILY_API_KEY:
        from tavily import TavilyClient

        results = TavilyClient(api_key=TAVILY_API_KEY).search(query, max_results=max_results)
        return "\n\n".join(r["content"] for r in results["results"])

    from ddgs import DDGS

    with DDGS() as ddgs:
        hits = list(ddgs.text(query, max_results=max_results))
    return "\n\n".join(h["body"] for h in hits if h.get("body"))


def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


# ============================================================
# Reciprocal Rank Fusion (Cormack et al., 2009)
# ============================================================
def reciprocal_rank_fusion(results: list[list[Document]], k: int = 60) -> list[Document]:
    """Fuse several ranked lists into one: score(d) = sum over lists of 1 / (k + rank)."""
    scores: dict[tuple[str, str], float] = {}
    first_seen: dict[tuple[str, str], Document] = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            key = (doc.page_content, repr(sorted((doc.metadata or {}).items(), key=str)))
            first_seen.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank + k)
    ranked = sorted(scores, key=scores.__getitem__, reverse=True)
    return [first_seen[key] for key in ranked]


# ============================================================
# Shared grounded-answer generation
# ============================================================
ABSTAIN = "I don't know."

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question based ONLY on the context below.
If the context doesn't contain the answer, reply exactly: "I don't know."

Context:
{context}

Question: {question}

Answer:"""
)


def generate_answer(question: str, context, llm=None) -> str:
    """Generate a grounded answer from a list of Documents or a context string."""
    if not isinstance(context, str):
        context = format_docs(context)
    chain = ANSWER_PROMPT | (llm or get_llm()) | StrOutputParser()
    return chain.invoke({"context": context, "question": question})
