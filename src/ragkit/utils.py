"""Shared utilities — Groq + HuggingFace + LangSmith."""

import logging
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragkit.config import (
    CHROMA_DIR,
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

logger = logging.getLogger(__name__)


def get_llm(temperature: float = 0, model: str | None = None, **kwargs):
    """Free LLM via Groq (default) or Gemini."""
    chosen_model = model or get_default_llm_model()

    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY missing. Get a free key at https://console.groq.com")
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=chosen_model,
            temperature=temperature,
            groq_api_key=GROQ_API_KEY,
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


@lru_cache(maxsize=4)
def get_embeddings(model: str = EMBED_MODEL):
    """Free embeddings via HuggingFace Inference API."""
    if not HUGGINGFACEHUB_API_TOKEN:
        raise ValueError(
            "HUGGINGFACEHUB_API_TOKEN missing. Get a free token at "
            "https://huggingface.co/settings/tokens"
        )
    from langchain_huggingface import HuggingFaceEndpointEmbeddings

    logger.info("Loading embedding model: %s", model)
    return HuggingFaceEndpointEmbeddings(
        model=model,
        task="feature-extraction",
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )


def load_web(url: str):
    return WebBaseLoader(url).load()


def load_pdf(path: str):
    return PyPDFLoader(path).load()


def split_docs(
    docs, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    ).split_documents(docs)


def get_vectorstore(splits, name: str, embeddings=None):
    """Build or load a persistent Chroma vectorstore."""
    persist_dir = CHROMA_DIR / name
    embeddings = embeddings or get_embeddings()

    if (persist_dir / "chroma.sqlite3").exists():
        logger.info("Loading existing vector store: %s", name)
        return Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)

    logger.info("Building new vector store: %s (%d splits)", name, len(splits))
    return Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )


def web_search(query: str, max_results: int = 3) -> str:
    """Free web search: Tavily if key set, else DuckDuckGo."""
    if TAVILY_API_KEY:
        from tavily import TavilyClient

        results = TavilyClient(api_key=TAVILY_API_KEY).search(query, max_results=max_results)
        return "\n\n".join(r["content"] for r in results["results"])

    from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        hits = list(ddgs.text(query, max_results=max_results))
    return "\n\n".join(h["body"] for h in hits if h.get("body"))


def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)
