"""Global configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project root (one level above src/ragkit/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# LangSmith tracing
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY or ""
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "rag-from-scratch-to-agentic")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv(
    "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
)
os.environ.setdefault("USER_AGENT", "ragkit/0.1.0")

# Provider + model
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Chunking + retrieval
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_K = 4


def get_default_llm_model() -> str:
    return GROQ_MODEL if LLM_PROVIDER == "groq" else GEMINI_MODEL
