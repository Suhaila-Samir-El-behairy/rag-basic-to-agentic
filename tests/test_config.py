"""Tests for config module."""
import pytest

from ragkit import config


def test_project_root_exists():
    assert config.PROJECT_ROOT.exists()
    assert config.PROJECT_ROOT.is_dir()

def test_directories_created():
    assert config.DATA_DIR.exists()
    assert config.CHROMA_DIR.exists()

def test_default_llm_model():
    # Set a known provider
    import os
    os.environ["LLM_PROVIDER"] = "groq"
    assert config.get_default_llm_model() == "llama-3.3-70b-versatile"

def test_missing_api_key_raises(monkeypatch):
    """Verify get_llm() raises when GROQ_API_KEY is missing."""
    # Patch the config module's cached value (it's already loaded)
    monkeypatch.setattr(config, "GROQ_API_KEY", None)
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")

    from ragkit.utils import get_llm

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        get_llm()
