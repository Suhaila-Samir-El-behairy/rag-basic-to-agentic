"""Tests for config module."""

from ragkit import config


def test_project_root_exists():
    assert config.PROJECT_ROOT.exists()
    assert config.PROJECT_ROOT.is_dir()


def test_directories_created():
    assert config.DATA_DIR.exists()
    assert config.CHROMA_DIR.exists()


def test_default_llm_model_groq(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    assert config.get_default_llm_model() == config.GROQ_MODEL


def test_default_llm_model_gemini(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    assert config.get_default_llm_model() == config.GEMINI_MODEL
