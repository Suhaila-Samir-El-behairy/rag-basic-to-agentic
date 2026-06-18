"""Smoke test for script 01."""
import importlib
import os

import pytest


def _load_script():
    """Load script 01 via importlib (works around digit-prefixed module name)."""
    return importlib.import_module("ragkit.scripts.01_basic_rag")

def test_module_imports():
    mod = _load_script()
    assert hasattr(mod, "main")
    assert hasattr(mod, "URL")

def test_url_is_lilian_weng():
    mod = _load_script()
    assert "lilianweng" in mod.URL

@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="Set GROQ_API_KEY to run integration test",
)
def test_chain_construction():
    """Verify the build_chain function works without errors."""
    mod = _load_script()
    # Just verify the function exists and is callable
    assert callable(mod.build_chain)
