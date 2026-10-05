"""Task definitions for invoke — cross-platform replacement for Make."""

import shlex
import sys

from invoke import task


@task
def install(c):
    """Install package + dependencies."""
    c.run("pip install -e .")


@task
def install_dev(c):
    """Install with dev dependencies."""
    c.run('pip install -e ".[dev]"')


@task
def test(c):
    """Run pytest."""
    c.run("pytest")


@task
def lint(c):
    """Run ruff linter."""
    c.run("ruff check src tests")


@task
def format(c):  # noqa: A001 - task name used as `inv format`
    """Run ruff formatter."""
    c.run("ruff format src tests")


@task
def clean(c):
    """Remove cache + vector stores."""
    c.run("find . -type d -name __pycache__ -exec rm -rf {} +", warn=True)
    c.run("find . -type f -name '*.pyc' -delete", warn=True)
    c.run("rm -rf chroma_db .pytest_cache .ruff_cache .mypy_cache", warn=True)
    c.run("rm -f bm25_cache*.pkl", warn=True)


@task(help={"n": "Script number, e.g. 01", "q": "Question", "e": "Extra script flags"})
def run(c, n, q, e=""):
    """Run script N with question Q.

    Example: inv run -n 05 -q "What is task decomposition?" -e "--method hybrid"
    """
    c.run(f"{shlex.quote(sys.executable)} main.py {n} --question {shlex.quote(q)} {e}")


@task
def install_optional(c):
    """Install invoke itself."""
    c.run("pip install invoke")
