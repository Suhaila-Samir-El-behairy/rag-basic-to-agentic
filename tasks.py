"""Task definitions for invoke — cross-platform replacement for Make."""
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
def format(c):
    """Run ruff formatter."""
    c.run("ruff format src tests")

@task
def clean(c):
    """Remove cache + vector stores."""
    c.run("find . -type d -name __pycache__ -exec rm -rf {} +", warn=True)
    c.run("find . -type f -name '*.pyc' -delete", warn=True)
    c.run("rm -rf chroma_db .pytest_cache .ruff_cache .mypy_cache", warn=True)
    c.run("rm -f bm25_cache*.pkl", warn=True)

@task
def run(c, n, q):
    """Run script N with question Q. Example: inv run -n 01 -q 'What is task decomposition?'"""
    c.run(f"python main.py {n} --question \"{q}\"")

@task
def install_optional(c):
    """Install invoke itself."""
    c.run("pip install invoke")