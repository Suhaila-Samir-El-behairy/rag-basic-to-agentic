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
    """Run ruff linter + mypy."""
    c.run("ruff check src tests")
    c.run("mypy src")


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


@task(help={"n": "Script number, e.g. 01", "q": "Question", "e": "Extra script flags"})
def run(c, n, q, e=""):
    """Run script N with question Q.

    Example: inv run -n 05 -q "What is task decomposition?" -e "--method hybrid"
    """
    c.run(f"{shlex.quote(sys.executable)} main.py {n} --question {shlex.quote(q)} {e}")


@task(
    help={
        "p": "Pipelines (space-separated), 'default' or 'all'",
        "limit": "Only the first N questions",
        "run_id": "Run directory name under results/ (resumes if it exists)",
    }
)
def eval(c, p="default", limit=0, run_id=""):  # noqa: A001 - task name used as `inv eval`
    """Benchmark pipelines on the QA set. Example: inv eval -p "01:basic 07:agent" -l 5"""
    cmd = f"{shlex.quote(sys.executable)} main.py eval --pipelines {p}"
    if limit:
        cmd += f" --limit {int(limit)}"
    if run_id:
        cmd += f" --run-id {shlex.quote(run_id)}"
    c.run(cmd, pty=False)


@task
def eval_check(c):
    """Check that every gold-evidence phrase occurs in the corpus."""
    c.run(f"{shlex.quote(sys.executable)} main.py eval-check")
