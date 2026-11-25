"""Run pipelines over a dataset and score every (pipeline, question) pair."""

import json
import subprocess
import sys
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ragkit import config
from ragkit.eval import metrics
from ragkit.eval.datasets import EvalExample
from ragkit.eval.judge import Judge
from ragkit.pipeline import RAGResult, get_pipeline, run_pipeline

ROWS_FILE = "per_question.jsonl"
META_FILE = "run.json"
RECALL_K = 4
# Pipelines whose tool_calls reflect a tool/source *choice* worth scoring.
TOOL_CHOOSING = {"03:routing", "07:agent"}


def score(
    example: EvalExample, result: RAGResult, judge: Judge | None, pipeline: str = ""
) -> dict[str, Any]:
    """All metrics for one row. None means "not applicable" and is excluded from means."""
    in_corpus = example.category in ("single_hop", "multi_hop")
    abstained = metrics.is_abstention(result.answer)
    m: dict[str, Any] = {
        "recall@4": metrics.recall_at_k(result.contexts, example.gold_evidence, RECALL_K)
        if in_corpus
        else None,
        "mrr": metrics.reciprocal_rank(result.contexts, example.gold_evidence)
        if in_corpus
        else None,
        "abstained": abstained,
        "error": result.error is not None,
        "correct": None,
        "faithful": None,
        "tool_correct": None,
    }

    if result.error:
        m["correct"] = False  # a crash is a wrong answer, not a skipped question
    elif not example.answerable:
        m["correct"] = abstained  # the only right answer is to abstain
    elif abstained:
        m["correct"] = False
    elif judge is not None:
        verdict = judge.correctness(example.question, example.gold_answer, result.answer)
        m["correct"] = verdict["correct"]
        m["correct_reason"] = verdict["reasoning"]

    if judge is not None and not result.error and not abstained and result.contexts:
        verdict = judge.faithfulness(result.answer, result.contexts)
        m["faithful"] = verdict["faithful"]
        m["faithful_reason"] = verdict["reasoning"]

    if pipeline in TOOL_CHOOSING:
        m["tool_correct"] = metrics.tool_selection_correct(
            result.tool_calls, example.expected_tools
        )
    if "hit_step_limit" in result.metadata:
        m["hit_step_limit"] = result.metadata["hit_step_limit"]
    return m


def load_rows(run_dir: Path) -> list[dict]:
    """Rows of a run; if a pair was re-run, the latest row wins."""
    path = run_dir / ROWS_FILE
    if not path.exists():
        return []
    latest: dict[tuple[str, str], dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[(row["pipeline"], row["example_id"])] = row
    return list(latest.values())


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=config.PROJECT_ROOT,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_eval(
    pipelines: Iterable[str],
    examples: list[EvalExample],
    run_dir: Path,
    judge: Judge | None,
    pipeline_fns: dict[str, Callable[[str], RAGResult]] | None = None,
) -> list[dict]:
    """Run every pipeline on every example, appending scored rows to run_dir.

    Pairs that already have an error-free row are skipped, so an interrupted run
    (rate limit, Ctrl-C) resumes where it stopped.
    """
    pipelines = list(pipelines)
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / META_FILE
    if not meta_path.exists():
        meta = {
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "llm_provider": config.LLM_PROVIDER,
            "llm_model": config.get_default_llm_model(),
            "judge_model": judge.model if judge else None,
            "embed_model": config.EMBED_MODEL,
            "pipelines": pipelines,
            "n_examples": len(examples),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    done = {
        (r["pipeline"], r["example_id"]) for r in load_rows(run_dir) if not r["result"]["error"]
    }
    rows_path = run_dir / ROWS_FILE
    total = len(pipelines) * len(examples)
    step = 0
    for name in pipelines:
        fn = (pipeline_fns or {}).get(name) or get_pipeline(name)
        for ex in examples:
            step += 1
            if (name, ex.id) in done:
                continue
            result = run_pipeline(fn, ex.question)
            row: dict[str, Any] = {
                "pipeline": name,
                "example_id": ex.id,
                "category": ex.category,
                "question": ex.question,
                "gold_answer": ex.gold_answer,
                "result": result.to_dict(),
                "metrics": score(ex, result, judge, name),
            }
            with rows_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
            status = "ERROR" if result.error else f"correct={row['metrics']['correct']}"
            print(
                f"[{step}/{total}] {name} {ex.id}: {status} ({result.latency_s:.1f}s)",
                file=sys.stderr,
            )
    return load_rows(run_dir)
