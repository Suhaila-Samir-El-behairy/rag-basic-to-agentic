"""Command-line entry points for the evaluation harness."""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

from ragkit import config
from ragkit.eval import report
from ragkit.eval.datasets import CATEGORIES, DEFAULT_DATASET, load_jsonl
from ragkit.eval.runner import load_rows, run_eval
from ragkit.pipeline import DEFAULT_SUITE, PIPELINES

JUDGE_CACHE = "judge_cache.jsonl"


def _resolve_pipelines(names: list[str]) -> list[str]:
    if names in (["default"], []):
        return list(DEFAULT_SUITE)
    if names == ["all"]:
        return list(PIPELINES)
    unknown = [n for n in names if n not in PIPELINES]
    if unknown:
        raise SystemExit(f"Unknown pipeline(s): {unknown}. Choose from: {', '.join(PIPELINES)}")
    return names


def cmd_eval(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="ragkit eval", description="Benchmark the pipelines")
    parser.add_argument(
        "--pipelines",
        nargs="*",
        default=["default"],
        help="Pipeline names, 'default' (one per technique) or 'all'. "
        f"Available: {', '.join(PIPELINES)}",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--categories", nargs="*", choices=CATEGORIES, help="Only these types")
    parser.add_argument("--limit", type=int, help="Only the first N questions (smoke runs)")
    parser.add_argument("--run-id", help="Directory name under results/ (resumes if it exists)")
    parser.add_argument(
        "--rpm", type=float, default=25, help="LLM requests/minute cap (Groq free tier: 30)"
    )
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM-judged metrics")
    args = parser.parse_args(argv)

    from ragkit.eval.judge import Judge
    from ragkit.utils import set_rate_limit

    examples = load_jsonl(args.dataset)
    if args.categories:
        examples = [e for e in examples if e.category in args.categories]
    if args.limit:
        examples = examples[: args.limit]

    set_rate_limit(args.rpm)
    run_dir = config.RESULTS_DIR / (args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S"))
    judge = None if args.no_judge else Judge(cache_path=config.RESULTS_DIR / JUDGE_CACHE)
    pipelines = _resolve_pipelines(args.pipelines)
    print(f"[eval] {len(pipelines)} pipelines x {len(examples)} questions → {run_dir}")

    rows = run_eval(pipelines, examples, run_dir, judge)
    out = report.write_report(run_dir, rows)
    print(out.read_text(encoding="utf-8"))
    print(f"[eval] Report written to {out}")
    return 0


def cmd_report(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="ragkit eval-report")
    parser.add_argument("--run", type=Path, required=True, help="results/<run_id>")
    args = parser.parse_args(argv)
    rows = load_rows(args.run)
    if not rows:
        print(f"No rows in {args.run}", file=sys.stderr)
        return 1
    out = report.write_report(args.run, rows)
    print(out.read_text(encoding="utf-8"))
    return 0


def cmd_label(argv: list[str]) -> int:
    """Hand-label a random sample of judged answers, to measure judge reliability."""
    parser = argparse.ArgumentParser(prog="ragkit eval-label")
    parser.add_argument("--run", type=Path, required=True, help="results/<run_id>")
    parser.add_argument("--n", type=int, default=30, help="How many answers to label")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    labels_path = args.run / report.HUMAN_LABELS_FILE
    labelled = set()
    if labels_path.exists():
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                labelled.add((d["pipeline"], d["example_id"]))

    candidates = [
        r
        for r in load_rows(args.run)
        if "correct_reason" in r["metrics"] and (r["pipeline"], r["example_id"]) not in labelled
    ]
    random.Random(args.seed).shuffle(candidates)
    todo = candidates[: max(0, args.n - len(labelled))]
    print(f"{len(labelled)} labelled so far; {len(todo)} to go. Answer y / n / s(kip) / q(uit).")
    print("The judge's verdict is hidden so it can't bias you.\n")

    for i, row in enumerate(todo, 1):
        print("=" * 70)
        print(f"[{i}/{len(todo)}] {row['pipeline']} · {row['example_id']}")
        print(f"Q:    {row['question']}")
        print(f"Gold: {row['gold_answer']}")
        print(f"Answer:\n{row['result']['answer']}\n")
        choice = ""
        while choice not in {"y", "n", "s", "q"}:
            choice = input("Correct? [y/n/s/q] ").strip().lower()
        if choice == "q":
            break
        if choice == "s":
            continue
        with labels_path.open("a", encoding="utf-8") as f:
            label = {
                "pipeline": row["pipeline"],
                "example_id": row["example_id"],
                "correct": choice == "y",
            }
            f.write(json.dumps(label) + "\n")

    agreement = report.judge_agreement(load_rows(args.run), labels_path)
    if agreement:
        print(
            f"\nJudge vs you on {agreement['n']} answers: {100 * agreement['agreement']:.0f}% "
            f"agreement, Cohen's κ = {agreement['kappa']:.2f}"
        )
    return 0


def missing_evidence(examples, corpus_text: str) -> list[tuple[str, str]]:
    """(example id, phrase) for every gold-evidence phrase absent from the corpus."""
    from ragkit.eval.metrics import normalize

    haystack = normalize(corpus_text)
    return [(e.id, p) for e in examples for p in e.gold_evidence if normalize(p) not in haystack]


def cmd_check(argv: list[str]) -> int:
    """Validate the dataset: every gold-evidence phrase must occur in the corpus."""
    parser = argparse.ArgumentParser(prog="ragkit eval-check")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args(argv)

    from ragkit import corpus

    examples = load_jsonl(args.dataset)
    text = "\n".join(d.page_content for d in corpus.load_docs())
    missing = missing_evidence(examples, text)
    for ex_id, phrase in missing:
        print(f"  {ex_id}: evidence not found in corpus: {phrase!r}")
    n = sum(len(e.gold_evidence) for e in examples)
    print(f"{n - len(missing)}/{n} evidence phrases found in the corpus.")
    return 1 if missing else 0


COMMANDS = {
    "eval": cmd_eval,
    "eval-report": cmd_report,
    "eval-label": cmd_label,
    "eval-check": cmd_check,
}


def main(command: str, argv: list[str]) -> int:
    return COMMANDS[command](argv)
