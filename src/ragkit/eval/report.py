"""Turn scored rows into a summary table, significance tests, and a cost plot."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ragkit.eval import metrics
from ragkit.eval.datasets import CATEGORIES

HUMAN_LABELS_FILE = "human_labels.jsonl"
SUMMARY_FILE = "summary.md"
PLOT_FILE = "quality_vs_cost.png"
BASELINE = "01:basic"

# (column header, metric key) for the main table
QUALITY_METRICS = [
    ("Correct", "correct"),
    ("Faithful", "faithful"),
    ("Recall@4", "recall@4"),
    ("MRR", "mrr"),
    ("Tool choice", "tool_correct"),
]


def _by_pipeline(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[r["pipeline"]].append(r)
    for group in grouped.values():
        group.sort(key=lambda r: r["example_id"])
    return grouped


def summarize(rows: list[dict]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for name, group in _by_pipeline(rows).items():
        s: dict[str, Any] = {"n": len(group)}
        for _, key in QUALITY_METRICS:
            s[key] = metrics.bootstrap_ci([r["metrics"].get(key) for r in group])
        for cat in CATEGORIES:
            s[f"correct/{cat}"] = metrics.mean(
                [r["metrics"]["correct"] for r in group if r["category"] == cat]
            )
        s["errors"] = sum(bool(r["result"]["error"]) for r in group)
        s["llm_calls"] = metrics.mean([r["result"]["llm_calls"] for r in group])
        s["tokens"] = metrics.mean([r["result"]["total_tokens"] for r in group])
        s["latency_s"] = metrics.mean([r["result"]["latency_s"] for r in group])
        steps = [r["metrics"].get("hit_step_limit") for r in group]
        if any(v is not None for v in steps):
            s["hit_step_limit"] = metrics.mean(steps)
        summary[name] = s
    return summary


def compare_to_baseline(rows: list[dict], baseline: str = BASELINE, key: str = "correct"):
    """Paired bootstrap of each pipeline against the baseline on the shared questions."""
    grouped = _by_pipeline(rows)
    if baseline not in grouped:
        return {}
    base = {r["example_id"]: r["metrics"].get(key) for r in grouped[baseline]}
    out = {}
    for name, group in grouped.items():
        if name == baseline:
            continue
        shared = [r for r in group if r["example_id"] in base]
        out[name] = metrics.paired_bootstrap(
            [r["metrics"].get(key) for r in shared], [base[r["example_id"]] for r in shared]
        )
    return out


def judge_agreement(rows: list[dict], labels_path: Path) -> dict[str, Any] | None:
    """Cohen's kappa between human correctness labels and the LLM judge's."""
    if not labels_path.exists():
        return None
    judged = {
        (r["pipeline"], r["example_id"]): r["metrics"]["correct"]
        for r in rows
        if "correct_reason" in r["metrics"]  # decided by the judge, not by a rule
    }
    human, judge = [], []
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        label = json.loads(line)
        key = (label["pipeline"], label["example_id"])
        if key in judged:
            human.append(bool(label["correct"]))
            judge.append(bool(judged[key]))
    if not human:
        return None
    agree = sum(h == j for h, j in zip(human, judge, strict=True)) / len(human)
    return {"n": len(human), "agreement": agree, "kappa": metrics.cohen_kappa(human, judge)}


def _fmt(value, pct=True) -> str:
    if value is None:
        return "–"
    return f"{100 * value:.0f}" if pct else f"{value:.1f}"


def _fmt_ci(ci) -> str:
    m, lo, hi = ci
    if m is None:
        return "–"
    return f"{100 * m:.0f} [{100 * lo:.0f}, {100 * hi:.0f}]"


def to_markdown(rows: list[dict], meta: dict | None = None, labels_path: Path | None = None) -> str:
    summary = summarize(rows)
    order = sorted(summary, key=lambda n: (n.split(":")[0], n))
    lines = []
    if meta:
        lines.append(
            f"Run: {meta.get('created')} · commit `{meta.get('git_commit')}` · "
            f"LLM `{meta.get('llm_model')}` · judge `{meta.get('judge_model')}` · "
            f"embeddings `{meta.get('embed_model')}`\n"
        )

    lines.append("### Quality (%, mean with 95% bootstrap CI)\n")
    header = [
        "Pipeline",
        "n",
        *[h for h, _ in QUALITY_METRICS],
        "LLM calls",
        "Tokens",
        "Latency (s)",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for name in order:
        s = summary[name]
        cells = [f"`{name}`", str(s["n"])]
        cells += [_fmt_ci(s[key]) for _, key in QUALITY_METRICS]
        cells += [
            _fmt(s["llm_calls"], False),
            _fmt(s["tokens"], False),
            _fmt(s["latency_s"], False),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("\n### Correctness by question type (%)\n")
    lines.append("| Pipeline | " + " | ".join(CATEGORIES) + " | Errors |")
    lines.append("|" + "---|" * (len(CATEGORIES) + 2))
    for name in order:
        s = summary[name]
        cells = [_fmt(s[f"correct/{c}"]) for c in CATEGORIES]
        lines.append(f"| `{name}` | " + " | ".join(cells) + f" | {s['errors']} |")

    comparisons = compare_to_baseline(rows)
    if comparisons:
        lines.append(f"\n### Correctness vs `{BASELINE}` (paired bootstrap)\n")
        lines.append("| Pipeline | Δ points | 95% CI | p |")
        lines.append("|---|---|---|---|")
        for name in order:
            c = comparisons.get(name)
            if c:
                lines.append(
                    f"| `{name}` | {100 * c['diff']:+.0f} | "
                    f"[{100 * c['lo']:+.0f}, {100 * c['hi']:+.0f}] | {c['p']:.3f} |"
                )

    if labels_path is not None:
        agreement = judge_agreement(rows, labels_path)
        lines.append("\n### LLM-judge reliability\n")
        if agreement:
            lines.append(
                f"On {agreement['n']} hand-labelled answers the judge agreed with the human "
                f"label {100 * agreement['agreement']:.0f}% of the time "
                f"(Cohen's κ = {agreement['kappa']:.2f})."
            )
        else:
            lines.append("No human labels yet: run `ragkit eval-label --run <dir>`.")
    return "\n".join(lines) + "\n"


def plot_quality_vs_cost(rows: list[dict], path: Path) -> Path | None:
    """Scatter of correctness against mean LLM calls per question (needs matplotlib)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    summary = summarize(rows)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for name, s in summary.items():
        m, lo, hi = s["correct"]
        if m is None or s["llm_calls"] is None:
            continue
        ax.errorbar(
            s["llm_calls"], 100 * m, yerr=[[100 * (m - lo)], [100 * (hi - m)]],
            fmt="o", color="#4C72B0", ecolor="#B0B0B0", capsize=3,
        )  # fmt: skip
        ax.annotate(name, (s["llm_calls"], 100 * m), textcoords="offset points", xytext=(5, 4),
                    fontsize=8)  # fmt: skip
    ax.set_xlim(left=0)
    ax.set_xlabel("Mean LLM calls per question")
    ax.set_ylabel("Correct (%)")
    ax.set_title("Answer quality vs cost")
    ax.grid(alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def write_report(run_dir: Path, rows: list[dict]) -> Path:
    meta_path = run_dir / "run.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    md = to_markdown(rows, meta, run_dir / HUMAN_LABELS_FILE)
    if plot_quality_vs_cost(rows, run_dir / PLOT_FILE):
        md += f"\n![Answer quality vs cost]({PLOT_FILE})\n"
    out = run_dir / SUMMARY_FILE
    out.write_text(md, encoding="utf-8")
    return out
