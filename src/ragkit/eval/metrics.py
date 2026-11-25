"""Metrics that need no LLM, plus the statistics used to report them."""

import random
import re
from collections.abc import Sequence

# Script 03 reports routes; map them to the agent's tool names so both are comparable.
ROUTE_TO_TOOL = {"vectorstore": "vector_search", "web": "web_search", "arxiv": "arxiv_search"}

_ABSTAIN_PATTERNS = re.compile(
    r"i don'?t know|i do not know|not enough information|don'?t have enough information"
    r"|does not (contain|provide|mention)|doesn'?t (contain|provide|mention)"
    r"|no information|cannot (be )?answer|can'?t answer|unable to (answer|find)",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    text = text.lower().replace("’", "'").replace("–", "-").replace("—", "-")
    return " ".join(text.split())


def evidence_rank(contexts: Sequence[str], phrase: str) -> int | None:
    """1-based rank of the first context containing the phrase, or None."""
    needle = normalize(phrase)
    for rank, ctx in enumerate(contexts, 1):
        if needle in normalize(ctx):
            return rank
    return None


def recall_at_k(
    contexts: Sequence[str], evidence: Sequence[str], k: int | None = None
) -> float | None:
    """Fraction of gold evidence phrases found in the top-k contexts (None if no evidence)."""
    if not evidence:
        return None
    top = list(contexts)[:k] if k else list(contexts)
    return sum(evidence_rank(top, e) is not None for e in evidence) / len(evidence)


def reciprocal_rank(contexts: Sequence[str], evidence: Sequence[str]) -> float | None:
    """1 / rank of the first context containing any gold evidence phrase (0 if none)."""
    if not evidence:
        return None
    ranks = [r for r in (evidence_rank(contexts, e) for e in evidence) if r is not None]
    return 1.0 / min(ranks) if ranks else 0.0


def is_abstention(answer: str) -> bool:
    return not answer.strip() or bool(_ABSTAIN_PATTERNS.search(answer))


def tool_selection_correct(tool_calls: Sequence[str], expected: Sequence[str]) -> bool:
    """Did the pipeline use one of the expected tools (or no tool, if none is expected)?"""
    used = {ROUTE_TO_TOOL.get(t, t) for t in tool_calls}
    if not expected:
        return not used
    return bool(used & set(expected))


# ============================================================
# Statistics
# ============================================================
def _clean(values: Sequence[float | bool | None]) -> list[float]:
    return [float(v) for v in values if v is not None]


def mean(values: Sequence[float | bool | None]) -> float | None:
    vals = _clean(values)
    return sum(vals) / len(vals) if vals else None


def bootstrap_ci(
    values: Sequence[float | bool | None],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float | None, float | None, float | None]:
    """Mean with a percentile-bootstrap (1 - alpha) confidence interval."""
    vals = _clean(values)
    if not vals:
        return None, None, None
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(rng.choices(vals, k=n)) / n for _ in range(n_resamples))
    lo = means[int((alpha / 2) * n_resamples)]
    hi = means[min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)]
    return sum(vals) / n, lo, hi


def paired_bootstrap(
    a: Sequence[float | bool | None],
    b: Sequence[float | bool | None],
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, float] | None:
    """Compare two systems on the same questions: mean(a - b), its 95% CI, and a p-value.

    Pairs where either side is missing are dropped. The p-value is two-sided: twice
    the fraction of resampled mean differences on the other side of zero.
    """
    pairs = [
        (float(x), float(y)) for x, y in zip(a, b, strict=True) if x is not None and y is not None
    ]
    if not pairs:
        return None
    diffs = [x - y for x, y in pairs]
    rng = random.Random(seed)
    n = len(diffs)
    boot = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(n_resamples))
    observed = sum(diffs) / n
    if observed >= 0:
        tail = sum(d <= 0 for d in boot) / n_resamples
    else:
        tail = sum(d >= 0 for d in boot) / n_resamples
    return {
        "diff": observed,
        "lo": boot[int(0.025 * n_resamples)],
        "hi": boot[int(0.975 * n_resamples)],
        "p": min(1.0, 2 * tail),
        "n": n,
    }


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float | None:
    """Agreement between two binary raters, corrected for chance."""
    if len(a) != len(b):
        raise ValueError("raters must label the same items")
    n = len(a)
    if n == 0:
        return None
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - expected) / (1 - expected)
