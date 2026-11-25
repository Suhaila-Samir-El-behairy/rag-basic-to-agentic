"""Evaluation datasets."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from ragkit.config import EVAL_DATA_DIR

DEFAULT_DATASET = EVAL_DATA_DIR / "agent_blog_qa.jsonl"
CATEGORIES = ("single_hop", "multi_hop", "out_of_corpus", "unanswerable")


@dataclass
class EvalExample:
    id: str
    question: str
    gold_answer: str
    category: str
    # Short phrases that a context must contain to count as supporting evidence.
    gold_evidence: list[str] = field(default_factory=list)
    # Tools an ideal agent would call; [] means "answer without tools" is ideal.
    expected_tools: list[str] = field(default_factory=list)

    @property
    def answerable(self) -> bool:
        return self.category != "unanswerable"


def load_jsonl(path: str | Path = DEFAULT_DATASET) -> list[EvalExample]:
    examples = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            ex = EvalExample(**json.loads(line))
            if ex.category not in CATEGORIES:
                raise ValueError(f"{path}:{line_no}: unknown category {ex.category!r}")
            examples.append(ex)
    ids = [e.id for e in examples]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate example ids")
    return examples
