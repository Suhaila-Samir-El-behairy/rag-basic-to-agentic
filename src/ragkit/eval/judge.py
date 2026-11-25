"""LLM-as-judge for answer correctness and faithfulness, with an on-disk cache.

Judging is the expensive part of a benchmark run, so every verdict is cached by
a hash of (judge model, prompt kind, inputs). Re-running a report, or resuming an
interrupted run, costs nothing for questions that were already judged.

Use a different (ideally stronger) model than the pipelines when you can
(``RAGKIT_JUDGE_MODEL``): a model grading its own answers tends to be lenient.
How far the judge can be trusted is measured, not assumed: label a sample by
hand with ``ragkit eval-label`` and the report shows Cohen's kappa.
"""

import hashlib
import json
import os
import threading
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ragkit.config import get_default_llm_model

MAX_CONTEXT_CHARS = 6000


class CorrectnessVerdict(BaseModel):
    reasoning: str = Field(description="One or two sentences comparing answer and reference")
    correct: bool = Field(description="True if the answer matches the reference answer")


class FaithfulnessVerdict(BaseModel):
    reasoning: str = Field(description="One or two sentences naming any unsupported claim")
    faithful: bool = Field(description="True if every claim in the answer is supported")


CORRECTNESS_PROMPT = ChatPromptTemplate.from_template(
    """You are grading a system's answer against a reference answer.

Question: {question}
Reference answer: {gold}
System answer: {answer}

The system answer is correct if it contains the key facts of the reference answer
and does not contradict it. Extra correct detail is fine. Wording may differ.
An answer that says it doesn't know is not correct."""
)

FAITHFULNESS_PROMPT = ChatPromptTemplate.from_template(
    """You are checking whether an answer is supported by the retrieved context.

Context:
{context}

Answer: {answer}

The answer is faithful if every factual claim in it is stated in or directly
implied by the context. Ignore whether the claims are true in the real world."""
)


class Judge:
    def __init__(self, llm=None, cache_path: str | Path | None = None, model: str | None = None):
        self.model = model or os.getenv("RAGKIT_JUDGE_MODEL") or get_default_llm_model()
        self._llm = llm
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, dict] = {}
        self._lock = threading.Lock()
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self._cache[entry["key"]] = entry["value"]

    @property
    def llm(self):
        if self._llm is None:
            from ragkit.utils import get_llm

            self._llm = get_llm(model=self.model)
        return self._llm

    def _key(self, kind: str, **inputs) -> str:
        payload = json.dumps([self.model, kind, inputs], sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _judge(self, kind: str, prompt, schema, **inputs) -> dict:
        key = self._key(kind, **inputs)
        if key in self._cache:
            return self._cache[key]
        verdict = (prompt | self.llm.with_structured_output(schema)).invoke(inputs)
        value = verdict.model_dump()
        with self._lock:
            self._cache[key] = value
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "value": value}) + "\n")
        return value

    def correctness(self, question: str, gold: str, answer: str) -> dict:
        return self._judge(
            "correctness",
            CORRECTNESS_PROMPT,
            CorrectnessVerdict,
            question=question,
            gold=gold,
            answer=answer,
        )

    def faithfulness(self, answer: str, contexts: list[str]) -> dict:
        context = "\n\n---\n\n".join(contexts)[:MAX_CONTEXT_CHARS]
        return self._judge(
            "faithfulness", FAITHFULNESS_PROMPT, FaithfulnessVerdict, context=context, answer=answer
        )
