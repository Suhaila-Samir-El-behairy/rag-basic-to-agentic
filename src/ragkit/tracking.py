"""Count LLM calls and tokens for everything run inside a ``with track_usage():`` block.

Works like ``get_openai_callback``: a context-var callback handler is attached to
every LangChain run started in the block (including runs in ``.batch`` threads),
so pipelines need no changes to be measured.
"""

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.tracers.context import register_configure_hook


@dataclass
class UsageHandler(BaseCallbackHandler):
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        with self._lock:
            self.llm_calls += 1

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
        with self._lock:
            self.llm_calls += 1

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        prompt = completion = 0
        for generations in response.generations:
            for gen in generations:
                usage = getattr(getattr(gen, "message", None), "usage_metadata", None)
                if usage:
                    prompt += usage.get("input_tokens", 0)
                    completion += usage.get("output_tokens", 0)
        with self._lock:
            self.prompt_tokens += prompt
            self.completion_tokens += completion


_usage_var: ContextVar[UsageHandler | None] = ContextVar("ragkit_usage", default=None)
register_configure_hook(_usage_var, inheritable=True)


@contextmanager
def track_usage() -> Iterator[UsageHandler]:
    handler = UsageHandler()
    token = _usage_var.set(handler)
    try:
        yield handler
    finally:
        _usage_var.reset(token)
