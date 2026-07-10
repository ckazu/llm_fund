"""Shared LLM-backend fakes for tests.

`FakeBackend` is a deterministic `LlmBackend` stub: it returns (or raises) each
scripted item in order on successive `complete()` calls and records every call's
arguments. Payload dicts are serialised to JSON text, mirroring the real
"JSON-only output" contract (llm/structured.py). Exceptions in the script are
raised as-is, so `LlmBackendError` entries exercise the tenacity retry paths.

`FakeRouter` satisfies the small surface `cli.py` uses (`for_role` /
`label_for`) so CLI tests can monkeypatch `llm_fund.cli.build_router`.
"""

import json
from typing import Any

from llm_fund.llm.backend import LlmResponse

FAKE_BACKEND_NAME = "fake"
FAKE_MODEL = "fake-model"
FAKE_MODEL_LABEL = f"{FAKE_BACKEND_NAME}:{FAKE_MODEL}"


class FakeBackend:
    """Scripted `LlmBackend`: each item is a payload dict, raw text, or exception."""

    def __init__(self, script: list[Any], *, repeat_last: bool = False) -> None:
        self._script = list(script)
        self._repeat_last = repeat_last
        self.calls: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def complete(
        self, system: str, prompt: str, *, max_tokens: int, temperature: float
    ) -> LlmResponse:
        self.calls.append(
            {
                "system": system,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if self._repeat_last and len(self._script) == 1:
            item = self._script[0]
        else:
            item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        return LlmResponse(
            text=text,
            model=FAKE_MODEL,
            backend=FAKE_BACKEND_NAME,
            usage={"input_tokens": 10, "output_tokens": 5},
        )


class FakeRouter:
    """`LlmRouter` stand-in returning the same fake backend for every role."""

    def __init__(self, backend: FakeBackend) -> None:
        self._backend = backend

    def for_role(self, role: str) -> tuple[FakeBackend, str]:
        return self._backend, FAKE_MODEL

    def label_for(self, role: str) -> str:
        return FAKE_MODEL_LABEL
