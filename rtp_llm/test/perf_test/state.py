"""Run/case state machines and stable outcome-to-exit-code mapping."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from rtp_llm.test.perf_test.errors import PerfTestError


class RunState(str, Enum):
    CREATED = "created"
    PLANNED = "planned"
    RUNNING = "running"
    COLLECTING = "collecting"
    REPORTING = "reporting"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class CaseState(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    OK = "ok"
    REQUEST_FAILED = "request_failed"
    INVALID_SHAPE = "invalid_shape"
    INVALID_REUSE = "invalid_reuse"
    INVALID_TIMING = "invalid_timing"


@dataclass(frozen=True)
class RunOutcome:
    state: RunState
    failure: Optional[PerfTestError] = None
    signal_number: Optional[int] = None

    @property
    def exit_code(self) -> int:
        if self.state == RunState.COMPLETED:
            if self.failure is not None or self.signal_number is not None:
                raise ValueError("completed outcome cannot contain a failure or signal")
            return 0
        if self.state == RunState.INTERRUPTED:
            if self.failure is not None or self.signal_number is None:
                raise ValueError(
                    "interrupted outcome requires only a positive signal number"
                )
            if (
                isinstance(self.signal_number, bool)
                or not isinstance(self.signal_number, int)
                or self.signal_number <= 0
            ):
                raise ValueError("signal_number must be a positive integer")
            return 128 + self.signal_number
        if self.state not in {RunState.PARTIAL, RunState.FAILED}:
            raise ValueError(
                f"non-terminal run state has no exit code: {self.state.value}"
            )
        if self.signal_number is not None:
            raise ValueError("only interrupted outcomes may contain a signal number")
        if self.failure is None:
            return 5
        if not isinstance(self.failure, PerfTestError):
            raise ValueError("failure must be a typed PerfTestError")
        return self.failure.exit_code
