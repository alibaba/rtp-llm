"""Run/case state machines and stable outcome-to-exit-code mapping."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, Optional

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
    TARGET_TPOT_NOT_MET = "target_tpot_not_met"


class StateTransitionError(ValueError):
    """Raised when persisted state attempts an undeclared transition."""


_RUN_TRANSITIONS: Dict[RunState, FrozenSet[RunState]] = {
    RunState.CREATED: frozenset(
        {RunState.PLANNED, RunState.FAILED, RunState.INTERRUPTED}
    ),
    RunState.PLANNED: frozenset(
        {RunState.RUNNING, RunState.REPORTING, RunState.FAILED, RunState.INTERRUPTED}
    ),
    RunState.RUNNING: frozenset(
        {
            RunState.COLLECTING,
            RunState.REPORTING,
            RunState.COMPLETED,
            RunState.PARTIAL,
            RunState.FAILED,
            RunState.INTERRUPTED,
        }
    ),
    RunState.COLLECTING: frozenset(
        {
            RunState.REPORTING,
            RunState.COMPLETED,
            RunState.PARTIAL,
            RunState.FAILED,
            RunState.INTERRUPTED,
        }
    ),
    RunState.REPORTING: frozenset(
        {
            RunState.COMPLETED,
            RunState.PARTIAL,
            RunState.FAILED,
            RunState.INTERRUPTED,
        }
    ),
    RunState.COMPLETED: frozenset(),
    RunState.PARTIAL: frozenset(),
    RunState.FAILED: frozenset(),
    RunState.INTERRUPTED: frozenset(),
}

_CASE_TRANSITIONS: Dict[CaseState, FrozenSet[CaseState]] = {
    CaseState.PLANNED: frozenset({CaseState.RUNNING}),
    CaseState.RUNNING: frozenset(
        {
            CaseState.OK,
            CaseState.REQUEST_FAILED,
            CaseState.INVALID_SHAPE,
            CaseState.INVALID_REUSE,
            CaseState.INVALID_TIMING,
        }
    ),
    CaseState.OK: frozenset(),
    CaseState.REQUEST_FAILED: frozenset({CaseState.RUNNING}),
    CaseState.INVALID_SHAPE: frozenset({CaseState.RUNNING}),
    CaseState.INVALID_REUSE: frozenset({CaseState.RUNNING}),
    CaseState.INVALID_TIMING: frozenset({CaseState.RUNNING}),
}


def transition_run(current: RunState, target: RunState) -> RunState:
    if target not in _RUN_TRANSITIONS[current]:
        raise StateTransitionError(
            f"invalid run state transition: {current.value} -> {target.value}"
        )
    return target


def transition_case(current: CaseState, target: CaseState) -> CaseState:
    if target not in _CASE_TRANSITIONS[current]:
        raise StateTransitionError(
            f"invalid case state transition: {current.value} -> {target.value}"
        )
    return target


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
