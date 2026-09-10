"""Shared extension contracts for scenario adapters and child-runner integration.

Adapters live in scenario/actions and export HANDLERS, without mutating the
registry at import time. The core explicitly collects those descriptors.
"""

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class StageHandler:
    """validate(params, plan) -> normalized params; execute(ctx, params, deadline).

    plan.reference(value, expected_kind) resolves prior-stage output types.
    Runtime output handles use ctx.register_resource(), never raw user IDs.
    Execute must honor remaining deadline/cancellation in actual I/O calls.
    """

    name: str
    validate: Callable
    execute: Callable
    outputs: dict[str, str]
    requires: frozenset[str] = frozenset()
    checks: frozenset[str] = frozenset()
    max_dynamic_additions: object = 0  # nonnegative int or normalized-params -> int
    # Maximum fresh worker population requested by this stage (not additions).
    # Callable receives normalized params and the selected profile.
    max_environment_workers: object = 0
    next_environment: object = None  # normalized params -> next raw environment


@dataclass(frozen=True)
class CheckResult:
    id: str
    status: str  # PASS | FAIL | ERROR | SKIP; only ordinary FAIL can match a finding
    detail: str = ""
    actual: object = None
    expected: object = None
    evidence: dict = field(default_factory=dict)


@dataclass
class StageOutput:
    output: dict = field(default_factory=dict)
    checks: list[CheckResult] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CheckHandler:
    name: str
    validate: Callable  # (params, plan) -> normalized params
    evaluate: Callable  # (ctx, params, deadline) -> CheckResult


@dataclass(frozen=True)
class ResourceHandle:
    kind: str
    id: str
    env_epoch: int

    def to_dict(self):
        return {"kind": self.kind, "id": self.id, "env_epoch": self.env_epoch}


@dataclass
class PlanContext:
    path: str
    outputs: dict
    environment: dict = field(default_factory=dict)
    profiles: tuple[str, ...] = ()

    def reference(self, value, expected_kind=None):
        from .compiler import reference

        return reference(value, self.path, self.outputs, expected_kind)
