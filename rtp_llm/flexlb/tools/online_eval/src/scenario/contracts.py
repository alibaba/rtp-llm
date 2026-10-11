"""Shared extension contracts for scenario adapters and child-runner integration.

Foundations export HANDLERS from scenario/actions. Case programs declare
ACTION_HANDLERS from cases/<case>/actions.py. The catalog collects descriptors explicitly;
modules do not mutate a registry at import time.
"""

from dataclasses import dataclass, field
from typing import Callable
from analysis.checks import CheckResult


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
    owners: frozenset[str] = frozenset()  # empty means foundational


@dataclass
class StageOutput:
    output: dict = field(default_factory=dict)
    checks: list[CheckResult] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)


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
        from scenario.stage_compiler import reference

        return reference(value, self.path, self.outputs, expected_kind)
