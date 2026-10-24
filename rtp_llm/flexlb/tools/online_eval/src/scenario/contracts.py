"""Shared authoring, execution and evidence contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol

from analysis.checks import CheckResult

if TYPE_CHECKING:
    from runtime.deadline import Deadline
    from scenario.builder import CaseBuilder
    from scenario.context import RuntimeContext


@dataclass(frozen=True)
class ExecutionRequirements:
    request_evidence: bool = False
    monitoring: bool = False

    def __post_init__(self):
        if type(self.request_evidence) is not bool or type(self.monitoring) is not bool:
            raise ValueError("execution requirements must be boolean")

    def validate(self, execution):
        if self.request_evidence and execution.get("collection", "request") == "aggregate":
            raise ValueError("execution.collection requires request or diagnostic evidence")
        if self.monitoring and (
            not isinstance(execution.get("monitoring"), dict)
            or execution["monitoring"].get("capture_metrics", True) is False
        ):
            raise ValueError("execution.monitoring.capture_metrics must be true for this consumer")


class StageValidator(Protocol):
    def __call__(self, params: dict, plan: PlanContext) -> dict: ...


class StageExecutor(Protocol):
    def __call__(self, context: RuntimeContext, params: dict, deadline: Deadline) -> StageOutput: ...


class ProgramBuilder(Protocol):
    def __call__(self, case: CaseBuilder) -> None: ...


class ViewValidator(Protocol):
    def __call__(self, path: str, data: dict, fail: Callable[[str, str], None]) -> None: ...


class ViewRenderer(Protocol):
    def __call__(
        self, directory: Path, analysis: dict, presentation: dict, deadline: Deadline | None = None
    ) -> Path: ...


class SourceFactory(Protocol):
    def __call__(self, environment: object, fields: set[str]) -> dict: ...


@dataclass(frozen=True)
class StageHandler:
    """validate(params, plan) -> normalized params; execute(ctx, params, deadline).

    plan.reference(value, expected_kind) resolves prior-stage output types.
    Runtime output handles use ctx.register_resource(), never raw user IDs.
    Execute must honor remaining deadline/cancellation in actual I/O calls.
    """

    name: str
    validate: StageValidator
    execute: StageExecutor
    outputs: dict[str, str]
    requires: frozenset[str] = frozenset()
    checks: frozenset[str] | Callable[[dict], frozenset[str]] = frozenset()
    max_dynamic_additions: object = 0  # nonnegative int or normalized-params -> int
    # Maximum fresh worker population requested by this stage (not additions).
    # Callable receives normalized params and the selected profile.
    max_environment_workers: object = 0
    next_environment: object = None  # normalized params -> next raw environment
    execution_requirements: ExecutionRequirements = field(default_factory=ExecutionRequirements)
    owners: frozenset[str] = frozenset()  # empty means foundational

    def __post_init__(self):
        if not isinstance(self.execution_requirements, ExecutionRequirements):
            raise ValueError("stage execution_requirements must be an ExecutionRequirements")


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
    case: str = ""

    @property
    def env(self):
        from scenario.environment_view import EnvironmentView

        if not hasattr(self, "_environment_view"):
            self._environment_view = EnvironmentView(self.environment, self.profiles, path=self.path)
        return self._environment_view

    def reference(self, value, expected_kind=None):
        from scenario.references import reference

        return reference(value, self.path, self.outputs, expected_kind)
