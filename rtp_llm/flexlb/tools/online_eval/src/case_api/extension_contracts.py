"""Author-facing callable contracts, checked without executing capabilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
from case_api.call_contract import validate_call as validate_call
from typing import TYPE_CHECKING, Callable, Mapping, Protocol

if TYPE_CHECKING:
    from case_api.builder import CaseBuilder
    from scenario.context import RuntimeContext
    from scenario.contracts import PlanContext, StageOutput
    from runtime.deadline import Deadline


@dataclass(frozen=True)
class FinalizationContext:
    """One producer's isolated snapshot; returned metadata is published by the caller."""
    directory: Path
    run: Mapping[str, object]
    deadline: object = None

    @classmethod
    def snapshot(cls, directory, run, deadline=None):
        return cls(Path(directory), MappingProxyType(deepcopy(run)), deadline)


@dataclass(frozen=True)
class ResourceEvidence:
    requests: dict = field(default_factory=dict)
    outages: tuple = ()
    producer: object = None
    errors: tuple = ()


class StageValidator(Protocol):
    def __call__(self, params: dict, plan: PlanContext) -> dict: ...


class StageExecutor(Protocol):
    def __call__(self, context: RuntimeContext, params: dict, deadline: Deadline) -> StageOutput: ...


class ProgramBuilder(Protocol):
    def __call__(self, case: CaseBuilder) -> None: ...


class ViewValidator(Protocol):
    def __call__(self, path: str, data: dict, fail: Callable[[str, str], None]) -> None: ...


class ViewRenderer(Protocol):
    def __call__(self, directory: Path, analysis: dict, presentation: dict) -> Path: ...


class ProducerDescription(Protocol):
    def __call__(self, producer: str, metric_id: str, calculation: dict | None) -> dict: ...


class GateProducer(Protocol):
    def __call__(self, directory: Path) -> None: ...


class FinalizeProducer(Protocol):
    def __call__(self, directory: Path, context: FinalizationContext) -> dict: ...


class SourceFactory(Protocol):
    def __call__(self, environment: object, fields: set[str]) -> dict: ...



class SnapshotReader(Protocol):
    def read(self, deadline: Deadline) -> object: ...


class SnapshotFactory(Protocol):
    def __call__(self, services: object, params: dict) -> SnapshotReader: ...


class RuntimeClient(Protocol):
    def cleanup(self, deadline: Deadline) -> None: ...


class RuntimeClientFactory(Protocol):
    def __call__(self, services: object, params: dict, deadline: Deadline) -> RuntimeClient: ...


class EvidenceExporter(Protocol):
    def __call__(self, client: object, profile: str, deadline: Deadline) -> ResourceEvidence: ...
