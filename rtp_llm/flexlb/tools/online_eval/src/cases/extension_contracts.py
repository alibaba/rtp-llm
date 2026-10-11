"""Author-facing callable contracts, checked without executing capabilities."""

from inspect import signature
from typing import Any, Protocol


class ProgramBuilder(Protocol):
    def __call__(self, case: Any) -> None: ...


class ViewValidator(Protocol):
    def __call__(self, path: Any, data: dict, fail: Any) -> None: ...


class ViewRenderer(Protocol):
    def __call__(self, directory: Any, analysis: dict, presentation: dict) -> Any: ...


class ProducerDescription(Protocol):
    def __call__(self, producer: str, metric_id: str, calculation: Any) -> dict: ...


class GateProducer(Protocol):
    def __call__(self, directory: Any) -> Any: ...


class FinalizeProducer(Protocol):
    def __call__(self, directory: Any, context: dict) -> Any: ...


class SourceFactory(Protocol):
    def __call__(self, environment: Any, fields: set[str]) -> dict: ...


def validate_call(function, argument_count, path):
    """Reject incompatible positional calls at discovery, never run user code."""
    try:
        signature(function).bind(*(object() for _ in range(argument_count)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: callable must accept {argument_count} positional arguments: {exc}") from exc
