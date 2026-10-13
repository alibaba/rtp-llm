"""Core action parameters; case adapters own their extensible payload contracts."""
from __future__ import annotations

from dataclasses import dataclass, fields
import json
from typing import Protocol, Optional, Union

from document_io import DocumentError as ScenarioError
from scenario.validation import mapping
from scenario.references import OutputReference


class StageParameters(Protocol):
    def to_dict(self) -> dict: ...


@dataclass(frozen=True)
class EmptyParameters:
    def to_dict(self):
        return {}


@dataclass(frozen=True)
class RequestParameters:
    input_len: int
    output_len: int
    count: int
    consume: str
    block_keys: Optional[tuple[int, ...]] = None
    priority: Optional[int] = None
    qos_level: Optional[int] = None
    schedule_timeout_s: Optional[float] = None
    stream_timeout_s: Optional[float] = None
    post_issue_delay_s: Optional[float] = None

    def to_dict(self):
        result = {field.name: getattr(self, field.name) for field in fields(self)
                  if getattr(self, field.name) is not None}
        if self.block_keys is not None:
            result["block_keys"] = list(self.block_keys)
        return result


@dataclass(frozen=True)
class RequestReferenceParameters:
    requests: OutputReference

    def to_dict(self):
        return dict(requests=self.requests.to_dict())


@dataclass(frozen=True)
class ComparisonParameters:
    actual: OutputReference
    op: str
    expected: Union[bool, int, float, str]

    def to_dict(self):
        return dict(actual=self.actual.to_dict(), op=self.op, expected=self.expected)


@dataclass(frozen=True)
class AdapterParameters:
    """Frozen transport for an action whose schema is owned by its registered adapter."""
    document: str

    def to_dict(self):
        return json.loads(self.document)


def decode_parameters(action, value):
    mapping(value, "stage.params", set(value) if isinstance(value, dict) else set())
    if action in {"setup", "teardown"}:
        mapping(value, "stage.params", set())
        return EmptyParameters()
    if action == "request":
        from scenario.request_parameters import normalize_request_parameters
        required = {"input_len", "output_len", "count", "consume"}
        if not required <= set(value):
            raise ScenarioError("compiled request parameters must be complete")
        normalized = dict(value)
        normalize_request_parameters(normalized, "stage")
        if "block_keys" in normalized:
            normalized["block_keys"] = tuple(normalized["block_keys"])
        return RequestParameters(**normalized)
    if action in {"wait", "cancel"}:
        mapping(value, "stage.params", {"requests"}, {"requests"})
        return RequestReferenceParameters(OutputReference.read(value["requests"]))
    if action == "check":
        mapping(value, "stage.params", {"actual", "op", "expected"}, {"actual", "op", "expected"})
        from analysis.checks import validate_comparison
        validate_comparison(value["op"], value["expected"])
        return ComparisonParameters(OutputReference.read(value["actual"]), value["op"], value["expected"])
    return AdapterParameters(json.dumps(value, separators=(",", ":"), allow_nan=False))
