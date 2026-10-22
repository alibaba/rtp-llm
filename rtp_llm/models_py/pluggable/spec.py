"""Immutable, device-runtime-free descriptions for module construction."""

import importlib
import json
from dataclasses import dataclass
from typing import Any, FrozenSet, Mapping, Optional, Tuple

from rtp_llm.device.device_type import DeviceType


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_entrypoint(path: str):
    module, separator, name = path.partition(":")
    if not separator or not module or not name or ":" in name:
        raise ValueError(f"Expected module:callable entrypoint, got {path!r}")
    result = getattr(importlib.import_module(module), name)
    if not callable(result):
        raise TypeError(f"Entrypoint {path!r} is not callable")
    return result


@dataclass(frozen=True)
class SupportResult:
    supported: bool
    reason: str = ""

    def __post_init__(self):
        if type(self.supported) is not bool or not isinstance(self.reason, str):
            raise TypeError("SupportResult requires a boolean and a reason string")


@dataclass(frozen=True)
class ModuleSpec:
    module_id: str
    api_version: int
    contract_id: str
    required_methods: Tuple[str, ...] = ("forward",)
    # Includes concrete base-class/protocol validation where the boundary needs it.
    validate_instance: Optional[str] = None

    def __post_init__(self):
        if not self.module_id or not self.contract_id or self.api_version < 1:
            raise ValueError(
                "Module identity, contract and positive API version required"
            )
        object.__setattr__(self, "required_methods", tuple(self.required_methods))


@dataclass(frozen=True)
class ModuleImplSpec:
    module_id: str
    impl_id: str
    api_version: int
    builder: str
    supported_devices: FrozenSet[DeviceType]
    predicate: str
    priority: int
    contract_id: str
    weight_format_id: str
    state_format_id: str
    collective_protocol_id: str
    capabilities: FrozenSet[str] = frozenset()
    auto_selectable: bool = False
    describe_build_requests: Optional[str] = None
    # Model/backend-owned descriptors; evaluated before allocation or loading.
    describe_resources: Optional[str] = None
    prepare_weights: Optional[str] = None
    # Implementation-owned readiness checks after each engine resource binding.
    validate_initialized: Optional[str] = None

    def __post_init__(self):
        for field in (
            "module_id",
            "impl_id",
            "builder",
            "predicate",
            "contract_id",
            "weight_format_id",
            "state_format_id",
            "collective_protocol_id",
        ):
            if not getattr(self, field):
                raise ValueError(f"Implementation requires {field}")
        devices = frozenset(self.supported_devices)
        if not devices or any(not isinstance(d, DeviceType) for d in devices):
            raise TypeError("supported_devices must contain DeviceType values")
        object.__setattr__(self, "supported_devices", devices)
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))


@dataclass(frozen=True)
class BuildRequest:
    module_id: str
    path: str
    weight_format_id: str
    state_format_id: str
    required_capabilities: FrozenSet[str] = frozenset()
    # A canonical snapshot prevents the caller mutating selection inputs later.
    metadata_json: str = "{}"

    def __post_init__(self):
        if not all(
            (self.module_id, self.path, self.weight_format_id, self.state_format_id)
        ):
            raise ValueError("Build request requires identity, path and formats")
        metadata = json.loads(self.metadata_json)
        if not isinstance(metadata, dict):
            raise TypeError("Build request metadata must be a JSON object")
        object.__setattr__(self, "metadata_json", canonical_json(metadata))
        object.__setattr__(
            self, "required_capabilities", frozenset(self.required_capabilities)
        )

    @classmethod
    def create(cls, *, metadata: Optional[Mapping[str, Any]] = None, **kwargs):
        return cls(metadata_json=canonical_json(dict(metadata or {})), **kwargs)

    @property
    def metadata(self):
        return json.loads(self.metadata_json)


@dataclass(frozen=True)
class ModuleBinding:
    request: BuildRequest
    implementation: ModuleImplSpec
    selection_source: str

    def protocol_record(self):
        impl = self.implementation
        return {
            "path": self.request.path,
            "module_id": impl.module_id,
            "impl_id": impl.impl_id,
            "api_version": impl.api_version,
            "contract_id": impl.contract_id,
            "weight_format_id": impl.weight_format_id,
            "state_format_id": impl.state_format_id,
            "collective_protocol_id": impl.collective_protocol_id,
            "capabilities": sorted(impl.capabilities),
            "required_capabilities": sorted(self.request.required_capabilities),
            "metadata": self.request.metadata,
            "describe_resources": impl.describe_resources,
            "prepare_weights": impl.prepare_weights,
            "validate_initialized": impl.validate_initialized,
        }
