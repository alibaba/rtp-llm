"""Immutable execution plans and deterministic case identities."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple, Union

from rtp_llm.test.perf_test.capabilities import CapabilityValidator, validate_capability
from rtp_llm.test.perf_test.config_resolver import ResolvedConfig, canonical_json
from rtp_llm.test.perf_test.errors import PlanningError, ResourceError

PLAN_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FrozenMap(Mapping[str, Any]):
    """A recursively immutable, deterministically ordered JSON object."""

    _items: Tuple[Tuple[str, Any], ...]

    def __iter__(self) -> Iterator[str]:
        return (name for name, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, key: str) -> Any:
        for name, value in self._items:
            if name == key:
                return value
        raise KeyError(key)


def freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        items = []
        for name in sorted(value):
            if not isinstance(name, str):
                raise PlanningError("plan mappings require string keys")
            items.append((name, freeze(value[name])))
        return FrozenMap(tuple(items))
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise PlanningError(f"plan value is not JSON-compatible: {type(value).__name__}")


def thaw(value: Any) -> Any:
    if isinstance(value, FrozenMap):
        return {name: thaw(item) for name, item in value._items}
    if isinstance(value, tuple):
        return [thaw(item) for item in value]
    return value


def case_id(mode: str, identity: Mapping[str, Any]) -> str:
    """Derive a stable identifier only from mode and normalized case identity."""
    if not isinstance(mode, str) or not mode:
        raise PlanningError("case mode must be a non-empty string")
    if not isinstance(identity, Mapping) or not identity:
        raise PlanningError("case identity must be a non-empty object")
    if "case_id" in identity:
        raise PlanningError("case identity must not contain case_id")
    payload = {"mode": mode, "identity": thaw(freeze(identity))}
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"{mode}-{digest[:24]}"


@dataclass(frozen=True)
class PlannedCase:
    case_id: str
    identity: FrozenMap

    def to_dict(self) -> Dict[str, Any]:
        return {"case_id": self.case_id, "identity": thaw(self.identity)}


@dataclass(frozen=True)
class LocalResourcePlan:
    backend: str
    devices: Tuple[int, ...]
    world_size: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "devices": list(self.devices),
            "world_size": self.world_size,
        }


@dataclass(frozen=True)
class AOneResourcePlan:
    backend: str
    resource: str
    gpu_type: str
    gpus_per_shard: int
    shard_count: Optional[int]
    max_parallel: int
    # Gang layout fields; the defaults keep pre-gang construction sites valid.
    # The plan builders always set them to the derived gang topology, and
    # persisted plans without them reconstruct with the single-machine layout.
    gpus_per_machine: int = 0
    machine_count: int = 1
    local_world_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "resource": self.resource,
            "gpu_type": self.gpu_type,
            "gpus_per_shard": self.gpus_per_shard,
            "shard_count": self.shard_count,
            "max_parallel": self.max_parallel,
            "gpus_per_machine": self.gpus_per_machine,
            "machine_count": self.machine_count,
            "local_world_size": self.local_world_size,
        }


ResourcePlan = Union[LocalResourcePlan, AOneResourcePlan]


class ResourcePlanner(Protocol):
    """Validate backend restrictions and return a serializable resource DTO."""

    def __call__(
        self,
        *,
        mode: str,
        resolved_config: Mapping[str, Any],
        world_size: int,
        case_count: int,
    ) -> ResourcePlan: ...


@dataclass(frozen=True)
class ExecutionPlan:
    schema_version: int
    mode: str
    action: str
    backend: str
    resolved_config: FrozenMap
    config_fingerprint: str
    workload_fingerprint: str
    cases: Tuple[PlannedCase, ...]
    resource_plan: ResourcePlan
    result_directory: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "action": self.action,
            "backend": self.backend,
            "resolved_config": thaw(self.resolved_config),
            "config_fingerprint": self.config_fingerprint,
            "workload_fingerprint": self.workload_fingerprint,
            "cases": [case.to_dict() for case in self.cases],
            "resource_plan": self.resource_plan.to_dict(),
            "result_directory": self.result_directory,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionPlan":
        try:
            schema_version = value["schema_version"]
            mode = value["mode"]
            action = value["action"]
            backend = value["backend"]
            raw_config = value["resolved_config"]
            config_fingerprint = value["config_fingerprint"]
            workload_fingerprint = value["workload_fingerprint"]
            raw_cases = value["cases"]
            raw_resource = value["resource_plan"]
            result_directory = value["result_directory"]
        except (KeyError, TypeError) as error:
            raise PlanningError(f"invalid execution plan: missing {error}") from error
        if schema_version != PLAN_SCHEMA_VERSION:
            raise PlanningError(
                f"unsupported execution plan schema_version: {schema_version!r}"
            )
        # Historical remote plans are data, not authorization for new execution.
        if backend == "aone":
            if (mode, action) not in {
                ("prefill", "run"),
                ("prefill", "report"),
                ("decode", "run"),
            }:
                raise PlanningError("invalid historical remote plan capability")
        else:
            validate_capability(mode, action, backend)
        if not isinstance(raw_cases, list):
            raise PlanningError("execution plan cases must be an array")
        cases = []
        for index, item in enumerate(raw_cases):
            if not isinstance(item, Mapping) or not isinstance(
                item.get("identity"), Mapping
            ):
                raise PlanningError(f"execution plan cases[{index}] is invalid")
            expected_id = case_id(mode, item["identity"])
            if item.get("case_id") != expected_id:
                raise PlanningError(
                    f"execution plan cases[{index}].case_id does not match identity"
                )
            cases.append(PlannedCase(expected_id, freeze(item["identity"])))
        resource_plan = _resource_from_dict(raw_resource)
        _validate_persisted_resource_plan(
            resource_plan, raw_config, backend, len(cases)
        )
        return cls(
            schema_version=schema_version,
            mode=mode,
            action=action,
            backend=backend,
            resolved_config=freeze(raw_config),
            config_fingerprint=config_fingerprint,
            workload_fingerprint=workload_fingerprint,
            cases=tuple(cases),
            resource_plan=resource_plan,
            result_directory=result_directory,
        )


def _world_size(resolved_config: Mapping[str, Any]) -> int:
    topology = resolved_config["model"]["topology"]
    return topology["tp"] * topology["dp"] * topology["pp"]


def _local_resource_plan(
    resolved_config: Mapping[str, Any], world_size: int
) -> LocalResourcePlan:
    configured = tuple(resolved_config["execution"]["devices"])
    devices = configured or tuple(range(world_size))
    if len(devices) != world_size:
        raise ResourceError(
            f"model topology requires {world_size} local devices; got {len(devices)}"
        )
    if len(set(devices)) != len(devices):
        raise ResourceError("execution.devices must contain unique device identifiers")
    return LocalResourcePlan("local", devices, world_size)


def _resource_from_dict(value: Any) -> ResourcePlan:
    if not isinstance(value, Mapping):
        raise PlanningError("execution plan resource_plan must be an object")
    backend = value.get("backend")
    try:
        if backend == "local":
            return LocalResourcePlan(
                backend="local",
                devices=tuple(value["devices"]),
                world_size=value["world_size"],
            )
        if backend == "aone":
            gpus_per_shard = value["gpus_per_shard"]
            return AOneResourcePlan(
                backend="aone",
                resource=value["resource"],
                gpu_type=value["gpu_type"],
                gpus_per_shard=gpus_per_shard,
                shard_count=value["shard_count"],
                max_parallel=value["max_parallel"],
                gpus_per_machine=value.get("gpus_per_machine", gpus_per_shard),
                machine_count=value.get("machine_count", 1),
                local_world_size=value.get("local_world_size", gpus_per_shard),
            )
    except KeyError as error:
        raise PlanningError(f"invalid resource plan: missing {error}") from error
    raise PlanningError(f"invalid resource plan backend: {backend!r}")


def _validate_persisted_resource_plan(
    resource_plan: ResourcePlan,
    resolved_config: Mapping[str, Any],
    backend: str,
    case_count: int,
) -> None:
    if resource_plan.backend != backend:
        raise ResourceError(
            "resource plan backend does not match execution plan backend"
        )
    world_size = _world_size(resolved_config)
    if isinstance(resource_plan, LocalResourcePlan):
        if len(resource_plan.devices) != world_size:
            raise ResourceError(
                f"model topology requires {world_size} local devices; "
                f"got {len(resource_plan.devices)}"
            )
        if len(set(resource_plan.devices)) != len(resource_plan.devices):
            raise ResourceError(
                "resource plan devices must contain unique device identifiers"
            )
        if resource_plan.world_size != world_size:
            raise ResourceError(
                "resource plan world_size does not match model topology"
            )
        expected = _local_resource_plan(resolved_config, world_size)
    else:
        execution = resolved_config["execution"]
        per_machine = execution.get("gpus_per_machine")
        if per_machine is None:
            per_machine = world_size
        if per_machine <= 0 or world_size % per_machine:
            raise ResourceError("persisted resource topology is invalid")
        expected = AOneResourcePlan(
            backend="aone",
            resource=execution["resource"],
            gpu_type=execution["gpu_type"],
            gpus_per_shard=world_size,
            shard_count=execution.get("shard_count"),
            max_parallel=execution["max_parallel"],
            gpus_per_machine=per_machine,
            machine_count=world_size // per_machine,
            local_world_size=per_machine,
        )
    if resource_plan != expected:
        raise ResourceError(
            "resource plan does not match the resolved execution configuration"
        )


def build_execution_plan(
    *,
    mode: str,
    action: str,
    resolved: ResolvedConfig,
    cases: Sequence[Mapping[str, Any]],
    result_directory: str,
    resource_planner: Optional[ResourcePlanner] = None,
    capability_validator: CapabilityValidator = validate_capability,
) -> ExecutionPlan:
    """Build and validate an immutable plan without touching the filesystem."""
    if not isinstance(result_directory, str) or not result_directory:
        raise PlanningError("result_directory must be a non-empty string")
    backend = resolved.resolved_config["execution"]["backend"]
    capability_validator(mode, action, backend)
    if not cases:
        raise PlanningError("execution plan requires at least one case")

    planned_cases = []
    seen = set()
    for identity in cases:
        identifier = case_id(mode, identity)
        if identifier in seen:
            raise PlanningError(f"duplicate case identity: {identifier}")
        seen.add(identifier)
        planned_cases.append(PlannedCase(identifier, freeze(identity)))
    grids = resolved.resolved_config["workload"].get("grids", ())
    if not (mode == "prefill" and grids):
        planned_cases.sort(key=lambda item: item.case_id)

    world_size = _world_size(resolved.resolved_config)
    if resource_planner is not None:
        resource_plan = resource_planner(
            mode=mode,
            resolved_config=resolved.resolved_config,
            world_size=world_size,
            case_count=len(planned_cases),
        )
        if resource_plan.backend != backend:
            raise ResourceError(
                "resource plan backend does not match execution plan backend"
            )
    elif backend == "local":
        resource_plan: ResourcePlan = _local_resource_plan(
            resolved.resolved_config, world_size
        )
    else:
        raise PlanningError("non-local execution requires an explicit resource planner")
    return ExecutionPlan(
        schema_version=PLAN_SCHEMA_VERSION,
        mode=mode,
        action=action,
        backend=backend,
        resolved_config=freeze(resolved.resolved_config),
        config_fingerprint=resolved.config_fingerprint,
        workload_fingerprint=resolved.workload_fingerprint,
        cases=tuple(planned_cases),
        resource_plan=resource_plan,
        result_directory=result_directory,
    )
