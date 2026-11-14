"""Typed decode configuration, policies, and frozen case identities."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple, Union


@dataclass(frozen=True)
class FixedBatchConfig:
    strategy: str
    sizes: Tuple[int, ...]
    unit: str = "global"


@dataclass(frozen=True)
class TargetTPOTConfig:
    strategy: str
    max_size: int
    candidate_step: int
    target_tpot_ms: float


DecodeBatchConfig = Union[FixedBatchConfig, TargetTPOTConfig]


@dataclass(frozen=True)
class DecodeProfilingConfig:
    enabled: bool = False
    runs: int = 1
    steps: int = 3
    enable_all_ranks: bool = True
    flush_timeout_seconds: int = 60


@dataclass(frozen=True)
class ExactBatchPolicy:
    strategy: str
    size: int


@dataclass(frozen=True)
class TargetTPOTPolicy:
    strategy: str
    max_size: int
    candidate_step: int
    target_tpot_ms: float


CaseBatchPolicy = Union[ExactBatchPolicy, TargetTPOTPolicy]


@dataclass(frozen=True)
class DecodeCaseSpec:
    workload_kind: str
    input_lengths: Tuple[int, ...]
    output_length: int
    batch_policy: CaseBatchPolicy
    measure_runs: int
    warmup_runs: int
    generate_config: Dict[str, Any] = field(default_factory=dict)
    profiling: DecodeProfilingConfig = field(default_factory=DecodeProfilingConfig)
    distribution: Optional[Dict[str, Any]] = None
    batch_unit: str = "global"

    def to_identity(self) -> Dict[str, Any]:
        identity = {
            "workload_kind": self.workload_kind,
            "input_lengths": list(self.input_lengths),
            "output_length": self.output_length,
            "batch_policy": asdict(self.batch_policy),
            "measure_runs": self.measure_runs,
            "warmup_runs": self.warmup_runs,
            "generate_config": copy.deepcopy(self.generate_config),
            "profiling": asdict(self.profiling),
            "distribution": copy.deepcopy(self.distribution),
        }
        if self.batch_unit != "global":
            identity["batch_unit"] = self.batch_unit
        return identity

    @classmethod
    def from_identity(cls, value: Mapping[str, Any]) -> "DecodeCaseSpec":
        raw_policy = dict(value["batch_policy"])
        policy: CaseBatchPolicy
        if raw_policy["strategy"] == "fixed":
            policy = ExactBatchPolicy(strategy="fixed", size=int(raw_policy["size"]))
        elif raw_policy["strategy"] == "target_tpot":
            policy = TargetTPOTPolicy(
                strategy="target_tpot",
                max_size=int(raw_policy["max_size"]),
                candidate_step=int(raw_policy["candidate_step"]),
                target_tpot_ms=float(raw_policy["target_tpot_ms"]),
            )
        else:
            raise ValueError(f"unknown decode batch policy: {raw_policy['strategy']!r}")
        return cls(
            workload_kind=str(value["workload_kind"]),
            input_lengths=tuple(int(item) for item in value["input_lengths"]),
            output_length=int(value["output_length"]),
            batch_policy=policy,
            measure_runs=int(value["measure_runs"]),
            warmup_runs=int(value["warmup_runs"]),
            generate_config=copy.deepcopy(dict(value["generate_config"])),
            profiling=DecodeProfilingConfig(**dict(value["profiling"])),
            distribution=(
                None
                if value.get("distribution") is None
                else copy.deepcopy(dict(value["distribution"]))
            ),
            batch_unit=str(value.get("batch_unit", "global")),
        )


@dataclass(frozen=True)
class DecodeBatchMeasurement:
    batch_size: int
    input_lengths: Tuple[int, ...]
    worker_allocations: Tuple[Dict[str, Any], ...]
    measurements: Tuple[Dict[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "input_lengths": list(self.input_lengths),
            "worker_allocations": copy.deepcopy(list(self.worker_allocations)),
            "measurements": copy.deepcopy(list(self.measurements)),
        }


@dataclass(frozen=True)
class DecodeStrategyOutcome:
    selected: Optional[DecodeBatchMeasurement]
    search: Optional[Dict[str, Any]]
    error_code: Optional[str] = None
