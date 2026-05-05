"""Typed dataclass models for smoke / perf cases.

Wraps the existing dict-of-dict layout in `smoke_defs_*.py` without changing
its on-disk shape. Use `SmokeCase.from_dict(name, dict_)` when you want type
safety + IDE autocomplete; the existing dicts continue to work unchanged.

Validation lives in `validation.py` — `verify_smoke_suites.py` calls those
checkers as a CI prepare-source gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

DataRoot = Literal["oss", "internal"]
EnvArgs = Union[List[str], Dict[str, List[str]]]
SmokeArgs = Union[str, Dict[str, str]]


@dataclass
class SmokeCase:
    """Typed view of one entry in `SMOKE_TESTS[<suite>][<case>]`."""

    name: str
    task_info: str
    smoke_args: SmokeArgs
    gpu_type: str
    platform: str = "cuda"
    markers: List[str] = field(default_factory=list)
    timeout: int = 600
    envs: EnvArgs = field(default_factory=list)
    # data_root is optional today; B0 split per-suite makes it explicit per file.
    data_root: Optional[DataRoot] = None
    # Optional fields consumed by run_smoke_test as kwargs to CaseRunner.
    sleep_time_qr: Optional[int] = None
    kill_remote: Optional[bool] = None
    concurrency_test: Optional[bool] = None
    features: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, name: str, raw: Mapping[str, Any]) -> "SmokeCase":
        return cls(
            name=name,
            task_info=raw["task_info"],
            smoke_args=raw.get("smoke_args", ""),
            gpu_type=raw.get("gpu_type", "gpu_cuda12"),
            platform=raw.get("platform", "cuda"),
            markers=list(raw.get("markers", [])),
            timeout=int(raw.get("timeout", 600)),
            envs=raw.get("envs", []),
            data_root=raw.get("data_root"),
            sleep_time_qr=raw.get("sleep_time_qr"),
            kill_remote=raw.get("kill_remote"),
            concurrency_test=raw.get("concurrency_test"),
            features=list(raw.get("features", [])),
        )


@dataclass
class PerfCase:
    """Typed view of one entry in `PERF_TESTS[<case>]`."""

    name: str
    mode: Literal["grid", "distribution"]
    model_type: str
    checkpoint_path: str
    gpu_type: str
    gpu_count: int
    perf_args: List[str]
    baseline: str = ""
    markers: List[str] = field(default_factory=list)
    timeout: int = 7200
    envs: Dict[str, str] = field(default_factory=dict)
    tokenizer_path: Optional[str] = None

    @classmethod
    def from_dict(cls, name: str, raw: Mapping[str, Any]) -> "PerfCase":
        return cls(
            name=name,
            mode=raw["mode"],
            model_type=raw["model_type"],
            checkpoint_path=raw["checkpoint_path"],
            gpu_type=raw.get("gpu_type", "H20"),
            gpu_count=int(raw.get("gpu_count", 1)),
            perf_args=list(raw.get("perf_args", [])),
            baseline=raw.get("baseline", ""),
            markers=list(raw.get("markers", [])),
            timeout=int(raw.get("timeout", 7200)),
            envs=dict(raw.get("envs", {})),
            tokenizer_path=raw.get("tokenizer_path"),
        )
