"""Shared helpers for DSV4 fused-kernel perf tests."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch.profiler import ProfilerActivity, profile, record_function


DEFAULT_M_SWEEP = [
    1, 2, 3, 5, 7, 8, 11, 16, 17, 31, 32, 37, 61, 64, 67, 97, 127,
    128, 191, 251, 256, 257, 509, 512, 769, 1021, 1024, 1531, 2039,
    2048, 3079, 4093, 4096, 6151, 8191, 8192, 12289, 16381, 16384,
    24593, 32749, 32768, 49157, 65521, 65536,
]


@dataclass
class KernelMeasurement:
    event_us: float
    kernel_span_us: float
    kernel_sum_us: float
    kernel_count: int
    kernel_names: List[str]
    kernel_name_counts_per_iter: Dict[str, float]
    idle_gap_us: float
    trace_path: str
    measure_method: str


@dataclass
class TimelineStats:
    span_us: float
    kernel_sum_us: float
    kernel_union_us: float
    idle_gap_us: float
    kernel_count: int
    kernel_names: List[str]
    kernel_name_counts_per_iter: Dict[str, float]
    trace_path: str


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def parse_int_list(name: str, default: Iterable[int]) -> List[int]:
    value = os.environ.get(name)
    if not value:
        return list(default)
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def iters_for_m(M: int) -> int:
    if M <= 256:
        return 200
    if M <= 4096:
        return 60
    if M <= 16384:
        return 20
    return 6


def bench_cuda_event(
    fn: Callable[[], object],
    *,
    warmup: int = 20,
    iters: int = 100,
    samples: int = 5,
) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    timings = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000.0 / iters)
    timings.sort()
    return timings[len(timings) // 2]


def _merged_interval_duration(intervals: List[Tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0.0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    total += cur_end - cur_start
    return total


def parse_timeline(
    trace_path: str,
    *,
    kernel_regex: Optional[str] = None,
    iters: int = 1,
) -> TimelineStats:
    with open(trace_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    events = data if isinstance(data, list) else data.get("traceEvents", [])
    pattern = re.compile(kernel_regex) if kernel_regex else None
    kernels = []
    for event in events:
        if event.get("ph") != "X" or "ts" not in event or "dur" not in event:
            continue
        cat = str(event.get("cat", ""))
        name = str(event.get("name", ""))
        if "kernel" not in cat.lower():
            continue
        if pattern is not None and pattern.search(name) is None:
            continue
        kernels.append(event)

    if not kernels:
        suffix = f" matching {kernel_regex!r}" if kernel_regex else ""
        raise AssertionError(f"profile trace has no GPU kernel events{suffix}: {trace_path}")

    scale = max(iters, 1)
    intervals = [(float(e["ts"]), float(e["ts"]) + float(e["dur"])) for e in kernels]
    name_counts: Dict[str, int] = {}
    for event in kernels:
        name = str(event.get("name", ""))
        name_counts[name] = name_counts.get(name, 0) + 1
    span_us = max(end for _, end in intervals) - min(start for start, _ in intervals)
    kernel_sum_us = sum(float(e["dur"]) for e in kernels)
    kernel_union_us = _merged_interval_duration(intervals)
    return TimelineStats(
        span_us=span_us / scale,
        kernel_sum_us=kernel_sum_us / scale,
        kernel_union_us=kernel_union_us / scale,
        idle_gap_us=(span_us - kernel_union_us) / scale,
        kernel_count=len(kernels) // scale,
        kernel_names=sorted(name_counts),
        kernel_name_counts_per_iter={name: count / scale for name, count in sorted(name_counts.items())},
        trace_path=trace_path,
    )


def measure_kernel(
    fn: Callable[[], object],
    *,
    label: str,
    trace_dir: str,
    kernel_regex: Optional[str] = None,
    warmup: int = 20,
    iters: int = 100,
    profile_enabled: bool = True,
    profile_iters: int = 1,
) -> KernelMeasurement:
    event_us = bench_cuda_event(fn, warmup=warmup, iters=iters)
    if not profile_enabled:
        return KernelMeasurement(
            event_us=event_us,
            kernel_span_us=event_us,
            kernel_sum_us=event_us,
            kernel_count=0,
            kernel_names=[],
            kernel_name_counts_per_iter={},
            idle_gap_us=0.0,
            trace_path="",
            measure_method="cuda_event",
        )

    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(trace_dir, safe_name(label) + ".json")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function(label):
            for _ in range(profile_iters):
                fn()
            torch.cuda.synchronize()
    prof.export_chrome_trace(trace_path)
    stats = parse_timeline(trace_path, kernel_regex=kernel_regex, iters=profile_iters)
    return KernelMeasurement(
        event_us=event_us,
        kernel_span_us=stats.span_us,
        kernel_sum_us=stats.kernel_sum_us,
        kernel_count=stats.kernel_count,
        kernel_names=stats.kernel_names,
        kernel_name_counts_per_iter=stats.kernel_name_counts_per_iter,
        idle_gap_us=stats.idle_gap_us,
        trace_path=stats.trace_path,
        measure_method="torch_profiler_kernel_span",
    )


def report_path_from_env(env_name: str, default_name: str) -> str:
    explicit = os.environ.get(env_name)
    if explicit:
        return explicit
    return os.path.join(os.getcwd(), "build_logs", default_name)


def trace_dir_from_report(report_path: str, env_name: str) -> str:
    explicit = os.environ.get(env_name)
    if explicit:
        return explicit
    return os.path.splitext(report_path)[0] + "_traces"


def write_json_report(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def measurement_payload(measurement: KernelMeasurement) -> dict:
    return asdict(measurement)


def device_payload() -> dict:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    device = torch.cuda.current_device()
    return {
        "cuda_available": True,
        "cuda_device": torch.cuda.get_device_name(device),
        "cuda_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def git_commit(cwd: Optional[str] = None) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def ensure_triton_cc() -> None:
    """Set a usable C compiler for Triton's launcher build in slim containers."""
    cc = os.environ.get("CC")
    if cc and os.path.exists(cc):
        return
    detected = shutil.which("gcc")
    if detected:
        os.environ["CC"] = detected
