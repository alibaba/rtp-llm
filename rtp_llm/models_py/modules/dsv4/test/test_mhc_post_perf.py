"""Manual benchmark for DSV4 TileLang mHC post variants."""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import statistics
import time
import unittest
from contextlib import contextmanager
from typing import Callable, Iterable, List

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from dsv4_kernel_perf_utils import (
    device_payload,
    git_commit,
    parse_int_list,
    report_path_from_env,
    safe_name,
    trace_dir_from_report,
    write_json_report,
)


def _prepare_tilelang_env() -> None:
    kernel_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "tilelang_kernels.py",
    )
    spec = importlib.util.spec_from_file_location("_dsv4_tilelang_kernels", kernel_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    if hasattr(mod, "_ensure_libz3_loadable"):
        mod._ensure_libz3_loadable()
    if hasattr(mod, "_ensure_tvm_tmpdir_writable"):
        mod._ensure_tvm_tmpdir_writable()


_prepare_tilelang_env()

_POST = importlib.import_module(
    "rtp_llm.models_py.3rdparty.tile_kernels.mhc.post_kernel"
)

M_LIST = [
    1,
    2,
    3,
    5,
    7,
    11,
    17,
    31,
    37,
    61,
    67,
    97,
    127,
    128,
    191,
    251,
    257,
    509,
    512,
    769,
    1021,
    1024,
    1531,
    2039,
    2048,
    3079,
    4093,
    4096,
    6151,
    8191,
    8192,
    12289,
    16381,
    16384,
    24593,
    32749,
    32768,
    49157,
    65521,
    65536,
]

HIDDEN = 4096
MHC = 4
EXPECTED_KERNELS_PER_ITER = 1


@contextmanager
def _env(name: str, value: str | None):
    old = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def _parse_candidates() -> List[str]:
    raw = os.environ.get("DSV4_MHC_POST_CANDIDATES", "baseline,auto,small,mid,large")
    candidates = [x.strip().lower() for x in raw.split(",") if x.strip()]
    invalid = [x for x in candidates if x not in _POST._MHC_POST_VARIANTS]
    if invalid:
        raise ValueError(f"invalid DSV4_MHC_POST_CANDIDATES values: {invalid}")
    return candidates


def _report_path() -> str:
    return (
        os.environ.get("DSV4_MHC_POST_JSON")
        or os.environ.get("PERF_JSON")
        or report_path_from_env("DSV4_MHC_POST_JSON", "dsv4_mhc_post_perf.json")
    )


def _make_inputs(m: int):
    torch.manual_seed(20260516 + m)
    x = (torch.randn(1, m, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
    residual = (
        torch.randn(1, m, MHC, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.2
    ).contiguous()
    post = torch.rand(1, m, MHC, 1, device="cuda", dtype=torch.float32).contiguous()
    comb = (torch.randn(1, m, MHC, MHC, device="cuda", dtype=torch.float32) * 0.1).contiguous()
    return x, residual, post, comb


def _stats_us(samples: List[float]) -> dict:
    samples = sorted(samples)
    p90_idx = min(len(samples) - 1, math.ceil(len(samples) * 0.90) - 1)
    return {
        "median_us": statistics.median(samples),
        "p90_us": samples[p90_idx],
        "min_us": samples[0],
        "max_us": samples[-1],
    }


def _bench_cuda_events(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    measure: int,
) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(measure):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return _stats_us(samples)


def _kernel_events_from_trace(trace_path: str) -> List[dict]:
    with open(trace_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    events = data if isinstance(data, list) else data.get("traceEvents", [])
    kernels = []
    for event in events:
        if event.get("ph") != "X" or "ts" not in event or "dur" not in event:
            continue
        if "kernel" not in str(event.get("cat", "")).lower():
            continue
        kernels.append(event)
    return kernels


def _profile_kernel(
    fn: Callable[[], torch.Tensor],
    *,
    label: str,
    trace_dir: str,
    profile_iters: int,
) -> dict:
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(trace_dir, safe_name(label) + ".json")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function(label):
            for _ in range(profile_iters):
                fn()
            torch.cuda.synchronize()
    prof.export_chrome_trace(trace_path)
    kernels = _kernel_events_from_trace(trace_path)
    kernel_names = sorted({str(event.get("name", "")) for event in kernels})
    if len(kernels) != profile_iters * EXPECTED_KERNELS_PER_ITER:
        raise AssertionError(
            f"{label}: expected {EXPECTED_KERNELS_PER_ITER} CUDA kernel per iter, "
            f"got {len(kernels) / max(profile_iters, 1):.2f}; names={kernel_names}"
        )
    if not any("mhc" in name.lower() and "post" in name.lower() for name in kernel_names):
        raise AssertionError(f"{label}: unexpected mHC post kernel names: {kernel_names}")
    kernel_sum_us = sum(float(event["dur"]) for event in kernels) / profile_iters
    return {
        "kernel_sum_us_per_iter": kernel_sum_us,
        "kernel_count": len(kernels) // profile_iters,
        "kernel_names": kernel_names,
        "trace_path": trace_path,
        "kernel_path_verified": True,
    }


def _traffic_bytes(m: int) -> int:
    x_read = m * HIDDEN * 2
    residual_read = m * MHC * HIDDEN * 2
    post_read = m * MHC * 4
    comb_read = m * MHC * MHC * 4
    out_write = m * MHC * HIDDEN * 2
    return x_read + residual_read + post_read + comb_read + out_write


def _effective_gbps(bytes_: int, median_us: float) -> float:
    if median_us <= 0:
        return 0.0
    return bytes_ / (median_us * 1e-6) / 1e9


def _make_runner(candidate: str, x, residual, post, comb, out):
    def _run():
        return _POST.mhc_post_fwd(x, residual, post, comb, out=out)

    return _run


def _check_candidate(candidate: str, baseline: torch.Tensor, got: torch.Tensor, m: int) -> None:
    if m > 1024:
        return
    torch.testing.assert_close(got, baseline, atol=2e-2, rtol=2e-2)


def _benchmark_one(
    m: int,
    candidate: str,
    *,
    trace_dir: str,
    warmup: int,
    measure: int,
    profile_iters: int,
    baseline_out: torch.Tensor | None,
) -> tuple[dict, torch.Tensor]:
    x, residual, post, comb = _make_inputs(m)
    out = torch.empty_like(residual)
    fn = _make_runner(candidate, x, residual, post, comb, out)
    with _env("DSV4_MHC_POST_VARIANT", candidate), torch.inference_mode():
        fn()
        torch.cuda.synchronize()
        selected_variant = _POST.mhc_post_last_selected_variant()
        if baseline_out is not None:
            _check_candidate(candidate, baseline_out, out, m)
        event_stats = _bench_cuda_events(fn, warmup=warmup, measure=measure)
        profile_stats = _profile_kernel(
            fn,
            label=f"mhc_post_M{m}_{candidate}_{selected_variant}",
            trace_dir=trace_dir,
            profile_iters=profile_iters,
        )

    traffic = _traffic_bytes(m)
    row = {
        "case": "mhc_post_hidden4096_mhc4",
        "m": m,
        "candidate": candidate,
        "selected_variant": selected_variant,
        "shape": [1, m, MHC, HIDDEN],
        "warmup": warmup,
        "measure": measure,
        "torch_compile_included": False,
        "compile_warmup_excluded": True,
        "allocation_outside_timed_region": True,
        "timing_source": "cuda_event",
        "pure_kernel_timing_source": "torch_profiler",
        "estimated_traffic_bytes": traffic,
        "effective_gbps": _effective_gbps(traffic, event_stats["median_us"]),
        **event_stats,
        **profile_stats,
    }
    return row, out.detach()


def _print_summary(rows: Iterable[dict]) -> None:
    print("\n[mhc_post]")
    print(
        "  {:>8} {:>9} {:>9} {:>11} {:>10} {:>9} {:>9}".format(
            "M",
            "candidate",
            "selected",
            "median_us",
            "kernel_us",
            "GB/s",
            "kernels",
        )
    )
    for row in rows:
        print(
            "  {m:8d} {candidate:>9} {selected_variant:>9} "
            "{median_us:11.3f} {kernel_sum_us_per_iter:10.3f} "
            "{effective_gbps:9.1f} {kernel_count:9d}".format(**row)
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class MHCPostPerfTest(unittest.TestCase):
    def test_perf_report(self):
        started = time.time()
        m_list = parse_int_list("DSV4_MHC_POST_M_LIST", M_LIST)
        candidates = _parse_candidates()
        warmup = int(os.environ.get("DSV4_MHC_POST_WARMUP", "30"))
        measure = int(os.environ.get("DSV4_MHC_POST_MEASURE", "100"))
        profile_iters = int(os.environ.get("DSV4_MHC_POST_PROFILE_ITERS", "10"))
        report_path = _report_path()
        trace_dir = trace_dir_from_report(report_path, "DSV4_MHC_POST_TRACE_DIR")

        rows: List[dict] = []
        invalid: List[dict] = []
        for m in m_list:
            baseline_out = None
            for candidate in candidates:
                try:
                    row, out = _benchmark_one(
                        m,
                        candidate,
                        trace_dir=trace_dir,
                        warmup=warmup,
                        measure=measure,
                        profile_iters=profile_iters,
                        baseline_out=baseline_out,
                    )
                    rows.append(row)
                    if candidate == "baseline":
                        baseline_out = out
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    invalid.append({"m": m, "candidate": candidate, "invalid_reason": f"OOM: {exc}"})
                torch.cuda.empty_cache()

        _print_summary(rows)
        payload = {
            "elapsed_sec": time.time() - started,
            "git_commit": git_commit(os.getcwd()),
            "device": device_payload(),
            "baseline_path": "DSV4_MHC_POST_VARIANT=baseline",
            "candidate_path": "DSV4_MHC_POST_VARIANT=auto|small|mid|large",
            "m_list": m_list,
            "candidates": candidates,
            "dims": {"hidden": HIDDEN, "mhc": MHC},
            "benchmark_contract": {
                "torch_compile_included": False,
                "compile_warmup_excluded": True,
                "allocation_outside_timed_region": True,
                "event_timing": "CUDA events around each invocation",
                "pure_kernel_timing": "Torch Profiler CUDA kernel event sum",
                "expected_kernel_count_per_iter": EXPECTED_KERNELS_PER_ITER,
            },
            "results": rows,
            "invalid": invalid,
        }
        write_json_report(report_path, payload)
        print(f"\nWrote mHC post perf JSON: {report_path}")
        self.assertTrue(rows)


if __name__ == "__main__":
    unittest.main()
