"""Kernel-only benchmark for DSV4 indexer Q FP8 quant + weight fold."""

from __future__ import annotations

import importlib.util
import os
import time
import unittest
from typing import List

import torch
import triton

from dsv4_kernel_perf_utils import (
    device_payload,
    env_flag,
    ensure_triton_cc,
    git_commit,
    iters_for_m,
    measure_kernel,
    measurement_payload,
    parse_int_list,
    report_path_from_env,
    trace_dir_from_report,
    write_json_report,
)


ensure_triton_cc()

N_HEADS = 64
KERNEL_REGEX = r".*indexer_q_fp8_fold.*"
M_SWEEP = [
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


def _load_indexer_q_quant_module():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.abspath(os.path.join(here, "..", "fp8", "_indexer_q_quant_triton.py"))
    spec = importlib.util.spec_from_file_location("_indexer_q_quant_triton", src)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_KERNEL_MOD = _load_indexer_q_quant_module()
DEFAULT_GROUP_HEADS = _KERNEL_MOD.DEFAULT_GROUP_HEADS
FP8_E4M3_MAX = _KERNEL_MOD.FP8_E4M3_MAX
INDEXER_HEAD_DIM = _KERNEL_MOD.INDEXER_HEAD_DIM
_indexer_q_fp8_fold_kernel = _KERNEL_MOD._indexer_q_fp8_fold_kernel
_indexer_q_fp8_fold_group_heads_kernel = (
    _KERNEL_MOD._indexer_q_fp8_fold_group_heads_kernel
)


def _parse_group_heads_list() -> List[int]:
    values = parse_int_list("DSV4_INDEXER_Q_FP8_GROUP_HEADS_LIST", (1, 2, 4, 8))
    invalid = [v for v in values if v not in (1, 2, 4, 8)]
    if invalid:
        raise ValueError(f"invalid GROUP_HEADS values: {invalid}")
    return values


def _traffic_bytes(M: int, weight_dtype: torch.dtype) -> int:
    weight_bytes = 2 if weight_dtype == torch.bfloat16 else 4
    return M * N_HEADS * (INDEXER_HEAD_DIM * 2 + INDEXER_HEAD_DIM + weight_bytes + 4)


def _launch(
    q: torch.Tensor,
    weights: torch.Tensor,
    q_fp8: torch.Tensor,
    w_fold: torch.Tensor,
    group_heads: int,
) -> None:
    M = q.shape[1]
    BSH = M * N_HEADS
    if group_heads == 1:
        _indexer_q_fp8_fold_kernel[(BSH,)](
            q,
            weights,
            q_fp8,
            w_fold,
            BSH=BSH,
            D=INDEXER_HEAD_DIM,
            fp8_max=FP8_E4M3_MAX,
            num_warps=4,
        )
    else:
        _indexer_q_fp8_fold_group_heads_kernel[
            (M, triton.cdiv(N_HEADS, group_heads))
        ](
            q,
            weights,
            q_fp8,
            w_fold,
            H=N_HEADS,
            D=INDEXER_HEAD_DIM,
            fp8_max=FP8_E4M3_MAX,
            GROUP_HEADS=group_heads,
            num_warps=4,
        )


def _bench_m(
    M: int,
    *,
    group_heads_list: List[int],
    weight_dtype: torch.dtype,
    trace_dir: str,
    profile_enabled: bool,
) -> List[dict]:
    torch.manual_seed(2026 + M)
    q = torch.randn(
        1,
        M,
        N_HEADS,
        INDEXER_HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    ).contiguous()
    weights = torch.randn(1, M, N_HEADS, dtype=weight_dtype, device="cuda").contiguous()

    rows: List[dict] = []
    traffic_bytes = _traffic_bytes(M, weight_dtype)
    warmup = 30 if M <= 4096 else 10
    iters = iters_for_m(M)
    baseline_span_us = None

    for group_heads in group_heads_list:
        q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
        w_fold = torch.empty(1, M, N_HEADS, dtype=torch.float32, device="cuda")
        fn = lambda gh=group_heads: _launch(q, weights, q_fp8, w_fold, gh)
        measurement = measure_kernel(
            fn,
            label=f"indexer_q_fp8_quant_M{M}_gh{group_heads}_{weight_dtype}",
            trace_dir=trace_dir,
            kernel_regex=KERNEL_REGEX,
            warmup=warmup,
            iters=iters,
            profile_enabled=profile_enabled,
        )
        if group_heads == 1:
            baseline_span_us = measurement.kernel_span_us
        speedup = (
            baseline_span_us / measurement.kernel_span_us
            if baseline_span_us
            else 1.0
        )
        rows.append(
            {
                "M": M,
                "shape": [1, M, N_HEADS, INDEXER_HEAD_DIM],
                "weight_dtype": str(weight_dtype),
                "impl": f"group_heads_{group_heads}",
                "group_heads": group_heads,
                "baseline_impl": "group_heads_1",
                "traffic_bytes": traffic_bytes,
                "effective_gbps": traffic_bytes / measurement.kernel_span_us / 1000.0,
                "speedup_vs_baseline": speedup,
                **measurement_payload(measurement),
            }
        )
    return rows


def _print_summary(rows: List[dict]) -> None:
    print("\n[indexer_q_fp8_quant]")
    print(
        "  {:>8} {:>8} {:>14} {:>12} {:>12} {:>9}".format(
            "M", "GH", "kernel_span", "kernel_sum", "GB/s", "speedup"
        )
    )
    for row in rows:
        print(
            "  {:8d} {:8d} {:14.3f} {:12.3f} {:12.1f} {:8.3f}x".format(
                row["M"],
                row["group_heads"],
                row["kernel_span_us"],
                row["kernel_sum_us"],
                row["effective_gbps"],
                row["speedup_vs_baseline"],
            )
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class IndexerQFP8QuantPerfTest(unittest.TestCase):
    def test_perf_report(self):
        started = time.time()
        m_list = parse_int_list("DSV4_INDEXER_Q_FP8_M_LIST", M_SWEEP)
        group_heads_list = _parse_group_heads_list()
        report_path = report_path_from_env(
            "DSV4_INDEXER_Q_FP8_JSON", "dsv4_indexer_q_fp8_quant_perf.json"
        )
        trace_dir = trace_dir_from_report(report_path, "DSV4_INDEXER_Q_FP8_TRACE_DIR")
        profile_enabled = env_flag("DSV4_INDEXER_Q_FP8_PROFILE", True)

        rows: List[dict] = []
        for M in m_list:
            rows.extend(
                _bench_m(
                    M,
                    group_heads_list=group_heads_list,
                    weight_dtype=torch.bfloat16,
                    trace_dir=trace_dir,
                    profile_enabled=profile_enabled,
                )
            )
            torch.cuda.empty_cache()

        _print_summary(rows)
        payload = {
            "elapsed_sec": time.time() - started,
            "git_commit": git_commit(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
            ),
            "device": device_payload(),
            "m_list": m_list,
            "group_heads_list": group_heads_list,
            "profile_enabled": profile_enabled,
            "metric_for_acceptance": "kernel_span_us",
            "results": rows,
        }
        write_json_report(report_path, payload)
        print(f"\nWrote indexer-Q FP8 quant perf JSON: {report_path}")
        self.assertTrue(rows)

        if env_flag("DSV4_INDEXER_Q_FP8_PERF_ASSERT", False):
            default_rows = [r for r in rows if r["group_heads"] == DEFAULT_GROUP_HEADS]
            small_rows = [r for r in default_rows if r["M"] <= 17]
            self.assertTrue(small_rows)
            self.assertLessEqual(max(r["kernel_span_us"] for r in small_rows), 3.0)
            large_rows = [r for r in default_rows if r["M"] >= 16384]
            self.assertTrue(large_rows)
            self.assertGreaterEqual(min(r["effective_gbps"] for r in large_rows), 1000.0)


if __name__ == "__main__":
    unittest.main()
