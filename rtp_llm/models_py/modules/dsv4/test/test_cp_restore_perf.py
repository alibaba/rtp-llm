"""Benchmark UT for DSV4 CP restore fast path."""

from __future__ import annotations

import time
import unittest
from typing import List

import torch

from dsv4_kernel_perf_utils import (
    device_payload,
    env_flag,
    git_commit,
    measure_kernel,
    measurement_payload,
    parse_int_list,
    report_path_from_env,
    trace_dir_from_report,
    write_json_report,
)
from rtp_llm.models_py.modules.dsv4.cp import CPContext, _cp_restore_gathered_full_2d


DEFAULT_T_SWEEP = [1024, 4096, 16384, 65536]
HIDDEN = 1024


def _make_ctx(T: int, *, prefix: bool) -> CPContext:
    if prefix:
        unpad_restore = torch.arange(T, device="cuda", dtype=torch.long)
    else:
        # Deterministic non-prefix restore to keep index_select in the baseline.
        unpad_restore = torch.arange(T, device="cuda", dtype=torch.long)
        unpad_restore = torch.roll(unpad_restore, shifts=1)
    return CPContext(
        cp_size=2,
        cp_rank=0,
        chunk_length=(T + 1) // 2,
        padded_seq_len=T,
        seq_len_full=T,
        relative_positions=torch.empty(0, device="cuda", dtype=torch.long),
        prefix_length=0,
        global_positions=torch.empty(0, device="cuda", dtype=torch.long),
        local_is_real=torch.empty(0, device="cuda", dtype=torch.bool),
        unpad_restore=unpad_restore,
        seq_len_total=T,
        cp_info=object(),
        unpad_restore_is_prefix=prefix,
    )


def _bench_t(T: int, trace_dir: str, profile_enabled: bool) -> List[dict]:
    torch.manual_seed(7100 + T)
    gathered = torch.randn(T, HIDDEN, device="cuda", dtype=torch.float32)
    prefix_ctx = _make_ctx(T, prefix=True)
    gather_ctx = _make_ctx(T, prefix=False)

    prefix_out = _cp_restore_gathered_full_2d(gathered, prefix_ctx)
    torch.testing.assert_close(prefix_out, gathered, rtol=0, atol=0)
    assert prefix_out.data_ptr() == gathered.data_ptr()

    warmup = 20
    iters = 100 if T <= 4096 else 20
    prefix_measure = measure_kernel(
        lambda: _cp_restore_gathered_full_2d(gathered, prefix_ctx),
        label=f"cp_restore_T{T}_prefix_view",
        trace_dir=trace_dir,
        warmup=warmup,
        iters=iters,
        profile_enabled=profile_enabled,
    )
    gather_measure = measure_kernel(
        lambda: _cp_restore_gathered_full_2d(gathered, gather_ctx),
        label=f"cp_restore_T{T}_index_select",
        trace_dir=trace_dir,
        warmup=warmup,
        iters=iters,
        profile_enabled=profile_enabled,
    )
    common = {"T": T, "hidden": HIDDEN, "baseline_impl": "index_select"}
    return [
        {
            **common,
            "impl": "index_select",
            "speedup_vs_baseline": 1.0,
            **measurement_payload(gather_measure),
        },
        {
            **common,
            "impl": "prefix_view",
            "speedup_vs_baseline": gather_measure.event_us / max(prefix_measure.event_us, 1e-6),
            **measurement_payload(prefix_measure),
        },
    ]


def _print_summary(rows: List[dict]) -> None:
    print("\n[cp_restore]")
    print("  {:>8} {:>16} {:>12} {:>9}".format("T", "impl", "event_us", "speedup"))
    for row in rows:
        print(
            "  {:8d} {:>16} {:12.4f} {:8.3f}x".format(
                row["T"],
                row["impl"],
                row["event_us"],
                row["speedup_vs_baseline"],
            )
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CPRestorePerfTest(unittest.TestCase):
    def test_perf_report(self):
        started = time.time()
        t_list = parse_int_list("DSV4_CP_RESTORE_T_LIST", DEFAULT_T_SWEEP)
        report_path = report_path_from_env("DSV4_CP_RESTORE_JSON", "dsv4_cp_restore_perf.json")
        trace_dir = trace_dir_from_report(report_path, "DSV4_CP_RESTORE_TRACE_DIR")
        profile_enabled = env_flag("DSV4_CP_RESTORE_PROFILE", False)

        rows: List[dict] = []
        for T in t_list:
            rows.extend(_bench_t(T, trace_dir, profile_enabled))
            torch.cuda.empty_cache()

        _print_summary(rows)
        payload = {
            "elapsed_sec": time.time() - started,
            "git_commit": git_commit(),
            "device": device_payload(),
            "t_list": t_list,
            "profile_enabled": profile_enabled,
            "results": rows,
        }
        write_json_report(report_path, payload)
        print(f"\nWrote CP restore perf JSON: {report_path}")
        self.assertTrue(rows)

        if env_flag("DSV4_PERF_ASSERT", False):
            prefix_rows = [r for r in rows if r["impl"] == "prefix_view"]
            self.assertTrue(prefix_rows)
            self.assertGreaterEqual(min(r["speedup_vs_baseline"] for r in prefix_rows), 5.0)


if __name__ == "__main__":
    unittest.main()
