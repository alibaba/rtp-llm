"""Benchmark UT for the DSV4 RoPE-only indexer kernel."""

from __future__ import annotations

import importlib.util
import os
import time
import unittest
from typing import List

import torch

from dsv4_kernel_perf_utils import (
    DEFAULT_M_SWEEP,
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
HEAD_DIM = 128
ROPE_DIM = 64
ROPE_ONLY_KERNEL_REGEX = r".*rope_only.*"


def _load_rope_only():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.abspath(os.path.join(here, "..", "_rope_only_triton.py"))
    spec = importlib.util.spec_from_file_location("_rope_only_triton", src)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.rope_only_inplace


def _make_freqs(rows: int) -> torch.Tensor:
    angle = torch.rand(rows, ROPE_DIM // 2, device="cuda") * 6.28
    return torch.polar(torch.ones_like(angle), angle).to(torch.complex64).contiguous()


def _eager_apply_rope_inplace(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    y = x
    xc = torch.view_as_complex(x.float().unflatten(-1, (x.size(-1) // 2, 2)))
    freqs = freqs_cis.view(xc.size(0), xc.size(1), 1, xc.size(-1))
    out = torch.view_as_real(xc * freqs).flatten(-2)
    y.copy_(out)
    return y


def _bench_m(M: int, trace_dir: str, profile_enabled: bool) -> List[dict]:
    torch.manual_seed(9000 + M)
    x_eager = torch.randn(1, M, N_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    x_rope = x_eager.clone()
    freqs = _make_freqs(M)
    rope_only = _load_rope_only()

    if M <= 1024:
        ref = x_eager.clone()
        cand = x_rope.clone()
        _eager_apply_rope_inplace(ref[..., -ROPE_DIM:], freqs)
        rope_only(cand[..., -ROPE_DIM:], freqs)
        torch.cuda.synchronize()
        torch.testing.assert_close(cand, ref, rtol=0, atol=3e-2)

    warmup = 20 if M <= 4096 else 8
    iters = iters_for_m(M)
    eager = lambda: _eager_apply_rope_inplace(x_eager[..., -ROPE_DIM:], freqs)
    fused = lambda: rope_only(x_rope[..., -ROPE_DIM:], freqs)

    eager_measure = measure_kernel(
        eager,
        label=f"rope_only_M{M}_eager",
        trace_dir=trace_dir,
        warmup=warmup,
        iters=iters,
        profile_enabled=profile_enabled,
    )
    fused_measure = measure_kernel(
        fused,
        label=f"rope_only_M{M}_triton",
        trace_dir=trace_dir,
        kernel_regex=ROPE_ONLY_KERNEL_REGEX,
        warmup=warmup,
        iters=iters,
        profile_enabled=profile_enabled,
    )
    speedup = eager_measure.kernel_span_us / fused_measure.kernel_span_us
    common = {
        "M": M,
        "shape": [1, M, N_HEADS, HEAD_DIM],
        "rope_dim": ROPE_DIM,
        "baseline_impl": "eager_complex",
    }
    return [
        {
            **common,
            "impl": "eager_complex",
            "speedup_vs_baseline": 1.0,
            **measurement_payload(eager_measure),
        },
        {
            **common,
            "impl": "rope_only_triton",
            "speedup_vs_baseline": speedup,
            **measurement_payload(fused_measure),
        },
    ]


def _print_summary(rows: List[dict]) -> None:
    print("\n[rope_only]")
    print("  {:>8} {:>18} {:>12} {:>9}".format("M", "impl", "span_us", "speedup"))
    for row in rows:
        print(
            "  {:8d} {:>18} {:12.2f} {:8.3f}x".format(
                row["M"],
                row["impl"],
                row["kernel_span_us"],
                row["speedup_vs_baseline"],
            )
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class RopeOnlyPerfTest(unittest.TestCase):
    def test_perf_report(self):
        started = time.time()
        m_list = parse_int_list("DSV4_ROPE_ONLY_M_LIST", DEFAULT_M_SWEEP)
        report_path = report_path_from_env("DSV4_ROPE_ONLY_JSON", "dsv4_rope_only_perf.json")
        trace_dir = trace_dir_from_report(report_path, "DSV4_ROPE_ONLY_TRACE_DIR")
        profile_enabled = env_flag("DSV4_ROPE_ONLY_PROFILE", True)

        rows: List[dict] = []
        for M in m_list:
            rows.extend(_bench_m(M, trace_dir, profile_enabled))
            torch.cuda.empty_cache()

        _print_summary(rows)
        payload = {
            "elapsed_sec": time.time() - started,
            "git_commit": git_commit(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")),
            "device": device_payload(),
            "m_list": m_list,
            "profile_enabled": profile_enabled,
            "results": rows,
        }
        write_json_report(report_path, payload)
        print(f"\nWrote RoPE-only perf JSON: {report_path}")
        self.assertTrue(rows)

        if env_flag("DSV4_PERF_ASSERT", False):
            fused_rows = [r for r in rows if r["impl"] == "rope_only_triton" and r["M"] >= 1024]
            self.assertTrue(fused_rows)
            min_speedup = min(r["speedup_vs_baseline"] for r in fused_rows)
            self.assertGreaterEqual(min_speedup, 2.0)


if __name__ == "__main__":
    unittest.main()
