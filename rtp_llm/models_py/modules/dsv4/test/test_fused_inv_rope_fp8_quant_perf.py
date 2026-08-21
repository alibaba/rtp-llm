"""Performance guard for DSV4 fused inverse-RoPE + FP8 quantization."""

from __future__ import annotations

import importlib.util
import os
import time
import unittest
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Tuple

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

_KERNEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "_fused_inv_rope_fp8_quant_triton.py",
)
_SPEC = importlib.util.spec_from_file_location("_fused_inv_rope_fp8_quant_triton", _KERNEL_PATH)
_KERNEL = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_KERNEL)
fused_inv_rope_fp8_quant_legacy = _KERNEL.fused_inv_rope_fp8_quant_legacy
fused_inv_rope_fp8_quant_optimized = _KERNEL.fused_inv_rope_fp8_quant_optimized
ensure_triton_cc()


N_HEADS = 64
HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
N_GROUPS = 8
HEADS_PER_GROUP = N_HEADS // N_GROUPS
KERNEL_REGEX = r".*fused_inv_rope_fp8_quant.*"


@dataclass
class BenchCase:
    name: str
    mode: str
    M: int
    input_shape: Tuple[int, ...]


def _parse_heads_per_cta(default: Iterable[int] = (1, 2, 4, 8)) -> List[int]:
    values = parse_int_list("DSV4_INV_ROPE_HEADS_PER_CTA_LIST", default)
    invalid = [x for x in values if x not in (1, 2, 4, 8)]
    if invalid:
        raise ValueError(f"invalid HEADS_PER_CTA values: {invalid}")
    return values


def _make_freqs(rows: int) -> torch.Tensor:
    ang = torch.rand(rows, ROPE_DIM // 2, device="cuda") * 6.28
    return torch.polar(torch.ones_like(ang), ang).to(torch.complex64).contiguous()


def _make_case(case: BenchCase) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(31 + case.M + len(case.name))
    o = (torch.randn(*case.input_shape, dtype=torch.bfloat16, device="cuda") * 0.3).contiguous()
    freqs = _make_freqs(case.M)
    if case.mode == "prefill":
        o = o.view(case.M, N_HEADS, HEAD_DIM)
    return o, freqs


def _run_legacy(o: torch.Tensor, freqs: torch.Tensor):
    return fused_inv_rope_fp8_quant_legacy(
        o,
        freqs,
        n_groups=N_GROUPS,
        heads_per_group=HEADS_PER_GROUP,
        nope_dim=NOPE_DIM,
        rope_head_dim=ROPE_DIM,
    )


def _run_optimized(o: torch.Tensor, freqs: torch.Tensor, heads_per_cta: int):
    return fused_inv_rope_fp8_quant_optimized(
        o,
        freqs,
        n_groups=N_GROUPS,
        heads_per_group=HEADS_PER_GROUP,
        nope_dim=NOPE_DIM,
        rope_head_dim=ROPE_DIM,
        heads_per_cta=heads_per_cta,
    )


def _check_outputs(legacy, optimized, label: str) -> None:
    fp8_ref, scale_ref = legacy
    fp8_opt, scale_opt = optimized
    fp8_diff = (
        fp8_ref.contiguous().view(torch.uint8).to(torch.int16)
        - fp8_opt.contiguous().view(torch.uint8).to(torch.int16)
    ).abs()
    scale_diff = (
        scale_ref.contiguous().view(torch.uint8).to(torch.int16)
        - scale_opt.contiguous().view(torch.uint8).to(torch.int16)
    ).abs()
    fp8_exact = (fp8_diff == 0).float().mean().item()
    scale_exact = (scale_diff == 0).float().mean().item()
    print(
        f"[{label}] fp8_exact={fp8_exact * 100:.2f}% max_ulp={fp8_diff.max().item()} "
        f"scale_exact={scale_exact * 100:.2f}% max_scale_byte={scale_diff.max().item()}"
    )
    if fp8_exact < 0.95:
        raise AssertionError(f"{label}: fp8 exact ratio too low: {fp8_exact:.4f}")
    if scale_exact < 0.99 or scale_diff.max().item() > 1:
        raise AssertionError(f"{label}: scale mismatch")


def _build_cases(M: int) -> List[BenchCase]:
    return [
        BenchCase(f"decode_M={M}", "decode", M, (M, 1, N_HEADS, HEAD_DIM)),
        BenchCase(f"prefill_M={M}", "prefill", M, (1, M, N_HEADS, HEAD_DIM)),
    ]


def _bench_case(
    case: BenchCase,
    heads_per_cta_values: List[int],
    trace_dir: str,
    profile_enabled: bool,
) -> List[dict]:
    o, freqs = _make_case(case)
    legacy_out = _run_legacy(o, freqs)
    torch.cuda.synchronize()

    warmup = 20 if case.M <= 4096 else 8
    iters = iters_for_m(case.M)
    legacy_measure = measure_kernel(
        lambda: _run_legacy(o, freqs),
        label=f"{case.mode}_M{case.M}_legacy",
        trace_dir=trace_dir,
        kernel_regex=KERNEL_REGEX,
        warmup=warmup,
        iters=iters,
        profile_enabled=profile_enabled,
    )
    rows = [
        {
            "case": case.name,
            "mode": case.mode,
            "M": case.M,
            "shape": list(case.input_shape),
            "impl": "legacy",
            "heads_per_cta": 1,
            "baseline_impl": "legacy",
            "speedup_vs_baseline": 1.0,
            **measurement_payload(legacy_measure),
        }
    ]

    for heads_per_cta in heads_per_cta_values:
        opt_out = _run_optimized(o, freqs, heads_per_cta)
        torch.cuda.synchronize()
        _check_outputs(legacy_out, opt_out, f"{case.name}_hpc{heads_per_cta}")
        opt_measure = measure_kernel(
            lambda h=heads_per_cta: _run_optimized(o, freqs, h),
            label=f"{case.mode}_M{case.M}_optimized_hpc{heads_per_cta}",
            trace_dir=trace_dir,
            kernel_regex=KERNEL_REGEX,
            warmup=warmup,
            iters=iters,
            profile_enabled=profile_enabled,
        )
        rows.append(
            {
                "case": case.name,
                "mode": case.mode,
                "M": case.M,
                "shape": list(case.input_shape),
                "impl": "optimized",
                "heads_per_cta": heads_per_cta,
                "baseline_impl": "legacy",
                "speedup_vs_baseline": legacy_measure.kernel_span_us / opt_measure.kernel_span_us,
                **measurement_payload(opt_measure),
            }
        )
    return rows


def _print_summary(rows: List[dict]) -> None:
    print("\n[fused_inv_rope_fp8_quant]")
    print("  {:>8} {:>10} {:>10} {:>5} {:>12} {:>9}".format("M", "mode", "impl", "HPC", "span_us", "speedup"))
    for row in rows:
        print(
            "  {:8d} {:>10} {:>10} {:5d} {:12.2f} {:8.3f}x".format(
                row["M"],
                row["mode"],
                row["impl"],
                row["heads_per_cta"],
                row["kernel_span_us"],
                row["speedup_vs_baseline"],
            )
        )


def _default_hpc2_by_case(rows: List[dict]) -> Dict[Tuple[str, int], Tuple[float, float]]:
    out: Dict[Tuple[str, int], List[float]] = {}
    for row in rows:
        key = (row["mode"], row["M"])
        if row["impl"] == "legacy":
            out.setdefault(key, [0.0, 0.0])[0] = row["kernel_span_us"]
        elif row["impl"] == "optimized" and row["heads_per_cta"] == 2:
            out.setdefault(key, [0.0, 0.0])[1] = row["kernel_span_us"]
    return {k: (v[0], v[1]) for k, v in out.items() if v[0] > 0 and v[1] > 0}


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FusedInvRopeFp8QuantPerfTest(unittest.TestCase):
    def test_perf_report(self):
        started = time.time()
        m_list = parse_int_list("DSV4_INV_ROPE_M_LIST", DEFAULT_M_SWEEP)
        heads_per_cta_values = _parse_heads_per_cta()
        report_path = report_path_from_env(
            "DSV4_INV_ROPE_JSON", "dsv4_fused_inv_rope_fp8_quant_perf.json"
        )
        trace_dir = trace_dir_from_report(report_path, "DSV4_INV_ROPE_TRACE_DIR")
        profile_enabled = env_flag("DSV4_INV_ROPE_PROFILE", True)

        rows: List[dict] = []
        invalid: List[dict] = []
        for M in m_list:
            for case in _build_cases(M):
                try:
                    rows.extend(_bench_case(case, heads_per_cta_values, trace_dir, profile_enabled))
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    invalid.append({**asdict(case), "invalid_reason": f"OOM: {exc}"})
                torch.cuda.empty_cache()

        _print_summary(rows)
        payload = {
            "elapsed_sec": time.time() - started,
            "git_commit": git_commit(os.getcwd()),
            "device": device_payload(),
            "m_list": m_list,
            "heads_per_cta_values": heads_per_cta_values,
            "profile_enabled": profile_enabled,
            "dims": {
                "n_heads": N_HEADS,
                "head_dim": HEAD_DIM,
                "rope_dim": ROPE_DIM,
                "n_groups": N_GROUPS,
                "heads_per_group": HEADS_PER_GROUP,
            },
            "results": rows,
            "invalid": invalid,
        }
        write_json_report(report_path, payload)
        print(f"\nWrote fused inverse-RoPE FP8 quant perf JSON: {report_path}")

        if env_flag("DSV4_INV_ROPE_STRICT_GATE", False):
            gated = _default_hpc2_by_case(rows)
            failures = []
            for key in (
                ("decode", 1),
                ("decode", 16),
                ("decode", 256),
                ("prefill", 256),
                ("prefill", 4096),
            ):
                if key not in gated:
                    continue
                legacy_us, optimized_us = gated[key]
                limit = 0.80 if key[0] == "decode" else 0.92
                if optimized_us > legacy_us * limit:
                    failures.append(
                        f"{key}: optimized_hpc2={optimized_us:.2f}us, "
                        f"legacy={legacy_us:.2f}us"
                    )
            self.assertFalse(failures, "perf gate failed: " + "; ".join(failures))
        self.assertTrue(rows)


if __name__ == "__main__":
    unittest.main()
