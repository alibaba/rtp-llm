"""Performance benchmark for DSV4 fused RMSNorm + partial RoPE."""

from __future__ import annotations

import importlib.util
import os
import time
import unittest
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional, Tuple

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
    "_fused_rmsnorm_rope_triton.py",
)
_SPEC = importlib.util.spec_from_file_location("_fused_rmsnorm_rope_triton", _KERNEL_PATH)
_KERNEL = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_KERNEL)
fused_rmsnorm_rope = _KERNEL.fused_rmsnorm_rope
ensure_triton_cc()


N_HEADS = 64
Q_HEAD_DIM = 128
KV_HEAD_DIM = 512
ROPE_DIM = 64
FUSED_KERNEL_REGEX = r".*fused_rmsnorm_rope.*"


def _group_heads_list() -> List[int]:
    values = parse_int_list("DSV4_RMSNORM_ROPE_GROUP_HEADS_LIST", (1, 2, 4, 8))
    invalid = [x for x in values if x not in (1, 2, 4, 8)]
    if invalid:
        raise ValueError(f"invalid GROUP_HEADS values: {invalid}")
    return values


@dataclass
class BenchCase:
    name: str
    path: str
    M: int
    shape: Tuple[int, ...]
    has_weight: bool
    inverse: bool


def _make_freqs(rows: int) -> torch.Tensor:
    angle = torch.rand(rows, ROPE_DIM // 2, device="cuda") * 6.28
    return torch.polar(torch.ones_like(angle), angle).to(torch.complex64).contiguous()


def _make_case(case: BenchCase) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    torch.manual_seed(17 + case.M + len(case.name))
    x = (torch.randn(*case.shape, dtype=torch.bfloat16, device="cuda") * 0.3).contiguous()
    weight = None
    if case.has_weight:
        weight = torch.rand(case.shape[-1], dtype=torch.bfloat16, device="cuda") + 0.5
    freqs = _make_freqs(case.M)
    return x, weight, freqs


def _apply_rope_tail(y: torch.Tensor, freqs: torch.Tensor, rope_dim: int, inverse: bool) -> torch.Tensor:
    flat = y.view(-1, y.shape[-1])
    num_rows = flat.shape[0]
    freqs_flat = freqs.view(-1, freqs.shape[-1])
    assert num_rows % freqs_flat.shape[0] == 0
    freq_stride = num_rows // freqs_flat.shape[0]
    freq_idx = torch.arange(num_rows, device=y.device) // freq_stride

    tail = flat[:, -rope_dim:].float().view(num_rows, rope_dim // 2, 2)
    real = tail[..., 0]
    imag = tail[..., 1]
    cos = freqs_flat.real.index_select(0, freq_idx)
    sin = freqs_flat.imag.index_select(0, freq_idx)
    if inverse:
        sin = -sin
    out = torch.empty_like(tail)
    out[..., 0] = real * cos - imag * sin
    out[..., 1] = real * sin + imag * cos
    flat[:, -rope_dim:] = out.view(num_rows, rope_dim).to(y.dtype)
    return y


def _run_eager(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    freqs: torch.Tensor,
    *,
    inverse: bool,
) -> torch.Tensor:
    x32 = x.float()
    inv = torch.rsqrt(x32.square().mean(-1, keepdim=True) + 1e-6)
    y = x32 * inv
    if weight is not None:
        y = y * weight.float()
    out = y.to(x.dtype).contiguous()
    return _apply_rope_tail(out, freqs, ROPE_DIM, inverse)


def _run_fused(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    freqs: torch.Tensor,
    *,
    inverse: bool,
) -> torch.Tensor:
    return fused_rmsnorm_rope(x, weight, freqs, ROPE_DIM, inverse=inverse)


def _build_cases(M: int) -> List[BenchCase]:
    return [
        BenchCase("q_decode_no_weight", "q", M, (M, 1, N_HEADS, Q_HEAD_DIM), False, False),
        BenchCase("q_prefill_no_weight", "q", M, (1, M, N_HEADS, Q_HEAD_DIM), False, False),
        BenchCase("q_prefill_inverse", "q", M, (1, M, N_HEADS, Q_HEAD_DIM), False, True),
        BenchCase("kv_decode_weight", "kv", M, (M, 1, KV_HEAD_DIM), True, False),
        BenchCase("kv_prefill_weight", "kv", M, (1, M, KV_HEAD_DIM), True, False),
        BenchCase("kv_prefill_no_weight_inverse", "kv", M, (1, M, KV_HEAD_DIM), False, True),
    ]


def _check_correctness(case: BenchCase, x: torch.Tensor, weight: Optional[torch.Tensor], freqs: torch.Tensor) -> None:
    ref = _run_eager(x, weight, freqs, inverse=case.inverse)
    cand = _run_fused(x, weight, freqs, inverse=case.inverse)
    torch.cuda.synchronize()
    diff = (ref.float() - cand.float()).abs()
    max_abs = float(diff.max().item())
    if max_abs > 5e-2:
        raise AssertionError(f"{case.name} M={case.M} max_abs={max_abs:.4e}")

    out = torch.empty_like(x)
    cand_out = fused_rmsnorm_rope(
        x, weight, freqs, ROPE_DIM, inverse=case.inverse, out=out
    )
    inplace_x = x.clone()
    cand_inplace = fused_rmsnorm_rope(
        inplace_x, weight, freqs, ROPE_DIM, inverse=case.inverse, inplace=True
    )
    out_diff = (ref.float() - cand_out.float()).abs().max().item()
    inplace_diff = (ref.float() - cand_inplace.float()).abs().max().item()
    if out_diff > 5e-2 or inplace_diff > 5e-2:
        raise AssertionError(
            f"{case.name} M={case.M} output mode mismatch: "
            f"out={out_diff:.4e} inplace={inplace_diff:.4e}"
        )
    if case.path == "q":
        for group_heads in _group_heads_list():
            if group_heads == 1:
                continue
            cand_grouped = fused_rmsnorm_rope(
                x,
                weight,
                freqs,
                ROPE_DIM,
                inverse=case.inverse,
                group_heads=group_heads,
            )
            grouped_diff = (ref.float() - cand_grouped.float()).abs().max().item()
            if grouped_diff > 5e-2:
                raise AssertionError(
                    f"{case.name} M={case.M} group_heads={group_heads} "
                    f"mismatch: {grouped_diff:.4e}"
                )
            grouped_inplace_x = x.clone()
            cand_grouped_inplace = fused_rmsnorm_rope(
                grouped_inplace_x,
                weight,
                freqs,
                ROPE_DIM,
                inverse=case.inverse,
                group_heads=group_heads,
                inplace=True,
            )
            grouped_inplace_diff = (
                ref.float() - cand_grouped_inplace.float()
            ).abs().max().item()
            if grouped_inplace_diff > 5e-2:
                raise AssertionError(
                    f"{case.name} M={case.M} group_heads={group_heads} inplace "
                    f"mismatch: {grouped_inplace_diff:.4e}"
                )

def _bench_case(case: BenchCase, report_trace_dir: str, profile_enabled: bool) -> List[dict]:
    x, weight, freqs = _make_case(case)
    if case.M <= 256:
        _check_correctness(case, x, weight, freqs)

    warmup = 20 if case.M <= 4096 else 8
    iters = iters_for_m(case.M)
    eager: Callable[[], torch.Tensor] = lambda: _run_eager(x, weight, freqs, inverse=case.inverse)
    fused: Callable[[], torch.Tensor] = lambda: _run_fused(x, weight, freqs, inverse=case.inverse)
    out_buf = torch.empty_like(x)
    inplace_x = x.clone()
    fused_out: Callable[[], torch.Tensor] = lambda: fused_rmsnorm_rope(
        x, weight, freqs, ROPE_DIM, inverse=case.inverse, out=out_buf
    )
    fused_inplace: Callable[[], torch.Tensor] = lambda: fused_rmsnorm_rope(
        inplace_x, weight, freqs, ROPE_DIM, inverse=case.inverse, inplace=True
    )

    eager_measure = measure_kernel(
        eager,
        label=f"{case.name}_M{case.M}_eager",
        trace_dir=report_trace_dir,
        warmup=warmup,
        iters=iters,
        profile_enabled=profile_enabled,
    )
    fused_measure = measure_kernel(
        fused,
        label=f"{case.name}_M{case.M}_fused",
        trace_dir=report_trace_dir,
        kernel_regex=FUSED_KERNEL_REGEX,
        warmup=warmup,
        iters=iters,
        profile_enabled=profile_enabled,
    )
    fused_out_measure = measure_kernel(
        fused_out,
        label=f"{case.name}_M{case.M}_fused_out",
        trace_dir=report_trace_dir,
        kernel_regex=FUSED_KERNEL_REGEX,
        warmup=warmup,
        iters=iters,
        profile_enabled=profile_enabled,
    )
    fused_inplace_measure = measure_kernel(
        fused_inplace,
        label=f"{case.name}_M{case.M}_fused_inplace",
        trace_dir=report_trace_dir,
        kernel_regex=FUSED_KERNEL_REGEX,
        warmup=warmup,
        iters=iters,
        profile_enabled=profile_enabled,
    )
    grouped_rows = []
    if case.path == "q":
        for group_heads in _group_heads_list():
            if group_heads == 1:
                continue
            grouped_measure = measure_kernel(
                lambda gh=group_heads: fused_rmsnorm_rope(
                    x, weight, freqs, ROPE_DIM, inverse=case.inverse, group_heads=gh
                ),
                label=f"{case.name}_M{case.M}_fused_group_heads_{group_heads}",
                trace_dir=report_trace_dir,
                kernel_regex=FUSED_KERNEL_REGEX,
                warmup=warmup,
                iters=iters,
                profile_enabled=profile_enabled,
            )
            grouped_rows.append(
                {
                    "group_heads": group_heads,
                    "impl": f"fused_triton_group_heads_{group_heads}",
                    "speedup_vs_baseline": eager_measure.kernel_span_us
                    / grouped_measure.kernel_span_us,
                    **measurement_payload(grouped_measure),
                }
            )
    speedup = eager_measure.kernel_span_us / fused_measure.kernel_span_us
    common = {
        "case": case.name,
        "path": case.path,
        "M": case.M,
        "shape": list(case.shape),
        "has_weight": case.has_weight,
        "inverse": case.inverse,
        "rope_dim": ROPE_DIM,
        "baseline_impl": "eager",
    }
    return [
        {
            **common,
            "impl": "eager",
            "speedup_vs_baseline": 1.0,
            **measurement_payload(eager_measure),
        },
        {
            **common,
            "impl": "fused_triton",
            "speedup_vs_baseline": speedup,
            **measurement_payload(fused_measure),
        },
        {
            **common,
            "impl": "fused_triton_out",
            "speedup_vs_baseline": eager_measure.kernel_span_us
            / fused_out_measure.kernel_span_us,
            **measurement_payload(fused_out_measure),
        },
        {
            **common,
            "impl": "fused_triton_inplace",
            "speedup_vs_baseline": eager_measure.kernel_span_us
            / fused_inplace_measure.kernel_span_us,
            **measurement_payload(fused_inplace_measure),
        },
    ] + [{**common, **row} for row in grouped_rows]


def _print_summary(rows: List[dict]) -> None:
    print("\n[fused_rmsnorm_rope]")
    print("  {:>8} {:>28} {:>14} {:>12} {:>9}".format("M", "case", "impl", "span_us", "speedup"))
    for row in rows:
        print(
            "  {:8d} {:>28} {:>14} {:12.2f} {:8.3f}x".format(
                row["M"],
                row["case"],
                row["impl"],
                row["kernel_span_us"],
                row["speedup_vs_baseline"],
            )
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FusedRmsnormRopePerfTest(unittest.TestCase):
    def test_perf_report(self):
        started = time.time()
        m_list = parse_int_list("DSV4_RMSNORM_ROPE_M_LIST", DEFAULT_M_SWEEP)
        report_path = report_path_from_env(
            "DSV4_RMSNORM_ROPE_JSON", "dsv4_fused_rmsnorm_rope_perf.json"
        )
        trace_dir = trace_dir_from_report(report_path, "DSV4_RMSNORM_ROPE_TRACE_DIR")
        profile_enabled = env_flag("DSV4_RMSNORM_ROPE_PROFILE", True)

        rows: List[dict] = []
        invalid: List[dict] = []
        for M in m_list:
            for case in _build_cases(M):
                try:
                    rows.extend(_bench_case(case, trace_dir, profile_enabled))
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    invalid.append({**asdict(case), "invalid_reason": f"OOM: {exc}"})
                torch.cuda.empty_cache()

        _print_summary(rows)
        payload = {
            "elapsed_sec": time.time() - started,
            "git_commit": git_commit(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")),
            "device": device_payload(),
            "m_list": m_list,
            "profile_enabled": profile_enabled,
            "results": rows,
            "invalid": invalid,
        }
        write_json_report(report_path, payload)
        print(f"\nWrote fused RMSNorm/RoPE perf JSON: {report_path}")
        self.assertTrue(rows)


if __name__ == "__main__":
    unittest.main()
