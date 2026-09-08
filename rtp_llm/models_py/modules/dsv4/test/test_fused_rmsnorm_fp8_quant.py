"""Correctness test for DSV4 fused RMSNorm + FP8 UE8M0 quantization."""

from __future__ import annotations

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_fp8_quant_triton import (
    rmsnorm_fp8_quant_ue8m0,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


def _cpp_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    out = torch.empty_like(x)
    rtp_llm_ops.rmsnorm(out, x, weight, eps, torch.cuda.current_stream().cuda_stream)
    return out


def _raw_fp8_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (
        a.contiguous().view(torch.uint8).to(torch.int16)
        - b.contiguous().view(torch.uint8).to(torch.int16)
    ).abs()


def _raw_scale_byte_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (
        a.contiguous().view(torch.uint8).to(torch.int16)
        - b.contiguous().view(torch.uint8).to(torch.int16)
    ).abs()


def _assert_matches(m: int, n: int) -> None:
    eps = 1.0e-6
    group_size = 128
    torch.manual_seed(20260703 + m * 17 + n)
    x = (torch.randn((m, n), device="cuda", dtype=torch.bfloat16) * 0.3).contiguous()
    weight = (
        (torch.randn((n,), device="cuda") * 0.1 + 1.0)
        .abs()
        .to(torch.bfloat16)
        .contiguous()
    )

    ref_norm = _cpp_rmsnorm(x, weight, eps)
    ref_q, ref_s = sgl_per_token_group_quant_fp8(
        ref_norm,
        group_size=group_size,
        eps=1.0e-4,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    got_norm, got_q, got_s = rmsnorm_fp8_quant_ue8m0(
        x, weight, eps=eps, group_size=group_size, clamp_eps=1.0e-4
    )
    torch.cuda.synchronize()

    norm_abs = (ref_norm.float() - got_norm.float()).abs()
    norm_bit_diff = (
        ref_norm.contiguous().view(torch.int16) != got_norm.contiguous().view(torch.int16)
    ).sum().item()
    q_diff = _raw_fp8_diff(ref_q, got_q)
    s_diff = _raw_scale_byte_diff(ref_s, got_s)

    total = ref_norm.numel()
    norm_exact = 1.0 - norm_bit_diff / total
    q_exact = (q_diff == 0).float().mean().item()
    s_exact = (s_diff == 0).float().mean().item()
    print(
        f"[m={m} n={n}] "
        f"norm_max={float(norm_abs.max()):.8f} norm_exact={norm_exact:.8f} "
        f"q_max_ulp={int(q_diff.max())} q_exact={q_exact:.8f} "
        f"s_max_byte={int(s_diff.max())} s_exact={s_exact:.8f}"
    )

    assert float(norm_abs.max()) <= 0.03125
    assert norm_exact >= 0.9999
    assert q_exact >= 0.99999
    assert int(q_diff.max()) <= 1
    assert s_exact >= 0.99999
    assert int(s_diff.max()) <= 1


def test_small_real_hidden() -> None:
    _assert_matches(1, 7168)


def test_prefill_real_hidden() -> None:
    _assert_matches(384, 7168)


def test_long_prefill_real_hidden() -> None:
    _assert_matches(4096, 7168)


if __name__ == "__main__":
    test_small_real_hidden()
    test_prefill_real_hidden()
    test_long_prefill_real_hidden()
    print("OK")
