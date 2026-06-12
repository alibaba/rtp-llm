"""Benchmark and correctness test: old path (mxfp8_quant_act + pack_mxfp8_scale) vs new path (_mxfp8_quant_act_v2).

Old path (production before optimization):
    q, s_fp32 = mxfp8_quant_act(x)        # fp32 power-of-two scale
    s_packed = pack_mxfp8_scale(s_fp32)   # int32 packed for DeepGEMM

New path (optimized):
    q, s_packed = _mxfp8_quant_act_v2(x)  # directly outputs int32 packed scale

Usage:
    python rtp_llm/models_py/kernels/cuda/test/test_mxfp8_quant_benchmark.py

Requirements:
    - CUDA GPU (SM89+ for v2 kernel, SM100+ for DeepGEMM GEMM)
    - rtp_llm package importable
"""

import sys
from typing import Tuple

import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MX_BLOCK = 32


def _old_path(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Old production path: mxfp8_quant_act + pack_mxfp8_scale."""
    from rtp_llm.models_py.kernels.cuda.mxfp8_ops import (
        mxfp8_quant_act,
        pack_mxfp8_scale,
    )

    M, K = x.shape
    q, s_fp32 = mxfp8_quant_act(x)
    s_packed = pack_mxfp8_scale(s_fp32, mn=M, k=K)
    return q, s_packed


def _new_path(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """New optimized path: _mxfp8_quant_act_v2 directly outputs packed int32."""
    from rtp_llm.models_py.kernels.cuda.mxfp8_ops import _mxfp8_quant_act_v2

    return _mxfp8_quant_act_v2(x)


def _unpack_int32_to_fp32(s_packed: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Unpack int32 packed UE8M0 scales to fp32 power-of-two for comparison."""
    s_u8 = s_packed.contiguous().view(torch.uint8)
    s_u8 = s_u8[:M, : K // MX_BLOCK]
    return torch.exp2(s_u8.to(torch.float32) - 127.0)


def _dequant(
    q: torch.Tensor, scale: torch.Tensor, block: int = MX_BLOCK
) -> torch.Tensor:
    """Dequantize e4m3 + scale back to fp32 for comparison."""
    M, K = q.shape
    q_fp32 = q.to(torch.float32).view(M, K // block, block)
    return (q_fp32 * scale.unsqueeze(-1)).view(M, K)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


def test_correctness(M: int, K: int, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 2.0

    q_old, s_old_packed = _old_path(x)
    q_new, s_new_packed = _new_path(x)

    # 1) e4m3 bit-exact comparison
    q_old_bytes = q_old.view(torch.uint8)
    q_new_bytes = q_new.view(torch.uint8)
    q_mismatch_count = (q_old_bytes != q_new_bytes).sum().item()
    q_total = q_old.numel()
    q_mismatch_pct = 100.0 * q_mismatch_count / q_total

    # 2) Packed scale comparison — unpack both to fp32 for readable diff
    s_old_fp32 = _unpack_int32_to_fp32(s_old_packed, M, K)
    s_new_fp32 = _unpack_int32_to_fp32(s_new_packed, M, K)
    scale_max_diff = (s_old_fp32 - s_new_fp32).abs().max().item()
    scale_match = torch.allclose(s_old_fp32, s_new_fp32, atol=0, rtol=0)

    # 3) Dequantized value comparison
    x_old = _dequant(q_old, s_old_fp32)
    x_new = _dequant(q_new, s_new_fp32)
    dequant_max_diff = (x_old - x_new).abs().max().item()
    dequant_mean_diff = (x_old - x_new).abs().mean().item()

    # 4) Relative error vs original input
    old_err = (x_old - x).abs()
    new_err = (x_new - x).abs()
    old_max_rel_err = (old_err / x.abs().clamp(min=1e-6)).max().item()
    new_max_rel_err = (new_err / x.abs().clamp(min=1e-6)).max().item()

    return {
        "M": M,
        "K": K,
        "scale_max_diff": scale_max_diff,
        "scale_exact_match": scale_match,
        "q_mismatch_count": q_mismatch_count,
        "q_total": q_total,
        "q_mismatch_pct": f"{q_mismatch_pct:.4f}%",
        "dequant_max_diff": dequant_max_diff,
        "dequant_mean_diff": dequant_mean_diff,
        "old_max_rel_err": f"{old_max_rel_err:.6f}",
        "new_max_rel_err": f"{new_max_rel_err:.6f}",
    }


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def _benchmark(fn, x: torch.Tensor, warmup: int = 50, iters: int = 200) -> float:
    """Returns average time in microseconds."""
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters  # ms → us


def benchmark(M: int, K: int) -> dict:
    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    t_old = _benchmark(_old_path, x)
    t_new = _benchmark(_new_path, x)
    speedup = t_old / t_new if t_new > 0 else float("inf")

    return {
        "M": M,
        "K": K,
        "numel": M * K,
        "old_us": f"{t_old:.1f}",
        "new_us": f"{t_new:.1f}",
        "speedup": f"{speedup:.2f}x",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"Device: {device_name} (SM{cap[0]}{cap[1]})")
    print(f"PyTorch: {torch.__version__}")
    print()

    # ---- Correctness ----
    print("=" * 80)
    print("CORRECTNESS TEST")
    print("=" * 80)

    # MiniMax-M3 shapes: hidden=6144, moe_inter=3072 (gate_up: 6144, down: 3072)
    # Decode scenarios (small batch)
    decode_shapes = [
        (1, 6144, "decode batch=1 hidden"),
        (1, 3072, "decode batch=1 MoE down"),
        (1, 12288, "decode batch=1 gate_up"),
        (2, 6144, "decode batch=2 hidden"),
        (2, 3072, "decode batch=2 MoE down"),
        (2, 12288, "decode batch=2 gate_up"),
        (4, 6144, "decode batch=4 hidden"),
        (4, 3072, "decode batch=4 MoE down"),
        (8, 6144, "decode batch=8 hidden"),
        (8, 3072, "decode batch=8 MoE down"),
        (16, 6144, "decode batch=16 hidden"),
        (32, 6144, "decode batch=32 hidden"),
        (64, 6144, "decode batch=64 hidden"),
    ]

    # Prefill scenarios (large batch)
    prefill_shapes = [
        (32, 6144, "small prefill hidden"),
        (128, 6144, "medium prefill hidden"),
        (256, 6144, "large prefill hidden"),
        (512, 6144, "xlarge prefill hidden"),
        (1024, 6144, "xxlarge prefill hidden"),
        (2048, 6144, "huge prefill hidden"),
        (4096, 6144, "massive prefill hidden"),
        (128, 3072, "prefill MoE down"),
        (512, 3072, "prefill MoE down large"),
        (1024, 3072, "prefill MoE down xlarge"),
        (256, 12288, "prefill gate_up"),
        (512, 12288, "prefill gate_up large"),
    ]

    shapes = decode_shapes + prefill_shapes

    for M, K, desc in shapes:
        try:
            result = test_correctness(M, K)
            status = (
                "✓"
                if result["scale_exact_match"] and result["q_mismatch_count"] == 0
                else "⚠"
            )
            print(f"\n{status} M={M:>5}, K={K:>5} ({desc})")
            print(
                f"    scale: exact_match={result['scale_exact_match']}, max_diff={result['scale_max_diff']:.2e}"
            )
            print(
                f"    e4m3:  mismatch={result['q_mismatch_count']}/{result['q_total']} ({result['q_mismatch_pct']})"
            )
            print(
                f"    dequant: max_diff={result['dequant_max_diff']:.4e}, mean_diff={result['dequant_mean_diff']:.4e}"
            )
            print(
                f"    max_rel_err: old={result['old_max_rel_err']}, new={result['new_max_rel_err']}"
            )
        except Exception as e:
            print(f"\n✗ M={M:>5}, K={K:>5} ({desc}): {e}")

    # ---- Performance ----
    print()
    print("=" * 80)
    print("PERFORMANCE BENCHMARK")
    print("=" * 80)
    print(
        f"{'Shape':>25} {'numel':>10} {'old (us)':>10} {'new (us)':>10} {'speedup':>10}"
    )
    print("-" * 70)

    for M, K, desc in shapes:
        try:
            result = benchmark(M, K)
            print(
                f"  {desc:>22} {result['numel']:>10} {result['old_us']:>10} {result['new_us']:>10} {result['speedup']:>10}"
            )
        except Exception as e:
            print(f"  {desc:>22} ERROR: {e}")


if __name__ == "__main__":
    main()
