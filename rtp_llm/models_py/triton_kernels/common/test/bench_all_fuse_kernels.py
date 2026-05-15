"""Comprehensive benchmark + accuracy test for all 9 fuse kernels.

Covers Qwen3.5-397B-A17B per-rank shapes (TP=2).
Produces markdown tables for performance and accuracy reporting.

Run:
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/tmp/sitecustom:.:bazel-bin \
        /opt/conda310/bin/python3 \
        rtp_llm/models_py/triton_kernels/common/test/bench_all_fuse_kernels.py
"""

import sys
import time
from typing import Callable, Optional

import flashinfer.norm
import torch
from torch.profiler import ProfilerActivity, profile

# Baseline imports
from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.modules.base.cuda.activation import FusedSiluAndMul
from rtp_llm.models_py.modules.base.cuda.norm import FusedQKRMSNorm
from rtp_llm.models_py.triton_kernels.common.activation import (
    silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd,
)

# ---------------------------------------------------------------------------
# Fused kernel imports
# ---------------------------------------------------------------------------
from rtp_llm.models_py.triton_kernels.common.attn_output_gate import (
    sigmoid_mul_fp8_quant_fwd,
    sigmoid_mul_inplace_triton,
)
from rtp_llm.models_py.triton_kernels.common.fused_add_rmsnorm_fp8_quant import (
    fused_add_rmsnorm_fp8_quant,
    fused_add_rmsnorm_fp8_quant_with_bf16_output,
)
from rtp_llm.models_py.triton_kernels.common.fused_qk_rmsnorm import (
    fused_qk_rmsnorm_triton,
)
from rtp_llm.models_py.triton_kernels.common.fused_rmsnorm_gated_fp8_quant import (
    fused_rmsnorm_gated_fp8_quant,
)
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import RmsNormGated

# ---------------------------------------------------------------------------
# Qwen3.5-397B-A17B per-rank (TP=2) dimensions
# ---------------------------------------------------------------------------
H = 4096  # hidden_size
INTER = 1024  # shared_expert_intermediate_size
HEAD_NUM = 16  # num_attention_heads per rank
KV_HEAD_NUM = 2  # num_key_value_heads per rank
HEAD_DIM = 256  # head_dim (ATTN layers)
QK_SIZE_PER_HEAD = 128  # qk_norm size_per_head
LINEAR_V_HEADS = 32  # linear_num_value_heads per rank
LINEAR_V_DIM = 128  # linear_value_head_dim
EPS = 1e-6
GROUP_SIZE = 128
SCALE_UE8M0 = False  # fp32 scale (H20)

# T values
T_DECODE = [1, 2, 4, 8, 16, 32]
T_PREFILL = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
T_ALL = T_DECODE + T_PREFILL

WARMUP = 10
REPEAT = 50

# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------


def _benchmark_us(fn: Callable, warmup: int = WARMUP, repeat: int = REPEAT) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(repeat):
            fn()
    torch.cuda.synchronize()
    total_us = sum(e.device_time_total for e in prof.key_averages())
    return total_us / repeat


# ---------------------------------------------------------------------------
# Dequantize helper (for accuracy comparison)
# ---------------------------------------------------------------------------


def _dequantize_fp8(
    fp8: torch.Tensor,
    scale: torch.Tensor,
    group_size: int = 128,
    scale_ue8m0: bool = False,
) -> torch.Tensor:
    T, Hq = fp8.shape
    n_groups = Hq // group_size
    if scale_ue8m0:
        scales_f = torch.empty((T, n_groups), dtype=torch.float32, device=fp8.device)
        scale_int = scale.to(torch.int32)
        for g in range(n_groups):
            packed_idx = g // 4
            byte_idx = g % 4
            shift = byte_idx * 8
            exp_byte = (scale_int[:, packed_idx] >> shift) & 0xFF
            f32_bits = (exp_byte << 23).to(torch.int32)
            scales_f[:, g] = f32_bits.view(torch.float32)
    else:
        scales_f = scale.float()
    return fp8.float().view(T, n_groups, group_size) * scales_f.unsqueeze(-1)


# ---------------------------------------------------------------------------
# RMSNorm reference (fp32)
# ---------------------------------------------------------------------------


def _rmsnorm_fp32(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps) * weight.float()).to(x.dtype)


# ---------------------------------------------------------------------------
# Per-fusion bench & accuracy functions
# ---------------------------------------------------------------------------


def bench_fusion1(T: int):
    """Fusion 1: sigmoid_mul"""
    attn = (torch.randn(T, H, device="cuda") * 2.0).to(torch.bfloat16)
    gate = (torch.randn(T, H, device="cuda") * 4.0).to(torch.bfloat16)

    def baseline_fn():
        return attn * torch.sigmoid(gate)

    def fused_fn():
        sigmoid_mul_inplace_triton(attn.clone(), gate)

    base_us = _benchmark_us(baseline_fn)
    fused_us = _benchmark_us(fused_fn)

    ref = (attn.float() * torch.sigmoid(gate.float())).to(torch.bfloat16)
    actual = sigmoid_mul_inplace_triton(attn.clone(), gate)
    max_diff = (actual.float() - ref.float()).abs().max().item()
    return base_us, fused_us, max_diff, "max_abs"


def bench_fusion2(T: int):
    """Fusion 2: silu_and_mul + fp8_quant"""
    x = (torch.randn(T, 2 * INTER, device="cuda") * 1.5).to(torch.bfloat16).contiguous()

    silu_and_mul_op = FusedSiluAndMul()

    def baseline_fn():
        activated = silu_and_mul_op(x)
        sgl_per_token_group_quant_fp8(
            activated,
            group_size=GROUP_SIZE,
            eps=1e-10,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=SCALE_UE8M0,
        )

    def fused_fn():
        silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd(
            x,
            quant_group_size=GROUP_SIZE,
            scale_ue8m0=SCALE_UE8M0,
        )

    base_us = _benchmark_us(baseline_fn)
    fused_us = _benchmark_us(fused_fn)

    # Accuracy: compare dequantized fused vs float ref (use fp32 Python silu for reference)
    gate_part = x[:, :INTER]
    up_part = x[:, INTER:]
    silu_fp32 = gate_part.float() / (1.0 + torch.exp(-gate_part.float()))
    ref_activated = (up_part.float() * silu_fp32).to(torch.bfloat16)
    fp8_out, scale_out = silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd(
        x,
        quant_group_size=GROUP_SIZE,
        scale_ue8m0=SCALE_UE8M0,
    )
    deq = _dequantize_fp8(fp8_out, scale_out, GROUP_SIZE, SCALE_UE8M0).reshape(T, INTER)
    rel_err = (
        ((deq - ref_activated.float()).abs() / (ref_activated.float().abs() + 1e-6))
        .mean()
        .item()
    )
    return base_us, fused_us, rel_err, "mean_rel"


def bench_fusion3(T: int):
    """Fusion 3/4: add + RMSNorm + fp8_quant (single output)"""
    hidden = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(H, dtype=torch.bfloat16, device="cuda")

    def baseline_fn():
        res = residual.clone()
        res.add_(hidden)
        normed = flashinfer.norm.rmsnorm(res, weight, eps=EPS)
        sgl_per_token_group_quant_fp8(
            normed.contiguous(),
            group_size=GROUP_SIZE,
            eps=1e-10,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=SCALE_UE8M0,
        )

    def fused_fn():
        res = residual.clone()
        fused_add_rmsnorm_fp8_quant(
            hidden.clone(), res, weight, EPS, GROUP_SIZE, SCALE_UE8M0
        )

    base_us = _benchmark_us(baseline_fn)
    fused_us = _benchmark_us(fused_fn)

    # Accuracy
    res_ref = residual.clone()
    res_ref.add_(hidden)
    normed_ref = _rmsnorm_fp32(res_ref, weight, EPS)
    res_fused = residual.clone()
    fp8_fused, scale_fused = fused_add_rmsnorm_fp8_quant(
        hidden.clone(), res_fused, weight, EPS, GROUP_SIZE, SCALE_UE8M0
    )
    deq = _dequantize_fp8(fp8_fused, scale_fused, GROUP_SIZE, SCALE_UE8M0).reshape(T, H)
    rel_err = (
        ((deq - normed_ref.float()).abs() / (normed_ref.float().abs() + 1e-6))
        .mean()
        .item()
    )
    return base_us, fused_us, rel_err, "mean_rel"


def bench_fusion5(T: int):
    """Fusion 5/7: add + RMSNorm + fp8_quant + bf16 dual output"""
    hidden = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(H, dtype=torch.bfloat16, device="cuda")

    def baseline_fn():
        res = residual.clone()
        res.add_(hidden)
        normed = flashinfer.norm.rmsnorm(res, weight, eps=EPS)
        _ = normed.clone()  # bf16 output
        sgl_per_token_group_quant_fp8(
            normed.contiguous(),
            group_size=GROUP_SIZE,
            eps=1e-10,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=SCALE_UE8M0,
        )

    def fused_fn():
        res = residual.clone()
        fused_add_rmsnorm_fp8_quant_with_bf16_output(
            hidden.clone(), res, weight, EPS, GROUP_SIZE, SCALE_UE8M0
        )

    base_us = _benchmark_us(baseline_fn)
    fused_us = _benchmark_us(fused_fn)

    # Accuracy
    res_ref = residual.clone()
    res_ref.add_(hidden)
    normed_ref = _rmsnorm_fp32(res_ref, weight, EPS)
    res_fused = residual.clone()
    bf16_fused, fp8_fused, scale_fused = fused_add_rmsnorm_fp8_quant_with_bf16_output(
        hidden.clone(), res_fused, weight, EPS, GROUP_SIZE, SCALE_UE8M0
    )
    bf16_diff = (bf16_fused.float() - normed_ref.float()).abs().max().item()
    deq = _dequantize_fp8(fp8_fused, scale_fused, GROUP_SIZE, SCALE_UE8M0).reshape(T, H)
    fp8_rel = (
        ((deq - normed_ref.float()).abs() / (normed_ref.float().abs() + 1e-6))
        .mean()
        .item()
    )
    return base_us, fused_us, max(bf16_diff, fp8_rel), "max(bf16_abs,fp8_rel)"


def bench_fusion6(T: int):
    """Fusion 6: RmsNormGated + fp8_quant"""
    num_heads = LINEAR_V_HEADS
    head_v_dim = LINEAR_V_DIM
    M = T * num_heads
    x = torch.randn(M, head_v_dim, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(M, head_v_dim, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(head_v_dim, dtype=torch.bfloat16, device="cuda")
    norm_op = RmsNormGated(weight, eps=EPS, group_size=head_v_dim)

    def baseline_fn():
        normed = norm_op(x.clone(), gate.clone())
        flat = normed.reshape(T, num_heads * head_v_dim).contiguous()
        sgl_per_token_group_quant_fp8(
            flat,
            group_size=GROUP_SIZE,
            eps=1e-10,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=SCALE_UE8M0,
        )

    def fused_fn():
        fused_rmsnorm_gated_fp8_quant(
            x.clone(),
            gate.clone(),
            weight,
            EPS,
            num_heads,
            GROUP_SIZE,
            SCALE_UE8M0,
        )

    base_us = _benchmark_us(baseline_fn)
    fused_us = _benchmark_us(fused_fn)

    # Accuracy
    normed_ref = norm_op(x.clone(), gate.clone())
    flat_ref = normed_ref.reshape(T, num_heads * head_v_dim).contiguous()
    fp8_fused, scale_fused = fused_rmsnorm_gated_fp8_quant(
        x.clone(),
        gate.clone(),
        weight,
        EPS,
        num_heads,
        GROUP_SIZE,
        SCALE_UE8M0,
    )
    deq = _dequantize_fp8(fp8_fused, scale_fused, GROUP_SIZE, SCALE_UE8M0).reshape(
        T, num_heads * head_v_dim
    )
    rel_err = (
        ((deq - flat_ref.float()).abs() / (flat_ref.float().abs() + 1e-6)).mean().item()
    )
    return base_us, fused_us, rel_err, "mean_rel"


def bench_fusion8(T: int):
    """Fusion 8: sigmoid_mul + fp8_quant"""
    attn = (torch.randn(T, H, device="cuda") * 2.0).to(torch.bfloat16)
    gate = (torch.randn(T, H, device="cuda") * 4.0).to(torch.bfloat16)

    def baseline_fn():
        out = sigmoid_mul_inplace_triton(attn.clone(), gate)
        sgl_per_token_group_quant_fp8(
            out.contiguous(),
            group_size=GROUP_SIZE,
            eps=1e-10,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=SCALE_UE8M0,
        )

    def fused_fn():
        sigmoid_mul_fp8_quant_fwd(attn.clone(), gate, GROUP_SIZE, SCALE_UE8M0)

    base_us = _benchmark_us(baseline_fn)
    fused_us = _benchmark_us(fused_fn)

    # Accuracy
    ref = attn.float() * torch.sigmoid(gate.float())
    fp8_fused, scale_fused = sigmoid_mul_fp8_quant_fwd(
        attn.clone(), gate, GROUP_SIZE, SCALE_UE8M0
    )
    num_groups = H // GROUP_SIZE
    deq = (
        fp8_fused.float().view(T, num_groups, GROUP_SIZE)
        * scale_fused.float().unsqueeze(-1)
    ).view(T, H)
    rel_err = ((deq - ref).abs() / (ref.abs() + 1e-6)).mean().item()
    return base_us, fused_us, rel_err, "mean_rel"


def bench_fusion9(T: int):
    """Fusion 9: QK RMSNorm merge"""
    head_num = HEAD_NUM
    kv_head_num = KV_HEAD_NUM
    spd = QK_SIZE_PER_HEAD
    total_dim = (head_num + kv_head_num * 2) * spd
    qkv = torch.randn(T, total_dim, dtype=torch.bfloat16, device="cuda")
    q_weight = torch.randn(spd, dtype=torch.bfloat16, device="cuda")
    k_weight = torch.randn(spd, dtype=torch.bfloat16, device="cuda")
    fused_op = FusedQKRMSNorm(q_weight, k_weight, head_num, kv_head_num, spd, EPS)

    def baseline_fn():
        qkv_3d = qkv.clone().reshape(T, head_num + kv_head_num * 2, spd)
        q = qkv_3d[:, :head_num, :].contiguous()
        k = qkv_3d[:, head_num : head_num + kv_head_num, :].contiguous()
        flashinfer.norm.rmsnorm(q, q_weight, eps=EPS, out=q)
        flashinfer.norm.rmsnorm(k, k_weight, eps=EPS, out=k)

    def fused_fn():
        fused_op(qkv.clone())

    base_us = _benchmark_us(baseline_fn)
    fused_us = _benchmark_us(fused_fn)

    # Accuracy
    def ref_qk_rmsnorm(qkv_in):
        qkv_3d = qkv_in.clone().reshape(T, head_num + kv_head_num * 2, spd)
        q = qkv_3d[:, :head_num, :].float()
        k = qkv_3d[:, head_num : head_num + kv_head_num, :].float()

        def rn(x, w):
            var = x.pow(2).mean(dim=-1, keepdim=True)
            return (x * torch.rsqrt(var + EPS) * w.float().unsqueeze(0)).to(
                torch.bfloat16
            )

        qkv_3d[:, :head_num, :] = rn(q, q_weight)
        qkv_3d[:, head_num : head_num + kv_head_num, :] = rn(k, k_weight)
        return qkv_3d.reshape(T, total_dim)

    ref = ref_qk_rmsnorm(qkv)
    actual = fused_op(qkv.clone())
    max_diff = (actual.float() - ref.float()).abs().max().item()
    return base_us, fused_us, max_diff, "max_abs"


# ---------------------------------------------------------------------------
# Main: run all benchmarks and produce report
# ---------------------------------------------------------------------------

FUSION_DEFS = [
    ("F1: sigmoid_mul", bench_fusion1, T_ALL),
    ("F2: silu_mul+fp8q", bench_fusion2, T_ALL),
    ("F3/4: add+norm+fp8q", bench_fusion3, T_ALL),
    ("F5/7: add+norm+dual", bench_fusion5, T_ALL),
    (
        "F6: normGated+fp8q",
        bench_fusion6,
        [t for t in T_ALL if t * LINEAR_V_HEADS * LINEAR_V_DIM <= 512 * 1024 * 1024],
    ),
    ("F8: sigm_mul+fp8q", bench_fusion8, T_ALL),
    ("F9: QK RMSNorm", bench_fusion9, T_ALL),
]


def _phase(T: int) -> str:
    return "decode" if T <= 32 else "prefill"


def main():
    torch.manual_seed(42)
    assert torch.cuda.is_available(), "CUDA required"

    print("=" * 100)
    print("  Qwen3.5-397B-A17B Fuse Kernel Benchmark (per-rank TP=2)")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(
        f"  H={H}, INTER={INTER}, heads={HEAD_NUM}q/{KV_HEAD_NUM}kv, qk_spd={QK_SIZE_PER_HEAD}"
    )
    print(f"  LINEAR_V_HEADS={LINEAR_V_HEADS}, LINEAR_V_DIM={LINEAR_V_DIM}")
    print(f"  GROUP_SIZE={GROUP_SIZE}, SCALE_UE8M0={SCALE_UE8M0}")
    print(f"  Warmup={WARMUP}, Repeat={REPEAT}")
    print("=" * 100)

    perf_results = []
    acc_results = []

    for fusion_name, bench_fn, t_values in FUSION_DEFS:
        print(f"\n--- {fusion_name} ---")
        for T in t_values:
            try:
                base_us, fused_us, err_val, err_type = bench_fn(T)
                speedup = base_us / fused_us if fused_us > 0 else float("inf")
                phase = _phase(T)
                perf_results.append((fusion_name, T, phase, base_us, fused_us, speedup))
                passed = (
                    (err_val < 0.15)
                    if "bf16" in err_type
                    else ((err_val < 0.05) if "rel" in err_type else (err_val < 0.15))
                )
                acc_results.append(
                    (
                        fusion_name,
                        T,
                        phase,
                        err_type,
                        err_val,
                        "PASS" if passed else "FAIL",
                    )
                )
                print(
                    f"  T={T:6d} ({phase:7s})  base={base_us:9.2f}us  fused={fused_us:9.2f}us  "
                    f"speedup={speedup:5.2f}x  err={err_val:.3e} ({err_type}) {'PASS' if passed else 'FAIL'}"
                )
            except Exception as e:
                print(f"  T={T:6d}  ERROR: {e}")
                perf_results.append((fusion_name, T, _phase(T), 0, 0, 0))
                acc_results.append(
                    (fusion_name, T, _phase(T), "error", 0, f"ERROR: {e}")
                )

    # Print markdown performance table
    print("\n\n")
    print("## Performance Report")
    print()
    print("| Fusion | T | Phase | Baseline (us) | Fused (us) | Speedup |")
    print("|--------|---:|-------|-------------:|----------:|--------:|")
    for name, T, phase, base, fused, spd in perf_results:
        spd_str = f"{spd:.2f}x" if spd > 0 else "N/A"
        print(f"| {name} | {T} | {phase} | {base:.2f} | {fused:.2f} | {spd_str} |")

    # Print markdown accuracy table
    print()
    print("## Accuracy Report")
    print()
    print("| Fusion | T | Phase | Metric | Error | Status |")
    print("|--------|---:|-------|--------|------:|--------|")
    for name, T, phase, metric, err, status in acc_results:
        print(f"| {name} | {T} | {phase} | {metric} | {err:.3e} | {status} |")

    # Summary
    print()
    print("## Summary")
    decode_perf = [
        (n, b, f, s) for n, t, p, b, f, s in perf_results if p == "decode" and s > 0
    ]
    prefill_perf = [
        (n, b, f, s) for n, t, p, b, f, s in perf_results if p == "prefill" and s > 0
    ]
    if decode_perf:
        avg_spd = sum(s for _, _, _, s in decode_perf) / len(decode_perf)
        print(f"- Decode average speedup: {avg_spd:.2f}x")
    if prefill_perf:
        avg_spd = sum(s for _, _, _, s in prefill_perf) / len(prefill_perf)
        print(f"- Prefill average speedup: {avg_spd:.2f}x")

    all_pass = all(
        s == "PASS"
        for _, _, _, _, _, s in acc_results
        if s not in ("PASS", "FAIL") or s == "PASS"
    )
    fail_count = sum(1 for _, _, _, _, _, s in acc_results if s == "FAIL")
    error_count = sum(1 for _, _, _, _, _, s in acc_results if "ERROR" in s)
    print(
        f"- Accuracy: {len(acc_results) - fail_count - error_count}/{len(acc_results)} PASS, {fail_count} FAIL, {error_count} ERROR"
    )


if __name__ == "__main__":
    main()
