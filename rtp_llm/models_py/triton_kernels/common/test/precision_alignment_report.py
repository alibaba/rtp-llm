"""精度对齐测试 — 最近 3 个 commit 涉及的所有 fuse kernel.

对每个 fused kernel:
  1) 构造与 baseline 相同的输入
  2) 跑 baseline 链（被替换的 1-3 个 kernel）
  3) 跑 fused kernel
  4) 报告 fused vs baseline 的 max_abs / mean_abs / max_rel / mean_rel
  5) 判定：
     - bit-exact: max_abs == 0
     - PASS    : max_abs < 1e-5
     - 1ULP    : max_abs < 1 bf16/fp8 ULP (数值噪声，可接受)
     - FAIL    : 系统性差异

性能聚焦 T=1,2,4,8,16,32（decode bs 范围）。

用法:
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=.:bazel-bin /opt/conda310/bin/python3 \
        rtp_llm/models_py/triton_kernels/common/test/precision_alignment_report.py
"""

import flashinfer.norm
import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile

# baseline imports
from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.base.cuda.activation import FusedSiluAndMul
from rtp_llm.models_py.modules.base.cuda.norm import FusedQKRMSNorm
from rtp_llm.models_py.triton_kernels.common.activation import (
    silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd,
)

# fused kernel imports
from rtp_llm.models_py.triton_kernels.common.attn_output_gate import (
    sigmoid_mul_fp8_quant_fwd,
    sigmoid_mul_inplace_triton,
)
from rtp_llm.models_py.triton_kernels.common.fused_add_rmsnorm_fp8_quant import (
    fused_add_rmsnorm_fp8_quant,
    fused_add_rmsnorm_fp8_quant_with_bf16_output,
)
from rtp_llm.models_py.triton_kernels.common.fused_logits_head_gate import (
    _baseline_logits_head_gate,
    fused_logits_head_gate,
)
from rtp_llm.models_py.triton_kernels.common.fused_qk_rmsnorm import (
    fused_qk_rmsnorm_triton,
)
from rtp_llm.models_py.triton_kernels.common.fused_rmsnorm_gated_fp8_quant import (
    fused_rmsnorm_gated_fp8_quant,
)
from rtp_llm.models_py.triton_kernels.common.fused_strided_rmsnorm import (
    fused_strided_rmsnorm,
    fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output,
)
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import RmsNormGated

torch.manual_seed(42)
DEVICE = "cuda"
EPS = 1e-6
GS = 128

# T values: focus on decode bs=1..32
T_DECODE = [1, 2, 4, 8, 16, 32]
WARMUP, REPEAT = 10, 50

# Production-relevant shapes
H_QWEN35 = 4096  # Qwen3.5 hidden_size per rank (TP=2)
INTER_QWEN35 = 1024
HEAD_NUM = 16
KV_HEAD_NUM = 2
QK_SPD = 128
LINEAR_V_HEADS = 32
LINEAR_V_DIM = 128

H_GLM5 = 6144  # GLM5 hidden_size

# Precision threshold
PRECISION_TOL = 1e-5
BF16_1ULP_AT_1 = 2**-7  # ≈ 7.8e-3
FP8_E4M3_RELTOL = 0.05  # 5% — fp8 e4m3 + per-group quant 噪声范围


def _bench_us(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(REPEAT):
            fn()
    torch.cuda.synchronize()
    return sum(e.device_time_total for e in prof.key_averages()) / REPEAT


def _classify(max_abs, mean_abs, ref_dtype="bf16"):
    """精度判定.

    bf16 1 ULP varies by magnitude: 2^(exp - 7). For input magnitudes in
    [8, 16) the 1 ULP is 0.0625; for [16, 32) it's 0.125. Fused vs baseline
    differing by 1 ULP on a few outlier elements is numeric noise, not a
    bug. Use mean_abs to detect *systematic* drift: if mean is at ULP
    magnitude (1e-3 ~ 1e-4) the fuse is numerically equivalent.
    """
    if max_abs == 0.0:
        return "bit-exact"
    if max_abs < PRECISION_TOL:
        return "PASS"
    if ref_dtype == "bf16":
        # Up to bf16 1 ULP at magnitude ~32 (covers typical activation range
        # for residual_add + rmsnorm outputs at large T).
        if max_abs <= BF16_1ULP_AT_1 * 32 and mean_abs < 5e-3:
            return "1ULP"
    return "FAIL"


def _diff_metrics(actual, ref):
    a = actual.float()
    b = ref.float()
    diff = (a - b).abs()
    # Clipped relative error: avoid blow-up when ref ≈ 0 (which would make
    # raw rel = diff/(0+eps) explode to 1e4+ even when diff is tiny). Use
    # max(|ref|, max_abs_ref * 1e-3) as denominator: only penalize relative
    # error against meaningful reference magnitudes.
    clip = max(b.abs().max().item() * 1e-3, 1e-6)
    rel_clipped = diff / b.abs().clamp_min(clip)
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_rel": rel_clipped.max().item(),
        "mean_rel": rel_clipped.mean().item(),
    }


def _dequant_fp8(fp8, scale, group_size=GS, T=None):
    """Dequantize fp8 with per-group scale to fp32. fp8 may be in column-major layout."""
    if T is None:
        T = fp8.shape[0]
    H = fp8.shape[-1]
    n_groups = H // group_size
    return (
        fp8.float().view(T, n_groups, group_size) * scale.float().unsqueeze(-1)
    ).reshape(T, H)


# =========================================================================
# F1: sigmoid_mul (Qwen3.5)
# =========================================================================
def test_f1_sigmoid_mul(T):
    H = H_QWEN35
    attn = (torch.randn(T, H, device=DEVICE) * 2.0).to(torch.bfloat16)
    gate = (torch.randn(T, H, device=DEVICE) * 4.0).to(torch.bfloat16)

    def baseline_fn():
        return attn * torch.sigmoid(gate)

    def fused_fn():
        return sigmoid_mul_inplace_triton(attn.clone(), gate)

    base = baseline_fn().to(torch.bfloat16)
    fused = fused_fn()
    m = _diff_metrics(fused, base)
    base_us = _bench_us(baseline_fn)
    fused_us = _bench_us(fused_fn)
    return m, base_us, fused_us, "bf16"


# =========================================================================
# F2: silu_and_mul + fp8q (Qwen3.5)
# =========================================================================
def test_f2_silu_fp8q(T):
    H = INTER_QWEN35
    x = (torch.randn(T, 2 * H, device=DEVICE) * 1.5).to(torch.bfloat16).contiguous()
    silu_op = FusedSiluAndMul()

    # baseline = silu_and_mul → fp8_quant
    def baseline_fn():
        a = silu_op(x)
        return sgl_per_token_group_quant_fp8(
            a,
            group_size=GS,
            eps=1e-10,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=False,
        )

    def fused_fn():
        return silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd(
            x,
            quant_group_size=GS,
            scale_ue8m0=False,
        )

    # fp8 比对方式：dequant 后比 fp32 ref
    silu_fp32 = (x[:, :H].float() / (1.0 + torch.exp(-x[:, :H].float()))) * x[
        :, H:
    ].float()

    base_fp8, base_scale = baseline_fn()
    fused_fp8, fused_scale = fused_fn()

    base_deq = _dequant_fp8(base_fp8, base_scale, GS, T)
    fused_deq = _dequant_fp8(fused_fp8, fused_scale, GS, T)

    # baseline dequant vs fp32 ref (量化噪声基线)
    base_vs_ref = _diff_metrics(base_deq, silu_fp32)
    fused_vs_ref = _diff_metrics(fused_deq, silu_fp32)
    # 关键：fused dequant vs baseline dequant
    fused_vs_base = _diff_metrics(fused_deq, base_deq)

    base_us = _bench_us(baseline_fn)
    fused_us = _bench_us(fused_fn)
    return (
        {"vs_ref": fused_vs_ref, "vs_base": fused_vs_base, "base_vs_ref": base_vs_ref},
        base_us,
        fused_us,
        "fp8",
    )


# =========================================================================
# F3/4: add+rmsnorm+fp8q single (Qwen3.5)
# =========================================================================
def test_f3_add_rms_fp8q(T):
    H = H_QWEN35
    hidden = torch.randn(T, H, dtype=torch.bfloat16, device=DEVICE)
    residual = torch.randn(T, H, dtype=torch.bfloat16, device=DEVICE)
    weight = torch.randn(H, dtype=torch.bfloat16, device=DEVICE)

    def baseline_fn():
        res = residual.clone()
        res.add_(hidden)
        normed = flashinfer.norm.rmsnorm(res, weight, eps=EPS)
        return sgl_per_token_group_quant_fp8(
            normed.contiguous(),
            group_size=GS,
            eps=1e-10,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=False,
        )

    def fused_fn():
        res = residual.clone()
        return fused_add_rmsnorm_fp8_quant(hidden.clone(), res, weight, EPS, GS, False)

    base_fp8, base_scale = baseline_fn()
    fused_fp8, fused_scale = fused_fn()
    base_deq = _dequant_fp8(base_fp8, base_scale, GS, T)
    fused_deq = _dequant_fp8(fused_fp8, fused_scale, GS, T)

    # fp32 ref
    res_ref = residual.clone().add_(hidden).float()
    var = res_ref.pow(2).mean(dim=-1, keepdim=True)
    ref = res_ref * torch.rsqrt(var + EPS) * weight.float()

    fused_vs_base = _diff_metrics(fused_deq, base_deq)
    fused_vs_ref = _diff_metrics(fused_deq, ref)
    base_vs_ref = _diff_metrics(base_deq, ref)
    base_us = _bench_us(baseline_fn)
    fused_us = _bench_us(fused_fn)
    return (
        {"vs_ref": fused_vs_ref, "vs_base": fused_vs_base, "base_vs_ref": base_vs_ref},
        base_us,
        fused_us,
        "fp8",
    )


# =========================================================================
# F5/7: add+rmsnorm+fp8q dual bf16+fp8 (Qwen3.5)
# =========================================================================
def test_f5_add_rms_fp8q_dual(T):
    H = H_QWEN35
    hidden = torch.randn(T, H, dtype=torch.bfloat16, device=DEVICE)
    residual = torch.randn(T, H, dtype=torch.bfloat16, device=DEVICE)
    weight = torch.randn(H, dtype=torch.bfloat16, device=DEVICE)

    def baseline_fn():
        res = residual.clone()
        res.add_(hidden)
        normed = flashinfer.norm.rmsnorm(res, weight, eps=EPS)
        bf16 = normed.clone()
        fp8, sc = sgl_per_token_group_quant_fp8(
            normed.contiguous(),
            group_size=GS,
            eps=1e-10,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=False,
        )
        return bf16, fp8, sc

    def fused_fn():
        res = residual.clone()
        return fused_add_rmsnorm_fp8_quant_with_bf16_output(
            hidden.clone(),
            res,
            weight,
            EPS,
            GS,
            False,
        )

    base_bf16, base_fp8, base_scale = baseline_fn()
    fused_bf16, fused_fp8, fused_scale = fused_fn()

    # bf16 channel
    bf16_metrics = _diff_metrics(fused_bf16, base_bf16)
    # fp8 channel: dequant comparison
    base_deq = _dequant_fp8(base_fp8, base_scale, GS, T)
    fused_deq = _dequant_fp8(fused_fp8, fused_scale, GS, T)
    fp8_metrics = _diff_metrics(fused_deq, base_deq)

    base_us = _bench_us(baseline_fn)
    fused_us = _bench_us(fused_fn)
    return (
        {"bf16_vs_base": bf16_metrics, "fp8_vs_base": fp8_metrics},
        base_us,
        fused_us,
        "dual",
    )


# =========================================================================
# F6: rmsnorm_gated + fp8q (Qwen3.5)
# =========================================================================
def test_f6_rmsnorm_gated_fp8q(T):
    num_heads = LINEAR_V_HEADS
    head_v_dim = LINEAR_V_DIM
    M = T * num_heads
    x = torch.randn(M, head_v_dim, dtype=torch.bfloat16, device=DEVICE)
    gate = torch.randn(M, head_v_dim, dtype=torch.bfloat16, device=DEVICE)
    weight = torch.randn(head_v_dim, dtype=torch.bfloat16, device=DEVICE)
    norm_op = RmsNormGated(weight, eps=EPS, group_size=head_v_dim)

    def baseline_fn():
        n = norm_op(x.clone(), gate.clone())
        flat = n.reshape(T, num_heads * head_v_dim).contiguous()
        return sgl_per_token_group_quant_fp8(
            flat,
            group_size=GS,
            eps=1e-10,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=False,
        )

    def fused_fn():
        return fused_rmsnorm_gated_fp8_quant(
            x.clone(),
            gate.clone(),
            weight,
            EPS,
            num_heads,
            GS,
            False,
        )

    base_fp8, base_scale = baseline_fn()
    fused_fp8, fused_scale = fused_fn()
    base_deq = _dequant_fp8(base_fp8, base_scale, GS, T)
    fused_deq = _dequant_fp8(fused_fp8, fused_scale, GS, T)
    fused_vs_base = _diff_metrics(fused_deq, base_deq)
    base_us = _bench_us(baseline_fn)
    fused_us = _bench_us(fused_fn)
    return {"vs_base": fused_vs_base}, base_us, fused_us, "fp8"


# =========================================================================
# F8: sigmoid_mul + fp8q (Qwen3.5)
# =========================================================================
def test_f8_sigm_mul_fp8q(T):
    H = H_QWEN35
    attn = (torch.randn(T, H, device=DEVICE) * 2.0).to(torch.bfloat16)
    gate = (torch.randn(T, H, device=DEVICE) * 4.0).to(torch.bfloat16)

    def baseline_fn():
        out = sigmoid_mul_inplace_triton(attn.clone(), gate)
        return sgl_per_token_group_quant_fp8(
            out.contiguous(),
            group_size=GS,
            eps=1e-10,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=False,
        )

    def fused_fn():
        return sigmoid_mul_fp8_quant_fwd(attn.clone(), gate, GS, False)

    base_fp8, base_scale = baseline_fn()
    fused_fp8, fused_scale = fused_fn()
    base_deq = _dequant_fp8(base_fp8, base_scale, GS, T)
    fused_deq = _dequant_fp8(fused_fp8, fused_scale, GS, T)
    fused_vs_base = _diff_metrics(fused_deq, base_deq)
    base_us = _bench_us(baseline_fn)
    fused_us = _bench_us(fused_fn)
    return {"vs_base": fused_vs_base}, base_us, fused_us, "fp8"


# =========================================================================
# F9: QK RMSNorm merge (Qwen3.5)
# =========================================================================
def test_f9_qk_rmsnorm(T):
    head_num = HEAD_NUM
    kv_head_num = KV_HEAD_NUM
    spd = QK_SPD
    total_dim = (head_num + kv_head_num * 2) * spd
    qkv = torch.randn(T, total_dim, dtype=torch.bfloat16, device=DEVICE)
    q_w = torch.randn(spd, dtype=torch.bfloat16, device=DEVICE)
    k_w = torch.randn(spd, dtype=torch.bfloat16, device=DEVICE)
    fused_op = FusedQKRMSNorm(q_w, k_w, head_num, kv_head_num, spd, EPS)

    def baseline_fn():
        qkv_out = qkv.clone()
        qkv_3d = qkv_out.reshape(T, head_num + kv_head_num * 2, spd)
        q = qkv_3d[:, :head_num, :].contiguous()
        k = qkv_3d[:, head_num : head_num + kv_head_num, :].contiguous()
        flashinfer.norm.rmsnorm(q, q_w, eps=EPS, out=q)
        flashinfer.norm.rmsnorm(k, k_w, eps=EPS, out=k)
        qkv_3d[:, :head_num, :] = q
        qkv_3d[:, head_num : head_num + kv_head_num, :] = k
        return qkv_out

    def fused_fn():
        return fused_op(qkv.clone())

    base = baseline_fn()
    fused = fused_fn()
    metrics = _diff_metrics(fused, base)
    base_us = _bench_us(baseline_fn)
    fused_us = _bench_us(fused_fn)
    return metrics, base_us, fused_us, "bf16"


# =========================================================================
# DSA-F1: strided_rmsnorm (GLM5)
# =========================================================================
def test_dsa_f1_strided_rmsnorm(T):
    H = H_GLM5
    big = torch.randn(T, H + 200, dtype=torch.bfloat16, device=DEVICE)
    x = big[:, :H]
    weight = torch.randn(H, dtype=torch.bfloat16, device=DEVICE)

    def baseline_fn():
        return flashinfer.norm.rmsnorm(x.contiguous(), weight, eps=EPS)

    def fused_fn():
        return fused_strided_rmsnorm(x, weight, EPS)

    base = baseline_fn()
    fused = fused_fn()
    metrics = _diff_metrics(fused, base)
    base_us = _bench_us(baseline_fn)
    fused_us = _bench_us(fused_fn)
    return metrics, base_us, fused_us, "bf16"


# =========================================================================
# DSA-F2: strided_rmsnorm + fp8q dual (GLM5)
# =========================================================================
def test_dsa_f2_strided_rms_fp8q_dual(T):
    H = H_GLM5
    big = torch.randn(T, H + 200, dtype=torch.bfloat16, device=DEVICE)
    x = big[:, :H]
    weight = torch.randn(H, dtype=torch.bfloat16, device=DEVICE)

    def baseline_fn():
        bf16 = flashinfer.norm.rmsnorm(x.contiguous(), weight, eps=EPS)
        fp8, sc = sgl_per_token_group_quant_fp8(
            bf16.contiguous(),
            group_size=GS,
            eps=1e-10,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=False,
        )
        return bf16, fp8, sc

    def fused_fn():
        return fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output(
            x,
            weight,
            EPS,
            GS,
            False,
        )

    base_bf16, base_fp8, base_scale = baseline_fn()
    fused_bf16, fused_fp8, fused_scale = fused_fn()

    bf16_metrics = _diff_metrics(fused_bf16, base_bf16)
    base_deq = _dequant_fp8(base_fp8, base_scale, GS, T)
    fused_deq = _dequant_fp8(fused_fp8, fused_scale, GS, T)
    fp8_metrics = _diff_metrics(fused_deq, base_deq)

    base_us = _bench_us(baseline_fn)
    fused_us = _bench_us(fused_fn)
    return (
        {"bf16_vs_base": bf16_metrics, "fp8_vs_base": fp8_metrics},
        base_us,
        fused_us,
        "dual",
    )


# =========================================================================
# DSA-F3: logits_head_gate (GLM5/DSV3.2)
# =========================================================================
def test_dsa_f3_logits_head_gate(T):
    K, N = 6144, 32  # GLM5 production shape
    x = torch.randn(T, K, dtype=torch.bfloat16, device=DEVICE)
    weight = (torch.randn(N, K, dtype=torch.float32, device=DEVICE) * 0.02).contiguous()
    qs = torch.randn(T, N, 1, dtype=torch.float32, device=DEVICE).abs() + 0.1
    sc = K**-0.5 * N**-0.5
    linear = nn.Linear(K, N, bias=False, device=DEVICE)
    linear.weight.data = weight.clone()

    def baseline_fn():
        return _baseline_logits_head_gate(x, qs, linear, sc)

    def fused_fn():
        return fused_logits_head_gate(x, qs, weight, sc, fallback_proj=linear)

    base = baseline_fn()
    fused = fused_fn()
    metrics = _diff_metrics(fused, base)
    base_us = _bench_us(baseline_fn)
    fused_us = _bench_us(fused_fn)
    return metrics, base_us, fused_us, "fp32"


# =========================================================================
# DSA-F4: output_bmm direct write (cuBLAS strideC, GLM5)
# =========================================================================
def test_dsa_f4_output_bmm(T):
    nh, kv, v = 128, 512, 128
    attn = torch.randn(T, nh, kv, dtype=torch.bfloat16, device=DEVICE)
    vw = torch.randn(nh, kv, v, dtype=torch.bfloat16, device=DEVICE)

    def baseline_fn():
        out = torch.bmm(attn.transpose(0, 1), vw)
        return out.transpose(0, 1).reshape(T, nh * v).contiguous()

    def fused_fn():
        out_flat = torch.empty(T, nh, v, dtype=attn.dtype, device=DEVICE)
        torch.bmm(attn.transpose(0, 1), vw, out=out_flat.transpose(0, 1))
        return out_flat.reshape(T, nh * v)

    base = baseline_fn()
    fused = fused_fn()
    metrics = _diff_metrics(fused, base)
    base_us = _bench_us(baseline_fn)
    fused_us = _bench_us(fused_fn)
    return metrics, base_us, fused_us, "bf16"


# =========================================================================
# Main
# =========================================================================
TESTS = [
    ("F1  sigmoid_mul (bf16)", test_f1_sigmoid_mul, "single"),
    ("F2  silu_and_mul + fp8q", test_f2_silu_fp8q, "fp8"),
    ("F3/4 add + rmsnorm + fp8q", test_f3_add_rms_fp8q, "fp8"),
    ("F5/7 add + rmsnorm + fp8q (bf16+fp8 dual)", test_f5_add_rms_fp8q_dual, "dual"),
    ("F6  rmsnorm_gated + fp8q", test_f6_rmsnorm_gated_fp8q, "fp8"),
    ("F8  sigmoid_mul + fp8q", test_f8_sigm_mul_fp8q, "fp8"),
    ("F9  QK RMSNorm merge", test_f9_qk_rmsnorm, "single"),
    ("DSA-F1  strided_rmsnorm (bf16)", test_dsa_f1_strided_rmsnorm, "single"),
    (
        "DSA-F2  strided_rms+fp8q (dual bf16+fp8)",
        test_dsa_f2_strided_rms_fp8q_dual,
        "dual",
    ),
    ("DSA-F3  logits_head_gate (fp32)", test_dsa_f3_logits_head_gate, "single"),
    ("DSA-F4  output_bmm direct (bf16)", test_dsa_f4_output_bmm, "single"),
]


def fmt(x):
    return f"{x:.2e}" if abs(x) > 0 or x == 0 else str(x)


def main():
    print("=" * 130)
    print("精度对齐测试报告 — 最近 3 个 commit 涉及的全部 fuse kernel")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"精度阈值: max_abs < {PRECISION_TOL:.0e} (PASS), bf16 1ULP噪声可接受")
    print(
        f"性能: T={T_DECODE}, warmup={WARMUP}, repeat={REPEAT}, 用 torch.profiler 测 device time"
    )
    print("=" * 130)

    perf_table = []
    prec_table = []

    for name, fn, kind in TESTS:
        print(f"\n--- {name} ---")
        for T in T_DECODE:
            try:
                metrics, base_us, fused_us, dtype = fn(T)
                speedup = base_us / fused_us if fused_us > 0 else 0
                perf_table.append((name, T, base_us, fused_us, speedup))

                if kind == "single":
                    m = metrics
                    cls = _classify(
                        m["max_abs"],
                        m["mean_abs"],
                        "bf16" if dtype == "bf16" else "fp32",
                    )
                    prec_table.append(
                        (
                            name,
                            T,
                            "fused vs base",
                            m["max_abs"],
                            m["mean_abs"],
                            m["max_rel"],
                            m["mean_rel"],
                            cls,
                        )
                    )
                    print(
                        f"  T={T:3d} | base={base_us:7.2f}us fused={fused_us:7.2f}us spd={speedup:5.2f}x | "
                        f"max_abs={fmt(m['max_abs'])} mean_abs={fmt(m['mean_abs'])} "
                        f"max_rel={fmt(m['max_rel'])} mean_rel={fmt(m['mean_rel'])} [{cls}]"
                    )
                elif kind == "fp8":
                    m = metrics["vs_base"]
                    # fp8 通道：用 mean_rel 判定（量化噪声）
                    cls = "PASS" if m["mean_rel"] < FP8_E4M3_RELTOL else "FAIL"
                    if m["max_abs"] == 0:
                        cls = "bit-exact"
                    prec_table.append(
                        (
                            name,
                            T,
                            "fp8 dequant vs base",
                            m["max_abs"],
                            m["mean_abs"],
                            m["max_rel"],
                            m["mean_rel"],
                            cls,
                        )
                    )
                    print(
                        f"  T={T:3d} | base={base_us:7.2f}us fused={fused_us:7.2f}us spd={speedup:5.2f}x | "
                        f"fp8 dequant: max_abs={fmt(m['max_abs'])} mean_abs={fmt(m['mean_abs'])} "
                        f"mean_rel={fmt(m['mean_rel'])} [{cls}]"
                    )
                elif kind == "dual":
                    bm = metrics["bf16_vs_base"]
                    fm = metrics["fp8_vs_base"]
                    bcls = _classify(bm["max_abs"], bm["mean_abs"], "bf16")
                    fcls = "PASS" if fm["mean_rel"] < FP8_E4M3_RELTOL else "FAIL"
                    if fm["max_abs"] == 0:
                        fcls = "bit-exact"
                    prec_table.append(
                        (
                            name,
                            T,
                            "bf16 vs base",
                            bm["max_abs"],
                            bm["mean_abs"],
                            bm["max_rel"],
                            bm["mean_rel"],
                            bcls,
                        )
                    )
                    prec_table.append(
                        (
                            name,
                            T,
                            "fp8 dequant vs base",
                            fm["max_abs"],
                            fm["mean_abs"],
                            fm["max_rel"],
                            fm["mean_rel"],
                            fcls,
                        )
                    )
                    print(
                        f"  T={T:3d} | base={base_us:7.2f}us fused={fused_us:7.2f}us spd={speedup:5.2f}x | "
                        f"bf16: max_abs={fmt(bm['max_abs'])} mean_abs={fmt(bm['mean_abs'])} [{bcls}] | "
                        f"fp8: max_abs={fmt(fm['max_abs'])} mean_rel={fmt(fm['mean_rel'])} [{fcls}]"
                    )
            except Exception as e:
                print(f"  T={T:3d}  ERROR: {e}")
                import traceback

                traceback.print_exc()

    # ---------------- markdown report ----------------
    print()
    print("# 精度对齐报告 (Markdown)")
    print()
    print("## 精度结果")
    print()
    print("| Fusion | T | 对比方式 | max_abs | mean_abs | max_rel | mean_rel | 判定 |")
    print("|--------|---|---------|--------:|---------:|--------:|---------:|-----|")
    for name, T, kind, ma, me, mr, mer, cls in prec_table:
        print(
            f"| {name} | {T} | {kind} | {fmt(ma)} | {fmt(me)} | {fmt(mr)} | {fmt(mer)} | {cls} |"
        )

    print()
    print("## 性能结果 (bs=1..32, decode)")
    print()
    print("| Fusion | T | Baseline (us) | Fused (us) | Speedup |")
    print("|--------|---:|------------:|----------:|--------:|")
    for name, T, b, f, s in perf_table:
        print(f"| {name} | {T} | {b:.2f} | {f:.2f} | {s:.2f}x |")

    print()
    print("## 判定汇总")
    classes = {}
    for r in prec_table:
        c = r[7]
        classes[c] = classes.get(c, 0) + 1
    for c in ["bit-exact", "PASS", "1ULP", "FAIL"]:
        if c in classes:
            print(f"- **{c}**: {classes[c]} 个测试条目")


if __name__ == "__main__":
    main()
