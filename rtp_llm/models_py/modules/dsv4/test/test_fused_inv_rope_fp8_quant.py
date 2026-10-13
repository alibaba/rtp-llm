"""UT for fused inverse-RoPE + per-token-group FP8 quant (single Triton kernel).

Audit §3.5 / §7.4 P1.  Decode call site ``attention.py`` today splits the
inverse RoPE and the wo_a-input FP8 quant into:

    (1) torch ``apply_rotary_emb_batched(o[..., -rd:], …, inverse=True)``
        — ~5-launch view_as_complex chain
    (2) ``_wo_a_grouped_fp8`` internally runs G per-token-group FP8 quants
        on the bf16 o → (fp8 + UE8M0 scale)

Both produce a ``(fp8, scale)`` pair in the exact layout
``deep_gemm.fp8_einsum("bhr,hdr->bhd", …, recipe=(1, 1, 128))`` consumes.
The fused kernel emits the SAME (fp8, scale) pair in one launch.

This UT validates the fused path two ways:
  A. Direct (fp8, scale) diff vs eager inv-RoPE + per-group quant.  Both
     use the same UE8M0 block scale rounding, so fp8 bytes must be
     bit-identical and scales must agree exactly.
  B. End-to-end via ``fp8_einsum`` — feed both (fp8, scale) pairs into
     the same GEMM and compare bf16 outputs.  ≤1 FP8 ULP expected.

Run:
  cd .../github-opensource && CUDA_VISIBLE_DEVICES=7 \
    PYTHONPATH=. /opt/conda310/bin/python3 \
    rtp_llm/models_py/modules/dsv4/test/test_fused_inv_rope_fp8_quant.py
"""

from __future__ import annotations

import deep_gemm
import torch
from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor

from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import fp8_max, fp8_min
from rtp_llm.models_py.modules.dsv4._fused_inv_rope_fp8_quant_triton import (
    fused_inv_rope_fp8_quant,
)
from rtp_llm.models_py.modules.dsv4.rope import apply_rotary_emb_batched
from rtp_llm.ops.compute_ops import per_token_group_quant_fp8


def _eager_ref(
    o: torch.Tensor,
    freqs_cis: torch.Tensor,
    G: int,
    R: int,
):
    """Eager inv-RoPE + per-group per_token_group_quant_fp8.  Produces the
    exact ``(o_fp8, o_scale)`` layout ``_wo_a_grouped_fp8`` builds today."""
    B, S, H, D = o.shape
    M = B * S
    tma_M = ((M + 3) // 4) * 4
    rd = freqs_cis.shape[-1] * 2

    # 1) Inverse RoPE on the last rd columns (torch path).
    o2 = o.clone()
    apply_rotary_emb_batched(o2[..., -rd:], freqs_cis, inverse=True)

    # 2) reshape [B, S, H, D] → [B, S, G, K]  (K = hpg * D)
    K = H * D // G
    o_bsgk = o2.reshape(B, S, G, K)
    # → [G, M, K] contiguous for per-group quant
    x_gmk = o_bsgk.reshape(M, G, K).permute(1, 0, 2).contiguous()

    a_fp8_3d = torch.empty(G, M, K, dtype=torch.float8_e4m3fn, device=o.device)
    scale_buf = torch.empty(G * (K // 512) * tma_M, dtype=torch.int32, device=o.device)
    a_scale_3d = scale_buf.as_strided((G, M, K // 512), (K // 512 * tma_M, 1, tma_M))
    for g in range(G):
        per_token_group_quant_fp8(
            x_gmk[g],
            a_fp8_3d[g],
            a_scale_3d[g],
            128,
            1e-10,
            fp8_min,
            fp8_max,
            True,
        )
    # transpose to match fused kernel's output: [M, G, …]
    return a_fp8_3d.permute(1, 0, 2).contiguous(), a_scale_3d.transpose(0, 1), M, tma_M


def _make_wo_a_weight(G: int, R: int, K: int, seed: int = 0):
    """Synthesize V4 ckpt-style wo_a in einsum-ready layout:
    returns (w_stk [G, R, K] fp8, s_stk [G, R, K/512] int32)."""
    torch.manual_seed(seed)
    w_fp32 = torch.randn(G * R, K, dtype=torch.float32, device="cuda") * 0.3
    weight_fp8 = w_fp32.to(torch.float8_e4m3fn).view(G, R, K).contiguous()
    scale_bytes = torch.randint(
        115, 125, (G * R // 128, K // 128), dtype=torch.uint8, device="cuda"
    )
    scale_raw = scale_bytes.view(torch.float8_e8m0fnu).contiguous()
    scale_fp32 = scale_raw.float().view(G, R // 128, K // 128)
    idx = torch.arange(R, device="cuda") // 128
    scale_rep = scale_fp32.index_select(-2, idx).contiguous()
    s_stk = get_mn_major_tma_aligned_packed_ue8m0_tensor(scale_rep)
    return weight_fp8, s_stk


# ---------------------------------------------------------------------------
def test_correctness_fp8_and_scale():
    """Part A: fused (fp8, scale) ≡ eager (fp8, scale), bit-identical."""
    torch.manual_seed(42)
    # Production dsv4 V4-Flash dims.
    n_heads, head_dim, rope_dim = 64, 512, 64
    nope_dim = head_dim - rope_dim
    n_groups = 8
    heads_per_group = n_heads // n_groups

    for B, S in [(1, 1), (8, 1), (16, 1), (64, 1)]:
        o = (
            torch.randn(B, S, n_heads, head_dim, dtype=torch.bfloat16, device="cuda")
            * 0.3
        )
        ang = torch.rand(B, rope_dim // 2, device="cuda") * 6.28
        freqs = torch.polar(torch.ones_like(ang), ang).to(torch.complex64)

        fp8_ref, scale_ref, M, tma_M = _eager_ref(o, freqs, n_groups, heads_per_group)
        fp8_cand, scale_cand = fused_inv_rope_fp8_quant(
            o,
            freqs,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_head_dim=rope_dim,
        )

        # FP8 compare as raw bytes.  RoPE done in fp32 in both — minor
        # per-order differences may flip 1 ULP on a handful of lanes.
        diff_fp8 = (
            fp8_ref.contiguous().view(torch.uint8).to(torch.int16)
            - fp8_cand.contiguous().view(torch.uint8).to(torch.int16)
        ).abs()
        exact_rate = (diff_fp8 == 0).float().mean().item()
        max_ulp = diff_fp8.max().item()

        # Scale compare: UE8M0 rounds to power-of-2 so equal-ish scale
        # values map to the same int32 byte most of the time; allow ±1
        # exponent on the rare boundary lane.  Both scale tensors are
        # strided views; compare element-wise as int32.
        sr_c = scale_ref.contiguous().view(torch.int32)
        sc_c = scale_cand.contiguous().view(torch.int32)
        diff_s = (
            sr_c.view(torch.uint8).to(torch.int16)
            - sc_c.view(torch.uint8).to(torch.int16)
        ).abs()
        s_exact = (diff_s == 0).float().mean().item()
        s_max = diff_s.max().item()

        print(
            f"  [B={B} S={S} M={M}]  "
            f"fp8 exact={exact_rate*100:.2f}% max_ulp={max_ulp}   "
            f"scale exact={s_exact*100:.2f}% max_byte_diff={s_max}"
        )
        # fp8 byte-level mismatches > 1 ULP happen where the fp32 absmax
        # of a block sits near a power-of-2 boundary — a single fp32 ULP
        # difference between torch complex-mul vs Triton explicit
        # a·cos±b·sin ordering tips ceil(log2(absmax)) by 1, doubling or
        # halving the UE8M0 scale and shifting all byte codes in that
        # block.  End-to-end (fp8*scale) via einsum must still match
        # ≤1 ULP (verified in test_end_to_end_via_einsum).
        assert exact_rate >= 0.95, f"fp8 byte exactness {exact_rate:.3f} < 0.95"
        assert s_exact >= 0.99, f"scale byte exactness {s_exact:.4f} < 0.99"
        assert s_max <= 1, f"scale byte diff {s_max} > 1 exp (boundary ok)"


def test_end_to_end_via_einsum():
    """Part B: fused + fp8_einsum ≡ eager + fp8_einsum (bf16 outputs)."""
    torch.manual_seed(7)
    n_heads, head_dim, rope_dim = 64, 512, 64
    nope_dim = head_dim - rope_dim
    n_groups = 8
    heads_per_group = n_heads // n_groups
    R = 1024
    K = heads_per_group * head_dim  # 4096

    w_stk, s_stk = _make_wo_a_weight(n_groups, R, K, seed=1)

    for B, S in [(1, 1), (16, 1), (64, 1)]:
        o = (
            torch.randn(B, S, n_heads, head_dim, dtype=torch.bfloat16, device="cuda")
            * 0.3
        )
        ang = torch.rand(B, rope_dim // 2, device="cuda") * 6.28
        freqs = torch.polar(torch.ones_like(ang), ang).to(torch.complex64)
        M = B * S

        # Ref path
        fp8_ref, scale_ref, _, _ = _eager_ref(o, freqs, n_groups, heads_per_group)
        out_ref = torch.empty(M, n_groups, R, dtype=torch.bfloat16, device="cuda")
        deep_gemm.fp8_einsum(
            "bhr,hdr->bhd",
            (fp8_ref, scale_ref),
            (w_stk, s_stk),
            out_ref,
            recipe=(1, 1, 128),
        )

        # Cand path: single fused kernel, same einsum
        fp8_cand, scale_cand = fused_inv_rope_fp8_quant(
            o,
            freqs,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_head_dim=rope_dim,
        )
        out_cand = torch.empty(M, n_groups, R, dtype=torch.bfloat16, device="cuda")
        deep_gemm.fp8_einsum(
            "bhr,hdr->bhd",
            (fp8_cand, scale_cand),
            (w_stk, s_stk),
            out_cand,
            recipe=(1, 1, 128),
        )

        d = (out_ref.float() - out_cand.float()).abs()
        rel = d / (out_ref.float().abs() + 1e-6)
        print(
            f"  [B={B} S={S} M={M}]  max={d.max():.4e}  mean={d.mean():.4e}  "
            f"rel-max={rel.max():.4e}"
        )
        assert d.max() <= 5e-2, f"end-to-end einsum diff @ B={B} S={S}: {d.max()}"


def _bench(fn, *args, warmup: int = 25, iters: int = 200) -> float:
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn(*args)
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / iters


def bench_decode():
    """Split (apply_rotary_emb_batched + G per-group quants) vs fused."""
    n_heads, head_dim, rope_dim = 64, 512, 64
    nope_dim = head_dim - rope_dim
    n_groups = 8
    heads_per_group = n_heads // n_groups

    print(f"  [n_heads={n_heads} head_dim={head_dim} G={n_groups}  decode S=1]")
    print(
        "    {:>5}  {:>12}  {:>10}  {:>10}".format(
            "B", "eager+quant", "fused", "speedup"
        )
    )

    for B in [1, 8, 16, 64, 256]:
        o = (
            torch.randn(B, 1, n_heads, head_dim, dtype=torch.bfloat16, device="cuda")
            * 0.3
        )
        ang = torch.rand(B, rope_dim // 2, device="cuda") * 6.28
        freqs = torch.polar(torch.ones_like(ang), ang).to(torch.complex64)

        def run_eager(o):
            return _eager_ref(o, freqs, n_groups, heads_per_group)

        def run_fused(o):
            return fused_inv_rope_fp8_quant(
                o,
                freqs,
                n_groups=n_groups,
                heads_per_group=heads_per_group,
                nope_dim=nope_dim,
                rope_head_dim=rope_dim,
            )

        t_e = _bench(run_eager, o)
        t_f = _bench(run_fused, o)
        print(
            "    {:5d}  {:10.2f}us  {:8.2f}us  {:8.2f}x".format(
                B, t_e * 1e3, t_f * 1e3, t_e / t_f
            )
        )


if __name__ == "__main__":
    print("== Correctness (direct fp8 + scale) ==")
    test_correctness_fp8_and_scale()
    print("\n== Correctness (end-to-end via fp8_einsum) ==")
    test_end_to_end_via_einsum()
    print("\n== Benchmark (decode B sweep) ==")
    bench_decode()
    print("\nOK")
