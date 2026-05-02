"""UT for stage ②: per-group for-loop → single `deep_gemm.fp8_einsum` call.

Current `attention.py:_wo_a_grouped_fp8` loops `for g in range(G)` calling
`fp8_gemm_nt` each iteration (G launches).  vLLM does this as a single
batched FP8 einsum (`deepseek_v4_attention.py:325`):

    fp8_einsum("bhr,hdr->bhd",
               (o_fp8,  o_scale),
               (wo_a_fp8, wo_a_scale),
               out,
               recipe=(1, 1, 128))    # SM100

`deep_gemm.fp8_einsum` is available in our vendored DeepGEMM — same API
vLLM uses.  One launch, same FP8 numerics as the per-group loop.

Run:
  cd .../github-opensource && CUDA_VISIBLE_DEVICES=7 \
    PYTHONPATH=. /opt/conda310/bin/python3 \
    rtp_llm/models_py/modules/dsv4/test/test_wo_a_batched_vs_loop.py
"""

from __future__ import annotations

import deep_gemm
import torch
from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import fp8_gemm_nt
from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import fp8_max, fp8_min
from rtp_llm.models_py.modules.dsv4.weight_loader import _repack_v4_fp8_scale_to_int32
from rtp_llm.ops.compute_ops import per_token_group_quant_fp8


# ---------------------------------------------------------------------------
# Weight fixture: V4 ckpt layout = [G*R, K] fp8 + [G*R/128, K/128] e8m0fnu.
# ---------------------------------------------------------------------------
def _build_wo_a_fixture(G: int, R: int, K: int, *, seed: int = 0):
    torch.manual_seed(seed)
    w_fp32 = torch.randn(G * R, K, dtype=torch.float32, device="cuda") * 0.3
    weight_fp8 = w_fp32.to(torch.float8_e4m3fn).contiguous()
    # UE8M0 scale bytes near 120 (~2^-7) — finite & in-range.
    scale_bytes = torch.randint(
        115, 125, (G * R // 128, K // 128), dtype=torch.uint8, device="cuda"
    )
    scale_raw = scale_bytes.view(torch.float8_e8m0fnu).contiguous()
    return weight_fp8, scale_raw


# ---------------------------------------------------------------------------
# Reference: per-group loop (mirrors today's `_wo_a_grouped_fp8`).
# ---------------------------------------------------------------------------
def ref_wo_a_loop(o: torch.Tensor, weight_fp8, scale_raw, R: int) -> torch.Tensor:
    B, S, G, K = o.shape
    M = B * S
    out_full = torch.empty(B, S, G, R, dtype=o.dtype, device=o.device)
    for g in range(G):
        w_g = weight_fp8[g * R : (g + 1) * R].contiguous()
        s_g_raw = scale_raw[g * R // 128 : (g + 1) * R // 128].contiguous()
        s_g = _repack_v4_fp8_scale_to_int32(s_g_raw)
        x_g = o[:, :, g, :].contiguous().view(M, K)
        x_fp8, x_scale = sgl_per_token_group_quant_fp8(
            x_g,
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        out_g = torch.empty(M, R, dtype=torch.bfloat16, device=o.device)
        fp8_gemm_nt(
            (x_fp8, x_scale),
            (w_g, s_g),
            out_g,
            c=None,
            disable_ue8m0_cast=False,
        )
        out_full[:, :, g, :].copy_(out_g.view(B, S, R))
    return out_full


# ---------------------------------------------------------------------------
# Weight prep for the batched einsum: stack once at init.
# ---------------------------------------------------------------------------
def prepare_wo_a_stacked(weight_fp8, scale_raw, G: int, R: int, K: int):
    """Returns (w_stk [G, R, K] fp8, s_stk [G, R, K/512] int32).

    Weight stack is free (view). Scale: e8m0fnu [G*R/128, K/128] →
    fp32 [G, R, K/128] (row-repeat by 128) → pack via
    `get_mn_major_tma_aligned_packed_ue8m0_tensor` → [G, R, K/512] int32
    stride (K/512 * tma_R, 1, tma_R). Exactly the layout DeepGEMM's
    einsum expects for the B-operand (is_sfa=False)."""
    w_stk = weight_fp8.view(G, R, K).contiguous()
    scale_fp32 = scale_raw.float().view(G, R // 128, K // 128)
    idx = torch.arange(R, device=scale_raw.device) // 128
    scale_rep = scale_fp32.index_select(-2, idx).contiguous()  # [G, R, K/128]
    s_stk = get_mn_major_tma_aligned_packed_ue8m0_tensor(scale_rep)
    return w_stk, s_stk


# ---------------------------------------------------------------------------
# Candidate: one `fp8_einsum` call.
# ---------------------------------------------------------------------------
def cand_wo_a_einsum(
    o: torch.Tensor, w_stk: torch.Tensor, s_stk: torch.Tensor, R: int
) -> torch.Tensor:
    """o: [B, S, G, K] bf16 → [B, S, G, R] bf16 via one fp8_einsum call.

    Quant: per-group via the low-level ``per_token_group_quant_fp8`` binding
    writing directly into pre-allocated 3D buffers whose per-group 2D views
    match what the 2D kernel expects. That gives us G small quant launches
    + ONE einsum (instead of G big FP8 GEMMs). Matches vLLM's layout
    (``deepseek_v4_attention.py:325`` + ``fused_inv_rope_fp8_quant.py:236``)
    so recipe=(1, 1, 128) SM100 UE8M0 path kicks in.

    Scale buffer stride ``(K/512*tma_M, 1, tma_M)`` = "G-major, TMA-aligned
    M runs at innermost"; `.transpose(0, 1)` yields the
    ``(1, K/512*tma_M, tma_M)`` layout the einsum consumes as A-scale."""
    B, S, G, K = o.shape
    M = B * S
    tma_M = ((M + 3) // 4) * 4

    # [B, S, G, K] → [M, G, K] → permute to G-major for per-group quant.
    # Using reshape (not view) as the intermediate may not be contiguous.
    x_gmk = o.reshape(M, G, K).permute(1, 0, 2).contiguous()

    a_fp8_3d = torch.empty(G, M, K, dtype=torch.float8_e4m3fn, device=o.device)
    scale_buf = torch.empty(G * (K // 512) * tma_M, dtype=torch.int32, device=o.device)
    a_scale_3d = scale_buf.as_strided((G, M, K // 512), (K // 512 * tma_M, 1, tma_M))
    for g in range(G):
        per_token_group_quant_fp8(
            x_gmk[g],  # [M, K] contiguous bf16
            a_fp8_3d[g],  # [M, K] contiguous fp8
            a_scale_3d[g],  # [M, K/512] stride (1, tma_M) int32
            128,
            1e-4,
            fp8_min,
            fp8_max,
            True,
        )

    # For einsum "bhr,hdr->bhd": transpose G-major → batch-major.
    a_fp8 = a_fp8_3d.permute(1, 0, 2).contiguous()  # [M, G, K]
    a_scale = a_scale_3d.transpose(0, 1)  # [M, G, K/512] strided

    out = torch.empty(M, G, R, dtype=torch.bfloat16, device=o.device)
    deep_gemm.fp8_einsum(
        "bhr,hdr->bhd",
        (a_fp8, a_scale),
        (w_stk, s_stk),
        out,
        recipe=(1, 1, 128),  # SM100 INT32-packed UE8M0 (B300/L20D)
    )
    return out.view(B, S, G, R)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
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


def test_correctness():
    """Loop vs einsum must agree — both do identical quant + FP8 math,
    only scheduling differs. Diff should be at most block-FP8 ULP noise."""
    torch.manual_seed(0)
    G, R, K = 2, 512, 4096  # dsv4 Flash: n_heads=64, head_dim=128, G=2 → K=4096, R=512
    weight_fp8, scale_raw = _build_wo_a_fixture(G, R, K)
    w_stk, s_stk = prepare_wo_a_stacked(weight_fp8, scale_raw, G, R, K)

    for B, S in [(1, 1), (8, 1), (64, 1), (1, 64), (4, 256)]:
        o = torch.randn(B, S, G, K, dtype=torch.bfloat16, device="cuda") * 0.5
        y_ref = ref_wo_a_loop(o, weight_fp8, scale_raw, R)
        y_cand = cand_wo_a_einsum(o, w_stk, s_stk, R)
        d = (y_ref.float() - y_cand.float()).abs()
        print(
            f"  [B={B} S={S} G={G} R={R} K={K}]  "
            f"max={d.max():.4e}  mean={d.mean():.4e}  "
            f"rel-max={(d / (y_ref.float().abs() + 1e-6)).max():.4e}"
        )
        assert (
            d.max() <= 1e-2
        ), f"einsum vs loop diff too large @ B={B} S={S}: {d.max()}"


def bench_decode_B_sweep():
    """Decode shape sweep. G=2, head_dim=128, n_heads=64, o_lora_rank=512."""
    G, R, K = 2, 512, 4096
    weight_fp8, scale_raw = _build_wo_a_fixture(G, R, K, seed=1)
    w_stk, s_stk = prepare_wo_a_stacked(weight_fp8, scale_raw, G, R, K)

    print(f"  [G={G} R={R} K={K}  decode S=1]")
    print("    {:>5}  {:>10}  {:>10}  {:>10}".format("B", "loop", "einsum", "speedup"))
    Blist = [1, 8, 16, 64, 256]
    results = []
    for B in Blist:
        o = torch.randn(B, 1, G, K, dtype=torch.bfloat16, device="cuda")

        def run_loop(o):
            return ref_wo_a_loop(o, weight_fp8, scale_raw, R)

        def run_einsum(o):
            return cand_wo_a_einsum(o, w_stk, s_stk, R)

        t_l = _bench(run_loop, o)
        t_b = _bench(run_einsum, o)
        results.append((B, t_l, t_b))
        print(
            "    {:5d}  {:8.2f}us  {:8.2f}us  {:8.2f}x".format(
                B, t_l * 1e3, t_b * 1e3, t_l / t_b
            )
        )
    return results


if __name__ == "__main__":
    print("== Correctness ==")
    test_correctness()
    print("\n== Benchmark (decode B sweep) ==")
    results = bench_decode_B_sweep()
    fail = False
    for B, t_l, t_b in results:
        if not (t_b <= t_l * 1.05):
            print(f"  [FAIL] B={B}: einsum={t_b*1e3:.2f}us > loop={t_l*1e3:.2f}us")
            fail = True
    assert not fail, "einsum must not regress vs loop at any B"
    print("\nOK")
