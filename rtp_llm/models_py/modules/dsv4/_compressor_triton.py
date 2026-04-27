"""DeepSeek-V4 Compressor pool Triton kernel — fused softmax + weighted-sum.

Replaces the per-layer `(kv * score.softmax(dim=2)).sum(dim=2)` chain in
`compressor.py` (final_plan.md P2):

    score = score.softmax(dim=2)                  # [B, NB, G, D] fp32 — per-D softmax over G
    kv    = (kv * score).sum(dim=2)               # [B, NB, D]    fp32

with a single Triton kernel that streams per-D tiles, computes softmax
in-place on score, and accumulates the weighted sum across G.

Why only the pool step?  RMSNorm + RoPE downstream operate on the
already-pooled [B, NB, D] tensor and use existing batched ops; folding
them in would require either a 512KB shared-memory tile (HCA G=128, D=512)
or a two-pass scratch buffer.  The pool step alone accounts for the 10+
launches that dominate compressor REF; RMSNorm and apply_rotary_emb add
~7 launches but stay in Python via the same paths the REF uses.

CSA (overlap=True, ratio=4)   → G = 2*ratio = 8,  D = head_dim = 512
HCA (overlap=False, ratio=128) → G = ratio    = 128, D = head_dim = 512

The wrapper accepts post-overlap-transform shapes — overlap_transform
itself stays in Python (kv_state book-keeping needs the same ops).

State writes (kv_state, score_state for incremental decode), the wkv /
wgate FP32 GEMMs, and the post-pool RMSNorm + RoPE all live in
`compressor.py` unchanged.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _v4_compressor_pool_fwd(
    kv_ptr,         # [B, NB, G, D] fp32
    score_ptr,      # [B, NB, G, D] fp32
    out_ptr,        # [B, NB, D]    fp32

    # strides — D-stride==1 assumed (caller asserts contiguous)
    kv_b: tl.constexpr, kv_n: tl.constexpr, kv_g: tl.constexpr,
    score_b: tl.constexpr, score_n: tl.constexpr, score_g: tl.constexpr,
    out_b: tl.constexpr, out_n: tl.constexpr,

    # geometry
    G: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program handles one (b, nb, d_tile).

    Loads score[G, BLOCK_D] and kv[G, BLOCK_D] tiles, computes per-D
    softmax over G, weighted sum, writes [BLOCK_D] fp32 to out.

    -inf entries in score (carried in by overlap_transform's fill value)
    contribute zero after exp(-inf - max) = 0; the divisor sum still
    behaves correctly since at least one entry per (b, nb, d_pos) is
    finite (the kernel doesn't need a special path for them).
    """
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_d = tl.program_id(2)

    g_off = tl.arange(0, G)
    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_off < D

    score_ptrs = (
        score_ptr
        + pid_b * score_b
        + pid_n * score_n
        + g_off[:, None] * score_g
        + d_off[None, :]
    )
    # Mask invalid d columns to -inf so softmax denom matches REF (those
    # cols won't be read — we mask the store too).
    score = tl.load(
        score_ptrs,
        mask=d_mask[None, :],
        other=float("-inf"),
    ).to(tl.float32)

    # Per-D softmax over G axis.
    score_max = tl.max(score, axis=0)                       # [BLOCK_D]
    score = tl.exp(score - score_max[None, :])              # [G, BLOCK_D]
    score_sum = tl.sum(score, axis=0)                       # [BLOCK_D]
    # Guard against all-(-inf) columns (would give 0/0); REF gets nan in
    # the same case so we let it propagate, but avoid crashing on div.
    score = score / score_sum[None, :]

    kv_ptrs = (
        kv_ptr
        + pid_b * kv_b
        + pid_n * kv_n
        + g_off[:, None] * kv_g
        + d_off[None, :]
    )
    kv = tl.load(kv_ptrs, mask=d_mask[None, :], other=0.0).to(tl.float32)

    out = tl.sum(kv * score, axis=0)                        # [BLOCK_D]

    out_ptrs = out_ptr + pid_b * out_b + pid_n * out_n + d_off
    tl.store(out_ptrs, out, mask=d_mask)


def v4_compressor_pool(
    kv: torch.Tensor,      # [B, NB, G, D] fp32
    score: torch.Tensor,   # [B, NB, G, D] fp32
) -> torch.Tensor:
    """Fused per-D softmax over G + weighted-sum.

    Returns ``out [B, NB, D]`` in fp32, matching the REF math
    ``(kv * score.softmax(dim=2)).sum(dim=2)``.

    Both inputs must be contiguous in the last (D) axis.  The G axis
    must be a power of 2 ≤ 256 (CSA: 8, HCA: 128 in V4-Flash).
    """
    assert kv.dim() == 4 and score.dim() == 4
    assert kv.shape == score.shape
    B, NB, G, D = kv.shape
    assert kv.dtype == torch.float32 and score.dtype == torch.float32, (
        f"expected fp32 inputs, got kv={kv.dtype} score={score.dtype}"
    )
    assert kv.stride(-1) == 1 and score.stride(-1) == 1, (
        "kv and score must be contiguous along D"
    )
    assert G & (G - 1) == 0 and G <= 256, (
        f"G must be a power of 2, ≤ 256 (got {G})"
    )

    out = torch.empty((B, NB, D), dtype=torch.float32, device=kv.device)
    if NB == 0:
        return out

    # BLOCK_D: HCA (G=128) keeps tile small to fit shared mem; CSA (G=8)
    # can take a much larger tile.  Both cap at D so we don't pay grid-
    # launch overhead on tiny D.
    if G >= 64:
        BLOCK_D = min(64, triton.next_power_of_2(D))
    else:
        BLOCK_D = min(256, triton.next_power_of_2(D))

    grid = (B, NB, triton.cdiv(D, BLOCK_D))

    _v4_compressor_pool_fwd[grid](
        kv, score, out,
        kv.stride(0), kv.stride(1), kv.stride(2),
        score.stride(0), score.stride(1), score.stride(2),
        out.stride(0), out.stride(1),
        G=G, D=D, BLOCK_D=BLOCK_D,
    )
    return out
