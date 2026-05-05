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

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _v4_compressor_pool_fwd(
    kv_ptr,  # raw state ptr — see OVERLAP for layout
    score_ptr,
    out_ptr,  # [B, NB, D]    fp32
    # strides — D-stride==1 assumed (caller asserts contiguous)
    kv_b: tl.constexpr,
    kv_n: tl.constexpr,
    kv_g: tl.constexpr,
    score_b: tl.constexpr,
    score_n: tl.constexpr,
    score_g: tl.constexpr,
    out_b: tl.constexpr,
    out_n: tl.constexpr,
    # geometry
    G: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    # P2: when OVERLAP=True the input is the *raw* CSA state of shape
    # [B, NB=1, 2r, 2d] in memory and we synthesize the post-cat view
    # ``cat([state[:, :ratio, :d], state[:, ratio:, d:]], dim=g)`` directly
    # in the load step.  G must equal 2*ratio in that case, and the caller
    # must pass the *raw-state* d-axis stride (== 2*D).
    OVERLAP: tl.constexpr = False,
    SPLIT_D: tl.constexpr = 0,  # = D when OVERLAP, else unused
):
    """One program handles one (b, nb, d_tile).

    OVERLAP=False (HCA decode + indexer + post-_overlap_transform inputs):
        Loads score[G, BLOCK_D] and kv[G, BLOCK_D] tiles directly.
    OVERLAP=True  (CSA decode raw-state path, P2):
        ``g < ratio`` rows take the LOWER half of the raw state's d axis;
        ``g >= ratio`` rows take the UPPER half.  Result is identical to
        feeding the kernel the post-``torch.cat`` view, but skips the
        2× alloc+copy.

    -inf entries in score (overlap_transform fill value) contribute zero
    after exp(-inf - max) = 0; the softmax denom stays correct.
    """
    pid_b = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    pid_d = tl.program_id(2).to(tl.int64)

    g_off = tl.arange(0, G)
    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_off < D

    if OVERLAP:
        # Raw state d-index: lower half rows read [d_off]; upper half read
        # [d_off + SPLIT_D].  Encoded as a per-row offset.
        ratio = G // 2
        upper = (g_off >= ratio).to(tl.int64)  # [G] 0/1
        d_idx = d_off[None, :] + upper[:, None] * SPLIT_D  # [G, BLOCK_D]
    else:
        d_idx = d_off[None, :] + (
            g_off[:, None] - g_off[:, None]
        )  # broadcast to [G, BLOCK_D]

    score_ptrs = (
        score_ptr + pid_b * score_b + pid_n * score_n + g_off[:, None] * score_g + d_idx
    )
    score = tl.load(
        score_ptrs,
        mask=d_mask[None, :],
        other=float("-inf"),
    ).to(tl.float32)

    # Per-D softmax over G axis.
    score_max = tl.max(score, axis=0)  # [BLOCK_D]
    score = tl.exp(score - score_max[None, :])  # [G, BLOCK_D]
    score_sum = tl.sum(score, axis=0)  # [BLOCK_D]
    score = score / score_sum[None, :]

    kv_ptrs = kv_ptr + pid_b * kv_b + pid_n * kv_n + g_off[:, None] * kv_g + d_idx
    kv = tl.load(kv_ptrs, mask=d_mask[None, :], other=0.0).to(tl.float32)

    out = tl.sum(kv * score, axis=0)  # [BLOCK_D]

    out_ptrs = out_ptr + pid_b * out_b + pid_n * out_n + d_off
    tl.store(out_ptrs, out, mask=d_mask)


def v4_compressor_pool(
    kv: torch.Tensor,
    score: torch.Tensor,
    *,
    overlap: bool = False,
    out_d: Optional[int] = None,
) -> torch.Tensor:
    """Fused per-D softmax over G + weighted-sum.

    Two modes (selected by ``overlap``):

    ``overlap=False`` (default — HCA decode, indexer compressor, prefill
    post-``_overlap_transform``):
        ``kv``, ``score`` shape ``[B, NB, G, D]`` fp32, contiguous in D.
        Returns ``out [B, NB, D]`` fp32 == ``(kv * score.softmax(dim=2)).sum(dim=2)``.

    ``overlap=True`` (P2 — CSA decode raw-state path):
        ``kv``, ``score`` shape ``[B, NB, 2r, 2d]`` fp32, contiguous in D.
        Effective ``G = 2r``, output ``D = d`` (= ``out_d``, default ``2d // 2``).
        Result equals feeding the kernel the post-``torch.cat`` view
            ``torch.cat([state[:, :, :r, :d], state[:, :, r:, d:]], dim=2)``
        but skips the 2× alloc+copy.

    G must be a power of 2 ≤ 256.  CSA: G=8, HCA: G=128 in V4-Flash.
    """
    assert kv.dim() == 4 and score.dim() == 4
    assert kv.shape == score.shape
    B, NB, G, D_in = kv.shape
    assert (
        kv.dtype == torch.float32 and score.dtype == torch.float32
    ), f"expected fp32 inputs, got kv={kv.dtype} score={score.dtype}"
    assert (
        kv.stride(-1) == 1 and score.stride(-1) == 1
    ), "kv and score must be contiguous along D"
    assert G & (G - 1) == 0 and G <= 256, f"G must be a power of 2, ≤ 256 (got {G})"

    if overlap:
        assert G % 2 == 0, "overlap requires G = 2*ratio (even)"
        D = out_d if out_d is not None else D_in // 2
        assert (
            D * 2 == D_in
        ), f"overlap raw-state D_in must equal 2*out_d (got D_in={D_in}, out_d={D})"
        SPLIT_D = D
    else:
        D = D_in
        SPLIT_D = 0

    out = torch.empty((B, NB, D), dtype=torch.float32, device=kv.device)
    if NB == 0:
        return out

    if G >= 64:
        BLOCK_D = min(64, triton.next_power_of_2(D))
    else:
        BLOCK_D = min(256, triton.next_power_of_2(D))

    grid = (B, NB, triton.cdiv(D, BLOCK_D))

    _v4_compressor_pool_fwd[grid](
        kv,
        score,
        out,
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        out.stride(0),
        out.stride(1),
        G=G,
        D=D,
        BLOCK_D=BLOCK_D,
        OVERLAP=overlap,
        SPLIT_D=SPLIT_D,
    )
    return out
