"""Leaner single-pass Triton sparse MLA forward (per-iteration overhead variant).

Semantically identical to
:func:`rtp_llm.models_py.triton_kernels.lsa_attention.sparse_mla_triton.sparse_mla_fwd`
(same signature, same padded-slot / empty-row behaviour, same natural-log LSE),
but the inner loop over the slot blocks is restructured to cut the per-iteration
scalar/address work that competes with the gather:

* **int32 slot ids, loaded once per tile.** ``indices`` are normalised to int32
  in the wrapper and the per-tile load is unmasked whenever ``S % BLOCK_N == 0``
  (the steady-state case), so no slot predicate arithmetic is emitted. All row
  address math below stays in 32-bit.
* **Hoisted row offsets / masks.** ``row_ok`` (validity) is computed once per
  tile; the clamped ``idx_safe * DQ`` row offset is computed once and reused by
  the four value gathers, the rope gather and (fp8) the scale gathers.
* **Unmasked gathers.** Every KV load is issued unmasked with a clamped row
  index: invalid rows read a valid row and are removed afterwards by the same
  ``-inf`` score mask (and by ``p == 0``) the baseline applies, so no predicated
  gather path is generated.
* **``tl.static_range`` over the four 128-wide ``c_val`` chunks** (DV = 512).
  The value tile is gathered/dotted as 4 x [BLOCK_N, 128] chunks, the rope tile
  is loaded once, and the output accumulator is kept as four [H, 128] chunk
  accumulators; this keeps each gather/dot unit small and lets the unrolled
  chunk pipeline overlap loads with MMA.
* **Rope offsets computed once** (``rope_off = DV + arange(ROPE)``) instead of
  being rebuilt inside every tile iteration.

Softmax, rescaling and LSE follow the baseline expression-for-expression.

Not covered here: split-KV decode (see ``sparse_mla_fwd_split``), CUDA graphs,
TMA. FP8 KV is supported and uses the same per-128-block scales as the baseline.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.lsa_attention.reference import (
    DEFAULT_KV_LORA_RANK,
    FP8_KV_BLOCK,
)
from rtp_llm.models_py.triton_kernels.lsa_attention.sparse_mla_triton import (
    _OUT_DTYPES,
    _validate,
    sparse_mla_fwd,
)

#: Width of one ``c_val`` sub-tile (DV=512 -> 4 chunks of 128).
CHUNK_WIDTH = 128


@triton.jit
def _sparse_fast_fwd_kernel(
    Q,
    KV,
    IDX,
    SCALES,
    OUT,
    LSE,
    T,
    N,
    NB,
    softmax_scale,
    DQ: tl.constexpr,
    DV: tl.constexpr,
    ROPE: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
    USE_FP8: tl.constexpr,
    FPB: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """One query token x all ``H`` heads per program (fast single pass)."""
    NCHUNK: tl.constexpr = DV // CHUNK
    t = tl.program_id(0)
    offs_h = tl.arange(0, H)
    offs_n = tl.arange(0, BLOCK_N)
    offs_c = tl.arange(0, CHUNK)
    offs_r = tl.arange(0, ROPE)

    # Query chunks + rope loaded once per program; rope column offsets are
    # hoisted so tile iterations only add the (clamped) row offset.
    q_row = Q + t * (H * DQ) + offs_h[:, None] * DQ
    q_chunks = ()
    for c in tl.static_range(NCHUNK):
        q_chunks = q_chunks + (tl.load(q_row + c * CHUNK + offs_c[None, :]),)
    q_rope = tl.load(q_row + DV + offs_r[None, :])
    rope_off = DV + offs_r

    idx_base = IDX + t * S

    m_i = tl.full([H], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([H], dtype=tl.float32)
    accs = ()
    for c in tl.static_range(NCHUNK):
        accs = accs + (tl.zeros([H, CHUNK], dtype=tl.float32),)

    for n0 in range(0, S, BLOCK_N):
        if S % BLOCK_N == 0:
            # Steady state: no slot predicate, ids land directly in registers.
            idx = tl.load(idx_base + n0 + offs_n)
        else:
            slots = n0 + offs_n
            idx = tl.load(idx_base + slots, mask=slots < S, other=-1)

        # Row validity + row offset computed once per tile, reused by every
        # gather below (clamped so the unmasked loads stay in bounds).
        row_ok = (idx >= 0) & (idx < N)
        idx_safe = tl.minimum(tl.maximum(idx, 0), N - 1)
        row_off = idx_safe * DQ

        v_chunks = ()
        s_block = tl.zeros([H, BLOCK_N], dtype=tl.float32)
        for c in tl.static_range(NCHUNK):
            v_c = tl.load(KV + row_off[:, None] + c * CHUNK + offs_c[None, :])
            if USE_FP8:
                sc = tl.load(SCALES + idx_safe * NB + (c * CHUNK) // FPB)
                v_c = (v_c.to(tl.float32) * sc[:, None]).to(tl.bfloat16)
            v_chunks = v_chunks + (v_c,)
            s_block = tl.dot(q_chunks[c], tl.trans(v_c), s_block)

        k_rope = tl.load(KV + row_off[:, None] + rope_off[None, :])
        if USE_FP8:
            sc_r = tl.load(SCALES + idx_safe * NB + (NB - 1))
            k_rope = (k_rope.to(tl.float32) * sc_r[:, None]).to(tl.bfloat16)
        s_block = tl.dot(q_rope, tl.trans(k_rope), s_block)
        s_block = s_block * softmax_scale
        s_block = tl.where(row_ok[None, :], s_block, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s_block, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_safe))
        p = tl.where(row_ok[None, :], tl.exp(s_block - m_safe[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new
        p16 = p.to(v_chunks[0].dtype)

        new_accs = ()
        for c in tl.static_range(NCHUNK):
            new_accs = new_accs + (
                tl.dot(p16, v_chunks[c], accs[c] * alpha[:, None]),
            )
        accs = new_accs

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    out_row = OUT + t * (H * DV) + offs_h[:, None] * DV
    for c in tl.static_range(NCHUNK):
        o_c = accs[c] / l_safe[:, None]
        tl.store(out_row + c * CHUNK + offs_c[None, :], o_c.to(OUT_DTYPE))
    lse = tl.where(l_i > 0, m_i + tl.log(l_safe), float("-inf"))
    tl.store(LSE + t * H + offs_h, lse)


def sparse_fast_fwd(
    q_abs: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    scale: Optional[float] = None,
    kv_lora_rank: int = DEFAULT_KV_LORA_RANK,
    out_dtype: Optional[torch.dtype] = None,
    block_n: int = 16,
    num_warps: int = 4,
    num_stages: int = 1,
    kv_scales: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drop-in replacement for ``sparse_mla_fwd`` (see module docstring).

    Delegates to the baseline kernel when ``kv_lora_rank`` cannot be split into
    chunks of at least 16 lanes (``kv_lora_rank < 16``), which ``tl.dot`` could
    not consume either way; every other shape accepted by ``_validate`` runs on
    the chunked schedule.
    """
    q_abs, kv, indices, t_count, heads, dq, dv = _validate(
        q_abs, kv, indices, kv_lora_rank
    )

    chunk = CHUNK_WIDTH if dv % CHUNK_WIDTH == 0 else dv
    if chunk < 16 or dv % chunk or kv.shape[0] == 0:
        # Unrepresentable chunking, or an empty KV buffer (the clamped row
        # offset would step before the buffer); the baseline handles both.
        return sparse_mla_fwd(
            q_abs,
            kv,
            indices,
            scale=scale,
            kv_lora_rank=kv_lora_rank,
            out_dtype=out_dtype,
            block_n=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
            kv_scales=kv_scales,
        )

    resolved_out_dtype = out_dtype or q_abs.dtype
    if resolved_out_dtype not in _OUT_DTYPES:
        raise ValueError(f"unsupported out_dtype {resolved_out_dtype}")

    use_fp8 = kv.dtype == torch.float8_e4m3fn
    fpb = FP8_KV_BLOCK
    if use_fp8:
        if dv % fpb != 0 or (dq - dv) > fpb:
            raise ValueError("fp8 KV requires kv_lora_rank % 128 == 0 and rope <= 128")
        if kv_scales is None:
            raise ValueError("fp8 KV requires kv_scales [N, ceil(DQ/128)]")
        nb = (dq + fpb - 1) // fpb
        if tuple(kv_scales.shape) != (kv.shape[0], nb):
            raise ValueError(f"kv_scales must be [{kv.shape[0]}, {nb}]")
        scales_arg = kv_scales.contiguous()
        nb_arg = nb
        kv_arg = kv
    else:
        if kv_scales is not None:
            raise ValueError("kv_scales is only valid with an fp8 KV tensor")
        scales_arg = kv  # unused placeholder keeps one uniform launch signature
        nb_arg = 1
        kv_arg = kv

    out = torch.empty(
        (t_count, heads, dv), device=q_abs.device, dtype=resolved_out_dtype
    )
    lse = torch.empty((t_count, heads), device=q_abs.device, dtype=torch.float32)
    _sparse_fast_fwd_kernel[(t_count,)](
        q_abs,
        kv_arg,
        indices,
        scales_arg,
        out,
        lse,
        t_count,
        kv.shape[0],
        nb_arg,
        scale if scale is not None else dq**-0.5,
        DQ=dq,
        DV=dv,
        ROPE=dq - dv,
        H=heads,
        S=indices.shape[1],
        BLOCK_N=block_n,
        CHUNK=chunk,
        USE_FP8=use_fp8,
        FPB=fpb,
        OUT_DTYPE=_OUT_DTYPES[resolved_out_dtype],
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, lse