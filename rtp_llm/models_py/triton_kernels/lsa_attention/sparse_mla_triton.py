"""Triton implementation of the LongCat LSA sparse MLA forward operator.

Implements the same contract as :mod:`rtp_llm.models_py.triton_kernels.lsa_attention.reference`:

    q_abs  [T, H, DQ]   absorbed queries (DQ = kv_lora_rank + rope)
    kv     [N, DQ]      latent KV cache rows ([c_kv ; k_rope])
    idx    [T, S]       per-row slot indices (S = index topk budget, -1 = pad)
    ->  out [T, H, DV], lse [T, H] (natural log)

Design (v2 baseline):

* One program per **query token**, with the attention-head dimension as the
  ``tl.dot`` M axis. Required because every query row carries its own slot
  list: a shared ``[BLOCK_N, DQ]`` K tile cannot serve rows with different
  slots (same structure as FlashMLA's sparse kernel).
* Query tiles are loaded once per program and reused for all slot blocks.
* Each slot block loads its latent row **once**: the ``[BLOCK_N, DV]`` tile
  serves both the QK scores (nope part) and the ``P @ V`` step; only the rope
  slice is loaded separately. This halves KV traffic versus the naive version.
* FA2-style online softmax over slot blocks.

Not yet covered (run plan P2/P3): split-KV for decode parallelism, FP8 KV,
page-table indirection, SWA/SFA dual-stream overlap, CUDA graphs, and
indexer/top-k fusion.
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

_OUT_DTYPES = {
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
    torch.float32: tl.float32,
}


@triton.jit
def _sparse_mla_fwd_kernel(
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
    USE_FP8: tl.constexpr,
    FPB: tl.constexpr,
    NBCP: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """One query token x all ``H`` heads per program.

    ``ROPE = DQ - DV`` is the rope slice of each latent row. The ``[BLOCK_N,
    DV]`` value tile is loaded once and reused for scores and ``P @ V``.
    """
    t = tl.program_id(0)
    offs_h = tl.arange(0, H)
    offs_n = tl.arange(0, BLOCK_N)
    offs_dv = tl.arange(0, DV)
    offs_r = tl.arange(0, ROPE)

    q_base = Q + t * (H * DQ)
    q_nope = tl.load(q_base + offs_h[:, None] * DQ + offs_dv[None, :])
    q_rope = tl.load(q_base + offs_h[:, None] * DQ + DV + offs_r[None, :])
    idx_base = IDX + t * S

    m_i = tl.full([H], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([H], dtype=tl.float32)
    acc_o = tl.zeros([H, DV], dtype=tl.float32)

    for n0 in range(0, S, BLOCK_N):
        slots = n0 + offs_n
        idx = tl.load(idx_base + slots, mask=slots < S, other=-1)
        row_ok = (idx >= 0) & (idx < N)
        idx_safe = tl.where(row_ok, idx, 0)

        v_raw = tl.load(
            KV + idx_safe[:, None] * DQ + offs_dv[None, :],
            mask=row_ok[:, None],
            other=0.0,
        )
        k_rope_raw = tl.load(
            KV + idx_safe[:, None] * DQ + DV + offs_r[None, :],
            mask=row_ok[:, None],
            other=0.0,
        )
        if USE_FP8:
            offs_nb = tl.arange(0, NBCP)
            sc_all = tl.load(
                SCALES + idx_safe[:, None] * NB + offs_nb[None, :],
                mask=row_ok[:, None] & (offs_nb[None, :] < NB),
                other=1.0,
            )
            blocks = tl.broadcast_to(offs_dv[None, :] // FPB, (BLOCK_N, DV))
            sc = tl.gather(sc_all, blocks, axis=1)
            v = (v_raw.to(tl.float32) * sc).to(tl.bfloat16)
            sc_r = tl.load(SCALES + idx_safe * NB + (NB - 1), mask=row_ok, other=1.0)
            k_rope = (k_rope_raw.to(tl.float32) * sc_r[:, None]).to(tl.bfloat16)
        else:
            v = v_raw
            k_rope = k_rope_raw

        s_block = tl.dot(q_nope, tl.trans(v))
        s_block = tl.dot(q_rope, tl.trans(k_rope), s_block)
        s_block = s_block * softmax_scale
        s_block = tl.where(row_ok[None, :], s_block, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s_block, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_safe))
        p = tl.where(row_ok[None, :], tl.exp(s_block - m_safe[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc_o = acc_o * alpha[:, None]
        m_i = m_new
        acc_o = tl.dot(p.to(v.dtype), v, acc_o)

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    o = acc_o / l_safe[:, None]
    tl.store(
        OUT + t * (H * DV) + offs_h[:, None] * DV + offs_dv[None, :],
        o.to(OUT_DTYPE),
    )
    lse = tl.where(l_i > 0, m_i + tl.log(l_safe), float("-inf"))
    tl.store(LSE + t * H + offs_h, lse)


def _validate(
    q_abs: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    kv_lora_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int]:
    if not q_abs.is_cuda:
        raise ValueError("sparse MLA kernels require CUDA tensors")
    if q_abs.dim() != 3:
        raise ValueError(f"q_abs must be [T, H, DQ], got {tuple(q_abs.shape)}")
    if kv.dim() != 2 or kv.shape[1] != q_abs.shape[2]:
        raise ValueError("kv must be [N, DQ] matching q_abs' last dim")
    if indices.dim() == 3:
        if indices.shape[1] != 1:
            raise ValueError("indices with 3 dims must be [T, 1, S] (MQA broadcast)")
        indices = indices[:, 0, :]
    if indices.dim() != 2 or indices.shape[0] != q_abs.shape[0]:
        raise ValueError("indices must be [T, S] or [T, 1, S]")

    q_abs = q_abs.contiguous()
    kv = kv.contiguous()
    if indices.dtype != torch.int32:
        indices = indices.to(torch.int32)
    indices = indices.contiguous()

    t_count, heads, dq = q_abs.shape
    dv = int(kv_lora_rank)
    rope = dq - dv
    if rope <= 0 or (rope & (rope - 1)):
        raise ValueError(f"rope slice (DQ - kv_lora_rank) must be a power of two, got {rope}")
    if dv & (dv - 1):
        raise ValueError(f"kv_lora_rank {dv} must be a power of two")
    if heads < 16:
        raise ValueError(f"baseline requires heads >= 16 as the dot M axis, got {heads}")
    return q_abs, kv, indices, t_count, heads, dq, dv


def sparse_mla_fwd(
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
    """Sparse MLA forward (absorbed MQA) over a fixed per-row slot set.

    Mirrors ``flash_mla_sparse_fwd``'s calling convention: ``indices`` are
    per-row slot indices into ``kv`` (``-1`` marks padding), and the output is
    the latent attention result per head before the ``W_uv`` up-projection.
    """
    q_abs, kv, indices, t_count, heads, dq, dv = _validate(
        q_abs, kv, indices, kv_lora_rank
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
        nbcp_arg = triton.next_power_of_2(nb)
        kv_arg = kv
    else:
        if kv_scales is not None:
            raise ValueError("kv_scales is only valid with an fp8 KV tensor")
        scales_arg = kv  # unused placeholder keeps one uniform launch signature
        nb_arg = 1
        nbcp_arg = 1
        kv_arg = kv

    out = torch.empty((t_count, heads, dv), device=q_abs.device, dtype=resolved_out_dtype)
    lse = torch.empty((t_count, heads), device=q_abs.device, dtype=torch.float32)
    _sparse_mla_fwd_kernel[(t_count,)](
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
        USE_FP8=use_fp8,
        FPB=fpb,
        NBCP=nbcp_arg,
        OUT_DTYPE=_OUT_DTYPES[resolved_out_dtype],
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, lse


@triton.jit
def _sparse_mla_fwd_split_kernel(
    Q,
    KV,
    IDX,
    PART_O,
    PART_M,
    PART_L,
    T,
    N,
    softmax_scale,
    DQ: tl.constexpr,
    DV: tl.constexpr,
    ROPE: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLITS: tl.constexpr,
    HALF_DV: tl.constexpr,
):
    """Partial pass: one (query token, slot-split) pair per program."""
    t = tl.program_id(0)
    split = tl.program_id(1)
    per = tl.cdiv(S, SPLITS)
    start = split * per
    end = tl.minimum(start + per, S)

    offs_h = tl.arange(0, H)
    offs_n = tl.arange(0, BLOCK_N)
    offs_dv2 = tl.arange(0, HALF_DV)
    offs_r = tl.arange(0, ROPE)

    q_base = Q + t * (H * DQ)
    q_nope0 = tl.load(q_base + offs_h[:, None] * DQ + offs_dv2[None, :])
    q_nope1 = tl.load(q_base + offs_h[:, None] * DQ + HALF_DV + offs_dv2[None, :])
    q_rope = tl.load(q_base + offs_h[:, None] * DQ + DV + offs_r[None, :])
    idx_base = IDX + t * S

    m_i = tl.full([H], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([H], dtype=tl.float32)
    acc_o0 = tl.zeros([H, HALF_DV], dtype=tl.float32)
    acc_o1 = tl.zeros([H, HALF_DV], dtype=tl.float32)

    for n0 in range(start, end, BLOCK_N):
        slots = n0 + offs_n
        idx = tl.load(idx_base + slots, mask=slots < end, other=-1)
        row_ok = (idx >= 0) & (idx < N)
        idx_safe = tl.where(row_ok, idx, 0)

        v0 = tl.load(
            KV + idx_safe[:, None] * DQ + offs_dv2[None, :],
            mask=row_ok[:, None],
            other=0.0,
        )
        v1 = tl.load(
            KV + idx_safe[:, None] * DQ + HALF_DV + offs_dv2[None, :],
            mask=row_ok[:, None],
            other=0.0,
        )
        k_rope = tl.load(
            KV + idx_safe[:, None] * DQ + DV + offs_r[None, :],
            mask=row_ok[:, None],
            other=0.0,
        )
        s_block = tl.dot(q_nope0, tl.trans(v0))
        s_block = tl.dot(q_nope1, tl.trans(v1), s_block)
        s_block = tl.dot(q_rope, tl.trans(k_rope), s_block)
        s_block = s_block * softmax_scale
        s_block = tl.where(row_ok[None, :], s_block, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s_block, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_safe))
        p = tl.where(row_ok[None, :], tl.exp(s_block - m_safe[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc_o0 = acc_o0 * alpha[:, None]
        acc_o1 = acc_o1 * alpha[:, None]
        m_i = m_new
        acc_o0 = tl.dot(p.to(v0.dtype), v0, acc_o0)
        acc_o1 = tl.dot(p.to(v1.dtype), v1, acc_o1)

    out_base = (t * SPLITS + split) * (H * DV)
    tl.store(PART_O + out_base + offs_h[:, None] * DV + offs_dv2[None, :], acc_o0)
    tl.store(PART_O + out_base + offs_h[:, None] * DV + HALF_DV + offs_dv2[None, :], acc_o1)
    stat_base = (t * SPLITS + split) * H + offs_h
    tl.store(PART_M + stat_base, m_i)
    tl.store(PART_L + stat_base, l_i)


@triton.jit
def _sparse_mla_merge_kernel(
    PART_O,
    PART_M,
    PART_L,
    OUT,
    LSE,
    T,
    H: tl.constexpr,
    DV: tl.constexpr,
    SPLITS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """Merge slot-split partials: one (token, head) pair per program."""
    pid = tl.program_id(0)
    t = pid // H
    head = pid % H
    offs_s = tl.arange(0, SPLITS)
    offs_dv = tl.arange(0, DV)

    m = tl.load(PART_M + (t * SPLITS + offs_s) * H + head)
    l = tl.load(PART_L + (t * SPLITS + offs_s) * H + head)
    m_max = tl.max(m, axis=0)
    e = tl.where(m == float("-inf"), 0.0, tl.exp(m - m_max))
    denom = tl.sum(l * e, axis=0)
    denom_safe = tl.where(denom > 0, denom, 1.0)
    # Partial outputs are unnormalized numerators scaled to each split's local
    # max; rescale by exp(m_s - m_max) and normalize by the merged denominator.
    weight = e / denom_safe

    o = tl.load(
        PART_O + (t * SPLITS + offs_s)[:, None] * (H * DV) + head * DV + offs_dv[None, :]
    )
    merged = tl.sum(o * weight[:, None], axis=0)
    tl.store(OUT + t * (H * DV) + head * DV + offs_dv, merged.to(OUT_DTYPE))
    lse = tl.where(denom > 0, m_max + tl.log(denom_safe), float("-inf"))
    tl.store(LSE + t * H + head, lse)


def sparse_mla_fwd_split(
    q_abs: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    scale: Optional[float] = None,
    kv_lora_rank: int = DEFAULT_KV_LORA_RANK,
    out_dtype: Optional[torch.dtype] = None,
    splits: int = 8,
    block_n: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split-KV sparse MLA for decode-like shapes with few query rows.

    The slot dimension is partitioned into ``splits`` chunks processed in
    parallel, then merged (flash-decoding style). Falls back to the plain
    kernel when ``splits <= 1``.
    """
    q_abs, kv, indices, t_count, heads, dq, dv = _validate(
        q_abs, kv, indices, kv_lora_rank
    )
    if splits <= 1:
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
        )
    resolved_out_dtype = out_dtype or q_abs.dtype
    if resolved_out_dtype not in _OUT_DTYPES:
        raise ValueError(f"unsupported out_dtype {resolved_out_dtype}")

    slots = indices.shape[1]
    # Partials are carried in the input dtype (bf16): they are unnormalized
    # numerators of the same value range as the KV rows, and halving their
    # traffic matters more than the extra rounding for split-KV decode.
    partial_o = torch.empty(
        (t_count, splits, heads, dv), device=q_abs.device, dtype=q_abs.dtype
    )
    partial_m = torch.empty(
        (t_count, splits, heads), device=q_abs.device, dtype=torch.float32
    )
    partial_l = torch.empty_like(partial_m)
    out = torch.empty((t_count, heads, dv), device=q_abs.device, dtype=resolved_out_dtype)
    lse = torch.empty((t_count, heads), device=q_abs.device, dtype=torch.float32)

    _sparse_mla_fwd_split_kernel[(t_count, splits)](
        q_abs,
        kv,
        indices,
        partial_o,
        partial_m,
        partial_l,
        t_count,
        kv.shape[0],
        scale if scale is not None else dq**-0.5,
        DQ=dq,
        DV=dv,
        ROPE=dq - dv,
        H=heads,
        S=slots,
        BLOCK_N=block_n,
        SPLITS=splits,
        HALF_DV=dv // 2,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _sparse_mla_merge_kernel[(t_count * heads,)](
        partial_o,
        partial_m,
        partial_l,
        out,
        lse,
        t_count,
        H=heads,
        DV=dv,
        SPLITS=splits,
        OUT_DTYPE=_OUT_DTYPES[resolved_out_dtype],
        num_warps=4,
    )
    return out, lse

def sparse_mla_attention_auto(
    q: torch.Tensor,
    kv: torch.Tensor,
    slots: torch.Tensor,
    *,
    kv_lora_rank: int = 512,
    out_dtype: torch.dtype = torch.float32,
    single_min_t: int = 128,
    splits: int = 4,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adaptive sparse-MLA entry: single-pass for long queries, split+merge otherwise.

    Measured on B300 with the merge included in the timing; clean-GPU numbers
    (see build_logs/longcat-lsa-b300-20261017/baseline-log.md #5.52/#5.53):

    * ``T >= 128``: single pass wins (T=128: 130.0 vs 133.1 us at 128k ctx and
      114.3 vs 126.3 us at 32k; T=192: 167.4 vs 185.5; T=256: 169.6 vs 233.1),
      because the split path's partial traffic + merge stops paying off;
    * ``T <= 112``: split+merge wins (T=112: 122.0 vs 129.1; T=96: 104.8 vs
      126.6; T=80: 93.2 vs 125.2; T=64: 73.0 vs 142; T=32: 56.6 vs 133).

    The crossover sits between T=112 and T=128, hence the default threshold.
    For the split branch ``splits=4`` is equal or better than ``8`` in clean-GPU
    re-sweeps (b64 kernel 70.2 vs 73.1 us; t=32 56.8 vs 56.1 us e2e; t=112
    121.9 vs 122.0 us), hence the default.
    """
    from rtp_llm.models_py.triton_kernels.lsa_attention.merge_triton import (
        merge_attention_triton,
    )

    # FlashMLA (when runnable) beats every Triton variant across the whole batch
    # range measured on B300: t=64 41.4 us vs 126 us single-pass / 73 us split+merge;
    # t=192 84.8 us vs 162 us (baseline-log #5.133/#5.136), so it short-circuits the
    # whole dispatch below.
    try:
        from rtp_llm.models_py.triton_kernels.lsa_attention.sparse_mla_flashmla import (
            flashmla_available,
            sparse_flashmla_fwd,
        )

        if flashmla_available():
            return sparse_flashmla_fwd(
                q, kv, slots, kv_lora_rank=kv_lora_rank, out_dtype=out_dtype,
                block_n=block_n, num_warps=num_warps, num_stages=num_stages,
            )
    except Exception:
        pass

    if q.shape[0] >= single_min_t:
        # ``sparse_fast_triton`` is an opt-in, bit-identical refactor of the
        # single-pass kernel that measured 3-7% faster on B300 (baseline-log
        # #5.89); fall back to the frozen kernel if it is unavailable.
        try:
            from rtp_llm.models_py.triton_kernels.lsa_attention.sparse_mla_flashmla import (
                flashmla_available,
                sparse_flashmla_fwd,
            )

            _single_pass = sparse_flashmla_fwd if flashmla_available() else None
        except Exception:
            _single_pass = None
        if _single_pass is None:
            try:
                from rtp_llm.models_py.triton_kernels.lsa_attention.sparse_fast_triton import (
                    sparse_fast_fwd as _single_pass,
                )
            except Exception:
                _single_pass = sparse_mla_fwd
        return _single_pass(
            q, kv, slots, kv_lora_rank=kv_lora_rank, out_dtype=out_dtype,
            block_n=block_n, num_warps=num_warps, num_stages=num_stages,
        )
    part_out, part_lse = sparse_mla_fwd_split(
        q, kv, slots, kv_lora_rank=kv_lora_rank, out_dtype=out_dtype,
        splits=splits, block_n=block_n, num_warps=num_warps, num_stages=num_stages,
    )
    merged_out, merged_lse = merge_attention_triton(
        [(part_out, part_lse)], out_dtype=out_dtype,
    )
    return merged_out, merged_lse
