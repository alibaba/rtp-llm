"""Flash-decoding attention for the Gemma4 FULL layers (head_dim 512).

The graph-mode FULL-layer decode previously ran a page-chunked SDPA loop
(~45 host ops per chunk, ~500 chunks per 64K request, scores recomputed a
second time for the LSE merge). This module replaces the whole loop with
the standard flash-decoding pair:

- ``_full_decode_partial``: grid (batch * q_heads, splits); each program
  strides its split of the request's pages, computes the causal scores of
  the q rows against every page tile with online fp32 softmax, and
  accumulates the unnormalized PV output for both 256-wide V halves.
- ``_full_decode_combine``: grid (batch * q_heads,); merges the per-split
  (m, l, acc) partials with the standard max-shifted LSE formula.

All metadata (page table, indptr, last-page-len, q positions) is read on
device, so a fixed-grid launch is CUDA-graph capturable; per-replay KV
lengths flow through the device buffers. The score math is fp32
accumulate with scale 1.0 (Gemma4 has no 1/sqrt(d)); P is rounded to
bf16 before the PV dot, matching the eager path's numeric class.
"""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.gemma4.decode_padding import pad_decode_queries


@triton.jit
def _full_decode_partial(
    Q,
    KC,
    VC,
    PAGES,
    INDPTR,
    LPL,
    QPOS,
    M_ACC,
    L_ACC,
    O_ACC,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kp,
    stride_kh,
    stride_ks,
    stride_os,
    stride_om,
    H: tl.constexpr,
    GROUP: tl.constexpr,
    DH: tl.constexpr,
    PS: tl.constexpr,
    CACHE_PAGES: tl.constexpr,
    M: tl.constexpr,
    SPLITS: tl.constexpr,
    WINDOW: tl.constexpr = 0,
    WINPAGES: tl.constexpr = 0,
):
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    kvh = h // GROUP

    row_start = tl.load(INDPTR + b).to(tl.int32)
    row_end = tl.load(INDPTR + b + 1).to(tl.int32)
    row_pages = row_end - row_start
    kvlen = tl.maximum(row_pages - 1, 0) * PS + tl.load(LPL + b)
    kvlen = tl.maximum(kvlen, 0)
    # WINDOW > 0 (swa layers): only each request's trailing win_pages pages
    # can hold any query's window, so skip the earlier pages entirely; the
    # constexpr default keeps the full-attention binary unchanged.
    win_start = row_start
    if WINDOW > 0:
        win_start = tl.maximum(row_end - WINPAGES, row_start)

    offs_m = tl.arange(0, M)
    offs_d = tl.arange(0, DH)
    q_ptrs = (
        Q
        + b.to(tl.int64) * stride_qb
        + h.to(tl.int64) * stride_qh
        + offs_m[:, None].to(tl.int64) * stride_qm
        + offs_d[None, :].to(tl.int64)
    )
    q_a = tl.load(q_ptrs)
    q_b = tl.load(q_ptrs + DH)
    qpos = tl.load(QPOS + b * M + offs_m)
    # draft-model queries sit at or past the cache end; clamping their
    # effective position to the last valid key anchors the causal and window
    # bounds at the cache end (the eager path's visibility semantics) and is
    # a no-op for target decode, whose queries never pass the last KV entry.
    qpos = tl.minimum(qpos, kvlen - 1)

    m_i = tl.full([M], float("-inf"), tl.float32)
    l_i = tl.zeros([M], tl.float32)
    acc_a = tl.zeros([M, DH], tl.float32)
    acc_b = tl.zeros([M, DH], tl.float32)

    SBT: tl.constexpr = 64  # page subtile width (smem budget)
    for p_idx in range(pid_s + win_start, row_end, SPLITS):
        page = tl.load(PAGES + p_idx)
        page_ok = (page >= 0) & (page < CACHE_PAGES)
        page = tl.where(page_ok, page, 0)
        base = page.to(tl.int64) * stride_kp + kvh.to(tl.int64) * stride_kh
        for sub in tl.static_range(PS // SBT):
            offs_s = sub * SBT + tl.arange(0, SBT)
            kt_ptrs = (
                base
                + offs_s[:, None].to(tl.int64) * stride_ks
                + offs_d[None, :].to(tl.int64)
            )
            key_pos = (p_idx - row_start) * PS + offs_s
            key_valid = page_ok & (key_pos < kvlen)
            kt_a = tl.load(KC + kt_ptrs, mask=key_valid[:, None], other=0.0)
            kt_b = tl.load(KC + kt_ptrs + DH, mask=key_valid[:, None], other=0.0)

            s = tl.dot(q_a, tl.trans(kt_a))
            s += tl.dot(q_b, tl.trans(kt_b))

            # key position is the offset inside THIS request's page run (row
            # indices are global across the batch; using p_idx directly would
            # shift every non-first request's positions by row_start*PS).
            allowed = key_valid[None, :] & (key_pos[None, :] <= qpos[:, None])
            if WINDOW > 0:
                allowed = allowed & (key_pos[None, :] > qpos[:, None] - WINDOW)
            s = tl.where(allowed, s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, 1))
            # an all-masked subtile leaves m_new at -inf; subtracting -inf from
            # -inf would produce NaN in alpha/p, so clamp to a finite sentinel
            # (l_i and the accumulators are still zero there, so any finite
            # rescale is a no-op).
            m_new = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            p_bf = p.to(tl.bfloat16)
            # mask the V load to the valid keys: the tail of the last page (past
            # lpl) is unwritten cache memory, and p is exactly 0 there but
            # 0 * NaN = NaN would poison the tensor-core PV dot (the eager
            # gather path sanitizes with nan_to_num for the same reason).
            v_mask = key_valid[:, None]
            vt_a = tl.load(VC + kt_ptrs, mask=v_mask, other=0.0)
            vt_b = tl.load(VC + kt_ptrs + DH, mask=v_mask, other=0.0)
            acc_a = acc_a * alpha[:, None] + tl.dot(p_bf, vt_a)
            acc_b = acc_b * alpha[:, None] + tl.dot(p_bf, vt_b)
            m_i = m_new

    m_base = (pid_bh.to(tl.int64) * SPLITS + pid_s).to(tl.int64)
    tl.store(M_ACC + m_base * M + offs_m, m_i)
    tl.store(L_ACC + m_base * M + offs_m, l_i)
    o_ptrs = (
        O_ACC
        + m_base * stride_os
        + offs_m[:, None].to(tl.int64) * stride_om
        + offs_d[None, :].to(tl.int64)
    )
    tl.store(o_ptrs, acc_a)
    tl.store(o_ptrs + DH, acc_b)


@triton.jit
def _full_decode_combine(
    M_ACC,
    L_ACC,
    O_ACC,
    OUT,
    stride_os,
    stride_om,
    stride_ob,
    stride_oh,
    stride_om_out,
    H: tl.constexpr,
    DH: tl.constexpr,
    M: tl.constexpr,
    SPLITS: tl.constexpr,
    TWOHALF: tl.constexpr = True,
    Q_LEN: tl.constexpr = None,
):
    pid_bh = tl.program_id(0)
    base = pid_bh.to(tl.int64) * SPLITS

    offs_m = tl.arange(0, M)
    m_star = tl.full([M], float("-inf"), tl.float32)
    for s in range(SPLITS):
        m_s = tl.load(M_ACC + (base + s).to(tl.int64) * M + offs_m)
        m_star = tl.maximum(m_star, m_s)
    # rows whose every split stayed all-masked have m_star = -inf; clamp to a
    # finite sentinel so exp(m_s - m_star) never computes -inf - -inf = NaN
    # (their l contributions are exp(-inf) = 0, so the merge stays exact).
    m_star = tl.where(m_star == float("-inf"), 0.0, m_star)

    l_star = tl.zeros([M], tl.float32)
    for s in range(SPLITS):
        m_s = tl.load(M_ACC + (base + s).to(tl.int64) * M + offs_m)
        l_s = tl.load(L_ACC + (base + s).to(tl.int64) * M + offs_m)
        l_star += tl.exp(m_s - m_star) * l_s

    offs_d = tl.arange(0, DH)
    acc_a = tl.zeros([M, DH], tl.float32)
    # TWOHALF: the D=512 callers split the head dim into two DH=256 halves
    # stored contiguously; single-half (D=128) callers must NOT touch the
    # +DH lane or they read/write out of bounds.
    acc_b = tl.zeros([M, DH], tl.float32)
    for s in range(SPLITS):
        m_s = tl.load(M_ACC + (base + s).to(tl.int64) * M + offs_m)
        w_s = tl.exp(m_s - m_star)  # [M]
        o_ptrs = (
            O_ACC
            + (base + s).to(tl.int64) * stride_os
            + offs_m[:, None].to(tl.int64) * stride_om
            + offs_d[None, :].to(tl.int64)
        )
        acc_a += tl.load(o_ptrs) * w_s[:, None]
        if TWOHALF:
            acc_b += tl.load(o_ptrs + DH) * w_s[:, None]

    safe_l = tl.where(l_star > 0.0, l_star, 1.0)
    out_a = acc_a / safe_l[:, None]
    zero_rows = l_star <= 0.0
    out_a = tl.where(zero_rows[:, None], 0.0, out_a)

    b = pid_bh // H
    h = pid_bh % H
    out_ptrs = (
        OUT
        + b.to(tl.int64) * stride_ob
        + h.to(tl.int64) * stride_oh
        + offs_m[:, None].to(tl.int64) * stride_om_out
        + offs_d[None, :].to(tl.int64)
    )
    # Legacy callers store every padded row; compact callers expose only Q_LEN.
    live_rows: tl.constexpr = M if Q_LEN is None else Q_LEN
    tl.static_assert(live_rows > 0, "Q_LEN must be positive")
    tl.static_assert(live_rows <= M, "Q_LEN must not exceed the padded M extent")
    store_mask = (offs_m < live_rows)[:, None]
    tl.store(out_ptrs, out_a.to(tl.bfloat16), mask=store_mask)
    if TWOHALF:
        out_b = acc_b / safe_l[:, None]
        out_b = tl.where(zero_rows[:, None], 0.0, out_b)
        tl.store(out_ptrs + DH, out_b.to(tl.bfloat16), mask=store_mask)


_KERNEL_VERSION = "v8-token-major-output"


def _next_pow2(value: int) -> int:
    result = 1
    while result < value:
        result *= 2
    return result


def _decode_layout_supported(
    q,
    k_cache,
    v_cache,
    pages,
    indptr,
    last_page_len,
    q_pos,
    head_num,
    kv_head_num,
    page_size,
    head_dim,
) -> bool:
    """Check launch geometry without reading metadata values or synchronizing."""
    if q.ndim != 4 or k_cache.ndim != 4 or kv_head_num <= 0:
        return False
    batch, heads, q_len, dim = q.shape
    # The fused caller reshapes packed BQHD storage and transposes H/Q.
    # Admit that exact view; arbitrary strided or broadcast Q still falls back.
    token_major_view = q.stride() == (q_len * heads * dim, dim, heads * dim, 1) and all(
        stride > 0 for stride in q.stride()
    )
    return (
        q.is_cuda
        and q.dtype == torch.bfloat16
        and (q.is_contiguous() or token_major_view)
        and batch > 0
        and q_len > 0
        and heads == head_num
        and head_num % kv_head_num == 0
        and dim == head_dim
        and page_size >= 64
        and page_size % 64 == 0
        and k_cache.shape[0] > 0
        and k_cache.shape[1:] == (kv_head_num, page_size, head_dim)
        and k_cache.dtype == v_cache.dtype == torch.bfloat16
        and k_cache.shape == v_cache.shape
        and k_cache.stride() == v_cache.stride()
        and k_cache.stride(-1) == 1
        and all(
            t.device == q.device
            for t in (k_cache, v_cache, pages, indptr, last_page_len, q_pos)
        )
        and all(
            t.ndim == 1 and t.is_contiguous() and t.dtype == torch.int32
            for t in (pages, indptr, last_page_len)
        )
        and indptr.numel() >= batch + 1
        and last_page_len.numel() >= batch
        and q_pos.shape == (batch, q_len)
        and q_pos.dtype in (torch.int32, torch.int64)
    )


def full_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    pages: torch.Tensor,
    indptr: torch.Tensor,
    last_page_len: torch.Tensor,
    q_pos: torch.Tensor,
    head_num: int,
    kv_head_num: int,
    page_size: int,
) -> torch.Tensor:
    """q: [B, H, M, 512] bf16; caches: [pages, KVH, PS, 512].

    Q may be contiguous BHMD or the canonical BHMD view of contiguous BMHD
    storage. Its strides flow directly into query padding and the partial kernel,
    so the fused caller needs no native contiguous-query copy.

    Returns a [B, H, q_len, 512] bf16 view of contiguous [B, q_len, H, 512]
    storage, or None when unsupported (the caller falls back). The consumer's
    transpose/reshape to packed NHD shares that storage. q_pos: [B, M] int.
    All metadata stays on
    device, so a fixed-grid launch replays correctly under CUDA graphs.

    The kernel tiles each page in SBT=64 sub-blocks, so only page sizes that
    are positive multiples of 64 are eligible; smaller or unaligned sizes
    would silently drop keys and must take the fallback path instead.
    """
    if not _decode_layout_supported(
        q,
        k_cache,
        v_cache,
        pages,
        indptr,
        last_page_len,
        q_pos,
        head_num,
        kv_head_num,
        page_size,
        512,
    ):
        return None
    batch, _, q_len, head_dim = q.shape
    group = head_num // kv_head_num
    # tl.dot requires M >= 16; decode q_len is 1..8 typically, so pad.
    m_pow2 = max(16, _next_pow2(q_len))
    # splits trade parallelism for merge work; o_acc is fp32 [B, H, splits, M,
    # D], so large decode buckets prefer fewer splits to bound the graph-pool
    # transient. Small batches are latency-bound, so 32 splits keep ~500
    # programs in flight to hide the paged-load latency.
    splits = 32 if batch <= 2 else (16 if batch <= 4 else (8 if batch <= 16 else 4))

    m_acc = torch.empty(
        (batch, head_num, splits, m_pow2), dtype=torch.float32, device=q.device
    )
    l_acc = torch.empty(
        (batch, head_num, splits, m_pow2), dtype=torch.float32, device=q.device
    )
    o_acc = torch.empty(
        (batch, head_num, splits, m_pow2, head_dim),
        dtype=torch.float32,
        device=q.device,
    )

    q_full, qpos_full = pad_decode_queries(q, q_pos, m_pow2)

    grid = (batch * head_num, splits)
    _full_decode_partial[grid](
        q_full,
        k_cache,
        v_cache,
        pages,
        indptr,
        last_page_len,
        qpos_full,
        m_acc,
        l_acc,
        o_acc,
        q_full.stride(0),
        q_full.stride(1),
        q_full.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        o_acc.stride(2),
        o_acc.stride(3),
        H=head_num,
        GROUP=group,
        DH=head_dim // 2,
        PS=page_size,
        CACHE_PAGES=k_cache.shape[0],
        M=m_pow2,
        SPLITS=splits,
        num_warps=4,
        num_stages=1,
    )
    output = torch.empty(
        (batch, q_len, head_num, head_dim), dtype=torch.bfloat16, device=q.device
    )
    _full_decode_combine[(batch * head_num,)](
        m_acc,
        l_acc,
        o_acc,
        output,
        o_acc.stride(2),
        o_acc.stride(3),
        output.stride(0),
        output.stride(2),
        output.stride(1),
        H=head_num,
        DH=head_dim // 2,
        M=m_pow2,
        SPLITS=splits,
        TWOHALF=True,
        Q_LEN=q_len,
        num_warps=4,
    )
    return output.permute(0, 2, 1, 3)
