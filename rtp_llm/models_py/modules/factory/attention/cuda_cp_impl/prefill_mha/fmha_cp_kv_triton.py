"""Triton CP fmha prefix-cache K/V preparation.

``triton_build_dual_prefix_extend`` reads the shared paged prefix once and
scatters it directly into both CP-part ``[prefix || extend]`` outputs (the two
parts differ only in their extend tail), folding each part's extend into its
tail. This fuses what used to be a paged-extract + two ragged-concat passes,
moving the dominant prefix K/V 3x instead of 6x (every stage is HBM-bound).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _is_uniform_lens(lens: torch.Tensor) -> bool:
    if lens.numel() <= 1:
        return True
    return bool(torch.all(lens == lens[0]).item())


def _build_extract_page_map(
    prefix_lengths: torch.Tensor,
    kv_indices: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, int, int] | None:
    device = prefix_lengths.device
    batch_count = prefix_lengths.numel()
    num_pages_per_batch = prefix_lengths // page_size
    total_pages = int(num_pages_per_batch.sum().item())
    if total_pages == 0:
        return None

    if _is_uniform_lens(num_pages_per_batch):
        pages_per_seq = int(num_pages_per_batch[0].item())
        page_global = torch.arange(total_pages, device=device, dtype=torch.int64)
        batch_idx = page_global // pages_per_seq
        page_local = page_global % pages_per_seq
        block_ids = kv_indices[batch_idx, page_local].contiguous()
    else:
        batch_indices = torch.repeat_interleave(
            torch.arange(batch_count, device=device, dtype=torch.int64),
            num_pages_per_batch.to(torch.int64),
        )
        pages_cumsum = num_pages_per_batch.cumsum(0)
        page_global = torch.arange(total_pages, device=device, dtype=torch.int64)
        page_local_idx = page_global - torch.repeat_interleave(
            pages_cumsum - num_pages_per_batch, num_pages_per_batch.to(torch.int64)
        )
        block_ids = kv_indices[batch_indices, page_local_idx].contiguous()

    total_tokens = total_pages * page_size
    return block_ids, total_pages, total_tokens


def _page_size_log2(page_size: int) -> int:
    log2 = 0
    size = page_size
    while size > 1:
        if size & 1:
            raise ValueError(f"page_size must be power of 2, got {page_size}")
        size >>= 1
        log2 += 1
    return log2


def _pick_block_hd(hd: int) -> tuple[int, int]:
    block_hd = triton.next_power_of_2(hd)
    if block_hd > 16384:
        block_hd = 4096
    num_warps = 8 if block_hd >= 8192 else 4
    return block_hd, num_warps


# ---------------------------------------------------------------------------
# Fused build of BOTH CP-part outputs directly from the paged pool.
#
# The prefix cache is shared by part0 and part1 (they differ only in the
# ``extend`` tail), so the un-fused path (extract -> concat part0 -> concat
# part1) moves the prefix K/V ~6x: read paged + write flat, then read+write
# per concat, twice. Since every stage is HBM-bandwidth bound, that dominates.
#
# ``triton_build_dual_prefix_extend`` reads each paged prefix token ONCE and
# scatters it straight into both ``[prefix||extend]`` outputs (prefix traffic
# 6x -> 3x), then folds each part's extend into its own tail. No intermediate
# prefix buffer is materialised.
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["num_prefix_tokens"])
def _prefix_dual_scatter_uniform_kernel(
    cache_k_ptr,
    cache_v_ptr,
    block_ids_ptr,
    out0_k_ptr,
    out0_v_ptr,
    out1_k_ptr,
    out1_v_ptr,
    num_prefix_tokens,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_SIZE_LOG2: tl.constexpr,
    cache_stride_b,
    cache_stride_h,
    cache_stride_t,
    HD: tl.constexpr,
    BLOCK_HD: tl.constexpr,
    P: tl.constexpr,
    SEG0: tl.constexpr,
    SEG1: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    if token >= num_prefix_tokens:
        return
    page_id = token >> PAGE_SIZE_LOG2
    page_t = token & (PAGE_SIZE - 1)
    block_id = tl.load(block_ids_ptr + page_id).to(tl.int64)
    batch_id = token // P
    local = token % P
    dst0 = batch_id * SEG0 + local
    dst1 = batch_id * SEG1 + local

    pid_blk = tl.program_id(1)
    offs = pid_blk * BLOCK_HD + tl.arange(0, BLOCK_HD)
    h = offs // D
    d = offs % D
    mask = (h < H) & (d < D)
    src = block_id * cache_stride_b + h * cache_stride_h + page_t * cache_stride_t + d
    vk = tl.load(cache_k_ptr + src, mask=mask, other=0.0)
    vv = tl.load(cache_v_ptr + src, mask=mask, other=0.0)
    tl.store(out0_k_ptr + dst0 * HD + offs, vk, mask=mask)
    tl.store(out0_v_ptr + dst0 * HD + offs, vv, mask=mask)
    tl.store(out1_k_ptr + dst1 * HD + offs, vk, mask=mask)
    tl.store(out1_v_ptr + dst1 * HD + offs, vv, mask=mask)


@triton.jit(do_not_specialize=["num_prefix_tokens"])
def _prefix_dual_scatter_ragged_kernel(
    cache_k_ptr,
    cache_v_ptr,
    block_ids_ptr,
    dst0_row_ptr,
    dst1_row_ptr,
    out0_k_ptr,
    out0_v_ptr,
    out1_k_ptr,
    out1_v_ptr,
    num_prefix_tokens,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_SIZE_LOG2: tl.constexpr,
    cache_stride_b,
    cache_stride_h,
    cache_stride_t,
    HD: tl.constexpr,
    BLOCK_HD: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    if token >= num_prefix_tokens:
        return
    page_id = token >> PAGE_SIZE_LOG2
    page_t = token & (PAGE_SIZE - 1)
    block_id = tl.load(block_ids_ptr + page_id).to(tl.int64)
    dst0 = tl.load(dst0_row_ptr + token).to(tl.int64)
    dst1 = tl.load(dst1_row_ptr + token).to(tl.int64)

    pid_blk = tl.program_id(1)
    offs = pid_blk * BLOCK_HD + tl.arange(0, BLOCK_HD)
    h = offs // D
    d = offs % D
    mask = (h < H) & (d < D)
    src = block_id * cache_stride_b + h * cache_stride_h + page_t * cache_stride_t + d
    vk = tl.load(cache_k_ptr + src, mask=mask, other=0.0)
    vv = tl.load(cache_v_ptr + src, mask=mask, other=0.0)
    tl.store(out0_k_ptr + dst0 * HD + offs, vk, mask=mask)
    tl.store(out0_v_ptr + dst0 * HD + offs, vv, mask=mask)
    tl.store(out1_k_ptr + dst1 * HD + offs, vk, mask=mask)
    tl.store(out1_v_ptr + dst1 * HD + offs, vv, mask=mask)


@triton.jit(do_not_specialize=["num_extend_tokens"])
def _extend_scatter_uniform_kernel(
    extend_k_ptr,
    extend_v_ptr,
    out_k_ptr,
    out_v_ptr,
    num_extend_tokens,
    HD: tl.constexpr,
    BLOCK_HD: tl.constexpr,
    P: tl.constexpr,
    E: tl.constexpr,
    SEG: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    if token >= num_extend_tokens:
        return
    batch_id = token // E
    j = token % E
    dst = batch_id * SEG + P + j
    pid_blk = tl.program_id(1)
    offs = pid_blk * BLOCK_HD + tl.arange(0, BLOCK_HD)
    mask = offs < HD
    tl.store(out_k_ptr + dst * HD + offs, tl.load(extend_k_ptr + token * HD + offs, mask=mask), mask=mask)
    tl.store(out_v_ptr + dst * HD + offs, tl.load(extend_v_ptr + token * HD + offs, mask=mask), mask=mask)


@triton.jit(do_not_specialize=["num_extend_tokens"])
def _extend_scatter_ragged_kernel(
    extend_k_ptr,
    extend_v_ptr,
    dst_row_ptr,
    out_k_ptr,
    out_v_ptr,
    num_extend_tokens,
    HD: tl.constexpr,
    BLOCK_HD: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    if token >= num_extend_tokens:
        return
    dst = tl.load(dst_row_ptr + token).to(tl.int64)
    pid_blk = tl.program_id(1)
    offs = pid_blk * BLOCK_HD + tl.arange(0, BLOCK_HD)
    mask = offs < HD
    tl.store(out_k_ptr + dst * HD + offs, tl.load(extend_k_ptr + token * HD + offs, mask=mask), mask=mask)
    tl.store(out_v_ptr + dst * HD + offs, tl.load(extend_v_ptr + token * HD + offs, mask=mask), mask=mask)


def _build_prefix_dst_rows(prefix_lens, seg0, seg1, device):
    """Per-prefix-token destination rows in out0 / out1 (ragged)."""
    pl = prefix_lens.to(torch.int64)
    out0_starts = seg0.to(torch.int64).cumsum(0) - seg0.to(torch.int64)
    out1_starts = seg1.to(torch.int64).cumsum(0) - seg1.to(torch.int64)
    prefix_starts = pl.cumsum(0) - pl
    total_prefix = int(pl.sum().item())
    gidx = torch.arange(total_prefix, device=device, dtype=torch.int64)
    batch_ids = torch.repeat_interleave(
        torch.arange(pl.numel(), device=device, dtype=torch.int64), pl
    )
    local = gidx - torch.repeat_interleave(prefix_starts, pl)
    dst0 = (out0_starts[batch_ids] + local).to(torch.int32)
    dst1 = (out1_starts[batch_ids] + local).to(torch.int32)
    return dst0.contiguous(), dst1.contiguous()


def _build_extend_dst_rows(prefix_lens, extend_lens, seg_lens, device):
    """Per-extend-token destination row in the [prefix||extend] output (ragged)."""
    el = extend_lens.to(torch.int64)
    seg_starts = seg_lens.to(torch.int64).cumsum(0) - seg_lens.to(torch.int64)
    extend_starts = el.cumsum(0) - el
    total_ext = int(el.sum().item())
    gidx = torch.arange(total_ext, device=device, dtype=torch.int64)
    batch_ids = torch.repeat_interleave(
        torch.arange(el.numel(), device=device, dtype=torch.int64), el
    )
    j = gidx - torch.repeat_interleave(extend_starts, el)
    dst = (seg_starts[batch_ids] + prefix_lens.to(torch.int64)[batch_ids] + j).to(torch.int32)
    return dst.contiguous()


def _scatter_extend(extend_k, extend_v, out_k, out_v, prefix_lens, extend_lens, seg_lens, hd, block_hd, num_warps, device):
    total_ext = int(extend_lens.to(torch.int64).sum().item())
    if total_ext == 0:
        return
    n_hd = triton.cdiv(hd, block_hd)
    ek = extend_k.contiguous().view(-1)
    ev = extend_v.contiguous().view(-1)
    if _is_uniform_lens(prefix_lens) and _is_uniform_lens(extend_lens):
        P = int(prefix_lens[0].item())
        E = int(extend_lens[0].item())
        SEG = P + E
        _extend_scatter_uniform_kernel[(total_ext, n_hd)](
            ek, ev, out_k.view(-1), out_v.view(-1), total_ext,
            HD=hd, BLOCK_HD=block_hd, P=P, E=E, SEG=SEG, num_warps=num_warps,
        )
    else:
        dst = _build_extend_dst_rows(prefix_lens, extend_lens, seg_lens, device)
        _extend_scatter_ragged_kernel[(total_ext, n_hd)](
            ek, ev, dst, out_k.view(-1), out_v.view(-1), total_ext,
            HD=hd, BLOCK_HD=block_hd, num_warps=num_warps,
        )


def triton_build_dual_prefix_extend(
    kv_cache_tensor: torch.Tensor,
    prefix_lengths: torch.Tensor,
    kv_indices: torch.Tensor,
    page_size: int,
    extend_k0: torch.Tensor,
    extend_v0: torch.Tensor,
    kv_indptr0: torch.Tensor,
    extend_k1: torch.Tensor,
    extend_v1: torch.Tensor,
    kv_indptr1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused: read paged prefix once, build both ``[prefix||extend]`` CP-part
    outputs (part0 uses extend0/kv_indptr0, part1 uses extend1/kv_indptr1).

    Returns ``(k0_cat, v0_cat, k1_cat, v1_cat)``. Semantically each output is
    ``[prefix || extend_part]`` per batch segment; the shared prefix is read
    from the paged pool once and scattered into both, moving the dominant
    prefix K/V 3x instead of the 6x of a separate extract + two concats. See
    the torch reference in ``test_fmha_cp_kv_triton.py`` for the exact layout.
    """
    cache_k = kv_cache_tensor[:, 0]
    cache_v = kv_cache_tensor[:, 1]
    h, d = cache_k.shape[1], cache_k.shape[-1]
    hd = h * d
    device = prefix_lengths.device

    prefix_lens = prefix_lengths.to(torch.int32)
    e0 = (kv_indptr0[1:] - kv_indptr0[:-1]).to(torch.int32)
    e1 = (kv_indptr1[1:] - kv_indptr1[:-1]).to(torch.int32)
    seg0 = prefix_lens + e0
    seg1 = prefix_lens + e1
    total0 = int(seg0.to(torch.int64).sum().item())
    total1 = int(seg1.to(torch.int64).sum().item())

    out0_k = cache_k.new_empty(total0, h, d)
    out0_v = cache_v.new_empty(total0, h, d)
    out1_k = cache_k.new_empty(total1, h, d)
    out1_v = cache_v.new_empty(total1, h, d)

    block_hd, num_warps = _pick_block_hd(hd)
    n_hd = triton.cdiv(hd, block_hd)

    meta = _build_extract_page_map(prefix_lengths, kv_indices, page_size)
    if meta is not None:
        block_ids, total_pages, total_prefix_tokens = meta
        sb, sh, st = cache_k.stride(0), cache_k.stride(1), cache_k.stride(2)
        if _is_uniform_lens(prefix_lens) and _is_uniform_lens(e0) and _is_uniform_lens(e1):
            _prefix_dual_scatter_uniform_kernel[(total_prefix_tokens, n_hd)](
                cache_k, cache_v, block_ids,
                out0_k.view(-1), out0_v.view(-1), out1_k.view(-1), out1_v.view(-1),
                total_prefix_tokens, H=h, D=d, PAGE_SIZE=page_size,
                PAGE_SIZE_LOG2=_page_size_log2(page_size),
                cache_stride_b=sb, cache_stride_h=sh, cache_stride_t=st,
                HD=hd, BLOCK_HD=block_hd,
                P=int(prefix_lens[0].item()), SEG0=int(seg0[0].item()), SEG1=int(seg1[0].item()),
                num_warps=num_warps,
            )
        else:
            dst0, dst1 = _build_prefix_dst_rows(prefix_lens, seg0, seg1, device)
            _prefix_dual_scatter_ragged_kernel[(total_prefix_tokens, n_hd)](
                cache_k, cache_v, block_ids, dst0, dst1,
                out0_k.view(-1), out0_v.view(-1), out1_k.view(-1), out1_v.view(-1),
                total_prefix_tokens, H=h, D=d, PAGE_SIZE=page_size,
                PAGE_SIZE_LOG2=_page_size_log2(page_size),
                cache_stride_b=sb, cache_stride_h=sh, cache_stride_t=st,
                HD=hd, BLOCK_HD=block_hd, num_warps=num_warps,
            )

    _scatter_extend(extend_k0, extend_v0, out0_k, out0_v, prefix_lens, e0, seg0, hd, block_hd, num_warps, device)
    _scatter_extend(extend_k1, extend_v1, out1_k, out1_v, prefix_lens, e1, seg1, hd, block_hd, num_warps, device)
    return out0_k, out0_v, out1_k, out1_v
