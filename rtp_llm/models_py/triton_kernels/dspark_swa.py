"""Standalone paged BF16 GQA window attention; no DSpark mask is assumed."""

import torch
import triton
import triton.language as tl


@triton.jit
def _commit_paged_kv(
    K,
    V,
    CACHE,
    SLOTS,
    VALID,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCKS: tl.constexpr,
    STRIDE: tl.constexpr,
    K_ROW: tl.constexpr,
    V_ROW: tl.constexpr,
    TILE: tl.constexpr,
):
    token = tl.program_id(0)
    slot = tl.load(SLOTS + token).to(tl.int64)
    valid = tl.load(VALID + token) & (slot >= 0) & (slot < BLOCKS * PAGE)
    if valid:
        x = tl.arange(0, TILE)
        mask = x < H * D
        destination = slot // PAGE * STRIDE + (x // D * PAGE + slot % PAGE) * D + x % D
        key = tl.load(K + token * K_ROW + x, mask, 0)
        value = tl.load(V + token * V_ROW + x, mask, 0)
        tl.store(CACHE + destination, key, mask)
        tl.store(CACHE + destination + H * PAGE * D, value, mask)


def commit_paged_gqa_kv(k, v, cache, slot_ids, valid_mask):
    """Copy already-normalized/RoPE K and V into native BF16 paged cache.

    k/v are [T,H,D], contiguous within a row; their row strides may differ.
    This accepts views of interleaved projected/all-gathered K/V without copies.
    slot_ids[T] is the explicit *physical* token
    slot (physical_block * page_size + offset), not a logical position.
    valid_mask[T] is bool; false, negative and out-of-range slots never write.
    Cache uses [blocks,2,H,page,D], permitting padded physical-block stride.
    Metadata and input values may be updated in place during graph replay.
    Concurrent valid writes must have unique physical slots. Inputs must not
    alias destination cache. Callers own slot lifetime, acceptance, logical
    lengths and page mapping; rejection can overwrite the same slot on a later
    ordered call. No normalization, RoPE, quantization or allocation occurs.
    """
    if k.ndim != 3 or v.shape != k.shape or cache.ndim != 5:
        raise ValueError("expected k/v [T,H,D] and cache [blocks,2,H,page,D]")
    tokens, heads, dim = k.shape
    if cache.shape[1:3] != (2, heads) or cache.shape[-1] != dim:
        raise ValueError("cache head dimensions must match k/v")
    page = cache.shape[3]
    if heads <= 0 or dim <= 0 or page <= 0:
        raise ValueError("head count, head dimension and page size must be positive")
    if (
        cache.stride()[1:] != (heads * page * dim, page * dim, dim, 1)
        or cache.stride(0) < 2 * heads * page * dim
    ):
        raise ValueError("cache must be contiguous within each physical block")
    tensors = (k, v, cache, slot_ids, valid_mask)
    if any(not t.is_cuda or t.device != k.device for t in tensors):
        raise ValueError("all tensors must be CUDA tensors on the same device")
    if any(t.dtype != torch.bfloat16 for t in (k, v, cache)):
        raise TypeError("commit supports BF16 k/v/cache")
    if slot_ids.shape != (tokens,) or valid_mask.shape != (tokens,):
        raise ValueError("slot_ids and valid_mask must have shape [T]")
    if (
        slot_ids.dtype not in (torch.int32, torch.int64)
        or valid_mask.dtype != torch.bool
    ):
        raise TypeError("slot_ids must be int32/int64 and valid_mask must be bool")
    if any(t.stride()[1:] != (dim, 1) or t.stride(0) < heads * dim for t in (k, v)):
        raise ValueError("k/v must be contiguous within non-overlapping rows")
    if any(not t.is_contiguous() for t in (slot_ids, valid_mask)):
        raise ValueError("slot_ids/valid_mask must be contiguous")
    if tokens:
        _commit_paged_kv[(tokens,)](
            k,
            v,
            cache,
            slot_ids,
            valid_mask,
            heads,
            dim,
            page,
            cache.shape[0],
            cache.stride(0),
            k.stride(0),
            v.stride(0),
            triton.next_power_of_2(heads * dim),
        )


@triton.jit
def _paged_swa(
    Q,
    K,
    V,
    CACHE,
    TABLE,
    LENS,
    QLENS,
    OUT,
    WIDTH: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCKS: tl.constexpr,
    PHYSICAL: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    LEFT: tl.constexpr,
    CAUSAL: tl.constexpr,
    SCALE: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    b, h = tl.program_id(0), tl.program_id(1)
    group: tl.constexpr = HQ // HK
    rows = tl.program_id(2) * M + tl.arange(0, M)
    qr, qh = rows // group, h * group + rows % group
    d = tl.arange(0, D)
    length = tl.load(LENS + b)
    live = tl.load(QLENS + b)
    row_valid = (qr < WIDTH) & (qr < live)
    q = tl.load(
        Q + ((b * WIDTH + qr[:, None]) * HQ + qh[:, None]) * D + d[None, :],
        row_valid[:, None],
        0,
    )
    start = tl.maximum(length - LEFT, 0)
    end = length + live
    maximum = tl.full((M,), -float("inf"), tl.float32)
    denominator = tl.zeros((M,), tl.float32)
    acc = tl.zeros((M, D), tl.float32)
    for offset in range(start, end, N):
        pos = offset + tl.arange(0, N)
        historical = (pos < length) & (pos // PAGE < BLOCKS)
        block = tl.load(TABLE + b * BLOCKS + pos // PAGE, historical, 0)
        historical = historical & (block >= 0) & (block < PHYSICAL)
        cache_off = (
            block[:, None].to(tl.int64) * CACHE_STRIDE
            + (h * PAGE + pos[:, None] % PAGE) * D
            + d[None, :]
        )
        past_k = tl.load(CACHE + cache_off, historical[:, None], 0)
        past_v = tl.load(CACHE + cache_off + HK * PAGE * D, historical[:, None], 0)
        current = (pos >= length) & (pos < end) & (pos - length < WIDTH)
        local_off = ((b * WIDTH + pos[:, None] - length) * HK + h) * D + d[None, :]
        new_k = tl.load(K + local_off, current[:, None], 0)
        new_v = tl.load(V + local_off, current[:, None], 0)
        k = tl.where(historical[:, None], past_k, new_k)
        v = tl.where(historical[:, None], past_v, new_v)
        allowed = row_valid[:, None] & (historical | current)[None, :]
        allowed &= pos[None, :] >= length + qr[:, None] - LEFT
        if CAUSAL:
            allowed &= pos[None, :] <= length + qr[:, None]
        scores = tl.dot(q, tl.trans(k)) * SCALE
        scores = tl.where(allowed, scores, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(scores, 1))
        safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
        correction = tl.exp(maximum - safe_max)
        p = tl.exp(scores - safe_max[:, None])
        acc = acc * correction[:, None] + tl.dot(p.to(v.dtype), v)
        denominator = denominator * correction + tl.sum(p, 1)
        maximum = new_max
    result = acc / tl.where(denominator > 0, denominator, 1.0)[:, None]
    tl.store(
        OUT + ((b * WIDTH + qr[:, None]) * HQ + qh[:, None]) * D + d[None, :],
        result,
        (qr < WIDTH)[:, None],
    )


def _swa_launch_config(batch, width, hq, hk, page, window_left, causal, sm_major):
    rows = width * hq // hk
    tile_rows = min(32, max(16, triton.next_power_of_2(rows)))
    # Measured SM10 B16 proposal shape: reuse each history tile for more query
    # rows. Other shapes retain the lower shared-memory launch configuration.
    if (
        batch == 16
        and width == 7
        and hq == 64
        and hk == 4
        and page == 128
        and window_left == 4095
        and causal
        and sm_major == 10
    ):
        return 64, 128, 8
    return tile_rows, 64, 4


def paged_gqa_swa(
    q,
    k,
    v,
    cache,
    block_table,
    context_lens,
    query_lens,
    *,
    window_left: int,
    causal: bool,
    out=None
):
    """Attend cached history plus separate current K/V without history gathering.

    q: [B, W, Hq, 128]; k/v: [B, W, Hkv, 128], all contiguous BF16.
    cache: native [physical_blocks, 2, Hkv, page_size, 128], contiguous within
    each physical block; a padded block stride is supported without copying.
    context_lens[B] counts historical tokens, query_lens[B] counts live current
    rows (zero for padded requests). Int32/64 CUDA metadata updates in place
    are replay safe. Callers own valid lengths and mappings for historical KV.
    Position p sees keys >= p-window_left, and <= p if causal, otherwise all
    live current keys. Thus window_left=4095 means 4096 keys including self.
    Invalid query rows are zero. This function never writes the cache.
    Provide out for a graph-stable allocation-free launch. Mask choice must
    come from the model reference; this primitive does not choose DSpark math.
    """
    if q.ndim != 4 or k.ndim != 4 or cache.ndim != 5:
        raise ValueError("expected rank-4 q/k/v and rank-5 native KV cache")
    b, w, hq, d = q.shape
    hk = k.shape[2]
    if (
        d != 128
        or hq % hk
        or k.shape != (b, w, hk, d)
        or v.shape != k.shape
        or cache.shape[1:3] != (2, hk)
        or cache.shape[-1] != d
        or window_left < 0
    ):
        raise ValueError("incompatible GQA shapes or negative window_left")
    if not isinstance(causal, bool):
        raise TypeError("causal must explicitly be bool")
    if block_table.ndim != 2 or block_table.shape[0] != b:
        raise ValueError("block_table must be [B, max_blocks]")
    if context_lens.shape != (b,) or query_lens.shape != (b,):
        raise ValueError("length tensors must have shape [B]")
    tensors = (q, k, v, cache, block_table, context_lens, query_lens)
    if any(t.device != q.device or not t.is_cuda for t in tensors):
        raise ValueError("inputs must be CUDA tensors on one device")
    if any(
        not t.is_contiguous() for t in (q, k, v, block_table, context_lens, query_lens)
    ):
        raise ValueError("query and metadata tensors must be contiguous")
    page = cache.shape[3]
    if (
        cache.stride()[1:] != (hk * page * d, page * d, d, 1)
        or cache.stride(0) < 2 * hk * page * d
    ):
        raise ValueError("cache must be contiguous within each physical block")
    if any(t.dtype != torch.bfloat16 for t in (q, k, v, cache)):
        raise TypeError("initial primitive supports BF16 Q/K/V/cache only")
    if any(t.dtype not in (torch.int32, torch.int64) for t in tensors[4:]):
        raise TypeError("metadata must be int32 or int64")
    if out is None:
        out = torch.empty_like(q)
    if (
        out.shape != q.shape
        or out.dtype != q.dtype
        or out.device != q.device
        or not out.is_contiguous()
    ):
        raise ValueError("out must match q shape, dtype, device and contiguous layout")
    # Split independent query/head rows: a full W*GQA tile underfills the GPU
    # and consumes excessive registers at proposal widths such as seven.
    # Each tile retains the same history traversal and online softmax.
    rows = w * hq // hk
    tile_rows, tile_columns, num_warps = _swa_launch_config(
        b,
        w,
        hq,
        hk,
        page,
        window_left,
        causal,
        torch.cuda.get_device_capability(q.device)[0],
    )
    _paged_swa[(b, hk, triton.cdiv(rows, tile_rows))](
        q,
        k,
        v,
        cache,
        block_table,
        context_lens,
        query_lens,
        out,
        w,
        hq,
        hk,
        d,
        cache.shape[3],
        block_table.shape[1],
        cache.shape[0],
        cache.stride(0),
        window_left,
        causal,
        d**-0.5,
        tile_rows,
        tile_columns,
        num_warps=num_warps,
        num_stages=2,
    )
    return out
