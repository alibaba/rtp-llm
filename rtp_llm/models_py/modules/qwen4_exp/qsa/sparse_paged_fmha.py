"""Paged sparse GQA attention for Qwen4 decode and context-style prefill.

The QSA indexer returns request-local absolute token indices.  This kernel maps
each selected token through the main MHA cache block table and performs GQA
attention without materialising gathered K/V tensors.  It is intentionally
Qwen4-specific: BF16 or FP8 E4M3 HND cache layout, token-level selections, and block id zero
reserved as the allocator sentinel.

The same B/S-shaped primitive serves ordinary decode, target verification, and
the correctness-first MTP draft incremental-prefill bridge (which invokes it
once per ragged request with ``B=1``).  Model-level routing remains explicitly
gated until indexer side-pool scoring/writes and prefix semantics are connected.
"""

import os

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_cache_views import (
    qsa_assert_valid,
    qsa_strict_validation_enabled,
)

_BLOCK_K = 64


def _sparse_paged_indices_verdict(
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    selected: torch.Tensor,
    *,
    page_size: int,
    cache_blocks: int,
) -> torch.Tensor:
    """0-dim boolean verdict for _validate_sparse_paged_indices (True=valid).

    Computes the same combined predicate without any host synchronization so
    the production entry can consume it with a device-side assert.
    """
    table_width = int(block_table.shape[1])
    invalid_lengths = (kv_lens < 0) | (kv_lens > table_width * page_size)
    invalid_pool = block_table >= cache_blocks
    verdict = ~invalid_lengths.any() & ~invalid_pool.any()
    if selected.numel():
        invalid_negative = selected < -1
        invalid_visible = (selected >= 0) & (selected >= kv_lens.unsqueeze(-1))
        logical_blocks = torch.clamp_min(selected, 0) // page_size
        invalid_logical = logical_blocks >= table_width
        # An invalid selected index must never reach gather before its error
        # is reported. Clamp only the lookup; invalid_logical retains the
        # original value for the verdict and the diagnostic below.
        safe_logical_blocks = logical_blocks.clamp(max=table_width - 1)
        physical = torch.gather(
            block_table.unsqueeze(1).expand(-1, selected.shape[1], -1),
            2,
            safe_logical_blocks.to(torch.long),
        )
        invalid_missing = (selected >= 0) & (physical <= 0)
        invalid_selection = (
            invalid_negative | invalid_visible | invalid_logical | invalid_missing
        )
        verdict = verdict & ~invalid_selection.any()
    return verdict


def _validate_sparse_paged_indices(
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    selected: torch.Tensor,
    *,
    page_size: int,
    cache_blocks: int,
) -> None:
    """Check paged reads with one device-to-host verdict on valid inputs."""
    verdict = _sparse_paged_indices_verdict(
        block_table,
        kv_lens,
        selected,
        page_size=page_size,
        cache_blocks=cache_blocks,
    )
    if bool(verdict.item()):
        return

    # Error paths may synchronize again to preserve the existing specific
    # diagnostics. No main-cache write has happened at this point.
    invalid_lengths = (kv_lens < 0) | (kv_lens > int(block_table.shape[1]) * page_size)
    if bool(invalid_lengths.any().item()):
        raise ValueError("kv_lens exceed the block-table token capacity")
    if selected.numel():
        invalid_negative = selected < -1
        invalid_visible = (selected >= 0) & (selected >= kv_lens.unsqueeze(-1))
        logical_blocks = torch.clamp_min(selected, 0) // page_size
        invalid_logical = logical_blocks >= int(block_table.shape[1])
        safe_logical_blocks = logical_blocks.clamp(max=int(block_table.shape[1]) - 1)
        physical = torch.gather(
            block_table.unsqueeze(1).expand(-1, selected.shape[1], -1),
            2,
            safe_logical_blocks.to(torch.long),
        )
        invalid_missing = (selected >= 0) & (physical <= 0)
        if bool(invalid_negative.any().item()):
            raise ValueError("selected indices may only use -1 as padding")
        if bool(invalid_visible.any().item()):
            raise ValueError("selected contains an index outside its row's kv_len")
        if bool(invalid_logical.any().item()):
            raise ValueError("selected index exceeds the block-table width")
        if bool(invalid_missing.any().item()):
            raise ValueError("selected index resolves to an unallocated cache block")
    max_physical = int(block_table.max().item())
    raise ValueError(
        f"block table physical id {max_physical} exceeds cache blocks {cache_blocks}"
    )


def main_kv_scale_view(cache, scale, *, kv_heads, page_size):
    """Validate the native per-token/head K/V dequant scale layout.

    The fused native writer currently casts to E4M3 and writes unit scales.
    Readers consume the stored scale rather than assuming that value, so PD
    copies and future scaled writers preserve the same cache contract.
    No tensor contents are read back during capture.
    """
    if cache.dtype == torch.bfloat16:
        return None
    if cache.dtype != torch.float8_e4m3fn:
        raise ValueError("QSA main KV must be BF16 or FP8 E4M3")
    pages = int(cache.shape[0])
    if scale is None or scale.dtype != torch.float32 or scale.device != cache.device:
        raise ValueError("FP8 main KV requires device-local FP32 K/V scales")
    shape = (pages, 2, kv_heads, page_size)
    if scale.ndim == 2:
        width = 2 * kv_heads * page_size
        if scale.shape[0] != pages or scale.shape[1] < width or scale.stride(1) != 1:
            raise ValueError("FP8 main KV packed scale geometry is invalid")
        return torch.as_strided(
            scale, shape, (scale.stride(0), kv_heads * page_size, page_size, 1)
        )
    if tuple(scale.shape) != shape or tuple(scale.stride()[1:]) != (
        kv_heads * page_size,
        page_size,
        1,
    ):
        raise ValueError("FP8 main KV scale geometry must be HND per-token/head")
    return scale


@triton.jit
def _sparse_paged_online_split_kernel(
    q_ptr,
    cache_ptr,
    cache_scale_ptr,
    block_table_ptr,
    kv_lens_ptr,
    selected_ptr,
    partial_ptr,
    stride_q_b,
    stride_q_h,
    stride_q_s,
    stride_cache_page,
    stride_cache_kv,
    stride_cache_h,
    stride_cache_token,
    stride_scale_page,
    stride_scale_kv,
    stride_scale_h,
    stride_scale_token,
    stride_bt_b,
    stride_bt_block,
    stride_len_b,
    stride_len_s,
    stride_sel_b,
    stride_sel_s,
    selected_width,
    max_blocks,
    cache_blocks,
    HAS_COUNT: tl.constexpr,
    FP8_CACHE: tl.constexpr,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    # QUERY_LEN/BLOCKS_PER_SPLIT stay runtime args: making them constexpr
    # recompiles the kernel for every new request length.
    QUERY_LEN,
    PAGE_SIZE: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCKS_PER_SPLIT,
):
    partition = tl.program_id(0)
    batch = partition // (QUERY_LEN * NUM_SPLITS)
    head = tl.program_id(1)
    query = (partition // NUM_SPLITS) % QUERY_LEN
    split = partition % NUM_SPLITS
    kv_head = head // (H_Q // H_KV)
    rd = tl.arange(0, D)
    q = tl.load(
        q_ptr + batch * stride_q_b + head * stride_q_h + query * stride_q_s + rd
    ).to(tl.float32)
    row = selected_ptr + batch * stride_sel_b + query * stride_sel_s
    kv_len = tl.load(kv_lens_ptr + batch * stride_len_b + query * stride_len_s)
    maximum = tl.full((), float("-inf"), tl.float32)
    normalizer = tl.zeros((), tl.float32)
    acc = tl.zeros((D,), tl.float32)
    count = (
        tl.minimum(tl.maximum(tl.load(row + selected_width), 0), selected_width)
        if HAS_COUNT
        else selected_width
    )
    limit = tl.maximum(
        0,
        tl.minimum(
            BLOCKS_PER_SPLIT, tl.cdiv(count - split * BLOCK_K, NUM_SPLITS * BLOCK_K)
        ),
    )
    for block in range(limit):
        # Interleave chunks so a short valid prefix is not concentrated in
        # one partition when the selected tensor has a fixed wide capacity.
        offset = (block * NUM_SPLITS + split) * BLOCK_K + tl.arange(0, BLOCK_K)
        index = tl.load(row + offset, mask=offset < count, other=-1)
        logical = index // PAGE_SIZE
        candidate = (offset < count) & (index >= 0) & (index < kv_len)
        physical = tl.load(
            block_table_ptr + batch * stride_bt_b + logical * stride_bt_block,
            mask=candidate & (logical < max_blocks),
            other=0,
        )
        valid = (
            candidate
            & (logical < max_blocks)
            & (physical > 0)
            & (physical < cache_blocks)
        )
        key_ptr = (
            cache_ptr
            + physical[:, None] * stride_cache_page
            + kv_head * stride_cache_h
            + (index % PAGE_SIZE)[:, None] * stride_cache_token
            + rd[None, :]
        )
        key = tl.load(key_ptr, mask=valid[:, None], other=0.0).to(tl.float32)
        if FP8_CACHE:
            scale_ptr = (
                cache_scale_ptr
                + physical * stride_scale_page
                + kv_head * stride_scale_h
                + (index % PAGE_SIZE) * stride_scale_token
            )
            key *= tl.load(scale_ptr, mask=valid, other=1.0)[:, None]
        score = tl.sum(q[None, :] * key, axis=1) * SCALE
        score = tl.where(valid, score, float("-inf"))
        next_max = tl.maximum(maximum, tl.max(score, axis=0))
        alpha = tl.where(next_max == float("-inf"), 0.0, tl.exp(maximum - next_max))
        weights = tl.where(valid, tl.exp(score - next_max), 0.0)
        value = tl.load(key_ptr + stride_cache_kv, mask=valid[:, None], other=0.0).to(
            tl.float32
        )
        if FP8_CACHE:
            value *= tl.load(scale_ptr + stride_scale_kv, mask=valid, other=1.0)[
                :, None
            ]
        acc = acc * alpha + tl.sum(weights[:, None] * value, axis=0)
        normalizer = normalizer * alpha + tl.sum(weights)
        maximum = next_max
    output_row = (batch * H_Q + head) * QUERY_LEN + query
    if NUM_SPLITS == 1:
        out = tl.where(normalizer > 0, acc / normalizer, 0.0)
        tl.store(partial_ptr + output_row * D + rd, out.to(q_ptr.dtype.element_ty))
    else:
        base = partial_ptr + (output_row * NUM_SPLITS + split) * (D + 2)
        tl.store(base + rd, acc)
        tl.store(base + D, maximum)
        tl.store(base + D + 1, normalizer)


def _grouped_split_supported(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    selected: torch.Tensor,
    *,
    page_size: int,
    selected_width: int,
) -> bool:
    """Gate the tensor-core grouped kernel to proven SM90 geometry.

    Shape-only checks; no tensor contents are read and no device synchronisation
    happens, so the gate is safe during CUDA graph capture.
    """
    if torch.version.hip is not None or not q.is_cuda:
        return False
    if torch.cuda.get_device_capability(q.device) != (9, 0):
        return False
    if q.dtype != torch.bfloat16 or q.dim() != 4:
        return False
    if kv_cache.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        return False
    if kv_cache.dim() != 5 or int(kv_cache.shape[1]) != 2:
        return False
    heads, kv_heads = int(q.shape[1]), int(kv_cache.shape[2])
    if kv_heads <= 0 or heads % kv_heads:
        return False
    group = heads // kv_heads
    # tl.dot needs M >= 16; keep the register tile bounded.
    block_m = max(16, triton.next_power_of_2(group))
    if group > 32 or block_m > 32:
        return False
    if not (16 <= q.shape[3] <= 256) or q.shape[3] & (q.shape[3] - 1):
        return False
    if kv_cache.shape[3] != page_size or kv_cache.shape[4] != q.shape[3]:
        return False
    if page_size <= 0 or page_size & (page_size - 1):
        return False
    if selected.dim() != 3 or selected.dtype != torch.int32:
        return False
    if selected_width <= 0:
        return False
    return kv_cache.stride(-1) == 1


def _grouped_split_config(rows: int, kv_heads: int, selected_width: int) -> tuple:
    """(BLOCK_N, num_warps, num_stages, num_splits) for the grouped kernel.

    rows = batch * query_len. Splits keep small decode/verify batches parallel
    across SMs; prefill rows already saturate the device with one split.
    Starting table ported from the vLLM sm90 tuning shape; retuned per-benchmark.
    """
    base = rows * kv_heads
    if base > 4096:
        splits = 1
    elif base > 512:
        splits = 2
    elif base > 64:
        splits = 4
    else:
        splits = 8
    # BLOCK_N=128 measured fastest for prefill on SM90: fewer tiles amortise
    # the per-tile gather setup, and the bf16 dot operands still fit in shared
    # memory with two pipeline stages.
    block_n = 128 if base > 64 else 32
    tiles = triton.cdiv(selected_width, block_n)
    # The merge kernel indexes splits with tl.arange, so the split count itself
    # must be a power of two.
    splits = min(splits, 1 << (tiles.bit_length() - 1))
    return block_n, 4, 2, splits


@triton.jit
def _sparse_paged_grouped_split_kernel(
    q_ptr,
    cache_ptr,
    cache_scale_ptr,
    block_table_ptr,
    kv_lens_ptr,
    selected_ptr,
    partial_ptr,
    stride_q_b,
    stride_q_h,
    stride_q_s,
    stride_cache_page,
    stride_cache_kv,
    stride_cache_h,
    stride_cache_token,
    stride_scale_page,
    stride_scale_kv,
    stride_scale_h,
    stride_scale_token,
    stride_bt_b,
    stride_bt_block,
    stride_len_b,
    stride_len_s,
    stride_sel_b,
    stride_sel_s,
    selected_width,
    max_blocks,
    cache_blocks,
    HAS_COUNT: tl.constexpr,
    FP8_CACHE: tl.constexpr,
    H_Q: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    # QUERY_LEN/NUM_TILES stay runtime args: they vary with every request
    # length, and constexpr keys recompile the kernel per new length.
    QUERY_LEN,
    PAGE_SIZE: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Tensor-core grouped sparse paged GQA.

    One program owns a full GQA group (all query heads of one query row sharing
    one KV head), so each gathered K/V tile is loaded once per group instead of
    once per head, and QK^T / PV run on tensor cores via tl.dot. The group is
    the MMA M dimension; both dot operands keep the wgmma-friendly sourcing
    (A from registers, B from the pipelined gather tiles).
    """
    partition = tl.program_id(0)
    kv_head = tl.program_id(1)
    batch = partition // (QUERY_LEN * NUM_SPLITS)
    query = (partition // NUM_SPLITS) % QUERY_LEN
    split = partition % NUM_SPLITS

    heads = tl.arange(0, BLOCK_M)
    rd = tl.arange(0, D)
    live = heads < GROUP_SIZE
    q = tl.load(
        q_ptr
        + batch * stride_q_b
        + (kv_head * GROUP_SIZE + heads)[:, None] * stride_q_h
        + query * stride_q_s
        + rd[None, :],
        mask=live[:, None],
        other=0.0,
    )

    row = selected_ptr + batch * stride_sel_b + query * stride_sel_s
    kv_len = tl.load(kv_lens_ptr + batch * stride_len_b + query * stride_len_s)
    count = (
        tl.minimum(tl.maximum(tl.load(row + selected_width), 0), selected_width)
        if HAS_COUNT
        else selected_width
    )
    tiles = tl.cdiv(count, BLOCK_N)
    tile_start = split * NUM_TILES // NUM_SPLITS
    tile_end = tl.minimum((split + 1) * NUM_TILES // NUM_SPLITS, tiles)

    maximum = tl.full([BLOCK_M], float("-inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    columns = tl.arange(0, BLOCK_N)
    out_rows = (batch * H_Q + kv_head * GROUP_SIZE + heads) * QUERY_LEN + query
    for tile in range(tile_start, tile_end):
        offset = tile * BLOCK_N + columns
        index = tl.load(row + offset, mask=offset < count, other=-1)
        logical = index // PAGE_SIZE
        candidate = (offset < count) & (index >= 0) & (index < kv_len)
        physical = tl.load(
            block_table_ptr + batch * stride_bt_b + logical * stride_bt_block,
            mask=candidate & (logical < max_blocks),
            other=0,
        )
        valid = (
            candidate
            & (logical < max_blocks)
            & (physical > 0)
            & (physical < cache_blocks)
        )
        safe_page = tl.maximum(physical, 0).to(tl.int64)
        page_slot = tl.maximum(index, 0) % PAGE_SIZE
        keys = tl.load(
            cache_ptr
            + safe_page[None, :] * stride_cache_page
            + kv_head * stride_cache_h
            + page_slot[None, :] * stride_cache_token
            + rd[:, None],
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            cache_ptr
            + safe_page[:, None] * stride_cache_page
            + stride_cache_kv
            + kv_head * stride_cache_h
            + page_slot[:, None] * stride_cache_token
            + rd[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        if FP8_CACHE:
            # Per-token scales factor out of both dots: the K scale multiplies
            # the score column after the QK dot, and the V scale folds into the
            # attention weights before the PV dot. This keeps the tile-wide
            # work to a single exact fp8->bf16 conversion.
            key_scale = tl.load(
                cache_scale_ptr
                + safe_page * stride_scale_page
                + kv_head * stride_scale_h
                + page_slot * stride_scale_token,
                mask=valid,
                other=0.0,
            )
            value_scale = tl.load(
                cache_scale_ptr
                + safe_page * stride_scale_page
                + stride_scale_kv
                + kv_head * stride_scale_h
                + page_slot * stride_scale_token,
                mask=valid,
                other=0.0,
            )
            keys = keys.to(tl.bfloat16)
            values = values.to(tl.bfloat16)
            scores = tl.dot(q, keys)
            scores = scores * (key_scale * SCALE)[None, :]
            scores = tl.where(valid[None, :], scores, float("-inf"))
            next_max = tl.maximum(maximum, tl.max(scores, 1))
            alpha = tl.where(next_max == float("-inf"), 0.0, tl.exp(maximum - next_max))
            weights = tl.where(
                valid[None, :], tl.exp(scores - next_max[:, None]), 0.0
            )
            # The softmax normalizer stays in raw probability space; only the PV
            # operand carries the per-token V scale.
            normalizer = normalizer * alpha + tl.sum(weights, 1)
            pv = weights * value_scale[None, :]
            acc = acc * alpha[:, None] + tl.dot(pv.to(tl.bfloat16), values)
            maximum = next_max
        else:
            keys = keys.to(tl.bfloat16)
            values = values.to(tl.bfloat16)
            scores = tl.dot(q, keys) * SCALE
            scores = tl.where(valid[None, :], scores, float("-inf"))
            next_max = tl.maximum(maximum, tl.max(scores, 1))
            alpha = tl.where(next_max == float("-inf"), 0.0, tl.exp(maximum - next_max))
            weights = tl.where(
                valid[None, :], tl.exp(scores - next_max[:, None]), 0.0
            )
            acc = acc * alpha[:, None] + tl.dot(weights.to(tl.bfloat16), values)
            normalizer = normalizer * alpha + tl.sum(weights, 1)
            maximum = next_max

    if NUM_SPLITS == 1:
        out = tl.where(normalizer[:, None] > 0, acc / normalizer[:, None], 0.0)
        tl.store(
            partial_ptr + out_rows[:, None] * D + rd[None, :],
            out,
            mask=live[:, None],
        )
    else:
        base = (out_rows * NUM_SPLITS + split) * (D + 2)
        tl.store(partial_ptr + base[:, None] + rd[None, :], acc, mask=live[:, None])
        tl.store(partial_ptr + base + D, maximum, mask=live)
        tl.store(partial_ptr + base + D + 1, normalizer, mask=live)


@triton.jit
def _merge_sparse_paged_splits_kernel(
    partial_ptr,
    out_ptr,
    gate_ptr,
    D: tl.constexpr,
    # NUM_SPLITS indexes tl.arange so it must stay constexpr (bounded
    # power-of-two set); QUERIES varies per request and stays runtime.
    NUM_SPLITS: tl.constexpr,
    HEADS: tl.constexpr,
    QUERIES,
    WITH_GATE: tl.constexpr,
):
    row = tl.program_id(0)
    splits = tl.arange(0, NUM_SPLITS)
    rd = tl.arange(0, D)
    base = partial_ptr + row * NUM_SPLITS * (D + 2)
    maxima = tl.load(base + splits * (D + 2) + D)
    sums = tl.load(base + splits * (D + 2) + D + 1)
    maximum = tl.max(maxima, axis=0)
    # Empty partitions contribute exactly zero, including an all-empty row.
    scale = tl.where(sums > 0, tl.exp(maxima - maximum), 0.0)
    partial = tl.load(base + splits[:, None] * (D + 2) + rd[None, :])
    numerator = tl.sum(partial * scale[:, None], axis=0)
    denominator = tl.sum(sums * scale, axis=0)
    out = tl.where(denominator > 0, numerator / denominator, 0.0)
    if WITH_GATE:
        batch = row // (HEADS * QUERIES)
        head = (row // QUERIES) % HEADS
        query = row % QUERIES
        offset = ((batch * QUERIES + query) * HEADS + head) * D + rd
        gate = tl.load(gate_ptr + offset).to(tl.float32)
        sigmoid = tl.sigmoid(gate).to(tl.bfloat16).to(tl.float32)
        out = out.to(tl.bfloat16).to(tl.float32) * sigmoid
    else:
        offset = row * D + rd
    tl.store(out_ptr + offset, out.to(out_ptr.dtype.element_ty))


def sparse_paged_gqa_attn(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    selected: torch.Tensor,
    *,
    page_size: int,
    kv_head_num: int | None = None,
    kv_scale: torch.Tensor | None = None,
    block_k: int = _BLOCK_K,
    graph_capture: bool = False,
    output_gate: torch.Tensor | None = None,
    selected_has_count: bool = False,
    fp16_pv: bool = False,
) -> torch.Tensor:
    """Attend to request-local token indices in a paged main MHA cache.

    Args:
        q: ``[B, H_q, S, D]`` BF16 CUDA queries after RoPE.
        kv_cache: HND cache ``[pages, 2, H_kv, page_size, D]`` or its packed
            two-dimensional HybridPool view.
        block_table: ``[B, max_blocks]`` int32 CUDA logical-to-physical map.
        kv_lens: ``[B, S]`` int32 CUDA visible main-KV lengths.
        selected: ``[B, S, K]`` int32 CUDA request-local token indices; ``-1``
            is padding.
        page_size: tokens per main MHA cache page.
        kv_head_num: local KV head count. Required for a packed HybridPool view,
            whose row stride may be padded by a larger cache group and therefore
            cannot be inferred from ``kv_cache.shape[1]``.
        graph_capture: capture only. The caller must validate the initial
            metadata before capture; replayed values are masked by the kernel.
    """
    if q.dim() != 4 or q.dtype != torch.bfloat16 or not q.is_cuda:
        raise ValueError("q must be a rank-4 BF16 CUDA tensor")
    batch, q_heads, query_len, head_dim = q.shape
    if output_gate is not None and not (
        output_gate.shape == (batch, query_len, q_heads, head_dim)
        and output_gate.dtype == torch.bfloat16
        and output_gate.device == q.device
        and output_gate.is_contiguous()
    ):
        raise ValueError(
            "QSA output gate must be contiguous BF16 [B,S,H,D] on the query device"
        )
    if q_heads <= 0 or head_dim <= 0 or head_dim & (head_dim - 1):
        raise ValueError("query heads must be positive and head_dim a power of two")
    if page_size <= 0 or page_size & (page_size - 1):
        raise ValueError(f"page_size must be a positive power of two, got {page_size}")
    if block_k <= 0 or block_k & (block_k - 1):
        raise ValueError(f"block_k must be a positive power of two, got {block_k}")
    if (
        block_table.dim() != 2
        or tuple(block_table.shape[:1]) != (batch,)
        or block_table.dtype != torch.int32
    ):
        raise ValueError(f"block_table must be int32 [{batch}, max_blocks]")
    if kv_lens.shape != (batch, query_len) or kv_lens.dtype != torch.int32:
        raise ValueError(f"kv_lens must be int32 [{batch}, {query_len}]")
    if selected.dim() != 3 or selected.shape[:2] != (batch, query_len):
        raise ValueError(f"selected must have leading shape [{batch}, {query_len}]")
    if selected.dtype != torch.int32:
        raise ValueError("selected must be int32")

    if selected_has_count and selected.shape[-1] != 2052:
        raise ValueError("packed QSA requires 2051 token slots and one valid_count")

    if kv_cache.dim() == 2:
        if kv_head_num is None or kv_head_num <= 0:
            raise ValueError("kv_head_num is required for a packed HybridPool view")
        required_width = 2 * kv_head_num * page_size * head_dim
        if int(kv_cache.shape[1]) < required_width:
            raise ValueError(
                f"packed kv_cache width {kv_cache.shape[1]} is smaller than "
                f"the required main-cache width {required_width}"
            )
        if kv_cache.stride(1) != 1:
            raise ValueError("packed HybridPool rows must have contiguous elements")
        elem_stride = kv_cache.stride(1)
        kv_cache = torch.as_strided(
            kv_cache,
            (int(kv_cache.shape[0]), 2, kv_head_num, page_size, head_dim),
            (
                kv_cache.stride(0),
                kv_head_num * page_size * head_dim * elem_stride,
                page_size * head_dim * elem_stride,
                head_dim * elem_stride,
                elem_stride,
            ),
        )
    if kv_cache.dim() != 5 or int(kv_cache.shape[1]) != 2:
        raise ValueError("kv_cache must be [pages, 2, H_kv, page_size, D]")
    cache_blocks, _, kv_heads, cache_page_size, cache_head_dim = kv_cache.shape
    if kv_head_num is not None and kv_heads != kv_head_num:
        raise ValueError(
            f"kv_cache has {kv_heads} KV heads, expected kv_head_num={kv_head_num}"
        )
    if cache_page_size != page_size or cache_head_dim != head_dim:
        raise ValueError(
            "kv_cache page/head geometry does not match q: "
            f"cache={tuple(kv_cache.shape)}, page_size={page_size}, D={head_dim}"
        )
    if kv_heads <= 0 or q_heads % kv_heads:
        raise ValueError(
            f"query heads {q_heads} must be divisible by KV heads {kv_heads}"
        )
    tensors = (kv_cache, block_table, kv_lens, selected)
    if any(not tensor.is_cuda or tensor.device != q.device for tensor in tensors):
        raise ValueError(
            "q, kv_cache, block_table, kv_lens, and selected must share CUDA"
        )
    scale_view = main_kv_scale_view(
        kv_cache, kv_scale, kv_heads=kv_heads, page_size=page_size
    )
    scale_pointer = kv_cache if scale_view is None else scale_view
    scale_strides = (0, 0, 0, 0) if scale_view is None else scale_view.stride()
    expected_inner_strides = (
        kv_heads * page_size * head_dim,
        page_size * head_dim,
        head_dim,
        1,
    )
    if tuple(kv_cache.stride()[1:]) != expected_inner_strides:
        raise ValueError(
            "kv_cache must use HND inner layout; got strides "
            f"{tuple(kv_cache.stride())}"
        )
    if int(block_table.shape[1]) == 0:
        raise ValueError("block_table must contain at least one logical block")
    if graph_capture:
        warmup = os.environ.get("RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD") == "1"
        if not torch.cuda.is_current_stream_capturing() and not warmup:
            raise RuntimeError(
                "qwen4 sparse paged GQA graph_capture requires an active capture or graph warmup"
            )
    else:
        if selected_has_count:
            count = selected[..., -1]
            slots = torch.arange(2051, device=selected.device)
            tokens = selected[..., :-1]
            invalid = (count < 0) | (count > 2051) | ((tokens >= 0).sum(-1) != count)
            invalid = (
                invalid.any() | ((tokens >= 0) != (slots < count.unsqueeze(-1))).any()
            )
            if qsa_strict_validation_enabled():
                if bool(invalid.item()):
                    raise ValueError("invalid packed QSA prefix/count")
            else:
                qsa_assert_valid(~invalid, "invalid packed QSA prefix/count")
        paged_selected = selected[..., :-1] if selected_has_count else selected
        if qsa_strict_validation_enabled():
            _validate_sparse_paged_indices(
                block_table,
                kv_lens,
                paged_selected,
                page_size=page_size,
                cache_blocks=cache_blocks,
            )
        else:
            # FMHA-entry validation (the kept half of the pair with the
            # per-request planner checks): one enqueue-only device verdict.
            qsa_assert_valid(
                _sparse_paged_indices_verdict(
                    block_table,
                    kv_lens,
                    paged_selected,
                    page_size=page_size,
                    cache_blocks=cache_blocks,
                ),
                "qwen4_exp sparse paged GQA selection, kv_lens or block table "
                "is invalid",
            )

    q = q.contiguous()
    block_table = block_table.contiguous()
    kv_lens = kv_lens.contiguous()
    selected = selected.contiguous()
    output = torch.empty_like(q)
    if batch == 0 or query_len == 0:
        return output

    selected_width = int(selected.shape[2]) - int(selected_has_count)
    if selected_width == 0:
        output.zero_()
        if output_gate is not None:
            output = output * torch.sigmoid(output_gate.transpose(1, 2))
        return output
    if _grouped_split_supported(
        q, kv_cache, selected, page_size=page_size, selected_width=selected_width
    ):
        kv_heads = int(kv_cache.shape[2])
        group = q_heads // kv_heads
        block_n, warps, stages, splits = _grouped_split_config(
            batch * query_len, kv_heads, selected_width
        )
        partial = (
            torch.empty(
                (batch, q_heads, query_len, splits, head_dim + 2),
                dtype=torch.float32,
                device=q.device,
            )
            if splits > 1
            else output
        )
        _sparse_paged_grouped_split_kernel[(batch * query_len * splits, kv_heads)](
            q,
            kv_cache,
            scale_pointer,
            block_table,
            kv_lens,
            selected,
            partial,
            *q.stride()[:3],
            *kv_cache.stride()[:4],
            *scale_strides,
            *block_table.stride(),
            *kv_lens.stride(),
            *selected.stride()[:2],
            selected_width,
            int(block_table.shape[1]),
            cache_blocks,
            HAS_COUNT=selected_has_count,
            FP8_CACHE=scale_view is not None,
            H_Q=q_heads,
            GROUP_SIZE=group,
            QUERY_LEN=query_len,
            PAGE_SIZE=page_size,
            D=head_dim,
            SCALE=1.0 / float(head_dim) ** 0.5,
            NUM_SPLITS=splits,
            NUM_TILES=triton.cdiv(selected_width, block_n),
            BLOCK_M=max(16, triton.next_power_of_2(group)),
            BLOCK_N=block_n,
            num_warps=warps,
            num_stages=stages,
        )
        if splits > 1:
            if output_gate is not None:
                output = torch.empty(
                    (batch, query_len, q_heads, head_dim), device=q.device, dtype=q.dtype
                ).transpose(1, 2)
            _merge_sparse_paged_splits_kernel[(batch * q_heads * query_len,)](
                partial,
                output,
                output_gate if output_gate is not None else output,
                D=head_dim,
                NUM_SPLITS=splits,
                HEADS=q_heads,
                QUERIES=query_len,
                WITH_GATE=output_gate is not None,
            )
        elif output_gate is not None:
            output = output * torch.sigmoid(output_gate.transpose(1, 2))
        return output
    splits = min(8, triton.next_power_of_2(triton.cdiv(selected_width, block_k)))
    from .grouped_sparse_paged_fmha import preferred_splits

    tuned_splits = preferred_splits(
        q, kv_cache, selected, page_size=page_size, block_k=block_k
    )
    if tuned_splits is not None:
        splits = tuned_splits

    partial = (
        torch.empty(
            (batch, q_heads, query_len, splits, head_dim + 2),
            dtype=torch.float32,
            device=q.device,
        )
        if splits > 1
        else output
    )
    grouped = False
    from rtp_llm.models_py.modules.qwen4_exp.qsa.grouped_sparse_paged_fmha import (
        grouped_sparse_paged_split,
        is_supported,
    )

    grouped = is_supported(
        q,
        kv_cache,
        block_table,
        kv_lens,
        selected,
        partial,
        page_size=page_size,
        splits=splits,
        block_k=block_k,
        scale=scale_view,
        selected_has_count=selected_has_count,
    )
    if grouped:
        grouped_sparse_paged_split(
            q,
            kv_cache,
            block_table,
            kv_lens,
            selected,
            partial,
            page_size=page_size,
            splits=splits,
            scale=scale_view,
            selected_has_count=selected_has_count,
            fp16_pv=fp16_pv,
        )
    if not grouped:
        # CUDA grid Y/Z are limited to 65535. Keep all query partitions on X.
        _sparse_paged_online_split_kernel[(batch * query_len * splits, q_heads)](
            q,
            kv_cache,
            scale_pointer,
            block_table,
            kv_lens,
            selected,
            partial,
            *q.stride()[:3],
            *kv_cache.stride()[:4],
            *scale_strides,
            *block_table.stride(),
            *kv_lens.stride(),
            *selected.stride()[:2],
            selected_width,
            int(block_table.shape[1]),
            cache_blocks,
            HAS_COUNT=selected_has_count,
            FP8_CACHE=scale_view is not None,
            H_Q=q_heads,
            H_KV=kv_heads,
            QUERY_LEN=query_len,
            PAGE_SIZE=page_size,
            D=head_dim,
            SCALE=1.0 / float(head_dim) ** 0.5,
            BLOCK_K=block_k,
            NUM_SPLITS=splits,
            BLOCKS_PER_SPLIT=triton.cdiv(selected_width, block_k * splits),
        )
    if splits > 1:
        if output_gate is not None:
            # The owner consumes [B,S,H,D]. Store that layout directly, while
            # retaining the public attention result's logical [B,H,S,D] shape.
            output = torch.empty(
                (batch, query_len, q_heads, head_dim), device=q.device, dtype=q.dtype
            ).transpose(1, 2)
        _merge_sparse_paged_splits_kernel[(batch * q_heads * query_len,)](
            partial,
            output,
            output_gate if output_gate is not None else output,
            D=head_dim,
            NUM_SPLITS=splits,
            HEADS=q_heads,
            QUERIES=query_len,
            WITH_GATE=output_gate is not None,
        )
    elif output_gate is not None:
        output = output * torch.sigmoid(output_gate.transpose(1, 2))
    return output
