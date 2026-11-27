"""DSV4 SWA FP8 KV insert — vLLM-aligned Triton kernel.

Vendored verbatim (signature + math) from vLLM
``vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py``
(``quantize_and_insert_k_kernel`` / ``quantize_and_insert_k_cache``).

One Triton program per token: quantize BF16 K → UE8M0 FP8 NoPE + BF16 RoPE,
write packed bytes into ``[num_blocks, block_size, 584]`` uint8 cache.

Per-token byte layout (fp8_ds_mla / fp8_model1_mla — identical):
    [0,   448) NoPE FP8 e4m3fn (7 quant tiles × 64 elements)
    [448, 576) RoPE BF16        (64 elements)
    [scales region @ block end] 7 ue8m0 scale bytes + 1 padding

Replaces the legacy two-step write (torch slot_mapping calc + CUDA
``concat_and_cache_mla``) used in ``_prefill_write_swa_fp8_paged`` —
single Triton launch covers the full prefill chunk.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv4.fp8._swa_cp_byte_sliced import (
    CPByteSlicedSlotCompaction,
)


@triton.jit
def _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS: tl.constexpr) -> None:
    if TRAP_INVALID_KV_ACCESS:
        tl.inline_asm_elementwise(
            "trap; // dummy $0",
            "=r",
            [],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )


@triton.jit(do_not_specialize=["num_tokens", "block_stride", "num_cache_blocks"])
def _quantize_and_insert_k_kernel(
    # Inputs
    k_ptr,  # [num_tokens, 512] bf16
    slot_mapping_ptr,  # [num_tokens] int64; -1 = skip
    # Output (in-place)
    k_cache_ptr,  # [num_blocks, block_size, 584] uint8 (3D)
    # Dimensions / constexprs
    num_tokens,
    input_dim: tl.constexpr,  # 512
    fp8_dim: tl.constexpr,  # 448
    bf16_dim: tl.constexpr,  # 64
    scale_dim: tl.constexpr,  # 8 (7 real scale bytes + 1 pad)
    quant_block: tl.constexpr,  # 64 (NoPE quantization tile size)
    cache_block_size: tl.constexpr,  # tokens per pool block (RTP-LLM: 256)
    token_data_size: tl.constexpr,  # 576 = 448 + 128
    block_stride,  # bytes per block (TMA-padded; from k_cache.stride(0))
    num_cache_blocks,
    fp8_max: tl.constexpr,  # 448.0
    n_quant_blocks: tl.constexpr,  # 8 (7 real + 1 padding tile-loop iter)
    TRAP_INVALID_KV_ACCESS: tl.constexpr,
):
    """One program per token. Block layout per pool block:
    [0, cache_block_size * 576)             — token data (NoPE+RoPE interleaved)
    [cache_block_size * 576,
     cache_block_size * 576 + cache_block_size * 8)  — scales
    [..., block_stride)                      — TMA padding
    """
    pid = tl.program_id(0).to(tl.int64)
    if pid >= num_tokens:
        return

    slot_idx = tl.load(slot_mapping_ptr + pid)
    if slot_idx == -1:
        return

    block_idx = slot_idx // cache_block_size
    pos_in_block = slot_idx % cache_block_size
    if block_idx < 0:
        _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)
    if block_idx >= num_cache_blocks:
        _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)

    input_row_ptr = k_ptr + pid * input_dim

    # int64: block_idx * block_stride can overflow int32 with many blocks
    # (e.g. >= 14K at block_stride 149504 → 2^31). Matches dequant kernel.
    cache_block_ptr = k_cache_ptr + block_idx.to(tl.int64) * block_stride.to(tl.int64)

    token_data_ptr = cache_block_ptr + pos_in_block * token_data_size
    token_scale_ptr = (
        cache_block_ptr + cache_block_size * token_data_size + pos_in_block * scale_dim
    )

    token_fp8_ptr = token_data_ptr
    token_bf16_ptr = token_data_ptr + fp8_dim

    # Vectorized whole-row quantization: load the full [512] row once, view it
    # as [8, 64] quant groups (groups 0-6 = NoPE 448, group 7 = RoPE bytes,
    # computed but masked out of every store), and reduce each group's absmax
    # in a single tl.max(axis=1) instead of 7 serial cross-warp block
    # reductions + 7 dependent scalar scale stores. Math, dtypes, and output
    # bytes are identical to the former per-tile loop (max is exact and
    # order-independent; the UE8M0 ceil-log2 scale math is element-wise).
    row_offsets = tl.arange(0, input_dim)
    x_full = tl.load(input_row_ptr + row_offsets)
    x_groups = tl.reshape(x_full, (n_quant_blocks, quant_block))

    abs_x = tl.abs(x_groups)
    block_max = tl.max(abs_x, axis=1)
    block_max = tl.maximum(block_max, 1e-4)  # match CUDA fmaxf(amax, 1e-4)

    # UE8M0: scale = 2^ceil(log2(block_max / fp8_max)) per 64-element group.
    raw_scale = block_max / fp8_max
    log_scale = tl.log2(raw_scale)
    exponent = tl.ceil(log_scale)
    scale = tl.exp2(exponent)

    x_scaled = x_groups / scale[:, None]
    x_clamped = tl.clamp(x_scaled, -fp8_max, fp8_max)
    x_fp8 = x_clamped.to(tl.float8e4nv)
    x_uint8 = tl.reshape(x_fp8.to(tl.uint8, bitcast=True), (input_dim,))

    # NoPE fp8 bytes: one 512-wide masked store (lanes >= 448 masked out).
    tl.store(token_fp8_ptr + row_offsets, x_uint8, mask=row_offsets < fp8_dim)

    # UE8M0 scale bytes = exponent + 127 (clamp [0, 255]); one 8-byte store
    # where the real groups (fp8_dim / quant_block = 7) carry their encoded
    # scales and the padding lane carries 0.
    scale_offsets = tl.arange(0, scale_dim)
    encoded_scale = exponent + 127.0
    encoded_scale = tl.maximum(tl.minimum(encoded_scale, 255.0), 0.0)
    encoded_scale = tl.where(
        scale_offsets < fp8_dim // quant_block, encoded_scale, 0.0
    )
    tl.store(token_scale_ptr + scale_offsets, encoded_scale.to(tl.uint8))

    # RoPE: bf16 passthrough, 64 elements = 128 bytes in one load/store.
    bf16_input_offset = fp8_dim
    bf16_out_ptr = token_bf16_ptr.to(tl.pointer_type(tl.bfloat16))
    rope_offsets = tl.arange(0, bf16_dim)
    bf16_vals = tl.load(input_row_ptr + bf16_input_offset + rope_offsets)
    tl.store(bf16_out_ptr + rope_offsets, bf16_vals)


# Layout constants (mirror decode/fp8_kv_quant_decode_op.py).
_TOKEN_FP8_DIM = 448
_TOKEN_BF16_DIM = 64
_TOKEN_SCALE_DIM = 8
_QUANT_BLOCK_SIZE = 64
_FP8_MAX = 448.0
_TOKEN_DATA_SIZE = _TOKEN_FP8_DIM + _TOKEN_BF16_DIM * 2  # 576
_INPUT_DIM = 512


def quantize_and_insert_k_cache(
    k: torch.Tensor,  # [num_tokens, 512] bf16
    k_cache: torch.Tensor,  # [num_blocks, block_size, 584] uint8
    slot_mapping: torch.Tensor,  # [num_tokens] int64; -1 = skip
) -> None:
    """One-launch quantize-and-write into the FP8 SWA pool.

    Args mirror vLLM ``quantize_and_insert_k_cache`` exactly. ``block_size``
    is read from ``k_cache.shape[1]`` so this works with both vLLM (64)
    and RTP-LLM (256) page sizes. ``block_stride`` is read from
    ``k_cache.stride(0)`` so the C++-side TMA padding (576B alignment via
    ``block_size_bytes``) is honored automatically.

    Slots with value ``-1`` are skipped. The caller decides which logical
    blocks are writable by passing a per-token slot mapping; sparse positive
    block-table entries are written just like contiguous tail entries.
    """
    assert (
        k.dim() == 2 and k.shape[1] == _INPUT_DIM
    ), f"K must be [num_tokens, 512], got {tuple(k.shape)}"
    assert k.dtype == torch.bfloat16, f"K must be bf16, got {k.dtype}"
    assert (
        k_cache.dim() == 3 and k_cache.shape[-1] == 584 and k_cache.dtype == torch.uint8
    ), (
        f"k_cache must be [num_blocks, block_size, 584] uint8, got "
        f"{tuple(k_cache.shape)} / {k_cache.dtype}"
    )
    if slot_mapping.dtype != torch.long:
        slot_mapping = slot_mapping.to(torch.long)

    num_tokens = int(slot_mapping.shape[0])
    if num_tokens == 0:
        return

    block_size = int(k_cache.shape[1])
    block_stride = int(k_cache.stride(0))  # bytes per block (TMA-padded)

    grid = (num_tokens,)
    _quantize_and_insert_k_kernel[grid](
        k,
        slot_mapping,
        k_cache,
        num_tokens,
        input_dim=_INPUT_DIM,
        fp8_dim=_TOKEN_FP8_DIM,
        bf16_dim=_TOKEN_BF16_DIM,
        scale_dim=_TOKEN_SCALE_DIM,
        quant_block=_QUANT_BLOCK_SIZE,
        cache_block_size=block_size,
        token_data_size=_TOKEN_DATA_SIZE,
        block_stride=block_stride,
        num_cache_blocks=int(k_cache.shape[0]),
        fp8_max=_FP8_MAX,
        n_quant_blocks=8,
        TRAP_INVALID_KV_ACCESS=True,
    )


@triton.jit(
    do_not_specialize=[
        "N",
        "D",
        "M_dst",
        "src_stride_row",
        "src_stride_col",
        "dst_stride_row",
        "dst_stride_col",
        "slot_stride",
    ]
)
def _row_scatter_kernel(
    dst_ptr,
    slots_ptr,
    src_ptr,
    N,
    D,
    M_dst,
    src_stride_row,
    src_stride_col,
    dst_stride_row,
    dst_stride_col,
    slot_stride,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MASK_NEGATIVE: tl.constexpr,
    CONTIGUOUS_COLS: tl.constexpr,
    STRIDE_UNIT: tl.constexpr,
    TRAP_INVALID_KV_ACCESS: tl.constexpr,
):
    """Vectorized 2D row scatter: ``dst[slots[r]] = src[r]`` for ``0 <= r < N``.

    Each CTA processes up to ``BLOCK_N`` rows × ``BLOCK_D`` columns with
    128-bit vectorization hints when ``STRIDE_UNIT > 1``. Negative slots are
    skipped in-kernel when ``MASK_NEGATIVE`` is enabled without host sync.
    """
    pid_n = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)

    row_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = row_offs < N

    slots = tl.load(
        slots_ptr + row_offs * slot_stride.to(tl.int64),
        mask=row_mask,
        other=0,
    ).to(tl.int64)
    if MASK_NEGATIVE:
        active_row = row_mask & (slots >= 0)
        invalid = active_row & (slots >= M_dst)
    else:
        active_row = row_mask
        invalid = active_row & ((slots < 0) | (slots >= M_dst))

    if tl.max(invalid.to(tl.int32), axis=0) != 0:
        _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)
        if MASK_NEGATIVE:
            active_row = active_row & (slots < M_dst)
        else:
            active_row = active_row & (slots >= 0) & (slots < M_dst)

    if tl.max(active_row.to(tl.int32), axis=0) == 0:
        return

    safe_slots = tl.where(active_row, slots, 0)

    if STRIDE_UNIT > 1:
        col_offs = tl.max_contiguous(
            tl.multiple_of(pid_d * BLOCK_D + tl.arange(0, BLOCK_D), [STRIDE_UNIT]),
            [BLOCK_D],
        )
        col_mask = (col_offs // STRIDE_UNIT) < (D // STRIDE_UNIT)
        s_src = src_stride_row.to(tl.int64) * STRIDE_UNIT
        s_dst = dst_stride_row.to(tl.int64) * STRIDE_UNIT
        src_off = tl.max_contiguous(
            tl.multiple_of(
                row_offs[:, None] * s_src + col_offs[None, :],
                [1, STRIDE_UNIT],
            ),
            [1, BLOCK_D],
        )
        dst_off = tl.max_contiguous(
            tl.multiple_of(
                safe_slots[:, None] * s_dst + col_offs[None, :],
                [1, STRIDE_UNIT],
            ),
            [1, BLOCK_D],
        )
    else:
        col_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        col_mask = col_offs < D
        if CONTIGUOUS_COLS:
            src_off = row_offs[:, None] * src_stride_row.to(tl.int64) + col_offs[None, :]
            dst_off = (
                safe_slots[:, None] * dst_stride_row.to(tl.int64) + col_offs[None, :]
            )
        else:
            src_off = (
                row_offs[:, None] * src_stride_row.to(tl.int64)
                + col_offs[None, :].to(tl.int64) * src_stride_col.to(tl.int64)
            )
            dst_off = (
                safe_slots[:, None] * dst_stride_row.to(tl.int64)
                + col_offs[None, :].to(tl.int64) * dst_stride_col.to(tl.int64)
            )

    tile_mask = active_row[:, None] & col_mask[None, :]
    vals = tl.load(src_ptr + src_off, mask=tile_mask, other=0)
    tl.store(dst_ptr + dst_off, vals, mask=tile_mask)


def _can_reinterpret_row_width(t: torch.Tensor, target_bytes: int) -> bool:
    elem_bytes = t.element_size()
    if target_bytes <= elem_bytes or (target_bytes % elem_bytes) != 0:
        return False
    if t.stride(1) != 1:
        return False
    if (int(t.shape[1]) * elem_bytes) % target_bytes != 0:
        return False
    if (int(t.stride(0)) * elem_bytes) % target_bytes != 0:
        return False
    if (int(t.storage_offset()) * elem_bytes) % target_bytes != 0:
        return False
    return (t.data_ptr() % target_bytes) == 0


def _reinterpret_for_scatter(
    dst: torch.Tensor,
    src: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Widen aligned contiguous-column rows and return ``(dst_w, src_w, stride_unit)``."""
    if dst.stride(1) == 1 and src.stride(1) == 1:
        if _can_reinterpret_row_width(dst, 16) and _can_reinterpret_row_width(src, 16):
            dst_w = dst if dst.dtype == torch.int64 else dst.view(torch.int64)
            src_w = src if src.dtype == torch.int64 else src.view(torch.int64)
            return dst_w, src_w, 2
        for word_dtype, word_bytes in (
            (torch.int64, 8),
            (torch.int32, 4),
            (torch.int16, 2),
        ):
            if _can_reinterpret_row_width(
                dst, word_bytes
            ) and _can_reinterpret_row_width(src, word_bytes):
                return dst.view(word_dtype), src.view(word_dtype), 1
    if dst.element_size() == 1 and dst.dtype != torch.uint8:
        return dst.view(torch.uint8), src.view(torch.uint8), 1
    return dst, src, 1


def row_scatter(
    dst: torch.Tensor,
    slots: torch.Tensor,
    src: torch.Tensor,
    *,
    mask_negative: bool = False,
    trap_invalid_kv_access: bool = True,
) -> None:
    """In-place 2D row scatter ``dst[slots[r]] = src[r]``.

    Replaces native ``dst.index_copy_(0, slots, src)`` on CUDA with a wide
    vectorized Triton scatter that avoids ``int64`` slot-cast allocations,
    ``.contiguous()`` row-slice copies, and ``nonzero()`` host syncs.
    """
    assert dst.dim() == 2 and src.dim() == 2, (
        f"row_scatter expects 2D tensors, got dst={tuple(dst.shape)} "
        f"src={tuple(src.shape)}"
    )
    assert (
        dst.shape[1] == src.shape[1]
    ), f"row_scatter width mismatch: dst={tuple(dst.shape)} src={tuple(src.shape)}"
    assert (
        dst.dtype == src.dtype
    ), f"row_scatter dtype mismatch: dst={dst.dtype} src={src.dtype}"
    slots_1d = slots.reshape(-1)
    n_rows = int(src.shape[0])
    d_cols = int(src.shape[1])
    assert (
        int(slots_1d.numel()) == n_rows
    ), f"row_scatter slot/row mismatch: slots={int(slots_1d.numel())} rows={n_rows}"
    if n_rows == 0 or d_cols == 0:
        return

    if not (dst.is_cuda and src.is_cuda and slots_1d.is_cuda):
        slot_long = (
            slots_1d.to(torch.long) if slots_1d.dtype != torch.long else slots_1d
        )
        if not mask_negative:
            dst.index_copy_(0, slot_long, src)
            return
        valid = slot_long >= 0
        valid_idx = valid.nonzero(as_tuple=False).flatten()
        if valid_idx.numel() > 0:
            dst.index_copy_(
                0,
                slot_long.index_select(0, valid_idx),
                src.index_select(0, valid_idx),
            )
        return

    dst_w, src_w, stride_unit = _reinterpret_for_scatter(dst, src)
    d_eff = int(src_w.shape[1])
    m_dst = int(dst_w.shape[0])
    block_n = 8
    if d_eff <= 64:
        block_d = 64
    elif d_eff <= 128:
        block_d = 128
    else:
        block_d = 256
    num_warps = 4

    contiguous_cols = dst_w.stride(1) == 1 and src_w.stride(1) == 1
    grid = (triton.cdiv(n_rows, block_n), triton.cdiv(d_eff, block_d))
    _row_scatter_kernel[grid](
        dst_w,
        slots_1d,
        src_w,
        N=n_rows,
        D=d_eff,
        M_dst=m_dst,
        src_stride_row=int(src_w.stride(0)) // stride_unit,
        src_stride_col=int(src_w.stride(1)),
        dst_stride_row=int(dst_w.stride(0)) // stride_unit,
        dst_stride_col=int(dst_w.stride(1)),
        slot_stride=int(slots_1d.stride(0)),
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        MASK_NEGATIVE=mask_negative,
        CONTIGUOUS_COLS=contiguous_cols,
        STRIDE_UNIT=stride_unit,
        TRAP_INVALID_KV_ACCESS=trap_invalid_kv_access,
        num_warps=num_warps,
    )


def quantize_and_insert_k_cache_cp_byte_sliced(
    k: torch.Tensor,
    k_cache_raw: torch.Tensor,
    slot_mapping: torch.Tensor,
    full_entries_per_block: int,
    cp_rank: int,
    cp_size: int,
    compaction: CPByteSlicedSlotCompaction,
) -> None:
    assert (
        k.dim() == 2 and k.shape[1] == _INPUT_DIM
    ), f"K must be [num_tokens, 512], got {tuple(k.shape)}"
    assert k.dtype == torch.bfloat16, f"K must be bf16, got {k.dtype}"
    assert k_cache_raw.dim() == 2 and k_cache_raw.dtype == torch.uint8, (
        f"k_cache_raw must be [num_blocks, local_slice_bytes] uint8, got "
        f"{tuple(k_cache_raw.shape)} / {k_cache_raw.dtype}"
    )
    assert (
        k_cache_raw.stride(1) == 1
    ), f"k_cache_raw must be byte-contiguous, got {k_cache_raw.stride()}"
    full_entries_per_block = int(full_entries_per_block)
    cp_rank = int(cp_rank)
    cp_size = int(cp_size)
    assert full_entries_per_block > 0 and cp_size > 1 and 0 <= cp_rank < cp_size
    num_tokens = int(slot_mapping.shape[0])
    if num_tokens == 0:
        return
    assert (
        compaction is not None
    ), "CP byte-sliced SWA insert requires metadata-precomputed compaction"
    unique_blocks = compaction.unique_blocks
    compact_slots = compaction.compact_slots
    if unique_blocks.numel() == 0:
        return

    local_slice_bytes = int(k_cache_raw.shape[1])
    full_stride_bytes = local_slice_bytes * cp_size
    slice_start = cp_rank * local_slice_bytes
    slice_end = slice_start + local_slice_bytes

    full_raw = torch.zeros(
        (int(unique_blocks.numel()), full_stride_bytes),
        dtype=torch.uint8,
        device=k_cache_raw.device,
    )
    full_view = full_raw.as_strided(
        (int(unique_blocks.numel()), full_entries_per_block, 584),
        (full_stride_bytes, 584, 1),
    )
    quantize_and_insert_k_cache(
        k,
        full_view,
        compact_slots,
    )
    row_scatter(
        k_cache_raw,
        unique_blocks,
        full_raw[:, slice_start:slice_end],
    )
