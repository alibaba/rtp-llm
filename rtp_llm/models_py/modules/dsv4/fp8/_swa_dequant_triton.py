"""DSV4 SWA FP8 KV cache → BF16 dequant + gather (prefill side).

Vendored from vLLM ``vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py``
``_dequantize_and_gather_k_kernel`` / ``dequantize_and_gather_k_cache``.
The byte layout (584B/token = 448 fp8 e4m3fn + 128 bf16 + 8 ue8m0 scale
bytes, with scales placed at the END of every block) is identical to
RTP-LLM's existing decode-side ``fp8_kv_quant_decode_op.py`` / vLLM's
``fp8_ds_mla``, so the same kernel reads both.

Only used for prefill: decode reads packed FP8 directly via
``flash_mla_with_kvcache(is_fp8_kvcache=True)`` and never dequantizes.
Prefill goes through TileLang's ``sparse_attn`` which expects BF16 KV,
so we dequant a per-request SWA window into a workspace buffer first
(matches vLLM's prefill flow).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional, Tuple

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv4._profiler import record_function_range
from rtp_llm.models_py.modules.dsv4.fp8._swa_cp_byte_sliced import (
    CPByteSlicedSlotCompaction,
)


def _acquire_swa_cp_buffer(
    device: torch.device,
    slot_key: str,
    shape: Tuple[int, ...],
    dtype: torch.dtype = torch.uint8,
) -> torch.Tensor:
    from rtp_llm.models_py.distributed import collective_torch

    acq = getattr(collective_torch, "acquire_grow_only_buffer", None)
    if (
        acq is not None
        and torch.distributed.is_initialized()
        and (device.type != "cuda" or not torch.cuda.is_current_stream_capturing())
    ):
        return acq(device, slot_key, shape, dtype=dtype)
    return torch.empty(shape, dtype=dtype, device=device)


@dataclass
class CPByteSlicedSwaPrefixPending:
    cp_size: int
    B: int
    W: int
    offset: int
    full_entries_per_block: int
    gathered: torch.Tensor
    unique_blocks: torch.Tensor
    compact_slots: torch.Tensor
    gather_lens: torch.Tensor
    gather_lens_cpu: Tuple[int, ...]
    work: Any
    stream: Any
    producer_stream: Any
    completion_event: Any
    local_slices: torch.Tensor
    out: Optional[torch.Tensor] = None
    out_stream: Any = None
    ready_event: Any = None
    work_waited: bool = False
    cu_tiles: Optional[torch.Tensor] = None
    total_tiles: Optional[int] = None
    cu_dst_offsets: Optional[torch.Tensor] = None
    cu_dst_offsets_cpu: Optional[Tuple[int, ...]] = None


# Layout constants — must stay byte-aligned with
# ``decode/fp8_kv_quant_decode_op.py`` and vLLM ``cache_utils.py``.
NOPE_DIM = 448
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM  # 512
TILE_SIZE = 64  # quant tile (UE8M0 scale per 64 NoPE elements)
NOPE_TILES = NOPE_DIM // TILE_SIZE  # 7
NOPE_BYTES = NOPE_DIM
ROPE_BYTES = ROPE_DIM * 2  # bf16
TOKEN_DATA_SIZE = NOPE_BYTES + ROPE_BYTES  # 576
SCALE_BYTES_PER_TOKEN = 8  # 7 real + 1 padding
ENTRY_BYTES = TOKEN_DATA_SIZE + SCALE_BYTES_PER_TOKEN  # 584
FP8_MAX = 448.0
# Token-tile dequant: each program handles TILE_T contiguous gathered tokens
# (distinct from TILE_SIZE, the 64-element UE8M0 quant group). The fp8 row is
# padded to a power of two so one wide 2D load covers it; the padded lanes
# stay masked out of every load and store.
TILE_T = 4
FP8_DIM_POW2 = triton.next_power_of_2(NOPE_DIM)
N_SCALE_GROUPS = FP8_DIM_POW2 // TILE_SIZE
# Minimum cache row slice (bytes) that guarantees at most one CP-rank crossing
# across either the padded FP8 NoPE row or the BF16 RoPE segment.
_MIN_SINGLE_CARRY_SLICE_BYTES = max(FP8_DIM_POW2, ROPE_BYTES)
_DIRECT_TRITON_SUPPORTED_CC_MAJORS = frozenset((9, 10))


@lru_cache(maxsize=32)
def _direct_triton_fast_path_supported_cached(
    device_type: str, device_index: Optional[int]
) -> bool:
    if device_type != "cuda":
        return False
    try:
        major, _ = torch.cuda.get_device_capability(device_index)
    except Exception:
        return False
    return int(major) in _DIRECT_TRITON_SUPPORTED_CC_MAJORS


def direct_triton_fast_path_supported(device: torch.device) -> bool:
    """Return whether direct DSV4 Triton paths support ``device``.

    Capability discovery is cached by CUDA device index and never reads tensor
    data, so checking the gate on the per-layer path does not introduce D2H
    synchronization. Unsupported or unknown architectures retain the existing
    PyTorch fallback.
    """
    device = torch.device(device)
    return _direct_triton_fast_path_supported_cached(device.type, device.index)


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


def _select_dequant_num_workers(num_reqs: int, max_tokens_hint: int = 0) -> int:
    """Scale CTA grid dim-1 for small-batch SM occupancy without empty CTAs.

    The returned count is the number of TILE_T-token tile programs per request:
    2048 total tiles across the batch (e.g. 1024 CTAs x 4 tokens for B=2)
    covers large SMs while ``max_tokens_hint`` avoids spawning workers with no
    tiles to process on short prefills.
    """
    if num_reqs <= 0:
        return 128
    target = max(128, (2048 + num_reqs - 1) // num_reqs)
    if max_tokens_hint > 0:
        tile_hint = max(1, (int(max_tokens_hint) + TILE_T - 1) // TILE_T)
        target = min(target, tile_hint)
    return min(1024, target)


def build_dequant_cu_tiles_from_host(
    lens_host: Any,
    device: torch.device,
    tile_t: int = TILE_T,
) -> Tuple[torch.Tensor, int]:
    """Build ``cu_tiles[B+1]`` (int32) and total tile count from host lengths."""
    cu_list = [0]
    acc = 0
    for length in lens_host:
        val = int(length)
        if val > 0:
            acc += (val + tile_t - 1) // tile_t
        cu_list.append(acc)
    return torch.tensor(cu_list, dtype=torch.int32, device=device), acc


def build_dequant_cu_tiles_device(
    lens: torch.Tensor,
    tile_t: int = TILE_T,
) -> torch.Tensor:
    """Build ``cu_tiles[B+1]`` (int32) on ``lens.device`` without host sync."""
    B = int(lens.numel())
    cu = torch.zeros(B + 1, dtype=torch.int32, device=lens.device)
    if B > 0:
        tiles = (lens.to(torch.int32).clamp_min(0) + tile_t - 1) // tile_t
        torch.cumsum(tiles, dim=0, dtype=torch.int32, out=cu[1:])
    return cu


@triton.jit
def _resolve_flat_tile_batch_and_idx(cu_tiles_ptr, num_reqs):
    """Map 1D ``program_id(0)`` to ``(batch_idx, tile_in_req)`` via ``cu_tiles``."""
    tile_id = tl.program_id(0)
    lo = tile_id * 0
    hi = tile_id * 0 + num_reqs
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        mid_start = tl.load(cu_tiles_ptr + mid)
        go_left = tile_id < mid_start
        hi = tl.where(go_left, mid, hi)
        lo = tl.where(go_left, lo, mid)
    batch_idx = lo
    tile_idx = tile_id - tl.load(cu_tiles_ptr + batch_idx)
    return batch_idx, tile_idx


@triton.jit(
    do_not_specialize=[
        "out_stride0",
        "out_stride1",
        "offset",
        "max_blocks_per_seq",
        "num_reqs",
    ]
)
def _dequantize_and_gather_k_kernel(
    out_ptr,
    out_stride0,
    out_stride1,
    k_cache_ptr,
    seq_lens_ptr,
    block_table_ptr,
    offset,
    gather_lens_ptr,
    max_blocks_per_seq,
    cu_tiles_ptr,
    cu_dst_offsets_ptr,
    num_reqs,
    fp8_dim: tl.constexpr,
    bf16_dim: tl.constexpr,
    scale_dim: tl.constexpr,
    quant_block: tl.constexpr,
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,
    block_stride: tl.constexpr,
    num_cache_blocks: tl.constexpr,
    output_dim: tl.constexpr,
    fp8_max: tl.constexpr,
    n_quant_blocks: tl.constexpr,
    fp8_dim_pow2: tl.constexpr,
    n_scale_groups: tl.constexpr,
    TILE_T: tl.constexpr,
    TRAP_INVALID_KV_ACCESS: tl.constexpr,
    USE_FLAT_GRID: tl.constexpr,
    HAS_CU_DST_OFFSETS: tl.constexpr,
):
    """One program per (request, tile worker) or per active flat 1D tile.

    Each loop iteration dequantizes TILE_T contiguous gathered tokens with one
    wide 2D fp8 load, one (TILE_T, n_scale_groups) scale load, and one wide RoPE
    transaction. Mirrors vLLM's per-token math exactly — see the V4 paper or
    vLLM's cache_utils.py.
    """
    if USE_FLAT_GRID:
        batch_idx, worker_id = _resolve_flat_tile_batch_and_idx(cu_tiles_ptr, num_reqs)
        num_workers = 1
    else:
        batch_idx = tl.program_id(0)
        worker_id = tl.program_id(1)
        num_workers = tl.num_programs(1)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    if gather_lens_ptr is not None:  # noqa: SIM108
        gather_len = tl.load(gather_lens_ptr + batch_idx)
    else:
        gather_len = seq_len
    start_pos = seq_len - gather_len

    num_tiles = (gather_len + TILE_T - 1) // TILE_T
    if USE_FLAT_GRID:
        num_tiles = tl.minimum(num_tiles, worker_id + 1)
    for tile in range(worker_id, num_tiles, num_workers):
        i_vec = tile * TILE_T + tl.arange(0, TILE_T)
        # Rows past gather_len are never read downstream: their loads and
        # stores are masked and their token position is clamped to 0.
        row_valid = i_vec < gather_len
        pos_vec = start_pos + i_vec
        pos = tl.where(row_valid, pos_vec, 0)
        block_in_seq = pos // cache_block_size
        pos_in_block = pos % cache_block_size

        # TRAP_INVALID_KV_ACCESS bounds checks (valid rows only, before the
        # cache loads) — the same traps as the previous per-token kernel.
        bad_pos = row_valid & (pos_vec < 0)
        if tl.max(bad_pos.to(tl.int32), axis=0) > 0:
            _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)
        bad_block = row_valid & (block_in_seq >= max_blocks_per_seq)
        if tl.max(bad_block.to(tl.int32), axis=0) > 0:
            _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)

        block_table_row_ptr = block_table_ptr + batch_idx * max_blocks_per_seq
        physical_block_idx = tl.load(
            block_table_row_ptr + block_in_seq, mask=row_valid, other=0
        )
        bad_phys_neg = row_valid & (physical_block_idx < 0)
        if tl.max(bad_phys_neg.to(tl.int32), axis=0) > 0:
            _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)
        bad_phys = row_valid & (physical_block_idx >= num_cache_blocks)
        if tl.max(bad_phys.to(tl.int32), axis=0) > 0:
            _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)

        # int64: physical_block_idx * block_stride can exceed 2^31 with many
        # KV-cache blocks. Match vLLM.
        physical_block_idx = physical_block_idx.to(tl.int64)
        pos_in_block = pos_in_block.to(tl.int64)
        token_data_off = (
            physical_block_idx * block_stride + pos_in_block * token_data_size
        )
        token_scale_off = (
            physical_block_idx * block_stride
            + cache_block_size * token_data_size
            + pos_in_block * scale_dim
        )

        # Keep the row offset in int64. Production prefill workspaces can have
        # very large per-request strides, so batch_idx * out_stride0 may exceed
        # signed int32 even though the underlying tensor allocation is valid.
        output_row = offset.to(tl.int64) + i_vec.to(tl.int64)
        if HAS_CU_DST_OFFSETS:
            dst_base = tl.load(cu_dst_offsets_ptr + batch_idx).to(tl.int64)
            output_row_off = (dst_base + output_row) * out_stride0.to(tl.int64)
        else:
            batch_off = batch_idx.to(tl.int64) * out_stride0.to(tl.int64)
            output_row_off = batch_off + output_row * out_stride1.to(tl.int64)

        # NoPE dequant: one 2D fp8 load over the padded [TILE_T, fp8_dim_pow2]
        # tile plus one (TILE_T, n_scale_groups) scale load. Reshaping to
        # (TILE_T, n_scale_groups, quant_block) broadcasts each token's
        # per-group scale exactly like the previous 7-tile loop.
        fp8_cols = tl.arange(0, fp8_dim_pow2)
        fp8_mask = row_valid[:, None] & (fp8_cols < fp8_dim)[None, :]
        fp8_off = token_data_off[:, None] + fp8_cols[None, :]
        x_uint8 = tl.load(k_cache_ptr + fp8_off, mask=fp8_mask, other=0)
        x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
        x_float = x_fp8.to(tl.float32)

        scale_cols = tl.arange(0, n_scale_groups)
        scale_mask = row_valid[:, None] & (scale_cols < n_quant_blocks)[None, :]
        scale_off = token_scale_off[:, None] + scale_cols[None, :]
        encoded_scale = tl.load(k_cache_ptr + scale_off, mask=scale_mask, other=0)
        exponent = encoded_scale.to(tl.float32) - 127.0
        scale = tl.exp2(exponent)
        x_groups = tl.reshape(x_float, (TILE_T, n_scale_groups, quant_block))
        x_dequant = x_groups * scale[:, :, None]
        x_out = tl.reshape(x_dequant, (TILE_T, fp8_dim_pow2)).to(tl.bfloat16)
        fp8_store_off = output_row_off[:, None] + fp8_cols[None, :]
        tl.store(out_ptr + fp8_store_off, x_out, mask=fp8_mask)

        # RoPE: bf16 passthrough (single vectorized 64-elem transaction when
        # bf16_dim == 64; byte offsets on the uint8 pointer, then one cast).
        bf16_output_offset = fp8_dim
        if bf16_dim == 64:
            rope_offsets = tl.arange(0, 64)
            rope_mask = row_valid[:, None]
            rope_off = token_data_off[:, None] + fp8_dim + rope_offsets[None, :] * 2
            rope_ptrs = (k_cache_ptr + rope_off).to(tl.pointer_type(tl.bfloat16))
            bf16_vals = tl.load(rope_ptrs, mask=rope_mask, other=0)
            rope_store_off = (
                output_row_off[:, None] + bf16_output_offset + rope_offsets[None, :]
            )
            tl.store(out_ptr + rope_store_off, bf16_vals, mask=rope_mask)
        else:
            for j in tl.static_range(bf16_dim // 16):
                chunk_offsets = j * 16 + tl.arange(0, 16)
                chunk_mask = row_valid[:, None]
                chunk_off = (
                    token_data_off[:, None] + fp8_dim + chunk_offsets[None, :] * 2
                )
                chunk_ptrs = (k_cache_ptr + chunk_off).to(tl.pointer_type(tl.bfloat16))
                bf16_vals = tl.load(chunk_ptrs, mask=chunk_mask, other=0)
                chunk_store_off = (
                    output_row_off[:, None]
                    + bf16_output_offset
                    + chunk_offsets[None, :]
                )
                tl.store(out_ptr + chunk_store_off, bf16_vals, mask=chunk_mask)


def dequantize_and_gather_k_cache(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: Optional[torch.Tensor],
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
    *,
    cu_tiles: Optional[torch.Tensor] = None,
    total_tiles: Optional[int] = None,
    cu_dst_offsets: Optional[torch.Tensor] = None,
    max_tokens_hint: int = 0,
) -> None:
    """Dequantize + gather FP8 K cache into a BF16 workspace.

    Direct port of vLLM ``dequantize_and_gather_k_cache``, extended with
    optional ragged 2D destination offsets via ``cu_dst_offsets``.

    Args:
      out:             Either ``[num_reqs, max_num_tokens, 512]`` bf16 when
                       ``cu_dst_offsets is None`` (writes
                       ``out[:, offset:offset+gather_len, :]``) or flat ragged
                       ``[total_rows, 512]`` bf16 when ``cu_dst_offsets`` is
                       provided (writes
                       ``out[cu_dst_offsets[b]+offset : ...+gather_len, :]``).
      k_cache:         ``[num_blocks, block_size, 584]`` uint8 — the FP8 SWA pool
                       in ``fp8_ds_mla`` layout.
      seq_lens:        ``[num_reqs]`` int — total tokens per request in ``k_cache``.
      gather_lens:     optional ``[num_reqs]`` int — tokens to gather (suffix);
                       ``None`` ⇒ gather all ``seq_lens[r]`` tokens for request r.
      block_table:     ``[num_reqs, max_blocks_per_seq]`` int — physical block id
                       per logical block per request.
      block_size:      tokens per cache block (matches the 3D shape of k_cache).
      offset:          row offset relative to request base in ``out`` to start writing.
      cu_dst_offsets:  optional ``[>= num_reqs]`` int32 contiguous tensor on
                       ``out.device`` giving base row per request in 2D ``out``.
      max_tokens_hint: optional per-request token count hint used when launching
                       the non-flat grid against a 2D ragged workspace.
    """
    num_reqs = int(seq_lens.shape[0])
    has_cu_dst_offsets = cu_dst_offsets is not None
    if has_cu_dst_offsets:
        assert (
            out.dim() == 2 and out.shape[-1] == HEAD_DIM and out.dtype == torch.bfloat16
        )
        assert (
            out.stride(1) == 1
        ), f"2D out must have contiguous last dim; got stride={out.stride()}"
        assert (
            cu_dst_offsets.dim() == 1
            and int(cu_dst_offsets.numel()) >= num_reqs
            and cu_dst_offsets.dtype == torch.int32
            and cu_dst_offsets.device == out.device
            and cu_dst_offsets.is_contiguous()
        ), (
            f"cu_dst_offsets contract mismatch: shape={cu_dst_offsets.shape} "
            f"dtype={cu_dst_offsets.dtype} num_reqs={num_reqs}"
        )
    else:
        assert (
            out.dim() == 3 and out.shape[-1] == HEAD_DIM and out.dtype == torch.bfloat16
        )
    assert (
        k_cache.dim() == 3
        and k_cache.shape[-1] == ENTRY_BYTES
        and k_cache.dtype == torch.uint8
    )
    # k_cache.stride(0) is in bytes (uint8 elements). The C++ side may pad
    # the per-block stride to satisfy FlashMLA TMA alignment (e.g. 149504
    # → 149760 for SWA bs=256), so this MUST come from the actual tensor
    # stride and NOT be reconstructed as block_size * ENTRY_BYTES.
    assert k_cache.stride(2) == 1 and k_cache.stride(1) == ENTRY_BYTES, (
        "k_cache must be packed within each token (stride[1]=584, stride[2]=1); "
        f"got stride={k_cache.stride()}"
    )

    # Caller invariant — int32 contig block tables; negative physical ids are
    # rejected before launch.
    assert block_table.dtype == torch.int32 and block_table.is_contiguous(), (
        "block_table must be int32 and contiguous; "
        f"got dtype={block_table.dtype} contig={block_table.is_contiguous()} "
        f"dev={block_table.device} (k_cache dev={k_cache.device})"
    )

    if cu_tiles is not None and total_tiles is not None:
        total_tiles_int = int(total_tiles)
        if num_reqs == 0 or total_tiles_int <= 0:
            return
        assert (
            cu_tiles.dim() == 1
            and int(cu_tiles.numel()) == num_reqs + 1
            and cu_tiles.dtype == torch.int32
            and cu_tiles.device == out.device
            and cu_tiles.is_contiguous()
        ), f"cu_tiles contract mismatch: shape={cu_tiles.shape} dtype={cu_tiles.dtype}"
        grid: Tuple[int, ...] = (total_tiles_int,)
        use_flat_grid = True
        cu_tiles_arg: Optional[torch.Tensor] = cu_tiles
    else:
        if num_reqs == 0:
            return
        tokens_hint = (
            int(max_tokens_hint)
            if int(max_tokens_hint) > 0
            else (int(out.shape[1]) if not has_cu_dst_offsets else int(out.shape[0]))
        )
        num_workers = _select_dequant_num_workers(num_reqs, max_tokens_hint=tokens_hint)
        grid = (num_reqs, num_workers)
        use_flat_grid = False
        cu_tiles_arg = None

    _dequantize_and_gather_k_kernel[grid](
        out,
        out.stride(0),
        out.stride(1),
        k_cache,
        seq_lens,
        block_table,
        offset,
        gather_lens,
        max_blocks_per_seq=block_table.shape[-1],
        cu_tiles_ptr=cu_tiles_arg,
        cu_dst_offsets_ptr=cu_dst_offsets,
        num_reqs=num_reqs,
        fp8_dim=NOPE_DIM,
        bf16_dim=ROPE_DIM,
        scale_dim=SCALE_BYTES_PER_TOKEN,
        quant_block=TILE_SIZE,
        cache_block_size=block_size,
        token_data_size=TOKEN_DATA_SIZE,
        block_stride=k_cache.stride(0),
        num_cache_blocks=int(k_cache.shape[0]),
        output_dim=HEAD_DIM,
        fp8_max=FP8_MAX,
        n_quant_blocks=NOPE_TILES,
        fp8_dim_pow2=FP8_DIM_POW2,
        n_scale_groups=N_SCALE_GROUPS,
        TILE_T=TILE_T,
        TRAP_INVALID_KV_ACCESS=True,
        USE_FLAT_GRID=use_flat_grid,
        HAS_CU_DST_OFFSETS=has_cu_dst_offsets,
        num_warps=4,
    )


@triton.jit
def _packed_slot_byte_ptrs_2d(
    k_cache_ptr,
    physical_block_idx,
    row_byte_offset,
    col_offsets,
    block_stride,
    num_cache_blocks,
    CP_RANK_MAJOR_RAW: tl.constexpr,
    SLICE_GE_TILE_COLS: tl.constexpr,
):
    """2D (rows, len(col_offsets)) byte-pointer tensor for one token tile.

    Tile-shaped extension of the previous per-element packed-slot translation.
    The CP rank-major raw layout stays exact for every byte: a token's data
    span can cross rank (block_stride) boundaries, so each element's rank is
    the per-row rank plus at most one carry comparison when slices are wider
    than the column span (SLICE_GE_TILE_COLS), or a full per-element division
    otherwise.
    """
    if CP_RANK_MAJOR_RAW:
        byte_rank = row_byte_offset // block_stride
        byte_in_rank = row_byte_offset - byte_rank * block_stride
        rel = byte_in_rank[:, None] + col_offsets.to(tl.int64)[None, :]
        if SLICE_GE_TILE_COLS:
            carry = (rel >= block_stride).to(tl.int64)
        else:
            carry = rel // block_stride
        rank = byte_rank[:, None] + carry
        in_rank = rel - carry * block_stride
        rank_major_row = rank * num_cache_blocks + physical_block_idx[:, None]
        full_off = rank_major_row * block_stride + in_rank
        return k_cache_ptr + full_off
    linear_off = physical_block_idx * block_stride + row_byte_offset
    full_off = linear_off[:, None] + col_offsets[None, :]
    return k_cache_ptr + full_off


@triton.jit(
    do_not_specialize=[
        "out_stride0",
        "out_stride1",
        "slot_mapping_stride0",
        "offset",
        "max_gather_len",
        "block_stride",
        "num_cache_blocks",
        "num_reqs",
    ]
)
def _dequantize_and_gather_k_slots_kernel(
    out_ptr,
    out_stride0,
    out_stride1,
    k_cache_ptr,
    slot_mapping_ptr,
    slot_mapping_stride0,
    offset,
    gather_lens_ptr,
    max_gather_len,
    cu_tiles_ptr,
    cu_dst_offsets_ptr,
    num_reqs,
    fp8_dim: tl.constexpr,
    bf16_dim: tl.constexpr,
    scale_dim: tl.constexpr,
    quant_block: tl.constexpr,
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,
    block_stride,
    num_cache_blocks,
    fp8_max: tl.constexpr,
    n_quant_blocks: tl.constexpr,
    fp8_dim_pow2: tl.constexpr,
    n_scale_groups: tl.constexpr,
    TILE_T: tl.constexpr,
    TRAP_INVALID_KV_ACCESS: tl.constexpr,
    CP_RANK_MAJOR_RAW: tl.constexpr,
    SLICE_GE_TILE_COLS: tl.constexpr,
    USE_FLAT_GRID: tl.constexpr,
    HAS_CU_DST_OFFSETS: tl.constexpr,
):
    """Gather/dequant using pre-translated flat global slot ids, TILE_T
    contiguous gathered tokens per loop iteration.

    ``slot < 0`` is the sentinel used by writers and metadata builders; it
    zero-fills the output row (the masked loads return zeros that are still
    stored) and never touches the cache pointer. Rows past each request's
    gather length are never stored: workspace writers rely on the per-request
    range disjointness of this kernel.
    """
    if USE_FLAT_GRID:
        batch_idx, worker_id = _resolve_flat_tile_batch_and_idx(cu_tiles_ptr, num_reqs)
        num_workers = 1
    else:
        batch_idx = tl.program_id(0)
        worker_id = tl.program_id(1)
        num_workers = tl.num_programs(1)

    if gather_lens_ptr is not None:  # noqa: SIM108
        gather_len = tl.load(gather_lens_ptr + batch_idx)
    else:
        gather_len = max_gather_len

    slot_row_ptr = slot_mapping_ptr + batch_idx.to(tl.int64) * slot_mapping_stride0

    num_tiles = (gather_len + TILE_T - 1) // TILE_T
    if USE_FLAT_GRID:
        num_tiles = tl.minimum(num_tiles, worker_id + 1)
    for tile in range(worker_id, num_tiles, num_workers):
        i_vec = tile * TILE_T + tl.arange(0, TILE_T)
        row_valid = i_vec < gather_len
        # Sentinel -1 for rows past gather_len: those rows are never stored.
        slots = tl.load(slot_row_ptr + i_vec, mask=row_valid, other=-1).to(tl.int64)
        slot_valid = slots >= 0
        row_mask = row_valid[:, None] & slot_valid[:, None]

        output_row = offset.to(tl.int64) + i_vec.to(tl.int64)
        if HAS_CU_DST_OFFSETS:
            dst_base = tl.load(cu_dst_offsets_ptr + batch_idx).to(tl.int64)
            output_row_off = (dst_base + output_row) * out_stride0.to(tl.int64)
        else:
            batch_off = batch_idx.to(tl.int64) * out_stride0.to(tl.int64)
            output_row_off = batch_off + output_row * out_stride1.to(tl.int64)

        physical_block_idx = slots // cache_block_size
        pos_in_block = slots % cache_block_size

        bad_phys = slot_valid & (physical_block_idx >= num_cache_blocks.to(tl.int64))
        if tl.max(bad_phys.to(tl.int32), axis=0) > 0:
            _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)

        # Sentinel rows address a clamped (block 0, entry 0) base; every cache
        # load below is masked off for them, so nothing is dereferenced.
        phys64 = tl.where(slot_valid, physical_block_idx, 0)
        pos64 = tl.where(slot_valid, pos_in_block, 0)
        token_data_off = pos64 * token_data_size
        token_scale_off = cache_block_size * token_data_size + pos64 * scale_dim

        fp8_cols = tl.arange(0, fp8_dim_pow2)
        fp8_mask = row_mask & (fp8_cols < fp8_dim)[None, :]
        fp8_ptrs = _packed_slot_byte_ptrs_2d(
            k_cache_ptr,
            phys64,
            token_data_off,
            fp8_cols,
            block_stride,
            num_cache_blocks,
            CP_RANK_MAJOR_RAW,
            SLICE_GE_TILE_COLS,
        )
        x_uint8 = tl.load(fp8_ptrs, mask=fp8_mask, other=0)
        x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
        x_float = x_fp8.to(tl.float32)

        scale_cols = tl.arange(0, n_scale_groups)
        scale_mask = row_mask & (scale_cols < n_quant_blocks)[None, :]
        scale_ptrs = _packed_slot_byte_ptrs_2d(
            k_cache_ptr,
            phys64,
            token_scale_off,
            scale_cols,
            block_stride,
            num_cache_blocks,
            CP_RANK_MAJOR_RAW,
            SLICE_GE_TILE_COLS,
        )
        encoded_scale = tl.load(scale_ptrs, mask=scale_mask, other=0)
        exponent = encoded_scale.to(tl.float32) - 127.0
        scale = tl.exp2(exponent)
        x_groups = tl.reshape(x_float, (TILE_T, n_scale_groups, quant_block))
        x_dequant = x_groups * scale[:, :, None]
        x_out = tl.reshape(x_dequant, (TILE_T, fp8_dim_pow2)).to(tl.bfloat16)
        # Sentinel rows still store zeros (the masked loads returned zeros);
        # rows past gather_len are masked out of the stores entirely.
        fp8_store_mask = row_valid[:, None] & (fp8_cols < fp8_dim)[None, :]
        fp8_store_off = output_row_off[:, None] + fp8_cols[None, :]
        tl.store(out_ptr + fp8_store_off, x_out, mask=fp8_store_mask)

        # RoPE: bf16 passthrough (single vectorized 64-elem transaction when
        # bf16_dim == 64).
        bf16_output_offset = fp8_dim
        if bf16_dim == 64:
            rope_offsets = tl.arange(0, 64)
            rope_ptrs = _packed_slot_byte_ptrs_2d(
                k_cache_ptr,
                phys64,
                token_data_off + fp8_dim,
                rope_offsets * 2,
                block_stride,
                num_cache_blocks,
                CP_RANK_MAJOR_RAW,
                SLICE_GE_TILE_COLS,
            ).to(tl.pointer_type(tl.bfloat16))
            bf16_vals = tl.load(rope_ptrs, mask=row_mask, other=0)
            rope_store_off = (
                output_row_off[:, None] + bf16_output_offset + rope_offsets[None, :]
            )
            tl.store(out_ptr + rope_store_off, bf16_vals, mask=row_valid[:, None])
        else:
            for j in tl.static_range(bf16_dim // 16):
                chunk_offsets = j * 16 + tl.arange(0, 16)
                chunk_ptrs = _packed_slot_byte_ptrs_2d(
                    k_cache_ptr,
                    phys64,
                    token_data_off + fp8_dim,
                    chunk_offsets * 2,
                    block_stride,
                    num_cache_blocks,
                    CP_RANK_MAJOR_RAW,
                    SLICE_GE_TILE_COLS,
                ).to(tl.pointer_type(tl.bfloat16))
                bf16_vals = tl.load(chunk_ptrs, mask=row_mask, other=0)
                chunk_store_off = (
                    output_row_off[:, None]
                    + bf16_output_offset
                    + chunk_offsets[None, :]
                )
                tl.store(out_ptr + chunk_store_off, bf16_vals, mask=row_valid[:, None])


def _launch_dequantize_and_gather_k_slots_unchecked(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    gather_lens: Optional[torch.Tensor],
    offset: int,
    *,
    cache_block_size: int,
    block_stride: int,
    num_cache_blocks: int,
    cp_rank_major_raw: bool,
    cu_tiles: Optional[torch.Tensor] = None,
    total_tiles: Optional[int] = None,
    cu_dst_offsets: Optional[torch.Tensor] = None,
) -> None:
    """Launch after the caller has established the packed-slot contracts."""
    batch_size = int(slot_mapping.shape[0])
    max_gather_len = int(slot_mapping.shape[1])
    if batch_size == 0 or max_gather_len == 0:
        return

    if cu_tiles is not None and total_tiles is not None:
        total_tiles_int = int(total_tiles)
        if total_tiles_int <= 0:
            return
        assert (
            cu_tiles.dim() == 1
            and int(cu_tiles.numel()) == batch_size + 1
            and cu_tiles.dtype == torch.int32
            and cu_tiles.device == out.device
            and cu_tiles.is_contiguous()
        ), f"cu_tiles contract mismatch: shape={cu_tiles.shape} dtype={cu_tiles.dtype}"
        grid: Tuple[int, ...] = (total_tiles_int,)
        use_flat_grid = True
        cu_tiles_arg: Optional[torch.Tensor] = cu_tiles
    else:
        num_workers = _select_dequant_num_workers(
            batch_size, max_tokens_hint=max_gather_len
        )
        grid = (batch_size, num_workers)
        use_flat_grid = False
        cu_tiles_arg = None

    _dequantize_and_gather_k_slots_kernel[grid](
        out,
        out.stride(0),
        out.stride(1),
        k_cache,
        slot_mapping,
        slot_mapping.stride(0),
        int(offset),
        gather_lens,
        max_gather_len,
        cu_tiles_ptr=cu_tiles_arg,
        cu_dst_offsets_ptr=cu_dst_offsets,
        num_reqs=batch_size,
        fp8_dim=NOPE_DIM,
        bf16_dim=ROPE_DIM,
        scale_dim=SCALE_BYTES_PER_TOKEN,
        quant_block=TILE_SIZE,
        cache_block_size=int(cache_block_size),
        token_data_size=TOKEN_DATA_SIZE,
        block_stride=int(block_stride),
        num_cache_blocks=int(num_cache_blocks),
        fp8_max=FP8_MAX,
        n_quant_blocks=NOPE_TILES,
        fp8_dim_pow2=FP8_DIM_POW2,
        n_scale_groups=N_SCALE_GROUPS,
        TILE_T=TILE_T,
        TRAP_INVALID_KV_ACCESS=True,
        CP_RANK_MAJOR_RAW=bool(cp_rank_major_raw),
        # Rank slices wider than the 512-wide tile columns allow the exact
        # carry-based rank translation (at most one boundary crossing per
        # element); narrower slices fall back to per-element division.
        SLICE_GE_TILE_COLS=int(block_stride) >= _MIN_SINGLE_CARRY_SLICE_BYTES,
        USE_FLAT_GRID=use_flat_grid,
        HAS_CU_DST_OFFSETS=cu_dst_offsets is not None,
        num_warps=4,
    )


def _launch_dequantize_and_gather_k_slots_cp_rank_major_unchecked(
    out: torch.Tensor,
    gathered: torch.Tensor,
    compact_slots: torch.Tensor,
    gather_lens: torch.Tensor,
    offset: int,
    *,
    full_entries_per_block: int,
    num_unique_blocks: int,
    cu_tiles: Optional[torch.Tensor] = None,
    total_tiles: Optional[int] = None,
    cu_dst_offsets: Optional[torch.Tensor] = None,
) -> None:
    """Launch after start/compaction established rank-major raw contracts."""
    _launch_dequantize_and_gather_k_slots_unchecked(
        out,
        gathered,
        compact_slots,
        gather_lens,
        offset,
        cache_block_size=int(full_entries_per_block),
        block_stride=int(gathered.shape[1]),
        num_cache_blocks=int(num_unique_blocks),
        cp_rank_major_raw=True,
        cu_tiles=cu_tiles,
        total_tiles=total_tiles,
        cu_dst_offsets=cu_dst_offsets,
    )


def dequantize_and_gather_k_cache_slots(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    gather_lens: Optional[torch.Tensor],
    offset: int,
    *,
    cu_tiles: Optional[torch.Tensor] = None,
    total_tiles: Optional[int] = None,
    cu_dst_offsets: Optional[torch.Tensor] = None,
) -> None:
    """Dequantize + gather FP8 K cache using flat global slot ids.

    ``slot_mapping`` is request-major ``[B, max_gather_len]``. Entries with
    ``-1`` zero-fill the corresponding output row and skip all cache reads.
    This is the SWA prefill path for layouts where block-table row coverage
    differs from the per-block ring entry count.
    """
    has_cu_dst_offsets = cu_dst_offsets is not None
    assert (
        slot_mapping.dim() == 2
    ), f"slot_mapping must be [num_reqs, max_gather_len], got {slot_mapping.shape}"
    if has_cu_dst_offsets:
        assert (
            out.dim() == 2 and out.shape[-1] == HEAD_DIM and out.dtype == torch.bfloat16
        )
        assert (
            out.stride(1) == 1
        ), f"2D out must have contiguous last dim; got stride={out.stride()}"
        assert (
            cu_dst_offsets.dim() == 1
            and int(cu_dst_offsets.numel()) >= int(slot_mapping.shape[0])
            and cu_dst_offsets.dtype == torch.int32
            and cu_dst_offsets.device == out.device
            and cu_dst_offsets.is_contiguous()
        ), f"cu_dst_offsets contract mismatch: shape={cu_dst_offsets.shape}"
    else:
        assert (
            out.dim() == 3 and out.shape[-1] == HEAD_DIM and out.dtype == torch.bfloat16
        )
        assert int(slot_mapping.shape[0]) == int(out.shape[0]), (
            f"slot_mapping batch ({slot_mapping.shape[0]}) must match out batch "
            f"({out.shape[0]})"
        )
    assert (
        k_cache.dim() == 3
        and k_cache.shape[-1] == ENTRY_BYTES
        and k_cache.dtype == torch.uint8
    )
    assert k_cache.stride(2) == 1 and k_cache.stride(1) == ENTRY_BYTES, (
        "k_cache must be packed within each token (stride[1]=584, stride[2]=1); "
        f"got stride={k_cache.stride()}"
    )
    if gather_lens is not None:
        assert int(gather_lens.shape[0]) == int(slot_mapping.shape[0]), (
            f"gather_lens batch ({gather_lens.shape[0]}) must match slot_mapping "
            f"batch ({slot_mapping.shape[0]})"
        )

    max_gather_len = int(slot_mapping.shape[1])
    if max_gather_len == 0 or int(slot_mapping.shape[0]) == 0:
        return

    slots_i64 = (
        slot_mapping
        if (
            slot_mapping.device == out.device
            and slot_mapping.dtype == torch.int64
            and slot_mapping.is_contiguous()
        )
        else slot_mapping.to(device=out.device, dtype=torch.int64).contiguous()
    )
    if gather_lens is None:
        gather_lens_i32 = None
    elif (
        gather_lens.device == out.device
        and gather_lens.dtype == torch.int32
        and gather_lens.is_contiguous()
    ):
        gather_lens_i32 = gather_lens
    else:
        gather_lens_i32 = gather_lens.to(
            device=out.device, dtype=torch.int32
        ).contiguous()

    _launch_dequantize_and_gather_k_slots_unchecked(
        out,
        k_cache,
        slots_i64,
        gather_lens_i32,
        offset,
        cache_block_size=int(k_cache.shape[1]),
        block_stride=int(k_cache.stride(0)),
        num_cache_blocks=int(k_cache.shape[0]),
        cp_rank_major_raw=False,
        cu_tiles=cu_tiles,
        total_tiles=total_tiles,
        cu_dst_offsets=cu_dst_offsets,
    )


def cp_swa_direct_dequant_scatter_enabled() -> bool:
    """Whether CP byte-sliced SWA dequant writes directly to the workspace."""
    return os.environ.get("DSV4_CP_SWA_DIRECT_DEQUANT_SCATTER", "1") != "0"


def try_dequantize_and_gather_k_cache_slots_to_workspace(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    gather_lens: torch.Tensor,
    offset: int,
    *,
    cu_tiles: Optional[torch.Tensor] = None,
    total_tiles: Optional[int] = None,
    cu_dst_offsets: Optional[torch.Tensor] = None,
) -> bool:
    """Dequantize compact slots directly into their final workspace rows.

    Unsupported layouts return ``False`` without modifying ``out``. The
    caller can then retain the previous temporary-dequant + copy fallback.
    """
    if not cp_swa_direct_dequant_scatter_enabled():
        return False
    if not direct_triton_fast_path_supported(out.device):
        return False

    has_cu_dst_offsets = cu_dst_offsets is not None
    tensors = (out, k_cache, slot_mapping, gather_lens)
    if (
        any(not tensor.is_cuda for tensor in tensors)
        or any(tensor.device != out.device for tensor in tensors[1:])
        or out.dtype != torch.bfloat16
        or int(out.shape[-1]) != HEAD_DIM
        or k_cache.dim() != 3
        or k_cache.dtype != torch.uint8
        or int(k_cache.shape[-1]) != ENTRY_BYTES
        or int(k_cache.stride(2)) != 1
        or int(k_cache.stride(1)) != ENTRY_BYTES
        or int(k_cache.shape[0]) <= 0
        or int(k_cache.shape[1]) <= 0
        or slot_mapping.dim() != 2
        or slot_mapping.dtype != torch.int64
        or not slot_mapping.is_contiguous()
        or gather_lens.dim() != 1
        or gather_lens.dtype != torch.int32
        or not gather_lens.is_contiguous()
        or int(gather_lens.shape[0]) != int(slot_mapping.shape[0])
        or int(offset) < 0
    ):
        return False
    if has_cu_dst_offsets:
        if (
            not cu_dst_offsets.is_cuda
            or cu_dst_offsets.device != out.device
            or out.dim() != 2
            or int(out.stride(1)) != 1
            or cu_dst_offsets.dim() != 1
            or cu_dst_offsets.dtype != torch.int32
            or not cu_dst_offsets.is_contiguous()
            or int(cu_dst_offsets.numel()) < int(slot_mapping.shape[0])
        ):
            return False
    else:
        if (
            out.dim() != 3
            or int(out.stride(2)) != 1
            or int(slot_mapping.shape[0]) != int(out.shape[0])
            or int(offset) + int(slot_mapping.shape[1]) > int(out.shape[1])
        ):
            return False

    batch_size = int(slot_mapping.shape[0])
    max_gather_len = int(slot_mapping.shape[1])
    if batch_size == 0 or max_gather_len == 0:
        return True

    _launch_dequantize_and_gather_k_slots_unchecked(
        out,
        k_cache,
        slot_mapping,
        gather_lens,
        offset,
        cache_block_size=int(k_cache.shape[1]),
        block_stride=int(k_cache.stride(0)),
        num_cache_blocks=int(k_cache.shape[0]),
        cp_rank_major_raw=False,
        cu_tiles=cu_tiles,
        total_tiles=total_tiles,
        cu_dst_offsets=cu_dst_offsets,
    )
    return True


@triton.jit(
    do_not_specialize=[
        "out_stride0",
        "out_stride1",
        "offset",
        "max_blocks_per_seq",
        "block_stride",
    ]
)
def _gather_k_cache_packed_kernel(
    out_ptr,
    out_stride0,
    out_stride1,
    k_cache_ptr,
    seq_lens_ptr,
    block_table_ptr,
    offset,
    gather_lens_ptr,
    max_blocks_per_seq,
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,
    scale_dim: tl.constexpr,
    block_stride,
):
    """Gather paged FP8 cache into true per-token packed slots.

    The physical block layout stores all token data first and all per-token
    scales at the end of the block. The output layout is compact per token:
    ``[448 fp8 NoPE | 128 bf16 RoPE | 8 UE8M0 scale]``.
    """
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    if gather_lens_ptr is not None:  # noqa: SIM108
        gather_len = tl.load(gather_lens_ptr + batch_idx)
    else:
        gather_len = seq_len
    start_pos = seq_len - gather_len

    data_offsets = tl.arange(0, 1024)
    data_mask = data_offsets < token_data_size
    scale_offsets = tl.arange(0, 8)
    scale_mask = scale_offsets < scale_dim

    for i in range(worker_id, gather_len, num_workers):
        pos = start_pos + i
        block_in_seq = pos // cache_block_size
        pos_in_block = pos % cache_block_size

        block_table_row_ptr = block_table_ptr + batch_idx.to(
            tl.int64
        ) * max_blocks_per_seq.to(tl.int64)
        physical_block_idx = tl.load(block_table_row_ptr + block_in_seq)
        tl.device_assert(physical_block_idx >= 0, "block_table contains -1")

        cache_block_ptr = k_cache_ptr + physical_block_idx.to(
            tl.int64
        ) * block_stride.to(tl.int64)
        token_data_ptr = cache_block_ptr + pos_in_block * token_data_size
        token_scale_ptr = (
            cache_block_ptr
            + cache_block_size * token_data_size
            + pos_in_block * scale_dim
        )
        output_row_ptr = (
            out_ptr
            + batch_idx.to(tl.int64) * out_stride0.to(tl.int64)
            + (offset.to(tl.int64) + i.to(tl.int64)) * out_stride1.to(tl.int64)
        )

        data = tl.load(token_data_ptr + data_offsets, mask=data_mask, other=0)
        tl.store(output_row_ptr + data_offsets, data, mask=data_mask)
        scales = tl.load(token_scale_ptr + scale_offsets, mask=scale_mask, other=0)
        tl.store(
            output_row_ptr + token_data_size + scale_offsets,
            scales,
            mask=scale_mask,
        )


def gather_k_cache_packed(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: Optional[torch.Tensor],
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    """Gather FP8 K cache into compact per-token packed slots.

    Args mirror :func:`dequantize_and_gather_k_cache`, but ``out`` is
    ``[B, max_tokens, 584] uint8`` rather than BF16.
    """
    assert out.dim() == 3 and out.shape[-1] == ENTRY_BYTES and out.dtype == torch.uint8
    assert out.stride(2) == 1, f"out must have packed byte stride; got {out.stride()}"
    assert (
        k_cache.dim() == 3
        and k_cache.shape[-1] == ENTRY_BYTES
        and k_cache.dtype == torch.uint8
    )
    assert k_cache.stride(2) == 1 and k_cache.stride(1) == ENTRY_BYTES, (
        "k_cache must be packed within each token (stride[1]=584, stride[2]=1); "
        f"got stride={k_cache.stride()}"
    )
    assert (
        block_table.dtype == torch.int32
        and block_table.is_contiguous()
        and block_table.device == k_cache.device
    ), (
        "block_table must be int32, contiguous, and on the same device as k_cache; "
        f"got dtype={block_table.dtype} contig={block_table.is_contiguous()} "
        f"dev={block_table.device} (k_cache dev={k_cache.device})"
    )

    NUM_WORKERS = 128
    _gather_k_cache_packed_kernel[(seq_lens.shape[0], NUM_WORKERS)](
        out,
        out.stride(0),
        out.stride(1),
        k_cache,
        seq_lens,
        block_table,
        offset,
        gather_lens,
        max_blocks_per_seq=block_table.shape[-1],
        cache_block_size=block_size,
        token_data_size=TOKEN_DATA_SIZE,
        scale_dim=SCALE_BYTES_PER_TOKEN,
        block_stride=k_cache.stride(0),
    )


def cp_direct_flat_pack_enabled() -> bool:
    """Whether CP pool reads use the direct rank-local flat gather."""
    return os.environ.get("DSV4_CP_DIRECT_FLAT_PACK", "1") != "0"


@triton.jit(
    do_not_specialize=[
        "batch_size",
        "total_local_tokens",
        "block_table_stride_b",
        "max_blocks_per_request",
        "cache_block_size",
        "cache_stride_b",
        "num_cache_blocks",
    ]
)
def _gather_k_cache_packed_to_flat_kernel(
    out_ptr,
    k_cache_ptr,
    block_table_ptr,
    padded_lens_ptr,
    actual_lens_ptr,
    batch_size,
    total_local_tokens,
    block_table_stride_b,
    max_blocks_per_request,
    cache_block_size,
    cache_stride_b,
    num_cache_blocks,
    token_data_size: tl.constexpr,
    scale_dim: tl.constexpr,
    entry_bytes: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
    TRAP_INVALID_KV_ACCESS: tl.constexpr,
):
    """Gather grouped cache bytes directly into padded rank-local flat order."""
    token_idx = tl.program_id(0).to(tl.int64)
    if token_idx >= total_local_tokens:
        return

    batch_offsets = tl.arange(0, BATCH_BLOCK)
    batch_mask = batch_offsets < batch_size
    padded_lens = tl.load(padded_lens_ptr + batch_offsets, mask=batch_mask, other=0).to(
        tl.int64
    )
    padded_ends = tl.cumsum(padded_lens, axis=0)
    batch_idx = tl.sum((token_idx >= padded_ends).to(tl.int32), axis=0)
    padded_start = tl.sum(
        tl.where(batch_offsets == batch_idx, padded_ends - padded_lens, 0),
        axis=0,
    )
    token_in_request = token_idx - padded_start
    actual_len = tl.sum(
        tl.where(
            batch_offsets == batch_idx,
            tl.load(actual_lens_ptr + batch_offsets, mask=batch_mask, other=0).to(
                tl.int64
            ),
            0,
        ),
        axis=0,
    )

    is_actual = token_in_request < actual_len
    logical_block_idx = token_in_request // cache_block_size
    valid_table_lookup = is_actual & (logical_block_idx < max_blocks_per_request)
    cache_block_idx = tl.load(
        block_table_ptr
        + batch_idx.to(tl.int64) * block_table_stride_b
        + logical_block_idx,
        mask=valid_table_lookup,
        other=-1,
    ).to(tl.int64)
    valid_cache_block = (
        valid_table_lookup
        & (cache_block_idx >= 0)
        & (cache_block_idx < num_cache_blocks)
    )
    if valid_table_lookup & ~valid_cache_block:
        _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)

    safe_cache_block_idx = tl.where(valid_cache_block, cache_block_idx, 0)
    token_in_block = token_in_request % cache_block_size
    cache_block_base = k_cache_ptr + safe_cache_block_idx * cache_stride_b
    output_row = out_ptr + token_idx * entry_bytes

    data_offsets = tl.arange(0, 1024)
    data_mask = data_offsets < token_data_size
    cache_data = tl.load(
        cache_block_base + token_in_block * token_data_size + data_offsets,
        mask=valid_cache_block & data_mask,
        other=0,
    )
    tl.store(output_row + data_offsets, cache_data, mask=data_mask)

    scale_offsets = tl.arange(0, 8)
    scale_mask = scale_offsets < scale_dim
    cache_scales = tl.load(
        cache_block_base
        + cache_block_size * token_data_size
        + token_in_block * scale_dim
        + scale_offsets,
        mask=valid_cache_block & scale_mask,
        other=0,
    )
    tl.store(
        output_row + token_data_size + scale_offsets,
        cache_scales,
        mask=scale_mask,
    )


def try_gather_k_cache_packed_to_flat(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    per_req_padded_lens: torch.Tensor,
    per_req_actual_lens: torch.Tensor,
    *,
    block_size: int,
    has_actual_tokens: bool,
) -> bool:
    """Fuse paged gather, zero padding, and flattening for CP all-gather.

    Unsupported layouts return ``False`` without modifying ``out`` so callers
    can use the previous padded-gather implementation as a fallback.
    """
    if not cp_direct_flat_pack_enabled():
        return False
    if not direct_triton_fast_path_supported(out.device):
        return False

    batch_size = int(per_req_padded_lens.numel())
    total_local_tokens = int(out.shape[0]) if out.dim() == 2 else -1
    block_size = int(block_size)
    tensors = (
        out,
        k_cache,
        block_table,
        per_req_padded_lens,
        per_req_actual_lens,
    )
    if (
        batch_size <= 0
        or batch_size > 64
        or total_local_tokens < 0
        or any(not tensor.is_cuda for tensor in tensors)
        or any(tensor.device != out.device for tensor in tensors[1:])
        or out.dtype != torch.uint8
        or tuple(out.shape) != (total_local_tokens, ENTRY_BYTES)
        or not out.is_contiguous()
        or k_cache.dim() != 3
        or k_cache.dtype != torch.uint8
        or int(k_cache.shape[-1]) != ENTRY_BYTES
        or int(k_cache.stride(2)) != 1
        or int(k_cache.stride(1)) != ENTRY_BYTES
        or int(k_cache.shape[1]) != block_size
        or block_table.dim() != 2
        or block_table.dtype != torch.int32
        or int(block_table.shape[0]) < batch_size
        or int(block_table.stride(1)) != 1
        or per_req_padded_lens.dim() != 1
        or per_req_padded_lens.dtype != torch.int32
        or not per_req_padded_lens.is_contiguous()
        or per_req_actual_lens.dim() != 1
        or int(per_req_actual_lens.numel()) != batch_size
        or per_req_actual_lens.dtype != torch.int32
        or not per_req_actual_lens.is_contiguous()
    ):
        return False
    if total_local_tokens == 0:
        return True
    if (
        int(k_cache.shape[0]) <= 0
        or int(k_cache.shape[1]) <= 0
        or (bool(has_actual_tokens) and int(block_table.shape[1]) <= 0)
    ):
        return False

    batch_block = triton.next_power_of_2(batch_size)
    _gather_k_cache_packed_to_flat_kernel[(total_local_tokens,)](
        out,
        k_cache,
        block_table,
        per_req_padded_lens,
        per_req_actual_lens,
        batch_size,
        total_local_tokens,
        int(block_table.stride(0)),
        int(block_table.shape[1]),
        block_size,
        int(k_cache.stride(0)),
        int(k_cache.shape[0]),
        token_data_size=TOKEN_DATA_SIZE,
        scale_dim=SCALE_BYTES_PER_TOKEN,
        entry_bytes=ENTRY_BYTES,
        BATCH_BLOCK=batch_block,
        TRAP_INVALID_KV_ACCESS=True,
        num_warps=4,
    )
    return True


@triton.jit(do_not_specialize=["out_stride0", "n_tokens"])
def _dequantize_packed_k_cache_flat_kernel(
    out_ptr,
    out_stride0,
    packed_ptr,
    n_tokens,
    fp8_dim: tl.constexpr,
    fp8_dim_pow2: tl.constexpr,
    bf16_dim: tl.constexpr,
    quant_block: tl.constexpr,
    token_data_size: tl.constexpr,
    entry_bytes: tl.constexpr,
    n_quant_blocks: tl.constexpr,
    n_scale_groups: tl.constexpr,
    TILE_T_FLAT: tl.constexpr,
):
    tile_idx = tl.program_id(0).to(tl.int64)
    t_offs = tile_idx * TILE_T_FLAT + tl.arange(0, TILE_T_FLAT)
    row_valid = t_offs < n_tokens

    token_off = t_offs * entry_bytes
    output_row_off = t_offs * out_stride0.to(tl.int64)

    fp8_cols = tl.arange(0, fp8_dim_pow2)
    fp8_mask = row_valid[:, None] & (fp8_cols < fp8_dim)[None, :]
    x_uint8 = tl.load(
        packed_ptr + token_off[:, None] + fp8_cols[None, :],
        mask=fp8_mask,
        other=0,
    )
    x_float = x_uint8.to(tl.float8e4nv, bitcast=True).to(tl.float32)

    scale_cols = tl.arange(0, n_scale_groups)
    scale_mask = row_valid[:, None] & (scale_cols < n_quant_blocks)[None, :]
    encoded_scale = tl.load(
        packed_ptr + token_off[:, None] + token_data_size + scale_cols[None, :],
        mask=scale_mask,
        other=0,
    )
    scale = tl.exp2(encoded_scale.to(tl.float32) - 127.0)
    x_groups = tl.reshape(x_float, (TILE_T_FLAT, n_scale_groups, quant_block))
    x_dequant = x_groups * scale[:, :, None]
    x_out = tl.reshape(x_dequant, (TILE_T_FLAT, fp8_dim_pow2)).to(tl.bfloat16)
    tl.store(
        out_ptr + output_row_off[:, None] + fp8_cols[None, :], x_out, mask=fp8_mask
    )

    bf16_output_offset = fp8_dim
    if bf16_dim == 64:
        rope_offsets = tl.arange(0, 64)
        rope_mask = row_valid[:, None]
        rope_ptrs = (
            packed_ptr + token_off[:, None] + fp8_dim + rope_offsets[None, :] * 2
        ).to(tl.pointer_type(tl.bfloat16))
        bf16_vals = tl.load(rope_ptrs, mask=rope_mask, other=0)
        tl.store(
            out_ptr
            + output_row_off[:, None]
            + bf16_output_offset
            + rope_offsets[None, :],
            bf16_vals,
            mask=rope_mask,
        )
    else:
        for j in tl.static_range(bf16_dim // 16):
            chunk_offsets = j * 16 + tl.arange(0, 16)
            chunk_mask = row_valid[:, None]
            chunk_ptrs = (
                packed_ptr + token_off[:, None] + fp8_dim + chunk_offsets[None, :] * 2
            ).to(tl.pointer_type(tl.bfloat16))
            bf16_vals = tl.load(chunk_ptrs, mask=chunk_mask, other=0)
            tl.store(
                out_ptr
                + output_row_off[:, None]
                + bf16_output_offset
                + chunk_offsets[None, :],
                bf16_vals,
                mask=chunk_mask,
            )


@triton.jit(
    do_not_specialize=[
        "out_stride0",
        "out_stride1",
        "out_batch_size",
        "out_rows_per_batch",
        "gathered_rows",
        "batch_size",
        "n_tokens",
        "offset",
    ]
)
def _restore_dequantize_scatter_packed_k_cache_flat_kernel(
    out_ptr,
    out_stride0,
    out_stride1,
    out_batch_size,
    out_rows_per_batch,
    gathered_ptr,
    gathered_rows,
    restore_indices_ptr,
    seq_lens_ptr,
    cu_dst_offsets_ptr,
    batch_size,
    n_tokens,
    offset,
    fp8_dim: tl.constexpr,
    fp8_dim_pow2: tl.constexpr,
    bf16_dim: tl.constexpr,
    quant_block: tl.constexpr,
    token_data_size: tl.constexpr,
    entry_bytes: tl.constexpr,
    n_quant_blocks: tl.constexpr,
    n_scale_groups: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
    HAS_CU_DST_OFFSETS: tl.constexpr,
):
    """Restore, dequantize, and scatter one packed row per program.

    Every metadata-derived address is independently masked. The host validates
    the expected total before launch, while these guards keep stale or divergent
    device metadata from turning into gathered/output OOB accesses.
    """
    token_idx = tl.program_id(0).to(tl.int64)
    token_valid = token_idx < n_tokens

    batch_offsets = tl.arange(0, BATCH_BLOCK)
    batch_mask = batch_offsets < batch_size
    seq_lens = tl.load(seq_lens_ptr + batch_offsets, mask=batch_mask, other=0).to(
        tl.int64
    )
    seq_ends = tl.cumsum(seq_lens, axis=0)
    batch_idx = tl.sum((token_idx >= seq_ends).to(tl.int32), axis=0)
    valid_batch = token_valid & (batch_idx >= 0) & (batch_idx < batch_size)
    selected_seq_len = tl.sum(tl.where(batch_offsets == batch_idx, seq_lens, 0), axis=0)
    seq_start = tl.sum(
        tl.where(batch_offsets == batch_idx, seq_ends - seq_lens, 0), axis=0
    )
    token_in_request = token_idx - seq_start
    safe_batch_idx = tl.where(valid_batch, batch_idx, 0).to(tl.int64)
    if HAS_CU_DST_OFFSETS:
        dst_base = tl.load(
            cu_dst_offsets_ptr + safe_batch_idx, mask=valid_batch, other=0
        ).to(tl.int64)
        destination_row = dst_base + offset.to(tl.int64) + token_in_request
        valid_destination = (
            valid_batch
            & (selected_seq_len >= 0)
            & (token_in_request >= 0)
            & (token_in_request < selected_seq_len)
            & (destination_row >= 0)
            & (destination_row < out_rows_per_batch)
        )
        safe_destination_row = tl.where(valid_destination, destination_row, 0)
        output_row_ptr = out_ptr + safe_destination_row * out_stride0.to(tl.int64)
    else:
        destination_row = offset.to(tl.int64) + token_in_request
        valid_destination = (
            valid_batch
            & (batch_idx < out_batch_size)
            & (selected_seq_len >= 0)
            & (token_in_request >= 0)
            & (token_in_request < selected_seq_len)
            & (destination_row >= 0)
            & (destination_row < out_rows_per_batch)
        )
        safe_destination_row = tl.where(valid_destination, destination_row, 0)
        output_row_ptr = (
            out_ptr
            + safe_batch_idx * out_stride0.to(tl.int64)
            + safe_destination_row * out_stride1.to(tl.int64)
        )

    restored_idx = tl.load(
        restore_indices_ptr + token_idx, mask=token_valid, other=-1
    ).to(tl.int64)
    valid_restore = token_valid & (restored_idx >= 0) & (restored_idx < gathered_rows)
    valid_row = valid_destination & valid_restore
    safe_restored_idx = tl.where(valid_restore, restored_idx, 0)
    token_ptr = gathered_ptr + safe_restored_idx * entry_bytes

    fp8_cols = tl.arange(0, fp8_dim_pow2)
    fp8_mask = valid_row & (fp8_cols < fp8_dim)
    x_uint8 = tl.load(token_ptr + fp8_cols, mask=fp8_mask, other=0)
    x_float = x_uint8.to(tl.float8e4nv, bitcast=True).to(tl.float32)

    scale_cols = tl.arange(0, n_scale_groups)
    encoded_scale = tl.load(
        token_ptr + token_data_size + scale_cols,
        mask=valid_row & (scale_cols < n_quant_blocks),
        other=0,
    )
    scale = tl.exp2(encoded_scale.to(tl.float32) - 127.0)
    x_groups = tl.reshape(x_float, (n_scale_groups, quant_block))
    x_out = tl.reshape(x_groups * scale[:, None], (fp8_dim_pow2,)).to(tl.bfloat16)
    tl.store(output_row_ptr + fp8_cols, x_out, mask=fp8_mask)

    bf16_output_offset = fp8_dim
    bf16_cache_ptr = (token_ptr + fp8_dim).to(tl.pointer_type(tl.bfloat16))
    if bf16_dim == 64:
        rope_offsets = tl.arange(0, 64)
        bf16_vals = tl.load(bf16_cache_ptr + rope_offsets, mask=valid_row, other=0)
        tl.store(
            output_row_ptr + bf16_output_offset + rope_offsets,
            bf16_vals,
            mask=valid_row,
        )
    else:
        for j in tl.static_range(bf16_dim // 16):
            chunk_offsets = j * 16 + tl.arange(0, 16)
            bf16_vals = tl.load(bf16_cache_ptr + chunk_offsets, mask=valid_row, other=0)
            tl.store(
                output_row_ptr + bf16_output_offset + chunk_offsets,
                bf16_vals,
                mask=valid_row,
            )


def dequantize_packed_k_cache_flat(out: torch.Tensor, packed: torch.Tensor) -> None:
    """Dequant compact packed FP8 slots into flat BF16 rows.

    ``packed`` is ``[N, 584] uint8`` with per-token compact layout
    ``[576 data | 8 scale]``. ``out`` is ``[N, 512] bf16``.
    """
    assert out.dim() == 2 and out.shape[-1] == HEAD_DIM and out.dtype == torch.bfloat16
    assert out.stride(1) == 1, f"out must be contiguous by row; got {out.stride()}"
    assert (
        packed.dim() == 2
        and packed.shape[-1] == ENTRY_BYTES
        and packed.dtype == torch.uint8
    )
    assert packed.stride() == (
        ENTRY_BYTES,
        1,
    ), f"packed must be contiguous [N, {ENTRY_BYTES}]; got {packed.stride()}"
    if packed.numel() == 0:
        return
    n_tokens = int(packed.shape[0])
    _dequantize_packed_k_cache_flat_kernel[(triton.cdiv(n_tokens, TILE_T),)](
        out,
        int(out.stride(0)),
        packed,
        n_tokens,
        fp8_dim=NOPE_DIM,
        fp8_dim_pow2=FP8_DIM_POW2,
        bf16_dim=ROPE_DIM,
        quant_block=TILE_SIZE,
        token_data_size=TOKEN_DATA_SIZE,
        entry_bytes=ENTRY_BYTES,
        n_quant_blocks=NOPE_TILES,
        n_scale_groups=N_SCALE_GROUPS,
        TILE_T_FLAT=TILE_T,
        num_warps=4,
    )


def try_restore_dequantize_scatter_packed_k_cache_flat(
    out: torch.Tensor,
    gathered: torch.Tensor,
    restore_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    offset: int,
    *,
    seq_lens_total: int,
    cu_dst_offsets: Optional[torch.Tensor] = None,
) -> bool:
    """Fuse packed-row restore, BF16 dequantization, and workspace scatter.

    ``seq_lens_total`` is the host-side total computed while request metadata is
    built. It must match the restore row count; this validates the launch without
    synchronizing ``seq_lens`` back to the host on every layer. Unsupported
    layouts return ``False`` without modifying ``out`` so callers can use the
    exact existing PyTorch fallback.
    """
    batch_size = int(seq_lens.numel())
    n_tokens = int(restore_indices.numel())
    seq_lens_total = int(seq_lens_total)
    if seq_lens_total < 0:
        raise ValueError(f"seq_lens_total must be non-negative, got {seq_lens_total}")
    if n_tokens != seq_lens_total:
        raise ValueError(
            f"restore row count ({n_tokens}) != host seq_lens total "
            f"({seq_lens_total})"
        )
    if batch_size == 0:
        return n_tokens == 0
    if not direct_triton_fast_path_supported(out.device):
        return False
    has_cu_dst_offsets = cu_dst_offsets is not None
    tensors = (out, gathered, restore_indices, seq_lens)
    if (
        batch_size > 64
        or any(not tensor.is_cuda for tensor in tensors)
        or any(tensor.device != out.device for tensor in tensors[1:])
        or out.dtype != torch.bfloat16
        or int(out.shape[-1]) != HEAD_DIM
        or gathered.dim() != 2
        or gathered.dtype != torch.uint8
        or int(gathered.shape[-1]) != ENTRY_BYTES
        or tuple(gathered.stride()) != (ENTRY_BYTES, 1)
        or restore_indices.dim() != 1
        or restore_indices.dtype != torch.int64
        or not restore_indices.is_contiguous()
        or seq_lens.dim() != 1
        or seq_lens.dtype != torch.int32
        or not seq_lens.is_contiguous()
        or int(offset) < 0
    ):
        return False
    if has_cu_dst_offsets:
        if (
            not cu_dst_offsets.is_cuda
            or cu_dst_offsets.device != out.device
            or out.dim() != 2
            or int(out.stride(1)) != 1
            or cu_dst_offsets.dim() != 1
            or cu_dst_offsets.dtype != torch.int32
            or not cu_dst_offsets.is_contiguous()
            or int(cu_dst_offsets.numel()) < batch_size
            or n_tokens > int(out.shape[0])
        ):
            return False
    else:
        if (
            out.dim() != 3
            or int(out.shape[0]) < batch_size
            or int(out.stride(2)) != 1
            or int(offset) >= int(out.shape[1])
            or n_tokens > batch_size * (int(out.shape[1]) - int(offset))
        ):
            return False
    if n_tokens == 0:
        return True

    batch_block = triton.next_power_of_2(batch_size)
    _restore_dequantize_scatter_packed_k_cache_flat_kernel[(n_tokens,)](
        out,
        out.stride(0),
        out.stride(1),
        1 if has_cu_dst_offsets else int(out.shape[0]),
        int(out.shape[0]) if has_cu_dst_offsets else int(out.shape[1]),
        gathered,
        int(gathered.shape[0]),
        restore_indices,
        seq_lens,
        cu_dst_offsets,
        batch_size,
        n_tokens,
        int(offset),
        fp8_dim=NOPE_DIM,
        fp8_dim_pow2=FP8_DIM_POW2,
        bf16_dim=ROPE_DIM,
        quant_block=TILE_SIZE,
        token_data_size=TOKEN_DATA_SIZE,
        entry_bytes=ENTRY_BYTES,
        n_quant_blocks=NOPE_TILES,
        n_scale_groups=N_SCALE_GROUPS,
        BATCH_BLOCK=batch_block,
        HAS_CU_DST_OFFSETS=has_cu_dst_offsets,
        num_warps=4,
    )
    return True


def dequantize_swa_window_to_bf16(
    kv_cache_packed: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    win: int,
    block_size: int,
) -> torch.Tensor:
    """High-level convenience wrapper for the prefill SWA-only site.

    Reconstructs a ``[B, win, 512]`` BF16 tensor from the FP8 SWA pool, with
    the same column convention the BF16 prefill path expects (the
    ``_get_window_topk_idxs`` indices address columns ``[0, win)`` of this
    tensor in ring order — same as the live SWA pool).

    Args:
      kv_cache_packed: ``[num_blocks, block_size, 584]`` uint8.
      block_table:     ``[B, max_blocks_per_seq]`` int — per-request block id.
      seq_lens:        ``[B]`` int — number of valid tokens already in the
                       SWA ring for each request (i.e. ``min(start_pos +
                       seqlen, win)``).
      win:             logical SWA window length.
      block_size:      tokens per packed block (must match kv_cache_packed.shape[1]).

    Returns: ``[B, win, 512]`` bf16. Untouched (out-of-window) positions
      stay zero — matches the BF16 register_buffer's pre-write state.
    """
    assert kv_cache_packed.dim() == 3 and kv_cache_packed.shape[1] == block_size
    B = int(block_table.shape[0])
    out = torch.zeros(
        (B, win, HEAD_DIM), dtype=torch.bfloat16, device=kv_cache_packed.device
    )
    # Gather seq_lens[r] tokens (the entire SWA window for request r) starting
    # at ring position 0. The kernel addresses kv_cache_packed via block_table
    # so the physical layout is fully driven by the framework's allocation.
    seq_i32 = seq_lens.to(dtype=torch.int32, device=kv_cache_packed.device)
    if not seq_i32.is_contiguous():
        seq_i32 = seq_i32.contiguous()
    bt_i32 = block_table.to(dtype=torch.int32, device=kv_cache_packed.device)
    if not bt_i32.is_contiguous():
        bt_i32 = bt_i32.contiguous()
    dequantize_and_gather_k_cache(
        out=out,
        k_cache=kv_cache_packed,
        seq_lens=seq_i32,
        gather_lens=None,
        block_table=bt_i32,
        block_size=block_size,
        offset=0,
    )
    return out


# --------------------------------------------------------------------------
# Per-slot fancy-index dequant (striped-layout aware).
#
# Used by dual-pool decode paths that already have flat slot indices
# (`block_id * block_size + pos_in_block`) and need a small N of tokens
# dequant'd. Cannot use `pool[slot, :]` fancy-index because the canonical
# 584B layout is per-block striped (data in the first `bs*576` bytes,
# scales in the trailing `bs*8` bytes) — bytes [0:584] of one pool row
# do NOT correspond to one token's data + scale.
# --------------------------------------------------------------------------


@triton.jit(
    do_not_specialize=[
        "out_stride0",
        "pool_block_stride",
        "num_cache_blocks",
    ]
)
def _dequantize_slots_kernel(
    out_ptr,  # [N, 512] bf16
    out_stride0,
    pool_ptr,  # [num_blocks, block_size, 584] uint8
    pool_block_stride,  # bytes per block (may be padded > bs*584)
    slot_indices_ptr,  # [N] int64
    block_size: tl.constexpr,
    fp8_dim: tl.constexpr,  # 448
    bf16_dim: tl.constexpr,  # 64
    scale_dim: tl.constexpr,  # 8
    quant_block: tl.constexpr,  # 64
    token_data_size: tl.constexpr,  # 576
    num_cache_blocks,
    n_quant_blocks: tl.constexpr,  # 7
    TRAP_INVALID_KV_ACCESS: tl.constexpr,
):
    """One program per output slot. Sentinel slots (slot < 0) write zeros."""
    pid = tl.program_id(0)
    slot = tl.load(slot_indices_ptr + pid)

    output_row_ptr = out_ptr + pid.to(tl.int64) * out_stride0.to(tl.int64)

    if slot < 0:
        # Sentinel: zero the row.
        for j in tl.static_range((fp8_dim + bf16_dim) // 16):
            chunk_offsets = j * 16 + tl.arange(0, 16)
            zero_vec = tl.zeros((16,), dtype=tl.bfloat16)
            tl.store(output_row_ptr + chunk_offsets, zero_vec)
        return

    blk = (slot // block_size).to(tl.int64)
    off = slot % block_size
    if blk < 0:
        _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)
    if blk >= num_cache_blocks:
        _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)

    cache_block_ptr = pool_ptr + blk * pool_block_stride.to(tl.int64)
    token_data_ptr = cache_block_ptr + off * token_data_size
    token_scale_ptr = cache_block_ptr + block_size * token_data_size + off * scale_dim
    token_fp8_ptr = token_data_ptr
    token_bf16_ptr = token_data_ptr + fp8_dim

    # NoPE: 7 tiles × 64 elements, UE8M0 scale per tile.
    for qblock_idx in tl.static_range(n_quant_blocks):
        qblock_start = qblock_idx * quant_block
        if qblock_start < fp8_dim:
            offsets = qblock_start + tl.arange(0, quant_block)
            mask = offsets < fp8_dim
            x_uint8 = tl.load(token_fp8_ptr + offsets, mask=mask, other=0)
            x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
            x_float = x_fp8.to(tl.float32)
            encoded_scale = tl.load(token_scale_ptr + qblock_idx)
            exponent = encoded_scale.to(tl.float32) - 127.0
            scale = tl.exp2(exponent)
            x_dequant = x_float * scale
            tl.store(output_row_ptr + offsets, x_dequant.to(tl.bfloat16), mask=mask)

    # RoPE: bf16 passthrough.
    bf16_output_offset = fp8_dim
    bf16_cache_ptr = token_bf16_ptr.to(tl.pointer_type(tl.bfloat16))
    for j in tl.static_range(bf16_dim // 16):
        chunk_offsets = j * 16 + tl.arange(0, 16)
        bf16_vals = tl.load(bf16_cache_ptr + chunk_offsets)
        tl.store(output_row_ptr + bf16_output_offset + chunk_offsets, bf16_vals)


def dequantize_slots_to_bf16(
    pool_3d: torch.Tensor,  # [num_blocks, block_size, 584] uint8
    slot_indices: torch.Tensor,  # [N] int (flat slot ids; <0 = sentinel)
) -> torch.Tensor:
    """Per-slot fancy-index dequant of canonical 584B FP8 KV → [N, 512] bf16.

    Replacement for the deleted ``_kv_fp8_dequant_canonical.unpack_kv_fp8_canonical``.
    Striped-layout aware: addresses data and scale regions independently.
    Sentinel rows (slot < 0) are zero-filled.
    """
    assert (
        pool_3d.dim() == 3
        and pool_3d.shape[-1] == ENTRY_BYTES
        and pool_3d.dtype == torch.uint8
    ), (
        f"pool_3d must be [num_blocks, block_size, {ENTRY_BYTES}] uint8; "
        f"got shape={tuple(pool_3d.shape)} dtype={pool_3d.dtype}"
    )
    assert pool_3d.stride(2) == 1 and pool_3d.stride(1) == ENTRY_BYTES, (
        "pool_3d must be packed within each token (stride[1]=584, stride[2]=1); "
        f"got stride={pool_3d.stride()}"
    )
    block_size = int(pool_3d.shape[1])
    N = int(slot_indices.numel())
    out = torch.empty((N, HEAD_DIM), dtype=torch.bfloat16, device=pool_3d.device)
    if N == 0:
        return out
    slots_i64 = slot_indices.reshape(-1).to(torch.int64).contiguous()
    _dequantize_slots_kernel[(N,)](
        out,
        out.stride(0),
        pool_3d,
        pool_block_stride=int(pool_3d.stride(0)),
        slot_indices_ptr=slots_i64,
        block_size=block_size,
        fp8_dim=NOPE_DIM,
        bf16_dim=ROPE_DIM,
        scale_dim=SCALE_BYTES_PER_TOKEN,
        quant_block=TILE_SIZE,
        token_data_size=TOKEN_DATA_SIZE,
        num_cache_blocks=int(pool_3d.shape[0]),
        n_quant_blocks=NOPE_TILES,
        TRAP_INVALID_KV_ACCESS=True,
    )
    return out


def _device_gather_lens(
    gather_lens: Optional[torch.Tensor],
    *,
    batch_size: int,
    max_gather_len: int,
    device: torch.device,
) -> torch.Tensor:
    if gather_lens is None:
        return torch.full(
            (batch_size,),
            max_gather_len,
            dtype=torch.int32,
            device=device,
        )
    if (
        gather_lens.dim() == 1
        and gather_lens.dtype == torch.int32
        and gather_lens.device == device
        and gather_lens.is_contiguous()
    ):
        return gather_lens
    return gather_lens.reshape(-1).to(device=device, dtype=torch.int32).contiguous()


def _validate_runtime_gather_lens(
    gather_lens: Optional[torch.Tensor],
    compaction: CPByteSlicedSlotCompaction,
    *,
    batch_size: int,
    max_gather_len: int,
) -> Optional[torch.Tensor]:
    """Return compaction-owned lengths without reading runtime CUDA values."""
    expected = tuple(int(value) for value in compaction.gather_lens_cpu)
    saved = compaction.gather_lens
    if gather_lens is None:
        if expected or saved is not None:
            raise ValueError(
                "runtime gather_lens is None but CP SWA compaction captured lengths"
            )
        return None
    if saved is None or not expected:
        raise ValueError(
            "runtime gather_lens was provided but CP SWA compaction has no lengths"
        )
    if gather_lens.dim() != 1 or int(gather_lens.numel()) != batch_size:
        raise ValueError(
            "runtime gather_lens shape does not match the CP SWA batch: "
            f"shape={tuple(gather_lens.shape)}, batch_size={batch_size}"
        )
    if len(expected) != batch_size or int(saved.numel()) != batch_size:
        raise ValueError(
            "CP SWA compaction gather_lens metadata is inconsistent: "
            f"host_count={len(expected)}, device_count={saved.numel()}, "
            f"batch_size={batch_size}"
        )
    if any(value < 0 or value > max_gather_len for value in expected):
        raise ValueError(
            "CP SWA compaction gather_lens is outside the slot-mapping width: "
            f"lengths={expected}, max_gather_len={max_gather_len}"
        )
    return saved


@triton.jit(
    do_not_specialize=[
        "out_stride0",
        "src_stride0",
        "src_stride1",
        "offset",
        "batch_size",
        "width",
    ]
)
def _write_ragged_2d_slices_kernel(
    out_ptr,
    out_stride0,
    src_ptr,
    src_stride0,
    src_stride1,
    cu_dst_offsets_ptr,
    gather_lens_ptr,
    offset,
    batch_size,
    width,
    HAS_SRC: tl.constexpr,
    USE_LENS: tl.constexpr,
    TILE_W: tl.constexpr,
    HEAD_D: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    w_tile = tl.program_id(1).to(tl.int64)
    if b >= batch_size:
        return
    gl = tl.load(gather_lens_ptr + b).to(tl.int64) if USE_LENS else width.to(tl.int64)
    w_offs = w_tile * TILE_W + tl.arange(0, TILE_W)
    w_valid = w_offs < gl
    dst_base = tl.load(cu_dst_offsets_ptr + b).to(tl.int64) + offset.to(tl.int64)
    d_offs = tl.arange(0, HEAD_D)
    dst_ptrs = (
        out_ptr
        + (dst_base + w_offs[:, None]) * out_stride0.to(tl.int64)
        + d_offs[None, :]
    )
    if HAS_SRC:
        src_ptrs = (
            src_ptr
            + b * src_stride0.to(tl.int64)
            + w_offs[:, None] * src_stride1.to(tl.int64)
            + d_offs[None, :]
        )
        vals = tl.load(src_ptrs, mask=w_valid[:, None], other=0)
    else:
        vals = tl.zeros((TILE_W, HEAD_D), dtype=tl.bfloat16)
    tl.store(dst_ptrs, vals, mask=w_valid[:, None])


def _write_ragged_2d_slices(
    out: torch.Tensor,
    offset: int,
    batch_size: int,
    width: int,
    gather_lens_present: bool,
    gather_lens_cpu: Tuple[int, ...],
    cu_dst_offsets: torch.Tensor,
    cu_dst_offsets_cpu: Optional[Tuple[int, ...]],
    src_3d: Optional[torch.Tensor] = None,
    gather_lens: Optional[torch.Tensor] = None,
) -> None:
    """Copy or zero-fill ragged 2D per-request workspace slices."""
    if batch_size == 0 or width == 0:
        return
    use_lens = gather_lens_present and len(gather_lens_cpu) == batch_size
    if (
        out.is_cuda
        and out.dtype == torch.bfloat16
        and int(out.shape[1]) == HEAD_DIM
        and out.stride(1) == 1
        and (
            src_3d is None
            or (
                src_3d.is_cuda
                and src_3d.dtype == torch.bfloat16
                and src_3d.stride(2) == 1
            )
        )
        and (
            not use_lens
            or (
                gather_lens is not None
                and gather_lens.is_cuda
                and gather_lens.is_contiguous()
            )
        )
        and (batch_size > 1 or cu_dst_offsets_cpu is None)
    ):
        _write_ragged_2d_slices_kernel[(batch_size, triton.cdiv(width, TILE_T))](
            out,
            int(out.stride(0)),
            src_3d if src_3d is not None else out,
            int(src_3d.stride(0)) if src_3d is not None else 0,
            int(src_3d.stride(1)) if src_3d is not None else 0,
            cu_dst_offsets,
            gather_lens if use_lens else cu_dst_offsets,
            int(offset),
            batch_size,
            width,
            HAS_SRC=src_3d is not None,
            USE_LENS=use_lens,
            TILE_W=TILE_T,
            HEAD_D=HEAD_DIM,
            num_warps=4,
        )
        return
    offsets = (
        cu_dst_offsets_cpu
        if (cu_dst_offsets_cpu is not None and len(cu_dst_offsets_cpu) >= batch_size)
        else cu_dst_offsets[:batch_size].tolist()
    )
    for b in range(batch_size):
        gl = int(gather_lens_cpu[b]) if use_lens else width
        if gl > 0:
            base = int(offsets[b]) + offset
            dst = out[base : base + gl, :]
            if src_3d is None:
                dst.zero_()
            else:
                dst.copy_(src_3d[b, :gl, :])


def extract_swa_local_slices(
    *,
    k_cache_raw: torch.Tensor,
    slot_mapping: torch.Tensor,
    gather_lens: Optional[torch.Tensor],
    offset: int,
    full_entries_per_block: int,
    cp_rank: int,
    cp_size: int,
    compaction: CPByteSlicedSlotCompaction,
    cu_tiles: Optional[torch.Tensor] = None,
    total_tiles: Optional[int] = None,
    cu_dst_offsets: Optional[torch.Tensor] = None,
    cu_dst_offsets_cpu: Optional[Tuple[int, ...]] = None,
    slot_key: str = "swa_cp.local_slices",
) -> Optional[
    Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[int],
        int,
        int,
        CPByteSlicedSlotCompaction,
    ]
]:
    """Validate and extract compact local SWA byte slices for CP all-gather."""
    assert slot_mapping.dim() == 2
    full_entries_per_block = int(full_entries_per_block)
    cp_size = int(cp_size)
    cp_rank = int(cp_rank)
    assert full_entries_per_block > 0 and cp_size > 1 and 0 <= cp_rank < cp_size
    B = int(slot_mapping.shape[0])
    W = int(slot_mapping.shape[1])
    if B == 0 or W == 0:
        return None
    if cu_dst_offsets is not None:
        assert (
            cu_dst_offsets.dim() == 1
            and int(cu_dst_offsets.numel()) >= B
            and cu_dst_offsets.dtype == torch.int32
            and cu_dst_offsets.device == k_cache_raw.device
            and cu_dst_offsets.is_contiguous()
        ), f"cu_dst_offsets contract mismatch: shape={cu_dst_offsets.shape}"
    assert (
        compaction is not None
    ), "CP byte-sliced SWA gather requires metadata-precomputed compaction"
    saved_gather_lens = _validate_runtime_gather_lens(
        gather_lens,
        compaction,
        batch_size=B,
        max_gather_len=W,
    )
    unique_blocks = compaction.unique_blocks
    if unique_blocks.numel() == 0:
        return None
    device = k_cache_raw.device
    gather_lens_i32 = _device_gather_lens(
        saved_gather_lens,
        batch_size=B,
        max_gather_len=W,
        device=device,
    )
    if (
        (cu_tiles is None or total_tiles is None)
        and len(compaction.gather_lens_cpu) == B
        and (device.type != "cuda" or not torch.cuda.is_current_stream_capturing())
    ):
        cu_tiles, total_tiles = build_dequant_cu_tiles_from_host(
            compaction.gather_lens_cpu,
            device=device,
        )
    if compaction.contiguous_block_start >= 0 and k_cache_raw.is_contiguous():
        local_slices = k_cache_raw.narrow(
            0,
            compaction.contiguous_block_start,
            int(unique_blocks.numel()),
        )
    else:
        local_slices = _acquire_swa_cp_buffer(
            device,
            slot_key,
            (int(unique_blocks.numel()), int(k_cache_raw.shape[1])),
            dtype=k_cache_raw.dtype,
        )
        torch.index_select(
            k_cache_raw,
            0,
            unique_blocks.to(device=device),
            out=local_slices,
        )
    return (
        local_slices,
        gather_lens_i32,
        cu_tiles,
        total_tiles,
        B,
        W,
        compaction,
    )


def build_pre_gathered_swa_pending(
    *,
    gathered: torch.Tensor,
    local_slices: torch.Tensor,
    compaction: CPByteSlicedSlotCompaction,
    gather_lens_i32: torch.Tensor,
    B: int,
    W: int,
    offset: int,
    full_entries_per_block: int,
    cp_size: int,
    producer_stream: Any,
    completion_event: Optional[Any] = None,
    cu_tiles: Optional[torch.Tensor] = None,
    total_tiles: Optional[int] = None,
    cu_dst_offsets: Optional[torch.Tensor] = None,
    cu_dst_offsets_cpu: Optional[Tuple[int, ...]] = None,
) -> CPByteSlicedSwaPrefixPending:
    return CPByteSlicedSwaPrefixPending(
        cp_size=int(cp_size),
        B=int(B),
        W=int(W),
        offset=int(offset),
        full_entries_per_block=int(full_entries_per_block),
        gathered=gathered,
        unique_blocks=compaction.unique_blocks,
        compact_slots=compaction.compact_slots,
        gather_lens=gather_lens_i32,
        gather_lens_cpu=compaction.gather_lens_cpu,
        work=None,
        stream=producer_stream,
        producer_stream=producer_stream,
        completion_event=completion_event,
        local_slices=local_slices,
        work_waited=True,
        cu_tiles=cu_tiles,
        total_tiles=total_tiles,
        cu_dst_offsets=cu_dst_offsets,
        cu_dst_offsets_cpu=cu_dst_offsets_cpu,
    )


def start_dequantize_and_gather_k_cache_slots_cp_byte_sliced(
    *,
    k_cache_raw: torch.Tensor,
    slot_mapping: torch.Tensor,
    gather_lens: Optional[torch.Tensor],
    offset: int,
    full_entries_per_block: int,
    cp_rank: int,
    cp_size: int,
    compaction: CPByteSlicedSlotCompaction,
    stream: Optional[Any] = None,
    profile_name: str = "dsv4.cp.all_gather.swa_prefix",
    cu_tiles: Optional[torch.Tensor] = None,
    total_tiles: Optional[int] = None,
    cu_dst_offsets: Optional[torch.Tensor] = None,
    cu_dst_offsets_cpu: Optional[Tuple[int, ...]] = None,
) -> Optional[CPByteSlicedSwaPrefixPending]:
    """Launch only the NCCL stage for CP byte-sliced SWA prefix reads."""
    extracted = extract_swa_local_slices(
        k_cache_raw=k_cache_raw,
        slot_mapping=slot_mapping,
        gather_lens=gather_lens,
        offset=offset,
        full_entries_per_block=full_entries_per_block,
        cp_rank=cp_rank,
        cp_size=cp_size,
        compaction=compaction,
        cu_tiles=cu_tiles,
        total_tiles=total_tiles,
        cu_dst_offsets=cu_dst_offsets,
        cu_dst_offsets_cpu=cu_dst_offsets_cpu,
        slot_key="swa_cp.async.local_slices",
    )
    if extracted is None:
        return None
    (
        local_slices,
        gather_lens_i32,
        cu_tiles,
        total_tiles,
        B,
        W,
        compaction,
    ) = extracted
    unique_blocks = compaction.unique_blocks
    compact_slots = compaction.compact_slots
    assert k_cache_raw.is_cuda, "CP byte-sliced async SWA gather requires CUDA"
    if not torch.distributed.is_initialized():
        return None

    from rtp_llm.models_py.distributed import collective_torch
    from rtp_llm.models_py.distributed.collective_torch import Group

    process_group = collective_torch._get_group(Group.TP)
    world_size = torch.distributed.get_world_size(process_group)
    if world_size != cp_size:
        raise RuntimeError(
            f"CP byte-sliced SWA gather world_size({world_size}) != cp_size({cp_size})"
        )

    device = k_cache_raw.device
    current_stream = torch.cuda.current_stream(device)
    if stream is None:
        raise ValueError("CP byte-sliced SWA async gather requires a stream")
    gathered = _acquire_swa_cp_buffer(
        device,
        "swa_cp.async.gathered",
        (cp_size * int(unique_blocks.numel()), int(k_cache_raw.shape[1])),
        dtype=k_cache_raw.dtype,
    )
    gather_stream = stream
    gather_stream.wait_stream(current_stream)

    with torch.cuda.stream(gather_stream):
        with record_function_range(f"{profile_name}.launch"):
            work = torch.distributed.all_gather_into_tensor(
                gathered,
                local_slices,
                group=process_group,
                async_op=True,
            )
        try:
            completion_event = torch.cuda.Event()
            completion_event.record(gather_stream)
        except Exception:
            # Drain the in-flight NCCL Work before propagating; the caller
            # never sees the pending handle so nothing else will wait it.
            work.wait()
            current_stream.wait_stream(gather_stream)
            raise

    return CPByteSlicedSwaPrefixPending(
        cp_size=cp_size,
        B=B,
        W=W,
        offset=int(offset),
        full_entries_per_block=full_entries_per_block,
        gathered=gathered,
        unique_blocks=unique_blocks,
        compact_slots=compact_slots,
        gather_lens=gather_lens_i32,
        gather_lens_cpu=compaction.gather_lens_cpu,
        work=work,
        stream=gather_stream,
        producer_stream=current_stream,
        completion_event=completion_event,
        local_slices=local_slices,
        cu_tiles=cu_tiles,
        total_tiles=total_tiles,
        cu_dst_offsets=cu_dst_offsets,
        cu_dst_offsets_cpu=cu_dst_offsets_cpu,
    )


def prepare_dequantize_and_gather_k_cache_slots_cp_byte_sliced(
    pending: CPByteSlicedSwaPrefixPending,
    *,
    out: torch.Tensor,
    stream: Optional[Any] = None,
    cu_dst_offsets: Optional[torch.Tensor] = None,
    cu_dst_offsets_cpu: Optional[Tuple[int, ...]] = None,
) -> None:
    """Enqueue restore/dequant/copy for a pending CP byte-sliced SWA read."""
    if pending.ready_event is not None:
        return
    cu_dst_offsets_eff = (
        cu_dst_offsets if cu_dst_offsets is not None else pending.cu_dst_offsets
    )
    cu_dst_offsets_cpu_eff = (
        cu_dst_offsets_cpu
        if cu_dst_offsets_cpu is not None
        else pending.cu_dst_offsets_cpu
    )
    if cu_dst_offsets_eff is not None:
        assert (
            out.dim() == 2 and out.shape[-1] == HEAD_DIM and out.dtype == torch.bfloat16
        )
        assert out.stride(1) == 1
        assert (
            cu_dst_offsets_eff.dim() == 1
            and int(cu_dst_offsets_eff.numel()) >= pending.B
            and cu_dst_offsets_eff.dtype == torch.int32
            and cu_dst_offsets_eff.device == out.device
            and cu_dst_offsets_eff.is_contiguous()
        )
    else:
        assert (
            out.dim() == 3 and out.shape[-1] == HEAD_DIM and out.dtype == torch.bfloat16
        )
    device = pending.gathered.device
    if stream is None:
        raise ValueError("CP byte-sliced SWA async prepare requires a stream")
    assemble_stream = stream
    current_stream = torch.cuda.current_stream(out.device)
    # ``out`` is the shared attention workspace.  Keep postprocess ordered after
    # current-stream workspace writes; the caller decides how early to invoke
    # prepare based on which ranges are disjoint.
    assemble_stream.wait_stream(current_stream)
    pending.out = out
    pending.out_stream = current_stream
    with torch.cuda.stream(assemble_stream):
        if pending.completion_event is not None:
            assemble_stream.wait_event(pending.completion_event)
        # Fence NCCL on the stream that will read the gathered bytes below.
        with record_function_range("dsv4.cp.all_gather.swa_prefix.wait_host"):
            _wait_swa_prefix_work_once(pending)
        try:
            if (
                cp_swa_direct_dequant_scatter_enabled()
                and direct_triton_fast_path_supported(out.device)
            ):
                _launch_dequantize_and_gather_k_slots_cp_rank_major_unchecked(
                    out,
                    pending.gathered,
                    pending.compact_slots,
                    pending.gather_lens,
                    pending.offset,
                    full_entries_per_block=pending.full_entries_per_block,
                    num_unique_blocks=int(pending.unique_blocks.numel()),
                    cu_tiles=pending.cu_tiles,
                    total_tiles=pending.total_tiles,
                    cu_dst_offsets=cu_dst_offsets_eff,
                )
            else:
                full_raw = (
                    pending.gathered.view(
                        pending.cp_size,
                        int(pending.unique_blocks.numel()),
                        int(pending.gathered.shape[1]),
                    )
                    .permute(1, 0, 2)
                    .reshape(
                        int(pending.unique_blocks.numel()),
                        pending.cp_size * int(pending.gathered.shape[1]),
                    )
                    .contiguous()
                )
                full_view = full_raw.as_strided(
                    (
                        int(pending.unique_blocks.numel()),
                        pending.full_entries_per_block,
                        ENTRY_BYTES,
                    ),
                    (int(full_raw.shape[1]), ENTRY_BYTES, 1),
                )
                restored = dequantize_slots_to_bf16(
                    full_view,
                    pending.compact_slots.reshape(-1),
                )
                restored_3d = restored.view(pending.B, pending.W, HEAD_DIM)
                if cu_dst_offsets_eff is not None:
                    _write_ragged_2d_slices(
                        out,
                        pending.offset,
                        pending.B,
                        pending.W,
                        gather_lens_present=bool(pending.gather_lens_cpu),
                        gather_lens_cpu=pending.gather_lens_cpu,
                        cu_dst_offsets=cu_dst_offsets_eff,
                        cu_dst_offsets_cpu=cu_dst_offsets_cpu_eff,
                        src_3d=restored_3d,
                        gather_lens=pending.gather_lens,
                    )
                elif pending.gather_lens_cpu:
                    # CPU lengths are captured during compaction; avoid per-request
                    # device synchronizations in the fallback hot path.
                    for b, gl in enumerate(pending.gather_lens_cpu):
                        if gl > 0:
                            out[b, pending.offset : pending.offset + gl, :].copy_(
                                restored_3d[b, :gl, :]
                            )
                else:
                    out[:, pending.offset : pending.offset + pending.W, :].copy_(
                        restored_3d
                    )
        finally:
            try:
                ready_event = torch.cuda.Event()
                ready_event.record(assemble_stream)
                pending.ready_event = ready_event
            except Exception:
                # The caller cannot wait on a terminal event that was not
                # created/recorded. Drain queued post-gather reads/writes before
                # gathered/out ownership can unwind through the exception path.
                assemble_stream.synchronize()
                raise


def wait_dequantize_and_gather_k_cache_slots_cp_byte_sliced(
    pending: CPByteSlicedSwaPrefixPending,
) -> None:
    if pending.ready_event is None:
        raise RuntimeError("CP byte-sliced SWA prefix pending was not prepared")
    current_stream = torch.cuda.current_stream(pending.gathered.device)
    current_stream.wait_event(pending.ready_event)
    if pending.producer_stream != current_stream:
        pending.producer_stream.wait_event(pending.ready_event)
    if pending.out_stream != current_stream:
        pending.out_stream.wait_event(pending.ready_event)
    if pending.stream is not None:
        pending.stream.wait_event(pending.ready_event)
    _wait_swa_prefix_work_once(pending)


def discard_dequantize_and_gather_k_cache_slots_cp_byte_sliced(
    pending: Optional[CPByteSlicedSwaPrefixPending],
) -> None:
    if pending is None:
        return
    current_stream = torch.cuda.current_stream(pending.gathered.device)
    event = pending.ready_event or pending.completion_event
    if event is not None:
        current_stream.wait_event(event)
        if (
            pending.producer_stream is not None
            and pending.producer_stream != current_stream
        ):
            pending.producer_stream.wait_event(event)
        if pending.out_stream is not None and pending.out_stream != current_stream:
            pending.out_stream.wait_event(event)
    if pending.ready_event is not None and pending.stream is not None:
        pending.stream.wait_event(pending.ready_event)
    _wait_swa_prefix_work_once(pending)


def _wait_swa_prefix_work_once(pending: CPByteSlicedSwaPrefixPending) -> None:
    if not pending.work_waited:
        if pending.work is not None:
            pending.work.wait()
        pending.work_waited = True


def dequantize_and_gather_k_cache_slots_cp_byte_sliced(
    out: torch.Tensor,
    k_cache_raw: torch.Tensor,
    slot_mapping: torch.Tensor,
    gather_lens: Optional[torch.Tensor],
    offset: int,
    full_entries_per_block: int,
    cp_rank: int,
    cp_size: int,
    compaction: CPByteSlicedSlotCompaction,
    *,
    cu_tiles: Optional[torch.Tensor] = None,
    total_tiles: Optional[int] = None,
    cu_dst_offsets: Optional[torch.Tensor] = None,
    cu_dst_offsets_cpu: Optional[Tuple[int, ...]] = None,
) -> None:
    """Reconstruct CP byte-sliced SWA entries with all_gather, then dequantize.

    Args:
        out: BF16 destination workspace with shape ``[B, M, HEAD_DIM]`` when
            ``cu_dst_offsets is None`` or flat ragged ``[total_rows, HEAD_DIM]``
            when ``cu_dst_offsets`` is provided. Restored rows are written to
            ``out[:, offset:offset + W, :]`` (or
            ``out[cu_dst_offsets[b] + offset : ..., :]``), clipped per request
            by ``gather_lens`` when provided.
        k_cache_raw: Current rank's SWA_KV raw byte pool with shape
            ``[num_blocks, local_slice_bytes]``. Each row is this CP rank's
            byte slice of one physical SWA block.
        slot_mapping: Canonical SWA slot ids with shape ``[B, W]``. Each
            non-negative slot is encoded as
            ``physical_block_id * full_entries_per_block + offset_in_block``;
            ``-1`` means skip.
            W is max_b(min(prefix_len, window_size - 1))
        gather_lens: Optional per-request valid length with shape ``[B]``.
            If ``None``, all ``W`` columns are copied for every request.
        offset: Row offset relative to request base in ``out`` where restored
            SWA rows begin.
            This is ``0`` for SWA-only attention. For CSA/HCA workspace
            attention, compressed K precedes SWA rows, so SWA rows
            start after that compressed prefix.
        full_entries_per_block: Entry count of the reconstructed full SWA
            block, i.e. the divisor used to decode ``slot_mapping``.
        cp_rank: Current CP rank id. Used for contract validation only; the
            input pool already contains this rank's local byte slice.
        cp_size: Number of CP ranks whose byte slices form one full SWA block.
        cu_dst_offsets: Optional ``[>= B]`` int32 contiguous tensor on
            ``out.device`` giving per-request base row in 2D ``out``.
        cu_dst_offsets_cpu: Optional host tuple matching ``cu_dst_offsets`` to
            avoid D2H syncs on zero-fill/fallback paths.
    """
    has_cu_dst_offsets = cu_dst_offsets is not None
    assert slot_mapping.dim() == 2
    B = int(slot_mapping.shape[0])
    W = int(slot_mapping.shape[1])
    if has_cu_dst_offsets:
        assert (
            out.dim() == 2 and out.shape[-1] == HEAD_DIM and out.dtype == torch.bfloat16
        )
        assert out.stride(1) == 1
        assert (
            cu_dst_offsets.dim() == 1
            and int(cu_dst_offsets.numel()) >= B
            and cu_dst_offsets.dtype == torch.int32
            and cu_dst_offsets.device == out.device
            and cu_dst_offsets.is_contiguous()
        ), f"cu_dst_offsets contract mismatch: shape={cu_dst_offsets.shape}"
    else:
        assert (
            out.dim() == 3 and out.shape[-1] == HEAD_DIM and out.dtype == torch.bfloat16
        )
    full_entries_per_block = int(full_entries_per_block)
    cp_size = int(cp_size)
    cp_rank = int(cp_rank)
    assert full_entries_per_block > 0 and cp_size > 1 and 0 <= cp_rank < cp_size
    if B == 0 or W == 0:
        return
    assert (
        compaction is not None
    ), "CP byte-sliced SWA gather requires metadata-precomputed compaction"
    saved_gather_lens = _validate_runtime_gather_lens(
        gather_lens,
        compaction,
        batch_size=B,
        max_gather_len=W,
    )
    unique_blocks = compaction.unique_blocks
    compact_slots = compaction.compact_slots
    if unique_blocks.numel() == 0:
        if not has_cu_dst_offsets:
            out[:, offset : offset + W, :].zero_()
        elif (
            cu_dst_offsets_cpu is None
            and cp_swa_direct_dequant_scatter_enabled()
            and direct_triton_fast_path_supported(out.device)
        ):
            gather_lens_i32 = _device_gather_lens(
                saved_gather_lens, batch_size=B, max_gather_len=W, device=out.device
            )
            dummy_gathered = k_cache_raw.new_empty((cp_size, int(k_cache_raw.shape[1])))
            _launch_dequantize_and_gather_k_slots_cp_rank_major_unchecked(
                out,
                dummy_gathered,
                compact_slots,
                gather_lens_i32,
                offset,
                full_entries_per_block=full_entries_per_block,
                num_unique_blocks=1,
                cu_tiles=cu_tiles,
                total_tiles=total_tiles,
                cu_dst_offsets=cu_dst_offsets,
            )
        else:
            gather_lens_arg = (
                _device_gather_lens(
                    saved_gather_lens, batch_size=B, max_gather_len=W, device=out.device
                )
                if gather_lens is not None
                else None
            )
            _write_ragged_2d_slices(
                out,
                offset,
                B,
                W,
                gather_lens_present=gather_lens is not None,
                gather_lens_cpu=compaction.gather_lens_cpu,
                cu_dst_offsets=cu_dst_offsets,
                cu_dst_offsets_cpu=cu_dst_offsets_cpu,
                src_3d=None,
                gather_lens=gather_lens_arg,
            )
        return

    from rtp_llm.models_py.distributed import collective_torch
    from rtp_llm.models_py.distributed.collective_torch import Group, all_gather

    if compaction.contiguous_block_start >= 0 and k_cache_raw.is_contiguous():
        local_slices = k_cache_raw.narrow(
            0,
            compaction.contiguous_block_start,
            int(unique_blocks.numel()),
        )
    else:
        local_slices = _acquire_swa_cp_buffer(
            k_cache_raw.device,
            "swa_cp.sync.local_slices",
            (int(unique_blocks.numel()), int(k_cache_raw.shape[1])),
            dtype=k_cache_raw.dtype,
        )
        torch.index_select(
            k_cache_raw,
            0,
            unique_blocks.to(device=k_cache_raw.device),
            out=local_slices,
        )
    all_gather_into = getattr(collective_torch, "all_gather_into", None)
    if (
        all_gather_into is not None
        and torch.distributed.is_initialized()
        and (
            k_cache_raw.device.type != "cuda"
            or not torch.cuda.is_current_stream_capturing()
        )
    ):
        pg = collective_torch._get_group(Group.TP)
        world_size = torch.distributed.get_world_size(pg)
        gathered = _acquire_swa_cp_buffer(
            k_cache_raw.device,
            "swa_cp.sync.gathered",
            (world_size * int(unique_blocks.numel()), int(k_cache_raw.shape[1])),
            dtype=k_cache_raw.dtype,
        )
        with record_function_range("dsv4.cp.all_gather.swa_prefix.sync.launch"):
            all_gather_into(gathered, local_slices, group=Group.TP)
    else:
        with record_function_range("dsv4.cp.all_gather.swa_prefix.sync.launch"):
            gathered = all_gather(local_slices, group=Group.TP)
    gather_lens_i32 = _device_gather_lens(
        saved_gather_lens,
        batch_size=B,
        max_gather_len=W,
        device=out.device,
    )
    if cp_swa_direct_dequant_scatter_enabled() and direct_triton_fast_path_supported(
        out.device
    ):
        if (
            (cu_tiles is None or total_tiles is None)
            and len(compaction.gather_lens_cpu) == B
            and (
                out.device.type != "cuda"
                or not torch.cuda.is_current_stream_capturing()
            )
        ):
            cu_tiles, total_tiles = build_dequant_cu_tiles_from_host(
                compaction.gather_lens_cpu,
                device=out.device,
            )
        _launch_dequantize_and_gather_k_slots_cp_rank_major_unchecked(
            out,
            gathered,
            compact_slots,
            gather_lens_i32,
            offset,
            full_entries_per_block=full_entries_per_block,
            num_unique_blocks=int(unique_blocks.numel()),
            cu_tiles=cu_tiles,
            total_tiles=total_tiles,
            cu_dst_offsets=cu_dst_offsets,
        )
        return
    gathered = gathered.view(
        cp_size, int(unique_blocks.numel()), int(k_cache_raw.shape[1])
    )
    full_raw = (
        gathered.permute(1, 0, 2)
        .reshape(int(unique_blocks.numel()), cp_size * int(k_cache_raw.shape[1]))
        .contiguous()
    )  # [num_unique_blocks, full_block_bytes]
    full_view = full_raw.as_strided(
        (int(unique_blocks.numel()), full_entries_per_block, ENTRY_BYTES),
        (int(full_raw.shape[1]), ENTRY_BYTES, 1),
    )
    restored = dequantize_slots_to_bf16(full_view, compact_slots.reshape(-1))
    restored_3d = restored.view(B, W, HEAD_DIM)
    if has_cu_dst_offsets:
        if gather_lens is not None:
            assert (
                len(compaction.gather_lens_cpu) == B
            ), "CP byte-sliced SWA gather compaction must include per-request gather_lens_cpu"
        _write_ragged_2d_slices(
            out,
            offset,
            B,
            W,
            gather_lens_present=gather_lens is not None,
            gather_lens_cpu=compaction.gather_lens_cpu,
            cu_dst_offsets=cu_dst_offsets,
            cu_dst_offsets_cpu=cu_dst_offsets_cpu,
            src_3d=restored_3d,
        )
        return
    if gather_lens is None:
        out[:, offset : offset + W, :].copy_(restored_3d)
        return
    gather_lens_cpu = compaction.gather_lens_cpu
    assert (
        len(gather_lens_cpu) == B
    ), "CP byte-sliced SWA gather compaction must include per-request gather_lens_cpu"
    for b, gl in enumerate(gather_lens_cpu):
        if gl > 0:
            out[b, offset : offset + gl, :].copy_(restored_3d[b, :gl, :])
