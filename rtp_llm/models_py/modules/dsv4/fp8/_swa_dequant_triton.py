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

from typing import Optional

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv4.fp8._trap_utils import (
    invalid_kv_access_validation_enabled,
    trap_invalid_kv_access_enabled,
    validate_block_table_lookup,
    validate_slot_mapping,
)

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



@triton.jit(do_not_specialize=["offset", "max_blocks_per_seq"])
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
    TRAP_INVALID_KV_ACCESS: tl.constexpr,
):
    """One program per (request, worker); workers stripe across the gathered
    tokens in the request. Mirrors vLLM's kernel exactly — see the V4 paper
    or vLLM's cache_utils.py for the math."""
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    if gather_lens_ptr is not None:  # noqa: SIM108
        gather_len = tl.load(gather_lens_ptr + batch_idx)
    else:
        gather_len = seq_len
    start_pos = seq_len - gather_len

    for i in range(worker_id, gather_len, num_workers):
        pos = start_pos + i
        block_in_seq = pos // cache_block_size
        pos_in_block = pos % cache_block_size

        if pos < 0:
            _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)
        if block_in_seq >= max_blocks_per_seq:
            _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)

        block_table_row_ptr = block_table_ptr + batch_idx * max_blocks_per_seq
        physical_block_idx = tl.load(block_table_row_ptr + block_in_seq)
        if physical_block_idx < 0:
            _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)
        if physical_block_idx >= num_cache_blocks:
            _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)

        # int64: physical_block_idx * block_stride can exceed 2^31 with many
        # KV-cache blocks. Match vLLM.
        cache_block_ptr = k_cache_ptr + physical_block_idx.to(tl.int64) * block_stride

        token_data_ptr = cache_block_ptr + pos_in_block * token_data_size
        token_scale_ptr = (
            cache_block_ptr
            + cache_block_size * token_data_size
            + pos_in_block * scale_dim
        )
        token_fp8_ptr = token_data_ptr
        token_bf16_ptr = token_data_ptr + fp8_dim

        # Keep the row offset in int64. Production prefill workspaces can have
        # very large per-request strides, so batch_idx * out_stride0 may exceed
        # signed int32 even though the underlying tensor allocation is valid.
        output_row = offset.to(tl.int64) + i.to(tl.int64)
        output_row_ptr = (
            out_ptr
            + batch_idx.to(tl.int64) * out_stride0.to(tl.int64)
            + output_row * out_stride1.to(tl.int64)
        )

        # NoPE dequant (7 tiles × 64 elements, UE8M0 scale per tile)
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

        # RoPE: bf16 passthrough
        bf16_output_offset = fp8_dim
        bf16_cache_ptr = token_bf16_ptr.to(tl.pointer_type(tl.bfloat16))
        for j in tl.static_range(bf16_dim // 16):
            chunk_offsets = j * 16 + tl.arange(0, 16)
            bf16_vals = tl.load(bf16_cache_ptr + chunk_offsets)
            tl.store(output_row_ptr + bf16_output_offset + chunk_offsets, bf16_vals)


def dequantize_and_gather_k_cache(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: Optional[torch.Tensor],
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    """Dequantize + gather FP8 K cache into a BF16 workspace.

    Direct port of vLLM ``dequantize_and_gather_k_cache``.

    Args:
      out:          ``[num_reqs, max_num_tokens, 512]`` bf16 — workspace; this
                    function writes into ``out[:, offset:offset+gather_len, :]``.
      k_cache:      ``[num_blocks, block_size, 584]`` uint8 — the FP8 SWA pool
                    in ``fp8_ds_mla`` layout.
      seq_lens:     ``[num_reqs]`` int — total tokens per request in ``k_cache``.
      gather_lens:  optional ``[num_reqs]`` int — tokens to gather (suffix);
                    ``None`` ⇒ gather all ``seq_lens[r]`` tokens for request r.
      block_table:  ``[num_reqs, max_blocks_per_seq]`` int — physical block id
                    per logical block per request.
      block_size:   tokens per cache block (matches the 3D shape of k_cache).
      offset:       column offset in ``out`` to start writing.
    """
    assert out.dim() == 3 and out.shape[-1] == HEAD_DIM and out.dtype == torch.bfloat16
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
    assert (
        block_table.dtype == torch.int32
        and block_table.is_contiguous()
    ), (
        "block_table must be int32 and contiguous; "
        f"got dtype={block_table.dtype} contig={block_table.is_contiguous()} "
        f"dev={block_table.device} (k_cache dev={k_cache.device})"
    )

    num_reqs = seq_lens.shape[0]
    if invalid_kv_access_validation_enabled():
        seq_i64 = seq_lens.detach().reshape(-1).to(torch.int64)
        if gather_lens is None:
            gather_i64 = seq_i64
        else:
            gather_i64 = gather_lens.detach().reshape(-1).to(torch.int64)
        max_gather = int(gather_i64.max().item()) if int(gather_i64.numel()) > 0 else 0
        if max_gather > 0:
            steps = torch.arange(max_gather, device=seq_lens.device, dtype=torch.int64)
            pos = seq_i64[:, None] - gather_i64[:, None] + steps[None, :]
            use = steps[None, :] < gather_i64[:, None]
            req = torch.arange(num_reqs, device=seq_lens.device, dtype=torch.int64)
            validate_block_table_lookup(
                "swa.dequantize_and_gather.block_table",
                block_table,
                req[:, None].expand_as(pos),
                pos // int(block_size),
                use,
                num_blocks=int(k_cache.shape[0]),
            )
    NUM_WORKERS = 128
    _dequantize_and_gather_k_kernel[(num_reqs, NUM_WORKERS)](
        out,
        out.stride(0),
        out.stride(1),
        k_cache,
        seq_lens,
        block_table,
        offset,
        gather_lens,
        max_blocks_per_seq=block_table.shape[-1],
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
        TRAP_INVALID_KV_ACCESS=trap_invalid_kv_access_enabled(),
    )


@triton.jit
def _gather_k_cache_packed_kernel(
    out_ptr,
    out_stride0,
    out_stride1,
    k_cache_ptr,
    seq_lens_ptr,
    block_table_ptr,
    offset,
    gather_lens_ptr,
    max_blocks_per_seq: tl.constexpr,
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,
    scale_dim: tl.constexpr,
    block_stride: tl.constexpr,
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

        block_table_row_ptr = block_table_ptr + batch_idx * max_blocks_per_seq
        physical_block_idx = tl.load(block_table_row_ptr + block_in_seq)
        tl.device_assert(physical_block_idx >= 0, "block_table contains -1")

        cache_block_ptr = k_cache_ptr + physical_block_idx.to(tl.int64) * block_stride
        token_data_ptr = cache_block_ptr + pos_in_block * token_data_size
        token_scale_ptr = (
            cache_block_ptr
            + cache_block_size * token_data_size
            + pos_in_block * scale_dim
        )
        output_row_ptr = out_ptr + batch_idx * out_stride0 + (offset + i) * out_stride1

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


@triton.jit
def _dequantize_packed_k_cache_flat_kernel(
    out_ptr,
    out_stride0,
    packed_ptr,
    n_tokens,
    fp8_dim: tl.constexpr,
    bf16_dim: tl.constexpr,
    quant_block: tl.constexpr,
    token_data_size: tl.constexpr,
    entry_bytes: tl.constexpr,
    n_quant_blocks: tl.constexpr,
):
    token_idx = tl.program_id(0)
    if token_idx >= n_tokens:
        return

    token_ptr = packed_ptr + token_idx * entry_bytes
    token_fp8_ptr = token_ptr
    token_bf16_ptr = token_ptr + fp8_dim
    token_scale_ptr = token_ptr + token_data_size
    output_row_ptr = out_ptr + token_idx * out_stride0

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

    bf16_output_offset = fp8_dim
    bf16_cache_ptr = token_bf16_ptr.to(tl.pointer_type(tl.bfloat16))
    for j in tl.static_range(bf16_dim // 16):
        chunk_offsets = j * 16 + tl.arange(0, 16)
        bf16_vals = tl.load(bf16_cache_ptr + chunk_offsets)
        tl.store(output_row_ptr + bf16_output_offset + chunk_offsets, bf16_vals)


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
    _dequantize_packed_k_cache_flat_kernel[(packed.shape[0],)](
        out,
        out.stride(0),
        packed,
        packed.shape[0],
        fp8_dim=NOPE_DIM,
        bf16_dim=ROPE_DIM,
        quant_block=TILE_SIZE,
        token_data_size=TOKEN_DATA_SIZE,
        entry_bytes=ENTRY_BYTES,
        n_quant_blocks=NOPE_TILES,
    )


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


@triton.jit
def _dequantize_slots_kernel(
    out_ptr,  # [N, 512] bf16
    out_stride0,
    pool_ptr,  # [num_blocks, block_size, 584] uint8
    pool_block_stride: tl.constexpr,  # bytes per block (may be padded > bs*584)
    slot_indices_ptr,  # [N] int64
    block_size: tl.constexpr,
    fp8_dim: tl.constexpr,  # 448
    bf16_dim: tl.constexpr,  # 64
    scale_dim: tl.constexpr,  # 8
    quant_block: tl.constexpr,  # 64
    token_data_size: tl.constexpr,  # 576
    num_cache_blocks: tl.constexpr,
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

    cache_block_ptr = pool_ptr + blk * pool_block_stride
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
    validate_slot_mapping(
        "swa.dequantize_slots.slot_indices",
        slots_i64,
        block_size=block_size,
        num_blocks=int(pool_3d.shape[0]),
        negative_mode="skip_any",
    )
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
        TRAP_INVALID_KV_ACCESS=trap_invalid_kv_access_enabled(),
    )
    return out
