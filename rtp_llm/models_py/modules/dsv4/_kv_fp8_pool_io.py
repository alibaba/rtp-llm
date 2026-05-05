"""Production reader for the canonical 584B FP8 KV pool.

Inverse of ``_compressor_kv_fused_triton.v4_compressor_kv_fused`` —
gathers a contiguous tail window of FP8-quantized K from the per-block
striped layout into a bf16 buffer ready for ``flash_mla_sparse_fwd``.

  pool [num_blocks, block_size, 584] uint8
       │   per-block striped:
       │     bytes [0,         bs*576) — token data: 448 fp8 NoPE + 64 bf16 RoPE
       │     bytes [bs*576,  +bs*8)    — UE8M0 scales: 7 real + 1 pad per token
       │     trailing pad to TMA alignment
       ▼
  out  [num_reqs, max_T, 512] bf16
       │   first 448 dims: dequant'd NoPE  ( fp8 * 2^(scale_byte - 127) )
       │   last  64 dims:  bf16 RoPE direct copy

This kernel is **vendored from vLLM HEAD
``ff449b6426812d1e5e107715af899fcff5e81419``** (file
``vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py``,
``dequantize_and_gather_k_cache`` /
``_dequantize_and_gather_k_kernel``). Byte-level compatibility with the
rtp-llm 584B writer is locked by
``test/test_compressor_fp8_reader.py``.

A test-only frozen copy lives at ``test/_vllm_ref/cache_utils.py`` for
byte-level golden tests; this file is the production import path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Layout constants — must agree with _compressor_kv_fused_triton.py and
# vLLM's quantize_and_insert_k_cache. Any change here without a matching
# writer update will silently corrupt the read path.
_TOKEN_FP8_DIM = 448
_TOKEN_BF16_DIM = 64
_TOKEN_SCALE_DIM = 8  # 7 real + 1 pad
_QUANT_BLOCK_SIZE = 64
_TOKEN_DATA_SIZE = _TOKEN_FP8_DIM + _TOKEN_BF16_DIM * 2  # 576
_OUTPUT_DIM = 512  # head_dim
_FP8_MAX = 448.0
_N_QUANT_BLOCKS = 7  # fp8_dim / quant_block

_NUM_WORKERS_PER_REQ = 128


@triton.jit
def _dequantize_and_gather_k_kernel(
    out_ptr,
    out_stride0,
    out_stride1,
    k_cache_ptr,
    seq_lens_ptr,
    block_table_ptr,
    offset,
    gather_lens_ptr,
    max_blocks_per_seq: tl.constexpr,
    fp8_dim: tl.constexpr,
    bf16_dim: tl.constexpr,
    scale_dim: tl.constexpr,
    quant_block: tl.constexpr,
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,
    block_stride: tl.constexpr,
    output_dim: tl.constexpr,
    fp8_max: tl.constexpr,
    n_quant_blocks: tl.constexpr,
):
    """One program per (request, worker); workers split the gather range."""
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

        block_table_row_ptr = block_table_ptr + batch_idx * max_blocks_per_seq
        physical_block_idx = tl.load(block_table_row_ptr + block_in_seq)

        # int64: physical_block_idx * block_stride can exceed 2^31 with many
        # KV cache blocks.
        cache_block_ptr = k_cache_ptr + physical_block_idx.to(tl.int64) * block_stride
        token_data_ptr = cache_block_ptr + pos_in_block * token_data_size
        token_scale_ptr = (
            cache_block_ptr
            + cache_block_size * token_data_size
            + pos_in_block * scale_dim
        )
        token_fp8_ptr = token_data_ptr
        token_bf16_ptr = token_data_ptr + fp8_dim

        output_row_ptr = out_ptr + batch_idx * out_stride0 + (offset + i) * out_stride1

        # NoPE: fp8 → fp32 * 2^(scale_byte - 127) → bf16
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
                tl.store(
                    output_row_ptr + offsets,
                    x_dequant.to(tl.bfloat16),
                    mask=mask,
                )

        # RoPE: bf16 direct copy
        bf16_output_offset = fp8_dim
        bf16_cache_ptr = token_bf16_ptr.to(tl.pointer_type(tl.bfloat16))
        for j in tl.static_range(bf16_dim // 16):
            chunk_offsets = j * 16 + tl.arange(0, 16)
            bf16_vals = tl.load(bf16_cache_ptr + chunk_offsets)
            tl.store(output_row_ptr + bf16_output_offset + chunk_offsets, bf16_vals)


def dequantize_and_gather_k_cache(
    out: torch.Tensor,  # [num_reqs, max_num_tokens, 512] bf16
    k_cache: torch.Tensor,  # [num_blocks, block_size, 584] uint8
    seq_lens: torch.Tensor,  # [num_reqs] int32 — per-req total seq len
    gather_lens: (
        torch.Tensor | None
    ),  # [num_reqs] int32 — gather only the trailing N tokens; None = all
    block_table: torch.Tensor,  # [num_reqs, max_blocks_per_seq] int32
    block_size: int,
    offset: int,
) -> None:
    """Dequant the trailing window of each request's K cache into ``out``.

    Layout / scale convention: see module docstring. ``out[b, offset+i, :]``
    receives the i-th gathered token (i ∈ [0, gather_len[b])) of request b,
    where the gathered range is ``[seq_len - gather_len, seq_len)`` —
    i.e. the most recent ``gather_len`` tokens.

    In-place writes ``out``. No return.
    """
    num_reqs = seq_lens.shape[0]
    _dequantize_and_gather_k_kernel[(num_reqs, _NUM_WORKERS_PER_REQ)](
        out,
        out.stride(0),
        out.stride(1),
        k_cache,
        seq_lens,
        block_table,
        offset,
        gather_lens,
        max_blocks_per_seq=block_table.shape[-1],
        fp8_dim=_TOKEN_FP8_DIM,
        bf16_dim=_TOKEN_BF16_DIM,
        scale_dim=_TOKEN_SCALE_DIM,
        quant_block=_QUANT_BLOCK_SIZE,
        cache_block_size=block_size,
        token_data_size=_TOKEN_DATA_SIZE,
        block_stride=k_cache.stride(0),
        output_dim=_OUTPUT_DIM,
        fp8_max=_FP8_MAX,
        n_quant_blocks=_N_QUANT_BLOCKS,
    )
