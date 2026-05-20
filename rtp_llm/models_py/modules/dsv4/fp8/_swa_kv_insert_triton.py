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


@triton.jit
def _trap_invalid_kv_access() -> None:
    tl.inline_asm_elementwise(
        "trap; // dummy $0",
        "=r",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit(do_not_specialize=["num_tokens"])
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
    block_stride: tl.constexpr,  # bytes per block (TMA-padded; from k_cache.stride(0))
    num_cache_blocks: tl.constexpr,
    fp8_max: tl.constexpr,  # 448.0
    n_quant_blocks: tl.constexpr,  # 8 (7 real + 1 padding tile-loop iter)
):
    """One program per token. Block layout per pool block:
    [0, cache_block_size * 576)             — token data (NoPE+RoPE interleaved)
    [cache_block_size * 576,
     cache_block_size * 576 + cache_block_size * 8)  — scales
    [..., block_stride)                      — TMA padding
    """
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    slot_idx = tl.load(slot_mapping_ptr + pid)
    if slot_idx == -1:
        return

    block_idx = slot_idx // cache_block_size
    pos_in_block = slot_idx % cache_block_size
    if block_idx < 0:
        _trap_invalid_kv_access()
    if block_idx >= num_cache_blocks:
        _trap_invalid_kv_access()

    input_row_ptr = k_ptr + pid * input_dim

    # int64: block_idx * block_stride can overflow int32 with many blocks
    # (e.g. >= 14K at block_stride 149504 → 2^31). Matches dequant kernel.
    cache_block_ptr = k_cache_ptr + block_idx.to(tl.int64) * block_stride

    token_data_ptr = cache_block_ptr + pos_in_block * token_data_size
    token_scale_ptr = (
        cache_block_ptr + cache_block_size * token_data_size + pos_in_block * scale_dim
    )

    token_fp8_ptr = token_data_ptr
    token_bf16_ptr = token_data_ptr + fp8_dim

    # NoPE: 7 quantization tiles of 64 elements each (UE8M0 power-of-2 scale).
    for qblock_idx in tl.static_range(n_quant_blocks):
        qblock_start = qblock_idx * quant_block
        if qblock_start < fp8_dim:
            offsets = qblock_start + tl.arange(0, quant_block)
            mask = offsets < fp8_dim

            x = tl.load(input_row_ptr + offsets, mask=mask, other=0.0)

            abs_x = tl.abs(x)
            block_max = tl.max(abs_x, axis=0)
            block_max = tl.maximum(block_max, 1e-4)  # match CUDA fmaxf(amax, 1e-4)

            # UE8M0: scale = 2^ceil(log2(block_max / fp8_max))
            raw_scale = block_max / fp8_max
            log_scale = tl.log2(raw_scale)
            exponent = tl.ceil(log_scale)
            scale = tl.exp2(exponent)

            x_scaled = x / scale
            x_clamped = tl.clamp(x_scaled, -fp8_max, fp8_max)
            x_fp8 = x_clamped.to(tl.float8e4nv)
            x_uint8 = x_fp8.to(tl.uint8, bitcast=True)

            tl.store(token_fp8_ptr + offsets, x_uint8, mask=mask)

            # UE8M0 scale byte = exponent + 127 (clamp [0, 255]).
            encoded_scale = exponent + 127.0
            encoded_scale = tl.maximum(tl.minimum(encoded_scale, 255.0), 0.0)
            tl.store(token_scale_ptr + qblock_idx, encoded_scale.to(tl.uint8))

    # Padding scale at index 7 (8 - 1).
    tl.store(token_scale_ptr + 7, tl.zeros((), dtype=tl.uint8))

    # RoPE: bf16 passthrough, 64 elements = 128 bytes.
    bf16_input_offset = fp8_dim
    bf16_out_ptr = token_bf16_ptr.to(tl.pointer_type(tl.bfloat16))
    for i in tl.static_range(bf16_dim // 16):
        chunk_offsets = i * 16 + tl.arange(0, 16)
        bf16_vals = tl.load(input_row_ptr + bf16_input_offset + chunk_offsets)
        tl.store(bf16_out_ptr + chunk_offsets, bf16_vals)


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
    )
