"""Packed NVFP4 MLA KV writes and selected-token FP8 views for FlashMLA."""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.kernels.cuda.nvfp4_kv import FP8_MLA_ROW_BYTES as _FP8_ROW_BYTES
from rtp_llm.models_py.kernels.cuda.nvfp4_kv import MLA_ROW_BYTES as _NVFP4_ROW_BYTES
from rtp_llm.models_py.kernels.cuda.nvfp4_kv import pack_nvfp4_rows

_FLASH_MLA_PAGE_SIZE = 64
MAX_NVFP4_MLA_SELECTED_ROWS = 128 * 1024


def _paged_mla_rows(cache: torch.Tensor) -> torch.Tensor:
    # LayerKVCache can expose a singleton head dimension in the workspace.
    if cache.ndim == 4 and cache.shape[2] == 1:
        cache = cache.squeeze(2)
    if (
        cache.ndim != 3
        or cache.shape[-1] != _NVFP4_ROW_BYTES
        or cache.dtype != torch.uint8
        or cache.stride(-1) != 1
    ):
        raise ValueError(
            f"NVFP4 MLA cache must have shape [pages, slots, {_NVFP4_ROW_BYTES}]"
        )
    return cache


@triton.jit
def _encode_nvfp4_blocks(vals_f32, NUM_BLOCKS: tl.constexpr):
    abs_vals = tl.abs(vals_f32)
    is_neg = (vals_f32.to(tl.uint32, bitcast=True) >> 31) != 0
    maxima = tl.max(abs_vals, axis=1)
    row_max = tl.max(maxima, axis=0)
    # Match PyTorch CUDA's tensor / 2688.0 reciprocal-multiply lowering used in QAD training.
    row_scale = row_max * (1.0 / 2688.0)
    row_scale = tl.where(row_scale == 0.0, 1.0, row_scale)
    raw_scale = tl.div_rn(maxima, 6.0 * row_scale)
    raw_scale = tl.where(raw_scale == 0.0, 1.0, raw_scale)
    raw_scale = tl.minimum(tl.maximum(raw_scale, 1.0 / 512.0), 448.0)
    group_scale = raw_scale.to(tl.float8e4nv)
    eff_scale = group_scale.to(tl.float32) * row_scale
    scaled_abs = tl.div_rn(abs_vals, eff_scale[:, None])

    bins = tl.where(scaled_abs > 5.0, 7, 6)
    bins = tl.where(scaled_abs < 3.5, 5, bins)
    bins = tl.where(scaled_abs <= 2.5, 4, bins)
    bins = tl.where(scaled_abs < 1.75, 3, bins)
    bins = tl.where(scaled_abs <= 1.25, 2, bins)
    bins = tl.where(scaled_abs < 0.75, 1, bins)
    bins = tl.where(scaled_abs <= 0.25, 0, bins).to(tl.uint8)
    sign_bits = tl.where(is_neg & (bins != 0), 8, 0).to(tl.uint8)
    codes = tl.reshape(bins | sign_bits, (NUM_BLOCKS * 8, 2))
    lo, hi = tl.split(codes)
    packed = lo | (hi << 4)
    scale_u8 = group_scale.to(tl.uint8, bitcast=True)
    rscale_bits = row_scale.to(tl.uint32, bitcast=True)
    rscale_u8 = ((rscale_bits >> (tl.arange(0, 4) * 8)) & 0xFF).to(tl.uint8)
    return packed, scale_u8, rscale_u8


@triton.jit
def _pack_and_write_mla_rows(
    CompressedKv,
    KeyPe,
    Slots,
    Cache,
    kv_stride_0,
    kv_stride_1,
    kpe_stride_0,
    kpe_stride_1,
    page_stride,
    slot_stride,
    pages,
    PAGE_SIZE: tl.constexpr,
    DEINTERLEAVE_ROPE: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    slot = tl.load(Slots + token).to(tl.int64)
    if (slot < 0) | (slot >= pages * PAGE_SIZE):
        return
    base = slot // PAGE_SIZE * page_stride + slot % PAGE_SIZE * slot_stride

    kv_cols = tl.arange(0, 512)
    kv_vals = tl.load(
        CompressedKv + token * kv_stride_0 + kv_cols * kv_stride_1
    ).to(tl.float32)
    kv_packed, kv_scale, kv_rscale = _encode_nvfp4_blocks(
        tl.reshape(kv_vals, (32, 16)), 32
    )
    tl.store(Cache + base + tl.arange(0, 256), kv_packed)
    tl.store(Cache + base + 256 + tl.arange(0, 32), kv_scale)
    tl.store(Cache + base + 288 + tl.arange(0, 4), kv_rscale)

    d = tl.arange(0, 64)
    src_d = tl.where(d < 32, d * 2, (d - 32) * 2 + 1) if DEINTERLEAVE_ROPE else d
    kpe_vals = (
        tl.load(KeyPe + token * kpe_stride_0 + src_d * kpe_stride_1)
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    kpe_packed, kpe_scale, kpe_rscale = _encode_nvfp4_blocks(
        tl.reshape(kpe_vals, (4, 16)), 4
    )
    tl.store(Cache + base + 292 + tl.arange(0, 32), kpe_packed)
    tl.store(Cache + base + 324 + tl.arange(0, 4), kpe_scale)
    tl.store(Cache + base + 328 + tl.arange(0, 4), kpe_rscale)
    tl.store(
        Cache + base + 332 + tl.arange(0, 4),
        tl.zeros((4,), dtype=tl.uint8),
    )


def write_nvfp4_mla_cache(
    compressed_kv: torch.Tensor,
    key_pe: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
    *,
    deinterleave_rope: bool = True,
) -> None:
    cache = _paged_mla_rows(cache)
    num_tokens = slots.numel()
    if num_tokens == 0:
        return
    if (
        compressed_kv.shape != (num_tokens, 512)
        or key_pe.numel() != num_tokens * 64
        or key_pe.shape[-1] != 64
    ):
        raise ValueError("NVFP4 MLA cache writes require NoPE 512 and RoPE 64")
    slots = slots.flatten().contiguous()
    if key_pe.ndim == 3 and key_pe.shape[1] == 1:
        key_pe = key_pe.squeeze(1)
    elif key_pe.ndim != 2:
        key_pe = key_pe.reshape(num_tokens, 64)
    if not cache.is_cuda:
        rows = pack_nvfp4_rows(
            compressed_kv,
            key_pe.to(torch.bfloat16),
            deinterleave_rope=deinterleave_rope,
        )
        flat_cache = cache.view(-1, _NVFP4_ROW_BYTES)
        valid = (slots >= 0) & (slots < flat_cache.shape[0])
        flat_cache.index_copy_(0, slots[valid].long(), rows[valid])
        return
    _pack_and_write_mla_rows[(num_tokens,)](
        compressed_kv,
        key_pe,
        slots,
        cache,
        compressed_kv.stride(0),
        compressed_kv.stride(1),
        key_pe.stride(0),
        key_pe.stride(1),
        cache.stride(0),
        cache.stride(1),
        cache.shape[0],
        cache.shape[1],
        deinterleave_rope,
        num_warps=4,
    )


@triton.jit
def _gather_mla_rows(
    Cache,
    Slots,
    Rows,
    page_stride,
    slot_stride,
    pages,
    PAGE_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(Slots + row).to(tl.int64)
    valid = (slot >= 0) & (slot < pages * PAGE_SIZE)
    col = tl.arange(0, 512)
    offset = slot // PAGE_SIZE * page_stride + slot % PAGE_SIZE * slot_stride
    values = tl.load(Cache + offset + col, valid & (col < WIDTH), other=0)
    tl.store(Rows + row * WIDTH + col, values, col < WIDTH)


def gather_padded_nvfp4_mla_rows(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    actual_lengths: torch.Tensor,
    padded_lengths: torch.Tensor,
    starts: torch.Tensor,
    total_rows: int,
) -> torch.Tensor:
    """Gather valid owned CP tokens and leave each request's padding zero."""
    cache = _paged_mla_rows(cache)
    output = torch.empty(
        (total_rows, _NVFP4_ROW_BYTES), dtype=torch.uint8, device=block_table.device
    )
    if total_rows == 0:
        return output
    if block_table.shape[1] == 0:
        return output.zero_()
    req = torch.repeat_interleave(
        torch.arange(block_table.shape[0], device=block_table.device),
        padded_lengths,
        output_size=total_rows,
    )
    pos = torch.arange(total_rows, device=req.device) - starts.index_select(0, req)
    page = torch.div(pos, cache.shape[1], rounding_mode="floor")
    block = block_table[req, page.clamp(0, block_table.shape[1] - 1)].to(torch.int64)
    valid = (pos >= 0) & (pos < actual_lengths.index_select(0, req))
    valid &= (page < block_table.shape[1]) & (block >= 0)
    slots = torch.where(valid, block * cache.shape[1] + pos % cache.shape[1], -1)
    _gather_mla_rows[(total_rows,)](
        cache,
        slots,
        output,
        cache.stride(0),
        cache.stride(1),
        cache.shape[0],
        cache.shape[1],
        _NVFP4_ROW_BYTES,
    )
    return output


@triton.jit
def _selected_mla_rows_to_fp8(
    Packed,
    GlobalScales,
    Indices,
    Fp8Scales,
    Bytes,
    OutBf16,
    Remapped,
    page_stride,
    slot_stride,
    pages,
    PAGE_SIZE: tl.constexpr,
    OUT_WIDTH: tl.constexpr,
    DEINTERLEAVE_ROPE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1)
    slot = tl.load(Indices + row).to(tl.int64)
    valid = (slot >= 0) & (slot < pages * PAGE_SIZE)
    base = slot // PAGE_SIZE * page_stride + slot % PAGE_SIZE * slot_stride
    col = group * 128 + tl.arange(0, 128)

    packed = tl.load(Packed + base + col // 2, valid, other=0).to(tl.int32)
    code = (packed >> (4 * (col % 2))) & 15
    magnitude = code & 7
    value = tl.where(
        magnitude < 4,
        magnitude.to(tl.float32) / 2,
        tl.where(magnitude == 7, 6, magnitude - 2).to(tl.float32),
    )
    value = tl.where((code & 8) != 0, -value, value)
    scale_bits = tl.load(Packed + base + 256 + col // 16, valid, other=0)
    group_scale = scale_bits.to(tl.float8e4nv, bitcast=True)
    global_scale = tl.load(GlobalScales + base // 4 + 72, valid, other=0.0)
    value = (
        (value * (group_scale.to(tl.float32) * global_scale))
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    raw_scale = tl.maximum(tl.max(tl.abs(value), 0), 1.0e-4) / 448.0
    raw_bits = raw_scale.to(tl.int32, bitcast=True)
    ue8m0_bits = (raw_bits & 0x7F800000) + tl.where(
        (raw_bits & 0x007FFFFF) != 0, 0x00800000, 0
    )
    scale = ue8m0_bits.to(tl.float32, bitcast=True)
    quantized = tl.clamp(tl.div_rn(value, scale), -448.0, 448.0).to(tl.float8e4nv)
    tl.store(Bytes + row * OUT_WIDTH + col, quantized.to(tl.uint8, bitcast=True))
    tl.store(Fp8Scales + row * (OUT_WIDTH // 4) + 128 + group, scale)

    if group == 0:
        j = tl.arange(0, 64)
        src_idx = ((j >> 1) + (j & 1) * 32) if DEINTERLEAVE_ROPE else j
        r_packed = tl.load(Packed + base + 292 + src_idx // 2, valid, other=0).to(
            tl.int32
        )
        r_code = (r_packed >> (4 * (src_idx & 1))) & 15
        r_mag = r_code & 7
        r_val = tl.where(
            r_mag < 4,
            r_mag.to(tl.float32) / 2,
            tl.where(r_mag == 7, 6, r_mag - 2).to(tl.float32),
        )
        r_val = tl.where((r_code & 8) != 0, -r_val, r_val)
        r_sbits = tl.load(Packed + base + 324 + src_idx // 16, valid, other=0)
        r_bscale = r_sbits.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        r_gscale = tl.load(GlobalScales + base // 4 + 82, valid, other=0.0)
        r_out = tl.where(valid, r_val * (r_bscale * r_gscale), 0.0).to(tl.bfloat16)
        tl.store(OutBf16 + row * (OUT_WIDTH // 2) + 264 + j, r_out)
        tl.store(Remapped + row, tl.where(valid, row, -1).to(tl.int32))


def gather_selected_mla_fp8(
    cache: torch.Tensor,
    physical_indices: torch.Tensor,
    *,
    deinterleave_rope: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transcode only the addressed rows and remap indices to their scratch slots."""
    cache = _paged_mla_rows(cache)
    if cache.stride(0) % 4 or cache.stride(1) % 4:
        raise ValueError("NVFP4 MLA cache rows must be 4-byte aligned")
    physical_indices = physical_indices.contiguous()
    selected = physical_indices.numel()
    pages = max(1, triton.cdiv(selected, _FLASH_MLA_PAGE_SIZE))
    output = torch.empty(
        (pages, _FLASH_MLA_PAGE_SIZE, 1, _FP8_ROW_BYTES),
        dtype=torch.uint8,
        device=physical_indices.device,
    )
    remapped = torch.empty_like(physical_indices, dtype=torch.int32)
    if selected:
        _selected_mla_rows_to_fp8[(selected, 4)](
            cache,
            cache.view(torch.float32),
            physical_indices,
            output.view(torch.float32),
            output,
            output.view(torch.bfloat16),
            remapped,
            cache.stride(0),
            cache.stride(1),
            cache.shape[0],
            cache.shape[1],
            _FP8_ROW_BYTES,
            deinterleave_rope,
        )
    return output, remapped
