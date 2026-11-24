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
        raise ValueError("NVFP4 MLA cache must have shape [pages, slots, 432]")
    return cache


@triton.jit
def _write_mla_rows(
    Rows,
    Slots,
    Cache,
    page_stride,
    slot_stride,
    pages,
    PAGE_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    slot = tl.load(Slots + token).to(tl.int64)
    if (slot >= 0) & (slot < pages * PAGE_SIZE):
        col = tl.arange(0, 512)
        value = tl.load(Rows + token * WIDTH + col, col < WIDTH, other=0)
        offset = slot // PAGE_SIZE * page_stride + slot % PAGE_SIZE * slot_stride
        tl.store(Cache + offset + col, value, col < WIDTH)


def write_nvfp4_mla_cache(
    compressed_kv: torch.Tensor,
    key_pe: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    cache = _paged_mla_rows(cache)
    if slots.numel() == 0:
        return
    if (
        compressed_kv.shape != (slots.numel(), 512)
        or key_pe.numel() != slots.numel() * 64
        or key_pe.shape[-1] != 64
    ):
        raise ValueError("NVFP4 MLA cache writes require NoPE 512 and RoPE 64")
    slots = slots.flatten().contiguous()
    key_pe = key_pe.reshape(-1, 64)
    # Bound quantization scratch on long prefills as well as attention scratch.
    for start in range(0, slots.numel(), MAX_NVFP4_MLA_SELECTED_ROWS // 8):
        end = min(start + MAX_NVFP4_MLA_SELECTED_ROWS // 8, slots.numel())
        rows = pack_nvfp4_rows(
            compressed_kv[start:end], key_pe[start:end].to(torch.bfloat16)
        )
        _write_mla_rows[(end - start,)](
            rows,
            slots[start:end],
            cache,
            cache.stride(0),
            cache.stride(1),
            cache.shape[0],
            cache.shape[1],
            _NVFP4_ROW_BYTES,
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
    Remapped,
    page_stride,
    slot_stride,
    pages,
    PAGE_SIZE: tl.constexpr,
    OUT_WIDTH: tl.constexpr,
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
    global_scale = tl.load(GlobalScales + base // 4 + 72, valid, other=0)
    value = value * (group_scale.to(tl.float32) * global_scale)
    scale = tl.maximum(tl.max(tl.abs(value), 0), 1.0e-4) / 448.0
    quantized = tl.clamp(tl.div_rn(value, scale), -448.0, 448.0).to(tl.float8e4nv)
    tl.store(Bytes + row * OUT_WIDTH + col, quantized.to(tl.uint8, bitcast=True))
    tl.store(Fp8Scales + row * (OUT_WIDTH // 4) + 128 + group, scale)

    if group == 0:
        rope_col = tl.arange(0, 128)
        rope = tl.load(Packed + base + 292 + rope_col, valid, other=0)
        tl.store(Bytes + row * OUT_WIDTH + 528 + rope_col, rope)
        tl.store(Remapped + row, tl.where(valid, row, -1).to(tl.int32))


def gather_selected_mla_fp8(
    cache: torch.Tensor, physical_indices: torch.Tensor
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
            remapped,
            cache.stride(0),
            cache.stride(1),
            cache.shape[0],
            cache.shape[1],
            _FP8_ROW_BYTES,
        )
    return output, remapped
