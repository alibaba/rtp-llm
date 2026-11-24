"""Packed NVFP4 rows and FP8 staging for GLM sparse attention.

The latent cache stores 512 E2M1 values, 32 E4M3 block scales, one FP32
row scale, and the unquantized 64-element BF16 RoPE tail. The indexer
uses the same convention for 128 elements, with no RoPE tail.
"""

import torch
import triton
import triton.language as tl

MLA_ROW_BYTES = 432
INDEXER_ROW_BYTES = 80
FP8_MLA_ROW_BYTES = 656
FP8_INDEXER_ROW_BYTES = 132
_BLOCK_VALUES = 16
_E2M1_MAX = 6.0
_E4M3_MAX = 448.0
_E4M3_MIN = 2.0**-9


def _row_bytes(width: int) -> int:
    if width == 512:
        return MLA_ROW_BYTES
    if width == 128:
        return INDEXER_ROW_BYTES
    raise ValueError(f"Unsupported NVFP4 KV width {width}")


def pack_nvfp4_rows(
    values: torch.Tensor, rope: torch.Tensor | None = None
) -> torch.Tensor:
    """Encode E2M1 in low-nibble-first order with one scale per 16 values."""
    width = values.shape[-1]
    stride = _row_bytes(width)
    prefix = values.shape[:-1]
    if width == 512:
        if rope is None or rope.shape != (*prefix, 64) or rope.dtype != torch.bfloat16:
            raise ValueError("NVFP4 MLA requires a BF16 RoPE tail of width 64")
    elif rope is not None:
        raise ValueError("NVFP4 indexer rows do not have a RoPE tail")

    groups = values.float().reshape(*prefix, width // _BLOCK_VALUES, _BLOCK_VALUES)
    maxima = groups.abs().amax(dim=-1)
    row_max = maxima.amax(dim=-1, keepdim=True)
    row_scale = torch.where(row_max > 0, row_max / (_E2M1_MAX * _E4M3_MAX), 1.0)
    row_scale = row_scale.clamp_min(torch.finfo(torch.float32).tiny)
    group_scale = (maxima / row_scale / _E2M1_MAX).clamp(_E4M3_MIN, _E4M3_MAX)
    group_scale = group_scale.to(torch.float8_e4m3fn)
    scale = (group_scale.float() * row_scale).clamp_min(torch.finfo(torch.float32).tiny)
    normalized = groups / scale.unsqueeze(-1)
    magnitude = normalized.abs().clamp_max(_E2M1_MAX)
    # E2M1 magnitudes are [0, .5, 1, 1.5, 2, 3, 4, 6]. Round-to-even
    # in each exponent range gives the correct rounding at their boundaries.
    bins = torch.where(
        magnitude <= 2,
        (magnitude * 2).round(),
        torch.where(magnitude < 4, magnitude.round() + 2, (magnitude / 2).round() + 4),
    ).to(torch.uint8)
    encoded = (bins | ((normalized < 0).to(torch.uint8) << 3)).reshape(*prefix, width)
    packed = encoded[..., 0::2] | (encoded[..., 1::2] << 4)

    result = torch.zeros((*prefix, stride), dtype=torch.uint8, device=values.device)
    scale_start = width // 2
    row_scale_start = scale_start + width // _BLOCK_VALUES
    result[..., :scale_start] = packed
    result[..., scale_start:row_scale_start] = group_scale.view(torch.uint8)
    result[..., row_scale_start : row_scale_start + 4] = row_scale.view(torch.uint8)
    if rope is not None:
        result[..., 292:420] = rope.contiguous().view(torch.uint8)
    return result


def unpack_nvfp4_rows(rows: torch.Tensor, width: int) -> torch.Tensor:
    """Decode to float32 before requantizing for the FP8 consumers."""
    if rows.dtype != torch.uint8 or rows.size(-1) != _row_bytes(width):
        raise ValueError(f"Expected packed uint8 NVFP4 KV rows for width {width}")
    rows = rows.contiguous()
    scale_start = width // 2
    row_scale_start = scale_start + width // _BLOCK_VALUES
    packed = rows[..., :scale_start]
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    code = (codes & 7).float()
    value = torch.where(code < 4, code / 2, torch.where(code < 7, code - 2, 6))
    value = torch.where(codes & 8 != 0, -value, value)
    blocks = rows[..., scale_start:row_scale_start].view(torch.float8_e4m3fn).float()
    row_scale = rows[..., row_scale_start : row_scale_start + 4].view(torch.float32)
    grouped = value.reshape(*rows.shape[:-1], width // _BLOCK_VALUES, _BLOCK_VALUES)
    return (grouped * (blocks * row_scale).unsqueeze(-1)).flatten(-2)


def unpack_nvfp4_rope(rows: torch.Tensor) -> torch.Tensor:
    if rows.dtype != torch.uint8 or rows.size(-1) != MLA_ROW_BYTES:
        raise ValueError("Expected packed uint8 NVFP4 MLA rows")
    return rows[..., 292:420].contiguous().view(torch.bfloat16)


def _fp8_quantize(
    values: torch.Tensor,
    group_width: int,
    scale_ue8m0: bool = False,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    groups = values.unflatten(-1, (-1, group_width))
    scale = groups.abs().amax(-1).clamp_min(eps) / _E4M3_MAX
    if scale_ue8m0:
        scale = scale.log2().ceil().exp2()
    quantized = (groups / scale.unsqueeze(-1)).clamp(-_E4M3_MAX, _E4M3_MAX)
    return quantized.flatten(-2).to(torch.float8_e4m3fn), scale.contiguous()


def nvfp4_mla_rows_to_fp8(rows: torch.Tensor) -> torch.Tensor:
    """Produce the FlashMLA 656B row format from gathered NVFP4 rows."""
    values, scale = _fp8_quantize(unpack_nvfp4_rows(rows, 512), 128)
    return torch.cat(
        (values.view(torch.uint8), scale.view(torch.uint8), rows[..., 292:420]), dim=-1
    )


@triton.jit
def _write_rows_kernel(
    cache,
    rows,
    slots,
    count: tl.constexpr,
    capacity: tl.constexpr,
    row_bytes: tl.constexpr,
    block_bytes: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(slots + row)
    offset = tl.arange(0, block_bytes)
    value = tl.load(rows + row * row_bytes + offset, offset < row_bytes, 0)
    tl.store(
        cache + slot.to(tl.int64) * row_bytes + offset,
        value,
        (slot >= 0) & (slot < capacity) & (offset < row_bytes),
    )


def write_nvfp4_rows(
    cache: torch.Tensor, rows: torch.Tensor, slots: torch.Tensor
) -> None:
    """Scatter rows while skipping negative padding slots used by CUDA graphs."""
    row_bytes = rows.shape[-1]
    cached_rows = cache.view(-1, row_bytes)
    rows = rows.reshape(-1, row_bytes).contiguous()
    slots = slots.flatten().contiguous()
    if cache.dtype != torch.uint8 or slots.numel() != rows.shape[0]:
        raise ValueError("NVFP4 KV write requires a byte cache and one slot per row")
    if slots.numel() == 0:
        return
    if slots.is_cuda:
        _write_rows_kernel[(slots.numel(),)](
            cached_rows,
            rows,
            slots,
            slots.numel(),
            cached_rows.shape[0],
            row_bytes,
            triton.next_power_of_2(row_bytes),
            num_warps=4,
        )
    else:
        valid = (slots >= 0) & (slots < cached_rows.shape[0])
        cached_rows.index_copy_(0, slots[valid].long(), rows[valid])


@triton.jit
def _decode_indexer_fp8_kernel(
    cache,
    cache_f32,
    slots,
    out_bytes,
    out_scales,
    count: tl.constexpr,
    capacity: tl.constexpr,
    paged: tl.constexpr,
    page_size: tl.constexpr,
    power2_scale: tl.constexpr,
    rows_per_block: tl.constexpr,
    mxfp8: tl.constexpr,
    source_layout: tl.constexpr,
):
    row = tl.program_id(0) * rows_per_block + tl.arange(0, rows_per_block)
    j = tl.arange(0, 128)
    source = tl.load(slots + row, row < count, -1).to(tl.int64)
    valid = (row < count) & (source >= 0) & (source < capacity)
    if source_layout == "nvfp4":
        base = source[:, None] * 80
        packed = tl.load(cache + base + j[None, :] // 2, valid[:, None], 0)
        nibble = (packed >> ((j[None, :] & 1) * 4)) & 15
        code = (nibble & 7).to(tl.float32)
        v = tl.where(code < 4, code * 0.5, tl.where(code < 7, code - 2, 6.0))
        v = tl.where((nibble & 8) != 0, -v, v)
        scale_byte = tl.load(cache + base + 64 + j[None, :] // 16, valid[:, None], 0)
        block_scale = scale_byte.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        row_scale = tl.load(cache_f32 + source * 20 + 18, valid, 1.0)
        v = v * (block_scale * row_scale[:, None])
    else:
        if source_layout == "paged_fp8":
            src_page = source // page_size
            src_pos = source % page_size
            src_data = src_page * (page_size * 132) + src_pos * 128
            src_sf = src_page * (page_size * 33) + page_size * 32 + src_pos
        else:
            src_data = source * 128
            src_sf = source
        raw = tl.load(cache + src_data[:, None] + j[None, :], valid[:, None], 0)
        sf = tl.load(cache_f32 + src_sf, valid, 1.0)
        v = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32) * sf[:, None]

    if mxfp8:
        groups = tl.reshape(v, (rows_per_block, 4, 32))
        maxima = tl.max(tl.abs(groups), 2)
        exponents = tl.clamp(
            tl.ceil(tl.log2(tl.maximum(maxima, 1e-4) / 448.0)), -127.0, 127.0
        )
        scale = tl.sum(
            (exponents + 127.0).to(tl.uint32) << (tl.arange(0, 4)[None, :] * 8),
            1,
        ).to(tl.int32, bitcast=True)
        normalized = tl.reshape(
            groups / tl.exp2(exponents)[:, :, None], (rows_per_block, 128)
        )
    else:
        scale = tl.maximum(tl.max(tl.abs(v), 1), 1e-4) / 448.0
        if power2_scale:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        normalized = v / scale[:, None]
    fp8 = tl.clamp(normalized, -448.0, 448.0).to(tl.float8e4nv)
    row_offset = row.to(tl.int64)
    if paged:
        page = row_offset // page_size
        pos = row_offset % page_size
        data_offset = page * (page_size * 132) + pos * 128
        scale_offset = page * (page_size * 33) + page_size * 32 + pos
    else:
        data_offset = row_offset * 128
        scale_offset = row
    tl.store(
        out_bytes + data_offset[:, None] + j[None, :],
        fp8.to(tl.uint8, bitcast=True),
        row[:, None] < count,
    )
    tl.store(out_scales + scale_offset, scale, row < count)


def _transcode_indexer(
    cache: torch.Tensor,
    slots: torch.Tensor,
    out_bytes: torch.Tensor,
    out_scales: torch.Tensor,
    scale_ue8m0: bool,
    page_size: int = 0,
    *,
    mxfp8: bool = False,
    source_nvfp4: bool = True,
) -> None:
    source = cache.view(
        -1, INDEXER_ROW_BYTES if source_nvfp4 else FP8_INDEXER_ROW_BYTES
    )
    slots = slots.flatten().contiguous()
    if slots.numel() == 0:
        return
    if cache.is_cuda:
        _decode_indexer_fp8_kernel[(triton.cdiv(slots.numel(), 4),)](
            source,
            source.view(torch.float32),
            slots,
            out_bytes,
            out_scales,
            slots.numel(),
            source.shape[0],
            bool(page_size),
            page_size or 1,
            scale_ue8m0,
            4,
            mxfp8,
            "nvfp4" if source_nvfp4 else "paged_fp8",
            num_warps=4,
        )
        return

    if mxfp8 or not source_nvfp4:
        raise ValueError("MXFP8 KV staging requires CUDA")
    valid = (slots >= 0) & (slots < source.shape[0])
    rows = source.index_select(0, slots.clamp(0, source.shape[0] - 1).long())
    values = torch.where(valid[:, None], unpack_nvfp4_rows(rows, 128), 0)
    fp8, scale = _fp8_quantize(values, 128, scale_ue8m0)
    if page_size:
        page_bytes = out_bytes.view(-1, page_size * FP8_INDEXER_ROW_BYTES)
        page_bytes[:, : page_size * 128].copy_(
            fp8.view(torch.uint8).reshape(-1, page_size * 128)
        )
        page_bytes[:, page_size * 128 :].view(torch.float32).copy_(
            scale.reshape(-1, page_size)
        )
    else:
        out_bytes.view(-1, 128).copy_(fp8.view(torch.uint8))
        out_scales.view(-1).copy_(scale.flatten())


def indexer_paged_fp8(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    page_size: int,
    scale_ue8m0: bool,
    *,
    mxfp8: bool = False,
    source_nvfp4: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage the candidate pages in the paged FP8 MQA layout.

    The page table dimensions are stable during graph replay. Only valid
    tokens indicated by the live causal lengths read from the resident cache.
    """
    pages = block_table.numel()
    local_pages = torch.arange(pages, dtype=torch.int32, device=cache.device)
    local_pages = local_pages.view_as(block_table)
    offsets = torch.arange(page_size, dtype=torch.int32, device=cache.device)
    page_offsets = (local_pages % block_table.shape[-1]) * page_size
    logical_positions = page_offsets.unsqueeze(-1) + offsets
    max_lens = context_lens.amax(dim=1).view(-1, 1, 1)
    mask = (logical_positions < max_lens) & (block_table.unsqueeze(-1) >= 0)
    physical = block_table.unsqueeze(-1) * page_size + offsets
    slots = torch.where(mask, physical, -1)
    output = torch.empty(
        (pages, page_size, 1, FP8_INDEXER_ROW_BYTES),
        dtype=torch.uint8,
        device=cache.device,
    )
    out_scales = output.view(torch.int32 if mxfp8 else torch.float32)
    _transcode_indexer(
        cache,
        slots,
        output,
        out_scales,
        scale_ue8m0,
        page_size,
        mxfp8=mxfp8,
        source_nvfp4=source_nvfp4,
    )
    return output, local_pages


def gather_indexer_fp8(
    cache: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_kv_seqlens: torch.Tensor,
    page_size: int,
    scale_ue8m0: bool,
    *,
    mxfp8: bool = False,
) -> None:
    """Read the exact ragged KV rows for a prefill/CP batch into FP8."""
    if not k_fp8.numel():
        return
    positions = torch.arange(k_fp8.shape[0], dtype=torch.int32, device=cache.device)
    batches = torch.searchsorted(cu_kv_seqlens[1:].contiguous(), positions, right=True)
    offsets = positions - cu_kv_seqlens[batches]
    valid = (batches < block_table.shape[0]) & (offsets >= 0)
    batches = batches.clamp(0, block_table.shape[0] - 1)
    page_idx = (offsets // page_size).clamp(0, block_table.shape[1] - 1)
    pages = block_table[batches, page_idx]
    valid = valid & (offsets // page_size < block_table.shape[1]) & (pages >= 0)
    slots = torch.where(valid, pages * page_size + offsets % page_size, -1)
    out_scales = k_scale.view(torch.int32 if mxfp8 else torch.float32)
    _transcode_indexer(
        cache, slots, k_fp8.view(torch.uint8), out_scales, scale_ue8m0, mxfp8=mxfp8
    )


def ragged_fp8_to_mxfp8(
    values: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage legacy FP8 cache rows for the Blackwell DeepGEMM 2.8.2 API."""
    count = values.size(0)
    output = torch.empty_like(values)
    scales = torch.empty(count, device=values.device, dtype=torch.int32)
    if count:
        slots = torch.arange(count, dtype=torch.int32, device=values.device)
        _decode_indexer_fp8_kernel[(triton.cdiv(count, 4),)](
            values.view(torch.uint8),
            scale,
            slots,
            output.view(torch.uint8),
            scales,
            count,
            count,
            False,
            1,
            False,
            4,
            True,
            "ragged_fp8",
            num_warps=4,
        )
    return output, scales
