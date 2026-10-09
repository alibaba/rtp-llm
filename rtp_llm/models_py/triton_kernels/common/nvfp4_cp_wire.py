"""Model-local CP suffix wire v1: nine (64 code + 8 scale) byte records.

Quantization reads the existing BF16 rounding boundary. Receiver scatter never
converts scale bytes or requantizes. Persistent/working page ABIs are unchanged.
"""
import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.common.nvfp4_kv_cache import (
    _e2m1_encode, _scale_128x4_offset, _validate_cp_writer_planes,
    _TL_E2M1_MAX, _TL_SCALE_MIN, _TL_SCALE_MAX,
)

WIRE_BYTES = 648


@triton.jit
def _pack_wire(packed, wire, ROWS, INPUT_S0: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    plane = tl.program_id(1)
    pair = tl.arange(0, 8)[None, :]
    groups = tl.arange(0, 8)
    element = groups[:, None] * 16 + 2 * pair
    offset = row * INPUT_S0 + plane * 128 + element
    even_values = tl.load(packed + offset, mask=row < ROWS, other=0).to(tl.float32)
    odd_values = tl.load(packed + offset + 1, mask=row < ROWS, other=0).to(tl.float32)
    amax = tl.max(tl.maximum(tl.abs(even_values), tl.abs(odd_values)), axis=1)
    raw_scale = tl.minimum(
        tl.maximum(amax / _TL_E2M1_MAX, _TL_SCALE_MIN), _TL_SCALE_MAX
    )
    stored_scale = raw_scale.to(tl.float8e4nv)
    scale = stored_scale.to(tl.float32)[:, None]
    packed_codes = _e2m1_encode(even_values, scale) | (
        _e2m1_encode(odd_values, scale) << 4
    )
    base = row * 648 + plane * 72
    tl.store(wire + base + groups[:, None] * 8 + pair, packed_codes, mask=row < ROWS)
    tl.store(wire + base + 64 + groups,
             stored_scale.to(tl.uint8, bitcast=True), mask=row < ROWS)


@triton.jit
def _scatter_wire_multirow(wire, unpad, slots, persistent_slots,
                  k, ks, v, vs, idx, idxs, pk, pks, pv, pvs, pidx, pidxs,
                  ROWS, SOURCE_ROWS, WORK_PAGES, PERSIST_PAGES,
                  K_S0: tl.constexpr, S_S0: tl.constexpr,
                  I_S0: tl.constexpr, IS_S0: tl.constexpr,
                  PK_S0: tl.constexpr, PS_S0: tl.constexpr,
                  PI_S0: tl.constexpr, PIS_S0: tl.constexpr,
                  FI_WORKING_LAYOUT: tl.constexpr = False,
                  ROWS_PER_CTA: tl.constexpr = 8):
    row = (tl.program_id(0).to(tl.int64) * ROWS_PER_CTA
           + tl.arange(0, ROWS_PER_CTA).to(tl.int64))[:, None]
    plane = tl.program_id(1)
    is_k = plane < 4
    is_v = (plane >= 4) & (plane < 8)
    is_idx = plane == 8
    head = tl.where(is_k, plane, tl.where(is_v, plane - 4, 0))
    slot = tl.load(slots + row, mask=row < ROWS, other=-1).to(tl.int64)
    ps = tl.load(persistent_slots + row, mask=row < ROWS, other=-1).to(tl.int64)
    valid = (row < ROWS) & (slot >= 0) & (slot < WORK_PAGES * 128)
    pvalid = (row < ROWS) & (ps >= 0) & (ps < PERSIST_PAGES * 128)
    source = tl.load(unpad + row, mask=row < ROWS, other=0).to(tl.int64)
    source_valid = (source >= 0) & (source < SOURCE_ROWS)
    # Invalid source maps are contract failures, not a zero-filled fallback.
    tl.device_assert(~(valid | pvalid) | source_valid, "CP wire source row out of bounds")
    read = (valid | pvalid) & source_valid
    b = tl.arange(0, 64)[None, :]
    g = tl.arange(0, 8)[None, :]
    base = source * 648 + plane * 72
    codes = tl.load(wire + base + b, mask=read, other=0)
    scales = tl.load(wire + base + 64 + g, mask=read, other=0)
    block, off = slot // 128, slot % 128
    pblock, poff = ps // 128, ps % 128
    main_offset = block * K_S0 + (head * 128 + off) * 64 + b
    scale_offset = block * S_S0 + head * 128 * 8 + _scale_128x4_offset(off, g)
    if FI_WORKING_LAYOUT:
        # FI changes only working K/V scales; index and persistent remain MMA.
        scale_offset = block * S_S0 + head * 1024 + off * 8 + g
        v_scale_offset = (block * S_S0 + head * 1024
                          + ((off // 4) * 4 + g // 2) * 8
                          + (g % 2) * 4 + off % 4)
    else:
        v_scale_offset = scale_offset
    idx_offset = block * I_S0 + off * 64 + b
    idx_scale_offset = block * IS_S0 + _scale_128x4_offset(off, g)
    tl.store(k + main_offset, codes, mask=valid & source_valid & is_k)
    tl.store(v + main_offset, codes, mask=valid & source_valid & is_v)
    tl.store(ks + scale_offset, scales, mask=valid & source_valid & is_k)
    tl.store(vs + v_scale_offset, scales, mask=valid & source_valid & is_v)
    tl.store(idx + idx_offset, codes, mask=valid & source_valid & is_idx)
    tl.store(idxs + idx_scale_offset, scales, mask=valid & source_valid & is_idx)
    pmain_offset = pblock * PK_S0 + (head * 128 + poff) * 64 + b
    pscale_offset = pblock * PS_S0 + head * 128 * 8 + _scale_128x4_offset(poff, g)
    pidx_offset = pblock * PI_S0 + poff * 64 + b
    pidx_scale_offset = pblock * PIS_S0 + _scale_128x4_offset(poff, g)
    tl.store(pk + pmain_offset, codes, mask=pvalid & source_valid & is_k)
    tl.store(pv + pmain_offset, codes, mask=pvalid & source_valid & is_v)
    tl.store(pks + pscale_offset, scales, mask=pvalid & source_valid & is_k)
    tl.store(pvs + pscale_offset, scales, mask=pvalid & source_valid & is_v)
    tl.store(pidx + pidx_offset, codes, mask=pvalid & source_valid & is_idx)
    tl.store(pidxs + pidx_scale_offset, scales, mask=pvalid & source_valid & is_idx)


def pack_cp_nvfp4_wire(packed, out):
    """Encode fully produced BF16 padded rows into a caller-owned wire lease."""
    if (packed.ndim != 2 or packed.shape[1] != 1152
            or packed.dtype != torch.bfloat16 or packed.stride(1) != 1
            or packed.stride(0) < 1152):
        raise ValueError("CP wire input requires BF16 [rows,1152] with contiguous channels")
    if (out.shape != (packed.shape[0], WIRE_BYTES) or out.dtype != torch.uint8
            or not out.is_contiguous()):
        raise ValueError("CP wire output requires contiguous uint8 [rows,648]")
    if not packed.is_cuda or not out.is_cuda or packed.device != out.device:
        raise ValueError("CP wire input/output must share a CUDA device")
    if packed.shape[0] and packed.untyped_storage().data_ptr() == out.untyped_storage().data_ptr():
        raise ValueError("CP wire input/output must have independent storage")
    if packed.shape[0]:
        _pack_wire[(packed.shape[0], 9)](
            packed, out, packed.shape[0], INPUT_S0=packed.stride(0), num_warps=1)


def scatter_cp_nvfp4_wire(wire, unpad_indices, slots, persistent_slots,
                         working_planes, persistent_planes, *, fi_working_layout=False, rows_per_cta=16):
    """Copy wire bytes through existing logical maps into both page layouts."""
    if type(rows_per_cta) is not int or rows_per_cta not in (8, 16):
        raise ValueError("rows_per_cta must be 8 or 16")
    if (wire.ndim != 2 or wire.shape[1] != WIRE_BYTES
            or wire.dtype != torch.uint8 or not wire.is_contiguous()):
        raise ValueError("CP wire gathered input requires contiguous uint8 [rows,648]")
    if type(fi_working_layout) is not bool:
        raise ValueError("fi_working_layout must be a bool")
    maps = (unpad_indices, slots, persistent_slots)
    if any(t.ndim != 1 or t.dtype != torch.int64 or not t.is_contiguous() for t in maps):
        raise ValueError("CP wire maps require contiguous device int64")
    rows = slots.numel()
    if any(t.numel() != rows for t in maps):
        raise ValueError("CP wire maps must cover every logical row")
    working, persistent = tuple(working_planes), tuple(persistent_planes)
    pages = _validate_cp_writer_planes(working)
    persistent_pages = _validate_cp_writer_planes(persistent)
    if not wire.is_cuda or any(not t.is_cuda or t.device != wire.device
                              for t in (*maps, *working, *persistent)):
        raise ValueError("CP wire tensors must share one CUDA device")
    # Six planes can share a nonoverlapping parent; persistent may not alias
    # the working/input allocation, matching the original dual-writer guard.
    input_storage = {t.untyped_storage().data_ptr() for t in (wire, *maps) if t.numel()}
    if input_storage.intersection(t.untyped_storage().data_ptr() for t in working if t.numel()):
        raise ValueError("CP wire working destination must have independent input storage")
    source_storage = input_storage | {t.untyped_storage().data_ptr() for t in working if t.numel()}
    if source_storage.intersection(t.untyped_storage().data_ptr() for t in persistent if t.numel()):
        raise ValueError("CP wire persistent destination must have independent non-input storage")
    if rows:
        _scatter_wire_multirow[(triton.cdiv(rows, rows_per_cta), 9)](
            wire, *maps, *(t.view(torch.uint8) for t in (*working, *persistent)),
            rows, wire.shape[0], pages, persistent_pages,
            K_S0=working[0].stride(0), S_S0=working[1].stride(0),
            I_S0=working[4].stride(0), IS_S0=working[5].stride(0),
            PK_S0=persistent[0].stride(0), PS_S0=persistent[1].stride(0),
            PI_S0=persistent[4].stride(0), PIS_S0=persistent[5].stride(0),
            FI_WORKING_LAYOUT=fi_working_layout, ROWS_PER_CTA=rows_per_cta, num_warps=4, debug=True)
