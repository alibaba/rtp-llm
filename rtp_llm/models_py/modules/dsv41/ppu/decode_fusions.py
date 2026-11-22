"""Default-on PPU Decode fusions with the old paths as fallbacks."""

import os
from functools import lru_cache

import torch

from rtp_llm.models_py.modules.dsv41.ppu.quant_conversion import FP8_INV_MAX


def enabled(component):
    return (
        os.environ.get(
            "RTP_V41_PPU_DECODE_FUSE_" + component,
            os.environ.get("RTP_V41_PPU_DECODE_FUSIONS", "1"),
        )
        == "1"
    )


def compact_routes_enabled(engine):
    config = getattr(engine, "_fake_balance_config", None)
    return (
        enabled("ROUTES")
        and getattr(engine, "backend", None) == "megamoe"
        and config is not None
        and config[0]
    )


def pack_graph_top_k_enabled(engine, top_k):
    """Use compact native output only when no route override or mapping runs."""
    if os.environ.get("RTP_V41_PPU_MEGA_GATE_PACK_TOP_K") != "1":
        return False
    logical_experts = getattr(engine, "num_routed_experts", None)
    config = getattr(engine, "_fake_balance_config", None)
    return (
        (getattr(engine, "backend", None), getattr(engine, "precision", None))
        == ("megamoe", "w4a4")
        and type(logical_experts) is int
        and logical_experts > 0
        and logical_experts == getattr(engine, "num_physical_experts", None)
        and type(top_k) is int
        and top_k == getattr(engine, "top_k", None)
        and getattr(engine, "_eplb_layers", None) == {}
        and (config is None or config[0] is False)
    )


def _rows_supported(value, widths):
    return (
        type(value) is torch.Tensor
        and value.is_cuda
        and value.layout == torch.strided
        and value.dtype == torch.bfloat16
        and value.ndim == 2
        and value.shape[0] > 0
        and value.shape[1] in widths
        and value.is_contiguous()
        and not value.is_neg()
        and not value.is_conj()
        and not value.requires_grad
        and not torch.is_grad_enabled()
    )


@lru_cache(maxsize=1)
def _quant_kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def _store_row(
        values,
        q,
        scale,
        row,
        WIDTH: tl.constexpr,
        BLOCK: tl.constexpr,
        INV_MAX: tl.constexpr,
    ):
        columns = tl.arange(0, BLOCK)
        maximum = tl.maximum(tl.max(tl.abs(values), 0), 1.0e-10)
        inverse = tl.div_rn(1.0, maximum) * 448.0
        payload = tl.minimum(tl.maximum(values * inverse, -448.0), 448.0)
        tl.store(q + row * WIDTH + columns, payload, columns < WIDTH)
        tl.store(scale + row, maximum * INV_MAX)

    @triton.jit
    def silu_bf16_ptpc_quant(
        x,
        q,
        scale,
        WIDTH: tl.constexpr,
        LIMIT: tl.constexpr,
        BLOCK: tl.constexpr,
        INV_MAX: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        gate = tl.load(x + row * (2 * WIDTH) + columns, columns < WIDTH, 0).to(
            tl.float32
        )
        up = tl.load(x + row * (2 * WIDTH) + WIDTH + columns, columns < WIDTH, 0).to(
            tl.float32
        )
        if LIMIT > 0:
            gate = tl.where(gate > LIMIT, LIMIT, gate)
            up = tl.where(up > LIMIT, LIMIT, up)
            up = tl.where(up < -LIMIT, -LIMIT, up)
        # The legacy Graph path rounds the FP32 SwiGLU once into BF16 before
        # whole-row quantization. Do not adopt another provider's SiLU rounding.
        values = ((gate * tl.sigmoid(gate)) * up).to(tl.bfloat16).to(tl.float32)
        _store_row(values, q, scale, row, WIDTH, BLOCK, INV_MAX)

    @triton.jit
    def mask_ptpc_quant(
        x,
        valid,
        q,
        scale,
        WIDTH: tl.constexpr,
        BLOCK: tl.constexpr,
        INV_MAX: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        values = tl.load(x + row * WIDTH + columns, columns < WIDTH, 0).to(tl.float32)
        values = tl.where(tl.load(valid + row), values, 0.0)
        _store_row(values, q, scale, row, WIDTH, BLOCK, INV_MAX)

    return triton, silu_bf16_ptpc_quant, mask_ptpc_quant


def _allocate_quant(value, width):
    return (
        torch.empty(
            (value.shape[0], width), dtype=torch.float8_e4m3fn, device=value.device
        ),
        torch.empty((value.shape[0], 1), dtype=torch.float32, device=value.device),
    )


def try_shared_activation_quant(gate_up, limit):
    if not enabled("SHARED") or not _rows_supported(gate_up, (4608,)):
        return None
    if type(limit) not in (float, int) or limit < 0 or not float(limit) < float("inf"):
        return None
    payload, scale = _allocate_quant(gate_up, gate_up.shape[1] // 2)
    triton, kernel, _ = _quant_kernels()
    rows, width = payload.shape
    # A CTA owns a full row. The wider CTA reduces per-lane work on
    # the measured PPU Decode, Proposal and Verify buckets.
    num_warps = 4 if rows == 1 else 16
    kernel[(rows,)](
        gate_up,
        payload,
        scale,
        width,
        float(limit),
        triton.next_power_of_2(width),
        FP8_INV_MAX,
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    return payload, scale


def supports_mask_quant(linear, valid):
    from rtp_llm.models_py.modules.dsv41.ppu.linear import PPUPTPCLinear

    return (
        enabled("MASK")
        and type(linear) is PPUPTPCLinear
        and linear.weight.is_cuda
        and linear.weight.shape[1] == 8192
        and type(valid) is torch.Tensor
        and valid.is_cuda
        and valid.dtype == torch.bool
        and valid.ndim == 1
        and valid.numel() > 0
        and valid.is_contiguous()
        and valid.device == linear.weight.device
        and not torch.is_grad_enabled()
    )


def masked_output_projection(linear, output, valid):
    flat = output.flatten(1)
    if (not supports_mask_quant(linear, valid) or not _rows_supported(flat, (8192,))
            or flat.shape[0] != valid.numel()):
        raise ValueError("Decode masked PTPC projection changed its admitted metadata")
    payload, scale = _allocate_quant(flat, flat.shape[1])
    triton, _, kernel = _quant_kernels()
    kernel[(flat.shape[0],)](
        flat,
        valid,
        payload,
        scale,
        flat.shape[1],
        triton.next_power_of_2(flat.shape[1]),
        FP8_INV_MAX,
        num_warps=8,
        enable_fp_fusion=False,
    )
    return linear.forward_quantized(payload, scale)


@lru_cache(maxsize=1)
def _joint_slot_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def allocated(
        raw,
        row,
        table,
        COLUMNS: tl.constexpr,
        STRIDE_ROW: tl.constexpr,
        STRIDE_COL: tl.constexpr,
        SPAN: tl.constexpr,
        ENTRIES: tl.constexpr,
        RATIO: tl.constexpr,
        POOLS: tl.constexpr,
    ):
        page = raw // SPAN
        present = (raw >= 0) & (page < COLUMNS)
        block = tl.load(table + row * STRIDE_ROW + page * STRIDE_COL, present, 0).to(
            tl.int64
        )
        offset = (raw % SPAN) // RATIO
        live = present & (block > 0) & (block < POOLS) & (offset < ENTRIES)
        return tl.where(live, raw, -1)

    @triton.jit
    def joint_journal_slots(
        raw,
        table0,
        table1,
        out0,
        out1,
        COUNT: tl.constexpr,
        PER_ROW: tl.constexpr,
        LAYOUT0: tl.constexpr,
        LAYOUT1: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        position = tl.load(raw + index, index < COUNT, -1).to(tl.int64)
        row = index // PER_ROW
        value0 = allocated(
            position,
            row,
            table0,
            LAYOUT0[0],
            LAYOUT0[1],
            LAYOUT0[2],
            LAYOUT0[3],
            LAYOUT0[4],
            LAYOUT0[5],
            LAYOUT0[6],
        )
        value1 = allocated(
            position,
            row,
            table1,
            LAYOUT1[0],
            LAYOUT1[1],
            LAYOUT1[2],
            LAYOUT1[3],
            LAYOUT1[4],
            LAYOUT1[5],
            LAYOUT1[6],
        )
        tl.store(out0 + index, value0, index < COUNT)
        tl.store(out1 + index, value1, index < COUNT)

    return triton, joint_journal_slots


def try_joint_journal_slots(cache, owner, tag, raw):
    if not enabled("METADATA") or tag not in ("global_kv_1", "global_kv_2"):
        return None
    from rtp_llm.models_py.modules.dsv41.ppu.dspark_graph import VerifyGraphJournal
    from rtp_llm.models_py.modules.dsv41.ppu.verify_ops import _raw_supported

    if type(cache) is not VerifyGraphJournal:
        return None
    native = cache.native_cache
    tables, layouts = [], []
    for domain in (tag, "indexer_kv"):
        key = native._table_tag(owner, domain)
        table = cache.row_tables[key]
        span, entries = native.spans[key], native.entries[key]
        ratio, pools = native.geometry.ratio(owner), native.pool(owner, domain).shape[0]
        if not _raw_supported(raw, table, span, entries, ratio, True, pools):
            return None
        tables.append(table)
        layouts.append((table.shape[1], *table.stride(), span, entries, ratio, pools))
    outputs = torch.empty_like(raw), torch.empty_like(raw)
    if raw.numel():
        triton, kernel = _joint_slot_kernel()
        kernel[(triton.cdiv(raw.numel(), 128),)](
            raw,
            *tables,
            *outputs,
            raw.numel(),
            raw.numel() // raw.shape[0],
            *layouts,
            128,
            num_warps=4,
        )
    return outputs
