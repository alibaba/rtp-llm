"""MODEL1 sparse attention with the official 64-head reduction order."""

import os

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

# MODEL1 packed page sizes supported by the native decode kernel.
MODEL1_NATIVE_PAGES = (1, 2, 32, 64, 128, 130, 132, 134)


@triton.jit
def _schedule(
    Schedule,
    Splits,
    Lengths,
    ExtraLengths,
    Batch: tl.constexpr,
    Width: tl.constexpr,
    ExtraWidth: tl.constexpr,
    HasLengths: tl.constexpr,
    HasExtraLengths: tl.constexpr,
):
    row = tl.program_id(0)
    length = tl.load(Lengths + row) if HasLengths else Width
    if ExtraWidth:
        extra = tl.load(ExtraLengths + row) if HasExtraLengths else ExtraWidth
        end = tl.cdiv(length, 64) * 64 + extra
    else:
        end = length
    # Empty/fake requests still run one masked tile and produce zero output.
    end = tl.maximum(end, 1)
    columns = tl.arange(0, 8)
    values = tl.where(
        (columns == 0) | (columns == 2), row, tl.where(columns == 3, end, 0)
    )
    tl.store(Schedule + row * 8 + columns, values)
    tl.store(Splits + row, row)
    if row == 0:
        tl.store(Splits + Batch, Batch)


@triton.jit
def native_sum64(p):
    lanes = tl.arange(0, 8)
    partial = tl.full((16, 8), 0, tl.float32)
    for j in tl.static_range(8):
        columns = (lanes // 4) * 32 + (lanes % 4) * 2 + (j % 4) * 8 + j // 4
        partial = partial + tl.gather(p, tl.broadcast_to(columns[None, :], (16, 8)), 1)
    for offset in tl.static_range(3):
        partner = lanes ^ (4 >> offset)
        partial = partial + tl.gather(
            partial, tl.broadcast_to(partner[None, :], (16, 8)), 1
        )
    return tl.sum(tl.where(lanes[None, :] == 0, partial, 0.0), 1)


@triton.jit
def sparse_dual(
    Q,
    KV,
    KV2,
    Sink,
    Idx,
    Idx2,
    Lengths,
    Lengths2,
    O,
    H: tl.constexpr,
    D: tl.constexpr,
    WA: tl.constexpr,
    WB: tl.constexpr,
    SCALE: tl.constexpr,
    PAGE: tl.constexpr,
    PAGE2: tl.constexpr,
    BLOCK_STRIDE: tl.constexpr,
    STRIDE2: tl.constexpr,
    QLEN: tl.constexpr,
    DIV_MODE: tl.constexpr,
    HAS_LENGTHS: tl.constexpr = True,
    HAS_LENGTHS2: tl.constexpr = True,
    BF16_KV: tl.constexpr = False,
):
    t = tl.program_id(0)
    hh = tl.program_id(1) * 16 + tl.arange(0, 16)
    dd = tl.arange(0, D)
    kk = tl.arange(0, 64)
    q = tl.load(
        Q + t * H * D + hh[:, None] * D + dd[None, :],
        (hh[:, None] < H) if H < 16 else None,
        other=0 if H < 16 else None,
    )
    maximum = tl.full((16,), float("-inf"), tl.float32)
    denom = tl.zeros((16,), tl.float32)
    acc = tl.zeros((16, D), tl.float32)
    length = tl.load(Lengths + t // QLEN) if HAS_LENGTHS else WA
    length2 = tl.load(Lengths2 + t // QLEN) if HAS_LENGTHS2 else WB
    blocks = triton.cdiv(WA + WB, 64)
    if BF16_KV and HAS_LENGTHS and H == 8 and WB == 0:
        blocks = tl.maximum(1, tl.cdiv(tl.minimum(length, WA), 64))
    for block in range(blocks):
        column = block * 64 + kk
        use_first = column < WA
        ia = tl.load(Idx + t * WA + column, (column < WA) & (column < length), other=-1)
        ib = tl.load(
            Idx2 + t * WB + column - WA,
            (column >= WA) & (column < WA + WB) & (column - WA < length2),
            other=-1,
        )
        idx = tl.where(use_first, ia, ib)
        if BF16_KV:
            kv = tl.load(
                KV + idx[:, None] * D + dd[None, :], idx[:, None] >= 0, other=0
            )
        else:
            words = tl.where(
                use_first[:, None],
                KV.to(tl.pointer_type(tl.uint16)),
                KV2.to(tl.pointer_type(tl.uint16)),
            )
            page_id = tl.where(use_first, idx // PAGE, idx // PAGE2)
            row = tl.where(use_first, idx % PAGE, idx % PAGE2)
            stride = tl.where(use_first, BLOCK_STRIDE // 2, STRIDE2 // 2)
            page = tl.where(use_first, PAGE, PAGE2)
            packed = words + page_id[:, None] * stride[:, None] + row[:, None] * 288
            pair = tl.load(
                packed + dd[None, :] // 2,
                (idx[:, None] >= 0) & (dd[None, :] < 448),
                other=0,
            )
            byte = ((pair >> (8 * (dd[None, :] % 2))) & 255).to(tl.uint8)
            group = dd[None, :] // 64
            scales = tl.load(
                words
                + page_id[:, None] * stride[:, None]
                + page[:, None] * 288
                + row[:, None] * 4
                + group // 2,
                (idx[:, None] >= 0) & (dd[None, :] < 448),
                other=32639,
            )
            scale = ((scales >> (8 * (group % 2))) & 255).to(tl.float32)
            fp8 = byte.to(tl.float8e4nv, bitcast=True).to(tl.float32) * tl.exp2(
                scale - 127.0
            )
            rope = tl.load(
                packed + 224 + dd[None, :] - 448,
                (idx[:, None] >= 0) & (dd[None, :] >= 448),
                other=0,
            ).to(tl.bfloat16, bitcast=True)
            kv = tl.where(dd[None, :] < 448, fp8, rope.to(tl.float32)).to(tl.bfloat16)
        score = tl.dot(q, tl.trans(kv)) * SCALE
        score = tl.where(idx[None, :] >= 0, score, float("-inf"))
        newmax = tl.maximum(maximum, tl.max(score, 1))
        safe_max = tl.where(newmax == float("-inf"), 0.0, newmax)
        alpha = libdevice.exp(maximum - safe_max)
        p = libdevice.exp(score - safe_max[:, None])
        denom = tl.fma(denom, alpha, native_sum64(p))
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), kv, acc)
        maximum = newmax
    sink = tl.load(Sink + hh, (hh < H) if H < 16 else None, other=0 if H < 16 else None)
    denom += libdevice.exp(sink - maximum)
    if DIV_MODE == 1:
        out = tl.div_rn(acc, denom[:, None])
    elif DIV_MODE == 2:
        reciprocal = tl.div_rn(1.0, denom)
        quotient = acc * reciprocal[:, None]
        residual = tl.fma(-quotient, denom[:, None], acc)
        out = tl.fma(residual, reciprocal[:, None], quotient)
    else:
        out = acc / denom[:, None]
    out = tl.where(denom[:, None] == float("inf"), 0.0, out)
    tl.store(
        O + t * H * D + hh[:, None] * D + dd[None, :],
        out,
        (hh[:, None] < H) if H < 16 else None,
    )


def forward(
    q,
    kv_cache,
    attn_sink,
    topk_idxs,
    sched_meta=None,
    cache_seqlens=None,
    block_table=None,
    topk_length=None,
    extra_k_cache=None,
    extra_topk_idxs=None,
    extra_topk_length=None,
):
    # Preserve the official 64-head reduction order on MODEL1 packed pools.
    if q.ndim != 4 or q.shape[2:] != (64, 512) or q.dtype != torch.bfloat16:
        raise ValueError("expected BF16 [B,Q,64,512]")
    if not q.is_cuda or not q.is_contiguous():
        raise ValueError("expected contiguous device queries")
    b, nq, h, d = q.shape

    def check(t, dtype):
        if t.device != q.device or t.dtype != dtype or not t.is_contiguous():
            raise ValueError("tensor dtype/device/contiguity mismatch")

    check(attn_sink, torch.float32)
    if attn_sink.shape != (h,):
        raise ValueError("sink shape mismatch")

    def pool(cache, idx, length):
        if (
            cache.ndim != 3
            or cache.shape[2] != 584
            or cache.shape[1] not in MODEL1_NATIVE_PAGES
        ):
            raise ValueError("expected MODEL1 [blocks,page,584]")
        if cache.dtype != torch.uint8 or cache.device != q.device:
            raise ValueError("cache dtype/device mismatch")
        if (
            cache.stride(2) != 1
            or cache.stride(1) != 584
            or cache.stride(0) < cache.shape[1] * 584
            or cache.stride(0) % 2
            or cache.data_ptr() % 2
        ):
            raise ValueError("unsupported packed cache layout")
        if idx.ndim != 3 or idx.shape[:2] != (b, nq):
            raise ValueError("indices shape mismatch")
        check(idx, torch.int32)
        if length is not None:
            check(length, torch.int32)
            if length.shape != (b,):
                raise ValueError("lengths must be per batch")

    pool(kv_cache, topk_idxs, topk_length)
    if (extra_k_cache is None) != (extra_topk_idxs is None):
        raise ValueError("secondary cache and indices must be paired")
    if extra_k_cache is None:
        if extra_topk_length is not None:
            raise ValueError("secondary lengths without pool")
        extra_k_cache, extra_topk_idxs = kv_cache, topk_idxs
        wb = 0
    else:
        pool(extra_k_cache, extra_topk_idxs, extra_topk_length)
        wb = extra_topk_idxs.shape[-1]
    if (
        (
            b in (1, 2)
            or (b == 8 and os.getenv("DSV4_PPU_FUSED_COMPACT_MATERIALIZE") == "1")
        )
        and nq == 3
        and topk_idxs.shape[-1] == 128
        and kv_cache.shape[1] == 130
        and kv_cache.stride(0) == 76032
        and (wb, extra_k_cache.shape[1], extra_k_cache.stride(0))
        in ((512, 64, 37440), (384, 2, 1728))
        and topk_length is not None
        and extra_topk_length is not None
    ):
        return _forward_materialized_csa(
            q,
            kv_cache,
            extra_k_cache,
            attn_sink,
            topk_idxs,
            extra_topk_idxs,
            topk_length,
            extra_topk_length,
        )
    out = torch.empty_like(q)
    if b * nq:
        sparse_dual[(b * nq, h // 16)](
            q,
            kv_cache,
            extra_k_cache,
            attn_sink,
            topk_idxs,
            extra_topk_idxs,
            topk_length,
            extra_topk_length,
            out,
            h,
            d,
            topk_idxs.shape[-1],
            wb,
            512**-0.5,
            kv_cache.shape[1],
            extra_k_cache.shape[1],
            kv_cache.stride(0),
            extra_k_cache.stride(0),
            nq,
            2,
            topk_length is not None,
            extra_topk_length is not None,
            num_warps=4,
            num_stages=2,
            enable_fp_fusion=False,
        )
    return out


def forward_prefill(
    q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None, out=None
):
    """Use the native reduction order with contiguous BF16 Prefill KV."""
    if q.ndim != 3 or q.shape[-1] != 512 or q.shape[1] not in (8, 16, 32, 64, 128):
        raise ValueError("expected BF16 [M,H,512] with 8, 16, 32, 64, or 128 heads")
    rows, heads, dim = q.shape

    def check(tensor, dtype):
        if (
            tensor is None
            or tensor.device != q.device
            or tensor.dtype != dtype
            or not tensor.is_contiguous()
        ):
            raise ValueError("Prefill dtype/device/contiguity mismatch")

    if not q.is_cuda or d_v != dim:
        raise ValueError("requires device queries and output head_dim 512")
    check(q, torch.bfloat16)
    check(kv, torch.bfloat16)
    check(indices, torch.int32)
    check(attn_sink, torch.float32)
    if (
        kv.ndim != 3
        or kv.shape[1:] != (1, dim)
        or indices.ndim != 3
        or indices.shape[:2] != (rows, 1)
        or attn_sink.shape != (heads,)
    ):
        raise ValueError("Prefill KV/indices/sink shape mismatch")
    if topk_length is not None:
        check(topk_length, torch.int32)
        if topk_length.shape != (rows,):
            raise ValueError("Prefill lengths must be per Q row")
    if out is None:
        out = torch.empty_like(q)
    else:
        check(out, torch.bfloat16)
        if out.shape != q.shape:
            raise ValueError("Prefill output shape mismatch")
    if rows:
        sparse_dual[(rows, triton.cdiv(heads, 16))](
            q,
            kv,
            kv,
            attn_sink,
            indices,
            indices,
            topk_length,
            None,
            out,
            heads,
            dim,
            indices.shape[-1],
            0,
            sm_scale,
            1,
            1,
            dim * 2,
            dim * 2,
            1,
            2,
            topk_length is not None,
            False,
            BF16_KV=True,
            num_warps=4,
            num_stages=2,
            enable_fp_fusion=False,
        )
    return out, None, None


def _forward_materialized_csa(
    q,
    swa_cache,
    csa_cache,
    sink,
    swa_indices,
    csa_indices,
    swa_lengths,
    csa_lengths,
):
    from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
        dequantize_and_gather_k_cache_slots,
    )

    rows = q.shape[0] * q.shape[1]
    swa_width, csa_width = swa_indices.shape[-1], csa_indices.shape[-1]
    width = swa_width + csa_width
    swa_columns = torch.arange(swa_width, device=q.device)
    csa_columns = torch.arange(csa_width, device=q.device)
    swa_slots = swa_indices.masked_fill(swa_columns >= swa_lengths[:, None, None], -1)
    csa_slots = csa_indices.masked_fill(csa_columns >= csa_lengths[:, None, None], -1)
    swa_slots = swa_slots.view(rows, swa_width)
    csa_slots = csa_slots.view(rows, csa_width)
    workspace = torch.empty((rows, width, 512), device=q.device, dtype=torch.bfloat16)
    dequantize_and_gather_k_cache_slots(workspace, swa_cache, swa_slots, None, 0)
    dequantize_and_gather_k_cache_slots(
        workspace, csa_cache, csa_slots, None, swa_width
    )

    # Retain the original tile order and holes, including duplicates.
    valid = torch.cat((swa_slots >= 0, csa_slots >= 0), dim=-1)
    dense_indices = torch.arange(rows * width, device=q.device, dtype=torch.int32)
    dense_indices = dense_indices.view(rows, 1, width).masked_fill(~valid[:, None], -1)
    output, _, _ = forward_prefill(
        q.view(rows, 64, 512),
        workspace.view(-1, 1, 512),
        dense_indices,
        512**-0.5,
        attn_sink=sink,
    )
    return output.view_as(q)


from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
    _trap_invalid_kv_access,
    trap_invalid_kv_access_enabled,
    validate_slot_mapping,
)


@triton.jit
def _emit_compact_q3(
    SWA,
    CMP,
    SWA_LEN,
    CMP_LEN,
    STARTS,
    SLOTS_SWA,
    SLOTS_CMP,
    DENSE,
    B: tl.constexpr,
    CW: tl.constexpr,
    HCA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    tile = tl.program_id(1)
    columns = tl.arange(0, BLOCK)
    if tile < 2:
        col = tile * BLOCK + columns
        in_swa = col < 130
        start = tl.load(STARTS + req)
        base = tl.maximum(start - 127, 0)
        length = tl.load(SWA_LEN + req)
        slot_out = tl.full((BLOCK,), -1, tl.int32)
        for qi in tl.static_range(3):
            row = req * 3 + qi
            shift = tl.maximum(start - 127 + qi, 0) - base
            src = col - shift
            in_row = in_swa & (src >= 0) & (src < 128)
            slot = tl.load(SWA + row * 128 + src, in_row & (src < length), other=-1)
            tl.device_assert(
                (slot_out < 0) | (slot < 0) | (slot == slot_out),
                "non-chronological Q3 SWA slot metadata",
            )
            slot_out = tl.where(slot_out >= 0, slot_out, tl.maximum(slot, -1))
            dense = tl.where(slot >= 0, req * 130 + col, -1)
            tl.store(DENSE + row * (128 + CW) + src, dense, in_row)
        tl.store(SLOTS_SWA + req * 130 + col, slot_out.to(tl.int64), in_swa)
    else:
        col = (tile - 2) * BLOCK + columns
        in_cmp = col < CW
        length = tl.load(CMP_LEN + req)
        slot_out = tl.full((BLOCK,), -1, tl.int32)
        for qi in tl.static_range(3):
            row = req * 3 + qi
            slot = tl.load(CMP + row * CW + col, in_cmp & (col < length), other=-1)
            if HCA:
                tl.device_assert(
                    (slot_out < 0) | (slot < 0) | (slot == slot_out),
                    "non-dense Q3 HCA slot metadata",
                )
                slot_out = tl.where(slot_out >= 0, slot_out, tl.maximum(slot, -1))
                out_row = req
            else:
                out_row = row
                tl.store(
                    SLOTS_CMP + out_row * CW + col,
                    tl.maximum(slot, -1).to(tl.int64),
                    in_cmp,
                )
            dense = tl.where(slot >= 0, B * 130 + out_row * CW + col, -1)
            tl.store(DENSE + row * (128 + CW) + 128 + col, dense, in_cmp)
        if HCA:
            tl.store(SLOTS_CMP + req * CW + col, slot_out.to(tl.int64), in_cmp)


def _emit_compact_q3_slots(swa, cmp, swa_len, cmp_len, starts):
    batch = swa.shape[0]
    width = cmp.shape[-1]
    hca = width == 384
    if (
        batch not in (1, 8)
        or swa.shape != (batch, 3, 128)
        or cmp.shape != (batch, 3, width)
        or width not in (384, 512)
    ):
        raise ValueError(
            "requires B1/B8 Q3 chronological SWA and dense HCA or ranked CSA"
        )
    if any(t.shape != (batch,) for t in (swa_len, cmp_len, starts)):
        raise ValueError("requires per-request lengths and Q3 start positions")
    device = swa.device
    if device.type != "cuda" or any(
        t.device != device or t.dtype != torch.int32 or not t.is_contiguous()
        for t in (swa, cmp, swa_len, cmp_len, starts)
    ):
        raise ValueError("requires contiguous int32 device metadata")
    sw = torch.empty((batch, 130), device=device, dtype=torch.int64)
    co = torch.empty(
        (batch if hca else batch * 3, width), device=device, dtype=torch.int64
    )
    dense = torch.empty((batch * 3, 1, 128 + width), device=device, dtype=torch.int32)
    _emit_compact_q3[(batch, 2 + triton.cdiv(width, 128))](
        swa,
        cmp,
        swa_len,
        cmp_len,
        starts,
        sw,
        co,
        dense,
        batch,
        width,
        hca,
        128,
        num_warps=4,
    )
    return sw, co, dense


@triton.jit
def _gather_c4_q3_tiles(
    OUT, POOL, SLOTS, BLOCKS, TOKEN_TILE: tl.constexpr, TRAP: tl.constexpr
):
    row = tl.program_id(0) * TOKEN_TILE + tl.arange(0, TOKEN_TILE)
    slot = tl.load(SLOTS + row)
    block = slot // 64
    pos = slot % 64
    if TRAP:
        if tl.sum(((slot >= 0) & (block >= BLOCKS)).to(tl.int32), 0) != 0:
            _trap_invalid_kv_access(TRAP)
    valid = (slot >= 0) & (block < BLOCKS)
    block_bytes = block * 37440
    data = block_bytes + pos * 576

    group = tl.arange(0, 8)
    cols = tl.arange(0, 64)
    scale_bytes = block_bytes + 64 * 576 + pos * 8
    scale_u8 = tl.load(
        POOL + scale_bytes[:, None] + group[None, :],
        valid[:, None] & (group[None, :] < 7),
        other=127,
    )
    scale = tl.exp2(scale_u8.to(tl.float32) - 127.0)

    dim = group[None, :, None] * 64 + cols[None, None, :]
    value_u8 = tl.load(
        POOL + data[:, None, None] + dim, valid[:, None, None] & (dim < 448), other=0
    )
    fp32 = value_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    value = (fp32 * scale[:, :, None]).to(tl.bfloat16)
    tl.store(OUT + row[:, None, None] * 512 + dim, value, dim < 448)

    rope = (POOL + data[:, None] + 448 + cols[None, :] * 2).to(
        tl.pointer_type(tl.bfloat16)
    )
    value_bf16 = tl.load(rope, valid[:, None], other=0)
    tl.store(OUT + row[:, None] * 512 + 448 + cols[None, :], value_bf16)


def _gather_c4_q3_tiles_out(out, pool, slots, *, token_tile=8):
    if (
        out.shape != (24, 512, 512)
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
        or slots.shape != (24, 512)
        or slots.dtype != torch.int64
        or not slots.is_contiguous()
        or pool.shape[1:] != (64, 584)
        or pool.stride() != (37440, 584, 1)
        or pool.dtype != torch.uint8
        or pool.data_ptr() % 2
        or not out.is_cuda
        or slots.device != out.device
        or pool.device != out.device
        or token_tile not in (4, 8, 16)
    ):
        raise ValueError("requires pinned B8/Q3 rank-ordered C4 packed geometry")
    validate_slot_mapping(
        "swa.dequantize_and_gather.slot_mapping",
        slots.view(-1),
        block_size=64,
        num_blocks=pool.shape[0],
        negative_mode="skip_any",
    )
    _gather_c4_q3_tiles[(slots.numel() // token_tile,)](
        out,
        pool,
        slots,
        pool.shape[0],
        token_tile,
        trap_invalid_kv_access_enabled(),
        num_warps=4,
        enable_fp_fusion=False,
    )


def try_materialized_compact_q3(
    q, swa_cache, cmp_cache, sink, swa, cmp, swa_len, cmp_len, starts, ratio
):
    """Return None when the sealed Q3 materialized contract does not apply."""
    if (
        q.shape != (8, 3, 64, 512)
        or not q.is_cuda
        or q.dtype != torch.bfloat16
        or not q.is_contiguous()
    ):
        return None
    if torch.cuda.get_device_name(q.device) != "ZW-M890P":
        return None
    if ratio not in (4, 128):
        return None
    width, page, stride = (512, 64, 37440) if ratio == 4 else (384, 2, 1728)
    for pool, entries, block_stride in (
        (swa_cache, 130, 76032),
        (cmp_cache, page, stride),
    ):
        if (
            pool.shape[1:] != (entries, 584)
            or pool.stride() != (block_stride, 584, 1)
            or pool.device != q.device
            or pool.dtype != torch.uint8
            or pool.data_ptr() % 2
        ):
            return None
    if (
        sink.shape != (64,)
        or sink.device != q.device
        or sink.dtype != torch.float32
        or not sink.is_contiguous()
    ):
        return None
    for tensor, shape in (
        (swa, (8, 3, 128)),
        (cmp, (8, 3, width)),
        (swa_len, (8,)),
        (cmp_len, (8,)),
        (starts, (8,)),
    ):
        if (
            tensor is None
            or tensor.shape != shape
            or tensor.device != q.device
            or tensor.dtype != torch.int32
            or not tensor.is_contiguous()
        ):
            return None

    workspace, dense = _materialize_compact_q3(
        q, swa_cache, cmp_cache, sink, swa, cmp, swa_len, cmp_len, starts, ratio
    )
    output, _, _ = forward_prefill(
        q.view(-1, 64, 512),
        workspace.view(-1, 1, 512),
        dense,
        512**-0.5,
        attn_sink=sink,
    )
    return output.view_as(q)


def _materialize_compact_q3(
    q, swa_cache, cmp_cache, sink, swa, cmp, swa_len, cmp_len, starts, ratio
):
    """Gather the ordered sparse keys after the caller checks B8/Q3 eligibility."""
    from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
        dequantize_and_gather_k_cache_slots,
        invalid_kv_access_validation_enabled,
    )

    # Preserve the original host validation mode without synchronizing the fast path.
    if (
        os.environ.get("DSV4_PPU_FUSED_COMPACT_MATERIALIZE", "0") != "1"
        or invalid_kv_access_validation_enabled()
    ):
        sw, co, dense = _emit_compact_q3_slots(swa, cmp, swa_len, cmp_len, starts)
        workspace = torch.empty(
            (sw.numel() + co.numel(), 512), device=q.device, dtype=torch.bfloat16
        )
        dequantize_and_gather_k_cache_slots(
            workspace[: sw.numel()].view(*sw.shape, 512), swa_cache, sw, None, 0
        )
        cmp_out = workspace[sw.numel() :].view(*co.shape, 512)
        if ratio == 4:
            _gather_c4_q3_tiles_out(cmp_out, cmp_cache, co, token_tile=8)
        else:
            dequantize_and_gather_k_cache_slots(cmp_out, cmp_cache, co, None, 0)
        return workspace, dense
    width, page, stride = (512, 64, 37440) if ratio == 4 else (384, 2, 1728)
    hca = ratio == 128
    rows = 8 * 130 + (8 if hca else 24) * width
    workspace = torch.empty((rows, 512), device=q.device, dtype=torch.bfloat16)
    dense = torch.empty((24, 1, 128 + width), device=q.device, dtype=torch.int32)
    _emit_and_materialize_compact_q3[(8, 17 + (1 if hca else 3) * (width // 8))](
        swa,
        cmp,
        swa_len,
        cmp_len,
        starts,
        swa_cache,
        cmp_cache,
        workspace,
        dense,
        width,
        hca,
        page,
        stride,
        swa_cache.shape[0],
        cmp_cache.shape[0],
        trap_invalid_kv_access_enabled(),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return workspace, dense


@triton.jit
def _emit_and_materialize_compact_q3(
    SWA,
    CMP,
    SWA_LEN,
    CMP_LEN,
    STARTS,
    SWA_POOL,
    CMP_POOL,
    OUT,
    DENSE,
    CW: tl.constexpr,
    HCA: tl.constexpr,
    CMP_PAGE: tl.constexpr,
    CMP_STRIDE: tl.constexpr,
    SWA_BLOCKS,
    CMP_BLOCKS,
    TRAP: tl.constexpr,
):
    req = tl.program_id(0)
    tile = tl.program_id(1)
    lanes = tl.arange(0, 8)
    if tile < 17:
        col = tile * 8 + lanes
        active = col < 130
        start = tl.load(STARTS + req)
        base = tl.maximum(start - 127, 0)
        length = tl.load(SWA_LEN + req)
        slot_out = tl.full((8,), -1, tl.int32)
        for qi in tl.static_range(3):
            row = req * 3 + qi
            shift = tl.maximum(start - 127 + qi, 0) - base
            src = col - shift
            in_row = active & (src >= 0) & (src < 128)
            slot = tl.load(SWA + row * 128 + src, in_row & (src < length), other=-1)
            tl.device_assert(
                (slot_out < 0) | (slot < 0) | (slot == slot_out),
                "non-chronological Q3 SWA slot metadata",
            )
            slot_out = tl.where(slot_out >= 0, slot_out, tl.maximum(slot, -1))
            dense = tl.where(slot >= 0, req * 130 + col, -1)
            tl.store(DENSE + row * (128 + CW) + src, dense, in_row)
        _materialize_eight_rows(
            OUT,
            SWA_POOL,
            req * 130 + col,
            slot_out,
            active,
            SWA_BLOCKS,
            130,
            76032,
            TRAP,
        )
    else:
        local = tile - 17
        col = (local % (CW // 8)) * 8 + lanes
        length = tl.load(CMP_LEN + req)
        if HCA:
            slot_out = tl.full((8,), -1, tl.int32)
            for qi in tl.static_range(3):
                row = req * 3 + qi
                slot = tl.load(CMP + row * CW + col, col < length, other=-1)
                tl.device_assert(
                    (slot_out < 0) | (slot < 0) | (slot == slot_out),
                    "non-dense Q3 HCA slot metadata",
                )
                slot_out = tl.where(slot_out >= 0, slot_out, tl.maximum(slot, -1))
                dense = tl.where(slot >= 0, 8 * 130 + req * CW + col, -1)
                tl.store(DENSE + row * (128 + CW) + 128 + col, dense)
            output_row = req
        else:
            qi = local // (CW // 8)
            output_row = req * 3 + qi
            slot = tl.load(CMP + output_row * CW + col, col < length, other=-1)
            slot_out = tl.maximum(slot, -1)
            dense = tl.where(slot >= 0, 8 * 130 + output_row * CW + col, -1)
            tl.store(DENSE + output_row * (128 + CW) + 128 + col, dense)
        _materialize_eight_rows(
            OUT,
            CMP_POOL,
            8 * 130 + output_row * CW + col,
            slot_out,
            tl.full((8,), True, tl.int1),
            CMP_BLOCKS,
            CMP_PAGE,
            CMP_STRIDE,
            TRAP,
        )


@triton.jit
def _materialize_eight_rows(
    OUT,
    POOL,
    rows,
    slots,
    active,
    BLOCKS,
    PAGE: tl.constexpr,
    STRIDE: tl.constexpr,
    TRAP: tl.constexpr,
):
    # Promote before byte arithmetic: do not overflow int32 on large KV pools.
    slot = slots.to(tl.int64)
    block = slot // PAGE
    pos = slot % PAGE
    if TRAP:
        if tl.sum((active & (slot >= 0) & (block >= BLOCKS)).to(tl.int32), 0) != 0:
            _trap_invalid_kv_access(TRAP)
    valid = active & (slot >= 0) & (block < BLOCKS)
    data = block * STRIDE + pos * 576
    group = tl.arange(0, 8)
    cols = tl.arange(0, 64)
    scale_bytes = block * STRIDE + PAGE * 576 + pos * 8
    scale_u8 = tl.load(
        POOL + scale_bytes[:, None] + group[None, :],
        valid[:, None] & (group[None, :] < 7),
        other=127,
    )
    scale = tl.exp2(scale_u8.to(tl.float32) - 127.0)
    dim = group[None, :, None] * 64 + cols[None, None, :]
    value_u8 = tl.load(
        POOL + data[:, None, None] + dim, valid[:, None, None] & (dim < 448), other=0
    )
    fp32 = value_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    value = (fp32 * scale[:, :, None]).to(tl.bfloat16)
    tl.store(
        OUT + rows[:, None, None] * 512 + dim,
        value,
        active[:, None, None] & (dim < 448),
    )
    rope = (POOL + data[:, None] + 448 + cols[None, :] * 2).to(
        tl.pointer_type(tl.bfloat16)
    )
    value_bf16 = tl.load(rope, valid[:, None], other=0)
    tl.store(
        OUT + rows[:, None] * 512 + 448 + cols[None, :], value_bf16, active[:, None]
    )
