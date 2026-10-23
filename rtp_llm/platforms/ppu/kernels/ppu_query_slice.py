"""Gather one chronological MTP query and its metadata in one launch."""

from dataclasses import replace

import torch
import triton
import triton.language as tl

_FIELDS = ("positions", "b_idx", "state_slots", "kv_slots", "token_to_req")


@triton.jit
def _query_slice(
    X,
    Q,
    START,
    POS,
    REQ,
    STATE,
    KV,
    TOKEN,
    LENGTH,
    OX,
    OQ,
    OSTART,
    OPOS,
    OREQ,
    OSTATE,
    OKV,
    OTOKEN,
    OLENGTH,
    WIDTH: tl.constexpr,
    INDEX: tl.constexpr,
    XD: tl.constexpr,
    QD: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    source_row = row * WIDTH + INDEX
    if tile < XD // BLOCK:
        col = tile * BLOCK + cols
        tl.store(OX + row * XD + col, tl.load(X + source_row * XD + col))
    elif QD > 0:
        col = (tile - XD // BLOCK) * BLOCK + cols
        tl.store(OQ + row * QD + col, tl.load(Q + source_row * QD + col))
    if tile == 0:
        tl.store(OSTART + row, tl.load(START + row) + INDEX)
        tl.store(OPOS + row, tl.load(POS + source_row))
        tl.store(OREQ + row, tl.load(REQ + source_row))
        tl.store(OSTATE + row, tl.load(STATE + source_row))
        tl.store(OKV + row, tl.load(KV + source_row))
        tl.store(OTOKEN + row, tl.load(TOKEN + source_row))
        if HAS_LENGTH:
            tl.store(OLENGTH + row, tl.load(LENGTH + source_row))


def prepare_indexer_query(x, qr, start_pos, meta, index):
    """Return None for unsupported layouts, preserving the original gather.

    Only copy/reorder work is combined. Compressor ring updates and attention
    reads still execute in query order, including speculative rollback slots.
    ``qr=None`` prepares only the compressor inputs after batched Q projection.
    """
    if (
        x.ndim != 3
        or (qr is not None and (qr.ndim != 3 or x.shape[:2] != qr.shape[:2]))
        or x.shape[1] not in (2, 3, 4)
        or not 0 <= index < x.shape[1]
        or x.shape[-1] <= 0
        or x.shape[-1] % 256
        or (qr is not None and (qr.shape[-1] <= 0 or qr.shape[-1] % 256))
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or (qr is not None and qr.dtype != x.dtype)
    ):
        return None
    batch, width = x.shape[:2]
    fields = [getattr(meta, name) for name in _FIELDS]
    length = meta.compressed_lens_per_token
    metadata = fields + ([length] if length is not None else [])
    if (
        any(t.numel() != batch * width for t in metadata)
        or start_pos.shape != (batch,)
        or any(
            t.device != x.device or not t.is_contiguous()
            for t in [x, start_pos, *metadata, *([qr] if qr is not None else [])]
        )
        or any(
            t.dtype not in (torch.int32, torch.int64) for t in [start_pos, *metadata]
        )
    ):
        return None
    out_x = torch.empty((batch, 1, x.shape[-1]), device=x.device, dtype=x.dtype)
    q_dim = qr.shape[-1] if qr is not None else 0
    out_q = (
        torch.empty((batch, 1, q_dim), device=x.device, dtype=x.dtype)
        if qr is not None
        else None
    )
    out_start = torch.empty_like(start_pos)
    # contiguous() retains the original stride for zero/single-element views.
    stride = width if batch <= 1 else 1
    outputs = [
        torch.empty_strided((batch,), (stride,), device=x.device, dtype=t.dtype)
        for t in metadata
    ]
    out_length = outputs[-1] if length is not None else None
    if batch:
        _query_slice[(batch, (x.shape[-1] + q_dim) // 256)](
            x,
            qr,
            start_pos,
            *fields,
            length,
            out_x,
            out_q,
            out_start,
            *outputs[:5],
            out_length,
            width,
            index,
            x.shape[-1],
            q_dim,
            length is not None,
            256,
            num_warps=4,
        )
    values = dict(zip(_FIELDS, outputs[:5]))
    if length is not None:
        values["compressed_lens_per_token"] = out_length
    step = replace(
        meta,
        **values,
        is_batched=False,
        seq_start_per_req=values["positions"],
        cu_seq_per_req=None,
    )
    return out_x, out_q, out_start, step
