"""Preserve released short SWA masks through maintained SM120 attention.

This diagnostic selects explicit owners. It preserves cache writes and GLOBAL
selection, and reuses the maintained finite-max TileLang core in TP8 groups.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from unittest.mock import patch

import torch
from rtp_llm.models_py.modules.dsv4 import tilelang_kernels
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_decode_attention_sm120 as decode
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_attention_sm120 as prefill
from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import AttentionV41FP8
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_decode_tilelang_reference import (
    _ReferenceDecodeAttention,
)


def prefill_plan(indices, lengths, kv_rows, global_rows):
    rows = len(lengths)
    if (
        type(global_rows) is not int
        or not 0 <= global_rows <= rows
        or kv_rows != rows + global_rows
        or not 1 <= rows <= 128
        or indices.ndim != 3
        or indices.shape[:2] != (rows, 1)
        or indices.dtype != torch.int32
        or not indices.is_cuda
        or not indices.is_contiguous()
        or lengths.shape != (rows,)
        or lengths.dtype != torch.int32
        or lengths.device != indices.device
        or not lengths.is_contiguous()
    ):
        raise ValueError("Short prefill requires one initial SWA workspace")
    # Membership preserves the supplied GLOBAL selection and causal mask. The
    # fixed SWA segment retains holes rather than compacting its live entries.
    positions = torch.cat(
        (
            torch.arange(global_rows, kv_rows, device=indices.device),
            torch.arange(global_rows, device=indices.device),
        )
    ).int()
    live = (
        torch.arange(indices.shape[-1], device=indices.device)[None] < lengths[:, None]
    )
    matches = (indices[:, 0, :, None] == positions[None, None]) & live[:, :, None]
    return torch.where(matches.any(1), positions[None], -1).unsqueeze(1).contiguous()


def prefill_attention(query, kv, indices, lengths, sinks, *, global_rows):
    if (
        query.ndim != 3
        or query.shape[-1] != 512
        or not 1 <= query.shape[1] <= 64
        or query.dtype != torch.bfloat16
        or not query.is_cuda
        or not query.is_contiguous()
        or torch.cuda.get_device_capability(query.device) != (12, 0)
        or kv.shape != (len(query) + global_rows, 1, 512)
        or kv.dtype != query.dtype
        or kv.device != query.device
        or not kv.is_contiguous()
        or indices.device != query.device
        or lengths.shape != query.shape[:1]
        or sinks.shape != (query.shape[1],)
        or sinks.dtype != torch.float32
        or sinks.device != query.device
        or not sinks.is_contiguous()
    ):
        raise ValueError("Unsupported short SM120 sparse prefill geometry")
    plan = prefill_plan(indices, lengths, len(kv), global_rows)
    return torch.cat(
        [
            tilelang_kernels.sparse_attn(
                query[:, head : head + 8].unsqueeze(0).contiguous(),
                kv.reshape(1, -1, 512),
                sinks[head : head + 8].contiguous(),
                plan.reshape(1, len(query), -1),
                512**-0.5,
                initial_max=-1e30,
            )[0]
            for head in range(0, query.shape[1], 8)
        ],
        1,
    ).contiguous()


def decode_plans(plans, starts, *, swa_entries):
    slots = plans.get("swa")
    if (
        slots is None
        or set(plans) not in ({"swa"}, {"swa", "global"})
        or slots.ndim != 2
        or slots.shape[1] != 128
        or not slots.is_cuda
        or slots.dtype not in (torch.int32, torch.int64)
        or not slots.is_contiguous()
        or starts.shape != slots.shape[:1]
        or starts.dtype != torch.int32
        or starts.device != slots.device
        or not starts.is_contiguous()
        or swa_entries not in (128, 130)
    ):
        raise ValueError("Short decode requires device positions and a 128-slot ring")
    # Generation one adds MTP entries to each page. Physical page stride is
    # distinct from the 128-slot SWA ring and comes from the cache layout.
    ring = (starts[:, None] + 1 + torch.arange(128, device=slots.device)) % 128
    matches = (slots[:, :, None] >= 0) & (
        slots[:, :, None] % swa_entries == ring[:, None]
    )
    ordered = torch.where(matches, slots[:, :, None], -1).amax(1).contiguous()
    return dict(plans, swa=ordered)


@contextmanager
def short_sparse_plan_reference(enabled, *, attentions=()):
    if type(enabled) is not bool:
        raise ValueError("Short sparse selection requires a bool")
    if not enabled:
        yield
        return
    attentions = tuple(attentions)
    if not attentions or len({id(owner) for owner in attentions}) != len(attentions):
        raise ValueError("Select distinct V4.1 attention owners")
    for owner in attentions:
        if (
            type(owner) is not AttentionV41FP8
            or owner.dim not in (512, 5120)
            or (owner.head_dim, owner.rope_head_dim, owner.window_size)
            != (512, 64, 128)
            or owner.compress_ratio not in (0, 1, 2)
            or not owner.attn_sink.is_cuda
            or owner.attn_sink.dtype != torch.float32
            or torch.cuda.get_device_capability(owner.attn_sink.device) != (12, 0)
            or (owner._cp_ctx is not None and owner._cp_ctx.cp_size > 1)
        ):
            raise ValueError("Short sparse selection requires admitted SM120 owners")
    selected = {id(owner) for owner in attentions}
    prefill_rows = ContextVar("v41_short_sparse_prefill_rows", default=None)
    decode_starts = ContextVar("v41_short_sparse_decode_starts", default=None)
    original_prefill_scope = AttentionV41FP8._prefill_sparse_attention
    original_decode_scope = AttentionV41FP8.forward_decode
    original_prefill = prefill.sparse_prefill_attention
    original_decode = decode._attention
    provider = _ReferenceDecodeAttention()

    def prefill_scope(owner, qkv, common, **kwargs):
        if id(owner) not in selected or not len(qkv.qr):
            return original_prefill_scope(owner, qkv, common, **kwargs)
        rows = len(qkv.qr)
        global_rows = rows // owner.compress_ratio if owner.compress_ratio else 0
        prefix = common.prefix_lengths
        if (
            common.cp_on
            or common.batch_size != 1
            or rows > 128
            or prefix is None
            or prefix.numel() != 1
            or prefix.item() != 0
            or kwargs["kv"].shape[0] != rows + global_rows
        ):
            raise ValueError("Short sparse selection requires one initial prefill")
        token = prefill_rows.set(global_rows)
        try:
            return original_prefill_scope(owner, qkv, common, **kwargs)
        finally:
            prefill_rows.reset(token)

    def decode_scope(owner, x, metadata, kv_cache=None):
        if id(owner) not in selected:
            return original_decode_scope(owner, x, metadata, kv_cache)
        if (
            metadata is None
            or metadata.start_pos is None
            or x.shape[0] != 1
            or metadata.start_pos.shape != (1,)
            or not 1 <= metadata.start_pos.item() < 128
        ):
            raise ValueError("Short sparse selection requires decode in the first ring")
        token = decode_starts.set(metadata.start_pos)
        try:
            return original_decode_scope(owner, x, metadata, kv_cache)
        finally:
            decode_starts.reset(token)

    def selected_prefill(query, kv, indices, lengths, sinks):
        count = prefill_rows.get()
        if count is None:
            return original_prefill(query, kv, indices, lengths, sinks)
        return prefill_attention(query, kv, indices, lengths, sinks, global_rows=count)

    def selected_decode(
        query,
        sinks,
        swa,
        swa_slots,
        *,
        valid_rows,
        global_cache=None,
        global_slots=None,
        out=None
    ):
        starts = decode_starts.get()
        if starts is None:
            return original_decode(
                query,
                sinks,
                swa,
                swa_slots,
                valid_rows=valid_rows,
                global_cache=global_cache,
                global_slots=global_slots,
                out=out,
            )
        plans = {"swa": swa_slots}
        if global_cache is not None:
            plans["global"] = global_slots
        ordered = decode_plans(plans, starts, swa_entries=swa.layout.entries)
        return provider(
            query,
            sinks,
            swa,
            ordered["swa"],
            valid_rows=valid_rows,
            global_cache=global_cache,
            global_slots=global_slots,
            out=out,
        )

    with patch.object(
        AttentionV41FP8, "_prefill_sparse_attention", prefill_scope
    ), patch.object(AttentionV41FP8, "forward_decode", decode_scope), patch.object(
        prefill, "sparse_prefill_attention", selected_prefill
    ), patch.object(
        decode, "_attention", selected_decode
    ):
        yield
