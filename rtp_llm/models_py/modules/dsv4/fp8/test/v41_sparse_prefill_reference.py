"""Scoped short-prefill SWA-first/libdevice arithmetic diagnostic for SM120."""

from contextlib import contextmanager
from contextvars import ContextVar
from unittest.mock import patch

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_attention_sm120 as prefill
from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import AttentionV41FP8
from rtp_llm.tools.dsv41_sparse_precision_replay import _precision_sparse


@triton.jit
def _swa_first(
    IDX,
    LENS,
    OUT,
    OUT_LENS,
    COUNT,
    GLOBAL_ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, BLOCK)
    limit = tl.load(LENS + row)
    slot = tl.load(IDX + row * WIDTH + columns, columns < WIDTH, other=-1)
    live = (columns < WIDTH) & (columns < limit)
    tl.device_assert(
        ~live | ((slot >= -1) & (slot < COUNT)), "Invalid scoped prefill slot"
    )
    valid = live & (slot >= 0) & (slot < COUNT)
    swa = valid & (slot >= GLOBAL_ROWS)
    global_key = valid & (slot < GLOBAL_ROWS)
    swa_rank = tl.cumsum(swa.to(tl.int32), 0) - 1
    global_rank = (
        tl.cumsum(global_key.to(tl.int32), 0) - 1 + tl.sum(swa.to(tl.int32), 0)
    )
    destination = tl.where(swa, swa_rank, global_rank)
    tl.store(OUT + row * WIDTH + destination, slot, valid)
    tl.store(OUT_LENS + row, tl.sum(valid.to(tl.int32), 0))


def swa_first_plan(indices, lengths, count, global_rows):
    if type(global_rows) is not int or not 0 <= global_rows <= count:
        raise ValueError("Sparse reference requires a bounded GLOBAL segment")
    # Separate initialization and scatter launches order all writes on the
    # stream; initializing in the scatter CTA would race across its warps.
    output = torch.full_like(indices, -1)
    output_lengths = torch.empty_like(lengths)
    if len(lengths):
        _swa_first[(len(lengths),)](
            indices,
            lengths,
            output,
            output_lengths,
            count,
            GLOBAL_ROWS=global_rows,
            WIDTH=indices.shape[-1],
            BLOCK=triton.next_power_of_2(indices.shape[-1]),
            num_warps=4,
        )
    return output, output_lengths


class _ReferencePrefillKernel:
    def __init__(self, original, selected):
        self.original, self.selected = original, selected

    def __getitem__(self, grid):
        original_launch = self.original[grid]

        def reference(query, kv, indices, lengths, sinks, out, count, **kwargs):
            global_rows = self.selected.get()
            if global_rows is None:
                return original_launch(
                    query, kv, indices, lengths, sinks, out, count, **kwargs
                )
            ordered, live = swa_first_plan(indices, lengths, count, global_rows)
            kwargs.pop("DIRECT_SINK_DIV_RN", None)
            kwargs.update(EXP=1, enable_fp_fusion=True)
            return _precision_sparse[grid](
                query, kv, ordered, live, sinks, out, count, **kwargs
            )

        return reference


@contextmanager
def sparse_prefill_reference(enabled, *, global_rows=None):
    if not enabled:
        yield
        return
    if torch.cuda.get_device_capability() != (12, 0):
        raise ValueError("Sparse prefill reference is qualified only on SM120")
    if global_rows is not None and (type(global_rows) is not int or global_rows < 0):
        raise ValueError("Sparse reference requires a nonnegative GLOBAL segment")
    selected = ContextVar("v41_diagnostic_sparse_prefill", default=global_rows)
    original_owner = AttentionV41FP8._prefill_sparse_attention

    def owner_scope(owner, qkv, common, *, kv, indices, topk_length, profile_name):
        rows = qkv.qr.shape[0]
        arguments = dict(
            kv=kv, indices=indices, topk_length=topk_length, profile_name=profile_name
        )
        if not rows:
            return original_owner(owner, qkv, common, **arguments)
        prefix = common.prefix_lengths
        ratio = owner.compress_ratio
        expected = rows // ratio if ratio else 0
        if (
            common.cp_on
            or common.batch_size != 1
            or rows > min(128, owner.window_size)
            or prefix is None
            or prefix.numel() != 1
            or prefix.item() != 0
            or kv.shape[0] != rows + expected
        ):
            raise ValueError(
                "Sparse prefill reference requires a short single initial workspace"
            )
        token = selected.set(expected)
        try:
            return original_owner(owner, qkv, common, **arguments)
        finally:
            selected.reset(token)

    with patch.object(
        AttentionV41FP8, "_prefill_sparse_attention", owner_scope
    ), patch.object(
        prefill,
        "_sparse_prefill",
        _ReferencePrefillKernel(prefill._sparse_prefill, selected),
    ):
        yield
