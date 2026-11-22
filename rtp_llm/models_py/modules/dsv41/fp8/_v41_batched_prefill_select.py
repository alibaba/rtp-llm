"""Bounded request-batched index selection for high-reuse prefill."""

from __future__ import annotations

import os
from bisect import bisect_left
from dataclasses import dataclass, replace

import torch

from .._prefill_stream import prefill_stream_is_capturing


def _try_publish_with_tokens(
    shared, logits, visible, rows, bounds, target, block, count
):
    """Return whether publication is handled and any selected token indices."""
    from . import _v41_prefill_candidates as pool
    from . import _v41_prefill_topk as topk

    retained = shared.get("ced_candidate_rows")
    if retained is not None:
        first = bisect_left(retained, rows.start)
        if first == len(retained) or retained[first] >= rows.stop:
            return True, None
    candidates = shared["candidates"]
    cache = shared.get("prefill_candidate_mask")
    flags = (
        cache[1][rows]
        if cache is not None and cache[0] is candidates and cache[2] == block
        else None
    )
    kwargs = dict(
        out=candidates[rows],
        flags=flags,
        build_bitmap=not shared.get("prefill_sparse_candidates", False),
        token_indices=target,
        token_ends=bounds[1],
    )
    if not pool.can_select_candidates(logits, visible, block, count, **kwargs):
        return False, None
    selected = topk.try_select_tokens(
        logits, visible, target.shape[1], bounds=bounds, out=target, filter_finite=False
    )
    if selected is None:
        return False, None
    if (
        pool.select_candidates(logits, visible, block, count, mask_tail=True, **kwargs)
        is None
    ):
        return False, topk.finish_tokens(logits, bounds[1], selected)
    return True, selected


def prepare_retained_references(attn, plan):
    """Map compact L20 rows to their original score-group selector geometry."""
    from functools import partial

    from ._v41_grouped_prefill_score import _group_layout
    from ._v41_prefill_indexer import PrefillIndexerKeys

    by_request = attn._shared_attention.get("global", {}).get(
        attn.kv_source_layer_id, ()
    )
    keys = [key for _, key in by_request]
    if not all(isinstance(key, PrefillIndexerKeys) for key in keys):
        return None
    return _retained_references(
        attn, plan, tuple(map(len, keys)), partial(_group_layout, keys)
    )


def prepare_owned_references(attn, plan, widths, groups):
    """Plan references from the owned gather's padded, per-group K slabs."""
    from ._v41_grouped_prefill_score import _plan_group_layout

    slab_ids = tuple(g.first for g in groups for _ in range(g.first, g.stop))

    def adjacent(left, right):
        return slab_ids[left] == slab_ids[right]

    def get_layout(rows, *, cp_ctx=None):
        if len(rows) != len(widths) or any(n <= 0 for n in rows):
            return None
        return _plan_group_layout(
            widths, rows, adjacent, _ReferenceGroup, cp_ctx=cp_ctx
        )

    return _retained_references(attn, plan, widths, get_layout)


@dataclass(frozen=True)
class _ReferenceGroup:
    first: int
    stop: int
    rows: slice
    width: int
    last: int
    source_width: int | None = None


def _retained_references(attn, plan, widths, get_layout):
    from . import _v41_candidate_topk_v3 as candidate_topk
    from . import _v41_prefill_deepselect as deepselect
    from . import _v41_prefill_topk as topk
    from ._indexer_score import has_fp8_fp4_mqa_logits

    original = plan.original_context
    device = original.global_positions.device
    config = attn.v41_config or {}
    if (
        (attn.index_topk, attn.index_n_heads, attn.compress_ratio) != (512, 32, 1)
        or config.get("candidate_source_layer_id") != attn.layer_id
        or config.get("candidate_topk_blocks") != 2048
        or config.get("candidate_block_size") != 8
        or not widths
        or not all(512 <= width <= 1 << 20 for width in widths)
        or (max(widths) + 7) // 8 <= 2048
        or not topk.supports_bf16()
        or not hasattr(topk.rtp_llm_ops, "dsv41_topk_v3_with_reference")
        or not candidate_topk.is_available("prefill", device)
        # Retained publication writes the sparse candidate table directly.
        or not deepselect.is_available(device)
        or not has_fp8_fp4_mqa_logits()
        or get_layout(plan.context.chunk_lengths_per_req) is None
    ):
        return None

    use_deepselect = deepselect.is_available(device, torch.bfloat16)
    references = [None] * plan.context.chunk_length
    peers = (peer for peer, count in enumerate(plan.exchange.receive_sizes) if count)
    for owner, (compact_rows, source_rows) in zip(
        peers, plan.exchange.projection_groups_host, strict=True
    ):
        layout = get_layout(
            original.chunk_lengths_per_req,
            cp_ctx=replace(original, cp_rank=owner),
        )
        if (
            layout is None
            or not layout
            or layout[-1].rows.stop != original.chunk_length
        ):
            return None
        group_ends = tuple(group.rows.stop for group in layout)
        for target, source in zip(compact_rows, source_rows, strict=True):
            group = layout[bisect_left(group_ends, source + 1)]
            references[target] = (
                group.rows.stop - group.rows.start,
                group.width,
                use_deepselect and group.width % 512 == 0,
            )
    return tuple(references)


def _reference_spans(references):
    start = 0
    while start < len(references):
        geometry = references[start]
        stop = start + 1
        while stop < len(references) and references[stop] == geometry:
            stop += 1
        if geometry is not None:
            yield slice(start, stop), geometry
        start = stop


def _select_retained_group(logits, visible, bounds, target, candidates, references):
    from . import _v41_prefill_candidates as pool
    from . import _v41_prefill_deepselect as deepselect
    from . import _v41_prefill_topk as topk

    spans = tuple(_reference_spans(references))
    workspace = topk._get_topk_workspace(logits.device)
    select_reference = topk.rtp_llm_ops.dsv41_topk_v3_with_reference
    for span, (rows, width, use_deepselect) in spans:
        scores = logits[span, :width]
        if use_deepselect:
            if scores.shape[1] != width or scores.stride(0) % 512:
                # The reference TMA geometry still governs ties. Only the
                # visible prefix of a compact row participates in selection.
                padded = logits.new_full((scores.shape[0], width), -float("inf"))
                padded[:, : scores.shape[1]].copy_(scores)
                scores = padded
            if (
                deepselect.try_select_tokens(
                    scores, bounds[1][span], out=target[span], filter_finite=False
                )
                is None
            ):
                raise RuntimeError(
                    "CED reference DeepSelect rejected its planned geometry"
                )
        else:
            select_reference(
                scores,
                bounds[1][span],
                target[span],
                workspace,
                512,
                scores.shape[1],
                rows,
                width,
                False,
            )
        del scores

    def select_blocks(scores, lengths, out):
        for span, (rows, width, _) in spans:
            width = (width + 7) // 8
            pooled = scores[span, :width]
            select_reference(
                pooled,
                lengths[span],
                out[span],
                workspace,
                2048,
                pooled.shape[1],
                rows,
                width,
                False,
            )
        return out

    if (
        pool.select_candidates(
            logits,
            visible,
            8,
            2048,
            out=candidates,
            build_bitmap=False,
            mask_tail=True,
            token_indices=target,
            token_ends=bounds[1],
            select_blocks=select_blocks,
        )
        is None
    ):
        raise RuntimeError("CED candidate publication rejected its planned geometry")


def try_select_batched(
    attn,
    q_payload,
    q_sf,
    weights,
    globals_by_req,
    request_row_slices,
    positions,
    out,
    *,
    candidate_source,
    publish_candidates,
    candidate_size,
    candidate_blocks,
    req_ids=None,
    retained_references=None,
):
    """Finish all request rows, or reject the layout before changing outputs.

    Group boundaries bound temporary scores; they are independent of the
    number of Python request objects. Projection arithmetic remains outside
    this helper at its original local-batch M.
    """
    if not globals_by_req or request_row_slices is None or not q_payload.is_cuda:
        return False
    shared = attn._shared_attention
    cp = getattr(attn, "_cp_ctx", None)
    if (
        cp is None
        or req_ids is None
        or cp.prefix_lengths is None
        or cp.input_lengths_global is None
    ):
        return False
    # CED has already projected weights in original-owner geometry. Its compact
    # query positions/IDs change, while these write-side KV lengths stay full.
    cache = shared.setdefault("prefill_score_bounds", {})
    counts_key = ("batch_key_counts", attn.compress_ratio)
    entry = cache.get(counts_key)
    if (
        isinstance(entry, tuple)
        and len(entry) == 3
        and entry[0] is cp.prefix_lengths
        and entry[1] is cp.input_lengths_global
    ):
        key_counts = entry[2]
    else:
        # Replacing CP tensors invalidates bounds as well as the cached counts.
        if entry is not None:
            cache.clear()
            shared.pop("prefill_sparse_plans", None)
        key_counts = (
            cp.prefix_lengths + cp.input_lengths_global
        ) // attn.compress_ratio
        key_counts = key_counts.to(torch.int32)
        cache[counts_key] = (cp.prefix_lengths, cp.input_lengths_global, key_counts)
    candidates = shared.get("candidates")
    if (
        candidates is not None
        and candidate_source >= 0
        and attn.layer_id > candidate_source
    ):
        from ._v41_sparse_prefill_indexer import try_batched_sparse

        return try_batched_sparse(
            q_payload,
            q_sf,
            weights,
            globals_by_req,
            request_row_slices,
            positions,
            attn.compress_ratio,
            candidates,
            candidate_size,
            attn.index_topk,
            shared,
            out,
            req_ids=req_ids,
            key_counts=key_counts,
        )

    from . import _v41_prefill_topk as topk
    from ._indexer_score import balanced_mqa_schedule_builder
    from ._v41_grouped_prefill_score import _mask_tail_kernel, try_grouped_scores
    from .attention_v41 import _apply_prefill_candidates

    return_bf16 = topk.supports_bf16()
    grouped = try_grouped_scores(
        q_payload,
        q_sf,
        weights,
        globals_by_req,
        request_row_slices,
        positions,
        attn.compress_ratio,
        shared,
        req_ids=req_ids,
        key_counts=key_counts,
        cp_ctx=cp,
    )
    if grouped is None:
        return False
    publishing = candidates is not None and publish_candidates
    schedule_builder = (
        os.environ.get("DSV41_PREFILL_INDEXER_BALANCED_SCORE") == "1"
        and return_bf16
        and (attn.layer_id, attn.compress_ratio) in ((2, 2), (8, 2), (14, 2), (20, 1))
        and torch.cuda.get_device_capability(q_payload.device) == (10, 3)
        and not prefill_stream_is_capturing()
        and balanced_mqa_schedule_builder()
    )
    if (
        schedule_builder
        and os.environ.get("DSV41_PREFILL_INDEXER_NATIVE_GROUPED_TOPK") == "1"
        and attn.compress_ratio == 2
        and attn.index_topk == 512
        and not publishing
        and retained_references is None
        and grouped.try_native_topk(out, schedule_builder)
    ):
        return True
    score_options = {"balance_kv_tiles": True} if schedule_builder else {}
    targets = iter(grouped.split_rows(out))
    # Keep the score generator directly in the loop: zip/enumerate can retain
    # its previous yielded tuple while allocating the next score chunk.
    for rows, logits, visible, bounds in grouped.groups(
        mask_tail=False, return_bf16=return_bf16, **score_options
    ):
        target = next(targets)
        count = min(attn.index_topk, logits.shape[1])
        if retained_references is not None:
            _select_retained_group(
                logits,
                visible,
                bounds,
                target,
                candidates[rows],
                retained_references[rows],
            )
            del logits
            continue
        if count != target.shape[1]:
            target = target[:, :count]
        publication_handled, selected = False, None
        if publishing:
            publication_handled, selected = _try_publish_with_tokens(
                shared,
                logits,
                visible,
                rows,
                bounds,
                target,
                candidate_size,
                candidate_blocks,
            )
        if selected is None:
            selected = topk.try_select_tokens(
                logits, visible, count, bounds=bounds, out=target
            )
        use_torch_topk = selected is None
        if use_torch_topk:
            # Native selectors mask at load. Share this fill with publication
            # when the token selector requires torch.topk.
            _mask_tail_kernel[(logits.shape[0],)](
                logits, visible, logits.shape[1], logits.stride(0), 256
            )
        if publishing and not publication_handled:
            _apply_prefill_candidates(
                shared,
                logits,
                visible,
                rows,
                candidate_size,
                candidate_blocks,
                True,
                mask_tail=not use_torch_topk,
            )
        if use_torch_topk:
            scores, selected = logits.topk(count, dim=-1)
            selected = torch.where(scores.isfinite(), selected, -1).int()
        if selected is not target:
            target.copy_(selected)
        del logits
    return True
