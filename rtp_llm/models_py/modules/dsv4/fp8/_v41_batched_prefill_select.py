"""Bounded request-batched index selection for high-reuse prefill."""

from __future__ import annotations

from bisect import bisect_left

import torch


def _try_publish_with_tokens(
    shared, logits, visible, rows, bounds, target, block, count
):
    from . import _v41_prefill_candidates as pool
    from . import _v41_prefill_topk as topk

    retained = shared.get("ced_candidate_rows")
    if retained is not None:
        first = bisect_left(retained, rows.start)
        if first == len(retained) or retained[first] >= rows.stop:
            return False, None
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
):
    """Finish all request rows, or reject the layout before changing outputs.

    Group boundaries bound temporary scores; they are independent of the
    number of Python request objects. Projection arithmetic remains outside
    this helper at its original local-batch M.
    """
    if len(globals_by_req) < 2 or request_row_slices is None or not q_payload.is_cuda:
        return False
    shared = attn._shared_attention
    cp = getattr(attn, "_cp_ctx", None)
    if (
        shared.get("ced_indexer_projection") is not None
        or cp is None
        or req_ids is None
        or cp.prefix_lengths is None
        or cp.input_lengths_global is None
    ):
        return False
    cache = shared.setdefault("prefill_score_bounds", {})
    counts_key = ("batch_key_counts", attn.compress_ratio)
    entry = cache.get(counts_key)
    produced = cache.get(("producer_key_counts", attn.compress_ratio))
    if (
        isinstance(entry, tuple)
        and len(entry) == 3
        and entry[0] is cp.prefix_lengths
        and entry[1] is cp.input_lengths_global
    ):
        key_counts = entry[2]
    else:
        # Replacing CP tensors invalidates bounds as well as the cached counts.
        # A same-source producer may publish again at L8/L14; keep object identity.
        if entry is not None:
            cache.clear()
            shared.pop("prefill_sparse_plans", None)
        if (
            produced is not None
            and produced[0] is cp.prefix_lengths
            and produced[1] is cp.input_lengths_global
        ):
            key_counts = produced[2]
        else:
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
    from ._v41_grouped_prefill_score import _mask_tail_kernel, try_grouped_scores
    from .attention_v41 import _apply_prefill_candidates

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
    )
    if grouped is None:
        return False
    publishing = candidates is not None and publish_candidates
    for rows, logits, visible, bounds in grouped.groups(mask_tail=False):
        count = min(attn.index_topk, logits.shape[1])
        target = out[rows, :count]
        published, selected = False, None
        if publishing:
            published, selected = _try_publish_with_tokens(
                shared,
                logits,
                visible,
                rows,
                bounds,
                target,
                candidate_size,
                candidate_blocks,
            )
        if publishing and not published:
            _mask_tail_kernel[(logits.shape[0],)](
                logits, visible, logits.shape[1], logits.stride(0), 256
            )
            _apply_prefill_candidates(
                shared,
                logits,
                visible,
                rows,
                candidate_size,
                candidate_blocks,
                True,
            )
        if selected is None:
            selected = topk.try_select_tokens(
                logits, visible, count, bounds=bounds, out=target
            )
        if selected is None:
            # Native K512 consumes ends; torch.topk and publication read tails.
            if not publishing:
                _mask_tail_kernel[(logits.shape[0],)](
                    logits, visible, logits.shape[1], logits.stride(0), 256
                )
            scores, selected = logits.topk(count, dim=-1)
            selected = torch.where(scores.isfinite(), selected, -1).int()
        if selected is not target:
            target.copy_(selected)
        del logits
    return True
