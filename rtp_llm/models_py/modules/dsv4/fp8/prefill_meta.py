"""DSV4 FP8 prefill metadata broadcast helpers.

Free functions (NOT methods on ``V4Transformer``) that build the
layer-invariant prefill meta once per ``compress_ratio`` bucket
(0 = SWA-only, 4 = CSA, 128 = HCA) and broadcast each bucket's meta
to its layers' ``AttentionFP8._prefill_meta_shared``.

Lives under ``dsv4/fp8/`` because the meta build hard-assumes
FP8 KV-cache pools (``_build_shared_prefill_meta`` reads FP8-only
descriptors). Caller (``prefill/forward.py``) must gate the call with
``if v4.fp8_kv_cache:``; once we're inside, every ``layer.attn`` is
asserted to be ``AttentionFP8``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from rtp_llm.models_py.modules.dsv4._profiler import record_function_range
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import cached_swa_region

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rtp_llm.models_py.modules.dsv4.fp8.attention import PrefillMeta
    from rtp_llm.models_py.modules.dsv4.prefill_workspace import PrefillWorkspace
    from rtp_llm.models_py.modules.dsv4.transformer import V4Transformer


def _flat_optional(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if t is None else t.reshape(-1).contiguous()


def release_v41_prefill_shared(shared: Dict, layer_id: Optional[int] = None) -> None:
    """Release prefill tensors after their last consumer, or at forward exit.

    Calls run on the forward stream after a complete layer. Dropping references
    preserves allocator stream ordering without a device or host synchronization.
    The layer registry and paged KV pools are not per-forward scratch.
    """
    if layer_id is None:
        for key in (
            "ced_indexer_projection",
            "global",
            "topk",
            "candidates",
            "candidate_mask",
            "prefill_candidate_mask",
            "prefill_sparse_candidates",
            "prefill_sparse_plans",
            "prefill_score_bounds",
            "prefill_chunk_meta",
            "prefill_kv_workspace",
            "prefill_index_plan",
            "prefill_meta_common",
        ):
            shared.pop(key, None)
        return

    remaining = [attn for index, attn in shared["layers"].items() if index > layer_id]
    for key, source_attr, dependent_keys in (
        (
            "global",
            "kv_source_layer_id",
            ("prefill_chunk_meta", "prefill_kv_workspace"),
        ),
        ("topk", "index_source_layer_id", ("prefill_index_plan",)),
    ):
        values = shared.get(key, {})
        live_sources = {getattr(attn, source_attr, None) for attn in remaining}
        for source in tuple(values):
            if source not in live_sources:
                del values[source]
        if not values:
            for dependent in dependent_keys:
                shared.pop(dependent, None)

    if not any(getattr(attn, "is_index_source", False) for attn in remaining):
        for key in (
            "candidates",
            "candidate_mask",
            "prefill_candidate_mask",
            "prefill_sparse_candidates",
            "prefill_sparse_plans",
            "prefill_score_bounds",
        ):
            shared.pop(key, None)


def build_and_propagate_prefill_meta_fp8(
    v4: "V4Transformer",
    x_first_layer: torch.Tensor,
    start_pos: int,
    kv_cache,
    block_tables_by_type,
    *,
    sp_per_req: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    batch_size: int = 1,
    input_lengths: Optional[torch.Tensor] = None,
    prefix_lengths: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    req_id_per_token: Optional[torch.Tensor] = None,
    max_seqlen_q: int = 0,
    workspace: "PrefillWorkspace",
) -> None:
    """Build the layer-invariant prefill meta once per ``compress_ratio``
    bucket and broadcast each bucket's meta to its layers'
    ``AttentionFP8._prefill_meta_shared``.

    Called from ``prefill/forward.py::forward_layers`` once at the top of
    the layer loop, gated by ``if v4.fp8_kv_cache:``.

    The first layer of each unique ratio is picked as the rep to build
    the meta. ``kv_cache`` + ``block_tables_by_type`` are temporarily
    stashed on the rep attention so ``_pool_view`` /
    ``_pool_entries_per_block`` / FP8-pool-bound checks resolve without
    threading the framework handles through every signature.

    All three ratios must be prepared even if the request only exercises
    one of them, because every layer's ``forward`` reads its own
    ``_prefill_meta_shared`` and we propagate that here.
    """
    sp_per_req = _flat_optional(sp_per_req)
    cu_seqlens = _flat_optional(cu_seqlens)
    input_lengths = _flat_optional(input_lengths)
    prefix_lengths = _flat_optional(prefix_lengths)
    position_ids = _flat_optional(position_ids)
    req_id_per_token = _flat_optional(req_id_per_token)

    def bucket(attn):
        return (
            int(attn.compress_ratio),
            cached_swa_region(v4, kv_cache, attn.layer_id),
        )

    representatives: Dict[tuple, Any] = {}
    for layer in v4.layers:
        attn = getattr(layer, "attn", None)
        if attn is not None:
            representatives.setdefault(bucket(attn), attn)

    # SWA-only owns the superset metadata (common + SWA Group-1 + Group-2),
    # so build it first whenever the model has one. Compressed-only models use
    # their first ratio as the common source; later ratios still reuse the
    # ratio-independent tensors and SWA write metadata.
    ordered_ratios = sorted(representatives, key=lambda key: key[0] != 0)

    meta_by_ratio: Dict[tuple, "PrefillMeta"] = {}
    reusable_common: Dict[int, "PrefillMeta"] = {}
    reusable_freqs_by_rope_kind: Dict[bool, "PrefillMeta"] = {}
    # V4.1 buckets (ratios 0/2/1) build through ``AttentionV41FP8``'s SWA
    # planner override, which caches the built meta per (rope kind, input
    # identity) in the shared per-forward state. Drop that cache HERE, at the
    # top of every forward's build, so a stale entry from a previous (or
    # failed) forward can never be observed — the input-tensor identities in
    # the key are only unique while the cached meta pins them alive.
    first_attn = getattr(next(iter(v4.layers), None), "attn", None)
    first_shared = getattr(first_attn, "_shared_attention", None)
    if isinstance(first_shared, dict):
        first_shared.pop("prefill_meta_common", None)
    try:
        with record_function_range("dsv4.fp8.prefill_meta.build_all_ratios"):
            for key in ordered_ratios:
                r, swa_region = key
                attn = representatives[key]
                compressed_rope = r != 0
                from rtp_llm.models_py.modules.dsv4.fp8.attention import bind_attn_cache

                with bind_attn_cache(attn, kv_cache, block_tables_by_type):
                    with record_function_range(f"dsv4.fp8.prefill_meta.ratio_{r}"):
                        meta_by_ratio[key] = attn._build_shared_prefill_meta(
                            x_first_layer,
                            start_pos,
                            sp_per_req=sp_per_req,
                            cu_seqlens=cu_seqlens,
                            batch_size=batch_size,
                            input_lengths=input_lengths,
                            prefix_lengths=prefix_lengths,
                            position_ids=position_ids,
                            req_id_per_token=req_id_per_token,
                            max_seqlen_q=max_seqlen_q,
                            reuse_common_meta=reusable_common.get(swa_region),
                            reuse_freqs_meta=reusable_freqs_by_rope_kind.get(
                                compressed_rope
                            ),
                        )._replace(workspace=workspace)
                reusable_common.setdefault(swa_region, meta_by_ratio[key])
                reusable_freqs_by_rope_kind.setdefault(
                    compressed_rope, meta_by_ratio[key]
                )

        with record_function_range("dsv4.fp8.prefill_meta.propagate"):
            for layer in v4.layers:
                attn = getattr(layer, "attn", None)
                if attn is None:
                    continue
                # Each layer owns its own compressor / indexer; freqs_cis must
                # be bound per-layer (not just on the rep). Cheap idempotent
                # is-None set.
                attn._ensure_freqs_cis_bound()
                attn._set_prefill_meta_shared(meta_by_ratio.get(bucket(attn)))
    except BaseException:
        clear_prefill_meta_shared_fp8(v4)
        raise
    finally:
        # Bucket reuse ends here; modules retain their propagated metadata.
        # Also release the cache when a later bucket or propagation fails.
        if isinstance(first_shared, dict):
            first_shared.pop("prefill_meta_common", None)


def clear_prefill_meta_shared_fp8(v4: "V4Transformer") -> None:
    """Reverse of :func:`build_and_propagate_prefill_meta_fp8` — clears
    the per-layer ``AttentionFP8._prefill_meta_shared`` slot so a stale
    meta can't leak into the next forward."""
    for layer in v4.layers:
        attn = getattr(layer, "attn", None)
        if attn is None:
            continue
        attn._set_prefill_meta_shared(None)
