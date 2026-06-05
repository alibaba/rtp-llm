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

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rtp_llm.models_py.modules.dsv4.fp8.attention import PrefillMeta
    from rtp_llm.models_py.modules.dsv4.prefill_workspace import PrefillWorkspace
    from rtp_llm.models_py.modules.dsv4.transformer import V4Transformer


def _flat_optional(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if t is None else t.reshape(-1).contiguous()


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

    meta_by_ratio: Dict[int, "PrefillMeta"] = {}
    with record_function_range("dsv4.fp8.prefill_meta.build_all_ratios"):
        for layer in v4.layers:
            attn = getattr(layer, "attn", None)
            if attn is None:
                continue
            r = int(attn.compress_ratio)
            if r in meta_by_ratio:
                continue
            prev_kv = attn._kv_cache
            prev_bt = attn._block_tables_by_type
            if kv_cache is not None:
                attn._kv_cache = kv_cache
            if block_tables_by_type is not None:
                attn._block_tables_by_type = block_tables_by_type
            try:
                with record_function_range(f"dsv4.fp8.prefill_meta.ratio_{r}"):
                    meta_by_ratio[r] = attn._build_shared_prefill_meta(
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
                    )._replace(workspace=workspace)
            finally:
                attn._kv_cache = prev_kv
                attn._block_tables_by_type = prev_bt

    with record_function_range("dsv4.fp8.prefill_meta.propagate"):
        for layer in v4.layers:
            attn = getattr(layer, "attn", None)
            if attn is None:
                continue
            # Each layer owns its own compressor / indexer; freqs_cis must
            # be bound per-layer (not just on the rep). Cheap idempotent
            # is-None set.
            attn._ensure_freqs_cis_bound()
            attn._set_prefill_meta_shared(meta_by_ratio.get(int(attn.compress_ratio)))


def clear_prefill_meta_shared_fp8(v4: "V4Transformer") -> None:
    """Reverse of :func:`build_and_propagate_prefill_meta_fp8` — clears
    the per-layer ``AttentionFP8._prefill_meta_shared`` slot so a stale
    meta can't leak into the next forward."""
    for layer in v4.layers:
        attn = getattr(layer, "attn", None)
        if attn is None:
            continue
        attn._set_prefill_meta_shared(None)
