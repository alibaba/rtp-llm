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

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from rtp_llm.models_py.modules.dsv4._profiler import record_function_range

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rtp_llm.models_py.modules.dsv4.fp8.attention import PrefillMeta
    from rtp_llm.models_py.modules.dsv4.prefill_workspace import PrefillWorkspace
    from rtp_llm.models_py.modules.dsv4.transformer import V4Transformer


# Per-forward cross-bucket shared metadata build.  The 3 ratio buckets
# (SWA/CSA/HCA) are built from IDENTICAL arguments (same tensors, same
# scalars; only the rep attention module differs), so the bucket-invariant
# pieces — freqs/topk gather, SWA Group-1 write meta, ``row_seqlens_full``,
# the workspace-meta SWA half, and the CP full-positions build — are computed
# once and shared instead of 2-3x per chunk.  Values are byte-identical
# (same expressions over the same inputs, proven by identity probes);
# ``shared=None`` reproduces today's per-bucket builds exactly.
_META_SHARED_BUILD_FLAG = "DSV4_FP8_PREFILL_META_SHARED_BUILD"


def _meta_shared_build_enabled() -> bool:
    value = os.environ.get(_META_SHARED_BUILD_FLAG, "1")
    if value not in ("0", "1"):
        raise ValueError(f"{_META_SHARED_BUILD_FLAG} must be 0 or 1, got {value!r}")
    return value == "1"


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
    # one per-forward memo shared by every ratio-bucket build (None =
    # legacy per-bucket rebuild, byte-identical values either way).
    shared: Optional[Dict[str, Any]] = {} if _meta_shared_build_enabled() else None
    with record_function_range("dsv4.fp8.prefill_meta.build_all_ratios"):
        for layer in v4.layers:
            attn = getattr(layer, "attn", None)
            if attn is None:
                continue
            r = int(attn.compress_ratio)
            if r in meta_by_ratio:
                continue
            from rtp_llm.models_py.modules.dsv4.fp8.attention import bind_attn_cache

            with bind_attn_cache(attn, kv_cache, block_tables_by_type):
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
                        shared=shared,
                    )._replace(workspace=workspace)

    with record_function_range("dsv4.fp8.prefill_meta.propagate"):
        _propagate_meta_map(v4, meta_by_ratio)


def _propagate_meta_map(
    v4: "V4Transformer", meta_by_ratio: Dict[int, "PrefillMeta"]
) -> None:
    """Bind each layer's bucket meta + freqs (the shared propagate loop)."""
    for layer in v4.layers:
        attn = getattr(layer, "attn", None)
        if attn is None:
            continue
        # Each layer owns its own compressor / indexer; freqs_cis must
        # be bound per-layer (not just on the rep). Cheap idempotent
        # is-None set.
        attn._ensure_freqs_cis_bound()
        attn._set_prefill_meta_shared(meta_by_ratio.get(int(attn.compress_ratio)))


def propagate_prebuilt_prefill_meta_fp8(
    v4: "V4Transformer",
    bundle,
    kv_cache,
    block_tables_by_type,
    workspace: "PrefillWorkspace",
) -> bool:
    """patch + propagate a prebuilt (``DSV4_PREFILL_ASYNC_HEAD``) meta set.

    ``patch_prebuilt_metas`` rebuilds the block-table-dependent fields against
    the real per-chunk tables; the rest of the bundle is byte-identical to the
    eager build by construction (validated at consume time). Returns ``True``
    on engagement; any patch failure logs once and returns ``False`` so the
    caller runs the legacy build — the fail-closed direction is always the
    eager path.
    """
    from rtp_llm.models_py.modules.dsv4.fp8 import _head_prebuild

    try:
        _head_prebuild.patch_prebuilt_metas(bundle, kv_cache, block_tables_by_type)
    except Exception:
        if not getattr(propagate_prebuilt_prefill_meta_fp8, "_failure_logged", False):
            logger.exception(
                "[dsv4-async-head] patch failed; running the legacy meta build"
            )
            propagate_prebuilt_prefill_meta_fp8._failure_logged = True
        return False
    meta_by_ratio = {
        int(r): meta._replace(workspace=workspace)
        for r, meta in bundle.meta_by_ratio.items()
    }
    with record_function_range("dsv4.fp8.prefill_meta.propagate"):
        _propagate_meta_map(v4, meta_by_ratio)
    return True


def clear_prefill_meta_shared_fp8(v4: "V4Transformer") -> None:
    """Reverse of :func:`build_and_propagate_prefill_meta_fp8` — clears
    the per-layer ``AttentionFP8._prefill_meta_shared`` slot so a stale
    meta can't leak into the next forward."""
    for layer in v4.layers:
        attn = getattr(layer, "attn", None)
        if attn is None:
            continue
        attn._set_prefill_meta_shared(None)
