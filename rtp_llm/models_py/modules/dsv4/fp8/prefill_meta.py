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

import dataclasses
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from rtp_llm.models_py.modules.dsv4._profiler import record_function_range

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rtp_llm.models_py.modules.dsv4.fp8.attention import PrefillMeta
    from rtp_llm.models_py.modules.dsv4.transformer import V4Transformer


def _flat_optional(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if t is None else t.reshape(-1).contiguous()


# ----------------------------------------------------------------------
# Zero-SWA inverted-triangle trim helpers (Stage C)
# ----------------------------------------------------------------------
#
# The prefill meta is built ONCE per compress_ratio bucket over the full
# materialized token set [H-restore, P) (T_total rows). The inverted-triangle
# trim has each layer ``j`` run attention over only a contiguous SUFFIX
# ``h[ks:]`` of that set (``ks = k_start_t[j]``, the K-span front-trim count).
# Because we always front-trim, every per-layer token set is a suffix of the
# widest (bottom) set, so the per-layer meta is a cheap SUFFIX SLICE of the
# shared meta plus a handful of recomputed [B]/[B+1] scalars — never a rebuild.
#
# Field actions below are the authoritative contract verified field-by-field
# against the actual FP8 forward consumers (see
# ``zero_swa_less_compute/design/11_python_trim_writeskip.md`` and the
# field-audit work log). The load-bearing corrections to the original design
# note are called out inline:
#   * ``CompressorMeta.seq_start_per_req`` is [B] (NOT [N]) and must be
#     RECOMPUTED ``+ks`` (the fused compress kernel maps a token's kv_raw row
#     via ``flat_idx = cu_seq_per_req[req] + (pos - seq_start_per_req[req])``;
#     after trimming ``kv_raw = h[ks:]`` its row-0 is absolute pos sp0+ks, so
#     ``seq_start_per_req`` must advance by ks). Paired with ``cu_seq_per_req``
#     shift. Slicing it (per the original note) would empty the [B=1] tensor;
#     keeping it full mis-indexes every kept token by +ks.
#   * ``WorkspaceMeta.qsl`` (the workspace query_start_loc) must be SHIFTED via
#     ``_shift_cu_seqlens`` (the combine_topk kernel loops token_idx over
#     [0, query_len) into the SLICED cmp_topk/combined_lens buffers — a full
#     qsl would index T rows over a (T-ks)-row buffer = OOB).
#   * ``slot_in_flat`` / ``combined_indices`` / ``new_k_slot_in_flat`` are
#     anchored at the request token-0 origin (value at token ks == ks), so they
#     are SLICE-ONLY, NO rebase.
#   * The seq_len-side lengths (``swa_seq_lens`` / ``swa_gather_lens`` /
#     ``combined_gather_lens`` / ``M`` / ``N`` / ``cmp_seq_lens``) stay FULL:
#     the kernel derives start_pos = swa_seq_lens - query_len, so keeping them
#     full while query_len shrinks yields start_pos = sp + ks — exactly the
#     "K one window wider" anchor (correct, not a bug).


def _shift_cu_seqlens(
    cu: Optional[torch.Tensor], ks: int
) -> Optional[torch.Tensor]:
    """Front-trim ``ks`` tokens from a cumulative-offset tensor.

    For B==1 (the only shape the trim engages on) ``cu = [0, T]`` becomes
    ``[0, T-ks]``. ``clamp_min(0)`` keeps request-0's lower bound at 0.
    """
    if cu is None:
        return None
    return (cu - ks).clamp_min(0)


def _slice_meta(meta: "PrefillMeta", ks: int) -> "PrefillMeta":
    """Suffix-slice the shared ``PrefillMeta`` for the inverted-triangle trim.

    Returns a NEW ``PrefillMeta`` (NamedTuple ``_replace``) whose per-token
    tensors are sliced ``[ks:]`` and whose small [B]/[B+1] scalars are
    recomputed for the trimmed token set. ``ks`` is the K-span front-trim
    count (``k_start_t[j]``); the layer runs+writes back the full K-span
    ``[ks, T)``. No index rebasing — every workspace/pool coordinate is
    anchored at an absolute pool slot or the request token-0 origin
    (verified). B==1 / CP-off only.
    """
    if ks <= 0:
        return meta

    def _s(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return None if t is None else t[ks:]

    swa = meta.swa_meta
    if swa is not None:
        swa = swa._replace(
            slot_mapping=_s(swa.slot_mapping),
            query_start_loc=_shift_cu_seqlens(swa.query_start_loc, ks),
            combined_seq_lens=(
                None
                if swa.combined_seq_lens is None
                else swa.combined_seq_lens - ks
            ),
            topk_length_kv_full=_s(swa.topk_length_kv_full),
            combined_indices=_s(swa.combined_indices),
            combined_lens=_s(swa.combined_lens),
            slot_in_flat=_s(swa.slot_in_flat),
            # combined_gather_lens / combined_gather_len_max / M /
            # cache_seq_lens / cache_gather_lens / prefix_len_max /
            # cache_slot_mapping: kept full (K-side widths / prefix-tail
            # descriptors / workspace stride origin — trim-invariant).
        )

    csa = meta.csa_meta
    if csa is not None:
        csa = csa._replace(
            indexer_meta=_slice_indexer_meta(csa.indexer_meta, ks),
            compressor_meta=_slice_compressor_meta(csa.compressor_meta, ks),
            workspace_meta=_slice_workspace_meta(csa.workspace_meta, ks),
        )

    hca = meta.hca_meta
    if hca is not None:
        hca = hca._replace(
            compressor_meta=_slice_compressor_meta(hca.compressor_meta, ks),
            workspace_meta=_slice_workspace_meta(hca.workspace_meta, ks),
        )

    topk_idxs = meta.topk_idxs
    if topk_idxs is not None and topk_idxs.shape[0] == meta.seqlen:
        # Flat [T_total, win] rows into kv_full — dead on the via_concat hit
        # path (via_kv_full only). Slice for shape-consistency; no rebase.
        topk_idxs = topk_idxs[ks:]

    return meta._replace(
        seqlen=meta.seqlen - ks,
        seqlen_full=meta.seqlen_full - ks,
        freqs_cis=_s(meta.freqs_cis),
        topk_idxs=topk_idxs,
        sp_int=meta.sp_int + ks,
        row_seqlens_full=(
            None if meta.row_seqlens_full is None else meta.row_seqlens_full - ks
        ),
        cu_seqlens=_shift_cu_seqlens(meta.cu_seqlens, ks),
        input_lengths=(
            None if meta.input_lengths is None else meta.input_lengths - ks
        ),
        position_ids=_s(meta.position_ids),
        req_id_per_token=_s(meta.req_id_per_token),
        swa_meta=swa,
        csa_meta=csa,
        hca_meta=hca,
    )


def _slice_workspace_meta(wm: Optional[Any], ks: int) -> Optional[Any]:
    """Suffix-slice a ``WorkspaceMeta`` (CSA/HCA). Only the two per-token
    [T_total] fields slice; ``qsl`` shifts; everything else is trim-invariant
    (widths / pool geometry / prefix-tail descriptors)."""
    if wm is None:
        return None

    def _s(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return None if t is None else t[ks:]

    return wm._replace(
        qsl=_shift_cu_seqlens(wm.qsl, ks),
        dense_cmp_topk=_s(wm.dense_cmp_topk),
        new_k_slot_in_flat=_s(wm.new_k_slot_in_flat),
    )


def _slice_compressor_meta(cm: Optional[Any], ks: int) -> Optional[Any]:
    """Suffix-slice a ``CompressorMeta`` (host CSA/HCA + nested indexer).

    Per-token [N==T_total] fields slice ``[ks:]`` (absolute values — no
    rebase). ``seq_start_per_req`` is [B] and RECOMPUTES ``+ks`` (NOT a slice —
    it is the kv_raw row-0 absolute position); ``cu_seq_per_req`` shifts.
    The two MUST be applied together or the fused compress kernel mis-indexes
    ``kv_raw`` by +ks for every kept token."""
    if cm is None:
        return None

    def _s(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return None if t is None else t[ks:]

    return dataclasses.replace(
        cm,
        positions=_s(cm.positions),
        b_idx=_s(cm.b_idx),
        state_slots=_s(cm.state_slots),
        kv_slots=_s(cm.kv_slots),
        token_to_req=_s(cm.token_to_req),
        seq_start_per_req=(
            None if cm.seq_start_per_req is None else cm.seq_start_per_req + ks
        ),
        cu_seq_per_req=_shift_cu_seqlens(cm.cu_seq_per_req, ks),
    )


def _slice_indexer_meta(im: Optional[Any], ks: int) -> Optional[Any]:
    """Suffix-slice an ``_IndexerFP8PrefillMeta``. Per-Q-row fields slice; the
    compressed-K side (T / cu_kv_seqlens / block_table) is trim-invariant; the
    nested compressor_meta recurses (incl. the ``seq_start_per_req + ks``
    rebase)."""
    if im is None:
        return None

    def _s(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return None if t is None else t[ks:]

    return im._replace(
        seqlen=im.seqlen - ks,
        M=im.M - ks,
        freqs_cis_slice=_s(im.freqs_cis_slice),
        positions_d=_s(im.positions_d),
        ks=_s(im.ks),  # struct field: per-Q-row K-window start; slice rows, NOT a value rebase
        ke=_s(im.ke),  # struct field: per-Q-row K-window end
        cu_kv_per_token=_s(im.cu_kv_per_token),  # None on B==1
        compressor_meta=_slice_compressor_meta(im.compressor_meta, ks),
    )


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
    write_skip_restore_window: int = 0,
) -> Dict[int, "PrefillMeta"]:
    """Build the layer-invariant prefill meta once per ``compress_ratio``
    bucket and broadcast each bucket's meta to its layers'
    ``AttentionFP8._prefill_meta_shared``.

    Returns ``meta_by_ratio`` (the per-ratio shared meta) so the layer loop can
    hand each layer a per-layer SUFFIX SLICE of it under the inverted-triangle
    trim (Stage C). The broadcast still sets each layer's shared slot to the
    FULL meta here, so when the trim is OFF the loop never touches the shared
    slot and behavior is byte-identical to before; under trim the loop only
    OVERRIDES the trimmed layers via ``_set_prefill_meta_shared(_slice_meta(...))``.

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
                        write_skip_restore_window=write_skip_restore_window,
                    )
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

    return meta_by_ratio


def clear_prefill_meta_shared_fp8(v4: "V4Transformer") -> None:
    """Reverse of :func:`build_and_propagate_prefill_meta_fp8` — clears
    the per-layer ``AttentionFP8._prefill_meta_shared`` slot so a stale
    meta can't leak into the next forward."""
    for layer in v4.layers:
        attn = getattr(layer, "attn", None)
        if attn is None:
            continue
        attn._set_prefill_meta_shared(None)
