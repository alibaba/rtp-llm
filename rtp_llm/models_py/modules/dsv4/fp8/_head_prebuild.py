"""Predictively prebuild the next context-parallel prefill chunk on a worker.

Only geometry and metadata are prepared. Fresh block-table fields are patched
at consumption, after validating the request and predicted chunk. Unsupported
inputs and builder failures fall back to eager preparation. The worker mirrors
the caller inference mode; CUDA events order publication and consumption.

Enabled by default; DSV4_PREFILL_ASYNC_HEAD=0 disables prebuilding.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_ASYNC_HEAD_FLAG = "DSV4_PREFILL_ASYNC_HEAD"


def async_head_enabled() -> bool:
    """Fail-closed parse: only '0' and '1' are accepted."""
    value = os.environ.get(_ASYNC_HEAD_FLAG, "1")
    if value not in ("0", "1"):
        raise ValueError(f"{_ASYNC_HEAD_FLAG} must be 0 or 1, got {value!r}")
    return value == "1"


# ---------------------------------------------------------------------------
# Recipe domain gate + prediction key
# ---------------------------------------------------------------------------
#
# The synthesized cp_info uses ``_compact_cp_runtime._zigzag_constants`` (the
# CP4 / 4096-padded / full-chunk zigzag pattern) and the entry content check
# revalidates the REAL mask/restore against that exact pattern, so the
# prebuild only ever engages for the recipe geometry. Other geometries fail
# closed to the eager build.


def _recipe_domain_ok(cp_ctx) -> bool:
    return (
        cp_ctx is not None
        and int(cp_ctx.cp_size) == 4
        and int(cp_ctx.chunk_length) == 1024
        and int(cp_ctx.padded_seq_len) == 4096
        and int(cp_ctx.seq_len_full) == 4096
        and not bool(getattr(cp_ctx, "kv_cache_sharded", False))
        and tuple(getattr(cp_ctx, "chunk_lengths_per_req", None) or ()) == (1024,)
        and tuple(getattr(cp_ctx, "input_lengths_full_host", None) or ()) == (4096,)
        and getattr(cp_ctx, "prefix_lengths_full_host", None) is not None
        and len(cp_ctx.prefix_lengths_full_host) == 1
    )


def predicted_continuation_key(cp_ctx, device) -> Tuple:
    """Key for the NEXT chunk of the same request (prefix += padded_seq_len)."""
    return (
        int(cp_ctx.cp_size),
        int(cp_ctx.cp_rank),
        int(cp_ctx.chunk_length),
        int(cp_ctx.padded_seq_len),
        (1024,),
        (4096,),
        int(cp_ctx.prefix_lengths_full_host[0]) + int(cp_ctx.padded_seq_len),
        str(device),
    )


def key_from_forward_inputs(
    cp_info,
    prefix_lengths: Optional[torch.Tensor],
    device,
    cp_size: int,
    cp_rank: int,
    num_tokens: int,
) -> Optional[Tuple]:
    """The same key computed at forward entry from the real inputs.

    Host reads only: the mask shape is metadata, ``actual_input_lengths_cpu`` /
    ``cp_chunk_lengths`` are CPU tensors, and ``prefix_lengths`` is the
    framework's pinned host tensor in production. Returns ``None`` (decline)
    whenever any input is missing or device-resident — never pay a D2H here.
    """
    mask = getattr(cp_info, "prefill_qkv_padding_mask", None)
    lengths_cpu = getattr(cp_info, "prefill_actual_input_lengths_cpu", None)
    if mask is None or lengths_cpu is None or int(lengths_cpu.numel()) != 1:
        return None
    if prefix_lengths is None or int(prefix_lengths.numel()) != 1:
        return None
    if prefix_lengths.device.type != "cpu":
        # The production path hands a pinned host tensor; anything else means
        # a config we cannot read without a sync — decline.
        return None
    chunk_lengths_obj = getattr(cp_info, "prefill_cp_chunk_lengths", None)
    if (
        chunk_lengths_obj is not None
        and int(chunk_lengths_obj.numel()) > 0
        and chunk_lengths_obj.device.type == "cpu"
    ):
        chunk_lengths = tuple(int(v) for v in chunk_lengths_obj.reshape(-1).tolist())
    else:
        chunk_lengths = (int(num_tokens),)
    return (
        int(cp_size),
        int(cp_rank),
        int(num_tokens),
        int(mask.shape[0]),
        chunk_lengths,
        tuple(int(v) for v in lengths_cpu.reshape(-1).tolist()),
        int(prefix_lengths.reshape(-1)[0].item()),
        str(device),
    )


# ---------------------------------------------------------------------------
# The prebuilt bundle
# ---------------------------------------------------------------------------


class PrebuiltHead:
    """One predicted chunk's context + metadata (block-table fields stale).

    ``meta_by_ratio`` maps compress_ratio -> the rep attention's PrefillMeta
    whose block-table-dependent fields (slot mappings, ``block_table_i32``)
    were built against the CURRENT chunk's block table and are therefore
    stale; ``patch_prebuilt_metas`` rebuilds exactly those fields against the
    real table at consume time. ``rep_attns`` maps ratio -> the rep module.
    """

    __slots__ = (
        "key",
        "cp_ctx",
        "meta_by_ratio",
        "rep_attns",
        "meta_args",
        "sp_int",
        "event",
    )

    def __init__(self, key, cp_ctx, meta_by_ratio, rep_attns, meta_args, sp_int):
        self.key = key
        self.cp_ctx = cp_ctx
        self.meta_by_ratio = meta_by_ratio
        self.rep_attns = rep_attns
        self.meta_args = meta_args
        self.sp_int = sp_int
        self.event = None  # set by the builder wrapper on CUDA


# ---------------------------------------------------------------------------
# The builder thread
# ---------------------------------------------------------------------------


class AsyncHeadBuilder:
    """One daemon thread running one prediction build at a time.

    ``kick`` posts work (called at the end of ``forward_layers``); ``consume``
    joins and validates (called at the top of the next ``forward_layers``).
    The builder ALWAYS signals the done event, including on exceptions, so a
    consumer can never wedge. Not engage-rate-limited by design: a miss costs
    one discarded build, never a wrong value.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._pending: Optional[Tuple[Tuple, Any]] = None
        self._result: Optional[Tuple[Tuple, Optional[PrebuiltHead]]] = None
        self._busy = False
        self._done = threading.Event()
        self._done.set()
        self._thread: Optional[threading.Thread] = None
        self._kicked = 0
        self._built = 0
        self._failed = 0

    # -- main-thread API ----------------------------------------------------
    def kick(self, key: Tuple, build_fn) -> None:
        with self._cond:
            # One outstanding build at a time. A stale unconsumed result is
            # fenced+dropped by the caller (consume path) before we get here.
            while self._pending is not None or self._busy:
                self._cond.wait()
            self._result = None
            self._done.clear()
            self._pending = (key, build_fn)
            self._kicked += 1
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._loop, name="dsv4-async-head", daemon=True
                )
                self._thread.start()
            self._cond.notify_all()

    def consume(self, key: Tuple) -> Optional[PrebuiltHead]:
        with self._cond:
            idle = self._pending is None and not self._busy and self._result is None
        if idle:
            return None
        self._done.wait()  # join: the builder always signals, even on failure
        with self._cond:
            result = self._result
            self._result = None
        if result is None:
            return None
        _result_key, bundle = result
        if bundle is None or _result_key != key:
            return None
        return bundle

    # -- builder thread -----------------------------------------------------
    def _loop(self) -> None:
        while True:
            with self._cond:
                while self._pending is None:
                    self._cond.wait()
                key, build_fn = self._pending
                self._pending = None
                self._busy = True
            bundle: Optional[PrebuiltHead] = None
            try:
                bundle = build_fn()
                self._built += 1
            except Exception:
                self._failed += 1
                if not getattr(AsyncHeadBuilder, "_failure_logged", False):
                    logger.exception(
                        "[dsv4-async-head] predictive head build failed; "
                        "falling back to the eager per-chunk build"
                    )
                    AsyncHeadBuilder._failure_logged = True
                bundle = None
            with self._cond:
                self._result = (key, bundle)
                self._busy = False
                self._done.set()
                self._cond.notify_all()


_BUILDER: Optional[AsyncHeadBuilder] = None
_BUILDER_LOCK = threading.Lock()
_BUILDER_STREAMS: Dict[Any, Any] = {}


def _builder() -> AsyncHeadBuilder:
    global _BUILDER
    with _BUILDER_LOCK:
        if _BUILDER is None:
            _BUILDER = AsyncHeadBuilder()
        return _BUILDER


def _builder_stream(device):
    stream = _BUILDER_STREAMS.get(device)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _BUILDER_STREAMS[device] = stream
    return stream


# ---------------------------------------------------------------------------
# The predictive build (builder thread)
# ---------------------------------------------------------------------------


def _synthesized_cp_info():
    """Host-side cp_info for a full 4096-token CP4 chunk (zigzag pattern).

    The content is the exact ``_verify_content`` reference: an all-ones mask
    and the zigzag ``argsort`` restore. Byte-identical to the framework's
    per-chunk tensors whenever the production content check passes.
    """
    from rtp_llm.models_py.modules.dsv4.fp8._compact_cp_runtime import _zigzag_constants

    _, restore = _zigzag_constants()
    return types_simple_namespace(
        prefill_qkv_padding_mask=torch.ones(4096, dtype=torch.int32),
        prefill_qkv_restore_indice=restore.to(torch.int32).contiguous(),
        prefill_actual_input_lengths_cpu=torch.tensor([4096], dtype=torch.int32),
        prefill_cp_chunk_lengths=torch.tensor([1024], dtype=torch.int32),
    )


def types_simple_namespace(**kwargs):
    # Kept as a function so tests can swap it; production uses SimpleNamespace.
    import types as _types

    return _types.SimpleNamespace(**kwargs)


def _build_bundle_inner(
    *,
    key,
    v4,
    kv_cache,
    block_tables_by_type,
    device,
    cp_size: int,
    cp_rank: int,
    predicted_prefix: int,
    shared,
):
    """Run the REAL context + meta builders with the predicted inputs.

    Every block-table-dependent field produced here is stale (built against
    the current chunk's table) and rebuilt at consume time by
    :func:`patch_prebuilt_metas`; everything else is byte-identical to what
    the eager path would compute for the predicted chunk.
    """
    from rtp_llm.models_py.modules.dsv4.cp import (
        build_cp_context_for_forward,
        first_position_for_meta,
    )
    from rtp_llm.models_py.modules.dsv4.fp8.attention import bind_attn_cache

    cp_info = _synthesized_cp_info()
    prefix_host = torch.tensor([predicted_prefix], dtype=torch.int32)
    cp_ctx = build_cp_context_for_forward(
        cp_info,
        cp_size,
        cp_rank,
        1024,
        device,
        prefix_lengths=prefix_host,
        kv_cache_sharded=False,
    )

    # Meta-build arguments, mirroring prefill/forward.py's prep block.
    sp_int = first_position_for_meta(cp_ctx, cp_ctx.global_positions)
    sp_per_req = cp_ctx.prefix_lengths.to(device=device, dtype=torch.int64).contiguous()
    req_id_per_token = cp_ctx.req_id_per_token.to(
        device=device, dtype=torch.int32
    ).contiguous()
    cu_seqlens = torch.tensor([0, 1024], dtype=torch.int32, device=device)
    input_lengths = torch.tensor([1024], dtype=torch.int32, device=device)
    prefix_lengths = torch.tensor([predicted_prefix], dtype=torch.int32, device=device)
    position_ids = cp_ctx.global_positions.to(device=device, dtype=torch.long)
    meta_args = dict(
        sp_per_req=sp_per_req,
        cu_seqlens=cu_seqlens,
        batch_size=1,
        input_lengths=input_lengths,
        prefix_lengths=prefix_lengths,
        position_ids=position_ids,
        req_id_per_token=req_id_per_token,
        max_seqlen_q=1024,
    )
    # ``x`` is read by the meta build for ``shape[0]`` / ``device`` only
    # (verified in ``_build_shared_prefill_meta``); a tiny stand-in suffices.
    x_stand_in = torch.empty((1024,), dtype=torch.int8, device=device)

    meta_by_ratio: Dict[int, Any] = {}
    rep_attns: Dict[int, Any] = {}
    for layer in v4.layers:
        attn = getattr(layer, "attn", None)
        if attn is None:
            continue
        r = int(attn.compress_ratio)
        if r in meta_by_ratio:
            continue
        # The builders read ``self._cp_ctx``; bind the predicted context on
        # exactly the modules that read it (mirroring
        # ``V4Transformer._propagate_cp_ctx``'s per-module set), then restore.
        swapped = _swap_cp_ctx(attn, cp_ctx)
        try:
            with bind_attn_cache(attn, kv_cache, block_tables_by_type):
                meta_by_ratio[r] = attn._build_shared_prefill_meta(
                    x_stand_in,
                    sp_int,
                    sp_per_req=sp_per_req,
                    cu_seqlens=cu_seqlens,
                    batch_size=1,
                    input_lengths=input_lengths,
                    prefix_lengths=prefix_lengths,
                    position_ids=position_ids,
                    req_id_per_token=req_id_per_token,
                    max_seqlen_q=1024,
                    shared=shared,
                )
        finally:
            _restore_cp_ctx(attn, swapped)
        rep_attns[r] = attn
    return PrebuiltHead(key, cp_ctx, meta_by_ratio, rep_attns, meta_args, sp_int)


def _swap_cp_ctx(attn, new_ctx):
    """Bind ``new_ctx`` on one rep attention + its compressor/indexer children.

    Returns the bundle of previous contexts for restoration. Mirrors
    ``V4Transformer._propagate_cp_ctx``'s traversal exactly (attn → compressor
    → indexer → indexer.compressor).
    """
    prev = [getattr(attn, "_cp_ctx", None)]
    attn.set_cp_ctx(new_ctx)
    c = getattr(attn, "compressor", None)
    prev.append(getattr(c, "_cp_ctx", None) if c is not None else None)
    if c is not None:
        c.set_cp_ctx(new_ctx)
    idx = getattr(attn, "indexer", None)
    prev.append(getattr(idx, "_cp_ctx", None) if idx is not None else None)
    if idx is not None:
        idx.set_cp_ctx(new_ctx)
        ic = getattr(idx, "compressor", None)
        prev.append(getattr(ic, "_cp_ctx", None) if ic is not None else None)
        if ic is not None:
            ic.set_cp_ctx(new_ctx)
    else:
        prev.append(None)
    return prev


def _restore_cp_ctx(attn, prev) -> None:
    c = getattr(attn, "compressor", None)
    idx = getattr(attn, "indexer", None)
    ic = getattr(idx, "compressor", None) if idx is not None else None
    attn.set_cp_ctx(prev[0])
    if c is not None:
        c.set_cp_ctx(prev[1])
    if idx is not None:
        idx.set_cp_ctx(prev[2])
        if ic is not None:
            ic.set_cp_ctx(prev[3])


def _build_under_inference_mode(kwargs, inference_mode: bool):
    """Mirror the caller's inference mode on the builder thread.

    Pinned buffers allocated during inference are inference tensors. Updating
    them outside that mode raises, while non-inference callers must retain
    their original behavior.
    """
    with torch.inference_mode(inference_mode):
        return _build_bundle_inner(**kwargs)


def _build_with_stream_fence(
    kwargs, inference_mode: bool = True
) -> Optional[PrebuiltHead]:
    device = kwargs["device"]
    if device.type == "cuda":
        stream = _builder_stream(device)
        with torch.cuda.stream(stream):
            bundle = _build_under_inference_mode(kwargs, inference_mode)
        event = torch.cuda.Event()
        event.record(stream)
        bundle.event = event
        return bundle
    return _build_under_inference_mode(kwargs, inference_mode)


# ---------------------------------------------------------------------------
# Kick (end of forward_layers, main thread)
# ---------------------------------------------------------------------------


def maybe_kick_async_head(v4, kv_cache, block_tables_by_type, cp_ctx, device) -> None:
    """Kick the builder for the predicted next chunk. Never raises."""
    try:
        if not async_head_enabled():
            return
        if not _recipe_domain_ok(cp_ctx):
            return
        if kv_cache is None or not block_tables_by_type:
            return
        if not bool(getattr(v4, "fp8_kv_cache", False)):
            # The metadata prebuild only pays for itself on the FP8 path (the
            # BF16 path has no broadcast meta build to hide).
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
        from rtp_llm.models_py.modules.dsv4.fp8.prefill_meta import (
            _meta_shared_build_enabled,
        )

        key = predicted_continuation_key(cp_ctx, device)
        kwargs = dict(
            key=key,
            v4=v4,
            kv_cache=kv_cache,
            block_tables_by_type=block_tables_by_type,
            device=device,
            cp_size=int(cp_ctx.cp_size),
            cp_rank=int(cp_ctx.cp_rank),
            predicted_prefix=int(cp_ctx.prefix_lengths_full_host[0])
            + int(cp_ctx.padded_seq_len),
            shared={} if _meta_shared_build_enabled() else None,
        )
        # Capture the serving grad mode HERE (main thread, inside the
        # inference_mode forward) — the lambda body runs later on the builder
        # thread, where ``is_inference_mode_enabled()`` would read the wrong
        # (builder) state.
        inference_mode = torch.is_inference_mode_enabled()
        _builder().kick(key, lambda: _build_with_stream_fence(kwargs, inference_mode))
    except Exception:
        # The kick itself must never break a forward.
        if not getattr(maybe_kick_async_head, "_failure_logged", False):
            logger.exception("[dsv4-async-head] kick failed; async head disabled")
            maybe_kick_async_head._failure_logged = True


# ---------------------------------------------------------------------------
# Consume (top of forward_layers, main thread)
# ---------------------------------------------------------------------------


def _content_matches_recipe(cp_info) -> bool:
    """The real chunk's mask/restore equal the zigzag pattern the builder used.

    Device-resident sources take the pinned-nonblocking DtoH + event,
    resolved immediately: at forward entry the queue is empty (the previous
    chunk ended in a full device drain), so the resolve never stalls on the
    previous chunk's work (~tens of µs). CPU sources (tests) take the direct
    host comparison.
    """
    from rtp_llm.models_py.modules.dsv4.fp8._compact_cp_runtime import (
        _PendingVerify,
        _verify_content,
    )

    mask = getattr(cp_info, "prefill_qkv_padding_mask", None)
    restore = getattr(cp_info, "prefill_qkv_restore_indice", None)
    if mask is None or restore is None:
        return False
    if (
        mask.is_cuda
        and restore.is_cuda
        and torch.cuda.is_available()
        and not torch.cuda.is_current_stream_capturing()
    ):
        try:
            return bool(_PendingVerify(mask, restore).resolve())
        except Exception:
            return False
    return bool(
        _verify_content(
            mask.detach().to(device="cpu").reshape(-1),
            restore.detach().to(device="cpu", dtype=torch.long).reshape(-1),
        )
    )


def consume_async_head(
    cp_info,
    prefix_lengths: Optional[torch.Tensor],
    device,
    cp_size: int,
    cp_rank: int,
    num_tokens: int,
) -> Optional[PrebuiltHead]:
    """Join the builder and validate its bundle for THIS chunk.

    Returns ``None`` (legacy eager path) on any miss: flag off, no prediction
    outstanding, key mismatch, or mask/restore content mismatch. On a hit the
    caller must still run :func:`patch_prebuilt_metas` before using the meta.
    """
    if not async_head_enabled():
        return None
    if (
        torch.cuda.is_available()
        and device.type == "cuda"
        and torch.cuda.is_current_stream_capturing()
    ):
        # The join/fence/content-check are not capture-legal; decline.
        return None
    key = key_from_forward_inputs(
        cp_info, prefix_lengths, device, cp_size, cp_rank, num_tokens
    )
    if key is None:
        return None
    builder = _builder()
    bundle = builder.consume(key)
    if bundle is None:
        return None
    # Content check: the bundle's context content tensors were built from the
    # zigzag pattern; they equal the eager build's outputs iff the REAL
    # mask/restore match that pattern (the same verdict the compact-CP
    # ``verified_geometry`` produces, resolved here instead of at first use).
    if not _content_matches_recipe(cp_info):
        _fence_bundle(bundle, device)
        return None
    # Fence the builder's device work into the main stream before any consumer
    # reads (or the allocator recycles) the bundle tensors.
    _fence_bundle(bundle, device)
    return bundle


def _fence_bundle(bundle: PrebuiltHead, device) -> None:
    """Order the main stream after the builder's device work.

    Runs on hits (before consumption) and on content-misses (before the
    bundle's tensors can be freed), so the caching allocator never recycles
    bundle storage into main-stream work that isn't ordered after the
    builder's writes.
    """
    if bundle.event is not None and device.type == "cuda":
        torch.cuda.current_stream(device).wait_event(bundle.event)


# ---------------------------------------------------------------------------
# Patch (consume time): rebuild the block-table-dependent fields
# ---------------------------------------------------------------------------
#
# The prebuilt metas were built against the PREVIOUS chunk's block table; the
# slot-mapping fields below are the only ones that read the table. Each patch
# calls the SAME leaf builder the legacy code calls inline, with the SAME
# argument values (carried on the prebuilt meta / cp_ctx), so the patched field
# is byte-identical to the legacy eager build's. Anything outside the recipe
# domain raises ``_PatchOutOfDomain`` and the caller falls back to the full
# legacy build.


class _PatchOutOfDomain(RuntimeError):
    pass


def _patch_swa_meta(attn, meta, cp_ctx, tables):
    """Rebuild ``swa_meta``'s block-table fields (Group-1 + Group-2 tail)."""
    from rtp_llm.models_py.modules.dsv4.fp8 import _swa_ops_triton as _swa_ops
    from rtp_llm.models_py.modules.dsv4.fp8._kv_cache_utils import (
        require_pool_tokens_per_block,
    )
    from rtp_llm.models_py.modules.dsv4.fp8.attention import (
        _build_suffix_pool_slot_mapping,
        _flat_1d,
        _suffix_gather_lens_max_host,
    )
    from rtp_llm.models_py.modules.dsv4.kv_cache_utils import SWA_KV

    swa = meta.swa_meta
    if swa is None:
        return None
    device = meta.device
    win = attn.window_size
    eb = attn._swa_entries_per_block()
    swa_tpb = require_pool_tokens_per_block(attn._kv_cache, tag=SWA_KV)
    bt = attn._block_tables_by_type.get(SWA_KV) if attn._block_tables_by_type else None
    if bt is None or int(bt.numel()) == 0:
        raise _PatchOutOfDomain("SWA block table missing at patch time")

    # Group-1 write meta — mirrors _build_swa_prefill_meta_varlen's CP write
    # trio: the async-head domain gate requires cp_on_write.
    if not (
        cp_ctx is not None
        and cp_ctx.cp_size > 1
        and cp_ctx.cu_seqlens_global is not None
        and cp_ctx.input_lengths_global is not None
    ):
        raise _PatchOutOfDomain("SWA patch requires the CP write trio")
    write_B = int(cp_ctx.input_lengths_global.numel())
    write_query_start_loc = _flat_1d(
        cp_ctx.cu_seqlens_global.to(device=device, dtype=torch.int32)
    ).contiguous()
    write_combined_seq_lens = (
        meta.prefix_lengths.to(torch.int32)[:write_B]
        + _flat_1d(cp_ctx.input_lengths_global.to(torch.int32))
    ).contiguous()
    bt_swa = bt[:write_B].to(device=device, dtype=torch.int32).contiguous()
    slot_mapping = _swa_ops.compute_swa_slot_mapping(
        block_table=bt_swa,
        query_start_loc=write_query_start_loc,
        seq_lens=write_combined_seq_lens,
        num_tokens=cp_ctx.seq_len_full,
        pool_entries_per_block=eb,
        tokens_per_block_for_block_table=swa_tpb,
        ring_entries=eb,
    )
    slot_compaction = attn._build_swa_cp_byte_compaction(
        slot_mapping,
        full_entries_per_block=eb,
        validation_site="swa.quantize_and_insert_cp_byte.slot_mapping",
        negative_mode="skip_minus_one",
    )
    swa = swa._replace(slot_mapping=slot_mapping, slot_compaction=slot_compaction)

    # Group-2 cache tail — present exactly when the legacy build produced it
    # (SWA bucket on a continuation chunk).
    if swa.cache_slot_mapping is not None:
        cache_slot_mapping = _build_suffix_pool_slot_mapping(
            block_table=bt_swa,
            seq_lens=swa.cache_seq_lens,
            gather_lens=swa.cache_gather_lens,
            entries_per_block=eb,
            tokens_per_block_for_block_table=swa_tpb,
            ring_entries=eb,
            max_gather_host=_suffix_gather_lens_max_host(
                meta.prefix_lengths,
                cp_ctx.prefix_lengths_full_host,
                write_B,
                win,
            ),
        )
        cache_compaction = attn._build_swa_cp_byte_compaction(
            cache_slot_mapping,
            full_entries_per_block=eb,
            validation_site="swa.gather_cp_byte.slot_indices",
            negative_mode="skip_any",
            gather_lens=swa.cache_gather_lens,
        )
        swa = swa._replace(
            cache_slot_mapping=cache_slot_mapping,
            cache_compaction=cache_compaction,
        )
    return swa


def _patch_workspace_meta(attn, wm, meta, cp_ctx, tables):
    """Rebuild ``workspace_meta``'s block-table fields (bt casts + suffix map)."""
    from rtp_llm.models_py.modules.dsv4.fp8._kv_cache_utils import (
        require_pool_tokens_per_block,
    )
    from rtp_llm.models_py.modules.dsv4.fp8.attention import (
        _build_suffix_pool_slot_mapping,
        _suffix_gather_lens_max_host,
    )
    from rtp_llm.models_py.modules.dsv4.kv_cache_utils import CSA_KV, HCA_KV, SWA_KV

    if wm is None:
        return None
    # The SM120-paged maps + raw-q-merge gate are off in the recipe; if a
    # prebuilt meta ever carries them the patch does not reproduce them —
    # fail closed.
    if (
        getattr(wm, "swa_pool_slot_mapping", None) is not None
        or getattr(wm, "cmp_pool_slot_mapping", None) is not None
        or bool(getattr(wm, "use_cp_raw_q_merge", False))
    ):
        raise _PatchOutOfDomain("workspace meta carries non-recipe fields")
    ratio = int(attn.compress_ratio)
    cmp_at = CSA_KV if ratio == 4 else HCA_KV
    swa_bt = tables.get(SWA_KV)
    cmp_bt = tables.get(cmp_at)
    if swa_bt is None or cmp_bt is None:
        raise _PatchOutOfDomain("workspace patch requires SWA + compressed tables")
    B = int(meta.batch_size)
    device = meta.device
    swa_bt_int32 = swa_bt[:B].to(device=device, dtype=torch.int32).contiguous()
    cmp_bt_int32 = cmp_bt[:B].to(device=device, dtype=torch.int32).contiguous()
    swa_tpb = require_pool_tokens_per_block(attn._kv_cache, tag=SWA_KV)
    swa_cache_slot_mapping = _build_suffix_pool_slot_mapping(
        block_table=swa_bt_int32,
        seq_lens=wm.swa_cache_seq_lens,
        gather_lens=wm.swa_cache_gather_lens,
        entries_per_block=wm.swa_eb,
        tokens_per_block_for_block_table=swa_tpb,
        ring_entries=wm.swa_eb,
        max_gather_host=_suffix_gather_lens_max_host(
            meta.prefix_lengths,
            cp_ctx.prefix_lengths_full_host,
            B,
            attn.window_size,
        ),
    )
    swa_cache_compaction = attn._build_swa_cp_byte_compaction(
        swa_cache_slot_mapping,
        full_entries_per_block=wm.swa_eb,
        validation_site="swa.gather_cp_byte.slot_indices",
        negative_mode="skip_any",
        gather_lens=wm.swa_cache_gather_lens,
    )
    return wm._replace(
        swa_bt_int32=swa_bt_int32,
        cmp_bt_int32=cmp_bt_int32,
        swa_cache_slot_mapping=swa_cache_slot_mapping,
        swa_cache_compaction=swa_cache_compaction,
    )


def patch_prebuilt_metas(bundle: PrebuiltHead, kv_cache, block_tables_by_type) -> None:
    """Rebuild every block-table-dependent field of the bundle's metas.

    Called at consume time (main thread) with the real per-chunk block tables.
    Raises ``_PatchOutOfDomain`` for anything the patch does not reproduce
    byte-exactly; the caller then runs the full legacy build.
    """
    from rtp_llm.models_py.modules.dsv4.fp8.attention import bind_attn_cache
    from rtp_llm.models_py.modules.dsv4.kv_cache_utils import INDEXER_KV

    cp_ctx = bundle.cp_ctx
    for r, meta in bundle.meta_by_ratio.items():
        attn = bundle.rep_attns[r]
        with bind_attn_cache(attn, kv_cache, block_tables_by_type):
            new_swa = _patch_swa_meta(attn, meta, cp_ctx, block_tables_by_type)
            meta = meta._replace(swa_meta=new_swa)
            if int(attn.compress_ratio) == 4 and meta.csa_meta is not None:
                new_csa = _patch_csa_meta(attn, meta, cp_ctx, block_tables_by_type)
                meta = meta._replace(csa_meta=new_csa)
            elif int(attn.compress_ratio) == 128 and meta.hca_meta is not None:
                new_hca = _patch_hca_meta(attn, meta, cp_ctx, block_tables_by_type)
                meta = meta._replace(hca_meta=new_hca)
            bundle.meta_by_ratio[r] = meta


def _patch_csa_meta(attn, meta, cp_ctx, tables):
    from rtp_llm.models_py.modules.dsv4.kv_cache_utils import INDEXER_KV

    csa = meta.csa_meta
    old_cm = csa.compressor_meta
    old_im = csa.indexer_meta
    attn._set_compressor_pool_context()
    try:
        # Main CSA compressor meta: re-run ``prepare_metadata`` against the
        # prebuilt (block-table-independent) positions/b_idx — the returned
        # meta is byte-identical to the legacy build's by construction.
        new_cm = attn.compressor.prepare_metadata(
            old_cm.positions,
            old_cm.b_idx,
            has_prefix=old_cm.has_prefix,
            is_batched=old_cm.is_batched,
            seq_start_per_req=old_cm.seq_start_per_req,
            cu_seq_per_req=old_cm.cu_seq_per_req,
        )
        # Nested indexer compressor meta + the indexer block table.
        idx_bt = attn._block_tables_by_type.get(INDEXER_KV)
        idx_eb = attn._pool_entries_per_block(INDEXER_KV)
        B = int(meta.batch_size)
        device = meta.device
        if idx_bt is not None and idx_eb > 0:
            new_block_table_i32 = (
                idx_bt[:B].to(device=device, dtype=torch.int32).contiguous()
            )
        else:
            new_block_table_i32 = torch.empty((B, 0), dtype=torch.int32, device=device)
        old_nested = old_im.compressor_meta
        new_nested = old_nested
        if old_nested is not None:
            attn.indexer._propagate_pool_to_nested()
            try:
                new_nested = attn.indexer.compressor.prepare_metadata(
                    old_nested.positions,
                    old_nested.b_idx,
                    has_prefix=old_nested.has_prefix,
                    is_batched=old_nested.is_batched,
                    seq_start_per_req=old_nested.seq_start_per_req,
                    cu_seq_per_req=old_nested.cu_seq_per_req,
                )
            finally:
                attn.indexer._clear_nested_pool()
        new_im = old_im._replace(
            block_table_i32=new_block_table_i32, compressor_meta=new_nested
        )
    finally:
        attn._clear_compressor_pool_context()
    new_ws = _patch_workspace_meta(attn, csa.workspace_meta, meta, cp_ctx, tables)
    return csa._replace(
        compressor_meta=new_cm, indexer_meta=new_im, workspace_meta=new_ws
    )


def _patch_hca_meta(attn, meta, cp_ctx, tables):
    hca = meta.hca_meta
    old_cm = hca.compressor_meta
    attn._set_compressor_pool_context()
    try:
        new_cm = attn.compressor.prepare_metadata(
            old_cm.positions,
            old_cm.b_idx,
            has_prefix=old_cm.has_prefix,
            is_batched=old_cm.is_batched,
            seq_start_per_req=old_cm.seq_start_per_req,
            cu_seq_per_req=old_cm.cu_seq_per_req,
        )
    finally:
        attn._clear_compressor_pool_context()
    new_ws = _patch_workspace_meta(attn, hca.workspace_meta, meta, cp_ctx, tables)
    return hca._replace(compressor_meta=new_cm, workspace_meta=new_ws)
