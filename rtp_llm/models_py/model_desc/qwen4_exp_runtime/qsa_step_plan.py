"""Frozen per-step QSA prefill metadata, built once and reused by every layer.

This is the QSA counterpart of sglang's ``init_forward_metadata`` frozen
struct: a model step currently rebuilds the same layer-independent metadata
in every attention layer (length/prefix host mirrors, cu boundaries, logical
positions, visible/compressed lengths, both indexer RoPE tables and the
selection padding maps).  ``Qwen4ExpQSARuntimeContext`` is recreated per
layer inside the model's layer loop, so the plans live in this module-level
bounded cache instead of on the context instance.  The cache key is
``(id(main_inputs), token_count, mode)``: the engine rebuilds attention
input objects per batch, so object identity identifies one model step, and
every entry keeps its owning ``PyAttentionInputs`` alive (an ``id`` cannot
be recycled while its entry exists).  Layer-dependent state -- the per-layer
kv/state pool views, block tables, their verdicts and the write transaction
-- deliberately stays out of the plans.

Entries record ``(data_ptr, _version)`` fingerprints of every source tensor
they were derived from plus the length/prefix host values, and each hit
re-reads those host mirrors; an in-place edit or attribute rebind therefore
invalidates the entry and the next call takes the full legacy build.  Cross
cache-region consistency is re-checked on hits by comparing the same tensor
objects through their fingerprints (the build validated the values once with
the legacy ``torch.equal`` chain), replacing the per-layer value comparison
that hid host synchronizations.

``RTP_LLM_DEVICE_INPUT=1`` engines keep ``input_lengths``/``prefix_lengths``
on the device.  :func:`host_input_mirror` reads such a source back exactly
once per model step (keyed by the source tensor's identity and guarded by
its fingerprint) so the gate, the plan build, the hit validation, the
per-layer dispatch checks and the FMHA input planning all consume host
values without per-layer synchronizations.  Host sources pass through with
zero copies.

``RTP_LLM_QSA_STRICT_VALIDATION=1`` bypasses the cache entirely: every call
rebuilds through the legacy per-check host validation chain with its exact
diagnostics, so tests and CI keep per-call semantics.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import accumulate
from threading import local
from typing import Any

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_KV_TAG,
    INDEXER_STATE_TAG,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa import grid_prep_triton
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import (
    _normalized_rope_style_name,
    build_qsa_rope,
    build_qsa_rope_from_logical_positions,
    cached_inv_freq,
    is_qsa_rope_style,
    mrope_decode_tail_verdict_packed,
    mrope_equal_axis_verdict,
    validate_mrope_decode_tail,
)

from .qsa_cache_views import (
    _block_table,
    _side_tables_verdict,
    qsa_assert_valid,
    qsa_strict_validation_enabled,
)

MODE_PREFIX_REUSE = "prefix_reuse"
MODE_DRAFT_INCREMENTAL = "draft_incremental"
MODE_DRAFT_PREFIX_REUSE = "draft_prefix_reuse"
MODE_COLD_PREFILL = "cold_prefill"

# Main-inputs members every plan derives its metadata from.  Their
# fingerprints make in-place mutation between layers (or between calls that
# reuse one inputs object) fall back to the full rebuild.
_FINGERPRINT_SOURCES = (
    "input_lengths",
    "prefix_lengths",
    "sequence_lengths",
    "combo_position_ids",
    "cu_seqlens_device",
    "cu_kv_seqlens_device",
)

_METADATA_CONSISTENCY_SOURCES = (
    "input_lengths",
    "prefix_lengths",
    "sequence_lengths",
    "cu_seqlens_device",
    "cu_kv_seqlens_device",
    "combo_position_ids",
)

# Plans hold device tensors and the owning inputs object; a handful of
# recent steps is more than one model step ever needs.
_STEP_PLAN_CACHE_LIMIT = 8
_STEP_PLAN_CACHE: "OrderedDict[tuple, tuple[Any, Qwen4ExpPrefillStepPlan]]" = (
    OrderedDict()
)
_STEP_PLAN_STATS = {"hits": 0, "misses": 0, "invalidations": 0}

# A plan may survive multiple forwards in the bounded plan cache, whereas a
# block-table verdict may only be reused inside ONE eager model forward.
# The model entry establishes this explicit epoch; direct/component callers
# without it retain the original per-call checks. Thread-local state keeps
# concurrent model threads independent, and nested forwards restore the outer
# epoch when they finish.
_SIDE_VALIDATION_LOCAL = local()


@contextmanager
def qsa_validation_step():
    """Scope side-table validation reuse to one synchronous eager forward.

    Production block tables and required-column vectors are frozen for this
    forward after their producer's stream dependency has been established.
    Inference tensors have no version counter: their values MUST NOT be edited
    in place during this scope. A producer that changes them must end this
    scope and start a new one (or use the uncached path). Version-tracked
    normal tensor edits and rebinds are checked on every lookup; untracked
    native/data writes also require the frozen-metadata contract. The next
    forward always gets a new epoch, even when it reuses exactly the same
    inputs objects, tensor addresses and plan.

    Capture has no reusable host verdict: every graph records its own device
    checks. Query capture only once at entry, without initializing CUDA in CPU
    callers. Capturing starts outside the model-forward scope in production.
    """
    previous = getattr(_SIDE_VALIDATION_LOCAL, "entries", None)
    capturing = torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing()
    _SIDE_VALIDATION_LOCAL.entries = None if capturing else {}
    try:
        yield
    finally:
        _SIDE_VALIDATION_LOCAL.entries = previous


def _side_validation_tensor_key(tensor: torch.Tensor) -> tuple:
    # Metadata only: no tensor values, content hashes, copies or synchronization.
    # Include the owner as well as storage/layout; Tensor.data assignment and
    # differently-strided aliases must not inherit another view's verdict.
    return (
        id(tensor),
        tensor.data_ptr(),
        None if torch.is_inference(tensor) else tensor._version,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def _side_validation_stream_key(device: torch.device) -> tuple:
    if device.type == "cuda":
        return (device, torch.cuda.current_stream(device).cuda_stream)
    return (device, None)


def validate_side_tables_for_step(
    step: Any,
    kv_table: torch.Tensor,
    state_table: torch.Tensor,
    *,
    kv_pool_blocks: int,
    state_pool_blocks: int,
) -> None:
    """Enqueue the original verdict/assert once per equivalent eager-step use.

    A cache hit means the checks were ENQUEUED earlier on the same stream,
    never that the CUDA boolean was read on the host. This remains before
    every consumer's snapshot/write by stream ordering. Capacities are part
    of the key because pools themselves remain layer-local. Other modes,
    strict callers and callers without a model-forward epoch keep the old
    validation path.
    """
    tensors = (
        kv_table,
        step.required_kv_columns,
        state_table,
        step.required_state_columns_device,
    )
    entries = getattr(_SIDE_VALIDATION_LOCAL, "entries", None)
    reusable = (
        entries is not None
        and step.mode == MODE_PREFIX_REUSE
        and not bool(getattr(step.main_inputs, "is_cuda_graph", False))
        and not qsa_strict_validation_enabled()
    )
    key = None
    if reusable:
        key = (
            id(step),
            _side_validation_stream_key(kv_table.device),
            kv_pool_blocks,
            state_pool_blocks,
            tuple(_side_validation_tensor_key(tensor) for tensor in tensors),
        )
        if key in entries:
            return
    qsa_assert_valid(
        _side_tables_verdict(
            kv_table,
            step.required_kv_columns,
            state_table,
            step.required_state_columns_device,
            kv_pool_blocks=kv_pool_blocks,
            state_pool_blocks=state_pool_blocks,
        ),
        "QSA cache 'indexer_kv'/'indexer_state' block table does not cover "
        "or resolve the draft incremental tail",
    )
    if reusable:
        # Retain owners so ids/data_ptrs cannot be recycled inside this epoch.
        # Do not record if either verdict construction or assert dispatch throws.
        entries[key] = (
            step,
            tensors,
            # Retain the original storages too: a Tensor.data rebind can keep
            # its Python owner while releasing/recycling the old allocation.
            tuple(tensor.untyped_storage() for tensor in tensors),
        )


def step_plan_cache_stats() -> dict:
    """Cache counters for benches and diagnostics (hits/misses/invalidations)."""
    return dict(
        _STEP_PLAN_STATS,
        mirror_readbacks=_MIRROR_STATS["readbacks"],
        fact_computes=_PREFILL_FACTS_STATS["computes"],
    )


def clear_step_plan_cache() -> None:
    """Drop every cached plan (tests isolate cases that share inputs objects)."""
    _STEP_PLAN_CACHE.clear()
    _MIRROR_CACHE.clear()
    _PREFILL_FACTS_CACHE.clear()
    entries = getattr(_SIDE_VALIDATION_LOCAL, "entries", None)
    if entries is not None:
        entries.clear()


def _tensor_fingerprint(tensor: Any) -> tuple[int, int] | None:
    if not isinstance(tensor, torch.Tensor):
        return None
    if torch.is_inference(tensor):
        # Inference tensors do not track a version counter: reading
        # ``._version`` raises "Inference tensors do not track version
        # counter" during a real (inference_mode) forward. The storage
        # identity stands in for the version slot; the engine materialises
        # fresh inference tensors per step and the runtime never rewrites
        # the fingerprinted metadata tensors in place.
        return (int(tensor.data_ptr()), -1)
    return (int(tensor.data_ptr()), int(tensor._version))


# Per-step CPU mirrors for the [B] metadata tensors that
# ``RTP_LLM_DEVICE_INPUT=1`` engines keep on the device (input_lengths /
# prefix_lengths).  The P3 gate, the plan build and the per-layer hit
# validation all need those values; a device source is read back exactly once
# per step and every later read of the same (inputs object, attribute) pair
# reuses the mirror without synchronizing.
# One model step's working set is up to 12 source tensors (target + MTP
# draft forward, each with three cache-region inputs objects carrying
# prefix_lengths/input_lengths); a handful of recent steps makes the bound
# below comfortably exceed it, so the layers of one step never evict each
# other's mirrors.
_MIRROR_CACHE_LIMIT = 64
_MIRROR_CACHE: "OrderedDict[tuple, tuple[Any, torch.Tensor, tuple]]" = OrderedDict()
_MIRROR_STATS = {"readbacks": 0}


def host_input_mirror(main_inputs: Any, name: str) -> torch.Tensor | None:
    """CPU mirror of a small metadata tensor on the inputs object.

    Host sources pass through unchanged (the exact tensor, zero copies).  A
    device source is read back once per step: the mirror is keyed by the
    source tensor's identity and validated by its fingerprint, so every reader
    of the same tensor (the gate, the plan build, the per-layer dispatch
    checks and the three cache regions, which share one inputs tensor per
    attribute) reuses a single readback, and a fresh step that materialises
    new tensors refreshes it.  In-place device edits that keep the same storage
    are outside the fingerprint's reach -- the same bet the plan fingerprints
    already make (the engine materialises fresh inference tensors per step and
    the runtime never rewrites the fingerprinted metadata tensors in place).
    """
    tensor = getattr(main_inputs, name, None)
    if not isinstance(tensor, torch.Tensor):
        return None
    if tensor.device.type == "cpu":
        return tensor
    fingerprint = _tensor_fingerprint(tensor)
    key = id(tensor)
    entry = _MIRROR_CACHE.get(key)
    if entry is not None:
        owner, mirror, cached = entry
        if owner is tensor and cached == fingerprint:
            _MIRROR_CACHE.move_to_end(key)
            return mirror
        del _MIRROR_CACHE[key]
    mirror = tensor.detach().to("cpu").contiguous()
    _MIRROR_STATS["readbacks"] += 1
    # The entry holds the source tensor alive so its id stays unique while
    # the mirror exists.
    _MIRROR_CACHE[key] = (tensor, mirror, fingerprint)
    while len(_MIRROR_CACHE) > _MIRROR_CACHE_LIMIT:
        _MIRROR_CACHE.popitem(last=False)
    return mirror


def host_cu_seqlens_mirror(main_inputs: Any, batch_size: int) -> torch.Tensor | None:
    """The engine's pinned host mirror of ``cu_seqlens_device``, if usable.

    The host-input engine and the CUDA-graph runner keep ``cu_seqlens`` as
    the pinned CPU tensor whose contents they upload to
    ``cu_seqlens_device`` (the OpDefs.h naming convention); the device-input
    engine leaves it empty.  Only a fully-shaped int32 CPU mirror qualifies --
    anything else keeps the legacy device comparison, which also preserves
    the legacy behavior for a device tensor of the wrong length.
    """
    mirror = getattr(main_inputs, "cu_seqlens", None)
    if (
        not isinstance(mirror, torch.Tensor)
        or mirror.device.type != "cpu"
        or mirror.dtype != torch.int32
        or mirror.dim() != 1
        or int(mirror.numel()) != batch_size + 1
    ):
        return None
    return mirror


@dataclass(frozen=True)
class PrefillMirrorFacts:
    """Host facts over one step's input_lengths/prefix_lengths mirrors.

    Freezes the derived python values every layer of one step shares: the
    partition lists, the token count, the nonzero-prefix gate and the
    fail-fast sign checks.  The legacy paths recomputed these per layer with
    host tensor ops (``torch.any`` / ``tolist`` / ``sum().item()``); the
    facts compute them once per mirror pair.
    """

    lengths: list[int]
    prefixes: list[int]
    lengths_sum: int
    has_nonzero_prefix: bool
    lengths_positive: bool
    prefixes_nonneg: bool


# Per-step host facts over the pinned length/prefix mirror pair.  Keyed by
# both mirrors' identities and validated by their fingerprints, so an in-place
# edit (version bump) or a fresh step's tensors recompute the facts; the
# layers of one step only read the frozen values.
_PREFILL_FACTS_LIMIT = 64
_PREFILL_FACTS_CACHE: (
    "OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor, PrefillMirrorFacts, tuple]]"
) = OrderedDict()
_PREFILL_FACTS_STATS = {"computes": 0}


def prefill_mirror_facts(main_inputs: Any) -> PrefillMirrorFacts | None:
    """Frozen per-step host facts over the length/prefix mirrors.

    The mirror cache already collapses any device readback to once per step;
    this additionally freezes the derived values.  Returns ``None`` when the
    inputs object carries no usable mirrors (callers keep their legacy
    per-call computation in that case).
    """
    lengths_mirror = host_input_mirror(main_inputs, "input_lengths")
    prefixes_mirror = host_input_mirror(main_inputs, "prefix_lengths")
    if not isinstance(lengths_mirror, torch.Tensor) or not isinstance(
        prefixes_mirror, torch.Tensor
    ):
        return None
    fingerprint = (
        _tensor_fingerprint(lengths_mirror),
        _tensor_fingerprint(prefixes_mirror),
    )
    key = (id(lengths_mirror), id(prefixes_mirror))
    entry = _PREFILL_FACTS_CACHE.get(key)
    if entry is not None:
        cached_lengths, cached_prefixes, facts, cached_fp = entry
        if (
            cached_lengths is lengths_mirror
            and cached_prefixes is prefixes_mirror
            and cached_fp == fingerprint
        ):
            _PREFILL_FACTS_CACHE.move_to_end(key)
            return facts
        del _PREFILL_FACTS_CACHE[key]
    facts = PrefillMirrorFacts(
        lengths=[int(value) for value in lengths_mirror.tolist()],
        prefixes=[int(value) for value in prefixes_mirror.tolist()],
        lengths_sum=int(lengths_mirror.sum().item()),
        has_nonzero_prefix=bool(prefixes_mirror.numel())
        and bool(torch.any(prefixes_mirror != 0).item()),
        lengths_positive=not bool(torch.any(lengths_mirror <= 0).item()),
        prefixes_nonneg=not bool(torch.any(prefixes_mirror < 0).item()),
    )
    _PREFILL_FACTS_STATS["computes"] += 1
    _PREFILL_FACTS_CACHE[key] = (lengths_mirror, prefixes_mirror, facts, fingerprint)
    while len(_PREFILL_FACTS_CACHE) > _PREFILL_FACTS_LIMIT:
        _PREFILL_FACTS_CACHE.popitem(last=False)
    return facts


def has_nonzero_prefill_prefix(main_inputs: Any) -> bool:
    """Whether any request in this step carries a non-zero reused prefix.

    The per-layer sparse-GQA dispatch gate and the attention layer's two
    prefix gates (the prefix-reuse projection gate and the prefill dispatch)
    read this frozen verdict instead of recomputing the
    ``torch.any(prefixes != 0).item()`` host chain per call site per layer.
    Inputs objects without a usable prefix mirror (component-test adapters)
    decline: no prefix, no reuse.
    """
    facts = prefill_mirror_facts(main_inputs)
    if facts is None:
        prefixes = host_input_mirror(main_inputs, "prefix_lengths")
        if not isinstance(prefixes, torch.Tensor):
            return False
        return bool(prefixes.numel()) and bool(torch.any(prefixes != 0).item())
    return facts.has_nonzero_prefix


def rope_signature(rope_config: Any) -> tuple:
    """Host values that fully determine the QSA RoPE tables.

    Layers construct their own ``rope_config`` pybind objects with identical
    values, so the cache compares this signature instead of object identity.
    """
    return (
        _normalized_rope_style_name(rope_config.style),
        int(rope_config.index_factor),
        int(rope_config.dim),
        float(rope_config.base),
        float(getattr(rope_config, "scale", 1.0) or 1.0),
        int(getattr(rope_config, "mrope_dim1", 0) or 0),
        int(getattr(rope_config, "mrope_dim2", 0) or 0),
        int(getattr(rope_config, "mrope_dim3", 0) or 0),
        bool(getattr(rope_config, "mrope_interleaved", False)),
        bool(getattr(rope_config, "indexer_is_neox_style", True)),
    )


@dataclass(frozen=True)
class Qwen4ExpPrefillStepPlan:
    """Layer-independent QSA prefill metadata for one model step.

    Host fields are free to rebuild (pinned mirrors / python ints); device
    fields are the per-step constants whose per-layer rebuild P2 removes.
    Mode-specific fields are ``None`` when the mode does not produce them.
    """

    # Identity / applicability.
    mode: str
    main_inputs: Any
    is_mtp_draft: bool
    token_count: int
    device: torch.device
    rope_dtype: torch.dtype
    rope_signature: tuple
    source_fingerprints: dict[str, tuple[int, int] | None]
    indexer_kv_inputs: Any = field(default=None, repr=False, compare=False)
    indexer_state_inputs: Any = field(default=None, repr=False, compare=False)
    # Per-region fingerprints of the metadata tensors the legacy build
    # validated with torch.equal; hits compare fingerprints instead.
    region_fingerprints: tuple[dict[str, tuple[int, int] | None], ...] = ()
    # Keeps the fingerprinted storages alive so data_ptr stays meaningful.
    source_tensors: tuple = field(default=(), repr=False, compare=False)

    # Host group.
    lengths: list[int] = field(default_factory=list)
    prefix_values: list[int] = field(default_factory=list)
    cu_host: list[int] = field(default_factory=list)
    tokens: int = 0
    batch_size: int = 0
    max_len: int = 0
    max_ctx_len: int = 0
    head_dim: int = 0
    head_num: int = 0
    ratio: int = 0
    token_budget: int = 0
    kv_tokens_per_block: int = 0
    kv_entries_per_block: int = 0
    state_tokens_per_block: int = 0
    # Host int32 [B]; the per-layer state-table verdict moves it to device.
    required_state_columns: torch.Tensor | None = None

    # Device group.
    prefixes_i64: torch.Tensor | None = None
    prefixes_i32: torch.Tensor | None = None
    input_lengths_i32: torch.Tensor | None = None
    cu_seqlens: torch.Tensor | None = None
    logical_positions: torch.Tensor | None = None
    # int32 [B, max_len] padded visible lengths: the same grid serves
    # visible_lengths_i32 (flattened) and padded_kv_lens; no consumer reads
    # the int64 copy the legacy chain built and cast away.
    visible_lengths: torch.Tensor | None = None
    visible_lengths_i32: torch.Tensor | None = None
    compressed_lengths: torch.Tensor | None = None
    required_kv_columns: torch.Tensor | None = None
    required_state_columns_device: torch.Tensor | None = None
    # Indexer RoPE for the current rows and for the compression-group starts.
    current_cos: torch.Tensor | None = None
    current_sin: torch.Tensor | None = None
    rope_cos: torch.Tensor | None = None
    rope_sin: torch.Tensor | None = None
    # Selection padding constants (pure functions of the length partition).
    selection_row_map: torch.Tensor | None = None
    selection_valid_rows: torch.Tensor | None = None
    selection_valid: torch.Tensor | None = None
    score_weights: torch.Tensor | None = None
    padded_score_weights: torch.Tensor | None = None
    kv_lens_rows: tuple[torch.Tensor, ...] | None = None
    padded_kv_lens: torch.Tensor | None = None
    # Cold prefill keeps the padded [B, max_len, rotary_dim] RoPE table the
    # side-cache writers and per-request scoring consume.
    rope_table_cos: torch.Tensor | None = None
    rope_table_sin: torch.Tensor | None = None
    # Whether the build ran the cold-path cu partition preflight.
    cu_checked: bool = False
    # Step-geometry gate for the fused prefix-reuse preparation writer: the
    # writer's step-invariant preconditions hold, so the production caller
    # passes ``trusted_geometry`` and the writer re-checks only the
    # per-layer projection tensors.
    prefix_reuse_fused_supported: bool = False
    # Step-frozen writer geometry for the vectorized indexer writer
    # (:class:`...indexer_compressor.IndexerWriterGeometry`): the layer-
    # independent token/page resolution every layer's side-cache write needs.
    # Built with the plan (strict validation skips it and keeps the legacy
    # per-call chain); each layer's writer re-validates it host-side before
    # use, so a stale bundle can never feed a wrong write.
    writer_geometry: Any = field(default=None, repr=False, compare=False)


def _cache_regions(context: Any) -> tuple:
    return (
        context.main_inputs,
        context.indexer_kv_inputs,
        context.indexer_state_inputs,
    )


def _collect_region_fingerprints(
    regions: tuple,
) -> tuple[dict[str, tuple[int, int] | None], ...]:
    """Fingerprint the metadata tensors the legacy build value-compared.

    The legacy ``_validate_mode_and_metadata`` check compares the three cache
    regions' metadata tensors with ``torch.equal`` when they are distinct
    objects (a hidden host synchronization per layer).  The build validated
    those values once; a hit only needs the same tensor objects, unchanged.
    """
    return tuple(
        {
            name: _tensor_fingerprint(getattr(inputs, name, None))
            for name in _METADATA_CONSISTENCY_SOURCES
        }
        for inputs in regions
    )


def _regions_agree_metadata_only(context: Any, plan: Qwen4ExpPrefillStepPlan) -> bool:
    """Cross-region consistency without reading any tensor values.

    The plan build validated the region values once with the legacy
    ``torch.equal`` chain.  Production resolves one inputs object per cache
    region for the whole forward, so identical region objects with
    unchanged tensor attributes keep the build's verdict; anything swapped
    or rebound falls back to the full rebuild.  This replaces the legacy
    per-layer value comparison (up to 12 hidden host synchronizations per
    layer) with host metadata checks.
    """
    if (
        plan.indexer_kv_inputs is not context.indexer_kv_inputs
        or plan.indexer_state_inputs is not context.indexer_state_inputs
    ):
        return False
    main_inputs = context.main_inputs
    for inputs in (context.indexer_kv_inputs, context.indexer_state_inputs):
        for name in ("is_prefill", "is_target_verify", "is_cuda_graph", "is_s_padded"):
            if bool(getattr(inputs, name, False)) != bool(
                getattr(main_inputs, name, False)
            ):
                return False
        if getattr(inputs, "context_parallel_info", None) is not None:
            return False
    if (
        _collect_region_fingerprints(_cache_regions(context))
        != plan.region_fingerprints
    ):
        return False
    from .pd_publication import validate_pd_cache_inputs

    try:
        validate_pd_cache_inputs(_cache_regions(context))
    except RuntimeError:
        return False
    return True


def _step_plan_applies(
    plan: Qwen4ExpPrefillStepPlan,
    context: Any,
    *,
    indexer: Any,
    token_count: int,
    device: torch.device,
    rope_config: Any,
    rope_dtype: torch.dtype | None = None,
) -> bool:
    """Whether a cached plan still describes this layer's invocation.

    Every check is a host attribute read or pinned-mirror value read; a hit
    never synchronizes with the device.
    """
    if plan.main_inputs is not context.main_inputs:
        return False
    if plan.token_count != token_count or plan.device != device:
        return False
    if plan.is_mtp_draft != bool(context.is_mtp_draft):
        return False
    if plan.rope_signature != rope_signature(rope_config):
        return False
    if rope_dtype is not None and plan.rope_dtype != rope_dtype:
        return False
    for name, fingerprint in plan.source_fingerprints.items():
        if _tensor_fingerprint(getattr(context.main_inputs, name, None)) != fingerprint:
            return False
    # Per-step CPU mirrors: the build read the mirrors once and froze their
    # values into the plan.  The source fingerprints above already cover the
    # guarantee the per-layer value comparison used to re-establish (an
    # unchanged tensor keeps the build's verdict; anything swapped, rebound
    # or edited in place falls back to the full rebuild), so hits no longer
    # re-read mirror values.
    if (plan.kv_tokens_per_block, plan.state_tokens_per_block) != (
        int(context.indexer_kv_cache.seq_size_per_block),
        int(context.indexer_state_cache.seq_size_per_block),
    ):
        return False
    if plan.mode == MODE_COLD_PREFILL:
        # The zero-prefix build only consumes the indexer geometry the
        # legacy cold path read; component-test indexers may not expose more.
        if (plan.head_dim, plan.ratio) != (
            int(indexer.head_dim),
            int(indexer.compress_ratio),
        ):
            return False
    elif (
        plan.head_dim,
        plan.head_num,
        plan.ratio,
        plan.token_budget,
    ) != (
        int(indexer.head_dim),
        int(indexer.n_heads),
        int(indexer.compress_ratio),
        int(indexer.token_budget),
    ):
        return False
    return _regions_agree_metadata_only(context, plan)


def _collect_fingerprints(main_inputs: Any) -> dict[str, tuple[int, int] | None]:
    return {
        name: _tensor_fingerprint(getattr(main_inputs, name, None))
        for name in _FINGERPRINT_SOURCES
    }


def _collect_source_tensors(main_inputs: Any) -> tuple:
    return tuple(
        tensor
        for tensor in (
            getattr(main_inputs, name, None) for name in _FINGERPRINT_SOURCES
        )
        if isinstance(tensor, torch.Tensor)
    )


def _cache_plan(plan: Qwen4ExpPrefillStepPlan) -> None:
    key = (id(plan.main_inputs), plan.token_count, plan.mode)
    _STEP_PLAN_CACHE[key] = (plan.main_inputs, plan)
    _STEP_PLAN_CACHE.move_to_end(key)
    while len(_STEP_PLAN_CACHE) > _STEP_PLAN_CACHE_LIMIT:
        _STEP_PLAN_CACHE.popitem(last=False)


def _pinned_to_device(host: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Async H2D from pinned memory; the pinned source stays plan-owned."""
    if device.type == "cuda" and torch.cuda.is_available():
        return host.pin_memory().to(device=device, non_blocking=True)
    # CPU fixtures (component tests) have no pinned allocator; a plain copy
    # produces the same values.
    return host.to(device)


def _build_selection_padding(
    lengths: list[int], *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-step constants for ragged selection padding (legacy row_map loop).

    ``selection_row_map`` repeats the last valid row of short requests,
    ``selection_valid_rows`` lists the packed positions of real rows and
    ``selection_valid`` is the [B, max_len] validity mask.
    """
    batch_size = len(lengths)
    max_rows = max(lengths)
    offsets = []
    valid_rows = []
    offset = 0
    for request_idx, length in enumerate(lengths):
        offsets.extend(offset + min(row, length - 1) for row in range(max_rows))
        valid_rows.extend(request_idx * max_rows + row for row in range(length))
        offset += length
    row_map_host = torch.tensor(offsets, dtype=torch.int64)
    valid_rows_host = torch.tensor(valid_rows, dtype=torch.int64)
    valid_host = torch.zeros(batch_size, max_rows, dtype=torch.bool)
    for request_idx, length in enumerate(lengths):
        valid_host[request_idx, :length] = True
    return (
        _pinned_to_device(row_map_host, device),
        _pinned_to_device(valid_rows_host, device),
        _pinned_to_device(valid_host, device),
    )


@dataclass(frozen=True)
class _StepGridPrep:
    """Step-plan grid tensors produced by one batched prep pass.

    The [B]/[B+1]-grade outputs are host arithmetic on the length/prefix
    mirrors uploaded as one pinned staging batch per dtype; the
    [B, max_len]/[T]-grade grid tensors and both score-weight tables come
    from a single Triton launch (or the byte-exact eager aten fallback when
    the fused prep declines the geometry).
    """

    cu_host: list[int]
    prefixes_i64: torch.Tensor
    prefixes_i32: torch.Tensor
    input_lengths_i32: torch.Tensor
    cu_seqlens: torch.Tensor
    cu_kv_seqlens: torch.Tensor
    logical_positions: torch.Tensor
    block_starts: torch.Tensor
    mrope_sources: torch.Tensor
    visible_lengths: torch.Tensor
    visible_lengths_i32: torch.Tensor
    compressed_lengths: torch.Tensor
    required_kv_columns: torch.Tensor
    required_state_columns: torch.Tensor
    required_state_columns_device: torch.Tensor
    selection_valid: torch.Tensor
    selection_row_map: torch.Tensor
    selection_valid_rows: torch.Tensor
    score_weights: torch.Tensor
    padded_score_weights: torch.Tensor


def _int32_staging(values: list[int]) -> torch.Tensor:
    """Stage host ints as int32 without raising on overflow.

    Out-of-range values only occur on geometries the caller rejects right
    after the prep (its int32 headroom checks); the legacy chain staged the
    same tensors through silent int64->int32 casts, so those values are
    never consumed.
    """
    limit = torch.iinfo(torch.int32).max
    floor = torch.iinfo(torch.int32).min
    return torch.tensor(
        [
            value if floor <= value <= limit else (value & 0xFFFFFFFF) - (1 << 32)
            for value in values
        ],
        dtype=torch.int32,
    )


def _eager_step_grid(
    prefixes_i64: torch.Tensor,
    lengths: list[int],
    *,
    device: torch.device,
    ratio: int,
    max_len: int,
    token_count: int,
    head_num: int,
    head_dim: int,
):
    """The legacy aten chain as the byte-exact fallback for fused declines.

    The int32 padded grid replaces the legacy int64 ``visible_lengths``:
    every downstream consumer (``visible_lengths_i32``, ``padded_kv_lens``,
    the per-request FMHA rows) reads the int32 values, which the legacy
    chain itself produced by casting the int64 tensor right away.
    """
    row_offsets = torch.arange(max_len, dtype=torch.int64, device=device)
    lengths_i64 = torch.tensor(lengths, dtype=torch.int64, device=device)
    visible_lengths = prefixes_i64[:, None] + row_offsets[None, :] + 1
    valid_rows = row_offsets[None, :] < lengths_i64[:, None]
    visible_lengths = torch.where(valid_rows, visible_lengths, 0)
    grid = visible_lengths.to(torch.int32)
    compressed_lengths = (visible_lengths // ratio).to(torch.int32)
    selection_row_map, selection_valid_rows, selection_valid = _build_selection_padding(
        lengths, device=device
    )
    score_scale = 1.0 / math.sqrt(head_dim)
    score_weights = torch.full(
        (token_count, head_num), score_scale, dtype=torch.float32, device=device
    )
    # The legacy padded table was an index_select of the constant score
    # weights: every row holds the same 1/sqrt(head_dim) fill.
    padded_score_weights = torch.full(
        (len(lengths) * max_len, head_num),
        score_scale,
        dtype=torch.float32,
        device=device,
    )
    return (
        grid,
        compressed_lengths,
        selection_valid,
        selection_row_map,
        selection_valid_rows,
        score_weights,
        padded_score_weights,
    )


def _build_step_grid_prep(
    *,
    lengths: list[int],
    prefix_values: list[int],
    device: torch.device,
    ratio: int,
    head_num: int,
    head_dim: int,
    kv_entries_per_block: int,
    state_tokens_per_block: int,
) -> _StepGridPrep:
    """Compute the step-plan grid tensors in one batched prep.

    Everything the legacy chain built as [B]/[B+1]/[B, max_len]/[T]-grade
    aten launches (arange/add/where/floor_divide/remainder/fill plus the
    selection-padding host loops) is produced here as host arithmetic on
    the length/prefix mirrors, two pinned staging uploads and one Triton
    launch.  ``visible_lengths`` keeps only its int32 padded grid: no
    consumer ever reads the int64 copy the legacy chain built and cast
    away.
    """
    batch_size = len(lengths)
    max_len = max(lengths)
    token_count = sum(lengths)
    ends = [prefix + length for prefix, length in zip(prefix_values, lengths)]
    cu_host = [0] + list(accumulate(lengths))
    cu_kv_host = [0] + list(accumulate(ends))
    # Degenerate page sizes keep the legacy error path: the caller raises
    # before any of these tensors is consumed, so zero placeholders keep
    # the staging shapes intact without dividing by zero here.
    required_kv_values = (
        [
            (end // ratio + kv_entries_per_block - 1) // kv_entries_per_block
            for end in ends
        ]
        if kv_entries_per_block > 0
        else [0] * batch_size
    )
    required_state_values = (
        [
            (prefix + length + state_tokens_per_block - 1) // state_tokens_per_block
            for prefix, length in zip(prefix_values, lengths)
        ]
        if state_tokens_per_block > 0
        else [0] * batch_size
    )
    positions: list[int] = []
    block_start_values: list[int] = []
    source_values: list[int] = []
    token_index = 0
    for prefix, length in zip(prefix_values, lengths):
        for row in range(length):
            position = prefix + row
            positions.append(position)
            remainder = position % ratio
            block_start_values.append(position - remainder)
            source_values.append(token_index - remainder)
            token_index += 1

    # One pinned int32 batch: the [B]/[B+1]-grade outputs only.
    host_i32 = torch.cat(
        (
            _int32_staging(prefix_values),
            _int32_staging(lengths),
            _int32_staging(cu_host),
            _int32_staging(cu_kv_host),
            _int32_staging(required_kv_values),
            _int32_staging(required_state_values),
        )
    )
    device_i32 = _pinned_to_device(host_i32, device)
    prefixes_i32 = device_i32[:batch_size]
    input_lengths_i32 = device_i32[batch_size : 2 * batch_size]
    cursor = 2 * batch_size
    cu_seqlens = device_i32[cursor : cursor + batch_size + 1]
    cursor += batch_size + 1
    cu_kv_seqlens = device_i32[cursor : cursor + batch_size + 1]
    cursor += batch_size + 1
    required_kv_columns = device_i32[cursor : cursor + batch_size]
    cursor += batch_size
    required_state_columns_device = device_i32[cursor : cursor + batch_size]
    # The host mirror of the state-columns segment ([5B + 2, 6B + 2) of the
    # [prefixes, lengths, cu, cu_kv, required_kv, required_state] batch).
    required_state_columns = host_i32[5 * batch_size + 2 : 6 * batch_size + 2]

    # One pinned int64 batch: the prefixes plus the [T]-grade position
    # tables the rope builders consume (block starts and the MRoPE block
    # sources the legacy chain derived with remainder/sub/arange launches).
    host_i64 = torch.cat(
        (
            torch.tensor(prefix_values, dtype=torch.int64),
            torch.tensor(positions, dtype=torch.int64),
            torch.tensor(block_start_values, dtype=torch.int64),
            torch.tensor(source_values, dtype=torch.int64),
        )
    )
    device_i64 = _pinned_to_device(host_i64, device)
    prefixes_i64 = device_i64[:batch_size]
    cursor64 = batch_size
    logical_positions = device_i64[cursor64 : cursor64 + token_count]
    cursor64 += token_count
    block_starts = device_i64[cursor64 : cursor64 + token_count]
    cursor64 += token_count
    mrope_sources = device_i64[cursor64 : cursor64 + token_count]

    # The [B, max_len]-grade grid tensors and both score-weight tables in
    # one launch; the eager chain keeps identical values elsewhere.
    fused = grid_prep_triton.build_incremental_grid(
        prefixes_i32,
        input_lengths_i32,
        cu_seqlens,
        max_len=max_len,
        token_count=token_count,
        ratio=ratio,
        head_num=head_num,
        score_scale=1.0 / math.sqrt(head_dim),
    )
    if fused is None:
        fused = _eager_step_grid(
            prefixes_i64,
            lengths,
            device=device,
            ratio=ratio,
            max_len=max_len,
            token_count=token_count,
            head_num=head_num,
            head_dim=head_dim,
        )
    (
        visible_lengths,
        compressed_lengths,
        selection_valid,
        selection_row_map,
        selection_valid_rows,
        score_weights,
        padded_score_weights,
    ) = fused
    return _StepGridPrep(
        cu_host=cu_host,
        prefixes_i64=prefixes_i64,
        prefixes_i32=prefixes_i32,
        input_lengths_i32=input_lengths_i32,
        cu_seqlens=cu_seqlens,
        cu_kv_seqlens=cu_kv_seqlens,
        logical_positions=logical_positions,
        block_starts=block_starts,
        mrope_sources=mrope_sources,
        visible_lengths=visible_lengths,
        visible_lengths_i32=visible_lengths.reshape(-1),
        compressed_lengths=compressed_lengths,
        required_kv_columns=required_kv_columns,
        required_state_columns=required_state_columns,
        required_state_columns_device=required_state_columns_device,
        selection_valid=selection_valid,
        selection_row_map=selection_row_map,
        selection_valid_rows=selection_valid_rows,
        score_weights=score_weights,
        padded_score_weights=padded_score_weights,
    )


def prefix_reuse_fused_geometry_supported(
    *,
    batch_size: int,
    head_dim: int,
    head_num: int,
    ratio: int,
    lengths: list[int],
    prefixes: list[int],
    kv_tokens_per_block: int,
    state_tokens_per_block: int,
) -> bool:
    """Whether the fused prefix-reuse preparation writer's step geometry holds.

    Every precondition of :func:`write_prefix_reuse_prefill` that depends only
    on step metadata (page/dims geometry, bounds, page-aligned prefixes) is
    decided once per step here; the writer then re-checks only the per-layer
    projection tensors when the caller passes ``trusted_geometry``.
    """
    return bool(
        batch_size > 0
        and head_dim == 128
        and 0 < head_num <= 65535
        and all(isinstance(n, int) and n > 0 for n in lengths)
        and max(lengths, default=0) <= 65535
        and kv_tokens_per_block == 128
        and state_tokens_per_block == 128
        and ratio == 4
        and all(value % 128 == 0 for value in prefixes)
    )


def _build_step_writer_geometry(
    context: Any,
    *,
    cu_seqlens: torch.Tensor,
    start_positions: torch.Tensor,
    token_count: int,
    ratio: int,
    kv_tokens_per_block: int,
    state_tokens_per_block: int,
    host_lengths: list[int],
    host_prefixes: list[int],
    trusted_tables: bool = False,
) -> Any:
    """Freeze the vectorized indexer writer's geometry for this step plan.

    The layer-independent block tables come from the step's cache-region inputs
    and the pool shapes from the caches; every layer's writer then re-uses the
    bundle instead of re-deriving token/page resolution per call.  Strict
    validation skips the bundle so each call keeps the legacy per-call
    diagnostics.  The runtime's state ring is ``2 * ratio`` entries wide; a
    mismatching pool makes the writer's own applicability check fall back.
    ``trusted_tables`` (incremental plans, whose build proved the cu partition
    and ran the required-blocks verdicts over every column this geometry can
    read) drops the redundant resolve asserts.

    The geometry build itself decides which fast write form the bundle can
    serve: short windows prove the destination slots unique (full-shape
    write) and longer monotone windows take the ring-wrap batch write; a
    window neither form can serve keeps the legacy per-call writer (the
    bundle is not paid for nothing).
    """
    if qsa_strict_validation_enabled():
        return None
    from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_compressor import (
        build_indexer_writer_geometry,
    )

    kv_table = _block_table(
        context.indexer_kv_inputs,
        "kv_cache_kernel_block_id_device",
        tag=INDEXER_KV_TAG,
        batch_size=len(host_lengths),
        device=cu_seqlens.device,
    )
    state_table = _block_table(
        context.indexer_state_inputs,
        "kv_cache_block_id_device",
        tag=INDEXER_STATE_TAG,
        batch_size=len(host_lengths),
        device=cu_seqlens.device,
    )
    geometry = build_indexer_writer_geometry(
        cu_seqlens,
        start_positions,
        kv_table,
        state_table,
        token_count=token_count,
        ratio=ratio,
        kv_tokens_per_block=kv_tokens_per_block,
        state_tokens_per_block=state_tokens_per_block,
        state_ring_entries=2 * ratio,
        kv_pool_blocks=int(context.indexer_kv_cache.kv_cache_base.shape[0]),
        state_pool_blocks=int(context.indexer_state_cache.kv_cache_base.shape[0]),
        invalid_block_policy="raise",
        host_lengths=host_lengths,
        host_prefixes=host_prefixes,
        trusted_tables=trusted_tables,
    )
    if not (geometry.state_slots_unique or geometry.state_ring_wrap):
        # Neither fast write form applies to this window: keep the legacy
        # per-call writer instead of freezing an unused bundle.
        return None
    return geometry


def build_incremental_step_plan(
    context: Any,
    *,
    indexer: Any,
    token_count: int,
    device: torch.device,
    rope_config: Any,
    draft: bool,
    draft_prefix_reuse: bool,
) -> Qwen4ExpPrefillStepPlan:
    """Validate one incremental prefill step and freeze its metadata.

    Ports the legacy ``_draft_incremental_prefill_geometry`` validation body
    unchanged (identical checks, order and messages); the additions are the
    per-step selection constants the selection/FMHA consumers previously
    rebuilt per layer.  Layer-dependent state (pool views, block tables and
    their verdicts) stays with the per-layer geometry caller.
    """
    if draft_prefix_reuse:
        if draft or not context.is_mtp_draft:
            raise RuntimeError(
                "qwen4_exp QSA draft prefix-reuse requires an explicit MTP "
                "draft context"
            )
        label = "MTP draft prefix-reuse prefill"
    else:
        label = "draft incremental prefill" if draft else "prefix-reuse prefill"
    lengths = context._validate_mode_and_metadata(
        expect_prefill=True,
        allow_draft_incremental=draft,
        allow_prefix_reuse=not draft,
    )
    if context._target_verify_enabled():
        raise RuntimeError(
            "qwen4_exp QSA draft incremental prefill cannot be target verification"
        )
    if draft and not is_qsa_rope_style(rope_config, "Base"):
        # The MTP draft runs Base RoPE; the target keeps its own style
        # (MRoPE for the released checkpoint), which build_qsa_rope
        # dispatches on for both the writer and the scoring rotation.
        raise RuntimeError(
            "qwen4_exp QSA MTP draft incremental prefill requires Base RoPE"
        )

    main_inputs = context.main_inputs
    batch_size = len(lengths)
    head_dim = int(indexer.head_dim)
    head_num = int(indexer.n_heads)
    ratio = int(indexer.compress_ratio)
    token_budget = int(indexer.token_budget)
    if min(head_dim, head_num, ratio) <= 0:
        raise RuntimeError(
            "qwen4_exp QSA draft incremental prefill has invalid indexer geometry"
        )
    if token_budget <= 0 or token_budget % ratio:
        raise RuntimeError(
            "qwen4_exp QSA draft incremental prefill token budget must be "
            "positive and divisible by the compression ratio"
        )
    if draft and max(lengths) > ratio:
        raise RuntimeError(
            "qwen4_exp QSA draft incremental prefill requires "
            f"max(input_lengths)={max(lengths)} <= compress_ratio={ratio}"
        )
    if sum(lengths) != token_count:
        raise RuntimeError(
            f"qwen4_exp QSA input_lengths={lengths} do not partition "
            f"the {token_count} packed draft tokens"
        )

    prefix_source = main_inputs.prefix_lengths
    if (
        prefix_source.dim() != 1
        or int(prefix_source.numel()) != batch_size
        or prefix_source.dtype != torch.int32
    ):
        raise RuntimeError(
            "qwen4_exp QSA draft incremental prefill requires int32 "
            "prefix_lengths [B]"
        )
    # Request-data checks stay on the CPU mirror (free fail-fast) and the
    # host values replace every later device readback of the prefixes.  A
    # device source (RTP_LLM_DEVICE_INPUT) is read back once per step; the
    # device copies below cast the source tensor without a host round trip.
    prefixes_host = host_input_mirror(main_inputs, "prefix_lengths")
    prefix_values = prefixes_host.tolist()
    invalid_prefix = any(value <= 0 if draft else value < 0 for value in prefix_values)
    if invalid_prefix:
        requirement = "nonzero" if draft else "non-negative"
        raise RuntimeError(
            f"qwen4_exp QSA {label} requires every prefix length to be "
            f"{requirement}"
        )
    # One batched prep produces every grid/selection tensor the plan
    # freezes: host arithmetic on the mirrors, one pinned staging upload
    # per dtype and (on supported CUDA geometry) a single Triton launch,
    # replacing the per-step arange/add/where/floor_divide/remainder/fill
    # chain.  The page sizes only shape the host formulas here; their
    # validation below keeps the legacy errors.
    kv_tokens_per_block = int(context.indexer_kv_cache.seq_size_per_block)
    state_tokens_per_block = int(context.indexer_state_cache.seq_size_per_block)
    prep = _build_step_grid_prep(
        lengths=lengths,
        prefix_values=prefix_values,
        device=device,
        ratio=ratio,
        head_num=head_num,
        head_dim=head_dim,
        kv_entries_per_block=kv_tokens_per_block // ratio,
        state_tokens_per_block=state_tokens_per_block,
    )
    sequence_lengths = main_inputs.sequence_lengths
    if sequence_lengths.numel() != 0:
        raise RuntimeError(
            "qwen4_exp QSA draft incremental prefill requires empty " "sequence_lengths"
        )

    cu_seqlens = main_inputs.cu_seqlens_device
    # The expected boundaries are the batched prep's host arithmetic (python
    # accumulate) riding its single pinned int32 batch; the wrapped host copy
    # feeds the host-mirror validation below without another staging upload.
    expected_cu = prep.cu_seqlens
    expected_cu_host = _int32_staging(prep.cu_host)
    if (
        cu_seqlens.dim() != 1
        or cu_seqlens.dtype != torch.int32
        or cu_seqlens.device != device
    ):
        raise RuntimeError(
            "qwen4_exp QSA draft incremental prefill cu_seqlens do not "
            "match input_lengths"
        )
    # The engine uploads cu_seqlens_device from its pinned host mirror, so a
    # usable mirror answers the same invariant host-side: zero device work,
    # and the async verdict raises fail-fast instead of trapping.  A
    # device-only source (RTP_LLM_DEVICE_INPUT engines, component fixtures)
    # keeps the legacy device comparison against the pinned staging upload.
    cu_seqlens_host = host_cu_seqlens_mirror(main_inputs, batch_size)
    if cu_seqlens_host is not None and int(cu_seqlens.numel()) == batch_size + 1:
        if qsa_strict_validation_enabled():
            if not bool(torch.equal(cu_seqlens_host, expected_cu_host)):
                raise RuntimeError(
                    "qwen4_exp QSA draft incremental prefill cu_seqlens do not "
                    "match input_lengths"
                )
        else:
            qsa_assert_valid(
                (cu_seqlens_host == expected_cu_host).all(),
                "qwen4_exp QSA draft incremental prefill cu_seqlens do not "
                "match input_lengths",
            )
    elif qsa_strict_validation_enabled():
        if not bool(torch.equal(cu_seqlens, expected_cu)):
            raise RuntimeError(
                "qwen4_exp QSA draft incremental prefill cu_seqlens do not "
                "match input_lengths"
            )
    else:
        qsa_assert_valid(
            (cu_seqlens == expected_cu).all(),
            "qwen4_exp QSA draft incremental prefill cu_seqlens do not "
            "match input_lengths",
        )
    expected_cu_kv = prep.cu_kv_seqlens
    # int32 headroom is host arithmetic on the mirror values; no device read.
    int32_max = torch.iinfo(torch.int32).max
    if (
        max(prefix + length for prefix, length in zip(prefix_values, lengths))
        > int32_max
    ):
        raise RuntimeError(
            "qwen4_exp QSA draft incremental prefill positions exceed int32"
        )
    if (
        sum(prefix + length for prefix, length in zip(prefix_values, lengths))
        > int32_max
    ):
        raise RuntimeError(
            "qwen4_exp QSA draft incremental prefill cumulative KV lengths "
            "exceed int32"
        )
    cu_kv_seqlens = main_inputs.cu_kv_seqlens_device
    if (
        cu_kv_seqlens.dim() != 1
        or cu_kv_seqlens.dtype != torch.int32
        or cu_kv_seqlens.device != device
    ):
        raise RuntimeError(
            "qwen4_exp QSA draft incremental prefill cu_kv_seqlens do not "
            "provide device int32 [B + 1] metadata"
        )
    if qsa_strict_validation_enabled():
        if not bool(torch.equal(cu_kv_seqlens, expected_cu_kv)):
            if not draft_prefix_reuse:
                raise RuntimeError(
                    "qwen4_exp QSA draft incremental prefill cu_kv_seqlens do not "
                    "match prefix+input lengths"
                )
            if int(cu_kv_seqlens.numel()) != batch_size + 1 or bool(
                torch.any(cu_kv_seqlens[1:] < cu_kv_seqlens[:-1]).item()
            ):
                raise RuntimeError(
                    "qwen4_exp QSA MTP draft prefix-reuse has invalid local "
                    "cu_kv_seqlens metadata"
                )
    elif draft_prefix_reuse:
        # Local cu_kv coordinates only need monotonicity (equality with the
        # absolute expected ends is intentionally not required here).
        if int(cu_kv_seqlens.numel()) != batch_size + 1:
            raise RuntimeError(
                "qwen4_exp QSA MTP draft prefix-reuse has invalid local "
                "cu_kv_seqlens metadata"
            )
        qsa_assert_valid(
            torch.all(cu_kv_seqlens[1:] >= cu_kv_seqlens[:-1]),
            "qwen4_exp QSA MTP draft prefix-reuse has invalid local "
            "cu_kv_seqlens metadata",
        )
    else:
        qsa_assert_valid(
            (cu_kv_seqlens == expected_cu_kv).all(),
            "qwen4_exp QSA draft incremental prefill cu_kv_seqlens do not "
            "match prefix+input lengths",
        )

    # prefix_values already came from the pinned host mirror above; the
    # logical positions ride the prep's pinned int64 batch instead of a
    # per-request arange chain.
    logical_positions = prep.logical_positions
    position_ids = main_inputs.combo_position_ids
    index_factor = int(rope_config.index_factor)
    if position_ids is None or int(position_ids.numel()) != token_count * index_factor:
        actual = 0 if position_ids is None else int(position_ids.numel())
        raise RuntimeError(
            "qwen4_exp QSA draft incremental prefill position ids do not "
            f"match index_factor={index_factor}: got {actual} values for "
            f"{token_count} tokens"
        )
    target_mrope = (
        not draft
        and not context.is_mtp_draft
        and is_qsa_rope_style(rope_config, "Mrope")
    )
    # Host-only equal-axes verdict over the transported suffix rows.  Text
    # requests repeat one position across their three axes, image rows do
    # not; host-resident rows compare on the CPU (never touching the
    # device), and device-resident rows keep the eager MRoPE builder because
    # reading their axes back would synchronize the plan window.  The verdict
    # covers every block-start row the mrope sources select below.
    mrope_suffix_text = target_mrope and (
        mrope_equal_axis_verdict(
            position_ids, token_count=token_count, index_factor=index_factor
        )
        is True
    )
    try:
        if target_mrope and any(length % ratio for length in lengths):
            # Ratio-aligned requests have no unfinished tail to validate, so
            # only steps with at least one ragged request pay for the check.
            position_rows = position_ids.reshape(token_count, index_factor)
            if qsa_strict_validation_enabled():
                if not bool(
                    mrope_decode_tail_verdict_packed(position_rows, expected_cu, ratio)
                ):
                    # Slow path only on failure: the legacy per-request
                    # validation reproduces the exact error and its order.
                    offset = 0
                    for length in lengths:
                        validate_mrope_decode_tail(
                            position_rows[offset : offset + length], ratio
                        )
                        offset += length
            else:
                # One enqueue-only batched device assert replaces the
                # per-request assert chain; a violation traps fail-stop at the
                # next host synchronization.
                qsa_assert_valid(
                    mrope_decode_tail_verdict_packed(position_rows, expected_cu, ratio),
                    "qwen4_exp QSA unfinished MRoPE compression group requires "
                    "a consecutive equal-axis text tail ending at the maximum "
                    "position",
                )
        current_cos, current_sin = build_qsa_rope(
            position_ids,
            rope_config,
            token_count=token_count,
            dtype=torch.bfloat16,
            device=device,
            logical_positions=None if target_mrope else logical_positions,
            # Text-shaped transports (Base style, or MRoPE whose rows the
            # logical-positions check inside build_qsa_rope proves
            # equal-axis) engage the fused text kernel through the shared
            # cached inv_freq tables; target MRoPE proves its rows on the
            # host when the transport is host-resident, and multimodal or
            # device-resident rows keep the eager builder.
            inv_freq=None if target_mrope else cached_inv_freq(rope_config, device),
        )
    except ValueError as error:
        raise RuntimeError(
            f"qwen4_exp QSA draft incremental prefill {error}"
        ) from error

    # Page sizes were read once above for the batched prep; the legacy
    # validation keeps its exact checks and messages.
    if kv_tokens_per_block <= 0 or kv_tokens_per_block % ratio:
        raise RuntimeError(
            "qwen4_exp QSA indexer KV page size must be positive and divisible "
            f"by ratio={ratio}, got {kv_tokens_per_block}"
        )
    if state_tokens_per_block <= 0:
        raise RuntimeError("qwen4_exp QSA indexer state page size must be positive")
    # Page alignment is request data: host arithmetic on the prefix mirror.
    if not draft and any(value % kv_tokens_per_block for value in prefix_values):
        raise RuntimeError(
            "qwen4_exp QSA prefix reuse requires prefixes aligned to the "
            f"indexer KV page size ({kv_tokens_per_block})"
        )
    kv_entries_per_block = kv_tokens_per_block // ratio

    # The frozen grid tensors came from the batched prep above.  The int32
    # padded grid doubles as the padded KV lengths and (flattened) as the
    # visible-length rows the selection consumer reads; the per-request
    # [1, length] FMHA rows are contiguous views into it (a leading size-1
    # dim keeps them contiguous), so the plan never materializes the int64
    # grid the legacy chain built only to cast away.
    visible_lengths = prep.visible_lengths
    compressed_lengths = prep.compressed_lengths
    required_kv_columns = prep.required_kv_columns
    required_state_columns = prep.required_state_columns
    kv_lens_rows = tuple(
        visible_lengths[request_idx : request_idx + 1, :length]
        for request_idx, length in enumerate(lengths)
    )
    padded_kv_lens = visible_lengths

    block_starts = prep.block_starts
    try:
        if target_mrope:
            # Each page-aligned suffix contains the first row of every new
            # group, including image groups with unequal/nonlinear axes.
            # Multimodal positions keep the eager MRoPE builder (which now
            # reads the shared cached inv_freq table); the fused text kernel
            # engages for proven equal-axis text transports -- the suffix
            # verdict above covers every row the sources select.
            sources = prep.mrope_sources
            block_positions = (
                position_ids.to(device=device)
                .reshape(token_count, index_factor)
                .index_select(0, sources)
            )
            rope_cos, rope_sin = build_qsa_rope(
                block_positions.reshape(-1),
                rope_config,
                token_count=token_count,
                dtype=torch.bfloat16,
                device=device,
                # The selected rows are device-resident; only the caller's
                # suffix proof lets them reach the fused kernel.
                validate_position_axes=not mrope_suffix_text,
            )
        else:
            rope_cos, rope_sin = build_qsa_rope_from_logical_positions(
                block_starts,
                rope_config,
                token_count=token_count,
                dtype=torch.bfloat16,
                device=device,
                inv_freq=cached_inv_freq(rope_config, device),
            )
    except ValueError as error:
        raise RuntimeError(
            f"qwen4_exp QSA draft incremental prefill {error}"
        ) from error

    score_weights = prep.score_weights
    padded_score_weights = prep.padded_score_weights
    selection_row_map = prep.selection_row_map
    selection_valid_rows = prep.selection_valid_rows
    selection_valid = prep.selection_valid

    # Main-model fused prefix reuse consumes the plan's tables directly.
    # Skip the unused bundle for its candidates; draft paths still need it.
    # A consumer-side decline retains the legacy transactional fallback.
    prefix_reuse_fused_supported = prefix_reuse_fused_geometry_supported(
        batch_size=batch_size,
        head_dim=head_dim,
        head_num=head_num,
        ratio=ratio,
        lengths=lengths,
        prefixes=prefix_values,
        kv_tokens_per_block=kv_tokens_per_block,
        state_tokens_per_block=state_tokens_per_block,
    )
    prefix_reuse_fused_candidate = (
        not draft
        and not draft_prefix_reuse
        and not bool(context.is_mtp_draft)
        and device.type == "cuda"
        and torch.version.hip is None
        and prefix_reuse_fused_supported
        and selection_valid_rows is not None
        and padded_score_weights is not None
        and current_cos is not None
        and current_cos.shape[-1] == 64
        and rope_cos is not None
        and rope_cos.shape[-1] == 64
    )
    # Freeze the writer geometry for both fast write forms: short windows
    # (every request within the state ring) prove the destination slots
    # unique and take the full-shape write, while longer monotone windows --
    # including the production draft tail (b=1, N=1578) -- take the
    # ring-wrap batch write over each (request, block)'s tail rows.  The
    # builder itself declines windows neither form can serve.
    # The incremental build validated the cu partition (torch.equal against
    # the pinned mirrors) and both tables (required-blocks verdicts cover every
    # column the geometry reads), so the redundant resolve asserts are dropped.
    writer_geometry = None
    if not prefix_reuse_fused_candidate:
        writer_geometry = _build_step_writer_geometry(
            context,
            cu_seqlens=expected_cu,
            start_positions=prep.prefixes_i64,
            token_count=token_count,
            ratio=ratio,
            kv_tokens_per_block=kv_tokens_per_block,
            state_tokens_per_block=state_tokens_per_block,
            host_lengths=lengths,
            host_prefixes=prefix_values,
            trusted_tables=True,
        )

    return Qwen4ExpPrefillStepPlan(
        mode=(
            MODE_DRAFT_PREFIX_REUSE
            if draft_prefix_reuse
            else (MODE_DRAFT_INCREMENTAL if draft else MODE_PREFIX_REUSE)
        ),
        main_inputs=main_inputs,
        indexer_kv_inputs=context.indexer_kv_inputs,
        indexer_state_inputs=context.indexer_state_inputs,
        is_mtp_draft=bool(context.is_mtp_draft),
        token_count=token_count,
        device=device,
        rope_dtype=torch.bfloat16,
        rope_signature=rope_signature(rope_config),
        source_fingerprints=_collect_fingerprints(main_inputs),
        region_fingerprints=_collect_region_fingerprints(_cache_regions(context)),
        source_tensors=_collect_source_tensors(main_inputs),
        lengths=lengths,
        prefix_values=prefix_values,
        cu_host=prep.cu_host,
        tokens=token_count,
        batch_size=batch_size,
        max_len=max(lengths),
        max_ctx_len=max(
            (prefix + length) // ratio for prefix, length in zip(prefix_values, lengths)
        ),
        head_dim=head_dim,
        head_num=head_num,
        ratio=ratio,
        token_budget=token_budget,
        kv_tokens_per_block=kv_tokens_per_block,
        kv_entries_per_block=kv_entries_per_block,
        state_tokens_per_block=state_tokens_per_block,
        required_state_columns=required_state_columns,
        required_state_columns_device=prep.required_state_columns_device,
        prefixes_i64=prep.prefixes_i64,
        prefixes_i32=prep.prefixes_i32,
        input_lengths_i32=prep.input_lengths_i32,
        cu_seqlens=expected_cu,
        logical_positions=prep.logical_positions,
        visible_lengths=visible_lengths,
        visible_lengths_i32=prep.visible_lengths_i32,
        compressed_lengths=compressed_lengths,
        required_kv_columns=required_kv_columns,
        current_cos=current_cos,
        current_sin=current_sin,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        selection_row_map=selection_row_map,
        selection_valid_rows=selection_valid_rows,
        selection_valid=selection_valid,
        score_weights=score_weights,
        padded_score_weights=padded_score_weights,
        kv_lens_rows=kv_lens_rows,
        padded_kv_lens=padded_kv_lens,
        prefix_reuse_fused_supported=prefix_reuse_fused_supported,
        writer_geometry=writer_geometry,
    )


def build_cold_step_plan(
    context: Any,
    *,
    indexer: Any,
    token_count: int,
    device: torch.device,
    rope_config: Any,
    rope_dtype: torch.dtype,
) -> Qwen4ExpPrefillStepPlan:
    """Freeze the zero-prefix prefill metadata (legacy writer preflights).

    Ports the validation and padded RoPE table construction of
    ``write_prefill_indexer_cache`` that only depend on the step's inputs;
    pool/table binding and both writers stay per layer.  The cu partition
    preflight runs here exactly when the legacy path ran it before any
    writer (``not defer_cu``); the deferred error-path recheck stays with
    the caller.
    """
    lengths = context._validate_mode_and_metadata(expect_prefill=True)
    main_inputs = context.main_inputs
    batch_size = len(lengths)
    head_dim = int(indexer.head_dim)
    ratio = int(indexer.compress_ratio)
    if sum(lengths) != token_count:
        raise RuntimeError(
            f"qwen4_exp QSA input_lengths={lengths} do not partition "
            f"the {token_count} packed raw keys"
        )

    kv_tokens_per_block = int(context.indexer_kv_cache.seq_size_per_block)
    state_tokens_per_block = int(context.indexer_state_cache.seq_size_per_block)
    if kv_tokens_per_block <= 0 or kv_tokens_per_block % ratio:
        raise RuntimeError(
            "qwen4_exp QSA indexer KV page size must be positive and divisible "
            f"by ratio={ratio}, got {kv_tokens_per_block}"
        )
    prefix_tensor = main_inputs.prefix_lengths
    if (
        prefix_tensor is None
        or prefix_tensor.dim() != 1
        or int(prefix_tensor.numel()) != len(lengths)
        or prefix_tensor.dtype != torch.int32
    ):
        raise RuntimeError("qwen4_exp QSA prefill requires int32 prefix_lengths [B]")
    # Device sources (RTP_LLM_DEVICE_INPUT) read back once per step; host
    # sources pass through.  The device cast below uses the source tensor.
    prefix_values = host_input_mirror(main_inputs, "prefix_lengths").tolist()
    start_positions = prefix_tensor.to(
        device=device, dtype=torch.int64, non_blocking=True
    ).contiguous()

    cu_seqlens = main_inputs.cu_seqlens_device
    if (
        cu_seqlens.dim() != 1
        or cu_seqlens.dtype != torch.int32
        or int(cu_seqlens.numel()) != batch_size + 1
        or cu_seqlens.device != device
    ):
        raise RuntimeError("qwen4_exp QSA requires device int32 cu_seqlens [B + 1]")

    def check_cu_partition():
        expected_cu = torch.tensor(
            [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
            dtype=torch.int32,
            device=device,
        )
        if not bool(torch.equal(cu_seqlens, expected_cu)):
            raise RuntimeError("qwen4_exp QSA cu_seqlens do not match input_lengths")

    defer_cu = (
        ratio == 4
        and state_tokens_per_block == kv_tokens_per_block
        and device.type == "cuda"
    )
    if not defer_cu:
        check_cu_partition()

    position_ids = main_inputs.combo_position_ids
    index_factor = int(rope_config.index_factor)
    if position_ids is None or int(position_ids.numel()) != token_count * index_factor:
        actual = 0 if position_ids is None else int(position_ids.numel())
        raise RuntimeError(
            "qwen4_exp QSA position ids do not match the configured "
            f"index_factor={index_factor}: got {actual} values for "
            f"{token_count} tokens"
        )
    positions = position_ids.reshape(token_count, index_factor)
    # One batched verdict replaces the per-request tail validation inside the
    # loop below; only a failed verdict re-runs the legacy per-request checks,
    # so their exact per-request error attribution is preserved.
    mrope_tail_failed = False
    if (
        not context.is_mtp_draft
        and is_qsa_rope_style(rope_config, "Mrope")
        and any(seq_len % ratio for seq_len in lengths)
    ):
        packed_cu = torch.tensor(
            [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
            dtype=torch.int32,
            device=device,
        )
        mrope_tail_failed = not bool(
            mrope_decode_tail_verdict_packed(positions, packed_cu, ratio)
        )
    max_len = max(lengths)
    rotary_dim = int(rope_config.dim)
    rope_cos = torch.zeros(
        (batch_size, max_len, rotary_dim),
        dtype=rope_dtype,
        device=device,
    )
    rope_sin = torch.zeros_like(rope_cos)
    offset = 0
    for request_idx, seq_len in enumerate(lengths):
        end = offset + seq_len
        # The MTP draft prefill carries the engine's shifted positions
        # (prompt positions moved one row left plus a tail anchor), so its
        # RoPE positions intentionally differ from the logical cache rows.
        logical_positions = None
        if is_qsa_rope_style(rope_config, "Base") and not context.is_mtp_draft:
            prefix_len = 0
            logical_positions = prefix_len + torch.arange(
                seq_len, dtype=torch.int64, device=device
            )
        try:
            if mrope_tail_failed:
                validate_mrope_decode_tail(positions[offset:end], ratio)
            cos, sin = build_qsa_rope(
                positions[offset:end].reshape(-1),
                rope_config,
                token_count=seq_len,
                dtype=rope_dtype,
                device=device,
                logical_positions=logical_positions,
            )
        except ValueError as error:
            prefixes = main_inputs.prefix_lengths
            prefix_head = prefixes[:6].tolist() if prefixes is not None else None
            raise RuntimeError(
                f"qwen4_exp QSA prefill {error} "
                f"[request={request_idx} seq_len={seq_len} "
                f"lengths[:6]={lengths[:6]} prefix_lengths[:6]={prefix_head} "
                f"is_mtp_draft={context.is_mtp_draft}]"
            ) from error
        rope_cos[request_idx, :seq_len].copy_(cos)
        rope_sin[request_idx, :seq_len].copy_(sin)
        offset = end

    # The zero-prefix fused writer does not consume writer_geometry. Skip
    # that work only for shapes it may serve, using the plan's known facts.
    # A later per-layer layout or alias rejection retains the ordinary
    # legacy fallback, which needs no bundle.
    fused_writer_candidate = (
        defer_cu
        and torch.version.hip is None
        and kv_tokens_per_block in (64, 128)
        and head_dim == 128
        and rotary_dim == 64
        and rope_dtype == torch.bfloat16
        and batch_size > 0
        and all(n > 0 for n in lengths)
    )
    writer_geometry = None
    if not fused_writer_candidate:
        writer_geometry = _build_step_writer_geometry(
            context,
            cu_seqlens=cu_seqlens,
            start_positions=start_positions,
            token_count=token_count,
            ratio=ratio,
            kv_tokens_per_block=kv_tokens_per_block,
            state_tokens_per_block=state_tokens_per_block,
            host_lengths=lengths,
            host_prefixes=prefix_values,
        )

    return Qwen4ExpPrefillStepPlan(
        mode=MODE_COLD_PREFILL,
        main_inputs=main_inputs,
        indexer_kv_inputs=context.indexer_kv_inputs,
        indexer_state_inputs=context.indexer_state_inputs,
        is_mtp_draft=bool(context.is_mtp_draft),
        token_count=token_count,
        device=device,
        rope_dtype=rope_dtype,
        rope_signature=rope_signature(rope_config),
        source_fingerprints=_collect_fingerprints(main_inputs),
        region_fingerprints=_collect_region_fingerprints(_cache_regions(context)),
        source_tensors=_collect_source_tensors(main_inputs),
        lengths=lengths,
        prefix_values=prefix_values,
        cu_host=[0] + list(torch.tensor(lengths).cumsum(0).tolist()),
        tokens=token_count,
        batch_size=batch_size,
        max_len=max_len,
        head_dim=head_dim,
        ratio=ratio,
        kv_tokens_per_block=kv_tokens_per_block,
        kv_entries_per_block=kv_tokens_per_block // ratio,
        state_tokens_per_block=state_tokens_per_block,
        prefixes_i64=start_positions,
        rope_table_cos=rope_cos,
        rope_table_sin=rope_sin,
        cu_checked=not defer_cu,
        writer_geometry=writer_geometry,
    )


def get_prefill_step_plan(
    context: Any,
    *,
    indexer: Any,
    token_count: int,
    device: torch.device,
    rope_config: Any,
    draft: bool,
    draft_prefix_reuse: bool,
) -> Qwen4ExpPrefillStepPlan:
    """Return this step's frozen incremental-prefill plan (cache or build)."""
    mode = (
        MODE_DRAFT_PREFIX_REUSE
        if draft_prefix_reuse
        else (MODE_DRAFT_INCREMENTAL if draft else MODE_PREFIX_REUSE)
    )
    key = (id(context.main_inputs), token_count, mode)
    if not qsa_strict_validation_enabled():
        entry = _STEP_PLAN_CACHE.get(key)
        if entry is not None:
            _, plan = entry
            if _step_plan_applies(
                plan,
                context,
                indexer=indexer,
                token_count=token_count,
                device=device,
                rope_config=rope_config,
            ):
                _STEP_PLAN_STATS["hits"] += 1
                return plan
            _STEP_PLAN_STATS["invalidations"] += 1
            del _STEP_PLAN_CACHE[key]
    _STEP_PLAN_STATS["misses"] += 1
    plan = build_incremental_step_plan(
        context,
        indexer=indexer,
        token_count=token_count,
        device=device,
        rope_config=rope_config,
        draft=draft,
        draft_prefix_reuse=draft_prefix_reuse,
    )
    if not qsa_strict_validation_enabled():
        _cache_plan(plan)
    return plan


def get_cold_step_plan(
    context: Any,
    *,
    indexer: Any,
    token_count: int,
    device: torch.device,
    rope_config: Any,
    rope_dtype: torch.dtype,
) -> Qwen4ExpPrefillStepPlan:
    """Return this step's frozen zero-prefix prefill plan (cache or build)."""
    key = (id(context.main_inputs), token_count, MODE_COLD_PREFILL)
    if not qsa_strict_validation_enabled():
        entry = _STEP_PLAN_CACHE.get(key)
        if entry is not None:
            _, plan = entry
            if _step_plan_applies(
                plan,
                context,
                indexer=indexer,
                token_count=token_count,
                device=device,
                rope_config=rope_config,
                rope_dtype=rope_dtype,
            ):
                _STEP_PLAN_STATS["hits"] += 1
                return plan
            _STEP_PLAN_STATS["invalidations"] += 1
            del _STEP_PLAN_CACHE[key]
    _STEP_PLAN_STATS["misses"] += 1
    plan = build_cold_step_plan(
        context,
        indexer=indexer,
        token_count=token_count,
        device=device,
        rope_config=rope_config,
        rope_dtype=rope_dtype,
    )
    if not qsa_strict_validation_enabled():
        _cache_plan(plan)
    return plan


def lookup_prefill_step_plan(
    main_inputs: Any, token_count: int
) -> Qwen4ExpPrefillStepPlan | None:
    """Find the frozen plan for this step's inputs, any prefill mode.

    Strict validation bypasses the cache, so it never serves a plan there.
    """
    if qsa_strict_validation_enabled():
        return None
    for (inputs_id, tokens, _mode), (owner, plan) in _STEP_PLAN_CACHE.items():
        if tokens == token_count and owner is main_inputs:
            return plan
    return None


def prefill_step_plan_hit(
    main_inputs: Any,
    token_count: int,
    *,
    device: torch.device | None = None,
    batch_size: int | None = None,
) -> Qwen4ExpPrefillStepPlan | None:
    """Lookup a frozen plan and re-check only fingerprints (no value reads).

    The plan build validated the mirror values once; per-layer consumers call
    this instead of re-reading the mirrors (``tolist``/``any``/list
    comparisons).  The source fingerprints carry the same guarantee the
    per-layer value comparison used to establish: unchanged tensors keep the
    build's verdict, and anything swapped, rebound or edited in place falls
    back to the caller's legacy path.  Strict validation never serves a hit.
    """
    if qsa_strict_validation_enabled():
        return None
    plan = lookup_prefill_step_plan(main_inputs, token_count)
    if plan is None:
        return None
    if device is not None and plan.device != device:
        return None
    if batch_size is not None and plan.batch_size != batch_size:
        return None
    for name, fingerprint in plan.source_fingerprints.items():
        if _tensor_fingerprint(getattr(main_inputs, name, None)) != fingerprint:
            return None
    return plan
