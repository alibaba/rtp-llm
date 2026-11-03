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
from dataclasses import dataclass, field
from typing import Any

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import (
    _normalized_rope_style_name,
    build_qsa_rope,
    build_qsa_rope_from_logical_positions,
    is_qsa_rope_style,
    mrope_decode_tail_verdict,
    validate_mrope_decode_tail,
)

from .qsa_cache_views import (
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


def step_plan_cache_stats() -> dict:
    """Cache counters for benches and diagnostics (hits/misses/invalidations)."""
    return dict(_STEP_PLAN_STATS, mirror_readbacks=_MIRROR_STATS["readbacks"])


def clear_step_plan_cache() -> None:
    """Drop every cached plan (tests isolate cases that share inputs objects)."""
    _STEP_PLAN_CACHE.clear()
    _MIRROR_CACHE.clear()


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


def _cache_regions(context: Any) -> tuple:
    return (context.main_inputs, context.indexer_kv_inputs, context.indexer_state_inputs)


def _collect_region_fingerprints(regions: tuple) -> tuple[dict[str, tuple[int, int] | None], ...]:
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
    if _collect_region_fingerprints(_cache_regions(context)) != plan.region_fingerprints:
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
    # Per-step CPU mirrors: a device source (RTP_LLM_DEVICE_INPUT) was read
    # back once for this step; the layers of one step compare host values
    # without another synchronization.
    lengths_mirror = host_input_mirror(context.main_inputs, "input_lengths")
    if lengths_mirror is None:
        return False
    if [int(value) for value in lengths_mirror.tolist()] != plan.lengths:
        return False
    prefixes_mirror = host_input_mirror(context.main_inputs, "prefix_lengths")
    if prefixes_mirror is None:
        return False
    if [int(value) for value in prefixes_mirror.tolist()] != plan.prefix_values:
        return False
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


def _build_kv_lens_rows(
    prefixes_i32: torch.Tensor,
    lengths: list[int],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Per-request [1, length] visible-KV lengths (legacy fmha-prep rows)."""
    return tuple(
        (
            prefixes_i32[request_idx]
            + torch.arange(1, length + 1, dtype=torch.int32, device=device)
        )
        .unsqueeze(0)
        .contiguous()
        for request_idx, length in enumerate(lengths)
    )


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
    invalid_prefix = any(
        value <= 0 if draft else value < 0 for value in prefix_values
    )
    if invalid_prefix:
        requirement = "nonzero" if draft else "non-negative"
        raise RuntimeError(
            f"qwen4_exp QSA {label} requires every prefix length to be "
            f"{requirement}"
        )
    prefixes = prefix_source.to(
        device=device, dtype=torch.int64, non_blocking=True
    ).contiguous()
    prefixes_i32 = prefix_source.to(
        device=device, dtype=torch.int32, non_blocking=True
    ).contiguous()
    input_lengths_i32 = main_inputs.input_lengths.to(
        device=device, dtype=torch.int32, non_blocking=True
    ).contiguous()
    sequence_lengths = main_inputs.sequence_lengths
    if sequence_lengths.numel() != 0:
        raise RuntimeError(
            "qwen4_exp QSA draft incremental prefill requires empty " "sequence_lengths"
        )

    cu_seqlens = main_inputs.cu_seqlens_device
    expected_cu = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    expected_cu[1:] = torch.tensor(lengths, dtype=torch.int32, device=device).cumsum(0)
    if (
        cu_seqlens.dim() != 1
        or cu_seqlens.dtype != torch.int32
        or cu_seqlens.device != device
    ):
        raise RuntimeError(
            "qwen4_exp QSA draft incremental prefill cu_seqlens do not "
            "match input_lengths"
        )
    if qsa_strict_validation_enabled():
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
    expected_cu_kv = torch.zeros_like(expected_cu)
    visible_ends = prefixes + torch.tensor(lengths, dtype=torch.int64, device=device)
    # int32 headroom is host arithmetic on the mirror values; no device read.
    int32_max = torch.iinfo(torch.int32).max
    if max(
        prefix + length for prefix, length in zip(prefix_values, lengths)
    ) > int32_max:
        raise RuntimeError(
            "qwen4_exp QSA draft incremental prefill positions exceed int32"
        )
    if sum(prefix + length for prefix, length in zip(prefix_values, lengths)) > int32_max:
        raise RuntimeError(
            "qwen4_exp QSA draft incremental prefill cumulative KV lengths "
            "exceed int32"
        )
    expected_cu_kv[1:] = visible_ends.to(torch.int32).cumsum(0)
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

    # prefix_values already came from the pinned host mirror above; no device
    # readback is needed for the logical positions.
    logical_positions = torch.cat(
        [
            torch.arange(prefix, prefix + length, device=device)
            for prefix, length in zip(prefix_values, lengths)
        ]
    ).to(torch.int64)
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
    try:
        if target_mrope:
            offset = 0
            position_rows = position_ids.reshape(token_count, index_factor)
            for length in lengths:
                if qsa_strict_validation_enabled():
                    validate_mrope_decode_tail(
                        position_rows[offset : offset + length], ratio
                    )
                else:
                    # One enqueue-only device assert per request; a violation
                    # traps fail-stop at the next host synchronization.
                    qsa_assert_valid(
                        mrope_decode_tail_verdict(
                            position_rows[offset : offset + length], ratio
                        ),
                        "qwen4_exp QSA unfinished MRoPE compression group requires "
                        "a consecutive equal-axis text tail ending at the maximum "
                        "position",
                    )
                offset += length
        current_cos, current_sin = build_qsa_rope(
            position_ids,
            rope_config,
            token_count=token_count,
            dtype=torch.bfloat16,
            device=device,
            logical_positions=None if target_mrope else logical_positions,
        )
    except ValueError as error:
        raise RuntimeError(
            f"qwen4_exp QSA draft incremental prefill {error}"
        ) from error

    kv_tokens_per_block = int(context.indexer_kv_cache.seq_size_per_block)
    state_tokens_per_block = int(context.indexer_state_cache.seq_size_per_block)
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

    row_offsets = torch.arange(max(lengths), dtype=torch.int64, device=device)
    visible_lengths = prefixes[:, None] + row_offsets[None, :] + 1
    valid_rows = (
        row_offsets[None, :]
        < torch.tensor(lengths, dtype=torch.int64, device=device)[:, None]
    )
    visible_lengths = torch.where(valid_rows, visible_lengths, 0)
    compressed_lengths = (visible_lengths // ratio).to(torch.int32)
    final_compressed_lengths = (visible_ends // ratio).to(torch.int32)
    required_kv_columns = (
        final_compressed_lengths + kv_entries_per_block - 1
    ) // kv_entries_per_block

    required_state_columns = torch.tensor(
        [
            (prefix + length + state_tokens_per_block - 1) // state_tokens_per_block
            for prefix, length in zip(prefix_values, lengths)
        ],
        dtype=torch.int32,
    )

    block_starts = logical_positions - logical_positions.remainder(ratio)
    try:
        if target_mrope:
            # Each page-aligned suffix contains the first row of every new
            # group, including image groups with unequal/nonlinear axes.
            sources = torch.arange(token_count, device=device)
            sources = sources - logical_positions.remainder(ratio)
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
            )
        else:
            rope_cos, rope_sin = build_qsa_rope_from_logical_positions(
                block_starts,
                rope_config,
                token_count=token_count,
                dtype=torch.bfloat16,
                device=device,
            )
    except ValueError as error:
        raise RuntimeError(
            f"qwen4_exp QSA draft incremental prefill {error}"
        ) from error

    # Per-step selection constants that every layer previously rebuilt.
    selection_row_map, selection_valid_rows, selection_valid = _build_selection_padding(
        lengths, device=device
    )
    score_weights = torch.full(
        (token_count, head_num),
        1.0 / math.sqrt(head_dim),
        dtype=torch.float32,
        device=device,
    )
    padded_score_weights = score_weights.index_select(0, selection_row_map)
    kv_lens_rows = _build_kv_lens_rows(prefixes_i32, lengths, device=device)
    padded_kv_lens = torch.where(
        selection_valid,
        prefixes_i32.unsqueeze(1)
        + torch.arange(1, max(lengths) + 1, dtype=torch.int32, device=device),
        torch.zeros((), dtype=torch.int32, device=device),
    ).contiguous()

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
        cu_host=[0]
        + [
            total
            for total in torch.tensor(lengths).cumsum(0).tolist()
        ],
        tokens=token_count,
        batch_size=batch_size,
        max_len=max(lengths),
        max_ctx_len=max(
            (prefix + length) // ratio
            for prefix, length in zip(prefix_values, lengths)
        ),
        head_dim=head_dim,
        head_num=head_num,
        ratio=ratio,
        token_budget=token_budget,
        kv_tokens_per_block=kv_tokens_per_block,
        kv_entries_per_block=kv_entries_per_block,
        state_tokens_per_block=state_tokens_per_block,
        required_state_columns=required_state_columns,
        required_state_columns_device=required_state_columns.to(device),
        prefixes_i64=prefixes,
        prefixes_i32=prefixes_i32,
        input_lengths_i32=input_lengths_i32,
        cu_seqlens=expected_cu,
        logical_positions=logical_positions,
        visible_lengths=visible_lengths,
        visible_lengths_i32=visible_lengths.reshape(-1).to(torch.int32),
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
            if not context.is_mtp_draft and is_qsa_rope_style(rope_config, "Mrope"):
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

    Consumers (sparse-GQA FMHA input planning) re-read the pinned length
    mirrors and compare them with the plan before consuming any field, so a
    plan whose lengths/prefixes changed can never be used.  Strict validation
    bypasses the cache, so it never serves a plan there.
    """
    if qsa_strict_validation_enabled():
        return None
    for (inputs_id, tokens, _mode), (owner, plan) in _STEP_PLAN_CACHE.items():
        if tokens == token_count and owner is main_inputs:
            return plan
    return None
