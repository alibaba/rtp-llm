"""DSV4 FP8 ``PoolHandle`` — per-(layer_id, region) resolved descriptor.

Foundation module for M06 (Python address helpers). Single source of truth
for the per-region pool descriptor consumed by all five address-resolving
call sites (decode fused phase2b, prefill compressor non-CP/CP,
``_kv_cache_utils._compute_pool_slots`` shim, decode meta state cyclic
slot mapping, SWA prefill UTs).

Replaces (and centralises) the per-layer triplet:
  * ``AttentionFP8._pool_view``              (attention.py)
  * ``AttentionFP8._pool_view_3d_fp8``       (attention.py)
  * ``AttentionFP8._pool_entries_per_block`` (attention.py)

Two construction paths are provided by :func:`make_pool_handle`:

  * Path A (descriptor-driven): consumes ``PyKVCacheRegionDesc`` published
    by M05 — eb / stride / TMA-pad / cyclic ring depth come from the
    canonical descriptor, no runtime arithmetic.
  * Path B (legacy fallback): derives eb from ``base.stride[0] * element_size()``
    exactly as today's ``_pool_view*`` triplet does. ``max_state_blocks``
    stays None on Path B; cyclic addressing is *not* served by legacy
    handles.

The class is a frozen dataclass; it carries no per-iteration state and is
cheap to keep on per-AttentionFP8 / metadata objects. See
``docs/dsv4/kvcache-unify-final/modules/M06_python_address_helpers.md``
§2.1 for the full field-by-field contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch

# Canonical KVCacheRegionName ints (must mirror C++ enum at
# ``cpp/cache/CacheGroupType.h:50-57`` and the Python mirror at
# ``models_py/modules/dsv4/attn_type.py``). Pinned by M06 §3.1.4.
REGION_DEFAULT = 0
REGION_CSA_KV = 1
REGION_HCA_KV = 2
REGION_INDEXER_KV = 3
REGION_INDEXER_STATE = 4
REGION_CSA_STATE = 5
REGION_HCA_STATE = 6
REGION_SWA_KV = 7
REGION_COUNT = 8

_STATE_REGIONS = frozenset({REGION_INDEXER_STATE, REGION_CSA_STATE, REGION_HCA_STATE})

# Per-region compression ratio (CP path / boundary math).
# Mirrors C08 §35.1 + M06 §3.4 enumeration; ratio==1 = no compression.
_RATIO_BY_REGION = {
    REGION_SWA_KV: 1,
    REGION_CSA_KV: 4,
    REGION_HCA_KV: 128,
    REGION_INDEXER_KV: 1,
    REGION_INDEXER_STATE: 1,
    REGION_CSA_STATE: 1,
    REGION_HCA_STATE: 1,
}


@dataclass(frozen=True)
class PoolHandle:
    """Per-(layer_id, region) descriptor; computed once per layer.

    Dtype-polymorphic: STATE pools (fp32 vec_dim views) and KV pools
    (uint8 paged views) share the same dataclass shape — readers consume
    ``entry_dtype`` + ``entry_elements`` instead of branching on
    ``region.is_state``.
    """

    region_id: int  # KVCacheRegionName int (0..7)
    eb: int  # entries_per_block (KV: 256/64/2; state: 1..N)
    entry_logical_bytes: int  # bytes per entry (584 / 132 / fp32*vec_dim)
    entry_dtype: torch.dtype  # underlying tensor dtype (uint8 FP8 KV / fp32 STATE)
    entry_elements: int  # element count per entry (NOT bytes)
    scale_bytes: int  # 8 (canonical) or 0
    has_scale: bool  # parity with ``is_fp8_swa_slot_pool`` predicate
    base_2d: Optional[torch.Tensor]  # [num_slots, vec_dim] (None when TMA pad blocks)
    base_3d: Optional[torch.Tensor]  # [num_blocks, eb, entry_bytes]
    block_stride_bytes: int  # uint8-view stride(0) in BYTES
    is_state: bool  # True iff region ∈ {INDEXER_STATE, CSA_STATE, HCA_STATE}
    ratio: int  # SWA/INDEXER/STATE=1, CSA=4, HCA=128

    # STATE-only: pair width (kv_state||score_state) for compressor consumers.
    state_pair_dim: Optional[int] = None

    # Optional producer-consumer ordering event.
    writer_event: Optional[torch.cuda.Event] = None

    # F02-aware (placeholder; bps≡1 today ⇒ super_block_id == pool_block_id).
    bps: int = 1
    super_to_pool_block: Optional[torch.Tensor] = None

    # STATE-only cyclic ring depth. Populated from
    # ``PyKVCacheRegionDesc.max_state_blocks`` on Path A. Path B leaves it
    # None; callers MUST guard with ``is_state`` before consuming.
    max_state_blocks: Optional[int] = None

    def __post_init__(self) -> None:
        # Derived single source of truth (caller MUST NOT recompute from
        # (eb, ratio); per C01 19-2 / 19-4). Under bps≡1 the equality
        # ``tokens_per_block == eb * ratio`` is invariant.
        object.__setattr__(self, "tokens_per_block", self.eb * self.ratio)

        # Stride invariant: byte stride of the underlying uint8 view MUST
        # match block_stride_bytes. Drift here means a ``_pool_view_3d_fp8``
        # rebuild silently lost the TMA pad — exactly the regression class
        # T09's TMA fixture catches (per C02 47-5).
        if self.base_3d is not None and self.block_stride_bytes > 0:
            stride_bytes = self.base_3d.stride(0) * self.base_3d.element_size()
            assert stride_bytes == self.block_stride_bytes, (
                f"PoolHandle stride drift: base_3d.stride(0)*element_size()="
                f"{stride_bytes} != block_stride_bytes={self.block_stride_bytes}"
            )

        # INDEXER_KV (132B) is NOT TMA-padded; padding would corrupt the
        # striped data+scale layout (per B04 18-4).
        if (
            self.region_id == REGION_INDEXER_KV
            and self.block_stride_bytes > 0
            and self.entry_logical_bytes > 0
        ):
            assert (
                self.block_stride_bytes == self.eb * self.entry_logical_bytes
            ), (
                f"INDEXER_KV must not introduce TMA padding: "
                f"block_stride_bytes={self.block_stride_bytes} != "
                f"eb*entry_bytes={self.eb * self.entry_logical_bytes}"
            )


def _adhoc_kv_handle(eb: int, ratio: int = 1, is_state: bool = False) -> PoolHandle:
    """Degenerate PoolHandle for legacy callers that only know ``eb``.

    Used by Path B and by the ``compute_kv_pool_slot_mapping`` /
    ``cp_kv_slot_mapping`` shims when no descriptor is available.
    ``max_state_blocks`` stays None; this handle cannot serve cyclic
    addressing. Scheduled for removal in M06 Step 7 once M01/M05 land.
    """
    return PoolHandle(
        region_id=REGION_DEFAULT,
        eb=int(eb),
        entry_logical_bytes=0,
        entry_dtype=torch.float32,
        entry_elements=0,
        scale_bytes=0,
        has_scale=False,
        base_2d=None,
        base_3d=None,
        block_stride_bytes=0,
        is_state=bool(is_state),
        ratio=int(ratio),
    )


def _adhoc_state_handle(eb: int, max_state_blocks: Optional[int] = None) -> PoolHandle:
    """Degenerate STATE PoolHandle (ratio=1, is_state=True)."""
    return PoolHandle(
        region_id=REGION_DEFAULT,
        eb=int(eb),
        entry_logical_bytes=0,
        entry_dtype=torch.float32,
        entry_elements=0,
        scale_bytes=0,
        has_scale=False,
        base_2d=None,
        base_3d=None,
        block_stride_bytes=0,
        is_state=True,
        ratio=1,
        max_state_blocks=max_state_blocks,
    )


def _descriptor_for_region(
    region_descs: Any, region_id: int
) -> Optional[Any]:
    """Find ``PyKVCacheRegionDesc`` matching ``region_id`` (by ``region_name``)."""
    if region_descs is None:
        return None
    try:
        n = len(region_descs)
    except TypeError:
        return None
    for i in range(n):
        d = region_descs[i]
        try:
            if int(getattr(d, "region_name", -1)) == int(region_id):
                return d
        except (TypeError, ValueError):
            continue
    return None


def _build_pool_views_from_raw(
    raw: torch.Tensor, eb: int, entry_logical_bytes: int
):
    """Build (base_2d, base_3d) from raw [num_blocks, stride_bytes_or_elems].

    Mirrors today's ``AttentionFP8._pool_view`` / ``_pool_view_3d_fp8``
    branching:
      * FP8 (uint8 stride > useful bytes): base_2d=None (TMA pad blocks
        flat slice), base_3d=as_strided 3D view.
      * Non-FP8: base_2d=flat [num_slots, vec_dim], base_3d=None.
    """
    if raw is None or raw.numel() == 0 or raw.dim() != 2:
        return None, None
    num_blocks = int(raw.shape[0])
    stride_bytes = int(raw.shape[1]) * int(raw.element_size())
    useful_bytes = int(eb) * int(entry_logical_bytes)
    if useful_bytes <= 0 or stride_bytes < useful_bytes:
        return None, None
    raw_u8 = raw.view(torch.uint8) if raw.dtype != torch.uint8 else raw
    base_3d = raw_u8.as_strided(
        (num_blocks, int(eb), int(entry_logical_bytes)),
        (stride_bytes, int(entry_logical_bytes), 1),
    )
    if raw.dtype == torch.uint8 and stride_bytes > useful_bytes:
        # FP8 TMA-padded: flat 2D view non-viewable.
        return None, base_3d
    # Non-FP8 (or stride == useful_bytes): also expose a flat 2D form.
    base_2d_u8 = raw_u8[:, :useful_bytes]
    try:
        base_2d = base_2d_u8.view(raw.dtype).view(-1, int(eb) * int(raw.shape[1]) // int(eb))
    except RuntimeError:
        base_2d = None
    return base_2d, base_3d


def make_pool_handle(
    kv_cache: Any,
    layer_id: int,
    region_id: int,
    region_descs: Optional[Any] = None,
) -> Optional[PoolHandle]:
    """Resolve a ``PoolHandle`` for ``(layer_id, region_id)``.

    * **Path A** (descriptor-driven): when ``region_descs`` is provided
      and contains an entry for ``region_id``, take ``eb`` / stride /
      TMA-pad / ``max_state_blocks`` directly from
      ``PyKVCacheRegionDesc`` (M05 §3.1; binding ships
      ``PyKVCacheRegionDesc.max_state_blocks`` per F3 follow-up). This is
      the only path that populates ``max_state_blocks`` and is therefore
      the only path that can serve cyclic STATE-pool addressing.

    * **Path B** (legacy fallback): when ``region_descs`` is None /
      empty / has no entry for this region, fall back to runtime
      arithmetic (``eb = stride_bytes // bytes_per_entry``) — exactly
      today's behaviour in ``AttentionFP8._pool_view*``. ``max_state_blocks``
      is left None.

    Returns ``None`` when the layer does not participate in the region
    (today's polymorphic-probe RuntimeError early-out).
    """
    if kv_cache is None:
        return None

    is_state = int(region_id) in _STATE_REGIONS
    ratio = _RATIO_BY_REGION.get(int(region_id), 1)

    # ---- Path A: descriptor-driven --------------------------------------------
    desc = _descriptor_for_region(region_descs, region_id)
    if desc is not None:
        try:
            raw = kv_cache.get_raw_pool_tensor(int(layer_id), int(region_id))
        except (RuntimeError, AttributeError):
            raw = None
        eb = int(getattr(desc, "entries_per_block", 0))
        entry_logical_bytes = int(getattr(desc, "bytes_per_entry", 0))
        block_stride_bytes = int(getattr(desc, "kv_block_stride_bytes", 0))
        scale_bytes_stride = int(getattr(desc, "kv_scale_stride_bytes", 0))
        is_state_desc = bool(getattr(desc, "is_state_pool", is_state))
        max_state_blocks_raw = int(getattr(desc, "max_state_blocks", 0))
        max_state_blocks = (
            int(max_state_blocks_raw)
            if is_state_desc and max_state_blocks_raw > 0
            else None
        )
        bps = int(getattr(desc, "kernel_blocks_per_kv_block", 1)) or 1

        base_2d, base_3d = _build_pool_views_from_raw(raw, eb, entry_logical_bytes)
        if base_3d is None and base_2d is None and (raw is None or raw.numel() == 0):
            return None
        entry_dtype = raw.dtype if raw is not None else torch.uint8
        entry_elements = (
            int(entry_logical_bytes // raw.element_size())
            if raw is not None and raw.element_size() > 0
            else 0
        )
        return PoolHandle(
            region_id=int(region_id),
            eb=eb,
            entry_logical_bytes=entry_logical_bytes,
            entry_dtype=entry_dtype,
            entry_elements=entry_elements,
            scale_bytes=scale_bytes_stride,
            has_scale=scale_bytes_stride > 0,
            base_2d=base_2d,
            base_3d=base_3d,
            block_stride_bytes=block_stride_bytes,
            is_state=is_state_desc,
            ratio=ratio,
            bps=bps,
            max_state_blocks=max_state_blocks,
        )

    # ---- Path B: legacy fallback ----------------------------------------------
    # We need a vec_dim hint to compute eb. Modern callers pass it through the
    # AttentionFP8._pool_handle wrapper which already knows _pool_spec; bare
    # callers must use Path A.
    try:
        raw = kv_cache.get_raw_pool_tensor(int(layer_id), int(region_id))
    except (RuntimeError, AttributeError):
        return None
    if raw is None or raw.numel() == 0 or raw.dim() != 2:
        return None
    stride_bytes = int(raw.shape[1]) * int(raw.element_size())
    # Without a vec_dim hint we cannot derive eb on Path B; expose only the
    # raw stride so the caller can decide. Returning a partially-populated
    # handle would silently break downstream arithmetic, so refuse.
    return PoolHandle(
        region_id=int(region_id),
        eb=0,
        entry_logical_bytes=0,
        entry_dtype=raw.dtype,
        entry_elements=0,
        scale_bytes=0,
        has_scale=False,
        base_2d=None,
        base_3d=None,
        block_stride_bytes=stride_bytes,
        is_state=is_state,
        ratio=ratio,
    )


def make_pool_handle_from_raw(
    raw: Optional[torch.Tensor],
    region_id: int,
    vec_dtype: torch.dtype,
    vec_dim: int,
    *,
    max_state_blocks: Optional[int] = None,
) -> Optional[PoolHandle]:
    """Path B (vec_dim-hinted) construction used by ``AttentionFP8._pool_handle``.

    Mirrors today's ``_pool_view`` / ``_pool_view_3d_fp8`` derivation so a
    bit-equal handle can be built without a descriptor when M05 wiring is
    absent (warmup / pre-F02 / non-DSV4).
    """
    if raw is None or raw.numel() == 0 or raw.dim() != 2:
        return None
    is_state = int(region_id) in _STATE_REGIONS
    ratio = _RATIO_BY_REGION.get(int(region_id), 1)
    stride_bytes = int(raw.shape[1]) * int(raw.element_size())
    entry_logical_bytes = int(vec_dim) * int(vec_dtype.itemsize)
    if entry_logical_bytes <= 0 or stride_bytes < entry_logical_bytes:
        return None
    eb = stride_bytes // entry_logical_bytes
    base_2d, base_3d = _build_pool_views_from_raw(raw, eb, entry_logical_bytes)
    entry_dtype = raw.dtype
    entry_elements = (
        int(entry_logical_bytes // raw.element_size())
        if raw.element_size() > 0
        else int(vec_dim)
    )
    return PoolHandle(
        region_id=int(region_id),
        eb=int(eb),
        entry_logical_bytes=int(entry_logical_bytes),
        entry_dtype=entry_dtype,
        entry_elements=int(entry_elements),
        scale_bytes=0,
        has_scale=False,
        base_2d=base_2d,
        base_3d=base_3d,
        block_stride_bytes=int(stride_bytes),
        is_state=bool(is_state),
        ratio=int(ratio),
        max_state_blocks=max_state_blocks,
    )


def cyclic_state_pool_mapper(
    kv_cache: Any,
    layer_id: int,
    at_state: int,
    region_descs: Optional[Any] = None,
) -> int:
    """Canonical accessor: cyclic ring depth for a STATE pool.

    Source of truth: ``PyKVCacheRegionDesc.max_state_blocks`` (M05 §3.1
    field added by F3 follow-up; pybind binding present at
    ``OpDefs.cc:238``). Closes the Panel B P0 plumbing gap previously
    cited against a non-existent M06 symbol.

    Called by M09 §4.1 (``allocate_decode_metadata_fp8``) to populate
    ``DSv4DecodeAttnMetadataFP8.max_state_blocks[at_state]`` once per
    metadata allocation. Host-side scalar; no device work.
    """
    handle = make_pool_handle(
        kv_cache, layer_id, at_state, region_descs=region_descs
    )
    if handle is None or not handle.is_state:
        raise RuntimeError(
            f"cyclic_state_pool_mapper: (layer={layer_id}, at_state={at_state}) "
            f"is not a STATE-pool region in this KVCache layout"
        )
    if handle.max_state_blocks is None:
        raise RuntimeError(
            f"cyclic_state_pool_mapper: handle.max_state_blocks is None "
            f"(legacy Path B handle); F3 must surface "
            f"PyKVCacheRegionDesc.max_state_blocks for region {at_state}"
        )
    return int(handle.max_state_blocks)


def pool_view_or_raise(handle: Optional[PoolHandle], who: str) -> PoolHandle:
    """Centralised None-guard for bind/scatter call sites.

    ``make_pool_handle`` may legitimately return None (layer doesn't
    participate). Callers that already performed the ``gid_for(...) >= 0``
    precondition use this to assert. Per M06 §3.1.1.
    """
    if handle is None:
        raise RuntimeError(
            f"{who}: PoolHandle is None — region not owned by this layer; "
            f"caller forgot the gid_for / region-owned early-out."
        )
    return handle
