"""Direct views of framework pools, with no private cache or allocator.

NativeCacheBridge's coordinate/planar copy primitives are reused, while its
legacy logical-cache construction, import and export are never called.
"""

import hashlib
import logging
import math
import os
from types import SimpleNamespace
from typing import Callable, NamedTuple

import torch

from rtp_llm.models_py.modules.dsv41.kv_cache_utils import as_attention_inputs_by_tag
from rtp_llm.models_py.modules.dsv41.ppu.kv_codec import PoolDescriptor
from rtp_llm.models_py.modules.dsv41.ppu.native_cache import (
    NativeCacheBridge,
    _FULL,
    _Pool,
    _WIDTH,
)
from rtp_llm.models_py.modules.dsv41.ppu.optimization_config import (
    enabled as optimization_enabled,
)
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import PrefillPolicy
from rtp_llm.models_py.modules.dsv41.ppu.row_metadata import (
    is_plain_tensor as _local_swa_raw_tensor,
    tensor_metadata_key,
)


_LOGGER = logging.getLogger(__name__)
CP8_ROW_PREP_ENV = "RTP_V41_PPU_CP8_PREFILL_ROW_PREP"
CP8_GATHER_RECEIVE_ENV = "RTP_V41_PPU_CP8_REUSE_GATHER_RECEIVE"
LOCAL_PREFILL_SOURCE_CACHE_ENV = "RTP_V41_PPU_LOCAL_PREFILL_SOURCE_CACHE"
LOCAL_PREFILL_SWA_SEGMENTS_ENV = "RTP_V41_PPU_LOCAL_PREFILL_SWA_SEGMENTS"
LOCAL_PREFILL_LAZY_PRODUCER_SLOTS_ENV = "RTP_V41_PPU_LOCAL_PREFILL_LAZY_PRODUCER_SLOTS"
LOCAL_PREFILL_SG_ALIGN_ENV = "RTP_V41_PPU_LOCAL_PREFILL_SG_ALIGN"


def _source_tensor_key(tensor):
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        return type(tensor), id(tensor)
    return tensor_metadata_key(tensor)


class _StagedSwaSegments(NamedTuple):
    prefix: int
    tail: torch.Tensor
    fresh: torch.Tensor


class _LocalSwaStageRanges(NamedTuple):
    """Call-local capability; the caller supplies a creation-stamped guard."""
    cache: object
    ranges: tuple[tuple[int, int, int], ...]
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    input_keys: tuple
    prefixes: tuple[int, ...]
    totals: tuple[int, ...]
    retain_prefix_tail: bool
    bounded_starts: tuple[int, ...] | None
    pool: object
    pool_key: tuple
    owners: tuple
    guard: Callable[[], bool]


class _LocalSwaWriteGeometry(NamedTuple):
    rank: int
    full_entries: int
    slice_elements: int


def _latest_csa_indices(blocks, offsets, positions):
    """Stable maximum raw position per real physical slot; ties keep first."""
    latest = {}
    for index, (block, offset, position) in enumerate(zip(blocks, offsets, positions)):
        key = block, offset
        previous = latest.get(key)
        if previous is None or position > positions[previous]:
            latest[key] = index
    return sorted(latest.values())


def _verified_cache_table(
    cache,
    pool,
    *,
    source,
    field,
    mirrors,
    invalid,
    invalid_key,
    stats,
    counter,
    mirror_keys=None
):
    """Validate a constructor-owned table and reuse one exact host mirror.

    Pool-specific bindings are checked by the caller. A changed/rebound table
    or mirror is permanently declined for this forward, including mutations
    subsequently changed back. Read and write users share the same proof.
    """
    tag = pool.framework_tag
    if source is None or invalid_key in invalid:
        return None
    meta, owners, birth = source

    def current():
        live = cache.inputs.get(tag)
        values = (getattr(live, field, None), getattr(live, field + "_device", None))
        return (
            live is meta
            and all(_local_swa_raw_tensor(value) for value in (*values, pool.table))
            and tuple(_source_tensor_key(value) for value in values) == birth
            and tuple(_source_tensor_key(value) for value in owners) == birth
            and _source_tensor_key(pool.table) == birth[1]
        )

    if not current():
        invalid.add(invalid_key)
        return None
    host, device = owners
    if (
        host.device.type != "cpu"
        or host.ndim != 2
        or device.ndim != 2
        or host.dtype not in (torch.int32, torch.int64)
        or device.dtype != host.dtype
        or host.shape != device.shape
        or device.device != pool.tensor.device
        or any(stride <= 0 for stride in (*host.stride(), *device.stride()))
        or host.shape[0] < len(cache.lengths)
        or torch._C._is_alias_of(pool.tensor, device)
    ):
        return None
    saved = mirrors.get(tag)
    if saved is not None:
        mirror, stamp = saved if mirror_keys is None else (saved, mirror_keys.get(tag))
        if _source_tensor_key(mirror) != stamp:
            invalid.add(invalid_key)
            return None
        return mirror
    mirror = device.detach().to(device="cpu", copy=True)
    stats[counter] += 1
    if "mirror_bytes" in stats:
        stats["mirror_bytes"] += mirror.numel() * mirror.element_size()
    if not current() or not torch.equal(host, mirror):
        invalid.add(invalid_key)
        return None
    stamp = _source_tensor_key(mirror)
    if mirror_keys is None:
        mirrors[tag] = mirror, stamp
    else:
        mirrors[tag], mirror_keys[tag] = mirror, stamp
    return mirror


def _local_pool_binding_valid(key, pool, binding, layout, invalid):
    if binding is None or key in invalid:
        return False
    tensors = pool.tensor, pool.handle.kv_cache_base
    if (
        pool is not binding[0]
        or pool.handle is not binding[1]
        or not all(_local_swa_raw_tensor(value) for value in tensors)
        or tuple(_source_tensor_key(value)[:-1] for value in tensors) != binding[3]
        or layout != binding[4:]
    ):
        invalid.add(key)
        return False
    return True


class SharedNativeCache(NativeCacheBridge):
    """Per-forward resources. Slots are (request row, absolute token position)."""

    def __init__(
        self, config, geometry, native, inputs, lengths, *, device, prefill=False,
        defer_producer_slots=False,
    ):
        self.config, self.native, self.device = config, native, device
        self._is_prefill = prefill
        self.policy = PrefillPolicy.from_config(config)
        self.geometry = geometry
        self.logical = SimpleNamespace(geometry=geometry)
        self.inputs = as_attention_inputs_by_tag(inputs, native)
        self.lengths = tuple(map(int, lengths))
        self.pools = {}
        self._fresh_swa = {}
        self._swa_fresh_lengths = None
        self._source_reads = {}
        self._local_source_reads = {}
        self.shared_prefix_stripes = None
        self.prefill_plans = None
        self.gather_receive = None
        self.placements = {}
        self.pool_gathers = []
        self.cp = dict(config.get("_ppu_cp_cache", {}))
        self.sharded = bool(self.cp.get("sliced_prefill", False))
        self.writes = []
        self._local_swa_write_sources = {}
        self._local_swa_write_mirrors = {}
        self._local_swa_write_mirror_keys = {}
        self._local_swa_write_invalid = set()
        self._local_swa_native_write_stats = dict(mirrors=0, launches=0, requests=0, fallbacks=0)
        self._local_swa_read_enabled = prefill and not self.sharded
        self._local_swa_read_nonce = object()
        self._local_swa_read_rows = None
        self._local_swa_read_mirrors = {}
        self._local_swa_read_plans = {}
        self._local_swa_native_read_stats = dict(
            enabled=self._local_swa_read_enabled, launches=0, requests=0, direct=0,
            packet=0, plan_uploads=0, cache_hits=0, fallbacks=0, closed=False,
        )
        keys = [(i, "swa_kv") for i in range(geometry.num_layers)]
        for i in geometry.kv_source_layer_ids:
            keys.extend([(i, "global_kv_2" if geometry.ratio(i) == 2 else "global_kv_1"), (i, "indexer_kv")])
            if geometry.ratio(i) == 2:
                keys.append((i, "csa_state"))
        for owner, tag in keys:
            framework_tag = self.policy.framework_tag(owner, tag, geometry)
            if framework_tag not in self.inputs:
                raise ValueError(f"missing native tagged input {framework_tag}")
            handle = native.get_layer_cache(owner, framework_tag)
            tensor = handle.kv_cache_base
            if tensor.ndim < 2 or not tensor.is_contiguous():
                raise ValueError(f"invalid native pool layout {(owner, tag)}")
            tensor = tensor.view(tensor.shape[0], math.prod(tensor.shape[1:]))
            entries = int(native.get_entries_per_block(framework_tag))
            span = int(native.get_kernel_seq_size_per_block(framework_tag) if tag in _FULL else native.get_seq_size_per_block(framework_tag))
            dtype = torch.float32 if tag == "csa_state" else torch.uint8
            if entries <= 0 or span <= 0 or tensor.dtype != dtype or (not self.sharded and tensor.shape[1] < entries * _WIDTH[tag]):
                raise ValueError(f"invalid native pool geometry {(owner, tag)}")
            if tag == "swa_kv" and entries < geometry.window:
                raise ValueError("native SWA ring cannot retain the complete window")
            if tag == "csa_state" and entries * (int(self.cp["size"]) if self.sharded else 1) < 2:
                raise ValueError("native CSA ring cannot retain an unfinished pair")
            meta = self.inputs[framework_tag]
            field = "kv_cache_kernel_block_id" if tag in _FULL else "kv_cache_block_id"
            table = getattr(meta, field + "_device", None)
            if table is None or not table.numel():
                table = getattr(meta, field, None)
            if table is None or table.ndim != 2:
                raise ValueError(f"missing {framework_tag} block table")
            self.pools[owner, tag] = _Pool(framework_tag, handle, tensor, table.to(tensor.device), entries, span)
            if self.sharded:
                from rtp_llm.models_py.modules.dsv41.ppu.cp_sharding import PoolPlacement

                self.placements[owner, tag] = PoolPlacement.create(
                    self.cp, (owner, tag), self.pools[owner, tag], native, meta, geometry,
                )
        self._init_local_source_native_reads()
        full = [pool for key, pool in self.pools.items() if key[1] in _FULL]
        self._local_sg_source_enabled = (
            os.environ.get(LOCAL_PREFILL_SG_ALIGN_ENV, "0") == "1"
            and self._local_source_cache_eligible() and bool(full)
            and self._local_source_native_read_enabled
            and self._local_source_native_read_device(full[0])
        )
        self._local_sg_source_stats = dict(
            enabled=self._local_sg_source_enabled, split_reads=0, derived_splits=0,
            hits=0, packed_rebuilds=0, closed=False,
        )
        self._init_local_csa_native_writes()
        if self._is_prefill and not self.sharded:
            for (owner, tag), pool in self.pools.items():
                if tag != "swa_kv" or pool.framework_tag in self._local_swa_write_sources:
                    continue
                meta = self.inputs[pool.framework_tag]
                owners = (getattr(meta, "kv_cache_block_id", None),
                          getattr(meta, "kv_cache_block_id_device", None))
                if not all(_local_swa_raw_tensor(value) for value in owners):
                    continue
                self._local_swa_write_sources[pool.framework_tag] = (
                    meta, owners, tuple(_source_tensor_key(value) for value in owners),
                )
        self._local_producer_slot_stats = dict(
            enabled=optimization_enabled(LOCAL_PREFILL_LAZY_PRODUCER_SLOTS_ENV),
            deferred=False, materializations=0, closed=False,
        )
        self._slots = None
        if not defer_producer_slots and not (
            self.cp8_row_prep_enabled()
            and optimization_enabled("RTP_V41_PPU_CP8_LAZY_PRODUCER_SLOTS")
        ) and not self._local_lazy_producer_slots_enabled():
            self._slots = self._build_producer_slots()
            self._local_producer_slot_stats["materializations"] += 1
        self._local_producer_slot_stats["deferred"] = self._slots is None

    def _local_lazy_producer_slots_enabled(self):
        if (not self._local_producer_slot_stats["enabled"]
                or not self._local_source_native_read_business() or torch.is_grad_enabled()
                or any(length < 0 for length in self.lengths)):
            return False
        full = [(key, pool) for key, pool in self.pools.items() if key[1] in _FULL]
        if not full or not self._local_source_native_read_device(full[0][1]):
            return False
        device = full[0][1].tensor.device
        for key, pool in self.pools.items():
            if pool.tensor.device != device:
                return False
            if key[1] in _FULL:
                if self.geometry.ratio(key[0]) not in (1, 2) or not self._local_source_native_binding_valid(key, pool):
                    return False
                source = self._local_source_native_read_sources.get(pool.framework_tag)
            elif key[1] == "swa_kv":
                source = self._local_swa_write_sources.get(pool.framework_tag)
            else:
                continue
            if source is None:
                return False
            meta, owners, birth = source
            host, target = owners
            if (self.inputs.get(pool.framework_tag) is not meta
                    or not all(_local_swa_raw_tensor(value) for value in (*owners, pool.table))
                    or tuple(_source_tensor_key(value) for value in owners) != birth
                    or _source_tensor_key(pool.table) != birth[1]
                    or host.device.type != "cpu" or host.ndim != 2 or target.ndim != 2
                    or host.dtype not in (torch.int32, torch.int64) or target.dtype != host.dtype
                    or host.shape != target.shape or host.shape[0] < len(self.lengths)
                    or target.device != device
                    or any(stride <= 0 for stride in (*host.stride(), *target.stride()))):
                return False
        return True

    def _build_producer_slots(self):
        slots = {}
        for rid, length in enumerate(self.lengths):
            layer_slots = {}
            for source in self.geometry.kv_source_layer_ids:
                ratio = self.geometry.ratio(source)
                positions = torch.arange(length // ratio, device=self.device) * ratio
                layer_slots[source] = self.slots_at_positions(
                    source, "indexer_kv", rid, positions,
                )
            slots[rid] = SimpleNamespace(
                global_slots=layer_slots, indexer_slots=layer_slots,
            )
        return slots

    @property
    def slots(self):
        if self._slots is None:
            self._slots = self._build_producer_slots()
            stats = getattr(self, "_local_producer_slot_stats", None)
            if stats is not None:
                stats["materializations"] += 1
        return self._slots

    def owned_positions(self, key, positions):
        positions = positions.to(device=self.device, dtype=torch.int64)
        placement = self.placements.get(key)
        return placement.owned(key, positions) if placement is not None else torch.ones_like(positions, dtype=torch.bool)

    def _coordinates(self, key, request_row, positions):
        placement = self.placements.get(key)
        if placement is None:
            return super()._coordinates(key, request_row, positions)
        return placement.coordinates(key, self.pools[key], request_row, positions.to(self.device).long())

    def _copy_rows(self, key, request_row, positions, rows=None):
        if rows is not None and key[1] in _FULL:
            self._invalidate_local_source_reads()
        positions = positions.to(device=self.device, dtype=torch.int64)
        placement = self.placements.get(key)
        if placement is None:
            return super()._copy_rows(key, request_row, positions, rows)
        if rows is None:
            if not positions.numel():
                return self.pools[key].tensor.new_empty((0, _WIDTH[key[1]]))
            self.pool_gathers.append({"layer": key[0], "tag": key[1], "request": int(request_row),
                                      "logical_rows": positions.numel(), "cp_size": placement.size})
            return placement.gather(key, self.pools[key], request_row, positions, self.geometry)
        owned = placement.owned(key, positions)
        positions, rows = positions[owned], rows[owned]
        if key[1] not in _FULL:
            placement.write_ring(key, self.pools[key], request_row, positions, rows)
            return None
        return super()._copy_rows(key, request_row, positions, rows)

    def slots_at_positions(self, layer, tag, request, positions):
        positions = positions.to(device=self.device, dtype=torch.int64)
        return torch.stack((torch.full_like(positions, int(request)), positions), -1)

    def _prefill_ring_keep(self, pool, request, positions, live_start):
        columns = positions // pool.span
        if positions.numel() and (int(positions.min()) < 0 or int(columns.max()) >= pool.table.shape[1]):
            raise ValueError(f"native Prefill {pool.framework_tag} position outside block table")
        missing = pool.table[request, columns] == -1
        if bool((missing & (positions >= live_start)).any()):
            raise ValueError(f"native Prefill {pool.framework_tag} is missing a live ring block")
        return ~missing

    def insert_rows(self, layer, tag, slots, rows):
        for rid in slots[:, 0].unique().tolist():
            mask = slots[:, 0] == rid
            self.insert_request_rows(layer, tag, rid, slots[mask, 1], rows[mask])

    def cp8_row_prep_enabled(self):
        return (
            self._is_prefill
            and self.sharded
            and self.cp["size"] == 8
            and os.environ.get(CP8_ROW_PREP_ENV, "1") == "1"
        )

    def insert_request_rows(self, layer, tag, request, positions, rows, *, start=None,
                            proof=None, prefill_layout=None):
        """Select a qualified local/CP store, otherwise use logical row copying."""
        if not positions.numel():
            return
        key = layer, tag
        if start is not None and proof is not None and tag in _FULL and self.try_insert_local_source_rows(
            layer, tag, request, start, rows, proof=proof, prefill_layout=prefill_layout,
        ):
            return
        if start is not None and self._try_store_prefill_range(key, request, start, rows):
            return
        owned = self.owned_positions(key, positions)
        write = {
            "layer": layer,
            "tag": tag,
            "request": int(request),
            "rows": positions.numel(),
            "owned_rows": int(owned.sum()),
        }
        if tag == "csa_state":
            if getattr(self, "_is_prefill", False):
                length = self.lengths[request]
                owned &= self._prefill_ring_keep(
                    self.pools[key], request, positions, length - length % 2,
                )
            positions, rows = positions[owned], rows[owned]
            blocks, offsets = self._coordinates(key, request, positions)
            keep = _latest_csa_indices(
                blocks.cpu().tolist(), offsets.cpu().tolist(), positions.cpu().tolist()
            )
            selected = torch.tensor(keep, device=positions.device, dtype=torch.long)
            # Deduplicate physical destinations even when table columns alias.
            positions, rows = positions[selected], rows[selected]
            write["physical_rows"] = len(keep)
        self._copy_rows(key, request, positions, rows)
        self.writes.append(write)

    def try_insert_csa_fields(self, layer, request, start, values, scores, *, proof=None, prefill_layout=None):
        """Write separate producer fields directly into the existing CSA row."""
        if not (
            values.shape == scores.shape
            and values.dtype in (torch.bfloat16, torch.float32)
            and scores.dtype in (torch.bfloat16, torch.float32)
            and scores.is_contiguous()
            and scores.device == values.device
        ):
            return False
        if proof is not None and self.try_insert_local_csa_fields(
            layer, request, start, values, scores, proof=proof, prefill_layout=prefill_layout,
        ):
            return True
        return self._try_store_prefill_range((layer, "csa_state"), request, start, values, scores=scores)

    def _try_store_prefill_range(self, key, request, start, rows, *, scores=None):
        if key[1] in _FULL:
            # Native direct stores do not update Tensor version counters.
            # Clear before launch, including writes through aliased requests.
            self._invalidate_local_source_reads()
        if (
            not self.cp8_row_prep_enabled()
            or start < 0
            or not rows.is_cuda
            or not rows.is_contiguous()
        ):
            return False
        layer, tag = key
        pool, placement = self.pools[key], self.placements[key]
        full = tag in _FULL
        field = "kv_cache_kernel_block_id" if full else "kv_cache_block_id"
        host = getattr(self.inputs[pool.framework_tag], field, None)
        if host is None or host.device.type != "cpu" or host.shape != pool.table.shape:
            return False

        from rtp_llm.models_py.modules.dsv41.ppu.graph_cache_kernels import (
            try_write_cp_prefill_rows,
            try_write_prefill_csa,
        )

        write = {"layer": layer, "tag": tag, "request": request, "rows": rows.shape[0]}
        if full:
            owned = try_write_cp_prefill_rows(
                pool, placement, host, request, start, rows, planar=tag == "indexer_kv",
            )
            if owned is None:
                return False
        else:
            counts = try_write_prefill_csa(
                pool, placement, host, request, start, self.lengths[request], rows, scores=scores,
            )
            if counts is None:
                return False
            owned, write["physical_rows"] = counts
        write["owned_rows"] = owned
        self.writes.append(write)
        return True

    def request_length(self, request):
        return self.lengths[int(request)]

    def descriptors(self, layer):
        return {
            tag: PoolDescriptor(entry_bytes=_WIDTH[tag], block_entries=pool.entries, window=self.geometry.window,
                                kind="fp8_e4m3_ue8m0" if tag == "swa_kv" else "fp4_e2m1_ue8m0" if tag == "indexer_kv" else "fp4_e2m1_e4m3")
            for (owner, tag), pool in self.pools.items() if owner == layer
        }

    def _local_source_cache_eligible(self):
        return (
            os.environ.get(LOCAL_PREFILL_SOURCE_CACHE_ENV, "0") == "1"
            and type(self) is SharedNativeCache
            and self._is_prefill and not self.sharded
            and self.cp.get("size", 1) == 1
            # Unsharded CP8 also freezes placement size1; require the actual
            # local TP role rather than inferring topology from that field.
            and os.environ.get("TP_SIZE") == "1"
            and os.environ.get("PREFILL_CP_SIZE", "1") == "1"
            and any(self.lengths)
            and not any(getattr(meta, "is_cuda_graph", False) for meta in self.inputs.values())
        )

    def _invalidate_local_source_reads(self):
        if getattr(self, "_local_source_reads", None):
            # A main/indexer pool can alias another source or request. Do not
            # invalidate only the nominal writer's (source, request) pair.
            self._source_reads.clear()
            self._local_source_reads.clear()

    def _local_source_native_read_business(self):
        # Independent of the payload-cache switch; keep its existing business
        # scope while every miss still reads current native pool bytes.
        return (
            type(self) is SharedNativeCache and self._is_prefill and not self.sharded
            and self.cp.get("size", 1) == 1 and os.environ.get("TP_SIZE") == "1"
            and os.environ.get("PREFILL_CP_SIZE", "1") == "1" and any(self.lengths)
            and not any(getattr(meta, "is_cuda_graph", False) for meta in self.inputs.values())
        )

    def _local_source_native_read_device(self, pool):
        from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu

        target = torch.device(self.device)
        return (
            pool.tensor.is_cuda and not torch.is_grad_enabled() and is_ppu()
            and target.type == pool.tensor.device.type
            and target.index in (None, pool.tensor.device.index)
            and torch.cuda.get_device_capability(pool.tensor.device) == (8, 9)
            and torch.cuda.current_device() == pool.tensor.device.index
            and not torch.cuda.is_current_stream_capturing()
            and torch.cuda.current_stream(pool.tensor.device) == torch.cuda.default_stream(pool.tensor.device)
        )

    def _init_local_source_native_reads(self):
        # One constructor-owned binding serves reads and writes. No opposite
        # operation or rollout flag is needed to establish table provenance.
        eligible = self._local_source_native_read_business()
        self._local_source_native_read_enabled = eligible
        self._local_source_native_write_enabled = eligible
        self._local_source_native_write_stats = dict(
            enabled=self._local_source_native_write_enabled, writes=0, rows=0,
            launches=0, bytes=0, fallbacks=0, fallback_reasons={}, geometries=[], closed=False,
        )
        self._local_source_native_read_nonce = object()
        self._local_source_native_read_birth_nonce = self._local_source_native_read_nonce
        self._local_source_native_read_sources = {}
        self._local_source_native_read_bindings = {}
        self._local_source_native_read_mirrors = {}
        self._local_source_native_read_ids = {}
        self._local_source_native_read_plans = {}
        self._local_source_native_read_invalid = set()
        self._local_source_native_read_invalid_pools = set()
        self._local_source_native_read_stats = dict(
            enabled=self._local_source_native_read_enabled, reads=0, mirrors=0,
            mirror_bytes=0, ids_uploads=0, ids_bytes=0, ids_hits=0,
            plan_uploads=0, plan_bytes=0, plan_hits=0, packet_calls=0,
            restore_calls=0, packet_bytes=0, output_bytes=0, geometries=[],
            fallbacks=0, fallback_reasons={}, closed=False,
        )
        if not self._local_source_native_read_enabled or not self._local_source_native_read_business():
            return
        for key, pool in self.pools.items():
            if key[1] not in _FULL or not self._local_source_native_read_device(pool):
                continue
            meta = self.inputs[pool.framework_tag]
            owners = (getattr(meta, "kv_cache_kernel_block_id", None),
                      getattr(meta, "kv_cache_kernel_block_id_device", None))
            if not all(_local_swa_raw_tensor(value) for value in (*owners, pool.table, pool.tensor, pool.handle.kv_cache_base)):
                continue
            self._local_source_native_read_sources.setdefault(
                pool.framework_tag, (meta, owners, tuple(_source_tensor_key(value) for value in owners)),
            )
            # Payload writes are legal before/among reads. Bind identity/layout,
            # not payload version; native writes need not bump that version.
            tensors = pool.tensor, pool.handle.kv_cache_base
            self._local_source_native_read_bindings[key] = (
                pool, pool.handle, tensors, tuple(_source_tensor_key(value)[:-1] for value in tensors),
                pool.entries, pool.span, self.geometry.ratio(key[0]), self.lengths,
            )

    def close_local_source_native_reads(self):
        """Finally-only host close; query compaction does not end FULL reads."""
        for name in ("sources", "bindings", "mirrors", "ids", "plans"):
            value = getattr(self, "_local_source_native_read_" + name, None)
            if value is not None:
                value.clear()
        stats = getattr(self, "_local_source_native_read_stats", None)
        if stats is not None:
            stats["closed"] = True
        stats = getattr(self, "_local_sg_source_stats", None)
        if stats is not None:
            stats["closed"] = True
        write_stats = getattr(self, "_local_source_native_write_stats", None)
        if write_stats is not None:
            write_stats["closed"] = True
        self.close_local_csa_native_writes()
        stats = getattr(self, "_local_producer_slot_stats", None)
        if stats is not None:
            stats["closed"] = True

    def _init_local_csa_native_writes(self):
        self._local_csa_native_write_enabled = self._local_source_native_read_business()
        self._local_csa_native_write_nonce = object()
        self._local_csa_native_write_birth_nonce = self._local_csa_native_write_nonce
        self._local_csa_native_write_sources = {}
        self._local_csa_native_write_bindings = {}
        self._local_csa_native_write_mirrors = {}
        self._local_csa_native_write_invalid = set()
        self._local_csa_native_write_stats = dict(
            enabled=self._local_csa_native_write_enabled, closed=False, writes=0,
            rows=0, launches=0, bytes=0, table_mirrors=0, metadata_uploads=0,
            fallbacks=0, fallback_reasons={}, geometries=[],
        )
        if not self._local_csa_native_write_enabled or not self._local_source_native_read_business():
            return
        for key, pool in self.pools.items():
            if key[1] != "csa_state" or not self._local_source_native_read_device(pool):
                continue
            meta = self.inputs[pool.framework_tag]
            owners = (getattr(meta, "kv_cache_block_id", None), getattr(meta, "kv_cache_block_id_device", None))
            tensors = pool.tensor, pool.handle.kv_cache_base
            if not all(_local_swa_raw_tensor(value) for value in (*owners, pool.table, *tensors)):
                continue
            self._local_csa_native_write_sources.setdefault(
                pool.framework_tag, (meta, owners, tuple(_source_tensor_key(value) for value in owners)),
            )
            self._local_csa_native_write_bindings[key] = (
                pool, pool.handle, tensors, tuple(_source_tensor_key(value)[:-1] for value in tensors),
                pool.entries, pool.span, self.lengths,
            )

    def close_local_csa_native_writes(self):
        """Pure-host, finally-only close; FULL query compaction is unrelated."""
        for name in ("sources", "bindings", "mirrors"):
            value = getattr(self, "_local_csa_native_write_" + name, None)
            if value is not None:
                value.clear()
        stats = getattr(self, "_local_csa_native_write_stats", None)
        if stats is not None:
            stats["closed"] = True

    def _verified_local_csa_native_table(self, key, pool):
        if not _local_pool_binding_valid(
            key, pool, self._local_csa_native_write_bindings.get(key),
            (pool.entries, pool.span, self.lengths), self._local_csa_native_write_invalid,
        ):
            return None
        return _verified_cache_table(
            self, pool, source=self._local_csa_native_write_sources.get(pool.framework_tag),
            field="kv_cache_block_id", mirrors=self._local_csa_native_write_mirrors,
            invalid=self._local_csa_native_write_invalid, invalid_key=key,
            stats=self._local_csa_native_write_stats, counter="table_mirrors",
        )

    def try_insert_local_csa_fields(self, layer, request, start, values, scores, *, proof, prefill_layout):
        """Store the old FP32 cat fields with an independent CSA table birth."""
        if not getattr(self, "_local_csa_native_write_enabled", False):
            return False
        stats = self._local_csa_native_write_stats

        def decline(reason):
            stats["fallbacks"] += 1
            reasons = stats["fallback_reasons"]
            reasons[reason] = reasons.get(reason, 0) + 1
            return False

        if (stats["closed"] or not self._local_source_native_read_business() or torch.is_grad_enabled()):
            return decline("scope")
        if self._local_csa_native_write_nonce is not self._local_csa_native_write_birth_nonce:
            self.close_local_csa_native_writes()
            return decline("nonce")
        from rtp_llm.models_py.modules.dsv41.ppu.prefill import _valid_local_producer_rotary_proof

        if not _valid_local_producer_rotary_proof(proof, cache=self, layer_id=layer, prefill_layout=prefill_layout):
            return decline("proof")
        if type(request) is not int or not 0 <= request < len(proof.prefixes) or self.geometry.ratio(layer) != 2:
            return decline("request")
        pool = self.pools[layer, "csa_state"]
        if (not self._local_source_native_read_device(pool)
                or not all(_local_swa_raw_tensor(value) for value in (pool.tensor, values, scores))
                or pool.tensor.dtype != torch.float32 or pool.tensor.ndim != 2 or not pool.tensor.is_contiguous()
                or values.dtype != torch.float32 or scores.dtype != torch.float32
                or values.device != pool.tensor.device or scores.device != pool.tensor.device
                or values.ndim != 2 or values.shape[1] != 512 or scores.shape != values.shape
                or not values.is_contiguous() or not scores.is_contiguous()
                or not 0 < values.shape[0] <= 16384 or type(start) is not int
                or not proof.prefixes[request] <= start < self.lengths[request]
                or start + values.shape[0] != self.lengths[request]
                or pool.entries < 2 or pool.span <= 0 or pool.tensor.stride(0) < pool.entries * 1024):
            return decline("geometry")
        target, extent = pool.tensor.data_ptr(), pool.tensor.numel() * pool.tensor.element_size()
        for source in (values, scores, *(other.tensor for key, other in self.pools.items() if key[1] != "csa_state")):
            if source.device != pool.tensor.device:
                continue
            if not _local_swa_raw_tensor(source) or not source.is_contiguous():
                return decline("alias")
            if source.data_ptr() < target + extent and target < source.data_ptr() + source.numel() * source.element_size():
                return decline("alias")
        mirror = self._verified_local_csa_native_table((layer, "csa_state"), pool)
        end = start + values.shape[0]
        first_page, last_page = start // pool.span, (end - 1) // pool.span + 1
        if mirror is None or last_page > mirror.shape[1]:
            return decline("table")
        blocks = tuple(map(int, mirror[request, first_page:last_page].tolist()))
        physical = [block for block in blocks if block >= 0]
        if len(set(physical)) != len(physical):
            return decline("duplicate")
        plan = []
        for page, block in enumerate(blocks, first_page):
            begin, stop = max(start, page * pool.span), min(end, (page + 1) * pool.span)
            if block == -1 and stop <= self.lengths[request] - self.lengths[request] % 2:
                continue
            if not 0 <= block < pool.tensor.shape[0]:
                return decline("physical")
            for offset in range(pool.entries):
                last = stop - 1 - (stop - 1 - offset) % pool.entries
                if last >= begin:
                    plan.append((block * pool.tensor.stride(0) + offset * 1024, last - start))
        # Unique destinations and nonaliasing fields make launch ordering
        # unobservable; sort by original row to retain old selected order too.
        plan.sort(key=lambda item: item[1])
        if plan:
            try:
                from rtp_llm.models_py.modules.dsv41.ppu.graph_cache_kernels import _kernels
                kernels = _kernels()
            except ImportError:
                return decline("provider")
            indices = torch.tensor(plan, dtype=torch.int64).to(pool.tensor.device, non_blocking=True)
            stats["metadata_uploads"] += 1
            stats["launches"] += 1
            kernels.store_prefill_csa[(len(plan),)](pool.tensor, indices, values, scores, 1024, True)
        self.writes.append(dict(layer=layer, tag="csa_state", request=request,
                                rows=values.shape[0], owned_rows=values.shape[0], physical_rows=len(plan)))
        stats["writes"] += 1
        stats["rows"] += len(plan)
        stats["bytes"] += len(plan) * 1024 * 4
        stats["geometries"].append(dict(
            source=layer, request=request, input_rows=values.shape[0], selected_rows=len(plan),
            entries=pool.entries, span=pool.span, stride=pool.tensor.stride(0), blocks=pool.tensor.shape[0],
            rawstart=start, rawend=end, first=start, last=end - 1,
            first_page=first_page, written_pages=last_page - first_page, separate=True,
        ))
        return True

    def try_insert_local_source_rows(self, layer, tag, request, start, rows, *, proof, prefill_layout):
        """Bit-copy a proved local FULL suffix; no CP placement is constructed."""
        if not getattr(self, "_local_source_native_write_enabled", False):
            return False
        stats = self._local_source_native_write_stats

        def decline(reason):
            stats["fallbacks"] += 1
            reasons = stats["fallback_reasons"]
            reasons[reason] = reasons.get(reason, 0) + 1
            return False

        if (stats["closed"] or self._local_source_native_read_stats["closed"]
                or not self._local_source_native_read_business() or torch.is_grad_enabled()):
            return decline("scope")
        if self._local_source_native_read_nonce is not self._local_source_native_read_birth_nonce:
            self.close_local_source_native_reads()
            return decline("nonce")
        from rtp_llm.models_py.modules.dsv41.ppu.prefill import _valid_local_producer_rotary_proof

        if not _valid_local_producer_rotary_proof(
            proof, cache=self, layer_id=layer, prefill_layout=prefill_layout,
        ):
            return decline("proof")
        if tag not in _FULL or type(request) is not int or not 0 <= request < len(proof.prefixes):
            return decline("request")
        key = layer, tag
        pool, ratio = self.pools[key], self.geometry.ratio(layer)
        if not self._local_source_native_binding_valid(key, pool):
            return decline("binding")
        width = _WIDTH[tag]
        if (ratio not in (1, 2) or not self._local_source_native_read_device(pool)
                or not all(_local_swa_raw_tensor(value) for value in (pool.tensor, rows))
                or pool.tensor.ndim != 2 or pool.tensor.dtype != torch.uint8
                or not pool.tensor.is_contiguous() or pool.tensor.stride(0) != pool.tensor.shape[1]
                or rows.dtype != torch.uint8 or rows.device != pool.tensor.device
                or rows.ndim != 2 or rows.shape[1] != width or not rows.is_contiguous()
                or type(start) is not int or start < 0 or start % ratio
                or pool.span <= 0 or pool.span % ratio or pool.entries < pool.span // ratio
                or pool.tensor.shape[1] < pool.entries * width
                or not 0 < rows.shape[0] <= 16384
                or proof.prefixes[request] % ratio or proof.lengths[request] % ratio):
            return decline("geometry")
        first, count = start // ratio, rows.shape[0]
        begin = proof.prefixes[request] // ratio
        stop = (proof.prefixes[request] + proof.lengths[request]) // ratio
        if not begin <= first < stop or first + count != stop:
            return decline("range")
        source_begin, target_begin = rows.data_ptr(), pool.tensor.data_ptr()
        if (source_begin < target_begin + pool.tensor.numel() * pool.tensor.element_size()
                and target_begin < source_begin + rows.numel() * rows.element_size()):
            return decline("alias")
        try:
            from rtp_llm.models_py.modules.dsv41.ppu.graph_cache_kernels import _kernels
            kernels = _kernels()
        except ImportError:
            return decline("provider")
        mirror = self._verified_local_source_native_table(key, pool)
        page = pool.span // ratio
        first_page, page_end = first // page, (first + count - 1) // page + 1
        if mirror is None or page_end > mirror.shape[1]:
            return decline("table")
        physical = tuple(map(int, mirror[request, first_page:page_end].tolist()))
        if (any(block < 0 or block >= pool.tensor.shape[0] for block in physical)
                or len(set(physical)) != len(physical)):
            return decline("physical")
        self._invalidate_local_source_reads()
        stats["launches"] += 1
        kernels.store_cp_prefill_rows[(count,)](
            pool.tensor, pool.table[request], rows, first, first, pool.tensor.stride(0),
            pool.table.stride(1), width, pool.entries, page, page, 1, 0,
            tag == "indexer_kv", kernels.triton.next_power_of_2(width),
        )
        self.writes.append(dict(layer=layer, tag=tag, request=request, rows=count, owned_rows=count))
        stats["writes"] += 1
        stats["rows"] += count
        stats["bytes"] += count * width
        stats["geometries"].append(dict(
            source=layer, tag=tag, request=request, count=count, ratio=ratio,
            entries=pool.entries, span=pool.span, page_rows=page,
            stride=pool.tensor.shape[1], blocks=pool.tensor.shape[0], rawstart=start,
            first=first, first_page=first_page, written_pages=page_end - first_page,
        ))
        return True

    def _local_source_native_binding_valid(self, key, pool):
        return _local_pool_binding_valid(
            key, pool, self._local_source_native_read_bindings.get(key),
            (pool.entries, pool.span, self.geometry.ratio(key[0]), self.lengths),
            self._local_source_native_read_invalid_pools,
        )

    def _verified_local_source_native_table(self, key, pool):
        if not self._local_source_native_binding_valid(key, pool):
            return None
        return _verified_cache_table(
            self, pool, source=self._local_source_native_read_sources.get(pool.framework_tag),
            field="kv_cache_kernel_block_id", mirrors=self._local_source_native_read_mirrors,
            invalid=self._local_source_native_read_invalid, invalid_key=pool.framework_tag,
            stats=self._local_source_native_read_stats, counter="mirrors",
        )

    def _try_read_local_source_pool(self, key, request, count, *, split_indexer=False):
        if not getattr(self, "_local_source_native_read_enabled", False):
            return None
        stats = self._local_source_native_read_stats

        def decline(reason):
            stats["fallbacks"] += 1
            reasons = stats["fallback_reasons"]
            reasons[reason] = reasons.get(reason, 0) + 1
            return None

        if self._local_source_native_read_nonce is not self._local_source_native_read_birth_nonce:
            self.close_local_source_native_reads()
            return decline("nonce")
        if stats["closed"] or not self._local_source_native_read_business():
            return decline("scope")
        if type(request) is not int or type(count) is not int or not 0 < count <= 131072:
            return decline("geometry")
        pool = self.pools[key]
        if not self._local_source_native_binding_valid(key, pool):
            return decline("binding")
        ratio, width = self.geometry.ratio(key[0]), _WIDTH[key[1]]
        if (key[1] not in _FULL or not self._local_source_native_read_device(pool)
                or not _local_swa_raw_tensor(pool.tensor) or pool.tensor.ndim != 2
                or pool.tensor.dtype != torch.uint8 or not pool.tensor.is_contiguous()
                or pool.tensor.stride(0) != pool.tensor.shape[1]
                or ratio not in (1, 2) or pool.span <= 0 or pool.span % ratio
                or pool.entries < pool.span // ratio or pool.tensor.shape[1] < pool.entries * width
                or not 0 <= request < len(self.lengths)):
            return decline("geometry")
        pages = (count + pool.span // ratio - 1) // (pool.span // ratio)
        if pages * pool.tensor.shape[1] > 128 * 1024**2:
            return decline("packet_limit")
        # Resolve the existing provider before submitting mirror/upload work.
        try:
            from rtp_llm.models_py.modules.dsv41.ppu.prefill_readback import _restore_rows
            from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_metadata import _upload
        except ImportError:
            return decline("provider")

        mirror = self._verified_local_source_native_table(key, pool)
        if mirror is None or pages > mirror.shape[1]:
            return decline("table")
        nonce = self._local_source_native_read_nonce
        ids_key = nonce, pool.framework_tag, request, pages
        saved_ids = self._local_source_native_read_ids.get(ids_key)
        if saved_ids is None:
            physical = tuple(map(int, mirror[request, :pages].tolist()))
            low, high = min(physical), max(physical)
            if low < 0 or high >= pool.tensor.shape[0]:
                return decline("physical")  # Let old checks preserve exact errors.
            ids = _upload(physical, pool.tensor.device, torch.int64)
            saved_ids = ids, _source_tensor_key(ids), low, high
            self._local_source_native_read_ids[ids_key] = saved_ids
            stats["ids_uploads"] += 1
            stats["ids_bytes"] += ids.numel() * ids.element_size()
        else:
            ids, stamp, low, high = saved_ids
            if _source_tensor_key(ids) != stamp:
                self.close_local_source_native_reads()
                return decline("ids_changed")
            if low < 0 or high >= pool.tensor.shape[0]:
                return decline("physical")
            stats["ids_hits"] += 1
        plan_key = nonce, count, pool.tensor.device
        saved_plan = self._local_source_native_read_plans.get(plan_key)
        if saved_plan is None:
            plan = _upload([(0, 0, count, 0)], pool.tensor.device, torch.int64)
            self._local_source_native_read_plans[plan_key] = plan, _source_tensor_key(plan)
            stats["plan_uploads"] += 1
            stats["plan_bytes"] += plan.numel() * plan.element_size()
        else:
            plan, stamp = saved_plan
            if _source_tensor_key(plan) != stamp:
                self.close_local_source_native_reads()
                return decline("plan_changed")
            stats["plan_hits"] += 1
        packet = pool.tensor.index_select(0, saved_ids[0])
        stats["packet_calls"] += 1
        stats["packet_bytes"] += packet.numel() * packet.element_size()
        split_indexer = split_indexer and key[1] == "indexer_kv"
        output = pool.tensor.new_empty((count, 64 if split_indexer else width))
        scale_bytes = pool.tensor.new_empty((count, 4)) if split_indexer else None
        page_rows = pool.span // ratio
        stats["restore_calls"] += 1
        _restore_rows[(1, (count * width + 1023) // 1024)](
            packet, plan, output, scale_bytes, 1, page_rows, page_rows, 1,
            pool.tensor.shape[1], pool.tensor.shape[1], pool.entries, width,
            key[1] == "indexer_kv", split_indexer, 1024,
        )
        stats["reads"] += 1
        stats["output_bytes"] += count * width
        stats["geometries"].append(dict(
            source=key[0], tag=key[1], request=request, count=count, ratio=ratio,
            entries=pool.entries, span=pool.span, page_rows=page_rows, pages=pages,
            stride=pool.tensor.shape[1], blocks=pool.tensor.shape[0],
        ))
        if split_indexer:
            self._local_sg_source_stats["split_reads"] += 1
            return output.view(torch.int8), scale_bytes.view(torch.int32).reshape(count)
        return output

    def _local_source_signature(self, source, *, owners=None):
        ratio = self.geometry.ratio(source)
        if ratio not in (1, 2):
            return None
        tag = "global_kv_2" if ratio == 2 else "global_kv_1"
        pools = []
        for key in ((source, tag), (source, "indexer_kv")):
            pool = self.pools.get(key)
            if pool is None or pool.framework_tag not in self.inputs:
                return None
            meta = self.inputs[pool.framework_tag]
            tensors = (pool.tensor, pool.table, pool.handle.kv_cache_base,
                       getattr(meta, "kv_cache_kernel_block_id", None),
                       getattr(meta, "kv_cache_kernel_block_id_device", None))
            if owners is not None:
                # Keep the original tensor identities alive until invalidation;
                # allocator reuse must not disguise a replacement as a hit.
                owners.extend(tensors)
            pools.append((pool.framework_tag, pool.entries, pool.span,
                          tuple(_source_tensor_key(t) for t in tensors)))
        return self.lengths, ratio, tuple(pools)

    @staticmethod
    def _source_bank_tensors(saved):
        global_rows, indexer = saved
        return (global_rows, *indexer) if isinstance(indexer, tuple) else (global_rows, indexer)

    def _check_local_source_reads(self):
        if not self._local_source_cache_eligible():
            self._invalidate_local_source_reads()
            return False
        for (source, request), (signature, banks, owners) in self._local_source_reads.items():
            saved = self._source_reads.get((source, request))
            if (
                saved is None or self._local_source_signature(source) != signature
                or tuple(_source_tensor_key(t) for t in self._source_bank_tensors(saved)) != banks
            ):
                self._invalidate_local_source_reads()
                break
        return True

    def _cached_source_read(self, source, request, global_entries):
        local = self._check_local_source_reads()
        key = source, int(request)
        if local and (type(global_entries) is not int or global_entries < 0):
            self._invalidate_local_source_reads()
            local = False
        saved = self._source_reads.get(key)
        if local and saved is not None and (key not in self._local_source_reads or global_entries > saved[0].shape[0]):
            self._invalidate_local_source_reads()
            saved = None
        return local, saved

    def _read_source_rows(self, source, request, count, *, split_indexer=False):
        """Read both source pools through the same kernel/fallback boundary."""
        ratio = self.geometry.ratio(source)
        tag = "global_kv_2" if ratio == 2 else "global_kv_1"
        banks, positions = [], None
        for kind in (tag, "indexer_kv"):
            rows = self._try_read_local_source_pool(
                (source, kind), request, count,
                split_indexer=split_indexer and kind == "indexer_kv",
            )
            if rows is None:
                if positions is None:
                    positions = torch.arange(count, device=self.device) * ratio
                rows = self._copy_rows((source, kind), request, positions)
            banks.append(rows)
        return tuple(banks)

    def _read_local_sg_source(self, source, request, count):
        """Read-only banks; old packed ABI is assembled only when requested."""
        if (not getattr(self, "_local_sg_source_enabled", False)
                or self._local_sg_source_stats["closed"]
                or type(request) is not int or type(count) is not int or count <= 0):
            return None
        local, saved = self._cached_source_read(source, request, count)
        if not local:
            return None
        if saved is not None:
            self._local_sg_source_stats["hits"] += 1
            if isinstance(saved[1], tuple):
                return saved
        else:
            saved = self._read_source_rows(source, request, count, split_indexer=True)
        if not isinstance(saved[1], tuple):
            from rtp_llm.models_py.modules.dsv41.ppu._quant import _bytes_to_sf

            packed = saved[1]
            saved = saved[0], (packed[:, :64].contiguous().view(torch.int8), _bytes_to_sf(packed[:, 64:]))
            self._local_sg_source_stats["derived_splits"] += 1
        # Publish only after original main->indexer read/validation succeeds.
        banks = self._source_bank_tensors(saved)
        owners = list(banks)
        signature = self._local_source_signature(source, owners=owners)
        if signature is not None:
            key = source, request
            self._source_reads[key] = saved
            self._local_source_reads[key] = signature, tuple(_source_tensor_key(t) for t in banks), tuple(owners)
        return saved

    def read_source(self, source, request, *, global_entries):
        banks = self._read_local_sg_source(source, request, global_entries)
        if banks is not None:
            payload, scales = banks[1]
            indexer = torch.cat((payload[:global_entries].view(torch.uint8),
                                 scales[:global_entries].view(torch.uint8).reshape(-1, 4)), dim=1)
            self._local_sg_source_stats["packed_rebuilds"] += 1
            return banks[0][:global_entries], indexer
        local, saved = self._cached_source_read(source, request, global_entries)
        key = source, int(request)
        if saved is not None:
            global_rows, indexer = saved
            if isinstance(indexer, tuple):
                # The BF16 scorer still consumes the original byte layout.
                payload, scales = indexer
                indexer = torch.cat(
                    (payload[:global_entries].view(torch.uint8),
                     scales[:global_entries].view(torch.uint8).reshape(-1, 4)),
                    dim=1,
                )
            return global_rows[:global_entries], indexer[:global_entries]
        # Publish only AFTER both old packed reads and all their native checks
        # succeed. Never fill inactive requests or read beyond this call's count.
        packed = self._read_source_rows(source, request, global_entries)
        if local:
            banks = self._source_bank_tensors(packed)
            owners = list(banks)
            signature = self._local_source_signature(source, owners=owners)
            if signature is not None:
                self._source_reads[key] = packed
                # Cached banks are read-only views, as in the existing CP path.
                # Retain owners; normal mutation invalidates. Inference inputs
                # are immutable within this forward and native writes invalidate
                # explicitly before launch. A new forward owns a new cache.
                self._local_source_reads[key] = signature, tuple(_source_tensor_key(t) for t in banks), tuple(owners)
        return packed

    def read_global(self, source, request, *, global_entries):
        banks = self._read_local_sg_source(source, request, global_entries)
        if banks is not None:
            return banks[0][:global_entries]
        _, saved = self._cached_source_read(source, request, global_entries)
        if saved is not None:
            return saved[0][:global_entries]
        return self.read_source(source, request, global_entries=global_entries)[0]

    def read_indexer_fp4(self, source, request, *, global_entries):
        from rtp_llm.models_py.modules.dsv41.ppu._quant import _bytes_to_sf

        banks = self._read_local_sg_source(source, request, global_entries)
        if banks is not None:
            return banks[1][0][:global_entries], banks[1][1][:global_entries]
        _, saved = self._cached_source_read(source, request, global_entries)
        if saved is not None and isinstance(saved[1], tuple):
            payload, scales = saved[1]
            return payload[:global_entries], scales[:global_entries]
        packed = self.read_source(source, request, global_entries=global_entries)[1]
        return packed[:, :64].contiguous().view(torch.int8), _bytes_to_sf(packed[:, 64:])

    def prepare_source(self, source):
        """Uniform collectives before any rank-local selector/consumer branch."""
        if self.sharded:
            mode = os.environ.get("RTP_V41_PPU_CP8_BATCHED_KV_READ", "1")
            if (
                mode in ("1", "verify")
                and self._is_prefill
                and self.cp["size"] == 8
                and torch.device(self.device).type == "cuda"
            ):
                if not all(
                    (source, rid) in self._source_reads for rid in range(len(self.lengths))
                ):
                    self._prepare_source_batched(source, verify=mode == "verify")
                return
            for rid, length in enumerate(self.lengths):
                self._source_reads[source, rid] = self.read_source(
                    source, rid, global_entries=length // self.geometry.ratio(source),
                )

    def bind_shared_source_prefix(self, prefixes):
        """Agree on full stripes whose physical pages alias across requests."""
        self.shared_prefix_stripes = {}
        if len(self.lengths) < 2 or len(set(self.lengths)) != 1 or min(prefixes) <= 0:
            return
        keys, candidates = [], []
        for key, pool in self.pools.items():
            if key[1] not in _FULL:
                continue
            placement = self.placements[key]
            width = placement.physical_span * placement.size
            stripes = min(prefixes) // width
            host = getattr(self.inputs[pool.framework_tag], "kv_cache_block_id", None)
            if (
                stripes and isinstance(host, torch.Tensor) and host.device.type == "cpu"
                and host.shape == placement.table.shape
                and host.shape[0] >= len(prefixes) and host.shape[1] >= stripes
            ):
                pages = host[:len(prefixes), :stripes]
                canonical = pages[0]
                block_count = pool.tensor.shape[0] // placement.pages
                if not (
                    torch.equal(pages, canonical.expand_as(pages))
                    and bool(((canonical >= 0) & (canonical < block_count)).all())
                ):
                    stripes = 0
            else:
                stripes = 0
            keys.append(key)
            candidates.append(stripes)
        if not keys:
            return
        import torch.distributed as dist
        from rtp_llm.models_py.distributed.collective_torch import Group, _get_group

        # Never pick a shorter collective from a rank-local alias check alone.
        agreed = torch.tensor(candidates, dtype=torch.int32, device=self.device)
        dist.all_reduce(agreed, op=dist.ReduceOp.MIN, group=_get_group(Group.TP))
        self.shared_prefix_stripes = dict(zip(keys, agreed.tolist()))

    def _prepare_source_batched(self, source, *, verify):
        from rtp_llm.models_py.modules.dsv41.ppu.prefill_readback import gather_pool

        ratio = self.geometry.ratio(source)
        counts = tuple(length // ratio for length in self.lengths)
        tag = "global_kv_2" if ratio == 2 else "global_kv_1"
        packed = []
        split_indexer = os.environ.get("RTP_V41_PPU_CP8_SPLIT_INDEXER_READBACK", "0") == "1"
        for kind in (tag, "indexer_kv"):
            key = source, kind
            packed.append(
                gather_pool(
                    self.pools[key],
                    self.placements[key],
                    counts,
                    width=_WIDTH[kind],
                    planar=kind == "indexer_kv",
                    split_indexer=split_indexer and kind == "indexer_kv",
                    plans=getattr(self, "prefill_plans", None),
                    receive=getattr(self, "gather_receive", None),
                    shared_prefix_stripes=(self.shared_prefix_stripes or {}).get(key, 0),
                )
            )
        rows = tuple(zip(*packed))
        if verify:
            self._verify_source_readback(source, counts, rows)
        self._source_reads.update(((source, rid), row) for rid, row in enumerate(rows))

    def _verify_source_readback(self, source, counts, rows):
        from rtp_llm.models_py.distributed.collective_torch import Group, all_gather

        mismatches = []
        for rid, (count, actual) in enumerate(zip(counts, rows)):
            expected = self.read_source(source, rid, global_entries=count)
            for kind, new, old in zip(("global", "indexer"), actual, expected):
                if isinstance(new, tuple):
                    payload, scales = new
                    match = torch.equal(payload.view(torch.uint8), old[:, :64]) and torch.equal(
                        scales.view(torch.uint8).reshape(-1, 4), old[:, 64:],
                    )
                else:
                    match = torch.equal(new, old)
                if not match:
                    mismatches.append((rid, kind))
        failed = torch.tensor(
            [int(bool(mismatches))], dtype=torch.int32, device=self.device
        )
        if bool(all_gather(failed, Group.TP).any()):
            raise RuntimeError(f"CP8 KV readback mismatch at source {source}: {mismatches}")
        _LOGGER.info(
            "CP8 KV readback bytes match: rank=%s source=%s counts=%s",
            self.cp["rank"],
            source,
            counts,
        )

    def read_csa(self, layer, request, positions):
        return self._copy_rows((layer, "csa_state"), request, positions)

    def _batched_swa_enabled(self):
        return (
            os.environ.get("RTP_V41_PPU_CP8_BATCHED_SWA", "1") == "1"
            and getattr(self, "_is_prefill", False)
            and self.sharded
            and self.cp["size"] == 8
        )

    def bind_swa_fresh_lengths(self, lengths, *, keep_plans=False):
        """Describe the full CP gather's request-major rows, or clear on compaction."""
        self._swa_fresh_lengths = lengths
        if lengths is None:
            if not keep_plans:
                self.prefill_plans = None
            self.gather_receive = None
        elif (
            getattr(self, "prefill_plans", None) is None
            and torch.device(self.device).type == "cuda"
            and os.environ.get("RTP_V41_PPU_CP8_ASYNC_PREFILL_METADATA", "1") == "1"
        ):
            self.prefill_plans = {}
        if (
            lengths is not None
            and getattr(self, "gather_receive", None) is None
            and torch.device(self.device).type == "cuda"
            and os.environ.get(CP8_GATHER_RECEIVE_ENV, "1") == "1"
            and self.cp.get("size") == 8
        ):
            from rtp_llm.models_py.modules.dsv41.cp import CPGatherReceive

            self.gather_receive = CPGatherReceive(self.cp["size"], self.cp["rank"])

    def _local_swa_pool_key(self, layer):
        pool = self.pools[layer, "swa_kv"]
        meta = self.inputs[pool.framework_tag]
        owners = (pool.tensor, pool.table, pool.handle.kv_cache_base,
                  getattr(meta, "kv_cache_block_id", None),
                  getattr(meta, "kv_cache_block_id_device", None))
        return (pool.framework_tag, pool.entries, pool.span,
                tuple(_source_tensor_key(value) for value in owners)), owners

    def _local_swa_write_payload_eligible(self, pool, packed):
        if not all(_local_swa_raw_tensor(value) for value in (pool.tensor, packed)):
            return False
        if not pool.tensor.is_contiguous():
            return False
        begin = pool.tensor.data_ptr()
        end = begin + pool.tensor.numel() * pool.tensor.element_size()
        for (_, tag), other in self.pools.items():
            if tag not in _FULL or other.tensor.device != pool.tensor.device:
                continue
            if other.tensor.layout != torch.strided or not other.tensor.is_contiguous():
                return False
            other_begin = other.tensor.data_ptr()
            other_end = other_begin + other.tensor.numel() * other.tensor.element_size()
            if begin < other_end and other_begin < end:
                # SWA writes intentionally preserve source banks. Native stores
                # cannot update their version counters through a physical alias.
                # Preserve the old writer for this exceptional shared payload.
                return False
        return True

    def _local_swa_native_geometry(self, pool, packed):
        if (
            not self._local_swa_write_payload_eligible(pool, packed)
            or not pool.tensor.is_cuda or not packed.is_cuda or torch.is_grad_enabled()
            or torch.cuda.is_current_stream_capturing()
            or pool.tensor.ndim != 2 or pool.tensor.dtype != torch.uint8
            or not pool.tensor.is_contiguous() or pool.tensor.stride(0) != pool.tensor.shape[1]
            or packed.device != pool.tensor.device or packed.dtype != torch.uint8
            or packed.ndim != 2 or packed.shape[1] != _WIDTH["swa_kv"] or not packed.is_contiguous()
            or pool.entries < self.geometry.window or pool.span <= 0
            or pool.tensor.shape[1] < pool.entries * _WIDTH["swa_kv"]
        ):
            return None
        from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu

        if not is_ppu() or torch.cuda.get_device_capability(pool.tensor.device) != (8, 9):
            return None
        return _LocalSwaWriteGeometry(0, pool.entries, pool.tensor.shape[1])

    def _verified_local_swa_write_table(self, pool):
        return _verified_cache_table(
            self, pool, source=self._local_swa_write_sources.get(pool.framework_tag),
            field="kv_cache_block_id", mirrors=self._local_swa_write_mirrors,
            mirror_keys=self._local_swa_write_mirror_keys,
            invalid=self._local_swa_write_invalid, invalid_key=pool.framework_tag,
            stats=self._local_swa_native_write_stats, counter="mirrors",
        )

    def _checked_local_swa_ranges(self, layer, positions, requests, packed, prefixes,
                                  retain_rows, retain_prefix_tail, bounded_starts, ticket):
        if (
            os.environ.get(LOCAL_PREFILL_SWA_SEGMENTS_ENV, "0") != "1"
            or type(self) is not SharedNativeCache or not self._is_prefill or self.sharded
            or not retain_rows or not isinstance(ticket, _LocalSwaStageRanges)
            or len(ticket.inputs) != 3 or len(ticket.ranges) != len(self.lengths)
            or ticket.cache is not self or ticket.prefixes != tuple(prefixes)
            or ticket.totals != self.lengths or ticket.retain_prefix_tail != retain_prefix_tail
            or ticket.bounded_starts != bounded_starts or ticket.pool is not self.pools[layer, "swa_kv"]
            or ticket.pool_key != self._local_swa_pool_key(layer)[0]
            or any(left is not right for left, right in zip(ticket.inputs, (positions, requests, packed)))
            or ticket.input_keys != tuple(_source_tensor_key(t) for t in (positions, requests, packed))
            or not ticket.guard()
        ):
            return None
        if (
            positions.ndim != 1 or positions.dtype != torch.int64 or not positions.is_contiguous()
            or requests.shape != positions.shape or requests.dtype != torch.int64 or not requests.is_contiguous()
            or packed.shape != (positions.numel(), _WIDTH["swa_kv"]) or packed.dtype != torch.uint8
            or not packed.is_contiguous() or positions.device != packed.device or requests.device != packed.device
            or len(prefixes) != len(self.lengths)
            or (bounded_starts is not None and len(bounded_starts) != len(self.lengths))
        ):
            return None
        # Never admit metadata/payload that a native writer could overwrite.
        native_storage = {value.untyped_storage().data_ptr()
                          for pool in self.pools.values() for value in (pool.tensor, pool.table)}
        if any(value.numel() and value.untyped_storage().data_ptr() in native_storage
               for value in (positions, requests, packed)):
            return None
        offset = 0
        for rid, (begin, start, length) in enumerate(ticket.ranges):
            if (any(type(value) is not int or value < 0 for value in (begin, start, length))
                    or begin != offset or start + length != self.lengths[rid]
                    or (retain_prefix_tail and start != prefixes[rid])
                    or (not retain_prefix_tail and (
                        bounded_starts is None or start != bounded_starts[rid]))):
                return None
            offset += length
        return ticket.ranges if offset == positions.numel() else None

    def close_local_swa_native_reads(self):
        """Drop metadata owners and permanently end this forward's read scope."""
        plans = getattr(self, "_local_swa_read_plans", None)
        if plans is not None:
            plans.clear()
        self._local_swa_read_rows = None
        mirrors = getattr(self, "_local_swa_read_mirrors", None)
        if mirrors is not None:
            mirrors.clear()
        stats = getattr(self, "_local_swa_native_read_stats", None)
        if stats is not None:
            stats["closed"] = True

    def _local_swa_native_read_stream(self, pool):
        return (
            pool.tensor.is_cuda and torch.cuda.current_device() == pool.tensor.device.index
            and not torch.cuda.is_current_stream_capturing()
            and torch.cuda.current_stream(pool.tensor.device) == torch.cuda.default_stream(pool.tensor.device)
        )

    def _local_swa_read_rows_valid(self, ticket):
        if (not isinstance(ticket, _LocalSwaStageRanges) or ticket.cache is not self
                or not ticket.retain_prefix_tail or ticket.bounded_starts is not None
                or ticket.totals != self.lengths or not ticket.guard()):
            return False
        rows = ticket.inputs[:2]
        if (not all(_local_swa_raw_tensor(value) for value in rows)
                or tuple(_source_tensor_key(value) for value in rows) != ticket.input_keys[:2]):
            return False
        if self._local_swa_read_rows is None:
            self._local_swa_read_rows = rows, ticket.input_keys[:2]
            return True
        owners, keys = self._local_swa_read_rows
        if tuple(_source_tensor_key(value) for value in owners) != keys:
            return False
        # Each layer's slice(None) creates a different TensorImpl. Bind exact
        # same-domain aliases AND their original A08 guard, never shape alone.
        return all(torch._C._is_alias_of(current, owner)
                   and _source_tensor_key(current)[1:] == key[1:]
                   for current, owner, key in zip(rows, owners, keys))

    def _try_local_swa_native_tail(self, layer, request, prefix, ticket, local_write):
        if not getattr(self, "_local_swa_read_enabled", False):
            return None
        stats = self._local_swa_native_read_stats

        def decline(*, close=False):
            if close:
                self.close_local_swa_native_reads()
            stats["fallbacks"] += 1
            return None

        pool = self.pools[layer, "swa_kv"]
        if pool.framework_tag in self._local_swa_write_invalid:
            return decline(close=True)
        if (stats["closed"] or local_write is None
                or torch.is_grad_enabled() or not self._local_swa_native_read_stream(pool)):
            return decline()
        geometry, mirror = local_write
        if not self._local_swa_read_rows_valid(ticket):
            return decline(close=True)
        # Revalidate the original table birth on EVERY use, including hits.
        # This never obtains a new mirror after a changed-then-restored table.
        if self._verified_local_swa_write_table(pool) is not mirror:
            return decline(close=True)
        prior = self._local_swa_read_mirrors.get(pool.framework_tag)
        if prior is not None and (prior[0] is not mirror or prior[1] != _source_tensor_key(mirror)):
            return decline(close=True)
        if prior is None:
            self._local_swa_read_mirrors[pool.framework_tag] = mirror, _source_tensor_key(mirror)
        if (geometry.rank != 0 or geometry.full_entries != pool.entries
                or geometry.slice_elements != pool.tensor.shape[1]
                or not _local_swa_raw_tensor(pool.tensor) or not pool.tensor.is_contiguous()
                or pool.tensor.dtype != torch.uint8 or pool.tensor.ndim != 2
                or pool.tensor.stride(0) != pool.tensor.shape[1]
                or pool.entries < self.geometry.window or pool.span <= 0
                or pool.tensor.shape[1] < pool.entries * _WIDTH["swa_kv"]
                or type(prefix) is not int or prefix < 0
                or not 0 <= request < mirror.shape[0]
                or request >= len(ticket.prefixes) or prefix != ticket.prefixes[request]):
            return decline()
        start = max(0, prefix - self.geometry.window + 1)
        count = prefix - start
        if not count:
            stats["requests"] += 1
            return pool.tensor.new_empty((0, _WIDTH["swa_kv"]))
        first, stop = start // pool.span, (prefix - 1) // pool.span + 1
        if stop > mirror.shape[1]:
            return decline()
        physical = tuple(int(value) for value in mirror[request, first:stop].tolist())
        if any(block < 0 or block >= pool.tensor.shape[0] for block in physical):
            return decline()  # Preserve the old reader's precise error path.
        direct = len(physical) == 1
        source = self._local_swa_write_sources[pool.framework_tag]
        metadata_owners = source[0], *source[1], mirror
        lineage = id(source[0]), source[2], _source_tensor_key(mirror)
        key = (
            self._local_swa_read_nonce, pool.framework_tag, lineage,
            self._local_swa_read_rows[1], request, prefix,
            start, count, first, stop, physical, pool.span, pool.entries,
            pool.tensor.shape[1], pool.tensor.shape[0], pool.tensor.device, direct,
        )
        entry = self._local_swa_read_plans.get(key)
        if entry is None:
            from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_metadata import _upload

            plan = _upload([(start, count, physical[0] if direct else 0, 0)], pool.tensor.device, torch.int64)
            blocks = None if direct else _upload(physical, pool.tensor.device, torch.int64)
            # Only immutable metadata owners; no packed values, snapshots,
            # pool payload version or guard closure is retained by this cache.
            entry = metadata_owners, plan, blocks
            self._local_swa_read_plans[key] = entry
            stats["plan_uploads"] += 1 + int(blocks is not None)
        else:
            stats["cache_hits"] += 1
        _, plan, blocks = entry
        packet = pool.tensor if direct else pool.tensor.index_select(0, blocks)
        output = pool.tensor.new_empty((count, _WIDTH["swa_kv"]))
        from rtp_llm.models_py.modules.dsv41.ppu.prefill_swa import _restore_tails

        _restore_tails[(1, (count * _WIDTH["swa_kv"] + 511) // 512)](
            packet, plan, output, pool.tensor.shape[0] if direct else len(physical),
            pool.span, geometry.full_entries, geometry.slice_elements, _WIDTH["swa_kv"], 512,
        )
        stats["requests"] += 1
        stats["launches"] += 1
        stats["direct" if direct else "packet"] += 1
        return output

    def stage_swa(
        self, layer, positions, requests, packed, prefixes, *,
        retain_rows=True, retain_prefix_tail=True, bounded_starts=None, local_ranges=None,
    ):
        """Preserve the SWA rows visible to this forward before ring writes.

        Bounded starts describe contiguous suffixes in restored request order.
        """
        if not self.lengths:
            return
        local_ticket = local_ranges
        local_ranges = self._checked_local_swa_ranges(
            layer, positions, requests, packed, prefixes, retain_rows,
            retain_prefix_tail, bounded_starts, local_ranges,
        )
        local_write = None
        if local_ranges is not None:
            geometry = self._local_swa_native_geometry(self.pools[layer, "swa_kv"], packed)
            if geometry is not None:
                table = self._verified_local_swa_write_table(self.pools[layer, "swa_kv"])
                if table is not None:
                    local_write = geometry, table
        if (getattr(self, "_local_swa_read_enabled", False) and not self._local_swa_native_read_stats["closed"]
                and self._local_swa_read_rows is not None):
            owners, keys = self._local_swa_read_rows
            if (tuple(_source_tensor_key(value) for value in owners) != keys
                    or not isinstance(local_ticket, _LocalSwaStageRanges) or not local_ticket.guard()
                    or self.pools[layer, "swa_kv"].framework_tag in self._local_swa_write_invalid):
                self.close_local_swa_native_reads()
        batched = (
            self._batched_swa_enabled()
            and retain_rows
            and packed.is_cuda
        )
        batch_swa = batched and retain_prefix_tail
        bounded_range = (
            batched and not retain_prefix_tail and bounded_starts is not None
            and len(bounded_starts) == len(self.lengths)
            and os.environ.get("RTP_V41_PPU_CP8_BOUNDED_SWA_RANGE", "1") == "1"
        )
        pool = self.pools[layer, "swa_kv"]
        fresh_lengths = None
        host_table = None
        if batch_swa or bounded_range:
            from rtp_llm.models_py.modules.dsv41.ppu.prefill_swa import (
                gather_tails,
                write_prefill_range,
                write_rows,
            )

            placement = self.placements[layer, "swa_kv"]
            fresh_starts = bounded_starts if bounded_range else prefixes
            lengths = (
                tuple(end - start for start, end in zip(fresh_starts, self.lengths))
                if bounded_range else getattr(self, "_swa_fresh_lengths", None)
            )
            plans = getattr(self, "prefill_plans", None)
            if (
                lengths is not None
                and len(lengths) == len(fresh_starts) == len(self.lengths)
                and sum(lengths) == positions.numel() == requests.numel() == packed.shape[0]
                and all(
                    0 <= start <= end and start + size == end
                    for start, size, end in zip(fresh_starts, lengths, self.lengths)
                )
            ):
                fresh_lengths = lengths
                # Native eager metadata keeps the original host page table
                # alongside its device copy. Read it without synchronizing PPU.
                table = getattr(self.inputs[pool.framework_tag], "kv_cache_block_id", None)
                if (
                    table is not None
                    and table.device.type == "cpu"
                    and table.shape == pool.table.shape
                    and table.shape[0] >= len(lengths)
                    and packed.is_contiguous()
                ):
                    host_table = table
            if bounded_range and host_table is None:
                fresh_lengths = None
        if batch_swa:
            host_table_for_read = None
            if plans is not None:
                # Both tables belong to this forward's native request metadata.
                table = getattr(self.inputs[pool.framework_tag], "kv_cache_block_id", None)
                if (
                    table is not None and table.device.type == "cpu"
                    and table.shape == placement.table.shape
                    and table.dtype in (torch.int32, torch.int64)
                ):
                    host_table_for_read = table
            old_tails = gather_tails(
                pool, placement, prefixes, self.geometry.window,
                plans=plans, host_table=host_table_for_read,
                receive=getattr(self, "gather_receive", None),
            )
            logical_rows = sum(tail.shape[0] for tail in old_tails)
            if logical_rows:
                self.pool_gathers.append(
                    {
                        "layer": layer,
                        "tag": "swa_kv",
                        "requests": [
                            rid for rid, tail in enumerate(old_tails) if tail.numel()
                        ],
                        "logical_rows": logical_rows,
                        "cp_size": placement.size,
                    }
                )
        elif fresh_lengths is not None:
            # Bounded replay sees the contiguous retained suffix only.
            old_tails = (packed[:0],) * len(fresh_lengths)
        direct_swa = (
            fresh_lengths is not None
            and packed.is_contiguous()
            and self.cp8_row_prep_enabled()
            and os.environ.get("RTP_V41_PPU_CP8_DIRECT_SWA_WORKSPACE", "1") == "1"
        )
        offset = 0
        for rid in range(len(self.lengths)):
            key = layer, rid
            if local_ranges is not None:
                begin, start, length = local_ranges[rid]
                chosen = slice(begin, begin + length)
            elif fresh_lengths is None:
                chosen = requests == rid
            else:
                stop = offset + fresh_lengths[rid]
                chosen = slice(offset, stop)
                offset = stop
            fresh_pos, fresh = positions[chosen], packed[chosen]
            if local_ranges is not None:
                # Match the old boolean gather's ownership, before this rid's
                # tail read and native write. Do not hoist across aliased rids.
                fresh = fresh.clone()
            if retain_rows:
                prefix = int(prefixes[rid])
                if local_ranges is not None:
                    old = fresh[:0]
                    if retain_prefix_tail and prefix:
                        old = self._try_local_swa_native_tail(layer, rid, prefix, local_ticket, local_write)
                        if old is None:
                            old_pos = torch.arange(max(0, prefix - self.geometry.window + 1), prefix, device=self.device)
                            try:
                                old = self._copy_rows((layer, "swa_kv"), rid, old_pos)
                            except ValueError as error:
                                raise ValueError(f"SWA layer {layer}, request {rid}, prefix {prefix}: {error}") from error
                    self._fresh_swa[key] = _StagedSwaSegments(start, old, fresh)
                elif direct_swa:
                    self._fresh_swa[key] = _StagedSwaSegments(
                        int(fresh_starts[rid]), old_tails[rid], fresh,
                    )
                elif retain_prefix_tail and prefix:
                    old_pos = torch.arange(max(0, prefix - self.geometry.window + 1), prefix, device=self.device)
                    if batch_swa:
                        old = old_tails[rid]
                    else:
                        try:
                            old = self._copy_rows((layer, "swa_kv"), rid, old_pos)
                        except ValueError as error:
                            raise ValueError(f"SWA layer {layer}, request {rid}, prefix {prefix}: {error}") from error
                    fresh_pos_and_tail = torch.cat((old_pos, fresh_pos))
                    fresh_and_tail = torch.cat((old, fresh))
                    self._fresh_swa[key] = (fresh_pos_and_tail, fresh_and_tail)
                else:
                    self._fresh_swa[key] = (fresh_pos, fresh)
            written = None
            if local_write is not None:
                from rtp_llm.models_py.modules.dsv41.ppu.prefill_swa import write_prefill_range

                geometry, table = local_write
                written = write_prefill_range(
                    pool, geometry, table, rid, start, self.lengths[rid], self.geometry.window, fresh,
                )
                if written is not None:
                    self._local_swa_native_write_stats["requests"] += 1
                    self._local_swa_native_write_stats["launches"] += bool(written)
                    self.writes.append(dict(layer=layer, tag=pool.framework_tag, request=rid,
                                            rows=len(written), absolute_positions=written, owned_rows=len(written)))
                    continue
                self._local_swa_native_write_stats["fallbacks"] += 1
            if host_table is not None:
                written = write_prefill_range(
                    pool, placement, host_table, rid, int(fresh_starts[rid]),
                    self.lengths[rid], self.geometry.window, fresh,
                )
            if written is None:
                # One write per ring slot; retain each allocated block's checkpoint.
                column_end = ((fresh_pos // pool.span + 1) * pool.span).clamp_max(self.lengths[rid])
                keep = fresh_pos >= column_end - pool.entries
                if getattr(self, "_is_prefill", False):
                    # Historical SWA columns at -1 stay available in _fresh_swa.
                    keep &= self._prefill_ring_keep(
                        pool, rid, fresh_pos, self.lengths[rid] - self.geometry.window,
                    )
                if not batch_swa:
                    self._copy_rows((layer, "swa_kv"), rid, fresh_pos[keep], fresh[keep])
                    self.writes.append({"layer": layer, "tag": pool.framework_tag, "request": rid,
                                        "rows": int(keep.sum()), "absolute_positions": fresh_pos[keep].tolist(),
                                        "owned_rows": int(self.owned_positions((layer, "swa_kv"), fresh_pos[keep]).sum())})
                    continue
                retained = fresh_pos[keep]
                write_rows(pool, placement, rid, retained, fresh[keep])
                written = retained.tolist()

            owned_start = placement.rank * placement.slice_elements
            owned_end = owned_start + placement.slice_elements
            width = _WIDTH["swa_kv"]
            byte_starts = (pos % placement.full_entries * width for pos in written)
            owned_count = sum(
                start < owned_end and start + width > owned_start for start in byte_starts
            )
            self.writes.append(
                {
                    "layer": layer,
                    "tag": pool.framework_tag,
                    "request": rid,
                    "rows": len(written),
                    "absolute_positions": written,
                    "owned_rows": owned_count,
                }
            )

    def try_decode_swa_range(self, layer, request, first, out):
        saved = self._fresh_swa.get((layer, int(request)))
        if not isinstance(saved, _StagedSwaSegments):
            return False

        from rtp_llm.models_py.modules.dsv41.ppu._quant import try_dequantize_swa_segments

        start = first - saved.prefix + saved.tail.shape[0]
        return try_dequantize_swa_segments(saved.tail, saved.fresh, start, out)

    def read_swa(self, layer, request, positions):
        saved = self._fresh_swa.get((layer, int(request)))
        if saved is None:
            return self._copy_rows((layer, "swa_kv"), request, positions)
        if isinstance(saved, _StagedSwaSegments):
            prefix, tail, fresh = saved
            domain = torch.arange(prefix - tail.shape[0], prefix + fresh.shape[0], device=self.device)
            values = torch.cat((tail, fresh))
        else:
            domain, values = saved
        if (
            self._batched_swa_enabled()
            and values.is_cuda
            and domain.is_contiguous()
            and values.is_contiguous()
            and positions.ndim == 1
            and positions.is_contiguous()
        ):
            from rtp_llm.models_py.modules.dsv41.ppu.prefill_swa import read_staged_rows

            return read_staged_rows(domain, values, positions)
        index = torch.searchsorted(domain, positions)
        present = (index < domain.numel()) & (domain[index.clamp_max(max(domain.numel() - 1, 0))] == positions)
        result = values.new_zeros((positions.numel(), values.shape[-1]))
        result[present] = values[index[present]]
        return result

    def stage_query(self, layer, positions, requests, packed, prefixes):
        """Call-local draft query KV; committed native feature rings stay live."""
        for rid, prefix in enumerate(prefixes):
            chosen = requests == rid
            old_pos = torch.arange(max(0, int(prefix) - self.geometry.window), int(prefix), device=self.device)
            old = self._copy_rows((layer, "swa_kv"), rid, old_pos)
            self._fresh_swa[layer, rid] = (
                torch.cat((old_pos, positions[chosen])), torch.cat((old, packed[chosen])),
            )

    def release_swa(self, layer):
        for key in tuple(self._fresh_swa):
            if key[0] == layer:
                del self._fresh_swa[key]
        if not self._check_local_source_reads():
            self._source_reads.clear()

    def placement_evidence(self):
        result = []
        for key, pool in self.pools.items():
            placement = self.placements.get(key)
            for rid, length in enumerate(self.lengths):
                positions = self._positions(key, length).to(self.device)
                owned = self.owned_positions(key, positions)
                owned_pos = positions[owned]
                if placement is not None and key[1] not in _FULL:
                    start = (owned_pos % placement.full_entries) * _WIDTH[key[1]]
                    count = (torch.minimum(start + _WIDTH[key[1]], start.new_tensor((placement.rank + 1) * placement.slice_elements))
                             - torch.maximum(start, start.new_tensor(placement.rank * placement.slice_elements))).sum()
                    owned_elements = int(count)
                else:
                    owned_elements = owned_pos.numel() * _WIDTH[key[1]]
                physical_span = placement.physical_span if placement else int(self.native.get_seq_size_per_block(pool.framework_tag))
                logical_blocks = (length + physical_span - 1) // physical_span
                if placement is not None and key[1] in _FULL:
                    reserved = (logical_blocks + placement.size - 1) // placement.size
                    actual = len(range(placement.rank, logical_blocks, placement.size))
                    padding_blocks = reserved - actual
                else:
                    reserved, actual, padding_blocks = logical_blocks, logical_blocks, 0
                result.append({
                    "layer": key[0], "tag": key[1], "request": rid,
                    "framework_tag": pool.framework_tag, "sharded": self.sharded,
                    "cp_size": int(self.cp.get("size", 1)), "cp_rank": int(self.cp.get("rank", 0)),
                    "entries": pool.entries, "full_entries": placement.full_entries if placement else pool.entries,
                    "kernel_span": pool.span, "physical_span": physical_span,
                    "request_reserved_blocks": reserved, "request_owned_blocks": actual,
                    "request_padding_blocks": padding_blocks,
                    "placement_rule": "physical-block-round-robin" if key[1] in _FULL and self.sharded else "native-payload-slice" if key[1] == "csa_state" and self.sharded else "native-stride-slice" if self.sharded else "full-local",
                    "allocated_pool_bytes": pool.tensor.numel() * pool.tensor.element_size(),
                    "local_block_stride_bytes": pool.tensor.shape[1] * pool.tensor.element_size(),
                    "live_rows": positions.numel(), "owned_positions": owned_pos.tolist(),
                    "owned_live_bytes": owned_elements * pool.tensor.element_size(),
                    "logical_row_count_includes_partial_fragments": True,
                    "padding_is_live_content": False,
                })
        return result

    def fingerprint(self, lengths, *, global_only=False):
        """Optional smoke evidence over logical live bytes, independent of pages."""
        result = {}
        for key in self.pools:
            if global_only and key[1] not in _FULL:
                continue
            for rid, length in enumerate(lengths):
                positions = self._positions(key, int(length))
                if not positions.numel():
                    continue
                packed = self._copy_rows(key, rid, positions)
                value = packed.cpu().contiguous().view(torch.uint8)
                result[f"{key[0]}:{key[1]}:{rid}"] = {
                    "positions": positions.tolist(), "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
                    "bytes": value.numel(), "nonzero_bytes": int(torch.count_nonzero(value)),
                }
                if self.sharded:
                    result[f"{key[0]}:{key[1]}:{rid}"]["source_owner"] = self._owner_fingerprint(key, rid, positions, packed)
        return result

    def _owner_fingerprint(self, key, request, positions, canonical):
        """Check actual local fragments against the assembled logical domain."""
        pool, placement = self.pools[key], self.placements[key]
        positions = positions.to(self.device)
        owned = placement.owned(key, positions)
        selected, expected = positions[owned], canonical[owned]
        width = _WIDTH[key[1]]
        fragments = []
        if key[1] in _FULL:
            native = NativeCacheBridge._copy_rows(self, key, request, selected)
            fragments = [{"position": p, "begin": 0, "end": width * pool.tensor.element_size()} for p in selected.tolist()]
            actual_bytes = native.contiguous().view(torch.uint8).reshape(-1)
            expected_bytes = expected.contiguous().view(torch.uint8).reshape(-1)
        else:
            actual, reference = [], []
            for index, position in enumerate(selected.tolist()):
                row_start = (position % placement.full_entries) * width
                begin = max(0, placement.rank * placement.slice_elements - row_start)
                end = min(width, (placement.rank + 1) * placement.slice_elements - row_start)
                block = int(pool.table[request, position // pool.span])
                local = row_start + begin - placement.rank * placement.slice_elements
                actual.append(pool.tensor[block, local:local + end - begin])
                reference.append(expected[index, begin:end])
                fragments.append({"position": position, "begin": begin * pool.tensor.element_size(), "end": end * pool.tensor.element_size()})
            actual_bytes = (torch.cat(actual) if actual else pool.tensor.new_empty(0)).contiguous().view(torch.uint8)
            expected_bytes = (torch.cat(reference) if reference else pool.tensor.new_empty(0)).contiguous().view(torch.uint8)
        if not torch.equal(actual_bytes, expected_bytes):
            raise RuntimeError("native CP owner bytes disagree with canonical handoff domain")
        return {"cp_rank": placement.rank, "cp_size": placement.size, "fragments": fragments,
                "bytes": actual_bytes.numel(), "matches_canonical": True,
                "sha256": hashlib.sha256(actual_bytes.cpu().numpy().tobytes()).hexdigest()}
