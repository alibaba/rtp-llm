"""Direct views of framework pools, with no private cache or allocator.

NativeCacheBridge's coordinate/planar copy primitives are reused, while its
legacy logical-cache construction, import and export are never called.
"""

import hashlib
import logging
import math
import os
from types import SimpleNamespace
from typing import NamedTuple

import torch

from rtp_llm.models_py.modules.dsv41.kv_cache_utils import as_attention_inputs_by_tag
from rtp_llm.models_py.modules.dsv41.ppu.kv_codec import PoolDescriptor
from rtp_llm.models_py.modules.dsv41.ppu.native_cache import (
    NativeCacheBridge,
    _FULL,
    _Pool,
    _WIDTH,
)
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import PrefillPolicy


_LOGGER = logging.getLogger(__name__)
CP8_ROW_PREP_ENV = "RTP_V41_PPU_CP8_PREFILL_ROW_PREP"
CP8_WORKSPACE_DEQUANT_ENV = "RTP_V41_PPU_CP8_FUSED_WORKSPACE_DEQUANT"
CP8_GATHER_RECEIVE_ENV = "RTP_V41_PPU_CP8_REUSE_GATHER_RECEIVE"


class _StagedSwaSegments(NamedTuple):
    prefix: int
    tail: torch.Tensor
    fresh: torch.Tensor


def _latest_csa_indices(blocks, offsets, positions):
    """Stable maximum raw position per real physical slot; ties keep first."""
    latest = {}
    for index, (block, offset, position) in enumerate(zip(blocks, offsets, positions)):
        key = block, offset
        previous = latest.get(key)
        if previous is None or position > positions[previous]:
            latest[key] = index
    return sorted(latest.values())


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
        self.shared_prefix_stripes = None
        self.prefill_plans = None
        self.gather_receive = None
        self.placements = {}
        self.pool_gathers = []
        self.cp = dict(config.get("_ppu_cp_cache", {}))
        self.sharded = bool(self.cp.get("sliced_prefill", False))
        self.writes = []
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
        self._slots = None
        if not defer_producer_slots and not (
            self.cp8_row_prep_enabled()
            and os.environ.get("RTP_V41_PPU_CP8_LAZY_PRODUCER_SLOTS", "0") == "1"
        ):
            self._slots = self._build_producer_slots()

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

    def insert_request_rows(self, layer, tag, request, positions, rows, *, start=None):
        """Write raw positions already grouped by the caller's request."""
        if not positions.numel():
            return
        key = layer, tag
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

    def try_insert_csa_fields(self, layer, request, start, values, scores):
        """Write separate producer fields directly into the existing CSA row."""
        if not (
            values.shape == scores.shape
            and values.dtype in (torch.bfloat16, torch.float32)
            and scores.dtype in (torch.bfloat16, torch.float32)
            and scores.is_contiguous()
            and scores.device == values.device
        ):
            return False
        return self._try_store_prefill_range((layer, "csa_state"), request, start, values, scores=scores)

    def _try_store_prefill_range(self, key, request, start, rows, *, scores=None):
        if (
            os.environ.get("RTP_V41_PPU_CP8_FUSED_COMPRESSED_STORE", "1") != "1"
            or not self.cp8_row_prep_enabled()
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

    def read_source(self, source, request, *, global_entries):
        saved = self._source_reads.get((source, int(request)))
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
        positions = torch.arange(global_entries, device=self.device) * self.geometry.ratio(source)
        tag = "global_kv_2" if self.geometry.ratio(source) == 2 else "global_kv_1"
        return self._copy_rows((source, tag), request, positions), self._copy_rows((source, "indexer_kv"), request, positions)

    def read_global(self, source, request, *, global_entries):
        saved = self._source_reads.get((source, int(request)))
        if saved is not None:
            return saved[0][:global_entries]
        return self.read_source(source, request, global_entries=global_entries)[0]

    def read_indexer_fp4(self, source, request, *, global_entries):
        from rtp_llm.models_py.modules.dsv41.ppu._quant import _bytes_to_sf

        saved = self._source_reads.get((source, int(request)))
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

    def stage_swa(
        self, layer, positions, requests, packed, prefixes, *,
        retain_rows=True, retain_prefix_tail=True, bounded_starts=None,
    ):
        """Preserve the SWA rows visible to this forward before ring writes.

        Bounded starts describe contiguous suffixes in restored request order.
        """
        if not self.lengths:
            return
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
                and os.environ.get("RTP_V41_PPU_CP8_FUSED_SWA_WRITE", "1") == "1"
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
            and os.environ.get(CP8_WORKSPACE_DEQUANT_ENV, "1") == "1"
            and os.environ.get("RTP_V41_PPU_CP8_DIRECT_SWA_WORKSPACE", "1") == "1"
        )
        offset = 0
        for rid in range(len(self.lengths)):
            key = layer, rid
            if fresh_lengths is None:
                chosen = requests == rid
            else:
                stop = offset + fresh_lengths[rid]
                chosen = slice(offset, stop)
                offset = stop
            fresh_pos, fresh = positions[chosen], packed[chosen]
            if retain_rows:
                prefix = int(prefixes[rid])
                if direct_swa:
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
