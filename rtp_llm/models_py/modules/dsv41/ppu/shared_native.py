"""Direct views of framework pools, with no private cache or allocator.

NativeCacheBridge's coordinate/planar copy primitives are reused, while its
legacy logical-cache construction, import and export are never called.
"""

import hashlib
import math
from types import SimpleNamespace

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

    def __init__(self, config, geometry, native, inputs, lengths, *, device, prefill=False):
        self.config, self.native, self.device = config, native, device
        self._is_prefill = prefill
        self.policy = PrefillPolicy.from_config(config)
        self.geometry = geometry
        self.logical = SimpleNamespace(geometry=geometry)
        self.inputs = as_attention_inputs_by_tag(inputs, native)
        self.lengths = tuple(map(int, lengths))
        self.pools = {}
        self._fresh_swa = {}
        self._source_reads = {}
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
                from rtp_llm.models_py.modules.dsv41.ppu.shared_sharded import PoolPlacement

                self.placements[owner, tag] = PoolPlacement.create(
                    self.cp, (owner, tag), self.pools[owner, tag], native, meta, geometry,
                )
        self.slots = {
            rid: SimpleNamespace(global_slots={
                source: self.slots_at_positions(source, "indexer_kv", rid, torch.arange(n // geometry.ratio(source), device=device) * geometry.ratio(source))
                for source in geometry.kv_source_layer_ids
            }) for rid, n in enumerate(self.lengths)
        }
        for handle in self.slots.values():
            handle.indexer_slots = handle.global_slots

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
            physical_rows = None
            if tag == "csa_state":
                positions, values = slots[mask, 1], rows[mask]
                if getattr(self, "_is_prefill", False):
                    length = self.lengths[rid]
                    keep = self._prefill_ring_keep(
                        self.pools[layer, tag], rid, positions, length - length % 2,
                    )
                    positions, values = positions[keep], values[keep]
                owned = self.owned_positions((layer, tag), positions)
                positions, values = positions[owned], values[owned]
                blocks, offsets = self._coordinates((layer, tag), rid, positions)
                keep = _latest_csa_indices(blocks.cpu().tolist(), offsets.cpu().tolist(), positions.cpu().tolist())
                selected = torch.tensor(keep, device=positions.device, dtype=torch.long)
                # Every destination appears once, including aliased page-table
                # columns and unordered CP rows. Never rely on index_put order.
                self._copy_rows((layer, tag), rid, positions[selected], values[selected])
                physical_rows = len(keep)
            else:
                self._copy_rows((layer, tag), rid, slots[mask, 1], rows[mask])
            write = {"layer": layer, "tag": tag, "request": int(rid), "rows": int(mask.sum())}
            write["owned_rows"] = int(self.owned_positions((layer, tag), slots[mask, 1]).sum())
            if physical_rows is not None:
                # Preserve logical producer coverage; report retained writes too.
                write["physical_rows"] = physical_rows
            self.writes.append(write)

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
            return saved[0][:global_entries], saved[1][:global_entries]
        positions = torch.arange(global_entries, device=self.device) * self.geometry.ratio(source)
        tag = "global_kv_2" if self.geometry.ratio(source) == 2 else "global_kv_1"
        return self._copy_rows((source, tag), request, positions), self._copy_rows((source, "indexer_kv"), request, positions)

    def prepare_source(self, source):
        """Uniform collectives before any rank-local selector/consumer branch."""
        if self.sharded:
            for rid, length in enumerate(self.lengths):
                self._source_reads[source, rid] = self.read_source(
                    source, rid, global_entries=length // self.geometry.ratio(source),
                )

    def read_csa(self, layer, request, positions):
        return self._copy_rows((layer, "csa_state"), request, positions)

    def stage_swa(self, layer, positions, requests, packed, prefixes, *, retain_rows=True):
        """Preserve prefix tails and fresh rows before any ring is overwritten."""
        for rid in range(len(self.lengths)):
            key = layer, rid
            chosen = requests == rid
            fresh_pos, fresh = positions[chosen], packed[chosen]
            if retain_rows:
                prefix = int(prefixes[rid])
                old_pos = torch.arange(max(0, prefix - self.geometry.window + 1), prefix, device=self.device)
                old = self._copy_rows((layer, "swa_kv"), rid, old_pos)
                self._fresh_swa[key] = (torch.cat((old_pos, fresh_pos)), torch.cat((old, fresh)))
            pool = self.pools[layer, "swa_kv"]
            # One write per ring slot; retain each allocated block's checkpoint.
            column_end = ((fresh_pos // pool.span + 1) * pool.span).clamp_max(self.lengths[rid])
            keep = fresh_pos >= column_end - pool.entries
            if getattr(self, "_is_prefill", False):
                # The scheduler leaves historical SWA columns at -1. Those
                # rows remain available to this call through _fresh_swa.
                keep &= self._prefill_ring_keep(
                    pool, rid, fresh_pos, self.lengths[rid] - self.geometry.window,
                )
            self._copy_rows((layer, "swa_kv"), rid, fresh_pos[keep], fresh[keep])
            self.writes.append({"layer": layer, "tag": pool.framework_tag, "request": rid,
                                "rows": int(keep.sum()), "absolute_positions": fresh_pos[keep].tolist(),
                                "owned_rows": int(self.owned_positions((layer, "swa_kv"), fresh_pos[keep]).sum())})

    def read_swa(self, layer, request, positions):
        saved = self._fresh_swa.get((layer, int(request)))
        if saved is None:
            return self._copy_rows((layer, "swa_kv"), request, positions)
        domain, values = saved
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
