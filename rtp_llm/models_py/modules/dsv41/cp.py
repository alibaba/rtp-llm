"""CP8 target prefill over framework-owned V4.1 cache shards.

Queries and model rows remain in the framework's padded zigzag order. Only
compressor inputs, SWA windows, and selected compact KV rows cross the TP/CP
group. All receive storage is scoped to a source tile and its consumers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace

import torch
import torch.nn.functional as F

from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.modules.dsv4.cp import (
    CPContext,
    _cp_all_gather_into_empty,
    cp_all_gather_full,
)
from rtp_llm.models_py.modules.dsv4.fp8._cp_slot_mapping import cp_kv_slot_mapping
from rtp_llm.models_py.modules.dsv41.attention import (
    AttentionOwnerCache,
    V41AttentionCache,
    V41AttentionContext,
    attention_rope,
    is_supported,
)
from rtp_llm.models_py.modules.dsv41.cache_layout import (
    DRAFT_LAYERS,
    GLOBAL_OWNERS,
    PAIR_OWNERS,
    SWA_WINDOW,
    CacheRegion,
    RegionSlot,
    layer_sources,
)
from rtp_llm.models_py.modules.dsv41.ced import (
    AuxRowMap,
    ReplayConfig,
    ReplayMode,
    RowRange,
)
from rtp_llm.models_py.modules.dsv41.compact_reader import (
    MAX_GATHER_BYTES,
    CompactPages,
    GlobalBinding,
    ReaderResult,
    SwaBinding,
    compact_attention,
)
from rtp_llm.models_py.modules.dsv41.compact_writer import encode_compact
from rtp_llm.models_py.modules.dsv41.compressor import (
    OwnerPageBinding,
    PairCarry,
    prepare_owner_kv,
)
from rtp_llm.models_py.modules.dsv41.cprr_reader import (
    CprrReadIdentity,
    CprrReaderLease,
    bind_cprr_paged,
    restore_cprr_swa,
)
from rtp_llm.models_py.modules.dsv41.indexer import (
    CANDIDATE_BLOCKS,
    INDEX_TOPK,
    QUERY_TILE,
    SPARSE_BLOCK,
    IndexSelection,
    score_candidate_tile,
)
from rtp_llm.models_py.modules.dsv41.math import grouped_wo_a
from rtp_llm.models_py.modules.dsv41.decode_compressor import normalize_empty_pair_checkpoint
from rtp_llm.models_py.modules.dsv41.source_indexer import (
    SOURCE_QUERY_TILE,
    prepare_index_source,
    score_index_source,
)

_PAIR_BYTES = 4112
_SOURCE_ROWS = 512
_READ_QUERIES = 4
_REINDEX_BLOCKS = 512
_INDEX_ROWS = CANDIDATE_BLOCKS * SPARSE_BLOCK
# Per gathered row: int64 page/index vectors and transient mask/index results.
_SELECTED_METADATA_BYTES = 80


def _check(condition, message):
    if not bool(condition.all().item()):
        raise ValueError(message)


def _bytes(*tensors):
    return sum(value.numel() * value.element_size() for value in tensors)


@dataclass(frozen=True)
class V41CPSwaState:
    """Published range metadata; payload ownership stays in the framework pool."""

    page_ids: torch.Tensor
    valid_starts: torch.Tensor
    valid_ends: torch.Tensor


class V41CPAttentionContext(V41AttentionContext):
    def __init__(
        self,
        cache,
        cp,
        request_index,
        pools,
        tables,
        pair_pools,
        pair_tables,
        epoch,
        replay_floor=0,
    ):
        lengths = cp.input_lengths_global_host
        chunks = cp.chunk_lengths_per_req
        prefixes = cp.prefix_lengths_host
        if lengths is None or chunks is None or prefixes is None:
            raise ValueError("CP requests require the framework's host length metadata")
        if not 0 <= request_index < len(lengths):
            raise ValueError("CP request index exceeds its batched metadata")
        self.cp = cp
        self.request_index = request_index
        self.local_first = sum(chunks[:request_index])
        self.local_last = self.local_first + chunks[request_index]
        self.local_count = chunks[request_index]
        self.positions = cp.global_positions[self.local_first : self.local_last]
        self.valid = cp.local_is_real[self.local_first : self.local_last]
        start, end = (
            prefixes[request_index],
            prefixes[request_index] + lengths[request_index],
        )
        cache.active_epoch = epoch
        super().__init__(cache, epoch, start, end, replay_floor)
        self.decoder_only = False
        self.parent_row_indices = None
        self.parent_query_rows = None
        self.framework_row_indices = None
        self.framework_query_rows = None
        self._framework_context = None
        self.pools, self.tables = pools, tables
        self.pair_pools, self.pair_tables = pair_pools, pair_tables
        self.current = (end - 1) // cache.layout.reuse_unit
        self.previous = (start - 1) // cache.layout.reuse_unit
        self._page_specs = {page.slot: page for page in cache.layout.pages}
        self._pair_snapshots = {
            state.owner_layer: state.snapshots for state in cache.layout.pair_states
        }
        self._pair_initials = {}
        self._gather_plans = {}
        self.max_receive_bytes = 0
        self.max_gather_live_bytes = 0
        self.gather_count = 0
        self.read_identity = CprrReadIdentity(
            cache.request_id, cache.identity, epoch, start, end
        )
        begin = sum(lengths[:request_index])
        restore = cp.unpad_restore[begin : begin + end - start].detach().cpu().tolist()
        mapping = []
        for flat in restore:
            rank, local = divmod(flat, cp.chunk_length)
            local -= self.local_first
            if not 0 <= rank < 8 or not 0 <= local < self.local_count:
                raise ValueError("CP restore map crosses a request's local row range")
            mapping.append(rank * self.local_count + local)
        if len(set(mapping)) != end - start:
            raise ValueError("CP restore map repeats canonical source rows")
        self._restore_host = tuple(mapping)
        device = self.query_device
        self._restore = torch.tensor(mapping, dtype=torch.int64, device=device)
        rank_positions = torch.full(
            (8 * self.local_count,), -1, dtype=torch.int64, device=device
        )
        rank_positions[self._restore] = torch.arange(start, end, device=device)
        self._rank_positions = rank_positions.view(8, self.local_count)
        expected = self._rank_positions[cp.cp_rank]
        _check(
            (self.valid == (expected >= 0))
            & (~self.valid | (self.positions == expected)),
            "CP query positions/validity differ from canonical source restoration",
        )

    @property
    def query_rows(self):
        return self.local_count

    @property
    def query_device(self):
        return self.cp.global_positions.device

    @property
    def query_identity(self):
        return super().query_identity + (
            self.cp.cp_rank,
            self.local_first,
            self.local_last,
        )

    def tail(self, start, *, replay_floor):
        """Retain a current-chunk tail while preserving each token's CP owner."""
        self.validate()
        if not self.start <= start < self.end:
            raise ValueError("CP tail rows must belong to the current encoder chunk")
        selected = self.selection_for(20)
        rank_rows = [[] for _ in range(8)]
        for position in range(start, self.end):
            rank, local = divmod(
                self._restore_host[position - self.start], self.local_count
            )
            rank_rows[rank].append((position, local))
        context = self._remap_tail(start, replay_floor, rank_rows, self.query_rows)
        context.publish_selection(
            IndexSelection(
                context.select_rows(selected.topk, fill=-1),
                context.select_rows(selected.candidate_blocks, fill=-1),
                context.select_rows(selected.status),
                20,
                20,
                0,
                0,
                0,
            )
        )
        return context

    def _remap_tail(self, start, replay_floor, rank_rows, parent_query_rows):
        if (
            self.cache.identity.replay_fingerprint
            != ReplayConfig(ReplayMode.BOUNDED).fingerprint
            or not 0 <= replay_floor <= start < self.end
            or self.end - start > SWA_WINDOW
            or len(rank_rows) != 8
        ):
            raise ValueError(
                "CP late execution requires an explicit bounded 128-row plan"
            )
        return self._remap_range(
            start,
            self.end,
            replay_floor,
            rank_rows,
            parent_query_rows,
            epoch=self.epoch,
            decoder_only=True,
        )

    def encoder_slice(self, start, end, *, epoch):
        """Split an engine chunk at N without changing its canonical CP ownership."""
        if (
            self.cache.poisoned
            or self.decoder_only
            or not self.start <= start < end <= self.end
            or type(epoch) is not int
            or epoch <= self.cache.active_epoch
        ):
            raise ValueError(
                "CP encoder split requires a fresh contiguous execution epoch"
            )
        if any(owner.materialized_end != start for owner in self.cache.owners.values()):
            raise ValueError("CP encoder split must follow completed source writes")
        rank_rows = [[] for _ in range(8)]
        for position in range(start, end):
            rank, local = divmod(
                self._restore_host[position - self.start], self.local_count
            )
            rank_rows[rank].append((position, local))
        result = self._remap_range(
            start, end, 0, rank_rows, self.query_rows, epoch=epoch, decoder_only=False
        )
        result._pair_initials = {
            owner: self.cache.owners[owner].pair for owner in PAIR_OWNERS
        }
        return result

    def _remap_range(
        self,
        start,
        end,
        replay_floor,
        rank_rows,
        parent_query_rows,
        *,
        epoch,
        decoder_only,
    ):
        all_positions = sorted(
            position for values in rank_rows for position, _ in values
        )
        if all_positions != list(range(start, end)):
            raise ValueError(
                "CP retained tail must own every canonical token exactly once"
            )
        capacity = max(map(len, rank_rows))
        device = self.query_device
        positions = torch.full((8, capacity), end - 1, dtype=torch.int64, device=device)
        valid = torch.zeros((8, capacity), dtype=torch.bool, device=device)
        restore = torch.empty(end - start, dtype=torch.int64, device=device)
        for rank, values in enumerate(rank_rows):
            absolute = torch.tensor(
                [position for position, _ in values], dtype=torch.int64, device=device
            )
            positions[rank, : len(values)] = absolute
            valid[rank, : len(values)] = True
            restore[absolute - start] = rank * capacity + torch.arange(
                len(values), device=device
            )
        local_positions = positions[self.cp.cp_rank].contiguous()
        cp = replace(
            self.cp,
            chunk_length=capacity,
            padded_seq_len=8 * capacity,
            seq_len_full=end - start,
            relative_positions=(local_positions - start).contiguous(),
            prefix_length=start,
            global_positions=local_positions,
            local_is_real=valid[self.cp.cp_rank].contiguous(),
            unpad_restore=restore,
            seq_len_total=end,
            req_id_per_token=torch.zeros(capacity, dtype=torch.int32, device=device),
            prefix_lengths=torch.tensor([start], dtype=torch.int64, device=device),
            input_lengths_global=torch.tensor(
                [end - start], dtype=torch.int32, device=device
            ),
            cu_seqlens_global=torch.tensor(
                [0, end - start], dtype=torch.int32, device=device
            ),
            unpad_restore_is_prefix=False,
            chunk_lengths_per_req=(capacity,),
            input_lengths_global_host=(end - start,),
            prefix_lengths_host=(start,),
        )
        context = V41CPAttentionContext(
            self.cache,
            cp,
            0,
            self.pools,
            self.tables,
            self.pair_pools,
            self.pair_tables,
            epoch,
            replay_floor,
        )
        context.decoder_only = decoder_only
        context.parent_query_rows = parent_query_rows
        context.parent_row_indices = torch.full(
            (capacity,), -1, dtype=torch.int64, device=device
        )
        rows = rank_rows[self.cp.cp_rank]
        context.parent_row_indices[: len(rows)] = torch.tensor(
            [local for _, local in rows], dtype=torch.int64, device=device
        )
        # Retained decoder rows may precede this encoder slice but still belong
        # to the same engine input, whose sampler needs their original row IDs.
        framework = self if self._framework_context is None else self._framework_context
        context._framework_context = framework
        context.framework_query_rows = framework.query_rows
        context.framework_row_indices = torch.full_like(context.parent_row_indices, -1)
        for offset, (position, _) in enumerate(rows):
            if framework.start <= position < framework.end:
                rank, local = divmod(
                    framework._restore_host[position - framework.start],
                    framework.local_count,
                )
                if rank != self.cp.cp_rank:
                    raise ValueError(
                        "CP retained tail changed ownership of a current input row"
                    )
                context.framework_row_indices[offset] = local
        if decoder_only:
            context.published_sources = set(self.published_sources)
        return context

    def select_rows(self, tensor, *, fill=0):
        """Copy only retained local rows; padding never retains the encoder backing."""
        self.validate()
        if (
            self.parent_row_indices is None
            or tensor.ndim < 1
            or tensor.shape[0] != self.parent_query_rows
            or tensor.device != self.query_device
        ):
            raise ValueError("CP tail row selection requires its declared local parent")
        output = torch.full(
            (self.query_rows, *tensor.shape[1:]),
            fill,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        valid = self.parent_row_indices >= 0
        output[valid] = tensor.index_select(0, self.parent_row_indices[valid])
        return output

    def select_model_rows(self, rows):
        rows = type(rows)(
            *(
                self.select_rows(
                    getattr(rows, item.name),
                    fill=-1 if item.name == "token_types" else 0,
                )
                for item in fields(rows)
            )
        )
        rows.validate()
        return rows

    def select_image_features(self, features):
        if features is None:
            return None
        inverse = torch.full(
            (self.parent_query_rows,), -1, dtype=torch.int64, device=self.query_device
        )
        selected = self.parent_row_indices >= 0
        inverse[self.parent_row_indices[selected]] = torch.arange(
            self.query_rows, device=self.query_device
        )[selected]
        remapped = inverse[features.row_indices]
        keep = (remapped >= 0).nonzero().flatten()
        keep = keep[remapped[keep].argsort()]
        return replace(
            features,
            row_indices=remapped[keep].contiguous(),
            token_types=features.token_types[keep].contiguous(),
            values=features.values[keep].contiguous(),
        )

    def select_l20(self, l20):
        return replace(
            l20,
            rows=self.select_model_rows(l20.rows),
            hidden_states=self.select_rows(l20.hidden_states),
            pre_mix=self.select_rows(l20.pre_mix),
        )

    def scatter_rows(self, tensor, *, output=None):
        """Restore selected late output positions for the framework's existing sampler."""
        self.validate()
        if self.framework_row_indices is None or tensor.shape[0] != self.query_rows:
            raise ValueError("CP tail output must match the selected local row map")
        shape = (self.framework_query_rows, *tensor.shape[1:])
        if output is None:
            output = tensor.new_zeros(shape)
        if (
            output.shape != shape
            or output.device != tensor.device
            or output.dtype != tensor.dtype
        ):
            raise ValueError(
                "CP tail output destination has a different parent geometry"
            )
        valid = self.framework_row_indices >= 0
        output.index_copy_(0, self.framework_row_indices[valid], tensor[valid])
        return output

    def for_decoder(self, extend):
        """Apply the planner's actual late-stage gate after full source production."""
        self.validate()
        if (
            extend.encoder_rows != RowRange(self.start, self.end)
            or not set(range(21)).issubset(self.completed_layers)
            or self.published_sources != {2, 8, 14, 20}
        ):
            raise ValueError(
                "CP stage gating requires completed L0-L20 and all sources"
            )
        if extend.decoder_rows is None:
            if (
                self.cache.identity.replay_fingerprint
                != ReplayConfig(ReplayMode.BOUNDED).fingerprint
            ):
                raise ValueError("only bounded mode may omit the late decoder stage")
            self.observations.append(
                {
                    "phase": "decoder_gate",
                    "encoder_start": self.start,
                    "encoder_end": self.end,
                    "decoder_rows": 0,
                    "omitted_layers": 19,
                }
            )
            return None
        if extend.decoder_rows.end != self.end:
            raise ValueError(
                "CP decoder consumers must end at the current source boundary"
            )
        if (
            self.cache.identity.replay_fingerprint
            == ReplayConfig(ReplayMode.FULL).fingerprint
        ):
            if extend.decoder_rows != extend.encoder_rows or extend.replay_floor != 0:
                raise ValueError("full CP execution cannot truncate decoder rows")
            return self
        return self.tail(extend.decoder_rows.start, replay_floor=extend.replay_floor)

    def _record_gather(self, receive_bytes, live_bytes):
        if live_bytes > MAX_GATHER_BYTES:
            raise ValueError(
                "CP receive lifetime exceeds 64 MiB; reduce the source tile"
            )
        self.max_receive_bytes = max(self.max_receive_bytes, receive_bytes)
        self.max_gather_live_bytes = max(self.max_gather_live_bytes, live_bytes)
        self.gather_count += 1

    def _all_gather(self, local, *, retained_bytes=0, restored_bytes=0):
        receive_bytes = 8 * _bytes(local)
        self._record_gather(
            receive_bytes,
            receive_bytes + _bytes(local) + restored_bytes + retained_bytes,
        )
        return _cp_all_gather_into_empty(local.contiguous(), Group.TP)

    def gather_rows(self, local_rows, first, last):
        """Gather one bounded absolute canonical range from this request's rows."""
        self.validate()
        if (
            local_rows.ndim != 2
            or local_rows.shape[0] != self.local_count
            or local_rows.device != self.query_device
            or not self.start <= first <= last <= self.end
        ):
            raise ValueError(
                "CP source gather requires request-local rows and canonical bounds"
            )
        if first == last:
            return local_rows.new_empty((0, local_rows.shape[1]))
        key = (first, last)
        if key not in self._gather_plans:
            source = self._restore_host[first - self.start : last - self.start]
            per_rank = [[] for _ in range(8)]
            for flat in source:
                rank, local = divmod(flat, self.local_count)
                per_rank[rank].append(local)
            capacity = max(map(len, per_rank))
            inverse = {
                rank * self.local_count + local: rank * capacity + offset
                for rank, values in enumerate(per_rank)
                for offset, local in enumerate(values)
            }
            indices = torch.tensor(
                per_rank[self.cp.cp_rank], dtype=torch.int64, device=self.query_device
            )
            restore = torch.tensor(
                [inverse[flat] for flat in source],
                dtype=torch.int64,
                device=self.query_device,
            )
            self._gather_plans[key] = (capacity, indices, restore)
        capacity, indices, restore = self._gather_plans[key]
        row_bytes = local_rows.shape[1] * local_rows.element_size()
        self._record_gather(
            8 * capacity * row_bytes,
            (10 * capacity + last - first) * row_bytes,
        )
        packed = local_rows.new_zeros((capacity, local_rows.shape[1]))
        packed[: indices.numel()].copy_(local_rows.index_select(0, indices))
        tile_cp = replace(
            self.cp,
            chunk_length=capacity,
            padded_seq_len=8 * capacity,
            seq_len_full=last - first,
            unpad_restore=restore,
            unpad_restore_is_prefix=False,
        )
        return cp_all_gather_full(packed, tile_cp, profile_name="dsv41.cp.source")

    def _physical(self, table, logical, pool):
        if not 0 <= logical < table.shape[1]:
            raise ValueError("CP fixed state is missing its canonical checkpoint page")
        page = int(table[0, logical].item())
        if not 0 < page < pool.shape[0]:
            raise ValueError("CP fixed state requires an allocated rank-local page")
        return page

    def _fixed_receive(self, pool, table, logical):
        page = self._physical(table, logical, pool)
        local = torch.zeros((2, pool.shape[1]), dtype=torch.uint8, device=pool.device)
        local[1].copy_(pool[page])
        return self._all_gather(local, restored_bytes=8 * pool.shape[1]).view(
            8, 2, pool.shape[1]
        )

    def restore_swa(self, layer):
        slot = RegionSlot(CacheRegion.SWA, layer)
        spec, pool, table = self._page_specs[slot], self.pools[slot], self.tables[slot]
        if self.start and self.replay_floor < self.start:
            if self.cache.swa_ends.get(layer, 0) != self.start:
                raise ValueError(
                    "short CP decoder continuation requires complete prior SWA state"
                )
            received = self._fixed_receive(pool, table, self.previous)
            restored = restore_cprr_swa(
                self.cache.layout,
                layer,
                received,
                torch.ones((8, 1), dtype=torch.int32, device=self.query_device),
            )
            restored.check()
            pages = restored.pages
        else:
            pages = CompactPages(
                torch.zeros(
                    (2, spec.page_stride_bytes),
                    dtype=torch.uint8,
                    device=self.query_device,
                ),
                CacheRegion.SWA,
                spec.entries,
            )
        return SwaBinding(
            pages,
            torch.ones(1, dtype=torch.int32, device=self.query_device),
            torch.tensor(
                [max(self.replay_floor, self.start - spec.entries)],
                dtype=torch.int32,
                device=self.query_device,
            ),
            torch.tensor([self.start], dtype=torch.int32, device=self.query_device),
        )

    def publish_swa(self, layer, initial, encoded, *, first=None):
        spec = self._page_specs[RegionSlot(CacheRegion.SWA, layer)]
        if first is None:
            first = max(self.start, self.end - spec.entries)
        if not max(self.start, self.end - spec.entries) <= first < self.end:
            raise ValueError("CP SWA publication exceeds the computed ring tail")
        rows = self.gather_rows(encoded, first, self.end)
        positions = torch.arange(first, self.end, device=self.query_device)
        data = initial.pages.data[1, : spec.entries * 528].view(spec.entries, 528)
        data.index_copy_(0, positions % spec.entries, rows)
        slot = spec.slot
        destination = self._physical(self.tables[slot], self.current, self.pools[slot])
        begin, end = spec.swa_byte_slice(self.cp.cp_rank)
        self.pools[slot][destination].copy_(initial.pages.data[1, begin:end])
        self.cache.swa[layer] = V41CPSwaState(
            self.tables[slot][0, self.current : self.current + 1],
            torch.tensor(
                [
                    max(
                        self.replay_floor,
                        self.end - spec.entries,
                        int(initial.valid_starts[0]) if first == self.start else first,
                    )
                ],
                dtype=torch.int32,
                device=self.query_device,
            ),
            torch.tensor([self.end], dtype=torch.int32, device=self.query_device),
        )
        self.cache.swa_ends[layer] = self.end

    def commit_draft(self, draft_commit, output, rows):
        """Project real local aux, then publish the complete three-stage CP shards."""
        self.validate()
        if not self.cache.layout.draft_enabled or draft_commit is None:
            raise ValueError("CP draft history requires all three commit writers")
        indices = output.aux_row_indices
        expected = rows.valid.nonzero().flatten()
        if indices is None or not torch.equal(indices, expected):
            raise ValueError(
                "CP draft aux must contain precisely the valid computed rows"
            )
        positions = tuple(self.positions[indices].cpu().tolist())
        row_map = AuxRowMap(
            self.cache.request_id,
            self.epoch,
            self.cache.identity.replay_fingerprint,
            positions,
            positions,
            tuple(rows.image_mask[indices].cpu().tolist()),
        )
        first = max(self.start, self.end - SWA_WINDOW)
        required = tuple(position for position in positions if position >= first)
        bindings = {layer: self.restore_swa(layer) for layer in DRAFT_LAYERS}
        result = draft_commit.commit(
            output.aux_hidden_states,
            row_map,
            required_positions=required,
            swa_bindings=bindings,
            request_id=self.cache.request_id,
            forward_epoch=self.epoch,
            replay_fingerprint=self.cache.identity.replay_fingerprint,
            replay_floor=self.replay_floor,
        )
        if not result.write_completed or result.positions != required:
            raise ValueError(
                "CP draft writer did not complete every selected local row"
            )
        for layer, binding in bindings.items():
            encoded = (
                binding.pages.data[1, : binding.pages.entries_per_page * 528]
                .view(-1, 528)
                .index_select(0, self.positions % binding.pages.entries_per_page)
            )
            encoded.masked_fill_(~self.valid[:, None], 0)
            self.publish_swa(layer, binding, encoded, first=first)
        return row_map, result

    def restore_pair(self, owner, *, restored_state_ready=False):
        if not self.start:
            return PairCarry.empty(owner, self.cache.request_id, self.cache.identity)
        received = self._fixed_receive(
            self.pair_pools[owner], self.pair_tables[owner], self.previous
        )
        first = (self._pair_snapshots[owner] - 1) * _PAIR_BYTES
        region = received[:, 1].contiguous().view(-1)
        raw = region[first : first + _PAIR_BYTES]
        normalize_empty_pair_checkpoint(
            region[None, :],
            raw[None, :],
            torch.tensor([self.start], dtype=torch.int64, device=self.query_device),
            torch.tensor([restored_state_ready], dtype=torch.bool, device=self.query_device),
            reuse_unit=self.cache.layout.reuse_unit,
        )
        position = int(raw[4096:4104].view(torch.int64).item())
        valid = int(raw[4104:4108].view(torch.int32).item())
        if position != self.start or valid != position % 2:
            raise ValueError(
                "CP pair slices disagree with the restored execution boundary"
            )
        return PairCarry(
            owner,
            self.cache.request_id,
            self.cache.identity,
            position,
            raw[:2048].view(torch.float32).clone() if valid else None,
            raw[2048:4096].view(torch.float32).clone() if valid else None,
        )

    def publish_pair(self, owner, pair):
        pool = self.pair_pools[owner]
        raw = torch.zeros(
            pool.shape[1] * 8, dtype=torch.uint8, device=self.query_device
        )
        for snapshot, value in (
            (0, self._pair_initials[owner]),
            (self._pair_snapshots[owner] - 1, pair),
        ):
            view = raw[snapshot * _PAIR_BYTES : (snapshot + 1) * _PAIR_BYTES]
            value.validate(
                owner,
                self.cache.request_id,
                self.cache.identity,
                value.next_position,
                self.query_device,
            )
            if value.next_position % 2:
                view[:2048].view(torch.float32).copy_(value.partial_kv)
                view[2048:4096].view(torch.float32).copy_(value.partial_score)
            view[4096:4104].view(torch.int64).fill_(value.next_position)
            view[4104:4108].view(torch.int32).fill_(value.next_position % 2)
        destination = self._physical(self.pair_tables[owner], self.current, pool)
        first = self.cp.cp_rank * pool.shape[1]
        pool[destination].copy_(raw[first : first + pool.shape[1]])

    def lease(self, slot, resources, layer):
        lease = CprrReaderLease(
            self.read_identity, self.cache.layout, slot, resources, (layer,)
        )
        lease.acquire(layer, self.read_identity)
        return lease

    def release(self, lease, layer):
        lease.complete(layer, self.read_identity)
        lease.release(self.read_identity)

    def gather_paged(self, slot, first_entry, last_entry, layer):
        """Bind a bounded index scan without assuming peer physical page IDs."""
        spec, pool, table = self._page_specs[slot], self.pools[slot], self.tables[slot]
        if first_entry % (spec.entries * 8):
            raise ValueError("CP scan windows must align to a complete RR page cycle")
        first_virtual = first_entry // (spec.entries * 8)
        virtual_count = (last_entry - first_entry + spec.entries * 8 - 1) // (
            spec.entries * 8
        )
        local = torch.zeros(
            (virtual_count + 1, spec.page_stride_bytes),
            dtype=torch.uint8,
            device=self.query_device,
        )
        ids = table[0, first_virtual : first_virtual + virtual_count].long()
        if ids.numel() != virtual_count:
            raise ValueError("CP index scan exceeds the request's allocated page table")
        _check((ids >= 0) & (ids < pool.shape[0]), "invalid CP index physical page ID")
        local[1:].copy_(pool.index_select(0, ids))
        received = self._all_gather(local).view(
            8, virtual_count + 1, spec.page_stride_bytes
        )
        remapped = torch.where(
            ids > 0, torch.arange(1, virtual_count + 1, device=self.query_device), 0
        ).to(torch.int32)
        rank_tables = self._all_gather(
            remapped[None, :], retained_bytes=_bytes(received)
        ).view(8, 1, virtual_count)
        restored = bind_cprr_paged(self.cache.layout, slot, received, rank_tables)
        restored.check()
        return restored, self.lease(
            slot, (received, restored.page_table, restored.status), layer
        )

    def gather_selected(self, slot, positions, layer, *, retained_bytes=0):
        """Receive compact rows selected by each rank, retaining original quantization."""
        spec, pool, table = self._page_specs[slot], self.pools[slot], self.tables[slot]
        requests, width = positions.shape
        wanted = self._all_gather(positions, retained_bytes=retained_bytes).reshape(-1)
        output_bytes = requests * width * spec.encoding.entry_bytes
        self._record_gather(
            output_bytes,
            retained_bytes
            + _bytes(positions, wanted)
            + 9 * output_bytes
            + wanted.numel() * _SELECTED_METADATA_BYTES,
        )
        logical = wanted.clamp_min(0).long() // spec.entries
        owners, virtual = logical % 8, logical // 8
        owned = (wanted >= 0) & (owners == self.cp.cp_rank)
        in_table = virtual < table.shape[1]
        ids = table[0].index_select(0, virtual.clamp_max(table.shape[1] - 1)).long()
        _check(
            ~owned | (in_table & (ids > 0) & (ids < pool.shape[0])),
            "CP selected KV row has no allocated owner page",
        )
        page_rows = pool[:, : spec.entries * spec.encoding.entry_bytes].view(
            pool.shape[0], spec.entries, spec.encoding.entry_bytes
        )
        local = page_rows[
            ids.clamp(0, pool.shape[0] - 1), wanted.clamp_min(0).long() % spec.entries
        ]
        local.masked_fill_(~owned[:, None], 0)
        del logical, owners, virtual, owned, in_table, ids
        values = torch.empty(
            (requests * width, spec.encoding.entry_bytes),
            dtype=torch.uint8,
            device=self.query_device,
        )
        # Each byte has exactly one page owner; SUM preserves its bit pattern.
        # Rank-major requests make each scatter chunk the requesting rank's rows.
        torch.distributed.reduce_scatter_tensor(
            values, local, group=collective_torch._get_group(Group.TP)
        )
        values = values.view(requests, width, -1)
        values.masked_fill_((positions < 0)[:, :, None], 0)
        return values, self.lease(slot, (values,), layer)

    def swa_queries(self, layer, first, last, initial, encoded):
        """Restore each local query's causal ring before any later row overwrites it."""
        query_positions = self._rank_positions[:, first:last]
        offsets = torch.arange(1 - SWA_WINDOW, 1, device=self.query_device)
        wanted = (query_positions[:, :, None] + offsets).reshape(-1)
        active = (
            (query_positions[:, :, None] >= 0).expand(-1, -1, SWA_WINDOW).reshape(-1)
        )
        valid = active & (wanted >= self.replay_floor) & (wanted >= 0)
        current = valid & (wanted >= self.start)
        source = self._restore.index_select(
            0, (wanted - self.start).clamp(0, self.end - self.start - 1)
        )
        owners = torch.where(current, source // self.local_count, 0)
        local_indices = source % self.local_count
        local = encoded.index_select(0, local_indices).clone()
        past = valid & ~current
        _check(
            ~past | (wanted >= initial.valid_starts[0]),
            "CP query is missing restored SWA history",
        )
        old = initial.pages.data[1, : initial.pages.entries_per_page * 528].view(
            -1, 528
        )
        local[past] = old.index_select(0, wanted[past] % initial.pages.entries_per_page)
        local.masked_fill_((~valid | (owners != self.cp.cp_rank))[:, None], 0)
        receive = self._all_gather(
            local,
            retained_bytes=_bytes(initial.pages.data),
            restored_bytes=(last - first) * SWA_WINDOW * 528,
        ).view(8, wanted.numel(), 528)
        begin = self.cp.cp_rank * (last - first) * SWA_WINDOW
        row = torch.arange(
            begin, begin + (last - first) * SWA_WINDOW, device=self.query_device
        )
        values = receive[owners[begin : begin + row.numel()], row]
        values.masked_fill_(~valid[begin : begin + row.numel(), None], 0)
        spec = self._page_specs[RegionSlot(CacheRegion.SWA, layer)]
        pages = CompactPages(
            torch.zeros(
                (last - first + 1, spec.page_stride_bytes),
                dtype=torch.uint8,
                device=self.query_device,
            ),
            CacheRegion.SWA,
            spec.entries,
        )
        page_ids = torch.arange(
            1, last - first + 1, dtype=torch.int32, device=self.query_device
        )
        tokens = wanted[begin : begin + row.numel()].view(last - first, SWA_WINDOW)
        columns = torch.arange(528, device=self.query_device)
        byte_slots = (tokens.clamp_min(0) % spec.entries)[:, :, None] * 528 + columns
        # Invalid positions must not overwrite a real slot zero near sequence start.
        for query in range(last - first):
            keep = valid[begin : begin + row.numel()].view(last - first, SWA_WINDOW)[
                query
            ]
            pages.data[query + 1, byte_slots[query, keep]] = values.view(
                last - first, SWA_WINDOW, 528
            )[query, keep]
        positions = query_positions[self.cp.cp_rank]
        binding = SwaBinding(
            pages,
            page_ids,
            torch.where(
                positions >= 0,
                (positions - SWA_WINDOW + 1).clamp_min(self.replay_floor),
                0,
            ).to(torch.int32),
            (positions + 1).clamp_min(0).to(torch.int32),
        )
        lease = self.lease(spec.slot, (receive, values, pages.data), layer)
        self.release(lease, layer)
        return binding


@dataclass(frozen=True)
class V41CPL20Tail:
    """At most 128 canonical L20 rows, stored only on their original CP ranks."""

    request_id: str
    cache_fingerprint: str
    epoch: int
    positions: RowRange
    cp_rank: int
    rank_positions: tuple[tuple[int, ...], ...]
    l20: object
    selection: IndexSelection

    @classmethod
    @torch.inference_mode()
    def append(cls, previous, l20, context):
        context.validate()
        l20.rows.validate()
        if (
            context.decoder_only
            or context.cache.identity.replay_fingerprint
            != ReplayConfig(ReplayMode.BOUNDED).fingerprint
            or not set(range(21)).issubset(context.completed_layers)
            or context.published_sources != {2, 8, 14, 20}
            or l20.rows.token_ids.numel() != context.query_rows
            or l20.hidden_states.shape[0] != context.query_rows
            or l20.pre_mix.shape[0] != context.query_rows
        ):
            raise ValueError(
                "CP L20 tail requires completed full encoder rows and sources"
            )
        _check(
            l20.rows.valid == context.valid,
            "CP L20 validity differs from its canonical row map",
        )
        selected = context.selection_for(20)
        selected.check()
        start = max(0, context.end - SWA_WINDOW)
        if previous is not None:
            if (
                previous.request_id != context.cache.request_id
                or previous.cache_fingerprint != context.cache.identity.fingerprint
                or previous.epoch >= context.epoch
                or previous.positions.end != context.start
                or previous.cp_rank != context.cp.cp_rank
            ):
                raise ValueError(
                    "CP retained tail has stale request, epoch or source ownership"
                )
            start = max(start, previous.positions.start)
        else:
            start = max(start, context.start)
        rank_positions = [[] for _ in range(8)]
        old_indices = []
        if previous is not None:
            for rank, positions in enumerate(previous.rank_positions):
                for offset, position in enumerate(positions):
                    if position >= start:
                        rank_positions[rank].append(position)
                        if rank == context.cp.cp_rank:
                            old_indices.append(offset)
        new_indices = []
        for position in range(max(start, context.start), context.end):
            rank, offset = divmod(
                context._restore_host[position - context.start], context.local_count
            )
            rank_positions[rank].append(position)
            if rank == context.cp.cp_rank:
                new_indices.append(offset)
        old_indices = torch.tensor(
            old_indices, dtype=torch.int64, device=context.query_device
        )
        new_indices = torch.tensor(
            new_indices, dtype=torch.int64, device=context.query_device
        )

        def join(old, new):
            selected_new = new.index_select(0, new_indices)
            if old_indices.numel() == 0:
                return selected_new
            return torch.cat((old.index_select(0, old_indices), selected_new), dim=0)

        rows = type(l20.rows)(
            *(
                join(
                    None if previous is None else getattr(previous.l20.rows, item.name),
                    getattr(l20.rows, item.name),
                )
                for item in fields(l20.rows)
            )
        )
        rows.validate()
        result = cls(
            context.cache.request_id,
            context.cache.identity.fingerprint,
            context.epoch,
            RowRange(start, context.end),
            context.cp.cp_rank,
            tuple(tuple(positions) for positions in rank_positions),
            replace(
                l20,
                rows=rows,
                hidden_states=join(
                    None if previous is None else previous.l20.hidden_states,
                    l20.hidden_states,
                ),
                pre_mix=join(
                    None if previous is None else previous.l20.pre_mix, l20.pre_mix
                ),
            ),
            IndexSelection(
                join(
                    None if previous is None else previous.selection.topk, selected.topk
                ),
                join(
                    None if previous is None else previous.selection.candidate_blocks,
                    selected.candidate_blocks,
                ),
                join(
                    None if previous is None else previous.selection.status,
                    selected.status,
                ),
                20,
                20,
                0,
                0,
                0,
            ),
        )
        if result.storage_bytes > MAX_GATHER_BYTES:
            raise ValueError("CP retained L20 state exceeds the bounded workspace")
        return result

    @property
    def storage_bytes(self):
        return _bytes(
            *(getattr(self.l20.rows, item.name) for item in fields(self.l20.rows)),
            self.l20.hidden_states,
            self.l20.pre_mix,
            self.selection.topk,
            self.selection.candidate_blocks,
            self.selection.status,
        )

    @torch.inference_mode()
    def select(self, current_context, rows: RowRange, replay_floor):
        current_context.validate()
        if (
            self.request_id != current_context.cache.request_id
            or self.cache_fingerprint != current_context.cache.identity.fingerprint
            or self.epoch != current_context.epoch
            or self.cp_rank != current_context.cp.cp_rank
            or not self.positions.start
            <= rows.start
            < rows.end
            == self.positions.end
            == current_context.end
            or current_context.published_sources != {2, 8, 14, 20}
            or any(
                owner.materialized_end != rows.end
                for owner in current_context.cache.owners.values()
            )
        ):
            raise ValueError(
                "CP decoder tail must match the current complete source boundary"
            )
        rank_rows = [
            [
                (position, local)
                for local, position in enumerate(positions)
                if position >= rows.start
            ]
            for positions in self.rank_positions
        ]
        context = current_context._remap_tail(
            rows.start, replay_floor, rank_rows, self.l20.rows.token_ids.numel()
        )
        context.publish_selection(
            IndexSelection(
                context.select_rows(self.selection.topk, fill=-1),
                context.select_rows(self.selection.candidate_blocks, fill=-1),
                context.select_rows(self.selection.status),
                20,
                20,
                0,
                0,
                0,
            )
        )
        return context.select_l20(self.l20), context


def begin_cp_request(
    cp_ctx: CPContext,
    request_index: int,
    *,
    request_id,
    identity,
    layout,
    max_tokens,
    pools,
    tables,
    pair_pools,
    pair_tables,
    epoch=0,
    decoder_ready_end=None,
    restored_state_ready=False,
):
    """Bind one real request, including ranks with zero real zigzag rows.

    Tables are already sliced to one request. SWA/pair tables use canonical
    B*CP checkpoint IDs; global/index tables use this rank's compact RR IDs.
    A nonzero prefix must have its encoder state restored by the engine.
    Bounded orchestration supplies the independently materialized decoder end.
    """
    if (
        cp_ctx.cp_size != 8
        or not 0 <= cp_ctx.cp_rank < 8
        or not cp_ctx.kv_cache_sharded
        or layout.cp_size != 8
        or (layout.speculative_tokens, layout.draft_enabled)
        not in ((0, False), (5, True))
        or identity.layout_fingerprint != layout.fingerprint
        or identity.replay_fingerprint
        not in (
            ReplayConfig(ReplayMode.FULL).fingerprint,
            ReplayConfig(ReplayMode.BOUNDED).fingerprint,
        )
        or not request_id
        or type(restored_state_ready) is not bool
    ):
        raise ValueError(
            "CP prefill requires CP8 target-only or gamma5/three-draft layout and replay policy"
        )
    if (
        not torch.distributed.is_initialized()
        or torch.distributed.get_world_size(collective_torch._get_group(Group.TP)) != 8
    ):
        raise RuntimeError(
            "CP8 attention requires the actual eight-rank TP process group"
        )
    device = cp_ctx.global_positions.device
    if not is_supported(torch.empty((0, 5120), dtype=torch.bfloat16, device=device)):
        raise RuntimeError("CP8 attention requires Blackwell CUDA")
    cache = V41AttentionCache(request_id, identity, layout, max_tokens, {}, {})
    context = V41CPAttentionContext(
        cache, cp_ctx, request_index, pools, tables, pair_pools, pair_tables, epoch
    )
    if not 0 <= context.start < context.end <= max_tokens <= 1048576:
        raise ValueError("CP request must have a nonempty admitted canonical range")
    occupied = set()
    for page in layout.pages:
        pool, table = pools[page.slot], tables[page.slot]
        if (
            pool.dtype != torch.uint8
            or pool.device != device
            or pool.ndim != 2
            or pool.shape[1] != page.prefill_shard_bytes
            or pool.stride(1) != 1
            or table.shape[0] != 1
            or table.ndim != 2
            or table.dtype != torch.int32
            or table.device != device
            or not table.is_contiguous()
            or not table.shape[1]
        ):
            raise ValueError(
                "CP request pools/tables do not match declared physical shards"
            )
        if page.slot.region == CacheRegion.SWA:
            destination = context._physical(table, context.current, pool)
            if context.start:
                context._physical(table, context.previous, pool)
            key = (pool.data_ptr(), destination)
            if key in occupied:
                raise ValueError("CP owner regions cannot alias writable fixed state")
            occupied.add(key)
        else:
            CompactPages(pool, page.slot.region, page.entries).validate(device)
            block_count = (
                context.end + layout.token_block_size - 1
            ) // layout.token_block_size
            for logical in range(cp_ctx.cp_rank, block_count, 8):
                context._physical(table, logical // 8, pool)
    for owner in GLOBAL_OWNERS:
        global_slot, index_slot = RegionSlot(CacheRegion.GLOBAL, owner), RegionSlot(
            CacheRegion.INDEX_K, owner
        )
        global_spec, index_spec = (
            context._page_specs[global_slot],
            context._page_specs[index_slot],
        )
        pair = None
        if owner in PAIR_OWNERS:
            pair_size = (
                ((context._pair_snapshots[owner] * _PAIR_BYTES + 511) // 512) * 512 // 8
            )
            pool, table = pair_pools[owner], pair_tables[owner]
            if (
                pool.dtype != torch.uint8
                or pool.device != device
                or pool.ndim != 2
                or pool.shape[1] != pair_size
                or pool.stride(1) != 1
                or table.ndim != 2
                or table.shape[0] != 1
                or table.dtype != torch.int32
                or table.device != device
                or not table.is_contiguous()
            ):
                raise ValueError(
                    "CP pair state must match the layout's complete byte-sliced snapshots"
                )
            context._physical(table, context.current, pool)
            pair = context.restore_pair(owner, restored_state_ready=restored_state_ready)
            context._pair_initials[owner] = pair
        cache.owners[owner] = AttentionOwnerCache(
            GlobalBinding(
                CompactPages(
                    pools[global_slot], CacheRegion.GLOBAL, global_spec.entries
                ),
                tables[global_slot],
                global_spec.ratio,
            ),
            CompactPages(pools[index_slot], CacheRegion.INDEX_K, index_spec.entries),
            tables[index_slot],
            context.start,
            pair,
        )
    layers = range(43 if layout.draft_enabled else 40)
    cache.swa_ends = {layer: context.start for layer in layers}
    if decoder_ready_end is not None:
        if (
            type(decoder_ready_end) is not int
            or not 0 <= decoder_ready_end <= context.start
        ):
            raise ValueError(
                "CP decoder readiness must precede the current encoder range"
            )
        if (
            identity.replay_fingerprint == ReplayConfig(ReplayMode.FULL).fingerprint
            and decoder_ready_end != context.start
        ):
            raise ValueError("full CP prefill requires complete prior decoder state")
        cache.swa_ends.update(
            {layer: decoder_ready_end for layer in layers if layer >= 21}
        )
    return context


def _packed_pages(values, spec):
    queries, rows, row_bytes = values.shape
    pages_per_query = (rows + spec.entries - 1) // spec.entries
    storage = torch.zeros(
        (queries * pages_per_query + 1, spec.page_stride_bytes),
        dtype=torch.uint8,
        device=values.device,
    )
    table = torch.arange(
        1, queries * pages_per_query + 1, dtype=torch.int32, device=values.device
    ).view(queries, pages_per_query)
    row = torch.arange(rows, device=values.device)
    ids = table.long()[:, row // spec.entries]
    columns = (row % spec.entries)[:, None] * row_bytes + torch.arange(
        row_bytes, device=values.device
    )
    storage[ids[:, :, None], columns[None, :, :]] = values
    return CompactPages(storage, spec.slot.region, spec.entries), table


def _publish_owner(attention, hidden, context):
    owner, state = attention.layer, context.cache.owners[attention.layer]
    if state.materialized_end != context.start:
        raise ValueError("CP compressor must extend its canonical materialized prefix")
    for first in range(context.start, context.end, _SOURCE_ROWS):
        last = min(first + _SOURCE_ROWS, context.end)
        source_hidden = context.gather_rows(hidden, first, last)
        source = attention.compressor(
            source_hidden,
            start_pos=first,
            request_id=context.cache.request_id,
            identity=context.cache.identity,
            pair=state.pair,
        )
        rows = prepare_owner_kv(source, attention.index_wk, attention.index_norm)
        slots = []
        # OwnerCompressor names each group's first token; the shared CP mapper
        # accepts the last token which completes that compression group.
        positions = source.group_positions + attention.source.ratio - 1
        for pages, table in (
            (state.global_kv.pages, state.global_kv.page_table),
            (state.index_pages, state.index_table),
        ):
            slots.append(
                cp_kv_slot_mapping(
                    positions,
                    table,
                    torch.zeros_like(positions),
                    context.cache.layout.token_block_size,
                    pages.entries_per_page,
                    attention.source.ratio,
                    8,
                    context.cp.cp_rank,
                )
            )
        for result in rows.store(
            OwnerPageBinding(owner, context.cache.identity, state.global_kv.pages),
            OwnerPageBinding(owner, context.cache.identity, state.index_pages),
            *slots,
        ):
            result.check()
        state.pair = source.next_pair
        state.materialized_end = last
        del source_hidden, source, rows, slots
    if owner in PAIR_OWNERS:
        context.publish_pair(owner, state.pair)
    context.published_sources.add(owner)


def _score_queries(attention, hidden, qr, context):
    layer, source = attention.layer, attention.source
    if source.index_k_owner not in context.published_sources:
        raise ValueError("CP index queries require their published source owner")
    query = attention_rope(
        attention.index_wq_b(qr).reshape(-1, 32, 128),
        context.positions,
        global_branch=True,
    )
    weights = (
        F.linear(hidden, attention.index_weights) * (128**-0.5 * 32**-0.5)
    ).contiguous()
    visible = torch.where(context.valid, (context.positions + 1) // source.ratio, 0).to(
        torch.int32
    )
    slot = RegionSlot(CacheRegion.INDEX_K, source.index_k_owner)
    top, blocks, status = {}, {}, {}
    calls = max_logits = max_packed = 0
    query_tile = SOURCE_QUERY_TILE if layer <= 20 else QUERY_TILE
    source_tiles = (
        (0,)
        if layer > 20
        else range(0, max(1, context.end // source.ratio), _INDEX_ROWS)
    )
    candidates = context.selection_for(20).candidate_blocks if layer > 20 else None
    for source_first in source_tiles:
        restored = lease = None
        if layer <= 20:
            source_last = min(
                source_first + _INDEX_ROWS, max(1, context.end // source.ratio)
            )
            restored, lease = context.gather_paged(
                slot, source_first, source_last, layer
            )
            prepared_source = prepare_index_source(
                restored.pages,
                restored.page_table,
                capacity=source_last - source_first,
            )
        for first in range(0, hidden.shape[0], query_tile):
            last = min(first + query_tile, hidden.shape[0])
            local_visible = visible[first:last]
            if layer <= 20:
                tile_visible = (local_visible - source_first).clamp(
                    0, source_last - source_first
                )
                scores = score_index_source(
                    query[first:last].contiguous(),
                    weights[first:last].contiguous(),
                    prepared_source,
                    tile_visible,
                    layer=layer,
                    position_offset=source_first,
                )
                scores = replace(scores, visible_lengths=local_visible)
            else:
                chosen = candidates[first:last]
                row = torch.arange(
                    SPARSE_BLOCK, dtype=torch.int32, device=hidden.device
                )
                actual = (chosen[:, :, None] * SPARSE_BLOCK + row).flatten(1)
                actual = torch.where(
                    (chosen[:, :, None] >= 0).expand(-1, -1, SPARSE_BLOCK).flatten(1)
                    & (actual < local_visible[:, None]),
                    actual,
                    -1,
                ).contiguous()
                values = torch.empty(
                    (last - first, actual.shape[1], 68),
                    dtype=torch.uint8,
                    device=hidden.device,
                )
                # Retain the complete candidates while bounding the rank-major
                # answers, reduced output and gathered int32 request positions.
                transport_blocks = min(
                    _REINDEX_BLOCKS,
                    (MAX_GATHER_BYTES - _bytes(values, actual))
                    // (
                        (last - first)
                        * SPARSE_BLOCK
                        * (9 * 68 + 8 * (4 + _SELECTED_METADATA_BYTES) + 4)
                    ),
                )
                # Tile transport, then score the complete candidate set once.
                # This preserves the existing selector's tie behavior.
                for row_first in range(
                    0, actual.shape[1], transport_blocks * SPARSE_BLOCK
                ):
                    row_last = row_first + transport_blocks * SPARSE_BLOCK
                    received, row_lease = context.gather_selected(
                        slot,
                        actual[:, row_first:row_last].contiguous(),
                        layer,
                        retained_bytes=_bytes(values, actual),
                    )
                    values[:, row_first:row_last].copy_(received)
                    context.release(row_lease, layer)
                    del received, row_lease
                pages, table = _packed_pages(values, context._page_specs[slot])
                tile_visible = (chosen >= 0).sum(-1, dtype=torch.int32) * SPARSE_BLOCK
                ids = torch.arange(
                    CANDIDATE_BLOCKS, dtype=torch.int32, device=hidden.device
                )[None, :].expand(last - first, -1)
                ids = torch.where(
                    ids * SPARSE_BLOCK < tile_visible[:, None], ids, -1
                ).contiguous()
                request_ids = torch.arange(
                    last - first, dtype=torch.int32, device=hidden.device
                )
                scores = score_candidate_tile(
                    query[first:last].contiguous(),
                    weights[first:last].contiguous(),
                    pages,
                    table,
                    request_ids,
                    tile_visible,
                    ids,
                    layer=layer,
                )
                original = torch.full_like(scores.positions, -1)
                original[:, : actual.shape[1]].copy_(actual)
                scores = replace(
                    scores,
                    logits=torch.where(original >= 0, scores.logits, -torch.inf),
                    positions=original,
                    visible_lengths=local_visible,
                )
            top[first] = scores.topk(top.get(first))
            if layer == 20:
                blocks[first] = scores.block_topk(blocks.get(first))
            status[first] = torch.maximum(
                status.get(first, torch.zeros_like(scores.status)), scores.status
            )
            calls += 1
            max_logits = max(max_logits, scores.logits.numel())
            max_packed = max(max_packed, scores.packed_kv_bytes)
            ReaderResult(scores.logits, scores.status).check()
            if layer > 20:
                del values, pages, table
            del scores
        if lease is not None:
            context.release(lease, layer)
            del restored, lease, prepared_source
    context.publish_selection(
        IndexSelection(
            torch.cat([value.ordered_positions() for value in top.values()]),
            (
                torch.cat([value.ordered_positions() for value in blocks.values()])
                if layer == 20
                else None
            ),
            torch.cat(list(status.values())),
            layer,
            source.index_k_owner,
            calls,
            max_logits,
            max_packed,
        )
    )


@torch.inference_mode()
def forward_cp_attention(attention, hidden, context):
    context.validate()
    if os.environ.get("DSV41_ATTENTION") != "1" or not is_supported(hidden):
        raise RuntimeError("CP8 attention requires opt-in Blackwell BF16 execution")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "CP prefill collectives require eager scheduling outside Graph capture"
        )
    if hidden.shape != (context.query_rows, 5120) or not hidden.is_contiguous():
        raise ValueError(
            "CP attention must retain the framework's rank-local padded rows"
        )
    if attention.layer in context.completed_layers:
        raise ValueError("CP attention layer already completed this forward")
    if context.decoder_only and attention.layer <= 20:
        raise ValueError("bounded CP tail cannot rerun an encoder source layer")
    if (
        attention.compressor is not None
        and attention.compressor.layout != context.cache.layout
    ):
        raise ValueError("CP compressor and framework cache layouts differ")
    backend = os.environ.get("DSV41_ATTENTION_BACKEND", "native")
    if backend not in ("native", "flashmla"):
        raise ValueError("unknown V4.1 attention backend; no silent fallback")
    try:
        qr, query, kv = attention._project(hidden, context.positions)
        if attention.source.writes_global:
            _publish_owner(attention, hidden, context)
        if attention.source.scores_queries:
            _score_queries(attention, hidden, qr, context)
        encoded = encode_compact(kv, CacheRegion.SWA)
        encoded.check()
        initial = context.restore_swa(attention.layer)
        output = torch.empty_like(query)
        indices = (
            context.indices_for(attention.layer) if attention.source.ratio else None
        )
        for first in range(0, hidden.shape[0], _READ_QUERIES):
            last = min(first + _READ_QUERIES, hidden.shape[0])
            swa = context.swa_queries(
                attention.layer, first, last, initial, encoded.output
            )
            global_kv = global_indices = lease = None
            if indices is not None:
                selected = indices[first:last].contiguous()
                slot = RegionSlot(CacheRegion.GLOBAL, attention.source.global_owner)
                values, lease = context.gather_selected(
                    slot,
                    selected,
                    attention.layer,
                    retained_bytes=_bytes(swa.pages.data, initial.pages.data),
                )
                pages, table = _packed_pages(values, context._page_specs[slot])
                global_kv = GlobalBinding(pages, table, attention.source.ratio)
                dense = torch.arange(
                    INDEX_TOPK, dtype=torch.int32, device=hidden.device
                )[None, :].expand(last - first, -1)
                global_indices = torch.where(selected >= 0, dense, -1).contiguous()
            reader = compact_attention
            if backend == "flashmla":
                from rtp_llm.models_py.modules.dsv41.flashmla import (
                    flashmla_compact_attention,
                )

                reader = flashmla_compact_attention
            positions = torch.where(
                context.valid[first:last], context.positions[first:last], -1
            ).to(torch.int32)
            result = reader(
                query[first:last].contiguous(),
                torch.arange(last - first, dtype=torch.int32, device=hidden.device),
                positions,
                torch.full_like(positions, context.replay_floor),
                swa,
                attention.sinks,
                global_kv=global_kv,
                global_indices=global_indices,
            )
            result.check()
            output[first:last].copy_(result.output)
            if lease is not None:
                context.release(lease, attention.layer)
                del values, pages, table, lease
            del result, swa, global_kv
        context.publish_swa(attention.layer, initial, encoded.output)
        output = attention_rope(
            output,
            context.positions,
            global_branch=bool(attention.source.ratio),
            inverse=True,
        )
        output = attention.wo_b(
            grouped_wo_a(output.reshape(-1, 8, 4096), attention.wo_a).flatten(1)
        )
        output.masked_fill_(~context.valid[:, None], 0)
        if output.shape != hidden.shape or output.dtype != torch.bfloat16:
            raise ValueError(
                "CP attention output projection changed rank-local target geometry"
            )
        context.completed_layers.add(attention.layer)
        context.observations.append(
            {
                "layer": attention.layer,
                "reader_backend": backend,
                "query_identity": context.query_identity,
                "query_rows": hidden.shape[0],
                "source_rows": (
                    context.end - context.start if attention.source.writes_global else 0
                ),
                "index_rows": hidden.shape[0] if attention.source.scores_queries else 0,
                "cp_size": 8,
                "gather_count": context.gather_count,
                "max_receive_bytes": context.max_receive_bytes,
                "max_gather_live_bytes": context.max_gather_live_bytes,
            }
        )
        return output
    except Exception:
        context.cache.poisoned = True
        raise
