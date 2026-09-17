"""CP8 target prefill over framework-owned V4.1 cache shards.

Queries retain the framework's padded zigzag order. Row-local model and attention
operations pack only real rows without changing their CP owners. Only compressor
inputs, SWA windows, and selected compact KV rows cross the TP/CP group. All
receive storage is scoped to a source tile and its consumers.
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
from rtp_llm.models_py.modules.dsv41.compact_writer import (
    CompactWriteResult,
    encode_compact,
)
from rtp_llm.models_py.modules.dsv41.compressor import (
    OwnerPageBinding,
    PairCarry,
    prepare_owner_kv,
)
from rtp_llm.models_py.modules.dsv41.cprr_reader import (
    CprrReaderLease,
    CprrReadIdentity,
    bind_cprr_paged,
    restore_cprr_swa,
)
from rtp_llm.models_py.modules.dsv41.decode_compressor import (
    normalize_empty_pair_checkpoint,
)
from rtp_llm.models_py.modules.dsv41.indexer import (
    CANDIDATE_BLOCKS,
    INDEX_TOPK,
    QUERY_TILE,
    SPARSE_BLOCK,
    IndexSelection,
    candidate_score_plan,
    score_candidate_tile,
)
from rtp_llm.models_py.modules.dsv41.math import grouped_wo_a
from rtp_llm.models_py.modules.dsv41.source_indexer import (
    SOURCE_QUERY_TILE,
    index_source_plan,
    prepare_index_source,
    score_index_source,
)

_PAIR_BYTES = 4112
_SOURCE_ROWS = 8192
# Measured optimum (2026-09-17 GB200 same-wheel three-arm A/B, 32 -> 128 ->
# 512 rows: 16K miss 5.55-7.08s -> 4.68-4.82s -> 3.47-3.68s, 64K miss
# 64.18s -> 34.25s -> 22.54s; composition ATen 62,595 -> 35,331 -> 28,515,
# forced syncs 9,470 -> 6,446 -> 5,690, chunks=1 held, reuse pins held).
_READ_QUERIES = 512
# The candidate transport tile cap; the 1 GiB live-byte formula below stays
# the binding constraint (735 blocks at 512 queries x 8 rows x 156B), so 768
# lets one reindex tile cover the full budget instead of an extra partial tile.
_REINDEX_BLOCKS = 768
_INDEX_ROWS = CANDIDATE_BLOCKS * SPARSE_BLOCK
# Per gathered row: int64 page/index vectors and transient mask/index results.
_SELECTED_METADATA_BYTES = 80


def _check(condition, message):
    if not bool(condition.all().item()):
        raise ValueError(message)


def _bytes(*tensors):
    return sum(value.numel() * value.element_size() for value in tensors)


def _query_rows(tensor, indices):
    return tensor if indices is None else tensor.index_select(0, indices)


def _copy_query_rows(destination, indices, values):
    if indices is None:
        destination.copy_(values)
    else:
        destination.index_copy_(0, indices, values)


def _selected_local_rows(
    pool, table, wanted, entries, entry_bytes, rank, *, defer_status=None
):
    message = "CP selected KV row has no allocated owner page"
    if (
        pool.is_cuda
        and pool.stride(1) == 1
        and torch.cuda.get_device_capability(pool.device)[0] == 10
        and os.environ.get("DSV41_CP_FUSED_SELECTED", "1") == "1"
    ):
        from rtp_llm.models_py.modules.dsv41._cp_gather_triton import (
            gather_selected_kernel,
        )

        block_rows = 4
        blocks = (wanted.numel() + block_rows - 1) // block_rows
        local = torch.empty(
            (wanted.numel(), entry_bytes), dtype=torch.uint8, device=pool.device
        )
        status = torch.empty((blocks,), dtype=torch.int32, device=pool.device)
        if blocks:
            gather_selected_kernel[(blocks,)](
                pool,
                table,
                wanted,
                local,
                status,
                ROWS=wanted.numel(),
                ENTRIES=entries,
                ENTRY_BYTES=entry_bytes,
                POOL_PAGES=pool.shape[0],
                PAGE_STRIDE=pool.stride(0),
                TABLE_WIDTH=table.shape[1],
                TABLE_STRIDE=table.stride(1),
                WANTED_STRIDE=wanted.stride(0),
                RANK=rank,
                BLOCK_ROWS=block_rows,
                BLOCK_BYTES=1 << (entry_bytes - 1).bit_length(),
            )
            # The async assert is the measured optimum (P0-2 GB200 A/B): the
            # synchronous _check forced an NCCL drain per tile. The env remains
            # only as a diagnostic override.
            if defer_status is not None:
                # The caller batches one async assert over the whole transport
                # loop; the per-tile status tensors stay alive in the list.
                defer_status.append(status)
            elif os.environ.get("DSV41_CP_GATHER_CHECK_ASYNC", "1") == "1":
                torch._assert_async((status == 0).all(), message)
            else:
                _check(status == 0, message)
        return local

    logical = wanted.clamp_min(0).long() // entries
    owners, virtual = logical % 8, logical // 8
    owned = (wanted >= 0) & (owners == rank)
    in_table = virtual < table.shape[1]
    ids = table[0].index_select(0, virtual.clamp_max(table.shape[1] - 1)).long()
    _check(~owned | (in_table & (ids > 0) & (ids < pool.shape[0])), message)
    page_rows = pool[:, : entries * entry_bytes].view(
        pool.shape[0], entries, entry_bytes
    )
    local = page_rows[
        ids.clamp(0, pool.shape[0] - 1), wanted.clamp_min(0).long() % entries
    ]
    local.masked_fill_(~owned[:, None], 0)
    return local


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
        self._swa_index_plans = {}
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
        query_owners = {flat // self.local_count for flat in mapping}
        self.single_query_owner = (
            next(iter(query_owners))
            if len(query_owners) == 1
            and os.environ.get("DSV41_CP_SINGLE_OWNER_TRANSPORT", "1") != "0"
            else None
        )
        device = self.query_device
        model_rows = sorted(
            flat % self.local_count
            for flat in mapping
            if flat // self.local_count == cp.cp_rank
        )
        self._real_local_rows = tuple(model_rows)
        self._query_row_plans = {}
        self.attention_query_rows = (
            len(model_rows)
            if os.environ.get("DSV41_CP_COMPACT_QUERY_ROWS", "1") != "0"
            else self.local_count
        )
        self.model_row_indices = (
            torch.tensor(model_rows, dtype=torch.int64, device=device)
            if len(model_rows) != self.local_count
            and os.environ.get("DSV41_CP_COMPACT_MODEL_ROWS", "1") != "0"
            else None
        )
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
    def model_query_rows(self):
        return (
            self.query_rows
            if self.model_row_indices is None
            else self.model_row_indices.numel()
        )

    def pack_model_rows(self, tensor):
        """Select real local rows using the already validated host ownership map."""
        if (
            tensor.ndim < 1
            or tensor.shape[0] != self.query_rows
            or tensor.device != self.query_device
        ):
            raise ValueError("CP model packing requires the canonical local row shape")
        if self.model_row_indices is None:
            return tensor
        return tensor.index_select(0, self.model_row_indices)

    def unpack_model_rows(self, tensor):
        """Restore attention/L20 row geometry without assigning work to padding."""
        if (
            tensor.ndim < 1
            or tensor.shape[0] != self.model_query_rows
            or tensor.device != self.query_device
        ):
            raise ValueError("CP model output differs from the packed local row shape")
        if self.model_row_indices is None:
            return tensor
        output = tensor.new_zeros((self.query_rows, *tensor.shape[1:]))
        output.index_copy_(0, self.model_row_indices, tensor)
        return output

    def query_row_indices(self, first, last):
        """Select real rows within a padded collective tile without device sync."""
        if not 0 <= first <= last <= self.query_rows:
            raise ValueError("CP query tile exceeds its canonical local row range")
        if self.attention_query_rows == self.query_rows:
            return None
        key = (first, last)
        if key not in self._query_row_plans:
            rows = [row - first for row in self._real_local_rows if first <= row < last]
            self._query_row_plans[key] = (
                None
                if len(rows) == last - first
                else torch.tensor(rows, dtype=torch.int64, device=self.query_device)
            )
        return self._query_row_plans[key]

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
        # One indexed upload for the whole slice; per-row device writes cost a
        # forced host sync each (2,048 scalar HtoD copies for a 16K request).
        framework_offsets, framework_locals = [], []
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
                framework_offsets.append(offset)
                framework_locals.append(local)
        if framework_offsets:
            context.framework_row_indices[framework_offsets] = torch.tensor(
                framework_locals, dtype=torch.int64, device=device
            )
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
                "CP receive lifetime exceeds 1 GiB; reduce the source tile"
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

    def _physical_async(self, table, logical, pool):
        # The GPU-side twin of _physical: the page ID stays on device (no host
        # sync) and the bounds check rides the async assert stream.
        if not 0 <= logical < table.shape[1]:
            raise ValueError("CP fixed state is missing its canonical checkpoint page")
        page = table[0, logical : logical + 1]
        torch._assert_async(
            ((page > 0) & (page < pool.shape[0])).all(),
            "CP fixed state requires an allocated rank-local page",
        )
        # Clamp before indexing: the async assert reports after the write, so
        # an invalid page must land on the unmapped null page, never OOB.
        return page.clamp(0, pool.shape[0] - 1).long()

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
        destination = self._physical_async(
            self.tables[slot], self.current, self.pools[slot]
        )
        begin, end = spec.swa_byte_slice(self.cp.cp_rank)
        self.pools[slot].index_copy_(
            0, destination, initial.pages.data[1, begin:end].unsqueeze(0)
        )
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
            torch.tensor(
                [restored_state_ready], dtype=torch.bool, device=self.query_device
            ),
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
        destination = self._physical_async(self.pair_tables[owner], self.current, pool)
        first = self.cp.cp_rank * pool.shape[1]
        pool.index_copy_(0, destination, raw[first : first + pool.shape[1]].unsqueeze(0))

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

    def _query_destination(self, query_owner):
        if query_owner != self.single_query_owner or query_owner is None:
            raise ValueError("CP compact transport requires one canonical query owner")
        group = collective_torch._get_group(Group.TP)
        return group, torch.distributed.get_global_rank(group, query_owner)

    def gather_selected(
        self,
        slot,
        positions,
        layer,
        *,
        retained_bytes=0,
        query_owner=None,
        mask_negative=True,
        defer_status=None,
    ):
        """Receive compact rows selected by each rank, retaining original quantization."""
        spec, pool, table = self._page_specs[slot], self.pools[slot], self.tables[slot]
        requests, width = positions.shape
        if query_owner is None:
            wanted = self._all_gather(positions, retained_bytes=retained_bytes).reshape(
                -1
            )
        else:
            group, destination = self._query_destination(query_owner)
            wanted = positions.clone().reshape(-1)
            self._record_gather(
                _bytes(wanted), retained_bytes + _bytes(positions, wanted)
            )
            torch.distributed.broadcast(wanted, src=destination, group=group)
        output_bytes = requests * width * spec.encoding.entry_bytes
        self._record_gather(
            output_bytes,
            retained_bytes
            + _bytes(positions, wanted)
            + (9 if query_owner is None else 1) * output_bytes
            + wanted.numel() * _SELECTED_METADATA_BYTES,
        )
        local = _selected_local_rows(
            pool,
            table,
            wanted,
            spec.entries,
            spec.encoding.entry_bytes,
            self.cp.cp_rank,
            defer_status=defer_status,
        )
        # Each byte has exactly one page owner; SUM preserves its bit pattern.
        if query_owner is None:
            values = torch.empty(
                (requests * width, spec.encoding.entry_bytes),
                dtype=torch.uint8,
                device=self.query_device,
            )
            torch.distributed.reduce_scatter_tensor(
                values, local, group=collective_torch._get_group(Group.TP)
            )
        else:
            torch.distributed.reduce(local, dst=destination, group=group)
            values = local
        values = values.view(requests, width, -1)
        if mask_negative:
            values.masked_fill_((positions < 0)[:, :, None], 0)
        return values, self.lease(slot, (values,), layer)

    def swa_queries(
        self, layer, first, last, initial, encoded, *, query_rows=None, query_owner=None
    ):
        """Restore each local query's causal ring before any later row overwrites it."""
        count = last - first
        spec = self._page_specs[RegionSlot(CacheRegion.SWA, layer)]
        if query_owner is None:
            query_positions = self._rank_positions[:, first:last]
        else:
            group, destination = self._query_destination(query_owner)
            query_positions = self._rank_positions[
                query_owner : query_owner + 1, first:last
            ]
        # SWA query indices depend on request-scoped state and tile bounds, never
        # on the layer; per-tile reuse across layers is bitwise identical and on
        # by default, the env remaining only as a diagnostic override.
        plan_key = (first, last, query_owner)
        cache_indices = os.environ.get("DSV41_SWA_INDEX_CACHE", "1") == "1"
        plan = self._swa_index_plans.get(plan_key) if cache_indices else None
        if plan is None:
            offsets = torch.arange(1 - SWA_WINDOW, 1, device=self.query_device)
            wanted = (query_positions[:, :, None] + offsets).reshape(-1)
            active = (
                (query_positions[:, :, None] >= 0)
                .expand(-1, -1, SWA_WINDOW)
                .reshape(-1)
            )
            valid = active & (wanted >= self.replay_floor) & (wanted >= 0)
            current = valid & (wanted >= self.start)
            restore_index = (wanted - self.start).clamp(0, self.end - self.start - 1)
            source = self._restore.index_select(0, restore_index)
            owners = torch.where(current, source // self.local_count, 0)
            local_indices = source % self.local_count
            past = valid & ~current
            ring_index = wanted % spec.entries
            plan = [
                (
                    offsets,
                    wanted,
                    active,
                    valid,
                    current,
                    restore_index,
                    source,
                    owners,
                    local_indices,
                    past,
                    ring_index,
                ),
                {},
            ]
            if cache_indices:
                self._swa_index_plans[plan_key] = plan
        (
            offsets,
            wanted,
            active,
            valid,
            current,
            restore_index,
            source,
            owners,
            local_indices,
            past,
            ring_index,
        ) = plan[0]
        output_bytes = count * SWA_WINDOW * 528
        page_bytes = (count + 1) * spec.page_stride_bytes
        # The fixed-shape past/current selection briefly owns three input tiles.
        # Communication and ring packing have smaller, disjoint lifetimes.
        self._record_gather(
            output_bytes,
            _bytes(initial.pages.data)
            + wanted.numel() * _SELECTED_METADATA_BYTES
            + max(
                (24 if query_owner is None else 3) * output_bytes,
                output_bytes + 2 * page_bytes,
            ),
        )
        local = encoded.index_select(0, local_indices)
        if self.replay_floor < self.start:
            _check(
                ~past | (wanted >= initial.valid_starts[0]),
                "CP query is missing restored SWA history",
            )
            if self.cp.cp_rank == 0:
                old = initial.pages.data[
                    1, : initial.pages.entries_per_page * 528
                ].view(-1, 528)
                local = torch.where(
                    past[:, None],
                    old.index_select(0, ring_index),
                    local,
                )
        local.masked_fill_((~valid | (owners != self.cp.cp_rank))[:, None], 0)
        # Current rows have their canonical CP owner; restored rows use rank0.
        if query_owner is None:
            values = torch.empty(
                (count * SWA_WINDOW, 528), dtype=torch.uint8, device=self.query_device
            )
            torch.distributed.reduce_scatter_tensor(
                values, local, group=collective_torch._get_group(Group.TP)
            )
        else:
            torch.distributed.reduce(local, dst=destination, group=group)
            values = local
            if self.cp.cp_rank != query_owner:
                values.zero_()
        del local, wanted, active, valid, current, source, owners, local_indices, past
        values = values.view(count, SWA_WINDOW, 528)
        if query_rows is not None:
            if query_rows.numel() == 0:
                return None
            values = values.index_select(0, query_rows)
            count = query_rows.numel()
        final = plan[1].get(count) if cache_indices else None
        if final is None:
            positions = self._rank_positions[self.cp.cp_rank, first:last]
            if query_rows is not None:
                positions = positions.index_select(0, query_rows)
            page_ids = torch.arange(
                1, count + 1, dtype=torch.int32, device=self.query_device
            )
            valid_starts = torch.where(
                positions >= 0,
                (positions - SWA_WINDOW + 1).clamp_min(self.replay_floor),
                0,
            ).to(torch.int32)
            valid_ends = (positions + 1).clamp_min(0).to(torch.int32)
            if cache_indices:
                page_indices = _swa_query_page_indices(
                    positions, spec, self.replay_floor
                )
                plan[1][count] = (
                    positions,
                    page_ids,
                    valid_starts,
                    valid_ends,
                    page_indices,
                )
                pages = _swa_query_pages(
                    values, positions, spec, self.replay_floor, indices=page_indices
                )
            else:
                pages = _swa_query_pages(values, positions, spec, self.replay_floor)
        else:
            positions, page_ids, valid_starts, valid_ends, page_indices = final
            pages = _swa_query_pages(
                values, positions, spec, self.replay_floor, indices=page_indices
            )
        binding = SwaBinding(
            pages,
            page_ids,
            valid_starts,
            valid_ends,
        )
        lease = self.lease(spec.slot, (values, pages.data), layer)
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
            logicals = [
                logical // 8 for logical in range(cp_ctx.cp_rank, block_count, 8)
            ]
            # One batched read per table instead of a forced scalar sync per
            # logical page; the checks and their order match _physical.
            row = table[0].cpu().tolist()
            for logical in logicals:
                if not 0 <= logical < table.shape[1]:
                    raise ValueError(
                        "CP fixed state is missing its canonical checkpoint page"
                    )
                if not 0 < row[logical] < pool.shape[0]:
                    raise ValueError(
                        "CP fixed state requires an allocated rank-local page"
                    )
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
            pair = context.restore_pair(
                owner, restored_state_ready=restored_state_ready
            )
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


def _swa_query_page_indices(positions, spec, replay_floor):
    columns = torch.arange(spec.entries, device=positions.device)
    distance = (positions[:, None] - columns[None, :]) % spec.entries
    source_rows = (SWA_WINDOW - 1 - distance).clamp_min(0)
    tokens = positions[:, None] - distance
    valid = (
        (positions[:, None] >= 0)
        & (distance < SWA_WINDOW)
        & (tokens >= replay_floor)
        & (tokens >= 0)
    )
    return columns, distance, source_rows, tokens, valid


def _swa_query_pages(values, positions, spec, replay_floor, indices=None):
    count, _, row_bytes = values.shape
    if indices is None:
        indices = _swa_query_page_indices(positions, spec, replay_floor)
    columns, distance, source_rows, tokens, valid = indices
    packed = values.gather(1, source_rows[:, :, None].expand(-1, -1, row_bytes))
    packed.masked_fill_(~valid[:, :, None], 0)
    storage = torch.zeros(
        (count + 1, spec.page_stride_bytes), dtype=torch.uint8, device=values.device
    )
    storage[1:, : spec.entries * row_bytes].view(count, spec.entries, row_bytes).copy_(
        packed
    )
    return CompactPages(storage, CacheRegion.SWA, spec.entries)


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
    destination = storage[1:].view(queries, pages_per_query, spec.page_stride_bytes)
    full_pages, tail = divmod(rows, spec.entries)
    if full_pages:
        destination[:, :full_pages, : spec.entries * row_bytes].copy_(
            values[:, : full_pages * spec.entries].reshape(
                queries, full_pages, spec.entries * row_bytes
            )
        )
    if tail:
        destination[:, full_pages, : tail * row_bytes].copy_(
            values[:, full_pages * spec.entries :].reshape(queries, tail * row_bytes)
        )
    return CompactPages(storage, spec.slot.region, spec.entries), table


def _publish_owner(attention, hidden, context, source_rows=0):
    owner, state = attention.layer, context.cache.owners[attention.layer]
    if state.materialized_end != context.start:
        raise ValueError("CP compressor must extend its canonical materialized prefix")
    # The tile size is validated by the caller (forward_cp_attention); the
    # default here only covers direct test callers of this helper.
    source_rows = source_rows or _SOURCE_ROWS
    if source_rows < 2 or source_rows % 2:
        raise ValueError("CP source tile must be a positive even row count")
    pending = []
    for first in range(context.start, context.end, source_rows):
        last = min(first + source_rows, context.end)
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
        pending.extend(
            rows.store(
                OwnerPageBinding(owner, context.cache.identity, state.global_kv.pages),
                OwnerPageBinding(owner, context.cache.identity, state.index_pages),
                *slots,
            )
        )
        state.pair = source.next_pair
        state.materialized_end = last
        del source_hidden, source, rows, slots
    # One synchronous writer status check per layer, after every source tile.
    CompactWriteResult.check_all(pending)
    del pending
    if owner in PAIR_OWNERS:
        context.publish_pair(owner, state.pair)
    context.published_sources.add(owner)


def _score_queries(attention, hidden, qr, context):
    layer, source = attention.layer, attention.source
    if source.index_k_owner not in context.published_sources:
        raise ValueError("CP index queries require their published source owner")
    query = context.unpack_model_rows(
        attention_rope(
            attention.index_wq_b(context.pack_model_rows(qr)).reshape(-1, 32, 128),
            context.pack_model_rows(context.positions),
            global_branch=True,
        )
    )
    weights = context.unpack_model_rows(
        F.linear(context.pack_model_rows(hidden), attention.index_weights)
        * (128**-0.5 * 32**-0.5)
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
    # (first,last) row plans are invariant across the source tile scans below.
    query_tiles = []
    for first in range(0, hidden.shape[0], query_tile):
        last = min(first + query_tile, hidden.shape[0])
        query_rows = context.query_row_indices(first, last)
        count = last - first if query_rows is None else query_rows.numel()
        local_visible = _query_rows(visible[first:last], query_rows)
        query_tiles.append((first, last, query_rows, count, local_visible))
    if layer > 20:
        sparse_rows = torch.arange(
            SPARSE_BLOCK, dtype=torch.int32, device=hidden.device
        )
        candidate_ids = torch.arange(
            CANDIDATE_BLOCKS, dtype=torch.int32, device=hidden.device
        )
        candidate_starts = candidate_ids * SPARSE_BLOCK
        request_ids = torch.arange(query_tile, dtype=torch.int32, device=hidden.device)
        candidate_plan = candidate_score_plan(hidden.device)
    for source_first in source_tiles:
        restored = lease = None
        if layer <= 20:
            source_last = min(
                source_first + _INDEX_ROWS, max(1, context.end // source.ratio)
            )
            restored, lease = context.gather_paged(
                slot, source_first, source_last, layer
            )
            if context.attention_query_rows:
                prepared_source = prepare_index_source(
                    restored.pages,
                    restored.page_table,
                    capacity=source_last - source_first,
                )
                source_plan = index_source_plan(hidden.device, source_first)
            else:
                prepared_source = source_plan = None
        for first, last, query_rows, count, local_visible in query_tiles:
            if layer <= 20:
                if not count:
                    continue
                tile_visible = (local_visible - source_first).clamp(
                    0, source_last - source_first
                )
                scores = score_index_source(
                    _query_rows(query[first:last], query_rows).contiguous(),
                    _query_rows(weights[first:last], query_rows).contiguous(),
                    prepared_source,
                    tile_visible,
                    layer=layer,
                    position_offset=source_first,
                    plan=source_plan,
                )
                scores = replace(scores, visible_lengths=local_visible)
            else:
                chosen = candidates[first:last]
                actual = (chosen[:, :, None] * SPARSE_BLOCK + sparse_rows).flatten(1)
                actual = torch.where(
                    (chosen[:, :, None] >= 0).expand(-1, -1, SPARSE_BLOCK).flatten(1)
                    & (actual < visible[first:last, None]),
                    actual,
                    -1,
                ).contiguous()
                values = torch.empty(
                    (count, actual.shape[1], 68),
                    dtype=torch.uint8,
                    device=hidden.device,
                )
                # Size the common transport schedule for the largest rank buffer.
                padded_value_bytes = (last - first) * actual.shape[1] * 68
                peers = 8 if context.single_query_owner is None else 1
                transport_blocks = min(
                    _REINDEX_BLOCKS,
                    (MAX_GATHER_BYTES - padded_value_bytes - _bytes(actual))
                    // (
                        (last - first)
                        * SPARSE_BLOCK
                        * (
                            (9 if peers == 8 else 1) * 68
                            + peers * (4 + _SELECTED_METADATA_BYTES)
                            + 4
                        )
                    ),
                )
                # Tile transport, then score the complete candidate set once.
                # This preserves the existing selector's tie behavior. The
                # per-tile owner-page status asserts batch into one async
                # assert and the negative-position mask applies once over the
                # complete values tensor (bitwise identical to per-tile masks).
                statuses = []
                for row_first in range(
                    0, actual.shape[1], transport_blocks * SPARSE_BLOCK
                ):
                    row_last = row_first + transport_blocks * SPARSE_BLOCK
                    received, row_lease = context.gather_selected(
                        slot,
                        actual[:, row_first:row_last].contiguous(),
                        layer,
                        retained_bytes=_bytes(values, actual),
                        query_owner=context.single_query_owner,
                        mask_negative=False,
                        defer_status=statuses,
                    )
                    if count:
                        values[:, row_first:row_last].copy_(
                            _query_rows(received, query_rows)
                        )
                    context.release(row_lease, layer)
                    del received, row_lease
                if statuses:
                    torch._assert_async(
                        (torch.cat(statuses) == 0).all(),
                        "CP selected KV row has no allocated owner page",
                    )
                if not count:
                    del values, actual
                    continue
                chosen = _query_rows(chosen, query_rows)
                actual = _query_rows(actual, query_rows)
                # The mask follows the query-row-selected actual (values holds
                # exactly those rows); bitwise identical to the per-tile masks.
                values.masked_fill_((actual < 0)[:, :, None], 0)
                pages, table = _packed_pages(values, context._page_specs[slot])
                tile_visible = (chosen >= 0).sum(-1, dtype=torch.int32) * SPARSE_BLOCK
                ids = torch.where(
                    candidate_starts < tile_visible[:, None], candidate_ids, -1
                ).contiguous()
                scores = score_candidate_tile(
                    _query_rows(query[first:last], query_rows).contiguous(),
                    _query_rows(weights[first:last], query_rows).contiguous(),
                    pages,
                    table,
                    request_ids[:count],
                    tile_visible,
                    ids,
                    layer=layer,
                    plan=candidate_plan,
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
            status[first] = (
                torch.maximum(status[first], scores.status)
                if first in status
                else scores.status
            )
            calls += 1
            max_logits = max(max_logits, scores.logits.numel())
            max_packed = max(max_packed, scores.packed_kv_bytes)
            if layer > 20:
                del values, pages, table
            del scores
        if lease is not None:
            context.release(lease, layer)
            del restored, lease, prepared_source, source_plan
    topk = torch.full(
        (context.query_rows, INDEX_TOPK), -1, dtype=torch.int32, device=hidden.device
    )
    candidate_blocks = (
        torch.full(
            (context.query_rows, CANDIDATE_BLOCKS),
            -1,
            dtype=torch.int32,
            device=hidden.device,
        )
        if layer == 20
        else None
    )
    query_status = torch.zeros(
        context.query_rows, dtype=torch.int32, device=hidden.device
    )
    for first, ranked in top.items():
        last = min(first + query_tile, context.query_rows)
        query_rows = context.query_row_indices(first, last)
        _copy_query_rows(topk[first:last], query_rows, ranked.ordered_positions())
        _copy_query_rows(query_status[first:last], query_rows, status[first])
        if candidate_blocks is not None:
            _copy_query_rows(
                candidate_blocks[first:last],
                query_rows,
                blocks[first].ordered_positions(),
            )
    # One synchronous status check per layer, after every tile merged above.
    ReaderResult(topk, query_status).check()
    context.publish_selection(
        IndexSelection(
            topk,
            candidate_blocks,
            query_status,
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
    if not hidden.is_cuda or hidden.dtype != torch.bfloat16:
        raise ValueError("CP attention needs CUDA BF16 hidden rows")
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
    read_queries = int(os.environ.get("DSV41_CP_READ_QUERIES", _READ_QUERIES))
    # The cap tracks the 1 GiB gather-live-byte budget (512 rows x SWA_WINDOW x
    # 528B x 24 transient ~= 830 MiB). 512 is the measured optimum and the code
    # default (see _READ_QUERIES); the env stays only as a diagnostic override.
    if not 1 <= read_queries <= 512:
        raise ValueError("CP attention query batch must be between 1 and 512")
    if context.single_query_owner is not None:
        owner_batch = int(os.environ.get("DSV41_CP_SINGLE_OWNER_READ_QUERIES", "64"))
        if owner_batch not in (32, 64, 128):
            raise ValueError("CP single-owner query batch must be 32, 64 or 128")
        # Reader staging is additional to the per-gather live-byte accounting.
        read_queries = min(owner_batch, 8 * read_queries)
    # The 8192-row source tile is the measured optimum (GB200 p13 A/B
    # 2026-09-17: 512 vs 2048 vs 8192 same-wheel back-to-back arms; 8192 wins
    # the 16K composition counts and the 64K unprofiled latency, ties 16K) and
    # the code default; the gather live-byte accounting in _record_gather
    # bounds the tile by the 1 GiB budget. The env stays only as a diagnostic
    # override. Validated here (before the poisoned-on-failure body) like
    # read_queries.
    source_rows = int(os.environ.get("DSV41_CP_SOURCE_ROWS", _SOURCE_ROWS))
    if source_rows < 2 or source_rows % 2:
        raise ValueError("CP source tile must be a positive even row count")
    try:
        model_positions = context.pack_model_rows(context.positions)
        qr, query, kv = (
            context.unpack_model_rows(value)
            for value in attention._project(
                context.pack_model_rows(hidden), model_positions
            )
        )
        if attention.source.writes_global:
            _publish_owner(attention, hidden, context, source_rows)
        if attention.source.scores_queries:
            _score_queries(attention, hidden, qr, context)
        query_rows = context.query_row_indices(0, context.query_rows)
        encoded = encode_compact(_query_rows(kv, query_rows), CacheRegion.SWA)
        encoded.check()
        if query_rows is None:
            encoded_rows = encoded.output
        else:
            encoded_rows = encoded.output.new_zeros((context.query_rows, 528))
            encoded_rows.index_copy_(0, query_rows, encoded.output)
        initial = context.restore_swa(attention.layer)
        output = torch.zeros_like(query)
        indices = (
            context.indices_for(attention.layer) if attention.source.ratio else None
        )
        dense_base = (
            torch.arange(INDEX_TOPK, dtype=torch.int32, device=hidden.device)
            if indices is not None
            else None
        )
        # The per-tile reader status checks batch into one DtoH copy per layer
        # (measured: the per-tile .cpu() is the dominant forced host sync in
        # the p13-final py-spy, 126 samples vs single digits elsewhere). The
        # batched check keeps identical accept/reject semantics and still runs
        # before the output is consumed by the inverse RoPE below.
        read_results = []
        for first in range(0, hidden.shape[0], read_queries):
            last = min(first + read_queries, hidden.shape[0])
            query_rows = context.query_row_indices(first, last)
            count = last - first if query_rows is None else query_rows.numel()
            swa = context.swa_queries(
                attention.layer,
                first,
                last,
                initial,
                encoded_rows,
                query_rows=query_rows,
                query_owner=context.single_query_owner,
            )
            global_kv = global_indices = lease = None
            if indices is not None:
                selected = indices[first:last].contiguous()
                slot = RegionSlot(CacheRegion.GLOBAL, attention.source.global_owner)
                values, lease = context.gather_selected(
                    slot,
                    selected,
                    attention.layer,
                    retained_bytes=_bytes(initial.pages.data)
                    + (0 if swa is None else _bytes(swa.pages.data)),
                    query_owner=context.single_query_owner,
                )
                if not count:
                    context.release(lease, attention.layer)
                    del values, lease
                    continue
                selected = _query_rows(selected, query_rows)
                values = _query_rows(values, query_rows)
                pages, table = _packed_pages(values, context._page_specs[slot])
                global_kv = GlobalBinding(pages, table, attention.source.ratio)
                dense = dense_base[None, :].expand(count, -1)
                global_indices = torch.where(selected >= 0, dense, -1).contiguous()
            elif not count:
                continue
            reader = compact_attention
            if backend == "flashmla":
                from rtp_llm.models_py.modules.dsv41.flashmla import (
                    flashmla_compact_attention,
                )

                reader = flashmla_compact_attention
            positions = torch.where(
                _query_rows(context.valid[first:last], query_rows),
                _query_rows(context.positions[first:last], query_rows),
                -1,
            ).to(torch.int32)
            result = reader(
                _query_rows(query[first:last], query_rows).contiguous(),
                torch.arange(count, dtype=torch.int32, device=hidden.device),
                positions,
                torch.full_like(positions, context.replay_floor),
                swa,
                attention.sinks,
                global_kv=global_kv,
                global_indices=global_indices,
            )
            read_results.append(result)
            _copy_query_rows(output[first:last], query_rows, result.output)
            if lease is not None:
                context.release(lease, attention.layer)
                del values, pages, table, lease
            del result, swa, global_kv
        ReaderResult.check_all(read_results)
        del read_results
        context.publish_swa(attention.layer, initial, encoded_rows)
        output = attention_rope(
            context.pack_model_rows(output),
            model_positions,
            global_branch=bool(attention.source.ratio),
            inverse=True,
        )
        output = attention.wo_b(
            grouped_wo_a(output.reshape(-1, 8, 4096), attention.wo_a).flatten(1)
        )
        output = context.unpack_model_rows(output)
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
                "model_rows": context.model_query_rows,
                "reader_rows": context.attention_query_rows,
                "read_queries": read_queries,
                "query_transport_owner": context.single_query_owner,
                "source_rows": (
                    context.end - context.start if attention.source.writes_global else 0
                ),
                "index_rows": (
                    context.attention_query_rows
                    if attention.source.scores_queries
                    else 0
                ),
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
