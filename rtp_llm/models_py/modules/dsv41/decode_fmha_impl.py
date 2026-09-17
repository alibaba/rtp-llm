"""Standard GraphRunner adapter for V4.1 target Q1 and DSpark Q6 verification.

``prepare_model_inputs`` receives the original inputs before replay. The
attention-only prepare hook cannot replace canonical IDs or recovered SWA
ranges. Capture uses explicit invalid rows and never touches a request page.
"""

from contextlib import nullcontext

import torch

from rtp_llm.models_py.modules.dsv41._decode_state_triton import copy_state_bytes_kernel
from rtp_llm.models_py.modules.dsv41.attention import AttentionOwnerCache
from rtp_llm.models_py.modules.dsv41.cache_layout import (
    GLOBAL_OWNERS,
    PAIR_OWNERS,
    CacheRegion,
    RegionSlot,
    layer_sources,
)
from rtp_llm.models_py.modules.dsv41.compact_reader import GlobalBinding, SwaBinding
from rtp_llm.models_py.modules.dsv41.decode_attention import (
    V41DecodeAttention,
    V41DecodeAttentionContext,
    _same_pages,
)
from rtp_llm.models_py.modules.dsv41.decode_compressor import (
    PAIR_SNAPSHOT_BYTES,
    V41DecodePairState,
    _tensor,
    normalize_empty_pair_checkpoint,
)
from rtp_llm.models_py.modules.dsv41.inputs import V41GraphInputBuffers, V41ModelRows
from rtp_llm.models_py.modules.dsv41.linear import is_supported
from rtp_llm.ops.compute_ops import KVCacheRegionName


_REGIONS = {
    CacheRegion.SWA: KVCacheRegionName.SWA_KV,
    CacheRegion.GLOBAL: KVCacheRegionName.DSV41_GLOBAL_KV,
    CacheRegion.INDEX_K: KVCacheRegionName.DSV41_INDEX_KV,
}


def _host(value, shape, dtype, name):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be an explicit tensor")
    _tensor(value, shape, dtype, value.device, name)
    return value.detach().cpu().tolist()


def _copy_bytes(
    source,
    destination,
    source_ids,
    destination_ids,
    active,
    status,
    *,
    copy_bytes,
    source_offset=0,
    destination_offset=0,
    source_min=1,
    destination_min=1,
    zero_inactive=False,
    clear_padding=False,
):
    if (
        source.dtype != torch.uint8
        or destination.dtype != torch.uint8
        or source.ndim != 2
        or destination.ndim != 2
        or source.stride(1) != 1
        or destination.stride(1) != 1
        or source.device != destination.device
        or source_offset < 0
        or source_offset + copy_bytes > source.shape[1]
        or destination_offset < 0
        or destination_offset + copy_bytes > destination.shape[1]
    ):
        raise ValueError("raw decode state copy exceeds its byte layout")
    width = destination.shape[1] - destination_offset if clear_padding else copy_bytes
    copy_state_bytes_kernel[(active.numel(), (width + 511) // 512)](
        source,
        destination,
        source_ids,
        destination_ids,
        active,
        status,
        SOURCE_ROWS=source.shape[0],
        DESTINATION_ROWS=destination.shape[0],
        SOURCE_STRIDE=source.stride(0),
        DESTINATION_STRIDE=destination.stride(0),
        SOURCE_OFFSET=source_offset,
        DESTINATION_OFFSET=destination_offset,
        COPY_BYTES=copy_bytes,
        WRITE_BYTES=width,
        SOURCE_MIN=source_min,
        DESTINATION_MIN=destination_min,
        ZERO_INACTIVE=zero_inactive,
        BLOCK=512,
        num_warps=4,
    )


class V41DecodeFmhaImpl:
    """Bind a fixed B x Q graph bucket to mutable framework-owned request pages.

    Verify writes are tentative. The executor must call ``commit_retained_rows``
    after rejection/stop clipping and successful draft TAIL execution. Only that
    boundary can be returned as a native execution-state certificate.
    """

    def __init__(self, model, inputs, *, query_width=1, capture_capable=False):
        if query_width not in (1, 6):
            raise ValueError("V4.1 target graph requires Q1 or gamma5 verification")
        if getattr(model, "_cp_enabled", False) or model.kv_cache is None:
            raise ValueError("decode Graph requires real complete local engine pages")
        if query_width == 6 and (
            not model.layout.draft_enabled or model.layout.speculative_tokens != 5
        ):
            raise ValueError(
                "Q6 verification requires the complete gamma5 draft layout"
            )
        if not is_supported(model.target.embedding):
            raise RuntimeError("V4.1 decode Graph requires CUDA13 Blackwell")
        attn = inputs.attention_inputs
        if (attn.is_prefill and not attn.is_target_verify) or bool(
            attn.is_target_verify
        ) != (query_width == 6):
            raise ValueError("target graph role and captured query width disagree")
        self.model, self.layout = model, model.layout
        self.query_width = query_width
        self.device = model.target.embedding.device
        self.batch_size, remainder = divmod(inputs.input_ids.numel(), query_width)
        if self.batch_size <= 0 or remainder:
            raise ValueError("decode Graph must have a nonempty fixed bucket")
        self.input_buffers = V41GraphInputBuffers(
            self.batch_size * query_width, device=self.device
        )
        self.rows = self.input_buffers.rows
        self.lookup_outputs = {
            layer: torch.empty(
                (self.batch_size * query_width, 24, 256),
                dtype=torch.bfloat16,
                device=self.device,
            )
            for layer in (1, 14)
        }
        lookup = getattr(model, "_shared_lookup", None)
        self._engram_graph = None
        if lookup is not None and capture_capable:
            self._engram_graph = lookup.graph(external_only=True)
        self._sparse_warmup_stream = None
        self._groups = dict(model._groups)
        self._tables = {}
        supplied = attn.kv_cache_kernel_block_id_device_by_group
        for group in set(self._groups.values()):
            if group >= len(supplied):
                raise ValueError("capture is missing a native cache group table")
            table = supplied[group]
            if (
                table.ndim != 2
                or table.shape[0] != self.batch_size
                or table.shape[1] <= 0
            ):
                raise ValueError("capture cache table differs from the graph bucket")
            _tensor(table, table.shape, torch.int32, self.device, "capture page table")
            self._tables[group] = torch.zeros_like(table)
        self._swa, self._owners, self._floors = {}, {}, {}
        self._draft_pages = {}
        self._pair_pools = dict(model._pair_pools)
        self._current_pages, self._previous_pages, self._committed_pages = {}, {}, {}
        self._copy_status = {}
        self._pair_states, self._pair_load_status, self._pair_store_status = {}, {}, {}
        self._pair_commit_status = {}
        self._commit_copy_status = {}
        self._snapshot_count = self.layout.speculative_tokens + 2
        self._row_ids = torch.arange(
            self.batch_size, dtype=torch.int64, device=self.device
        )
        for group in self._tables:
            self._current_pages[group] = torch.zeros_like(self._row_ids)
            self._previous_pages[group] = torch.zeros_like(self._row_ids)
            self._committed_pages[group] = torch.zeros_like(self._row_ids)
        for layer in range(40):
            pages = model._pages[RegionSlot(CacheRegion.SWA, layer)]
            self._swa[layer] = SwaBinding(
                pages,
                torch.zeros(self.batch_size, dtype=torch.int32, device=self.device),
                torch.zeros(self.batch_size, dtype=torch.int32, device=self.device),
                torch.zeros(self.batch_size, dtype=torch.int32, device=self.device),
            )
            self._floors[layer] = torch.zeros(
                self.batch_size, dtype=torch.int32, device=self.device
            )
            self._copy_status[layer] = torch.zeros_like(self._floors[layer])
            self._commit_copy_status[layer] = torch.zeros_like(self._floors[layer])
        if self.layout.draft_enabled:
            for layer in range(40, 43):
                self._draft_pages[layer] = model._pages[
                    RegionSlot(CacheRegion.SWA, layer)
                ]
                self._commit_copy_status[layer] = torch.zeros(
                    self.batch_size, dtype=torch.int32, device=self.device
                )
        for layer in GLOBAL_OWNERS:
            self._owners[layer] = AttentionOwnerCache(
                GlobalBinding(
                    model._pages[RegionSlot(CacheRegion.GLOBAL, layer)],
                    self._table(layer, KVCacheRegionName.DSV41_GLOBAL_KV),
                    layer_sources(layer).ratio,
                ),
                model._pages[RegionSlot(CacheRegion.INDEX_K, layer)],
                self._table(layer, KVCacheRegionName.DSV41_INDEX_KV),
            )
        for layer, pool in self._pair_pools.items():
            expected = ((self._snapshot_count * PAIR_SNAPSHOT_BYTES + 511) // 512) * 512
            if pool.dtype != torch.uint8 or pool.ndim != 2 or pool.shape[1] != expected:
                raise ValueError("decode pair pool differs from its snapshot layout")
            self._pair_states[layer] = V41DecodePairState(
                torch.zeros(
                    (self.batch_size, PAIR_SNAPSHOT_BYTES),
                    dtype=torch.uint8,
                    device=self.device,
                )
            )
            self._pair_load_status[layer] = torch.zeros(
                self.batch_size, dtype=torch.int32, device=self.device
            )
            self._pair_store_status[layer] = torch.zeros_like(
                self._pair_load_status[layer]
            )
            self._pair_commit_status[layer] = torch.zeros_like(
                self._pair_load_status[layer]
            )
        self.context = V41DecodeAttentionContext(
            self.layout,
            self._swa,
            self._owners,
            batch_size=self.batch_size,
            query_width=query_width,
            max_tokens=model.config.max_seq_len,
        )
        for layer, block in enumerate(model.target.blocks):
            attention = getattr(block.attention, "attention", block.attention)
            V41DecodeAttention(attention, self.context)
        self.context.prepare(
            self.context.start_positions,
            self.context.valid_rows,
            swa=self._swa,
            owners=self._owners,
            pair_states=self._pair_states,
            replay_floors=self._floors,
        )
        self._prepare_generation = 0
        self._prepared_epoch = torch.zeros((), dtype=torch.int64, device=self.device)
        self._executed_epoch = torch.full_like(self._prepared_epoch, -1)
        self._committed_epoch = torch.full_like(self._prepared_epoch, -1)
        self._retained_rows = torch.zeros(
            self.batch_size, dtype=torch.int32, device=self.device
        )
        self._draft_committed = not self.layout.draft_enabled
        self._signature = None
        self._requests = []
        self._ranges = []

    def _table(self, layer, region):
        return self._tables[self._groups[(layer, int(region))]]

    def support_cuda_graph(self):
        return True

    def engram_capture_scope(self):
        if not torch.cuda.is_current_stream_capturing() or self._engram_graph is None:
            return nullcontext()
        return self._engram_graph.capture(external=True)

    def prepare_cuda_graph(self, attn_inputs):
        if bool(attn_inputs.is_target_verify) != (self.query_width == 6):
            raise ValueError("target graph received another attention role")
        self._warmup_sparse_indexer()
        # Only prepare_model_inputs has the current canonical rows and state.

    def _request_metadata(self, inputs):
        attn = inputs.attention_inputs
        if (
            (attn.is_prefill and not attn.is_target_verify)
            or bool(attn.is_target_verify) != (self.query_width == 6)
            or attn.context_parallel_info is not None
        ):
            raise ValueError("decode Graph received another execution role")
        batch = inputs.request_id.numel()
        if (
            batch > self.batch_size
            or inputs.input_ids.numel() != batch * self.query_width
        ):
            raise ValueError("request-major decode rows must fit this graph bucket")
        ids = _host(inputs.request_id, (batch,), torch.int64, "request IDs")
        positions = (
            attn.prefix_lengths if self.query_width == 6 else attn.sequence_lengths
        )
        starts = _host(positions, (batch,), torch.int32, "decode positions")
        fake = _host(inputs.v41_is_fake, (batch,), torch.bool, "fake flags")
        ready = _host(inputs.v41_state_ready, (batch,), torch.bool, "state-ready flags")
        if len({rid for rid, masked in zip(ids, fake) if not masked}) != sum(
            not value for value in fake
        ):
            raise ValueError("live decode request IDs must be unique")
        if any(
            not masked
            and (
                rid < 0
                or start < 0
                or start >= self.context.max_tokens
                or (start > 0 and not restored)
            )
            for rid, start, masked, restored in zip(ids, starts, fake, ready)
        ):
            raise ValueError(
                "decode Graph needs complete state at every live request start"
            )
        return ids, starts, fake

    @torch.inference_mode()
    def prepare_model_inputs(self, inputs):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare original model inputs before graph replay")
        self._warmup_sparse_indexer()
        ids, starts, fake = self._request_metadata(inputs)
        batch = len(ids)
        rows = V41ModelRows.from_model_inputs(inputs)
        row_valid = rows.valid.view(batch, self.query_width)
        counts_cpu = _host(
            row_valid.sum(1, dtype=torch.int32),
            (batch,),
            torch.int32,
            "valid row counts",
        )
        if any((count == 0) != masked for count, masked in zip(counts_cpu, fake)):
            raise ValueError("decode rows disagree with fake request flags")
        if any(
            start + count > self.context.max_tokens
            for start, count in zip(starts, counts_cpu)
        ):
            raise ValueError("live decode rows exceed the model context")
        execution = _host(
            inputs.v41_execution_context,
            (batch, 4),
            torch.int64,
            "execution boundaries",
        )
        ranges = _host(
            inputs.v41_swa_ranges, (batch, 43, 3), torch.int64, "restored SWA ranges"
        )
        for index, (start, masked) in enumerate(zip(starts, fake)):
            if masked:
                continue
            if execution[index][1] != start or execution[index][2] != start:
                raise ValueError(
                    "decode Graph cannot skip an incomplete execution boundary"
                )
            for layer in range(43 if self.layout.draft_enabled else 40):
                if start == 0:
                    ranges[index][layer] = [0, 0, 0]
                begin, end, floor = ranges[index][layer]
                if (
                    begin < 0
                    or begin > max(0, start - 128)
                    or end != start
                    or end - begin > self.layout.swa_entries
                    or floor < 0
                    or floor > begin
                    or (
                        (self.model.config.dsv41_replay_mode == "full" or layer <= 20)
                        and floor != 0
                    )
                ):
                    raise ValueError("decode Graph requires exact complete SWA ranges")
        if self.model.layout != self.layout:
            raise ValueError("model cache layout changed after graph creation")
        for layer in range(40):
            _same_pages(
                self._swa[layer].pages,
                self.model._pages[RegionSlot(CacheRegion.SWA, layer)],
            )
        for layer, pages in self._draft_pages.items():
            _same_pages(pages, self.model._pages[RegionSlot(CacheRegion.SWA, layer)])
        for layer, owner in self._owners.items():
            _same_pages(
                owner.global_kv.pages,
                self.model._pages[RegionSlot(CacheRegion.GLOBAL, layer)],
            )
            _same_pages(
                owner.index_pages,
                self.model._pages[RegionSlot(CacheRegion.INDEX_K, layer)],
            )
        for layer, pool in self._pair_pools.items():
            current = self.model._pair_pools[layer]
            if (
                current.data_ptr() != pool.data_ptr()
                or current.shape != pool.shape
                or current.stride() != pool.stride()
            ):
                raise ValueError("pair pool backing changed after graph creation")
        self.input_buffers.update(rows)
        padded_starts = torch.tensor(
            starts + [0] * (self.batch_size - batch),
            dtype=torch.int64,
            device=self.device,
        )
        counts = torch.tensor(
            counts_cpu + [0] * (self.batch_size - batch),
            dtype=torch.int32,
            device=self.device,
        )
        active = counts > 0
        tables = inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group
        for group, destination in self._tables.items():
            if group >= len(tables):
                raise ValueError("decode input is missing a native cache group")
            source = tables[group]
            if (
                source.ndim != 2
                or source.shape[0] < batch
                or source.shape[1] > destination.shape[1]
            ):
                raise ValueError("decode page table exceeds captured group capacity")
            _tensor(
                source, source.shape, torch.int32, self.device, "original page table"
            )
            destination.zero_()
            destination[:batch, : source.shape[1]].copy_(source[:batch])
            fixed = (
                self.model.kv_cache.group_seq_size_per_block[group]
                == self.layout.reuse_unit
            )
            unit = self.layout.reuse_unit if fixed else self.layout.token_block_size
            logical = (padded_starts + counts - 1).clamp_min(0) // unit
            previous = (padded_starts - 1).clamp_min(0) // unit
            selected = destination.gather(
                1, logical.clamp_max(destination.shape[1] - 1)[:, None]
            ).squeeze(1)
            old = destination.gather(
                1, previous.clamp_max(destination.shape[1] - 1)[:, None]
            ).squeeze(1)
            self._current_pages[group].copy_(selected)
            self._previous_pages[group].copy_(old)
        # Transfer validated host ranges once, before the per-layer device work.
        # Each pageable H2D tensor construction otherwise fences earlier copies.
        range_layers = tuple(self._swa)
        range_values = [
            [ranges[index][layer] for layer in range_layers]
            if not fake[index]
            else [[0, 0, 0] for _ in range_layers]
            for index in range(batch)
        ]
        range_values += [
            [[0, 0, 0] for _ in range_layers]
            for _ in range(self.batch_size - batch)
        ]
        device_ranges = torch.tensor(
            range_values, dtype=torch.int32, device=self.device
        ).unbind(1)
        for (layer, binding), values in zip(self._swa.items(), device_ranges):
            group = self._groups[(layer, int(KVCacheRegionName.SWA_KV))]
            previous, current = self._previous_pages[group], self._current_pages[group]
            binding.page_ids.copy_(current)
            binding.valid_starts.copy_(values[:, 0])
            binding.valid_ends.copy_(values[:, 1])
            self._floors[layer].copy_(values[:, 2])
            _copy_bytes(
                binding.pages.data,
                binding.pages.data,
                previous,
                current,
                active & (padded_starts > 0) & (previous != current),
                self._copy_status[layer],
                copy_bytes=binding.pages.data.shape[1],
            )
        for layer, pool in self._pair_pools.items():
            group = self._groups[(layer, int(KVCacheRegionName.DSV41_PAIR_STATE))]
            state = self._pair_states[layer]
            _copy_bytes(
                pool,
                state.storage,
                self._previous_pages[group],
                self._row_ids,
                active & (padded_starts > 0),
                self._pair_load_status[layer],
                copy_bytes=PAIR_SNAPSHOT_BYTES,
                source_offset=(self._snapshot_count - 1) * PAIR_SNAPSHOT_BYTES,
                destination_min=0,
                zero_inactive=True,
            )
            # Active requests have passed ready and exact execution/SWA boundary
            # validation above; zero memory checkpoints omit the absolute header.
            previous = self._previous_pages[group]
            normalize_empty_pair_checkpoint(
                pool.index_select(0, previous.clamp(0, pool.shape[0] - 1).long()),
                state.storage,
                padded_starts,
                active & (previous > 0) & (previous < pool.shape[0]),
                reuse_unit=self.layout.reuse_unit,
            )
        self.context.prepare(
            padded_starts,
            counts,
            swa=self._swa,
            owners=self._owners,
            pair_states=self._pair_states,
            replay_floors=self._floors,
        )
        self._requests = list(zip(ids, starts, fake))
        self._ranges = ranges
        self._signature = tuple(self._requests)
        self._prepare_generation += 1
        self._prepared_epoch.fill_(self._prepare_generation)
        self._committed_epoch.fill_(-1)
        self._retained_rows.zero_()
        self._draft_committed = not self.layout.draft_enabled
        self.model._active_v41_graph_impl = self

    def _warmup_sparse_indexer(self):
        stream = torch.cuda.current_stream(self.device)
        stream_id = stream.cuda_stream
        if self._sparse_warmup_stream == stream_id:
            return
        from rtp_llm.models_py.modules.dsv41.indexer import warmup_sparse_indexer

        warmup_sparse_indexer(self.device)
        self._sparse_warmup_stream = stream_id

    def begin_forward(self):
        self.context.begin_forward()

    @torch.inference_mode()
    def finish_forward(self):
        active = self.context.valid_rows > 0
        for layer, pool in self._pair_pools.items():
            group = self._groups[(layer, int(KVCacheRegionName.DSV41_PAIR_STATE))]
            snapshots = self.context.layers[layer].compressor.pair_snapshots.flatten(1)
            _copy_bytes(
                snapshots,
                pool,
                self._row_ids,
                self._current_pages[group],
                active,
                self._pair_store_status[layer],
                copy_bytes=(self.query_width + 1) * PAIR_SNAPSHOT_BYTES,
                source_min=0,
                clear_padding=True,
            )
        self._executed_epoch.copy_(self._prepared_epoch)
        if self.query_width == 1 and not self.layout.draft_enabled:
            self._commit(self.context.valid_rows)

    @torch.inference_mode()
    def _commit(self, retained_rows):
        """Device-only state selection; the caller supplies an executed round."""
        self._retained_rows.copy_(retained_rows)
        end = self.context.start_positions + retained_rows
        active = self.context.valid_rows > 0
        for group, table in self._tables.items():
            unit = self.model.kv_cache.group_seq_size_per_block[group]
            logical = (end - 1).clamp_min(0) // unit
            selected = table.gather(
                1, logical.clamp_max(table.shape[1] - 1)[:, None]
            ).squeeze(1)
            self._committed_pages[group].copy_(selected)
        for layer, binding in self.context.swa.items():
            group = self._groups[(layer, int(KVCacheRegionName.SWA_KV))]
            previous, current = self._current_pages[group], self._committed_pages[group]
            _copy_bytes(
                binding.pages.data,
                binding.pages.data,
                previous,
                current,
                active & (end > 0) & (previous != current),
                self._commit_copy_status[layer],
                copy_bytes=binding.pages.data.shape[1],
            )
            binding.page_ids.copy_(current)
            binding.valid_ends.copy_(torch.where(active, end, binding.valid_ends))
        for layer, pool in self._pair_pools.items():
            group = self._groups[(layer, int(KVCacheRegionName.DSV41_PAIR_STATE))]
            selected = self.context.layers[layer].compressor.select_pair(retained_rows)
            _copy_bytes(
                selected.storage,
                pool,
                self._row_ids,
                self._committed_pages[group],
                active & (end > 0),
                self._pair_commit_status[layer],
                copy_bytes=PAIR_SNAPSHOT_BYTES,
                destination_offset=(self._snapshot_count - 1) * PAIR_SNAPSHOT_BYTES,
                source_min=0,
                clear_padding=True,
            )
        for layer, pages in self._draft_pages.items():
            group = self._groups[(layer, int(KVCacheRegionName.SWA_KV))]
            previous, current = self._current_pages[group], self._committed_pages[group]
            _copy_bytes(
                pages.data,
                pages.data,
                previous,
                current,
                active & (end > 0) & (previous != current),
                self._commit_copy_status[layer],
                copy_bytes=pages.data.shape[1],
            )
        self._committed_epoch.copy_(self._prepared_epoch)

    @torch.inference_mode()
    def commit_retained_rows(self, retained_rows, *, draft_committed=False):
        """Commit actual materialized rows after rejection, EOS and draft TAIL.

        Counts include the executed anchor and accepted proposals only. The
        correction/bonus token has been sampled but has no target KV yet.
        This call is ordered on the model/replay stream before state publication.
        """
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("commit retained rows after verification replay")
        batch = len(self._requests)
        _tensor(retained_rows, (batch,), torch.int32, self.device, "retained rows")
        if (
            int(self._executed_epoch) != self._prepare_generation
            or self._signature is None
        ):
            raise RuntimeError("retained rows do not belong to an executed round")
        if int(self._committed_epoch) == self._prepare_generation:
            raise RuntimeError("the executed round has already been committed")
        if self.layout.draft_enabled and not draft_committed:
            raise RuntimeError(
                "target state cannot be committed before all draft stages"
            )
        counts = torch.zeros_like(self._retained_rows)
        counts[:batch].copy_(retained_rows)
        self._commit(counts)
        self._draft_committed = bool(draft_committed)
