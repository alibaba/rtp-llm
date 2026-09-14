"""Framework-owned V4.1 target execution and CP prefill publication.

The standard engine owns requests, allocation, copying and sampling. This
adapter binds its physical pages and returns normalized rank-local hidden
states. CP prefill uses the engine's canonical zigzag row and page metadata.
"""

from contextlib import nullcontext
from dataclasses import dataclass, fields
import os

import torch
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.base.common.kvcache_store import (
    create_write_cache_store_impl,
)
from rtp_llm.models_py.modules.dsv4.cp import build_cp_context_for_forward
from rtp_llm.models_py.modules.dsv41.attention import (
    AttentionOwnerCache,
    V41Attention,
    V41AttentionCache,
)
from rtp_llm.models_py.modules.dsv41.cache_layout import (
    GLOBAL_OWNERS,
    PAIR_OWNERS,
    CacheLayout,
    CacheRegion,
    RegionSlot,
    layer_sources,
)
from rtp_llm.models_py.modules.dsv41.ced import ReplayConfig, ReplayMode
from rtp_llm.models_py.modules.dsv41.compact_reader import (
    CompactPages,
    GlobalBinding,
    SwaBinding,
)
from rtp_llm.models_py.modules.dsv41.compressor import PairCarry
from rtp_llm.models_py.modules.dsv41.inputs import V41ModelRows
from rtp_llm.models_py.modules.dsv41.linear import warmup_block32_linears
from rtp_llm.models_py.modules.dsv41.moe import V41MoE
from rtp_llm.models_py.modules.dsv41.transformer import V41ImageFeatures, V41TargetModel
from rtp_llm.ops.compute_ops import KVCacheRegionName, PyModelOutputs
from torch import nn

_REGIONS = {
    CacheRegion.SWA: KVCacheRegionName.SWA_KV,
    CacheRegion.GLOBAL: KVCacheRegionName.DSV41_GLOBAL_KV,
    CacheRegion.INDEX_K: KVCacheRegionName.DSV41_INDEX_KV,
}
_PAIR_BYTES = 4112


def _host_vector(value, count, dtype, name):
    if count == 0 and value is None:
        return []
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != (count,)
        or value.dtype != dtype
        or not value.is_contiguous()
    ):
        raise ValueError(f"V4.1 {name} must contain one {dtype} value per request")
    return value.detach().cpu().tolist()


def _request_ranges(attn, row_count):
    if attn.is_target_verify:
        raise NotImplementedError("V4.1 target verify requires DSpark integration")
    if attn.context_parallel_info is not None:
        raise NotImplementedError(
            "V4.1 CP prefill requires actual CP page communication"
        )
    batch_size = attn.input_lengths.numel()
    if attn.is_prefill:
        lengths = _host_vector(attn.input_lengths, batch_size, torch.int32, "lengths")
        starts = _host_vector(attn.prefix_lengths, batch_size, torch.int32, "prefixes")
        if attn.sequence_lengths.numel():
            raise ValueError("V4.1 standard engine must separate prefill and decode")
    else:
        starts = _host_vector(
            attn.sequence_lengths, batch_size, torch.int32, "positions"
        )
        lengths = [1] * batch_size
        if attn.prefix_lengths.numel():
            raise ValueError("V4.1 standard engine must separate prefill and decode")
    if any(start < 0 or length < 0 for start, length in zip(starts, lengths)):
        raise ValueError("V4.1 request ranges cannot be negative")
    if sum(lengths) > row_count:
        raise ValueError("V4.1 request lengths exceed the supplied canonical rows")
    return starts, lengths


def _pair_view(pool, page, snapshot):
    begin = snapshot * _PAIR_BYTES
    return pool[page, begin : begin + _PAIR_BYTES]


def _read_pair(raw, owner, request_id, identity, position):
    stored_position = int(raw[4096:4104].view(torch.int64).item())
    valid = int(raw[4104:4108].view(torch.int32).item())
    if stored_position != position or valid != position % 2:
        raise ValueError("V4.1 pair bytes do not match the restored execution boundary")
    if not valid:
        return PairCarry.empty(owner, request_id, identity, position)
    return PairCarry(
        owner,
        request_id,
        identity,
        position,
        raw[:2048].view(torch.float32),
        raw[2048:4096].view(torch.float32),
    )


def _write_pair(raw, pair):
    raw.zero_()
    if pair.next_position % 2:
        raw[:2048].view(torch.float32).copy_(pair.partial_kv)
        raw[2048:4096].view(torch.float32).copy_(pair.partial_score)
    raw[4096:4104].view(torch.int64).fill_(pair.next_position)
    raw[4104:4108].view(torch.int32).fill_(pair.next_position % 2)


@dataclass
class _Request:
    first: int
    last: int
    context: object
    pair_outputs: dict
    pair_initials: dict
    batch: int = 0


class _BatchedAttention(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention

    def forward(self, hidden, requests):
        if hasattr(requests, "layers"):
            return requests.layers[self.attention.layer](hidden, requests)
        if hasattr(requests, "cache"):
            return self.attention(hidden, requests)
        result = torch.zeros_like(hidden)
        for request in requests:
            result[request.first : request.last] = self.attention(
                hidden[request.first : request.last].contiguous(), request.context
            )
        return result


class DeepSeekV41Model(GptModelBase):
    """Consume engine request IDs, execution lengths and state-ready metadata.

    For a nonzero execution start, ``v41_state_ready`` certifies all target SWA
    and ratio2 carry at that boundary. The engine must also privatize every page
    written by this forward; a global/index data hit alone cannot set the flag.
    ``v41_is_fake`` identifies scheduler placeholders whose rows must be invalid.
    They participate in the target's collectives without accessing request KV.
    Each successful forward returns normalized hidden states. The caller may
    publish that completed boundary after its usual CUDA completion ordering.
    """

    def __init__(
        self,
        config,
        parallelism_config,
        weight,
        *,
        tokenizer,
        shared_lookup,
        kv_cache_config,
        max_tokens_per_rank,
        max_generate_batch_size,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
        vision=None,
        prefill_draft=None,
    ):
        super().__init__(
            config,
            parallelism_config,
            weight,
            max_generate_batch_size,
            fmha_config,
            py_hw_kernel_config,
            device_resource_config,
        )
        cp = parallelism_config.prefill_cp_config
        if parallelism_config.get_attn_tp_size() != 1 or cp.prefill_cp_size != 8:
            raise ValueError(
                "V4.1 target requires attention TP1 and explicit CP8 layout"
            )
        self._cp_enabled = cp.is_enabled()
        self._cp_rank = parallelism_config.tp_rank
        self._forward_epoch = 0
        if self._cp_enabled and (
            parallelism_config.tp_size != 8
            or parallelism_config.ep_size != 8
            or not cp.kv_cache_sharded
        ):
            raise ValueError("V4.1 CP prefill requires sharded CP8/EP8 pages")
        self._capture_aux = tuple(config.capture_aux_hidden_layer_ids or ())
        if self._capture_aux and self._capture_aux != (37, 38, 39):
            raise ValueError("V4.1 DSpark requires target aux layers 37/38/39")
        self._replay = ReplayConfig(ReplayMode(config.dsv41_replay_mode))
        if config.dsv41_tail_policy_version != 1:
            raise ValueError("V4.1 requires tail policy version 1")
        if self._cp_enabled and bool(self._capture_aux) != (prefill_draft is not None):
            raise ValueError(
                "V4.1 speculative prefill requires its actual draft writer"
            )
        if type(max_tokens_per_rank) is not int or max_tokens_per_rank <= 0:
            raise ValueError(
                "V4.1 target needs a positive admitted per-rank token budget"
            )
        self.layout = CacheLayout(
            token_block_size=(
                kv_cache_config.seq_size_per_block
                if kv_cache_config.seq_size_per_block > 0
                else 128
            ),
            cp_size=8,
            speculative_tokens=5 if self._capture_aux else 0,
            draft_enabled=bool(self._capture_aux),
        )
        self.identity = self._replay.cache_identity(
            shared_lookup.shared.manifest["revision"], self.layout
        )
        self._pages, self._pair_pools = {}, {}
        self._raw_pages = {}
        self._groups = {}
        self._max_tokens = config.max_seq_len
        self._shared_lookup = shared_lookup
        self.prefill_draft = prefill_draft
        self._active_v41_graph_impl = None
        self.max_tokens_per_rank = max_tokens_per_rank
        self.target = V41TargetModel.from_model_weights(
            config.dsv41_config,
            weight,
            attention_factory=lambda layer, local: _BatchedAttention(
                V41Attention.from_weights(layer, local, layout=self.layout)
            ),
            moe_factory=lambda layer, local: V41MoE.from_weights(
                config.dsv41_config,
                layer,
                local,
                ep_size=parallelism_config.ep_size,
                ep_rank=parallelism_config.ep_rank,
                max_tokens_per_rank=max_tokens_per_rank,
            ),
            shared_lookup=shared_lookup,
            tokenizer=tokenizer.get_real_tokenizer(),
            vision=vision,
            head_tp_size=parallelism_config.tp_size,
            head_tp_rank=parallelism_config.tp_rank,
        )
        self._mtp_aux_buffer = (
            self.target.embedding.new_zeros(
                (max_generate_batch_size * 6, config.hidden_size * 3)
            )
            if self._capture_aux and not self._cp_enabled
            else None
        )
        warmup_block32_linears(self.target, max_rows=max_tokens_per_rank)
        if self.prefill_draft is not None:
            warmup_block32_linears(self.prefill_draft, max_rows=128)
        if os.environ.get("DSV41_SPARSE_INDEXER") == "1":
            from rtp_llm.models_py.modules.dsv41.indexer import warmup_sparse_indexer

            warmup_sparse_indexer(self.target.embedding.device)

    def initialize(self, init_resource):
        if bool(init_resource.is_speculative) != self.layout.draft_enabled:
            raise ValueError("V4.1 model and engine speculative cache layouts differ")
        super().initialize(init_resource)
        self._pages.clear()
        self._pair_pools.clear()
        self._raw_pages = {}
        self._groups.clear()
        if self.kv_cache is None:
            return True
        cache = self.kv_cache
        if (
            cache.seq_size_per_block != self.layout.token_block_size
            or cache.kernel_seq_size_per_block != self.layout.token_block_size
            or len(cache.layer_region_to_group_id)
            != (43 if self.layout.draft_enabled else 40)
        ):
            raise ValueError(
                "V4.1 framework cache differs from target block/layer geometry"
            )
        device = self.target.embedding.device
        for page in self.layout.pages:
            layer, region = page.slot.owner_layer, _REGIONS[page.slot.region]
            pool = cache.get_raw_pool_tensor(layer, region)
            self._raw_pages[page.slot] = pool
            self._groups[(layer, int(region))] = self._group(layer, region)
            if getattr(self, "_cp_enabled", False):
                expected = page.page_stride_bytes
                if page.slot.region == CacheRegion.SWA:
                    expected //= self.layout.cp_size
                if (
                    pool.dtype != torch.uint8
                    or pool.device != device
                    or pool.ndim != 2
                    or pool.shape[1] != expected
                    or pool.stride(1) != 1
                    or pool.stride(0) < expected
                ):
                    raise ValueError("V4.1 CP pool does not match its physical slice")
                continue
            pages = CompactPages(pool, page.slot.region, page.entries)
            pages.validate(device)
            if pool.shape[1] != page.page_stride_bytes:
                raise NotImplementedError(
                    "V4.1 eager adapter requires complete physical pages"
                )
            self._pages[page.slot] = pages
        for layer in PAIR_OWNERS:
            region = KVCacheRegionName.DSV41_PAIR_STATE
            pool = cache.get_raw_pool_tensor(layer, region)
            snapshots = self.layout.speculative_tokens + 2
            expected = ((snapshots * _PAIR_BYTES + 511) // 512) * 512
            if getattr(self, "_cp_enabled", False):
                expected //= self.layout.cp_size
            if (
                pool.dtype != torch.uint8
                or pool.device != device
                or pool.ndim != 2
                or pool.shape[1] != expected
                or pool.stride(1) != 1
                or pool.stride(0) < expected
                or pool.stride(0) % (64 if getattr(self, "_cp_enabled", False) else 512)
            ):
                raise ValueError(
                    "V4.1 pair pages differ from the configured snapshot count"
                )
            self._pair_pools[layer] = pool
            self._groups[(layer, int(region))] = self._group(layer, region)
        return True

    def _group(self, layer, region):
        gid = self.kv_cache.layer_region_to_group_id[layer][int(region)]
        if (
            gid < 0
            or gid >= len(self.kv_cache.group_region_names)
            or self.kv_cache.group_region_names[gid] != region
        ):
            raise ValueError(
                "V4.1 physical owner has no matching framework cache group"
            )
        expected = (
            self.layout.reuse_unit
            if region in (KVCacheRegionName.SWA_KV, KVCacheRegionName.DSV41_PAIR_STATE)
            else self.layout.token_block_size
        )
        if self.kv_cache.group_seq_size_per_block[gid] != expected:
            raise ValueError(
                "V4.1 cache group has a different canonical CP block namespace"
            )
        return gid

    def prepare_fmha_impl(self, inputs, is_cuda_graph=False):
        if self.kv_cache is None and not is_cuda_graph:
            return None
        attn = inputs.attention_inputs
        if (
            is_cuda_graph
            or (not attn.is_prefill and self.layout.draft_enabled)
            or attn.is_target_verify
        ):
            from rtp_llm.models_py.modules.dsv41.decode_fmha_impl import (
                V41DecodeFmhaImpl,
            )

            impl = V41DecodeFmhaImpl(
                self,
                inputs,
                query_width=6 if attn.is_target_verify else 1,
                capture_capable=is_cuda_graph,
            )
            if not is_cuda_graph:
                impl.prepare_model_inputs(inputs)
            return impl
        return None

    def get_execution_states(self, inputs):
        if self._active_v41_graph_impl is None:
            raise RuntimeError("V4.1 replay has no current execution context")
        return self._active_v41_graph_impl.get_execution_states(inputs)

    def commit_retained_rows(self, retained_rows, inputs):
        if self._active_v41_graph_impl is None:
            raise RuntimeError("V4.1 speculative commit has no active verify context")
        self._active_v41_graph_impl.commit_retained_rows(
            retained_rows, draft_committed=self.layout.draft_enabled
        )
        return self.get_execution_states(inputs)

    def get_mtp_target_hidden_states(self, num_tokens):
        if self._mtp_aux_buffer is None:
            return None
        if not 0 <= num_tokens <= self._mtp_aux_buffer.shape[0]:
            raise ValueError("V4.1 aux request exceeds the fixed decode buffer")
        return self._mtp_aux_buffer[:num_tokens]

    def has_mtp_hidden_buffer(self):
        return self._mtp_aux_buffer is not None

    def _table(self, attn, batch, layer, region):
        gid = self._groups[(layer, int(region))]
        tables = attn.kv_cache_kernel_block_id_device_by_group
        if gid >= len(tables):
            raise ValueError("V4.1 request is missing a physical owner block table")
        table = tables[gid]
        if (
            table.ndim != 2
            or table.dtype != torch.int32
            or table.device != self.target.embedding.device
            or batch >= table.shape[0]
        ):
            raise ValueError("V4.1 block table has invalid request geometry or device")
        return table[batch : batch + 1].contiguous()

    @staticmethod
    def _page_id(table, logical, pool):
        if logical < 0 or logical >= table.shape[1]:
            raise ValueError("V4.1 execution boundary has no allocated physical page")
        page = int(table[0, logical].item())
        if not 0 < page < pool.shape[0]:
            raise ValueError("V4.1 execution requires an allocated nonzero page")
        return page

    def _requests(self, inputs, rows):
        attn = inputs.attention_inputs
        if attn.context_parallel_info is not None:
            return self._cp_requests(inputs, rows)
        starts, lengths = _request_ranges(attn, rows.token_ids.numel())
        batch_size = len(starts)
        identities = _host_vector(
            inputs.request_id, batch_size, torch.int64, "request_id"
        )
        ready = _host_vector(
            inputs.v41_state_ready, batch_size, torch.bool, "state_ready"
        )
        fake = _host_vector(
            inputs.v41_is_fake, batch_size, torch.bool, "fake request flags"
        )
        requests, copies, occupied = [], [], set()
        first = 0
        for batch, (start, length) in enumerate(zip(starts, lengths)):
            last, end = first + length, start + length
            if fake[batch]:
                if bool(rows.valid[first:last].any()):
                    raise ValueError("V4.1 fake request rows must all be invalid")
                first = last
                continue
            if end > self._max_tokens or (start and not ready[batch]):
                raise ValueError(
                    "V4.1 execution requires restored state at its admitted start"
                )
            if not length:
                continue
            if not bool(rows.valid[first:last].all()):
                raise ValueError("V4.1 live request rows cannot contain padding")
            current = (end - 1) // self.layout.reuse_unit
            previous = (start - 1) // self.layout.reuse_unit
            swa, owners, pair_outputs, pair_initials = {}, {}, {}, {}
            for layer in range(40):
                pages = self._pages[RegionSlot(CacheRegion.SWA, layer)]
                table = self._table(attn, batch, layer, KVCacheRegionName.SWA_KV)
                destination = self._page_id(table, current, pages.data)
                key = (pages.data.data_ptr(), destination)
                if key in occupied:
                    raise ValueError(
                        "V4.1 active requests cannot share writable SWA pages"
                    )
                occupied.add(key)
                if start:
                    source = self._page_id(table, previous, pages.data)
                    if source != destination:
                        copies.append((pages.data[destination], pages.data[source]))
                swa[layer] = SwaBinding(
                    pages,
                    table[0, current : current + 1],
                    torch.tensor(
                        [max(0, start - pages.entries_per_page)],
                        device=table.device,
                        dtype=torch.int32,
                    ),
                    torch.tensor([start], device=table.device, dtype=torch.int32),
                )
            request_id = str(identities[batch])
            for layer in GLOBAL_OWNERS:
                global_pages = self._pages[RegionSlot(CacheRegion.GLOBAL, layer)]
                index_pages = self._pages[RegionSlot(CacheRegion.INDEX_K, layer)]
                pair = None
                if layer in PAIR_OWNERS:
                    pool = self._pair_pools[layer]
                    table = self._table(
                        attn, batch, layer, KVCacheRegionName.DSV41_PAIR_STATE
                    )
                    destination = self._page_id(table, current, pool)
                    key = (pool.data_ptr(), destination)
                    if key in occupied:
                        raise ValueError(
                            "V4.1 active requests cannot share writable pair pages"
                        )
                    occupied.add(key)
                    pair = PairCarry.empty(layer, request_id, self.identity)
                    if start:
                        source = self._page_id(table, previous, pool)
                        pair = _read_pair(
                            _pair_view(pool, source, 1),
                            layer,
                            request_id,
                            self.identity,
                            start,
                        )
                    pair_outputs[layer] = _pair_view(pool, destination, 1)
                    pair_initials[layer] = (_pair_view(pool, destination, 0), pair)
                owners[layer] = AttentionOwnerCache(
                    GlobalBinding(
                        global_pages,
                        self._table(
                            attn, batch, layer, KVCacheRegionName.DSV41_GLOBAL_KV
                        ),
                        layer_sources(layer).ratio,
                    ),
                    index_pages,
                    self._table(attn, batch, layer, KVCacheRegionName.DSV41_INDEX_KV),
                    start,
                    pair,
                )
                for pages, table in (
                    (global_pages, owners[layer].global_kv.page_table),
                    (index_pages, owners[layer].index_table),
                ):
                    for logical in range(
                        start // self.layout.token_block_size,
                        (end + self.layout.token_block_size - 1)
                        // self.layout.token_block_size,
                    ):
                        page_id = self._page_id(table, logical, pages.data)
                        key = (pages.data.data_ptr(), page_id)
                        if key in occupied:
                            raise ValueError(
                                "V4.1 requests cannot share writable global/index pages"
                            )
                        occupied.add(key)
            cache = V41AttentionCache(
                request_id,
                self.identity,
                self.layout,
                self._max_tokens,
                swa,
                owners,
                {layer: start for layer in range(40)},
            )
            requests.append(
                _Request(
                    first,
                    last,
                    cache.begin_forward(epoch=0, start=start, end=end),
                    pair_outputs,
                    pair_initials,
                    batch,
                )
            )
            first = last
        if bool(rows.valid[first:].any()):
            raise ValueError("V4.1 valid rows are missing their request boundary")
        # Resolve every request before touching framework-owned destination bytes.
        for destination, source in copies:
            destination.copy_(source)
        for request in requests:
            for destination, pair in request.pair_initials.values():
                _write_pair(destination, pair)
        return requests

    def _cp_requests(self, inputs, rows):
        from rtp_llm.models_py.modules.dsv41.cp import begin_cp_request

        if not self._cp_enabled or not inputs.attention_inputs.is_prefill:
            raise ValueError("V4.1 CP row metadata requires its CP8 prefill role")
        attn = inputs.attention_inputs
        cp_info = attn.context_parallel_info
        cp_context = build_cp_context_for_forward(
            cp_info,
            self.layout.cp_size,
            self._cp_rank,
            rows.token_ids.numel(),
            rows.token_ids.device,
            prefix_lengths=attn.prefix_lengths,
            prefix_lengths_host=cp_info.prefill_prefix_lengths_cpu,
            chunk_lengths_device=attn.input_lengths,
            kv_cache_sharded=True,
        )
        lengths = cp_context.chunk_lengths_per_req
        if lengths is None or sum(lengths) != rows.token_ids.numel():
            raise ValueError("V4.1 CP metadata does not cover every local row")
        batch_size = len(lengths)
        identities = _host_vector(
            inputs.request_id, batch_size, torch.int64, "request_id"
        )
        ready = _host_vector(
            inputs.v41_state_ready, batch_size, torch.bool, "state_ready"
        )
        fake = _host_vector(
            inputs.v41_is_fake, batch_size, torch.bool, "fake request flags"
        )
        requests, first, occupied = [], 0, set()
        self._forward_epoch += 1
        for batch, length in enumerate(lengths):
            last = first + length
            if fake[batch]:
                if bool(rows.valid[first:last].any()):
                    raise ValueError("V4.1 CP placeholder rows must all be invalid")
                first = last
                continue
            if not torch.equal(
                rows.valid[first:last], cp_context.local_is_real[first:last]
            ):
                raise ValueError("V4.1 canonical row validity differs from CP mapping")
            start = cp_context.prefix_lengths_host[batch]
            end = start + cp_context.input_lengths_global_host[batch]
            if end > self._max_tokens or (start and not ready[batch]):
                raise ValueError("V4.1 CP execution requires complete restored state")
            tables = {
                slot: self._table(attn, batch, slot.owner_layer, _REGIONS[slot.region])
                for slot in self._raw_pages
            }
            pair_tables = {
                layer: self._table(
                    attn, batch, layer, KVCacheRegionName.DSV41_PAIR_STATE
                )
                for layer in self._pair_pools
            }
            writable = []
            current = (end - 1) // self.layout.reuse_unit
            for slot, pool in self._raw_pages.items():
                logical_pages = (current,)
                if slot.region != CacheRegion.SWA:
                    first_block = start // self.layout.token_block_size
                    last_block = (end - 1) // self.layout.token_block_size
                    logical_pages = (
                        block // self.layout.cp_size
                        for block in range(first_block, last_block + 1)
                        if block % self.layout.cp_size == self._cp_rank
                    )
                writable.extend(
                    (pool, self._page_id(tables[slot], logical, pool))
                    for logical in logical_pages
                )
            writable.extend(
                (pool, self._page_id(pair_tables[layer], current, pool))
                for layer, pool in self._pair_pools.items()
            )
            for pool, physical in writable:
                key = (pool.data_ptr(), physical)
                if key in occupied:
                    raise ValueError("V4.1 CP requests cannot share writable pages")
                occupied.add(key)
            context = begin_cp_request(
                cp_context,
                batch,
                request_id=str(identities[batch]),
                identity=self.identity,
                layout=self.layout,
                max_tokens=self._max_tokens,
                pools=self._raw_pages,
                tables=tables,
                pair_pools=self._pair_pools,
                pair_tables=pair_tables,
                epoch=self._forward_epoch,
                decoder_ready_end=start,
            )
            requests.append(_Request(first, last, context, {}, {}, batch))
            first = last
        return requests

    def _cp_native_call(self, context, action):
        from rtp_llm.models_py.distributed import collective_torch
        from rtp_llm.models_py.distributed.collective_torch import Group

        torch.cuda.current_stream().synchronize()
        collective_torch.barrier(Group.TP)
        error = None
        status = torch.ones(1, dtype=torch.int32, device=context.query_device)
        if self._cp_rank == 0:
            try:
                if action() is False:
                    raise RuntimeError("V4.1 native checkpoint copy did not complete")
            except Exception as exc:
                error = exc
                status.zero_()
        collective_torch.broadcast(status, 0, Group.TP)
        if int(status.item()) != 1:
            raise RuntimeError(
                "V4.1 native state update failed on the scheduling rank"
            ) from error

    def _cp_protect(
        self, native, context, history, *, publication_end=None, swa_ranges=None
    ):
        from rtp_llm.models_py.distributed import collective_torch
        from rtp_llm.models_py.distributed.collective_torch import Group

        group_count = len(self.kv_cache.group_region_names)
        sizes = torch.zeros(group_count, dtype=torch.int64, device=context.query_device)
        if self._cp_rank == 0:
            sizes.copy_(
                torch.tensor(
                    [len(ids) for ids in native.block_ids_by_group],
                    dtype=torch.int64,
                    device=context.query_device,
                )
            )
        collective_torch.broadcast(sizes, 0, Group.TP)
        groups = {}
        for slot, table in context.tables.items():
            groups[self._groups[(slot.owner_layer, int(_REGIONS[slot.region]))]] = (
                table[0]
            )
        for layer, table in context.pair_tables.items():
            groups[self._groups[(layer, int(KVCacheRegionName.DSV41_PAIR_STATE))]] = (
                table[0]
            )
        counts = sizes.cpu().tolist()
        if set(groups) != set(range(group_count)) or any(
            count > groups[gid].numel() for gid, count in enumerate(counts)
        ):
            raise ValueError(
                "V4.1 native checkpoint groups differ from actual CP page tables"
            )
        flat = torch.cat([groups[gid][:count] for gid, count in enumerate(counts)])
        gathered = (
            collective_torch.all_gather(flat, Group.TP).reshape(8, -1).cpu().tolist()
        )
        per_rank = []
        for values in gathered:
            grouped, first = [], 0
            for count in counts:
                grouped.append(values[first : first + count])
                first += count
            per_rank.append(grouped)
        publication = self._context_state(
            context,
            history.token_ids,
            history.image_mask,
            publication_end=publication_end,
            swa_ranges=swa_ranges,
        )
        self._cp_native_call(
            context,
            lambda: native.protect_checkpoint(publication, per_rank[0], per_rank),
        )
        return True

    def _protect_restored_cp(self, native, context, rows, swa_ranges):
        from rtp_llm.models_py.modules.dsv41.prefill import V41CPHistory

        packed = torch.cat(
            (rows.history_ids, (~rows.history_valid).to(torch.int32)), dim=1
        )
        values = (
            context.gather_rows(packed, context.start, context.start + 1)[0]
            .cpu()
            .tolist()
        )
        for index, position in enumerate(range(context.start - 3, context.start)):
            if position < 0:
                values[index], values[index + 3] = -1, 0
        history = V41CPHistory(
            tuple(values[:3]), tuple(bool(value) for value in values[3:])
        )
        self._cp_protect(
            native,
            context,
            history,
            publication_end=context.start,
            swa_ranges=swa_ranges,
        )

    def _forward_cp(self, inputs, rows, requests):
        from rtp_llm.models_py.distributed import collective_torch
        from rtp_llm.models_py.distributed.collective_torch import Group
        from rtp_llm.models_py.modules.dsv41.prefill import V41CPPrefillExecutor

        handles = {
            item.request_id: item
            for item in getattr(inputs, "v41_execution_contexts", ())
            if item is not None
        }
        images = self._images(inputs, rows)
        hidden = self.target.embedding.new_zeros(
            (rows.token_ids.numel(), self.config.hidden_size)
        )
        states = []
        self._prefill_observations = []
        for request in requests:
            context = request.context
            native = handles.get(int(context.cache.request_id))
            bounds = torch.tensor(
                [0, context.end, 0], dtype=torch.int64, device=context.query_device
            )
            if native is not None:
                bounds.copy_(
                    torch.tensor(
                        [native.protected_prefix_end, native.final_handoff_end, 1],
                        dtype=torch.int64,
                        device=context.query_device,
                    )
                )
            collective_torch.broadcast(bounds, 0, Group.TP)
            checkpoint, final, has_native = bounds.cpu().tolist()
            if final != context.end:
                raise ValueError(
                    "V4.1 engine must supply the complete prefill before sampling"
                )
            current = V41ModelRows(
                *(
                    getattr(rows, field.name)[request.first : request.last].contiguous()
                    for field in fields(V41ModelRows)
                )
            )
            current_images = None
            if images is not None:
                keep = (images.row_indices >= request.first) & (
                    images.row_indices < request.last
                )
                current_images = V41ImageFeatures(
                    images.row_indices[keep] - request.first,
                    images.token_types[keep],
                    images.values[keep],
                )
            restored_protected = (
                has_native and checkpoint == context.start and checkpoint > 0
            )
            if restored_protected:
                self._protect_restored_cp(
                    native, context, current, inputs.v41_swa_ranges[request.batch]
                )
            executor = V41CPPrefillExecutor(
                self.target,
                request_id=context.cache.request_id,
                identity=self.identity,
                layout=self.layout,
                initial_encoder_end=context.start,
                initial_decoder_end=context.start,
                initial_protected_end=checkpoint if restored_protected else 0,
                draft_commit=self.prefill_draft,
                max_tokens_per_rank=self.max_tokens_per_rank,
            )

            def report(progress):
                from rtp_llm.ops.compute_ops import V41ExecutionProgress

                if progress.encoder_materialized_end == progress.decoder_checkpoint_end:
                    return
                value = V41ExecutionProgress()
                value.request_id = int(context.cache.request_id)
                value.encoder_materialized_end = progress.encoder_materialized_end
                value.decoder_checkpoint_end = progress.decoder_checkpoint_end
                self._cp_native_call(context, lambda: native.report_progress(value))

            output = executor.run_extend(
                current,
                context,
                protected_checkpoint_end=checkpoint,
                final_handoff_end=final,
                protect_checkpoint=(
                    (
                        lambda completed, history: self._cp_protect(
                            native, completed, history
                        )
                    )
                    if has_native
                    else None
                ),
                report_progress=report if has_native else None,
                image_features=current_images,
            )
            hidden[request.first : request.last].copy_(output.hidden_states)
            states.append(
                self._context_state(
                    output.decoder_context,
                    output.history.token_ids,
                    output.history.image_mask,
                )
            )
            self._prefill_observations.append(
                {
                    "request_id": context.cache.request_id,
                    "segments": executor.observations,
                }
            )
        result = PyModelOutputs(hidden)
        result.v41_execution_states = states
        return result

    @staticmethod
    def _context_state(
        context,
        history_token_ids,
        history_image_mask,
        *,
        publication_end=None,
        swa_ranges=None,
    ):
        from rtp_llm.ops.compute_ops import V41ExecutionState

        if context is None:
            raise RuntimeError("V4.1 final prefill has no completed decoder context")
        cache = context.cache
        end = context.end if publication_end is None else publication_end
        layers = 43 if cache.layout.draft_enabled else 40
        if any(cache.swa_ends.get(layer) != end for layer in range(layers)):
            raise RuntimeError(
                "V4.1 publication requires all target/draft SWA at the actual boundary"
            )
        state = V41ExecutionState()
        state.request_id = int(cache.request_id)
        state.materialized_end = end
        state.encoder_materialized_end = end
        state.decoder_checkpoint_end = end
        state.draft_layers = layers - 40
        state.global_entries = [
            cache.owners[layer].materialized_end // layer_sources(layer).ratio
            for layer in GLOBAL_OWNERS
        ]
        state.index_entries = list(state.global_entries)
        if swa_ranges is None:
            state.swa_valid_start = [
                int(cache.swa[layer].valid_starts[0]) for layer in range(layers)
            ]
            state.swa_valid_end = [
                int(cache.swa[layer].valid_ends[0]) for layer in range(layers)
            ]
            state.swa_replay_floor = [
                0 if layer <= 20 else context.replay_floor for layer in range(layers)
            ]
        else:
            if swa_ranges.dtype != torch.int64 or swa_ranges.shape != (layers, 3):
                raise ValueError("V4.1 restored state requires every native SWA range")
            ranges = swa_ranges.cpu().tolist()
            if any(last != end for _, last, _ in ranges):
                raise ValueError(
                    "V4.1 restored SWA does not match its checkpoint boundary"
                )
            state.swa_valid_start = [first for first, _, _ in ranges]
            state.swa_valid_end = [last for _, last, _ in ranges]
            state.swa_replay_floor = [floor for _, _, floor in ranges]
        state.pair_positions = [
            (
                cache.owners[layer].pair.next_position - 1
                if cache.owners[layer].pair.next_position % 2
                else -1
            )
            for layer in PAIR_OWNERS
        ]
        state.pair_valid = [
            cache.owners[layer].pair.next_position % 2 for layer in PAIR_OWNERS
        ]
        state.history_token_ids = list(history_token_ids)
        state.history_image_mask = [int(value) for value in history_image_mask]
        state.history_ready = True
        state.draft_committed = cache.layout.draft_enabled
        if state.draft_committed:
            state.aux_valid_start = max(0, end - 128)
            state.aux_valid_end = end
        return state

    @staticmethod
    def _images(inputs, rows):
        features = inputs.multimodal_features
        if not features:
            return None
        locations = _host_vector(
            inputs.mm_features_locs, len(features), torch.int32, "image locations"
        )
        if any(
            value.ndim != 2
            or start < 0
            or start + value.shape[0] > rows.token_ids.numel()
            for start, value in zip(locations, features)
        ):
            raise ValueError("V4.1 image features exceed the canonical input rows")
        indices = torch.cat(
            [
                torch.arange(
                    start, start + value.shape[0], device=rows.token_ids.device
                )
                for start, value in zip(locations, features)
            ]
        )
        return V41ImageFeatures(indices, rows.token_types[indices], torch.cat(features))

    @staticmethod
    def _execution_states(requests, rows):
        from rtp_llm.ops.compute_ops import V41ExecutionState

        states = []
        for request in requests:
            context = request.context
            cache = context.cache
            end = context.end
            local = slice(request.first, request.last)
            history = torch.cat(
                (rows.history_ids[local, -2:], rows.token_ids[local, None]), dim=1
            )
            image_mask = torch.cat(
                (~rows.history_valid[local, -2:], rows.image_mask[local, None]), dim=1
            )
            tail = torch.cat((history, image_mask.to(torch.int32)), dim=1)
            if hasattr(context, "gather_rows"):
                tail = context.gather_rows(tail, end - 1, end)
            else:
                tail = tail[-1:]
            tail = tail.cpu().tolist()[0]
            for index, position in enumerate(range(end - 3, end)):
                if position < 0:
                    tail[index], tail[index + 3] = -1, 0
            state = V41ExecutionState()
            state.request_id = int(cache.request_id)
            state.materialized_end = end
            state.encoder_materialized_end = end
            state.decoder_checkpoint_end = end
            state.draft_layers = 0
            state.global_entries = [
                cache.owners[layer].materialized_end // layer_sources(layer).ratio
                for layer in GLOBAL_OWNERS
            ]
            state.index_entries = list(state.global_entries)
            state.swa_valid_start = [
                int(cache.swa[layer].valid_starts[0]) for layer in range(40)
            ]
            state.swa_valid_end = [
                int(cache.swa[layer].valid_ends[0]) for layer in range(40)
            ]
            state.swa_replay_floor = [0] * 40
            state.pair_positions = [
                (
                    cache.owners[layer].pair.next_position - 1
                    if cache.owners[layer].pair.next_position % 2
                    else -1
                )
                for layer in PAIR_OWNERS
            ]
            state.pair_valid = [
                cache.owners[layer].pair.next_position % 2 for layer in PAIR_OWNERS
            ]
            state.history_token_ids = tail[:3]
            state.history_image_mask = tail[3:]
            state.history_ready = True
            state.draft_committed = False
            states.append(state)
        return states

    @torch.inference_mode()
    def forward(self, inputs, fmha_impl=None):
        if self.kv_cache is None:
            # NormalExecutor's allocation warmup runs before it binds any pool.
            return PyModelOutputs(
                self.target.embedding.new_zeros(
                    (max(inputs.input_ids.numel(), 1), self.config.hidden_size)
                )
            )
        if fmha_impl is not None:
            fmha_impl.begin_forward()
            rows = fmha_impl.rows
            capture_scope = getattr(
                fmha_impl, "engram_capture_scope", lambda: nullcontext()
            )
            with capture_scope():
                target_args = dict(
                    execution_mode="full",
                    aux_row_indices=(
                        torch.arange(
                            rows.token_ids.numel(), device=rows.token_ids.device
                        )
                        if self._capture_aux
                        else None
                    ),
                    dense_aux=bool(self._capture_aux),
                )
                if getattr(fmha_impl, "_engram_graph", None) is not None:
                    target_args["lookup_outputs"] = fmha_impl.lookup_outputs
                output = self.target(rows, fmha_impl.context, **target_args)
            if self._capture_aux:
                # All graph buckets write the same handoff allocation; Python
                # attributes are not updated by replay of a previously captured bucket.
                self._mtp_aux_buffer[: rows.token_ids.numel()].copy_(
                    output.aux_hidden_states
                )
            fmha_impl.finish_forward()
            result = PyModelOutputs(output.hidden_states)
            if (
                not torch.cuda.is_current_stream_capturing()
                and self._active_v41_graph_impl is fmha_impl
            ):
                result.v41_execution_states = fmha_impl.get_execution_states(inputs)
            return result
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("V4.1 capture requires its prepared decode context")
        self._active_v41_graph_impl = None
        is_cp = inputs.attention_inputs.context_parallel_info is not None
        if not is_cp and inputs.input_ids.numel() > self.max_tokens_per_rank:
            raise ValueError(
                "V4.1 input exceeds the admitted per-rank execution budget"
            )
        if inputs.input_ids.numel():
            rows = V41ModelRows.from_model_inputs(inputs)
        else:
            device = self.target.embedding.device
            rows = V41ModelRows(
                torch.empty(0, dtype=torch.int32, device=device),
                torch.empty(0, dtype=torch.int32, device=device),
                torch.empty(0, dtype=torch.bool, device=device),
                torch.empty((0, 3), dtype=torch.int32, device=device),
                torch.empty((0, 3), dtype=torch.bool, device=device),
            )
        requests = self._requests(inputs, rows)
        if is_cp:
            result = self._forward_cp(inputs, rows, requests)
            self._write_cache_store(inputs)
            return result
        output = self.target(
            rows,
            requests,
            execution_mode="full",
            image_features=self._images(inputs, rows),
        )
        if (
            output.hidden_states.shape
            != (rows.token_ids.numel(), self.config.hidden_size)
            or output.hidden_states.dtype != torch.bfloat16
            or output.hidden_states.device != self.target.embedding.device
        ):
            raise RuntimeError(
                "V4.1 target must return one normalized BF16 hidden row per input"
            )
        for request in requests:
            if request.context.completed_layers != set(range(40)):
                raise RuntimeError("V4.1 target did not materialize every layer")
            for layer, destination in request.pair_outputs.items():
                _write_pair(destination, request.context.cache.owners[layer].pair)
        self._write_cache_store(inputs)
        result = PyModelOutputs(output.hidden_states)
        result.v41_execution_states = self._execution_states(requests, rows)
        return result

    def _write_cache_store(self, inputs):
        if getattr(inputs.attention_inputs, "cache_store_inputs", None) is not None:
            writer = create_write_cache_store_impl(
                inputs.attention_inputs, self.kv_cache
            )
            if writer is not None:
                # Pair snapshots must be complete before the async writer records
                # its CUDA event for any region of the handoff boundary.
                for layer in range(43 if self.layout.draft_enabled else 40):
                    writer(self.kv_cache.get_layer_caches(layer))
