"""Framework-owned, complete-page V4.1 target execution.

The standard engine owns requests, allocation, copying and sampling. This
adapter binds its physical pages and returns normalized hidden states. CP
prefill communication, PD publication, Graph and DSpark remain separate work.
"""

from dataclasses import dataclass

import torch
from rtp_llm.models_py.model_desc.module_base import GptModelBase
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


class _BatchedAttention(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention

    def forward(self, hidden, requests):
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
        if cp.is_enabled():
            raise NotImplementedError(
                "V4.1 P execution requires CP communication integration"
            )
        if getattr(config, "capture_aux_hidden_layer_ids", None):
            raise NotImplementedError(
                "V4.1 target DSpark capture is not wired to the engine"
            )
        if config.dsv41_replay_mode != "full" or config.dsv41_tail_policy_version != 1:
            raise NotImplementedError("V4.1 standard adapter requires full execution")
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
            speculative_tokens=0,
            draft_enabled=False,
        )
        self.identity = ReplayConfig(ReplayMode.FULL).cache_identity(
            shared_lookup.shared.manifest["revision"], self.layout
        )
        self._pages, self._pair_pools = {}, {}
        self._groups = {}
        self._max_tokens = config.max_seq_len
        self._shared_lookup = shared_lookup
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
        )
        warmup_block32_linears(self.target, max_rows=max_tokens_per_rank)

    def initialize(self, init_resource):
        if init_resource.is_speculative:
            raise NotImplementedError(
                "V4.1 standard adapter currently executes target only"
            )
        super().initialize(init_resource)
        self._pages.clear()
        self._pair_pools.clear()
        self._groups.clear()
        if self.kv_cache is None:
            return True
        cache = self.kv_cache
        if (
            cache.seq_size_per_block != self.layout.token_block_size
            or cache.kernel_seq_size_per_block != self.layout.token_block_size
            or len(cache.layer_region_to_group_id) != 40
        ):
            raise ValueError(
                "V4.1 framework cache differs from target block/layer geometry"
            )
        device = self.target.embedding.device
        for page in self.layout.pages:
            layer, region = page.slot.owner_layer, _REGIONS[page.slot.region]
            pool = cache.get_raw_pool_tensor(layer, region)
            pages = CompactPages(pool, page.slot.region, page.entries)
            pages.validate(device)
            if pool.shape[1] != page.page_stride_bytes:
                raise NotImplementedError(
                    "V4.1 eager adapter requires complete physical pages"
                )
            self._pages[page.slot] = pages
            self._groups[(layer, int(region))] = self._group(layer, region)
        for layer in PAIR_OWNERS:
            region = KVCacheRegionName.DSV41_PAIR_STATE
            pool = cache.get_raw_pool_tensor(layer, region)
            expected = ((2 * _PAIR_BYTES + 511) // 512) * 512
            if (
                pool.dtype != torch.uint8
                or pool.device != device
                or pool.ndim != 2
                or pool.shape[1] != expected
                or pool.stride(1) != 1
                or pool.stride(0) < expected
                or pool.stride(0) % 512
            ):
                raise ValueError(
                    "V4.1 target requires complete two-snapshot pair pages"
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
        if is_cuda_graph:
            raise NotImplementedError(
                "V4.1 eager target is not a CUDA Graph implementation"
            )
        return None

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

    @torch.inference_mode()
    def forward(self, inputs, fmha_impl=None):
        if self.kv_cache is None:
            raise RuntimeError(
                "V4.1 target requires cache-backed engine warmup and execution"
            )
        if torch.cuda.is_current_stream_capturing():
            raise NotImplementedError(
                "V4.1 eager target cannot run inside CUDA Graph capture"
            )
        if inputs.input_ids.numel() > self.max_tokens_per_rank:
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
        return PyModelOutputs(output.hidden_states)
