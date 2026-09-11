"""Explicit eager target attention over the native V4.1 compact cache.

This component composes initialized projections, four owner compressors and
eight query scorers. Full local pages are required; distributed CP assembly and
Graph scheduling remain caller integration work. No legacy reader fallback is
selected. The local allocator is for component execution, not a CP deployment.
"""

import os
from dataclasses import dataclass, field, replace
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.models_py.modules.dsv41.cache_layout import (
    SWA_WINDOW,
    CacheIdentity,
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
    compact_attention,
)
from rtp_llm.models_py.modules.dsv41.compact_writer import write_compact
from rtp_llm.models_py.modules.dsv41.compressor import (
    CompressorRoPE,
    OwnerCompressor,
    OwnerPageBinding,
    PairCarry,
    prepare_owner_kv,
)
from rtp_llm.models_py.modules.dsv41.indexer import (
    QUERY_TILE,
    IndexSelection,
    select_index_positions,
)
from rtp_llm.models_py.modules.dsv41.math import grouped_wo_a, rms_norm


def is_supported(hidden: torch.Tensor) -> bool:
    return (
        hidden.is_cuda
        and hidden.dtype == torch.bfloat16
        and torch.cuda.get_device_capability(hidden.device)[0] == 10
    )


def attention_rope(values, positions, *, global_branch, inverse=False):
    """Rotate adjacent pairs in the final 64 channels without mutating the input."""
    if values.ndim not in (2, 3) or values.shape[0] != positions.numel():
        raise ValueError("attention RoPE requires one absolute position per row")
    if global_branch:
        frequencies = CompressorRoPE().frequencies(positions)
    else:
        dimensions = torch.arange(0, 64, 2, dtype=torch.float32, device=values.device)
        phase = torch.outer(positions.float(), 1.0 / (10000.0 ** (dimensions / 64)))
        frequencies = torch.polar(torch.ones_like(phase), phase)
    if inverse:
        frequencies = frequencies.conj()
    tail = torch.view_as_complex(values[..., -64:].float().unflatten(-1, (-1, 2)))
    if values.ndim == 3:
        frequencies = frequencies[:, None, :]
    result = values.clone()
    result[..., -64:] = torch.view_as_real(tail * frequencies).flatten(-2)
    return result


@dataclass
class AttentionOwnerCache:
    global_kv: GlobalBinding
    index_pages: CompactPages
    index_table: torch.Tensor
    materialized_end: int = 0
    pair: Optional[PairCarry] = None


@dataclass
class V41AttentionCache:
    request_id: str
    identity: CacheIdentity
    layout: CacheLayout
    max_tokens: int
    swa: dict[int, SwaBinding]
    owners: dict[int, AttentionOwnerCache]
    swa_ends: dict[int, int] = field(default_factory=dict)
    active_epoch: int = -1
    poisoned: bool = False

    @classmethod
    def allocate_local(cls, request_id, identity, layout, max_tokens, *, device):
        if not request_id or identity.layout_fingerprint != layout.fingerprint:
            raise ValueError("attention cache requires its actual request and layout")
        if type(max_tokens) is not int or not 0 < max_tokens <= 1048576:
            raise ValueError("attention capacity must fit the actual model context")
        swa, regions, tables = {}, {}, {}
        for page in layout.pages:
            count = (
                1
                if page.slot.region == CacheRegion.SWA
                else (max_tokens + layout.token_block_size - 1)
                // layout.token_block_size
            )
            storage = torch.zeros(
                (count + 1, page.page_stride_bytes), dtype=torch.uint8, device=device
            )
            pages = CompactPages(storage, page.slot.region, page.entries)
            table = torch.arange(1, count + 1, dtype=torch.int32, device=device)[
                None, :
            ]
            regions[page.slot] = pages
            tables[page.slot] = table
            if page.slot.region == CacheRegion.SWA:
                swa[page.slot.owner_layer] = SwaBinding(
                    pages,
                    table[0],
                    torch.zeros_like(table[0]),
                    torch.zeros_like(table[0]),
                )
        owners = {}
        for owner in (2, 8, 14, 20):
            g = RegionSlot(CacheRegion.GLOBAL, owner)
            i = RegionSlot(CacheRegion.INDEX_K, owner)
            owners[owner] = AttentionOwnerCache(
                GlobalBinding(regions[g], tables[g], layer_sources(owner).ratio),
                regions[i],
                tables[i],
            )
        return cls(request_id, identity, layout, max_tokens, swa, owners)

    def begin_forward(self, *, epoch, start, end, replay_floor=0):
        if self.poisoned:
            raise RuntimeError("failed attention cache must be discarded or restored")
        if type(epoch) is not int or epoch <= self.active_epoch:
            raise ValueError("attention forward epochs must strictly increase")
        if not 0 <= replay_floor <= start <= end <= self.max_tokens:
            raise ValueError("invalid attention query range or replay floor")
        if self.identity.replay_fingerprint not in (
            ReplayConfig(ReplayMode.FULL).fingerprint,
            ReplayConfig(ReplayMode.BOUNDED).fingerprint,
        ):
            raise ValueError("attention cache has an unsupported replay policy")
        if (
            replay_floor
            and self.identity.replay_fingerprint
            != ReplayConfig(ReplayMode.BOUNDED).fingerprint
        ):
            raise ValueError("full attention cannot truncate SWA history")
        self.active_epoch = epoch
        return V41AttentionContext(self, epoch, start, end, replay_floor)


@dataclass
class V41AttentionContext:
    cache: V41AttentionCache
    epoch: int
    start: int
    end: int
    replay_floor: int
    selections: dict[int, IndexSelection] = field(default_factory=dict)
    published_sources: set[int] = field(default_factory=set)
    completed_layers: set[int] = field(default_factory=set)
    observations: list[dict] = field(default_factory=list)
    planar_globals: dict = field(default_factory=dict)
    _identity: tuple = field(init=False, repr=False)

    def __post_init__(self):
        self._identity = (
            self.cache.request_id,
            self.cache.identity.fingerprint,
            self.epoch,
            self.start,
            self.end,
            self.replay_floor,
        )

    @property
    def query_identity(self):
        return self._identity

    def validate(self):
        if (
            self.cache.poisoned
            or self.epoch != self.cache.active_epoch
            or self._identity
            != (
                self.cache.request_id,
                self.cache.identity.fingerprint,
                self.epoch,
                self.start,
                self.end,
                self.replay_floor,
            )
        ):
            raise ValueError(
                "attention context has stale or failed request/epoch state"
            )

    def tail(self, start, *, replay_floor):
        """Own selected L20 query results before the encoder batch is released."""
        self.validate()
        if not self.start <= start <= self.end or not 0 <= replay_floor <= start:
            raise ValueError(
                "late attention rows must be retained by their source batch"
            )
        if (
            self.cache.identity.replay_fingerprint
            != ReplayConfig(ReplayMode.BOUNDED).fingerprint
        ):
            raise ValueError("tail selection requires bounded cache identity")
        if 20 not in self.selections or 20 not in self.published_sources:
            raise ValueError(
                "late attention requires complete L20 source and query state"
            )
        source = self.selection_for(20)
        offset = start - self.start
        selection = IndexSelection(
            source.topk[offset:].clone(),
            source.candidate_blocks[offset:].clone(),
            source.status[offset:].clone(),
            20,
            20,
            0,
            0,
            0,
        )
        result = V41AttentionContext(
            self.cache, self.epoch, start, self.end, replay_floor
        )
        result.published_sources = set(self.published_sources)
        result.publish_selection(selection)
        return result

    def publish_selection(self, selected):
        self.validate()
        source = layer_sources(selected.query_owner)
        if (
            not source.scores_queries
            or selected.key_owner != source.index_k_owner
            or source.global_owner not in self.published_sources
        ):
            raise ValueError(
                "index selection does not belong to a ready query/key owner"
            )
        if selected.query_owner in self.selections:
            raise ValueError("index query owner already published this forward")
        if selected.query_identity not in (None, self.query_identity):
            raise ValueError("index selection has stale request/epoch/query identity")
        self._validate_selection(selected)
        self.selections[selected.query_owner] = replace(
            selected, query_identity=self.query_identity
        )

    def _validate_selection(self, selected):
        rows = self.end - self.start
        device = self.cache.swa[selected.query_owner].pages.data.device
        tensors = [(selected.topk, (rows, 512)), (selected.status, (rows,))]
        if selected.query_owner == 20:
            if selected.candidate_blocks is None:
                raise ValueError("L20 selection requires its candidate block set")
            tensors.append((selected.candidate_blocks, (rows, 2048)))
        elif selected.candidate_blocks is not None:
            raise ValueError("only L20 owns the reindex candidate block set")
        if any(
            tensor.shape != shape
            or tensor.dtype != torch.int32
            or tensor.device != device
            or not tensor.is_contiguous()
            for tensor, shape in tensors
        ):
            raise ValueError("index selection tensors do not match the query range")

    def selection_for(self, layer):
        self.validate()
        source = layer_sources(layer)
        if source.global_owner not in self.published_sources:
            raise ValueError("attention owner has not published this forward's source")
        if source.topk_owner not in self.selections:
            raise ValueError("attention query owner has not scored this query range")
        selected = self.selections[source.topk_owner]
        if (
            selected.query_owner != source.topk_owner
            or selected.key_owner != source.index_k_owner
            or selected.query_identity != self.query_identity
        ):
            raise ValueError(
                "index selection has stale request/epoch/query/owner identity"
            )
        self._validate_selection(selected)
        return selected

    def indices_for(self, layer):
        return self.selection_for(layer).topk


class V41Attention(nn.Module):
    @classmethod
    def from_weights(cls, layer, weights, *, layout=None):
        """Bind the loader's checkpoint-local tensors to the block32 path."""
        from rtp_llm.models_py.modules.dsv41.linear import V41Block32Linear

        source = layer_sources(layer)

        def dense(name):
            return V41Block32Linear(weights[name + ".weight"], weights[name + ".scale"])

        owner = source.writes_global
        compressor = None
        if owner:
            compressor = OwnerCompressor(
                layer,
                weights["attn.compressor.wkv.weight"],
                weights["attn.compressor.norm.weight"],
                weights["attn.compressor.wgate.weight"] if source.ratio == 2 else None,
                layout=layout,
            )
        return cls(
            layer,
            wq_a=dense("attn.wq_a"),
            wq_b=dense("attn.wq_b"),
            wkv=dense("attn.wkv"),
            wo_b=dense("attn.wo_b"),
            wo_a=weights["attn.wo_a.weight"].view(8, 1024, 4096),
            q_norm=weights["attn.q_norm.weight"],
            kv_norm=weights["attn.kv_norm.weight"],
            sinks=weights["attn.attn_sink"],
            compressor=compressor,
            index_wq_b=dense("attn.indexer.wq_b") if source.scores_queries else None,
            index_weights=(
                weights["attn.indexer.weights_proj.weight"]
                if source.scores_queries
                else None
            ),
            index_wk=weights["attn.indexer.wk.weight"] if owner else None,
            index_norm=weights["attn.indexer.k_norm.weight"] if owner else None,
        )

    def __init__(
        self,
        layer,
        *,
        wq_a,
        wq_b,
        wkv,
        wo_b,
        wo_a,
        q_norm,
        kv_norm,
        sinks,
        compressor=None,
        index_wq_b=None,
        index_weights=None,
        index_wk=None,
        index_norm=None,
    ):
        super().__init__()
        self.layer = layer
        source = layer_sources(layer)
        if layer >= 40:
            raise ValueError(
                "target attention does not implement draft commit semantics"
            )
        self.source = source
        self.wq_a, self.wq_b, self.wkv, self.wo_b = wq_a, wq_b, wkv, wo_b
        for name, value, shape, dtype in (
            ("wo_a", wo_a, (8, 1024, 4096), torch.bfloat16),
            ("q_norm", q_norm, (1280,), torch.bfloat16),
            ("kv_norm", kv_norm, (512,), torch.bfloat16),
            ("sinks", sinks, (64,), torch.float32),
        ):
            if value.shape != shape or value.dtype != dtype:
                raise ValueError(f"invalid V4.1 {name} shape or dtype")
            self.register_buffer(name, value)
        if source.writes_global != isinstance(compressor, OwnerCompressor):
            raise ValueError("only the four KV owners require a compressor")
        if compressor is not None and compressor.owner_layer != layer:
            raise ValueError("compressor belongs to a different owner layer")
        if source.scores_queries != (
            index_wq_b is not None and index_weights is not None
        ):
            raise ValueError("only the eight query owners require index projections")
        if source.writes_index_k != (index_wk is not None and index_norm is not None):
            raise ValueError("only the four KV owners require index K weights")
        self.compressor = compressor
        self.index_wq_b = index_wq_b
        self.register_buffer("index_weights", index_weights)
        self.register_buffer("index_wk", index_wk)
        self.register_buffer("index_norm", index_norm)

    def _project(self, hidden, positions):
        qr = rms_norm(self.wq_a(hidden), self.q_norm)
        query = self.wq_b(qr).reshape(-1, 64, 512)
        kv = rms_norm(self.wkv(hidden), self.kv_norm)
        branch = self.source.ratio != 0
        return (
            qr,
            attention_rope(query, positions, global_branch=branch),
            attention_rope(kv, positions, global_branch=branch),
        )

    def _publish_owner(self, hidden, context):
        state = context.cache.owners[self.layer]
        if state.materialized_end != context.start:
            raise ValueError(
                "owner KV must extend its real contiguous materialized prefix"
            )
        source = self.compressor(
            hidden,
            start_pos=context.start,
            request_id=context.cache.request_id,
            identity=context.cache.identity,
            pair=state.pair,
        )
        rows = prepare_owner_kv(source, self.index_wk, self.index_norm)
        logical = source.group_positions // self.source.ratio
        global_slots, index_slots = [], []
        for pages, table, slots in (
            (state.global_kv.pages, state.global_kv.page_table, global_slots),
            (state.index_pages, state.index_table, index_slots),
        ):
            page_ids = table[0].index_select(
                0, (logical // pages.entries_per_page).long()
            )
            slots.append(
                (
                    page_ids * pages.entries_per_page + logical % pages.entries_per_page
                ).contiguous()
            )
        results = rows.store(
            OwnerPageBinding(self.layer, context.cache.identity, state.global_kv.pages),
            OwnerPageBinding(self.layer, context.cache.identity, state.index_pages),
            global_slots[0],
            index_slots[0],
        )
        for result in results:
            result.check()
        state.pair = source.next_pair
        state.materialized_end = context.end
        context.published_sources.add(self.layer)

    def _score(self, hidden, qr, positions, context):
        if self.source.global_owner not in context.published_sources:
            raise ValueError("index source is not ready for the current forward")
        state = context.cache.owners[self.source.index_k_owner]
        if state.materialized_end < context.end:
            raise ValueError("index K does not cover the query range")
        query = attention_rope(
            self.index_wq_b(qr).reshape(-1, 32, 128), positions, global_branch=True
        )
        weights = (
            F.linear(hidden, self.index_weights) * (128**-0.5 * 32**-0.5)
        ).contiguous()
        candidates = None
        if self.layer > 20:
            if 20 not in context.selections:
                raise ValueError(
                    "reindex requires L20 candidates for the same query range"
                )
            candidates = context.selection_for(20).candidate_blocks
        results = []
        for first in range(0, hidden.shape[0], QUERY_TILE):
            last = min(first + QUERY_TILE, hidden.shape[0])
            result = select_index_positions(
                query[first:last].contiguous(),
                weights[first:last],
                state.index_pages,
                state.index_table,
                torch.zeros(last - first, dtype=torch.int32, device=hidden.device),
                ((positions[first:last] + 1) // self.source.ratio).to(torch.int32),
                layer=self.layer,
                max_visible_length=max(1, context.end // self.source.ratio),
                candidate_blocks=(
                    None if candidates is None else candidates[first:last].contiguous()
                ),
            )
            result.check()
            results.append(result)
        if not results:
            topk = torch.empty((0, 512), dtype=torch.int32, device=hidden.device)
            blocks = (
                torch.empty((0, 2048), dtype=torch.int32, device=hidden.device)
                if self.layer == 20
                else None
            )
            status = torch.empty(0, dtype=torch.int32, device=hidden.device)
        else:
            topk = torch.cat([result.topk for result in results])
            blocks = (
                torch.cat([result.candidate_blocks for result in results])
                if self.layer == 20
                else None
            )
            status = torch.cat([result.status for result in results])
        context.publish_selection(
            IndexSelection(
                topk,
                blocks,
                status,
                self.layer,
                self.source.index_k_owner,
                sum(result.scorer_calls for result in results),
                max((result.max_logits_elements for result in results), default=0),
                max((result.max_packed_kv_bytes for result in results), default=0),
            )
        )

    @torch.inference_mode()
    def forward(self, hidden, context: V41AttentionContext):
        context.validate()
        if os.environ.get("DSV41_ATTENTION") != "1" or not is_supported(hidden):
            raise RuntimeError(
                "V4.1 attention requires opt-in Blackwell BF16 execution"
            )
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "eager attention composition cannot be captured; use reader Graph probes"
            )
        if (
            hidden.shape != (context.end - context.start, 5120)
            or not hidden.is_contiguous()
        ):
            raise ValueError(
                "attention hidden rows do not match the explicit query range"
            )
        if self.layer in context.completed_layers:
            raise ValueError("attention layer already executed in this forward")
        if (
            context.cache.identity.layout_fingerprint
            != context.cache.layout.fingerprint
        ):
            raise ValueError("attention cache layout identity changed")
        if (
            self.compressor is not None
            and self.compressor.layout != context.cache.layout
        ):
            raise ValueError("attention owner and cache layout disagree")
        swa = context.cache.swa[self.layer]
        previous = context.cache.swa_ends.get(self.layer, 0)
        history_start = int(swa.valid_starts[0].item())
        if previous != context.start:
            if not (
                self.layer > 20
                and context.replay_floor == context.start
                and context.cache.identity.replay_fingerprint
                == ReplayConfig(ReplayMode.BOUNDED).fingerprint
            ):
                raise ValueError(
                    "attention continuation is missing its complete SWA history"
                )
            history_start = context.start
        positions = torch.arange(
            context.start, context.end, dtype=torch.int64, device=hidden.device
        )
        backend = os.environ.get("DSV41_ATTENTION_BACKEND", "native")
        if backend not in ("native", "flashmla"):
            raise ValueError("unknown V4.1 attention backend; no silent fallback")
        try:
            qr, query, kv = self._project(hidden, positions)
            if self.source.writes_global:
                self._publish_owner(hidden, context)
            if self.source.scores_queries:
                self._score(hidden, qr, positions, context)
            indices = context.indices_for(self.layer) if self.source.ratio else None
            global_kv = (
                context.cache.owners[self.source.global_owner].global_kv
                if self.source.ratio
                else None
            )
            planar_swa = planar_global = None
            if backend == "flashmla":
                from rtp_llm.models_py.modules.dsv41.flashmla import (
                    PlanarGlobalBinding,
                    PlanarSwaBinding,
                    flashmla_attention,
                    to_planar,
                )

                planar_swa = PlanarSwaBinding.from_compact(swa)
                if global_kv is not None:
                    owner = self.source.global_owner
                    if owner not in context.planar_globals:
                        context.planar_globals[owner] = (
                            PlanarGlobalBinding.from_compact(global_kv)
                        )
                    planar_global = context.planar_globals[owner]
            outputs = []
            # Earlier queries must read the old ring before later writes wrap it.
            tile_rows = min(QUERY_TILE, swa.pages.entries_per_page - SWA_WINDOW + 1)
            for first in range(0, hidden.shape[0], tile_rows):
                last = min(first + tile_rows, hidden.shape[0])
                pos = positions[first:last]
                slots = (
                    swa.page_ids[0] * swa.pages.entries_per_page
                    + pos % swa.pages.entries_per_page
                ).contiguous()
                write_compact(kv[first:last].contiguous(), swa.pages, slots).check()
                valid_end = context.start + last
                swa.valid_starts.fill_(
                    max(
                        history_start,
                        context.replay_floor,
                        valid_end - swa.pages.entries_per_page,
                        0,
                    )
                )
                swa.valid_ends.fill_(valid_end)
                reader = compact_attention
                reader_swa, reader_global = swa, global_kv
                if backend == "flashmla":
                    to_planar(swa.pages, out=planar_swa.pages)
                    reader, reader_swa, reader_global = (
                        flashmla_attention,
                        planar_swa,
                        planar_global,
                    )
                result = reader(
                    query[first:last].contiguous(),
                    torch.zeros_like(pos, dtype=torch.int32),
                    pos.to(torch.int32),
                    torch.full_like(pos, context.replay_floor, dtype=torch.int32),
                    reader_swa,
                    self.sinks,
                    global_kv=reader_global,
                    global_indices=(
                        None if indices is None else indices[first:last].contiguous()
                    ),
                )
                result.check()
                outputs.append(result.output)
            output = torch.cat(outputs) if outputs else query.new_empty((0, 64, 512))
            output = attention_rope(
                output, positions, global_branch=bool(self.source.ratio), inverse=True
            )
            output = self.wo_b(
                grouped_wo_a(output.reshape(-1, 8, 4096), self.wo_a).flatten(1)
            )
            if output.shape != hidden.shape or output.dtype != torch.bfloat16:
                raise ValueError(
                    "attention output projection changed the target geometry or dtype"
                )
            context.cache.swa_ends[self.layer] = context.end
            context.completed_layers.add(self.layer)
            context.observations.append(
                {
                    "layer": self.layer,
                    "reader_backend": backend,
                    "query_identity": context.query_identity,
                    "query_rows": hidden.shape[0],
                    "source_rows": hidden.shape[0] if self.source.writes_global else 0,
                    "index_rows": hidden.shape[0] if self.source.scores_queries else 0,
                    "swa_write_tile": tile_rows,
                }
            )
            return output
        except Exception:
            context.cache.poisoned = True
            raise
