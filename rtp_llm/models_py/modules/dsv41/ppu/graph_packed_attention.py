"""Ordinary Decode Graph attention over the native packed KV pools."""

from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv41.ppu.packed_attention import (
    load_packed_backend,
    packed_pool_view,
)
from rtp_llm.models_py.modules.dsv41.ppu.packed_decode import packed_decode

GRAPH_PACKED_ATTENTION_ENV = "RTP_V41_PPU_GRAPH_PACKED_ATTENTION"
GRAPH_VISIBLE_INDEXER_ENV = "RTP_V41_PPU_GRAPH_VISIBLE_INDEXER"


class GraphPackedAttention:
    """Bind the provider and zero-copy pool views before bucket capture."""

    def __init__(self, cache, geometry):
        backend, _ = load_packed_backend()
        self.native = backend.flash_mla_with_kvcache
        self.metadata_factory = backend.get_mla_metadata
        self.cache = cache
        g = geometry
        if g.window != 128 or g.head_dim != 512 or not 0 < g.n_heads <= 128:
            raise ValueError("unsupported PPU packed Decode Graph geometry")
        self.offsets = torch.arange(g.window, dtype=torch.int64, device=cache.device)
        self.views = {}
        self.swa_domains = {}
        widths = {"swa_kv": 528, "global_kv_1": 288, "global_kv_2": 288}
        for owner, tag in cache.geometry.pool_keys():
            if tag not in widths:
                continue
            tensor = cache.pool(owner, tag)
            domain = cache._table_tag(owner, tag)
            entries = cache.entries[domain]
            self.views[owner, tag] = packed_pool_view(
                SimpleNamespace(tensor=tensor, entries=entries), widths[tag]
            )
            if tag == "swa_kv":
                self.swa_domains[owner] = (domain, tensor.shape[0])

    def _swa_indices(self, layer_id, positions, valid, shared):
        key = ("packed_swa", self.swa_domains[layer_id])
        if key not in shared:
            width = self.offsets.numel()
            end = positions + 1
            lengths = torch.where(valid, end.clamp_max(width), 0).int()
            raw = (end - width).clamp_min(0)[:, None] + self.offsets[None]
            raw = torch.where(self.offsets[None] < lengths[:, None], raw, -1)
            slots = self.cache.slots(layer_id, "swa_kv", raw).int()
            shared[key] = (slots[:, None], lengths)
        return shared[key]

    def _global_indices(self, source, index_source, tag, ratio, selected, positions, valid, shared):
        key = ("packed_global", source, index_source)
        if key not in shared:
            visible = (positions + 1) // ratio
            lengths = torch.where(valid, visible.clamp_max(selected.shape[-1]), 0).int()
            live = valid[:, None] & (selected >= 0) & (selected < visible[:, None])
            raw = torch.where(live, selected.long() * ratio, -1)
            slots = self.cache.slots(source, tag, raw).int()
            shared[key] = (slots[:, None], lengths)
        return shared[key]

    def forward(self, attn, q, selected, metadata, shared):
        positions, valid = metadata.positions, metadata.valid_mask
        swa_indices, swa_lengths = self._swa_indices(attn.layer_id, positions, valid, shared)
        global_view = global_indices = global_lengths = None
        if selected is not None:
            source, ratio = attn.kv_source_layer_id, attn.compress_ratio
            tag = "global_kv_2" if ratio == 2 else "global_kv_1"
            global_view = self.views[source, tag]
            global_indices, global_lengths = self._global_indices(
                source, attn.index_source_layer_id, tag, ratio, selected,
                positions, valid, shared,
            )
        # The provider masks -1 slots inside each live prefix, including
        # unmapped warmup pages. Both prefixes keep their original key order.
        # Fresh scheduler objects capture the device scheduler for every layer.
        plan = SimpleNamespace(
            native=self.native,
            metadata_factory=self.metadata_factory,
            swa_cache=self.views[attn.layer_id, "swa_kv"],
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
            global_cache=global_view,
            global_indices=global_indices,
            global_lengths=global_lengths,
            valid_mask=valid,
        )
        return packed_decode(q, plan, softmax_scale=attn.geometry.sm_scale(), attn_sink=attn.attn_sink)
