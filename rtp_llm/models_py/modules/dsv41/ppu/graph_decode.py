"""Device-driven full-model V4.1 PPU decode for eager and Graph execution.

Captured Python expands the fixed layer loop once. Request state, positions,
cache addresses and padding validity are tensor inputs refreshed by the
framework; replay needs no Python request histories or cache reconstruction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv41.backend import V41StepContext
from rtp_llm.models_py.modules.dsv41.ppu._quant import prepare_e2m1_codebooks
from rtp_llm.models_py.modules.dsv41.ppu.engram import engram_gate_inject, wkv_project
from rtp_llm.models_py.modules.dsv41.ppu.graph_attention import (
    decode_attention,
    indexer_topk,
)
from rtp_llm.models_py.modules.dsv41.ppu.moe import (
    combine_routed_and_shared,
    pad_topk_for_dispatch,
    shared_expert_forward,
    sqrtsoftplus_gate,
)
from rtp_llm.models_py.modules.dsv41.ppu.prefill import (
    _compress_pairs,
    _rms_norm,
    _rope_tail,
)


class GraphDecodeRunner:
    """One model's graph-safe decode core, bound after its weights are loaded."""

    def __init__(self, core, *, max_batch_size, dp_rank=0, dp_size=1):
        self.core = core
        self.ops, self.bundle = core.ops, core.weights
        self.geometry = core.geometry
        self.max_batch_size = int(max_batch_size)
        self.dp_rank, self.dp_size = int(dp_rank), int(dp_size)
        self._embedding_pointer = self.bundle.embedding.data_ptr()
        prepare_e2m1_codebooks(core.device)
        self.engram = None
        if self.geometry.engram_enabled():
            from rtp_llm.models_py.modules.dsv41.ppu.graph_engram import GraphEngram

            self.engram = GraphEngram(
                self.ops,
                self.geometry,
                device=core.device,
                max_batch_size=self.max_batch_size,
            )

    def validate(self):
        """Graph-external lifetime checks; these never read device values."""
        if self.bundle.embedding.data_ptr() != self._embedding_pointer:
            raise RuntimeError("PPU Graph embedding allocation changed; recapture is required")
        if self.engram is not None:
            self.engram.validate()

    @torch.inference_mode()
    def forward(self, inputs, impl):
        geometry = self.geometry
        metadata = impl.metadata
        cache = impl.cache
        # This hook records tensor metadata updates at the beginning of capture
        # and prepares the identical tensor values on graph-safe eager runs.
        impl.prepare_in_graph(inputs.attention_inputs)
        positions, valid = metadata.positions, metadata.valid_mask
        batch = positions.shape[0]
        if inputs.input_ids.numel() != batch:
            raise ValueError("PPU decode Graph requires one token per captured row")
        ids = inputs.input_ids.reshape(batch).long()
        ids = torch.where(valid, ids, torch.zeros_like(ids))
        hidden = self.bundle.embedding[ids]
        h = hidden[:, None].expand(
            batch, geometry.hc_mult, geometry.hidden_size
        ).contiguous()
        requests = torch.arange(batch, device=positions.device, dtype=torch.int64)
        context = V41StepContext(
            positions=positions,
            valid_mask=valid,
            request_ids=requests,
            owner_ranks=torch.full_like(requests, self.dp_rank),
            request_offsets=torch.arange(
                batch + 1, device=positions.device, dtype=torch.int64
            ),
            parallel_context=self,
        )
        hashes = None
        if self.engram is not None:
            windows = inputs.engram_token_windows
            if windows is None or tuple(windows.shape) != (
                batch, geometry.engram_max_ngram_size
            ):
                raise ValueError("PPU decode Graph requires scheduler Engram windows")
            hashes = self.engram.hash(windows)
        shared_topk = {}
        previous_mix = None
        for layer_id in range(geometry.num_layers):
            h, previous_mix = self._forward_layer(
                layer_id,
                h,
                cache,
                context,
                hashes,
                shared_topk,
                previous_mix,
            )
        head = self.ops.mhc(h, None, phase="head", state=previous_mix)
        output = _rms_norm(
            head.hidden.reshape(batch, geometry.hidden_size),
            self.bundle.final_norm,
            geometry.norm_eps,
        )
        return torch.where(valid[:, None], output, torch.zeros_like(output))

    def _produce_global(self, layer, x, cache, positions, valid, freqs):
        geometry = self.geometry
        layer_id = layer.layer_id
        ratio = geometry.ratio(layer_id)
        values = layer.compressor_wkv(x)
        if ratio == 1:
            latent = _rms_norm(
                values, layer.compressor_norm, geometry.norm_eps
            ).to(torch.bfloat16)
            entry_positions = positions
            commit = valid
        elif ratio == 2:
            scores = layer.compressor_wgate(x)
            # Only an odd-position token closes the even|odd pair. Reading a
            # masked predecessor for other rows preserves static tensor shape.
            commit = valid & ((positions & 1) == 1)
            previous_slots = cache.slots(
                layer_id, "csa_state", torch.where(commit, positions - 1, -1)
            )
            previous = cache.read_rows(layer_id, "csa_state", previous_slots)
            latent = _compress_pairs(
                torch.stack((previous[:, :512].to(values.dtype), values), dim=1),
                torch.stack((previous[:, 512:].to(scores.dtype), scores), dim=1),
                layer.compressor_norm,
                geometry.norm_eps,
            )
            entry_positions = positions - 1
            # Commit the raw compressor projection, including an unfinished
            # even-position half-pair, to the framework-owned fp32 CSA ring.
            state_slots = cache.slots(
                layer_id, "csa_state", torch.where(valid, positions, -1)
            )
            cache.write_rows(
                layer_id,
                "csa_state",
                state_slots,
                torch.cat((values, scores), dim=-1).float(),
            )
        else:
            raise ValueError("PPU Graph producer must use compression ratio 1 or 2")
        safe_positions = entry_positions.clamp(min=0, max=freqs.shape[0] - 1)
        global_keys = _rope_tail(latent, freqs[safe_positions])
        tag = "global_kv_2" if ratio == 2 else "global_kv_1"
        raw = torch.where(commit, entry_positions, -1)
        slots = cache.slots(layer_id, tag, raw)
        packed = self.ops.encode_kv(global_keys, pool_tag=tag)
        cache.write_rows(layer_id, tag, slots, packed)

        index = F.linear(latent, layer.index_wk.weight)
        index = _rms_norm(index, layer.index_k_norm, geometry.norm_eps)
        index = _rope_tail(index, freqs[safe_positions])
        packed = self.ops.encode_kv(index, pool_tag="indexer_kv")
        cache.write_rows(
            layer_id,
            "indexer_kv",
            cache.slots(layer_id, "indexer_kv", raw),
            packed,
        )

    def _indexer(self, layer, x, qr, cache, positions, valid, freqs):
        geometry = self.geometry
        source = geometry.kv_source_for(layer.layer_id)
        if source is None:
            raise ValueError("V4.1 index-source layer requires a KV source")
        query = layer.index_wq(qr).reshape(
            x.shape[0], geometry.index_n_heads, geometry.index_head_dim
        )
        query = _rope_tail(query, freqs[positions.clamp_max(freqs.shape[0] - 1)])
        weights = F.linear(x, layer.index_weights_proj).float()
        weights = weights * (geometry.index_head_dim * geometry.index_n_heads) ** -0.5
        return indexer_topk(
            query,
            weights,
            cache=cache,
            source_layer=source,
            ratio=geometry.ratio(source),
            positions=positions,
            valid_mask=valid,
            topk=geometry.index_topk,
            max_seq_len=min(geometry.max_seq_len, cache.max_seq_len),
        )

    def _inject_engram(self, layer_id, layer, h, hashes, valid):
        hash_index = self.geometry.engram_layer_ids.index(layer_id)
        rows = hashes[:, hash_index]
        embedded = self.engram.lookup(rows, layer_id=layer_id, valid_mask=valid)
        projected = wkv_project(
            embedded.reshape(embedded.shape[0], -1), layer.engram_wkv.weight
        )
        return engram_gate_inject(
            h,
            layer.engram_q_weight,
            layer.engram_k_weight,
            projected,
            token_mask=valid & (rows[:, 0] >= 0),
        )

    def _forward_layer(
        self, layer_id, h, cache, context, hashes, shared_topk, previous_mix
    ):
        geometry = self.geometry
        layer = self.bundle.layers[layer_id]
        positions, valid = context.positions, context.valid_mask
        batch = positions.shape[0]
        freqs = self.bundle.freqs_for(layer_id)
        rope = freqs[positions.clamp(min=0, max=freqs.shape[0] - 1)]
        if layer_id in geometry.engram_layer_ids:
            h = self._inject_engram(layer_id, layer, h, hashes, valid)
        residual = h
        attention_mix = self.ops.mhc(
            residual,
            None,
            phase="pre",
            parameters=layer.mhc_parameters("attn"),
            context=context,
            state=previous_mix,
        )
        x = _rms_norm(
            attention_mix.hidden.reshape(batch, geometry.hidden_size),
            layer.attn_norm,
            geometry.norm_eps,
        )
        qr = _rms_norm(layer.wq_a(x), layer.q_norm, geometry.norm_eps)
        q = layer.wq_b(qr).reshape(batch, geometry.n_heads, geometry.head_dim)
        q = _rope_tail(q, rope)
        kv = _rope_tail(
            _rms_norm(layer.wkv(x), layer.kv_norm, geometry.norm_eps), rope
        )
        slots = cache.slots(
            layer_id, "swa_kv", torch.where(valid, positions, -1)
        )
        cache.write_rows(
            layer_id, "swa_kv", slots, self.ops.encode_kv(kv, pool_tag="swa_kv")
        )
        if layer.is_kv_source:
            self._produce_global(layer, x, cache, positions, valid, freqs)
        if layer.is_index_source:
            selected = self._indexer(layer, x, qr, cache, positions, valid, freqs)
            shared_topk[layer_id] = selected
        elif geometry.ratio(layer_id):
            source = geometry.index_source_for(layer_id)
            if source not in shared_topk:
                raise ValueError("V4.1 decode layer is missing shared index selection")
            selected = shared_topk[source]
        else:
            selected = None
        attention = decode_attention(
            q,
            selected,
            cache=cache,
            layer_id=layer_id,
            source_layer=geometry.kv_source_for(layer_id),
            ratio=geometry.ratio(layer_id),
            positions=positions,
            valid_mask=valid,
            window=geometry.window,
            sm_scale=geometry.sm_scale(),
            attn_sink=layer.attn_sink,
        )
        attention = _rope_tail(attention, rope, inverse=True)
        grouped = attention.reshape(batch, geometry.o_groups, -1)
        rank = geometry.o_lora_rank
        projected = torch.stack(
            [
                F.linear(
                    grouped[:, group].contiguous(),
                    layer.wo_a.weight[group * rank : (group + 1) * rank].contiguous(),
                )
                for group in range(geometry.o_groups)
            ],
            dim=1,
        )
        branch = layer.wo_b(projected.reshape(batch, -1))
        h = self.ops.mhc(
            residual,
            branch,
            phase="post",
            parameters=layer.mhc_parameters("attn"),
            context=context,
            state=attention_mix,
        ).hidden
        residual = h
        ffn_mix = self.ops.mhc(
            residual,
            None,
            phase="pre",
            parameters=layer.mhc_parameters("ffn"),
            context=context,
            state=attention_mix,
        )
        x = _rms_norm(
            ffn_mix.hidden.reshape(batch, geometry.hidden_size),
            layer.ffn_norm,
            geometry.norm_eps,
        )
        ids, weights = sqrtsoftplus_gate(
            x,
            layer.router_weight,
            layer.router_bias,
            top_k=geometry.n_activated_experts,
            routed_scaling_factor=geometry.routed_scaling_factor,
        )
        ids, weights = pad_topk_for_dispatch(ids, weights)
        ids = torch.where(valid[:, None], ids, -1)
        weights = torch.where(valid[:, None], weights, torch.zeros_like(weights))
        x = x.to(torch.bfloat16)
        routed = self.ops.moe(x, ids, weights, layer_id=layer_id, context=context)
        if layer.shared_w13 is not None:
            shared = shared_expert_forward(
                x, layer.shared_w13, layer.shared_w2, geometry.swiglu_limit
            )
            branch = combine_routed_and_shared(routed, shared)
        else:
            branch = routed
        h = self.ops.mhc(
            residual,
            branch,
            phase="post",
            parameters=layer.mhc_parameters("ffn"),
            context=context,
            state=ffn_mix,
        ).hidden
        return h, ffn_mix
