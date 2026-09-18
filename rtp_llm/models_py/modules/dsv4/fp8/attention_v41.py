"""DeepSeek V4.1 shared global attention over RTP's typed CP/PD pools.

The initial eager implementation deliberately shares the established SWA and
FlashMLA kernels with V4. Global compression is non-overlapping (ratio 1/2),
and index keys are projected from its normalized, pre-RoPE latent. Only source
layers write global pools; consumers reuse source KV and index selections.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4.attn_type import (
    CSA_KV,
    CSA_STATE,
    HCA_KV,
    INDEXER_KV,
    SWA_KV,
)
from rtp_llm.models_py.modules.dsv4.cp import cp_all_gather_full_varlen
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_indexer as prefill_indexer
from rtp_llm.models_py.modules.dsv4.fp8._cp_slot_mapping import (
    cp_kv_slot_mapping,
    cp_state_slot_mapping,
)
from rtp_llm.models_py.modules.dsv4.fp8._kv_cache_utils import (
    require_pool_tokens_per_block,
)
from rtp_llm.models_py.modules.dsv4.fp8.attention import (
    _ATTN_TYPE_ENUM_BY_INT,
    AttentionFP8,
)
from rtp_llm.models_py.modules.dsv4.rope import apply_rotary_emb, precompute_freqs_cis
from rtp_llm.models_py.modules.dsv4.utils import _v4_fp8_linear


def rms_norm(x, weight, eps):
    """Preserve the BF16 normalization boundary used by V4.1."""
    xf = x.float()
    return (
        xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * weight.float()
    ).to(x.dtype)


def compress_pairs(values, scores, norm_weight, eps):
    """Pool consecutive non-overlapping pairs independently for each channel."""
    weights = torch.softmax(scores.float(), dim=-2)
    latent = (values.float() * weights).sum(-2)
    return rms_norm(latent, norm_weight, eps).to(torch.bfloat16)


def rope_only(x, freqs, rope_dim):
    apply_rotary_emb(x[..., -rope_dim:].unsqueeze(0), freqs)
    return x


def fp8_roundtrip(x):
    scale = (x.float().abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-12)
    return (x.float() / scale).to(torch.float8_e4m3fn).float() * scale


def select_candidate_blocks(logits, visible, block_size, topk_blocks):
    """V4.1 candidate max-pooling, including the newest visible block."""
    nblocks = (logits.shape[1] + block_size - 1) // block_size
    padded = F.pad(
        logits, (0, nblocks * block_size - logits.shape[1]), value=-torch.inf
    )
    scores = padded.view(logits.shape[0], nblocks, block_size).amax(-1)
    newest = ((visible - 1).clamp_min(0) // block_size).long()
    scores.scatter_(
        1, newest[:, None], torch.where(visible > 0, torch.inf, -torch.inf)[:, None]
    )
    values, ids = scores.topk(min(topk_blocks, nblocks), dim=-1)
    return torch.where(values > -torch.inf, ids, -1).int()


def mask_candidate_logits(logits, candidates, block_size):
    nblocks = (logits.shape[1] + block_size - 1) // block_size
    allowed = torch.zeros(
        logits.shape[0], nblocks + 1, dtype=torch.bool, device=logits.device
    )
    ids = torch.where(candidates >= 0, candidates, nblocks).long()
    allowed.scatter_(1, ids, True)
    columns = torch.arange(logits.shape[1], device=logits.device) // block_size
    return logits.masked_fill_(~allowed[:, columns], -torch.inf)


class AttentionV41FP8(AttentionFP8):
    def __init__(self, *args, v41_config, shared_attention, **kwargs):
        ratio = int(kwargs["compress_ratio"])
        if ratio not in (0, 1, 2):
            raise ValueError(f"V4.1 compression ratio must be 0, 1 or 2, got {ratio}")
        # The inherited constructor owns projections and SWA, not the V4
        # compressor/indexer (their checkpoint topology differs from V4.1).
        kwargs["compress_ratio"] = 0
        super().__init__(*args, **kwargs)
        self.compress_ratio = ratio
        self.skip_post_q_norm = True
        self.v41_config = v41_config
        # Store a plain dict: registering source modules would create cycles.
        self._shared_attention = shared_attention
        shared_attention.setdefault("layers", {})[self.layer_id] = self
        self.index_topk = int(kwargs["index_topk"])
        self.index_n_heads = int(kwargs["index_n_heads"])
        self.index_head_dim = int(kwargs["index_head_dim"])
        self.is_kv_source = self.layer_id in v41_config.get("kv_source_layer_ids", [])
        self.is_index_source = self.layer_id in v41_config.get(
            "index_source_layer_ids", []
        )
        self.kv_source_layer_id = None
        self.index_source_layer_id = None
        if ratio:
            self.kv_source_layer_id = max(
                i for i in v41_config["kv_source_layer_ids"] if i <= self.layer_id
            )
            self.index_source_layer_id = max(
                i for i in v41_config["index_source_layer_ids"] if i <= self.layer_id
            )
            self._rope_base = kwargs["compress_rope_theta"]
            self._rope_o_seq_len = kwargs["original_seq_len"]
            self.freqs_cis = precompute_freqs_cis(
                self._rope_dim,
                self._rope_max_seq_len,
                self._rope_o_seq_len,
                self._rope_base,
                self._rope_factor,
                self._rope_beta_fast,
                self._rope_beta_slow,
            )
        from rtp_llm.utils.model_weight import W

        w = kwargs["layer_weights"]
        if self.is_kv_source:
            self.global_wkv = w[W.v4_compressor_wkv]
            self.global_norm = w[W.v4_compressor_norm]
            self.global_wgate = w[W.v4_compressor_wgate] if ratio == 2 else None
            self.index_wk = w[W.v41_indexer_wk]
            self.index_k_norm = w[W.v41_indexer_k_norm]
        if self.is_index_source:
            self.index_wq = _v4_fp8_linear(
                w[W.v4_indexer_wq_b_w], w[W.v4_indexer_wq_b_s]
            )
            self.index_weights = w[W.v4_indexer_weights_proj_w]
        self._pool_spec[CSA_STATE] = (torch.float32, 2 * self.head_dim)
        self._wo_a_groups_v41 = torch.nn.ModuleList()
        scale_block = self.wo_a_w.shape[1] // self.wo_a_s.shape[1]
        for group in range(self.n_groups):
            begin, end = group * self.o_lora_rank, (group + 1) * self.o_lora_rank
            self._wo_a_groups_v41.append(
                _v4_fp8_linear(
                    self.wo_a_w[begin:end].contiguous(),
                    self.wo_a_s[begin // scale_block : end // scale_block].contiguous(),
                )
            )

    def _project_output(self, o, freqs, out=None):
        from rtp_llm.models_py.modules.dsv4.fp8 import _v41_output_projection

        o = o.reshape(-1, self.n_heads, self.head_dim)
        if _v41_output_projection.is_supported(
            o, freqs, self._wo_a_stk_w, self._wo_a_stk_s
        ):
            projected = _v41_output_projection.grouped_output_projection(
                o, freqs, self._wo_a_stk_w, self._wo_a_stk_s
            )
            return self.wo_b(projected, out=out)
        apply_rotary_emb(
            o[..., -self.rope_head_dim :].unsqueeze(0), freqs, inverse=True
        )
        grouped = o.reshape(o.shape[0], self.n_groups, -1)
        projected = torch.stack(
            [
                linear(grouped[:, g].contiguous())
                for g, linear in enumerate(self._wo_a_groups_v41)
            ],
            dim=1,
        )
        return self.wo_b(projected.flatten(1), out=out)

    def _prefill_output_proj_into(self, o, freqs_cis, *, out):
        self._project_output(o, freqs_cis, out=out)

    def _ensure_freqs_cis_bound(self):
        # V4.1 has no independent compressor module or nested index compressor.
        return

    def _build_shared_prefill_meta(self, *args, **kwargs):
        # Every V4.1 layer needs the full SWA prefix metadata, including global
        # layers. Build through the mature SWA planner with this layer's RoPE.
        ratio = self.compress_ratio
        self.compress_ratio = 0
        kwargs["reuse_common_meta"] = None
        try:
            return super()._build_shared_prefill_meta(*args, **kwargs)
        finally:
            self.compress_ratio = ratio

    def _materialize_prefill_q(self, qkv, common):
        if qkv.q is not None:
            return qkv
        rows = qkv.qr.shape[0]
        q_out = common.workspace.prefill_q(rows).view(
            rows, self.n_heads * self.head_dim
        )
        q = self._lin(self.wq_b, qkv.qr, out=q_out).view(
            -1, self.n_heads, self.head_dim
        )
        return qkv._replace(q=rope_only(q, common.freqs_cis, self.rope_head_dim))

    def _begin_forward(self):
        if self.layer_id == min(self._shared_attention["layers"]):
            self._shared_attention["global"] = {}
            self._shared_attention["topk"] = {}
            self._shared_attention["candidates"] = None

    def _owner(self):
        return self._shared_attention["layers"][self.kv_source_layer_id]

    def _global_region(self):
        return CSA_KV if self.compress_ratio == 2 else HCA_KV

    def _source_pool(self, region):
        if self._kv_cache is None:
            return None
        layer_id = self.layer_id if region == SWA_KV else self.kv_source_layer_id
        base = self._kv_cache.get_layer_cache(
            layer_id, _ATTN_TYPE_ENUM_BY_INT[region]
        ).kv_cache_base
        if base is None or base.numel() == 0:
            return None
        dtype, width = self._pool_spec[region]
        stride_bytes = base.shape[1] * base.element_size()
        eb = stride_bytes // (width * dtype.itemsize)
        raw = base.view(torch.uint8)
        if dtype == torch.uint8:
            return raw.as_strided((base.shape[0], eb, width), (stride_bytes, width, 1))
        return raw.view(dtype).view(-1, width)

    def _source_entries(self, region, pool):
        if region == CSA_STATE:
            raw = self._kv_cache.get_layer_cache(
                self.kv_source_layer_id, _ATTN_TYPE_ENUM_BY_INT[region]
            ).kv_cache_base
            return raw.shape[1] * raw.element_size() // (2 * self.head_dim * 4)
        return pool.shape[1]

    def _slots(self, region, positions, req_ids, *, state_end=None):
        pool = self._source_pool(region)
        if pool is None or self._block_tables_by_type is None:
            return None
        bt = self._block_tables_by_type[region]
        eb = self._source_entries(region, pool)
        tpb = require_pool_tokens_per_block(self._kv_cache, region=region)
        cp = self._cp_ctx
        sharded = cp is not None and cp.cp_size > 1 and cp.kv_cache_sharded
        if region == CSA_STATE:
            return cp_state_slot_mapping(
                positions,
                bt,
                req_ids,
                eb,
                tpb,
                cp.cp_size if sharded else 1,
                cp.cp_rank if sharded else 0,
                seq_end_per_req=state_end,
            )
        ratio = self.compress_ratio
        if sharded:
            return cp_kv_slot_mapping(
                positions,
                bt,
                req_ids,
                tpb,
                eb,
                ratio,
                cp.cp_size,
                cp.cp_rank,
                owner_tokens_per_block=self._kv_cache.seq_size_per_block,
            )
        col = positions // tpb
        valid = (col >= 0) & (col < bt.shape[1]) & ((positions + 1) % ratio == 0)
        block = bt[req_ids, col.clamp(0, bt.shape[1] - 1)].long()
        return torch.where(
            valid & (block > 0), block * eb + positions.remainder(tpb) // ratio, -1
        )

    def _gather_shards(self, value):
        if value.numel() == 0:
            return value
        cp = self._cp_ctx
        if cp is not None and cp.cp_size > 1 and cp.kv_cache_sharded:
            from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce

            all_reduce(value, Group.TP, inplace=True)
        return value

    def _read_state(self, positions, req_ids):
        slots = self._slots(CSA_STATE, positions, req_ids)
        if slots is None:
            return torch.zeros(
                len(positions), 2 * self.head_dim, device=positions.device
            )
        pool = self._source_pool(CSA_STATE)
        state = pool[slots.clamp_min(0)].clone()
        state.masked_fill_((slots < 0).unsqueeze(-1), 0)
        return self._gather_shards(state)

    def _write_states(self, values, scores, positions, req_ids, seq_ends):
        slots = self._slots(CSA_STATE, positions, req_ids, state_end=seq_ends)
        if slots is not None:
            valid = slots >= 0
            self._source_pool(CSA_STATE)[slots[valid]] = torch.cat(
                (values, scores), -1
            ).float()[valid]

    def _produce_global(self, x_full, positions, req_ids, starts, lengths, *, prefill):
        """Publish this owner's main/index pools, and materialize source keys."""
        owner = self._owner()
        ratio = self.compress_ratio
        from rtp_llm.models_py.modules.dsv4.fp8.compressor import _linear_bf16_bf16_fp32

        values = _linear_bf16_bf16_fp32(x_full, owner.global_wkv)
        if ratio == 2:
            scores = _linear_bf16_bf16_fp32(x_full, owner.global_wgate)
            # Read the previous token before tail writes can reuse its ring slot.
            previous = self._read_state(
                (starts - 1).clamp_min(0),
                torch.arange(len(starts), device=x_full.device),
            )
            value_prev = torch.cat((values[:1], values[:-1]), 0)
            score_prev = torch.cat((scores[:1], scores[:-1]), 0)
            first = positions == starts[req_ids]
            value_prev[first] = previous[req_ids[first], : self.head_dim].to(
                values.dtype
            )
            score_prev[first] = previous[req_ids[first], self.head_dim :].to(
                scores.dtype
            )
            boundary = (positions + 1).remainder(2) == 0
            latent = compress_pairs(
                torch.stack((value_prev[boundary], values[boundary]), 1),
                torch.stack((score_prev[boundary], scores[boundary]), 1),
                owner.global_norm,
                self.eps,
            )
            self._write_states(values, scores, positions, req_ids, starts + lengths)
        else:
            boundary = torch.ones_like(positions, dtype=torch.bool)
            latent = rms_norm(values, owner.global_norm, self.eps).to(torch.bfloat16)
        boundary_pos, boundary_req = positions[boundary], req_ids[boundary]
        freqs = self.freqs_cis[(boundary_pos // ratio) * ratio]
        global_keys = rope_only(latent.clone(), freqs, self.rope_head_dim)
        index_keys = rope_only(
            rms_norm(F.linear(latent, owner.index_wk), owner.index_k_norm, self.eps),
            freqs,
            self.rope_head_dim,
        )
        main_pool = self._source_pool(self._global_region())
        index_pool = self._source_pool(INDEXER_KV)
        if main_pool is not None:
            from rtp_llm.models_py.modules.dsv4.fp8._indexer_quant_triton import (
                quantize_indexer_k,
            )
            from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
                quantize_and_insert_k_cache,
            )

            quantize_and_insert_k_cache(
                global_keys.contiguous(),
                main_pool,
                self._slots(self._global_region(), boundary_pos, boundary_req),
            )
            quantize_indexer_k(
                index_keys.contiguous(),
                self._slots(INDEXER_KV, boundary_pos, boundary_req),
                index_pool,
            )
        result = []
        fused_indexer = prefill and prefill_indexer.is_supported(
            x_full.device, getattr(self, "index_n_heads", 0), index_keys.shape[-1]
        )
        for b, end in enumerate((starts + lengths).tolist()):
            count = int(end) // ratio
            idx = torch.arange(count, device=x_full.device, dtype=torch.long)
            pos = (idx + 1) * ratio - 1
            req = torch.full_like(pos, b)
            if main_pool is None:
                mask = boundary_req == b
                g = global_keys[mask]
                k = (
                    prefill_indexer.PrefillIndexerKeys(
                        *prefill_indexer.quantize_indexer_k_reference(index_keys[mask])
                    )
                    if fused_indexer
                    else fp8_roundtrip(index_keys[mask])
                )
            else:
                if fused_indexer:
                    k = prefill_indexer.gather_indexer_keys(
                        index_pool,
                        lambda p, r: self._slots(INDEXER_KV, p, r),
                        count,
                        b,
                        ratio,
                        self._cp_ctx,
                        self._kv_cache.seq_size_per_block,
                    )
                else:
                    from rtp_llm.models_py.modules.dsv4.fp8._indexer_quant_triton import (
                        dequantize_indexer_k,
                    )

                    k = dequantize_indexer_k(
                        index_pool, self._slots(INDEXER_KV, pos, req)
                    )
                    self._gather_shards(k)
                if prefill:
                    from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
                        dequantize_slots_to_bf16,
                    )

                    g = dequantize_slots_to_bf16(
                        main_pool, self._slots(self._global_region(), pos, req)
                    )
                    self._gather_shards(g)
                else:
                    g = None
            result.append((g, k))
        self._shared_attention["global"] = {self.layer_id: result}

    def _select_indices(self, x, qr, positions, req_ids):
        shared = self._shared_attention
        if not self.is_index_source:
            return shared["topk"][self.index_source_layer_id]
        globals_by_req = shared["global"][self.kv_source_layer_id]
        out = torch.full(
            (x.shape[0], self.index_topk), -1, dtype=torch.int32, device=x.device
        )
        config = getattr(self, "v41_config", {})
        candidate_source = int(config.get("candidate_source_layer_id", -1))
        candidate_blocks = int(config.get("candidate_topk_blocks", 0))
        candidate_size = int(config.get("candidate_block_size", 0))
        publish_candidates = (
            self.layer_id == candidate_source
            and candidate_blocks > 0
            and candidate_size > 0
        )
        if publish_candidates:
            max_blocks = max(
                (
                    (len(k) + candidate_size - 1) // candidate_size
                    for _, k in globals_by_req
                ),
                default=0,
            )
            shared["candidates"] = (
                torch.full(
                    (x.shape[0], candidate_blocks),
                    -1,
                    dtype=torch.int32,
                    device=x.device,
                )
                if max_blocks > candidate_blocks
                else None
            )
        # Short prompts select all causal candidates, as in the vLLM backend.
        if max((len(k) for _, k in globals_by_req), default=0) <= self.index_topk:
            ids = torch.arange(self.index_topk, device=x.device)
            out.copy_(
                torch.where(
                    ids[None] < ((positions + 1) // self.compress_ratio)[:, None],
                    ids[None],
                    -1,
                )
            )
        else:
            q = self._lin(self.index_wq, qr).view(
                -1, self.index_n_heads, self.index_head_dim
            )
            q = rope_only(q, self.freqs_cis[positions], self.rope_head_dim)
            weights = (
                F.linear(x, self.index_weights).float()
                * (self.index_head_dim * self.index_n_heads) ** -0.5
            )
            fused_indexer = isinstance(
                globals_by_req[0][1], prefill_indexer.PrefillIndexerKeys
            )
            if fused_indexer:
                q_fp8, weights_folded = prefill_indexer.quantize_indexer_q(q, weights)
            else:
                q = fp8_roundtrip(q)
            for b, (_, keys) in enumerate(globals_by_req):
                rows = torch.where(req_ids == b)[0]
                key_count = len(keys)
                chunk_rows = (
                    prefill_indexer.logits_chunk_rows(key_count)
                    if fused_indexer
                    else 64
                )
                for chunk in rows.split(chunk_rows):
                    visible = (positions[chunk] + 1) // self.compress_ratio
                    if fused_indexer:
                        from rtp_llm.models_py.modules.dsv4._profiler import (
                            record_function_range,
                        )

                        with record_function_range(
                            "dsv41.prefill.indexer.fused_logits"
                        ):
                            logits = prefill_indexer.score_indexer_chunk(
                                q_fp8.view(torch.uint8)[chunk].view(
                                    torch.float8_e4m3fn
                                ),
                                weights_folded[chunk],
                                keys.quant,
                                keys.scale,
                                visible,
                            )
                    else:
                        logits = torch.einsum(
                            "thd,kd->thk", q[chunk], keys.float()
                        ).relu_()
                        logits = (logits * weights[chunk, :, None]).sum(1)
                        ids = torch.arange(key_count, device=x.device)
                        logits.masked_fill_(ids[None] >= visible[:, None], -torch.inf)
                    candidates = shared.get("candidates")
                    if candidates is not None:
                        if publish_candidates:
                            block_ids = select_candidate_blocks(
                                logits,
                                visible,
                                candidate_size,
                                candidate_blocks,
                            )
                            candidates[chunk, : block_ids.shape[1]] = block_ids
                        elif candidate_source >= 0 and self.layer_id > candidate_source:
                            mask_candidate_logits(
                                logits, candidates[chunk], candidate_size
                            )
                    k = min(self.index_topk, key_count)
                    scores, selected = logits.topk(k, dim=-1)
                    out[chunk, :k] = torch.where(scores.isfinite(), selected, -1).int()
        shared["topk"] = {self.layer_id: out}
        return out

    def _swa_prefill_workspace(self, qkv, common):
        """Return BF16 [request, prefix tail + new tokens] for sparse prefill."""
        from rtp_llm.models_py.modules.dsv4.fp8 import _swa_dequant_triton as dq

        meta = common.swa_meta
        if not common.any_cont:
            lengths = (
                common.cp_ctx.input_lengths_global
                if common.cp_on
                else common.input_lengths
            ).tolist()
            return list(qkv.kv_full.split(lengths)), [0] * len(lengths)
        B, D = common.batch_size, self.head_dim
        buf = torch.zeros(B, meta.M, D, dtype=torch.bfloat16, device=qkv.kv_full.device)
        if meta.prefix_len_max > 0:
            if self._swa_cp_byte_sliced():
                dq.dequantize_and_gather_k_cache_slots_cp_byte_sliced(
                    out=buf,
                    k_cache_raw=self._pool_raw_u8(SWA_KV),
                    slot_mapping=meta.cache_slot_mapping,
                    gather_lens=meta.cache_gather_lens,
                    offset=0,
                    full_entries_per_block=self._swa_entries_per_block(),
                    cp_size=common.cp_ctx.cp_size,
                    cp_rank=common.cp_ctx.cp_rank,
                    compaction=meta.cache_compaction,
                )
            else:
                dq.dequantize_and_gather_k_cache_slots(
                    out=buf,
                    k_cache=self._pool_view_3d_fp8(SWA_KV),
                    slot_mapping=meta.cache_slot_mapping,
                    gather_lens=meta.cache_gather_lens,
                    offset=0,
                )
        buf.view(-1, D).index_copy_(0, meta.slot_in_flat, qkv.kv_full)
        prefixes = common.prefix_lengths.tolist()
        lengths = (
            common.cp_ctx.input_lengths_global if common.cp_on else common.input_lengths
        ).tolist()
        tails = [min(int(p), self.window_size - 1) for p in prefixes]
        return [buf[b, : tails[b] + int(lengths[b])] for b in range(B)], [
            int(p) - t for p, t in zip(prefixes, tails)
        ]

    def _forward_prefill(self, x, positions, shared_input_quant=None):
        self._begin_forward()
        common = self._prefill_common_setup(x, positions)
        qkv = self._prefill_compute_qkv(
            x, common, shared_input_quant=shared_input_quant
        )
        # Read old SWA tails before writes wrap over them in long prefill chunks.
        swa, swa_starts = (
            self._swa_prefill_workspace(qkv, common)
            if self.compress_ratio
            else (None, None)
        )
        self._prefill_write_swa_fp8_paged(common, qkv.kv_full)
        if not self.compress_ratio:
            return self._forward_prefill_swa_only(qkv, common)
        positions = (
            common.cp_ctx.global_positions.long()
            if common.cp_on
            else common.position_ids.long()
        )
        req_ids = common.req_id_per_token.long()
        starts = common.prefix_lengths.long()
        lengths = (
            common.cp_ctx.input_lengths_global if common.cp_on else common.input_lengths
        ).long()
        if self.is_kv_source:
            x_full = cp_all_gather_full_varlen(x, common.cp_ctx) if common.cp_on else x
            ids_full = torch.repeat_interleave(
                torch.arange(common.batch_size, device=x.device), lengths
            )
            cu = torch.cat(
                (torch.zeros(1, device=x.device, dtype=torch.long), lengths.cumsum(0))
            )
            pos_full = (
                torch.arange(x_full.shape[0], device=x.device)
                - cu[ids_full]
                + starts[ids_full]
            )
            self._produce_global(
                x_full, pos_full, ids_full, starts, lengths, prefill=True
            )
        selected = self._select_indices(x, qkv.qr, positions, req_ids)
        globals_by_req = self._shared_attention["global"][self.kv_source_layer_id]
        chunks, offsets, global_sizes = [], [], []
        offset = 0
        for (g, _), sw in zip(globals_by_req, swa):
            offsets.append(offset)
            global_sizes.append(g.shape[0])
            chunks.extend((g, sw))
            offset += g.shape[0] + sw.shape[0]
        offsets = torch.tensor(offsets, device=x.device)[req_ids, None]
        ns = torch.tensor(global_sizes, device=x.device)[req_ids, None]
        swstart = torch.tensor(swa_starts, device=x.device)[req_ids, None]
        swpos = (
            positions[:, None]
            - self.window_size
            + 1
            + torch.arange(self.window_size, device=x.device)[None]
        )
        swidx = torch.where(swpos >= swstart, offsets + ns + swpos - swstart, -1)
        global_idx = torch.where(selected >= 0, offsets + selected, -1)
        indices = torch.cat((global_idx, swidx), -1).int()
        # FlashMLA's length bounds count a compact valid prefix.
        indices = indices.gather(1, torch.argsort(indices < 0, dim=-1, stable=True))
        lens = (indices >= 0).sum(-1).int()
        if indices.shape[1] % 64:
            indices = F.pad(indices, (0, 64 - indices.shape[1] % 64), value=-1)
        qkv = self._materialize_prefill_q(qkv, common)
        return self._flash_mla_sparse_fwd_chunked_projected(
            q=qkv.q,
            kv=torch.cat(chunks).unsqueeze(1),
            indices=indices.unsqueeze(1),
            topk_length=lens,
            freqs_cis=common.freqs_cis,
            profile_name="dsv41.prefill.shared_global",
        )

    def _produce_global_decode(self, x, positions, req_ids, starts):
        """Fixed-shape global writes and index reads, safe for CUDA graphs."""
        from rtp_llm.models_py.modules.dsv4.fp8._indexer_quant_triton import (
            dequantize_indexer_k,
            quantize_indexer_k,
        )
        from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
            quantize_and_insert_k_cache,
        )
        from rtp_llm.models_py.modules.dsv4.fp8.compressor import _linear_bf16_bf16_fp32

        B, S, _ = x.shape
        from rtp_llm.models_py.modules.dsv4.fp8._v41_decode_global import (
            try_produce_global,
        )

        if not try_produce_global(self, x, positions, req_ids, starts):
            flat = x.reshape(B * S, -1)
            values = _linear_bf16_bf16_fp32(flat, self.global_wkv)
            if self.compress_ratio == 2:
                scores = _linear_bf16_bf16_fp32(flat, self.global_wgate)
                previous = self._read_state(
                    (starts - 1).clamp_min(0), torch.arange(B, device=x.device)
                )
                first = (torch.arange(B * S, device=x.device) % S == 0)[:, None]
                value_prev = torch.where(
                    first, previous[req_ids, : self.head_dim], values.roll(1, 0)
                )
                score_prev = torch.where(
                    first, previous[req_ids, self.head_dim :], scores.roll(1, 0)
                )
                latent = compress_pairs(
                    torch.stack((value_prev, values), 1),
                    torch.stack((score_prev, scores), 1),
                    self.global_norm,
                    self.eps,
                )
                slots = self._slots(CSA_STATE, positions, req_ids, state_end=starts + S)
                state_rows = torch.cat((values, scores), -1)
                # Slot zero is the allocator's unallocated sentinel; invalid rows
                # may overwrite it, but no live request ever reads that slot.
                self._source_pool(CSA_STATE).index_copy_(
                    0,
                    slots.clamp_min(0),
                    torch.where((slots >= 0)[:, None], state_rows, 0.0),
                )
            else:
                latent = rms_norm(values, self.global_norm, self.eps).to(torch.bfloat16)
            freqs = self.freqs_cis[
                (positions // self.compress_ratio) * self.compress_ratio
            ]
            global_keys = rope_only(latent.clone(), freqs, self.rope_head_dim)
            index_keys = rope_only(
                rms_norm(F.linear(latent, self.index_wk), self.index_k_norm, self.eps),
                freqs,
                self.rope_head_dim,
            )
            quantize_and_insert_k_cache(
                global_keys.contiguous(),
                self._source_pool(self._global_region()),
                self._slots(self._global_region(), positions, req_ids),
            )
            index_pool = self._source_pool(INDEXER_KV)
            quantize_indexer_k(
                index_keys.contiguous(),
                self._slots(INDEXER_KV, positions, req_ids),
                index_pool,
            )
        index_pool = self._source_pool(INDEXER_KV)
        capacity = self._rope_max_seq_len // self.compress_ratio
        from rtp_llm.models_py.modules.dsv4.fp8 import (
            _v41_decode_indexer as decode_indexer,
        )

        cp = self._cp_ctx
        sharded = cp is not None and cp.cp_size > 1 and cp.kv_cache_sharded
        table = self._block_tables_by_type[INDEXER_KV][:B]
        raw_tpb = require_pool_tokens_per_block(self._kv_cache, region=INDEXER_KV)
        logical_entries = raw_tpb // self.compress_ratio
        if (
            not sharded
            and raw_tpb % self.compress_ratio == 0
            and index_pool.is_contiguous()
            and logical_entries in (index_pool.shape[1], index_pool.shape[1] // 2)
            and capacity <= table.shape[1] * logical_entries
            and decode_indexer.is_supported(
                x.device, index_pool.shape[1], self.index_n_heads, self.index_head_dim
            )
        ):
            # Keep the owner's original FP8 bytes and paged layout. In V4.1's
            # uniform INDEXER_KV pool, ratio 2 only fills half of each page.
            self._shared_attention["global"] = {
                self.layer_id: decode_indexer.DecodeIndexerKeys(
                    index_pool, table, capacity, logical_entries
                )
            }
            return
        ids = torch.arange(capacity, device=x.device, dtype=torch.long).expand(B, -1)
        read_positions = (ids + 1) * self.compress_ratio - 1
        read_req = torch.arange(B, device=x.device)[:, None].expand_as(ids)
        slots = self._slots(
            INDEXER_KV, read_positions.reshape(-1), read_req.reshape(-1)
        ).view(B, capacity)
        slots = torch.where(
            ids < ((starts + S) // self.compress_ratio)[:, None], slots, -1
        )
        keys = dequantize_indexer_k(index_pool, slots.reshape(-1)).view(
            B, capacity, self.index_head_dim
        )
        self._shared_attention["global"] = {self.layer_id: keys}

    def _select_indices_decode(self, x, qr, positions):
        """Capture-static index scoring with device-side causal/candidate masks."""
        shared = self._shared_attention
        if not self.is_index_source:
            return shared["topk"][self.index_source_layer_id]
        B, S, _ = x.shape
        T = B * S
        keys = shared["global"][self.kv_source_layer_id]
        from rtp_llm.models_py.modules.dsv4.fp8 import (
            _v41_decode_indexer as decode_indexer,
        )

        paged = isinstance(keys, decode_indexer.DecodeIndexerKeys)
        capacity = keys.capacity if paged else keys.shape[1]
        flat = x.reshape(T, -1)
        q = self._lin(self.index_wq, qr.reshape(T, -1)).view(
            T, self.index_n_heads, self.index_head_dim
        )
        weights = (
            F.linear(flat, self.index_weights).float()
            * (self.index_head_dim * self.index_n_heads) ** -0.5
        )
        freqs = self.freqs_cis[positions]
        visible_per_token = (positions + 1) // self.compress_ratio
        if paged:
            all_logits = decode_indexer.score_decode_indexer(
                q.view(B, S, self.index_n_heads, self.index_head_dim),
                weights.view(B, S, self.index_n_heads),
                freqs,
                keys.pool,
                keys.block_table,
                visible_per_token.view(B, S),
                max_ctx_len=capacity,
                rope_head_dim=self.rope_head_dim,
                logical_entries_per_block=keys.logical_entries_per_block,
            )
            if all_logits is None:
                raise RuntimeError(
                    "V4.1 paged indexer support changed within a forward"
                )
        else:
            q = fp8_roundtrip(rope_only(q, freqs, self.rope_head_dim))
        config = self.v41_config
        candidate_source = int(config.get("candidate_source_layer_id", -1))
        candidate_blocks = int(config.get("candidate_topk_blocks", 0))
        candidate_size = int(config.get("candidate_block_size", 0))
        publish = (
            self.layer_id == candidate_source
            and candidate_blocks > 0
            and candidate_size > 0
        )
        use_candidates = (
            candidate_size > 0
            and (capacity + candidate_size - 1) // candidate_size > candidate_blocks
        )
        if publish:
            shared["candidates"] = (
                torch.empty(T, candidate_blocks, dtype=torch.int32, device=x.device)
                if use_candidates
                else None
            )
        output = torch.full(
            (T, self.index_topk), -1, dtype=torch.int32, device=x.device
        )
        columns = None if paged else torch.arange(capacity, device=x.device)
        for b in range(B):
            for start in range(b * S, (b + 1) * S, 16):
                end = min(start + 16, (b + 1) * S)
                if paged:
                    logits = all_logits[start:end]
                else:
                    logits = torch.einsum("thd,kd->thk", q[start:end], keys[b]).relu_()
                    logits = (logits * weights[start:end, :, None]).sum(1)
                visible = visible_per_token[start:end]
                if not paged:
                    logits.masked_fill_(columns[None] >= visible[:, None], -torch.inf)
                candidates = shared.get("candidates")
                if candidates is not None:
                    if publish:
                        candidates[start:end] = select_candidate_blocks(
                            logits, visible, candidate_size, candidate_blocks
                        )
                    elif candidate_source >= 0 and self.layer_id > candidate_source:
                        mask_candidate_logits(
                            logits, candidates[start:end], candidate_size
                        )
                count = min(self.index_topk, capacity)
                scores, selected = logits.topk(count, dim=-1)
                selected = torch.where(scores.isfinite(), selected, -1)
                # Preserve the short-context ascending-order shortcut without
                # a CPU branch; it also avoids unstable ties between zero keys.
                dense = torch.arange(count, device=x.device)[None].expand(
                    end - start, -1
                )
                dense = torch.where(dense < visible[:, None], dense, -1)
                selected = torch.where(
                    (visible <= self.index_topk)[:, None], dense, selected
                )
                output[start:end, :count] = selected.int()
        shared["topk"] = {self.layer_id: output}
        return output

    def _decode_global_slots(self, selected, req):
        """Translate compressed indices without materializing token positions."""
        region = self._global_region()
        pool = self._source_pool(region)
        table = self._block_tables_by_type[region]
        raw_tpb = require_pool_tokens_per_block(self._kv_cache, region=region)
        logical_entries = raw_tpb // self.compress_ratio
        cp = self._cp_ctx
        sharded = cp is not None and cp.cp_size > 1 and cp.kv_cache_sharded
        if (
            os.environ.get("DSV41_FUSED_DECODE_SLOTS", "1") != "0"
            and selected.is_cuda
            and selected.dtype == torch.int32
            and req.dtype == torch.int32
            and table.dtype == torch.int32
            and selected.shape[1] > 0
            and not sharded
            and raw_tpb % self.compress_ratio == 0
            and pool.shape[1] == logical_entries
        ):
            from rtp_llm.models_py.modules.dsv4.fp8.decode.paged_topk_translator import (
                translate_local_to_global_slots,
            )

            return translate_local_to_global_slots(
                req,
                table,
                selected,
                entries_per_block=pool.shape[1],
                tokens_per_block_for_block_table=logical_entries,
            )
        slots = self._slots(
            region,
            (selected.long().clamp_min(0) + 1).reshape(-1) * self.compress_ratio - 1,
            req[:, None].expand_as(selected).reshape(-1),
        ).reshape_as(selected)
        slots.masked_fill_(selected < 0, -1)
        return slots.int()

    def _forward_decode_body(self, x, metadata):
        from rtp_llm.models_py.modules.dsv4.fp8.decode.compute_qkv import (
            decode_compute_qkv,
        )

        self._begin_forward()
        B, S, _ = x.shape
        T = B * S
        positions = metadata.position_ids[:T].long()
        qkv = decode_compute_qkv(self, x, positions)
        self._decode_write_swa_fp8(qkv.kv, B, S, metadata)
        if not self.compress_ratio:
            o = self._forward_decode_swa_only(qkv.q, B, S, metadata)
        else:
            if self.is_kv_source:
                req = metadata.req_id_per_token[:T].long()
                starts = metadata.start_pos[:B].long()
                self._produce_global_decode(x, positions, req, starts)
            selected = self._select_indices_decode(x, qkv.qr, positions)
            from rtp_llm.models_py.modules.dsv4.fp8.decode.attention_kernels import (
                attn_fp8_dual_paged,
            )
            from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata import (
                get_or_build_sched_meta,
            )

            slots = self._decode_global_slots(selected, metadata.req_id_per_token[:T])
            o = attn_fp8_dual_paged(
                q=qkv.q,
                swa_pool_3d=self._pool_view_3d_fp8(SWA_KV),
                cmp_pool_3d=self._source_pool(self._global_region()),
                attn_sink=self.attn_sink,
                swa_topk_3d=metadata.swa_global_slots[:T]
                .view(B, S, self.window_size)
                .contiguous(),
                cmp_topk_3d=slots.int().view(B, S, self.index_topk).contiguous(),
                swa_block_table=metadata.pool_block_tables[SWA_KV][:B],
                sched_meta=get_or_build_sched_meta(
                    metadata,
                    batch_size=B,
                    q_len=S,
                    num_heads=self.n_heads,
                    topk=self.window_size,
                    extra_attn_type=self._global_region(),
                ),
                fp8_op=self._get_fp8_decode_op(),
                topk_length=None,
                extra_topk_length=None,
            )
        out = self._project_output(o, qkv.freqs_cis)
        self._prefill_output_all_reduce(out)
        return out.view(B, S, self.dim)
