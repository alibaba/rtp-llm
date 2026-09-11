"""PPU FP4 Indexer, retaining RTP's instance and pool lifecycle."""

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt
from rtp_llm.models_py.modules.dsv4._profiler import record_function_range
from rtp_llm.models_py.modules.dsv4.fp8.compressor import (
    CompressorFP8,
    _linear_bf16_bf16_fp32,
)
from rtp_llm.models_py.modules.dsv4.fp8.indexer import IndexerFP8
from rtp_llm.utils.model_weight import W

from ...kernels.cuda.ppu_fp4_indexer import (
    compress4,
    compress4_decode,
    norm_rope_store,
    paged_score,
    quantize_q,
    topk_bf16,
    topk_decode,
)
from ...kernels.ppu_fp4_indexer_cache import build_decode_plan, build_plans, gather_k
from .ppu_rope_attention import PpuRopeAttention


class PpuFP4Compressor(CompressorFP8):
    def __init__(self, *args, compressor_weights, **kwargs):
        super().__init__(*args, compressor_weights=compressor_weights, **kwargs)
        if (self.head_dim, self.rope_head_dim, self.compress_ratio) != (128, 64, 4):
            raise ValueError("FP4 Indexer compressor requires head128/rope64/C4")
        self._pool_entry_bytes = 68
        self.norm.weight = torch.nn.Parameter(
            compressor_weights["norm"].float().contiguous(), requires_grad=False
        )
        # SG reorders overlap/current APE rows once after loading, privately.
        self.ape = torch.nn.Parameter(
            torch.cat(self.ape.chunk(2, dim=-1), dim=0).contiguous(),
            requires_grad=False,
        )

    def _launch(self, kv_flat, score_flat, meta, seq_start=None):
        if self._state_pool_3d is None or self._kv_pool_view is None:
            return
        if self._cp_ctx is not None and self._cp_ctx.cp_size > 1:
            raise ValueError("FP4 Indexer CP is not qualified")
        n = kv_flat.shape[0]
        if not n:
            return
        # Parent projection supplies two views of one contiguous FP32 [N,512].
        if (
            kv_flat.stride() != (512, 1)
            or score_flat.stride() != (512, 1)
            or score_flat.data_ptr() != kv_flat.data_ptr() + 256 * 4
        ):
            raise ValueError("FP4 compressor requires the fused KV/score projection")
        fused = kv_flat.as_strided((n, 512), (512, 1))
        c, w, slots = build_plans(
            meta,
            self._state_block_table,
            self._state_eb,
            self._state_tokens_per_block,
            seq_start,
        )
        with record_function_range("dsv4.ppu.fp4.indexer.compress"):
            compressed = compress4(self._state_pool_3d, fused, self.ape, c, w)
        freqs = torch.view_as_real(self.freqs_cis).flatten(-2)
        with record_function_range("dsv4.ppu.fp4.indexer.norm_rope_store"):
            norm_rope_store(
                compressed,
                c,
                self.norm.weight,
                self.norm_eps,
                freqs,
                slots,
                self._kv_pool_view,
                self._kv_eb,
            )

    def forward_decode_vectorized(self, x, start_pos, meta=None, position_ids=None):
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("FP4 Decode compressor requires one token per request")
        if meta is None or meta.positions.numel() != x.shape[0]:
            raise ValueError("FP4 Decode compressor requires step-level metadata")
        if self._state_pool_3d is None or self._kv_pool_view is None:
            raise RuntimeError("FP4 Decode compressor pools were not bound")
        if not x.shape[0]:
            return
        fused = _linear_bf16_bf16_fp32(
            x, self._wkv_wgate_fused, linear_op=self._bf16_fp32_linear
        ).reshape(x.shape[0], 512)
        plan, slots = build_decode_plan(
            meta, self._state_block_table, self._state_eb, self._state_tokens_per_block
        )
        compressed = compress4_decode(self._state_pool_3d, fused, self.ape, plan)
        norm_rope_store(
            compressed,
            plan,
            self.norm.weight,
            self.norm_eps,
            torch.view_as_real(self.freqs_cis).flatten(-2),
            slots,
            self._kv_pool_view,
            self._kv_eb,
            is_decode=True,
        )


class PpuFP4Indexer(IndexerFP8):
    CACHE_ENTRY_BYTES = 68

    def __init__(self, *args, layer_weights, **kwargs):
        super().__init__(
            *args,
            layer_weights=layer_weights,
            compressor_factory=PpuFP4Compressor,
            **kwargs,
        )
        import deep_gemm

        if not callable(getattr(deep_gemm, "fp8_fp4_mqa_logits", None)):
            raise RuntimeError("PPU DeepGEMM FP4 MQA is required")
        self._score = deep_gemm.fp8_fp4_mqa_logits
        self.weights_proj = (
            layer_weights[W.v4_indexer_weights_proj_w].to(torch.bfloat16).contiguous()
        )
        self.weight_scale = self.softmax_scale * self.n_heads**-0.5

    def forward(
        self,
        x,
        qr,
        attention_inputs,
        *,
        workspace,
        cp_gather_stream=None,
        post_gather_stream=None,
    ):
        meta = attention_inputs
        if self._cp_ctx is not None and self._cp_ctx.cp_size > 1:
            raise ValueError("FP4 Indexer CP is not qualified")
        if (
            self._kv_pool_view is None
            or self._kv_block_table is None
            or self._kv_eb <= 0
        ):
            return torch.empty((*x.shape[:-1], 0), device=x.device, dtype=torch.int32)
        self.compressor.freqs_cis = self.freqs_cis
        self._propagate_pool_to_nested()
        try:
            with record_function_range("dsv4.ppu.fp4.indexer.q_projection"):
                q = (
                    self._compute_indexer_q(qr, meta.freqs_cis_slice, apply_rope=False)
                    .reshape(meta.M, self.n_heads, 128)
                    .contiguous()
                )
            self.compressor(
                x, meta.sp_int, meta=meta.compressor_meta, workspace=workspace
            )
            if meta.T == 0:
                return torch.empty(
                    (*x.shape[:-1], 0), device=x.device, dtype=torch.int32
                )
            with record_function_range("dsv4.ppu.fp4.indexer.weights_quant_q"):
                weights = F.linear(x.reshape(meta.M, -1), self.weights_proj)
                q, qs, weights = quantize_q(
                    q,
                    weights,
                    self.weight_scale,
                    torch.view_as_real(self.freqs_cis).flatten(-2),
                    meta.compressor_meta.positions,
                )
            with record_function_range("dsv4.ppu.fp4.indexer.gather_k"):
                k, ks = gather_k(
                    self._kv_pool_view,
                    meta.block_table_i32,
                    meta.cu_kv_seqlens,
                    meta.T,
                    self._kv_eb,
                )
            out = torch.empty(
                (meta.M, self.index_topk), device=x.device, dtype=torch.int32
            )
            chunk = self._prefill_score_chunk_rows or meta.M
            for start in range(0, meta.M, chunk):
                end = min(meta.M, start + chunk)
                with record_function_range("dsv4.ppu.fp4.indexer.score"):
                    logits = self._score(
                        (q[start:end], qs[start:end]),
                        (k, ks),
                        weights[start:end],
                        meta.ks[start:end],
                        meta.ke[start:end],
                        clean_logits=False,
                        logits_dtype=torch.bfloat16,
                    )
                label = getattr(self.compressor, "_profile_label", "")
                if (
                    _rt.LEVEL >= 2
                    and label.startswith("L")
                    and label[1:3].isdigit()
                    and _rt.should_record_layer(int(label[1:3]))
                ):
                    for suffix, tensor in (
                        ("q", q[start:end]),
                        ("q_scale", qs[start:end]),
                        ("w", weights[start:end]),
                        ("k", k),
                        ("k_scale", ks),
                        ("ks", meta.ks[start:end]),
                        ("ke", meta.ke[start:end]),
                        ("logits", logits),
                    ):
                        _rt.record_if_level(
                            2,
                            f"{label.replace('.', '_')}_score_{start}_{suffix}",
                            tensor,
                        )
                with record_function_range("dsv4.ppu.fp4.indexer.topk"):
                    topk_bf16(
                        logits, meta.ks[start:end], meta.ke[start:end], out[start:end]
                    )
                del logits
            return out.view(*x.shape[:-1], self.index_topk)
        finally:
            self._clear_nested_pool()

    def forward_decode_vectorized(
        self,
        x,
        qr,
        start_pos,
        out_topk_buffer,
        position_ids=None,
        compressor_meta=None,
        *,
        q_producer_stream=None,
        decode_streams=None,
    ):
        if (q_producer_stream is None) != (decode_streams is None):
            raise ValueError("Indexer overlap requires producer and auxiliary streams")
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("FP4 Indexer Decode requires one token per request")
        if compressor_meta is None:
            raise ValueError(
                "FP4 Indexer Decode requires step-level compressor metadata"
            )
        if self._cp_ctx is not None and self._cp_ctx.cp_size > 1:
            raise ValueError("FP4 Indexer CP is not qualified")
        if self._kv_pool_view is None or self._kv_block_table is None:
            raise RuntimeError("FP4 Indexer Decode pools were not bound")
        bsz = x.shape[0]
        if compressor_meta.positions.numel() != bsz:
            raise ValueError("FP4 Indexer Decode metadata row counts differ")
        if not bsz:
            return out_topk_buffer
        self.compressor.freqs_cis = self.freqs_cis
        self._propagate_pool_to_nested()
        try:
            if decode_streams is not None:
                from .ppu_decode_indexer import prepare_decode_indexer_overlap

                q, qs, weights = prepare_decode_indexer_overlap(
                    self,
                    x,
                    qr,
                    start_pos,
                    position_ids,
                    compressor_meta,
                    q_producer_stream,
                    decode_streams,
                )
            else:
                self.compressor.forward_decode_vectorized(
                    x, start_pos, meta=compressor_meta, position_ids=position_ids
                )
                q = (
                    self._compute_indexer_q(qr, None, apply_rope=False)
                    .reshape(bsz, self.n_heads, 128)
                    .contiguous()
                )
                weights = F.linear(x.reshape(bsz, -1), self.weights_proj)
                q, qs, weights = quantize_q(
                    q,
                    weights,
                    self.weight_scale,
                    torch.view_as_real(self.freqs_cis).flatten(-2),
                    compressor_meta.positions,
                )
            lengths = compressor_meta.compressed_lens_per_token
            if lengths is None:
                lengths = (compressor_meta.positions + 1) // self.compress_ratio
            lengths = lengths.reshape(bsz, 1).to(torch.int32).contiguous()
            table = self._kv_block_table[:bsz].to(torch.int32).contiguous()
            capacity = table.shape[1] * self._kv_eb
            width = max(32, min(capacity, self._kv_cache_t or capacity))
            logits = paged_score(
                q.reshape(bsz, 1, self.n_heads, 64),
                qs.reshape(bsz, 1, self.n_heads),
                weights,
                self._kv_pool_view,
                table,
                lengths,
                width,
            )
            topk_decode(
                logits.reshape(bsz, width),
                lengths.reshape(bsz),
                out_topk_buffer.reshape(bsz, self.index_topk),
            )
            return out_topk_buffer
        finally:
            self._clear_nested_pool()


class PpuFP4Attention(PpuRopeAttention):
    def __init__(
        self,
        *args,
        decode_stream_pool=None,
        decode_qkv_mode="separate",
        decode_indexer_mode="sequential",
        **kwargs,
    ):
        if decode_qkv_mode not in ("separate", "merged"):
            raise ValueError("PPU Decode QKV must be separate or merged")
        if decode_qkv_mode == "merged" and decode_stream_pool is None:
            raise ValueError("Merged PPU Decode QKV requires model-owned streams")
        if decode_indexer_mode not in ("sequential", "overlap"):
            raise ValueError("PPU Decode Indexer must be sequential or overlap")
        if decode_indexer_mode == "overlap" and decode_stream_pool is None:
            raise ValueError("PPU Indexer overlap requires model-owned streams")
        super().__init__(*args, indexer_factory=PpuFP4Indexer, **kwargs)
        self._decode_streams = None
        self._decode_indexer_streams = None
        self._decode_qkv_projection = None
        if decode_qkv_mode == "merged":
            from rtp_llm.platforms.ppu.modules.linear.fp8_linear import (
                concatenate_ppu_fp8_linears,
            )

            if (self.q_lora_rank, self.head_dim, self.rope_head_dim) != (1024, 512, 64):
                raise ValueError(
                    "Merged PPU Decode QKV requires Flash 1024/512/64 geometry"
                )
            self._decode_qkv_projection = concatenate_ppu_fp8_linears(
                (self.wq_a, self.wkv)
            )
        if decode_stream_pool is not None:
            self._decode_streams = {
                role: decode_stream_pool.get(
                    "attention_" + role, self.wq_a.weight.device
                )
                for role in ("kv", "compressor", "indexer")
            }
            if decode_indexer_mode == "overlap" and self.indexer is not None:
                self._decode_indexer_streams = {
                    role: decode_stream_pool.get(
                        "attention_indexer_" + role, self.wq_a.weight.device
                    )
                    for role in ("q", "weights")
                }

    def _decode_update_indexer(
        self,
        x,
        qr,
        bsz,
        q_len,
        start_pos,
        position_ids,
        attn_metadata,
        *,
        q_producer_stream=None,
    ):
        if q_producer_stream is None:
            return super()._decode_update_indexer(
                x, qr, bsz, q_len, start_pos, position_ids, attn_metadata
            )
        from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
            INDEXER_KV,
            INDEXER_STATE,
        )

        if self.indexer is None or self._decode_indexer_streams is None:
            raise ValueError("Indexer overlap was not prepared for this layer")
        meta = self._decode_compressor_meta_from_metadata(
            attn_metadata,
            state_attn_type=INDEXER_STATE,
            kv_attn_type=INDEXER_KV,
            bsz=bsz,
            q_len=q_len,
        )
        return self.indexer.forward_decode_vectorized(
            x,
            qr,
            start_pos,
            attn_metadata.topk_buffer_compressed[:bsz],
            position_ids=position_ids,
            compressor_meta=meta,
            q_producer_stream=q_producer_stream,
            decode_streams=self._decode_indexer_streams,
        )

    def _forward_decode_body(self, x, attn_metadata):
        if self._decode_streams is None:
            return super()._forward_decode_body(x, attn_metadata)
        from .ppu_decode_attention import decode_attention_overlap

        return decode_attention_overlap(self, x, attn_metadata, self._decode_streams)
