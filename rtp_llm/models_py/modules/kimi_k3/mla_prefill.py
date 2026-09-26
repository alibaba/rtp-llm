"""K3 expanded prefill MLA, including reads of cached latent vectors."""

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla import MlaFlashInferPrefillOp
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_fp8_kernels import gather_fp8_prefix
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_qkv_fp8_quant import quantize_qkv_fp8
from rtp_llm.models_py.modules.kimi_k3.linear import KimiK3Bf16Linear
from rtp_llm.utils.model_weight import W
from rtp_llm.models_py.modules.kimi_k3.native_mla_prefill import KimiK3TokenspeedPrefill
from rtp_llm.ops import KvCacheDataType

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
    MlaFlashInferPrefillImpl,
)


class KimiK3MlaPrefillOp(MlaFlashInferPrefillOp):
    def _attention_dtype(self):
        return torch.float8_e4m3fn if self.kv_cache_type == KvCacheDataType.FP8 else torch.bfloat16

    def _create_prefill_wrapper(self):
        return KimiK3TokenspeedPrefill(fp8_compute=self.kv_cache_type == KvCacheDataType.FP8)

    def _reuse_kv_cache_indexed_batched(self, compressed_kv, k_pe, kv_cache):
        if self.kv_cache_type == KvCacheDataType.FP8:
            if self.reuse_cache_page_indice is None or self.reuse_cache_page_indice.numel() == 0:
                return compressed_kv, k_pe
            if kv_cache is None:
                raise ValueError("K3 FP8 prefix reuse requires a target KV cache")
            latent = torch.empty((self.total_kv_lens, self.kv_lora_rank),
                                 dtype=torch.bfloat16, device=compressed_kv.device)
            suffix = torch.empty((self.total_kv_lens, self.qk_rope_head_dim),
                                 dtype=torch.bfloat16, device=compressed_kv.device)
            gather_fp8_prefix(
                latent, suffix, compressed_kv, k_pe,
                kv_cache.kv_cache_base.view(-1, self.token_per_block,
                                            self.kv_lora_rank + self.qk_rope_head_dim),
                self.reuse_cache_page_indice, self.batch_reuse_info_vec,
                self.qo_indptr, self.token_per_block, scale=1.0,
            )
            return latent, suffix
        latent, suffix = super()._reuse_kv_cache_indexed_batched(compressed_kv, k_pe, kv_cache)
        # The shared gather reserves full cache pages, but packs only each
        # request's prefix and query rows. TokenSpeed expects that valid extent.
        # Narrow before kv_b_proj so unused capacity is neither projected nor read.
        return latent[: self.total_kv_lens], suffix[: self.total_kv_lens]

    def _make_kv_b_proj(self, layer_id):
        weight = self.weights[layer_id][W.mla_kv_b_w]
        if weight.is_cuda and weight.dtype == torch.bfloat16:
            return KimiK3Bf16Linear(weight)
        return super()._make_kv_b_proj(layer_id)

    def forward(self, q, compressed_kv, k_pe, kv_cache, layer_id):
        if self.kv_cache_type != KvCacheDataType.FP8:
            return super().forward(q, compressed_kv, k_pe, kv_cache, layer_id)
        compressed_kv, k_pe = self._reuse_kv_cache_indexed_batched(
            compressed_kv, k_pe, kv_cache
        )
        kv = self._make_kv_b_proj(layer_id)(compressed_kv)
        kv = kv.view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k = self._concat_and_cast_mha_k(
            kv[:, :, : self.qk_nope_head_dim], k_pe.view(-1, 1, self.qk_rope_head_dim)
        )
        q_fp8, k_fp8, v_fp8 = quantize_qkv_fp8(
            q, k, kv[:, :, self.qk_nope_head_dim :]
        )
        return self.prefill_wrapper.run(q_fp8, k_fp8, v_fp8).view(
            -1, self.num_heads, self.v_head_dim
        )


class KimiK3MlaPrefillImpl(MlaFlashInferPrefillImpl):
    prefill_op_type = KimiK3MlaPrefillOp

    def __init__(
        self, config, parallelism, weights, inputs, fmha_config, is_cuda_graph
    ):
        if is_cuda_graph:
            raise ValueError(
                "K3 ordinary prefill requires eager planning; verify and draft "
                "updates use the separate paged MLA graph implementation"
            )
        attention = config.getAttentionConfigs(parallelism.get_attn_tp_size())
        inputs.headwise_config = getattr(config, "headwise_config", None)
        # Keep expanded attention for both full and reused prefixes. Native
        # TokenSpeed arithmetic runs under RTP's cache/state planning.
        super().__init__(
            attention,
            inputs,
            weights.weights,
            None,  # K3 uses NoPE.
            fmha_config,
            quant_config=config.attention_projection_quant_config,
            max_seq_len=config.max_seq_len,
            is_cuda_graph=False,
            parallelism_config=parallelism,
            allow_absorb=False,
        )
