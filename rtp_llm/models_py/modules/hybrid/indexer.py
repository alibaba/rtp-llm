from typing import Any, Dict, Optional

import deep_gemm
import flashinfer.rope as rope
import torch
from torch import nn

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules import LayerNorm
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.ops import AttentionConfigs, HWKernelConfig
from rtp_llm.ops.compute_ops import KVCache, rtp_llm_ops
from rtp_llm.utils.model_weight import W


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """
    Hadamard transform for activation rotation.
    Note: This is a simplified version. For production use, you may need
    to use optimized CUDA kernels like sgl_kernel.hadamard_transform.
    """
    assert x.dtype == torch.bfloat16
    from fast_hadamard_transform import hadamard_transform

    hidden_size = x.size(-1)
    assert (
        hidden_size & (hidden_size - 1)
    ) == 0, "Hidden size must be a power of 2 for Hadamard transform."

    return hadamard_transform(x, scale=hidden_size**-0.5)


def unpack_ue8m0_scale(sf_packed: torch.Tensor) -> torch.Tensor:
    # sf_packed: (..., num_scales), dtype=int32
    # UE8M0 format: scale is stored in the lowest byte of int32.

    # Extract the lowest byte via bitwise ops to avoid view.
    sf_u8 = (sf_packed & 0xFF).to(torch.int32)  # extract lowest byte
    # Shift left to float32 exponent position (bits 23-30).
    sf_i32 = sf_u8 << 23
    # Reinterpret as float32.
    sf_fp32 = sf_i32.view(torch.float32)
    return sf_fp32


class Indexer(nn.Module):
    """
    Indexer for DeepSeek-V3.2 DSA (DeepSeek Sparse Attention) mechanism.
    Adapted from sglang's Indexer implementation.
    """

    def __init__(
        self,
        attn_config: AttentionConfigs,
        weights: Dict[str, torch.Tensor],
        global_weights: Dict[str, torch.Tensor],
        layer_idx: int,
        layernorm_eps: float,
        quant_config: object,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        scale_fmt: Optional[str] = "none",
    ):
        super().__init__()
        self.layer_idx = layer_idx

        self.index_n_heads = attn_config.indexer_head_num
        self.index_head_dim = attn_config.indexer_head_dim
        self.index_topk = attn_config.indexer_topk

        self.rope_head_dim = attn_config.rope_head_dim
        self.block_size = 128  # quantization block size (128)
        self.head_kv = 1
        self.scale_fmt = scale_fmt  # FP8 quantization format
        self.softmax_scale = self.index_head_dim**-0.5
        self.weights_scale = self.index_n_heads**-0.5
        self.blocksize = attn_config.tokens_per_block  # page size, typically 64
        self.indexer_size = self.index_head_dim / 2 + self.index_head_dim / 128 * 2

        self.wq_b = LinearFactory.create_linear_from_weights(
            weights,
            W.mla_indexer_qb_w,
            W.mla_indexer_qb_s,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )

        self.wk = LinearFactory.create_linear_from_weights(
            weights,
            W.mla_indexer_k_w,
            W.mla_indexer_k_s,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )

        self.k_norm = LayerNorm(
            weights[W.mla_indexer_k_norm_w],
            weights[W.mla_indexer_k_norm_b],
            eps=layernorm_eps,
        )

        self.weights_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.mla_indexer_weights_proj_w,
            None,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )
        self.cos_sin_cache = global_weights[W.rope_cos_sin_cache]

    # TODO: fuse kernel here
    def _get_logits_head_gate(
        self, x: torch.Tensor, q_scale: torch.Tensor
    ) -> torch.Tensor:
        x = x.float()
        weights = self.weights_proj(x)
        scale = self.softmax_scale * self.weights_scale
        weights = weights.unsqueeze(-1) * q_scale * scale
        return weights

    def _topk_transform_decode(
        self,
        logits: torch.Tensor,
        fmha_params: Any,
        attention_inputs: Any,
    ) -> torch.Tensor:
        """TopK transform for decode phase (paged attention)."""
        from rtp_llm.models_py.kernels.cuda.fast_topk import fast_topk_transform_fused

        assert (
            fmha_params.expanded_seq_lens.device == logits.device
        ), "expanded_seq_lens must be on the same device as logits"
        assert (
            fmha_params.page_table_1.device == logits.device
        ), "page_table_1 must be on the same device as logits"
        assert (
            attention_inputs.decode_cu_seqlens_d.device == logits.device
        ), "cu_seqlens must be on the same device as logits"
        # NOTE(dark): if fused, we return a transformed page table directly
        return fast_topk_transform_fused(
            score=logits,
            lengths=fmha_params.expanded_seq_lens,  # expanded_seq_lens
            page_table_size_1=fmha_params.page_table_1,  # page_indices
            cu_seqlens_q=attention_inputs.decode_cu_seqlens_d,  # bs + 1
            topk=self.index_topk,
            row_starts=None,
        )

    def _topk_transform_prefill(
        self,
        logits: torch.Tensor,
        fmha_params: Any,
    ) -> torch.Tensor:
        """TopK transform for prefill phase (ragged attention)."""
        from rtp_llm.models_py.kernels.cuda.fast_topk import (
            fast_topk_transform_ragged_fused,
        )

        assert (
            fmha_params.expanded_seq_lens.device == logits.device
        ), "expanded_seq_lens must be on the same device as logits"
        assert (
            fmha_params.topk_indices_offset.device == logits.device
        ), "topk_indices_offset must be on the same device as logits"
        assert (
            fmha_params.ks.device == logits.device
        ), "ks must be on the same device as logits"
        return fast_topk_transform_ragged_fused(
            score=logits,
            lengths=fmha_params.expanded_seq_lens,
            topk_indices_offset=fmha_params.topk_indices_offset,
            topk=self.index_topk,
            row_starts=fmha_params.ks,
        )

    def _get_topk_paged(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        kv_cache: KVCache,
        fmha_params: Any,
        attention_inputs: Any,
    ) -> torch.Tensor:

        weights = weights.view(-1, self.index_n_heads)
        kv_cache_fp8 = kv_cache.kv_scale_base

        block_kv = self.blocksize
        num_heads_kv = 1
        head_dim_with_sf = (
            self.index_head_dim + self.index_head_dim // self.block_size * 4
        )
        kv_cache_fp8 = kv_cache_fp8.view(
            kv_cache_fp8.shape[0], block_kv, num_heads_kv, head_dim_with_sf
        ).view(dtype=torch.uint8)
        max_seq_len = (
            attention_inputs.kv_cache_block_id_device.shape[1] * self.blocksize
        )
        schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
            fmha_params.kvlen_d,
            self.blocksize,
            deep_gemm.get_num_sms(),
        )
        logits = deep_gemm.fp8_paged_mqa_logits(
            q_fp8.unsqueeze(1),
            kv_cache_fp8.view(dtype=torch.uint8),
            weights,
            fmha_params.kvlen_d,
            attention_inputs.kv_cache_block_id_device,
            schedule_metadata,
            max_seq_len,
            clean_logits=False,
        )
        topk_result = self._topk_transform_decode(logits, fmha_params, attention_inputs)
        return topk_result

    def _get_topk_ragged(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        k_fp8: torch.Tensor,
        k_scale: torch.Tensor,
        fmha_params: Any,
    ) -> torch.Tensor:
        weights = weights.squeeze(-1)
        kv_fp8 = (k_fp8, k_scale.view(torch.float32))
        assert (
            fmha_params.ks is not None and fmha_params.ke is not None
        ), "ks/ke must be prepared in prefill"
        logits = deep_gemm.fp8_mqa_logits(
            q_fp8,
            kv_fp8,
            weights,
            fmha_params.ks,
            fmha_params.ke,
            clean_logits=False,
        )
        topk_result = self._topk_transform_prefill(logits, fmha_params)

        return topk_result

    def _get_q_k_bf16(
        self,
        q_lora: torch.Tensor,
        x: torch.Tensor,
        flashmla_params: Any,
    ):
        q = self.wq_b(q_lora)
        q = q.view(-1, self.index_n_heads, self.index_head_dim)
        q_pe = q[:, :, : self.index_head_dim - self.rope_head_dim]

        k = self.wk(x)
        k = self.k_norm(k)
        k_pe = k[:, : self.index_head_dim - self.rope_head_dim]

        # same as vllm indexer rope
        rope._apply_rope_pos_ids_cos_sin_cache(
            q=q_pe,
            k=k_pe.unsqueeze(1),
            q_rope=q_pe,
            k_rope=k_pe.unsqueeze(1),
            cos_sin_cache=self.cos_sin_cache,
            pos_ids=flashmla_params.positions_d,
            interleave=False,
        )

        query = rotate_activation(q)
        key = rotate_activation(k)

        return query, key

    def _get_k_bf16(
        self,
        x: torch.Tensor,
        flashmla_params: Any,
    ):
        # Compute only key, skip query
        k = self.wk(x)
        k = self.k_norm(k)
        k_pe = k[:, : self.index_head_dim - self.rope_head_dim]

        # same as vllm indexer rope
        rope._apply_rope_pos_ids_cos_sin_cache(
            q=k_pe.unsqueeze(1),
            k=k_pe.unsqueeze(1),
            q_rope=k_pe.unsqueeze(1),
            k_rope=k_pe.unsqueeze(1),
            cos_sin_cache=self.cos_sin_cache,
            pos_ids=flashmla_params.positions_d,
            interleave=False,
        )
        key = rotate_activation(k)

        return key

    def _forward_cuda_k_only(
        self,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        fmha_params: Any,
    ) -> torch.Tensor:
        key = self._get_k_bf16(hidden_states, fmha_params)
        assert kv_cache is not None, "kv_cache is required"
        rtp_llm_ops.indexer_k_quant_and_cache(
            key,  # Original key in BF16/FP16 [num_tokens, index_head_dim]
            kv_cache.kv_scale_base,  # [num_blocks, block_size, cache_stride]
            fmha_params.slot_mapping,  # [num_tokens] physical slot indices
            self.block_size,  # quantization block size (128)
            self.scale_fmt,  # "ue8m0" for power-of-2 scaling
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        kv_cache: KVCache,
        fmha_params: Any,
        attention_inputs: Any,
        use_fast_path: bool,
    ) -> torch.Tensor:
        # fast path: only compute and store k cache, skip all q and weights ops
        if use_fast_path:
            self._forward_cuda_k_only(hidden_states, kv_cache, fmha_params)
            return None

        query, key = self._get_q_k_bf16(q_lora, hidden_states, fmha_params)
        query = query.view(-1, self.index_head_dim)
        q_fp8, q_scale = sgl_per_token_group_quant_fp8(
            query,
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=(self.scale_fmt == "ue8m0"),
        )
        q_fp8 = q_fp8.view(-1, self.index_n_heads, self.index_head_dim)
        if self.scale_fmt == "ue8m0":
            q_scale = unpack_ue8m0_scale(q_scale)
        q_scale = q_scale.view(-1, self.index_n_heads, 1)
        weights = self._get_logits_head_gate(hidden_states, q_scale)

        assert kv_cache is not None, "kv_cache is required"
        rtp_llm_ops.indexer_k_quant_and_cache(
            key,  # Original key in BF16/FP16 [num_tokens, index_head_dim]
            kv_cache.kv_scale_base,  # [num_blocks, block_size, cache_stride]
            fmha_params.slot_mapping,  # [num_tokens] physical slot indices
            self.block_size,  # quantization block size (128)
            self.scale_fmt,  # "ue8m0" for power-of-2 scaling
        )
        if attention_inputs.is_prefill:
            num_tokens = key.shape[0]
            k_fp8 = torch.empty(
                (num_tokens, self.index_head_dim),
                dtype=torch.float8_e4m3fn,
                device=key.device,
            )
            k_scale = torch.empty(
                (num_tokens, self.index_head_dim // self.block_size * 4),
                dtype=torch.uint8,
                device=key.device,
            )

            rtp_llm_ops.cp_gather_indexer_k_quant_cache(
                kv_cache.kv_scale_base,  # [num_blocks, block_size, cache_stride]
                k_fp8,  # output [num_tokens, index_head_dim]
                k_scale,  # output [num_tokens, scale_size]
                attention_inputs.kv_cache_block_id_device,  # [batch_size, num_blocks]
                attention_inputs.cu_kv_seqlens,
            )

        if not attention_inputs.is_prefill:
            topk_result = self._get_topk_paged(
                q_fp8, weights, kv_cache, fmha_params, attention_inputs
            )
        else:
            topk_result = self._get_topk_ragged(
                q_fp8, weights, k_fp8, k_scale, fmha_params
            )

        return topk_result
