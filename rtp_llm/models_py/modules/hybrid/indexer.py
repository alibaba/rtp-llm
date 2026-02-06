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
    # @torch.compile(dynamic=True)
    def _get_logits_head_gate(
        self, x: torch.Tensor, q_scale: torch.Tensor
    ) -> torch.Tensor:
        x = x.float()  # 一个小算子
        weights = self.weights_proj(x)  # 一个算子
        # weights = weights * self.weights_scale # 一个算子
        # weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale # 两个算子
        scale = self.softmax_scale * self.weights_scale
        weights = weights.unsqueeze(-1) * q_scale * scale  # 两个算子
        return weights

    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
        ks: Optional[torch.Tensor] = None,
        is_paged: bool = False,
        is_ragged: bool = False,
        use_nas_fuse_topk: bool = True,
        params: Any = None,
    ) -> torch.Tensor:
        from rtp_llm.models_py.kernels.cuda.fast_topk import (
            fast_topk_transform_fused,
            fast_topk_transform_ragged_fused,
            fast_topk_v2,
        )

        if not use_nas_fuse_topk:
            return fast_topk_v2(logits, params.expanded_seq_lens.to("cuda"), topk)
        elif is_paged:
            # NOTE(dark): if fused, we return a transformed page table directly
            return fast_topk_transform_fused(
                score=logits,
                lengths=params.expanded_seq_lens.to("cuda"),  # expanded_seq_lens
                page_table_size_1=params.page_table_1.to("cuda"),  # page_indices
                cu_seqlens_q=params.cu_q_seqlens.to("cuda"),  # bs + 1
                topk=topk,
                row_starts=None,
            )
        elif is_ragged:
            return fast_topk_transform_ragged_fused(
                score=logits,
                lengths=params.expanded_seq_lens.to("cuda"),
                topk_indices_offset=params.topk_indices_offset.to("cuda"),
                topk=topk,
                row_starts=ks,
            )
        else:
            assert False, f"Unsupported {self.topk_transform_method = }"

    def _get_topk_paged(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        kv_cache: KVCache,
        params: Any,
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
        max_seq_len = params.block_table.shape[1] * self.blocksize
        schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
            params.seq_lens,
            self.blocksize,
            deep_gemm.get_num_sms(),
        )
        logits = deep_gemm.fp8_paged_mqa_logits(
            q_fp8.unsqueeze(1),
            kv_cache_fp8.view(dtype=torch.uint8),
            weights,
            params.seq_lens,
            params.block_table,
            schedule_metadata,
            max_seq_len,
            clean_logits=False,
        )
        # NOTE(dark): logits should be cleaned in topk_transform（adapter from sglang）
        topk_result = self.topk_transform(
            logits, self.index_topk, is_paged=True, params=params
        )
        return topk_result

    def _get_topk_ragged(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        k_fp8: torch.Tensor,
        k_scale: torch.Tensor,
        params: Any,
    ) -> torch.Tensor:
        weights = weights.squeeze(-1)
        kv_fp8 = (k_fp8, k_scale.view(torch.float32))
        assert (
            params.ks is not None and params.ke is not None
        ), "ks/ke must be prepared in prefill"
        logits = deep_gemm.fp8_mqa_logits(
            q_fp8,
            kv_fp8,
            weights,
            params.ks,
            params.ke,
            clean_logits=False,
        )
        # NOTE(dark): logits should be cleaned in topk_transform（adapter from sglang）
        topk_result = self.topk_transform(
            logits, self.index_topk, ks=params.ks, is_ragged=True, params=params
        )

        return topk_result

    def _get_q_k_bf16(
        self,
        q_lora: torch.Tensor,
        x: torch.Tensor,
        params: Any,
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
            pos_ids=params.positions_d,
            interleave=False,
        )

        query = rotate_activation(q)
        key = rotate_activation(k)

        return query, key

    def _get_k_bf16(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        params: Any,
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
            pos_ids=params.positions_d,
            interleave=False,
        )
        key = rotate_activation(k)

        return key

    def _forward_cuda_k_only(
        self,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        params: Any,
    ) -> torch.Tensor:
        key = self._get_k_bf16(hidden_states, params.positions_d, params)
        assert kv_cache is not None, "kv_cache is required"
        rtp_llm_ops.indexer_k_quant_and_cache(
            key,  # Original key in BF16/FP16 [num_tokens, index_head_dim]
            kv_cache.kv_scale_base,  # [num_blocks, block_size, cache_stride]
            params.slot_mapping,  # [num_tokens] physical slot indices
            self.block_size,  # quantization block size (128)
            self.scale_fmt,  # "ue8m0" for power-of-2 scaling
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        kv_cache: KVCache,
        params: Any,
        use_fast_path: bool = False,
    ) -> torch.Tensor:
        # fast path: only compute and store k cache, skip all q and weights ops
        if use_fast_path:
            self._forward_cuda_k_only(hidden_states, kv_cache, params)
            return None

        query, key = self._get_q_k_bf16(q_lora, hidden_states, params)
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
            params.slot_mapping,  # [num_tokens] physical slot indices
            self.block_size,  # quantization block size (128)
            self.scale_fmt,  # "ue8m0" for power-of-2 scaling
        )
        if params.is_prefill:
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
                params.block_table,  # [batch_size, num_blocks]
                params.cu_kv_seqlens,
            )

        if not params.is_prefill:
            topk_result = self._get_topk_paged(q_fp8, weights, kv_cache, params)
        else:
            topk_result = self._get_topk_ragged(q_fp8, weights, k_fp8, k_scale, params)

        return topk_result
