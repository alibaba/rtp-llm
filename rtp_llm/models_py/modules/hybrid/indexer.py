# 在 mla_attention.py 中添加 Indexer 类和相关功能

from typing import Any, Dict, Optional, Tuple

import deep_gemm
import torch
from einops import rearrange
from torch import nn

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules import RMSNorm
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.rotary_emb import (
    MlaRotaryEmbeddingOp,
)
from rtp_llm.ops import AttentionConfigs, HWKernelConfig
from rtp_llm.ops.compute_ops import KVCache
from rtp_llm.utils.model_weight import W


def kv_cache_cast_to_fp8(x: torch.Tensor) -> torch.Tensor:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    x_fp8 = torch.empty(
        (num_blocks, block_size * (head_dim + 4)), device=x.device, dtype=torch.uint8
    )
    x_fp8[:, : block_size * head_dim] = x_scaled.view(
        num_blocks, block_size * head_dim
    ).view(dtype=torch.uint8)
    x_fp8[:, block_size * head_dim :] = sf.view(num_blocks, block_size).view(
        dtype=torch.uint8
    )
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """
    Hadamard transform for activation rotation.
    Note: This is a simplified version. For production use, you may need
    to use optimized CUDA kernels like sgl_kernel.hadamard_transform.
    """
    assert x.dtype == torch.bfloat16 or x.dtype == torch.float16
    hidden_size = x.size(-1)
    assert (
        hidden_size & (hidden_size - 1)
    ) == 0, "Hidden size must be a power of 2 for Hadamard transform."

    # Simplified Hadamard transform implementation
    # For better performance, consider using optimized CUDA kernels
    scale = hidden_size**-0.5
    # Apply Walsh-Hadamard transform approximation
    # This is a placeholder - you may need to implement proper Hadamard transform
    return x * scale


class Indexer(nn.Module):
    """
    Indexer for DeepSeek-V3.2 DSA (DeepSeek Sparse Attention) mechanism.
    Adapted from sglang's Indexer implementation.
    """

    def __init__(
        self,
        attn_config: AttentionConfigs,
        weights: Dict[str, torch.Tensor],
        cos_sin_cache: torch.Tensor,
        layer_idx: int,
        layernorm_eps: float,
        max_position_embeddings: int,
        rope_theta: float,
        quant_config: object,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        # Get dimensions from config or use defaults
        self.index_n_heads = attn_config.index_n_heads
        self.index_head_dim = attn_config.index_head_dim
        self.index_topk = attn_config.index_topk

        self.rope_head_dim = attn_config.rope_head_dim
        self.q_lora_rank = attn_config.q_lora_rank
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.block_size = 128
        self.scale_fmt = "ue8m0"  # FP8 quantization format
        self.softmax_scale = self.index_head_dim**-0.5
        self.max_model_len = 111 * 1000
        self.num_blocks = self.max_model_len * 3
        self.blocksize = 64

        # Create linear layers
        # wq_b: projects q_lora to index_n_heads * index_head_dim
        self.wq_b = LinearFactory.create_linear_from_weights(
            weights,
            W.mla_indexer_wq_b_w,
            W.mla_indexer_wq_b_s,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )

        # wk: projects hidden_states to index_head_dim
        self.wk = LinearFactory.create_linear_from_weights(
            weights,
            W.mla_indexer_wk_w,
            W.mla_indexer_wk_s,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )

        # k_norm: LayerNorm for keys
        self.k_norm = RMSNorm(
            weights.get(W.mla_indexer_k_norm_gamma, None), eps=layernorm_eps
        )

        # weights_proj: projects hidden_states to index_n_heads (for attention weights)
        self.weights_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.mla_indexer_weights_proj_w,
            W.mla_indexer_weights_proj_s,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )

        self.rotary_emb = MlaRotaryEmbeddingOp(
            head_size=attn_config.nope_head_dim,
            cos_sin_cache=cos_sin_cache,
            kv_lora_rank=attn_config.kv_lora_rank,
            rope_head_dim=attn_config.rope_head_dim,
            token_per_block=attn_config.tokens_per_block,
            is_neox_style=False,
        )

    @torch.compile(dynamic=True)
    def _get_logits_head_gate(self, x: torch.Tensor, q_scale: torch.Tensor):
        weights, _ = self.weights_proj(x.float())
        weights = weights * self.n_heads**-0.5
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
        return weights

    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
        seq_len: int,
        is_paged: bool,
        is_ragged: bool,
        use_nas_fuse_topk: bool = True,
    ) -> torch.Tensor:
        # todo: adapter topk from sglang
        from sgl_kernel import (
            fast_topk_transform_fused,
            fast_topk_transform_ragged_fused,
            fast_topk_v2,
        )

        bs = 1
        if not use_nas_fuse_topk:
            return fast_topk_v2(logits, seq_len, topk)
        elif is_paged:
            src_page_table = torch.arange(
                0, seq_len, dtype=torch.int32, device=logits.device
            )
            src_page_table = src_page_table.unsqueeze(0).expand(bs, -1)
            cu_seqlens_q = torch.arange(
                0, bs + 1, dtype=torch.int32, device=logits.device
            )
            # NOTE(dark): if fused, we return a transformed page table directly
            return fast_topk_transform_fused(
                score=logits,
                lengths=seq_len,
                page_table_size_1=src_page_table,
                cu_seqlens_q=cu_seqlens_q,
                topk=topk,
            )
        elif is_ragged:
            topk_indices_offset = torch.randint(
                0, 1024, (bs,), dtype=torch.int32, device=logits.device
            )
            return fast_topk_transform_ragged_fused(
                score=logits,
                lengths=seq_len,
                topk_indices_offset=topk_indices_offset,
                topk=topk,
            )
        else:
            assert False, f"Unsupported {self.topk_transform_method = }"

    def _get_topk_paged(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:

        weights = weights.squeeze(2)
        seq_len = q_fp8.shape[0]
        batch_size = 1
        # todo: use kv cache from cache manager
        kv_cache = torch.randn(
            (self.num_blocks, self.blocksize, 1, self.index_head_dim),
            device=q_fp8.device,
            dtype=torch.bfloat16,
        )
        kv_cache_fp8 = kv_cache_cast_to_fp8(kv_cache)

        avg_kv = 8192
        context_lens = (
            torch.randint(int(0.7 * avg_kv), int(1.3 * avg_kv), (batch_size,))
            .cuda()
            .to(torch.int32)
        )
        max_block_len = (
            (context_lens.max().item() + self.blocksize - 1)
            // self.blocksize
            * self.blocksize
        )
        block_tables = torch.zeros(
            (batch_size, max_block_len), device=q_fp8.device, dtype=torch.int32
        )
        schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
            context_lens, self.blocksize, deep_gemm.get_num_sms()
        )
        logits = deep_gemm.fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8,
            weights,
            context_lens,
            block_tables,
            schedule_metadata,
            self.max_model_len,
            clean_logits=False,
        )

        # NOTE(dark): logits should be cleaned in topk_transform
        topk_result = self.topk_transform(
            logits, self.index_topk, seq_len, is_paged=True
        )
        return topk_result

    def _get_topk_ragged(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        k_fp8: torch.Tensor,
        k_scale: torch.Tensor,
    ) -> torch.Tensor:

        weights = weights.squeeze(-1)
        seq_len = q_fp8.shape[0]
        kv_fp8 = (k_fp8, k_scale)
        seq_len_kv = k_fp8.shape[0]
        ks = torch.zeros(seq_len, dtype=torch.int, device=kv_fp8.device)
        ke = torch.arange(seq_len, dtype=torch.int, device=kv_fp8.device) + (
            seq_len_kv - seq_len
        )

        logits = deep_gemm.fp8_mqa_logits(
            q_fp8,
            kv_fp8,
            weights,
            ks,
            ke,
            clean_logits=False,
        )
        token_nums, _, _ = q_fp8.shape
        raw_topk_result = self.topk_transform(
            logits, self.index_topk, seq_len, is_ragged=True
        )
        topk_result = torch.full(
            (token_nums, self.index_topk), -1, device=q_fp8.device, dtype=torch.int32
        )
        topk_result = raw_topk_result
        return topk_result

    def _get_q_k_bf16(
        self,
        q_lora: torch.Tensor,
        x: torch.Tensor,
        positions: torch.Tensor,
    ):

        query, _ = self.wq_b(q_lora)
        query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
        q_rope, _ = torch.split(
            query, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        key, _ = self.wk(x)
        key = self.k_norm(key)
        k_rope, _ = torch.split(
            key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )

        q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)

        query[..., : self.rope_head_dim] = q_rope
        key[..., : self.rope_head_dim] = k_rope

        query = rotate_activation(query)
        key = rotate_activation(key)

        return query, key

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        query, key = self._get_q_k_bf16(q_lora, hidden_states, positions)
        q_fp8, q_scale = sgl_per_token_group_quant_fp8(
            query,
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=self.scale_fmt,
        )
        k_fp8, k_scale = sgl_per_token_group_quant_fp8(
            key,
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=self.scale_fmt,
        )

        weights = self._get_logits_head_gate(hidden_states, q_scale)

        if not self.config.is_prefill:
            topk_result = self._get_topk_paged(q_fp8, weights)
        else:
            topk_result = self._get_topk_ragged(q_fp8, weights)

        return topk_result
