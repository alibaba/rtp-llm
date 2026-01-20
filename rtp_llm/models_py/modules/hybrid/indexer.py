# 在 mla_attention.py 中添加 Indexer 类和相关功能

from typing import Dict, Optional

import deep_gemm
import torch
from einops import rearrange
from torch import nn

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules import LayerNorm
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.rotary_emb import (
    MlaRotaryEmbeddingOp,
)
from rtp_llm.ops import AttentionConfigs, HWKernelConfig
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs
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


class Indexer(nn.Module):
    """
    Indexer for DeepSeek-V3.2 DSA (DeepSeek Sparse Attention) mechanism.
    Adapted from sglang's Indexer implementation.
    """

    def __init__(
        self,
        attn_config: AttentionConfigs,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
        layernorm_eps: float,
        quant_config: object,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        # Get dimensions from config or use defaults
        self.index_n_heads = attn_config.indexer_head_num
        self.index_head_dim = attn_config.indexer_head_dim
        self.index_topk = attn_config.indexer_topk

        self.rope_head_dim = attn_config.rope_head_dim
        self.q_lora_rank = attn_config.q_lora_rank
        self.block_size = 128
        self.scale_fmt = "ue8m0"  # FP8 quantization format
        self.softmax_scale = self.index_head_dim**-0.5
        self.weights_scale = self.index_n_heads**-0.5
        self.max_model_len = 111 * 1000
        self.num_blocks = self.max_model_len * 3
        self.blocksize = attn_config.tokens_per_block
        self.indexer_size = self.index_head_dim / 2 + self.index_head_dim / 128 * 2

        # Create linear layers
        # wq_b: projects q_lora to index_n_heads * index_head_dim
        self.wq_b = LinearFactory.create_linear_from_weights(
            weights,
            W.mla_indexer_qb_w,
            W.mla_indexer_qb_s,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )

        # wk: projects hidden_states to index_head_dim
        self.wk = LinearFactory.create_linear_from_weights(
            weights,
            W.mla_indexer_k_w,
            W.mla_indexer_k_s,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )

        # k_norm: LayerNorm for keys
        self.k_norm = LayerNorm(
            weights[W.mla_indexer_k_norm_w],
            weights[W.mla_indexer_k_norm_b],
            eps=layernorm_eps,
        )

        # weights_proj: projects hidden_states to index_n_heads (for attention weights)
        self.weights_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.mla_indexer_weights_proj_w,
            None,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )

        self.rotary_emb = MlaRotaryEmbeddingOp(
            head_size=attn_config.nope_head_dim,
            cos_sin_cache=weights[W.rope_cos_sin_cache],
            kv_lora_rank=attn_config.kv_lora_rank,
            rope_head_dim=attn_config.rope_head_dim,
            token_per_block=attn_config.tokens_per_block,
            is_neox_style=False,
        )

    def prepare(self, attention_inputs: PyAttentionInputs):
        """Prepare indexer parameters from attention inputs"""
        self.params = self._prepare_params(attention_inputs)

    def _prepare_params(self, attention_inputs: PyAttentionInputs):
        """
        Prepare indexer parameters from attention inputs.

        Returns a simple namespace with:
            - expanded_seq_lens: torch.Tensor - Expanded sequence lengths for each token
            - page_table_1: torch.Tensor - Page table with page size = 1
            - cu_seqlens_q: torch.Tensor - Cumulative sequence lengths for query
            - topk_indices_offset: torch.Tensor - Offset for topk indices in ragged mode
            - batch_size: int - Batch size
            - seq_lens: torch.Tensor - Sequence lengths
            - extend_seq_lens: List[int] - Extend sequence lengths per request
        """
        from types import SimpleNamespace

        # Extract basic information
        is_prefill = attention_inputs.is_prefill
        input_lengths = attention_inputs.input_lengths  # [batch_size]
        sequence_lengths = attention_inputs.sequence_lengths  # [decode_batch_size]

        if is_prefill:
            # Prefill mode
            batch_size = input_lengths.size(0)
            seq_lens = input_lengths.to(torch.int32)
            extend_seq_lens = input_lengths.cpu().tolist()

            # cu_seqlens_q: cumulative sequence lengths [0, len1, len1+len2, ...]
            cu_seqlens_q = attention_inputs.cu_seqlens.to(torch.int32)

            # expanded_seq_lens: repeat each seq_len for each token in that sequence
            # For example: if seq_lens = [3, 2], expanded = [3, 3, 3, 2, 2]
            expanded_seq_lens = torch.repeat_interleave(seq_lens, seq_lens)

            # topk_indices_offset: cumulative offset for ragged KV layout
            # cu_seqlens_q[:-1] gives the starting position of each sequence
            topk_indices_offset = torch.repeat_interleave(
                cu_seqlens_q[:-1], seq_lens
            ).to(torch.int32)

            # page_table_1: use kv_cache_block_id_device if available
            # In prefill mode, this is typically used for paged attention
            if attention_inputs.kv_cache_block_id_device is not None:
                page_table_1 = attention_inputs.kv_cache_block_id_device.to(torch.int32)
            else:
                # Fallback: create identity mapping
                max_seq_len = seq_lens.max().item()
                page_table_1 = (
                    torch.arange(
                        max_seq_len, dtype=torch.int32, device=input_lengths.device
                    )
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
        else:
            # Decode mode
            batch_size = sequence_lengths.size(0)
            seq_lens = sequence_lengths.to(torch.int32)
            extend_seq_lens = [1] * batch_size

            # In decode mode, each request generates 1 token
            cu_seqlens_q = torch.arange(
                0, batch_size + 1, dtype=torch.int32, device=sequence_lengths.device
            )

            # expanded_seq_lens: in decode mode, it's same as seq_lens
            expanded_seq_lens = seq_lens

            # topk_indices_offset: not used in decode mode, but set for compatibility
            topk_indices_offset = cu_seqlens_q[:-1]

            # page_table_1: kv cache block indices
            if attention_inputs.kv_cache_block_id_device is not None:
                page_table_1 = attention_inputs.kv_cache_block_id_device.to(torch.int32)
            else:
                max_seq_len = seq_lens.max().item()
                page_table_1 = (
                    torch.arange(
                        max_seq_len, dtype=torch.int32, device=sequence_lengths.device
                    )
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )

        return SimpleNamespace(
            expanded_seq_lens=expanded_seq_lens,
            page_table_1=page_table_1,
            cu_seqlens_q=cu_seqlens_q,
            topk_indices_offset=topk_indices_offset,
            batch_size=batch_size,
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
        )

    @torch.compile(dynamic=True)
    def _get_logits_head_gate(self, x: torch.Tensor, q_scale: torch.Tensor):
        weights, _ = self.weights_proj(x.float())
        weights = weights * self.weights_scale
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
        return weights

    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
        ks: Optional[torch.Tensor] = None,
        is_paged: bool = False,
        is_ragged: bool = False,
        use_nas_fuse_topk: bool = True,
    ) -> torch.Tensor:
        from rtp_llm.models_py.kernels.cuda.fast_topk import (
            fast_topk_transform_fused,
            fast_topk_transform_ragged_fused,
            fast_topk_v2,
        )

        if not use_nas_fuse_topk:
            return fast_topk_v2(logits, self.params.expanded_seq_lens, topk)
        elif is_paged:
            # NOTE(dark): if fused, we return a transformed page table directly
            return fast_topk_transform_fused(
                score=logits,
                lengths=self.params.expanded_seq_lens,  # expanded_seq_lens
                page_table_size_1=self.params.page_table_1,  # page_indices
                cu_seqlens_q=self.params.cu_seqlens_q,  # bs + 1
                topk=topk,
                row_starts=None,
            )
        elif is_ragged:
            return fast_topk_transform_ragged_fused(
                score=logits,
                lengths=self.params.expanded_seq_lens,
                topk_indices_offset=self.params.topk_indices_offset,
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
    ) -> torch.Tensor:

        weights = weights.squeeze(2)
        kv_cache_fp8 = kv_cache.kv_cache_base[..., -self.indexer_size :]

        block_kv = 1
        num_heads_kv = 1
        head_dim_with_sf = 132
        kv_cache_fp8 = kv_cache_fp8.view(
            kv_cache_fp8.shape[0], block_kv, num_heads_kv, head_dim_with_sf
        )
        # len of k
        seqlens_32 = self.params.seq_lens.to(torch.int32)
        max_block_len = (
            (seqlens_32.max().item() + self.blocksize - 1)
            // self.blocksize
            * self.blocksize
        )
        block_tables = torch.zeros(
            (self.params.batch_size, max_block_len),
            device=q_fp8.device,
            dtype=torch.int32,
        )
        schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
            seqlens_32, self.blocksize, deep_gemm.get_num_sms()
        )
        max_seq_len = block_tables.shape[1] * self.blocksize

        logits = deep_gemm.fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8,
            weights,
            seqlens_32,
            block_tables,
            schedule_metadata,
            max_seq_len,
            clean_logits=False,
        )
        # NOTE(dark): logits should be cleaned in topk_transform
        topk_result = self.topk_transform(logits, self.index_topk, is_paged=True)
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

        # 收集所有 batch 的 ks 和 ke
        ks_list = []
        ke_list = []

        q_offset = 0  # query 的累积偏移
        k_offset = 0  # KV cache 的累积偏移

        for i in range(self.params.batch_size):
            seq_len = self.params.seq_lens[i]  # KV cache 总长度
            extend_seq_len = self.params.extend_seq_lens[i]  # 新增 token 数

            # 当前 batch 的所有新 token 起始位置相同
            ks = torch.full(
                (extend_seq_len,), k_offset, dtype=torch.int32, device=q_fp8.device
            )

            # 每个新 token 能看到的 KV 长度递增
            seq_lens_expanded = torch.arange(
                seq_len - extend_seq_len + 1,
                seq_len + 1,
                dtype=torch.int32,
                device=q_fp8.device,
            )

            # 结束位置 = 起始位置 + 可见长度
            ke = ks + seq_lens_expanded

            ks_list.append(ks)
            ke_list.append(ke)

            # 更新偏移量
            q_offset += extend_seq_len
            k_offset += seq_len

        # 拼接所有 batch
        ks = torch.cat(ks_list, dim=0)
        ke = torch.cat(ke_list, dim=0)

        logits = deep_gemm.fp8_mqa_logits(
            q_fp8,
            kv_fp8,
            weights,
            ks,
            ke,
            clean_logits=False,
        )
        topk_result = self.topk_transform(
            logits, self.index_topk, ks=ks, is_ragged=True
        )

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

        # TODO: write k_fp8 and k_scale to kv cache
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
        attention_inputs: PyAttentionInputs,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        positions = None  # TODO: get positions from attention_inputs
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
            topk_result = self._get_topk_paged(q_fp8, weights, kv_cache)
        else:
            topk_result = self._get_topk_ragged(q_fp8, weights, k_fp8, k_scale)

        return topk_result
