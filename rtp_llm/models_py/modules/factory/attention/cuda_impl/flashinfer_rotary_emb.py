from typing import Any, Optional

import flashinfer.page as page
import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.base_rotary_embedding_op import (
    BaseRotaryEmbeddingOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla import (
    check_attention_inputs,
)
from rtp_llm.ops import AttentionConfigs, RopeStyle
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs, fill_mla_params


class MhaRotaryEmbeddingOp(BaseRotaryEmbeddingOp):
    """Rotary positional embedding for Multi-Head Attention (MHA)."""

    def __init__(
        self,
        head_size: int,
        cos_sin_cache: torch.Tensor | None,
        token_per_block: int,
        attn_config: AttentionConfigs,
        num_kv_heads: int = 1,
        max_position_embeddings: int = 32768,
        return_qkv: bool = False,
    ) -> None:
        """
        Initialize MHA Rotary Embedding operator.

        Args:
            head_size: Dimension of each attention head
            cos_sin_cache: Precomputed cos/sin cache for RoPE, shape [max_seq_len, head_dim].
                          Layout: [cos_0, cos_1, ..., cos_{d/2-1}, sin_0, sin_1, ..., sin_{d/2-1}]
                          where d = head_dim. First half stores cosine values, second half stores sine values.
                          dtype should be torch.float32 for numerical stability.
                          If None, will auto-generate using attn_config.rope_config.
            token_per_block: Number of tokens per KV cache block (page size), typically 16 or 32
            attn_config: Attention configuration containing rope_config for determining interleave style
            num_kv_heads: Number of key-value heads (for GQA/MQA support)
            max_position_embeddings: Maximum position embeddings for auto-generating cache (default: 32768)
            return_qkv: If True, return concatenated qkv tensor; if False, return only query tensor
        """
        super().__init__(
            head_size,
            cos_sin_cache,
            token_per_block,
            is_neox_style=False,
            rope_config=attn_config.rope_config,
            max_position_embeddings=max_position_embeddings,
        )
        self.num_heads = attn_config.head_num
        self.num_kv_heads = num_kv_heads
        self.use_rope = attn_config.rope_config.style != RopeStyle.No
        self.seq_size_per_block = attn_config.tokens_per_block
        self.params = None
        self.return_qkv = return_qkv

    def set_params(self, params: Any):
        """Set the params object to be filled by this op."""
        self.params = params

    def prepare(self, attn_inputs: PyAttentionInputs):
        check_attention_inputs(attn_inputs)
        assert self.params is not None
        self.params.fill_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_block_id_host,
            self.seq_size_per_block,
        )
        return self.params

    def forward(  # type: ignore
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
        rope_params: Any,
    ) -> torch.Tensor:
        """
        Apply RoPE and append KV cache for MHA.

        Args:
            qkv: QKV tensor [total_tokens, hidden_size] where hidden_size = (num_heads + 2*num_kv_heads) * head_dim
            fmha_type: FMHA type (not used in this implementation)
            kv_cache: KV cache [num_pages, 2, page_size, num_kv_heads, head_dim]
            rope_params: RoPE parameters containing batch indices, positions, etc.

        Returns:
            If return_qkv=True: concatenated qkv tensor [total_tokens, hidden_size] after RoPE
            If return_qkv=False: query tensor [total_tokens, num_heads, head_dim] after RoPE
        """
        # Split QKV tensor into Q, K, V
        # qkv shape: [total_tokens, (num_heads + 2*num_kv_heads) * head_dim]
        qkv = qkv.reshape(qkv.shape[0], -1)
        q, k, v = torch.split(
            qkv,
            [
                self.head_size * self.num_heads,
                self.head_size * self.num_kv_heads,
                self.head_size * self.num_kv_heads,
            ],
            dim=-1,
        )
        # Reshape to [total_tokens, num_heads, head_dim]
        query = q.reshape(q.shape[0], self.num_heads, self.head_size)
        key = k.reshape(k.shape[0], self.num_kv_heads, self.head_size)
        value = v.reshape(v.shape[0], self.num_kv_heads, self.head_size)

        # Apply RoPE to Q and K if RoPE is enabled (rope_config.style != No)
        if self.use_rope:
            self._apply_rope(query, key, rope_params)

        # Append KV to cache
        if kv_cache is not None:
            # For MHA, KV cache has shape [num_pages, 2, num_kv_heads, page_size, head_dim] (HND layout)
            # Split into K and V caches
            k_cache = kv_cache.kv_cache_base[
                :, 0, :, :, :
            ]  # [num_pages, num_kv_heads, page_size, head_dim]
            v_cache = kv_cache.kv_cache_base[
                :, 1, :, :, :
            ]  # [num_pages, num_kv_heads, page_size, head_dim]

            # Append K and V to paged cache using HND layout
            page.append_paged_kv_cache(  # type: ignore
                key,  # append_key: [total_tokens, num_kv_heads, head_dim]
                value,  # append_value: [total_tokens, num_kv_heads, head_dim]
                rope_params.batch_indice_d,
                rope_params.positions_d,
                (k_cache, v_cache),  # paged_kv_cache: tuple of K and V caches
                rope_params.page_indice_d,
                rope_params.decode_page_indptr_d,
                rope_params.paged_kv_last_page_len_d,
                "HND",  # kv_layout: HND layout (num_pages, num_kv_heads, page_size, head_dim)
            )
        else:
            # For warmup/JIT compilation - create dummy KV cache
            (
                batch_indices,
                positions,
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_len,
                max_num_pages,
            ) = self._prepare_warmup_cache_indices(value.size(0), value.device)

            # Create MHA KV cache: [num_pages, num_kv_heads, page_size, head_dim] (HND layout)
            k_cache = torch.empty(
                (
                    max_num_pages,
                    self.num_kv_heads,
                    self.token_per_block,
                    self.head_size,
                ),
                dtype=value.dtype,
                device=value.device,
            )
            v_cache = torch.empty(
                (
                    max_num_pages,
                    self.num_kv_heads,
                    self.token_per_block,
                    self.head_size,
                ),
                dtype=value.dtype,
                device=value.device,
            )

            # Append K and V to paged cache using HND layout
            page.append_paged_kv_cache(  # type: ignore
                key,
                value,
                batch_indices,
                positions,
                (k_cache, v_cache),  # paged_kv_cache: tuple of K and V caches
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_len,
                "HND",  # kv_layout: HND layout (num_pages, num_kv_heads, page_size, head_dim)
            )

        if self.return_qkv:
            return qkv
        else:
            return query

    @staticmethod
    def support(attn_inputs: PyAttentionInputs) -> bool:
        """Check if MHA RoPE implementation supports the given inputs.

        Args:
            attn_inputs: Attention inputs to check

        Returns:
            True if supported (MHA RoPE always supports all inputs)
        """
        return True
