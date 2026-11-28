from typing import Any

import torch

from rtp_llm.models_py.kernels.cuda.scaled_rope import scaled_rope_qk_inplace
from rtp_llm.models_py.modules.factory.attention.cuda_impl.base_rotary_embedding_op import (
    BaseRotaryEmbeddingOp,
)
from rtp_llm.ops import AttentionConfigs, RopeStyle


class MhaRotaryEmbeddingOp(BaseRotaryEmbeddingOp):
    """Rotary positional embedding for Multi-Head Attention (MHA)."""

    def __init__(
        self,
        attn_config: AttentionConfigs,
        cos_sin_cache: torch.Tensor | None = None,
    ) -> None:
        """
        Initialize MHA Rotary Embedding operator.

        Note: This op only applies RoPE. For KV cache writing, use KVCacheWriteOp separately.

        Args:
            attn_config: Attention configuration containing all necessary parameters
            cos_sin_cache: Precomputed cos/sin cache for RoPE, shape [max_seq_len, head_dim].
                          Layout: [cos_0, cos_1, ..., cos_{d/2-1}, sin_0, sin_1, ..., sin_{d/2-1}]
                          where d = head_dim. First half stores cosine values, second half stores sine values.
                          dtype should be torch.float32 for numerical stability.
                          If None, will auto-generate using attn_config.rope_config.
        """
        # Base/YaRN caches cannot represent Llama3 or request-dependent NTK.
        # Keep positions and sequence lengths device-resident, including replay.
        self.scaled_rope_config = (
            attn_config.rope_config
            if attn_config.rope_config.style in (RopeStyle.Llama3, RopeStyle.DynamicNTK)
            else None
        )
        super().__init__(
            attn_config.size_per_head,
            cos_sin_cache,
            attn_config.kernel_tokens_per_block,
            is_neox_style=False,
            rope_config=(
                attn_config.rope_config if self.scaled_rope_config is None else None
            ),
            max_position_embeddings=attn_config.max_seq_len
            + attn_config.gen_num_per_cycle
            + 1,
        )
        self.num_heads = attn_config.head_num
        self.num_kv_heads = attn_config.kv_head_num
        self.seq_size_per_block = attn_config.kernel_tokens_per_block
        self.params = None

    def set_params(self, params: Any):
        """Set the params object to be filled by this op."""
        self.params = params

    def forward(  # type: ignore
        self,
        qkv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to QKV tensor for MHA.

        Note: This op only applies RoPE. KV cache writing should be done by KVCacheWriteOp separately.

        Args:
            qkv: QKV tensor [total_tokens, hidden_size] where hidden_size = (num_heads + 2*num_kv_heads) * head_dim

        Returns:
            Tuple of (query, key, value) tensors after RoPE:
                - query: [total_tokens, num_heads, head_dim]
                - key: [total_tokens, num_kv_heads, head_dim]
                - value: [total_tokens, num_kv_heads, head_dim]
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

        # Apply RoPE to Q and K
        if self.scaled_rope_config is None:
            self._apply_rope(query, key, self.params)
        else:
            config = self.scaled_rope_config
            scaled_rope_qk_inplace(
                query,
                key,
                self.params.positions_d,
                self.params.batch_indice_d,
                self.params.kvlen_d,
                rotary_dim=config.dim,
                base=config.base,
                scale=config.scale,
                original_max_position=config.max_pos,
                llama3=config.style == RopeStyle.Llama3,
                low_freq_factor=config.factor1,
                high_freq_factor=config.factor2,
            )

        return query, key, value
