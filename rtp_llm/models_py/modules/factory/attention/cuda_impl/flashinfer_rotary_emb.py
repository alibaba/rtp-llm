from typing import Any

import torch
from flashinfer import rope

from rtp_llm.models_py.modules.factory.attention.cuda_impl.base_rotary_embedding_op import (
    BaseRotaryEmbeddingOp,
)
from rtp_llm.ops import AttentionConfigs, RopeConfig, RopeStyle, get_rope_cache_once
from rtp_llm.ops.compute_ops import PyAttentionInputs


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
        super().__init__(
            attn_config.size_per_head,
            cos_sin_cache,
            attn_config.kernel_tokens_per_block,
            is_neox_style=False,
            rope_config=attn_config.rope_config,
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
        self._apply_rope(query, key, self.params)

        return query, key, value


class TextMropeEmbeddingOp(MhaRotaryEmbeddingOp):
    """Apply Base RoPE to MTP text tokens whose T/H/W positions are equal."""

    def __init__(
        self, attn_config: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> None:
        rope_config = RopeConfig()
        rope_config.style = RopeStyle.Base
        rope_config.dim = attn_config.rope_config.dim
        rope_config.base = attn_config.rope_config.base
        rope_config.scale = attn_config.rope_config.scale
        max_position_embeddings = (
            attn_config.max_seq_len + attn_config.gen_num_per_cycle + 1
        )
        rope_cache = get_rope_cache_once(
            rope_config,
            max_position_embeddings,
            is_cuda=True,
            interleave=False,
        )
        super().__init__(attn_config, cos_sin_cache=rope_cache.data)
        self.position_ids = self._text_position_ids(attn_inputs).contiguous()

    @staticmethod
    def _text_position_ids(attn_inputs: PyAttentionInputs) -> torch.Tensor:
        return attn_inputs.combo_position_ids.view(-1, 3)[:, 0]

    def _apply_rope(
        self, query: torch.Tensor, key: torch.Tensor, _rope_params: Any
    ) -> None:
        assert self.cos_sin_cache is not None
        rope._apply_rope_pos_ids_cos_sin_cache(  # type: ignore
            q=query,
            k=key,
            q_rope=query,
            k_rope=key,
            cos_sin_cache=self.cos_sin_cache,
            pos_ids=self.position_ids.narrow(0, 0, query.shape[0]),
            interleave=self.is_neox_style,
        )

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs) -> None:
        self.position_ids.copy_(self._text_position_ids(attn_inputs), non_blocking=True)


class InterleavedMropeEmbeddingOp(MhaRotaryEmbeddingOp):
    """Apply interleaved MRoPE (Qwen3.5) with arbitrary per-axis T/H/W positions.

    Unlike TextMropeEmbeddingOp (which requires all three axes to share one
    position id), this op rotates each rotary pair with the position of its own
    axis, matching the fused dynamic FP8 decode kernel's MRoPE path exactly.
    Used by FP8 KV cache mode 2 prefill, where RoPE runs separately from the
    per-token-scale KV write.
    """

    def __init__(
        self, attn_config: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(attn_config)
        rope_config = attn_config.rope_config
        device = attn_inputs.combo_position_ids.device
        rotary_pairs = rope_config.dim // 2
        # Section assignment mirrors the CUDA kernel: pairs round-robin across
        # the three T/H/W sections, bounded by the per-axis section counts.
        axes = torch.zeros(rotary_pairs, dtype=torch.long)
        axes[1 : 3 * rope_config.mrope_dim2 : 3] = 1
        axes[2 : 3 * rope_config.mrope_dim3 : 3] = 2
        self.mrope_axes = axes.to(device)
        self.inv_freq = rope_config.base ** (
            -2.0
            * torch.arange(rotary_pairs, dtype=torch.float32, device=device)
            / rope_config.dim
        )
        self.rope_dim = rope_config.dim
        self.rotary_pairs = rotary_pairs
        self.position_ids = attn_inputs.combo_position_ids.contiguous()

    def _apply_rope(
        self, query: torch.Tensor, key: torch.Tensor, _rope_params: Any
    ) -> None:
        num_tokens = query.shape[0]
        positions = (
            self.position_ids.reshape(-1, 3)[:num_tokens, self.mrope_axes].to(torch.float32)
        )
        angles = positions * self.inv_freq.unsqueeze(0)
        cos = angles.cos().unsqueeze(1)
        sin = angles.sin().unsqueeze(1)
        for tensor in (query, key):
            values = tensor.float()
            low = values[..., : self.rotary_pairs]
            high = values[..., self.rotary_pairs : self.rope_dim]
            tail = values[..., self.rope_dim :]
            rotated = torch.cat(
                [low * cos - high * sin, high * cos + low * sin, tail], dim=-1
            )
            tensor.copy_(rotated.to(tensor.dtype))
