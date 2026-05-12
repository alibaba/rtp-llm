import torch

from rtp_llm.models_py.modules.factory.attention.ascend_impl.ascend_rope import apply_rope_pos_ids_nhd
from rtp_llm.ops import AttentionConfigs


class AscendRotaryEmbeddingOp:
    """Ascend RoPE using pure PyTorch implementation (replaces flashinfer.rope)."""

    def __init__(self, attn_config: AttentionConfigs, cos_sin_cache: torch.Tensor | None = None):
        self.head_size = attn_config.size_per_head
        self.is_neox_style = False
        self.token_per_block = attn_config.kernel_tokens_per_block
        self.rope_config = attn_config.rope_config
        self.cos_sin_cache = cos_sin_cache
        self.num_heads = attn_config.head_num
        self.num_kv_heads = attn_config.kv_head_num
        self.params = None

    def set_params(self, params):
        self.params = params

    def forward(self, qkv):
        qkv = qkv.reshape(qkv.shape[0], -1)
        q, k, v = torch.split(qkv, [
            self.head_size * self.num_heads,
            self.head_size * self.num_kv_heads,
            self.head_size * self.num_kv_heads,
        ], dim=-1)
        query = q.reshape(q.shape[0], self.num_heads, self.head_size)
        key = k.reshape(k.shape[0], self.num_kv_heads, self.head_size)
        value = v.reshape(v.shape[0], self.num_kv_heads, self.head_size)

        self._apply_rope(query, key, self.params)

        return query, key, value

    def _apply_rope(self, query, key, rope_params):
        if self.cos_sin_cache is not None:
            apply_rope_pos_ids_nhd(
                query, key,
                self.cos_sin_cache,
                rope_params.positions_d,
                is_neox_style=self.is_neox_style,
            )
        else:
            raise RuntimeError("AscendRotaryEmbeddingOp requires cos_sin_cache")

    def _prepare_warmup_cache_indices(self, num_tokens, device):
        batch_indices = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        positions = torch.arange(num_tokens, dtype=torch.int32, device=device)

        max_num_pages = (num_tokens + self.token_per_block - 1) // self.token_per_block
        kv_page_indices = positions // self.token_per_block
        kv_page_indptr = torch.tensor([0, max_num_pages], dtype=torch.int32, device=device)

        last_page_len = num_tokens % self.token_per_block
        if last_page_len == 0:
            last_page_len = self.token_per_block
        kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device=device)

        return batch_indices, positions, kv_page_indices, kv_page_indptr, kv_last_page_len, max_num_pages
