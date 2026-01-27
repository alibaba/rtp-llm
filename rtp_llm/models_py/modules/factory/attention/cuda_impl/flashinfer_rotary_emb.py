from typing import Any, Optional

import flashinfer.page as page
import torch

from rtp_llm.models_py.modules.factory.attention.common import BaseRotaryEmbeddingOp
from rtp_llm.ops.compute_ops import KVCache


class MhaRotaryEmbeddingOp(BaseRotaryEmbeddingOp):
    """Rotary positional embedding for Multi-Head Attention (MHA)."""

    def __init__(
        self,
        head_size: int,
        cos_sin_cache: torch.Tensor | None,
        token_per_block: int,
        is_neox_style: bool,
        num_kv_heads: int = 1,
    ) -> None:
        super().__init__(head_size, cos_sin_cache, token_per_block, is_neox_style)
        self.num_kv_heads = num_kv_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        rope_params: Any,
        kv_cache: Optional[KVCache] = None,
    ):
        """
        Apply RoPE and append KV cache for MHA.

        Args:
            query: Query tensor [total_tokens, num_heads, head_dim]
            key: Key tensor [total_tokens, num_kv_heads, head_dim]
            value: Value tensor [total_tokens, num_kv_heads, head_dim]
            rope_params: RoPE parameters containing batch indices, positions, etc.
            kv_cache: KV cache [num_pages, 2, page_size, num_kv_heads, head_dim]
        """
        # Apply RoPE to Q and K
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
