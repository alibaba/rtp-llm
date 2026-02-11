from typing import Any, Optional

import flashinfer.page as page
import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.base_rotary_embedding_op import (
    BaseRotaryEmbeddingOp,
)
from rtp_llm.ops.compute_ops import KVCache


class MlaRotaryEmbeddingOp(BaseRotaryEmbeddingOp):
    """Rotary positional embedding for Multi-Latent Attention (MLA)."""

    def __init__(
        self,
        head_size: int,
        cos_sin_cache: torch.Tensor | None,
        kv_lora_rank: int,
        rope_head_dim: int,
        token_per_block: int,
        is_neox_style: bool,
    ) -> None:
        super().__init__(head_size, cos_sin_cache, token_per_block, is_neox_style)
        self.kv_lora_rank = kv_lora_rank
        self.rope_head_dim = rope_head_dim

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        append_ckv_t: torch.Tensor,
        rope_params: Any,
        kv_cache: Optional[KVCache] = None,
    ):
        """
        Apply RoPE and append KV cache for MLA.

        Args:
            query: Query tensor
            key: Key tensor for RoPE (will be unsqueezed)
            append_ckv_t: Compressed KV tensor to append
            rope_params: RoPE parameters containing batch indices, positions, etc.
            kv_cache: MLA KV cache with compressed layout
        """
        # Apply RoPE to Q and K (MLA requires key.unsqueeze(1))
        self._apply_rope(query, key.unsqueeze(1), rope_params)

        # Append compressed KV to cache
        if kv_cache is not None:
            # Split MLA cache into compressed K and position-encoded V
            k_cache, v_cache = torch.split(
                kv_cache.kv_cache_base, [self.kv_lora_rank, self.rope_head_dim], dim=-1
            )

            page.append_paged_mla_kv_cache(
                append_ckv_t,
                key,
                rope_params.batch_indice_d,
                rope_params.positions_d,
                k_cache,
                v_cache,
                rope_params.page_indice_d,
                rope_params.decode_page_indptr_d,
                rope_params.paged_kv_last_page_len_d,
            )
        else:
            # For warmup/JIT compilation - create dummy MLA KV cache
            (
                batch_indices,
                positions,
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_len,
                max_num_pages,
            ) = self._prepare_warmup_cache_indices(
                append_ckv_t.size(0), append_ckv_t.device
            )

            # Create MLA cache: [num_pages, page_size, kv_lora_rank + rope_head_dim]
            cache = torch.empty(
                [
                    max_num_pages,
                    self.token_per_block,
                    self.kv_lora_rank + self.rope_head_dim,
                ],
                dtype=append_ckv_t.dtype,
                device=append_ckv_t.device,
            )
            k_cache, v_cache = torch.split(
                cache, [self.kv_lora_rank, self.rope_head_dim], dim=-1
            )

            page.append_paged_mla_kv_cache(
                append_ckv_t,
                key,
                batch_indices,
                positions,
                k_cache,
                v_cache,
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_len,
            )
