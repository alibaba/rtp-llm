"""KV cache write operation for Multi-Latent Attention (MLA).

This module provides the KV cache writing operation specifically for MLA architecture,
which uses a compressed KV cache layout.
"""

from typing import Any, Optional, Tuple

import flashinfer.page as page
import torch

from rtp_llm.ops.compute_ops import LayerKVCache


class MlaKVCacheWriteOp:
    """Write compressed KV cache for Multi-Latent Attention."""

    def __init__(
        self,
        kv_lora_rank: int,
        rope_head_dim: int,
        token_per_block: int,
    ) -> None:
        """
        Args:
            kv_lora_rank: Rank of KV LoRA compression
            rope_head_dim: Dimension of RoPE-applied head
            token_per_block: Number of tokens per KV cache block (page size)
        """
        self.kv_lora_rank = kv_lora_rank
        self.rope_head_dim = rope_head_dim
        self.token_per_block = token_per_block
        self.params: Any = None

    def set_params(self, params: Any) -> None:
        """Set parameters for KV cache writing.

        Args:
            params: FlashInferMlaAttnParams containing batch indices, positions, etc.
        """
        self.params = params

    def forward(
        self,
        append_ckv_t: torch.Tensor,
        key_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
    ) -> None:
        """Write compressed KV and position-encoded key to MLA cache.

        Args:
            append_ckv_t: Compressed KV tensor to append [num_tokens, kv_lora_rank]
            key_pe: Position-encoded key tensor [num_tokens, rope_head_dim]
            kv_cache: MLA KV cache with compressed layout
        """
        if kv_cache is not None:
            # Split MLA cache into compressed K and position-encoded V
            k_cache, v_cache = torch.split(
                kv_cache.kv_cache_base, [self.kv_lora_rank, self.rope_head_dim], dim=-1
            )

            page.append_paged_mla_kv_cache(
                append_ckv_t,
                key_pe,
                self.params.batch_indice_d,
                self.params.positions_d,
                k_cache,
                v_cache,
                self.params.page_indice_d,
                self.params.decode_page_indptr_d,
                self.params.paged_kv_last_page_len_d,
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
                key_pe,
                batch_indices,
                positions,
                k_cache,
                v_cache,
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_len,
            )

    def _prepare_warmup_cache_indices(
        self,
        num_tokens: int,
        device: torch.device,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int
    ]:
        """Prepare dummy cache indices for warmup/JIT compilation.

        Args:
            num_tokens: Number of tokens to process
            device: Device to create tensors on

        Returns:
            Tuple of (batch_indices, positions, kv_page_indices, kv_page_indptr,
                     kv_last_page_len, max_num_pages)
        """
        num_pages = (num_tokens + self.token_per_block - 1) // self.token_per_block
        batch_indices = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        positions = torch.arange(num_tokens, dtype=torch.int32, device=device)
        kv_page_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
        kv_page_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
        kv_last_page_len = torch.tensor(
            [num_tokens % self.token_per_block or self.token_per_block],
            dtype=torch.int32,
            device=device,
        )
        return (
            batch_indices,
            positions,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_len,
            num_pages,
        )
