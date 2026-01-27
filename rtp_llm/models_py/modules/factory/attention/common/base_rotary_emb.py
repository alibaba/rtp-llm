"""Base class for Rotary Positional Embedding operations."""

import math
from abc import ABC, abstractmethod
from typing import Any, Tuple

import flashinfer.rope as rope
import torch
from flashinfer import get_batch_indices_positions, get_seq_lens


class BaseRotaryEmbeddingOp(ABC):
    """Base class for rotary positional embedding with FlashInfer.

    This class provides common functionality for both MHA and MLA implementations,
    including RoPE application and warmup cache preparation.
    """

    def __init__(
        self,
        head_size: int,
        cos_sin_cache: torch.Tensor | None,
        token_per_block: int,
        is_neox_style: bool,
    ) -> None:
        """
        Args:
            head_size: Dimension of each attention head
            cos_sin_cache: Precomputed cos/sin cache for RoPE [max_seq_len, rope_dim]
            token_per_block: Number of tokens per KV cache block (page size)
            is_neox_style: Whether to use GPT-NeoX style RoPE (interleave) or LLaMA style (non-interleave)
        """
        if cos_sin_cache is None:
            raise Exception(f"RotaryEmbedding need cos_sin_cache but got none")
        super().__init__()
        self.head_size = head_size
        self.is_neox_style = is_neox_style
        self.cos_sin_cache = cos_sin_cache
        self.token_per_block = token_per_block

    def _apply_rope(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        rope_params: Any,
    ) -> None:
        """Apply RoPE to query and key tensors in-place.

        Args:
            query: Query tensor to apply RoPE to
            key: Key tensor to apply RoPE to
            rope_params: Parameters containing position IDs
        """
        rope._apply_rope_pos_ids_cos_sin_cache(  # type: ignore
            q=query,
            k=key,
            q_rope=query,
            k_rope=key,
            cos_sin_cache=self.cos_sin_cache,
            pos_ids=rope_params.positions_d,
            interleave=self.is_neox_style,
        )

    def _prepare_warmup_cache_indices(
        self,
        num_tokens: int,
        device: torch.device,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int
    ]:
        """Prepare indices and metadata for warmup KV cache creation.

        This creates dummy batch indices, positions, page indices, etc. required
        for JIT compilation warmup when no real KV cache is provided.

        Args:
            num_tokens: Number of tokens to process
            device: Device to create tensors on

        Returns:
            Tuple of (batch_indices, positions, kv_page_indices, kv_page_indptr,
                     kv_last_page_len, max_num_pages)
        """
        kv_len = [num_tokens]
        num_pages_per_req = torch.tensor(
            [math.ceil(length / self.token_per_block) for length in kv_len],
            dtype=torch.int32,
            device=device,
        )
        kv_append_length = torch.tensor(kv_len, dtype=torch.int32, device=device)
        kv_append_indptr = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.cumsum(kv_append_length, dim=0),
            ]
        )

        max_num_pages = int(sum(num_pages_per_req))
        kv_page_indptr = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.cumsum(num_pages_per_req, dim=0),
            ]
        )
        kv_page_indices = torch.arange(max_num_pages, dtype=torch.int32, device=device)

        kv_last_page_len = torch.tensor(
            [
                (
                    length % self.token_per_block
                    if length % self.token_per_block != 0
                    else self.token_per_block
                )
                for length in kv_len
            ],
            dtype=torch.int32,
            device=device,
        )

        batch_indices, positions = get_batch_indices_positions(
            kv_append_indptr,
            get_seq_lens(kv_page_indptr, kv_last_page_len, self.token_per_block),
            num_tokens,
        )

        return (
            batch_indices,
            positions,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_len,
            max_num_pages,
        )

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> None:
        """Forward pass - must be implemented by subclasses.

        Subclasses should:
        1. Call self._apply_rope() to apply RoPE to Q and K
        2. Implement their specific KV cache append logic
        3. Use self._prepare_warmup_cache_indices() for warmup cache preparation
        """
        pass
