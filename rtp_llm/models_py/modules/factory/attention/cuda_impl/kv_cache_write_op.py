"""KV Cache Write Operation for paged KV cache."""

from typing import Any, Optional, Tuple

import flashinfer.page as page
import torch

from rtp_llm.ops.compute_ops import KVCache


class KVCacheWriteOp:
    """Operator for writing key-value pairs to paged KV cache."""

    def __init__(
        self,
        num_kv_heads: int,
        head_size: int,
        token_per_block: int,
    ) -> None:
        """
        Initialize KV Cache Write operator.

        Args:
            num_kv_heads: Number of key-value heads
            head_size: Dimension of each attention head
            token_per_block: Number of tokens per KV cache block (page size)
        """
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.token_per_block = token_per_block
        self.params = None

    def set_params(self, params: Any):
        """Set the params object to be used by this op."""
        self.params = params

    def forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[KVCache],
    ) -> None:
        """
        Write key and value tensors to paged KV cache.

        Args:
            key: Key tensor [total_tokens, num_kv_heads, head_dim]
            value: Value tensor [total_tokens, num_kv_heads, head_dim]
            kv_cache: KV cache [num_pages, 2, num_kv_heads, page_size, head_dim] (HND layout)
        """
        if kv_cache is not None:
            # For real execution - use provided KV cache
            # KV cache has shape [num_pages, 2, num_kv_heads, page_size, head_dim] (HND layout)
            # In hybrid cache mode (reuse_cache with multiple layer groups), the per-layer tensor
            # may arrive as a raw 2D buffer [num_pages, kv_block_stride_elems]. Reshape it to 5D
            # [num_pages, 2, num_kv_heads, page_size, head_dim].
            kv_cache_base = kv_cache.kv_cache_base
            if kv_cache_base.dim() == 2:
                block_num = kv_cache_base.shape[0]
                expected_elems = (
                    2 * self.num_kv_heads * self.token_per_block * self.head_size
                )
                kv_cache_base = kv_cache_base[:, :expected_elems].reshape(
                    block_num,
                    2,
                    self.num_kv_heads,
                    self.token_per_block,
                    self.head_size,
                )
            # Split into K and V caches
            k_cache = kv_cache_base[
                :, 0, :, :, :
            ]  # [num_pages, num_kv_heads, page_size, head_dim]
            v_cache = kv_cache_base[
                :, 1, :, :, :
            ]  # [num_pages, num_kv_heads, page_size, head_dim]

            # Append K and V to paged cache using HND layout
            page.append_paged_kv_cache(  # type: ignore
                key,  # append_key: [total_tokens, num_kv_heads, head_dim]
                value,  # append_value: [total_tokens, num_kv_heads, head_dim]
                self.params.batch_indice_d,
                self.params.positions_d,
                (k_cache, v_cache),  # paged_kv_cache: tuple of K and V caches
                self.params.page_indice_d,
                self.params.decode_page_indptr_d,
                self.params.paged_kv_last_page_len_d,
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

    def _prepare_warmup_cache_indices(
        self, num_tokens: int, device: torch.device
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int
    ]:
        """
        Prepare dummy cache indices for warmup/JIT compilation.

        Args:
            num_tokens: Number of tokens to process
            device: Device to create tensors on

        Returns:
            Tuple of (batch_indices, positions, kv_page_indices, kv_page_indptr, kv_last_page_len, max_num_pages)
        """
        # Assume 1 batch, sequential tokens
        batch_indices = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        positions = torch.arange(num_tokens, dtype=torch.int32, device=device)

        # Calculate required pages
        max_num_pages = (num_tokens + self.token_per_block - 1) // self.token_per_block

        # Page indices: [0, 0, 0, ..., 1, 1, 1, ..., 2, 2, 2, ...]
        kv_page_indices = (
            torch.arange(num_tokens, dtype=torch.int32, device=device)
            // self.token_per_block
        )

        # Page indptr: [0, max_num_pages] for single batch
        kv_page_indptr = torch.tensor(
            [0, max_num_pages], dtype=torch.int32, device=device
        )

        # Last page length
        last_page_len = num_tokens % self.token_per_block
        if last_page_len == 0:
            last_page_len = self.token_per_block
        kv_last_page_len = torch.tensor(
            [last_page_len], dtype=torch.int32, device=device
        )

        return (
            batch_indices,
            positions,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_len,
            max_num_pages,
        )
