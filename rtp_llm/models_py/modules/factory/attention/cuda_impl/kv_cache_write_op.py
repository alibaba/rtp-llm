"""KV Cache Write Operation for paged KV cache."""

from typing import Any, Optional, Tuple

import flashinfer.page as page
import torch

from rtp_llm.ops.compute_ops import LayerKVCache, rtp_llm_ops


class KVCacheWriteOp:
    """Operator for writing key-value pairs to paged KV cache."""

    def __init__(
        self,
        num_kv_heads: int,
        head_size: int,
        physical_page_size: Optional[int] = None,
        kernel_page_size: Optional[int] = None,
        dynamic_mode: bool = False,
        token_per_block: Optional[int] = None,
    ) -> None:
        """Initialize the KV cache writer for physical/kernel page geometry.

        ``token_per_block`` remains as a compatibility alias for callers that
        predate separate physical and kernel page sizes.
        """
        if physical_page_size is None:
            physical_page_size = token_per_block
        elif token_per_block is not None and token_per_block != physical_page_size:
            raise ValueError("token_per_block must match physical_page_size")
        if physical_page_size is None or physical_page_size <= 0:
            raise ValueError("physical_page_size must be positive")
        if kernel_page_size is None:
            kernel_page_size = physical_page_size
        if kernel_page_size <= 0 or physical_page_size % kernel_page_size != 0:
            raise ValueError(
                "physical_page_size must be divisible by kernel_page_size, got "
                f"{physical_page_size} and {kernel_page_size}"
            )

        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.physical_page_size = physical_page_size
        self.kernel_page_size = kernel_page_size
        self.subdivision = physical_page_size // kernel_page_size
        self.dynamic_mode = dynamic_mode
        # Keep the old attribute for warmup and compatibility users.
        self.token_per_block = kernel_page_size
        self.params = None

    def set_params(self, params: Any):
        """Set the params object to be used by this op."""
        self.params = params

    def forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
    ) -> None:
        """
        Write key and value tensors to paged KV cache.

        Args:
            key: Key tensor [total_tokens, num_kv_heads, head_dim]
            value: Value tensor [total_tokens, num_kv_heads, head_dim]
            kv_cache: KV cache [num_pages, 2, num_kv_heads, page_size, head_dim] (HND layout)
        """
        if kv_cache is not None:
            # FlashInfer requires batch_indices/positions size == nnz. Device
            # planner buffers can be oversized, so narrow without a host sync.
            nnz = key.size(0)
            batch_indices = self.params.batch_indice_d.narrow(0, 0, nnz)
            positions = self.params.positions_d.narrow(0, 0, nnz)

            if self.dynamic_mode:
                kv_scales = getattr(kv_cache, "kv_scale_base", None)
                if kv_scales is None or kv_scales.numel() == 0:
                    raise ValueError(
                        "FP8 KV cache mode 2 requires a non-empty kv_scale_base"
                    )

                page_slots = self.params.decode_page_indptr_d[batch_indices.long()]
                page_slots = page_slots + torch.div(
                    positions, self.kernel_page_size, rounding_mode="floor"
                )
                target_kernel_pages = self.params.page_indice_d[page_slots.long()]
                target_physical_pages = torch.div(
                    target_kernel_pages,
                    self.subdivision,
                    rounding_mode="floor",
                ).contiguous()
                physical_token_offsets = (
                    torch.remainder(target_kernel_pages, self.subdivision)
                    * self.kernel_page_size
                    + torch.remainder(positions, self.kernel_page_size)
                ).contiguous()
                rtp_llm_ops.quantize_and_write_fp8_kv_cache(
                    key.contiguous(),
                    value.contiguous(),
                    kv_cache.kv_cache_base,
                    kv_scales,
                    target_physical_pages,
                    physical_token_offsets,
                    self.physical_page_size,
                    self.kernel_page_size,
                    self.subdivision,
                )
                return

            # For legacy/base execution, cache dtype must already match K/V.
            k_cache = kv_cache.kv_cache_base[:, 0, :, :, :]
            v_cache = kv_cache.kv_cache_base[:, 1, :, :, :]
            if key.dtype != k_cache.dtype:
                raise ValueError(
                    f"key dtype {key.dtype} must match K cache dtype {k_cache.dtype}"
                )
            if value.dtype != v_cache.dtype:
                raise ValueError(
                    f"value dtype {value.dtype} must match V cache dtype {v_cache.dtype}"
                )

            # Append K and V to paged cache using HND layout.
            page.append_paged_kv_cache(  # type: ignore
                key,
                value,
                batch_indices,
                positions,
                (k_cache, v_cache),
                self.params.page_indice_d,
                self.params.decode_page_indptr_d,
                self.params.paged_kv_last_page_len_d,
                "HND",
            )
        elif not self.dynamic_mode:
            # For legacy/base warmup/JIT compilation - create dummy KV cache.
            # Dynamic mode intentionally performs no write without a real cache.
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
