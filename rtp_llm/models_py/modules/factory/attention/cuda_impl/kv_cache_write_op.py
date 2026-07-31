"""KV Cache Write Operation for paged KV cache."""

import logging
from typing import Any, Optional, Tuple

import flashinfer.page as page
import torch

from rtp_llm.ops.compute_ops import LayerKVCache

logger = logging.getLogger(__name__)


class KVCacheWriteOp:
    """Operator for writing key-value pairs to paged KV cache."""

    _fp8_scale_warning_emitted = False

    @classmethod
    def _warn_fp8_implicit_scale_once(cls) -> None:
        if cls._fp8_scale_warning_emitted:
            return
        logger.warning(
            "PyFlashinfer FP8 KV cache uses implicit scale=1; values outside "
            "the finite E4M3 range are saturated. This mode can lose accuracy "
            "for models with KV outliers; disable fp8_kv_cache to fall back "
            "to the configured non-FP8 cache dtype"
        )
        cls._fp8_scale_warning_emitted = True

    def __init__(
        self,
        num_kv_heads: int,
        head_size: int,
        token_per_block: int,
        kv_cache_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Initialize KV Cache Write operator.

        Args:
            num_kv_heads: Number of key-value heads
            head_size: Dimension of each attention head
            token_per_block: Number of tokens per KV cache block (page size)
            kv_cache_dtype: Cache dtype used by warmup. ``None`` keeps the
                activation dtype for non-FP8 cache configurations.
        """
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.token_per_block = token_per_block
        self.kv_cache_dtype = kv_cache_dtype
        self.params = None
        if kv_cache_dtype == torch.float8_e4m3fn:
            self._warn_fp8_implicit_scale_once()

    @staticmethod
    def _cast_for_cache(tensor: torch.Tensor, cache_dtype: torch.dtype) -> torch.Tensor:
        if tensor.dtype == cache_dtype:
            return tensor
        if cache_dtype != torch.float8_e4m3fn:
            raise TypeError(
                "KVCacheWriteOp only converts activations for FP8 E4M3 cache; "
                f"got activation dtype {tensor.dtype} and cache dtype {cache_dtype}"
            )
        # PyFlashinfer FP8 KV uses an implicit scale of 1. Values outside the
        # finite E4M3 range are saturated before conversion, so overflow does
        # not create NaN; NaN inputs remain NaN. This conversion intentionally
        # has no per-token scale.
        limit = torch.finfo(cache_dtype).max
        return tensor.clamp(min=-limit, max=limit).to(cache_dtype)

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
            # For real execution - use provided KV cache
            # KV cache has shape [num_pages, 2, num_kv_heads, page_size, head_dim] (HND layout)
            k_cache = kv_cache.kv_cache_base[
                :, 0, :, :, :
            ]  # [num_pages, num_kv_heads, page_size, head_dim]
            v_cache = kv_cache.kv_cache_base[
                :, 1, :, :, :
            ]  # [num_pages, num_kv_heads, page_size, head_dim]

            if self.kv_cache_dtype not in (None, k_cache.dtype):
                raise RuntimeError(
                    "PyFlashinfer KV cache dtype mismatch: configured "
                    f"{self.kv_cache_dtype}, actual {k_cache.dtype}"
                )

            if k_cache.dtype == torch.float8_e4m3fn:
                self._warn_fp8_implicit_scale_once()
                # MemoryLayoutStrategy initializes every FP8 scale entry to
                # 1.0 before execution. PyFlashinfer writes direct-cast values,
                # so checking tensor contents here would add a device sync to
                # every layer and is unsafe during CUDA graph capture. Keep the
                # hot-path check structural and rely on that allocator contract.
                kv_scale = kv_cache.kv_scale_base
                if kv_scale is None or kv_scale.numel() == 0:
                    raise RuntimeError(
                        "PyFlashinfer FP8 KV cache requires an initialized "
                        "kv_scale_base buffer with implicit scale=1"
                    )

            # append_paged_kv_cache copies elements without converting dtype.
            key = self._cast_for_cache(key, k_cache.dtype)
            value = self._cast_for_cache(value, v_cache.dtype)

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

            cache_dtype = self.kv_cache_dtype or value.dtype
            key = self._cast_for_cache(key, cache_dtype)
            value = self._cast_for_cache(value, cache_dtype)

            # Create MHA KV cache: [num_pages, num_kv_heads, page_size, head_dim] (HND layout)
            k_cache = torch.empty(
                (
                    max_num_pages,
                    self.num_kv_heads,
                    self.token_per_block,
                    self.head_size,
                ),
                dtype=cache_dtype,
                device=value.device,
            )
            v_cache = torch.empty(
                (
                    max_num_pages,
                    self.num_kv_heads,
                    self.token_per_block,
                    self.head_size,
                ),
                dtype=cache_dtype,
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
