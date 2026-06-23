"""KV Cache Write Operation for paged KV cache."""

from typing import Any, Optional, Tuple

import flashinfer.page as page
import torch

from rtp_llm.ops import KvCacheDataType
from rtp_llm.ops.compute_ops import LayerKVCache

FP8_E4M3_MAX = 448.0


class KVCacheWriteOp:
    """Operator for writing key-value pairs to paged KV cache."""

    def __init__(
        self,
        num_kv_heads: int,
        head_size: int,
        token_per_block: int,
        fp8_kv_cache_scale_mode: str = "per_tensor",
        kv_cache_dtype: KvCacheDataType = KvCacheDataType.BASE,
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
        self.fp8_kv_cache_scale_mode = fp8_kv_cache_scale_mode
        self.kv_cache_dtype = kv_cache_dtype
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
            # For real execution - use provided KV cache
            # KV cache has shape [num_pages, 2, num_kv_heads, page_size, head_dim] (HND layout)
            k_cache = kv_cache.kv_cache_base[
                :, 0, :, :, :
            ]  # [num_pages, num_kv_heads, page_size, head_dim]
            v_cache = kv_cache.kv_cache_base[
                :, 1, :, :, :
            ]  # [num_pages, num_kv_heads, page_size, head_dim]

            if self._use_fp8_per_token_head(kv_cache):
                key, value = self._quantize_and_store_scales(key, value, kv_cache)

            # flashinfer.page.append_paged_kv_cache does a raw element copy and
            # does not cast dtypes. RoPE produces BF16/FP16 K/V while an FP8 KV
            # cache stores e4m3, so cast before appending; otherwise activation
            # bytes are reinterpreted as FP8 and freshly-written cache tokens are
            # corrupted. Stage 1 uses scale 1.0, matching the read path.
            if key.dtype != k_cache.dtype:
                key = key.to(k_cache.dtype)
                value = value.to(v_cache.dtype)

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

    def _use_fp8_per_token_head(self, kv_cache: LayerKVCache) -> bool:
        return (
            self.kv_cache_dtype == KvCacheDataType.FP8
            and self.fp8_kv_cache_scale_mode == "per_token_head"
            and kv_cache.kv_scale_base is not None
            and kv_cache.kv_scale_base.numel() > 0
        )

    def _scale_views(self, kv_cache: LayerKVCache) -> tuple[torch.Tensor, torch.Tensor]:
        scale = kv_cache.kv_scale_base
        num_blocks = scale.size(0)
        scale_flat = scale.reshape(num_blocks, -1)
        scale_per_kv = self.num_kv_heads * self.token_per_block
        k_flat = scale_flat[:, :scale_per_kv]
        v_flat = scale_flat[:, scale_per_kv : 2 * scale_per_kv]
        k_scale = torch.as_strided(
            k_flat,
            (num_blocks, self.token_per_block, self.num_kv_heads),
            (scale_flat.stride(0), 1, self.token_per_block),
        )
        v_scale = torch.as_strided(
            v_flat,
            (num_blocks, self.token_per_block, self.num_kv_heads),
            (scale_flat.stride(0), 1, self.token_per_block),
        )
        return k_scale, v_scale

    def _slot_mapping(self, token_num: int, device: torch.device) -> torch.Tensor:
        batch_indices = self.params.batch_indice_d[:token_num].to(torch.long)
        positions = self.params.positions_d[:token_num].to(torch.long)
        block_offsets = positions // self.token_per_block
        offsets = positions % self.token_per_block
        page_indptr = self.params.decode_page_indptr_d.to(torch.long)
        page_indices = self.params.page_indice_d.to(torch.long)
        blocks = page_indices[page_indptr[batch_indices] + block_offsets]
        return blocks.to(device) * self.token_per_block + offsets.to(device)

    def _quantize_and_store_scales(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: LayerKVCache,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k_scale = torch.clamp(key.float().abs().amax(dim=-1) / FP8_E4M3_MAX, min=1e-6)
        v_scale = torch.clamp(value.float().abs().amax(dim=-1) / FP8_E4M3_MAX, min=1e-6)
        k_quant = torch.clamp(
            key.float() / k_scale.unsqueeze(-1), -FP8_E4M3_MAX, FP8_E4M3_MAX
        ).to(torch.float8_e4m3fn)
        v_quant = torch.clamp(
            value.float() / v_scale.unsqueeze(-1), -FP8_E4M3_MAX, FP8_E4M3_MAX
        ).to(torch.float8_e4m3fn)

        k_scale_cache, v_scale_cache = self._scale_views(kv_cache)
        slot_mapping = self._slot_mapping(key.size(0), key.device)
        block_ids = slot_mapping // self.token_per_block
        offsets = slot_mapping % self.token_per_block
        head_ids = torch.arange(self.num_kv_heads, device=key.device, dtype=torch.long)
        k_scale_cache[block_ids[:, None], offsets[:, None], head_ids[None, :]] = (
            k_scale.float()
        )
        v_scale_cache[block_ids[:, None], offsets[:, None], head_ids[None, :]] = (
            v_scale.float()
        )
        return k_quant, v_quant

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
