"""Common utilities for attention implementations.

This module contains:
1. Base class for Rotary Positional Embedding operations
2. Helper functions for FMHA implementations (cache store, parameter updates, etc.)
"""

import math
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import flashinfer
import flashinfer.rope as rope
import torch
from flashinfer import get_batch_indices_positions, get_seq_lens

from rtp_llm.models_py.modules.base.common.kvcache_store import WriteCacheStoreOp
from rtp_llm.ops import RopeConfig, get_rope_cache_once
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs


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
        rope_config: Optional[RopeConfig] = None,
        max_position_embeddings: int = 32768,
    ) -> None:
        """
        Args:
            head_size: Dimension of each attention head
            cos_sin_cache: Precomputed cos/sin cache for RoPE [max_seq_len, rope_dim].
                          If None and rope_config is provided, will auto-generate using get_rope_cache_once.
            token_per_block: Number of tokens per KV cache block (page size)
            is_neox_style: RoPE interleave style:
                          - True (GPT-NeoX/interleave): Rotate adjacent pairs of dimensions together,
                            i.e., (x[0], x[1]), (x[2], x[3]), ..., (x[d-2], x[d-1])
                          - False (LLaMA/non-interleave): Rotate first and second halves separately,
                            i.e., (x[0], x[d/2]), (x[1], x[d/2+1]), ..., (x[d/2-1], x[d-1])
                          Most modern models (LLaMA, Qwen, DeepSeek, Mistral, etc.) use False.
                          Only specific models like GPT-NeoX use True.
            rope_config: RoPE configuration for auto-generating cos_sin_cache if not provided (optional)
            max_position_embeddings: Maximum position embeddings for auto-generating cache
        """
        super().__init__()
        self.head_size = head_size
        self.is_neox_style = is_neox_style
        self.token_per_block = token_per_block
        self.rope_config = rope_config

        # Try to get cos_sin_cache from C++ RopeCache if not provided
        if cos_sin_cache is None and rope_config is not None:
            # Save original interleave value
            original_interleave = rope_config.interleave
            try:
                rope_config.interleave = False
                rope_cache = get_rope_cache_once(
                    rope_config, max_position_embeddings, is_cuda=True
                )
                self.cos_sin_cache = rope_cache.data
            except Exception:
                # If get_rope_cache_once fails, fallback to dynamic computation in _apply_rope
                self.cos_sin_cache = None
            finally:
                # Always restore original interleave value
                rope_config.interleave = original_interleave
        else:
            self.cos_sin_cache = cos_sin_cache

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
        if self.cos_sin_cache is not None:
            rope._apply_rope_pos_ids_cos_sin_cache(  # type: ignore
                q=query,
                k=key,
                q_rope=query,
                k_rope=key,
                cos_sin_cache=self.cos_sin_cache,
                pos_ids=rope_params.positions_d,
                interleave=self.is_neox_style,
            )
        else:
            rope_theta = (
                self.rope_config.base if self.rope_config is not None else 10000
            )
            flashinfer.apply_rope_pos_ids_inplace(
                query, key, rope_params.positions_d, rope_theta=rope_theta
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


def create_write_cache_store_impl(
    attn_inputs: PyAttentionInputs,
) -> Optional[WriteCacheStoreOp]:
    """Create write cache store implementation if needed.

    Args:
        attn_inputs: Attention calculation input parameters

    Returns:
        WriteCacheStoreOp instance if cache store is needed, None otherwise
    """
    if attn_inputs.is_prefill and attn_inputs.cache_store_inputs:
        return WriteCacheStoreOp(
            attn_inputs.input_lengths,
            attn_inputs.prefix_lengths,
            attn_inputs.kv_cache_block_id_host,
            attn_inputs.cache_store_inputs,
        )
    return None


def apply_write_cache_store(
    write_cache_store_impl: Optional[WriteCacheStoreOp],
    attn_inputs: PyAttentionInputs,
    kv_cache: Optional[KVCache],
) -> None:
    """Apply write cache store operation if needed.

    Args:
        write_cache_store_impl: Write cache store implementation
        attn_inputs: Attention calculation input parameters
        kv_cache: KV Cache to write to
    """
    if (
        attn_inputs.is_prefill
        and attn_inputs.cache_store_inputs
        and write_cache_store_impl is not None
    ):
        write_cache_store_impl(kv_cache)


def copy_kv_cache_offset(old_offset: torch.Tensor, new_offset: torch.Tensor) -> None:
    """Copy KV Cache offset data.

    Used for CUDA graph parameter update scenarios.
    Copies new offset data into old offset tensor. If shapes match, copies directly,
    otherwise only copies the matching portion (slicing from the first dimension).

    Args:
        old_offset: Target offset tensor, data will be updated
        new_offset: Source offset tensor, provides new data
    """
    if new_offset.shape == old_offset.shape:
        old_offset.copy_(new_offset, non_blocking=True)
    else:
        # Build slice indices dynamically
        slice_indices = [
            slice(0, new_offset.size(dim)) for dim in range(new_offset.dim())
        ]
        target_slice = old_offset[tuple(slice_indices)]
        target_slice.copy_(new_offset, non_blocking=True)


def update_trt_params(
    fmha_impl: Any,
    rope_kvcache_impl: Any,
    fmha_params: Any,
    rope_params: Any,
    attn_inputs: PyAttentionInputs,
) -> None:
    """Update TRT-related parameters.

    Updates FMHA and RoPE parameters based on new input parameters, maintaining KV Cache offset consistency.
    Mainly used for CUDA graph parameter update scenarios.

    Args:
        fmha_impl: FMHA implementation object
        rope_kvcache_impl: RoPE KV Cache implementation object
        fmha_params: Current FMHA parameters
        rope_params: Current RoPE parameters
        attn_inputs: New attention calculation input parameters
    """
    new_fmha_params = fmha_impl.prepare(attn_inputs)
    new_offset = new_fmha_params.kv_cache_offset
    old_offset = fmha_params.kv_cache_offset
    copy_kv_cache_offset(old_offset, new_offset)

    new_rope_params = rope_kvcache_impl.prepare(attn_inputs)
    new_offset = new_rope_params.kv_cache_offset
    old_offset = rope_params.kv_cache_offset
    copy_kv_cache_offset(old_offset, new_offset)
