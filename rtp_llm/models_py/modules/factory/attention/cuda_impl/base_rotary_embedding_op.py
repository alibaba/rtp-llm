"""Base class for Rotary Positional Embedding operations.

This module provides the abstract base class for implementing RoPE (Rotary Positional Embedding)
operations with FlashInfer. It includes common functionality for both MHA and MLA implementations.
"""

import math
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import flashinfer
import flashinfer.rope as rope
import torch
from flashinfer import get_batch_indices_positions, get_seq_lens

from rtp_llm.ops import (
    RopeConfig,
    check_rope_cache,
    get_rope_cache,
    get_rope_cache_once,
)

# cos/sin caches keyed by the rope parameters they were built from, see
# _resolve_cos_sin_cache().
_COS_SIN_CACHE_BY_CONFIG: dict[tuple, torch.Tensor] = {}


def _cos_sin_cache_key(
    rope_config: RopeConfig, max_position_embeddings: int, interleave: bool
) -> tuple:
    return (
        str(rope_config.style),
        int(rope_config.dim),
        int(rope_config.base),
        float(rope_config.scale),
        int(rope_config.max_pos),
        float(rope_config.factor1),
        float(rope_config.factor2),
        float(rope_config.extrapolation_factor),
        float(rope_config.mscale),
        int(max_position_embeddings),
        bool(interleave),
    )


def _resolve_cos_sin_cache(
    rope_config: RopeConfig, max_position_embeddings: int, interleave: bool
) -> Optional[torch.Tensor]:
    """Build the cos/sin cache that matches ``rope_config``.

    ``get_rope_cache_once`` keeps a single process-wide cache per interleave mode
    (``std::call_once``) and returns it for every later call regardless of the config
    passed in. Models whose layers use different rope parameters would therefore all get
    whichever config asked first -- MiMo V2.5 has theta 1e7 on its 9 global-attention
    layers and 1e4 on its 39 sliding-window layers, so the SWA layers would silently be
    positioned with the GA theta.

    Take the singleton only when it actually matches (``check_rope_cache`` plus a length
    check, which that helper does not cover), otherwise build a dedicated cache and
    memoize it per config.
    """
    key = _cos_sin_cache_key(rope_config, max_position_embeddings, interleave)
    cached = _COS_SIN_CACHE_BY_CONFIG.get(key)
    if cached is not None:
        return cached

    try:
        shared = get_rope_cache_once(
            rope_config, max_position_embeddings, is_cuda=True, interleave=interleave
        )
        if check_rope_cache(rope_config, shared) and shared.data.size(0) >= (
            max_position_embeddings
        ):
            data = shared.data
        else:
            data = get_rope_cache(rope_config, max_position_embeddings, interleave)
    except Exception:
        # Fall back to dynamic computation in _apply_rope.
        return None

    _COS_SIN_CACHE_BY_CONFIG[key] = data
    return data


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
            # FlashInfer uses non-interleaved format (False)
            self.cos_sin_cache = _resolve_cos_sin_cache(
                rope_config, max_position_embeddings, interleave=False
            )
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
        # narrow() to query.size(0) — no-op when fillParams already shrank
        # positions_d to nnz, correct slice when fill_params_mha_device left
        # the buffer at its alloc-time size.
        nnz = query.size(0)
        pos_ids = rope_params.positions_d.narrow(0, 0, nnz)

        if self.cos_sin_cache is not None:
            rope._apply_rope_pos_ids_cos_sin_cache(  # type: ignore
                q=query,
                k=key,
                q_rope=query,
                k_rope=key,
                cos_sin_cache=self.cos_sin_cache,
                pos_ids=pos_ids,
                interleave=self.is_neox_style,
            )
        else:
            rope_theta = (
                self.rope_config.base if self.rope_config is not None else 10000
            )
            flashinfer.apply_rope_pos_ids_inplace(
                query, key, pos_ids, rope_theta=rope_theta
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
