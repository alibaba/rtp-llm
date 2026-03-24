from typing import Optional

import flashinfer.rope as rope
import torch

from rtp_llm.ops import KvCacheDataType, compute_ops
from rtp_llm.ops.compute_ops import KVCache, rtp_llm_ops


class NewMlaRotaryEmbeddingOp(object):
    """Original rotary positional embedding."""

    def __init__(
        self,
        cos_sin_cache: torch.Tensor | None,
        is_neox_style: bool,
    ) -> None:
        if cos_sin_cache is None:
            raise Exception(f"RotaryEmbedding need cos_sin_cache but got none")
        super().__init__()
        self.is_neox_style = is_neox_style
        self.cos_sin_cache = cos_sin_cache

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        fmha_params: rtp_llm_ops.SparseMlaParams,
    ):

        rope._apply_rope_pos_ids_cos_sin_cache(
            q=query,
            k=key.unsqueeze(1),
            q_rope=query,
            k_rope=key.unsqueeze(1),
            cos_sin_cache=self.cos_sin_cache,
            pos_ids=fmha_params.positions_d,
            interleave=not self.is_neox_style,
        )
