import math
from typing import Any, Optional

import flashinfer.page as page
import flashinfer.rope as rope
import torch
from flashinfer import get_batch_indices_positions, get_seq_lens

from rtp_llm.ops import KvCacheDataType, compute_ops
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs, rtp_llm_ops

from .flashinfer_mla import check_attention_inputs


class NewMlaRotaryEmbeddingParams(object):
    def __init__(
        self,
        fmha_params: rtp_llm_ops.SparseMlaParams,
    ):
        self.params = fmha_params


class NewMlaRotaryEmbeddingOp(object):
    """Original rotary positional embedding."""

    def __init__(
        self,
        head_size: int,
        cos_sin_cache: torch.Tensor | None,
        kv_lora_rank: int,
        rope_head_dim: int,
        token_per_block: int,
        is_neox_style: bool,
        kv_cache_dtype: KvCacheDataType,
    ) -> None:
        if cos_sin_cache is None:
            raise Exception(f"RotaryEmbedding need cos_sin_cache but got none")
        super().__init__()
        self.head_size = head_size
        self.is_neox_style = is_neox_style
        self.cos_sin_cache = cos_sin_cache
        self.kv_lora_rank = kv_lora_rank
        self.rope_head_dim = rope_head_dim
        self.token_per_block = token_per_block
        self.kv_cache_type = (
            "fp8_ds_mla" if kv_cache_dtype == KvCacheDataType.FP8 else "auto"
        )
        # Scale tensor is required for concat_and_cache_mla even in non-FP8 mode
        self.scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        append_ckv_t: torch.Tensor,
        rope_params: NewMlaRotaryEmbeddingParams,
        kv_cache: Optional[KVCache] = None,
    ):

        rope._apply_rope_pos_ids_cos_sin_cache(
            q=query,
            k=key.unsqueeze(1),
            q_rope=query,
            k_rope=key.unsqueeze(1),
            cos_sin_cache=self.cos_sin_cache,
            pos_ids=rope_params.params.positions_d,
            interleave=self.is_neox_style,
        )

        if kv_cache is not None:
            compute_ops.concat_and_cache_mla(
                append_ckv_t,
                key,
                kv_cache.kv_cache_base,
                rope_params.params.slot_mapping,
                self.kv_cache_type,
                self.scale,
            )
