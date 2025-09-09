from typing import Any, Optional

import torch
from flashinfer.page import append_paged_mla_kv_cache as append_paged_mla_kv_cache
from flashinfer.rope import (
    _apply_rope_pos_ids_cos_sin_cache as _apply_rope_pos_ids_cos_sin_cache,
)

from rtp_llm.models_py.modules.flashinfer_mla import fill_flash_params
from rtp_llm.ops import KVCache, PyAttentionInputs


class MlaRotaryEmbeddingOp(object):
    """Original rotary positional embedding."""

    def __init__(
        self,
        head_size: int,
        cos_sin_cache: torch.Tensor | None,
        kv_lora_rank: int,
        rope_head_dim: int,
        token_per_block: int,
        is_neox_style: bool,
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

    def prepare(self, attention_inputs: PyAttentionInputs):
        return fill_flash_params(
            self.token_per_block, attention_inputs, self.cos_sin_cache.device
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        append_ckv_t: torch.Tensor,
        rope_params: Any,
        kv_cache: Optional[KVCache] = None,
    ):
        query_out = torch.empty_like(query)
        key_out = torch.empty_like(key)

        _apply_rope_pos_ids_cos_sin_cache(
            query,
            key.unsqueeze(1),
            query_out,
            key_out.unsqueeze(1),
            cos_sin_cache=self.cos_sin_cache,
            pos_ids=rope_params.positions,
            interleave=self.is_neox_style,
        )

        if kv_cache is not None:
            k_cache, v_cache = torch.split(
                kv_cache.k_cache_base, [self.kv_lora_rank, self.rope_head_dim], dim=-1
            )

            append_paged_mla_kv_cache(
                append_ckv_t,
                key_out,
                rope_params.batch_indice,
                rope_params.positions,
                k_cache,
                v_cache,
                rope_params.page_indice,
                rope_params.page_indptr,
                rope_params.paged_kv_last_page_len,
            )

        return query_out, key_out