import logging
from typing import Any, Optional

import torch

from rtp_llm.models_py.modules.mla.flashinfer_mla import (
    check_attention_inputs,
    flashinfer_python,
)
from rtp_llm.ops import KVCache, PyAttentionInputs, rtp_llm_ops


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
        check_attention_inputs(attention_inputs)
        return rtp_llm_ops.fill_mla_params(
            attention_inputs.prefix_lengths,
            attention_inputs.sequence_lengths,
            attention_inputs.input_lengths,
            attention_inputs.kv_cache_block_id_host,
            self.token_per_block,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        append_ckv_t: torch.Tensor,
        rope_params: Any,
        kv_cache: Optional[KVCache] = None,
    ):

        flashinfer_python.rope._apply_rope_pos_ids_cos_sin_cache(
            q=query,
            k=key.unsqueeze(1),
            q_rope=query,
            k_rope=key.unsqueeze(1),
            cos_sin_cache=self.cos_sin_cache,
            pos_ids=rope_params.positions,
            interleave=self.is_neox_style,
        )

        if kv_cache is not None:
            k_cache, v_cache = torch.split(
                kv_cache.k_cache_base, [self.kv_lora_rank, self.rope_head_dim], dim=-1
            )

            flashinfer_python.page.append_paged_mla_kv_cache(
                append_ckv_t,
                key,
                rope_params.batch_indice,
                rope_params.positions,
                k_cache,
                v_cache,
                rope_params.page_indice,
                rope_params.page_indptr,
                rope_params.paged_kv_last_page_len,
            )
