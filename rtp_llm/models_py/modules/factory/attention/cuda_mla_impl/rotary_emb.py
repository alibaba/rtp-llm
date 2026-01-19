import math
from typing import Any, Optional

import flashinfer.page as page
import flashinfer.rope as rope
import torch
from flashinfer import get_batch_indices_positions, get_seq_lens

from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs, rtp_llm_ops

from .flashinfer_mla import check_attention_inputs


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

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        append_ckv_t: torch.Tensor,
        rope_params: Any,
        kv_cache: Optional[KVCache] = None,
    ):

        rope._apply_rope_pos_ids_cos_sin_cache(
            q=query,
            k=key.unsqueeze(1),
            q_rope=query,
            k_rope=key.unsqueeze(1),
            cos_sin_cache=self.cos_sin_cache,
            pos_ids=rope_params.positions_d,
            interleave=self.is_neox_style,
        )

        if kv_cache is not None:
            k_cache, v_cache = torch.split(
                kv_cache.kv_cache_base, [self.kv_lora_rank, self.rope_head_dim], dim=-1
            )

            page.append_paged_mla_kv_cache(
                append_ckv_t,
                key,
                rope_params.batch_indice_d,
                rope_params.positions_d,
                k_cache,
                v_cache,
                rope_params.page_indice_d,
                rope_params.decode_page_indptr_d,
                rope_params.paged_kv_last_page_len_d,
            )
        else:
            # for warm up jit
            kv_len = [append_ckv_t.size(0)]
            num_pages_per_req = torch.tensor(
                [math.ceil(len / self.token_per_block) for len in kv_len],
                dtype=torch.int32,
                device=append_ckv_t.device,
            )
            kv_append_length = torch.tensor(
                kv_len, dtype=torch.int32, device=append_ckv_t.device
            )
            kv_append_indptr = (
                torch.cat(
                    [
                        torch.zeros(1).int().to(append_ckv_t.device),
                        torch.cumsum(kv_append_length, dim=0),
                    ],
                )
                .int()
                .to(append_ckv_t.device)
            )

            max_num_pages = sum(num_pages_per_req)
            kv_page_indptr = (
                torch.cat(
                    [
                        torch.zeros(1).int().to(append_ckv_t.device),
                        torch.cumsum(num_pages_per_req, dim=0),
                    ],
                )
                .int()
                .to(append_ckv_t.device)
            )
            kv_page_indices = torch.arange(
                sum(num_pages_per_req), dtype=torch.int32, device=append_ckv_t.device
            )

            kv_last_page_len = torch.tensor(
                [
                    (
                        len % self.token_per_block
                        if len % self.token_per_block != 0
                        else self.token_per_block
                    )
                    for len in kv_len
                ],
                dtype=torch.int32,
                device=append_ckv_t.device,
            )
            batch_indices, positions = get_batch_indices_positions(
                kv_append_indptr,
                get_seq_lens(kv_page_indptr, kv_last_page_len, self.token_per_block),
                append_ckv_t.size(0),
            )
            cache = torch.empty(
                [
                    max_num_pages,
                    self.token_per_block,
                    self.kv_lora_rank + self.rope_head_dim,
                ],
                dtype=append_ckv_t.dtype,
                device=append_ckv_t.device,
            )
            k_cache, v_cache = torch.split(
                cache, [self.kv_lora_rank, self.rope_head_dim], dim=-1
            )

            page.append_paged_mla_kv_cache(
                append_ckv_t,
                key,
                batch_indices,
                positions,
                k_cache,
                v_cache,
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_len,
            )
