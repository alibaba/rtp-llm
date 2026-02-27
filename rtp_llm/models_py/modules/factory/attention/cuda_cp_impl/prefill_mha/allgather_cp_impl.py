import logging
from enum import Enum, auto
from functools import cached_property
from typing import Any, Dict, Optional

import torch
from numpy import append

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather, recv, send
from rtp_llm.models_py.distributed.user_buffers import get_user_buffers_communicator
from rtp_llm.ops import AttentionConfigs, CPRotateMethod, FMHAType, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    KVCache,
    ParamsBase,
    PyAttentionInputs,
    PyContextParallelParams,
    fill_mla_params,
)

logger = logging.getLogger(__name__)

from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
from flashinfer.cascade import merge_state
from flashinfer.page import append_paged_kv_cache

from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.cp_utils import (
    generate_kv_indices,
    generate_q_indices,
    get_workspace_buffer,
)


class PCPAllGatherAttnOp:
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
        backend: str = "auto",  # "auto", "fa2", or "fa3"
        causal: bool = True,
        kv_layout: str = "NHD",  # "NHD" or "HND"
    ):
        """
        Args:
            config: Model configuration
            num_heads: Number of query heads
            num_kv_heads: Number of key/value heads (for GQA/MQA)
            head_dim: Dimension of each head
            backend: FlashInfer backend ("auto", "fa2", or "fa3")
            causal: Whether to use causal masking
            kv_layout: KV cache layout ("NHD" or "HND")
        """
        super().__init__()
        self.attn_inputs = attn_inputs
        self.attn_configs = attn_configs
        self.num_qo_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.backend = backend
        self.kv_layout = kv_layout

        assert causal == True
        self.device = torch.cuda.current_device()
        self.workspace_buffer = get_workspace_buffer(self.device)

        self.cp_info = attn_inputs.context_parallel_info

        self.prefill_wrappers = {}

        self.prefill_cp_rank = parallelism_config.tp_rank
        self.prefill_cp_size = parallelism_config.tp_size

        self.q0_idx = self.q1_idx = None
        self.kv0_idx = self.kv1_idx = None
        # write kv cache params
        self.kv_restore_unpad_indices = None

        # init flashinfer attention wrapper
        for wrapper_name in ["part0", "part1"]:
            self.prefill_wrappers[wrapper_name] = BatchPrefillWithRaggedKVCacheWrapper(
                self.workspace_buffer,
                kv_layout=kv_layout,
                backend=backend,
            )

    def support(self, attention_inputs: PyAttentionInputs) -> bool:
        return attention_inputs.is_prefill

    def prepare(self, attention_inputs: PyAttentionInputs) -> ParamsBase:
        cu_seqlens = attention_inputs.cu_seqlens[
            : attention_inputs.input_lengths.size(0) + 1
        ]
        padding_mask = self.cp_info.prefill_qkv_padding_mask
        kv_restore_indices = self.cp_info.prefill_qkv_restore_indice
        self.kv_restore_unpad_indices = kv_restore_indices[padding_mask == 1]

        qo_indptr = cu_seqlens // 2
        kv_indptr_part0 = qo_indptr * (self.prefill_cp_rank + 1)
        kv_indptr_part1 = qo_indptr * (2 * self.prefill_cp_size - self.prefill_cp_rank)

        common_params = {
            "num_qo_heads": self.num_qo_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim_qk": self.head_dim,
            "causal": True,
            "q_data_type": torch.bfloat16,
        }
        configs = [
            {
                "wrapper_name": "part0",
                "qo_indptr": qo_indptr,
                "kv_indptr": kv_indptr_part0,
            },
            {
                "wrapper_name": "part1",
                "qo_indptr": qo_indptr,
                "kv_indptr": kv_indptr_part1,
            },
        ]
        for config in configs:
            wrapper_name = config.pop("wrapper_name")
            self.prefill_wrappers[wrapper_name].plan(**config, **common_params)

        q0_idx, q1_idx = generate_q_indices(self.cp_info.prefill_cp_chunk_lengths)
        kv0_idx, kv1_idx = generate_kv_indices(
            self.cp_info.prefill_cp_chunk_lengths,
            self.prefill_cp_rank,
            self.prefill_cp_size,
        )

        self.kv0_idx = kv_restore_indices[kv0_idx]
        self.kv1_idx = kv_restore_indices[kv1_idx]
        self.q0_idx = torch.tensor(q0_idx, device=self.device)
        self.q1_idx = torch.tensor(q1_idx, device=self.device)

        params = fill_mla_params(
            self.attn_inputs.prefix_lengths,
            self.attn_inputs.sequence_lengths,
            self.cp_info.prefill_actual_input_lengths_cpu,
            self.attn_inputs.kv_cache_block_id_host,
            self.attn_configs.tokens_per_block,
        )
        return params

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        params: ParamsBase = None,
    ) -> torch.Tensor:
        # reshape qkv to q, k, v
        qkv = qkv.reshape(qkv.shape[0], -1)
        q, k, v = torch.split(
            qkv,
            [
                self.head_dim * self.num_qo_heads,
                self.head_dim * self.num_kv_heads,
                self.head_dim * self.num_kv_heads,
            ],
            dim=-1,
        )
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        all_keys = all_gather(k, group=Group.TP).reshape(
            k.shape[0] * self.prefill_cp_size, self.num_kv_heads, self.head_dim
        )
        all_values = all_gather(v, group=Group.TP).reshape(
            v.shape[0] * self.prefill_cp_size, self.num_kv_heads, self.head_dim
        )
        q_reshaped = q.reshape(-1, self.num_qo_heads, self.head_dim)

        # TODO: make write local kvcache async
        restore_k = all_keys[self.kv_restore_unpad_indices]
        restore_v = all_values[self.kv_restore_unpad_indices]
        append_paged_kv_cache(
            append_key=restore_k,
            append_value=restore_v,
            batch_indices=params.batch_indice_d,
            positions=params.positions_d,
            paged_kv_cache=kv_cache.kv_cache_base,
            kv_indices=params.page_indice_d,
            kv_indptr=params.prefill_ragged_kv_len_indptr_d,
            kv_last_page_len=params.paged_kv_last_page_len_d,
            kv_layout="HND",
        )

        q0 = torch.index_select(q_reshaped, 0, self.q0_idx).contiguous()
        q1 = torch.index_select(q_reshaped, 0, self.q1_idx).contiguous()
        k0 = torch.index_select(all_keys, 0, self.kv0_idx).contiguous()
        k1 = torch.index_select(all_keys, 0, self.kv1_idx).contiguous()
        v0 = torch.index_select(all_values, 0, self.kv0_idx).contiguous()
        v1 = torch.index_select(all_values, 0, self.kv1_idx).contiguous()

        attn_output_part0 = self.prefill_wrappers["part0"].run(q0, k0, v0)
        attn_output_part1 = self.prefill_wrappers["part1"].run(q1, k1, v1)

        out = torch.cat([attn_output_part0, attn_output_part1], dim=0)
        return out
