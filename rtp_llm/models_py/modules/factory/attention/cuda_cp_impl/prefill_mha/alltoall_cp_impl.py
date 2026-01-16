import logging
from typing import Optional

import torch
from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
from flashinfer.cascade import merge_state
from flashinfer.page import append_paged_kv_cache

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather, recv, send
from rtp_llm.models_py.distributed.user_buffers import get_user_buffers_communicator
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.cp_utils import (
    generate_kv_indices,
    generate_q_indices,
    get_workspace_buffer,
)
from rtp_llm.ops import AttentionConfigs, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    KVCache,
    ParamsBase,
    PyAttentionInputs,
    fill_mla_params,
)

logger = logging.getLogger(__name__)


class PCPAll2AllAttnOp:
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
        assert causal == True
        self.kv_layout = kv_layout

        self.device = torch.cuda.current_device()
        self.workspace_buffer = get_workspace_buffer(self.device)

        self.cp_info = attn_inputs.context_parallel_info
        self.prefill_cp_rank = parallelism_config.tp_rank
        self.prefill_cp_size = parallelism_config.tp_size

        self.communication_stream = torch.cuda.Stream(device=self.device)
        self.comm_events = [torch.cuda.Event() for _ in range(self.prefill_cp_size)]
        self.math_events = [torch.cuda.Event() for _ in range(self.prefill_cp_size)]

        self.all_shuffle_indices = None
        self.half_q_idx = self.half_kv_idx = None

        # Init flashinfer attention wrappers
        wrapper_names = ["causal", "non_causal_pattern_0", "non_causal_pattern_1"]
        self.prefill_wrappers = {
            name: BatchPrefillWithRaggedKVCacheWrapper(
                self.workspace_buffer,
                kv_layout=kv_layout,
                backend=backend,
            )
            for name in wrapper_names
        }
        self.ub_communicator = get_user_buffers_communicator()

    def support(self, attention_inputs: PyAttentionInputs) -> bool:
        return attention_inputs.is_prefill

    def prepare(self, attention_inputs: PyAttentionInputs) -> ParamsBase:
        """Prepare for ALLTOALL rotation method (Ring Attention)."""
        cu_seqlens = attention_inputs.cu_seqlens[
            : attention_inputs.input_lengths.size(0) + 1
        ]
        prefill_cp_chunk_lengths = self.cp_info.prefill_cp_chunk_lengths
        shuffle_indices = self.cp_info.prefill_shuffle_indices
        self.all_shuffle_indices = all_gather(shuffle_indices, group=Group.TP).reshape(
            -1, shuffle_indices.shape[0]
        )

        common_params = {
            "num_qo_heads": self.num_qo_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim_qk": self.head_dim,
            "q_data_type": torch.bfloat16,
        }
        configs = [
            {
                "wrapper_name": "causal",
                "qo_indptr": cu_seqlens,
                "kv_indptr": cu_seqlens,
                "causal": True,
            },
            {
                "wrapper_name": "non_causal_pattern_0",
                "qo_indptr": cu_seqlens,
                "kv_indptr": cu_seqlens // 2,
                "causal": False,
            },
            {
                "wrapper_name": "non_causal_pattern_1",
                "qo_indptr": cu_seqlens // 2,
                "kv_indptr": cu_seqlens,
                "causal": False,
            },
        ]
        for config in configs:
            wrapper_name = config.pop("wrapper_name")
            self.prefill_wrappers[wrapper_name].plan(**config, **common_params)

        half_q_indices = generate_half_q_indices(prefill_cp_chunk_lengths)
        half_kv_indices = generate_half_kv_indices(prefill_cp_chunk_lengths)
        self.half_q_idx = torch.tensor(half_q_indices, device=self.device)
        self.half_kv_idx = torch.tensor(half_kv_indices, device=self.device)

        return fill_mla_params(
            self.attn_inputs.prefix_lengths,
            self.attn_inputs.sequence_lengths,
            self.attn_inputs.input_lengths,
            self.attn_inputs.kv_cache_block_id_host,
            self.attn_configs.tokens_per_block,
        )

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

        # init kv buffer
        kv_buffer = torch.cat([k, v], dim=0)
        remote_kv_buffer = torch.empty_like(kv_buffer)
        self.communication_stream.wait_stream(torch.cuda.current_stream())

        out_buffer = torch.empty(
            [q.shape[0], self.num_qo_heads, self.head_dim],
            dtype=q.dtype,
            device=q.device,
        )
        lse_buffer = torch.empty(
            [q.shape[0], self.num_qo_heads],
            dtype=torch.float32,
            device=q.device,
        )
        for round_id in range(0, self.prefill_cp_size):
            if round_id > 0:
                out_buffer.zero_()
                lse_buffer.fill_(float("-inf"))

            if round_id < self.prefill_cp_size - 1:
                with torch.cuda.stream(self.communication_stream):
                    prev_rank_id = (
                        self.prefill_cp_rank - round_id - 1
                    ) % self.prefill_cp_size
                    next_rank_id = (
                        self.prefill_cp_rank + round_id + 1
                    ) % self.prefill_cp_size

                    if (
                        self.ub_communicator is not None
                        and self.ub_communicator.can_handle_tensor(kv_buffer)
                    ):
                        self.ub_communicator.send(kv_buffer, dst=next_rank_id)
                        self.ub_communicator.recv(remote_kv_buffer, src=prev_rank_id)
                    else:
                        # Fall back to standard collective communication
                        if self.prefill_cp_rank < next_rank_id:
                            send(kv_buffer, dst=next_rank_id, group=Group.TP)
                            recv(remote_kv_buffer, src=prev_rank_id, group=Group.TP)
                        else:
                            recv(remote_kv_buffer, src=prev_rank_id, group=Group.TP)
                            send(kv_buffer, dst=next_rank_id, group=Group.TP)
                    self.comm_events[round_id].record()

            if round_id == 0:  # local attention
                self.math_events[round_id].record()
                # TODO: make write local kvcache async

                k = k.reshape(-1, self.num_kv_heads, self.head_dim)
                v = v.reshape(-1, self.num_kv_heads, self.head_dim)

                append_paged_kv_cache(
                    append_key=k,
                    append_value=v,
                    batch_indices=params.batch_indice_d,
                    positions=self.all_shuffle_indices[self.prefill_cp_rank],
                    paged_kv_cache=kv_cache.kv_cache_base,
                    kv_indices=params.page_indice_d,
                    kv_indptr=params.prefill_ragged_kv_len_indptr_d,
                    kv_last_page_len=params.paged_kv_last_page_len_d,
                    kv_layout="HND",
                )
                merged_out, merged_lse = self.prefill_wrappers["causal"].run(
                    q.reshape(-1, self.num_qo_heads, self.head_dim),
                    k,
                    v,
                    return_lse=True,
                )
            else:
                torch.cuda.current_stream().wait_event(self.comm_events[round_id - 1])
                self.comm_events[round_id - 1].synchronize()
                remote_k, remote_v = torch.split(
                    remote_kv_buffer, [k.shape[0], v.shape[0]], dim=0
                )
                remote_k = remote_k.contiguous().reshape(
                    -1, self.num_kv_heads, self.head_dim
                )
                remote_v = remote_v.contiguous().reshape(
                    -1, self.num_kv_heads, self.head_dim
                )
                # TODO: make write local kvcache async
                src_rank = (self.prefill_cp_rank - round_id - 1) % self.prefill_cp_size
                append_paged_kv_cache(
                    append_key=remote_k,
                    append_value=remote_v,
                    batch_indices=params.batch_indice_d,
                    positions=self.all_shuffle_indices[src_rank],
                    paged_kv_cache=kv_cache.kv_cache_base,
                    kv_indices=params.page_indice_d,
                    kv_indptr=params.prefill_ragged_kv_len_indptr_d,
                    kv_last_page_len=params.paged_kv_last_page_len_d,
                    kv_layout="HND",
                )
                if round_id > self.prefill_cp_rank:
                    # half q and full kv
                    q_split = (
                        torch.index_select(q, 0, self.half_q_indices)
                        .contiguous()
                        .reshape(-1, self.num_qo_heads, self.head_dim)
                    )
                    k_split = remote_k
                    v_split = remote_v
                    self.math_events[round_id].record()
                    (
                        out_buffer[self.half_q_indices, :, :],
                        lse_buffer[self.half_q_indices, :],
                    ) = self.prefill_wrappers["non_causal_pattern_1"].run(
                        q=q_split,
                        k=k_split,
                        v=v_split,
                        return_lse=True,
                    )
                    merged_out, merged_lse = merge_state(
                        v_a=merged_out,
                        s_a=merged_lse,
                        v_b=out_buffer,
                        s_b=lse_buffer,
                    )
                else:
                    # half kv and full q
                    k_split = torch.index_select(
                        remote_k, 0, self.half_kv_indices
                    ).contiguous()
                    v_split = torch.index_select(
                        remote_v, 0, self.half_kv_indices
                    ).contiguous()
                    q_split = q.contiguous().reshape(
                        -1, self.num_qo_heads, self.head_dim
                    )
                    self.math_events[round_id].record()
                    out_buffer, lse_buffer = self.prefill_wrappers[
                        "non_causal_pattern_0"
                    ].run(
                        q=q_split,
                        k=k_split,
                        v=v_split,
                        return_lse=True,
                    )
                    merged_out, merged_lse = merge_state(
                        v_a=merged_out,
                        s_a=merged_lse,
                        v_b=out_buffer,
                        s_b=lse_buffer,
                    )
            if round_id < self.prefill_cp_size - 1:
                self.communication_stream.wait_event(self.math_events[round_id])
                self.math_events[round_id].synchronize()

        return merged_out
