import logging
from typing import Optional

import torch
from flashinfer import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)
from flashinfer.cascade import merge_state
from flashinfer.page import append_paged_kv_cache

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather, recv, send
from rtp_llm.models_py.distributed.user_buffers import get_user_buffers_communicator
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.cp_utils import (
    generate_half_kv_indices,
    generate_half_q_indices,
    plan_prefix_paged_attention,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    get_py_flashinfer_workspace_buffer,
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
        self.workspace_buffer = get_py_flashinfer_workspace_buffer()

        self.cp_info = attn_inputs.context_parallel_info
        self.prefill_cp_rank = parallelism_config.tp_rank
        self.prefill_cp_size = parallelism_config.tp_size

        self.seq_size_per_block = attn_configs.tokens_per_block

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
        self.prefix_paged_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer,
            kv_layout="HND",
            backend=backend,
        )
        self.ub_communicator = get_user_buffers_communicator()
        self.use_ub = self.ub_communicator is not None

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

        params = fill_mla_params(
            self.attn_inputs.prefix_lengths,
            self.attn_inputs.sequence_lengths,
            self.cp_info.prefill_actual_input_lengths_cpu,
            self.attn_inputs.kv_cache_kernel_block_id_host,
            self.attn_configs.kernel_tokens_per_block,
        )

        chunk_lens = prefill_cp_chunk_lengths.tolist()
        self.append_batch_indice = torch.cat(
            [
                torch.full((cl,), i, dtype=torch.int32, device=self.device)
                for i, cl in enumerate(chunk_lens)
            ]
        )

        self.has_prefix = self.attn_inputs.prefix_lengths.any().item()
        if self.has_prefix:
            plan_prefix_paged_attention(
                self.prefix_paged_wrapper,
                cu_seqlens,
                attention_inputs.prefix_lengths,
                params,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=self.seq_size_per_block,
                device=self.device,
            )

        return params

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        params: ParamsBase = None,
    ) -> torch.Tensor:
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

        kv_buffer = torch.cat([k, v], dim=0)
        remote_kv_buffers = [torch.empty_like(kv_buffer) for _ in range(2)]
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
        kv_cache_tensor = kv_cache.kv_cache_base.view(
            -1, 2, self.num_kv_heads, self.seq_size_per_block, self.head_dim
        )
        for round_id in range(0, self.prefill_cp_size):
            if round_id > 0:
                out_buffer.zero_()
                lse_buffer.fill_(float("-inf"))

            if round_id < self.prefill_cp_size - 1:
                recv_buf = remote_kv_buffers[round_id % 2]
                with torch.cuda.stream(self.communication_stream):
                    prev_rank_id = (
                        self.prefill_cp_rank - round_id - 1
                    ) % self.prefill_cp_size
                    next_rank_id = (
                        self.prefill_cp_rank + round_id + 1
                    ) % self.prefill_cp_size

                    if self.use_ub and self.ub_communicator.can_handle_tensor(
                        kv_buffer
                    ):
                        self.ub_communicator.send(kv_buffer, dst=next_rank_id)
                        self.ub_communicator.recv(recv_buf, src=prev_rank_id)
                    else:
                        if self.prefill_cp_rank < next_rank_id:
                            send(kv_buffer, dst=next_rank_id, group=Group.TP)
                            recv(recv_buf, src=prev_rank_id, group=Group.TP)
                        else:
                            recv(recv_buf, src=prev_rank_id, group=Group.TP)
                            send(kv_buffer, dst=next_rank_id, group=Group.TP)
                    self.comm_events[round_id].record()

            if round_id == 0:  # local attention
                # TODO: make write local kvcache async

                k = k.reshape(-1, self.num_kv_heads, self.head_dim)
                v = v.reshape(-1, self.num_kv_heads, self.head_dim)

                append_paged_kv_cache(
                    append_key=k,
                    append_value=v,
                    batch_indices=self.append_batch_indice,
                    positions=self.all_shuffle_indices[self.prefill_cp_rank],
                    paged_kv_cache=kv_cache_tensor,
                    kv_indices=params.page_indice_d,
                    kv_indptr=params.decode_page_indptr_d,
                    kv_last_page_len=params.paged_kv_last_page_len_d,
                    kv_layout="HND",
                )

                q_reshaped = q.reshape(-1, self.num_qo_heads, self.head_dim)
                merged_out, merged_lse = self.prefill_wrappers["causal"].run(
                    q_reshaped,
                    k,
                    v,
                    return_lse=True,
                )

                if self.has_prefix:
                    prefix_out, prefix_lse = self.prefix_paged_wrapper.run(
                        q_reshaped,
                        kv_cache_tensor,
                        return_lse=True,
                    )
                    merged_out, merged_lse = merge_state(
                        v_a=merged_out,
                        s_a=merged_lse,
                        v_b=prefix_out,
                        s_b=prefix_lse,
                    )
                self.math_events[round_id].record()
            else:
                torch.cuda.current_stream().wait_event(self.comm_events[round_id - 1])
                compute_buf = remote_kv_buffers[(round_id - 1) % 2]
                remote_k, remote_v = torch.split(
                    compute_buf, [k.shape[0], v.shape[0]], dim=0
                )
                remote_k = remote_k.contiguous().reshape(
                    -1, self.num_kv_heads, self.head_dim
                )
                remote_v = remote_v.contiguous().reshape(
                    -1, self.num_kv_heads, self.head_dim
                )
                # TODO: make write local kvcache async
                src_rank = (self.prefill_cp_rank - round_id) % self.prefill_cp_size
                append_paged_kv_cache(
                    append_key=remote_k,
                    append_value=remote_v,
                    batch_indices=self.append_batch_indice,
                    positions=self.all_shuffle_indices[src_rank],
                    paged_kv_cache=kv_cache_tensor,
                    kv_indices=params.page_indice_d,
                    kv_indptr=params.decode_page_indptr_d,
                    kv_last_page_len=params.paged_kv_last_page_len_d,
                    kv_layout="HND",
                )
                if round_id > self.prefill_cp_rank:
                    q_split = (
                        torch.index_select(q, 0, self.half_q_idx)
                        .contiguous()
                        .reshape(-1, self.num_qo_heads, self.head_dim)
                    )
                    k_split = remote_k
                    v_split = remote_v
                    (
                        out_buffer[self.half_q_idx, :, :],
                        lse_buffer[self.half_q_idx, :],
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
                    k_split = torch.index_select(
                        remote_k, 0, self.half_kv_idx
                    ).contiguous()
                    v_split = torch.index_select(
                        remote_v, 0, self.half_kv_idx
                    ).contiguous()
                    q_split = q.contiguous().reshape(
                        -1, self.num_qo_heads, self.head_dim
                    )
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
                self.math_events[round_id].record()
            if round_id < self.prefill_cp_size - 1:
                self.communication_stream.wait_event(self.math_events[round_id])

        return merged_out
