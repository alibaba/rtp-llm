import logging
from enum import Enum, auto
from functools import cached_property
from typing import Any, Dict, Optional

import torch
from numpy import append

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather, recv, send
from rtp_llm.models_py.distributed.user_buffers import get_user_buffers_communicator
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHAPrefillImplBase,
)
from rtp_llm.ops import AttentionConfigs, CPRotateMethod, FMHAType, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCachePrefillOp,
    KVCache,
    ParamsBase,
    PyAttentionInputs,
    PyContextParallelParams,
    fill_mla_params,
)

logger = logging.getLogger(__name__)

# Global workspace buffer shared across all wrappers
_g_workspace_buffer = None
_g_workspace_size = 512 * 1024 * 1024  # 512MB


def get_workspace_buffer(device: torch.device) -> torch.Tensor:
    """Get or create global workspace buffer for FlashInfer."""
    global _g_workspace_buffer
    if _g_workspace_buffer is None:
        _g_workspace_buffer = torch.empty(
            _g_workspace_size,
            dtype=torch.uint8,
            device=device,
        )
    return _g_workspace_buffer


from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
from flashinfer.cascade import merge_state
from flashinfer.page import append_paged_kv_cache


class ContextParallelFlashInferRaggedPrefillOp:
    """
    FlashInfer Ragged KV Cache Prefill Attention Operator for Context Parallel MHA.

    This operator supports three context parallel rotation methods for distributed attention computation
    with zig-zag load balancing to optimize causal attention patterns.

    ## Context Parallel Rotation Methods

    ### 1. ALL_GATHER Method

    Zig-zag Attention Partitioning for Causal Attention:
    For a sequence of length N distributed across cp_size ranks:

    1. Tokens are split using zig-zag shuffle: alternating chunks from start/end
       Example (cp_size=4, N=16, chunk=2): [0,1, 14,15, 2,3, 12,13, 4,5, 10,11, 6,7, 8,9]
                                             └─┘  └──┘  └─┘  └──┘  └─┘  └──┘  └─┘  └─┘
                                       rank0(r0)   r0  r1   r1    r2   r2    r3   r3

    2. Each rank holds a subset of Q and KV:
       - Rank i has Q_i (queries for its token chunk)
       - Rank i has KV_i (keys/values for its token chunk)

    3. Causal Attention Matrix (Q attends to KV where token_j <= token_i):

                  KV0  KV1  KV2  KV3  KV4  KV5  KV6  KV7  KV8  KV9  KV10 KV11 KV12 KV13 KV14 KV15
       Q0   (r0) [ C   ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗   ]
       Q1   (r0) [ C   C    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗   ] 2
       Q2   (r1) [ C   C    C    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗   ]
       Q3   (r1) [ C   C    C    C    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗   ] 4
       Q4   (r2) [ C   C    C    C    C    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗   ]
       Q5   (r2) [ C   C    C    C    C    C    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗   ] 6
       Q6   (r3) [ C   C    C    C    C    C    C    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗   ]
       Q7   (r3) [ C   C    C    C    C    C    C    C    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗   ] 8
       Q8   (r3) [ C   C    C    C    C    C    C    C    C    ✗    ✗    ✗    ✗    ✗    ✗    ✗   ]
       Q9   (r3) [ C   C    C    C    C    C    C    C    C    C    ✗    ✗    ✗    ✗    ✗    ✗   ] 10
       Q10  (r2) [ C   C    C    C    C    C    C    C    C    C    C    ✗    ✗    ✗    ✗    ✗   ]
       Q11  (r2) [ C   C    C    C    C    C    C    C    C    C    C    C    ✗    ✗    ✗    ✗   ] 12
       Q12  (r1) [ C   C    C    C    C    C    C    C    C    C    C    C    C    ✗    ✗    ✗   ]
       Q13  (r1) [ C   C    C    C    C    C    C    C    C    C    C    C    C    C    ✗    ✗   ] 14
       Q14  (r0) [ C   C    C    C    C    C    C    C    C    C    C    C    C    C    C    ✗   ]
       Q15  (r0) [ C   C    C    C    C    C    C    C    C    C    C    C    C    C    C    C   ] 16

    4. All gather without overlap:
       - rank_i_part_0: q_len=chunk_size, kv_len=chunk_size*(rank_id + 1)
       - rank_i_part_1: q_len=chunk_size, kv_len=chunk_size*(2 * cp_size - rank_id)

    ### 2. ALLTOALL Method (Ring Attention)

    Ring attention with zig-zag load balancing: KV chunks rotate across ranks while Q stays local.
    Each rank computes attention between its local Q and the received KV chunks.

    Example (cp_size=4, N=16, chunk_size=4):

                  rank0                rank1                     rank2                 rank3
    --------------------------------------------------------------------------------------------------
    iter0: Calculate local causal attention
                     kv0 kv1 kv14 kv15      kv2 kv3 kv12 kv13       kv4 kv5 kv10 kv11      kv6 kv7 kv8 kv9
               [q0    Y   X   X    X ][q2    Y   X   X    X ][q4    Y   X   X    X ][q6    Y   X   X    X ]
               [q1    Y   Y   X    X ][q3    Y   Y   X    X ][q5    Y   Y   X    X ][q7    Y   Y   X    X ]
               [q14   Y   Y   Y    X ][q12   Y   Y   Y    X ][q10   Y   Y   Y    X ][q8    Y   Y   Y    X ]
               [q15   Y   Y   Y    Y ][q13   Y   Y   Y    Y ][q11   Y   Y   Y    Y ][q9    Y   Y   Y    Y ]
    ---------------------------------------------------------------------------------------------------
    iter1: KV rotates right (rank0->rank1, rank1->rank2, rank2->rank3, rank3->rank0)
                     kv6 kv7 kv8 kv9        kv0 kv1 kv14 kv15      kv2 kv3 kv12 kv13     kv4 kv5 kv10 kv11
               [q0    X   X   X    X ][q2    Y   Y   X    X ][q4    Y   Y   X    X ][q6    Y   Y   X    X ]
               [q1    X   X   X    X ][q3    Y   Y   X    X ][q5    Y   Y   X    X ][q7    Y   Y   X    X ]
               [q14   Y   Y   Y    Y ][q12   Y   Y   X    X ][q10   Y   Y   X    X ][q8    Y   Y   X    X ]
               [q15   Y   Y   Y    Y ][q13   Y   Y   X    X ][q11   Y   Y   X    X ][q9    Y   Y   X    X ]
    ---------------------------------------------------------------------------------------------------
    iter2: KV rotates right again
                     kv4 kv5 kv10 kv11       kv6 kv7 kv8 kv9        kv0 kv1 kv14 kv15     kv2 kv3 kv12 kv13
               [q0    X   X   X    X ][q2    X   X   X    X ][q4    Y   Y   X    X ][q6    Y   Y   X    X ]
               [q1    X   X   X    X ][q3    X   X   X    X ][q5    Y   Y   X    X ][q7    Y   Y   X    X ]
               [q14   Y   Y   Y    Y ][q12   Y   Y   Y    Y ][q10   Y   Y   X    X ][q8    Y   Y   X    X ]
               [q15   Y   Y   Y    Y ][q13   Y   Y   Y    Y ][q11   Y   Y   X    X ][q9    Y   Y   X    X ]
    ---------------------------------------------------------------------------------------------------
    iter3: KV rotates right again (final iteration)
                     kv2 kv3 kv12 kv13      kv4 kv5 kv10 kv11       kv6 kv7 kv8 kv9      kv0 kv1 kv14 kv15
               [q0    X   X   X    X ][q2    X   X   X    X ][q4    X   X   X    X ][q6    Y   Y   X    X ]
               [q1    X   X   X    X ][q3    X   X   X    X ][q5    X   X   X    X ][q7    Y   Y   X    X ]
               [q14   Y   Y   Y    Y ][q12   Y   Y   Y    Y ][q10   Y   Y   Y    Y ][q8    Y   Y   X    X ]
               [q15   Y   Y   Y    Y ][q13   Y   Y   Y    Y ][q11   Y   Y   Y    Y ][q9    Y   Y   X    X ]
    ---------------------------------------------------------------------------------------------------

    All chunk attention has 3 patterns:
    1. Local rank (i.e. iter=0): all ranks compute causal attention
    2. iter_i <= cp_rank: compute non-causal attention with half of the chunk kv
    3. iter_i > cp_rank: compute non-causal attention with half of the chunk q

    ### 3. ALL_GATHER_WITH_OVERLAP Method

    Similar to ALL_GATHER but overlaps communication with computation for better performance.

    Reference: https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/experimental/_context_parallel/_attention.py
    """

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
        self.causal = causal
        self.kv_layout = kv_layout

        self.device = torch.cuda.current_device()
        self.workspace_buffer = get_workspace_buffer(self.device)

        # Set rotate method from parallelism_config or use default
        if parallelism_config is not None and hasattr(
            parallelism_config, "cp_rotate_method"
        ):
            self.rotate_method = parallelism_config.cp_rotate_method
        else:
            self.rotate_method = CPRotateMethod.ALLTOALL

        self.context_parallel_info: PyContextParallelParams = (
            attn_inputs.context_parallel_info
        )
        self.prefill_wrappers = {}

        if parallelism_config is not None:
            self.cp_rank = parallelism_config.cp_rank
            self.cp_size = parallelism_config.cp_size
        else:
            self.cp_rank = 0
            self.cp_size = 1

        self.communication_stream = torch.cuda.Stream(device=self.device)
        self.comm_events = [torch.cuda.Event() for _ in range(self.cp_size)]
        self.math_events = [torch.cuda.Event() for _ in range(self.cp_size)]

        self.all_shuffle_indices = None
        self.kv_restore_indices = None
        self.q_part_0_indices = self.q_part_1_indices = None
        self.kv_part_0_indices = self.kv_part_1_indices = None
        self.half_q_indices = self.half_kv_indices = None

        # write kv cache params
        self.kv_restore_unpad_indices = None
        self.actual_input_lengths = None
        self.actual_cu_seqlens = None

        # init flashinfer attention wrapper
        wrapper_configs = {
            CPRotateMethod.ALL_GATHER: ["part0", "part1"],
            CPRotateMethod.ALLTOALL: [
                "causal",
                "non_causal_pattern_0",
                "non_causal_pattern_1",
            ],
            CPRotateMethod.ALL_GATHER_WITH_OVERLAP: [
                "causal",
                "non_causal_part_0",
                "non_causal_part_1",
            ],
        }
        if self.rotate_method not in wrapper_configs:
            raise ValueError(f"Unsupported rotate method: {self.rotate_method}")

        for wrapper_name in wrapper_configs[self.rotate_method]:
            self.prefill_wrappers[wrapper_name] = BatchPrefillWithRaggedKVCacheWrapper(
                self.workspace_buffer,
                kv_layout=kv_layout,
                backend=backend,
            )
        self.ub_communicator = get_user_buffers_communicator()

    def support(self, attention_inputs: PyAttentionInputs) -> bool:
        return attention_inputs.is_prefill

    def prepare(self, attention_inputs: PyAttentionInputs) -> ParamsBase:
        # Get batch information
        batch_size = attention_inputs.input_lengths.size(0)
        device = torch.cuda.current_device()
        cu_seqlens = attention_inputs.cu_seqlens[
            : attention_inputs.input_lengths.size(0) + 1
        ]
        cp_info = attention_inputs.context_parallel_info
        prefill_cp_chunk_lengths = cp_info.prefill_cp_chunk_lengths
        padding_mask = cp_info.prefill_qkv_padding_mask
        self.kv_restore_indices = cp_info.prefill_qkv_restore_indice
        self.kv_restore_unpad_indices = self.kv_restore_indices[padding_mask == 1]
        self.actual_input_lengths = cp_info.prefill_actual_input_lengths_cpu
        shuffle_indices = cp_info.prefill_shuffle_indices
        self.all_shuffle_indices = all_gather(shuffle_indices, group=Group.CP).reshape(
            -1, shuffle_indices.shape[0]
        )

        if self.rotate_method == CPRotateMethod.ALL_GATHER:
            # Plan for both part0 and part1 wrappers
            qo_indptr = cu_seqlens // 2
            kv_indptr_part0 = qo_indptr * (self.cp_rank + 1)
            kv_indptr_part1 = qo_indptr * (2 * self.cp_size - self.cp_rank)
            # Part0: First part of attention computation
            self.prefill_wrappers["part0"].plan(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr_part0,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_dim,
                causal=True,  # Part0 uses causal attention
                q_data_type=torch.bfloat16,
            )
            # Part1: Second part of attention computation
            self.prefill_wrappers["part1"].plan(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr_part1,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_dim,
                causal=True,
                q_data_type=torch.bfloat16,
            )
            q_part_0_indices, q_part_1_indices = self._generate_q_indices(
                prefill_cp_chunk_lengths
            )
            kv_part_0_indices, kv_part_1_indices = self._generate_kv_indices(
                prefill_cp_chunk_lengths, self.cp_rank, self.cp_size
            )
            self.kv_part_0_indices = self.kv_restore_indices[kv_part_0_indices]
            self.kv_part_1_indices = self.kv_restore_indices[kv_part_1_indices]
            self.q_part_0_indices = torch.tensor(q_part_0_indices, device=device)
            self.q_part_1_indices = torch.tensor(q_part_1_indices, device=device)

            params = fill_mla_params(
                self.attn_inputs.prefix_lengths,
                self.attn_inputs.sequence_lengths,
                self.actual_input_lengths,
                self.attn_inputs.kv_cache_block_id_host,
                self.attn_configs.tokens_per_block,
            )
            return params

        elif self.rotate_method == CPRotateMethod.ALLTOALL:
            # local attention
            self.prefill_wrappers["causal"].plan(
                qo_indptr=cu_seqlens,
                kv_indptr=cu_seqlens,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_dim,
                causal=True,
                q_data_type=torch.bfloat16,
            )
            # non-causal attention with full-chunk-q and half-chunk-kv
            self.prefill_wrappers["non_causal_pattern_0"].plan(
                qo_indptr=cu_seqlens,
                kv_indptr=cu_seqlens // 2,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_dim,
                causal=False,
                q_data_type=torch.bfloat16,
            )
            # non-causal attention with half-chunk-q and full-chunk-kv
            self.prefill_wrappers["non_causal_pattern_1"].plan(
                qo_indptr=cu_seqlens // 2,
                kv_indptr=cu_seqlens,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_dim,
                causal=False,
                q_data_type=torch.bfloat16,
            )
            half_q_indices = self._generate_half_q_indices(prefill_cp_chunk_lengths)
            half_kv_indices = self._generate_half_kv_indices(prefill_cp_chunk_lengths)

            self.half_q_indices = torch.tensor(half_q_indices, device=self.device)
            self.half_kv_indices = torch.tensor(half_kv_indices, device=self.device)
            return fill_mla_params(
                self.attn_inputs.prefix_lengths,
                self.attn_inputs.sequence_lengths,
                self.attn_inputs.input_lengths,
                self.attn_inputs.kv_cache_block_id_host,
                self.attn_configs.tokens_per_block,
            )

        elif self.rotate_method == CPRotateMethod.ALL_GATHER_WITH_OVERLAP:
            # Plan for both part0 and part1 wrappers
            qo_indptr = cu_seqlens // 2
            kv_indptr_part0 = qo_indptr * self.cp_rank
            kv_indptr_part1 = qo_indptr * (2 * self.cp_size - self.cp_rank - 1)
            # local attention
            self.prefill_wrappers["causal"].plan(
                qo_indptr=cu_seqlens,
                kv_indptr=cu_seqlens,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_dim,
                causal=True,
                q_data_type=torch.bfloat16,
            )
            # Part0: First part of attention computation
            self.prefill_wrappers["non_causal_part_0"].plan(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr_part0,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_dim,
                causal=False,
                q_data_type=torch.bfloat16,
            )
            # Part1: Second part of attention computation
            self.prefill_wrappers["non_causal_part_1"].plan(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr_part1,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_dim,
                causal=False,
                q_data_type=torch.bfloat16,
            )
            q_part_0_indices, q_part_1_indices = self._generate_q_indices(
                prefill_cp_chunk_lengths
            )
            kv_part_0_indices, kv_part_1_indices = self._generate_kv_indices(
                prefill_cp_chunk_lengths, self.cp_rank, self.cp_size, is_non_local=True
            )
            self.q_part_0_indices = torch.tensor(q_part_0_indices, device=device)
            self.q_part_1_indices = torch.tensor(q_part_1_indices, device=device)
            self.kv_part_0_indices = self.kv_restore_indices[kv_part_0_indices]
            self.kv_part_1_indices = self.kv_restore_indices[kv_part_1_indices]

            return fill_mla_params(
                self.attn_inputs.prefix_lengths,
                self.attn_inputs.sequence_lengths,
                self.actual_input_lengths,
                self.attn_inputs.kv_cache_block_id_host,
                self.attn_configs.tokens_per_block,
            )
        else:
            raise ValueError(f"Unsupported rotate method: {self.rotate_method}")
        return ParamsBase()

    def _generate_half_q_indices(self, cp_chunk_lengths):
        half_q_indices = []
        offset = 0
        for chunk_len in cp_chunk_lengths:
            assert chunk_len % 2 == 0
            half_q_indices.extend(range(offset + (chunk_len) // 2, offset + chunk_len))
            offset += chunk_len
        return half_q_indices

    def _generate_half_kv_indices(self, cp_chunk_lengths):
        half_kv_indices = []
        offset = 0
        for chunk_len in cp_chunk_lengths:
            assert chunk_len % 2 == 0
            half_kv_indices.extend(range(offset, offset + (chunk_len) // 2))
            offset += chunk_len
        return half_kv_indices

    def _generate_kv_restore_indices(
        self, cp_chunk_lengths, all_cp_indices, cp_rank, cp_size
    ):
        input_length = sum(cp_chunk_lengths).item()
        all_cp_indices_cpu = all_cp_indices.tolist()
        all_cp_indices = all_cp_indices.reshape(-1)

        for i in range(cp_size):
            start_offset = i * input_length
            chunk_offset = 0
            for chunk_len in cp_chunk_lengths:
                for j in range(chunk_len):
                    all_cp_indices[start_offset + chunk_offset + j - 1] = (
                        all_cp_indices[start_offset + chunk_offset + j - 1]
                        + chunk_offset * cp_size
                    )
                chunk_offset += chunk_len
        sort_indices = torch.argsort(all_cp_indices)
        return sort_indices

    def _generate_kv_indices(
        self, cp_chunk_lengths, cp_rank, cp_size, is_non_local=False
    ):
        cp_chunk_lengths_cpu = cp_chunk_lengths.tolist()
        restore_seq_len = [x * cp_size for x in cp_chunk_lengths_cpu]

        kv_part_0_indices = []
        kv_part_1_indices = []
        seq_offset = 0
        for i in range(len(restore_seq_len)):
            assert cp_chunk_lengths_cpu[i] % 2 == 0
            half_chunk_len = cp_chunk_lengths_cpu[i] // 2
            start_pos_part0 = 0
            end_pos_part0 = half_chunk_len * (cp_rank + 1 - int(is_non_local))
            start_pos_part1 = 0
            end_pos_part1 = half_chunk_len * (2 * cp_size - cp_rank - int(is_non_local))
            if end_pos_part0 > start_pos_part0:
                kv_part_0_indices.extend(
                    range(start_pos_part0 + seq_offset, end_pos_part0 + seq_offset)
                )
            kv_part_1_indices.extend(
                range(start_pos_part1 + seq_offset, end_pos_part1 + seq_offset)
            )
            seq_offset += restore_seq_len[i]
        return kv_part_0_indices, kv_part_1_indices

    def _generate_q_indices(self, cp_chunk_lengths):
        """Generate two sets of indices by splitting each chunk in half.
        Args:
            cp_chunk_lengths: List of chunk lengths for each CP rank
        Returns:
            indices0: List of first half indices from each chunk (gets extra element if odd)
            indices1: List of second half indices from each chunk
        Example 1:
            cp_chunk_lengths = [8, 4, 4]
            indices0 = [0, 1, 2, 3, 8, 9, 12, 13]
            indices1 = [4, 5, 6, 7, 10, 11, 14, 15]
        Example 2:
            cp_chunk_lengths = [1, 8, 4, 4, 1]
            indices0 = [0, 1, 2, 3, 4, 9, 10, 13, 14, 17]
            indices1 = [5, 6, 7, 8, 11, 12, 15, 16]
        """
        indices0 = []
        indices1 = []
        offset = 0
        for chunk_len in cp_chunk_lengths:
            # Use ceiling division for first half (gets extra element if odd)
            half0 = (chunk_len + 1) // 2
            indices0.extend(range(offset, offset + half0))
            indices1.extend(range(offset + half0, offset + chunk_len))
            offset += chunk_len

        return indices0, indices1

    def forward_all_gather(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        params: ParamsBase = None,
    ) -> torch.Tensor:
        all_keys = all_gather(k, group=Group.CP).reshape(
            k.shape[0] * self.cp_size, self.num_kv_heads, self.head_dim
        )
        all_values = all_gather(v, group=Group.CP).reshape(
            v.shape[0] * self.cp_size, self.num_kv_heads, self.head_dim
        )
        q_reshaped = q.reshape(-1, self.num_qo_heads, self.head_dim)

        # TODO: make write local kvcache async
        restore_k = all_keys[self.kv_restore_unpad_indices]
        restore_v = all_values[self.kv_restore_unpad_indices]
        append_paged_kv_cache(
            append_key=restore_k,
            append_value=restore_v,
            batch_indices=params.batch_indice,
            positions=params.positions,
            paged_kv_cache=kv_cache.kv_cache_base,
            kv_indices=params.page_indice,
            kv_indptr=params.prefill_page_indptr,
            kv_last_page_len=params.paged_kv_last_page_len,
            kv_layout="HND",
        )
        q0 = torch.index_select(q_reshaped, 0, self.q_part_0_indices).contiguous()
        q1 = torch.index_select(q_reshaped, 0, self.q_part_1_indices).contiguous()
        k0 = torch.index_select(all_keys, 0, self.kv_part_0_indices).contiguous()
        k1 = torch.index_select(all_keys, 0, self.kv_part_1_indices).contiguous()
        v0 = torch.index_select(all_values, 0, self.kv_part_0_indices).contiguous()
        v1 = torch.index_select(all_values, 0, self.kv_part_1_indices).contiguous()

        attn_output_part0 = self.prefill_wrappers["part0"].run(q0, k0, v0)
        attn_output_part1 = self.prefill_wrappers["part1"].run(q1, k1, v1)
        attn_output = torch.cat([attn_output_part0, attn_output_part1], dim=0)
        return attn_output

    def forward_all_to_all(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        params: ParamsBase = None,
    ) -> torch.Tensor:
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
        for round_id in range(0, self.cp_size):
            if round_id > 0:
                out_buffer.zero_()
                lse_buffer.fill_(float("-inf"))

            if round_id < self.cp_size - 1:
                with torch.cuda.stream(self.communication_stream):
                    prev_rank_id = (self.cp_rank - round_id - 1) % self.cp_size
                    next_rank_id = (self.cp_rank + round_id + 1) % self.cp_size

                    if (
                        self.ub_communicator is not None
                        and self.ub_communicator.can_handle_tensor(kv_buffer)
                    ):
                        self.ub_communicator.send(kv_buffer, dst=next_rank_id)
                        self.ub_communicator.recv(remote_kv_buffer, src=prev_rank_id)
                    else:
                        # Fall back to standard collective communication
                        if self.cp_rank < next_rank_id:
                            send(kv_buffer, dst=next_rank_id, group=Group.CP)
                            recv(remote_kv_buffer, src=prev_rank_id, group=Group.CP)
                        else:
                            recv(remote_kv_buffer, src=prev_rank_id, group=Group.CP)
                            send(kv_buffer, dst=next_rank_id, group=Group.CP)
                    self.comm_events[round_id].record()

            if round_id == 0:  # local attention
                self.math_events[round_id].record()
                # TODO: make write local kvcache async

                k = k.reshape(-1, self.num_kv_heads, self.head_dim)
                v = v.reshape(-1, self.num_kv_heads, self.head_dim)

                append_paged_kv_cache(
                    append_key=k,
                    append_value=v,
                    batch_indices=params.batch_indice,
                    positions=self.all_shuffle_indices[self.cp_rank],
                    paged_kv_cache=kv_cache.kv_cache_base,
                    kv_indices=params.page_indice,
                    kv_indptr=params.prefill_page_indptr,
                    kv_last_page_len=params.paged_kv_last_page_len,
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
                src_rank = (self.cp_rank - round_id - 1) % self.cp_size
                append_paged_kv_cache(
                    append_key=remote_k,
                    append_value=remote_v,
                    batch_indices=params.batch_indice,
                    positions=self.all_shuffle_indices[src_rank],
                    paged_kv_cache=kv_cache.kv_cache_base,
                    kv_indices=params.page_indice,
                    kv_indptr=params.prefill_page_indptr,
                    kv_last_page_len=params.paged_kv_last_page_len,
                    kv_layout="HND",
                )
                if round_id > self.cp_rank:
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
            if round_id < self.cp_size - 1:
                self.communication_stream.wait_event(self.math_events[round_id])
                self.math_events[round_id].synchronize()
        return merged_out

    def forward_all_gather_with_overlap(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        params: ParamsBase = None,
    ) -> torch.Tensor:
        self.communication_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.communication_stream):
            if (
                self.ub_communicator is not None
                and self.ub_communicator.can_handle_tensor(k)
            ):
                all_keys = self.ub_communicator.all_gather(k).reshape(
                    k.shape[0] * self.cp_size, self.num_kv_heads, self.head_dim
                )
                all_values = self.ub_communicator.all_gather(v).reshape(
                    k.shape[0] * self.cp_size, self.num_kv_heads, self.head_dim
                )
            else:
                all_keys = all_gather(k, group=Group.CP).reshape(
                    k.shape[0] * self.cp_size, self.num_kv_heads, self.head_dim
                )
                all_values = all_gather(v, group=Group.CP).reshape(
                    k.shape[0] * self.cp_size, self.num_kv_heads, self.head_dim
                )
        q_reshaped = q.reshape(-1, self.num_qo_heads, self.head_dim)
        output, lse = self.prefill_wrappers["causal"].run(
            q_reshaped,
            k.reshape(-1, self.num_kv_heads, self.head_dim),
            v.reshape(-1, self.num_kv_heads, self.head_dim),
            return_lse=True,
        )
        torch.cuda.current_stream().wait_stream(self.communication_stream)

        # TODO: make write local kvcache async
        restore_k = all_keys[self.kv_restore_unpad_indices]
        restore_v = all_values[self.kv_restore_unpad_indices]
        append_paged_kv_cache(
            append_key=restore_k,
            append_value=restore_v,
            batch_indices=params.batch_indice,
            positions=params.positions,
            paged_kv_cache=kv_cache.kv_cache_base,
            kv_indices=params.page_indice,
            kv_indptr=params.prefill_page_indptr,
            kv_last_page_len=params.paged_kv_last_page_len,
            kv_layout="HND",
        )

        q0 = torch.index_select(q_reshaped, 0, self.q_part_0_indices).contiguous()
        q1 = torch.index_select(q_reshaped, 0, self.q_part_1_indices).contiguous()
        k0 = torch.index_select(all_keys, 0, self.kv_part_0_indices).contiguous()
        k1 = torch.index_select(all_keys, 0, self.kv_part_1_indices).contiguous()
        v0 = torch.index_select(all_values, 0, self.kv_part_0_indices).contiguous()
        v1 = torch.index_select(all_values, 0, self.kv_part_1_indices).contiguous()

        out_buffer = torch.zeros(
            [q.shape[0], self.num_qo_heads, self.head_dim],
            dtype=q.dtype,
            device=q.device,
        )
        lse_buffer = torch.full(
            [q.shape[0], self.num_qo_heads],
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )
        if k0.numel() > 0:
            (
                out_buffer[self.q_part_0_indices, :, :],
                lse_buffer[self.q_part_0_indices, :],
            ) = self.prefill_wrappers["non_causal_part_0"].run(
                q=q0, k=k0, v=v0, return_lse=True
            )
        if k1.numel() > 0:
            (
                out_buffer[self.q_part_1_indices, :, :],
                lse_buffer[self.q_part_1_indices, :],
            ) = self.prefill_wrappers["non_causal_part_1"].run(
                q=q1, k=k1, v=v1, return_lse=True
            )
        merged_output, merged_lse = merge_state(
            v_a=output,
            s_a=lse,
            v_b=out_buffer,
            s_b=lse_buffer,
        )
        return merged_output

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

        if self.rotate_method == CPRotateMethod.ALL_GATHER:
            attn_output = self.forward_all_gather(q, k, v, kv_cache, params)
        elif self.rotate_method == CPRotateMethod.ALLTOALL:
            attn_output = self.forward_all_to_all(q, k, v, kv_cache, params)
        elif self.rotate_method == CPRotateMethod.ALL_GATHER_WITH_OVERLAP:
            attn_output = self.forward_all_gather_with_overlap(
                q, k, v, kv_cache, params
            )
        return attn_output


class PrefillContextParallelFlashInferImpl(FMHAPrefillImplBase):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ):
        super().__init__(
            fmha_impl=ContextParallelFlashInferRaggedPrefillOp(
                attn_configs, attn_inputs, parallelism_config
            ),
            rope_kvcache_impl=FusedRopeKVCachePrefillOp(attn_configs),
            attn_inputs=attn_inputs,
        )
        self.attn_inputs = attn_inputs

    def support(self) -> bool:
        """Check if this implementation supports current inputs."""
        return self.fmha_impl.support(self.attn_inputs)

    def fmha_type(self) -> FMHAType:
        return FMHAType.CP_FLASH_INFER

    def support_cuda_graph(self) -> bool:
        return False

    @property
    def cp_rank(self) -> int:
        return self.fmha_impl.cp_rank

    @cp_rank.setter
    def cp_rank(self, value: int):
        self.fmha_impl.cp_rank = value

    @property
    def cp_size(self) -> int:
        return self.fmha_impl.cp_size

    @cp_size.setter
    def cp_size(self, value: int):
        self.fmha_impl.cp_size = value

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
        need_rope_kv_cache: bool = True,
    ) -> torch.Tensor:
        assert self.rope_kvcache_impl is not None and self.rope_params is not None
        if need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(
                qkv, self.fmha_type(), kv_cache, self.rope_params
            )
        else:
            fmha_input = qkv

        assert self.fmha_impl is not None
        res = self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)

        # delay write cachestore until local kvcache finish writing
        if (
            self.attn_inputs.cache_store_inputs
            and self.write_cache_store_impl is not None
        ):
            self.write_cache_store_impl(kv_cache)
        return res
