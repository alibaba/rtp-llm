from typing import Optional

import torch

from rtp_llm.models_py.modules.factory.attention import common

# Select implementation based on CP method
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.allgather_cp_impl import (
    PCPAllGatherAttnOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.allgather_overlap_impl import (
    PCPAllGatherOverlapAttnOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.alltoall_cp_impl import (
    PCPAll2AllAttnOp,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs, CPRotateMethod, FMHAType, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCachePrefillOpQKVOut,
    KVCache,
    PyAttentionInputs,
)

impl_map = {
    CPRotateMethod.ALL_GATHER: PCPAllGatherAttnOp,
    CPRotateMethod.ALLTOALL: PCPAll2AllAttnOp,
    CPRotateMethod.ALL_GATHER_WITH_OVERLAP: PCPAllGatherOverlapAttnOp,
}


class CPFlashInferImpl(FMHAImplBase):
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
    ):
        # Create implementations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache

        method = parallelism_config.prefill_cp_config.method
        self.fmha_impl = impl_map[method](attn_configs, attn_inputs, parallelism_config)
        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpQKVOut(attn_configs)

        # Store input info
        self.attn_inputs = attn_inputs

        # Create params
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

        self.attn_inputs = attn_inputs

    def support(self) -> bool:
        """Check if this implementation supports current inputs."""
        return self.fmha_impl.support(self.attn_inputs)

    @classmethod
    def support(cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs):
        return True

    def fmha_type(self) -> FMHAType:
        return FMHAType.CP_FLASH_INFER

    @classmethod
    def support_prefill_cp(cls) -> bool:
        return True

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
        need_rope_kv_cache: bool = True,
    ) -> torch.Tensor:
        assert self.rope_kvcache_impl is not None and self.rope_params is not None
        if need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        assert self.fmha_impl is not None
        output = self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)

        # Delay write to cache store until local kv cache finishes writing
        if (
            self.attn_inputs.cache_store_inputs
            and self.write_cache_store_impl is not None
        ):
            self.write_cache_store_impl(kv_cache)
        return output
