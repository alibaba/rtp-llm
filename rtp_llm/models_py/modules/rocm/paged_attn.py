"""
* Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
* Copyright (C) 2024-2025, The vLLM team.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
"""

from typing import List, Optional, Tuple
import torch
import aiter

_PARTITION_SIZE_ROCM = 256
_DEVICE_PROPERTIES = torch.cuda.get_device_properties("cuda")
_ON_NAVI = (
    hasattr(_DEVICE_PROPERTIES, "gcnArchName")
    and "gfx1" in torch.cuda.get_device_properties("cuda").gcnArchName
)
def _use_rocm_custom_paged_attention(
    qtype: torch.dtype,
    head_size: int,
    block_size: int,
    gqa_ratio: int,
    max_seq_len: int,
) -> bool:
    # rocm custom page attention not support on navi (gfx1*)
    return (
        not _ON_NAVI
        and (qtype == torch.float16 or qtype == torch.bfloat16)
        and (head_size == 64 or head_size == 128)
        and (block_size == 16 or block_size == 32)
        and (gqa_ratio >= 1 and gqa_ratio <= 16)
        and max_seq_len <= 65536
    )



def forward_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    kv_cache_dtype: str,
    num_kv_heads: int,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
    fp8_out_scale=None,
) -> torch.Tensor:
    # Whether to use rocm custom paged attention or not
    # cache dimensions:
    # rpt-llm  [block_nums,local_head_num_kv, seq_size_per_block,size_per_head]
    # aiter support [num_blocks, num_heads, head_size, block_size]  
    num_seqs, num_heads, head_size = query.shape
    block_size = value_cache.shape[2]
    gqa_ratio = num_heads // num_kv_heads
    use_custom = _use_rocm_custom_paged_attention(
        query.dtype, head_size, block_size, gqa_ratio, max_seq_len
    )
    output = torch.empty_like(query)
    if use_custom:
        max_num_partitions = (
            max_seq_len + _PARTITION_SIZE_ROCM - 1
        ) // _PARTITION_SIZE_ROCM
        assert _PARTITION_SIZE_ROCM % block_size == 0
        tmp_output = torch.empty(
            size=(num_seqs, num_heads, max_num_partitions, head_size),
            dtype=output.dtype,
            device=output.device,
        )
        exp_sums = torch.empty(
            size=(num_seqs, num_heads, max_num_partitions),
            dtype=torch.float32,
            device=output.device,
        )
        max_logits = torch.ones_like(exp_sums)
        cpa_fp8_out = False
        # if fp8_out_scale is not None:
        #     output = torch.empty_like(output, dtype= dtypes.fp8)
        #     cpa_fp8_out = True
        kv_cache_dtype ="auto"
        key_cache_reshaped = key_cache.permute(0,1,3,2)
        value_cache_reshaped = value_cache.permute(0,1,3,2)
                
        aiter.paged_attention_rocm(
            output,
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache_reshaped,
            value_cache_reshaped,
            num_kv_heads,
            float(scale),
            block_tables,
            seq_lens,
            block_size,
            max_seq_len,
            alibi_slopes,
            kv_cache_dtype,
            k_scale,
            v_scale,
            fp8_out_scale if cpa_fp8_out else None,
            _PARTITION_SIZE_ROCM,
        )
        if cpa_fp8_out:
            return output.view(num_seqs, num_heads * head_size)
    else:
        assert use_custom==True,"rocm custom paged attention should be used"
    
    #output shape [num_seqs, num_heads, head_size]  
    output_reshaped = output.view(num_seqs, -1)
    return output_reshaped
