/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>

#if USING_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif
#if USING_ROCM
#include <hip/hip_runtime.h>
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#endif

namespace rtp_llm {

// KV cache operations for paged attention
void invokeConvertOffsetToBlockArrayData(int32_t*     offset_addr,  // [b, 2, m]
                                         const int*   offset,       // [b, m]
                                         int          batch_size,
                                         int          max_block_num,
                                         cudaStream_t stream);

// Reuse KV cache indexed batched kernel
template<typename T>
void invokeReuseKVCacheIndexedBatched(
    T*             final_compressed_kv,      // [total_len, compressed_kv_dim]
    T*             final_k_pe,               // [total_len, k_pe_dim]
    const T*       compressed_kv,            // [compressed_kv_len, compressed_kv_dim]
    const T*       k_pe,                     // [compressed_kv_len, k_pe_dim]
    const T*       kv_cache_base,            // [num_blocks, tokens_per_block, kv_dim]
    const int32_t* reuse_cache_page_indice,  // [num_reuse_blocks]
    const int32_t* batch_reuse_info_vec,     // [num_batches, 4] (batch_idx, reuse_len, block_start_idx, blocks_needed)
    const int32_t* qo_indptr,                // [batch_size + 1]
    int            num_batches,
    int            total_final_len,  // final_compressed_kv.size(0)
    int            compressed_kv_dim,
    int            k_pe_dim,
    int            tokens_per_block,
    int            kv_dim,  // compressed_kv_dim + k_pe_dim
    cudaStream_t   stream);

}  // namespace rtp_llm