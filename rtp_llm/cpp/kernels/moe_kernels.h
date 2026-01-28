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

// MoE (Mixture of Experts) operations

// Scatter-add operation for expert routing
template<typename T>
void invokeScatterAdd(
    T const* src, int N, int K, int32_t const* index, T* out, bool use_stable_scatter_add, cudaStream_t stream);

// Slice and copy operation along dimension 1
template<typename T>
void invokeSliceDim1Copy(T const* src, int dim0, int dim1, int dim1_start, int dim1_size, T* out, cudaStream_t stream);

// Fake expert load balancing for testing/debugging
void fake_balance_expert(int*         expert,
                         float*       expert_scales,
                         int          dp_rank,
                         int          dp_size,
                         int          ep_size,
                         int          expert_num,
                         int          size,
                         cudaStream_t stream);

void fake_balance_expert(int64_t*     expert,
                         float*       expert_scales,
                         int          dp_rank,
                         int          dp_size,
                         int          ep_size,
                         int          expert_num,
                         int          size,
                         cudaStream_t stream);

// Expert indexing operations
void genSourceRowRevert(
    int64_t* expert_rows, int* expert_rows_dst, size_t token_num, size_t top_k, int start_expert, cudaStream_t stream);

}  // namespace rtp_llm