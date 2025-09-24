/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <assert.h>

#if USING_CUDA
#include <cuda_runtime.h>
#endif

#if USING_ROCM
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#endif

#include <iostream>

#define MAX_ALL_REDUCE_BLOCKS 24
#define FLAG(a) ((uint32_t)((a) % 0x146))
#define MAX_RANKS_PER_NODE 16
#define WARP_SIZE 32
#define DEFAULT_BLOCK_SIZE 1024

namespace rtp_llm {

enum AllReduceStrategyType {
    ONESHOT,
    TWOSHOT,
};

struct CustomAllReduceParameters {
    size_t                elts_total_num;
    size_t                data_type_size;
    size_t                elts_per_rank;
    size_t                elts_per_block;
    size_t                rank_offset;
    size_t                rank, local_rank;
    size_t                ranks_per_node;
    size_t                max_elts_total_size;
    uint32_t              barrier_flag;
    uint32_t*             peer_barrier_ptrs[MAX_RANKS_PER_NODE];
    void*                 peer_comm_buffer_ptrs[MAX_RANKS_PER_NODE];
    void*                 local_output_buffer_ptr;
    AllReduceStrategyType kernel_algo;
};

template<typename T1, typename T2>
inline size_t divUp(const T1& a, const T2& n) {
    size_t tmp_a = static_cast<size_t>(a);
    size_t tmp_n = static_cast<size_t>(n);
    return (tmp_a + tmp_n - 1) / tmp_n;
}

inline size_t roundUp(size_t a, size_t n) {
    return divUp(a, n) * n;
}

template<typename T, size_t RANKS_PER_NODE>
void invokeCustomAllReduceKernel(CustomAllReduceParameters* param, uint32_t barrier_flag, cudaStream_t stream);

template<typename T>
void invokeCustomAllReduceDispatch(CustomAllReduceParameters* param, uint32_t barrier_flag, cudaStream_t stream);

void kernelLaunchConfig(CustomAllReduceParameters* param, size_t& blocks_per_grid, size_t& threads_per_block);

}  // namespace rtp_llm
