/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <assert.h>
#include <type_traits>
#include <memory>

#include "rtp_llm/models_py/bindings/cuda/cuda_type_utils.cuh"
#include "rtp_llm/models_py/bindings/cuda/cuda_fp8_utils.h"
#if USING_CUDA
#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif
#endif
#include "rtp_llm/models_py/bindings/common/kernels/moe_kernels.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

#if USING_CUDA
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#endif

#if USING_ROCM
#include "rtp_llm/models_py/bindings/rocm/cuda_shims.h"
#include "rtp_llm/models_py/bindings/rocm/hip_host_utils.h"
#endif

namespace rtp_llm {

template<typename T>
__global__ void fakeBalanceExpertKernel(T*     expert,
                                        float* expert_scales,
                                        int    ep_size,
                                        int    local_expert_num,
                                        int    dest_ep_rank_offset,
                                        int    dest_local_expert_idx_offset,
                                        int    size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        int dest_ep_rank          = (dest_ep_rank_offset + index % ep_size) % ep_size;
        int dest_local_expert_idx = (dest_local_expert_idx_offset + index / ep_size) % local_expert_num;
        expert[index]             = dest_ep_rank * local_expert_num + dest_local_expert_idx;
        expert_scales[index]      = 1.0f;
    }
}

void fake_balance_expert(int*         expert,
                         float*       expert_scales,
                         int          dp_rank,
                         int          dp_size,
                         int          ep_size,
                         int          expert_num,
                         int          size,
                         cudaStream_t stream) {

    int local_expert_num    = expert_num / ep_size;
    int dest_ep_rank_offset = ep_size * dp_rank / dp_size;
    int dest_local_expert_idx_offset =
        static_cast<int>(dp_rank * std::max(static_cast<float>(local_expert_num) / dp_size, 1.0f)) % local_expert_num;
    fakeBalanceExpertKernel<int><<<(size + 255) / 256, 256, 0, stream>>>(
        expert, expert_scales, ep_size, local_expert_num, dest_ep_rank_offset, dest_local_expert_idx_offset, size);
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
}

void fake_balance_expert(int64_t*     expert,
                         float*       expert_scales,
                         int          dp_rank,
                         int          dp_size,
                         int          ep_size,
                         int          expert_num,
                         int          size,
                         cudaStream_t stream) {
    int local_expert_num    = expert_num / ep_size;
    int dest_ep_rank_offset = ep_size * dp_rank / dp_size;
    int dest_local_expert_idx_offset =
        static_cast<int>(dp_rank * std::max(static_cast<float>(local_expert_num) / dp_size, 1.0f)) % local_expert_num;
    fakeBalanceExpertKernel<int64_t><<<(size + 255) / 256, 256, 0, stream>>>(
        expert, expert_scales, ep_size, local_expert_num, dest_ep_rank_offset, dest_local_expert_idx_offset, size);
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
}

}  // namespace rtp_llm
