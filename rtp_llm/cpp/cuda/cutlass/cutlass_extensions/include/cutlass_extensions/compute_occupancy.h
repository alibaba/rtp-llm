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
#pragma once

#include <cuda_runtime_api.h>

#include "cutlass/device_kernel.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"

namespace tensorrt_llm
{
namespace cutlass_extensions
{

template <typename GemmKernel, bool enable_cutlass_3x = false>
inline int compute_occupancy_for_kernel()
{

    int smem_size = int(sizeof(typename GemmKernel::SharedStorage));

    if (smem_size > (48 << 10))
    {
        cudaFuncAttributes attr;
        int max_smem_per_block = rtp_llm::getMaxSharedMemoryPerBlockOptin();
        if constexpr (enable_cutlass_3x)
        {
            check_cuda_value(cudaFuncGetAttributes(&attr, cutlass::device_kernel<GemmKernel>));
        }
        else
        {
            check_cuda_value(cudaFuncGetAttributes(&attr, cutlass::Kernel<GemmKernel>));
        }
        if (smem_size + attr.sharedSizeBytes >= static_cast<size_t>(max_smem_per_block))
        {
            // This should mean that
            // cudaFuncSetAttribute(cutlass::Kernel<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size)
            // wouldn't work. In that case, we return an occupancy of 0. This will cause the heuristic to ignore this
            // configuration.
            return 0;
        }

        if constexpr (enable_cutlass_3x)
        {
            check_cuda_value(cudaFuncSetAttribute(
                cutlass::device_kernel<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        else
        {
            check_cuda_value(cudaFuncSetAttribute(
                cutlass::Kernel<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
    }

    int max_active_blocks = -1;
    if constexpr (enable_cutlass_3x)
    {
        check_cuda_value(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, cutlass::device_kernel<GemmKernel>,
                128 * (GemmKernel::NumLoadWarpGroups + GemmKernel::NumMmaWarpGroups), smem_size));
    }
    else
    {
        check_cuda_value(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks, cutlass::Kernel<GemmKernel>, GemmKernel::kThreadCount, smem_size));
    }

    return max_active_blocks;
}

} // namespace cutlass_extensions
} // namespace tensorrt_llm
