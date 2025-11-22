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
#include "rtp_llm/cpp/cuda/cuda_type_utils.cuh"
#include "rtp_llm/cpp/cuda/cuda_fp8_utils.h"
#if USING_CUDA
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif
#endif
#include "rtp_llm/cpp/kernels/kv_cache_kernels.h"

namespace rtp_llm {

// input: offset [b, m](kv_cache_block_id):
// ┌─────────────────┐
// │ batch0: [0,1,2] │
// │ batch1: [3,4,5] │
// │ batch2: [6,7,8] │
// └─────────────────┘
// flatten to (k,v block offset) -> k block id and v block id:
// output: offset_addr [b, 2, m](kv_block_array.data, kv_block_array.mPrimaryPool is the real kv cache data address):
// ┌─────────────────────────────────────────---┐
// │ batch0_K: [0,1,2]  batch0_V: [0+Δ,1+Δ,2+Δ] │
// │ batch1_K: [3,4,5]  batch1_V: [3+Δ,4+Δ,5+Δ] │
// │ batch2_K: [6,7,8]  batch2_V: [6+Δ,7+Δ,8+Δ] │
// └─────────────────────────────────────────---┘
// Δ is kv_block_offset (layer_num * block_num_per_layer).

__global__ void ConvertOffsetToBlockArrayData(int32_t*   offset_addr,
                                              const int* offset,  // [b, m]
                                              int        batch_size,
                                              int        max_block_num) {
    const int batch_stride = 2 * max_block_num;
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size * max_block_num;
         index += blockDim.x * gridDim.x) {
        const int     batch_index                     = index / max_block_num;
        const int     col_index                       = index % max_block_num;
        const int32_t block_offset                    = (int32_t)offset[batch_index * max_block_num + col_index];
        const int32_t block_addr_index                = (int32_t)batch_index * batch_stride + col_index;
        offset_addr[block_addr_index]                 = block_offset * 2;
        offset_addr[block_addr_index + max_block_num] = block_offset * 2 + 1;
    }
}

void invokeConvertOffsetToBlockArrayData(int32_t*     offset_addr,  // [b, 2, m]
                                         const int*   offset,       // [b, m]
                                         int          batch_size,
                                         int          max_block_num,
                                         cudaStream_t stream) {
    dim3 grid(min(batch_size, 65536));
    dim3 block(min(max_block_num, 1024));
    ConvertOffsetToBlockArrayData<<<grid, block, 0, stream>>>(offset_addr,  // [b, 2, m]
                                                              offset,       // [b, m]
                                                              batch_size,
                                                              max_block_num);
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
}

}  // namespace rtp_llm