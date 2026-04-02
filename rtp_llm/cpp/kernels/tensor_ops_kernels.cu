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
#include "rtp_llm/cpp/utils/math_utils.h"
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
#include "rtp_llm/cpp/kernels/tensor_ops_kernels.h"

#if USING_CUDA
#include "rtp_llm/cpp/kernels/vec_dtypes.cuh"
#endif

#if USING_ROCM
#include "rtp_llm/cpp/kernels/rocm_utils/vec_dtypes_hip.h"
#endif

namespace rtp_llm {

template<typename T>
__global__ void transposeAxis12(
    T* out, T* in, const size_t dim0, const size_t dim1, const size_t dim2, const size_t dim3, const size_t elem_num) {
    size_t index = static_cast<size_t>(threadIdx.x) + static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x);
    if (index < elem_num) {
        const size_t input_dim3_index = index % dim3;
        index                         = (index - input_dim3_index) / dim3;
        const size_t input_dim2_index = index % dim2;
        index                         = (index - input_dim2_index) / dim2;
        const size_t input_dim1_index = index % dim1;
        index                         = (index - input_dim1_index) / dim1;
        const size_t input_dim0_index = index % dim0;

        const size_t src_idx = input_dim0_index * dim1 * dim2 * dim3 + input_dim1_index * dim2 * dim3
                               + input_dim2_index * dim3 + input_dim3_index;
        const size_t dst_idx = input_dim0_index * dim1 * dim2 * dim3 + input_dim2_index * dim1 * dim3
                               + input_dim1_index * dim3 + input_dim3_index;
        out[dst_idx] = in[src_idx];
    }
}

template<typename T>
void invokeTransposeAxis12(
    T* out, T* in, const size_t dim0, const size_t dim1, const size_t dim2, const size_t dim_3, cudaStream_t stream) {
    const size_t elem_num   = dim0 * dim1 * dim2 * dim_3;
    const size_t thread_num = 512;
    const size_t block_num  = ceil_div<size_t>(elem_num, thread_num);
    dim3         block(thread_num);
    dim3         grid(block_num);
    transposeAxis12<<<grid, block, 0, stream>>>(out, in, dim0, dim1, dim2, dim_3, elem_num);
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
}

template void invokeTransposeAxis12(float*       out,
                                    float*       in,
                                    const size_t dim0,
                                    const size_t dim1,
                                    const size_t dim2,
                                    const size_t dim_3,
                                    cudaStream_t stream);

template void invokeTransposeAxis12(half*        out,
                                    half*        in,
                                    const size_t dim0,
                                    const size_t dim1,
                                    const size_t dim2,
                                    const size_t dim_3,
                                    cudaStream_t stream);

template void invokeTransposeAxis12(size_t*      out,
                                    size_t*      in,
                                    const size_t dim0,
                                    const size_t dim1,
                                    const size_t dim2,
                                    const size_t dim_3,
                                    cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeTransposeAxis12(__nv_bfloat16* out,
                                    __nv_bfloat16* in,
                                    const size_t   dim0,
                                    const size_t   dim1,
                                    const size_t   dim2,
                                    const size_t   dim_3,
                                    cudaStream_t   stream);
#endif

}  // namespace rtp_llm
