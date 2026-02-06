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

// Tensor transpose operations
template<typename T>
void invokeTransposeAxis012(
    T* out, T* in, const size_t dim0, const size_t dim1, const size_t dim2, cudaStream_t stream);

// from [b, s, h, d] to [b, h, s, d]
template<typename T>
void invokeTransposeAxis12(
    T* out, T* in, const size_t dim0, const size_t dim1, const size_t dim2, const size_t dim_3, cudaStream_t stream);

template<typename T>
void invokeTransposeAxis01(T* out, T* in, const size_t dim0, const size_t dim1, cudaStream_t stream);

// Sequence operations
template<typename T>
void invokeLookupHiddenStateOfLastToken(T*           from_tensor,
                                        const T*     hidden_state,
                                        const int*   input_lengths,
                                        const size_t batch_size,
                                        const size_t hidden_units,
                                        const size_t idx_offset,
                                        cudaStream_t stream);

}  // namespace rtp_llm
