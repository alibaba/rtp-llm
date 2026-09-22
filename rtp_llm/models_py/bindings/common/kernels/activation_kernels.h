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

#if USING_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif
#if USING_ROCM
#include "rtp_llm/models_py/bindings/rocm/cuda_shims.h"
#endif
#include <cstdint>
#include <stdlib.h>

namespace rtp_llm {

template<typename T>
void invokeAddBias(T* output, const T* bias, size_t numel, size_t hidden_size, cudaStream_t stream);

template<typename T>
void invokeAddBiasGelu(T* output, const T* bias, size_t numel, size_t hidden_size, cudaStream_t stream);

#if USING_CUDA
template<typename T>
void invokeAddBiasGeluQuantFp8(const T*     input,
                               const T*     bias,
                               void*        output,
                               uint32_t*    scales,
                               size_t       rows,
                               size_t       hidden_size,
                               size_t       scale_stride,
                               cudaStream_t stream);
#endif

template<typename T>
void invokeAddBiasSoftMax(T*           logits,
                          const T*     bias,
                          const int*   end_ids,
                          const bool*  finished,
                          const int    m,
                          const int    n_padded,
                          const int    n,
                          cudaStream_t stream);

}  // namespace rtp_llm
