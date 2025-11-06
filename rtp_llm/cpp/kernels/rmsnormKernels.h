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

#include <assert.h>

#if USING_ROCM
#include "rtp_llm/cpp/cuda/cuda_type_utils.cuh"
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#endif
#if USING_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#endif

namespace rtp_llm {
template<typename T, typename QUANT_OUT_T>
void invokeGeneralRmsNorm(T*           out,
                          const T*     input,
                          const T*     gamma,
                          const T*     beta,
                          const float  eps,
                          const size_t tokens,
                          const size_t hidden_dim,
                          cudaStream_t stream        = 0,
                          const float* scale         = nullptr,
                          float*       dynamic_scale = nullptr,
                          QUANT_OUT_T* out_quant     = nullptr);

template<typename T, typename QUANT_OUT_T>
void invokeAddBiasResidualRmsNorm(T*           output,
                                  T*           normed_output,
                                  const T*     input,
                                  const T*     bias,
                                  const T*     residual,
                                  const T*     residual2,
                                  const T*     gamma,
                                  const T*     beta,
                                  const float  eps,
                                  const size_t tokens,
                                  const size_t hidden_dim,
                                  cudaStream_t stream        = 0,
                                  const float* scale         = nullptr,
                                  float*       dynamic_scale = nullptr,
                                  QUANT_OUT_T* out_quant     = nullptr);

template<typename T>
void invokeRmsNormWithStride(T* __restrict output,
                             const size_t out_stride,
                             const T* __restrict input,
                             const size_t in_stride,
                             const T* __restrict gamma,
                             const T* __restrict beta,
                             const float  layernorm_eps,
                             const size_t    tokens,
                             const size_t    hidden_size,
                             const size_t    norm_size,
                             cudaStream_t stream);

}  // namespace rtp_llm
