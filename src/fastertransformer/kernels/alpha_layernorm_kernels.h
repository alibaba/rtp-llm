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
#include "src/fastertransformer/cuda/cuda_type_utils.cuh"
#include <assert.h>
#if USING_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

namespace fastertransformer{

template<typename T>
void invokeAlphaAddBiasResidualLayerNorm(T*           out,
                                         const T*     input,
                                         const T*     residual1,
                                         const T*     bias,
                                         const T*     gamma,
                                         const T*     beta,
                                         const T      alpha,
                                         const int    m,
                                         const int    n,
                                         cudaStream_t stream);

template<typename T>
void invokeAddBiasResidualLayerNorm(T*           out,
                                    const T*     input,
                                    const T*     bias,
                                    const T*     gamma,
                                    const T*     beta,
                                    const float  layernorm_eps,
                                    const int    m,
                                    const int    n,
                                    cudaStream_t stream);

template<typename T>
void invokeGeneralLayerNormWithPadding(T*           out,
                                       const T*     input,
                                       const T*     gamma,
                                       const T*     beta,
                                       const float  layernorm_eps,
                                       const int    m,
                                       const int    real_n,
                                       const int    padding_n,
                                       float*       scale,
                                       float*       dynamic_scale,
                                       const int    int8_mode,
                                       cudaStream_t stream,
                                       int          opt_version = 2);
}