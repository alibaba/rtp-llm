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
#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif
#endif
#if USING_ROCM
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#ifdef ENABLE_FP8
#include <hip/hip_fp8.h>
#endif
#endif

namespace rtp_llm {

template<typename T>
void invokeQuantization(
    int8_t* dst, const T* src, const int64_t size, const float* scalePtr, cudaStream_t stream = 0, int maxGirdSize = 0);

template<typename T>
void invokePerTokenQuantization(int8_t*       dst,
                                const T*      src,
                                const int64_t numRows,
                                const int64_t numCols,
                                float*        scalePtr,
                                const float*  smoother,
                                const float*  shift,
                                cudaStream_t  stream = 0);

}  // namespace rtp_llm
