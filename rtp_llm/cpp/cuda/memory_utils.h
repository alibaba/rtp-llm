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

#include "rtp_llm/cpp/cuda/cuda_fp8_utils.h"
#if USING_CUDA
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#endif
#if USING_ROCM
#include "rtp_llm/cpp/rocm/hip_host_utils.h"
#endif

namespace rtp_llm {

template<typename T>
void deviceMalloc(T** ptr, size_t size, bool is_random_initialize = true);

template<typename T>
void deviceFree(T*& ptr);


}  // namespace rtp_llm
