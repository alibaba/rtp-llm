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

#include "rtp_llm/cpp/cuda/cuda_type_utils.cuh"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cuda/memory_utils.h"

namespace rtp_llm {

template<typename T>
void deviceMalloc(T** ptr, size_t size, bool is_random_initialize) {
    // RTP_LLM_CHECK_WITH_INFO(size >= ((size_t)0), "Ask deviceMalloc size " + std::to_string(size) + "< 0 is
    // invalid.");
    check_cuda_value(cudaMalloc((void**)(ptr), sizeof(T) * size));
    if (is_random_initialize) {
        // Random initialization removed - function was unused
        check_cuda_value(cudaMemset(static_cast<void*>(*ptr), 0, sizeof(T) * size));
    }
}

template void deviceMalloc(float** ptr, size_t size, bool is_random_initialize);
template void deviceMalloc(half** ptr, size_t size, bool is_random_initialize);
#ifdef ENABLE_BF16
template void deviceMalloc(__nv_bfloat16** ptr, size_t size, bool is_random_initialize);
#endif
template void deviceMalloc(uint16_t** ptr, size_t size, bool is_random_initialize);
template void deviceMalloc(int** ptr, size_t size, bool is_random_initialize);
template void deviceMalloc(bool** ptr, size_t size, bool is_random_initialize);
template void deviceMalloc(char** ptr, size_t size, bool is_random_initialize);
template void deviceMalloc(int8_t** ptr, size_t size, bool is_random_initialize);
template void deviceMalloc(uint8_t** ptr, size_t size, bool is_random_initialize);
#ifdef ENABLE_FP8
template void deviceMalloc(__nv_fp8_e4m3** ptr, size_t size, bool is_random_initialize);
#endif

template<typename T>
void deviceFree(T*& ptr) {
    if (ptr != NULL) {
        check_cuda_value(cudaFree(ptr));
        ptr = NULL;
    }
}

template void deviceFree(float*& ptr);
template void deviceFree(half*& ptr);
#ifdef ENABLE_BF16
template void deviceFree(__nv_bfloat16*& ptr);
#endif
template void deviceFree(unsigned short*& ptr);
template void deviceFree(int*& ptr);
template void deviceFree(bool*& ptr);
template void deviceFree(char*& ptr);
template void deviceFree(int8_t*& ptr);
template void deviceFree(uint8_t*& ptr);
#ifdef ENABLE_FP8
template void deviceFree(__nv_fp8_e4m3*& ptr);
#endif

}  // namespace rtp_llm
