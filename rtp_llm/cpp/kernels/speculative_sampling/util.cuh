/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_UTILS_CUH_
#define FLASHINFER_UTILS_CUH_
#include <cuda_bf16.h>
#include <cuda_device_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>

#include "rtp_llm/cpp/kernels/speculative_sampling/exception.h"

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

// macro to turn off fp16 qk reduction to reduce binary
#ifndef FLASHINFER_ALWAYS_DISUSE_FP16_QK_REDUCTION
#define FLASHINFER_ALWAYS_DISUSE_FP16_QK_REDUCTION 0
#endif

#ifndef NDEBUG
#define FLASHINFER_CUDA_CALL(func, ...)                                                                                \
    {                                                                                                                  \
        cudaError_t e = (func);                                                                                        \
        if (e != cudaSuccess) {                                                                                        \
            std::cerr << "CUDA Error: " << cudaGetErrorString(e) << " (" << e << ") " << __FILE__ << ": line "         \
                      << __LINE__ << " at function " << STR(func) << std::endl;                                        \
            return e;                                                                                                  \
        }                                                                                                              \
    }
#else
#define FLASHINFER_CUDA_CALL(func, ...)                                                                                \
    {                                                                                                                  \
        cudaError_t e = (func);                                                                                        \
        if (e != cudaSuccess) {                                                                                        \
            return e;                                                                                                  \
        }                                                                                                              \
    }
#endif

#define DISPATCH_ALIGNED_VEC_SIZE(aligned_vec_size, ALIGNED_VEC_SIZE, ...)                                             \
    switch (aligned_vec_size) {                                                                                        \
        case 16: {                                                                                                     \
            constexpr size_t ALIGNED_VEC_SIZE = 16;                                                                    \
            __VA_ARGS__                                                                                                \
            break;                                                                                                     \
        }                                                                                                              \
        case 8: {                                                                                                      \
            constexpr size_t ALIGNED_VEC_SIZE = 8;                                                                     \
            __VA_ARGS__                                                                                                \
            break;                                                                                                     \
        }                                                                                                              \
        case 4: {                                                                                                      \
            constexpr size_t ALIGNED_VEC_SIZE = 4;                                                                     \
            __VA_ARGS__                                                                                                \
            break;                                                                                                     \
        }                                                                                                              \
        case 2: {                                                                                                      \
            constexpr size_t ALIGNED_VEC_SIZE = 2;                                                                     \
            __VA_ARGS__                                                                                                \
            break;                                                                                                     \
        }                                                                                                              \
        case 1: {                                                                                                      \
            constexpr size_t ALIGNED_VEC_SIZE = 1;                                                                     \
            __VA_ARGS__                                                                                                \
            break;                                                                                                     \
        }                                                                                                              \
        default: {                                                                                                     \
            std::ostringstream err_msg;                                                                                \
            err_msg << "Unsupported aligned_vec_size: " << aligned_vec_size;                                           \
            FLASHINFER_ERROR(err_msg.str());                                                                           \
        }                                                                                                              \
    }

namespace flashinfer {

template<typename T1, typename T2>
__forceinline__ __device__ __host__ T1 ceil_div(const T1 x, const T2 y) {
    return (x + y - 1) / y;
}

inline std::pair<int, int> GetCudaComputeCapability() {
    int device_id = 0;
    cudaGetDevice(&device_id);
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id);
    return std::make_pair(major, minor);
}
}  // namespace flashinfer

#endif  // FLASHINFER_UTILS_CUH_
