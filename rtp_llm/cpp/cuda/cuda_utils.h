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

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/cuda/launch_utils.h"

#include <cstddef>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nvml.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <functional>

#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif

#ifndef _WIN32  // Linux
#include <sys/sysinfo.h>
#endif  // not WIN32
#include <vector>
#ifdef _WIN32  // Windows
#include <windows.h>
#undef ERROR  // A Windows header file defines ERROR as 0, but it's used in our logger.h enum. Logging breaks without
              // this undef.
#endif        // WIN32

namespace rtp_llm {

// workspace for cublas gemm : 32MB
#define CUBLAS_WORKSPACE_SIZE 33554432

typedef struct __align__(4) {
    half x, y, z, w;
} half4;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct __align__(16) Float4_ {
    float2 x;
    float2 y;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct __align__(32) Float8_ {
    float2 x;
    float2 y;
    float2 z;
    float2 w;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_BF16
struct __align__(8) bf16_4_t {
    __nv_bfloat162 x;
    __nv_bfloat162 y;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct __align__(16) bf16_8_t {
    __nv_bfloat162 x;
    __nv_bfloat162 y;
    __nv_bfloat162 z;
    __nv_bfloat162 w;
};
#endif

#ifdef ENABLE_FP8
using fp8_2_t = __nv_fp8x2_e4m3;
using fp8_4_t = __nv_fp8x4_e4m3;

struct __align__(8) fp8_8_t {
    __nv_fp8x2_e4m3 x;
    __nv_fp8x2_e4m3 y;
    __nv_fp8x2_e4m3 z;
    __nv_fp8x2_e4m3 w;
};
#endif

/* **************************** type definition ***************************** */

enum CublasDataType {
    FLOAT_DATATYPE    = 0,
    HALF_DATATYPE     = 1,
    BFLOAT16_DATATYPE = 2,
    INT8_DATATYPE     = 3,
    FP8_DATATYPE      = 4
};

enum FtCudaDataType {
    FP32 = 0,
    FP16 = 1,
    BF16 = 2,
    INT8 = 3,
    FP8  = 4
};

enum class OperationType {
    FP32,
    FP16,
    BF16,
    INT8,
    FP8
};

template<typename T>
inline T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

inline int div_up(int a, int b) {
    return ceil_div<int>(a, b);
}

template<typename T,
         typename U,
         typename = std::enable_if_t<std::is_integral<T>::value>,
         typename = std::enable_if_t<std::is_integral<U>::value>>
auto constexpr ceilDiv(T numerator, U denominator) {
    return (numerator + denominator - 1) / denominator;
}

inline size_t pad(const size_t& input, const size_t& alignment) {
    return alignment * ((input + alignment - 1) / alignment);
}

inline size_t pad_to_multiple_of_16(const size_t& input) {
    return pad(input, 16);
}

inline size_t pad_to_multiple_of_128(const size_t& input) {
    return pad(input, 128);
}

template<typename T>
void check(T result, const char* const file, int const line);
#define check_cuda_value(val) rtp_llm::check((val), __FILE__, __LINE__)

void syncAndCheckInDebug(const char* const file, int const line);
#define check_cuda_error() rtp_llm::syncAndCheckInDebug(__FILE__, __LINE__)

int  get_sm();
bool is_sm70();
bool is_sm8x();
bool is_sm90();
bool is_sm100();

float                      timing_function(const std::function<void(cudaStream_t)>& operation,
                                           int64_t                                  timing_iterations,
                                           cudaStream_t                             stream);
int                        getDevice();
int                        getDeviceCount();
int                        currentDeviceId();
void                       priorityRange(int* low_priority, int* high_priority, int device_id = -1);
std::tuple<size_t, size_t> getDeviceMemoryInfo(bool const useUvm);
bool                       shared_mem_sufficient(int smem_size);
std::string                getDriverVersion();
int                        getCudaVersion();
bool                       checkAllNVLinks(std::vector<size_t> device_ids);
bool                       checkOnSameNumaNodes(std::vector<size_t> device_ids);
int                        getVisibleDeviceNum();
bool                       checkP2PAvailable(const std::vector<size_t>& tp_ranks, size_t rank);
int                        getMultiProcessorCount(int device_id = -1);
int                        getMaxSharedMemoryPerMultiprocessor(int device_id = -1);
int                        getMaxSharedMemoryPerBlockOptin(int device_id = -1);
int                        getMaxThreadsPerMultiprocessor(int device_id = -1);
int                        getMaxBlocksPerMultiprocessor(int device_id = -1);
int                        getComputeCapabilityMajor(int device_id = -1);
int                        getComputeCapabilityMinor(int device_id = -1);
std::pair<int, int>        getComputeCapability(int device_id = -1);

template<typename T>
T getCudaValue(const T* ptr, int index) {
    T tmp;
    check_cuda_value(cudaMemcpy(&tmp, ptr + index, sizeof(T), cudaMemcpyDeviceToHost));
    return tmp;
}

template<typename T>
void setCudaValue(T* ptr, int index, T value) {
    check_cuda_value(cudaMemcpy(ptr + index, &value, sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
CublasDataType getCublasDataType() {
    if (std::is_same<T, half>::value) {
        return HALF_DATATYPE;
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        return BFLOAT16_DATATYPE;
    }
#endif
    else if (std::is_same<T, float>::value) {
        return FLOAT_DATATYPE;
    } else {
        RTP_LLM_CHECK(false);
        return FLOAT_DATATYPE;
    }
}

template<typename T>
cudaDataType_t getCudaDataType() {
    if (std::is_same<T, half>::value) {
        return CUDA_R_16F;
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        return CUDA_R_16BF;
    }
#endif
    else if (std::is_same<T, float>::value) {
        return CUDA_R_32F;
    } else {
        RTP_LLM_CHECK(false);
        return CUDA_R_32F;
    }
}

template<CublasDataType T>
struct getTypeFromCudaDataType {
    using Type = float;
};

template<>
struct getTypeFromCudaDataType<HALF_DATATYPE> {
    using Type = half;
};

#ifdef ENABLE_BF16
template<>
struct getTypeFromCudaDataType<BFLOAT16_DATATYPE> {
    using Type = __nv_bfloat16;
};
#endif

// clang-format off
template<typename T> struct packed_type_2;
template <>          struct packed_type_2<float>         { using type = float; }; // we don't need to pack float by default
template <>          struct packed_type_2<half>          { using type = half2; };

#ifdef ENABLE_BF16
template<>
struct packed_type_2<__nv_bfloat16> {
    using type = __nv_bfloat162;
};
#endif

template <typename T, int N>
struct packed_type;

template <typename T>
struct packed_type<T, 1>
{
    using type = T;
};

template <>
struct packed_type<int8_t, 1>
{
    using type = int8_t;
};

template <>
struct packed_type<int8_t, 2>
{
    using type = int16_t;
};

template <>
struct packed_type<int8_t, 4>
{
    using type = int32_t;
};

template <>
struct packed_type<int8_t, 8>
{
    using type = int64_t;
};

#ifdef ENABLE_FP8

template <>
struct packed_type<__nv_fp8_e4m3, 1>
{
    using type = __nv_fp8_e4m3;
};

template <>
struct packed_type<__nv_fp8_e4m3, 2>
{
    using type = fp8_2_t;
};

template <>
struct packed_type<__nv_fp8_e4m3, 4>
{
    using type = fp8_4_t;
};

template <>
struct packed_type<__nv_fp8_e4m3, 8>
{
    using type = fp8_8_t;
};
#endif // ENABLE_FP8

template <>
struct packed_type<uint16_t, 2>
{
    using type = uint32_t;
};

template <>
struct packed_type<uint16_t, 4>
{
    using type = uint2;
};

template <>
struct packed_type<uint16_t, 8>
{
    using type = uint4;
};

template <>
struct packed_type<half, 2>
{
    using type = uint32_t;
};

template <>
struct packed_type<half, 4>
{
    using type = uint2;
};

template <>
struct packed_type<half, 8>
{
    using type = uint4;
};

#ifdef ENABLE_BF16
template <>
struct packed_type<__nv_bfloat16, 2>
{
    using type = __nv_bfloat162;
};

template <>
struct packed_type<__nv_bfloat16, 4>
{
    using type = bf16_4_t;
};

template <>
struct packed_type<__nv_bfloat16, 8>
{
    using type = bf16_8_t;
};
#endif
template <>
struct packed_type<float, 2>
{
    using type = float2;
};

template <>
struct packed_type<float, 4>
{
    using type = float4;
};

template <>
struct packed_type<float, 8>
{
    using type = Float8_;
};

template<typename T> struct num_elems;
template <>          struct num_elems<float>           { static constexpr int value = 1; };
template <>          struct num_elems<float2>          { static constexpr int value = 2; };
template <>          struct num_elems<float4>          { static constexpr int value = 4; };
template <>          struct num_elems<Float4_>          { static constexpr int value = 4; };
template <>          struct num_elems<Float8_>          { static constexpr int value = 8; };

template <>          struct num_elems<half>            { static constexpr int value = 1; };
template <>          struct num_elems<half2>           { static constexpr int value = 2; };
template <>          struct num_elems<uint32_t>           { static constexpr int value = 2; };
template <>          struct num_elems<int32_t>           { static constexpr int value = 2; };
template <>          struct num_elems<int64_t>           { static constexpr int value = 4; };
template <>          struct num_elems<uint2>           { static constexpr int value = 4; };
template <>          struct num_elems<uint4>           { static constexpr int value = 8; };

#ifdef ENABLE_BF16
template <>          struct num_elems<__nv_bfloat16>   { static constexpr int value = 1; };
template <>          struct num_elems<__nv_bfloat162>  { static constexpr int value = 2; };
template <>          struct num_elems<bf16_4_t>  { static constexpr int value = 4; };
template <>          struct num_elems<bf16_8_t>  { static constexpr int value = 8; };
#endif

#ifdef ENABLE_FP8
template <>          struct num_elems<__nv_fp8_e4m3>   { static constexpr int value = 1; };
template <>          struct num_elems<fp8_2_t>   { static constexpr int value = 2; };
template <>          struct num_elems<fp8_4_t>   { static constexpr int value = 4; };
template <>          struct num_elems<fp8_8_t>   { static constexpr int value = 8; };
#endif

template<typename T, int num> struct packed_as;
template<typename T>          struct packed_as<T, 1>              { using type = T; };
template<>                    struct packed_as<half,  2>          { using type = half2; };
template<>                    struct packed_as<float,  2>         { using type = float2; };
template<>                    struct packed_as<int8_t, 2>         { using type = int16_t; };
template<>                    struct packed_as<int32_t, 2>        { using type = int2; };
template<>                    struct packed_as<half2, 1>          { using type = half; };
template<>                    struct packed_as<float2, 1>         { using type = float; };
#ifdef ENABLE_BF16
template<> struct packed_as<__nv_bfloat16,  2> { using type = __nv_bfloat162; };
template<> struct packed_as<__nv_bfloat162, 1> { using type = __nv_bfloat16;  };
#endif

#ifdef ENABLE_FP8
template<> struct packed_as<__nv_fp8_e4m3,  2> { using type = __nv_fp8x2_e4m3; };
template<> struct packed_as<__nv_fp8x2_e4m3, 1> { using type = __nv_fp8_e4m3;  };
template<> struct packed_as<__nv_fp8_e5m2,  2> { using type = __nv_fp8x2_e5m2; };
template<> struct packed_as<__nv_fp8x2_e5m2, 1> { using type = __nv_fp8_e5m2;  };
#endif

inline __device__ float2 operator*(float2 a, float2 b) { return make_float2(a.x * b.x, a.y * b.y); }
inline __device__ float2 operator+(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
inline __device__ float2 operator-(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }

inline __device__ float2 operator*(float2 a, float  b) { return make_float2(a.x * b, a.y * b); }
inline __device__ float2 operator+(float2 a, float  b) { return make_float2(a.x + b, a.y + b); }
inline __device__ float2 operator-(float2 a, float  b) { return make_float2(a.x - b, a.y - b); }

bool should_print();

template<typename T>
void print_bshd(const int   layer_id,
                const char* name,
                const T*    ptr,
                int         batch_size,
                int         seq_len,
                int         num_heads,
                int         hidden_size_per_head,
                int         total_num_heads = 0,
                int         heads_offset = 0,
                bool        is_device_ptr = true);
template<typename T>
void print_bhsd(const int   layer_id,
                const char* name,
                const T*    ptr,
                int         batch_size,
                int         num_heads,
                int         seq_len,
                int         hidden_size_per_head,
                bool        is_device_ptr = true);
template<typename T>
void print_bhss(const int   layer_id,
                const char* name,
                const T*    ptr,
                int         batch_size,
                int         num_heads,
                int         seq_len,
                int         seq_len2,
                bool        is_device_ptr = true);
template<typename T>
void print_bsd(const int   layer_id,
               const char* name,
               const T*    ptr,
               int         batch_size,
               int         seq_len,
               int         hidden_size,
               int         start         = 0,
               int         end           = 20,
               bool        is_device_ptr = true);
template<typename T>
void print_bsd_sum_and_square(const int   layer_id,
                              const char* name,
                              const T*    ptr,
                              int         batch_size,
                              int         seq_len,
                              int         hidden_size,
                              int         start         = 0,
                              int         end           = 20,
                              bool        is_device_ptr = true);
template<typename T>
void print_kv_cache(const int   layer_id,
                    const char* name,
                    const T*    ptr,
                    int         dim1,
                    int         dim2,
                    int         dim3,
                    int         dim4,
                    int         dim5,
                    int         dim6,
                    bool        print_all     = true,
                    bool        is_device_ptr = true);

}  // namespace rtp_llm
