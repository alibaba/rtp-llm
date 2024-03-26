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

#include "src/fastertransformer/utils/assert_utils.h"
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/core/Types.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#ifdef SPARSITY_ENABLED
#include <cusparseLt.h>
#endif

#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

namespace fastertransformer {

#define MAX_CONFIG_NUM 20
#define COL32_ 32
// workspace for cublas gemm : 32MB
#define CUBLAS_WORKSPACE_SIZE 33554432

typedef struct __align__(4) {
    half x, y, z, w;
}
half4;

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

/* **************************** debug tools ********************************* */
static const char* _cudaGetErrorEnum(cudaError_t error) {
    return cudaGetErrorString(error);
}

static const char* _cudaGetErrorEnum(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "<unknown>";
}

template<typename T>
void check(T result, char const* const func, const char* const file, int const line) {
    if (result) {
        FT_LOG_ERROR(std::string("[FT][ERROR] CUDA runtime error: ") + (_cudaGetErrorEnum(result)) + " " + file + ":"
                     + std::to_string(line) + " \n");
        fflush(stdout);
        throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: ") + (_cudaGetErrorEnum(result)) + " "
                                 + file + ":" + std::to_string(line) + " \n");
        abort();
    }
}

#define check_cuda_error(val) fastertransformer::check((val), #val, __FILE__, __LINE__)
#define check_cuda_error_2(val, file, line) fastertransformer::check((val), #val, file, line)

inline void syncAndCheck(const char* const file, int const line) {
    // When FT_DEBUG_LEVEL=DEBUG, must check error
    static char* level_name = std::getenv("FT_DEBUG_LEVEL");
    if (level_name != nullptr) {
        static std::string level = std::string(level_name);
        if (level == "DEBUG") {
            cudaDeviceSynchronize();
            cudaError_t result = cudaGetLastError();
            if (result) {
                throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: ") + (_cudaGetErrorEnum(result))
                                         + " " + file + ":" + std::to_string(line) + " \n");
            }
            FT_LOG_DEBUG(fmtstr("run syncAndCheck at %s:%d", file, line));
        }
    }

#ifndef NDEBUG
    cudaDeviceSynchronize();
    cudaError_t result = cudaGetLastError();
    if (result) {
        throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: ") + (_cudaGetErrorEnum(result)) + " "
                                 + file + ":" + std::to_string(line) + " \n");
    }
#endif
}

inline void CheckError(const char* const file, int const line) {
    cudaError_t result = cudaGetLastError();
    if (result) {
        FT_LOG_ERROR(std::string("[FT][ERROR] CUDA runtime error: ") + (_cudaGetErrorEnum(result)) + " " + file + ":"
                     + std::to_string(line) + " \n");
        fflush(stdout);
        abort();
    }
}

#define sync_check_cuda_error() fastertransformer::syncAndCheck(__FILE__, __LINE__)

#define final_check_error() CheckError(__FILE__, __LINE__)

inline int get_sm() {
    static int sm = []() {
        int device;
        check_cuda_error(cudaGetDevice(&device));
        cudaDeviceProp deviceProp;
        check_cuda_error(cudaGetDeviceProperties(&deviceProp, device));
        return deviceProp.major * 10 + deviceProp.minor;
    }();
    return sm;
}

inline bool is_sm70() {
    static bool IS_SM70 = []() {
        return get_sm() == 70;
    }();
    return IS_SM70;
}

inline bool is_sm8x() {
    static bool IS_SM8X = []() {
        return (get_sm() >= 80) && (get_sm() <= 89);
    }();
    return IS_SM8X;
}

inline bool is_sm90() {
    static bool IS_SM90 = []() {
        return get_sm() == 90;
    }();
    return IS_SM90;
}

#define checkCUDNN(expression)                                                                                         \
    {                                                                                                                  \
        cudnnStatus_t status = (expression);                                                                           \
        if (status != CUDNN_STATUS_SUCCESS) {                                                                          \
            std::cerr << "Error on file " << __FILE__ << " line " << __LINE__ << ": " << cudnnGetErrorString(status)   \
                      << std::endl;                                                                                    \
            std::exit(EXIT_FAILURE);                                                                                   \
        }                                                                                                              \
    }

template<typename T>
void print_to_file(const T*           result,
                   const int          size,
                   const char*        file,
                   cudaStream_t       stream    = 0,
                   std::ios::openmode open_mode = std::ios::out);

template<typename T>
void print_abs_mean(const T* buf, uint size, cudaStream_t stream, std::string name = "");

template<typename T>
void print_to_screen(const T* result, const int size);

template<typename T>
void printMatrix(T* ptr, int m, int k, int stride, bool is_device_ptr);

void printMatrix(unsigned long long* ptr, int m, int k, int stride, bool is_device_ptr);
void printMatrix(int* ptr, int m, int k, int stride, bool is_device_ptr);
void printMatrix(size_t* ptr, int m, int k, int stride, bool is_device_ptr);

template<typename T>
void check_max_val(const T* result, const int size);

template<typename T>
void check_abs_mean_val(const T* result, const int size);

#define PRINT_FUNC_NAME_()                                                                                             \
    do {                                                                                                               \
        std::cout << "[FT][CALL] " << __FUNCTION__ << " " << std::endl;                                                \
    } while (0)

#ifdef SPARSITY_ENABLED
#define CHECK_CUSPARSE(func)                                                                                           \
    {                                                                                                                  \
        cusparseStatus_t status = (func);                                                                              \
        if (status != CUSPARSE_STATUS_SUCCESS) {                                                                       \
            throw std::runtime_error(std::string("[FT][ERROR] CUSPARSE API failed at line ")                           \
                                     + std::to_string(__LINE__) + " in file " + __FILE__ + ": "                        \
                                     + cusparseGetErrorString(status) + " " + std::to_string(status));                 \
        }                                                                                                              \
    }
#endif

/*************Time Handling**************/
class CudaTimer {
private:
    cudaEvent_t  event_start_;
    cudaEvent_t  event_stop_;
    cudaStream_t stream_;

public:
    explicit CudaTimer(cudaStream_t stream = 0) {
        stream_ = stream;
    }
    void start() {
        check_cuda_error(cudaEventCreate(&event_start_));
        check_cuda_error(cudaEventCreate(&event_stop_));
        check_cuda_error(cudaEventRecord(event_start_, stream_));
    }
    float stop() {
        float time;
        check_cuda_error(cudaEventRecord(event_stop_, stream_));
        check_cuda_error(cudaEventSynchronize(event_stop_));
        check_cuda_error(cudaEventElapsedTime(&time, event_start_, event_stop_));
        check_cuda_error(cudaEventDestroy(event_start_));
        check_cuda_error(cudaEventDestroy(event_stop_));
        return time;
    }
    ~CudaTimer() {}
};

static double diffTime(timeval start, timeval end) {
    return (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
}

/* ***************************** common utils ****************************** */

inline void print_mem_usage(std::string time = "after allocation") {
    size_t free_bytes, total_bytes;
    check_cuda_error(cudaMemGetInfo(&free_bytes, &total_bytes));
    float free  = static_cast<float>(free_bytes) / 1024.0 / 1024.0 / 1024.0;
    float total = static_cast<float>(total_bytes) / 1024.0 / 1024.0 / 1024.0;
    float used  = total - free;
    printf("%-20s: free: %5.2f GB, total: %5.2f GB, used: %5.2f GB\n", time.c_str(), free, total, used);
}

inline int getSMVersion() {
    int device{-1};
    check_cuda_error(cudaGetDevice(&device));
    int sm_major = 0;
    int sm_minor = 0;
    check_cuda_error(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
    check_cuda_error(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device));
    return sm_major * 10 + sm_minor;
}

inline int getMaxSharedMemoryPerBlock() {
    int device{-1};
    check_cuda_error(cudaGetDevice(&device));
    int max_shared_memory_size = 0;
    check_cuda_error(cudaDeviceGetAttribute(&max_shared_memory_size, cudaDevAttrMaxSharedMemoryPerBlock, device));
    return max_shared_memory_size;
}

inline std::string getDeviceName() {
    int device{-1};
    check_cuda_error(cudaGetDevice(&device));
    cudaDeviceProp props;
    check_cuda_error(cudaGetDeviceProperties(&props, device));
    return std::string(props.name);
}

inline int div_up(int a, int n) {
    return (a + n - 1) / n;
}

template <typename T, typename U, typename = std::enable_if_t<std::is_integral<T>::value>,
    typename = std::enable_if_t<std::is_integral<U>::value>>
auto constexpr ceilDiv(T numerator, U denominator)
{
    return (numerator + denominator - 1) / denominator;
}

cudaError_t getSetDevice(int i_device, int* o_device = NULL);

inline int getDevice() {
    int current_dev_id = 0;
    check_cuda_error(cudaGetDevice(&current_dev_id));
    return current_dev_id;
}

inline int getDeviceCount() {
    int count = 0;
    check_cuda_error(cudaGetDeviceCount(&count));
    return count;
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
        FT_CHECK(false);
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
        FT_CHECK(false);
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

inline __device__ float2 operator*(float2 a, float2 b) { return make_float2(a.x * b.x, a.y * b.y); }
inline __device__ float2 operator+(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
inline __device__ float2 operator-(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }

inline __device__ float2 operator*(float2 a, float  b) { return make_float2(a.x * b, a.y * b); }
inline __device__ float2 operator+(float2 a, float  b) { return make_float2(a.x + b, a.y + b); }
inline __device__ float2 operator-(float2 a, float  b) { return make_float2(a.x - b, a.y - b); }

// clang-format on

template<typename T1, typename T2>
void compareTwoTensor(
    const T1* pred, const T2* ref, const int size, const int print_size = 0, const std::string filename = "") {
    T1* h_pred = new T1[size];
    T2* h_ref  = new T2[size];
    check_cuda_error(cudaMemcpy(h_pred, pred, size * sizeof(T1), cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(h_ref, ref, size * sizeof(T2), cudaMemcpyDeviceToHost));

    FILE* fd = nullptr;
    if (filename != "") {
        fd = fopen(filename.c_str(), "w");
        fprintf(fd, "| %10s | %10s | %10s | %10s | \n", "pred", "ref", "abs_diff", "rel_diff(%)");
    }

    if (print_size > 0) {
        FT_LOG_INFO("  id |   pred  |   ref   |abs diff | rel diff (%) |");
    }
    float mean_abs_diff = 0.0f;
    float mean_rel_diff = 0.0f;
    int   count         = 0;
    for (int i = 0; i < size; i++) {
        if (i < print_size) {
            FT_LOG_INFO("%4d | % 6.4f | % 6.4f | % 6.4f | % 7.4f |",
                        i,
                        (float)h_pred[i],
                        (float)h_ref[i],
                        abs((float)h_pred[i] - (float)h_ref[i]),
                        abs((float)h_pred[i] - (float)h_ref[i]) / (abs((float)h_ref[i]) + 1e-6f) * 100.f);
        }
        if ((float)h_pred[i] == 0) {
            continue;
        }
        count += 1;
        mean_abs_diff += abs((float)h_pred[i] - (float)h_ref[i]);
        mean_rel_diff += abs((float)h_pred[i] - (float)h_ref[i]) / (abs((float)h_ref[i]) + 1e-6f) * 100.f;

        if (fd != nullptr) {
            fprintf(fd,
                    "| %10.5f | %10.5f | %10.5f | %11.5f |\n",
                    (float)h_pred[i],
                    (float)h_ref[i],
                    abs((float)h_pred[i] - (float)h_ref[i]),
                    abs((float)h_pred[i] - (float)h_ref[i]) / (abs((float)h_ref[i]) + 1e-6f) * 100.f);
        }
    }
    mean_abs_diff = mean_abs_diff / (float)count;
    mean_rel_diff = mean_rel_diff / (float)count;
    FT_LOG_INFO("mean_abs_diff: % 6.4f, mean_rel_diff: % 6.4f (%%)", mean_abs_diff, mean_rel_diff);

    if (fd != nullptr) {
        fprintf(fd, "mean_abs_diff: % 6.4f, mean_rel_diff: % 6.4f (%%)", mean_abs_diff, mean_rel_diff);
        fclose(fd);
    }
    delete[] h_pred;
    delete[] h_ref;
}

static inline size_t pad_to_multiple_of_16(const size_t& input) {
    static constexpr int ALIGNMENT = 16;
    return ALIGNMENT * ((input + ALIGNMENT - 1) / ALIGNMENT);
}

static inline bool shared_mem_sufficient(int smem_size) {
    //
    // Determine SMEM requirements and waive if not satisfied
    //
    cudaDeviceProp properties;
    int            device_idx;
    cudaError_t    result = cudaGetDevice(&device_idx);

    if (result != cudaSuccess) {
        throw std::runtime_error("cudaGetDevice() API call failed.");
    }

    result = cudaGetDeviceProperties(&properties, device_idx);

    if (result != cudaSuccess) {
        throw std::runtime_error("cudaGetDeviceProperties() failed");
    }

    if (properties.sharedMemPerMultiprocessor < smem_size) {
        return false;
    }

    return true;
}

static inline bool should_print() {
    static char* level_name = std::getenv("FT_DEBUG_PRINT_LEVEL");
    if (!level_name || (strcmp(level_name, "DEBUG") != 0)) {
        return false;
    }

    static char* tp_rank = std::getenv("WORLD_RANK");
    if (tp_rank && (strcmp(tp_rank, "0") != 0)) {
        return false;
    }

    return true;
}

template<typename T>
void print_bshd(const int   layer_id,
                const char* name,
                const T*    ptr,
                int         batch_size,
                int         seq_len,
                int         num_heads,
                int         hidden_size_per_head,
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

template<typename T>
T getCudaValue(const T* ptr, int index) {
    T tmp;
    check_cuda_error(cudaMemcpy(&tmp, ptr + index, sizeof(T), cudaMemcpyDeviceToHost));
    return tmp;
}

template<typename T>
void setCudaValue(T* ptr, int index, T value) {
    check_cuda_error(cudaMemcpy(ptr + index, &value, sizeof(T), cudaMemcpyHostToDevice));
}

/* **************************** type dispatch ***************************** */

#define DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type, function, ...)                              \
    do {                                                                                        \
        switch (data_type) {                                                                    \
            case DataType::TYPE_FP32:                                                           \
                function<float>(__VA_ARGS__);                                                   \
                break;                                                                          \
            case DataType::TYPE_FP16:                                                           \
                function<half>(__VA_ARGS__);                                                    \
                break;                                                                          \
            case DataType::TYPE_BF16:                                                           \
                function<__nv_bfloat16>(__VA_ARGS__);                                           \
                break;                                                                          \
            default:                                                                            \
                FT_CHECK(false);                                                                \
        }                                                                                       \
    } while (0)

/* ************************** end of common utils ************************** */
}  // namespace fastertransformer
