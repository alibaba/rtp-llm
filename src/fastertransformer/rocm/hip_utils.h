#pragma once


#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/utils/assert_utils.h"
#include "src/fastertransformer/utils/string_utils.h"

#include <hip/hip_runtime.h>
#include "cuda_shims.h"

#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>


#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fastertransformer {

#define ROCM_CHECK(val) rocm::check((val), #val, __FILE__, __LINE__)
#define ROCM_SYNC_AND_CHECK() rocm::sync_and_check(__FILE__, __LINE__)
#define ROCM_CHECK_VALUE(val, info, ...)                              \
do {                                                                \
    bool is_valid_val = (val);                                      \
    if (!is_valid_val) {						                    \
        rocm::throwRocmError(__FILE__, __LINE__, fmtstr(info, ##__VA_ARGS__)); \
    }								                                \
} while (0)
#define ROCM_FAIL(info, ...) rocm::throwRocmError(__FILE__, __LINE__, fmtstr(info, ##__VA_ARGS__))

#define MAX_CONFIG_NUM 20
#define COL32_ 32
#define HIPBLAS_WORKSPACE_SIZE 33554432 // 32MB

typedef struct __attribute__((aligned(4))) {
    half x, y, z, w;
} half4;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct __attribute__((aligned(16))) Float4_ {
    float2 x;
    float2 y;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct __attribute__((aligned(32))) Float8_ {
    float2 x;
    float2 y;
    float2 z;
    float2 w;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_BF16
struct __attribute__((aligned(8))) bf16_4_t {
    __nv_bfloat162 x;
    __nv_bfloat162 y;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct __attribute__((aligned(16))) bf16_8_t {
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

namespace rocm {
enum FtHipDataType {
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
static const char* _hipGetErrorEnum(hipError_t error)
{
    return hipGetErrorString(error);
}
static const char* _hipGetErrorEnum(hipblasStatus_t error)
{
    switch (error) {
        case HIPBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case HIPBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case HIPBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case HIPBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case HIPBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case HIPBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case HIPBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case HIPBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case HIPBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case HIPBLAS_STATUS_UNKNOWN:
            return "CUBLAS_STATUS_LICENSE_ERROR";

        case HIPBLAS_STATUS_HANDLE_IS_NULLPTR:
            return "HIPBLAS_STATUS_HANDLE_IS_NULLPTR";

        case HIPBLAS_STATUS_INVALID_ENUM:
            return "HIPBLAS_STATUS_INVALID_ENUM";
    }
    return "<unknown>";
}

static void throwRocmError(const char* const file, int const line, std::string const& info = "") {
    auto error_msg = std::string("[ROCm][ERROR] ") + info + " Assertion fail: " + file + ":" + std::to_string(line) + " \n";
    std::printf("%s", error_msg.c_str());
    fflush(stdout);
    fflush(stderr);
    if (std::getenv("FT_CORE_DUMP_ON_EXCEPTION")) {
        abort();
    }
    throw NEW_FT_EXCEPTION(error_msg);
}

template<typename T>
void check(T result, char const* const func, const char* const file, int const line)
{
    if (result) {
        throwRocmError(file, line, _hipGetErrorEnum(result));
    }
}
inline void sync_and_check(const char* const file, int const line) {
    hipDeviceSynchronize();
    hipError_t result = hipGetLastError();
    if(result) {
        throwRocmError(file, line, _hipGetErrorEnum(result));;
    }
}

/*************Time Handling**************/
class HipTimer {
private:
    hipEvent_t  event_start_;
    hipEvent_t  event_stop_;
    hipStream_t stream_;

public:
    explicit HipTimer(hipStream_t stream = 0)
    {
        stream_ = stream;
    }
    void start()
    {
        ROCM_CHECK(hipEventCreate(&event_start_));
        ROCM_CHECK(hipEventCreate(&event_stop_));
        ROCM_CHECK(hipEventRecord(event_start_, stream_));
    }
    float stop()
    {
        float time;
        ROCM_CHECK(hipEventRecord(event_stop_, stream_));
        ROCM_CHECK(hipEventSynchronize(event_stop_));
        ROCM_CHECK(hipEventElapsedTime(&time, event_start_, event_stop_));
        ROCM_CHECK(hipEventDestroy(event_start_));
        ROCM_CHECK(hipEventDestroy(event_stop_));
        return time;
    }
    ~HipTimer() {}
};

static double diffTime(timeval start, timeval end)
{
    return (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
}

/* ***************************** common utils ****************************** */

inline void print_mem_usage(std::string time = "after allocation")
{
    size_t free_bytes, total_bytes;
    ROCM_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
    float free  = static_cast<float>(free_bytes) / 1024.0 / 1024.0 / 1024.0;
    float total = static_cast<float>(total_bytes) / 1024.0 / 1024.0 / 1024.0;
    float used  = total - free;
    printf("%-20s: free: %5.2f GB, total: %5.2f GB, used: %5.2f GB\n", time.c_str(), free, total, used);
}

inline int getSMVersion()
{
    int device{-1};
    ROCM_CHECK(hipGetDevice(&device));
    int sm_major = 0;
    int sm_minor = 0;
    ROCM_CHECK(hipDeviceGetAttribute(&sm_major, hipDeviceAttributeComputeCapabilityMajor, device));
    ROCM_CHECK(hipDeviceGetAttribute(&sm_minor, hipDeviceAttributeComputeCapabilityMinor, device));
    return sm_major * 10 + sm_minor;
}

inline int getMaxSharedMemoryPerBlock()
{
    int device{-1};
    ROCM_CHECK(hipGetDevice(&device));
    int max_shared_memory_size = 0;
    ROCM_CHECK(hipDeviceGetAttribute(&max_shared_memory_size, hipDeviceAttributeMaxSharedMemoryPerBlock, device));
    return max_shared_memory_size;
}

inline std::string getDeviceName()
{
    int device{-1};
    ROCM_CHECK(hipGetDevice(&device));
    hipDeviceProp_t props;
    ROCM_CHECK(hipGetDeviceProperties(&props, device));
    return std::string(props.name);
}

inline int div_up(int a, int n)
{
    return (a + n - 1) / n;
}

hipError_t getSetDevice(int i_device, int* o_device = NULL);

inline int getDevice()
{
    int current_dev_id = 0;
    ROCM_CHECK(hipGetDevice(&current_dev_id));
    return current_dev_id;
}

inline int getDeviceCount()
{
    int count = 0;
    ROCM_CHECK(hipGetDeviceCount(&count));
    return count;
}


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
inline __device__ float2 operator*(float2 a, float  b) { return make_float2(a.x * b, a.y * b); }
// clang-format on

template<typename T1, typename T2>
void compareTwoTensor(
    const T1* pred, const T2* ref, const int size, const int print_size = 0, const std::string filename = "")
{
    T1* h_pred = new T1[size];
    T2* h_ref  = new T2[size];
    ROCM_CHECK(hipMemcpy(h_pred, pred, size * sizeof(T1), hipMemcpyDeviceToHost));
    ROCM_CHECK(hipMemcpy(h_ref, ref, size * sizeof(T2), hipMemcpyDeviceToHost));

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

/* ************************** end of common utils ************************** */
}  // namespace rocm
}  // namespace fastertransformer
