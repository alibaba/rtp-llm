// Adapted from sglang
#pragma once

#include <ATen/Tensor.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#define _DISPATCH_CASE_F16(c_type, ...)                                                                                \
    case at::ScalarType::Half: {                                                                                       \
        using c_type = nv_half;                                                                                        \
        return __VA_ARGS__();                                                                                          \
    }

#define _DISPATCH_CASE_BF16(c_type, ...)                                                                               \
    case at::ScalarType::BFloat16: {                                                                                   \
        using c_type = nv_bfloat16;                                                                                    \
        return __VA_ARGS__();                                                                                          \
    }

#define CHECK_EQ(a, b) TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                                                 \
    CHECK_CUDA(x);                                                                                                     \
    CHECK_CONTIGUOUS(x)

#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(pytorch_dtype, c_type, ...)                                         \
    [&]() -> bool {                                                                                                    \
        switch (pytorch_dtype) {                                                                                       \
            case at::ScalarType::Float: {                                                                              \
                using c_type = float;                                                                                  \
                return __VA_ARGS__();                                                                                  \
            }                                                                                                          \
                _DISPATCH_CASE_F16(c_type, __VA_ARGS__)                                                                \
                _DISPATCH_CASE_BF16(c_type, __VA_ARGS__)                                                               \
            default:                                                                                                   \
                std::ostringstream oss;                                                                                \
                oss << __PRETTY_FUNCTION__ << " failed to dispatch data type " << pytorch_dtype;                       \
                TORCH_CHECK(false, oss.str());                                                                         \
                return false;                                                                                          \
        }                                                                                                              \
    }()

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

#define WARP_SIZE 32

// add FP8 support
#include <c10/util/Float8_e4m3fn.h>
using FP8_TYPE                              = c10::Float8_e4m3fn;
C10_HOST_DEVICE constexpr auto FP8_E4M3_MAX = std::numeric_limits<FP8_TYPE>::max();

#define FULL_MASK 0xffffffff

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
                         __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
    return old;
}

__device__ __forceinline__ float warpReduceMax(float value) {
    value = fmaxf(value, __shfl_xor_sync(FULL_MASK, value, 16));
    value = fmaxf(value, __shfl_xor_sync(FULL_MASK, value, 8));
    value = fmaxf(value, __shfl_xor_sync(FULL_MASK, value, 4));
    value = fmaxf(value, __shfl_xor_sync(FULL_MASK, value, 2));
    value = fmaxf(value, __shfl_xor_sync(FULL_MASK, value, 1));
    return value;
}

__device__ __forceinline__ float blockReduceMax(float value) {
    static __shared__ float warpLevelMaxs[WARP_SIZE];
    const int               laneId = threadIdx.x % WARP_SIZE;
    const int               warpId = threadIdx.x / WARP_SIZE;

    value = warpReduceMax(value);

    if (laneId == 0)
        warpLevelMaxs[warpId] = value;
    __syncthreads();

    value = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelMaxs[laneId] : 0;
    if (warpId == 0)
        value = warpReduceMax(value);

    return value;
}
