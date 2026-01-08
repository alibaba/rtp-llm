#pragma once

#include <ATen/Tensor.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#if USING_CUDA
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

#if USING_ROCM
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#endif

#include <sstream>

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

#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(pytorch_dtype, c_type, ...)                                               \
    [&]() -> bool {                                                                                                    \
        switch (pytorch_dtype) {                                                                                       \
            _DISPATCH_CASE_F16(c_type, __VA_ARGS__)                                                                    \
            _DISPATCH_CASE_BF16(c_type, __VA_ARGS__)                                                                   \
            default:                                                                                                   \
                std::ostringstream oss;                                                                                \
                oss << __PRETTY_FUNCTION__ << " failed to dispatch data type " << pytorch_dtype;                       \
                TORCH_CHECK(false, oss.str());                                                                         \
                return false;                                                                                          \
        }                                                                                                              \
    }()

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_EQ(a, b) TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

#define CHECK_INPUT(x)                                                                                                 \
    CHECK_CUDA(x);                                                                                                     \
    CHECK_CONTIGUOUS(x)

// SGLANG_SHFL_XOR_* adapted from https://github.com/vllm-project/vllm/blob/v0.7.3/csrc/cuda_compat.h#L19-L28
#ifndef USE_ROCM
#define SGLANG_SHFL_XOR_SYNC(mask, var, lane_mask) __shfl_xor_sync((mask), (var), (lane_mask))
#define SGLANG_SHFL_XOR_SYNC_WIDTH(mask, var, lane_mask, width) __shfl_xor_sync((mask), (var), (lane_mask), (width))
#else
#define SGLANG_SHFL_XOR_SYNC(mask, var, lane_mask) __shfl_xor((var), (lane_mask))
#define SGLANG_SHFL_XOR_SYNC_WIDTH(mask, var, lane_mask, width) __shfl_xor((var), (lane_mask), (width))
#endif

#define WARP_SIZE 32

#if USING_CUDA
#ifndef CUDART_INF_FP16
#define CUDART_INF_FP16 __ushort_as_half((unsigned short)0x7C00U)
#endif
#ifndef CUDART_INF_BF16
#define CUDART_INF_BF16 __ushort_as_bfloat16((unsigned short)0x7F80U)
#endif
#endif

#if USING_ROCM
#ifndef HIP_INF_FP16
#define HIP_INF_FP16 __ushort_as_half((unsigned short)0x7C00U)
#endif
#ifndef HIP_INF_BF16
#define HIP_INF_BF16 __ushort_as_bfloat16((unsigned short)0x7F80U)
#endif
#define CUDART_INF_FP16 HIP_INF_FP16
#define CUDART_INF_BF16 HIP_INF_BF16
#endif

namespace rtp_llm {

template<typename T>
__device__ __forceinline__ T NegativeInfinity() {
    return -INFINITY;
}

template<>
__device__ __forceinline__ __half NegativeInfinity<__half>() {
    return -CUDART_INF_FP16;
}

template<>
__device__ __forceinline__ __nv_bfloat16 NegativeInfinity<__nv_bfloat16>() {
    return -CUDART_INF_BF16;
}

}  // namespace rtp_llm
