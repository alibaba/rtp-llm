#pragma once
#include <vector>
#include <torch/extension.h>
#include <torch/all.h>
#include "rtp_llm/cpp/core/Types.h"

#if USING_ROCM
#include <rtp_llm/cpp/rocm/amd_bfloat16.h>
#include <hip/hip_runtime.h>
#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#include "rtp_llm/cpp/kernels/rocm/fused_qk_rmsnorm.h"
#include "rtp_llm/cpp/kernels/rocm/layernorm_kernels.h"
using bf16_type  = amd_bfloat16;
using StreamType = hipStream_t;
#define GET_CURRENT_STREAM() at::hip::getCurrentHIPStream().stream()
#elif USING_CUDA
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_device_runtime_api.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include "rtp_llm/cpp/kernels/layernorm_kernels.h"
#include "rtp_llm/cpp/kernels/fused_qk_rmsnorm.h"
#include "rtp_llm/cpp/kernels/rmsnormKernels.h"
using bf16_type  = nv_bfloat16;
using StreamType = cudaStream_t;
#define GET_CURRENT_STREAM() at::cuda::getCurrentCUDAStream(at::cuda::current_device()).stream()
#endif

#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(pytorch_dtype, c_type, ...)                                               \
    [&]() -> bool {                                                                                                    \
        switch (pytorch_dtype) {                                                                                       \
            case at::ScalarType::Half: {                                                                               \
                using c_type = __half;                                                                                 \
                return __VA_ARGS__();                                                                                  \
            }                                                                                                          \
            case at::ScalarType::BFloat16: {                                                                           \
                using c_type = bf16_type;                                                                              \
                return __VA_ARGS__();                                                                                  \
            }                                                                                                          \
            default:                                                                                                   \
                std::ostringstream oss;                                                                                \
                oss << __PRETTY_FUNCTION__ << " failed to dispatch data type " << pytorch_dtype;                       \
                TORCH_CHECK(false, oss.str());                                                                         \
                return false;                                                                                          \
        }                                                                                                              \
    }()

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_LAST_DIM_CONTIGUOUS(x)                                                                                   \
    TORCH_CHECK(x.strides()[x.strides().size() - 1] == 1, #x "must be contiguous at last dimension")

#define CHECK_INPUT(x)                                                                                                 \
    CHECK_CUDA(x);                                                                                                     \
    CHECK_CONTIGUOUS(x)
#define CHECK_LAST_DIM_CONTIGUOUS_INPUT(x)                                                                             \
    CHECK_CUDA(x);                                                                                                     \
    CHECK_LAST_DIM_CONTIGUOUS(x)

#define CHECK_DIM(d, x) TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")

#define CHECK_SHAPE(a, b) check_shape(a, b, #a, #b)

#define CHECK_EQ(a, b) TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

#define CHECK_GE(a, b) TORCH_CHECK((a) >= (b), "CHECK_GE(" #a ", " #b ") failed. ", a, " vs ", b)