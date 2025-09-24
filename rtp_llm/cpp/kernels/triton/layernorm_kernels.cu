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

#include "rtp_llm/cpp/kernels/triton/layernorm_kernels.h"

#ifdef ENABLE_TRITON
extern "C" {
#include "rtp_llm/cpp/kernels/triton/aot/layernorm_kernel_bf16.h"
#include "rtp_llm/cpp/kernels/triton/aot/layernorm_kernel_fp16.h"
#include "rtp_llm/cpp/kernels/triton/aot/layernorm_kernel_fp32.h"
}
#endif

// wont't support new features
namespace rtp_llm {
unsigned int nextPowerOf2(unsigned int n) {
    if (n == 0)
        return 1;

    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;

    return n + 1;
}

#ifdef ENABLE_TRITON
#define TRITON_LAYERNORM_KERNEL(T, LEN, WARPS, HAS_BIAS, ...)                                                          \
    do {                                                                                                               \
        if constexpr (std::is_same_v<T, float>) {                                                                      \
            if constexpr (HAS_BIAS)                                                                                    \
                check_cuda_value(layernorm_kernel_fp32_1x1x##LEN##_##warps##WARPS##xstages3(__VA_ARGS__));             \
            else                                                                                                       \
                check_cuda_value(layernorm_kernel_fp32_0x0x##LEN##_##warps##WARPS##xstages3(__VA_ARGS__));             \
        } else if constexpr (std::is_same_v<T, half>) {                                                                \
            if constexpr (HAS_BIAS)                                                                                    \
                check_cuda_value(layernorm_kernel_fp16_1x1x##LEN##_##warps##WARPS##xstages3(__VA_ARGS__));             \
            else                                                                                                       \
                check_cuda_value(layernorm_kernel_fp16_0x0x##LEN##_##warps##WARPS##xstages3(__VA_ARGS__));             \
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {                                                       \
            if constexpr (HAS_BIAS)                                                                                    \
                check_cuda_value(layernorm_kernel_bf16_1x1x##LEN##_##warps##WARPS##xstages3(__VA_ARGS__));             \
            else                                                                                                       \
                check_cuda_value(layernorm_kernel_bf16_0x0x##LEN##_##warps##WARPS##xstages3(__VA_ARGS__));             \
        }                                                                                                              \
    } while (0)
#else
#define TRITON_LAYERNORM_KERNEL(T, LEN, WARPS, HAS_BIAS, ...)                                                          \
    do {                                                                                                               \
        /* Triton kernels not available */                                                                             \
    } while (0)
#endif

template<typename T, typename QUANT_OUT_T, bool HAS_BIAS>
void invokeTritonLayerNorm(T*           out,
                           T*           norm_output,
                           const T*     input,
                           const T*     bias,
                           const T*     residual,
                           const T*     gamma,
                           const T*     beta,
                           const float  eps,
                           const int    tokens,
                           const int    hidden_dim,
                           cudaStream_t stream,
                           bool         use_diff_of_squares,
                           const float* scale,
                           float*       dynamic_scale,
                           QUANT_OUT_T* out_quant,
                           bool         return_normed_output) {
    unsigned hdim = nextPowerOf2(hidden_dim);
    if (hdim <= 1024)
        hdim = 1024;

    if (hdim == 1024)
        TRITON_LAYERNORM_KERNEL(T,
                                1024,
                                4,
                                HAS_BIAS,
                                stream,
                                reinterpret_cast<CUdeviceptr>(const_cast<T*>(input)),
                                reinterpret_cast<CUdeviceptr>(norm_output),
                                reinterpret_cast<CUdeviceptr>(out),
                                reinterpret_cast<CUdeviceptr>(const_cast<T*>(gamma)),
                                reinterpret_cast<CUdeviceptr>(const_cast<T*>(beta)),
                                reinterpret_cast<CUdeviceptr>(const_cast<T*>(residual)),
                                reinterpret_cast<CUdeviceptr>(const_cast<T*>(bias)),
                                hidden_dim,
                                hidden_dim,
                                hidden_dim,
                                tokens,
                                hidden_dim,
                                eps,
                                out != nullptr);
    else if (hdim == 2048)
        TRITON_LAYERNORM_KERNEL(T,
                                2048,
                                8,
                                HAS_BIAS,
                                stream,
                                reinterpret_cast<CUdeviceptr>(const_cast<T*>(input)),
                                reinterpret_cast<CUdeviceptr>(norm_output),
                                reinterpret_cast<CUdeviceptr>(out),
                                reinterpret_cast<CUdeviceptr>(const_cast<T*>(gamma)),
                                reinterpret_cast<CUdeviceptr>(const_cast<T*>(beta)),
                                reinterpret_cast<CUdeviceptr>(const_cast<T*>(residual)),
                                reinterpret_cast<CUdeviceptr>(const_cast<T*>(bias)),
                                hidden_dim,
                                hidden_dim,
                                hidden_dim,
                                tokens,
                                hidden_dim,
                                eps,
                                out != nullptr);
    else
        TRITON_LAYERNORM_KERNEL(T,
                                4096,
                                8,
                                HAS_BIAS,
                                stream,
                                reinterpret_cast<CUdeviceptr>(const_cast<T*>(input)),
                                reinterpret_cast<CUdeviceptr>(norm_output),
                                reinterpret_cast<CUdeviceptr>(out),
                                reinterpret_cast<CUdeviceptr>(const_cast<T*>(gamma)),
                                reinterpret_cast<CUdeviceptr>(const_cast<T*>(beta)),
                                reinterpret_cast<CUdeviceptr>(const_cast<T*>(residual)),
                                reinterpret_cast<CUdeviceptr>(const_cast<T*>(bias)),
                                hidden_dim,
                                hidden_dim,
                                hidden_dim,
                                tokens,
                                hidden_dim,
                                eps,
                                out != nullptr);
}

#define INSTANTIATE_TRITON_ADD_BIAS_RESIDUAL_LAYERNORM(T, QUANT_OUT_T, HAS_BIAS)                                       \
    template void invokeTritonLayerNorm<T, QUANT_OUT_T, HAS_BIAS>(T * out,                                             \
                                                                  T * norm_output,                                     \
                                                                  const T*     input,                                  \
                                                                  const T*     bias,                                   \
                                                                  const T*     residual,                               \
                                                                  const T*     gamma,                                  \
                                                                  const T*     beta,                                   \
                                                                  const float  eps,                                    \
                                                                  const int    tokens,                                 \
                                                                  const int    hidden_dim,                             \
                                                                  cudaStream_t stream,                                 \
                                                                  bool         use_diff_of_squares,                    \
                                                                  const float* scale,                                  \
                                                                  float*       dynamic_scale,                          \
                                                                  QUANT_OUT_T* out_quant,                              \
                                                                  bool         return_normed_output);

INSTANTIATE_TRITON_ADD_BIAS_RESIDUAL_LAYERNORM(float, int8_t, true);
INSTANTIATE_TRITON_ADD_BIAS_RESIDUAL_LAYERNORM(float, int8_t, false);
INSTANTIATE_TRITON_ADD_BIAS_RESIDUAL_LAYERNORM(half, int8_t, true);
INSTANTIATE_TRITON_ADD_BIAS_RESIDUAL_LAYERNORM(half, int8_t, false);
#ifdef ENABLE_BF16
INSTANTIATE_TRITON_ADD_BIAS_RESIDUAL_LAYERNORM(__nv_bfloat16, int8_t, true);
INSTANTIATE_TRITON_ADD_BIAS_RESIDUAL_LAYERNORM(__nv_bfloat16, int8_t, false);
#endif

}  // namespace rtp_llm