#pragma once

#include <assert.h>

#if USING_ROCM
#include "rtp_llm/models_py/bindings/cuda/cuda_type_utils.cuh"
#include "rtp_llm/models_py/bindings/rocm/cuda_shims.h"
#endif
#if USING_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#endif

namespace rtp_llm {

// Drop-in faster replacement for invokeFusedQkRmsNorm, ROCm-only.
// Layout assumption (matches the baseline):
//   input is in-place fused QKV [m, n] BF16, where
//     n   = q_group_num * norm_size + k_group_num * norm_size + v_size
//   q occupies columns [0, q_group_num*norm_size)
//   k occupies columns [q_group_num*norm_size, (q_group_num+k_group_num)*norm_size)
//   v slice (n - the above) is left untouched.
// q_gamma, k_gamma are per-head weights of size [norm_size].
// One wave (64 threads) handles one (token, head) work item; 4 waves per block.
template<typename T>
void invokeFusedQkRmsNormV2(T* __restrict       input,
                            const T* __restrict q_gamma,
                            const T* __restrict q_bias,
                            const T* __restrict k_gamma,
                            const T* __restrict k_bias,
                            const float         layernorm_eps,
                            const int           q_group_num,
                            const int           k_group_num,
                            const int           m,
                            const int           n,
                            const int           norm_size,
                            cudaStream_t        stream);

}  // namespace rtp_llm
