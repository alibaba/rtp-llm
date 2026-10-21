#pragma once

#include <assert.h>

#if USING_ROCM
#include "rtp_llm/models_py/bindings/cuda/cuda_type_utils.cuh"
#include "rtp_llm/models_py/bindings/rocm/cuda_shims.h"
#endif
#if USING_CUDA
#include <cuda_fp16.h>
#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif
#include <cuda_runtime.h>
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#endif

namespace rtp_llm {

template<typename T>
void invokeFusedQkRmsNorm(T* __restrict input,
                          const T* __restrict q_gamma,
                          const T* __restrict q_bias,
                          const T* __restrict k_gamma,
                          const T* __restrict k_bias,
                          const float  layernorm_eps,
                          const int    q_group_num,
                          const int    k_group_num,
                          const int    m,
                          const int    n,
                          const int    norm_size,
                          cudaStream_t stream);

#if USING_CUDA && defined(ENABLE_BF16)
// CUDA-only, BF16 Q16/K8/D128 packed input; PDL is deliberately disabled.
cudaError_t invokeFusedQkRmsNormCudaVec8(__nv_bfloat16*       input,
                                         const __nv_bfloat16* q_weight,
                                         const __nv_bfloat16* k_weight,
                                         uint32_t             m,
                                         float                eps,
                                         int                  device_id,
                                         int                  sm_count,
                                         cudaStream_t         stream);
#endif

}  // namespace rtp_llm
