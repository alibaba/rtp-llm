#pragma once

#include <assert.h>

#if USING_ROCM
#include "rtp_llm/cpp/cuda/cuda_type_utils.cuh"
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#endif
#if USING_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
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

}  // namespace rtp_llm
