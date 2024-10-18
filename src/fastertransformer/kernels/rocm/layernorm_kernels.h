#pragma once

#include "src/fastertransformer/utils/layernorm_types.h"
#include "src/fastertransformer/cuda/cuda_type_utils.cuh"
#include <assert.h>

namespace fastertransformer {

template<typename T>
struct LayerNormWeight {
    const T* gamma = nullptr;
    const T* beta  = nullptr;
};

template<typename T>
void invokeGeneralAddBiasResidualLayerNorm(T*           out,
                                           T*           norm_output,
                                           const T*     input,
                                           const T*     bias,
                                           const T*     residual,
                                           const T*     gamma,
                                           const T*     beta,
                                           const float  eps,
                                           const int    tokens,
                                           const int    hidden_dim,
                                           cudaStream_t stream               = 0,
                                           bool         use_diff_of_squares  = true,
                                           const float* scale                = nullptr,
                                           float*       dynamic_scale        = nullptr,
                                           int8_t*      out_quant            = nullptr,
                                           bool         return_normed_output = false);

template<typename T>
void invokeGeneralLayerNorm(T*           out,
                            T*           normed_output,
                            const T*     input,
                            const T*     gamma,
                            const T*     beta,
                            const float  eps,
                            const int    tokens,
                            const int    hidden_dim,
                            cudaStream_t stream               = 0,
                            bool         use_diff_of_squares  = true,
                            const float* scale                = nullptr,
                            float*       dynamic_scale        = nullptr,
                            int8_t*      out_quant            = nullptr,
                            bool         return_normed_output = false);

template<typename T>
void invokeQkLayerNorm(T* __restrict qkv,
                       const T* __restrict gamma,
                       const float  layernorm_eps,
                       const int    tokens,
                       const int    head_num,
                       const int    head_num_kv,
                       const int    size_per_head,
                       cudaStream_t stream = 0);

template<typename T>
void invokeLayerNormWithStride(T* __restrict data,
                               const T* __restrict gamma,
                               const T* __restrict beta,
                               const float  layernorm_eps,
                               const int    tokens,
                               const int    hidden_size,
                               const int    stride,
                               const int    offset,
                               cudaStream_t stream);

}  // namespace fastertransformer
