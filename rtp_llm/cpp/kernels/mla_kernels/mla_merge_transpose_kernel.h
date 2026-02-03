#pragma once

#ifdef USING_ROCM
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#endif

#if USING_CUDA
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "rtp_llm/cpp/kernels/mla_kernels/utils.cuh"
#endif
#include <cstdint>

namespace rtp_llm {

template<typename T>
void invokeMlaMergeTranspose(T*           q,
                             T*           k_nope,
                             T*           k_rope,
                             T*           v,
                             T*           qkv,
                             int          token_num,
                             int          head_num,
                             int          nope_head_dim,
                             int          rope_head_dim,
                             int          v_head_dim,
                             cudaStream_t stream);

template<typename T>
void invokeMlaQKVMerge(T*           q,
                       T*           k_nope,
                       T*           k_rope,
                       T*           v,
                       T*           qkv,
                       int          token_num,
                       int          head_num,
                       int          nope_head_dim,
                       int          rope_head_dim,
                       int          v_head_dim,
                       cudaStream_t stream);

#if USING_CUDA
// Fused kernel to concatenate k_nope and k_pe in one operation
template<typename T>
void invokeMlaKMerge(T*            k_out,
                     T*            k_nope,
                     T*            k_pe,
                     const int     num_tokens,
                     const int64_t k_stride_0,
                     const int     k_stride_1,
                     const int64_t k_nope_stride_0,
                     const int     k_nope_stride_1,
                     const int64_t k_rope_stride_0,
                     cudaStream_t  stream);

// Fused concat a[..., :512] + b[..., :64] -> out[..., :576]. Same as sglang concat_mla_absorb_q.
template<typename T>
void invokeMlaQMerge(T*            a,
                     T*            b,
                     T*            out,
                     const int     num_items,
                     const int     dim_1,
                     const int64_t a_stride_0,
                     const int     a_stride_1,
                     const int64_t b_stride_0,
                     const int     b_stride_1,
                     const int64_t out_stride_0,
                     const int     out_stride_1,
                     cudaStream_t  stream);
#endif
}  // namespace rtp_llm