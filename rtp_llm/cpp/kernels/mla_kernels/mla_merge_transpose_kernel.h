#pragma once

#ifdef USING_ROCM
#include <hip/hip_fp16.h>
#include "rtp_llm/cpp/rocm/hip_utils.h"
#else
#include <cuda_fp16.h>
#include "rtp_llm/cpp/cuda/cuda_utils.h"
#endif

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
}