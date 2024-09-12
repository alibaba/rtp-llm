#pragma once

#include <cuda_fp16.h>
#include "src/fastertransformer/cuda/cuda_utils.h"

namespace fastertransformer {

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
}