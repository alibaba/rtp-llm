#pragma once

#include "src/fastertransformer/core/Tensor.h"

namespace fastertransformer {

class AttentionBuffers {
public:
    Tensor  qkv_buf;
    Tensor  qkv_buf_2;
    Tensor  q_buf_2;
    Tensor  k_buf_2;
    Tensor  v_buf_2;
    Tensor  qk_buf;
    Tensor  partial_out;
    Tensor  partial_sum;   // float
    Tensor  partial_max;   // float
    Tensor  block_counter; // int
    Tensor  softmax_lse;   // float
    Tensor  qk_buf_float;  // float

    Tensor  mixed_gemm_workspace;
    Tensor  int8_gemm_workspace;
};

class FfnBuffers {
    Tensor inter_buf;
    Tensor inter_buf_2;
    Tensor inter_buf_normed;

    Tensor mixed_gemm_workspace;

    Tensor moe_gates_buf;
    Tensor moe_fc_workspace;
};

class SamplerBuffers {
};

}  // namespace fastertransformer
