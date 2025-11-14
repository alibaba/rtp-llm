#pragma once

#include <vector>
#include <torch/extension.h>
#include <torch/all.h>
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/kernels/rocm/layernorm_kernels.h"

namespace rtp_llm {

void fused_add_layernorm(at::Tensor& input,
                         at::Tensor& residual,
                         at::Tensor& bias,
                         at::Tensor& weight,
                         at::Tensor& beta,
                         double      eps,
                         int64_t     hip_stream);

void layernorm(
    at::Tensor& output, at::Tensor& input, at::Tensor& weight, at::Tensor& beta, double eps, int64_t hip_stream);

}  // namespace rtp_llm
