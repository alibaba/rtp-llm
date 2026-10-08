// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <ATen/ATen.h>
namespace rtp_llm {
void kimi_k3_fused_a_gemm(at::Tensor& output, const at::Tensor& input, const at::Tensor& weight, bool enable_pdl);
}
