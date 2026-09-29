// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <ATen/ATen.h>
namespace rtp_llm {
at::Tensor kimi_k3_rms_norm(const at::Tensor& input, const at::Tensor& weight, double epsilon);
}
