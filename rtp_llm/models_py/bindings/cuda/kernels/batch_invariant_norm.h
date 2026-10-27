// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/ATen.h>

namespace torch_ext {
// Input is the already BF16-rounded residual sum. Uses the scalar reduction
// selected by vLLM 0.29.0 fused_add_rms_norm in batch-invariant mode.
at::Tensor batch_invariant_post_residual_rmsnorm(const at::Tensor& input, const at::Tensor& weight, double epsilon);
}  // namespace torch_ext
