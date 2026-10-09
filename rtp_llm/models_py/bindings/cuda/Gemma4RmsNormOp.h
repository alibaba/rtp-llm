#pragma once

#include <torch/extension.h>

namespace torch_ext {

at::Tensor gemma4_rms_square_bf16(const at::Tensor& input);
at::Tensor gemma4_rms_mean_fp32(const at::Tensor& input);
at::Tensor gemma4_rms_inv_fp32(const at::Tensor& input, double eps);
at::Tensor gemma4_rms_apply_bf16(const at::Tensor& input, const at::Tensor& inv_rms, const at::Tensor& weight);
at::Tensor gemma4_rms_apply_unweighted_bf16(const at::Tensor& input, const at::Tensor& inv_rms);

}  // namespace torch_ext
