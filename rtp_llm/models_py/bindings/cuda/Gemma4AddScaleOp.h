#pragma once

#include <torch/extension.h>

namespace torch_ext {

at::Tensor gemma4_add_bf16(const at::Tensor& lhs, const at::Tensor& rhs);
at::Tensor gemma4_scale_bf16(const at::Tensor& input, double scale);
at::Tensor gemma4_add_scale_bf16(const at::Tensor& residual, const at::Tensor& hidden, const at::Tensor& scale);
at::Tensor gemma4_router_scale_bf16(const at::Tensor& input, const at::Tensor& channel_scale, double scalar_scale);

}  // namespace torch_ext
