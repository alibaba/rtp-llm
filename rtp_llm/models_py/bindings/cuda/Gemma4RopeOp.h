#pragma once

#include <torch/extension.h>

#include <tuple>

namespace torch_ext {

std::tuple<at::Tensor, at::Tensor> gemma4_rope_cos_sin_bf16(const at::Tensor& positions, const at::Tensor& inv_freq);
std::tuple<at::Tensor, at::Tensor>
gemma4_qk_rope_bf16(const at::Tensor& q, const at::Tensor& k, const at::Tensor& cos, const at::Tensor& sin);

}  // namespace torch_ext
