#pragma once

#include <torch/extension.h>

#include <tuple>

namespace torch_ext {

std::tuple<at::Tensor, at::Tensor> gemma4_expand_kv_heads_8_bf16(const at::Tensor& k, const at::Tensor& v);
std::tuple<at::Tensor, at::Tensor> gemma4_expand_kv_heads_2_bf16(const at::Tensor& k, const at::Tensor& v);

}  // namespace torch_ext
