#pragma once

#include <torch/extension.h>

namespace torch_ext {
std::tuple<torch::Tensor, torch::Tensor>
deepselect_fp32_topk(const torch::Tensor& logits, int64_t k, bool* used_deepselect = nullptr);
}
