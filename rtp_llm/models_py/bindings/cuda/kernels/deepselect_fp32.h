#pragma once

#include <torch/extension.h>

namespace torch_ext {
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
deferred_compact_topk_topp(const torch::Tensor& logits, int64_t k, float top_p);
// Returns false without modifying logits when this fast path is unsupported.
bool deferred_topk_topp_filter(const torch::Tensor& logits, int64_t k, float top_p);
std::tuple<torch::Tensor, torch::Tensor>
deepselect_fp32_topk(const torch::Tensor& logits, int64_t k, bool* used_deepselect = nullptr);
}
