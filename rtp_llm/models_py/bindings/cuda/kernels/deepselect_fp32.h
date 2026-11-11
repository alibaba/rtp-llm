#pragma once

#include <torch/extension.h>

namespace torch_ext {
// Temperature 1 skips scaling; otherwise uses the existing 1/(T+1e-6) convention.
// Unsupported inputs return undefined tensors without modifying logits.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
deferred_compact_topk_topp(const torch::Tensor& logits, int64_t k, float top_p, float temperature = 1.0f);
// Returns false without modifying logits when this fast path is unsupported.
bool deferred_topk_topp_filter(const torch::Tensor& logits, int64_t k, float top_p);
std::tuple<torch::Tensor, torch::Tensor>
deepselect_fp32_topk(const torch::Tensor& logits, int64_t k, bool* used_deepselect = nullptr);
}
