#pragma once

#include <torch/extension.h>

namespace torch_ext {
// Temperature 1 skips scaling; otherwise uses the existing 1/(T+1e-6) convention.
// Unsupported inputs return undefined tensors without modifying logits.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
deferred_compact_topk_topp(const torch::Tensor& logits,
                           int64_t              request_top_k,
                           float                top_p,
                           float                temperature    = 1.0f,
                           torch::Tensor*       log_normalizer = nullptr);
// Returns false without modifying logits when this fast path is unsupported.
bool deferred_topk_topp_filter(const torch::Tensor& logits, float top_p);
}  // namespace torch_ext
