#pragma once

#include <ATen/ATen.h>
#include <torch/all.h>

namespace rtp_llm {

void persistent_topk(const torch::Tensor& logits,
                     const torch::Tensor& lengths,
                     torch::Tensor&       output,
                     torch::Tensor&       workspace,
                     int64_t              k,
                     int64_t              max_seq_len);

}  // namespace rtp_llm
