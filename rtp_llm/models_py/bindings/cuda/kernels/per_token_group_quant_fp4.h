#pragma once

#include <torch/extension.h>

namespace rtp_llm {

// Per-token-group FP4 e2m1 quant (indexer / deep_gemm layout).
// group_size must be 32 (v1). When use_packed_ue8m0 is true, output_s is int32
// with 4 UE8M0 exponent bytes packed per scale slot (deep_gemm convention).
void per_token_group_quant_fp4(torch::Tensor& input,
                               torch::Tensor& output_q,
                               torch::Tensor& output_s,
                               int64_t        group_size,
                               double         eps,
                               bool           use_packed_ue8m0);

}  // namespace rtp_llm
