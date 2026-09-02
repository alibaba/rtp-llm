#pragma once

#include <torch/extension.h>

namespace rtp_llm {

void FlashMlaMergeAttentionStatesSegmentedInPlace(torch::Tensor output,
                                                  torch::Tensor output_lse,
                                                  torch::Tensor partial_output,
                                                  torch::Tensor partial_lse,
                                                  torch::Tensor partial_q_indptr,
                                                  torch::Tensor destination_starts);

}  // namespace rtp_llm
