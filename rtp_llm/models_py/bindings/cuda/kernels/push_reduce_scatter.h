#pragma once

#include <torch/extension.h>

namespace rtp_llm {

// TP8 BF16 SUM, rank-contiguous row shards. Peer buffers and phase counters
// are allocated/zeroed collectively before use and retained through graph replay.
// Calls sharing these buffers must be serialized, in the same order on all ranks.
void push_reduce_scatter(const torch::Tensor&              input,
                         torch::Tensor&                    output,
                         const std::vector<torch::Tensor>& peers,
                         torch::Tensor&                    counters,
                         int64_t                           rank,
                         int64_t                           blocks,
                         int64_t                           threads);

}  // namespace rtp_llm
