#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rtp_llm {

struct ExpertLoadLayerMetrics {
    bool                 valid             = false;
    double               moe_lb            = 0.0;
    double               moe_lb_window     = 0.0;
    int64_t              total_activations = 0;
    std::vector<int64_t> rank_activations;
};

// rank_history is already summed across unique source contributions and is
// laid out as [sample, layer, destination_ep_rank]. This function does not
// perform communication and must not be called separately before source SUM.
// Empty/all-zero samples yield valid=false. Negative counts, bad shapes and
// int64 count overflow are rejected.
std::vector<ExpertLoadLayerMetrics> reduceExpertLoadWindow(const std::vector<int64_t>& rank_history,
                                                           size_t                      sample_count,
                                                           size_t                      layer_count,
                                                           size_t                      ep_size);

}  // namespace rtp_llm
