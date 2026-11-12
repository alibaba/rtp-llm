#include "rtp_llm/cpp/models/eplb/test/ExpertLoadMetricsReference.h"

#include <algorithm>
#include <stdexcept>

namespace rtp_llm {

ExpertLoadWindowMetrics reduceExpertLoadWindowReference(
    const std::vector<int64_t>& packed, size_t sample_count, size_t layer_count, size_t world_size, size_t ep_size) {
    if (layer_count == 0 || world_size == 0 || (ep_size != 1 && ep_size != world_size)
        || packed.size() != world_size * sample_count * layer_count) {
        throw std::invalid_argument("invalid expert load window shape or EP layout");
    }
    ExpertLoadWindowMetrics result;
    result.layers.resize(layer_count);
    result.aggregate.rank_activations.resize(ep_size, 0);
    for (auto& layer : result.layers) {
        layer.rank_activations.resize(ep_size, 0);
    }
    std::vector<int64_t> rank_counts(ep_size);
    for (size_t sample = 0; sample < sample_count; ++sample) {
        for (size_t layer = 0; layer < layer_count; ++layer) {
            std::fill(rank_counts.begin(), rank_counts.end(), 0);
            for (size_t rank = 0; rank < world_size; ++rank) {
                rank_counts[rank % ep_size] += packed[(rank * sample_count + sample) * layer_count + layer];
            }
            auto& value = result.layers[layer];
            value.max_rank_activations += *std::max_element(rank_counts.begin(), rank_counts.end());
            for (size_t rank = 0; rank < ep_size; ++rank) {
                value.rank_activations[rank] += rank_counts[rank];
                value.total_activations += rank_counts[rank];
            }
        }
    }
    for (auto& layer : result.layers) {
        if (layer.total_activations != 0) {
            layer.moe_lb = static_cast<double>(layer.max_rank_activations) * ep_size / layer.total_activations;
        }
        result.aggregate.total_activations += layer.total_activations;
        result.aggregate.max_rank_activations += layer.max_rank_activations;
        for (size_t rank = 0; rank < ep_size; ++rank) {
            result.aggregate.rank_activations[rank] += layer.rank_activations[rank];
        }
    }
    if (result.aggregate.total_activations != 0) {
        result.aggregate.moe_lb =
            static_cast<double>(result.aggregate.max_rank_activations) * ep_size / result.aggregate.total_activations;
    }
    return result;
}

}  // namespace rtp_llm
