#include "rtp_llm/cpp/models/eplb/ExpertLoadMetrics.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace rtp_llm {
namespace {
size_t checkedSizeProduct(size_t a, size_t b) {
    if (b != 0 && a > std::numeric_limits<size_t>::max() / b) {
        throw std::invalid_argument("expert load history dimensions overflow");
    }
    return a * b;
}

int64_t checkedCountAdd(int64_t a, int64_t b) {
    if (b < 0) {
        throw std::invalid_argument("expert activation counts must be non-negative");
    }
    if (a > std::numeric_limits<int64_t>::max() - b) {
        throw std::overflow_error("expert activation count overflow");
    }
    return a + b;
}
}  // namespace

std::vector<ExpertLoadLayerMetrics> reduceExpertLoadWindow(const std::vector<int64_t>& rank_history,
                                                           size_t                      sample_count,
                                                           size_t                      layer_count,
                                                           size_t                      ep_size) {
    if (layer_count == 0 || ep_size == 0) {
        throw std::invalid_argument("expert load layer and EP counts must be positive");
    }
    const size_t layer_stride = checkedSizeProduct(layer_count, ep_size);
    if (rank_history.size() != checkedSizeProduct(sample_count, layer_stride)) {
        throw std::invalid_argument("expert load history shape mismatch");
    }

    std::vector<ExpertLoadLayerMetrics> result(layer_count);
    for (size_t layer = 0; layer < layer_count; ++layer) {
        auto& metrics = result[layer];
        metrics.rank_activations.assign(ep_size, 0);
        int64_t step_max_sum = 0;
        for (size_t sample = 0; sample < sample_count; ++sample) {
            int64_t step_max = 0;
            for (size_t rank = 0; rank < ep_size; ++rank) {
                const auto count               = rank_history[sample * layer_stride + layer * ep_size + rank];
                metrics.rank_activations[rank] = checkedCountAdd(metrics.rank_activations[rank], count);
                metrics.total_activations      = checkedCountAdd(metrics.total_activations, count);
                step_max                       = std::max(step_max, count);
            }
            step_max_sum = checkedCountAdd(step_max_sum, step_max);
        }
        if (metrics.total_activations == 0) {
            continue;
        }
        metrics.valid                 = true;
        const long double denominator = static_cast<long double>(metrics.total_activations) / ep_size;
        metrics.moe_lb                = static_cast<double>(static_cast<long double>(step_max_sum) / denominator);
        const auto window_max = *std::max_element(metrics.rank_activations.begin(), metrics.rank_activations.end());
        metrics.moe_lb_window = static_cast<double>(static_cast<long double>(window_max) / denominator);
    }
    return result;
}

}  // namespace rtp_llm
