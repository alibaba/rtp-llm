#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rtp_llm {

// One complete reporting window, not one source rank's destination column.
class RtpLLmEplbMetricsCollector final {
public:
    // Retained for the existing weight-update metrics; reporting does not trigger updates.
    int64_t ep_rank = 0;
    int64_t update_layer_id = 0;
    int64_t update_weights_latency_ms = 0;
    bool update_weights_qps = false;

    std::vector<std::vector<int64_t>> gpu_loads;  // [model layer][destination EP rank]
    std::vector<double> ib_layers;
    double ib_layer_mean = 0.0;
    int64_t window_steps = 0;
    int64_t active_layer_count = 0;
    bool has_valid_load = false;

    // tp_replication_size must describe proven identical dispatches, NOT arbitrary TP/DP/CP size.
    // Default 1 also covers producer-side TP representative selection (never deduplicate twice).
    void setGpuLoads(std::vector<std::vector<int64_t>> loads,
                     int64_t steps,
                     int64_t tp_replication_size = 1) {
        if (steps <= 0 || tp_replication_size <= 0 || loads.empty() || loads.front().empty()) {
            throw std::invalid_argument("Invalid EPLB reporting window");
        }
        const auto ep_size = loads.front().size();
        std::vector<double> imbalance(loads.size(), 0.0);
        int64_t active = 0;
        double imbalance_sum = 0.0;
        for (size_t layer = 0; layer < loads.size(); ++layer) {
            auto& row = loads[layer];
            if (row.size() != ep_size) {
                throw std::invalid_argument("Ragged EPLB GPU-load matrix");
            }
            long double total = 0;
            int64_t maximum = 0;
            for (auto& count : row) {
                if (count < 0 || count % tp_replication_size != 0) {
                    throw std::invalid_argument("Invalid count or non-identical TP dispatch replication");
                }
                count /= tp_replication_size;
                total += count;
                maximum = std::max(maximum, count);
            }
            if (total > 0) {
                imbalance[layer] = static_cast<double>(maximum * static_cast<long double>(ep_size) / total - 1);
                imbalance_sum += imbalance[layer];
                ++active;
            }
        }
        gpu_loads = std::move(loads);
        ib_layers = std::move(imbalance);
        window_steps = steps;
        active_layer_count = active;
        has_valid_load = active > 0;
        ib_layer_mean = active ? imbalance_sum / active : 0.0;
    }
};

}  // namespace rtp_llm
