#pragma once

#include <ATen/ATen.h>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "rtp_llm/cpp/metrics/EplbStatsMetricsCollector.h"

namespace rtp_llm {

// The producer's INT32 buffer is per step. This separate INT64 accumulator is per report window.
// All operations must be ordered on the producer stream (or after an explicit stream dependency).
class ExpertStatsReportWindow {
public:
    explicit ExpertStatsReportWindow(int64_t interval = 10): interval_(interval) {
        if (interval <= 0) {
            throw std::invalid_argument("EPLB report interval must be positive");
        }
    }

    bool add(const at::Tensor& per_step_counts) {
        TORCH_CHECK(per_step_counts.defined() && per_step_counts.dim() == 2,
                    "EPLB stats must be a defined [layer, ep_rank] tensor");
        TORCH_CHECK(per_step_counts.size(0) > 0 && per_step_counts.size(1) > 0,
                    "EPLB stats must have non-empty dimensions");
        TORCH_CHECK(per_step_counts.scalar_type() == at::kInt || per_step_counts.scalar_type() == at::kLong,
                    "EPLB stats must use INT32 or INT64 counts");
        TORCH_CHECK(steps_ < interval_, "Completed EPLB window must be reset before reuse");
        if (!counts_.defined()) {
            counts_ = at::zeros(per_step_counts.sizes(), per_step_counts.options().dtype(at::kLong));
        }
        TORCH_CHECK(counts_.sizes() == per_step_counts.sizes() && counts_.device() == per_step_counts.device(),
                    "EPLB stats shape/device changed within a model lifetime");
        counts_.add_(per_step_counts);
        return ++steps_ == interval_;
    }

    at::Tensor snapshot() const {
        TORCH_CHECK(counts_.defined() && steps_ == interval_, "EPLB report window is incomplete");
        return counts_.clone();  // Allreduce must not mutate the accumulator or the producer buffer.
    }

    void reset() {
        if (counts_.defined()) {
            counts_.zero_();  // Preserve the allocation/address.
        }
        steps_ = 0;
    }

    int64_t steps() const { return steps_; }

    static void fillCollector(const at::Tensor& aggregated,
                              int64_t steps,
                              int64_t tp_replication_size,
                              RtpLLmEplbMetricsCollector& collector) {
        TORCH_CHECK(aggregated.defined() && aggregated.dim() == 2 && aggregated.scalar_type() == at::kLong,
                    "Aggregated EPLB counts must be INT64 [layer, ep_rank]");
        const auto host = aggregated.cpu().contiguous();
        const auto* data = host.data_ptr<int64_t>();
        std::vector<std::vector<int64_t>> loads(host.size(0), std::vector<int64_t>(host.size(1)));
        for (int64_t layer = 0; layer < host.size(0); ++layer) {
            for (int64_t rank = 0; rank < host.size(1); ++rank) {
                loads[layer][rank] = data[layer * host.size(1) + rank];
            }
        }
        collector.setGpuLoads(std::move(loads), steps, tp_replication_size);
    }

private:
    int64_t interval_;
    int64_t steps_ = 0;
    at::Tensor counts_;
};

}  // namespace rtp_llm
