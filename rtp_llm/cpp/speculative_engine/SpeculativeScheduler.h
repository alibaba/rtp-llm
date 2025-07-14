#pragma once

#include "rtp_llm/cpp/schedulers/FIFOScheduler.h"

namespace rtp_llm {

class SpeculativeScheduler: public FIFOScheduler {
public:
    explicit SpeculativeScheduler(const rtp_llm::GptInitParameter&     params,
                                  const std::shared_ptr<CacheManager>& cache_manager,
                                  const kmonitor::MetricsReporterPtr   metrics_reporter = nullptr):
        FIFOScheduler(params, cache_manager, metrics_reporter) {}

    absl::StatusOr<std::list<GenerateStreamPtr>> schedule(size_t reserve_step = 0) override;
    bool                                         empty() override {
        return pending_sp_run_streams_.empty() && FIFOScheduler::empty();
    };

private:
    std::list<GenerateStreamPtr> pending_sp_run_streams_;
};

}  // namespace rtp_llm
