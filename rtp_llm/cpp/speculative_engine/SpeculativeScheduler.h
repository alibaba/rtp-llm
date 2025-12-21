#pragma once

#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"

namespace rtp_llm {

class SpeculativeScheduler: virtual public FIFOScheduler {
public:
    explicit SpeculativeScheduler(const RuntimeConfig&                 runtime_config,
                                  const ModelConfig&                   model_config,
                                  const PDSepConfig&                  pd_sep_config,
                                  const ParallelismConfig&            parallelism_config,
                                  const ModelSpecificConfig&          model_specific_config,
                                  const std::shared_ptr<KVCacheManager>& cache_manager,
                                  const kmonitor::MetricsReporterPtr   metrics_reporter = nullptr,
                                  const int                            max_score_len    = 1):
        FIFOScheduler(runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager, metrics_reporter, max_score_len) {}

    absl::StatusOr<std::list<GenerateStreamPtr>> schedule(size_t reserve_step = 0) override;
    bool                                         empty() override {
        return pending_sp_run_streams_.empty() && FIFOScheduler::empty();
    };

private:
    std::list<GenerateStreamPtr> pending_sp_run_streams_;
};

}  // namespace rtp_llm
