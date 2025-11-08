#include "rtp_llm/cpp/speculative_engine/SpeculativeScheduler.h"
#include "rtp_llm/cpp/engine_base/schedulers/GatherBatchScheduler.h"

namespace rtp_llm {

struct SpeculativeGatherBatchSchedulerConfigLocal: public autil::legacy::Jsonizable {
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("batch_size", batch_size_);
    }
    uint32_t batch_size_;
};

class SpeculativeGatherBatchScheduler: public SpeculativeScheduler, public GatherBatchScheduler {
public:
    explicit SpeculativeGatherBatchScheduler(const rtp_llm::GptInitParameter&        params,
                                             const std::shared_ptr<KVCacheManager>& cache_manager,
                                             const kmonitor::MetricsReporterPtr      metrics_reporter = nullptr,
                                             const int                               max_score_len    = 1):
        FIFOScheduler(params, cache_manager, metrics_reporter, max_score_len),
        SpeculativeScheduler(params, cache_manager, metrics_reporter, max_score_len),
        GatherBatchScheduler(params, cache_manager, metrics_reporter, max_score_len) {}
};
}  // namespace rtp_llm