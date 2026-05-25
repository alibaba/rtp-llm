#pragma once

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOSchedulerBase.h"
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/schedulers/EngineScheduleInfo.h"
namespace rtp_llm {

class FIFOScheduler: public FIFOSchedulerBase {
public:
    explicit FIFOScheduler(const RuntimeConfig&                   runtime_config,
                           const ModelConfig&                     model_config,
                           const PDSepConfig&                     pd_sep_config,
                           const ParallelismConfig&               parallelism_config,
                           const ModelSpecificConfig&             model_specific_config,
                           const std::shared_ptr<KVCacheManager>& cache_manager,
                           const kmonitor::MetricsReporterPtr     metrics_reporter = nullptr,
                           const int                              max_score_len    = 1,
                           bool                                   enable_batch_cache_reuse = false);

    ~FIFOScheduler() override;

    absl::StatusOr<std::list<GenerateStreamPtr>> schedule() override;

public:
    // for test
    using FIFOSchedulerBase::runningStreamsSize;
    using FIFOSchedulerBase::waitingStreamsSize;

private:
    const char* schedulerName() const override {
        return "FIFOScheduler";
    }
    bool evaluateRunningMemory(const std::list<GenerateStreamPtr>& streams,
                               const GenerateStreamPtr&            new_stream) const override;
    void accountBatchMetrics(const GenerateStreamPtr& new_stream);
    bool waitPredicate() override;
    void onRunningStream(const GenerateStreamPtr& stream) override;
    int64_t nextBatchEpoch() override;

    std::atomic<int64_t> batch_epoch_counter_{0};

    // Feature toggle: when false, epoch is always 0 (no batch-level cache reuse)
    bool enable_batch_cache_reuse_ = false;

    // TODO @wangyin support different beams run togather
};

}  // namespace rtp_llm
