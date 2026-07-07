#pragma once

#include <list>
#include <memory>
#include <string>
#include <vector>

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/schedulers/EngineScheduleInfo.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOSchedulerBase.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"

namespace rtp_llm {

class PDFusionRatioScheduler: public FIFOSchedulerBase {
public:
    explicit PDFusionRatioScheduler(const RuntimeConfig&                   runtime_config,
                                    const ModelConfig&                     model_config,
                                    const PDSepConfig&                     pd_sep_config,
                                    const ParallelismConfig&               parallelism_config,
                                    const ModelSpecificConfig&             model_specific_config,
                                    const std::shared_ptr<KVCacheManager>& cache_manager,
                                    const kmonitor::MetricsReporterPtr     metrics_reporter = nullptr,
                                    const int                              max_score_len    = 1);

    ~PDFusionRatioScheduler() override;

    absl::StatusOr<std::list<GenerateStreamPtr>> schedule() override;

    // for test
    using FIFOSchedulerBase::runningStreamsSize;
    using FIFOSchedulerBase::waitingStreamsSize;
    int64_t pendingDecodeStreamsSize();
    int64_t decodeSincePrefillForTest();

private:
    enum class RoundType {
        PREFILL,
        DECODE
    };

    const char* schedulerName() const override {
        return "PDFusionRatioScheduler";
    }
    bool      evaluateRunningMemory(const std::list<GenerateStreamPtr>& streams,
                                    const GenerateStreamPtr&            new_stream) const override;
    bool      waitPredicate() override;
    void      cancelExtraStreams() override;
    bool      hasExtraStreams() const override;
    int64_t   extraOnflightStreams() const override;
    void      fillExtraMetrics(RtpLLMSchedulerMetricsCollector& collector) const override;
    size_t    reapErroredWaitingStreams();
    size_t    reapFinished(std::list<GenerateStreamPtr>& streams);
    size_t    promotePendingDecodeStreams();
    RoundType chooseRound();

private:
    std::list<GenerateStreamPtr> pending_decode_streams_;
    int64_t                      decode_prefill_step_  = 1;
    int64_t                      decode_since_prefill_ = 0;
    int64_t                      prefill_since_decode_ = 0;
};

}  // namespace rtp_llm
