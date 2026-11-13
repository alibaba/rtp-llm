#pragma once

#include <list>
#include <future>
#include <memory>
#include <string>

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
                                    const std::shared_ptr<KVCacheManager>& cache_manager,
                                    const kmonitor::MetricsReporterPtr     metrics_reporter = nullptr,
                                    const int                              max_score_len    = 1);

    ~PDFusionRatioScheduler() override;

    absl::StatusOr<ScheduleOutput> schedule() override;

    ScheduleObservation lastScheduleObservation() override;

    // for test
    using FIFOSchedulerBase::runningStreamsSize;
    using FIFOSchedulerBase::waitingStreamsSize;
    int64_t pendingDecodeStreamsSize();
    int64_t decodeSincePrefillForTest();

protected:
    enum class RoundType {
        PREFILL,
        DECODE
    };

    const char* schedulerName() const override {
        return "PDFusionRatioScheduler";
    }
    bool      evaluateRunningMemory(const std::list<GenerateStreamPtr>& streams,
                                    const GenerateStreamPtr&            new_stream) override;
    bool      evaluatePrefillMemory(const std::list<GenerateStreamPtr>& streams,
                                    const GenerateStreamPtr&            new_stream,
                                    bool                                packed_tokens);
    bool      waitPredicate() override;
    void      cancelExtraStreams() override;
    int64_t   extraOnflightStreams() const override;
    void      fillExtraMetrics(RtpLLMSchedulerMetricsCollector& collector) const override;
    void      appendExtraRunningTaskList(std::vector<EngineScheduleInfo::TaskInfo>& task_list) const override;
    bool      classifyActivePrefillBatch();
    bool      promotePendingDecodeStreams();
    RoundType chooseRound();
    bool      tryAdmitKVForPrefill(const GenerateStreamPtr& new_stream);
    void      advanceLateMultimodal(std::unique_lock<std::mutex>& lock, bool allow_prepare);

    void                admitWaitingForCoordination(const std::list<GenerateStreamPtr>& ready_reservations = {});
    void                observeCommitted(const std::list<GenerateStreamPtr>& streams, bool prefill);
    const bool          current_kv_admission_;
    const bool          trace_enabled_;
    ScheduleObservation observation_;

    struct AdmissionPeakState;
    void buildAdmissionPeakState(const std::list<GenerateStreamPtr>& ready_reservations = {});
    bool tryAddToAdmissionPeakState(const GenerateStreamPtr& new_stream,
                                    int64_t                  initial_capacity,
                                    int64_t                  lifecycle_capacity);

    // new_streams_ keeps the admitted prefill cohort across chunks. The flag marks that the cohort
    // returned by the previous schedule round must be classified before it can be scheduled again.
    bool                         has_unclassified_prefill_batch_ = false;
    std::list<GenerateStreamPtr> pending_decode_streams_;
    const size_t late_multimodal_batch_limit_;
    // One preparation job per rank; retain its cohort until Prefill consumes it.
    std::vector<GenerateStreamPtr> late_multimodal_streams_;
    std::future<std::function<void()>> late_multimodal_future_;
    // Per-prefill-round scratch state, guarded by the scheduler lock.
    std::unique_ptr<AdmissionPeakState> admission_peak_state_;
    int64_t                             decode_prefill_step_  = 1;
    int64_t                             decode_since_prefill_ = 0;
    int64_t                             prefill_since_decode_ = 0;
};

}  // namespace rtp_llm
