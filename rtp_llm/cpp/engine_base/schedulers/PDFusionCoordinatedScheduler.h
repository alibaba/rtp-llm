#pragma once
// Single-host prepare/commit scheduler for the opt-in global cadence mode.
#include "rtp_llm/cpp/engine_base/schedulers/PDFusionRatioScheduler.h"
#include "rtp_llm/cpp/engine_base/schedulers/PDFusionScheduleCoordinator.h"
#include <unordered_map>
namespace rtp_llm {
class PDFusionCoordinatedScheduler: public PDFusionRatioScheduler {
public:
    PDFusionCoordinatedScheduler(const RuntimeConfig&                   runtime_config,
                                 const ModelConfig&                     model_config,
                                 const PDSepConfig&                     pd_sep_config,
                                 const ParallelismConfig&               parallelism_config,
                                 const ModelSpecificConfig&             model_specific_config,
                                 const std::shared_ptr<KVCacheManager>& cache_manager,
                                 const kmonitor::MetricsReporterPtr     metrics_reporter = nullptr);
    absl::StatusOr<std::list<GenerateStreamPtr>> schedule() override;
    // No network in these methods. Shared pointers and allocated KV form the reservation.
    PDFusionPreparedState        prepareLocal(int64_t epoch);
    std::list<GenerateStreamPtr> commitLocal(int64_t epoch, PDFusionPlan plan);
    void                         completeModelStep();
    absl::Status                 stop() override;
    void                         releaseStoppedStreams();  // Engine calls only after joining its loop thread.
    const PDFusionPreparedState& lastPrepared() const {
        return last_prepared_;
    }
    bool globalIdle() const {
        return final_real_count_ == 0;
    }
    int64_t controlEpoch() const {
        return epoch_;
    }
    PDFusionPlan plan() const {
        return plan_;
    }
    int64_t preparedMask() const {
        return ready_mask_;
    }
    int64_t committedMask() const {
        return commit_mask_;
    }
    int64_t prepareUs() const {
        return prepare_us_;
    }
    int64_t controlUs() const {
        return control_us_;
    }
    int64_t commitUs() const {
        return commit_us_;
    }

private:
    bool        evaluateRunningMemory(const std::list<GenerateStreamPtr>& streams,
                                      const GenerateStreamPtr&            candidate) override;
    int64_t     extraOnflightStreams() const override;
    void        cancelExtraStreams() override;
    const char* schedulerName() const override {
        return "PDFusionCoordinatedScheduler";
    }
    PDFusionPreparedState                        last_prepared_;
    std::string                                  run_id_;
    int                                          rank_, timeout_ms_;
    int64_t                                      decode_steps_;
    std::unique_ptr<PDFusionScheduleCoordinator> coordinator_;
    PDFusionGlobalCadence                        cadence_;
    int64_t                                      epoch_ = 0, prepared_epoch_ = 0, committed_epoch_ = 0;
    bool                                         model_completed_  = true;
    PDFusionPlan                                 plan_             = PDFusionPlan::IDLE;
    int64_t                                      final_real_count_ = 0, ready_mask_ = 0, commit_mask_ = 0;
    int64_t                                      prepare_us_ = 0, control_us_ = 0, commit_us_ = 0;
    int64_t                                      reserved_since_us_ = 0;
    std::unordered_map<int64_t, int64_t>         decode_last_served_us_;
    std::list<GenerateStreamPtr>                 prepared_decode_, committed_;
};
}  // namespace rtp_llm
