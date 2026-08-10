#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <tuple>
#include <unordered_set>
#include <vector>
#include <atomic>
#include <unordered_map>
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerBase.h"
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/schedulers/EngineScheduleInfo.h"
namespace rtp_llm {

enum class ForceBatchMode {
    NORMAL,
    WAITING,
    ATOMIC,
};

class FIFOScheduler: public SchedulerBase {
public:
    explicit FIFOScheduler(const RuntimeConfig&                   runtime_config,
                           const ModelConfig&                     model_config,
                           const PDSepConfig&                     pd_sep_config,
                           const ParallelismConfig&               parallelism_config,
                           const ModelSpecificConfig&             model_specific_config,
                           const std::shared_ptr<KVCacheManager>& cache_manager,
                           const kmonitor::MetricsReporterPtr     metrics_reporter = nullptr,
                           const int                              max_score_len    = 1);

    ~FIFOScheduler() override;

    // Enqueue a single stream. Returns OkStatus on success, InvalidArgumentError if checkInputLength fails.
    // On failure, the stream's error is reported via reportError() but the stream is NOT queued.
    // Caller must check the return status to know whether the stream was actually enqueued.
    absl::Status enqueue(const GenerateStreamPtr& stream) override;

    // Enqueue multiple streams. Returns the input vector unchanged so callers can index 1:1 with
    // their original input list. Streams that fail checkInputLength are NOT added to the waiting
    // queue, but their errors are reported via reportError() and they are still returned in the
    // output vector — collectStreamOutput / pollStreamOutput will surface the error per-stream.
    std::vector<std::shared_ptr<GenerateStream>> batchEnqueue(const std::vector<GenerateStreamPtr>& streams) override;
    absl::StatusOr<std::list<GenerateStreamPtr>> schedule() override;
    absl::Status                                 stop() override;
    bool                                         empty() override;

    void reportMetrics();

public:
    // for test
    int64_t                                   waitingStreamsSize();
    int64_t                                   runningStreamsSize();
    std::vector<EngineScheduleInfo::TaskInfo> waitingTaskList();
    std::vector<EngineScheduleInfo::TaskInfo> runningTaskList();
    int64_t                                   onflightStreams() override;

private:
    struct ForceBatchAdmissionIdentity {
        int64_t                                   group_id             = -1;
        int                                       expected_size        = 0;
        int64_t                                   fallback_deadline_ms = 0;
        bool                                      broken               = false;
        std::unordered_set<const GenerateStream*> members;
    };

    int64_t lastScheduleTime() override;
    bool evaluateRunningMemory(const std::list<GenerateStreamPtr>& streams, const GenerateStreamPtr& new_stream) const;
    void accountBatchMetrics(const GenerateStreamPtr& new_stream);
    bool waitPredicate();
    size_t coalescibleContextBatchSize() const;
    bool   shouldCoalesceContextBatch() const;
    void   waitForContextBatch(std::unique_lock<std::mutex>& lock);
    bool   contextLoadCohortEnabled() const;
    bool   waitForContextLoadCohort(std::unique_lock<std::mutex>& lock);
    void   removeFromContextLoadCohort(const GenerateStreamPtr& stream);
    ForceBatchMode effectiveForceBatchMode(const GenerateStreamPtr& stream,
                                           ForceBatchMode             base_mode,
                                           int64_t                    now_ms) const;
    void trackCompleteForceBatchAdmissions(const std::list<GenerateStreamPtr>& admitted_streams, int64_t now_ms);
    void removeFromForceBatchAdmission(const GenerateStreamPtr& stream, bool broken_before_run);
    void addStreamToNewState(const GenerateStreamPtr& stream, StreamState new_state);
    void evaluateWaitingStreams(std::list<GenerateStreamPtr>& streams);
    void cancelStreams(std::list<GenerateStreamPtr>& streams);
    bool checkInputLength(const GenerateStreamPtr& stream);

protected:
    void evaluateAndUpdateStreams(std::list<GenerateStreamPtr>& streams);
    virtual void onContextBatchCoalescingWait() {}

protected:
    PDSepConfig                     pd_sep_config_;
    ModelSpecificConfig             model_specific_config_;
    std::list<GenerateStreamPtr>    waiting_streams_;
    std::list<GenerateStreamPtr>    loading_cache_streams_;
    std::list<GenerateStreamPtr>    running_streams_;
    std::list<GenerateStreamPtr>    new_streams_;
    std::shared_ptr<KVCacheManager> cache_manager_;
    std::atomic<int64_t>            last_schedule_time_      = autil::TimeUtility::currentTimeInMilliSeconds();
    size_t                          max_seq_len_             = 0;
    size_t                          max_batch_tokens_size_   = 0;
    size_t                          max_generate_batch_size_ = 1;
    size_t                          max_context_batch_size_  = 1;
    int64_t                         context_batch_coalescing_window_ms_ = 0;
    std::unordered_set<const GenerateStream*> context_load_cohort_;
    std::optional<std::chrono::steady_clock::time_point> context_load_cohort_deadline_;
    std::unordered_map<const GenerateStream*, std::shared_ptr<ForceBatchAdmissionIdentity>>
        force_batch_admission_by_member_;
    std::unordered_set<int64_t> active_force_batch_group_ids_;
    bool                            admit_context_load_cohort_only_ = false;
    const bool                      need_fill_fake_stream_   = false;
    std::atomic<bool>               stop_                    = false;
    bool                            schedule_trigger_        = false;
    std::mutex                      lock_;
    std::condition_variable         cond_;
    kmonitor::MetricsReporterPtr    metrics_reporter_ = nullptr;

    std::vector<EngineScheduleInfo::TaskInfo> waiting_task_list_;
    std::vector<EngineScheduleInfo::TaskInfo> running_task_list_;

    // TODO @wangyin support different beams run togather
};

}  // namespace rtp_llm
