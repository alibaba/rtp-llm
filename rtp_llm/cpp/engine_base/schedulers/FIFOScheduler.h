#pragma once

#include <atomic>
#include <list>
#include <string>
#include <vector>

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/schedulers/EngineScheduleInfo.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerBase.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"

namespace rtp_llm {

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

    absl::Status enqueue(const GenerateStreamPtr& stream) override;
    std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
                                                 enqueueGroup(const std::vector<GenerateStreamPtr>& streams) override;
    absl::StatusOr<std::list<GenerateStreamPtr>> schedule() override;
    absl::Status                                 stop() override;
    void                                         wake() override;
    void setForcePoll(bool enable) override {
        force_poll_.store(enable, std::memory_order_relaxed);
    }
    bool                                         empty() override;

    void reportMetrics();

public:
    int64_t                                   waitingStreamsSize();
    int64_t                                   runningStreamsSize();
    std::vector<EngineScheduleInfo::TaskInfo> waitingTaskList();
    std::vector<EngineScheduleInfo::TaskInfo> runningTaskList();
    int64_t                                   onflightStreams() override;

private:
    using StreamGroup      = std::list<GenerateStreamPtr>;
    using StreamGroupQueue = std::list<StreamGroup>;

    enum class AdmissionLane {
        NONE,
        NORMAL,
        GROUP,
    };

    struct ScheduleRuntime {
        size_t  admitted_running_stream_count             = 0;
        size_t  admitted_prefill_token_size_with_cache    = 0;
        size_t  admitted_prefill_max_seq_len_with_cache   = 0;
        size_t  admitted_prefill_sequence_count           = 0;
        size_t  admitted_prefill_token_size_without_cache = 0;
        size_t  newly_inited_kv_streams                   = 0;
    };

    int64_t lastScheduleTime() override;
    bool    fitsPrefillTokenLimits(size_t                   admitted_stream_count,
                                   size_t                   admitted_tokens,
                                   size_t                   admitted_max_seq_len,
                                   size_t                   admitted_sequence_count,
                                   const GenerateStreamPtr& candidate) const;
    bool    evaluateRunningBatch(const ScheduleRuntime& schedule_runtime, const GenerateStreamPtr& new_stream) const;
    bool   evaluateRunningBatch(const std::list<GenerateStreamPtr>& streams, const GenerateStreamPtr& new_stream) const;
    size_t  prefillTokenCostWithoutCache(const GenerateStreamPtr& stream) const;
    size_t  prefillSeqLenWithCache(const GenerateStreamPtr& stream) const;
    size_t  prefillTokenCostWithCache(const GenerateStreamPtr& stream) const;
    size_t  countInitedKVCacheStreams() const;
    size_t  groupQueueStreamsSize(const StreamGroupQueue& group_queue) const;
    void    accountBatchMetrics(const GenerateStreamPtr& new_stream);
    bool    waitPredicate();
    void    addStreamToNewState(const GenerateStreamPtr& stream, StreamState new_state);
    bool    checkInputLength(const GenerateStreamPtr& stream);
    void    evaluateWaitingStreams(std::list<GenerateStreamPtr>&       streams,
                                  const std::list<GenerateStreamPtr>& already_admitted_streams);
    void   evaluateWaitingGroupQueue();
    void   evaluateLoadingCacheGroupQueue();
    bool   loadingGroupReady() const;
    void   advanceLoadingGroup(StreamGroup& group);
    void   moveGroupToNewStreams(StreamGroup& group);
    void   moveGroupToAllocatingGroup(StreamGroup& group);
    void   dispatchPreparedGroup(StreamGroup& group);
    void    cancelStreams(std::list<GenerateStreamPtr>& streams);
    void    cancelGroups(StreamGroupQueue& group_queue);
protected:
    void                            evaluateAndUpdateStreams(std::list<GenerateStreamPtr>& streams);
    PDSepConfig                     pd_sep_config_;
    ModelSpecificConfig             model_specific_config_;
    std::list<GenerateStreamPtr>    waiting_streams_;
    std::list<GenerateStreamPtr>    loading_cache_streams_;
    std::list<GenerateStreamPtr>    running_streams_;
    std::list<GenerateStreamPtr>    new_streams_;
    StreamGroupQueue                waiting_group_queue_;
    StreamGroupQueue                loading_cache_group_queue_;
    std::shared_ptr<KVCacheManager> cache_manager_;
    std::atomic<int64_t>            last_schedule_time_          = autil::TimeUtility::currentTimeInMilliSeconds();
    size_t                          max_seq_len_                    = 0;
    size_t                          max_batch_tokens_size_          = 0;
    size_t                          max_batch_tokens_without_cache_ = 0;
    size_t                          max_generate_batch_size_        = 1;
    size_t                          max_inited_kv_cache_streams_    = 0;
    const bool                      need_fill_fake_stream_          = false;
    const size_t                    prefill_cp_size_                = 1;
    // Keep polling while collective sleep-quiesce is armed so drained ranks
    // continue issuing the synchronization co-steps.
    std::atomic<bool>               force_poll_                     = false;
    std::atomic<bool>               stop_                        = false;
    bool                            schedule_trigger_            = false;
    std::mutex                      lock_;
    std::condition_variable         cond_;
    kmonitor::MetricsReporterPtr    metrics_reporter_                 = nullptr;
    int64_t                         last_admitted_context_batch_size_ = 0;
    int64_t                         last_admitted_context_token_size_ = 0;
    int64_t                         last_waiting_oldest_age_us_       = 0;
    std::atomic<int64_t>            pending_group_fallback_count_     = 0;
    AdmissionLane                   active_admission_lane_             = AdmissionLane::NONE;
    bool                            prefer_group_next_                 = false;

    std::vector<EngineScheduleInfo::TaskInfo> waiting_task_list_;
    std::vector<EngineScheduleInfo::TaskInfo> running_task_list_;
};

}  // namespace rtp_llm
