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

    int64_t lastScheduleTime() override;
    bool   evaluateRunningBatch(const std::list<GenerateStreamPtr>& streams, const GenerateStreamPtr& new_stream) const;
    size_t countInitedKVCacheStreams() const;
    size_t groupQueueStreamsSize(const StreamGroupQueue& group_queue) const;
    void   accountBatchMetrics(const GenerateStreamPtr& new_stream);
    bool   waitPredicate();
    void   addStreamToNewState(const GenerateStreamPtr& stream, StreamState new_state);
    bool   checkInputLength(const GenerateStreamPtr& stream);
    void   evaluateWaitingStreams(std::list<GenerateStreamPtr>&       streams,
                                  const std::list<GenerateStreamPtr>& already_admitted_streams);
    void   evaluateWaitingGroupQueue();
    void   evaluateLoadingCacheGroupQueue();
    void   advanceLoadingGroup(StreamGroup& group);
    void   moveGroupToNewStreams(StreamGroup& group);
    void   moveGroupToAllocatingGroup(StreamGroup& group);
    void   dissolveGroup(StreamGroup& group);
    void   cancelStreams(std::list<GenerateStreamPtr>& streams);
    void   cancelGroups(StreamGroupQueue& group_queue);

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
    size_t                          max_seq_len_                 = 0;
    size_t                          max_batch_tokens_size_       = 0;
    size_t                          max_generate_batch_size_     = 1;
    size_t                          max_inited_kv_cache_streams_ = 0;
    const bool                      need_fill_fake_stream_       = false;
    const bool                      cp_force_single_prefill_     = false;
    std::atomic<bool>               stop_                        = false;
    std::mutex                      lock_;
    std::condition_variable         cond_;
    kmonitor::MetricsReporterPtr    metrics_reporter_                 = nullptr;
    int64_t                         last_admitted_context_batch_size_ = 0;
    int64_t                         last_admitted_context_token_size_ = 0;
    int64_t                         last_waiting_oldest_age_us_       = 0;

    std::vector<EngineScheduleInfo::TaskInfo> waiting_task_list_;
    std::vector<EngineScheduleInfo::TaskInfo> running_task_list_;
};

}  // namespace rtp_llm
