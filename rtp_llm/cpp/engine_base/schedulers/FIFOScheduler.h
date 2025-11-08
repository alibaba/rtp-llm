#pragma once

#include <queue>
#include <tuple>
#include <vector>
#include <atomic>
#include "rtp_llm/cpp/cache_new/KVCacheManager.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerBase.h"
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "rtp_llm/cpp/engine_base/schedulers/EngineScheduleInfo.h"
namespace rtp_llm {

class FIFOScheduler: public SchedulerBase {
public:
    explicit FIFOScheduler(const rtp_llm::GptInitParameter&        params,
                           const std::shared_ptr<KVCacheManager>& cache_manager,
                           const kmonitor::MetricsReporterPtr      metrics_reporter = nullptr,
                           const int                               max_score_len    = 1);

    ~FIFOScheduler() override;

    absl::Status                                 enqueue(const GenerateStreamPtr& stream) override;
    absl::Status                                 batchEnqueue(const std::vector<GenerateStreamPtr>& streams) override;
    absl::StatusOr<std::list<GenerateStreamPtr>> schedule(size_t reserve_step = 0) override;
    absl::Status                                 stop() override;
    bool                                         empty() override;

    void reportMetrics(size_t fallback_stream_size);

public:
    // for test
    int64_t                                   waitingStreamsSize();
    int64_t                                   runningStreamsSize();
    std::vector<EngineScheduleInfo::TaskInfo> waitingTaskList();
    std::vector<EngineScheduleInfo::TaskInfo> runningTaskList();
    int64_t                                   onflightStreams() override;

protected:
    virtual std::list<GenerateStreamPtr> scheduleNew(size_t reserve_step);

private:
    void                 evictDoneStreams(std::list<GenerateStreamPtr>& streams);
    bool                 evaluateNewStream(const std::list<GenerateStreamPtr>& streams,
                                           const GenerateStreamPtr&            new_stream,
                                           size_t                              reserve_step);
    std::tuple<int, int> evaluateRunningNext(size_t reserve_step);
    void                 evaluateRunningRemote();
    int64_t              lastScheduleTime() override;
    int                  runningNextBlockNum(size_t reserve_step) const;
    bool evaluateRunningMemory(const std::list<GenerateStreamPtr>& streams, const GenerateStreamPtr& new_stream) const;
    void accountBatchMetrics(const std::list<GenerateStreamPtr>& new_streams,
                             const std::list<GenerateStreamPtr>& running_streams);    
    bool                         waitPredicate();

protected:
    rtp_llm::GptInitParameter        params_;
    std::list<GenerateStreamPtr>     waiting_streams_;
    std::list<GenerateStreamPtr>     running_streams_;
    std::list<GenerateStreamPtr>     remote_running_streams_;
    std::shared_ptr<KVCacheManager>  cache_manager_;
    std::atomic<int64_t>          last_schedule_time_       = autil::TimeUtility::currentTimeInMilliSeconds();
    size_t                        max_seq_len_              = 0;
    size_t                        max_batch_tokens_size_    = 0;
    size_t                        max_generate_batch_size_  = 1;
    int                           reserve_block_num_        = 0;
    bool                          enable_partial_fallback_  = false;
    bool                          enable_whole_fallback_    = true;
    bool                          enable_fast_gen_          = false;
    const bool                    need_fill_fake_stream_    = false;
    int                           fast_gen_max_context_len_ = 0;
    int                           token_capacity_           = 0;
    std::atomic<bool>             stop_                     = false;
    std::mutex                    lock_;
    std::condition_variable       cond_;
    kmonitor::MetricsReporterPtr  metrics_reporter_ = nullptr;

    std::vector<EngineScheduleInfo::TaskInfo> waiting_task_list_;
    std::vector<EngineScheduleInfo::TaskInfo> running_task_list_;

    // TODO @wangyin support different beams run togather
};

}  // namespace rtp_llm
