#pragma once

#include <queue>
#include <tuple>
#include "rtp_llm/cpp/cache/CacheManager.h"
#include "rtp_llm/cpp/dataclass/Query.h"
#include "rtp_llm/cpp/schedulers/SchedulerBase.h"
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/th_op/GptInitParameter.h"

namespace rtp_llm {

class FIFOScheduler: public SchedulerBase {
public:
    explicit FIFOScheduler(const rtp_llm::GptInitParameter&          params,
                           const std::shared_ptr<CacheManager>& cache_manager,
                           const kmonitor::MetricsReporterPtr   metrics_reporter = nullptr);

    ~FIFOScheduler() override;

    absl::Status                                 enqueue(const GenerateStreamPtr& stream) override;
    absl::StatusOr<std::list<GenerateStreamPtr>> schedule(size_t reserve_step = 0) override;
    absl::Status                                 stop() override;
    bool                                         empty() override;

    void reportMetrics(size_t fallback_stream_size);

public:
    // for test
    int64_t waitingStreamsSize();
    int64_t runningStreamsSize();
    int64_t onflightStreams() override;

private:
    void evictDoneStreams(std::list<GenerateStreamPtr>& streams) const;
    bool evaluateNewStream(const std::list<GenerateStreamPtr>& streams,
                            const GenerateStreamPtr& new_stream, size_t reserve_step);
    std::tuple<int, int> evaluateRunningNext(size_t reserve_step);
    void evaluateRunningRemote();
    int64_t lastScheduleTime() override;
    int  runningNextBlockNum(size_t reserve_step) const;
    bool evaluateRunningMemory(const std::list<GenerateStreamPtr>& streams, const GenerateStreamPtr& new_stream) const;
    void accountBatchMetrics(const std::list<GenerateStreamPtr>& new_streams, const std::list<GenerateStreamPtr>& running_streams);
    std::list<GenerateStreamPtr> scheduleNew(size_t reserve_step);
    bool waitPredicate();

private:
    rtp_llm::GptInitParameter     params_;
    std::list<GenerateStreamPtr>  waiting_streams_;
    std::list<GenerateStreamPtr>  running_streams_;
    std::list<GenerateStreamPtr>  remote_running_streams_;
    std::shared_ptr<CacheManager> cache_manager_;
    std::atomic<int64_t>          last_schedule_time_       = autil::TimeUtility::currentTimeInMilliSeconds();
    size_t                        max_seq_len_              = 0;
    size_t                        max_context_batch_size_   = 1;
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

    // TODO @wangyin support different beams run togather
};

}  // namespace rtp_llm
