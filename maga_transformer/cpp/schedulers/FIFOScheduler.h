#pragma once

#include <queue>
#include "maga_transformer/cpp/cache/CacheManager.h"
#include "maga_transformer/cpp/dataclass/Query.h"
#include "maga_transformer/cpp/schedulers/SchedulerBase.h"
#include "kmonitor/client/MetricsReporter.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"

namespace rtp_llm {

class FIFOScheduler: public SchedulerBase {
public:
    explicit FIFOScheduler(const ft::GptInitParameter&          params,
                           const std::shared_ptr<CacheManager>& cache_manager,
                           const kmonitor::MetricsReporterPtr   metrics_reporter = nullptr);

    ~FIFOScheduler();

    absl::Status enqueue(const GenerateStreamPtr& stream) override;
    absl::StatusOr<std::list<GenerateStreamPtr>> schedule() override;
    absl::Status stop() override;

    void reportMetrics(size_t fallback_stream_size);

public:
    // for test
    int waitingStreamsSize();
    int runningStreamsSize();

private:
    void evictDoneStreams(std::list<GenerateStreamPtr>& streams) const;
    bool evaluateNewStream(const std::list<GenerateStreamPtr>& streams, const GenerateStreamPtr& new_stream);
    void evaluateRunningNext();
    int  runningNextBlockNum() const;
    bool evaluateRunningMemory(const std::list<GenerateStreamPtr>& streams, const GenerateStreamPtr& new_stream) const;
    std::list<GenerateStreamPtr> scheduleNew();

private:
    std::list<GenerateStreamPtr>        waiting_streams_;
    std::list<GenerateStreamPtr>        running_streams_;
    std::shared_ptr<CacheManager>       cache_manager_;
    size_t                              max_seq_len_        = 0;
    size_t                              max_context_batch_size_ = 1;
    int                                 reserve_block_num_  = 0;
    // TODO(xinfei.sxf) add smoke for all fallback
    bool                                enable_partial_fallback_ = false;
    bool                                enable_fast_gen_    = false;
    int                                 max_context_len_    = 0;
    int                                 token_capacity_     = 0;
    std::atomic<bool>                   stop_               = false;
    std::mutex                          lock_;
    std::condition_variable             cond_;

    kmonitor::MetricsReporterPtr        metrics_reporter_ = nullptr;
    // TODO @wangyin support different beams run togather
};

}  // namespace rtp_llm
