#pragma once

#include <atomic>
#include <condition_variable>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/schedulers/EngineScheduleInfo.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerBase.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

namespace rtp_llm {

class FIFOSchedulerBase: public SchedulerBase {
public:
    explicit FIFOSchedulerBase(const RuntimeConfig&                   runtime_config,
                               const ModelConfig&                     model_config,
                               const PDSepConfig&                     pd_sep_config,
                               const ParallelismConfig&               parallelism_config,
                               const ModelSpecificConfig&             model_specific_config,
                               const std::shared_ptr<KVCacheManager>& cache_manager,
                               const kmonitor::MetricsReporterPtr     metrics_reporter = nullptr);

    ~FIFOSchedulerBase() override = default;

    absl::Status                                 enqueue(const GenerateStreamPtr& stream) override;
    std::vector<std::shared_ptr<GenerateStream>> batchEnqueue(const std::vector<GenerateStreamPtr>& streams) override;
    absl::Status                                 stop() override;
    void                                         wake() override;
    bool                                         empty() override;
    int64_t                                      lastScheduleTime() override;
    int64_t                                      onflightStreams() override;

    std::vector<EngineScheduleInfo::TaskInfo> waitingTaskList() override;
    std::vector<EngineScheduleInfo::TaskInfo> runningTaskList() override;

    void reportMetrics();

    // for test
    int64_t waitingStreamsSize();
    int64_t runningStreamsSize();

protected:
    virtual const char* schedulerName() const                                            = 0;
    virtual bool        evaluateRunningMemory(const std::list<GenerateStreamPtr>& streams,
                                              const GenerateStreamPtr&            new_stream) = 0;
    virtual bool        waitPredicate()                                                  = 0;
    virtual void        onRunningStream(const GenerateStreamPtr& stream) {}
    virtual void        cancelExtraStreams() {}
    virtual bool        hasExtraStreams() const {
        return false;
    }
    virtual int64_t extraOnflightStreams() const {
        return 0;
    }
    virtual void fillExtraMetrics(RtpLLMSchedulerMetricsCollector& collector) const {}

    bool   checkInputLength(const GenerateStreamPtr& stream);
    void   cancelStreams(std::list<GenerateStreamPtr>& streams);
    size_t evaluateAndUpdateStreams(std::list<GenerateStreamPtr>& streams);
    void   evaluateWaitingStreams(std::list<GenerateStreamPtr>& waiting_streams);
    void   addStreamToNewState(const GenerateStreamPtr& stream, StreamState new_state);

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
    bool                            need_fill_fake_stream_   = false;
    std::atomic<bool>               stop_                    = false;
    bool                            schedule_trigger_        = false;
    std::mutex                      lock_;
    std::condition_variable         cond_;
    kmonitor::MetricsReporterPtr    metrics_reporter_ = nullptr;

    std::vector<EngineScheduleInfo::TaskInfo> waiting_task_list_;
    std::vector<EngineScheduleInfo::TaskInfo> running_task_list_;
};

}  // namespace rtp_llm
