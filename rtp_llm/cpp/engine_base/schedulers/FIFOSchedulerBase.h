#pragma once

#include <atomic>
#include <condition_variable>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <tuple>

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/schedulers/EngineScheduleInfo.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerBase.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerUtils.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

namespace rtp_llm {

int64_t computeChunkGrant(int64_t budget, int64_t rows, int64_t remaining, int64_t alignment, int64_t prefix = 0);

class FIFOSchedulerBase: public SchedulerBase {
public:
    explicit FIFOSchedulerBase(const RuntimeConfig&                   runtime_config,
                               const ModelConfig&                     model_config,
                               const PDSepConfig&                     pd_sep_config,
                               const ParallelismConfig&               parallelism_config,
                               const std::shared_ptr<KVCacheManager>& cache_manager,
                               const kmonitor::MetricsReporterPtr     metrics_reporter = nullptr,
                               const ModelExecutionCapacity&          execution_capacity = ModelExecutionCapacity{});

    ~FIFOSchedulerBase() override = default;

    absl::Status enqueue(const GenerateStreamPtr& stream) override;
    // Default group enqueue: validates each stream and admits the valid ones as
    // ordinary individual streams (no co-scheduling). FIFOScheduler overrides this
    // with real group-queue semantics.
    std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
                 enqueueGroup(const std::vector<GenerateStreamPtr>& streams) override;
    absl::Status requestStop() override;
    absl::Status stop() override;
    bool         empty() override;
    int64_t      lastScheduleTime() override;
    int64_t      onflightStreams() override;

    std::vector<EngineScheduleInfo::TaskInfo> waitingTaskList() override;
    std::vector<EngineScheduleInfo::TaskInfo> runningTaskList() override;

    void reportMetrics();

    // for test
    int64_t waitingStreamsSize();
    int64_t runningStreamsSize();

protected:
    struct ScheduleRuntime {
        size_t  admitted_stream_count                     = 0;
        size_t  admitted_prefill_token_size_with_cache    = 0;
        size_t  admitted_prefill_max_seq_len_with_cache   = 0;
        size_t  admitted_prefill_sequence_count           = 0;
        size_t  admitted_prefill_token_size_without_cache = 0;
        size_t  newly_inited_kv_streams                   = 0;
        int64_t batch_size = 0, prefill_tokens = 0, draft_tokens = 0;
        size_t  prefill_streams = 0;
        // Retain charged rows so cancellation and cache hits release the original cost.
        std::unordered_map<const GenerateStream*, std::tuple<int64_t, int64_t, int64_t, bool>> counted_streams;
        std::list<GenerateStreamPtr>                                                           admitted_streams;
    };

    virtual const char* schedulerName() const                                            = 0;
    virtual bool        evaluateRunningMemory(const ScheduleRuntime& runtime, const GenerateStreamPtr& new_stream);
    virtual bool        waitPredicate()                                                  = 0;
    virtual void        onRunningStream(const GenerateStreamPtr& stream) {}
    virtual void        cancelExtraStreams() {}
    virtual int64_t extraOnflightStreams() const {
        return 0;
    }
    virtual void fillExtraMetrics(RtpLLMSchedulerMetricsCollector& collector) const {}
    virtual void appendExtraRunningTaskList(std::vector<EngineScheduleInfo::TaskInfo>& task_list) const {}

    int64_t admissionBatchSize(const GenerateStreamPtr& stream) const {
        // Legacy quotas count requests; fixed workspaces need expanded sequence bounds.
        return max_execution_tokens_ > 0 || max_prefill_draft_tokens_ > 0 ? stream->maxBatchSize() : 1;
    }
    bool   checkInputLength(const GenerateStreamPtr& stream);
    bool                                checkPrefillCapacity(const GenerateStreamPtr& stream);
    absl::Status checkChunkedPrefillRequest(const GenerateStreamPtr& stream);
    std::list<GenerateStreamPtr> selectPrefillPrefix(std::list<GenerateStreamPtr>& active_streams);
    virtual void cancelStreams(std::list<GenerateStreamPtr>& streams);
    virtual bool refreshAndReapTerminalStreams(std::list<GenerateStreamPtr>& streams);
    size_t                              evaluateAndUpdateStreams(std::list<GenerateStreamPtr>&              streams,
                                                                 const std::unordered_set<GenerateStream*>* admitted_streams = nullptr);
    std::unordered_set<GenerateStream*> evaluateWaitingStreams(std::list<GenerateStreamPtr>& waiting_streams,
                                                               bool include_running_mode = true);
    virtual void   addStreamToNewState(const GenerateStreamPtr& stream, StreamState new_state);
    size_t countInitedKVCacheStreams() const;
    virtual void                        appendExtraExecutionReservations(std::vector<GenerateStreamPtr>&) const {}
    ScheduleRuntime                     initScheduleRuntime(const std::list<GenerateStreamPtr>& admitted = {}) const;
    bool accountExecutionRows(ScheduleRuntime& runtime, const GenerateStreamPtr& stream, bool add = true) const;
    void accountAdmission(ScheduleRuntime& runtime, const GenerateStreamPtr& stream) const;
    void   appendTaskInfos(std::vector<EngineScheduleInfo::TaskInfo>& task_list,
                           const std::list<GenerateStreamPtr>&         streams) const;

protected:
    PDSepConfig                     pd_sep_config_;
    std::list<GenerateStreamPtr>    waiting_streams_;
    std::list<GenerateStreamPtr>    loading_cache_streams_;
    std::list<GenerateStreamPtr>    running_streams_;
    std::list<GenerateStreamPtr>    new_streams_;
    std::shared_ptr<KVCacheManager> cache_manager_;
    std::atomic<int64_t>            last_schedule_time_      = autil::TimeUtility::currentTimeInMilliSeconds();
    size_t                          max_seq_len_             = 0;
    size_t                          max_batch_tokens_size_   = 0;
    size_t                          max_generate_batch_size_ = 1;
    int64_t                         max_execution_tokens_        = 0;
    int64_t                         max_prefill_draft_tokens_    = 0;
    size_t                          draft_prefill_cp_size_       = 1;
    bool                            target_only_runs_draft_      = false;
    size_t                          max_inited_kv_cache_streams_ = 0;
    int64_t                         prefill_chunk_size_      = 0;
    int64_t                         prefill_token_alignment_     = 1;
    int64_t                         prefill_chunk_alignment_     = 0;
    int64_t                         prefill_chunk_cp_size_       = 0;
    int64_t                         prefill_chunk_padded_rows_   = 0;
    bool                            need_fill_fake_stream_   = false;
    std::atomic<bool>               stop_                    = false;
    bool                            schedule_trigger_        = false;
    std::mutex                      lock_;
    std::condition_variable         cond_;
    kmonitor::MetricsReporterPtr    metrics_reporter_ = nullptr;
    int64_t                         last_admitted_context_batch_size_ = 0;
    int64_t                         last_admitted_context_token_size_ = 0;
    int64_t                         last_waiting_oldest_age_us_       = 0;

    std::vector<EngineScheduleInfo::TaskInfo> waiting_task_list_;
    std::vector<EngineScheduleInfo::TaskInfo> running_task_list_;
};

}  // namespace rtp_llm
