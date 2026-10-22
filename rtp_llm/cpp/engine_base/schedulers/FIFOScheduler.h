#pragma once

#include <atomic>
#include <cstddef>
#include <list>

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOSchedulerBase.h"
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/schedulers/EngineScheduleInfo.h"
namespace rtp_llm {

class FIFOScheduler: public FIFOSchedulerBase {
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

    // Group-aware enqueue: streams enter the dedicated group queues and are
    // co-scheduled at an isolated execution boundary. Falls back to individual
    // FIFO admission when the group exceeds scheduler-wide limits.
    std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
    enqueueGroup(const std::vector<GenerateStreamPtr>& streams) override;

    absl::StatusOr<std::list<GenerateStreamPtr>> schedule() override;

public:
    // for test. Group-aware shadow of the FIFOSchedulerBase helper so that
    // streams parked in the group queues stay observable.
    int64_t waitingStreamsSize();
    using FIFOSchedulerBase::runningStreamsSize;
    std::vector<EngineScheduleInfo::TaskInfo> waitingTaskList() override;

private:
    using StreamGroup      = std::list<GenerateStreamPtr>;
    using StreamGroupQueue = std::list<StreamGroup>;

    // Which lane owns the currently running execution batch. Group batches are
    // isolated: ordinary streams must not join a group's boundary and vice versa.
    enum class AdmissionLane {
        NONE,
        NORMAL,
        GROUP,
    };

    // Per-round admission accounting for admitWaitingStreams(). Only streams that actually
    // made scheduling progress contribute, so a stream whose KV malloc failed and stayed
    // WAITING does not consume this round's token budget.
    struct ScheduleRuntime {
        size_t admitted_stream_count                     = 0;
        size_t admitted_prefill_token_size_with_cache    = 0;
        size_t admitted_prefill_max_seq_len_with_cache   = 0;
        size_t admitted_prefill_sequence_count           = 0;
        size_t admitted_prefill_token_size_without_cache = 0;
        size_t newly_inited_kv_streams                   = 0;
    };

    const char* schedulerName() const override {
        return "FIFOScheduler";
    }
    bool evaluateRunningMemory(const std::list<GenerateStreamPtr>& streams,
                               const GenerateStreamPtr&            new_stream) override;
    // Counter-based admission check used by admitWaitingStreams(). evaluateRunningMemory()
    // is the list-based FIFOSchedulerBase entry point and delegates here.
    bool   evaluateRunningBatch(const ScheduleRuntime& schedule_runtime, const GenerateStreamPtr& new_stream) const;
    // Overload for group-queue admission: the admitted streams are tracked as a list
    // because ScheduleRuntime is not built in the group-queue path.
    bool   evaluateRunningBatch(const std::list<GenerateStreamPtr>& streams, const GenerateStreamPtr& new_stream) const;
    bool   fitsPrefillTokenLimits(size_t                   admitted_stream_count,
                                  size_t                   admitted_tokens,
                                  size_t                   admitted_max_seq_len,
                                  size_t                   admitted_sequence_count,
                                  const GenerateStreamPtr& candidate) const;
    size_t prefillTokenCostWithoutCache(const GenerateStreamPtr& stream) const;
    size_t prefillSeqLenWithCache(const GenerateStreamPtr& stream) const;
    size_t prefillTokenCostWithCache(const GenerateStreamPtr& stream) const;
    // Group-aware shadow of FIFOSchedulerBase::countInitedKVCacheStreams(): streams parked
    // in the group queues also hold inited KV blocks and must count against the limit.
    size_t countInitedKVCacheStreams() const;
    size_t groupQueueStreamsSize(const StreamGroupQueue& group_queue) const;
    void   accountBatchMetrics(const GenerateStreamPtr& new_stream);
    bool   waitPredicate() override;
    void   onRunningStream(const GenerateStreamPtr& stream) override;
    // FIFO-specific replacement for FIFOSchedulerBase::evaluateWaitingStreams(): admission and
    // state transition happen in a single pass so that per-round token budgets only account for
    // streams that really advanced, and errored streams behind a saturated budget are still
    // finalized. FIFOSchedulerBase::evaluateWaitingStreams() is left untouched for the other
    // FIFOSchedulerBase subclasses.
    void admitWaitingStreams(std::list<GenerateStreamPtr>&       waiting_streams,
                             const std::list<GenerateStreamPtr>& already_admitted_streams);

    void cancelGroups(StreamGroupQueue& group_queue);
    void evaluateWaitingGroupQueue();
    void evaluateLoadingCacheGroupQueue();
    bool loadingGroupReady() const;
    void advanceLoadingGroup(StreamGroup& group);
    void moveGroupToNewStreams(StreamGroup& group);
    void moveGroupToAllocatingGroup(StreamGroup& group);
    void dispatchPreparedGroup(StreamGroup& group);

    void    cancelExtraStreams() override;
    bool    hasExtraStreams() const override;
    int64_t extraOnflightStreams() const override;
    void    fillExtraMetrics(RtpLLMSchedulerMetricsCollector& collector) const override;

    // Explicit request groups (enqueueGroup). Each group is admitted as a whole
    // to an isolated execution boundary; a partially admitted group keeps its
    // residual members at the queue head.
    StreamGroupQueue waiting_group_queue_;
    StreamGroupQueue loading_cache_group_queue_;

    // Context-parallel prefill can opt into single-request admission until
    // the model-side path supports per-request layouts.
    const bool cp_force_single_prefill_ = false;
    // Soft per-round quota on the tokens that are actually recomputed (prefix-cache hits
    // excluded). 0 disables it.
    const size_t max_batch_tokens_without_cache_ = 0;
    const size_t prefill_cp_size_                = 1;

    // Consumed (exchanged to 0) from the const fillExtraMetrics() reporting hook.
    mutable std::atomic<int64_t> pending_group_fallback_count_ = 0;
    AdmissionLane                active_admission_lane_        = AdmissionLane::NONE;
    bool                         prefer_group_next_            = false;

    // TODO @wangyin support different beams run togather
};

}  // namespace rtp_llm
