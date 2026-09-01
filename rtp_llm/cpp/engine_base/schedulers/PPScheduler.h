#pragma once

#include <cstdint>
#include <list>
#include <vector>

#include "rtp_llm/cpp/engine_base/schedulers/FIFOSchedulerBase.h"

namespace rtp_llm {

class PPScheduler: public FIFOSchedulerBase {
public:
    PPScheduler(const RuntimeConfig&                   runtime_config,
                const ModelConfig&                     model_config,
                const PDSepConfig&                     pd_sep_config,
                const ParallelismConfig&               parallelism_config,
                const ModelSpecificConfig&             model_specific_config,
                const std::shared_ptr<KVCacheManager>& cache_manager,
                const kmonitor::MetricsReporterPtr     metrics_reporter = nullptr);

    ~PPScheduler() override;

    absl::StatusOr<ScheduleOutput> schedule() override;

private:
    /** Per-round scheduling and admission accounting. */
    struct ScheduleRuntime {
        size_t inited_kv_stream_count                    = 0;
        size_t scheduled_stream_count                    = 0;
        size_t scheduled_prefill_token_size_with_cache   = 0;
        size_t scheduled_prefill_max_seq_len_with_cache  = 0;
        size_t scheduled_prefill_sequence_count          = 0;
        size_t admitted_prefill_token_size_without_cache = 0;
    };

    const char* schedulerName() const override {
        return "PPScheduler";
    }

    bool evaluateRunningMemory(const std::list<GenerateStreamPtr>& streams,
                               const GenerateStreamPtr&            new_stream) override;

    bool waitPredicate() override;

    void addStreamToNewState(const GenerateStreamPtr& stream, StreamState new_state) override;

    std::list<GenerateStreamPtr> evaluateRunningStreams();

    void admitWaitingStreams(std::list<GenerateStreamPtr>& scheduled_streams);

    void initScheduleRuntime(const std::list<GenerateStreamPtr>& scheduled_streams,
                             ScheduleRuntime&                    schedule_runtime) const;

    void updateScheduleRuntime(ScheduleRuntime&         schedule_runtime,
                               const GenerateStreamPtr& stream,
                               bool                     new_inited_kv = false) const;

    bool fitsCurrentBatch(const ScheduleRuntime& schedule_runtime, const GenerateStreamPtr& candidate) const;

    bool fitsPrefillTokenLimits(const ScheduleRuntime& schedule_runtime, const GenerateStreamPtr& candidate) const;

    size_t prefillSeqLenWithCache(const GenerateStreamPtr& stream) const;

    size_t prefillTokenCostWithCache(const GenerateStreamPtr& stream) const;

    size_t prefillTokenCostWithoutCache(const GenerateStreamPtr& stream) const;

    const size_t         max_batch_tokens_without_cache_ = 0;
    std::vector<int64_t> finished_request_ids_;
};

}  // namespace rtp_llm
