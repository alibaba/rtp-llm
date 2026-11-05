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
                const SpeculativeExecutionConfig&      sp_config,
                const std::shared_ptr<KVCacheManager>& cache_manager,
                const kmonitor::MetricsReporterPtr     metrics_reporter = nullptr);

    ~PPScheduler() override;

    absl::StatusOr<ScheduleOutput> schedule() override;

    void finalizeCancelledStreams() override;

private:
    /** Per-round scheduling and admission accounting. */
    struct ScheduleRuntime {
        size_t concurrency_stream_count = 0;
        size_t inited_kv_stream_count   = 0;
        size_t scheduled_stream_count                    = 0;
        size_t scheduled_prefill_token_size_with_cache   = 0;
        size_t scheduled_prefill_max_seq_len_with_cache  = 0;
        size_t scheduled_prefill_sequence_count          = 0;
        size_t admitted_prefill_token_size_without_cache = 0;
        bool   has_execution_mode                        = false;
        bool   prefill_only                              = false;
    };

    const char* schedulerName() const override {
        return "PPScheduler";
    }

    bool evaluateRunningMemory(const std::list<GenerateStreamPtr>& streams,
                               const GenerateStreamPtr&            new_stream) override;

    bool waitPredicate() override;

    void onRunningStream(const GenerateStreamPtr& stream) override;

    void initializeDecodeCandidates(const GenerateStreamPtr& stream) const;

    void addStreamToNewState(const GenerateStreamPtr& stream, StreamState new_state) override;

    void cancelStreams(std::list<GenerateStreamPtr>& streams) override;

    bool refreshAndReapTerminalStreams(std::list<GenerateStreamPtr>& streams) override;

    bool chunkedPrefillConfigured() const {
        return prefill_chunk_size_ > 0;
    }

    std::list<GenerateStreamPtr> evaluateRunningStreams();
    bool                         canSchedule(const GenerateStreamPtr& stream) const;
    bool                         grantChunk(const GenerateStreamPtr& stream, int64_t& budget);
    void                         moveMiddleChunksToBack(std::list<GenerateStreamPtr>& streams) const;
    size_t                       countInitedKVCacheStreams() const;
    void                         cancelExtraStreams() override;
    int64_t                      extraOnflightStreams() const override;
    void                         fillExtraMetrics(RtpLLMSchedulerMetricsCollector& collector) const override;
    void appendExtraRunningTaskList(std::vector<EngineScheduleInfo::TaskInfo>& task_list) const override;

    void resetAdmissionMetrics();

    void maintainWaitingStreams();

    void admitWaitingStreams(std::list<GenerateStreamPtr>& scheduled_streams);

    void initScheduleRuntime(const std::list<GenerateStreamPtr>& scheduled_streams,
                             ScheduleRuntime&                    schedule_runtime) const;

    void updateScheduleRuntime(ScheduleRuntime&         schedule_runtime,
                               const GenerateStreamPtr& stream,
                               bool                     new_admission,
                               bool                     new_inited_kv) const;

    bool fitsCurrentBatch(const ScheduleRuntime& schedule_runtime, const GenerateStreamPtr& candidate) const;

    bool fitsPrefillTokenLimits(const ScheduleRuntime& schedule_runtime, const GenerateStreamPtr& candidate) const;

    size_t prefillSeqLenWithCache(const GenerateStreamPtr& stream) const;

    size_t prefillTokenCostWithCache(const GenerateStreamPtr& stream) const;

    size_t prefillTokenCostWithoutCache(const GenerateStreamPtr& stream) const;

    const size_t                     max_batch_tokens_without_cache_ = 0;
    const bool                       cp_force_single_prefill_;
    const size_t                     prefill_cp_size_;
    const SpeculativeExecutionConfig sp_config_;
    std::vector<int64_t>             finished_request_ids_;
    std::list<GenerateStreamPtr>     pending_decode_streams_;
};

}  // namespace rtp_llm
