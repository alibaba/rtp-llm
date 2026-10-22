#pragma once

#include <array>
#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "kmonitor/client/MetricsReporter.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/RankLayout.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/engine_base/Executor.h"
#include "rtp_llm/cpp/engine_base/stream/SamplingState.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/model_utils/MlaConfig.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPBatchStreamProcessor.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPTransport.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPTypes.h"
#include "rtp_llm/cpp/normal_engine/speculative/MtpCompute.h"
#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"
#include "rtp_llm/models_py/bindings/core/TensorHolder.h"

namespace rtp_llm {

struct EngineInitParams;
struct GptModelInitParams;
struct ProposeModelEngineInitParams;
class KVCacheManager;
class ModelBase;
class Sampler;
class ExpertBalancer;
class ModelInputsLogger;
class SpecLogitsVerifyRunner;

using PPTickets = std::vector<std::unique_ptr<PPCommTicket>>;

/** Tail-stage request state retained and retired by PPExecutor across execution plans. */
struct PPSamplingState {
    SamplingState sampling;
    /** Committed history received from the first stage, one vector per return sequence. */
    std::vector<std::vector<int32_t>> token_history;
};
using SamplingStates = std::unordered_map<RequestId, PPSamplingState>;

class PPExecutor final: public Executor {
public:
    PPExecutor(const EngineInitParams&                params,
               const std::shared_ptr<KVCacheManager>& cache_manager,
               bool                                   warm_up             = false,
               MlaOpsType                             mla_ops_type        = MlaOpsType::AUTO,
               std::function<void()>                  profile_step_start  = nullptr,
               std::function<void()>                  profile_step_finish = nullptr,
               ProposeModelEngineInitParams*          propose_params      = nullptr);

    ~PPExecutor() override;

    absl::Status process(const ScheduleOutput& schedule_output, int64_t schedule_time_us = 0) override;

    /** First-stage root receives pending results in submission order without sending plans or advancing the ring. */
    absl::Status drainPendingResults();

    absl::Status finish() override;

    bool updateEplbConfig(const EPLBConfig& config) override;

    static GenerateStreamPtr createMinFakePrefillStream(const ModelConfig&                model_config,
                                                        const RuntimeConfig&              runtime_config,
                                                        const ResourceContext&            resource_context,
                                                        const SpeculativeExecutionConfig& sp_config,
                                                        RoleType                          role_type);

    static GenerateStreamPtr createMinFakeDecodeStream(const ModelConfig&                model_config,
                                                       const RuntimeConfig&              runtime_config,
                                                       const ResourceContext&            resource_context,
                                                       const SpeculativeExecutionConfig& sp_config);

    void setBatchProcessor(std::unique_ptr<PPBatchStreamProcessor> processor) {
        batch_stream_processor_ = std::move(processor);
    }

    void setModel(std::unique_ptr<ModelBase> model) {
        model_ = std::move(model);
    }

    using ModelFactory = std::function<std::unique_ptr<ModelBase>(const GptModelInitParams&)>;
    static ModelFactory test_model_factory;

private:
    static void validateConfig(const EngineInitParams& params);

    int64_t firstLocalMoeLayer(const ModelConfig& model_config) const;

    struct InflightBatch {
        bool                           result_pending = false;
        StreamGroups                   stream_groups;
        int64_t                        schedule_time_us = 0;
        RtpLLMExecutorMetricsCollector executor_collector;

        /** Keep both reporters active across process calls until this batch's result is handled. */
        MetricsLoopReporter<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector>::ActiveGuard tps_active_guard;
        WallClockMetricsLoopReporter<RtpLLMWallClockTokenPSMetrics, RtpLLMTokenPSMetricsCollector>::ActiveGuard
            wall_tps_active_guard;

        PPTickets plan_sends;
        PPTickets activation_sends;
        PPTickets execution_result_sends;

        void reset();
    };

    absl::Status warmUp(const ScheduleOutput& schedule_output);

    absl::Status processImpl(const ScheduleOutput& schedule_output, int64_t schedule_time_us, bool shutdown);

    void waitLastRun();

    void releaseAllModelBuffers();

    std::optional<ErrorInfo> validateStream(const GenerateStream& stream) const;

    void prepareStreams(std::list<GenerateStreamPtr>& streams);

    absl::StatusOr<PPExecutionPlan>
    buildPlan(const StreamGroups& stream_groups, const std::vector<RequestId>& finished_request_ids, bool shutdown);

    void sampleTokens(const PPExecutionPlan&  plan,
                      const GptModelOutputs&  model_output,
                      PPExecutionResult&      result,
                      GrammarPrefetchContext* grammar_prefetch = nullptr);

    void finishDSparkPrefillCachePublication(const GptModelInputs& model_input, PPExecutionResult& execution_result);

    void advanceSamplingStates(const PPSamplingPlan& sampling_plan, PPExecutionResult& result);

    void eraseFailedDSparkQ(const PPExecutionResult& result);

    absl::Status processExecutionResult(InflightBatch& batch);

    void verifyDraftTokens(const PPExecutionPlan&                        plan,
                           const torch::Tensor&                          target_logits,
                           SamplerOutput&                                draft_sampler_output,
                           const speculative::SpeculativeSamplingParams& verify_params,
                           SamplerOutput&                                target_sampler_output,
                           speculative::SpeculativeSamplerOutput&        accepted);

    void broadcastPostRejectionInputs(GptModelInputs& draft_input);

    GptModelOutputs forwardDraftModel(ModelBase& draft_model, GptModelInputs& draft_input);

    void runDraftStep(GptModelInputs&       target_input,
                      const torch::Tensor&  draft_next_position_ids,
                      const torch::Tensor&  target_hidden_states,
                      bool                  is_decode,
                      const PPSamplingPlan& sampling_plan,
                      PPExecutionResult&    execution_result);

    void fillFailedDraftRows(PPExecutionResult& result);

    void asyncSendPlan(const torch::Tensor& plan_payload, PPTickets& tickets);

    PPExecutionPlan receivePlan(torch::Tensor& plan_payload);

    void asyncSendTensors(const PPIntermediateTensors& tensors, PPTickets& tickets);

    PPIntermediateTensors receiveTensors();

    void asyncSendExecutionResult(const PPExecutionResult& result, PPTickets& tickets);

    void sendObject(const torch::Tensor& object, PPTickets& tickets);

    torch::Tensor receiveObject();

    void waitAll(PPTickets& tickets);

    void reapCompletedActivationSends();

    bool isFirstStage() const {
        return pp_layout_.isFirstStage();
    }

    bool isLastStage() const {
        return pp_layout_.isLastStage();
    }

    bool isStageRoot() const {
        return parallelism_config_.tp_rank == 0;
    }

    void collectExecutorMetrics(const GptModelInputs&           model_input,
                                const torch::Tensor&            sequence_lengths,
                                RtpLLMExecutorMetricsCollector& collector) const;

    void collectTokenCounts(const StreamGroups&                      stream_groups,
                            const PPExecutionResult&                 result,
                            StreamGroups::TokenCountsByPriority&     token_counts_by_priority,
                            RtpLLMSpeculativeEngineMetricsCollector& sp_collector) const;

    void reportResultMetrics(InflightBatch&                             batch,
                             const StreamGroups::TokenCountsByPriority& token_counts_by_priority,
                             RtpLLMSpeculativeEngineMetricsCollector&   sp_collector);

    /* Report this stage's local executor timings tagged with pp_rank. Called by non-first
       stages at round end; the first stage reports through reportResultMetrics. */
    void reportStageMetrics(RtpLLMExecutorMetricsCollector& collector);

private:
    const bool                              warm_up_;
    const RoleType                          role_type_;
    std::shared_ptr<KVCacheManager>         cache_manager_;
    std::unique_ptr<ModelBase>              model_;
    std::unique_ptr<Sampler>                sampler_;
    std::unique_ptr<PPBatchStreamProcessor> batch_stream_processor_;
    std::shared_ptr<ExpertBalancer>         expert_balancer_;
    // Holds executor-owned CPU sources for copies to the device; models and PPCommTicket own their buffers.
    TensorHolder   buffer_holder_;
    SamplingStates sampling_states_;
    /** History lengths included in sampling plans; the next plan carries only the suffix beyond these lengths. */
    std::unordered_map<RequestId, int32_t> token_history_lengths_;

    /** Keep draft-vocabulary Q on the last-stage GPU, keyed by request to survive pipeline-slot reuse. */
    std::unordered_map<RequestId, torch::Tensor> dspark_q_;

    /** A processed terminal plan prevents further execution while the engine waits for local stop(). */
    bool stopped_ = false;

    bool    sp_enabled_             = false;
    bool    is_dspark_              = false;
    int64_t propose_step_           = 0;
    size_t  position_id_len_factor_ = 1;
    /** PP currently publishes host state; keep layout selection on the shared device-state policy. */
    const mtp::DraftInputLayout                      draft_input_layout_ = mtp::selectDraftInputLayout(false);
    size_t                                           draft_vocab_size_ = 0;
    torch::Tensor                                    d2t_map_;
    torch::Tensor                                    dspark_markov_w1_;
    torch::Tensor                                    dspark_markov_w2_;
    std::unique_ptr<SpecLogitsVerifyRunner>          spec_logits_verify_runner_;
    std::unique_ptr<speculative::SpeculativeSampler> speculative_sampler_;
    std::unique_ptr<ModelBase>                       draft_model_;
    std::unique_ptr<ModelBase>                       draft_commit_model_;
    std::unique_ptr<speculative::FastTopKSampler>    fast_topk_sampler_;

    const ParallelismConfig      parallelism_config_;
    const RankLayout             pp_layout_;
    std::unique_ptr<PPTransport> transport_;
    std::vector<InflightBatch>   slots_;
    size_t                       current_slot_ = 0;

    /** Covers the last execution's model-buffer consumers, including outgoing activation copies. */
    torch::Event last_compute_done_ = cuda_graph::makeGraphEvent();

    /**
     * Receiving on the compute stream would make NCCL wait for the previous forward. A dedicated receive
     * stream enables overlap, while two persistent buffers protect active inputs and avoid per-batch
     * allocations once capacity is sufficient. Reuse relies on the intervening batch's waitLastRun()
     * completing prior readers and outgoing activation copies.
     */
    std::array<PPIntermediateTensors, 2> recv_buffers_;
    size_t                              next_recv_buffer_ = 0;
    torch::Stream                       recv_stream_;

    /** Match MTP publication completion on the DSpARK P tail, which owns the draft COMMIT. */
    const bool    track_dspark_cache_store_;
    torch::Stream dspark_cache_store_sync_stream_;
    torch::Tensor dspark_cache_store_status_;

    bool                               enable_detail_log_ = false;
    std::shared_ptr<ModelInputsLogger> model_inputs_logger_;
    std::function<void()>              profile_step_start_;
    std::function<void()>              profile_step_finish_;
    kmonitor::MetricsReporterPtr       metrics_reporter_ = nullptr;

    MetricsLoopReporter<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector>                   tps_reporter_;
    WallClockMetricsLoopReporter<RtpLLMWallClockTokenPSMetrics, RtpLLMTokenPSMetricsCollector> wall_tps_reporter_;
};

}  // namespace rtp_llm
