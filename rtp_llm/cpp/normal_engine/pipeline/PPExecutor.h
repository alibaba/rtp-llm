#pragma once

#include <functional>
#include <list>
#include <memory>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "kmonitor/client/MetricsReporter.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/RankLayout.h"
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

    void notifyShutdown() override {
        stopping_ = true;
    }

    /** True once the shutdown sentinel flow has completed for this stage (loop may exit). */
    bool shutdownCompleted() const {
        return shutdown_completed_;
    }

    using ModelFactory = std::function<std::unique_ptr<ModelBase>(const GptModelInitParams&)>;
    static ModelFactory test_model_factory;

private:
    struct InflightBatch {
        bool                          skip_run = true;
        StreamGroups                  stream_groups;
        int64_t                       schedule_time_us = 0;
        RtpLLMExecutorMetricsCollector executor_collector;

        PPTickets plan_sends;
        PPTickets activation_sends;
        PPTickets execution_result_sends;

        void reset();
    };

    absl::Status warmUp(const ScheduleOutput& schedule_output);

    void releaseAllModelBuffers();

    void prepareStreams(std::list<GenerateStreamPtr>& streams);

    absl::StatusOr<PPExecutionPlan> buildPlan(const StreamGroups&         stream_groups,
                                              const std::vector<int64_t>& finished_request_ids);

    void sampleTokens(const PPExecutionPlan& plan, const GptModelOutputs& model_output, PPExecutionResult& result);

    void advanceSamplingStates(const PPSamplingPlan& sampling_plan, PPExecutionResult& result);

    absl::Status processExecutionResult(InflightBatch& batch);

    void verifyDraftTokens(const PPExecutionPlan& plan, const torch::Tensor& target_logits, PPExecutionResult& result);

    void prepareDraftPrefillAfterTargetPrefill(GptModelInputs&      draft_prefill_input,
                                               const torch::Tensor& target_hidden_states,
                                               const torch::Tensor& sampled_token_ids,
                                               const torch::Tensor& next_position_ids);

    void prepareDraftPrefillAfterVerify(GptModelInputs&      draft_prefill_input,
                                        const torch::Tensor& target_hidden_states,
                                        const torch::Tensor& accepted_token_ids,
                                        const torch::Tensor& accepted_lengths);

    void prepareDSparkCommitInput(GptModelInputs& commit_input, const torch::Tensor& target_features);

    void broadcastPostRejectionInputs(GptModelInputs& draft_input);

    GptModelInputs prepareDSparkProposeInput(const GptModelInputs& target_input,
                                             bool                  is_decode,
                                             const torch::Tensor&  new_token_ids,
                                             const torch::Tensor&  new_token_lengths);

    GptModelOutputs forwardDraftModel(GptModelInputs& draft_input);

    torch::Tensor runDraftStep(const GptModelInputs& target_input,
                               const torch::Tensor&  draft_next_position_ids,
                               const torch::Tensor&  target_hidden_states,
                               const torch::Tensor&  new_token_ids,
                               const torch::Tensor&  new_token_lengths,
                               bool                  is_decode);

    void fillFailedDraftRows(PPExecutionResult& result);

    void asyncSendPlan(const PPExecutionPlan& plan, bool empty_plan, PPTickets& tickets);

    PPExecutionPlan receivePlan();

    void asyncSendTensors(const PPIntermediateTensors& tensors, PPTickets& tickets);

    PPIntermediateTensors receiveTensors(PPTickets& tickets);

    void asyncSendExecutionResult(const PPExecutionResult& result, PPTickets& tickets);

    void sendObject(const torch::Tensor& object, PPTickets& tickets);

    torch::Tensor receiveObject();

    void waitAll(PPTickets& tickets, const char* what, bool throw_on_timeout = true);

    void waitTicket(PPCommTicket& ticket, const char* what, bool throw_on_timeout = true);

    bool isFirstStage() const {
        return pp_layout_.hasEmbedding();
    }

    bool isLastStage() const {
        return pp_layout_.hasLmHead();
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

    /** PP comm watchdog: bounded waits once shutdown has been notified. */
    bool    stopping_                 = false;
    int64_t comm_watchdog_timeout_ms_ = 30000;

    /** Sentinel flow state; only touched from the engine loop thread. */
    size_t idle_streak_        = 0;
    bool   shutdown_completed_ = false;

    bool                                             sp_enabled_             = false;
    bool                                             is_dspark_              = false;
    int32_t                                          dspark_mask_token_id_   = -1;
    size_t                                           propose_step_           = 0;
    size_t                                           position_id_len_factor_ = 1;
    /** PP currently publishes host state; keep layout selection on the shared device-state policy. */
    const mtp::DraftInputLayout                      draft_input_layout_ = mtp::selectDraftInputLayout(false);
    mtp::DSparkProposeInputBuffers                   dspark_propose_input_buffers_;
    std::unique_ptr<SpecLogitsVerifyRunner>          spec_logits_verify_runner_;
    std::unique_ptr<speculative::SpeculativeSampler> speculative_sampler_;
    std::unique_ptr<ModelBase>                       draft_model_;
    std::unique_ptr<speculative::FastTopKSampler>    fast_topk_sampler_;

    const ParallelismConfig      parallelism_config_;
    const RankLayout             pp_layout_;
    std::unique_ptr<PPTransport> transport_;
    std::vector<InflightBatch>   slots_;
    size_t                       current_slot_ = 0;

    bool                               enable_detail_log_ = false;
    std::shared_ptr<ModelInputsLogger> model_inputs_logger_;
    std::function<void()>              profile_step_start_;
    std::function<void()>              profile_step_finish_;
    kmonitor::MetricsReporterPtr       metrics_reporter_ = nullptr;

    MetricsLoopReporter<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector>                   tps_reporter_;
    WallClockMetricsLoopReporter<RtpLLMWallClockTokenPSMetrics, RtpLLMTokenPSMetricsCollector> wall_tps_reporter_;
};

}  // namespace rtp_llm
