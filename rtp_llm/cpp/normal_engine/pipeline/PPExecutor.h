#pragma once

#include <functional>
#include <list>
#include <memory>
#include <vector>

#include <torch/torch.h>

#include "kmonitor/client/MetricsReporter.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/PPLayout.h"
#include "rtp_llm/cpp/engine_base/Executor.h"
#include "rtp_llm/cpp/engine_base/stream/SamplingState.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/model_utils/MlaConfig.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPBatchStreamProcessor.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPTransport.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPTypes.h"
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

    void setBatchProcessor(std::unique_ptr<PPBatchStreamProcessor> processor) {
        batch_stream_processor_ = std::move(processor);
    }

    void setModel(std::unique_ptr<ModelBase> model) {
        model_ = std::move(model);
    }

    using ModelFactory = std::function<std::unique_ptr<ModelBase>(const GptModelInitParams&)>;
    static ModelFactory test_model_factory;

private:
    struct InflightBatch {
        bool         skip_run = true;
        StreamGroups stream_groups;
        int64_t      schedule_time_us = 0;

        PPTickets plan_sends;
        PPTickets activation_sends;
        PPTickets execution_result_sends;

        void reset();
    };

    absl::Status warmUp(const ScheduleOutput& schedule_output);

    void prepareStreams(const std::list<GenerateStreamPtr>& streams) const;

    absl::StatusOr<PPExecutionPlan> buildPlan(const StreamGroups&         stream_groups,
                                              const std::vector<int64_t>& finished_request_ids);

    absl::StatusOr<PPExecutionResult> sampleTokens(const PPExecutionPlan& plan, const GptModelOutputs& model_output);

    void advanceSamplingStates(const PPSamplingPlan& sampling_plan, PPExecutionResult& result);

    absl::Status processExecutionResult(InflightBatch& batch);

    absl::StatusOr<PPExecutionResult> verifyDraftTokens(const PPExecutionPlan& plan,
                                                        const torch::Tensor&   target_logits);

    GptModelInputs prepareDraftInputForPrefill(const GptModelInputs&  target_input,
                                               const GptModelOutputs& target_output,
                                               const torch::Tensor&   sampled_token_ids,
                                               const torch::Tensor&   next_position_ids);

    GptModelInputs prepareDraftInputForDecode(const GptModelInputs&  target_input,
                                              const GptModelOutputs& target_output,
                                              const torch::Tensor&   accepted_token_ids,
                                              const torch::Tensor&   accepted_lengths);

    void draftSampleAndPropose(const PPExecutionPlan& plan,
                               const GptModelOutputs& model_output,
                               PPExecutionResult&     execution_result);

    torch::Tensor proposeDraftTokens(GptModelInputs draft_input, size_t num_draft_tokens);

    void asyncSendPlan(const PPExecutionPlan& plan, bool empty_plan, PPTickets& tickets);

    PPExecutionPlan receivePlan();

    void asyncSendTensors(const PPIntermediateTensors& tensors, PPTickets& tickets);

    PPIntermediateTensors receiveTensors(PPTickets& tickets);

    void asyncSendExecutionResult(const PPExecutionResult& result, PPTickets& tickets);

    void sendObject(const torch::Tensor& object, PPTickets& tickets);

    torch::Tensor receiveObject();

    static void waitAll(PPTickets& tickets);

    bool isFirstStage() const {
        return pp_layout_.hasEmbedding();
    }

    bool isLastStage() const {
        return pp_layout_.hasLmHead();
    }

    bool isStageRoot() const {
        return parallelism_config_.tp_rank == 0;
    }

private:
    const bool                                 warm_up_;
    std::shared_ptr<KVCacheManager>            cache_manager_;
    std::unique_ptr<ModelBase>                 model_;
    std::unique_ptr<Sampler>                   sampler_;
    std::unique_ptr<PPBatchStreamProcessor>    batch_stream_processor_;
    std::shared_ptr<ExpertBalancer>            expert_balancer_;
    TensorHolder                               buffer_holder_;
    SamplingStates                            sampling_states_;

    bool                                             mtp_enabled_            = false;
    size_t                                           propose_step_           = 0;
    size_t                                           position_id_len_factor_ = 1;
    std::unique_ptr<SpecLogitsVerifyRunner>          spec_logits_verify_runner_;
    std::unique_ptr<speculative::SpeculativeSampler> speculative_sampler_;
    std::unique_ptr<ModelBase>                       draft_model_;
    std::unique_ptr<speculative::FastTopKSampler>    fast_topk_sampler_;

    const ParallelismConfig      parallelism_config_;
    const PPLayout               pp_layout_;
    std::unique_ptr<PPTransport> transport_;
    std::vector<InflightBatch>   slots_;
    size_t                       current_slot_ = 0;

    bool                               enable_detail_log_ = false;
    std::shared_ptr<ModelInputsLogger> model_inputs_logger_;
    std::function<void()>              profile_step_start_;
    std::function<void()>              profile_step_finish_;
    kmonitor::MetricsReporterPtr       metrics_reporter_ = nullptr;

    MetricsLoopReporter<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector> tps_reporter_;
    WallClockMetricsLoopReporter<RtpLLMWallClockTokenPSMetrics, RtpLLMTokenPSMetricsCollector> wall_tps_reporter_;
};

}  // namespace rtp_llm
