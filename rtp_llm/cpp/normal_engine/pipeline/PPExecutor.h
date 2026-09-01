#pragma once

#include <functional>
#include <memory>
#include <unordered_map>
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
#include "rtp_llm/models_py/bindings/core/TensorHolder.h"

namespace rtp_llm {

struct EngineInitParams;
struct GptModelInitParams;
class KVCacheManager;
class ModelBase;
class Sampler;
class ExpertBalancer;
class ModelInputsLogger;

using PPTickets = std::vector<std::unique_ptr<PPCommTicket>>;

class PPExecutor final: public Executor {
public:
    PPExecutor(const EngineInitParams&                params,
               const std::shared_ptr<KVCacheManager>& cache_manager,
               MlaOpsType                             mla_ops_type        = MlaOpsType::AUTO,
               std::function<void()>                  profile_step_start  = nullptr,
               std::function<void()>                  profile_step_finish = nullptr);

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
        PPTickets    plan_sends;
        PPTickets    activation_sends;
        PPTickets    execution_result_sends;

        void reset();
    };

    void                            sendObject(const torch::Tensor& object, PPTickets& tickets);
    torch::Tensor                   receiveObject();
    void                            asyncSendPlan(const PPExecutionPlan& plan, bool empty_plan, PPTickets& tickets);
    PPExecutionPlan                 receivePlan();
    void                            asyncSendExecutionResult(const PPExecutionResult& result, PPTickets& tickets);
    void                            asyncSendTensors(const PPIntermediateTensors& tensors, PPTickets& tickets);
    PPIntermediateTensors           receiveTensors(PPTickets& tickets);
    static void                     waitAll(PPTickets& tickets);
    absl::Status                    processExecutionResult(InflightBatch& batch);
    absl::StatusOr<PPExecutionPlan> buildPlan(const StreamGroups&         stream_groups,
                                              const std::vector<int64_t>& finished_request_ids);
    absl::StatusOr<SamplerInputs>   makeSamplerInputs(const PPSamplingPlan& sampling_plan,
                                                      const PPOutputConfig& output_config,
                                                      const torch::Tensor&  logits);

    void advanceSamplingStates(const PPSamplingPlan& sampling_plan,
                               const SamplerOutput&  sampler_output,
                               PPExecutionResult&    result);

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
    std::unique_ptr<ModelBase>                                               model_;
    std::unique_ptr<Sampler>                                                 sampler_;
    std::unique_ptr<PPBatchStreamProcessor>                                  batch_stream_processor_;
    std::shared_ptr<KVCacheManager>                                          cache_manager_;
    std::shared_ptr<ModelInputsLogger>                                       model_inputs_logger_;
    std::shared_ptr<ExpertBalancer>                                          expert_balancer_;
    const int64_t                                                            processor_eos_token_id_;
    kmonitor::MetricsReporterPtr                                             metrics_reporter_ = nullptr;
    MetricsLoopReporter<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector> tps_reporter_;
    WallClockMetricsLoopReporter<RtpLLMWallClockTokenPSMetrics, RtpLLMTokenPSMetricsCollector> wall_tps_reporter_;
    bool                    enable_detail_log_ = false;
    const ParallelismConfig parallelism_config_;
    // Single source of stage-role truth (hasEmbedding/hasLmHead) and the
    // materialized layer partition, shared with cache creation and the
    // Python loader/model mirrors.
    const PPLayout                             pp_layout_;
    std::unique_ptr<PPTransport>               transport_;
    std::function<void()>                      profile_step_start_;
    std::function<void()>                      profile_step_finish_;
    std::vector<InflightBatch>                 slots_;
    TensorHolder                               buffer_holder_;
    std::unordered_map<int64_t, SamplingState> sampling_states_;
    size_t                                     current_slot_ = 0;
};

}  // namespace rtp_llm
