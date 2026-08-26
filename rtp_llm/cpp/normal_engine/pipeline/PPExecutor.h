#pragma once

#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/Executor.h"
#include "rtp_llm/cpp/model_utils/MlaConfig.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPTransport.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPTypes.h"
#include "rtp_llm/models_py/bindings/core/TensorHolder.h"

namespace rtp_llm {

struct EngineInitParams;
class KVCacheManager;
class ModelBase;
class Sampler;
class ExpertBalancer;
class NormalBatchStreamProcessor;

using PPTickets = std::vector<std::unique_ptr<PPCommTicket>>;

class PPExecutor final: public Executor {
public:
    PPExecutor(const EngineInitParams&                params,
               const std::shared_ptr<KVCacheManager>& cache_manager,
               MlaOpsType                             mla_ops_type        = MlaOpsType::AUTO,
               std::function<void()>                  profile_step_start  = nullptr,
               std::function<void()>                  profile_step_finish = nullptr);

    ~PPExecutor() override;

    absl::Status process(const std::list<GenerateStreamPtr>& streams, int64_t schedule_time_us = 0) override;

private:
    struct InflightBatch {
        std::list<GenerateStreamPtr> streams;
        PPTickets                    plan_sends;
        PPTickets                    activation_sends;
        PPTickets                    sample_result_sends;
    };

    struct RequestSamplingState {
        std::vector<BaseLogitsProcessorPtr> logits_processors;
        at::Generator                       generator;
        float                               cum_log_prob = 0.0f;
    };

    void                            sendObject(const torch::Tensor& object, PPTickets& tickets);
    torch::Tensor                   receiveObject();
    void                            asyncSendPlan(const PPExecutionPlan& plan, bool empty_plan, PPTickets& tickets);
    PPExecutionPlan                 receivePlan();
    void                            asyncSendSampleResult(const PPSampleResult& result, PPTickets& tickets);
    void                            asyncSendTensors(const PPIntermediateTensors& tensors, PPTickets& tickets);
    PPIntermediateTensors           receiveTensors(PPTickets& tickets);
    static void                     waitAll(PPTickets& tickets);
    absl::Status                    processSampleResult(InflightBatch& batch);
    absl::StatusOr<PPExecutionPlan> buildPlan(const std::list<GenerateStreamPtr>& streams);
    absl::StatusOr<SamplerInputs>   makeSamplerInputs(const PPSamplingData& sampling, const torch::Tensor& logits);
    void                            advanceSamplingStates(const PPSamplingData& sampling, PPSampleResult& result);

    bool isFirstStage() const {
        return pp_rank_ == 0;
    }

    bool isLastStage() const {
        return pp_rank_ + 1 == parallelism_config_.pp_size;
    }

    bool isStageRoot() const {
        return parallelism_config_.tp_rank == 0;
    }

private:
    const ParallelismConfig                           parallelism_config_;
    const int64_t                                     pp_rank_;
    std::shared_ptr<KVCacheManager>                   cache_manager_;
    std::unique_ptr<ModelBase>                        model_;
    std::unique_ptr<Sampler>                          sampler_;
    std::unique_ptr<NormalBatchStreamProcessor>       batch_stream_processor_;
    std::shared_ptr<ExpertBalancer>                   expert_balancer_;
    std::unique_ptr<PPTransport>                      transport_;
    std::function<void()>                             profile_step_start_;
    std::function<void()>                             profile_step_finish_;
    std::vector<std::optional<InflightBatch>>         slots_;
    TensorHolder                                      buffer_holder_;
    std::unordered_map<int64_t, RequestSamplingState> sampling_states_;
    std::vector<int64_t>                              output_vocab_ids_;
    int64_t                                           processor_eos_token_id_ = 0;
    size_t                                            current_slot_           = 0;
};

}  // namespace rtp_llm
