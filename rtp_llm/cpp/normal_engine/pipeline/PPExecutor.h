#pragma once

#include <atomic>
#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <vector>

#include <torch/torch.h>

#include "absl/status/status.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/Executor.h"
#include "rtp_llm/cpp/model_utils/MlaConfig.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPTransport.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPTypes.h"

namespace rtp_llm {

struct EngineInitParams;
class KVCacheManager;
class ModelBase;
class Sampler;
class ExpertBalancer;

// PPExecutor owns only stage-local execution and communication state. It must
// not schedule or retain GenerateStream/StreamGroup objects. The first stage
// consumes streams selected by NormalEngine; later stages receive a plan.
class PPExecutor final: public Executor {
public:
    PPExecutor(const EngineInitParams&                params,
               const std::shared_ptr<KVCacheManager>& cache_manager,
               MlaOpsType                             mla_ops_type        = MlaOpsType::AUTO,
               std::function<void()>                  profile_step_start  = nullptr,
               std::function<void()>                  profile_step_finish = nullptr);

    ~PPExecutor() override;

    absl::Status process(const std::list<GenerateStreamPtr>& streams, int64_t schedule_time_us = 0) override;
    void         requestStop() override;

private:
    struct InflightBatch {
        explicit InflightBatch(PPExecutionPlan execution_plan);

        InflightBatch(InflightBatch&&) noexcept            = default;
        InflightBatch& operator=(InflightBatch&&) noexcept = default;
        InflightBatch(const InflightBatch&)                = delete;
        InflightBatch& operator=(const InflightBatch&)     = delete;

        PPExecutionPlan                      plan;
        std::optional<PPIntermediateTensors> input_tensors;
        GptModelOutputs                      model_output;
        std::optional<PPIntermediateTensors> output_tensors;
        std::optional<PPSampleResult>        sample_result;
        torch::Event                         forward_done;
        std::unique_ptr<PPCommTicket>        plan_send;
        std::unique_ptr<PPCommTicket>        tensors_recv;
        std::unique_ptr<PPCommTicket>        output_send;
    };

    void runStage(InflightBatch& batch, GptModelInputs& local_model_input);

private:
    const ParallelismConfig                   parallelism_config_;
    const int64_t                             pp_rank_;
    std::shared_ptr<KVCacheManager>           cache_manager_;
    std::unique_ptr<ModelBase>                model_;
    std::unique_ptr<Sampler>                  sampler_;
    std::shared_ptr<ExpertBalancer>           expert_balancer_;
    std::unique_ptr<PPTransport>              transport_;
    std::function<void()>                     profile_step_start_;
    std::function<void()>                     profile_step_finish_;
    std::vector<std::optional<InflightBatch>> slots_;
    size_t                                    current_slot_ = 0;
    std::atomic<bool>                         stop_requested_{false};
};

}  // namespace rtp_llm
