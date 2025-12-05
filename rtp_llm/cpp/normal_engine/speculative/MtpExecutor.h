#pragma once

#include <memory>
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/engine_base/Executor.h"
#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/models/lora/LoraManager.h"
#include "rtp_llm/cpp/models/eplb/ExpertBalancer.h"
#include "rtp_llm/cpp/normal_engine/speculative/MtpBatchStreamProcessor.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"

namespace rtp_llm {

struct MtpMetricsCollector {
    RtpLLMExecutorMetricsCollector          executor_collector;
    RtpLLMTokenPSMetricsCollector           tps_collector;
    RtpLLMSpeculativeEngineMetricsCollector sp_engine_collector;

    bool not_skip = false;
};

class MtpExecutor: public Executor {
public:
    explicit MtpExecutor(const EngineInitParams&                           params,
                         std::unique_ptr<ProposeModelEngineInitParams>&    propose_params,
                         const std::shared_ptr<CacheManager>&              cache_manager,
                         const std::vector<std::shared_ptr<CacheManager>>& mtp_cache_managers,
                         rtp_llm::DeviceBase*                              device,
                         const std::shared_ptr<lora::LoraManager>&         lora_manager,
                         bool                                              warm_up = false);

    absl::Status process(const std::list<GenerateStreamPtr>& streams) override;
    bool         updateEplbConfig(const EplbConfig& config) override;

    void setTargetModel(std::unique_ptr<GptModel> model) {
        model_ = std::move(model);
    }

    void setDraftModel(std::unique_ptr<GptModel> model) {
        draft_model_ = std::move(model);
    }

    void setBatchProcessor(std::unique_ptr<MtpBatchStreamProcessor> processor) {
        batch_stream_processor_ = std::move(processor);
    }

protected:
    bool isTpRank0() const;

    void maybePrintModelInput(const GptModelInputs& model_input, const std::string& prefix) const;

    absl::Status prefillStep(const std::list<GenerateStreamPtr>& streams, MtpMetricsCollector& metrics_collector);

    absl::Status decodeStep(const std::list<GenerateStreamPtr>& streams, MtpMetricsCollector& metrics_collector);

    void draftModelSample(const BufferPtr& logits,
                          SamplerOutput&   sampler_output,
                          torch::Tensor&   draft_probs,
                          torch::Tensor&   draft_token_ids);

    void draftModelDecode(GptModelInputs&             model_input,
                          const StreamGroups&         stream_groups,
                          std::vector<torch::Tensor>& draft_probs_list,
                          torch::Tensor&              draft_token_ids_t);

    std::tuple<torch::Tensor, torch::Tensor> fastTopK(const torch::Tensor& probs, int top_k, int dim);

private:
    std::unique_ptr<GptModel>                model_;
    std::unique_ptr<Sampler>                 sampler_;
    std::unique_ptr<MtpBatchStreamProcessor> batch_stream_processor_;
    std::shared_ptr<CacheManager>            cache_manager_;
    std::shared_ptr<lora::LoraManager>       lora_manager_;
    bool                                     enable_ffn_disaggregate_ = false;
    bool                                     enable_detail_log_       = false;
    kmonitor::MetricsReporterPtr             metrics_reporter_        = nullptr;
    std::shared_ptr<ExpertBalancer>          expert_balancer_;
    size_t                                   vocab_size_;

    // for mtp
    size_t                                           propose_step_;
    size_t                                           propose_vocab_size_;
    std::unique_ptr<GptModel>                        draft_model_;
    std::vector<std::shared_ptr<CacheManager>>       mtp_cache_managers_;
    std::unique_ptr<speculative::SpeculativeSampler> speculative_sampler_;

    bool warm_up_;
};
};  // namespace rtp_llm