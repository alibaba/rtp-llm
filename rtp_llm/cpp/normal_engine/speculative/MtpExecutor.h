#pragma once

#include <memory>
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
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

class MtpBufferHolder {
public:
    void hold(const GptModelInputs& model_input) {
        host_buffers_holder_.push_back(model_input.combo_tokens);
        host_buffers_holder_.push_back(model_input.input_lengths);
        host_buffers_holder_.push_back(model_input.sequence_lengths);
        host_buffers_holder_.push_back(model_input.lm_output_indexes);
        host_buffers_holder_.push_back(model_input.prefix_lengths);
    }

    void hold(const BufferPtr& buffer) {
        host_buffers_holder_.push_back(buffer);
    }

    void release() {
        host_buffers_holder_.clear();
    }

private:
    std::vector<BufferPtr> host_buffers_holder_;
};

class MtpExecutor: public Executor {
public:
    explicit MtpExecutor(const EngineInitParams&                        params,
                         std::unique_ptr<ProposeModelEngineInitParams>& propose_params,
                         const std::shared_ptr<KVCacheManager>&         cache_manager,
                         rtp_llm::DeviceBase*                           device,
                         const std::shared_ptr<lora::LoraManager>&      lora_manager,
                         bool                                           warm_up = false);

    absl::Status process(const std::list<GenerateStreamPtr>& streams) override;
    bool         updateEplbConfig(const EPLBConfig& config) override;

    void setTargetModel(std::unique_ptr<GptModel> model) {
        model_ = std::move(model);
    }

    void setDraftModel(std::unique_ptr<GptModel> model) {
        draft_model_ = std::move(model);
    }

    void setBatchProcessor(std::unique_ptr<MtpBatchStreamProcessor> processor) {
        batch_stream_processor_ = std::move(processor);
    }

public:
    static GenerateStreamPtr createMinFakePrefillStream(int                    max_new_tokens,
                                                        const ModelConfig&     model_config,
                                                        const RuntimeConfig&   runtime_config,
                                                        const ResourceContext& resource_context,
                                                        DeviceBase*            device);
    static GenerateStreamPtr createMinFakeDecodeStream(int                    max_new_tokens,
                                                       const ModelConfig&     model_config,
                                                       const RuntimeConfig&   runtime_config,
                                                       const ResourceContext& resource_context,
                                                       DeviceBase*            device);

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

    void prepareStreams(const std::list<GenerateStreamPtr>& streams,
                        std::list<GenerateStreamPtr>&       prefill_streams,
                        std::list<GenerateStreamPtr>&       decode_streams);

private:
    std::unique_ptr<GptModel>                model_;
    std::unique_ptr<Sampler>                 sampler_;
    std::unique_ptr<MtpBatchStreamProcessor> batch_stream_processor_;
    std::shared_ptr<KVCacheManager>          cache_manager_;
    std::shared_ptr<lora::LoraManager>       lora_manager_;
    bool                                     enable_ffn_disaggregate_ = false;
    bool                                     enable_detail_log_       = false;
    kmonitor::MetricsReporterPtr             metrics_reporter_        = nullptr;
    std::shared_ptr<ExpertBalancer>          expert_balancer_;
    size_t                                   vocab_size_;

    // for mtp
    DataType                                         data_type_;
    size_t                                           hidden_size_;
    size_t                                           propose_step_;
    size_t                                           propose_vocab_size_;
    std::unique_ptr<GptModel>                        draft_model_;
    std::unique_ptr<speculative::SpeculativeSampler> speculative_sampler_;

    // holder for host buffers to avoid early free before H2D copy kernel execution
    MtpBufferHolder buffer_holder_;

    bool     warm_up_;
    RoleType role_type_;
};
};  // namespace rtp_llm