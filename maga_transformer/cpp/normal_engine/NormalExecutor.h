#pragma once

#include <memory>
#include "kmonitor/client/MetricsReporter.h"
#include "maga_transformer/cpp/engine_base/Executor.h"
#include "maga_transformer/cpp/deprecated/ParallelModelWrapper.h"
#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"
#include "maga_transformer/cpp/normal_engine/NormalBatchStreamProcessor.h"

namespace rtp_llm {

class NormalExecutor: public Executor {
public:
    explicit NormalExecutor(const EngineInitParams& params, ft::DeviceBase* device);
    absl::Status process(const std::list<GenerateStreamPtr>& streams) override;
    absl::Status addLoRA(const int64_t                                                           lora_id,
                         const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_a_weights,
                         const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_b_weights) override;
    absl::Status removeLoRA(const int64_t lora_id) override;
    void         reportMetrics(const StreamGroups& stream_groups);

private:
    // TODO: remove this
    ModelRequest generateOldModelRequest(GptModelInputs& model_input);

private:
    std::unique_ptr<GptModel>                   model_;
    std::unique_ptr<Sampler>                    sampler_;
    std::unique_ptr<NormalBatchStreamProcessor> batch_stream_processor_;
    std::unique_ptr<ParallelModelWrapper>       model_wrapper_;
    kmonitor::MetricsReporterPtr                metrics_reporter_ = nullptr;

    bool                                        use_new_device_impl_ = false;
};

}  // namespace rtp_llm
