#pragma once

#include "maga_transformer/cpp/deprecated/ParallelModelWrapper.h"
#include "maga_transformer/cpp/dataclass/MagaInitParameter.h"
#include "maga_transformer/cpp/dataclass/MergedQuery.h"
#include "maga_transformer/cpp/engine_base/Executor.h"
#include "maga_transformer/cpp/normal_engine/NormalBatchStreamProcessor.h"

#include <memory>
namespace rtp_llm {

class NormalExecutor: public Executor {
public:
    explicit NormalExecutor(
            const MagaInitParams&                                                   params,
            ft::NcclParam                                                           tensor_para,
            ft::NcclParam                                                           pipeline_para,
            const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layer_weights,
            const std::unordered_map<std::string, ft::ConstBufferPtr>&              weights);
    absl::Status process(const std::list<GenerateStreamPtr>& streams) override;
    void addLoRA(const int64_t                                                   lora_id,
                 const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_a_weights,
                 const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_b_weights) override;
    void removeLoRA(const int64_t lora_id) override;

private:
    // TODO: remove this
    ModelRequest generateOldModelRequest(GptModelInputs& model_input);

private:
    std::unique_ptr<GptModel>             model_;
    std::unique_ptr<Sampler>              sampler_;
    std::unique_ptr<NormalBatchStreamProcessor> batch_stream_processor_;
    std::unique_ptr<ParallelModelWrapper> model_wrapper_;
    ft::NcclParam                         tensor_para_;
    ft::NcclParam                         pipeline_para_;
};

}  // namespace rtp_llm
