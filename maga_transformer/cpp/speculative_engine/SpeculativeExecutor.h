#pragma once

#include "maga_transformer/cpp/deprecated/ParallelModelWrapper.h"
#include "maga_transformer/cpp/common/fatal_util.h"
#include "maga_transformer/cpp/dataclass/MagaInitParameter.h"
#include "maga_transformer/cpp/dataclass/MergedQuery.h"
#include "maga_transformer/cpp/models/GptModel.h"
#include "maga_transformer/cpp/models/Sampler.h"
#include "maga_transformer/cpp/normal_engine/NormalExecutor.h"
#include "maga_transformer/cpp/speculative_engine/SpeculativeBatchStreamProcessor.h"
#include "maga_transformer/cpp/speculative_engine/SpeculativeSampler.h"

namespace rtp_llm {

class SpeculativeExecutor: public Executor {
public:
    explicit SpeculativeExecutor(const MagaInitParams&                                                   params,
                                 ft::NcclParam                                                           tensor_para,
                                 ft::NcclParam                                                           pipeline_para,
                                 const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layer_weights,
                                 const std::unordered_map<std::string, ft::ConstBufferPtr>&              weights);
    absl::Status process(const std::list<GenerateStreamPtr>& streams) override;
    absl::Status addLoRA(const int64_t                                                           lora_id,
                         const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_a_weights,
                         const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_b_weights) override {
        return absl::UnimplementedError("speculative not support lora yet");
    }

    absl::Status removeLoRA(const int64_t lora_id) override {
        return absl::UnimplementedError("speculative not support lora yet");
    }

private:
    absl::StatusOr<std::list<GenerateStreamPtr>> getTargetStreams(const std::list<GenerateStreamPtr>& streams);
    absl::Status updateTargetProb(const std::list<GenerateStreamPtr>& streams, const ft::Buffer& logits);
    ModelRequest generateOldModelRequest(GptModelInputs& model_input);

private:
    std::unique_ptr<GptModel>                        model_;
    std::unique_ptr<SpeculativeSampler>              sampler_;
    std::unique_ptr<SpeculativeBatchStreamProcessor> batch_stream_processor_;
    std::unique_ptr<ParallelModelWrapper>            model_wrapper_;
    ft::NcclParam                                    tensor_para_;
    ft::NcclParam                                    pipeline_para_;
};

}  // namespace rtp_llm
