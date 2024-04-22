#pragma once

#include "maga_transformer/cpp/dataclass/MagaInitParameter.h"
#include "maga_transformer/cpp/dataclass/MergedQuery.h"
#include "maga_transformer/cpp/engine_base/Executor.h"
#include "maga_transformer/cpp/models/GptModel.h"
#include "maga_transformer/cpp/models/Sampler.h"
#include "maga_transformer/cpp/common/fatal_util.h"

namespace rtp_llm {

class SpeculativeExecutor: public Executor {
public:
    SpeculativeExecutor(const MagaInitParams&                                           params,
                        const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layer_weights,
                        const std::unordered_map<std::string, ft::ConstBufferPtr>&              weights);
    absl::Status process(const std::list<GenerateStreamPtr>& streams) override;
    void addLoRA(const int64_t                                                   lora_id,
                 const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_a_weights,
                 const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_b_weights) override {
        RAISE_FATAL_ERROR("Speculative not support lora now");
    }
    void removeLoRA(const int64_t lora_id) override {
        RAISE_FATAL_ERROR("Speculative not support lora now");
    }

private:
    std::unique_ptr<GptModel> sp_model_;
    std::unique_ptr<Sampler>  sp_sampler_;
    int                       gen_num_;
};

}  // namespace rtp_llm
