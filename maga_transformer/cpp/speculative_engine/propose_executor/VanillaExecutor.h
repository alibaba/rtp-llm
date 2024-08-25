#pragma once

#include "maga_transformer/cpp/cache/CacheManager.h"
#include "maga_transformer/cpp/dataclass/MergedQuery.h"
#include "maga_transformer/cpp/models/GptModel.h"
#include "maga_transformer/cpp/speculative_engine/propose_executor/ProposeExecutor.h"
#include "maga_transformer/cpp/normal_engine/NormalExecutor.h"
namespace rtp_llm {

class VanillaExecutor: public ProposeExecutor {
public:
    explicit VanillaExecutor(std::unique_ptr<ProposeModelEngineInitParams>& propose_model_engine_init_params,
                             ft::DeviceBase*                                device,
                             const std::shared_ptr<CacheManager>&           cache_manager,
                             const std::shared_ptr<lora::LoraManager>&      lora_manager):
        ProposeExecutor(device),
        propose_step_(propose_model_engine_init_params->vanilla_model_params->gpt_init_parameter.gen_num_per_circle_),
        normal_executor_(
            *propose_model_engine_init_params->vanilla_model_params, cache_manager, device_, lora_manager, false) {}

    ~VanillaExecutor() {}

    absl::StatusOr<ProposeOutput> propose(const std::list<GenerateStreamPtr>& streams) override;

    void dynamicUpdateConfig(const ProposeDynamicConfig& config) override {}

    // TODO(xyz): consider bonus tokens
    size_t reserveStep() const override {
        return propose_step_;
    }

private:
    size_t         propose_step_;
    NormalExecutor normal_executor_;
};

}  // namespace rtp_llm