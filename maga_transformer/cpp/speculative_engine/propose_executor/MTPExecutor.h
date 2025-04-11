#pragma once

#include "maga_transformer/cpp/cache/CacheManager.h"
#include "maga_transformer/cpp/dataclass/MergedQuery.h"
#include "maga_transformer/cpp/models/GptModel.h"
#include "maga_transformer/cpp/models/MTPModel.h"
#include "maga_transformer/cpp/speculative_engine/propose_executor/ProposeExecutor.h"
#include "maga_transformer/cpp/normal_engine/NormalExecutor.h"
#include "maga_transformer/cpp/speculative_engine/propose_executor/MTPBatchStreamProcessor.h"


namespace rtp_llm {

class MTPExecutor: public ProposeExecutor {
public:
    explicit MTPExecutor(std::unique_ptr<ProposeModelEngineInitParams>& propose_model_engine_init_params,
                             ft::DeviceBase*                                device,
                             const std::vector<std::shared_ptr<CacheManager>>& mtp_cache_managers,
                             const std::shared_ptr<lora::LoraManager>&      lora_manager,
                             bool                                           warm_up = false):
        ProposeExecutor(device)
    {
        propose_step_ = std::min(propose_model_engine_init_params->gen_num_per_circle,
                                 propose_model_engine_init_params->mtp_model_params_->size());

        FT_LOG_INFO("create MTPExecutor use propose_step_ is %d, gen_num_per_circle is %d, mtp_model_num is %d",
            propose_step_,
            propose_model_engine_init_params->gen_num_per_circle,
            propose_model_engine_init_params->mtp_model_params_->size());

        size_t index = 0;
        for (auto& mtp_params : *propose_model_engine_init_params->mtp_model_params_) {
            auto cache_manager = (index < mtp_cache_managers.size()) ? mtp_cache_managers[index] : nullptr;
            auto executor = std::make_shared<NormalExecutor>(*mtp_params, cache_manager, device_, lora_manager, warm_up);

            auto norm_executor = std::make_shared<NormalExecutor>(*mtp_params, cache_manager, device_, lora_manager, warm_up);
            const auto& cache_config = cache_manager ? cache_manager->cacheConfig() : CacheConfig();
            executor->setBatchProcessor(std::move(std::make_unique<MTPBatchStreamProcessor>(
                mtp_params->gpt_init_parameter, cache_config, warm_up)));
            auto model_params = GptModelInitParams({device_, mtp_params->gpt_weights,
                Executor::genModelDescription(mtp_params->gpt_init_parameter),
                cache_manager ? ((std::optional<CacheManager::KVCacheBuffer>)cache_manager->kvCacheBuffer()) : std::nullopt});
            auto new_model = std::make_unique<MTPModel>(model_params);
            executor->setGptModel(std::move(new_model));
            normal_mtp_executors_.push_back(norm_executor);
            mtp_executors_.push_back((executor));
            index++;
        }
        FT_LOG_INFO("mtp executor init index is %d", index);

    }

    ~MTPExecutor() {};


    // care about hidden states.
    absl::Status normalProcess(const std::list<GenerateStreamPtr>& streams) override {

        for (auto& executor : normal_mtp_executors_) {
            auto status = executor->process(streams);
            if (!status.ok()) {
                return status;
            }
        }
        return absl::OkStatus();
    }

    absl::StatusOr<ProposeOutput> propose(const std::list<GenerateStreamPtr>& streams) override;

    void dynamicUpdateConfig(const ProposeDynamicConfig& config) override {}

    size_t reserveStep() const override {
        return propose_step_;
    }

private:
    size_t         propose_step_;
    std::vector<std::shared_ptr<NormalExecutor>> mtp_executors_;
    std::vector<std::shared_ptr<NormalExecutor>> normal_mtp_executors_;
};

}  // namespace rtp_llm