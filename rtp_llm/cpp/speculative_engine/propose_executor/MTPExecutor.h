#pragma once

#include "rtp_llm/cpp/cache_new/KVCacheManager.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/models/GptModel.h"
#include "rtp_llm/cpp/models/MTPModel.h"
#include "rtp_llm/cpp/models/Eagle3Model.h"
#include "rtp_llm/cpp/speculative_engine/propose_executor/ProposeExecutor.h"
#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"
#include "rtp_llm/cpp/speculative_engine/propose_executor/MTPBatchStreamProcessor.h"

namespace rtp_llm {

class MTPExecutor: public ProposeExecutor {
public:
    explicit MTPExecutor(const std::string&                                  sp_type,
                         std::unique_ptr<ProposeModelEngineInitParams>&      propose_model_engine_init_params,
                         rtp_llm::DeviceBase*                                device,
                         const std::vector<std::shared_ptr<KVCacheManager>>& mtp_cache_managers,
                         const std::shared_ptr<lora::LoraManager>&           lora_manager,
                         bool                                                warm_up = false):
        ProposeExecutor(device), sp_type_(sp_type) {

        if (sp_type_ == "eagle") {
            propose_step_ = propose_model_engine_init_params->gen_num_per_circle;
        } else {
            propose_step_ = std::min(propose_model_engine_init_params->gen_num_per_circle,
                                     propose_model_engine_init_params->mtp_model_params_->size());
        }

        RTP_LLM_LOG_INFO("create MTPExecutor use propose_step_ is %d, gen_num_per_circle is %d, mtp_model_num is %d",
                         propose_step_,
                         propose_model_engine_init_params->gen_num_per_circle,
                         propose_model_engine_init_params->mtp_model_params_->size());

        size_t index = 0;
        for (auto& mtp_params : *propose_model_engine_init_params->mtp_model_params_) {
            RTP_LLM_LOG_INFO("index %d, mtp model_id %d", index, mtp_params->model_id);
            auto cache_manager = (index < mtp_cache_managers.size()) ? mtp_cache_managers[index] : nullptr;
            auto executor =
                std::make_shared<NormalExecutor>(*mtp_params, cache_manager, device_, lora_manager, warm_up);

            auto norm_executor =
                std::make_shared<NormalExecutor>(*mtp_params, cache_manager, device_, lora_manager, warm_up);
            const auto& cache_config = cache_manager ? cache_manager->cacheConfig() : CacheConfig();
            executor->setBatchProcessor(std::move(
                std::make_unique<MTPBatchStreamProcessor>(mtp_params->model_config_, mtp_params->pd_sep_config, mtp_params->profiling_debug_logging_config, cache_config, warm_up)));
            auto model_params = GptModelInitParams(
                {device_,
                 mtp_params->gpt_weights,
                 Executor::genModelDescription(mtp_params->model_config_, mtp_params->parallelism_config, mtp_params->eplb_config, mtp_params->moe_config),
                 cache_manager ? std::make_optional(cache_manager->kvCacheBuffer()) : std::nullopt,
                                 std::nullopt,
                 mtp_params->model_id});
            std::unique_ptr<GptModel> new_model;
            if (sp_type_ == "mtp" || sp_type_ == "eagle") {
                RTP_LLM_LOG_INFO("prepare mtp model");
                executor->setGptModel(std::make_unique<MTPModel>(model_params));
                norm_executor->setGptModel(std::make_unique<MTPModel>(model_params));
            } else if (sp_type == "eagle3") {
                RTP_LLM_LOG_INFO("prepare eagle3 model");
                executor->setGptModel(std::make_unique<Eagle3Model>(model_params));
                norm_executor->setGptModel(std::make_unique<Eagle3Model>(model_params));
            } else {
                RTP_LLM_LOG_ERROR("unknown sp_type %s", sp_type_.c_str());
            }
            normal_mtp_executors_.push_back(norm_executor);
            mtp_executors_.push_back((executor));
            index++;
        }
        RTP_LLM_LOG_INFO("mtp executor init index is %d", index);
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

    absl::Status propose(const std::list<GenerateStreamPtr>& streams, bool skip_check = false) override;

    size_t reserveStep() const override {
        return propose_step_;
    }

    bool updateEplbConfig(const EPLBConfig& config) override;

protected:
    std::string                                  sp_type_;
    size_t                                       propose_step_;
    std::vector<std::shared_ptr<NormalExecutor>> mtp_executors_;
    std::vector<std::shared_ptr<NormalExecutor>> normal_mtp_executors_;
};

}  // namespace rtp_llm