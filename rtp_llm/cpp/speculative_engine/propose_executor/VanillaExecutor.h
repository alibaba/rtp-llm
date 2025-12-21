#pragma once

#include "rtp_llm/cpp/cache_new/KVCacheManager.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/models/GptModel.h"
#include "rtp_llm/cpp/speculative_engine/propose_executor/ProposeExecutor.h"
#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"

namespace rtp_llm {

class VanillaExecutor: public ProposeExecutor {
public:
    explicit VanillaExecutor(std::unique_ptr<ProposeModelEngineInitParams>& propose_model_engine_init_params,
                             rtp_llm::DeviceBase*                           device,
                             const std::shared_ptr<KVCacheManager>&         cache_manager,
                             const std::shared_ptr<lora::LoraManager>&      lora_manager,
                             bool                                           warm_up = false):
        ProposeExecutor(device),
        propose_step_(propose_model_engine_init_params->gen_num_per_circle),
        normal_executor_(
            *propose_model_engine_init_params->vanilla_model_params, cache_manager, device_, lora_manager, warm_up) {
        RTP_LLM_LOG_INFO("VanillaExecutor propose step is %ld", propose_step_);
    }

    ~VanillaExecutor() {}

    absl::Status normalProcess(const std::list<GenerateStreamPtr>& streams) override {
        return normal_executor_.process(streams);
    }

    absl::Status propose(const std::list<GenerateStreamPtr>& streams, bool skip_check = false) override;

    size_t reserveStep() const override {
        return propose_step_;
    }

    void tpSyncStopFinishedStream(bool& need_stop) {
        if (device_->getDeviceProperties().tp_size <= 1) {
            return;
        }
        auto disable_sp_run =
            device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {1}, rtp_llm::AllocationType::HOST});
        auto disable_sp_run_ptr       = disable_sp_run->data<int32_t>();
        disable_sp_run_ptr[(size_t)0] = need_stop;

        device_->broadcast({{disable_sp_run}, 0});
        device_->syncCommunication(false);
        device_->syncAndCheck();
        need_stop = disable_sp_run_ptr[(size_t)0];
    };

private:
    size_t         propose_step_;
    NormalExecutor normal_executor_;
};

}  // namespace rtp_llm