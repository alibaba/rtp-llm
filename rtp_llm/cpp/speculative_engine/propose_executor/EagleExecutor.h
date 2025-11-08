#pragma once

#include "rtp_llm/cpp/cache_new/KVCacheManager.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/models/GptModel.h"
#include "rtp_llm/cpp/models/MTPModel.h"
#include "rtp_llm/cpp/models/Eagle3Model.h"
#include "rtp_llm/cpp/speculative_engine/propose_executor/ProposeExecutor.h"
#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"
#include "rtp_llm/cpp/speculative_engine/propose_executor/MTPBatchStreamProcessor.h"
#include "rtp_llm/cpp/speculative_engine/propose_executor/MTPExecutor.h"

namespace rtp_llm {

class EagleExecutor: public MTPExecutor {
public:
    explicit EagleExecutor(const std::string&                                   sp_type,
                           std::unique_ptr<ProposeModelEngineInitParams>&       propose_model_engine_init_params,
                           rtp_llm::DeviceBase*                                 device,
                           const std::vector<std::shared_ptr<KVCacheManager>>& mtp_cache_managers,
                           const std::shared_ptr<lora::LoraManager>&            lora_manager,
                           bool                                                 warm_up = false):
        MTPExecutor(sp_type, propose_model_engine_init_params, device, mtp_cache_managers, lora_manager, warm_up) {}

    absl::Status propose(const std::list<GenerateStreamPtr>& streams, bool skip_check) override;
};

}  // namespace rtp_llm