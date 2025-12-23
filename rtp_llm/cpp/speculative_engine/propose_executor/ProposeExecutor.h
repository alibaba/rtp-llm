#pragma once

#include "absl/status/status.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/Executor.h"
#include "rtp_llm/cpp/models/lora/LoraManager.h"

namespace rtp_llm {

class ProposeExecutor {
public:
    ProposeExecutor(rtp_llm::DeviceBase* device): device_(device) {}
    virtual ~ProposeExecutor() {};

    virtual absl::Status propose(const std::list<GenerateStreamPtr>& streams, bool skip_check = false) = 0;
    virtual size_t       reserveStep() const                                                           = 0;
    virtual absl::Status normalProcess(const std::list<GenerateStreamPtr>& streams)                    = 0;

    virtual bool updateEplbConfig(const EPLBConfig& config) {
        return true;
    }

protected:
    rtp_llm::DeviceBase* device_;
};

std::unique_ptr<ProposeExecutor>
createProposeExecutor(const EngineInitParams&                        score_model_engine_init_params,
                      std::unique_ptr<ProposeModelEngineInitParams>& propose_model_engine_init_params,
                      rtp_llm::DeviceBase*                           device,
                      const std::shared_ptr<KVCacheManager>&         cache_manager,
                      const std::shared_ptr<lora::LoraManager>&      lora_manager);

}  // namespace rtp_llm