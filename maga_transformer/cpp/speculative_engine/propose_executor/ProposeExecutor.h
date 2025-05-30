#pragma once

#include "absl/status/status.h"
#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"
#include "maga_transformer/cpp/cache/CacheManager.h"
#include "maga_transformer/cpp/stream/GenerateStream.h"
#include "maga_transformer/cpp/engine_base/Executor.h"
#include "maga_transformer/cpp/lora/LoraManager.h"
#include "maga_transformer/cpp/speculative_engine/propose_executor/ProposeDynamicConfig.h"
#include "maga_transformer/cpp/speculative_engine/propose_executor/ProposeOutput.h"
#include "maga_transformer/cpp/speculative_engine/score_executor/ScoreOutput.h"
#include "maga_transformer/cpp/speculative_engine/speculative_sampler/SpeculativeSamplerOutput.h"

namespace rtp_llm {

class ProposeExecutor {
public:
    ProposeExecutor(rtp_llm::DeviceBase* device): device_(device) {}
    virtual ~ProposeExecutor(){};

    virtual absl::StatusOr<ProposeOutput> propose(const std::list<GenerateStreamPtr>& streams)       = 0;
    virtual void                          dynamicUpdateConfig(const ProposeDynamicConfig& config)    = 0;
    virtual size_t                        reserveStep() const                                        = 0;
    virtual absl::Status                  normalProcess(const std::list<GenerateStreamPtr>& streams) = 0;

protected:
    rtp_llm::DeviceBase* device_;
};

std::unique_ptr<ProposeExecutor>
createProposeExecutor(const EngineInitParams&                        score_model_engine_init_params,
                      std::unique_ptr<ProposeModelEngineInitParams>& propose_model_engine_init_params,
                      rtp_llm::DeviceBase*                                device,
                      const std::shared_ptr<CacheManager>&           cache_manager,
                      const std::vector<std::shared_ptr<CacheManager>>&           mtp_cache_managers,
                      const std::shared_ptr<lora::LoraManager>&      lora_manager);

}  // namespace rtp_llm