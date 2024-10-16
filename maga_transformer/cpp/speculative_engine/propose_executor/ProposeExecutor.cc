#include "maga_transformer/cpp/speculative_engine/propose_executor/ProposeExecutor.h"
#include "maga_transformer/cpp/speculative_engine/propose_executor/VanillaExecutor.h"
#include "maga_transformer/cpp/speculative_engine/propose_executor/DeterministicExecutor.h"

namespace rtp_llm {

std::unique_ptr<ProposeExecutor>
createProposeExecutor(const EngineInitParams&                        score_model_engine_init_params,
                      std::unique_ptr<ProposeModelEngineInitParams>& propose_model_engine_init_params,
                      ft::DeviceBase*                                device,
                      const std::shared_ptr<CacheManager>&           cache_manager,
                      const std::shared_ptr<lora::LoraManager>&      lora_manager) {
    const std::string&               sp_type          = propose_model_engine_init_params->sp_type;
    std::unique_ptr<ProposeExecutor> propose_executor = nullptr;
    if (sp_type == "vanilla") {
        propose_executor.reset(
            new VanillaExecutor(propose_model_engine_init_params, device, cache_manager, lora_manager));
    } else if (sp_type == "deterministic") {
        propose_executor.reset(
            new DeterministicExecutor(score_model_engine_init_params, propose_model_engine_init_params, device));
    } else {
        FT_FAIL("invalid sp_type: %s", sp_type);
    }
    return propose_executor;
}

};  // namespace rtp_llm