#include "rtp_llm/cpp/speculative_engine/propose_executor/ProposeExecutor.h"
#include "rtp_llm/cpp/speculative_engine/propose_executor/EagleExecutor.h"
#include "rtp_llm/cpp/speculative_engine/propose_executor/VanillaExecutor.h"
#include "rtp_llm/cpp/speculative_engine/propose_executor/DeterministicExecutor.h"
#include "rtp_llm/cpp/speculative_engine/propose_executor/MTPExecutor.h"

namespace rtp_llm {

std::unique_ptr<ProposeExecutor>
createProposeExecutor(const EngineInitParams&                              score_model_engine_init_params,
                      std::unique_ptr<ProposeModelEngineInitParams>&       propose_model_engine_init_params,
                      rtp_llm::DeviceBase*                                 device,
                      const std::shared_ptr<KVCacheManager>&               cache_manager,
                      const std::vector<std::shared_ptr<KVCacheManager>>& mtp_cache_managers,
                      const std::shared_ptr<lora::LoraManager>&            lora_manager) {
    const std::string&               sp_type          = propose_model_engine_init_params->sp_type;
    std::unique_ptr<ProposeExecutor> propose_executor = nullptr;
    if (sp_type == "vanilla") {
        propose_executor.reset(
            new VanillaExecutor(propose_model_engine_init_params, device, cache_manager, lora_manager));
    } else if (sp_type == "deterministic") {
        propose_executor.reset(
            new DeterministicExecutor(score_model_engine_init_params, propose_model_engine_init_params, device));
    } else if (sp_type == "mtp") {
        propose_executor.reset(
            new MTPExecutor(sp_type, propose_model_engine_init_params, device, mtp_cache_managers, lora_manager));
    } else if (sp_type == "eagle" || sp_type == "eagle3") {
        propose_executor.reset(
            new EagleExecutor(sp_type, propose_model_engine_init_params, device, mtp_cache_managers, lora_manager));
    } else {
        RTP_LLM_FAIL("invalid sp_type: %s", sp_type);
    }
    return propose_executor;
}

};  // namespace rtp_llm