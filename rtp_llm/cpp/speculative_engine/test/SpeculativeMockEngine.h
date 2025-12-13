#pragma once

#include <memory>

#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#include "rtp_llm/cpp/speculative_engine/SpeculativeEngine.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"

using namespace std;
namespace rtp_llm {

std::shared_ptr<SpeculativeEngine>
createVanillaSpeculativeEngine(DeviceBase* device, const CustomConfig& config) {
    rtp_llm::ModelConfig model_config;
    rtp_llm::RuntimeConfig runtime_config;
    rtp_llm::KVCacheConfig kv_cache_config;
    EngineInitParams                              score_params   = createEngineInitParams(device, config, model_config, runtime_config, kv_cache_config);
    EngineInitParams                              vanilla_params = createEngineInitParams(device, config, model_config, runtime_config, kv_cache_config);
    std::unique_ptr<ProposeModelEngineInitParams> propose_params = std::make_unique<ProposeModelEngineInitParams>(
        0, SP_TYPE_VANILLA, 1, vanilla_params.model_config_,
        vanilla_params,
        std::move(vanilla_params.gpt_weights),
        py::none(), vanilla_params.py_eplb);
    std::shared_ptr<SpeculativeEngine> engine = make_shared<SpeculativeEngine>(score_params, std::move(propose_params));
    THROW_IF_STATUS_ERROR(engine->init());
    return engine;
}

}  // namespace rtp_llm
