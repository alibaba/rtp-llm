#pragma once

#include <memory>

#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#include "rtp_llm/cpp/speculative_engine/SpeculativeEngine.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"

using namespace std;
namespace rtp_llm {

std::shared_ptr<SpeculativeEngine>
createVanillaSpeculativeEngine(DeviceBase* device, const CustomConfig& config, GptInitParameter& params) {
    EngineInitParams                              score_params   = createEngineInitParams(device, config, params);
    EngineInitParams                              vanilla_params = createEngineInitParams(device, config, params);
    std::unique_ptr<ProposeModelEngineInitParams> propose_params = std::make_unique<ProposeModelEngineInitParams>(
        0, "vanilla", 1, vanilla_params.gpt_init_parameter, std::move(vanilla_params.gpt_weights));
    std::shared_ptr<SpeculativeEngine> engine = make_shared<SpeculativeEngine>(score_params, std::move(propose_params));
    THROW_IF_STATUS_ERROR(engine->init());
    return engine;
}

}  // namespace rtp_llm
