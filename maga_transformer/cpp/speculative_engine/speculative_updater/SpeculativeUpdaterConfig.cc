#include "maga_transformer/cpp/speculative_engine/speculative_updater/SpeculativeUpdaterConfig.h"
#include <torch/csrc/autograd/forward_grad.h>
namespace rtp_llm {

SpeculativeUpdaterConfig createSpeculativeUpdaterConfig(std::unique_ptr<ProposeModelEngineInitParams>& propose_model_engine_init_params) {
    std::string sp_type = propose_model_engine_init_params->sp_type;
    if (sp_type == "vanilla") {
        return {true, true, false};
    } else if (sp_type == "prompt_lookup") {
        return {false, true, false};
    } else {
        FT_FAIL("Invalid sp_type: %s", sp_type);
    }
    return SpeculativeUpdaterConfig();
}

}