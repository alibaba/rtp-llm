#include "maga_transformer/cpp/speculative_engine/speculative_sampler/SpeculativeSampler.h"
#include "maga_transformer/cpp/speculative_engine/speculative_sampler/RejectionSampler.h"
#include "maga_transformer/cpp/utils/AssertUtils.h"

namespace ft = fastertransformer;

namespace rtp_llm {

std::unique_ptr<SpeculativeSampler> createSpeculativeSampler(const std::unique_ptr<ProposeModelEngineInitParams>& propose_model_engine_init_params, ft::DeviceBase* device) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    const std::string& sp_type = propose_model_engine_init_params->sp_type;
    std::unique_ptr<SpeculativeSampler> speculative_sampler = nullptr;
    if (sp_type == "vanilla" || sp_type == "deterministic" || sp_type == "mtp") {
        speculative_sampler.reset(new RejectionSampler(device));
    } else {
        FT_FAIL("Invalid sp_type: %s", sp_type);
    }
    return speculative_sampler;
}

};