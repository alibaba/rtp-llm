#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "gtest/gtest.h"

namespace rtp_llm {

class EplbStartupIntegrationTest: public DeviceTestBase {};

TEST_F(EplbStartupIntegrationTest, MissingNativeExpertWeightsFailWithActionableError) {
    ModelConfig   model;
    RuntimeConfig runtime;
    KVCacheConfig cache;
    auto          params                 = createEngineInitParams(CustomConfig{}, model, runtime, cache);
    params.model_config_.moe_style       = 1;
    params.model_config_.moe_layer_index = {0};
    params.gpt_weights.layers[0].ffn_weights.moe_gate_weight = nullptr;
    params.eplb_config.eplb_mode                             = EplbMode::STATS;
    try {
        (void)NormalExecutor(params, nullptr);
        FAIL() << "missing expert weights must reject EPLB before model initialization";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string(error.what()).find("EPLB requires native MoE expert weights"), std::string::npos);
    }
}

}  // namespace rtp_llm
