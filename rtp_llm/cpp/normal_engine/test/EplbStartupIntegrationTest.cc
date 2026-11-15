#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "gtest/gtest.h"

namespace rtp_llm {

class EplbStartupIntegrationTest: public DeviceTestBase {};

TEST_F(EplbStartupIntegrationTest, MissingNativeExpertWeightsFailWithActionableError) {
    ModelConfig model;
    RuntimeConfig runtime;
    KVCacheConfig cache;
    auto params = createEngineInitParams(CustomConfig{}, model, runtime, cache);
    params.model_config_.moe_style = 1;
    params.model_config_.moe_layer_index = {0};
    params.gpt_weights.layers[0].ffn_weights.moe_gate_weight = nullptr;
    params.eplb_config.eplb_mode = EplbMode::STATS;
    EXPECT_THROW((void)NormalExecutor(params, nullptr), std::exception);
}

}  // namespace rtp_llm
