#include "gtest/gtest.h"
#include "rtp_llm/cpp/models/FullPrefillCudaGraphEligibility.h"

namespace rtp_llm {

namespace {

GptModelDescription makeFullPrefillMoeDescription() {
    GptModelDescription description;
    description.data_type   = DataType::TYPE_BF16;
    description.act_qscheme = QScheme::Qfp8PerTokenBlock;
    MoeConfigs model_moe_config;
    model_moe_config.expert_num     = 96;
    model_moe_config.top_k          = 8;
    model_moe_config.use_all_gather = true;
    description.ffn_conf.moe_configs.emplace(model_moe_config);
    return description;
}

MoeConfig makeFullPrefillMoeRuntimeConfig() {
    MoeConfig config;
    config.moe_strategy           = "fp8_per_block_no_dp_masked";
    config.use_all_gather         = true;
    config.use_deepep_moe         = false;
    config.use_deepep_internode   = false;
    config.use_deepep_low_latency = false;
    return config;
}

}  // namespace

TEST(FullPrefillCudaGraphEligibilityTest, testFullPrefillCudaGraphSupportsDenseModel) {
    GptModelDescription description;
    EXPECT_TRUE(supportsFullPrefillCudaGraphMoe(description, ParallelismConfig{}, MoeConfig{}));
}

TEST(FullPrefillCudaGraphEligibilityTest, testFullPrefillCudaGraphSupportsSingleGpuFp8MaskedMoe) {
    EXPECT_TRUE(supportsFullPrefillCudaGraphMoe(
        makeFullPrefillMoeDescription(), ParallelismConfig{}, makeFullPrefillMoeRuntimeConfig()));
}

TEST(FullPrefillCudaGraphEligibilityTest, testFullPrefillCudaGraphRejectsAutoMoeStrategy) {
    auto config         = makeFullPrefillMoeRuntimeConfig();
    config.moe_strategy = "auto";
    EXPECT_FALSE(supportsFullPrefillCudaGraphMoe(makeFullPrefillMoeDescription(), ParallelismConfig{}, config));
}

TEST(FullPrefillCudaGraphEligibilityTest, testFullPrefillRocmPerChannelDoesNotChangeCudaEligibility) {
    auto description        = makeFullPrefillMoeDescription();
    auto config             = makeFullPrefillMoeRuntimeConfig();
    description.act_qscheme = QScheme::Qfp8PerToken;
    config.moe_strategy     = "auto";
    EXPECT_TRUE(supportsFullPrefillCudaGraphMoe(description, ParallelismConfig{}, config, true));
    EXPECT_FALSE(supportsFullPrefillCudaGraphMoe(description, ParallelismConfig{}, config));
    // A CUDA strategy name must not masquerade as the validated AITER path.
    EXPECT_FALSE(supportsFullPrefillCudaGraphMoe(
        makeFullPrefillMoeDescription(), ParallelismConfig{}, makeFullPrefillMoeRuntimeConfig(), true));
    config.use_deepep_low_latency = true;
    EXPECT_FALSE(supportsFullPrefillCudaGraphMoe(description, ParallelismConfig{}, config, true));
    config.use_deepep_low_latency = false;
    auto parallelism = ParallelismConfig{};
    parallelism.tp_size = 2;
    EXPECT_FALSE(supportsFullPrefillCudaGraphMoe(description, parallelism, config, true));
    description.ffn_conf.moe_configs->enable_eplb = true;
    EXPECT_FALSE(supportsFullPrefillCudaGraphMoe(description, ParallelismConfig{}, config, true));
}

TEST(FullPrefillCudaGraphEligibilityTest, testFullPrefillCudaGraphRejectsNonFp8PerBlockMoe) {
    auto description        = makeFullPrefillMoeDescription();
    description.act_qscheme = QScheme::NoQuantize;
    EXPECT_FALSE(supportsFullPrefillCudaGraphMoe(description, ParallelismConfig{}, makeFullPrefillMoeRuntimeConfig()));
}

TEST(FullPrefillCudaGraphEligibilityTest, testFullPrefillCudaGraphRejectsGraphUnsafeMoeTransport) {
    auto config                   = makeFullPrefillMoeRuntimeConfig();
    config.use_deepep_low_latency = true;
    EXPECT_FALSE(supportsFullPrefillCudaGraphMoe(makeFullPrefillMoeDescription(), ParallelismConfig{}, config));

    config                = makeFullPrefillMoeRuntimeConfig();
    config.use_all_gather = false;
    EXPECT_FALSE(supportsFullPrefillCudaGraphMoe(makeFullPrefillMoeDescription(), ParallelismConfig{}, config));
}

TEST(FullPrefillCudaGraphEligibilityTest, testFullPrefillCudaGraphRejectsDistributedOrEplbMoe) {
    auto parallelism       = ParallelismConfig{};
    parallelism.ep_size    = 2;
    parallelism.dp_size    = 2;
    parallelism.world_size = 2;
    EXPECT_FALSE(supportsFullPrefillCudaGraphMoe(
        makeFullPrefillMoeDescription(), parallelism, makeFullPrefillMoeRuntimeConfig()));

    auto description                              = makeFullPrefillMoeDescription();
    description.ffn_conf.moe_configs->enable_eplb = true;
    EXPECT_FALSE(supportsFullPrefillCudaGraphMoe(description, ParallelismConfig{}, makeFullPrefillMoeRuntimeConfig()));
}

}  // namespace rtp_llm
