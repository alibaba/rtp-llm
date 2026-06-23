#include "rtp_llm/cpp/config/ModelConfig.h"

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

ModelConfig makeModelConfig() {
    ModelConfig model_config;
    model_config.attn_config.head_num      = 1;
    model_config.attn_config.kv_head_num   = 1;
    model_config.attn_config.size_per_head = 1;
    model_config.attn_config.q_lora_rank   = 0;
    model_config.attn_config.kv_lora_rank  = 0;
    model_config.attn_config.nope_head_dim = 0;
    model_config.attn_config.rope_head_dim = 0;
    model_config.attn_config.v_head_dim    = 0;
    return model_config;
}

struct BiasFusionCase {
    bool      qk_norm;
    bool      use_kvcache;
    RopeStyle rope_style;
    bool      expected_fusion;
};

class ModelConfigBiasFusionTest: public ::testing::TestWithParam<BiasFusionCase> {};

TEST_P(ModelConfigBiasFusionTest, SelectsFusionWithoutChangingRopeKvCache) {
    const auto& test_case                      = GetParam();
    auto        model_config                   = makeModelConfig();
    model_config.qk_norm                       = test_case.qk_norm;
    model_config.use_kvcache                   = test_case.use_kvcache;
    model_config.attn_config.rope_config.style = test_case.rope_style;

    const auto attention_config = model_config.getAttentionConfigs(1);

    EXPECT_EQ(attention_config.fuse_qkv_add_bias, test_case.expected_fusion);
    EXPECT_TRUE(attention_config.need_rope_kv_cache);
}

INSTANTIATE_TEST_SUITE_P(AttentionModes,
                         ModelConfigBiasFusionTest,
                         ::testing::Values(BiasFusionCase{false, false, RopeStyle::No, false},
                                           BiasFusionCase{false, true, RopeStyle::No, true},
                                           BiasFusionCase{false, false, RopeStyle::Base, true},
                                           BiasFusionCase{true, true, RopeStyle::Base, false},
                                           BiasFusionCase{true, false, RopeStyle::No, false}));

TEST(ModelConfigTest, BertEmbeddingAttentionPreservesDisabledRopeKvCache) {
    auto model_config                           = makeModelConfig();
    model_config.use_kvcache                    = false;
    model_config.attn_config.rope_config.style  = RopeStyle::No;
    model_config.attn_config.need_rope_kv_cache = false;

    const auto attention_config = model_config.getAttentionConfigs(1);

    EXPECT_FALSE(attention_config.need_rope_kv_cache);
}

TEST(ModelConfigTest, PartitionsKvHeadsForDivisibleAndGcdTensorParallelSizes) {
    auto model_config                    = makeModelConfig();
    model_config.attn_config.head_num    = 12;
    model_config.attn_config.kv_head_num = 6;

    const auto tp2 = model_config.getAttentionConfigs(2);
    EXPECT_EQ(tp2.head_num, 6);
    EXPECT_EQ(tp2.kv_head_num, 3);

    const auto tp3 = model_config.getAttentionConfigs(3);
    EXPECT_EQ(tp3.head_num, 4);
    EXPECT_EQ(tp3.kv_head_num, 2);

    model_config.attn_config.kv_head_num = 4;
    const auto gcd_tp3                   = model_config.getAttentionConfigs(3);
    EXPECT_EQ(gcd_tp3.head_num, 4);
    EXPECT_EQ(gcd_tp3.kv_head_num, 4);
}

TEST(ModelConfigTest, RejectsHeadCountsNotDivisibleByKvHeads) {
    auto model_config                    = makeModelConfig();
    model_config.attn_config.head_num    = 10;
    model_config.attn_config.kv_head_num = 4;

    EXPECT_THROW(model_config.getAttentionConfigs(2), std::runtime_error);
}

TEST(ModelConfigTest, UsesTokensPerBlockAsKernelFallback) {
    auto model_config                                = makeModelConfig();
    model_config.attn_config.tokens_per_block        = 32;
    model_config.attn_config.kernel_tokens_per_block = 0;

    EXPECT_EQ(model_config.getAttentionConfigs(1).kernel_tokens_per_block, 32);
}

}  // namespace
}  // namespace rtp_llm
