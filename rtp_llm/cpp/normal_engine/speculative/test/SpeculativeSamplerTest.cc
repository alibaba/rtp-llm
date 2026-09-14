#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"

#include <gtest/gtest.h>
#include <torch/all.h>

namespace rtp_llm {
namespace speculative {
namespace {

TEST(FastTopKSamplerTest, TopKOneReturnsArgmaxIndex) {
    FastTopKSampler sampler;
    auto            logits = torch::tensor({{1.0f, 2.0f, 5.0f, 3.0f}});
    auto            out    = sampler.forward(logits, 1);

    ASSERT_EQ(out.token_ids.dim(), 2);
    ASSERT_EQ(out.token_ids.size(0), 1);
    ASSERT_EQ(out.token_ids.size(1), 1);
    EXPECT_EQ(out.token_ids[0][0].item<int64_t>(), 2);
    EXPECT_TRUE(torch::equal(out.all_probs, torch::tensor({{0.0f, 0.0f, 1.0f, 0.0f}})));
}

TEST(FastTopKSamplerTest, TopKGreaterThanOneReturnsTopKIndices) {
    FastTopKSampler sampler;
    auto            logits = torch::tensor({{1.0f, 2.0f, 5.0f, 3.0f}});
    auto            out    = sampler.forward(logits, 2);

    ASSERT_EQ(out.token_ids.dim(), 2);
    ASSERT_EQ(out.token_ids.size(0), 1);
    ASSERT_EQ(out.token_ids.size(1), 2);
    // softmax preserves ordering: indices 2 (5.0) then 3 (3.0).
    EXPECT_EQ(out.token_ids[0][0].item<int64_t>(), 2);
    EXPECT_EQ(out.token_ids[0][1].item<int64_t>(), 3);
}

TEST(DFlashSamplerTest, GreedyUsesExactPointMass) {
    GenerateConfig config;
    config.do_sample = false;
    auto logits      = torch::tensor({{1.0f, 2.0f, 5.0f, 3.0f}, {5.0f, 1.0f, 0.0f, -1.0f}});
    auto q           = dflashDraftProbabilities(logits, config);
    EXPECT_TRUE(torch::equal(q, torch::tensor({{0.0f, 0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f}})));
    config.do_sample   = true;
    config.temperature = 0.0f;
    EXPECT_TRUE(torch::equal(q, dflashDraftProbabilities(logits, config)));
}

TEST(DFlashSamplerTest, TemperatureTopKAndNucleusMatchReferenceDistribution) {
    // Before filtering the temperature-scaled scores represent [0.4,0.3,0.2,0.1].
    auto           logits = torch::log(torch::tensor({{0.4f, 0.3f, 0.2f, 0.1f}})) * 2.0;
    GenerateConfig config;
    config.temperature = 2.0f;
    config.top_k       = 3;
    config.top_p       = 0.6f;
    // Top-k normalization gives [4/9,3/9,2/9]; nucleus retains its crossing token.
    auto q = dflashDraftProbabilities(logits, config);
    EXPECT_TRUE(torch::allclose(q, torch::tensor({{4.0f / 7, 3.0f / 7, 0.0f, 0.0f}}), 1e-5, 1e-6));
    config.top_k = 0;
    config.top_p = 1.0f;
    EXPECT_TRUE(torch::allclose(
        dflashDraftProbabilities(logits, config), torch::tensor({{0.4f, 0.3f, 0.2f, 0.1f}}), 1e-5, 1e-6));
}

TEST(DFlashSamplerTest, TinyNucleusAlwaysRetainsLargestToken) {
    GenerateConfig config;
    config.top_p = 0.001f;
    EXPECT_TRUE(torch::equal(dflashDraftProbabilities(torch::tensor({{1.0f, 2.0f, 0.0f}}), config),
                             torch::tensor({{0.0f, 1.0f, 0.0f}})));
    config.top_p = 0.0f;
    EXPECT_THROW(dflashDraftProbabilities(torch::tensor({{1.0f, 2.0f}}), config), c10::Error);
}

TEST(DFlashConfigTest, TypeIsDistinctAndRoundTrips) {
    EXPECT_EQ(SpeculativeExecutionConfig::from_string("dflash"), SP_TYPE_DFLASH);
    EXPECT_EQ(SpeculativeExecutionConfig::to_string(SP_TYPE_DFLASH), "dflash");
    EXPECT_NE(SP_TYPE_DSPARK, SP_TYPE_DFLASH);
    EXPECT_TRUE(isBlockDraftType(SP_TYPE_DFLASH));
    EXPECT_TRUE(isBlockDraftType(SP_TYPE_DSPARK));
    EXPECT_FALSE(isBlockDraftType(SP_TYPE_MTP));
}

}  // namespace
}  // namespace speculative
}  // namespace rtp_llm
