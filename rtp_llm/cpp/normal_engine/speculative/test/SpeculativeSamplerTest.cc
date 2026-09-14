#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"

#include <limits>

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
    // Admission passes top_p = 0 through unchanged (OpenAI passthrough), so
    // it normalizes to "no nucleus cut" (top_p = 1.0) instead of throwing
    // inside the engine loop thread.
    config.top_p = 0.0f;
    auto plain = torch::tensor({{1.0f, 2.0f}});
    EXPECT_TRUE(torch::allclose(dflashDraftProbabilities(plain, config), torch::softmax(plain, -1), 1e-6, 1e-6));
}

TEST(DFlashSamplerTest, NormalizesAdmissionTolerantSamplingParameters) {
    // Admission lets unusable sampling parameters through: the repo's own
    // smoke load sends {temperature: 0, top_p: 0}, the OpenAI endpoint keeps
    // top_p = 0 / > 1 unchanged and is_positive_number accepts inf
    // temperatures. The engine loop thread must normalize all of them instead
    // of throwing; see dflashDraftProbabilities.
    auto           logits = torch::tensor({{0.0f, 1.0f, 2.0f, 3.0f}});
    GenerateConfig config;
    // Smoke-load shape: a greedy request with top_p = 0 stays an exact point mass.
    config.do_sample   = false;
    config.temperature = 0.0f;
    config.top_p       = 0.0f;
    config.top_k       = 1;
    EXPECT_TRUE(torch::equal(dflashDraftProbabilities(logits, config), torch::tensor({{0.0f, 0.0f, 0.0f, 1.0f}})));
    // Stochastic with a non-finite temperature, negative top_k and top_p > 1:
    // everything falls back to the plain softmax distribution (1.0 / 0 / 1.0).
    config.do_sample   = true;
    config.temperature = std::numeric_limits<float>::infinity();
    config.top_k       = -3;
    config.top_p       = 2.0f;
    EXPECT_TRUE(
        torch::allclose(dflashDraftProbabilities(logits, config), torch::softmax(logits, -1), 1e-6, 1e-6));
    // A denormal temperature survives the finite/positive checks but would
    // overflow scores/temperature to inf and softmax into NaN; the 1e-6 floor
    // (same as the DSpARK sanitize) keeps the distribution finite and argmax-like.
    config.temperature = std::numeric_limits<float>::denorm_min();
    config.top_k       = 0;
    config.top_p       = 1.0f;
    auto floored       = dflashDraftProbabilities(logits, config);
    EXPECT_TRUE(floored.isfinite().all().item<bool>());
    EXPECT_TRUE(torch::equal(floored, torch::tensor({{0.0f, 0.0f, 0.0f, 1.0f}})));
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
