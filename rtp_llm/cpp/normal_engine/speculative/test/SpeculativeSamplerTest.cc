#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"
#include "rtp_llm/cpp/config/StaticConfig.h"
#include "rtp_llm/cpp/utils/Exception.h"

#include <gtest/gtest.h>
#include <torch/all.h>

namespace rtp_llm {
namespace speculative {
namespace {

class FastTopKSamplerValidationTest: public ::testing::Test {
protected:
    void SetUp() override {
        old_core_dump_on_exception_                  = StaticConfig::user_ft_core_dump_on_exception;
        StaticConfig::user_ft_core_dump_on_exception = false;
    }

    void TearDown() override {
        StaticConfig::user_ft_core_dump_on_exception = old_core_dump_on_exception_;
    }

private:
    bool old_core_dump_on_exception_{false};
};

TEST(FastTopKSamplerTest, TopKOneReturnsMappedArgmaxAndExplicitPointMass) {
    auto            d2t_map = torch::tensor({0, 1, 3, 2}, torch::kInt64);
    FastTopKSampler sampler(d2t_map, 5);
    auto            logits   = torch::tensor({{1.0f, 2.0f, 5.0f, 3.0f}, {4.0f, 3.0f, 2.0f, 5.0f}});
    auto            expected = torch::tensor({{0.0f, 0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 1.0f}});
    auto            out      = sampler.forward(logits, 1);

    ASSERT_EQ(out.token_ids.dim(), 2);
    ASSERT_EQ(out.token_ids.size(0), 2);
    ASSERT_EQ(out.token_ids.size(1), 1);
    EXPECT_EQ(out.token_ids[0][0].item<int64_t>(), 3);
    EXPECT_EQ(out.token_ids[1][0].item<int64_t>(), 2);
    ASSERT_EQ(out.all_probs.dim(), 2);
    ASSERT_EQ(out.all_probs.size(0), 2);
    ASSERT_EQ(out.all_probs.size(1), 4);
    EXPECT_TRUE(torch::allclose(out.all_probs, expected));
}

TEST(SpeculativeSamplerTest, MapsProposalDistributionWithManyToOneVocabMap) {
    auto draft_probs = torch::tensor({{{0.1f, 0.2f, 0.3f, 0.4f}}, {{0.4f, 0.3f, 0.2f, 0.1f}}});
    auto d2t_map     = torch::tensor({0, 1, 1, 2}, torch::kInt64);
    auto expected    = torch::tensor({{{0.1f, 0.5f, 0.4f}}, {{0.4f, 0.5f, 0.1f}}});

    auto mapped = SpeculativeSampler::mapDraftProbsToTarget(draft_probs, d2t_map, 3);

    EXPECT_TRUE(torch::allclose(mapped, expected));
    EXPECT_TRUE(torch::allclose(mapped.sum(-1), draft_probs.sum(-1)));
}

TEST(SpeculativeSamplerTest, MapsEqualWidthNonIdentityVocabMap) {
    auto draft_probs = torch::tensor({{{0.1f, 0.2f, 0.3f, 0.4f}}});
    auto d2t_map     = torch::tensor({0, 2, 1, 3}, torch::kInt64);
    auto expected    = torch::tensor({{{0.1f, 0.3f, 0.2f, 0.4f}}});

    auto mapped = SpeculativeSampler::mapDraftProbsToTarget(draft_probs, d2t_map, 4);

    EXPECT_TRUE(torch::allclose(mapped, expected));
}

TEST(SpeculativeSamplerTest, ReusesLargerTargetProbabilityBuffer) {
    auto draft_probs = torch::tensor({{{0.1f, 0.2f, 0.3f, 0.4f}}});
    auto d2t_map     = torch::tensor({0, 1, 1, 2}, torch::kInt64);
    auto buffer      = torch::full({3, 1, 3}, -1.0f);

    auto mapped = SpeculativeSampler::mapDraftProbsToTarget(draft_probs, d2t_map, 3, &buffer);

    EXPECT_EQ(mapped.data_ptr<float>(), buffer.data_ptr<float>());
    ASSERT_EQ(mapped.dim(), 3);
    EXPECT_EQ(mapped.size(0), 1);
    EXPECT_EQ(mapped.size(1), 1);
    EXPECT_EQ(mapped.size(2), 3);
    EXPECT_TRUE(torch::allclose(mapped, torch::tensor({{{0.1f, 0.5f, 0.4f}}})));
}

TEST(SpeculativeSamplerTest, ReallocatesForDifferentStepCountToKeepOutputContiguous) {
    auto draft_probs = torch::tensor({{{0.1f, 0.2f, 0.3f, 0.4f}}, {{0.4f, 0.3f, 0.2f, 0.1f}}});
    auto d2t_map     = torch::tensor({0, 1, 1, 2}, torch::kInt64);
    auto buffer      = torch::full({3, 2, 3}, -1.0f);

    auto mapped = SpeculativeSampler::mapDraftProbsToTarget(draft_probs, d2t_map, 3, &buffer);

    ASSERT_EQ(mapped.dim(), 3);
    EXPECT_EQ(mapped.size(0), 2);
    EXPECT_EQ(mapped.size(1), 1);
    EXPECT_EQ(mapped.size(2), 3);
    EXPECT_TRUE(mapped.is_contiguous());
    EXPECT_TRUE(torch::allclose(mapped, torch::tensor({{{0.1f, 0.5f, 0.4f}}, {{0.4f, 0.5f, 0.1f}}})));
}

TEST_F(FastTopKSamplerValidationTest, TopKGreaterThanOneIsRejected) {
    FastTopKSampler sampler;
    auto            logits = torch::tensor({{1.0f, 2.0f, 5.0f, 3.0f}});

    EXPECT_THROW(sampler.forward(logits, 2), RTPException);
}

TEST_F(FastTopKSamplerValidationTest, VocabMismatchWithoutMapIsRejected) {
    FastTopKSampler sampler(torch::Tensor(), 5);
    auto            logits = torch::tensor({{1.0f, 2.0f, 5.0f, 3.0f}});

    EXPECT_THROW(sampler.forward(logits, 1), RTPException);
}

}  // namespace
}  // namespace speculative
}  // namespace rtp_llm
