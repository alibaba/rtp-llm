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

TEST(SpeculativeSamplerTest, SelectsPerRequestProbabilityModeWithoutChangingRejectionInput) {
    SamplerOutput target;
    target.all_probs = torch::zeros({3, 2, 4}, torch::kFloat32);
    target.all_probs.select(2, 3).fill_(1.0f);
    target.original_all_probs = torch::full({3, 2, 4}, 0.25f, torch::kFloat32);

    auto mixed = SpeculativeSampler::targetResponseProbabilities(
        {ReturnAllProbsMode::DEFAULT, ReturnAllProbsMode::ORIGINAL, ReturnAllProbsMode::NONE}, target);
    EXPECT_TRUE(torch::equal(mixed.select(0, 0), target.all_probs.select(0, 0)));
    EXPECT_TRUE(torch::equal(mixed.select(0, 1), target.original_all_probs.select(0, 1)));
    EXPECT_TRUE(torch::equal(target.all_probs.select(0, 1),
                             torch::tensor({{0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f}})));

    auto original = SpeculativeSampler::targetResponseProbabilities(
        {ReturnAllProbsMode::ORIGINAL, ReturnAllProbsMode::ORIGINAL, ReturnAllProbsMode::NONE}, target);
    EXPECT_TRUE(torch::equal(original, target.original_all_probs));
    auto filtered = SpeculativeSampler::targetResponseProbabilities(
        {ReturnAllProbsMode::DEFAULT, ReturnAllProbsMode::DEFAULT, ReturnAllProbsMode::NONE}, target);
    EXPECT_TRUE(torch::equal(filtered, target.all_probs));
    EXPECT_FALSE(SpeculativeSampler::targetResponseProbabilities(
                     {ReturnAllProbsMode::NONE, ReturnAllProbsMode::NONE, ReturnAllProbsMode::NONE}, target)
                     .defined());
}

}  // namespace
}  // namespace speculative
}  // namespace rtp_llm
