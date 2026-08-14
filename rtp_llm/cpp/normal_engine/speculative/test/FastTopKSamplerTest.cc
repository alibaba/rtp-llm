#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"

#include <gtest/gtest.h>

namespace rtp_llm::speculative {

TEST(FastTopKSamplerTest, ReportsDistributionOfActualGreedyProposal) {
    const auto logits = torch::tensor({2.0f, 1.0f, 0.0f, -1.0f, 3.0f, 2.0f}).reshape({2, 3});

    FastTopKSampler sampler;
    const auto      output = sampler.forward(logits);

    const auto expected_token_ids = torch::tensor({0, 1}, torch::kInt64).reshape({2, 1});
    const auto expected_proposal_probs =
        torch::tensor({1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}).reshape({2, 3});

    EXPECT_TRUE(torch::equal(output.token_ids, expected_token_ids));
    EXPECT_TRUE(torch::equal(output.all_probs, expected_proposal_probs));
}

TEST(FastTopKSamplerTest, RejectsMultipleCandidatesWithoutAProposalSamplingRule) {
    const auto logits = torch::tensor({2.0f, 1.0f, 0.0f}).reshape({1, 3});

    FastTopKSampler sampler;
    EXPECT_ANY_THROW(sampler.forward(logits, 2));
}

}  // namespace rtp_llm::speculative
