#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"

#include <gtest/gtest.h>
#include <torch/all.h>
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {
namespace speculative {
namespace {

TEST(FastTopKSamplerTest, TopKOneReturnsArgmaxWithPointMassProbability) {
    FastTopKSampler sampler;
    auto            logits = torch::tensor({{1.0f, 2.0f, 5.0f, 3.0f}});
    auto            out    = sampler.forward(logits, 1);

    ASSERT_EQ(out.token_ids.dim(), 2);
    ASSERT_EQ(out.token_ids.size(0), 1);
    ASSERT_EQ(out.token_ids.size(1), 1);
    EXPECT_EQ(out.token_ids[0][0].item<int64_t>(), 2);
    EXPECT_TRUE(torch::equal(out.all_probs, torch::tensor({{0.f, 0.f, 1.f, 0.f}})));
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

TEST(PointMassVerifierTest, FixedCandidatesUseTargetAcceptanceAndResidualDistribution) {
    constexpr int64_t batch = 7;
    constexpr int64_t width = 4;
    const auto i32 = torch::TensorOptions(torch::kInt32).device(torch::kCUDA);
    const auto f32 = torch::TensorOptions(torch::kFloat32).device(torch::kCUDA);
    auto drafts = torch::ones({batch, width - 1}, i32);
    auto targets = torch::tensor({1, 1, 2, 3, 2, 1, 1, 3, 1, 1, 1, 3,
                                  1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 3}, i32).reshape({batch * width, 1});
    auto probs = torch::zeros({batch, width, 4}, f32);
    probs.select(2, 1).fill_(1);
    auto uniforms = torch::full({batch, width}, 0.1, f32);
    /** Q(candidate)=1. P(candidate)=.25, so u=.1 accepts and u=.5 rejects. */
    for (int64_t row : {3, 4, 5}) {
        probs[row][0].copy_(torch::tensor({0.f, .25f, .50f, .25f}, f32));
    }
    uniforms[4][0] = .5;
    uniforms[5][0] = .5;
    uniforms[4][1] = .5;  // Residual CDF: token 2 has mass 2/3.
    uniforms[5][1] = .9;  // Token 3 has the remaining 1/3.
    probs[6][0].copy_(torch::tensor({0.f, 0.f, 1.f, 0.f}, f32));  // Processor-masked candidate.
    auto stochastic = torch::tensor({false, false, false, true, true, true, true},
                                     torch::TensorOptions(torch::kBool).device(torch::kCUDA));
    auto output = torch::full({batch, width}, -7, i32);
    auto lengths = torch::zeros({batch}, i32);
    execRejectionSampling({{}, drafts, uniforms, probs, targets, output, lengths, stochastic, true});
    EXPECT_TRUE(torch::equal(lengths.cpu(), torch::tensor({3, 1, 4, 4, 1, 1, 1}, torch::kInt32)));
    const std::vector<std::vector<int32_t>> expected{
        {1, 1, 2}, {2}, {1, 1, 1, 3}, {1, 1, 1, 3}, {2}, {3}, {2}};
    const auto cpu = output.cpu();
    for (int64_t row = 0; row < batch; ++row) {
        EXPECT_TRUE(torch::equal(cpu[row].narrow(0, 0, expected[row].size()),
                                 torch::tensor(expected[row], torch::kInt32))) << "row=" << row;
    }
}

TEST(PointMassVerifierTest, ExplicitForceAcceptKeepsItsOverride) {
    SpeculativeSampler sampler({}, 3);
    SamplerOutput draft;
    draft.token_ids = torch::ones({1, 3}, torch::kInt32);
    draft.token_ids_are_point_mass = true;
    SamplerOutput target;
    target.token_ids = torch::full({4, 1}, 2, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
    target.all_probs = torch::zeros({1, 4, 4}, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));
    target.all_probs.select(2, 2).fill_(1);
    SpeculativeSamplingParams params;
    params.do_sample = torch::zeros({1}, torch::kBool);
    params.force_accept = torch::ones({1}, torch::kBool);
    const auto result = sampler.forward(params, draft, target);
    EXPECT_EQ(result.accept_len.item<int32_t>(), 4);
    EXPECT_TRUE(torch::equal(result.accept_tokens.cpu(), torch::tensor({{1, 1, 1, 2}}, torch::kInt32)));
}

}  // namespace
}  // namespace speculative
}  // namespace rtp_llm
