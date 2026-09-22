#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"

#include <gtest/gtest.h>
#include <torch/all.h>

namespace rtp_llm::speculative {
namespace {

TEST(DSparkSamplerTest, AppliesMarkovBiasAutoregressivelyAndMapsDraftTokens) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is required";
    }

    constexpr int64_t batch_size   = 1;
    constexpr int64_t propose_step = 3;
    constexpr int64_t target_vocab = 6;
    constexpr int64_t draft_vocab  = 4;
    constexpr int64_t rank         = 2;
    auto              options      = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);

    // draft ids [0,1,2,3] map to target ids [1,3,4,5].  The transition
    // sequence is anchor target 0 -> draft 1(target 3) -> draft 2(target 4)
    // -> draft 3(target 5).  Each next Markov lookup must therefore consume
    // the mapped target id, not the previous draft id.
    auto d2t = torch::tensor({1, 3, 4, 5}, torch::TensorOptions().dtype(torch::kInt64)).cuda();
    auto w1  = torch::zeros({target_vocab, rank}, options);
    w1.index_put_({0, 0}, 1.0f);
    w1.index_put_({3, 0}, 2.0f);
    w1.index_put_({4, 0}, 3.0f);
    auto w2 = torch::tensor({{0.0f, 0.0f}, {10.0f, 0.0f}, {20.0f, 0.0f}, {30.0f, 0.0f}}, options);

    // Cancel the larger later columns at each step so that the expected
    // transition is unique after adding W1[token] @ W2.T.
    auto base_logits = torch::tensor(
        {{0.0f, 0.0f, -100.0f, -100.0f}, {0.0f, -100.0f, -10.0f, -100.0f}, {0.0f, -100.0f, -100.0f, -20.0f}}, options);
    auto anchors     = torch::tensor({0}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));
    auto temperature = torch::full({batch_size}, 1.0e-6f, options);

    SpeculativeSampler sampler(d2t, propose_step, DraftProposalMode::LEGACY);
    auto               output = sampler.sampleDSparkDraft(base_logits, anchors, temperature, w1, w2, draft_vocab);

    ASSERT_EQ(output.token_ids.sizes(), torch::IntArrayRef({batch_size, propose_step}));
    EXPECT_TRUE(torch::equal(output.token_ids.cpu(), torch::tensor({{3, 4, 5}}, torch::kInt32)));
    ASSERT_EQ(output.all_probs.sizes(), torch::IntArrayRef({batch_size, propose_step, draft_vocab}));
    EXPECT_TRUE(torch::allclose(output.all_probs.sum(-1).cpu(), torch::ones({batch_size, propose_step})));
}

}  // namespace
}  // namespace rtp_llm::speculative
