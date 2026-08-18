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

    // The declared proposal distribution is the one that actually selected the token,
    // not the full softmax over the draft vocabulary.
    const auto expected_probs = torch::tensor({{0.0f, 0.0f, 1.0f, 0.0f}});
    EXPECT_TRUE(torch::equal(out.all_probs, expected_probs));
}

TEST(FastTopKSamplerTest, ReportsOneHotDistributionForEveryBatchRow) {
    FastTopKSampler sampler;
    auto            logits = torch::tensor({2.0f, 1.0f, 0.0f, -1.0f, 3.0f, 2.0f}).reshape({2, 3});
    auto            out    = sampler.forward(logits);

    const auto expected_token_ids = torch::tensor({0, 1}, torch::kInt64).reshape({2, 1});
    const auto expected_probs     = torch::tensor({1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}).reshape({2, 3});

    EXPECT_TRUE(torch::equal(out.token_ids, expected_token_ids));
    EXPECT_TRUE(torch::equal(out.all_probs, expected_probs));
}

TEST(FastTopKSamplerTest, OneHotUsesDraftVocabIndexBeforeTargetMapping) {
    // d2t_map rewrites token_ids into target-vocabulary ids, but all_probs stays indexed in
    // draft-vocabulary space. Scattering after the mapping would either write the wrong column
    // or index out of bounds.
    const auto      d2t_map = torch::tensor({70, 80, 90}, torch::kInt64);
    FastTopKSampler sampler(d2t_map);

    auto logits = torch::tensor({0.0f, 5.0f, 1.0f}).reshape({1, 3});
    auto out    = sampler.forward(logits);

    EXPECT_EQ(out.token_ids[0][0].item<int64_t>(), 80);

    const auto expected_probs = torch::tensor({0.0f, 1.0f, 0.0f}).reshape({1, 3});
    EXPECT_TRUE(torch::equal(out.all_probs, expected_probs));
}

TEST(FastTopKSamplerTest, RejectsTopKGreaterThanOne) {
    // Multiple candidates have no declared proposal-sampling rule, so they must fail fast
    // instead of silently claiming a full softmax q for an argmax-selected token.
    FastTopKSampler sampler;
    auto            logits = torch::tensor({{1.0f, 2.0f, 5.0f, 3.0f}});

    EXPECT_ANY_THROW(sampler.forward(logits, 2));
}

// F-A1: FastTopKSampler::forward leaves the one-hot proposal probabilities in draft-vocabulary
// space and rewrites token_ids into target-vocabulary ids. Rejection sampling then reads the
// proposal probability q at the sampled target id, so whenever a d2t map is defined the proposal
// probs must be scattered into target space via the same map. If that remap is skipped, q at the
// sampled id is a spurious 0.0 and the acceptance test u * p < q fires unconditionally. These
// tests exercise the product remap itself, so reverting it turns them red.
TEST(DraftProbsNeedTargetVocabRemapTest, HoldsForEqualWidthMapThatTheOldWidthGuardSkipped) {
    // The bug: the decision used to be "draft vocab width != target vocab width", so a defined map
    // over equal widths was never remapped. The predicate must depend on the map alone.
    const auto equal_width_map = torch::tensor({3, 2, 1, 0}, torch::kInt64);
    EXPECT_TRUE(draftProbsNeedTargetVocabRemap(equal_width_map));

    const auto different_width_map = torch::tensor({4, 2, 0}, torch::kInt64);
    EXPECT_TRUE(draftProbsNeedTargetVocabRemap(different_width_map));
}

TEST(DraftProbsNeedTargetVocabRemapTest, FailsClosedWithoutAUsableMap) {
    EXPECT_FALSE(draftProbsNeedTargetVocabRemap(torch::Tensor()));
    // mappingDraft2Target treats a defined but empty map as "no mapping"; disagreeing here would
    // send an empty index tensor into index_put_ and throw on a path that used to be skipped.
    EXPECT_FALSE(draftProbsNeedTargetVocabRemap(torch::empty({0}, torch::kInt64)));
}

TEST(RemapDraftProbsToTargetVocabTest, RemapsProposalIntoTargetVocabForEqualWidthD2tMap) {
    // Non-identity permutation with equal draft/target vocab width (4 == 4): this is exactly the
    // case the old vocab-width guard failed to remap.
    const auto d2t_map = torch::tensor({3, 2, 1, 0}, torch::kInt64);

    // Proposal one-hot in draft space: draft id 1 was sampled -> target id d2t_map[1] == 2.
    const auto    draft_probs       = torch::tensor({0.0f, 1.0f, 0.0f, 0.0f}).reshape({1, 1, 4});
    const int64_t sampled_target_id = d2t_map[1].item<int64_t>();
    ASSERT_EQ(sampled_target_id, 2);

    // Unremapped, q read at the sampled target id would be 0.0 -> unconditional accept.
    ASSERT_EQ(draft_probs[0][0][sampled_target_id].item<float>(), 0.0f);

    torch::Tensor buffer;
    const auto    remapped = remapDraftProbsToTargetVocab(draft_probs, d2t_map, 1, 4, buffer);

    EXPECT_EQ(remapped[0][0][sampled_target_id].item<float>(), 1.0f);
    const auto expected = torch::tensor({0.0f, 0.0f, 1.0f, 0.0f}).reshape({1, 1, 4});
    EXPECT_TRUE(torch::equal(remapped, expected));
}

TEST(RemapDraftProbsToTargetVocabTest, RemapsProposalIntoTargetVocabForDifferentWidthD2tMap) {
    // Different width (draft vocab 3 < target vocab 5): the one-hot must land on the mapped target
    // column and stay zero elsewhere, yielding a non-degenerate q == 1.0 at the sampled id.
    const auto d2t_map = torch::tensor({4, 2, 0}, torch::kInt64);

    const auto    draft_probs       = torch::tensor({0.0f, 1.0f, 0.0f}).reshape({1, 1, 3});
    const int64_t sampled_target_id = d2t_map[1].item<int64_t>();
    ASSERT_EQ(sampled_target_id, 2);

    torch::Tensor buffer;
    const auto    remapped = remapDraftProbsToTargetVocab(draft_probs, d2t_map, 1, 5, buffer);

    EXPECT_EQ(remapped[0][0][sampled_target_id].item<float>(), 1.0f);
    const auto expected = torch::tensor({0.0f, 0.0f, 1.0f, 0.0f, 0.0f}).reshape({1, 1, 5});
    EXPECT_TRUE(torch::equal(remapped, expected));
}

TEST(RemapDraftProbsToTargetVocabTest, KeepsRowsAndStepsIndependent) {
    // Production runs GEN_NUM_PER_CIRCLE=3 speculative steps over a batch, so per-row and per-step
    // offsets must not collapse: a single-row single-step fixture cannot catch a stride mistake.
    const auto d2t_map = torch::tensor({3, 2, 1, 0}, torch::kInt64);

    auto draft_probs = torch::zeros({2, 3, 4});
    draft_probs.index_put_({0, 0, 1}, 1.0f);
    draft_probs.index_put_({0, 1, 3}, 1.0f);
    draft_probs.index_put_({0, 2, 0}, 1.0f);
    draft_probs.index_put_({1, 0, 2}, 1.0f);
    draft_probs.index_put_({1, 1, 1}, 1.0f);
    draft_probs.index_put_({1, 2, 3}, 1.0f);

    torch::Tensor buffer;
    const auto    remapped = remapDraftProbsToTargetVocab(draft_probs, d2t_map, 2, 4, buffer);

    auto expected = torch::zeros({2, 3, 4});
    expected.index_put_({0, 0, 2}, 1.0f);
    expected.index_put_({0, 1, 0}, 1.0f);
    expected.index_put_({0, 2, 3}, 1.0f);
    expected.index_put_({1, 0, 1}, 1.0f);
    expected.index_put_({1, 1, 2}, 1.0f);
    expected.index_put_({1, 2, 0}, 1.0f);

    EXPECT_TRUE(torch::equal(remapped, expected));
}

TEST(RemapDraftProbsToTargetVocabTest, ReuseAcrossCallsLeavesNoStaleMass) {
    // The buffer is reused across decode steps, so a later smaller batch must not read probability
    // mass left over from an earlier larger one.
    const auto    d2t_map = torch::tensor({1, 0}, torch::kInt64);
    torch::Tensor buffer;

    auto first = torch::zeros({2, 1, 2});
    first.index_put_({0, 0, 0}, 1.0f);
    first.index_put_({1, 0, 1}, 1.0f);
    remapDraftProbsToTargetVocab(first, d2t_map, 2, 2, buffer);
    ASSERT_TRUE(buffer.defined());

    auto second = torch::zeros({1, 1, 2});
    second.index_put_({0, 0, 1}, 1.0f);
    const auto remapped = remapDraftProbsToTargetVocab(second, d2t_map, 1, 2, buffer);

    const auto expected = torch::tensor({1.0f, 0.0f}).reshape({1, 1, 2});
    EXPECT_TRUE(torch::equal(remapped, expected));
    EXPECT_EQ(remapped.sum().item<float>(), 1.0f);
}

TEST(RemapDraftProbsToTargetVocabTest, GrowsBufferWithoutShrinkingIt) {
    const auto    d2t_map = torch::tensor({1, 0}, torch::kInt64);
    torch::Tensor buffer;

    auto small = torch::zeros({1, 1, 2});
    small.index_put_({0, 0, 0}, 1.0f);
    remapDraftProbsToTargetVocab(small, d2t_map, 1, 2, buffer);
    const auto grown_once = buffer.sizes().vec();

    auto large = torch::zeros({3, 2, 2});
    large.index_put_({2, 1, 0}, 1.0f);
    remapDraftProbsToTargetVocab(large, d2t_map, 3, 2, buffer);
    EXPECT_GE(buffer.size(0), 3);
    EXPECT_GE(buffer.size(1), 2);

    // Going back down must not shrink the buffer, otherwise every step reallocates.
    remapDraftProbsToTargetVocab(small, d2t_map, 1, 2, buffer);
    EXPECT_GE(buffer.size(0), 3);
    EXPECT_GE(buffer.size(1), 2);
    EXPECT_GE(buffer.size(0), grown_once[0]);
}

}  // namespace
}  // namespace speculative
}  // namespace rtp_llm
