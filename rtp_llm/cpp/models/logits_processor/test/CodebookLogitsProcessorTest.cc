#include "gtest/gtest.h"

#include <algorithm>
#include <limits>

#include "rtp_llm/cpp/models/logits_processor/CodebookLogitsProcessor.h"

namespace rtp_llm {
namespace {

SamplerInputs inputsWith(torch::Tensor logits) {
    SamplerInputs inputs;
    inputs.logits     = std::move(logits);
    inputs.batch_size = inputs.logits.size(0);
    inputs.vocab_size = inputs.logits.size(1);
    return inputs;
}

void expectAllowed(const torch::Tensor& logits, int64_t row, const std::vector<int64_t>& allowed) {
    auto cpu = logits.cpu();
    for (int64_t token = 0; token < cpu.size(1); ++token) {
        const bool should_be_finite = std::find(allowed.begin(), allowed.end(), token) != allowed.end();
        EXPECT_EQ(torch::isfinite(cpu[row][token]).item<bool>(), should_be_finite) << "token " << token;
        if (!should_be_finite) {
            EXPECT_TRUE(torch::isneginf(cpu[row][token]).item<bool>()) << "token " << token;
        }
    }
}

TEST(CodebookLogitsProcessorTest, MasksOnlySliceAndSupportsOverlappingGroups) {
    CodebookLogitsProcessor processor({{1, 3}, {3, 4}}, 6, 2);
    auto                    logits   = torch::arange(24, torch::kFloat32).reshape({4, 6});
    auto                    original = logits.clone();
    auto                    inputs   = inputsWith(logits);

    ASSERT_FALSE(processor.process(inputs, 1, 3).has_value());
    EXPECT_TRUE(torch::equal(logits[0], original[0]));
    EXPECT_TRUE(torch::equal(logits[3], original[3]));
    expectAllowed(logits, 1, {1, 3});
    expectAllowed(logits, 2, {1, 3});

    ASSERT_FALSE(processor.updateStatus(torch::zeros({2, 1}, torch::kInt32), 1).has_value());
    auto next = inputsWith(torch::zeros({2, 6}));
    ASSERT_FALSE(processor.process(next, 0, 2).has_value());
    expectAllowed(next.logits, 0, {3, 4});
    expectAllowed(next.logits, 1, {3, 4});
}

TEST(CodebookLogitsProcessorTest, PreservesFinishedRowsInsideBatchSlice) {
    CodebookLogitsProcessor processor({{1, 3}}, 5, 3);
    auto                    inputs   = inputsWith(torch::arange(25, torch::kFloat32).reshape({5, 5}));
    auto                    original = inputs.logits.clone();
    inputs.finished_mask             = torch::tensor({true, false, true, false, false}, torch::kBool);
    ASSERT_FALSE(processor.process(inputs, 1, 4).has_value());
    EXPECT_TRUE(torch::equal(inputs.logits[0], original[0]));
    EXPECT_TRUE(torch::equal(inputs.logits[2], original[2]));
    EXPECT_TRUE(torch::equal(inputs.logits[4], original[4]));
    expectAllowed(inputs.logits, 1, {1, 3});
    expectAllowed(inputs.logits, 3, {1, 3});
}

TEST(CodebookLogitsProcessorTest, ValidatesBoundsAndExactStageCount) {
    CodebookLogitsProcessor processor({{0}, {1}}, 3, 1);
    auto                    inputs = inputsWith(torch::zeros({2, 3}));
    EXPECT_TRUE(processor.process(inputs, 1, 1).has_value());
    EXPECT_TRUE(processor.process(inputs, 1, 3).has_value());
    EXPECT_TRUE(processor.updateStatus(torch::zeros({1, 2}, torch::kInt32), 3).has_value());
    EXPECT_EQ(*processor.committedOutputLen(), 0);
    EXPECT_FALSE(processor.updateStatus(torch::zeros({1, 2}, torch::kInt32), 2).has_value());
    EXPECT_EQ(*processor.committedOutputLen(), 2);
    EXPECT_TRUE(processor.process(inputs, 0, 1).has_value());
    EXPECT_TRUE(processor.updateStatus(torch::zeros({1, 1}, torch::kInt32), 1).has_value());
    EXPECT_TRUE(processor.isStateful());
    EXPECT_FALSE(processor.supportsNormalAsyncDeviceState());
}

TEST(CodebookLogitsProcessorTest, BeamExpansionAndReorderKeepOneSharedStage) {
    CodebookLogitsProcessor processor({{0, 1}, {2}}, 4, 1);
    processor.updateMultiSeqStatus({0, 0, 0});
    auto expanded = inputsWith(torch::zeros({3, 4}));
    ASSERT_FALSE(processor.process(expanded, 0, 3).has_value());
    for (int row = 0; row < 3; ++row) {
        expectAllowed(expanded.logits, row, {0, 1});
    }
    ASSERT_FALSE(processor.updateStatus(torch::tensor({{0}, {1}, {0}}, torch::kInt32), 1).has_value());
    EXPECT_EQ(*processor.committedOutputLen(), 1);
    processor.updateMultiSeqStatus({2, 0});

    auto inputs = inputsWith(torch::zeros({2, 4}));
    ASSERT_FALSE(processor.process(inputs, 0, 2).has_value());
    expectAllowed(inputs.logits, 0, {2});
    expectAllowed(inputs.logits, 1, {2});
    EXPECT_ANY_THROW(processor.updateMultiSeqStatus({2}));
}

TEST(CodebookLogitsProcessorTest, SeparateProcessorsCanMaskMixedStagesInOneBatch) {
    CodebookLogitsProcessor first({{0}, {1}}, 4, 1);
    CodebookLogitsProcessor second({{0}, {1}}, 4, 1);
    ASSERT_FALSE(second.updateStatus(torch::zeros({1, 1}, torch::kInt32), 1).has_value());
    auto inputs = inputsWith(torch::zeros({2, 4}));
    ASSERT_FALSE(first.process(inputs, 0, 1).has_value());
    ASSERT_FALSE(second.process(inputs, 1, 2).has_value());
    expectAllowed(inputs.logits, 0, {0});
    expectAllowed(inputs.logits, 1, {1});
}

TEST(CodebookLogitsProcessorTest, TwoStepBeamSearchMatchesCanonicalFullVocabReference) {
    constexpr int64_t                       canonical_vocab  = 10;
    constexpr int64_t                       compact_vocab    = 4;
    const auto                              union_ids        = torch::tensor({1, 3, 6, 8}, torch::kLong);
    const std::vector<std::vector<int64_t>> compact_groups   = {{0, 2}, {1, 3}};
    const std::vector<std::vector<int64_t>> canonical_groups = {{1, 6}, {3, 8}};

    auto compare_beam_step = [&](CodebookLogitsProcessor& processor,
                                 const torch::Tensor&     full_logits,
                                 const torch::Tensor&     cumulative_scores,
                                 size_t                   level,
                                 int64_t                  beam_width) {
        auto compact_logits = full_logits.index_select(1, union_ids);
        auto inputs         = inputsWith(compact_logits.clone());
        EXPECT_FALSE(processor.process(inputs, 0, full_logits.size(0)).has_value());

        auto canonical_mask = torch::ones({1, canonical_vocab}, torch::kBool);
        for (const auto token_id : canonical_groups[level]) {
            canonical_mask[0][token_id] = false;
        }
        auto masked_canonical = full_logits.clone();
        masked_canonical.masked_fill_(canonical_mask, -std::numeric_limits<float>::infinity());

        auto canonical_scores =
            (torch::log_softmax(masked_canonical, -1) + cumulative_scores.unsqueeze(1)).reshape({-1});
        auto compact_scores = (torch::log_softmax(inputs.logits, -1) + cumulative_scores.unsqueeze(1)).reshape({-1});
        auto canonical_top_indices = std::get<1>(torch::topk(canonical_scores, beam_width));
        auto compact_top_indices   = std::get<1>(torch::topk(compact_scores, beam_width));

        auto canonical_parents = torch::floor_divide(canonical_top_indices, canonical_vocab);
        auto canonical_tokens  = canonical_top_indices.remainder(canonical_vocab);
        auto compact_parents   = torch::floor_divide(compact_top_indices, compact_vocab);
        auto compact_tokens    = compact_top_indices.remainder(compact_vocab);
        auto remapped_tokens   = union_ids.index_select(0, compact_tokens);

        EXPECT_TRUE(torch::equal(compact_parents, canonical_parents));
        EXPECT_TRUE(torch::equal(remapped_tokens, canonical_tokens));
        EXPECT_TRUE(torch::allclose(std::get<0>(torch::topk(compact_scores, beam_width)),
                                    std::get<0>(torch::topk(canonical_scores, beam_width))));
        return compact_parents;
    };

    CodebookLogitsProcessor processor(compact_groups, compact_vocab, 2);
    auto                    first_logits = torch::tensor(
        {{0.2, 2.4, -0.3, 0.5, 1.0, -0.7, 1.8, -0.2, 0.9, 0.1}, {1.1, 0.7, -0.4, 1.5, 0.3, 0.2, 2.1, -0.8, 1.2, -0.1}});
    auto first_parents = compare_beam_step(processor, first_logits, torch::tensor({0.4, -0.2}), 0, 3);

    ASSERT_FALSE(processor.updateStatus(torch::zeros({3, 1}, torch::kInt32), 1).has_value());
    std::vector<int> parent_mapping(first_parents.numel());
    auto             first_parents_cpu = first_parents.to(torch::kCPU);
    for (int64_t i = 0; i < first_parents_cpu.numel(); ++i) {
        parent_mapping[i] = first_parents_cpu[i].item<int64_t>();
    }
    processor.updateMultiSeqStatus(parent_mapping);

    auto second_logits_by_parent = torch::tensor(
        {{0.5, 1.4, -0.2, 2.3, 0.1, -0.5, 0.7, 1.0, 1.9, -0.8}, {-0.4, 0.8, 0.3, 1.7, -0.1, 0.6, 1.2, -0.7, 2.5, 0.2}});
    auto second_logits = second_logits_by_parent.index_select(0, first_parents);
    compare_beam_step(processor, second_logits, torch::tensor({-0.1, 0.6, 0.2}), 1, 2);
}

TEST(CodebookLogitsProcessorTest, MasksCudaLogitsWithCachedDeviceMask) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is unavailable";
    }
    CodebookLogitsProcessor processor({{1, 4}}, 6, 2);
    auto                    inputs = inputsWith(torch::randn({2, 6}, torch::TensorOptions().device(torch::kCUDA)));
    ASSERT_FALSE(processor.process(inputs, 0, 2).has_value());
    expectAllowed(inputs.logits, 0, {1, 4});

    auto second = inputsWith(torch::randn({2, 6}, torch::TensorOptions().device(torch::kCUDA)));
    ASSERT_FALSE(processor.process(second, 0, 2).has_value());
    expectAllowed(second.logits, 1, {1, 4});
}

TEST(CodebookLogitsProcessorTest, SharedMasksKeepIndependentStagesAndRemainImmutable) {
    for (const auto device : {torch::kCPU, torch::kCUDA}) {
        if (device == torch::kCUDA && !torch::cuda::is_available()) {
            continue;
        }
        auto                    masks    = CodebookLogitsProcessor::createMasks({{1, 3}, {2, 4}}, 6).to(device);
        auto                    original = masks.clone();
        CodebookLogitsProcessor first(masks, 1);
        CodebookLogitsProcessor second(masks, 1);
        ASSERT_FALSE(first.updateStatus(torch::zeros({1, 1}, torch::kInt32), 1).has_value());
        auto inputs = inputsWith(torch::zeros({2, 6}, torch::TensorOptions().device(device)));
        ASSERT_FALSE(first.process(inputs, 0, 1).has_value());
        ASSERT_FALSE(second.process(inputs, 1, 2).has_value());
        expectAllowed(inputs.logits, 0, {2, 4});
        expectAllowed(inputs.logits, 1, {1, 3});
        EXPECT_EQ(first.masks_.data_ptr(), masks.data_ptr());
        EXPECT_EQ(second.masks_.data_ptr(), masks.data_ptr());
        EXPECT_TRUE(torch::equal(masks, original));
        masks = torch::Tensor();  // Request references keep the mask alive.
        inputs.logits.zero_();
        ASSERT_FALSE(second.process(inputs, 1, 2).has_value());
        expectAllowed(inputs.logits, 1, {1, 3});
    }
}

}  // namespace
}  // namespace rtp_llm
