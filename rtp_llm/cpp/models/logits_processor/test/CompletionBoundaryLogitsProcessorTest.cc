#include "rtp_llm/cpp/models/logits_processor/CompletionBoundaryLogitsProcessor.h"

#include <gtest/gtest.h>

#include <vector>

namespace rtp_llm {
namespace {

SamplerInputs makeInputs(size_t vocab_size) {
    SamplerInputs inputs;
    inputs.logits     = torch::zeros({1, static_cast<int64_t>(vocab_size)}, torch::kFloat32);
    inputs.vocab_size = vocab_size;
    return inputs;
}

bool bitmaskAllows(const int32_t* row, int32_t token_id) {
    return (static_cast<uint32_t>(row[token_id / 32]) & (1u << (token_id % 32))) != 0u;
}

TEST(CompletionBoundaryLogitsProcessorTest, MasksOnlyStopsUntilFullBoundary) {
    CompletionBoundaryLogitsProcessor processor(
        {CompletionBoundaryState({7, 8, 9}, /*input_length=*/0, /*is_beam_search=*/false)}, {0, 6});

    auto inputs = makeInputs(16);
    processor.process(inputs, 0, 1);
    EXPECT_EQ(inputs.logits[0][0].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(inputs.logits[0][6].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(inputs.logits[0][7].item<float>(), 0.0f);
    EXPECT_EQ(inputs.logits[0][5].item<float>(), 0.0f);

    processor.updateStatus(torch::tensor({{7, 8}}, torch::kInt32), 2);
    inputs.logits.zero_();
    processor.process(inputs, 0, 1);
    EXPECT_EQ(inputs.logits[0][0].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(processor.boundaryStatus()[0], 2);

    processor.updateStatus(torch::tensor({{9}}, torch::kInt32), 1);
    inputs.logits.zero_();
    processor.process(inputs, 0, 1);
    EXPECT_EQ(inputs.logits[0][0].item<float>(), 0.0f);
    EXPECT_EQ(inputs.logits[0][6].item<float>(), 0.0f);
    EXPECT_EQ(processor.acceptedTokenLen(), 3);
}

TEST(CompletionBoundaryLogitsProcessorTest, UsesStreamingKmpForPartialBoundary) {
    CompletionBoundaryLogitsProcessor processor(
        {CompletionBoundaryState({7, 8, 7, 9}, 0, false)}, {0});

    processor.updateStatus(torch::tensor({{7, 8, 7, 8, 7, 9}}, torch::kInt32), 6);
    EXPECT_EQ(processor.boundaryStatus()[0], 4);

    auto inputs = makeInputs(16);
    processor.process(inputs, 0, 1);
    EXPECT_EQ(inputs.logits[0][0].item<float>(), 0.0f);
}

TEST(CompletionBoundaryLogitsProcessorTest, SpecMaskOpensAfterBoundaryInsideDraft) {
    CompletionBoundaryLogitsProcessor processor(
        {CompletionBoundaryState({7, 8}, 0, false)}, {0, 6});
    const int            propose_step = 3;
    const size_t         words = SpecLogitsProcessor::bitmaskWordCount(16);
    std::vector<int32_t> draft = {7, 8, 6};
    std::vector<int32_t> bitmask((propose_step + 1) * words, 0);

    SpecLogitsProcessorRequest request;
    request.draft_tokens       = draft.data();
    request.propose_step       = propose_step;
    request.bitmask_cpu_out    = bitmask.data();
    request.bitmask_size_int32 = words;
    request.vocab_size         = 16;

    EXPECT_EQ(processor.tryAcceptAndFillBitmask(request), propose_step);
    EXPECT_FALSE(bitmaskAllows(bitmask.data(), 0));
    EXPECT_FALSE(bitmaskAllows(bitmask.data() + words, 6));
    EXPECT_TRUE(bitmaskAllows(bitmask.data() + 2 * words, 6));
    EXPECT_TRUE(bitmaskAllows(bitmask.data() + 3 * words, 0));
    EXPECT_EQ(processor.boundaryStatus()[0], 0);
    EXPECT_EQ(processor.acceptedTokenLen(), 0);
}

TEST(CompletionBoundaryLogitsProcessorTest, SpecRejectsPrematureStopWithoutMutation) {
    CompletionBoundaryLogitsProcessor processor(
        {CompletionBoundaryState({7, 8}, 0, false)}, {0});
    const int            propose_step = 2;
    const size_t         words = SpecLogitsProcessor::bitmaskWordCount(16);
    std::vector<int32_t> draft = {7, 0};
    std::vector<int32_t> bitmask((propose_step + 1) * words, 0);

    SpecLogitsProcessorRequest request;
    request.draft_tokens       = draft.data();
    request.propose_step       = propose_step;
    request.bitmask_cpu_out    = bitmask.data();
    request.bitmask_size_int32 = words;
    request.vocab_size         = 16;

    EXPECT_EQ(processor.tryAcceptAndFillBitmask(request), 1);
    EXPECT_EQ(processor.boundaryStatus()[0], 0);
    EXPECT_EQ(processor.acceptedTokenLen(), 0);
}

}  // namespace
}  // namespace rtp_llm
