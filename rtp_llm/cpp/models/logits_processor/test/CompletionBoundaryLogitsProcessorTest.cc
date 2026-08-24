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

CompletionBoundarySpec makeK3CompletionSpec(bool starts_in_think) {
    CompletionBoundarySpec spec;
    spec.think_close_token_ids    = {10, 11};
    spec.response_open_token_ids  = {12, 13};
    spec.response_close_token_ids = {14, 15};
    spec.tools_open_token_ids     = {16, 17};
    spec.tools_close_token_ids    = {18, 19};
    spec.message_close_token_ids  = {20, 21};
    spec.whitespace_token_ids     = {2};
    spec.starts_in_think          = starts_in_think;
    return spec;
}

CompletionBoundarySpec makeProductionK3CompletionSpec() {
    CompletionBoundarySpec spec;
    spec.think_close_token_ids    = {163588, 39964, 163589};
    spec.response_open_token_ids  = {163587, 12092, 163589};
    spec.response_close_token_ids = {163588, 12092, 163589};
    spec.message_close_token_ids  = {163588, 2778, 163589};
    spec.starts_in_think          = true;
    return spec;
}

void expectOnlyStopAllowed(CompletionBoundaryLogitsProcessor& processor, int32_t stop_token_id = 0) {
    auto inputs = makeInputs(32);
    processor.process(inputs, 0, 1);
    EXPECT_EQ(inputs.logits[0][stop_token_id].item<float>(), 0.0f);
    EXPECT_EQ(inputs.logits[0][5].item<float>(), BaseLogitsProcessor::neg_inf);
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
    CompletionBoundaryLogitsProcessor processor({CompletionBoundaryState({7, 8, 7, 9}, 0, false)}, {0});

    processor.updateStatus(torch::tensor({{7, 8, 7, 8, 7, 9}}, torch::kInt32), 6);
    EXPECT_EQ(processor.boundaryStatus()[0], 4);

    auto inputs = makeInputs(16);
    processor.process(inputs, 0, 1);
    EXPECT_EQ(inputs.logits[0][0].item<float>(), 0.0f);
}

TEST(CompletionBoundaryLogitsProcessorTest, SpecMaskOpensAfterBoundaryInsideDraft) {
    CompletionBoundaryLogitsProcessor processor({CompletionBoundaryState({7, 8}, 0, false)}, {0, 6});
    const int                         propose_step = 3;
    const size_t                      words        = SpecLogitsProcessor::bitmaskWordCount(16);
    std::vector<int32_t>              draft        = {7, 8, 6};
    std::vector<int32_t>              bitmask((propose_step + 1) * words, 0);

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
    CompletionBoundaryLogitsProcessor processor({CompletionBoundaryState({7, 8}, 0, false)}, {0});
    const int                         propose_step = 2;
    const size_t                      words        = SpecLogitsProcessor::bitmaskWordCount(16);
    std::vector<int32_t>              draft        = {7, 0};
    std::vector<int32_t>              bitmask((propose_step + 1) * words, 0);

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

TEST(CompletionBoundaryLogitsProcessorTest, StatefulGuardRejectsMessageCloseBeforeResponseClose) {
    CompletionBoundaryLogitsProcessor processor({CompletionBoundaryState(makeK3CompletionSpec(true), 0, false)}, {0});

    processor.updateStatus(torch::tensor({{10, 11, 12, 13, 20, 21}}, torch::kInt32), 6);
    expectOnlyStopAllowed(processor);
}

TEST(CompletionBoundaryLogitsProcessorTest, StatefulGuardRequiresResponseOpenAfterThinkClose) {
    CompletionBoundaryLogitsProcessor processor({CompletionBoundaryState(makeK3CompletionSpec(true), 0, false)}, {0});

    processor.updateStatus(torch::tensor({{10, 11, 5, 14, 15, 20, 21}}, torch::kInt32), 7);
    expectOnlyStopAllowed(processor);
}

TEST(CompletionBoundaryLogitsProcessorTest, ProductionK3TraceTailForcesStopAfterInvalidMessageClose) {
    CompletionBoundaryLogitsProcessor processor({CompletionBoundaryState(makeProductionK3CompletionSpec(), 0, false)},
                                                {163586});

    // Real failed tail: think closes, response opens, then message closes
    // without response content or <|close|>response<|sep|>.
    processor.updateStatus(
        torch::tensor({{163588, 39964, 163589, 163587, 12092, 163589, 163588, 2778, 163589}}, torch::kInt32), 9);

    auto inputs = makeInputs(163600);
    processor.process(inputs, 0, 1);
    EXPECT_EQ(inputs.logits[0][163586].item<float>(), 0.0f);
    EXPECT_EQ(inputs.logits[0][163588].item<float>(), BaseLogitsProcessor::neg_inf);
}

TEST(CompletionBoundaryLogitsProcessorTest, ProductionK3DirectResponseCloseFromThinkForcesStop) {
    CompletionBoundaryLogitsProcessor processor({CompletionBoundaryState(makeProductionK3CompletionSpec(), 0, false)},
                                                {163586});

    // Exact invalid transition observed online: reasoning text is followed by
    // <|close|>response<|sep|> while the state is still inside think.
    processor.updateStatus(torch::tensor({{42, 163588, 12092, 163589}}, torch::kInt32), 4);

    auto inputs = makeInputs(163600);
    processor.process(inputs, 0, 1);
    EXPECT_EQ(inputs.logits[0][163586].item<float>(), 0.0f);
    EXPECT_EQ(inputs.logits[0][163588].item<float>(), BaseLogitsProcessor::neg_inf);
}

TEST(CompletionBoundaryLogitsProcessorTest, ProductionK3ValidResponseForcesEndOfMessage) {
    CompletionBoundaryLogitsProcessor processor({CompletionBoundaryState(makeProductionK3CompletionSpec(), 0, false)},
                                                {163586});

    processor.updateStatus(
        torch::tensor({{163588, 39964, 163589, 163587, 12092, 163589, 42, 163588, 12092, 163589, 163588, 2778, 163589}},
                      torch::kInt32),
        13);

    auto inputs = makeInputs(163600);
    processor.process(inputs, 0, 1);
    EXPECT_EQ(inputs.logits[0][163586].item<float>(), 0.0f);
    EXPECT_EQ(inputs.logits[0][163588].item<float>(), BaseLogitsProcessor::neg_inf);
}

TEST(CompletionBoundaryLogitsProcessorTest, StatefulGuardRejectsEmptyOrWhitespaceOnlyResponse) {
    CompletionBoundaryLogitsProcessor empty_processor({CompletionBoundaryState(makeK3CompletionSpec(false), 0, false)},
                                                      {0});
    empty_processor.updateStatus(torch::tensor({{14, 15, 20, 21}}, torch::kInt32), 4);
    expectOnlyStopAllowed(empty_processor);

    CompletionBoundaryLogitsProcessor whitespace_processor(
        {CompletionBoundaryState(makeK3CompletionSpec(false), 0, false)}, {0});
    whitespace_processor.updateStatus(torch::tensor({{2, 14, 15, 20, 21}}, torch::kInt32), 5);
    expectOnlyStopAllowed(whitespace_processor);
}

TEST(CompletionBoundaryLogitsProcessorTest, StatefulGuardAcceptsClosedVisibleResponse) {
    CompletionBoundaryLogitsProcessor processor({CompletionBoundaryState(makeK3CompletionSpec(true), 0, false)}, {0});

    processor.updateStatus(torch::tensor({{10, 11, 12, 13, 5, 14, 15, 20, 21}}, torch::kInt32), 9);
    expectOnlyStopAllowed(processor);
}

TEST(CompletionBoundaryLogitsProcessorTest, StatefulGuardMasksOnlyDeclaredStops) {
    CompletionBoundaryLogitsProcessor processor({CompletionBoundaryState(makeK3CompletionSpec(false), 0, false)},
                                                {0, 6});

    auto inputs = makeInputs(32);
    processor.process(inputs, 0, 1);
    EXPECT_EQ(inputs.logits[0][0].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(inputs.logits[0][6].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(inputs.logits[0][5].item<float>(), 0.0f);
    EXPECT_EQ(inputs.logits[0][14].item<float>(), 0.0f);
}

TEST(CompletionBoundaryLogitsProcessorTest, StatefulGuardAcceptsToolsAfterEmptyResponse) {
    CompletionBoundaryLogitsProcessor processor({CompletionBoundaryState(makeK3CompletionSpec(false), 0, false)}, {0});

    processor.updateStatus(torch::tensor({{14, 15, 16, 17, 6, 18, 19, 20, 21}}, torch::kInt32), 9);
    expectOnlyStopAllowed(processor);
}

TEST(CompletionBoundaryLogitsProcessorTest, StatefulGuardRejectsEmptyToolsChannel) {
    CompletionBoundaryLogitsProcessor processor({CompletionBoundaryState(makeK3CompletionSpec(false), 0, false)}, {0});

    processor.updateStatus(torch::tensor({{14, 15, 16, 17, 18, 19, 20, 21}}, torch::kInt32), 8);
    expectOnlyStopAllowed(processor);

    CompletionBoundaryLogitsProcessor whitespace_processor(
        {CompletionBoundaryState(makeK3CompletionSpec(false), 0, false)}, {0});
    whitespace_processor.updateStatus(torch::tensor({{14, 15, 16, 17, 2, 18, 19, 20, 21}}, torch::kInt32), 9);
    expectOnlyStopAllowed(whitespace_processor);
}

TEST(CompletionBoundaryLogitsProcessorTest, StatefulGuardRejectsOpenedButUnclosedToolsChannel) {
    CompletionBoundaryLogitsProcessor processor({CompletionBoundaryState(makeK3CompletionSpec(false), 0, false)}, {0});

    processor.updateStatus(torch::tensor({{14, 15, 16, 17, 6, 20, 21}}, torch::kInt32), 7);
    expectOnlyStopAllowed(processor);
}

TEST(CompletionBoundaryLogitsProcessorTest, StatefulGuardRejectsEmptyToolsAfterVisibleResponse) {
    CompletionBoundaryLogitsProcessor processor({CompletionBoundaryState(makeK3CompletionSpec(false), 0, false)}, {0});

    processor.updateStatus(torch::tensor({{5, 14, 15, 16, 17, 18, 19, 20, 21}}, torch::kInt32), 9);
    expectOnlyStopAllowed(processor);
}

TEST(CompletionBoundaryLogitsProcessorTest, StatefulGuardCopiesIndependentMultiSequenceState) {
    CompletionBoundaryLogitsProcessor processor({CompletionBoundaryState(makeK3CompletionSpec(false), 0, false)}, {0});

    processor.updateStatus(torch::tensor({{5, 14, 15}}, torch::kInt32), 3);
    processor.updateMultiSeqStatus({0, 0});
    processor.updateStatus(torch::tensor({{20, 21}, {16, 17}}, torch::kInt32), 2);

    SamplerInputs inputs;
    inputs.logits     = torch::zeros({2, 32}, torch::kFloat32);
    inputs.vocab_size = 32;
    processor.process(inputs, 0, 2);
    EXPECT_EQ(inputs.logits[0][0].item<float>(), 0.0f);
    EXPECT_EQ(inputs.logits[1][0].item<float>(), BaseLogitsProcessor::neg_inf);
}

TEST(CompletionBoundaryLogitsProcessorTest, StatefulGuardSpecMaskOpensOnlyAfterValidCompletion) {
    CompletionBoundaryLogitsProcessor processor({CompletionBoundaryState(makeK3CompletionSpec(false), 0, false)}, {0});
    const std::vector<int32_t>        draft        = {5, 14, 15, 20, 21, 0};
    const int                         propose_step = static_cast<int>(draft.size());
    const size_t                      words        = SpecLogitsProcessor::bitmaskWordCount(32);
    std::vector<int32_t>              bitmask((propose_step + 1) * words, 0);

    SpecLogitsProcessorRequest request;
    request.draft_tokens       = draft.data();
    request.propose_step       = propose_step;
    request.bitmask_cpu_out    = bitmask.data();
    request.bitmask_size_int32 = words;
    request.vocab_size         = 32;

    EXPECT_EQ(processor.tryAcceptAndFillBitmask(request), propose_step);
    EXPECT_FALSE(bitmaskAllows(bitmask.data(), 0));
    EXPECT_FALSE(bitmaskAllows(bitmask.data() + 4 * words, 0));
    EXPECT_TRUE(bitmaskAllows(bitmask.data() + 5 * words, 0));
    EXPECT_TRUE(bitmaskAllows(bitmask.data() + 6 * words, 0));
    EXPECT_FALSE(bitmaskAllows(bitmask.data() + 5 * words, 5));
}

TEST(CompletionBoundaryLogitsProcessorTest, StatefulGuardSpecRejectsLoopAfterInvalidTransition) {
    CompletionBoundaryLogitsProcessor processor({CompletionBoundaryState(makeK3CompletionSpec(true), 0, false)}, {0});
    const std::vector<int32_t>        draft        = {14, 15, 20, 21};
    const int                         propose_step = static_cast<int>(draft.size());
    const size_t                      words        = SpecLogitsProcessor::bitmaskWordCount(32);
    std::vector<int32_t>              bitmask((propose_step + 1) * words, 0);

    SpecLogitsProcessorRequest request;
    request.draft_tokens       = draft.data();
    request.propose_step       = propose_step;
    request.bitmask_cpu_out    = bitmask.data();
    request.bitmask_size_int32 = words;
    request.vocab_size         = 32;

    EXPECT_EQ(processor.tryAcceptAndFillBitmask(request), 2);
    EXPECT_TRUE(bitmaskAllows(bitmask.data() + 2 * words, 0));
    EXPECT_FALSE(bitmaskAllows(bitmask.data() + 2 * words, 20));
}

}  // namespace
}  // namespace rtp_llm
