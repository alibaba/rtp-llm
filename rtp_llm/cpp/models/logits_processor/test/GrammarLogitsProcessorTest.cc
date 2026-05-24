#include "rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include <xgrammar/tokenizer_info.h>

#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackendCpp.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"

namespace rtp_llm {
namespace {

std::string makeTokenizerInfoJson() {
    std::vector<std::string> vocab;
    vocab.reserve(128);
    for (int i = 0; i < 128; ++i) {
        vocab.emplace_back(1, static_cast<char>(i));
    }
    xgrammar::TokenizerInfo info(vocab,
                                 xgrammar::VocabType::RAW,
                                 /*vocab_size=*/128,
                                 /*stop_token_ids=*/std::vector<int32_t>{0});
    return info.SerializeJSON();
}

XGrammarBackendCpp makeBackend() {
    XGrammarBackendOptions options;
    options.max_compiler_threads = 1;
    return XGrammarBackendCpp(makeTokenizerInfoJson(), options);
}

bool packedBitmaskAllowsToken(const int32_t* bitmask, int32_t token_id) {
    const int32_t word = bitmask[token_id / 32];
    return (static_cast<uint32_t>(word) & (1u << (token_id % 32))) != 0u;
}

TEST(GrammarLogitsProcessorTest, ProcessMasksDisallowedTokens) {
    auto backend  = makeBackend();
    auto compiled = backend.compileNow({"regex", "a"}).compiled;
    ASSERT_TRUE(compiled);

    auto                   matcher = backend.createMatcher(compiled, false, std::nullopt);
    GrammarLogitsProcessor processor(matcher, /*eos_token_id=*/0);

    SamplerInputs inputs;
    inputs.logits        = torch::zeros({1, 128}, torch::kFloat32);
    inputs.finished_mask = torch::zeros({1}, torch::kBool);
    processor.process(inputs, 0, 1);

    EXPECT_GT(inputs.logits[0][static_cast<int>('a')].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(inputs.logits[0][static_cast<int>('b')].item<float>(), BaseLogitsProcessor::neg_inf);
}

TEST(GrammarLogitsProcessorTest, UpdateStatusAdvancesMatcherToTerminal) {
    auto backend  = makeBackend();
    auto compiled = backend.compileNow({"regex", "a"}).compiled;
    ASSERT_TRUE(compiled);

    auto                   matcher = backend.createMatcher(compiled, false, std::nullopt);
    GrammarLogitsProcessor processor(matcher, /*eos_token_id=*/0);

    processor.updateStatus(torch::tensor({{static_cast<int32_t>('a')}}, torch::kInt32), 1);
    EXPECT_TRUE(processor.isStateful());
    EXPECT_EQ(processor.acceptedTokenLen(), 1);
    EXPECT_FALSE(matcher->isTerminated());

    processor.updateStatus(torch::tensor({{0}}, torch::kInt32), 1);
    EXPECT_EQ(processor.acceptedTokenLen(), 2);
    EXPECT_TRUE(matcher->isTerminated());
}

TEST(GrammarLogitsProcessorTest, TerminateWithoutStopTokenForcesEosAndAcceptsCommit) {
    auto backend  = makeBackend();
    auto compiled = backend.compileNow({"regex", "a"}).compiled;
    ASSERT_TRUE(compiled);

    auto matcher = backend.createMatcher(compiled, false, std::nullopt, /*terminate_without_stop_token=*/true);
    GrammarLogitsProcessor processor(matcher, /*eos_token_id=*/0);

    processor.updateStatus(torch::tensor({{static_cast<int32_t>('a')}}, torch::kInt32), 1);
    ASSERT_TRUE(matcher->isTerminated());

    SamplerInputs inputs;
    inputs.logits        = torch::zeros({1, 128}, torch::kFloat32);
    inputs.finished_mask = torch::zeros({1}, torch::kBool);
    processor.process(inputs, 0, 1);

    EXPECT_EQ(inputs.logits[0][0].item<float>(), 1.0f);
    EXPECT_EQ(inputs.logits[0][static_cast<int>('a')].item<float>(), BaseLogitsProcessor::neg_inf);

    processor.updateStatus(torch::tensor({{0}}, torch::kInt32), 1);
    EXPECT_EQ(processor.acceptedTokenLen(), 2);
}

TEST(GrammarLogitsProcessorTest, ReasoningModeWaitsForFullThinkEndSequence) {
    auto backend  = makeBackend();
    auto compiled = backend.compileNow({"regex", "a"}).compiled;
    ASSERT_TRUE(compiled);

    auto matcher =
        backend.createMatcher(compiled, true, std::vector<int>{static_cast<int>('x'), static_cast<int>('y')});
    matcher->initReasoning(true);
    GrammarLogitsProcessor processor(matcher, /*eos_token_id=*/0);

    SamplerInputs inputs;
    inputs.logits        = torch::zeros({1, 128}, torch::kFloat32);
    inputs.finished_mask = torch::zeros({1}, torch::kBool);

    processor.process(inputs, 0, 1);
    EXPECT_EQ(inputs.logits[0][static_cast<int>('b')].item<float>(), 0.0f);
    EXPECT_EQ(inputs.logits[0][0].item<float>(), BaseLogitsProcessor::neg_inf);

    processor.updateStatus(torch::tensor({{static_cast<int32_t>('x')}}, torch::kInt32), 1);
    inputs.logits = torch::zeros({1, 128}, torch::kFloat32);
    processor.process(inputs, 0, 1);
    EXPECT_EQ(inputs.logits[0][static_cast<int>('b')].item<float>(), 0.0f);
    EXPECT_EQ(inputs.logits[0][0].item<float>(), BaseLogitsProcessor::neg_inf);

    processor.updateStatus(torch::tensor({{static_cast<int32_t>('y')}}, torch::kInt32), 1);
    processor.process(inputs, 0, 1);
    EXPECT_GT(inputs.logits[0][static_cast<int>('a')].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(inputs.logits[0][static_cast<int>('b')].item<float>(), BaseLogitsProcessor::neg_inf);
}

TEST(GrammarLogitsProcessorTest, ReasoningModeNormalizesThinkEndPaddingTokens) {
    constexpr int32_t kQwenGlmNewlineTokenId = 198;
    auto              backend                = makeBackend();
    auto              compiled               = backend.compileNow({"regex", "a"}).compiled;
    ASSERT_TRUE(compiled);

    auto matcher = backend.createMatcher(
        compiled, true, std::vector<int>{kQwenGlmNewlineTokenId, static_cast<int>('x'), kQwenGlmNewlineTokenId});
    matcher->initReasoning(true);
    GrammarLogitsProcessor processor(matcher, /*eos_token_id=*/0);

    processor.updateStatus(torch::tensor({{static_cast<int32_t>('x')}}, torch::kInt32), 1);

    SamplerInputs inputs;
    inputs.logits        = torch::zeros({1, 128}, torch::kFloat32);
    inputs.finished_mask = torch::zeros({1}, torch::kBool);
    processor.process(inputs, 0, 1);

    EXPECT_GT(inputs.logits[0][static_cast<int>('a')].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(inputs.logits[0][static_cast<int>('b')].item<float>(), BaseLogitsProcessor::neg_inf);
}

TEST(GrammarLogitsProcessorTest, SpeculativePrefixMasksRowsAndRollsBack) {
    auto backend  = makeBackend();
    auto compiled = backend.compileNow({"regex", "ab"}).compiled;
    ASSERT_TRUE(compiled);

    auto matcher   = backend.createMatcher(compiled, false, std::nullopt);
    auto processor = std::make_shared<GrammarLogitsProcessor>(matcher, /*eos_token_id=*/0);

    SamplerInputs inputs;
    inputs.logits        = torch::zeros({2, 128}, torch::kFloat32);
    inputs.finished_mask = torch::zeros({2}, torch::kBool);

    LogitsProcessorStates states;
    states.insertSpeculative(processor, 0, 1, {});
    states.insertSpeculative(processor, 1, 2, {static_cast<int32_t>('a')});
    states.batchProcess(inputs);

    EXPECT_GT(inputs.logits[0][static_cast<int>('a')].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(inputs.logits[0][static_cast<int>('b')].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(inputs.logits[1][static_cast<int>('a')].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_GT(inputs.logits[1][static_cast<int>('b')].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(processor->acceptedTokenLen(), 0);

    inputs.logits = torch::zeros({1, 128}, torch::kFloat32);
    processor->process(inputs, 0, 1);
    EXPECT_GT(inputs.logits[0][static_cast<int>('a')].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(inputs.logits[0][static_cast<int>('b')].item<float>(), BaseLogitsProcessor::neg_inf);
}

TEST(GrammarLogitsProcessorTest, SpecTryAcceptBuildsOffsetMasksAndCap) {
    auto backend  = makeBackend();
    auto compiled = backend.compileNow({"regex", "ab"}).compiled;
    ASSERT_TRUE(compiled);

    auto                   matcher = backend.createMatcher(compiled, false, std::nullopt);
    GrammarLogitsProcessor processor(matcher, /*eos_token_id=*/0);

    const int            P     = 2;
    const size_t         W     = SpecLogitsProcessor::bitmaskWordCount(128);
    std::vector<int32_t> draft = {static_cast<int32_t>('a'), static_cast<int32_t>('x')};
    std::vector<int32_t> bitmask((P + 1) * W, SpecLogitsProcessor::kBitmaskAllowAll);

    SpecLogitsProcessorRequest request;
    request.draft_tokens       = draft.data();
    request.propose_step       = P;
    request.bitmask_cpu_out    = bitmask.data();
    request.bitmask_size_int32 = W;
    request.vocab_size         = 128;

    EXPECT_EQ(processor.tryAcceptAndFillBitmask(request), 1);
    EXPECT_TRUE(packedBitmaskAllowsToken(bitmask.data(), static_cast<int32_t>('a')));
    EXPECT_FALSE(packedBitmaskAllowsToken(bitmask.data(), static_cast<int32_t>('b')));
    EXPECT_FALSE(packedBitmaskAllowsToken(bitmask.data() + W, static_cast<int32_t>('a')));
    EXPECT_TRUE(packedBitmaskAllowsToken(bitmask.data() + W, static_cast<int32_t>('b')));
    EXPECT_EQ(processor.acceptedTokenLen(), 0);
}

TEST(GrammarLogitsProcessorTest, ReasoningModeUsesKmpForSelfOverlappingThinkEnd) {
    auto backend  = makeBackend();
    auto compiled = backend.compileNow({"regex", "a"}).compiled;
    ASSERT_TRUE(compiled);

    auto matcher = backend.createMatcher(
        compiled, true, std::vector<int>{static_cast<int>('x'), static_cast<int>('y'), static_cast<int>('x')});
    matcher->initReasoning(true);
    GrammarLogitsProcessor processor(matcher, /*eos_token_id=*/0);

    SamplerInputs inputs;
    inputs.logits        = torch::zeros({1, 128}, torch::kFloat32);
    inputs.finished_mask = torch::zeros({1}, torch::kBool);

    processor.updateStatus(torch::tensor({{static_cast<int32_t>('x'),
                                           static_cast<int32_t>('y'),
                                           static_cast<int32_t>('y'),
                                           static_cast<int32_t>('x')}},
                                         torch::kInt32),
                           4);
    processor.process(inputs, 0, 1);
    EXPECT_EQ(inputs.logits[0][static_cast<int>('b')].item<float>(), 0.0f);
    EXPECT_EQ(inputs.logits[0][0].item<float>(), BaseLogitsProcessor::neg_inf);

    processor.updateStatus(torch::tensor({{static_cast<int32_t>('y'), static_cast<int32_t>('x')}}, torch::kInt32), 2);
    inputs.logits = torch::zeros({1, 128}, torch::kFloat32);
    processor.process(inputs, 0, 1);
    EXPECT_GT(inputs.logits[0][static_cast<int>('a')].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(inputs.logits[0][static_cast<int>('b')].item<float>(), BaseLogitsProcessor::neg_inf);
}

}  // namespace
}  // namespace rtp_llm
