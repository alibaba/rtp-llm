#include "rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include <xgrammar/tokenizer_info.h>

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackendCpp.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"
#include "rtp_llm/cpp/models/logits_processor/ReasoningGrammarLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/ThinkModeLogitsProcessor.h"

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

class MaskAllSpecProcessor: public BaseLogitsProcessor, public SpecLogitsProcessor {
public:
    void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override {
        inputs.logits.narrow(0, start_idx, finish_idx - start_idx).fill_(BaseLogitsProcessor::neg_inf);
    }
    void updateMultiSeqStatus(const std::vector<int>&) override {}
    void updateStatus(const torch::Tensor&, int32_t) override {}
    bool isSpecVerifyEligible() const override {
        return true;
    }
    int tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) override {
        return request.propose_step;
    }
};

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

TEST(GrammarLogitsProcessorTest, JsonObjectReasoningModeConstrainsOnlyAfterThinkEnd) {
    auto backend  = makeBackend();
    auto compiled = backend.compileNow({"json", R"({"type":"object"})"}).compiled;
    ASSERT_TRUE(compiled);

    auto matcher =
        backend.createMatcher(compiled, true, std::vector<int>{static_cast<int>('x'), static_cast<int>('y')});
    matcher->initReasoning(true);
    GrammarLogitsProcessor processor(matcher, /*eos_token_id=*/0);

    SamplerInputs inputs;
    inputs.logits        = torch::zeros({1, 128}, torch::kFloat32);
    inputs.finished_mask = torch::zeros({1}, torch::kBool);

    processor.process(inputs, 0, 1);
    EXPECT_EQ(inputs.logits[0][static_cast<int>('a')].item<float>(), 0.0f);
    EXPECT_EQ(inputs.logits[0][static_cast<int>('{')].item<float>(), 0.0f);
    EXPECT_EQ(inputs.logits[0][0].item<float>(), BaseLogitsProcessor::neg_inf);

    processor.updateStatus(torch::tensor({{static_cast<int32_t>('x'), static_cast<int32_t>('y')}}, torch::kInt32), 2);
    inputs.logits = torch::zeros({1, 128}, torch::kFloat32);
    processor.process(inputs, 0, 1);

    EXPECT_GT(inputs.logits[0][static_cast<int>('{')].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(inputs.logits[0][static_cast<int>('a')].item<float>(), BaseLogitsProcessor::neg_inf);
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

TEST(GrammarLogitsProcessorTest, SpeculativePrefixPathReportsInsteadOfRollingMatcher) {
    auto backend  = makeBackend();
    auto compiled = backend.compileNow({"regex", "ab"}).compiled;
    ASSERT_TRUE(compiled);

    auto matcher   = backend.createMatcher(compiled, false, std::nullopt);
    bool reported  = false;
    auto processor = std::make_shared<GrammarLogitsProcessor>(
        matcher,
        /*eos_token_id=*/0,
        [&reported](ErrorCode, const std::string& message, bool) {
            reported = message.find("precomputed MTP verify bitmask") != std::string::npos;
        });

    SamplerInputs inputs;
    inputs.logits        = torch::zeros({2, 128}, torch::kFloat32);
    inputs.finished_mask = torch::zeros({2}, torch::kBool);

    LogitsProcessorStates states;
    states.insertSpeculative(processor, 0, 1, {});
    states.insertSpeculative(processor, 1, 2, {static_cast<int32_t>('a')});
    states.batchProcess(inputs);

    EXPECT_GT(inputs.logits[0][static_cast<int>('a')].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(inputs.logits[0][static_cast<int>('b')].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_TRUE(reported);
    EXPECT_EQ(inputs.logits[1][static_cast<int>('a')].item<float>(), 0.0f);
    EXPECT_EQ(inputs.logits[1][static_cast<int>('b')].item<float>(), 0.0f);
    EXPECT_EQ(processor->acceptedTokenLen(), 0);

    inputs.logits = torch::zeros({1, 128}, torch::kFloat32);
    processor->process(inputs, 0, 1);
    EXPECT_GT(inputs.logits[0][static_cast<int>('a')].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(inputs.logits[0][static_cast<int>('b')].item<float>(), BaseLogitsProcessor::neg_inf);
}

TEST(GrammarLogitsProcessorTest, SpecArtifactSkipsOnlyAppliedProcessorId) {
    auto processor = std::make_shared<MaskAllSpecProcessor>();

    SamplerInputs inputs;
    inputs.phase               = LogitsProcessorPhase::MTP_VERIFY;
    inputs.logits              = torch::zeros({1, 4}, torch::kFloat32);
    inputs.spec_vocab_mask_gpu = torch::zeros({1, 4}, torch::kBool);
    inputs.spec_applied_processors.push_back({7, 3});

    LogitsProcessorStates states;
    states.insert(processor, 0, 1, /*stream_id=*/7, /*processor_idx=*/3);
    states.batchProcess(inputs);
    EXPECT_EQ(inputs.logits[0][0].item<float>(), 0.0f);

    inputs.logits.zero_();
    inputs.spec_applied_processors = {{7, 4}};
    states.batchProcess(inputs);
    EXPECT_EQ(inputs.logits[0][0].item<float>(), BaseLogitsProcessor::neg_inf);
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

TEST(GrammarLogitsProcessorTest, SpecTryAcceptMasksModelVocabTailBeyondGrammarVocab) {
    auto backend  = makeBackend();
    auto compiled = backend.compileNow({"regex", "a"}).compiled;
    ASSERT_TRUE(compiled);

    auto                   matcher = backend.createMatcher(compiled, false, std::nullopt);
    GrammarLogitsProcessor processor(matcher, /*eos_token_id=*/0);

    const int            P           = 1;
    const size_t         model_vocab = 160;
    const size_t         W           = SpecLogitsProcessor::bitmaskWordCount(model_vocab);
    std::vector<int32_t> draft       = {static_cast<int32_t>('a')};
    std::vector<int32_t> bitmask((P + 1) * W, SpecLogitsProcessor::kBitmaskAllowAll);

    SpecLogitsProcessorRequest request;
    request.draft_tokens       = draft.data();
    request.propose_step       = P;
    request.bitmask_cpu_out    = bitmask.data();
    request.bitmask_size_int32 = W;
    request.vocab_size         = model_vocab;

    EXPECT_EQ(processor.tryAcceptAndFillBitmask(request), 1);
    EXPECT_FALSE(packedBitmaskAllowsToken(bitmask.data(), 128));
    EXPECT_FALSE(packedBitmaskAllowsToken(bitmask.data(), 159));
    EXPECT_FALSE(packedBitmaskAllowsToken(bitmask.data() + W, 128));
    EXPECT_FALSE(packedBitmaskAllowsToken(bitmask.data() + W, 159));
}

TEST(GrammarLogitsProcessorTest, SpecTryAcceptRejectsInvalidFirstDraftToken) {
    auto backend  = makeBackend();
    auto compiled = backend.compileNow({"regex", "ab"}).compiled;
    ASSERT_TRUE(compiled);

    auto                   matcher = backend.createMatcher(compiled, false, std::nullopt);
    GrammarLogitsProcessor processor(matcher, /*eos_token_id=*/0);

    const int            P     = 2;
    const size_t         W     = SpecLogitsProcessor::bitmaskWordCount(128);
    std::vector<int32_t> draft = {static_cast<int32_t>('x'), static_cast<int32_t>('b')};
    std::vector<int32_t> bitmask((P + 1) * W, SpecLogitsProcessor::kBitmaskAllowAll);

    SpecLogitsProcessorRequest request;
    request.draft_tokens       = draft.data();
    request.propose_step       = P;
    request.bitmask_cpu_out    = bitmask.data();
    request.bitmask_size_int32 = W;
    request.vocab_size         = 128;

    EXPECT_EQ(processor.tryAcceptAndFillBitmask(request), 0);
    EXPECT_TRUE(packedBitmaskAllowsToken(bitmask.data(), static_cast<int32_t>('a')));
    EXPECT_FALSE(packedBitmaskAllowsToken(bitmask.data(), static_cast<int32_t>('x')));
    EXPECT_EQ(processor.acceptedTokenLen(), 0);
}

TEST(GrammarLogitsProcessorTest, SpecTryAcceptJsonObjectReasoningPassthroughUntilThinkEnd) {
    auto backend  = makeBackend();
    auto compiled = backend.compileNow({"json", R"({"type":"object"})"}).compiled;
    ASSERT_TRUE(compiled);

    auto matcher =
        backend.createMatcher(compiled, true, std::vector<int>{static_cast<int>('x'), static_cast<int>('y')});
    matcher->initReasoning(true);
    GrammarLogitsProcessor processor(matcher, /*eos_token_id=*/0);

    const int            P     = 4;
    const size_t         W     = SpecLogitsProcessor::bitmaskWordCount(128);
    std::vector<int32_t> draft = {
        static_cast<int32_t>('a'),
        static_cast<int32_t>('x'),
        static_cast<int32_t>('y'),
        static_cast<int32_t>('a'),
    };
    std::vector<int32_t> bitmask((P + 1) * W, SpecLogitsProcessor::kBitmaskAllowAll);

    SpecLogitsProcessorRequest request;
    request.draft_tokens       = draft.data();
    request.propose_step       = P;
    request.bitmask_cpu_out    = bitmask.data();
    request.bitmask_size_int32 = W;
    request.vocab_size         = 128;

    EXPECT_EQ(processor.tryAcceptAndFillBitmask(request), 3);
    EXPECT_TRUE(packedBitmaskAllowsToken(bitmask.data(), static_cast<int32_t>('a')));
    EXPECT_TRUE(packedBitmaskAllowsToken(bitmask.data() + W, static_cast<int32_t>('x')));
    EXPECT_TRUE(packedBitmaskAllowsToken(bitmask.data() + 2 * W, static_cast<int32_t>('y')));
    EXPECT_FALSE(packedBitmaskAllowsToken(bitmask.data(), 0));
    EXPECT_FALSE(packedBitmaskAllowsToken(bitmask.data() + W, 0));
    EXPECT_FALSE(packedBitmaskAllowsToken(bitmask.data() + 2 * W, 0));
    EXPECT_TRUE(packedBitmaskAllowsToken(bitmask.data() + 3 * W, static_cast<int32_t>('{')));
    EXPECT_FALSE(packedBitmaskAllowsToken(bitmask.data() + 3 * W, static_cast<int32_t>('a')));
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

TEST(ReasoningGrammarLogitsProcessorTest, JsonObjectConstrainsOnlyAfterThinkEnd) {
    auto backend  = makeBackend();
    auto compiled = backend.compileNow({"json", R"({"type":"object"})"}).compiled;
    ASSERT_TRUE(compiled);

    auto matcher = backend.createMatcher(
        compiled, /*require_reasoning=*/false, std::nullopt, /*terminate_without_stop_token=*/true);
    ReasoningGrammarLogitsProcessor processor(matcher,
                                              /*eos_token_id=*/0,
                                              /*max_thinking_tokens=*/32,
                                              {static_cast<int>('<')},
                                              {static_cast<int>('x'), static_cast<int>('y')},
                                              /*input_length=*/0);

    SamplerInputs inputs;
    inputs.logits           = torch::zeros({1, 128}, torch::kFloat32);
    inputs.finished_mask    = torch::zeros({1}, torch::kBool);
    inputs.input_lengths    = torch::tensor({0}, torch::kInt32);
    inputs.sequence_lengths = torch::tensor({0}, torch::kInt32);
    inputs.vocab_size       = 128;
    processor.process(inputs, 0, 1);

    EXPECT_EQ(inputs.logits[0][static_cast<int>('a')].item<float>(), 0.0f);
    EXPECT_EQ(inputs.logits[0][static_cast<int>('{')].item<float>(), 0.0f);
    EXPECT_EQ(inputs.logits[0][0].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(inputs.logits[0][static_cast<int>('<')].item<float>(), BaseLogitsProcessor::neg_inf);

    processor.updateStatus(
        torch::tensor({{static_cast<int32_t>('b'), static_cast<int32_t>('x'), static_cast<int32_t>('y')}},
                      torch::kInt32),
        3);

    inputs.logits           = torch::zeros({1, 128}, torch::kFloat32);
    inputs.sequence_lengths = torch::tensor({3}, torch::kInt32);
    processor.process(inputs, 0, 1);

    EXPECT_GT(inputs.logits[0][static_cast<int>('{')].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(inputs.logits[0][static_cast<int>('a')].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(processor.acceptedTokenLen(), 3);
}

TEST(ReasoningGrammarLogitsProcessorTest, NaturalCloseForcesTrailingPadBeforeGrammar) {
    // end_think = [</think>='y', 271 (pad)]; pad stays in DFA, so a natural
    // </think> only matches the prefix and CLOSING_THINK force-emits 271 via
    // the existing forceThinkEndTokenLocked path before grammar takes over.
    constexpr int kPadToken  = 271;
    constexpr int kVocabSize = 512;
    auto          backend    = makeBackend();
    auto          compiled   = backend.compileNow({"json", R"({"type":"object"})"}).compiled;
    ASSERT_TRUE(compiled);

    auto matcher = backend.createMatcher(
        compiled, /*require_reasoning=*/false, std::nullopt, /*terminate_without_stop_token=*/true);
    ReasoningGrammarLogitsProcessor processor(matcher,
                                              /*eos_token_id=*/0,
                                              /*max_thinking_tokens=*/32,
                                              {static_cast<int>('<')},
                                              {static_cast<int>('y'), kPadToken},
                                              /*input_length=*/0);

    processor.updateStatus(torch::tensor({{static_cast<int32_t>('y')}}, torch::kInt32), 1);

    SamplerInputs inputs;
    inputs.logits           = torch::zeros({1, kVocabSize}, torch::kFloat32);
    inputs.finished_mask    = torch::zeros({1}, torch::kBool);
    inputs.input_lengths    = torch::tensor({0}, torch::kInt32);
    inputs.sequence_lengths = torch::tensor({1}, torch::kInt32);
    inputs.vocab_size       = kVocabSize;
    processor.process(inputs, 0, 1);

    EXPECT_GT(inputs.logits[0][kPadToken].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(inputs.logits[0][static_cast<int>('{')].item<float>(), BaseLogitsProcessor::neg_inf);

    processor.updateStatus(torch::tensor({{kPadToken}}, torch::kInt32), 1);

    inputs.logits           = torch::zeros({1, kVocabSize}, torch::kFloat32);
    inputs.sequence_lengths = torch::tensor({2}, torch::kInt32);
    processor.process(inputs, 0, 1);

    EXPECT_GT(inputs.logits[0][static_cast<int>('{')].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(inputs.logits[0][static_cast<int>('a')].item<float>(), BaseLogitsProcessor::neg_inf);
}

TEST(ReasoningGrammarLogitsProcessorTest, SpecTryAcceptPassthroughThenGrammar) {
    auto backend  = makeBackend();
    auto compiled = backend.compileNow({"regex", "a"}).compiled;
    ASSERT_TRUE(compiled);

    auto matcher = backend.createMatcher(compiled, /*require_reasoning=*/false, std::nullopt);
    ReasoningGrammarLogitsProcessor processor(matcher,
                                              /*eos_token_id=*/0,
                                              /*max_thinking_tokens=*/32,
                                              {static_cast<int>('<')},
                                              {static_cast<int>('x'), static_cast<int>('y')},
                                              /*input_length=*/0);

    const int            P     = 4;
    const size_t         W     = SpecLogitsProcessor::bitmaskWordCount(128);
    std::vector<int32_t> draft = {
        static_cast<int32_t>('b'),
        static_cast<int32_t>('x'),
        static_cast<int32_t>('y'),
        static_cast<int32_t>('b'),
    };
    std::vector<int32_t> bitmask((P + 1) * W, SpecLogitsProcessor::kBitmaskAllowAll);

    SpecLogitsProcessorRequest request;
    request.draft_tokens       = draft.data();
    request.propose_step       = P;
    request.bitmask_cpu_out    = bitmask.data();
    request.bitmask_size_int32 = W;
    request.vocab_size         = 128;

    EXPECT_EQ(processor.tryAcceptAndFillBitmask(request), 3);
    EXPECT_TRUE(packedBitmaskAllowsToken(bitmask.data(), static_cast<int32_t>('b')));
    EXPECT_FALSE(packedBitmaskAllowsToken(bitmask.data(), 0));
    EXPECT_TRUE(packedBitmaskAllowsToken(bitmask.data() + W, static_cast<int32_t>('x')));
    EXPECT_TRUE(packedBitmaskAllowsToken(bitmask.data() + 2 * W, static_cast<int32_t>('y')));
    EXPECT_FALSE(packedBitmaskAllowsToken(bitmask.data() + 2 * W, static_cast<int32_t>('a')));
    EXPECT_TRUE(packedBitmaskAllowsToken(bitmask.data() + 3 * W, static_cast<int32_t>('a')));
    EXPECT_FALSE(packedBitmaskAllowsToken(bitmask.data() + 3 * W, static_cast<int32_t>('b')));
    EXPECT_EQ(processor.acceptedTokenLen(), 0);
}

TEST(ReasoningGrammarLogitsProcessorTest, BudgetForceCloseThenGrammar) {
    auto backend  = makeBackend();
    auto compiled = backend.compileNow({"regex", "a"}).compiled;
    ASSERT_TRUE(compiled);

    auto matcher = backend.createMatcher(compiled, /*require_reasoning=*/false, std::nullopt);
    ReasoningGrammarLogitsProcessor processor(matcher,
                                              /*eos_token_id=*/0,
                                              /*max_thinking_tokens=*/4,
                                              {7},
                                              {8, 9},
                                              /*input_length=*/0);

    EXPECT_EQ(processor.finishedThinkOutputLen(), -1);
    processor.updateStatus(torch::tensor({{5}}, torch::kInt32), 1);

    SamplerInputs inputs;
    inputs.logits           = torch::zeros({1, 128}, torch::kFloat32);
    inputs.finished_mask    = torch::zeros({1}, torch::kBool);
    inputs.input_lengths    = torch::tensor({0}, torch::kInt32);
    inputs.sequence_lengths = torch::tensor({1}, torch::kInt32);
    inputs.vocab_size       = 128;
    processor.process(inputs, 0, 1);

    EXPECT_EQ(inputs.logits[0][8].item<float>(), 1.0f);
    EXPECT_EQ(inputs.logits[0][9].item<float>(), BaseLogitsProcessor::neg_inf);

    processor.updateStatus(torch::tensor({{8}}, torch::kInt32), 1);
    inputs.logits           = torch::zeros({1, 128}, torch::kFloat32);
    inputs.sequence_lengths = torch::tensor({2}, torch::kInt32);
    processor.process(inputs, 0, 1);

    EXPECT_EQ(inputs.logits[0][8].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(inputs.logits[0][9].item<float>(), 1.0f);

    processor.updateStatus(torch::tensor({{9}}, torch::kInt32), 1);
    EXPECT_EQ(processor.finishedThinkOutputLen(), 3);
    inputs.logits           = torch::zeros({1, 128}, torch::kFloat32);
    inputs.sequence_lengths = torch::tensor({3}, torch::kInt32);
    processor.process(inputs, 0, 1);

    EXPECT_GT(inputs.logits[0][static_cast<int>('a')].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(inputs.logits[0][static_cast<int>('b')].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(processor.acceptedTokenLen(), 3);
}

TEST(LogitsProcessorFactoryTest, GrammarThinkingCreatesReasoningGrammarAndSkipsThinkMode) {
    GrammarConfig grammar_config;
    grammar_config.grammar_backend     = "xgrammar";
    grammar_config.tokenizer_info_json = makeTokenizerInfoJson();
    LogitsProcessorFactory::init("", "", grammar_config);

    auto generate_input                                    = std::make_shared<GenerateInput>();
    generate_input->generate_config                        = std::make_shared<GenerateConfig>();
    generate_input->generate_config->response_format       = R"({"type":"json_object"})";
    generate_input->generate_config->max_new_tokens        = 4;
    generate_input->generate_config->in_think_mode         = true;
    generate_input->generate_config->max_thinking_tokens   = -1;
    generate_input->generate_config->begin_think_token_ids = {static_cast<int>('<')};
    generate_input->generate_config->end_think_token_ids   = {static_cast<int>('x'), static_cast<int>('y')};
    generate_input->input_ids                              = torch::tensor({1, 2}, torch::kInt32);

    auto processors = LogitsProcessorFactory::createLogitsProcessors(
        generate_input, /*init_batch_size=*/1, /*max_batch_size=*/1, /*eos_token_id=*/0);

    ASSERT_EQ(processors.size(), 1);
    EXPECT_EQ(generate_input->generate_config->max_thinking_tokens, -1);
    auto reasoning_processor = std::dynamic_pointer_cast<ReasoningGrammarLogitsProcessor>(processors[0]);
    ASSERT_NE(reasoning_processor, nullptr);
    EXPECT_EQ(std::dynamic_pointer_cast<GrammarLogitsProcessor>(processors[0]), nullptr);
    EXPECT_EQ(std::dynamic_pointer_cast<ThinkModeLogitsProcessor>(processors[0]), nullptr);

    const int            propose_step = 3;
    const size_t         words        = SpecLogitsProcessor::bitmaskWordCount(128);
    std::vector<int32_t> draft_tokens = {
        static_cast<int32_t>('a'), static_cast<int32_t>('x'), static_cast<int32_t>('y')};
    std::vector<int32_t> bitmask((propose_step + 1) * words, SpecLogitsProcessor::kBitmaskAllowAll);

    SpecLogitsProcessorRequest request;
    request.draft_tokens       = draft_tokens.data();
    request.propose_step       = propose_step;
    request.bitmask_cpu_out    = bitmask.data();
    request.bitmask_size_int32 = words;
    request.vocab_size         = 128;

    EXPECT_EQ(reasoning_processor->tryAcceptAndFillBitmask(request), propose_step);
    reasoning_processor->updateStatus(torch::tensor(draft_tokens, torch::kInt32).reshape({1, propose_step}),
                                      propose_step);
    EXPECT_EQ(reasoning_processor->finishedThinkOutputLen(), propose_step);
}

TEST(LogitsProcessorFactoryTest, GrammarThinkingWithoutEndIdsReportsInvalidParams) {
    GrammarConfig grammar_config;
    grammar_config.grammar_backend     = "xgrammar";
    grammar_config.tokenizer_info_json = makeTokenizerInfoJson();
    LogitsProcessorFactory::init("", "", grammar_config);

    auto generate_input                                    = std::make_shared<GenerateInput>();
    generate_input->generate_config                        = std::make_shared<GenerateConfig>();
    generate_input->generate_config->response_format       = R"({"type":"json_object"})";
    generate_input->generate_config->in_think_mode         = true;
    generate_input->generate_config->max_thinking_tokens   = 32;
    generate_input->generate_config->begin_think_token_ids = {static_cast<int>('<')};
    generate_input->input_ids                              = torch::tensor({1, 2}, torch::kInt32);

    bool        reported = false;
    ErrorCode   code     = ErrorCode::UNKNOWN_ERROR;
    std::string message;
    auto        processors = LogitsProcessorFactory::createLogitsProcessors(
        generate_input,
        /*init_batch_size=*/1,
        /*max_batch_size=*/1,
        /*eos_token_id=*/0,
        [&](ErrorCode error_code, const std::string& error_message, bool) {
            reported = true;
            code     = error_code;
            message  = error_message;
        });

    EXPECT_TRUE(processors.empty());
    EXPECT_TRUE(reported);
    EXPECT_EQ(code, ErrorCode::INVALID_PARAMS);
    EXPECT_NE(message.find("end_think_token_ids"), std::string::npos);
}

TEST(LogitsProcessorFactoryTest, GrammarThinkingBackendMissingSuppressesThinkModeFallback) {
    GrammarConfig grammar_config;
    grammar_config.grammar_backend = "none";
    LogitsProcessorFactory::init("", "", grammar_config);

    auto generate_input                                    = std::make_shared<GenerateInput>();
    generate_input->generate_config                        = std::make_shared<GenerateConfig>();
    generate_input->generate_config->response_format       = R"({"type":"json_object"})";
    generate_input->generate_config->in_think_mode         = true;
    generate_input->generate_config->max_thinking_tokens   = 32;
    generate_input->generate_config->begin_think_token_ids = {static_cast<int>('<')};
    generate_input->generate_config->end_think_token_ids   = {static_cast<int>('x'), static_cast<int>('y')};
    generate_input->input_ids                              = torch::tensor({1, 2}, torch::kInt32);

    bool        reported = false;
    ErrorCode   code     = ErrorCode::UNKNOWN_ERROR;
    std::string message;
    auto        processors = LogitsProcessorFactory::createLogitsProcessors(
        generate_input,
        /*init_batch_size=*/1,
        /*max_batch_size=*/1,
        /*eos_token_id=*/0,
        [&](ErrorCode error_code, const std::string& error_message, bool) {
            reported = true;
            code     = error_code;
            message  = error_message;
        });

    EXPECT_TRUE(processors.empty());
    EXPECT_TRUE(reported);
    EXPECT_EQ(code, ErrorCode::INVALID_PARAMS);
    EXPECT_NE(message.find("xgrammar backend is not initialized"), std::string::npos);
}

TEST(LogitsProcessorFactoryTest, GrammarThinkingInvalidResponseFormatSuppressesThinkModeFallback) {
    auto generate_input                                    = std::make_shared<GenerateInput>();
    generate_input->generate_config                        = std::make_shared<GenerateConfig>();
    generate_input->generate_config->response_format       = "{invalid-json";
    generate_input->generate_config->in_think_mode         = true;
    generate_input->generate_config->max_thinking_tokens   = 32;
    generate_input->generate_config->begin_think_token_ids = {static_cast<int>('<')};
    generate_input->generate_config->end_think_token_ids   = {static_cast<int>('x'), static_cast<int>('y')};
    generate_input->input_ids                              = torch::tensor({1, 2}, torch::kInt32);

    bool        reported = false;
    ErrorCode   code     = ErrorCode::UNKNOWN_ERROR;
    std::string message;
    auto        processors = LogitsProcessorFactory::createLogitsProcessors(
        generate_input,
        /*init_batch_size=*/1,
        /*max_batch_size=*/1,
        /*eos_token_id=*/0,
        [&](ErrorCode error_code, const std::string& error_message, bool) {
            reported = true;
            code     = error_code;
            message  = error_message;
        });

    EXPECT_TRUE(processors.empty());
    EXPECT_TRUE(reported);
    EXPECT_EQ(code, ErrorCode::INVALID_PARAMS);
    EXPECT_NE(message.find("invalid grammar response_format"), std::string::npos);
}

}  // namespace
}  // namespace rtp_llm
