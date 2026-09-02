#include "rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include <xgrammar/tokenizer_info.h>

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
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

std::string makeKimiK3TokenizerInfoJson() {
    std::vector<std::string> vocab;
    vocab.reserve(260);
    for (int i = 0; i < 256; ++i) {
        vocab.emplace_back(1, static_cast<char>(i));
    }
    vocab.emplace_back("<|open|>");
    vocab.emplace_back("<|close|>");
    vocab.emplace_back("<|sep|>");
    vocab.emplace_back("<|end_of_msg|>");
    xgrammar::TokenizerInfo info(vocab,
                                 xgrammar::VocabType::RAW,
                                 /*vocab_size=*/260,
                                 /*stop_token_ids=*/std::vector<int32_t>{259});
    return info.SerializeJSON();
}

XGrammarBackendCpp makeBackend() {
    XGrammarBackendOptions options;
    options.max_compiler_threads = 1;
    return XGrammarBackendCpp(makeTokenizerInfoJson(), options);
}

XGrammarBackendCpp makeKimiK3Backend() {
    XGrammarBackendOptions options;
    options.max_compiler_threads = 1;
    return XGrammarBackendCpp(makeKimiK3TokenizerInfoJson(), options);
}

struct ModelThinkBoundaryCase {
    const char*      name;
    std::vector<int> begin_think_token_ids;
    std::vector<int> end_think_token_ids;
    int              vocab_size;
};

class ReasoningGrammarModelBoundaryTest: public testing::TestWithParam<ModelThinkBoundaryCase> {};

std::string modelThinkBoundaryCaseName(const testing::TestParamInfo<ModelThinkBoundaryCase>& info) {
    return info.param.name;
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

TEST(GrammarLogitsProcessorTest, CompilesKimiK3XmlArguments) {
    auto backend = makeKimiK3Backend();
    auto compiled = backend.compileNow(
        {"structural_tag",
         R"({"type":"structural_tag","format":{"type":"json_schema","json_schema":{"type":"object",)"
         R"("properties":{"name":{"type":"string"}},"required":["name"],"additionalProperties":false},)"
         R"("style":"kimi_k3_xml"}})"})
                        .compiled;
    ASSERT_TRUE(compiled);

    auto matcher = backend.createMatcher(compiled, false, std::nullopt);
    ASSERT_TRUE(matcher->acceptToken(256));
    for (char c : std::string("argument key=\"name\" type=\"string\"")) {
        ASSERT_TRUE(matcher->acceptToken(static_cast<uint8_t>(c)));
    }
    ASSERT_TRUE(matcher->acceptToken(258));
    for (char c : std::string("Bob")) {
        ASSERT_TRUE(matcher->acceptToken(static_cast<uint8_t>(c)));
    }
    ASSERT_TRUE(matcher->acceptToken(257));
    for (char c : std::string("argument")) {
        ASSERT_TRUE(matcher->acceptToken(static_cast<uint8_t>(c)));
    }
    ASSERT_TRUE(matcher->acceptToken(258));
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

TEST(ReasoningGrammarLogitsProcessorTest, ZeroBudgetStartsInGrammarMode) {
    auto backend  = makeBackend();
    auto compiled = backend.compileNow({"json", R"({"type":"object"})"}).compiled;
    ASSERT_TRUE(compiled);

    auto matcher = backend.createMatcher(
        compiled, /*require_reasoning=*/false, std::nullopt, /*terminate_without_stop_token=*/true);
    ReasoningGrammarLogitsProcessor processor(matcher,
                                              /*eos_token_id=*/0,
                                              /*max_thinking_tokens=*/0,
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

    EXPECT_GT(inputs.logits[0][static_cast<int>('{')].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(inputs.logits[0][static_cast<int>('a')].item<float>(), BaseLogitsProcessor::neg_inf);
}

TEST(ReasoningGrammarLogitsProcessorTest, MasksReasoningStopsAndThinkBeginFirstToken) {
    auto backend  = makeBackend();
    auto compiled = backend.compileNow({"json", R"({"type":"object"})"}).compiled;
    ASSERT_TRUE(compiled);

    auto matcher = backend.createMatcher(
        compiled, /*require_reasoning=*/false, std::nullopt, /*terminate_without_stop_token=*/true);
    ReasoningGrammarLogitsProcessor processor(matcher,
                                              /*eos_token_id=*/0,
                                              /*max_thinking_tokens=*/32,
                                              {static_cast<int>('{'), static_cast<int>('q')},
                                              {static_cast<int>('x'), static_cast<int>('y')},
                                              /*input_length=*/0,
                                              /*error_reporter=*/nullptr,
                                              /*stop_words_list=*/{{static_cast<int>('!')}},
                                              /*model_type=*/"kimi_k3");

    SamplerInputs inputs;
    inputs.logits           = torch::zeros({1, 128}, torch::kFloat32);
    inputs.finished_mask    = torch::zeros({1}, torch::kBool);
    inputs.input_lengths    = torch::tensor({0}, torch::kInt32);
    inputs.sequence_lengths = torch::tensor({0}, torch::kInt32);
    inputs.vocab_size       = 128;
    processor.process(inputs, 0, 1);

    EXPECT_EQ(inputs.logits[0][0].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(inputs.logits[0][static_cast<int>('!')].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(inputs.logits[0][static_cast<int>('{')].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(inputs.logits[0][static_cast<int>('x')].item<float>(), 0.0f);

    processor.updateStatus(torch::tensor({{static_cast<int32_t>('{')}}, torch::kInt32), 1);
    inputs.logits.zero_();
    inputs.sequence_lengths = torch::tensor({1}, torch::kInt32);
    processor.process(inputs, 0, 1);
    EXPECT_EQ(inputs.logits[0][static_cast<int>('q')].item<float>(), 0.0f);

    processor.updateStatus(torch::tensor({{static_cast<int32_t>('x'), static_cast<int32_t>('y')}}, torch::kInt32), 2);
    inputs.logits.zero_();
    inputs.sequence_lengths = torch::tensor({3}, torch::kInt32);
    processor.process(inputs, 0, 1);

    EXPECT_GT(inputs.logits[0][static_cast<int>('{')].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(inputs.logits[0][static_cast<int>('a')].item<float>(), BaseLogitsProcessor::neg_inf);
}

TEST(ReasoningGrammarLogitsProcessorTest, SpecMasksAllSingleTokenReasoningStops) {
    auto backend  = makeBackend();
    auto compiled = backend.compileNow({"regex", "a"}).compiled;
    ASSERT_TRUE(compiled);

    auto matcher = backend.createMatcher(compiled, /*require_reasoning=*/false, std::nullopt);
    ReasoningGrammarLogitsProcessor processor(matcher,
                                              /*eos_token_id=*/0,
                                              /*max_thinking_tokens=*/32,
                                              {7, 1},
                                              {8, 2},
                                              /*input_length=*/0,
                                              /*error_reporter=*/nullptr,
                                              /*stop_words_list=*/{{10}},
                                              /*model_type=*/"kimi_k3");

    const size_t         words = SpecLogitsProcessor::bitmaskWordCount(16);
    std::vector<int32_t> draft = {10};
    std::vector<int32_t> bitmask(2 * words, SpecLogitsProcessor::kBitmaskAllowAll);
    SpecLogitsProcessorRequest request;
    request.draft_tokens       = draft.data();
    request.propose_step       = 1;
    request.bitmask_cpu_out    = bitmask.data();
    request.bitmask_size_int32 = words;
    request.vocab_size         = 16;

    EXPECT_EQ(0, processor.tryAcceptAndFillBitmask(request));
    EXPECT_FALSE(packedBitmaskAllowsToken(bitmask.data(), 0));
    EXPECT_FALSE(packedBitmaskAllowsToken(bitmask.data(), 10));
    EXPECT_FALSE(packedBitmaskAllowsToken(bitmask.data(), 7));
    EXPECT_TRUE(packedBitmaskAllowsToken(bitmask.data(), 8));
}

TEST(ThinkModeLogitsProcessorTest, KimiK3NaturalBoundaryAndReasoningStops) {
    constexpr int kEosTokenId     = 163585;
    constexpr int kEndOfMessageId = 163586;
    constexpr int kOpenTokenId    = 163587;
    constexpr int kCloseTokenId   = 163588;
    auto generate_input = std::make_shared<GenerateInput>();
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    generate_input->generate_config->in_think_mode         = true;
    generate_input->generate_config->max_thinking_tokens   = 32;
    generate_input->generate_config->begin_think_token_ids = {kOpenTokenId, 39964, 163589};
    generate_input->generate_config->end_think_token_ids = {
        kCloseTokenId, 39964, 163589, kOpenTokenId, 12092, 163589};
    generate_input->generate_config->stop_words_list = {{kEndOfMessageId}};
    generate_input->input_ids = torch::tensor({1, 2, 3}, torch::kInt32);

    auto processor = ThinkModeLogitsProcessor::fromGenerateInput(
        generate_input, 1, kEosTokenId, "kimi_k3");
    ASSERT_NE(processor, nullptr);

    SamplerInputs inputs;
    inputs.logits           = torch::zeros({1, 163600}, torch::kFloat32);
    inputs.input_lengths    = torch::tensor({3}, torch::kInt32);
    inputs.sequence_lengths = torch::tensor({3}, torch::kInt32);
    inputs.vocab_size       = 163600;
    processor->process(inputs, 0, 1);

    EXPECT_EQ(inputs.logits[0][kEosTokenId].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(inputs.logits[0][kEndOfMessageId].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(inputs.logits[0][kOpenTokenId].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(inputs.logits[0][kCloseTokenId].item<float>(), 0.0f);

    processor->updateStatus(torch::tensor({{kCloseTokenId, 39964, 163589, kOpenTokenId, 12092, 163589}},
                                          torch::kInt32),
                            6);
    inputs.logits.zero_();
    inputs.sequence_lengths = torch::tensor({9}, torch::kInt32);
    processor->process(inputs, 0, 1);

    EXPECT_EQ(inputs.logits[0][kEosTokenId].item<float>(), 0.0f);
    EXPECT_EQ(inputs.logits[0][kEndOfMessageId].item<float>(), 0.0f);
    EXPECT_EQ(inputs.logits[0][kOpenTokenId].item<float>(), 0.0f);
    EXPECT_EQ(inputs.logits[0][kCloseTokenId].item<float>(), 0.0f);
}

TEST(ThinkModeLogitsProcessorTest, SpecMasksReasoningStops) {
    auto generate_input = std::make_shared<GenerateInput>();
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    generate_input->generate_config->in_think_mode         = true;
    generate_input->generate_config->max_thinking_tokens   = 32;
    generate_input->generate_config->begin_think_token_ids = {7, 1};
    generate_input->generate_config->end_think_token_ids   = {8, 2};
    generate_input->generate_config->stop_words_list       = {{10}};
    generate_input->input_ids = torch::tensor({1, 2, 3}, torch::kInt32);

    auto processor = ThinkModeLogitsProcessor::fromGenerateInput(
        generate_input, 1, /*eos_token_id=*/11, "kimi_k3");
    ASSERT_NE(processor, nullptr);

    const size_t         words = SpecLogitsProcessor::bitmaskWordCount(16);
    std::vector<int32_t> draft = {10};
    std::vector<int32_t> bitmask(2 * words, SpecLogitsProcessor::kBitmaskAllowAll);
    SpecLogitsProcessorRequest request;
    request.draft_tokens       = draft.data();
    request.propose_step       = 1;
    request.bitmask_cpu_out    = bitmask.data();
    request.bitmask_size_int32 = words;
    request.vocab_size         = 16;

    EXPECT_EQ(0, processor->tryAcceptAndFillBitmask(request));
    EXPECT_FALSE(packedBitmaskAllowsToken(bitmask.data(), 10));
    EXPECT_FALSE(packedBitmaskAllowsToken(bitmask.data(), 11));
    EXPECT_FALSE(packedBitmaskAllowsToken(bitmask.data(), 7));
    EXPECT_TRUE(packedBitmaskAllowsToken(bitmask.data(), 8));
}

TEST(ThinkModeLogitsProcessorTest, K3GuardMasksThinkTokenAfterSharedPrefix) {
    auto generate_input = std::make_shared<GenerateInput>();
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    generate_input->generate_config->in_think_mode         = false;
    generate_input->generate_config->begin_think_token_ids = {7, 1};
    generate_input->generate_config->end_think_token_ids   = {8, 2};
    generate_input->input_ids = torch::tensor({3}, torch::kInt32);

    auto processor = ThinkModeLogitsProcessor::fromGenerateInput(
        generate_input, 1, /*eos_token_id=*/-1, "kimi_k3");
    ASSERT_NE(processor, nullptr);

    SamplerInputs inputs;
    inputs.logits     = torch::zeros({1, 16}, torch::kFloat32);
    inputs.vocab_size = 16;
    processor->process(inputs, 0, 1);
    EXPECT_EQ(inputs.logits[0][7].item<float>(), 0.0f);
    EXPECT_EQ(inputs.logits[0][8].item<float>(), 0.0f);

    processor->updateStatus(torch::tensor({{7}}, torch::kInt32), 1);
    inputs.logits.zero_();
    processor->process(inputs, 0, 1);
    EXPECT_EQ(inputs.logits[0][7].item<float>(), 0.0f);
    EXPECT_EQ(inputs.logits[0][1].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(inputs.logits[0][8].item<float>(), 0.0f);

    processor->updateStatus(torch::tensor({{8}}, torch::kInt32), 1);
    inputs.logits.zero_();
    processor->process(inputs, 0, 1);
    EXPECT_EQ(inputs.logits[0][2].item<float>(), BaseLogitsProcessor::neg_inf);
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

TEST_P(ReasoningGrammarModelBoundaryTest, NaturalBoundaryCompletesBeforeGrammar) {
    const auto& profile = GetParam();
    auto        backend = makeBackend();
    auto        compiled = backend.compileNow({"regex", "a"}).compiled;
    ASSERT_TRUE(compiled);

    auto matcher = backend.createMatcher(
        compiled, /*require_reasoning=*/false, std::nullopt, /*terminate_without_stop_token=*/true);
    bool reported = false;
    ReasoningGrammarLogitsProcessor processor(matcher,
                                              /*eos_token_id=*/0,
                                              /*max_thinking_tokens=*/32,
                                              profile.begin_think_token_ids,
                                              profile.end_think_token_ids,
                                              /*input_length=*/0,
                                              [&reported](ErrorCode, const std::string&, bool) { reported = true; });

    processor.updateStatus(torch::tensor(profile.end_think_token_ids, torch::kInt32).unsqueeze(0),
                           static_cast<int32_t>(profile.end_think_token_ids.size()));
    EXPECT_FALSE(reported);
    EXPECT_EQ(processor.acceptedTokenLen(), profile.end_think_token_ids.size());

    SamplerInputs inputs;
    inputs.logits           = torch::zeros({1, profile.vocab_size}, torch::kFloat32);
    inputs.finished_mask    = torch::zeros({1}, torch::kBool);
    inputs.input_lengths    = torch::tensor({0}, torch::kInt32);
    inputs.sequence_lengths = torch::tensor({static_cast<int>(profile.end_think_token_ids.size())}, torch::kInt32);
    inputs.vocab_size       = profile.vocab_size;
    processor.process(inputs, 0, 1);

    EXPECT_GT(inputs.logits[0][static_cast<int>('a')].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(inputs.logits[0][static_cast<int>('b')].item<float>(), BaseLogitsProcessor::neg_inf);

    processor.updateStatus(torch::tensor({{static_cast<int32_t>('a')}}, torch::kInt32), 1);
    EXPECT_FALSE(reported);
    EXPECT_EQ(processor.acceptedTokenLen(), profile.end_think_token_ids.size() + 1);
}

TEST_P(ReasoningGrammarModelBoundaryTest, NaturalCloseForcesTrailingBoundaryToken) {
    const auto& profile = GetParam();
    ASSERT_EQ(profile.end_think_token_ids.size(), 2);
    auto backend  = makeBackend();
    auto compiled = backend.compileNow({"regex", "a"}).compiled;
    ASSERT_TRUE(compiled);

    auto matcher = backend.createMatcher(
        compiled, /*require_reasoning=*/false, std::nullopt, /*terminate_without_stop_token=*/true);
    bool reported = false;
    ReasoningGrammarLogitsProcessor processor(matcher,
                                              /*eos_token_id=*/0,
                                              /*max_thinking_tokens=*/32,
                                              profile.begin_think_token_ids,
                                              profile.end_think_token_ids,
                                              /*input_length=*/0,
                                              [&reported](ErrorCode, const std::string&, bool) { reported = true; });

    processor.updateStatus(torch::tensor({{profile.end_think_token_ids.front()}}, torch::kInt32), 1);

    SamplerInputs inputs;
    inputs.logits           = torch::zeros({1, profile.vocab_size}, torch::kFloat32);
    inputs.finished_mask    = torch::zeros({1}, torch::kBool);
    inputs.input_lengths    = torch::tensor({0}, torch::kInt32);
    inputs.sequence_lengths = torch::tensor({1}, torch::kInt32);
    inputs.vocab_size       = profile.vocab_size;
    processor.process(inputs, 0, 1);

    const int trailing_token_id = profile.end_think_token_ids.back();
    EXPECT_EQ(inputs.logits[0][trailing_token_id].item<float>(), 1.0f);
    EXPECT_EQ(inputs.logits[0][static_cast<int>('a')].item<float>(), BaseLogitsProcessor::neg_inf);

    processor.updateStatus(torch::tensor({{trailing_token_id}}, torch::kInt32), 1);
    inputs.logits.zero_();
    inputs.sequence_lengths = torch::tensor({2}, torch::kInt32);
    processor.process(inputs, 0, 1);

    EXPECT_FALSE(reported);
    EXPECT_GT(inputs.logits[0][static_cast<int>('a')].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(inputs.logits[0][static_cast<int>('b')].item<float>(), BaseLogitsProcessor::neg_inf);
}

INSTANTIATE_TEST_SUITE_P(
    ModelBoundaries,
    ReasoningGrammarModelBoundaryTest,
    testing::Values(ModelThinkBoundaryCase{"DeepSeekV4", {128821, 198}, {128822, 271}, 248100},
                    ModelThinkBoundaryCase{"Qwen35", {248068, 198}, {248069, 271}, 248100}),
    modelThinkBoundaryCaseName);

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
                                              /*max_thinking_tokens=*/1,
                                              {7},
                                              {8, 9},
                                              /*input_length=*/0);

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
    ModelConfig model_config;
    LogitsProcessorFactory::init(model_config, "", grammar_config);

    auto generate_input                                    = std::make_shared<GenerateInput>();
    generate_input->generate_config                        = std::make_shared<GenerateConfig>();
    generate_input->generate_config->response_format       = R"({"type":"json_object"})";
    generate_input->generate_config->in_think_mode         = true;
    generate_input->generate_config->max_thinking_tokens   = 32;
    generate_input->generate_config->begin_think_token_ids = {static_cast<int>('<')};
    generate_input->generate_config->end_think_token_ids   = {static_cast<int>('x'), static_cast<int>('y')};
    generate_input->input_ids                              = torch::tensor({1, 2}, torch::kInt32);

    auto processors = LogitsProcessorFactory::createLogitsProcessors(
        generate_input, /*init_batch_size=*/1, /*max_batch_size=*/1, /*eos_token_id=*/0, model_config.model_type);

    ASSERT_EQ(processors.size(), 1);
    EXPECT_NE(std::dynamic_pointer_cast<ReasoningGrammarLogitsProcessor>(processors[0]), nullptr);
    EXPECT_EQ(std::dynamic_pointer_cast<GrammarLogitsProcessor>(processors[0]), nullptr);
    EXPECT_EQ(std::dynamic_pointer_cast<ThinkModeLogitsProcessor>(processors[0]), nullptr);
}

TEST(LogitsProcessorFactoryTest, GrammarWithoutThinkingSkipsThinkMode) {
    GrammarConfig grammar_config;
    grammar_config.grammar_backend     = "xgrammar";
    grammar_config.tokenizer_info_json = makeTokenizerInfoJson();
    ModelConfig model_config;
    model_config.model_type            = "kimi_k3";
    LogitsProcessorFactory::init(model_config, "", grammar_config);

    auto generate_input                              = std::make_shared<GenerateInput>();
    generate_input->generate_config                  = std::make_shared<GenerateConfig>();
    generate_input->generate_config->response_format = R"({"type":"json_object"})";
    generate_input->input_ids                        = torch::tensor({1, 2}, torch::kInt32);

    auto processors = LogitsProcessorFactory::createLogitsProcessors(
        generate_input, /*init_batch_size=*/1, /*max_batch_size=*/1, /*eos_token_id=*/0, model_config.model_type);

    ASSERT_EQ(processors.size(), 1);
    EXPECT_NE(std::dynamic_pointer_cast<GrammarLogitsProcessor>(processors[0]), nullptr);
    EXPECT_EQ(std::dynamic_pointer_cast<ThinkModeLogitsProcessor>(processors[0]), nullptr);
}

TEST(LogitsProcessorFactoryTest, KimiK3RequestBoundariesGuardThinkTagWithoutRenderer) {
    constexpr int kOpenTokenId  = 10;
    constexpr int kCloseTokenId = 11;
    constexpr int kThinkTokenId = 12;
    constexpr int kSepTokenId   = 13;

    ModelConfig model_config;
    model_config.model_type = "kimi_k3";
    GrammarConfig grammar_config;
    grammar_config.grammar_backend = "none";
    LogitsProcessorFactory::init(model_config, "", grammar_config);

    auto generate_input             = std::make_shared<GenerateInput>();
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    generate_input->generate_config->begin_think_token_ids = {kOpenTokenId, kThinkTokenId, kSepTokenId};
    generate_input->generate_config->end_think_token_ids   = {kCloseTokenId, kThinkTokenId, kSepTokenId};
    generate_input->input_ids       = torch::tensor({1, 2}, torch::kInt32);

    auto processors = LogitsProcessorFactory::createLogitsProcessors(
        generate_input, /*init_batch_size=*/1, /*max_batch_size=*/1, /*eos_token_id=*/0, model_config.model_type);
    ASSERT_EQ(processors.size(), 1);
    auto processor = std::dynamic_pointer_cast<ThinkModeLogitsProcessor>(processors[0]);
    ASSERT_NE(processor, nullptr);

    SamplerInputs inputs;
    inputs.logits     = torch::zeros({1, 32}, torch::kFloat32);
    inputs.vocab_size = 32;
    processor->process(inputs, 0, 1);
    EXPECT_EQ(inputs.logits[0][kOpenTokenId].item<float>(), 0.0f);
    EXPECT_EQ(inputs.logits[0][kCloseTokenId].item<float>(), 0.0f);

    processor->updateStatus(torch::tensor({{kOpenTokenId}}, torch::kInt32), 1);
    inputs.logits.zero_();
    processor->process(inputs, 0, 1);
    EXPECT_EQ(inputs.logits[0][kThinkTokenId].item<float>(), BaseLogitsProcessor::neg_inf);
}

TEST(LogitsProcessorFactoryTest, Eagle3DraftTypeUsesFirstTokenGuard) {
    constexpr int kOpenTokenId  = 10;
    constexpr int kCloseTokenId = 11;
    constexpr int kThinkTokenId = 12;
    constexpr int kSepTokenId   = 13;

    ModelConfig model_config;
    model_config.model_type = "kimi_k3_mla_swa_eagle3";
    GrammarConfig grammar_config;
    grammar_config.grammar_backend = "none";
    LogitsProcessorFactory::init(model_config, "", grammar_config);

    auto generate_input             = std::make_shared<GenerateInput>();
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    generate_input->generate_config->begin_think_token_ids = {kOpenTokenId, kThinkTokenId, kSepTokenId};
    generate_input->generate_config->end_think_token_ids   = {kCloseTokenId, kThinkTokenId, kSepTokenId};
    generate_input->input_ids       = torch::tensor({1, 2}, torch::kInt32);

    auto processors = LogitsProcessorFactory::createLogitsProcessors(
        generate_input, /*init_batch_size=*/1, /*max_batch_size=*/1, /*eos_token_id=*/0, model_config.model_type);
    ASSERT_EQ(processors.size(), 1);
    auto processor = std::dynamic_pointer_cast<ThinkModeLogitsProcessor>(processors[0]);
    ASSERT_NE(processor, nullptr);

    SamplerInputs inputs;
    inputs.logits     = torch::zeros({1, 32}, torch::kFloat32);
    inputs.vocab_size = 32;
    processor->process(inputs, 0, 1);
    EXPECT_EQ(inputs.logits[0][kOpenTokenId].item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(inputs.logits[0][kCloseTokenId].item<float>(), BaseLogitsProcessor::neg_inf);
}

TEST(LogitsProcessorFactoryTest, GrammarThinkingWithoutEndIdsReportsInvalidParams) {
    GrammarConfig grammar_config;
    grammar_config.grammar_backend     = "xgrammar";
    grammar_config.tokenizer_info_json = makeTokenizerInfoJson();
    ModelConfig model_config;
    LogitsProcessorFactory::init(model_config, "", grammar_config);

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
        model_config.model_type,
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
    ModelConfig model_config;
    LogitsProcessorFactory::init(model_config, "", grammar_config);

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
        model_config.model_type,
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
        std::string{},
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
