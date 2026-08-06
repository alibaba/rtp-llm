// Unit tests for GrammarLogitsProcessor over a 128-char ASCII vocab.

#include "rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.h"
#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackend.h"
#include "rtp_llm/cpp/models/logits_processor/BitmaskUtils.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <xgrammar/tokenizer_info.h>

namespace rtp_llm {

static_assert(std::is_base_of_v<BaseLogitsProcessor, GrammarLogitsProcessor>);
namespace {

xgrammar::TokenizerInfo makeAsciiTokenizerInfo(int vocab_size = 128) {
    std::vector<std::string> vocab;
    vocab.reserve(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        vocab.emplace_back(1, static_cast<char>(i));
    }
    return xgrammar::TokenizerInfo(vocab,
                                   xgrammar::VocabType::RAW,
                                   vocab_size,
                                   /*stop_token_ids=*/std::vector<int32_t>{0});
}

GrammarConfig grammarConfig(bool terminate_without_stop_token = true) {
    GrammarConfig cfg;
    cfg.num_workers                  = 2;
    cfg.compiler_cache_bytes         = -1;
    cfg.terminate_without_stop_token = terminate_without_stop_token;
    return cfg;
}

std::shared_ptr<XGrammarBackend> makeBackend(bool terminate_without_stop_token = true, int vocab_size = 128) {
    return XGrammarBackend::create(makeAsciiTokenizerInfo(vocab_size).SerializeJSON(),
                                   grammarConfig(terminate_without_stop_token));
}

struct ProcessorBundle {
    std::shared_ptr<GrammarLogitsProcessor> proc;
    std::shared_ptr<RtpGrammarMatcher>      matcher;

    GrammarLogitsProcessor* operator->() const noexcept {
        return proc.get();
    }
};

ProcessorBundle makeProcessorFromKey(const std::shared_ptr<XGrammarBackend>& backend,
                                     const GrammarKeyCpp&                    key,
                                     int64_t                                 eos_token_id = 0) {
    EXPECT_TRUE(backend);
    if (!backend) {
        return {};
    }
    auto matcher_or = backend->createMatcherFromKey(key);
    EXPECT_TRUE(matcher_or.ok()) << matcher_or.status().ToString();
    if (!matcher_or.ok()) {
        return {};
    }
    auto matcher = matcher_or.value();
    auto proc    = std::make_shared<GrammarLogitsProcessor>(matcher, eos_token_id);
    return {std::move(proc), std::move(matcher)};
}

// The test backend enables terminate_without_stop_token, so the matcher flips IsTerminated()
// the moment the regex completes.
ProcessorBundle makeProcessor(const std::shared_ptr<XGrammarBackend>& backend, const std::string& regex) {
    return makeProcessorFromKey(backend, {"regex", regex});
}

bool rowAllows(const std::vector<int32_t>& bm, size_t words, int row, int token) {
    const int32_t word = bm[static_cast<size_t>(row) * words + token / 32];
    return (static_cast<uint32_t>(word) & (1u << (token % 32))) != 0u;
}

SamplerInputs makeSamplerInputs(torch::Tensor logits) {
    SamplerInputs inputs;
    inputs.logits = std::move(logits);
    return inputs;
}

std::vector<float> logitsVec(const torch::Tensor& logits) {
    auto cpu = logits.to(torch::kFloat32).cpu().contiguous();
    return std::vector<float>(cpu.data_ptr<float>(), cpu.data_ptr<float>() + cpu.numel());
}

void expectTokenAllowed(const std::vector<float>& logits, int token) {
    ASSERT_LT(token, static_cast<int>(logits.size()));
    EXPECT_FLOAT_EQ(logits[token], 0.0f);
}

void expectTokenMasked(const std::vector<float>& logits, int token) {
    ASSERT_LT(token, static_cast<int>(logits.size()));
    EXPECT_LT(logits[token], -1e20f);
}

bool matcherTerminated(const RtpGrammarMatcher& matcher) {
    auto terminated = matcher.isTerminated();
    EXPECT_TRUE(terminated.ok());
    return terminated.ok() && terminated.value();
}

bool acceptMatcherToken(RtpGrammarMatcher& matcher, int32_t token) {
    auto accepted = matcher.acceptToken(token);
    EXPECT_TRUE(accepted.ok());
    return accepted.ok() && accepted.value();
}

std::string makeReasoningStructuralTag(int budget) {
    return R"({"type":"structural_tag","format":{"type":"sequence","elements":[)"
           R"({"type":"tag","begin":"","content":{"type":"any_text","max_tokens":)"
           + std::to_string(budget) + R"(},"end":"z"},{"type":"regex","pattern":"a"}]}})";
}

std::string makeReasoningStructuralTagWithTokenEnd(int budget, int end_token_id) {
    return R"({"type":"structural_tag","format":{"type":"sequence","elements":[)"
           R"({"type":"tag","begin":"","content":{"type":"any_text","max_tokens":)"
           + std::to_string(budget) + R"(},"end":{"type":"token","token":)" + std::to_string(end_token_id)
           + R"(}},{"type":"regex","pattern":"a"}]}})";
}

std::string makeUnboundedAnyTextStructuralTag() {
    return R"({"type":"structural_tag","format":{"type":"any_text"}})";
}

constexpr int kA   = 'a';  // token id 97
constexpr int kB   = 'b';  // token id 98
constexpr int kC   = 'c';  // token id 99
constexpr int kD   = 'd';  // token id 100
constexpr int kX   = 'x';  // token id 120
constexpr int kEos = 0;    // stop token in makeAsciiTokenizerInfo
constexpr int kZ   = 'z';  // structural-tag think end in makeReasoningStructuralTag

int expectCapOk(ErrorResult<int>&& result) {
    EXPECT_TRUE(result.ok()) << result.status().ToString();
    return result.ok() ? result.value() : -1;
}

}  // namespace

TEST(GrammarLogitsProcessorTest, ProcessMasksInitialDecodeState) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);
    auto proc = makeProcessor(backend, "ab");

    auto logits = torch::zeros({1, 128}, torch::kFloat32);
    auto inputs = makeSamplerInputs(logits);
    auto error  = proc->process(inputs, 0, 1);

    auto values = logitsVec(logits);
    expectTokenAllowed(values, kA);
    expectTokenMasked(values, kB);
    expectTokenMasked(values, kX);
    EXPECT_FALSE(error.has_value());
}

TEST(GrammarLogitsProcessorTest, ProcessRejectsGrammarVocabExceedingModelVocab) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);
    auto proc = makeProcessor(backend, "ab");

    auto logits = torch::zeros({1, 127}, torch::kFloat32);
    auto error  = proc->process(makeSamplerInputs(logits), 0, 1);

    ASSERT_TRUE(error.has_value());
    EXPECT_EQ(error->code(), ErrorCode::GRAMMAR_VOCAB_EXCEEDS_MODEL_VOCAB);
}

TEST(GrammarLogitsProcessorTest, ProcessAppliesPackedMaskOnGpuAcrossLogitDtypes) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);

    for (const auto dtype : {torch::kFloat32, torch::kFloat16, torch::kBFloat16}) {
        auto proc   = makeProcessor(backend, "ab");
        auto logits = torch::zeros({1, 128}, torch::TensorOptions().dtype(dtype).device(torch::kCUDA));
        auto inputs = makeSamplerInputs(logits);

        ASSERT_FALSE(proc->process(inputs, 0, 1).has_value());
        auto values = logitsVec(logits);
        expectTokenAllowed(values, kA);
        expectTokenMasked(values, kB);
        expectTokenMasked(values, kX);
    }
}

TEST(GrammarLogitsProcessorTest, ProcessKeepsReasoningFreeUntilBudgetThenAppliesFinalGrammarOnGpu) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);
    auto proc = makeProcessorFromKey(backend, {"structural_tag", makeReasoningStructuralTag(/*budget=*/1)});

    auto reasoning_logits = torch::zeros({1, 128}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    ASSERT_FALSE(proc->process(makeSamplerInputs(reasoning_logits), 0, 1).has_value());
    expectTokenAllowed(logitsVec(reasoning_logits), kX);

    ASSERT_FALSE(proc->updateStatus(torch::tensor({kX}, torch::kInt32).reshape({1, 1}), 1).has_value());
    auto end_logits = torch::zeros({1, 128}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    ASSERT_FALSE(proc->process(makeSamplerInputs(end_logits), 0, 1).has_value());
    auto end_values = logitsVec(end_logits);
    expectTokenAllowed(end_values, kZ);
    expectTokenMasked(end_values, kX);
    expectTokenMasked(end_values, kA);

    ASSERT_FALSE(proc->updateStatus(torch::tensor({kZ}, torch::kInt32).reshape({1, 1}), 1).has_value());
    auto final_logits = torch::zeros({1, 128}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    ASSERT_FALSE(proc->process(makeSamplerInputs(final_logits), 0, 1).has_value());
    auto final_values = logitsVec(final_logits);
    expectTokenAllowed(final_values, kA);
    expectTokenMasked(final_values, kB);
}

TEST(GrammarLogitsProcessorTest, AllTrueXGrammarMaskIsANoopRatherThanFailure) {
    auto backend = makeBackend(/*terminate_without_stop_token=*/false);
    ASSERT_TRUE(backend);
    auto proc = makeProcessorFromKey(backend, {"structural_tag", makeUnboundedAnyTextStructuralTag()});

    const size_t         words = SpecLogitsProcessorRequest::bitmaskWordCount(128);
    std::vector<int32_t> bitmask(words, SpecLogitsProcessorRequest::kBitmaskAllowAll);
    int64_t              dl_shape[2];
    DLTensor             dl     = makeSingleRowBitmaskView(bitmask.data(), static_cast<int32_t>(words), dl_shape);
    auto                 filled = proc.matcher->fillBitmask(&dl, 0);
    ASSERT_TRUE(filled.ok()) << filled.status().ToString();
    ASSERT_FALSE(filled.value()) << "unbounded any_text should produce an all-true mask";

    auto logits = torch::zeros({1, 128}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    ASSERT_FALSE(proc->process(makeSamplerInputs(logits), 0, 1).has_value());
    auto values = logitsVec(logits);
    expectTokenAllowed(values, kA);
    expectTokenAllowed(values, kX);
    expectTokenAllowed(values, kEos);
}

TEST(GrammarLogitsProcessorTest, InstanceDeclaresMtpAndCommittedStateContract) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);
    auto incremental = makeProcessor(backend, "ab");

    EXPECT_EQ(incremental->mtpCapability().mode, MtpProcessorMode::SPEC_VERIFY);
    ASSERT_TRUE(incremental->committedOutputLen().has_value());
    EXPECT_EQ(incremental->committedOutputLen().value(), 0);
}

TEST(GrammarLogitsProcessorTest, UpdateStatusAdvancesDecodeMaskState) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);
    auto proc = makeProcessor(backend, "ab");

    auto token_a      = torch::tensor({kA}, torch::kInt32).reshape({1, 1});
    auto update_error = proc->updateStatus(token_a, 1);
    EXPECT_EQ(proc->committedOutputLen().value(), 1);
    ASSERT_FALSE(update_error.has_value());

    auto logits        = torch::zeros({1, 128}, torch::kFloat32);
    auto inputs        = makeSamplerInputs(logits);
    auto process_error = proc->process(inputs, 0, 1);
    ASSERT_FALSE(process_error.has_value());

    auto values = logitsVec(logits);
    expectTokenMasked(values, kA);
    expectTokenAllowed(values, kB);
    expectTokenMasked(values, kX);
}

TEST(GrammarLogitsProcessorTest, ProcessForcesEosAfterGrammarTerminates) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);
    auto proc = makeProcessor(backend, "ab");

    ASSERT_FALSE(proc->updateStatus(torch::tensor({kA}, torch::kInt32).reshape({1, 1}), 1).has_value());
    ASSERT_FALSE(proc->updateStatus(torch::tensor({kB}, torch::kInt32).reshape({1, 1}), 1).has_value());
    EXPECT_EQ(proc->committedOutputLen().value(), 2);
    EXPECT_TRUE(matcherTerminated(*proc.matcher));

    auto logits        = torch::zeros({1, 128}, torch::kFloat32);
    auto inputs        = makeSamplerInputs(logits);
    auto process_error = proc->process(inputs, 0, 1);
    ASSERT_FALSE(process_error.has_value());

    auto values = logitsVec(logits);
    expectTokenAllowed(values, kEos);
    expectTokenMasked(values, kA);
    expectTokenMasked(values, kB);

    ASSERT_FALSE(proc->updateStatus(torch::tensor({kEos}, torch::kInt32).reshape({1, 1}), 1).has_value());
    EXPECT_EQ(proc->committedOutputLen().value(), 3);
    EXPECT_FALSE(proc.matcher->finished());

    auto next_logits = torch::zeros({1, 128}, torch::kFloat32);
    auto next_inputs = makeSamplerInputs(next_logits);
    ASSERT_FALSE(proc->process(next_inputs, 0, 1).has_value());
    auto next_values = logitsVec(next_logits);
    expectTokenAllowed(next_values, kEos);
    expectTokenMasked(next_values, kA);

    ASSERT_FALSE(proc->updateStatus(torch::tensor({kEos}, torch::kInt32).reshape({1, 1}), 1).has_value());
    EXPECT_EQ(proc->committedOutputLen().value(), 4);
    EXPECT_EQ(proc.matcher->numAcceptedTokens(), 2);
}

TEST(GrammarLogitsProcessorTest, ProcessRejectsOutOfVocabEosAfterGrammarTerminates) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);
    auto proc = makeProcessorFromKey(backend, {"regex", "ab"}, /*eos_token_id=*/128);

    ASSERT_FALSE(proc->updateStatus(torch::tensor({kA, kB}, torch::kInt32).reshape({1, 2}), 2).has_value());
    ASSERT_TRUE(matcherTerminated(*proc.matcher));

    auto logits = torch::zeros({1, 128}, torch::kFloat32);
    auto error  = proc->process(makeSamplerInputs(logits), 0, 1);

    ASSERT_TRUE(error.has_value());
    EXPECT_EQ(error->code(), ErrorCode::GRAMMAR_EOS_OUT_OF_VOCAB);
    EXPECT_NE(error->ToString().find("eos_token_id (128)"), std::string::npos);
    EXPECT_NE(error->ToString().find("vocab=128"), std::string::npos);
}

TEST(GrammarLogitsProcessorTest, UpdateStatusReportsInvalidCommittedToken) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);
    auto proc = makeProcessor(backend, "ab");

    auto error = proc->updateStatus(torch::tensor({kX}, torch::kInt32).reshape({1, 1}), 1);

    ASSERT_TRUE(error.has_value());
    EXPECT_EQ(error->code(), ErrorCode::GRAMMAR_PARSER_REJECTED_TOKEN);
    EXPECT_EQ(proc->committedOutputLen().value(), 0);
}

TEST(GrammarLogitsProcessorTest, UpdateStatusRollsBackEntireRejectedBatch) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);
    auto proc = makeProcessor(backend, "ab");

    auto initial_logits = torch::zeros({1, 128}, torch::kFloat32);
    ASSERT_FALSE(proc->process(makeSamplerInputs(initial_logits), 0, 1).has_value());
    expectTokenAllowed(logitsVec(initial_logits), kA);
    expectTokenMasked(logitsVec(initial_logits), kB);

    auto error = proc->updateStatus(torch::tensor({kA, kX}, torch::kInt32).reshape({1, 2}), 2);

    ASSERT_TRUE(error.has_value());
    EXPECT_EQ(error->code(), ErrorCode::GRAMMAR_PARSER_REJECTED_TOKEN);
    EXPECT_EQ(proc->committedOutputLen().value(), 0);
    EXPECT_EQ(proc.matcher->numAcceptedTokens(), 0);

    // Both matcher state and the cached mask still describe the pre-commit state.
    auto logits_after_rollback = torch::zeros({1, 128}, torch::kFloat32);
    ASSERT_FALSE(proc->process(makeSamplerInputs(logits_after_rollback), 0, 1).has_value());
    expectTokenAllowed(logitsVec(logits_after_rollback), kA);
    expectTokenMasked(logitsVec(logits_after_rollback), kB);

    ASSERT_FALSE(proc->updateStatus(torch::tensor({kA}, torch::kInt32).reshape({1, 1}), 1).has_value());
    EXPECT_EQ(proc->committedOutputLen().value(), 1);
}

// regex "ab": legal sequence is 'a' then 'b'. A fully-legal draft chain should
// return cap == propose_step with each row constraining to the expected token.
TEST(GrammarLogitsProcessorTest, AcceptsLegalDraftChain) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);
    auto proc = makeProcessor(backend, "ab");

    const int            propose_step = 2;
    const size_t         words        = SpecLogitsProcessorRequest::bitmaskWordCount(128);
    std::vector<int32_t> bm(static_cast<size_t>(propose_step + 1) * words, 0);
    std::vector<int32_t> draft{kA, kB};

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = draft.data();
    req.propose_step       = propose_step;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 128;

    const int cap = expectCapOk(proc->prepareSpeculative(req));
    EXPECT_EQ(cap, propose_step) << "every draft token is grammar-legal";
    EXPECT_TRUE(rowAllows(bm, words, 0, kA)) << "row 0 must allow 'a'";
    EXPECT_FALSE(rowAllows(bm, words, 0, kB)) << "row 0 must NOT allow 'b' at the start";
    EXPECT_TRUE(rowAllows(bm, words, 1, kB)) << "row 1 (after 'a') must allow 'b'";
}

TEST(GrammarLogitsProcessorTest, SpecVerifyKeepsAllTrueXGrammarRowsUnconstrained) {
    auto backend = makeBackend(/*terminate_without_stop_token=*/false);
    ASSERT_TRUE(backend);
    auto proc = makeProcessorFromKey(backend, {"structural_tag", makeUnboundedAnyTextStructuralTag()});

    const int            propose_step = 2;
    const size_t         words        = SpecLogitsProcessorRequest::bitmaskWordCount(128);
    std::vector<int32_t> bm(static_cast<size_t>(propose_step + 1) * words, 0);
    std::vector<int32_t> draft{kA, kX};

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = draft.data();
    req.propose_step       = propose_step;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 128;

    EXPECT_EQ(expectCapOk(proc->prepareSpeculative(req)), propose_step);
    for (int row = 0; row <= propose_step; ++row) {
        EXPECT_TRUE(rowAllows(bm, words, row, kA));
        EXPECT_TRUE(rowAllows(bm, words, row, kX));
    }
}

// A draft token that violates the grammar caps at that offset.
TEST(GrammarLogitsProcessorTest, CapsAtFirstIllegalDraftToken) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);
    auto proc = makeProcessor(backend, "ab");

    const int            propose_step = 2;
    const size_t         words        = SpecLogitsProcessorRequest::bitmaskWordCount(128);
    std::vector<int32_t> bm(static_cast<size_t>(propose_step + 1) * words, 0);
    std::vector<int32_t> draft{kA, kX};  // 'a' ok, then 'x' illegal (expected 'b')

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = draft.data();
    req.propose_step       = propose_step;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 128;

    const int cap = expectCapOk(proc->prepareSpeculative(req));
    EXPECT_EQ(cap, 1) << "draft[1]='x' is illegal after 'a', so cap == 1";
}

// Regression for the terminated-matcher guard: once the grammar terminates
// (after "ab"), verify must stop with cap == 2 and must NOT call acceptToken on
// the already-terminated matcher for trailing draft tokens (including EOS).
TEST(GrammarLogitsProcessorTest, TerminatedMatcherLeavesAllowAllWithoutCrash) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);
    auto proc = makeProcessor(backend, "ab");

    const int            propose_step = 3;  // one past the grammar's natural end
    const size_t         words        = SpecLogitsProcessorRequest::bitmaskWordCount(128);
    std::vector<int32_t> bm(static_cast<size_t>(propose_step + 1) * words,
                            SpecLogitsProcessorRequest::kBitmaskAllowAll);
    std::vector<int32_t> draft{kA, kB, kA, kB};

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = draft.data();
    req.propose_step       = propose_step;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 128;

    int cap = 0;
    ASSERT_NO_THROW({
        auto cap_or = proc->prepareSpeculative(req);
        ASSERT_TRUE(cap_or.ok()) << cap_or.status().ToString();
        cap = cap_or.value();
    });
    EXPECT_EQ(cap, 2) << "grammar completes after 'ab'; trailing draft must not advance matcher";
    EXPECT_FALSE(matcherTerminated(*proc.matcher)) << "provisional accepts must roll back committed matcher state";
    // Unfilled rows stay at allow-all (init + fill_row never reached offset 2+).
    EXPECT_TRUE(rowAllows(bm, words, 3, kX));
}

TEST(GrammarLogitsProcessorTest, VerifyCapStopsWhenDraftContinuesWithEos) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);
    auto proc = makeProcessor(backend, "ab");

    const int            propose_step = 3;
    const size_t         words        = SpecLogitsProcessorRequest::bitmaskWordCount(128);
    std::vector<int32_t> bm(static_cast<size_t>(propose_step + 1) * words, 0);
    std::vector<int32_t> draft{kA, kB, kEos, kX};

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = draft.data();
    req.propose_step       = propose_step;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 128;

    const int cap = expectCapOk(proc->prepareSpeculative(req));
    EXPECT_EQ(cap, 2);
    EXPECT_FALSE(matcherTerminated(*proc.matcher));
}

TEST(GrammarLogitsProcessorTest, VerifyCapZeroWhenGrammarAlreadyComplete) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);
    auto proc = makeProcessor(backend, "ab");

    ASSERT_TRUE(acceptMatcherToken(*proc.matcher, kA));
    ASSERT_TRUE(acceptMatcherToken(*proc.matcher, kB));
    EXPECT_TRUE(matcherTerminated(*proc.matcher));

    const int            propose_step = 2;
    const size_t         words        = SpecLogitsProcessorRequest::bitmaskWordCount(128);
    std::vector<int32_t> bm(static_cast<size_t>(propose_step + 1) * words, 0);
    std::vector<int32_t> draft{kX, kX, kX};

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = draft.data();
    req.propose_step       = propose_step;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 128;

    EXPECT_EQ(expectCapOk(proc->prepareSpeculative(req)), 0);
    EXPECT_TRUE(matcherTerminated(*proc.matcher));
}

TEST(GrammarLogitsProcessorTest, SpecVerifyCountsEosCommittedAcrossTerminatedRounds) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);
    auto proc = makeProcessor(backend, "ab");

    ASSERT_FALSE(proc->updateStatus(torch::tensor({kA, kB}, torch::kInt32).reshape({1, 2}), 2).has_value());
    ASSERT_TRUE(matcherTerminated(*proc.matcher));
    ASSERT_EQ(proc->committedOutputLen().value(), 2);

    const int            propose_step = 2;
    const size_t         words        = SpecLogitsProcessorRequest::bitmaskWordCount(128);
    std::vector<int32_t> bm(static_cast<size_t>(propose_step + 1) * words, 0);
    std::vector<int32_t> draft{kX, kX};

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = draft.data();
    req.propose_step       = propose_step;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 128;

    EXPECT_EQ(expectCapOk(proc->prepareSpeculative(req)), 0);
    EXPECT_TRUE(rowAllows(bm, words, 0, kEos));
    EXPECT_FALSE(rowAllows(bm, words, 0, kX));
    ASSERT_FALSE(proc->updateStatus(torch::tensor({kEos}, torch::kInt32).reshape({1, 1}), 1).has_value());
    EXPECT_EQ(proc->committedOutputLen().value(), 3);

    EXPECT_EQ(expectCapOk(proc->prepareSpeculative(req)), 0);
    ASSERT_FALSE(proc->updateStatus(torch::tensor({kEos}, torch::kInt32).reshape({1, 1}), 1).has_value());
    EXPECT_EQ(proc->committedOutputLen().value(), 4);
    EXPECT_EQ(proc.matcher->numAcceptedTokens(), 2);
}

// prepareSpeculative must leave the matcher's committed state unchanged
// (it rolls back any provisional accepts): a second identical call yields the
// same cap.
TEST(GrammarLogitsProcessorTest, RollsBackProvisionalAccepts) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);
    auto proc = makeProcessor(backend, "ab");

    const int            propose_step = 2;
    const size_t         words        = SpecLogitsProcessorRequest::bitmaskWordCount(128);
    std::vector<int32_t> bm(static_cast<size_t>(propose_step + 1) * words, 0);
    std::vector<int32_t> draft{kA, kB};

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = draft.data();
    req.propose_step       = propose_step;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 128;

    const int64_t committed_matcher_len = proc.matcher->numAcceptedTokens();
    const int     r1                    = expectCapOk(proc->prepareSpeculative(req));
    EXPECT_EQ(proc.matcher->numAcceptedTokens(), committed_matcher_len);
    const int r2 = expectCapOk(proc->prepareSpeculative(req));
    EXPECT_EQ(r1, r2) << "state must be unchanged across calls (rollback)";
    EXPECT_EQ(proc.matcher->numAcceptedTokens(), committed_matcher_len);
    EXPECT_EQ(proc->committedOutputLen().value(), 0);
}

TEST(GrammarLogitsProcessorTest, VerifyCapIsDraftRejectIndex) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);
    auto proc = makeProcessor(backend, "ab");

    const int            propose_step = 2;
    const size_t         words        = SpecLogitsProcessorRequest::bitmaskWordCount(128);
    std::vector<int32_t> bm(static_cast<size_t>(propose_step + 1) * words, 0);
    std::vector<int32_t> bad_draft{kX, kB};

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = bad_draft.data();
    req.propose_step       = propose_step;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 128;

    EXPECT_EQ(expectCapOk(proc->prepareSpeculative(req)), 0);
}

TEST(GrammarLogitsProcessorTest, SpecVerifyRejectsGrammarVocabExceedingModelVocabInSameBitmaskWord) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);
    auto proc = makeProcessor(backend, "ab");

    const int            propose_step = 1;
    const size_t         words        = SpecLogitsProcessorRequest::bitmaskWordCount(127);
    std::vector<int32_t> bm(static_cast<size_t>(propose_step + 1) * words, 0);
    std::vector<int32_t> draft{kA};

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = draft.data();
    req.propose_step       = propose_step;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 127;

    auto cap_or = proc->prepareSpeculative(req);
    ASSERT_FALSE(cap_or.ok());
    EXPECT_EQ(cap_or.status().code(), ErrorCode::GRAMMAR_VOCAB_EXCEEDS_MODEL_VOCAB);
    EXPECT_TRUE(rowAllows(bm, words, 0, kEos));
    EXPECT_FALSE(rowAllows(bm, words, 0, kA));
}

TEST(GrammarLogitsProcessorTest, SpecVerifyRejectsEosOutsideModelVocabInSameBitmaskWord) {
    auto backend = makeBackend(/*terminate_without_stop_token=*/true, /*vocab_size=*/127);
    ASSERT_TRUE(backend);
    auto proc = makeProcessorFromKey(backend, {"regex", "ab"}, /*eos_token_id=*/127);

    const size_t         words = SpecLogitsProcessorRequest::bitmaskWordCount(127);
    std::vector<int32_t> bm(2 * words, 0);
    std::vector<int32_t> draft{kA};

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = draft.data();
    req.propose_step       = 1;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 127;

    auto cap_or = proc->prepareSpeculative(req);
    ASSERT_FALSE(cap_or.ok());
    EXPECT_EQ(cap_or.status().code(), ErrorCode::GRAMMAR_EOS_OUT_OF_VOCAB);
}

TEST(GrammarLogitsProcessorTest, ClearBitmaskTokenRangeClearsFullWordsAndEdges) {
    const size_t         words = SpecLogitsProcessorRequest::bitmaskWordCount(96);
    std::vector<int32_t> bm(words, SpecLogitsProcessorRequest::kBitmaskAllowAll);

    clearBitmaskTokenRange(bm.data(), words, 35, 70);

    EXPECT_TRUE(rowAllows(bm, words, 0, 34));
    EXPECT_FALSE(rowAllows(bm, words, 0, 35));
    EXPECT_FALSE(rowAllows(bm, words, 0, 63));
    EXPECT_FALSE(rowAllows(bm, words, 0, 64));
    EXPECT_FALSE(rowAllows(bm, words, 0, 69));
    EXPECT_TRUE(rowAllows(bm, words, 0, 70));
}

TEST(GrammarLogitsProcessorTest, StructuralTagReasoningBudgetForcesEndAndFinalGrammar) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);
    auto proc = makeProcessorFromKey(backend, {"structural_tag", makeReasoningStructuralTag(/*budget=*/1)});

    const int            propose_step = 3;
    const size_t         words        = SpecLogitsProcessorRequest::bitmaskWordCount(128);
    std::vector<int32_t> bm(static_cast<size_t>(propose_step + 1) * words, 0);
    std::vector<int32_t> draft{kX, kZ, kA};

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = draft.data();
    req.propose_step       = propose_step;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 128;

    const int cap = expectCapOk(proc->prepareSpeculative(req));
    EXPECT_EQ(cap, propose_step);

    // Before budget is consumed, xgrammar permits ordinary think content.
    EXPECT_TRUE(rowAllows(bm, words, 0, kX));

    // After one think token, AnyTextFormat(max_tokens=1) suppresses body tokens
    // and forces the structural tag end string.
    EXPECT_TRUE(rowAllows(bm, words, 1, kZ));
    EXPECT_FALSE(rowAllows(bm, words, 1, kX));
    EXPECT_FALSE(rowAllows(bm, words, 1, kA));

    // Once the end string is accepted, the final regex grammar owns the mask.
    EXPECT_TRUE(rowAllows(bm, words, 2, kA));
    EXPECT_FALSE(rowAllows(bm, words, 2, kB));

    EXPECT_EQ(proc->committedOutputLen().value(), 0);
    EXPECT_EQ(proc.matcher->numAcceptedTokens(), 0);
}

TEST(GrammarLogitsProcessorTest, StructuralTagReasoningBudgetForcesTokenEndAndFinalGrammar) {
    auto backend = makeBackend();
    ASSERT_TRUE(backend);
    auto proc =
        makeProcessorFromKey(backend, {"structural_tag", makeReasoningStructuralTagWithTokenEnd(/*budget=*/1, kZ)});

    const int            propose_step = 3;
    const size_t         words        = SpecLogitsProcessorRequest::bitmaskWordCount(128);
    std::vector<int32_t> bm(static_cast<size_t>(propose_step + 1) * words, 0);
    std::vector<int32_t> draft{kX, kZ, kA};

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = draft.data();
    req.propose_step       = propose_step;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 128;

    const int cap = expectCapOk(proc->prepareSpeculative(req));
    EXPECT_EQ(cap, propose_step);

    EXPECT_TRUE(rowAllows(bm, words, 0, kX));
    EXPECT_TRUE(rowAllows(bm, words, 1, kZ));
    EXPECT_FALSE(rowAllows(bm, words, 1, kX));
    EXPECT_FALSE(rowAllows(bm, words, 1, kA));
    EXPECT_TRUE(rowAllows(bm, words, 2, kA));
    EXPECT_FALSE(rowAllows(bm, words, 2, kB));

    EXPECT_EQ(proc->committedOutputLen().value(), 0);
    EXPECT_EQ(proc.matcher->numAcceptedTokens(), 0);
}

}  // namespace rtp_llm
