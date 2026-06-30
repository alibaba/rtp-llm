// CPU unit tests for GrammarLogitsProcessor::tryAcceptAndFillBitmask over a 128-char ASCII vocab.

#include "rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.h"
#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackend.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include <xgrammar/tokenizer_info.h>

namespace rtp_llm {
namespace {

std::string makeAsciiTokenizerInfoJson() {
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

XGrammarBackendOptions defaultOptions() {
    XGrammarBackendOptions opts;
    opts.any_whitespace        = true;
    opts.strict_mode           = true;
    opts.max_compiler_threads  = 2;
    opts.enable_compiler_cache = true;
    opts.compiler_cache_bytes  = -1;
    return opts;
}

struct ProcessorBundle {
    std::shared_ptr<GrammarLogitsProcessor> proc;
    std::shared_ptr<RtpGrammarMatcher>      matcher;

    GrammarLogitsProcessor* operator->() const noexcept {
        return proc.get();
    }
};

// terminate_without_stop_token=true so the matcher flips IsTerminated() the moment the regex completes.
ProcessorBundle makeProcessor(XGrammarBackend& backend, const std::string& regex) {
    auto compiled = backend.compileNow({"regex", regex}).compiled;
    EXPECT_TRUE(compiled);
    std::shared_ptr<RtpGrammarMatcher> matcher = backend.createMatcher(compiled,
                                                                       /*require_reasoning=*/false,
                                                                       /*think_end_token_ids=*/std::nullopt,
                                                                       /*terminate_without_stop_token=*/true);
    auto                               proc    = std::make_shared<GrammarLogitsProcessor>(matcher);
    return {std::move(proc), std::move(matcher)};
}

// Reasoning matcher: starts in passthrough (think body) until think_end matches.
ProcessorBundle makeReasoningProcessor(XGrammarBackend& backend, const std::string& regex) {
    auto compiled = backend.compileNow({"regex", regex}).compiled;
    EXPECT_TRUE(compiled);
    // think_end 'z' (id 122): single-byte exit, disjoint from {a,b,x,eos}.
    std::shared_ptr<RtpGrammarMatcher> matcher = backend.createMatcher(compiled,
                                                                       /*require_reasoning=*/true,
                                                                       /*think_end_token_ids=*/std::vector<int>{'z'},
                                                                       /*terminate_without_stop_token=*/true);
    matcher->initReasoning(/*in_think_body=*/true);
    auto proc = std::make_shared<GrammarLogitsProcessor>(matcher);
    return {std::move(proc), std::move(matcher)};
}

bool rowAllows(const std::vector<int32_t>& bm, size_t words, int row, int token) {
    const int32_t word = bm[static_cast<size_t>(row) * words + token / 32];
    return (static_cast<uint32_t>(word) & (1u << (token % 32))) != 0u;
}

constexpr int kA   = 'a';  // token id 97
constexpr int kB   = 'b';  // token id 98
constexpr int kX   = 'x';  // token id 120
constexpr int kEos = 0;    // stop token in makeAsciiTokenizerInfoJson

}  // namespace

// regex "ab": legal sequence is 'a' then 'b'. A fully-legal draft chain should
// return cap == propose_step with each row constraining to the expected token.
TEST(GrammarLogitsProcessorTest, AcceptsLegalDraftChain) {
    XGrammarBackend backend(makeAsciiTokenizerInfoJson(), defaultOptions());
    auto            proc = makeProcessor(backend, "ab");

    const int            propose_step = 2;
    const size_t         words        = SpecLogitsProcessor::bitmaskWordCount(128);
    std::vector<int32_t> bm(static_cast<size_t>(propose_step + 1) * words, 0);
    std::vector<int32_t> draft{kA, kB};

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = draft.data();
    req.propose_step       = propose_step;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 128;

    const int cap = proc->tryAcceptAndFillBitmask(req);
    EXPECT_EQ(cap, propose_step) << "every draft token is grammar-legal";
    EXPECT_TRUE(rowAllows(bm, words, 0, kA)) << "row 0 must allow 'a'";
    EXPECT_FALSE(rowAllows(bm, words, 0, kB)) << "row 0 must NOT allow 'b' at the start";
    EXPECT_TRUE(rowAllows(bm, words, 1, kB)) << "row 1 (after 'a') must allow 'b'";
}

// A draft token that violates the grammar caps at that offset.
TEST(GrammarLogitsProcessorTest, CapsAtFirstIllegalDraftToken) {
    XGrammarBackend backend(makeAsciiTokenizerInfoJson(), defaultOptions());
    auto            proc = makeProcessor(backend, "ab");

    const int            propose_step = 2;
    const size_t         words        = SpecLogitsProcessor::bitmaskWordCount(128);
    std::vector<int32_t> bm(static_cast<size_t>(propose_step + 1) * words, 0);
    std::vector<int32_t> draft{kA, kX};  // 'a' ok, then 'x' illegal (expected 'b')

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = draft.data();
    req.propose_step       = propose_step;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 128;

    const int cap = proc->tryAcceptAndFillBitmask(req);
    EXPECT_EQ(cap, 1) << "draft[1]='x' is illegal after 'a', so cap == 1";
}

// Regression for the terminated-matcher guard: once the grammar terminates
// (after "ab"), verify must stop with cap == 2 and must NOT call acceptToken on
// the already-terminated matcher for trailing draft tokens (including EOS).
TEST(GrammarLogitsProcessorTest, TerminatedMatcherLeavesAllowAllWithoutCrash) {
    XGrammarBackend backend(makeAsciiTokenizerInfoJson(), defaultOptions());
    auto            proc = makeProcessor(backend, "ab");

    const int            propose_step = 3;  // one past the grammar's natural end
    const size_t         words        = SpecLogitsProcessor::bitmaskWordCount(128);
    std::vector<int32_t> bm(static_cast<size_t>(propose_step + 1) * words, SpecLogitsProcessor::kBitmaskAllowAll);
    std::vector<int32_t> draft{kA, kB, kA, kB};

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = draft.data();
    req.propose_step       = propose_step;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 128;

    int cap = 0;
    ASSERT_NO_THROW({ cap = proc->tryAcceptAndFillBitmask(req); });
    EXPECT_EQ(cap, 2) << "grammar completes after 'ab'; trailing draft must not advance matcher";
    EXPECT_FALSE(proc.matcher.get()->isTerminated()) << "provisional accepts must roll back committed matcher state";
    // Unfilled rows stay at allow-all (init + fill_row never reached offset 2+).
    EXPECT_TRUE(rowAllows(bm, words, 3, kX));
}

TEST(GrammarLogitsProcessorTest, VerifyCapStopsWhenDraftContinuesWithEos) {
    XGrammarBackend backend(makeAsciiTokenizerInfoJson(), defaultOptions());
    auto            proc = makeProcessor(backend, "ab");

    const int            propose_step = 3;
    const size_t         words        = SpecLogitsProcessor::bitmaskWordCount(128);
    std::vector<int32_t> bm(static_cast<size_t>(propose_step + 1) * words, 0);
    std::vector<int32_t> draft{kA, kB, kEos, kX};

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = draft.data();
    req.propose_step       = propose_step;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 128;

    const int cap = proc->tryAcceptAndFillBitmask(req);
    EXPECT_EQ(cap, 2);
    EXPECT_FALSE(proc.matcher.get()->isTerminated());
}

TEST(GrammarLogitsProcessorTest, VerifyCapZeroWhenGrammarAlreadyComplete) {
    XGrammarBackend backend(makeAsciiTokenizerInfoJson(), defaultOptions());
    auto            proc = makeProcessor(backend, "ab");

    ASSERT_TRUE(proc.matcher.get()->acceptToken(kA));
    ASSERT_TRUE(proc.matcher.get()->acceptToken(kB));
    EXPECT_TRUE(proc.matcher->isTerminated());

    const int            propose_step = 2;
    const size_t         words        = SpecLogitsProcessor::bitmaskWordCount(128);
    std::vector<int32_t> bm(static_cast<size_t>(propose_step + 1) * words, 0);
    std::vector<int32_t> draft{kX, kX, kX};

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = draft.data();
    req.propose_step       = propose_step;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 128;

    EXPECT_EQ(proc->tryAcceptAndFillBitmask(req), 0);
    EXPECT_TRUE(proc.matcher->isTerminated());
}

// tryAcceptAndFillBitmask must leave the matcher's committed state unchanged
// (it rolls back any provisional accepts): a second identical call yields the
// same cap.
TEST(GrammarLogitsProcessorTest, RollsBackProvisionalAccepts) {
    XGrammarBackend backend(makeAsciiTokenizerInfoJson(), defaultOptions());
    auto            proc = makeProcessor(backend, "ab");

    const int            propose_step = 2;
    const size_t         words        = SpecLogitsProcessor::bitmaskWordCount(128);
    std::vector<int32_t> bm(static_cast<size_t>(propose_step + 1) * words, 0);
    std::vector<int32_t> draft{kA, kB};

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = draft.data();
    req.propose_step       = propose_step;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 128;

    const int r1 = proc->tryAcceptAndFillBitmask(req);
    const int r2 = proc->tryAcceptAndFillBitmask(req);
    EXPECT_EQ(r1, r2) << "state must be unchanged across calls (rollback)";
}

// In reasoning passthrough fill_row must allow all non-EOS but mask EOS (else draft EOS prematurely closes).
TEST(GrammarLogitsProcessorTest, PassthroughRowsAllowAllExceptEos) {
    XGrammarBackend backend(makeAsciiTokenizerInfoJson(), defaultOptions());
    auto            proc = makeReasoningProcessor(backend, "ab");
    ASSERT_TRUE(proc.matcher->isPassthroughForMask()) << "reasoning matcher must start in passthrough before think_end";

    const int            propose_step = 3;
    const size_t         words        = SpecLogitsProcessor::bitmaskWordCount(128);
    std::vector<int32_t> bm(static_cast<size_t>(propose_step + 1) * words, 0);
    // 'x' / 'a' / EOS / 'b' — all should be allowed by passthrough except EOS,
    // which forces cap == 2.
    std::vector<int32_t> draft{kX, kA, kEos, kB};

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = draft.data();
    req.propose_step       = propose_step;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 128;

    const int cap = proc->tryAcceptAndFillBitmask(req);
    EXPECT_EQ(cap, 2) << "draft[2]=EOS is masked in passthrough → cap stops at 2";

    for (int row = 0; row < propose_step; ++row) {
        EXPECT_TRUE(rowAllows(bm, words, row, kA)) << "row " << row << " must allow 'a'";
        EXPECT_TRUE(rowAllows(bm, words, row, kB)) << "row " << row << " must allow 'b'";
        EXPECT_TRUE(rowAllows(bm, words, row, kX)) << "row " << row << " must allow 'x'";
        EXPECT_FALSE(rowAllows(bm, words, row, kEos)) << "row " << row << " must mask EOS in passthrough";
    }
    EXPECT_TRUE(proc.matcher->isPassthroughForMask()) << "verify must roll back: matcher still in passthrough";
}

TEST(GrammarLogitsProcessorTest, VerifyCapIsDraftRejectIndex) {
    XGrammarBackend backend(makeAsciiTokenizerInfoJson(), defaultOptions());
    auto            proc = makeProcessor(backend, "ab");

    const int            propose_step = 2;
    const size_t         words        = SpecLogitsProcessor::bitmaskWordCount(128);
    std::vector<int32_t> bm(static_cast<size_t>(propose_step + 1) * words, 0);
    std::vector<int32_t> bad_draft{kX, kB};

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = bad_draft.data();
    req.propose_step       = propose_step;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 128;

    EXPECT_EQ(proc->tryAcceptAndFillBitmask(req), 0);
}

// Undersized bitmask buffer must error out as GRAMMAR_BITMASK_BUFFER_TOO_SMALL, not corrupt the caller's heap.
TEST(GrammarLogitsProcessorTest, RejectsUndersizedBitmaskBuffer) {
    XGrammarBackend backend(makeAsciiTokenizerInfoJson(), defaultOptions());
    auto            proc = makeProcessor(backend, "ab");

    const int propose_step = 1;
    // Allocate a deliberately too-small buffer for vocab=128 (needs 4 words).
    const size_t         words = 1;
    std::vector<int32_t> bm(static_cast<size_t>(propose_step + 1) * words, 0);
    std::vector<int32_t> draft{kA};

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = draft.data();
    req.propose_step       = propose_step;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 128;

    EXPECT_EQ(proc->tryAcceptAndFillBitmask(req), 0);
    ASSERT_TRUE(proc->hasError());
    EXPECT_EQ(proc->error().code(), ErrorCode::GRAMMAR_BITMASK_BUFFER_TOO_SMALL);
}

}  // namespace rtp_llm
