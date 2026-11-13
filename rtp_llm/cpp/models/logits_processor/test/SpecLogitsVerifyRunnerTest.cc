#include <gtest/gtest.h>

#include <torch/torch.h>
#include <limits>
#include <xgrammar/tokenizer_info.h>
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackendCpp.h"
#include "rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/ReasoningGrammarLogitsProcessor.h"

#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.h"

namespace rtp_llm {
namespace {

class StubSpecProcessor: public SpecLogitsProcessor {
public:
    StubSpecProcessor(bool eligible, int cap, int masked_token): eligible_(eligible), cap_(cap), masked_token_(masked_token) {}

    bool isSpecVerifyEligible() const override {
        return eligible_;
    }

    int tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) override {
        ++fill_calls_;
        if (masked_token_ >= 0) {
            for (int row = 0; row <= request.propose_step; ++row) {
                int32_t* row_ptr = request.bitmask_cpu_out + row * request.bitmask_size_int32;
                row_ptr[masked_token_ / 32] &= ~(1 << (masked_token_ % 32));
            }
        }
        return cap_;
    }

    int fillCalls() const {
        return fill_calls_;
    }

private:
    bool eligible_;
    int  cap_;
    int  masked_token_;
    int  fill_calls_ = 0;
};

SpecLogitsVerifyRunner::LaunchTask makeTask(const std::vector<SpecLogitsProcessorPtr>& processors,
                                            int                                        propose_step,
                                            size_t                                     vocab_size) {
    SpecLogitsVerifyRunner::LaunchTask task;
    task.total_streams = processors.size();
    task.propose_step  = propose_step;
    task.vocab_size    = vocab_size;
    task.draft_tokens =
        torch::zeros({static_cast<int64_t>(processors.size()), propose_step}, torch::kInt32);
    for (size_t i = 0; i < processors.size(); ++i) {
        task.active.push_back({processors[i], /*stream_idx=*/i, /*processor_idx=*/0,
                               /*stream_id=*/100 + i, /*base_seq_len=*/4, /*base_output_len=*/2});
    }
    return task;
}

}  // namespace

TEST(SpecLogitsVerifyRunnerTest, MixedBatchSkipsIneligibleWithoutDroppingArtifact) {
    const int    P = 2;
    const size_t V = 64;

    auto eligible   = std::make_shared<StubSpecProcessor>(/*eligible=*/true, /*cap=*/1, /*masked_token=*/7);
    auto ineligible = std::make_shared<StubSpecProcessor>(/*eligible=*/false, /*cap=*/0, /*masked_token=*/3);

    SpecLogitsVerifyRunner runner;
    auto result = runner.buildInline(makeTask({eligible, ineligible}, P, V));

    EXPECT_TRUE(result.has_active_processor);
    EXPECT_EQ(result.skipped_ineligible_processors, 1u);
    ASSERT_EQ(result.applied_processors.size(), 1u);
    EXPECT_EQ(result.applied_processors[0].stream_id, 100u);
    EXPECT_EQ(eligible->fillCalls(), 1);
    EXPECT_EQ(ineligible->fillCalls(), 0);

    ASSERT_TRUE(result.spec_vocab_mask_gpu.defined());
    ASSERT_TRUE(result.spec_cap_gpu.defined());
    if (result.ready_event) {
        result.ready_event->synchronize();
    }
    auto mask = result.spec_vocab_mask_gpu.cpu();
    auto cap  = result.spec_cap_gpu.cpu();

    ASSERT_EQ(mask.sizes(), (torch::IntArrayRef{2 * (P + 1), static_cast<int64_t>(V)}));
    // Stream 0 (eligible): exactly token 7 masked in each of its P+1 rows.
    for (int row = 0; row <= P; ++row) {
        for (size_t tok = 0; tok < V; ++tok) {
            EXPECT_EQ(mask[row][static_cast<int64_t>(tok)].item<bool>(), tok == 7)
                << "stream0 row=" << row << " tok=" << tok;
        }
    }
    // Stream 1 (skipped): rows stay all-allow.
    for (int row = P + 1; row < 2 * (P + 1); ++row) {
        EXPECT_FALSE(mask[row].any().item<bool>()) << "stream1 row=" << row;
    }
    EXPECT_EQ(cap[0].item<int32_t>(), 1);
    EXPECT_EQ(cap[1].item<int32_t>(), P);
}

TEST(SpecLogitsVerifyRunnerTest, AllIneligibleReturnsNoArtifactWithSkipCount) {
    auto a = std::make_shared<StubSpecProcessor>(/*eligible=*/false, /*cap=*/0, /*masked_token=*/-1);
    auto b = std::make_shared<StubSpecProcessor>(/*eligible=*/false, /*cap=*/0, /*masked_token=*/-1);

    SpecLogitsVerifyRunner runner;
    auto result = runner.buildInline(makeTask({a, b}, /*propose_step=*/2, /*vocab_size=*/64));

    EXPECT_FALSE(result.has_active_processor);
    EXPECT_EQ(result.skipped_ineligible_processors, 2u);
    EXPECT_TRUE(result.applied_processors.empty());
    EXPECT_FALSE(result.spec_vocab_mask_gpu.defined());
    EXPECT_FALSE(result.spec_cap_gpu.defined());
    EXPECT_EQ(a->fillCalls(), 0);
    EXPECT_EQ(b->fillCalls(), 0);
}

TEST(SpecLogitsVerifyRunnerTest, TimedOutStreamPreservesThirdArtifactAndDoesNotAffectOtherStreams) {
    std::vector<std::string> vocab;
    for (int i = 0; i < 128; ++i) {
        vocab.emplace_back(1, static_cast<char>(i));
    }
    xgrammar::TokenizerInfo info(vocab, xgrammar::VocabType::RAW, 128, std::vector<int32_t>{0});
    XGrammarBackendOptions  options;
    options.compile_timeout_ms = 0;
    XGrammarBackendCpp backend(info.SerializeJSON(), options);
    auto               compiled = backend.compileNow({"regex", "a*"}).compiled;
    ASSERT_TRUE(compiled);
    for (bool reasoning : {false, true}) {
        BaseLogitsProcessorPtr slow;
        auto                   matcher = backend.createMatcher(compiled, false, std::nullopt);
        if (reasoning) {
            auto p = std::make_shared<ReasoningGrammarLogitsProcessor>(
                matcher, 0, 0, std::vector<int>{}, std::vector<int>{}, 0);
            p->timeout_state_.timeout_ms_ = 0;
            slow                          = p;
        } else {
            auto p                        = std::make_shared<GrammarLogitsProcessor>(matcher, 0);
            p->timeout_state_.timeout_ms_ = 0;
            slow                          = p;
        }
        auto fast = std::make_shared<GrammarLogitsProcessor>(
            backend.createMatcher(compiled, false, std::nullopt), 0, nullptr, std::numeric_limits<int>::max());
        auto task         = makeTask({std::dynamic_pointer_cast<SpecLogitsProcessor>(slow), fast}, 2, 128);
        task.draft_tokens = torch::tensor({{'a', 'x'}, {'a', 'x'}}, torch::kInt32);
        SpecLogitsVerifyRunner runner;
        for (int round = 1; round <= 4; ++round) {
            auto result = runner.buildInline(task);
            ASSERT_TRUE(result.has_active_processor);
            EXPECT_EQ(result.skipped_ineligible_processors, 0);
            ASSERT_EQ(result.applied_processors.size(), 2);
            if (result.ready_event) {
                result.ready_event->synchronize();
            }
            auto mask = result.spec_vocab_mask_gpu.cpu();
            auto cap  = result.spec_cap_gpu.cpu();
            EXPECT_EQ(cap[0].item<int32_t>(), round <= 3 ? 1 : 2);
            EXPECT_EQ(mask[0]['x'].item<bool>(), round <= 3);
            EXPECT_EQ(cap[1].item<int32_t>(), 1);
            EXPECT_TRUE(mask[3]['x'].item<bool>());
            EXPECT_EQ(slow->xgrammarUpdateTimedOut(), round >= 3);
            EXPECT_FALSE(fast->xgrammarUpdateTimedOut());
        }
    }
}

}  // namespace rtp_llm
