#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackend.h"
#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.h"
#include "rtp_llm/cpp/models/logits_processor/ThinkModeLogitsProcessor.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

#if USING_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <future>
#endif

namespace rtp_llm {
namespace {

// Minimal SPEC_VERIFY-capable processor. Main gates MTP eligibility through
// mtpCapability() (the MtpExecutor-level filtering is covered by
// MtpExecutorTest::testErroredSpecLogitsStreamDoesNotAbortExecutor); at the
// runner level every task.active entry is already eligible, so this stub only
// exercises prepareSpeculative() packed-mask emission, caps, and errors.
class StubSpecVerifyProcessor: public BaseLogitsProcessor {
public:
    StubSpecVerifyProcessor(int masked_token, int cap = -1, bool fail = false):
        masked_token_(masked_token), cap_(cap), fail_(fail) {}

    std::optional<ErrorInfo> process(const SamplerInputs&, size_t, size_t) override {
        return std::nullopt;
    }
    void                     updateMultiSeqStatus(const std::vector<int>&) override {}
    std::optional<ErrorInfo> updateStatus(const torch::Tensor&, int32_t) override {
        return std::nullopt;
    }

    MtpProcessorCapability mtpCapability() const override {
        return {MtpProcessorMode::SPEC_VERIFY, "stub spec-verify processor"};
    }

    ErrorResult<int> prepareSpeculative(const SpecLogitsProcessorRequest& request) override {
        ++prepare_calls_;
        observed_tokens_.assign(request.draft_tokens, request.draft_tokens + request.propose_step);
        if (fail_) {
            return ErrorResult<int>(ErrorCode::INVALID_PARAMS, "stub grammar rejected draft tokens");
        }
        if (masked_token_ >= 0) {
            for (int row = 0; row <= request.propose_step; ++row) {
                int32_t* row_ptr = request.bitmask_cpu_out + static_cast<size_t>(row) * request.bitmask_size_int32;
                row_ptr[masked_token_ / 32] &= ~(1 << (masked_token_ % 32));
            }
        }
        return ErrorResult<int>(cap_ < 0 ? request.propose_step : static_cast<int>(cap_));
    }

    int prepareCalls() const {
        return prepare_calls_;
    }
    const std::vector<int32_t>& observedTokens() const {
        return observed_tokens_;
    }

private:
    int                  masked_token_;
    int                  cap_;
    bool                 fail_;
    int                  prepare_calls_ = 0;
    std::vector<int32_t> observed_tokens_;
};

using StubPtr = std::shared_ptr<StubSpecVerifyProcessor>;

SpecLogitsVerifyRunner::LaunchTask makeTask(size_t total_streams, int propose_step, size_t vocab_size) {
    SpecLogitsVerifyRunner::LaunchTask task;
    task.total_streams = total_streams;
    task.propose_step  = propose_step;
    task.vocab_size    = vocab_size;
    task.draft_tokens  = torch::zeros({static_cast<int64_t>(total_streams), propose_step}, torch::kInt32);
    return task;
}

// bit=1 allow, bit=0 masked (packed allow-mask convention of SpecLogitsProcessorRequest).
bool allowedBit(const torch::Tensor& packed_mask, int64_t row, int64_t token) {
    const auto word = static_cast<uint32_t>(packed_mask[row][token / 32].item<int32_t>());
    return ((word >> (token % 32)) & 1u) != 0u;
}

std::shared_ptr<GrammarLogitsProcessor> makeGrammarProcessor() {
    std::vector<std::string> vocab;
    for (int i = 0; i < 128; ++i) {
        vocab.emplace_back(1, static_cast<char>(i));
    }
    xgrammar::TokenizerInfo tokenizer_info(vocab, xgrammar::VocabType::RAW, 128, std::vector<int32_t>{0});
    GrammarConfig           config;
    config.num_workers = 1;
    auto backend       = XGrammarBackend::create(tokenizer_info.SerializeJSON(), config);
    auto matcher       = backend->createMatcherFromKey({"regex", "a*"});
    EXPECT_TRUE(matcher.ok()) << matcher.status().ToString();
    if (!matcher.ok()) {
        return nullptr;
    }
    return std::make_shared<GrammarLogitsProcessor>(
        matcher.value(), /*eos_token_id=*/0, std::numeric_limits<int>::max());
}

void syncResult(const SpecLogitsVerifyRunner::LaunchResult& result) {
    if (result.ready_event) {
        result.ready_event->synchronize();
    }
}

}  // namespace

TEST(SpecLogitsVerifyRunnerTest, EmptyActiveReturnsInertResult) {
    SpecLogitsVerifyRunner runner;
    auto                   result = runner.run(makeTask(/*total_streams=*/2, /*propose_step=*/1, /*vocab_size=*/32));
    EXPECT_FALSE(result.has_active_processor);
    EXPECT_FALSE(result.packed_allow_mask_gpu.defined());
    EXPECT_FALSE(result.packed_allow_mask_cpu_lifetime.defined());
    EXPECT_TRUE(result.processor_errors.empty());
}

// Sparse verification: with one active stream out of three, the runner must
// emit compact rows only for the active stream and route them to the right
// logits rows; multiple processors on the same stream share one compact slot
// with AND-merged masks and min-merged caps.
TEST(SpecLogitsVerifyRunnerTest, SparseActiveStreamMasksOnlyItsCompactRows) {
    constexpr size_t kStreams = 3;
    constexpr int    kStep    = 1;
    constexpr size_t kVocab   = 35;  // 2 bitmask words; exercises the tail word

    auto mask_34 = std::make_shared<StubSpecVerifyProcessor>(/*masked_token=*/34);
    auto mask_33 = std::make_shared<StubSpecVerifyProcessor>(/*masked_token=*/33, /*cap=*/0);

    auto task = makeTask(kStreams, kStep, kVocab);
    task.active.push_back({mask_34, /*stream_idx=*/1});
    task.active.push_back({mask_33, /*stream_idx=*/1});

    SpecLogitsVerifyRunner runner;
    auto                   result = runner.run(task);
    syncResult(result);

    EXPECT_TRUE(result.has_active_processor);
    ASSERT_EQ(result.processor_errors.size(), kStreams);
    for (const auto& error : result.processor_errors) {
        EXPECT_FALSE(error.has_value());
    }
    EXPECT_EQ(mask_34->prepareCalls(), 1);
    EXPECT_EQ(mask_33->prepareCalls(), 1);

    const auto& packed      = result.packed_allow_mask_cpu_lifetime;
    const auto& row_indices = result.logits_row_indices_cpu_lifetime;
    ASSERT_TRUE(packed.defined());
    // Only the active stream contributes rows: (propose_step + 1) rows total.
    ASSERT_EQ(packed.size(0), kStep + 1);
    ASSERT_EQ(packed.size(1), static_cast<int64_t>(SpecLogitsProcessorRequest::bitmaskWordCount(kVocab)));
    // Rows map to the full logits layout of stream 1: rows 2 and 3.
    ASSERT_EQ(row_indices.size(0), kStep + 1);
    EXPECT_EQ(row_indices[0].item<int32_t>(), 2);
    EXPECT_EQ(row_indices[1].item<int32_t>(), 3);

    for (int64_t row = 0; row <= kStep; ++row) {
        for (size_t token = 0; token < kVocab; ++token) {
            const bool expect_allowed = token != 33 && token != 34;
            EXPECT_EQ(allowedBit(packed, row, static_cast<int64_t>(token)), expect_allowed)
                << "row=" << row << " token=" << token;
        }
    }

    // Caps: inactive streams keep propose_step; the active stream takes the
    // min across its processors.
    const auto& cap = result.spec_cap_cpu;
    ASSERT_EQ(cap.size(0), static_cast<int64_t>(kStreams));
    EXPECT_EQ(cap[0].item<int32_t>(), kStep);
    EXPECT_EQ(cap[1].item<int32_t>(), 0);
    EXPECT_EQ(cap[2].item<int32_t>(), kStep);
}

// All-active verification: every stream contributes compact rows; masks stay
// per-stream and row indices cover the full logits layout in active-order.
TEST(SpecLogitsVerifyRunnerTest, AllActiveStreamsKeepPerStreamMasksAndRowOrder) {
    constexpr size_t  kStreams     = 2;
    constexpr int     kStep        = 2;
    constexpr size_t  kVocab       = 35;
    constexpr int32_t kFirstToken  = 0;
    constexpr int32_t kSecondToken = 34;

    // Push stream 1 first to verify compact-slot ordering follows task.active.
    auto stream1_proc = std::make_shared<StubSpecVerifyProcessor>(kSecondToken, /*cap=*/1);
    auto stream0_proc = std::make_shared<StubSpecVerifyProcessor>(kFirstToken);

    auto task = makeTask(kStreams, kStep, kVocab);
    task.active.push_back({stream1_proc, /*stream_idx=*/1});
    task.active.push_back({stream0_proc, /*stream_idx=*/0});

    SpecLogitsVerifyRunner runner;
    auto                   result = runner.run(task);
    syncResult(result);

    const auto& packed      = result.packed_allow_mask_cpu_lifetime;
    const auto& row_indices = result.logits_row_indices_cpu_lifetime;
    ASSERT_TRUE(packed.defined());
    ASSERT_EQ(packed.size(0), static_cast<int64_t>(kStreams) * (kStep + 1));
    // Compact rows 0..2 belong to stream 1 (logits rows 3..5), rows 3..5 to
    // stream 0 (logits rows 0..2).
    const std::vector<int32_t> expected_rows{3, 4, 5, 0, 1, 2};
    ASSERT_EQ(row_indices.size(0), static_cast<int64_t>(expected_rows.size()));
    for (size_t i = 0; i < expected_rows.size(); ++i) {
        EXPECT_EQ(row_indices[static_cast<int64_t>(i)].item<int32_t>(), expected_rows[i]) << "compact_row=" << i;
    }

    for (int64_t row = 0; row < packed.size(0); ++row) {
        const bool stream1_row = row < kStep + 1;
        EXPECT_EQ(allowedBit(packed, row, kSecondToken), !stream1_row) << "row=" << row;
        EXPECT_EQ(allowedBit(packed, row, kFirstToken), stream1_row) << "row=" << row;
        EXPECT_TRUE(allowedBit(packed, row, 1)) << "row=" << row;
    }

    const auto& cap = result.spec_cap_cpu;
    EXPECT_EQ(cap[0].item<int32_t>(), kStep);
    EXPECT_EQ(cap[1].item<int32_t>(), 1);

    // GPU mirrors must match the CPU views once ready_event fired.
    if (result.packed_allow_mask_gpu.defined()) {
        EXPECT_TRUE(torch::equal(result.packed_allow_mask_gpu.cpu(), packed.contiguous()));
        EXPECT_TRUE(torch::equal(result.logits_row_indices_gpu.cpu(), row_indices.contiguous()));
        EXPECT_TRUE(torch::equal(result.spec_cap_gpu.cpu(), cap.contiguous()));
    }
}

// Main's replacement for dev's `skipped_ineligible_processors` abort guard:
// eligibility is gated upstream via mtpCapability() == SPEC_VERIFY, and a
// processor that errors during prepareSpeculative() must surface a per-stream
// entry in processor_errors without aborting the whole batch — the healthy
// stream keeps its mask and cap, the errored stream degrades to all-allow
// with cap 0.
TEST(SpecLogitsVerifyRunnerTest, ErroredStreamDoesNotAbortBatch) {
    constexpr size_t  kStreams     = 2;
    constexpr int     kStep        = 2;
    constexpr size_t  kVocab       = 64;
    constexpr int32_t kMaskedToken = 7;

    auto failing = std::make_shared<StubSpecVerifyProcessor>(/*masked_token=*/3, /*cap=*/-1, /*fail=*/true);
    auto healthy = std::make_shared<StubSpecVerifyProcessor>(kMaskedToken);

    auto task = makeTask(kStreams, kStep, kVocab);
    task.active.push_back({failing, /*stream_idx=*/0});
    task.active.push_back({healthy, /*stream_idx=*/1});

    SpecLogitsVerifyRunner runner;
    auto                   result = runner.run(task);
    syncResult(result);

    EXPECT_TRUE(result.has_active_processor);
    ASSERT_EQ(result.processor_errors.size(), kStreams);
    ASSERT_TRUE(result.processor_errors[0].has_value());
    EXPECT_EQ(result.processor_errors[0]->code(), ErrorCode::INVALID_PARAMS);
    EXPECT_FALSE(result.processor_errors[1].has_value());

    const auto& packed = result.packed_allow_mask_cpu_lifetime;
    ASSERT_TRUE(packed.defined());
    ASSERT_EQ(packed.size(0), static_cast<int64_t>(kStreams) * (kStep + 1));
    for (int64_t row = 0; row < packed.size(0); ++row) {
        const bool errored_stream = row < kStep + 1;
        for (size_t token = 0; token < kVocab; ++token) {
            const bool expect_allowed = errored_stream || token != static_cast<size_t>(kMaskedToken);
            EXPECT_EQ(allowedBit(packed, row, static_cast<int64_t>(token)), expect_allowed)
                << "row=" << row << " token=" << token;
        }
    }

    // The errored stream must not accept any speculative token.
    EXPECT_EQ(result.spec_cap_cpu[0].item<int32_t>(), 0);
    EXPECT_EQ(result.spec_cap_cpu[1].item<int32_t>(), kStep);
}

// End-to-end packed-mask application: masked positions become -FLT_MAX, and
// rows without an active processor stay untouched.
TEST(SpecLogitsVerifyRunnerTest, ApplyMaskToLogitsMasksOnlyActiveRows) {
    constexpr size_t  kStreams     = 2;
    constexpr int     kStep        = 1;
    constexpr size_t  kVocab       = 40;
    constexpr int32_t kMaskedToken = 33;

    auto processor = std::make_shared<StubSpecVerifyProcessor>(kMaskedToken);

    auto task = makeTask(kStreams, kStep, kVocab);
    task.active.push_back({processor, /*stream_idx=*/1});

    SpecLogitsVerifyRunner runner;
    auto                   result = runner.run(task);
    syncResult(result);

    auto options = torch::TensorOptions().dtype(torch::kFloat32);
#if USING_CUDA
    options = options.device(torch::kCUDA);
#endif
    auto logits = torch::zeros({static_cast<int64_t>(kStreams) * (kStep + 1), static_cast<int64_t>(kVocab)}, options);
    SpecLogitsVerifyRunner::applyMaskToLogits(logits, result, kVocab);
    auto logits_cpu = logits.cpu();

    const float masked_value = -std::numeric_limits<float>::max();
    for (int64_t row = 0; row < logits_cpu.size(0); ++row) {
        const bool active_row = row >= kStep + 1;
        for (size_t token = 0; token < kVocab; ++token) {
            const float expected = active_row && token == static_cast<size_t>(kMaskedToken) ? masked_value : 0.0f;
            EXPECT_EQ(logits_cpu[row][static_cast<int64_t>(token)].item<float>(), expected)
                << "row=" << row << " token=" << token;
        }
    }
}

// Draft tensors may arrive as [B, P+1] with a leading verify anchor; the
// runner must strip it and hand processors exactly the P proposed tokens.
TEST(SpecLogitsVerifyRunnerTest, LeadingVerifyAnchorIsNotPassedAsProposal) {
    constexpr int    kStep  = 3;
    constexpr size_t kVocab = 35;

    auto processor = std::make_shared<StubSpecVerifyProcessor>(/*masked_token=*/-1);

    auto task         = makeTask(/*total_streams=*/1, kStep, kVocab);
    task.draft_tokens = torch::tensor({99, 11, 12, 13}, torch::kInt32).reshape({1, kStep + 1});
    task.active.push_back({processor, /*stream_idx=*/0});

    SpecLogitsVerifyRunner runner;
    auto                   result = runner.run(task);
    syncResult(result);

    EXPECT_TRUE(result.has_active_processor);
    EXPECT_EQ((std::vector<int32_t>{11, 12, 13}), processor->observedTokens());
}

TEST(SpecLogitsVerifyRunnerTest, ThreeTimeoutsDegradeOnlyTheirRequestAndKeepIndependentThinkingMasks) {
    constexpr int kStep     = 2;
    constexpr int kVocab    = 128;
    constexpr int kBegin    = 7;
    constexpr int kBoundary = 8;
    constexpr int kSuffix   = 9;

    auto input                   = std::make_shared<GenerateInput>();
    input->input_ids             = torch::tensor({11, 12}, torch::kInt32);
    input->generate_config       = std::make_shared<GenerateConfig>();
    auto& config                 = *input->generate_config;
    config.in_think_mode         = true;
    config.max_thinking_tokens   = 3;
    config.begin_think_token_ids = {kBegin};
    config.end_think_token_ids   = {kBoundary, kSuffix};

    // Hand-compose independent processors; the baseline factory still selects
    // grammar instead of installing a separate ThinkMode processor.
    auto degraded         = makeGrammarProcessor();
    auto healthy          = makeGrammarProcessor();
    auto thinking_grammar = makeGrammarProcessor();
    auto thinking         = ThinkModeLogitsProcessor::fromGenerateInput(input, 1);
    ASSERT_TRUE(degraded);
    ASSERT_TRUE(healthy);
    ASSERT_TRUE(thinking_grammar);
    ASSERT_TRUE(thinking);
    degraded->update_timeout_ms_         = 0;
    thinking_grammar->update_timeout_ms_ = 0;
    const auto accepted                  = torch::full({1, 1}, 'a', torch::kInt32);
    for (int round = 0; round < 3; ++round) {
        SCOPED_TRACE(round);
        ASSERT_FALSE(degraded->updateStatus(accepted, 1).has_value());
        ASSERT_FALSE(healthy->updateStatus(accepted, 1).has_value());
        ASSERT_FALSE(thinking_grammar->updateStatus(accepted, 1).has_value());
        ASSERT_FALSE(thinking->updateStatus(accepted, 1).has_value());
        EXPECT_EQ(degraded->updateTimedOut(), round == 2);
        EXPECT_FALSE(healthy->updateTimedOut());
        EXPECT_EQ(thinking_grammar->updateTimedOut(), round == 2);
        EXPECT_TRUE(degraded->isStateful());
        EXPECT_TRUE(thinking_grammar->isStateful());
    }

    auto task = makeTask(/*total_streams=*/3, kStep, kVocab);
    task.draft_tokens =
        torch::tensor(std::vector<int32_t>{'b', 'b', 'a', 'b', kBoundary, kBegin}, torch::kInt32).reshape({3, kStep});
    task.active = {{degraded, 0}, {healthy, 1}, {thinking_grammar, 2}, {thinking, 2}};
    SpecLogitsVerifyRunner runner;
    auto                   result = runner.run(task);
    syncResult(result);
    ASSERT_EQ(result.processor_errors.size(), 3);
    for (const auto& error : result.processor_errors) {
        ASSERT_FALSE(error.has_value());
    }
    ASSERT_EQ(result.packed_allow_mask_cpu_lifetime.size(0), 3 * (kStep + 1));
    EXPECT_EQ(result.spec_cap_cpu[0].item<int32_t>(), kStep);
    EXPECT_EQ(result.spec_cap_cpu[1].item<int32_t>(), 1);
    EXPECT_EQ(result.spec_cap_cpu[2].item<int32_t>(), 1);

    const auto& packed = result.packed_allow_mask_cpu_lifetime;
    for (int row = 0; row <= kStep; ++row) {
        for (int token = 0; token < kVocab; ++token) {
            EXPECT_TRUE(allowedBit(packed, row, token)) << "row=" << row << " token=" << token;
        }
    }
    for (int row = 0; row < 2; ++row) {
        EXPECT_TRUE(allowedBit(packed, kStep + 1 + row, 'a'));
        EXPECT_FALSE(allowedBit(packed, kStep + 1 + row, 'b'));
        for (int token = 0; token < kVocab; ++token) {
            const bool expected = token == (row == 0 ? kBoundary : kSuffix);
            EXPECT_EQ(allowedBit(packed, 2 * (kStep + 1) + row, token), expected)
                << "thinking row=" << row << " token=" << token;
        }
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat32);
#if USING_CUDA
    options = options.device(torch::kCUDA);
#endif
    auto logits = torch::zeros({3 * (kStep + 1), kVocab}, options);
    SpecLogitsVerifyRunner::applyMaskToLogits(logits, result, kVocab);
    const auto logits_cpu = logits.cpu();
    EXPECT_TRUE(torch::equal(logits_cpu.narrow(0, 0, kStep + 1), torch::zeros({kStep + 1, kVocab})));
    EXPECT_LT(logits_cpu[kStep + 1]['b'].item<float>(), -1e20f);
    EXPECT_EQ(logits_cpu[2 * (kStep + 1)][kBoundary].item<float>(), 0.0f);
    EXPECT_LT(logits_cpu[2 * (kStep + 1)][0].item<float>(), -1e20f);
    EXPECT_EQ(logits_cpu[2 * (kStep + 1) + 1][kSuffix].item<float>(), 0.0f);
    EXPECT_LT(logits_cpu[2 * (kStep + 1) + 1][0].item<float>(), -1e20f);
    EXPECT_LT(logits_cpu[2 * (kStep + 1) + 1][kBegin].item<float>(), -1e20f);
    EXPECT_EQ(degraded->committedOutputLen(), 3);
    EXPECT_EQ(healthy->committedOutputLen(), 3);
    EXPECT_EQ(thinking_grammar->committedOutputLen(), 3);
    EXPECT_EQ(thinking->committedOutputLen(), 3);

    // Degraded grammar stops rejecting commits. The independent baseline
    // ThinkMode processor still consumes its complete forced ending sequence.
    ASSERT_FALSE(degraded->updateStatus(torch::full({1, 2}, 'b', torch::kInt32), 2).has_value());
    const auto after_thinking =
        torch::tensor(std::vector<int32_t>{kBoundary, kSuffix, 'b'}, torch::kInt32).reshape({1, 3});
    ASSERT_FALSE(thinking_grammar->updateStatus(after_thinking, 3).has_value());
    ASSERT_FALSE(thinking->updateStatus(after_thinking, 3).has_value());
    EXPECT_EQ(degraded->committedOutputLen(), 5);
    EXPECT_EQ(healthy->committedOutputLen(), 3);
    EXPECT_EQ(thinking_grammar->committedOutputLen(), 6);
    EXPECT_EQ(thinking->committedOutputLen(), 6);
    EXPECT_FALSE(healthy->updateTimedOut());
}

TEST(SpecLogitsVerifyRunnerTest, FastVerifyResetsStreakAndThirdTimeoutPreservesCurrentResults) {
    constexpr int kStep    = 2;
    constexpr int kVocab   = 128;
    auto          degraded = makeGrammarProcessor();
    auto          healthy  = makeGrammarProcessor();
    ASSERT_TRUE(degraded);
    ASSERT_TRUE(healthy);
    auto task         = makeTask(/*total_streams=*/2, kStep, kVocab);
    task.draft_tokens = torch::tensor(std::vector<int32_t>{'a', 'b', 'a', 'b'}, torch::kInt32).reshape({2, kStep});
    task.active       = {{degraded, 0}, {healthy, 1}};
    SpecLogitsVerifyRunner runner;
    const auto             normal = runner.run(task);
    syncResult(normal);
    ASSERT_FALSE(normal.processor_errors[0].has_value());
    ASSERT_FALSE(normal.processor_errors[1].has_value());
    // The runner reuses its buffers. Keep an independent snapshot of all rows,
    // including the unused suffix, to compare with the third slow result.
    const auto normal_mask = normal.packed_allow_mask_cpu_lifetime.clone();
    const auto normal_caps = normal.spec_cap_cpu.clone();
    for (int round = 0; round < 7; ++round) {
        SCOPED_TRACE(round);
        degraded->update_timeout_ms_ = round == 2 || round == 6 ? std::numeric_limits<int>::max() : 0;
        auto result                  = runner.run(task);
        syncResult(result);
        ASSERT_EQ(result.processor_errors.size(), 2);
        ASSERT_FALSE(result.processor_errors[0].has_value());
        ASSERT_FALSE(result.processor_errors[1].has_value());
        const bool timed_out = round >= 5;
        const bool bypass    = round == 6;
        EXPECT_EQ(degraded->updateTimedOut(), timed_out);
        EXPECT_FALSE(healthy->updateTimedOut());
        EXPECT_EQ(degraded->committedOutputLen(), round * kStep);
        EXPECT_EQ(result.spec_cap_cpu[0].item<int32_t>(), bypass ? kStep : 1);
        EXPECT_EQ(result.spec_cap_cpu[1].item<int32_t>(), 1);
        const auto& packed = result.packed_allow_mask_cpu_lifetime;
        ASSERT_EQ(packed.size(0), 2 * (kStep + 1));
        if (!bypass) {
            EXPECT_TRUE(torch::equal(packed, normal_mask));
            EXPECT_TRUE(torch::equal(result.spec_cap_cpu, normal_caps));
        } else {
            EXPECT_TRUE(
                packed.narrow(0, 0, kStep + 1).eq(SpecLogitsProcessorRequest::kBitmaskAllowAll).all().item<bool>());
        }
        EXPECT_TRUE(torch::equal(packed.narrow(0, kStep + 1, kStep + 1), normal_mask.narrow(0, kStep + 1, kStep + 1)));
        if (result.packed_allow_mask_gpu.defined()) {
            EXPECT_TRUE(torch::equal(result.packed_allow_mask_gpu.cpu(), packed));
            EXPECT_TRUE(torch::equal(result.spec_cap_gpu.cpu(), result.spec_cap_cpu));
        }
        for (int row = 0; row <= kStep; ++row) {
            if (bypass) {
                for (int token = 0; token < kVocab; ++token) {
                    EXPECT_TRUE(allowedBit(packed, row, token)) << "row=" << row << " token=" << token;
                }
            } else if (row < kStep) {
                EXPECT_TRUE(allowedBit(packed, row, 'a'));
                EXPECT_FALSE(allowedBit(packed, row, 'b'));
            }
            if (row < kStep) {
                EXPECT_TRUE(allowedBit(packed, kStep + 1 + row, 'a'));
                EXPECT_FALSE(allowedBit(packed, kStep + 1 + row, 'b'));
            }
        }
        if (round == 1 || round == 4) {
            for (const int budget : {0, std::numeric_limits<int>::max()}) {
                degraded->update_timeout_ms_ = budget;
                ASSERT_FALSE(degraded->updateStatus(torch::empty({1, 0}, torch::kInt32), 0).has_value());
                EXPECT_FALSE(degraded->updateTimedOut());
                EXPECT_EQ(degraded->committedOutputLen(), round * kStep);
            }
        }
        // Fast commits reset only the update streak. The third slow verify
        // still commits its constrained result; only the bypass round uses 'b'.
        degraded->update_timeout_ms_ = std::numeric_limits<int>::max();
        ASSERT_FALSE(
            degraded->updateStatus(torch::full({1, kStep}, bypass ? 'b' : 'a', torch::kInt32), kStep).has_value());
        ASSERT_FALSE(healthy->updateStatus(torch::full({1, kStep}, 'a', torch::kInt32), kStep).has_value());
        EXPECT_EQ(degraded->committedOutputLen(), (round + 1) * kStep);
        EXPECT_EQ(healthy->committedOutputLen(), (round + 1) * kStep);
        EXPECT_TRUE(degraded->isStateful());
    }
}

TEST(SpecLogitsVerifyRunnerTest, ThirdSlowVerifyPreservesCapBoundariesAndBonusMaskUntilNextRound) {
    constexpr int kStep  = 2;
    constexpr int kVocab = 128;
    for (const int expected_cap : {0, kStep}) {
        SCOPED_TRACE(expected_cap);
        auto processor = makeGrammarProcessor();
        ASSERT_TRUE(processor);
        auto task         = makeTask(/*total_streams=*/1, kStep, kVocab);
        task.draft_tokens = torch::full({1, kStep}, 'a', torch::kInt32);
        if (expected_cap == 0) {
            task.draft_tokens[0][0] = 'b';
        }
        task.active = {{processor, 0}};
        SpecLogitsVerifyRunner runner;
        const auto             normal = runner.run(task);
        syncResult(normal);
        ASSERT_FALSE(normal.processor_errors[0].has_value());
        ASSERT_EQ(normal.spec_cap_cpu[0].item<int32_t>(), expected_cap);
        const auto normal_mask = normal.packed_allow_mask_cpu_lifetime.clone();
        const auto normal_caps = normal.spec_cap_cpu.clone();
        EXPECT_TRUE(allowedBit(normal_mask, expected_cap, 'a'));
        EXPECT_FALSE(allowedBit(normal_mask, expected_cap, 'b'));

        for (int round = 0; round < 4; ++round) {
            SCOPED_TRACE(round);
            const bool bypass             = round == 3;
            processor->update_timeout_ms_ = bypass ? std::numeric_limits<int>::max() : 0;
            auto result                   = runner.run(task);
            syncResult(result);
            ASSERT_FALSE(result.processor_errors[0].has_value());
            EXPECT_EQ(processor->updateTimedOut(), round >= 2);
            EXPECT_EQ(processor->committedOutputLen(), 0);
            if (bypass) {
                EXPECT_EQ(result.spec_cap_cpu[0].item<int32_t>(), kStep);
                EXPECT_TRUE(result.packed_allow_mask_cpu_lifetime.eq(SpecLogitsProcessorRequest::kBitmaskAllowAll)
                                .all()
                                .item<bool>());
            } else {
                EXPECT_TRUE(torch::equal(result.packed_allow_mask_cpu_lifetime, normal_mask));
                EXPECT_TRUE(torch::equal(result.spec_cap_cpu, normal_caps));
            }
            if (result.packed_allow_mask_gpu.defined()) {
                EXPECT_TRUE(torch::equal(result.packed_allow_mask_gpu.cpu(), result.packed_allow_mask_cpu_lifetime));
                EXPECT_TRUE(torch::equal(result.spec_cap_gpu.cpu(), result.spec_cap_cpu));
            }
        }
    }
}

TEST(SpecLogitsVerifyRunnerTest, RealErrorOnThirdSlowVerifyIsNotHiddenByTimeout) {
    constexpr int kStep   = 2;
    constexpr int kVocab  = 128;
    auto          failing = makeGrammarProcessor();
    auto          healthy = makeGrammarProcessor();
    ASSERT_TRUE(failing);
    ASSERT_TRUE(healthy);
    failing->update_timeout_ms_ = 0;
    auto task                   = makeTask(/*total_streams=*/2, kStep, kVocab);
    task.draft_tokens = torch::tensor(std::vector<int32_t>{'a', 'b', 'a', 'b'}, torch::kInt32).reshape({2, kStep});
    task.active       = {{failing, 0}, {healthy, 1}};
    SpecLogitsVerifyRunner runner;
    for (int round = 0; round < 2; ++round) {
        auto result = runner.run(task);
        syncResult(result);
        ASSERT_EQ(result.processor_errors.size(), 2);
        ASSERT_FALSE(result.processor_errors[0].has_value());
        ASSERT_FALSE(result.processor_errors[1].has_value());
        EXPECT_FALSE(failing->updateTimedOut());
        EXPECT_EQ(result.spec_cap_cpu[0].item<int32_t>(), 1);
    }
    // Exercise the real verify validation path after two slow successes.
    failing->eos_token_id_ = kVocab;
    auto result            = runner.run(task);
    syncResult(result);
    ASSERT_EQ(result.processor_errors.size(), 2);
    ASSERT_TRUE(result.processor_errors[0].has_value());
    EXPECT_EQ(result.processor_errors[0]->code(), ErrorCode::GRAMMAR_EOS_OUT_OF_VOCAB);
    EXPECT_FALSE(result.processor_errors[1].has_value());
    EXPECT_EQ(result.spec_cap_cpu[0].item<int32_t>(), 0);
    EXPECT_EQ(result.spec_cap_cpu[1].item<int32_t>(), 1);
    EXPECT_FALSE(failing->updateTimedOut());
    EXPECT_FALSE(healthy->updateTimedOut());
    EXPECT_EQ(failing->committedOutputLen(), 0);
    EXPECT_EQ(healthy->committedOutputLen(), 0);
    for (int row = 0; row < kStep; ++row) {
        EXPECT_TRUE(allowedBit(result.packed_allow_mask_cpu_lifetime, kStep + 1 + row, 'a'));
        EXPECT_FALSE(allowedBit(result.packed_allow_mask_cpu_lifetime, kStep + 1 + row, 'b'));
    }
}

#if USING_CUDA
namespace {

// A bounded wait makes an accidental synchronous upload fail without hanging
// the test. Release the stream even when an assertion exits the scope early.
class BlockedCudaStream {
public:
    explicit BlockedCudaStream(c10::cuda::CUDAStream stream): stream_(stream), ready_(release_.get_future().share()) {
        C10_CUDA_CHECK(cudaLaunchHostFunc(
            stream_.stream(),
            [](void* data) {
                auto& blocker     = *static_cast<BlockedCudaStream*>(data);
                blocker.timed_out = blocker.ready_.wait_for(std::chrono::seconds(5)) == std::future_status::timeout;
            },
            this));
    }

    ~BlockedCudaStream() {
        release_.set_value();
        stream_.synchronize();
    }

    std::atomic<bool> timed_out{false};

private:
    c10::cuda::CUDAStream    stream_;
    std::promise<void>       release_;
    std::shared_future<void> ready_;
};

}  // namespace

TEST(SpecLogitsVerifyRunnerTest, PendingGrammarMaskSurvivesThirdCommitTimeoutAndNextRoundAllowsAllTokens) {
    constexpr int              kStep  = 2;
    constexpr int              kVocab = 128;
    const auto                 stream = c10::cuda::getStreamFromPool();
    c10::cuda::CUDAStreamGuard guard(stream);
    auto                       processor = makeGrammarProcessor();
    ASSERT_TRUE(processor);
    processor->update_timeout_ms_ = 0;
    const auto accepted           = torch::full({1, 1}, 'a', torch::kInt32);
    for (int round = 0; round < 2; ++round) {
        ASSERT_FALSE(processor->updateStatus(accepted, 1).has_value());
        EXPECT_FALSE(processor->updateTimedOut());
    }

    auto task         = makeTask(/*total_streams=*/1, kStep, kVocab);
    task.draft_tokens = torch::tensor(std::vector<int32_t>{'a', 'b'}, torch::kInt32).reshape({1, kStep});
    task.active       = {{processor, 0}};
    SpecLogitsVerifyRunner runner;
    // Allocate the runner's pinned/device buffers and warm the mask kernel
    // before the callback prevents any work on this non-default CUDA stream.
    processor->update_timeout_ms_ = std::numeric_limits<int>::max();
    auto warm                     = runner.run(task);
    ASSERT_FALSE(warm.processor_errors[0].has_value());
    auto logits = torch::ones({kStep + 1, kVocab}, torch::TensorOptions().device(torch::kCUDA));
    SpecLogitsVerifyRunner::applyMaskToLogits(logits, warm, kVocab);
    warm.consumed_event->record(stream.unwrap());
    logits.fill_(1.0f);
    stream.synchronize();

    SpecLogitsVerifyRunner::LaunchResult pending;
    {
        BlockedCudaStream blocker(stream);
        pending = runner.run(task);
        ASSERT_FALSE(pending.processor_errors[0].has_value());
        ASSERT_TRUE(pending.ready_event);
        EXPECT_FALSE(pending.ready_event->query());
        EXPECT_EQ(pending.spec_cap_cpu[0].item<int32_t>(), 1);
        const auto*                mask = pending.packed_allow_mask_cpu_lifetime.data_ptr<int32_t>();
        const std::vector<int32_t> snapshot(mask, mask + pending.packed_allow_mask_cpu_lifetime.numel());

        // Only the commit path is timed out in this regression. The preceding
        // fast verify calls must not reset its two prior slow commits.
        processor->update_timeout_ms_ = 0;
        ASSERT_FALSE(processor->updateStatus(accepted, 1).has_value());
        EXPECT_TRUE(processor->updateTimedOut());
        EXPECT_TRUE(processor->isStateful());
        EXPECT_EQ(processor->committedOutputLen(), 3);
        std::vector<int32_t>             allow_mask((kStep + 1) * (kVocab / 32), 0);
        const int32_t                    next_draft[] = {'b', 'b'};
        const SpecLogitsProcessorRequest request{next_draft, kStep, allow_mask.data(), kVocab / 32, kVocab};
        const auto                       next_cap = processor->prepareSpeculative(request);
        ASSERT_TRUE(next_cap.ok()) << next_cap.status().ToString();
        EXPECT_EQ(next_cap.value(), kStep);
        EXPECT_TRUE(std::all_of(allow_mask.begin(), allow_mask.end(), [](int32_t word) {
            return word == SpecLogitsProcessorRequest::kBitmaskAllowAll;
        }));
        ASSERT_FALSE(processor->updateStatus(torch::full({1, 2}, 'b', torch::kInt32), 2).has_value());
        EXPECT_EQ(processor->committedOutputLen(), 5);
        EXPECT_TRUE(std::equal(snapshot.begin(), snapshot.end(), mask));
        EXPECT_EQ(pending.spec_cap_cpu[0].item<int32_t>(), 1);

        SpecLogitsVerifyRunner::applyMaskToLogits(logits, pending, kVocab);
        pending.consumed_event->record(stream.unwrap());
        EXPECT_FALSE(blocker.timed_out.load()) << "MTP mask upload waited for preceding GPU work";
        EXPECT_FALSE(pending.ready_event->query());
    }
    EXPECT_TRUE(torch::equal(pending.packed_allow_mask_gpu.cpu(), pending.packed_allow_mask_cpu_lifetime));
    EXPECT_EQ(pending.spec_cap_gpu.cpu()[0].item<int32_t>(), 1);
    const auto constrained_logits = logits.cpu();
    for (int row = 0; row < 2; ++row) {
        EXPECT_EQ(constrained_logits[row]['a'].item<float>(), 1.0f);
        EXPECT_LT(constrained_logits[row]['b'].item<float>(), -1e20f);
    }

    // The previous snapshot has now been consumed. Reusing the same runner must
    // overwrite every old mask row, including the bonus row, and restore cap=P.
    task.draft_tokens = torch::full({1, kStep}, 'b', torch::kInt32);
    auto next         = runner.run(task);
    ASSERT_FALSE(next.processor_errors[0].has_value());
    EXPECT_EQ(next.spec_cap_cpu[0].item<int32_t>(), kStep);
    EXPECT_TRUE(
        next.packed_allow_mask_cpu_lifetime.eq(SpecLogitsProcessorRequest::kBitmaskAllowAll).all().item<bool>());
    logits.fill_(1.0f);
    SpecLogitsVerifyRunner::applyMaskToLogits(logits, next, kVocab);
    next.consumed_event->record(stream.unwrap());
    stream.synchronize();
    EXPECT_EQ(next.spec_cap_gpu.cpu()[0].item<int32_t>(), kStep);
    EXPECT_TRUE(torch::equal(logits.cpu(), torch::ones({kStep + 1, kVocab})));
    EXPECT_TRUE(processor->updateTimedOut());
    EXPECT_EQ(processor->committedOutputLen(), 5);
}

TEST(SpecLogitsVerifyRunnerTest, PendingMaskSurvivesThirdVerifyTimeoutWithCurrentResultsPreserved) {
    constexpr int              kStep     = 2;
    constexpr int              kVocab    = 128;
    constexpr int              kBegin    = 7;
    constexpr int              kBoundary = 8;
    const auto                 stream    = c10::cuda::getStreamFromPool();
    c10::cuda::CUDAStreamGuard guard(stream);
    auto                       input = std::make_shared<GenerateInput>();
    input->input_ids                 = torch::tensor({11, 12}, torch::kInt32);
    input->generate_config           = std::make_shared<GenerateConfig>();
    auto& config                     = *input->generate_config;
    config.in_think_mode             = true;
    config.max_thinking_tokens       = 3;
    config.begin_think_token_ids     = {kBegin};
    config.end_think_token_ids       = {kBoundary};
    auto degraded                    = makeGrammarProcessor();
    auto healthy                     = makeGrammarProcessor();
    auto thinking                    = ThinkModeLogitsProcessor::fromGenerateInput(input, 1);
    ASSERT_TRUE(degraded);
    ASSERT_TRUE(healthy);
    ASSERT_TRUE(thinking);
    const std::vector<BaseLogitsProcessorPtr> processors{degraded, healthy, thinking};
    const auto                                accepted = torch::full({1, 1}, 'a', torch::kInt32);
    // Close thinking independently. Its existing post-thinking policy keeps
    // masking the opening marker after grammar is disabled.
    ASSERT_FALSE(degraded->updateStatus(accepted, 1).has_value());
    ASSERT_FALSE(healthy->updateStatus(accepted, 1).has_value());
    ASSERT_FALSE(thinking->updateStatus(torch::full({1, 1}, kBoundary, torch::kInt32), 1).has_value());

    auto task         = makeTask(/*total_streams=*/2, kStep, kVocab);
    task.draft_tokens = torch::tensor(std::vector<int32_t>{'a', 'b', 'a', 'b'}, torch::kInt32).reshape({2, kStep});
    task.active       = {{degraded, 0}, {healthy, 1}, {thinking, 0}};
    SpecLogitsVerifyRunner runner;
    auto                   warm = runner.run(task);
    ASSERT_FALSE(warm.processor_errors[0].has_value());
    ASSERT_FALSE(warm.processor_errors[1].has_value());
    auto logits = torch::ones({2 * (kStep + 1), kVocab}, torch::TensorOptions().device(torch::kCUDA));
    SpecLogitsVerifyRunner::applyMaskToLogits(logits, warm, kVocab);
    warm.consumed_event->record(stream.unwrap());
    logits.fill_(1.0f);
    stream.synchronize();

    degraded->update_timeout_ms_ = 0;
    auto first                   = runner.run(task);
    syncResult(first);
    ASSERT_FALSE(first.processor_errors[0].has_value());
    ASSERT_FALSE(first.processor_errors[1].has_value());
    EXPECT_EQ(first.spec_cap_cpu[0].item<int32_t>(), 1);
    EXPECT_FALSE(degraded->updateTimedOut());
    degraded->update_timeout_ms_ = std::numeric_limits<int>::max();
    for (const auto& processor : processors) {
        ASSERT_FALSE(processor->updateStatus(accepted, 1).has_value());
    }

    SpecLogitsVerifyRunner::LaunchResult pending;
    {
        BlockedCudaStream blocker(stream);
        degraded->update_timeout_ms_ = 0;
        pending                      = runner.run(task);  // second slow verify, upload remains pending
        ASSERT_FALSE(pending.processor_errors[0].has_value());
        ASSERT_FALSE(pending.processor_errors[1].has_value());
        ASSERT_TRUE(pending.ready_event);
        EXPECT_FALSE(pending.ready_event->query());
        EXPECT_FALSE(degraded->updateTimedOut());
        const auto*                mask = pending.packed_allow_mask_cpu_lifetime.data_ptr<int32_t>();
        const std::vector<int32_t> snapshot(mask, mask + pending.packed_allow_mask_cpu_lifetime.numel());
        degraded->update_timeout_ms_ = std::numeric_limits<int>::max();
        for (const auto& processor : processors) {
            ASSERT_FALSE(processor->updateStatus(accepted, 1).has_value());
        }

        // A fast commit cannot erase the verify streak. The third verification
        // keeps the same mask and truncated cap as a healthy grammar request.
        const int32_t        draft[] = {'a', 'b'};
        std::vector<int32_t> normal_mask((kStep + 1) * (kVocab / 32), SpecLogitsProcessorRequest::kBitmaskAllowAll);
        const SpecLogitsProcessorRequest normal_request{draft, kStep, normal_mask.data(), kVocab / 32, kVocab};
        const auto                       normal_cap = healthy->prepareSpeculative(normal_request);
        ASSERT_TRUE(normal_cap.ok()) << normal_cap.status().ToString();
        ASSERT_EQ(normal_cap.value(), 1);
        degraded->update_timeout_ms_ = 0;
        std::vector<int32_t>             allow_mask(normal_mask.size(), SpecLogitsProcessorRequest::kBitmaskAllowAll);
        const SpecLogitsProcessorRequest request{draft, kStep, allow_mask.data(), kVocab / 32, kVocab};
        const auto                       cap = degraded->prepareSpeculative(request);
        ASSERT_TRUE(cap.ok()) << cap.status().ToString();
        EXPECT_EQ(cap.value(), normal_cap.value());
        EXPECT_EQ(allow_mask, normal_mask);
        EXPECT_TRUE(degraded->updateTimedOut());
        EXPECT_TRUE(degraded->isStateful());
        EXPECT_FALSE(healthy->updateTimedOut());
        EXPECT_EQ(degraded->committedOutputLen(), 3);

        // Bypass starts with the next call and overwrites every row, even when
        // its destination initially rejects every token and the draft is illegal.
        std::fill(allow_mask.begin(), allow_mask.end(), 0);
        const int32_t                    next_draft[] = {'b', 'b'};
        const SpecLogitsProcessorRequest next_request{next_draft, kStep, allow_mask.data(), kVocab / 32, kVocab};
        const auto                       next_cap = degraded->prepareSpeculative(next_request);
        ASSERT_TRUE(next_cap.ok()) << next_cap.status().ToString();
        EXPECT_EQ(next_cap.value(), kStep);
        EXPECT_TRUE(std::all_of(allow_mask.begin(), allow_mask.end(), [](int32_t word) {
            return word == SpecLogitsProcessorRequest::kBitmaskAllowAll;
        }));
        EXPECT_TRUE(std::equal(snapshot.begin(), snapshot.end(), mask));
        EXPECT_EQ(pending.spec_cap_cpu[0].item<int32_t>(), 1);
        EXPECT_EQ(pending.spec_cap_cpu[1].item<int32_t>(), 1);
        SpecLogitsVerifyRunner::applyMaskToLogits(logits, pending, kVocab);
        pending.consumed_event->record(stream.unwrap());
        EXPECT_FALSE(blocker.timed_out.load());
        EXPECT_FALSE(pending.ready_event->query());
    }
    EXPECT_TRUE(torch::equal(pending.packed_allow_mask_gpu.cpu(), pending.packed_allow_mask_cpu_lifetime));
    EXPECT_TRUE(torch::equal(pending.spec_cap_gpu.cpu(), pending.spec_cap_cpu));
    const auto healthy_mask = pending.packed_allow_mask_cpu_lifetime.narrow(0, kStep + 1, kStep + 1).clone();
    const auto old_logits   = logits.cpu();
    for (int stream_idx = 0; stream_idx < 2; ++stream_idx) {
        for (int row = 0; row < 2; ++row) {
            EXPECT_EQ(old_logits[stream_idx * (kStep + 1) + row]['a'].item<float>(), 1.0f);
            EXPECT_LT(old_logits[stream_idx * (kStep + 1) + row]['b'].item<float>(), -1e20f);
        }
    }

    degraded->update_timeout_ms_ = std::numeric_limits<int>::max();
    task.draft_tokens = torch::tensor(std::vector<int32_t>{'b', 'b', 'a', 'b'}, torch::kInt32).reshape({2, kStep});
    auto next         = runner.run(task);
    ASSERT_FALSE(next.processor_errors[0].has_value());
    ASSERT_FALSE(next.processor_errors[1].has_value());
    EXPECT_EQ(next.spec_cap_cpu[0].item<int32_t>(), kStep);
    EXPECT_EQ(next.spec_cap_cpu[1].item<int32_t>(), 1);
    EXPECT_TRUE(torch::equal(next.packed_allow_mask_cpu_lifetime.narrow(0, kStep + 1, kStep + 1), healthy_mask));
    logits.fill_(1.0f);
    SpecLogitsVerifyRunner::applyMaskToLogits(logits, next, kVocab);
    next.consumed_event->record(stream.unwrap());
    stream.synchronize();
    const auto next_logits = logits.cpu();
    for (int row = 0; row <= kStep; ++row) {
        for (int token = 0; token < kVocab; ++token) {
            const bool allowed = token != kBegin;
            EXPECT_EQ(allowedBit(next.packed_allow_mask_cpu_lifetime, row, token), allowed);
            if (allowed) {
                EXPECT_EQ(next_logits[row][token].item<float>(), 1.0f);
            } else {
                EXPECT_LT(next_logits[row][token].item<float>(), -1e20f);
            }
        }
    }
    for (int row = 0; row < 2; ++row) {
        EXPECT_EQ(next_logits[kStep + 1 + row]['a'].item<float>(), 1.0f);
        EXPECT_LT(next_logits[kStep + 1 + row]['b'].item<float>(), -1e20f);
    }
    EXPECT_TRUE(torch::equal(next.packed_allow_mask_gpu.cpu(), next.packed_allow_mask_cpu_lifetime));
    EXPECT_TRUE(torch::equal(next.spec_cap_gpu.cpu(), next.spec_cap_cpu));
    EXPECT_TRUE(degraded->updateTimedOut());
    EXPECT_FALSE(healthy->updateTimedOut());
    for (const auto& processor : processors) {
        EXPECT_EQ(processor->committedOutputLen(), 3);
    }
}
#endif

}  // namespace rtp_llm
