#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

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

}  // namespace rtp_llm
