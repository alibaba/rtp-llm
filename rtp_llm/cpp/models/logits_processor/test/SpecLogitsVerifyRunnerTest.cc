#include <limits>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"

namespace rtp_llm {
namespace {

class StubSpecProcessor: public SpecLogitsProcessor {
public:
    StubSpecProcessor(int masked_token, int cap): masked_token_(masked_token), cap_(cap) {}

    bool isSpecVerifyEligible() const override {
        return true;
    }

    int tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) override {
        observed_tokens_.assign(request.draft_tokens, request.draft_tokens + request.propose_step);
        for (int row = 0; row <= request.propose_step; ++row) {
            auto* row_ptr = request.bitmask_cpu_out + static_cast<size_t>(row) * request.bitmask_size_int32;
            row_ptr[masked_token_ / 32] &= ~(1u << (masked_token_ % 32));
        }
        return cap_;
    }

    const std::vector<int32_t>& observedTokens() const {
        return observed_tokens_;
    }

private:
    int                  masked_token_;
    int                  cap_;
    std::vector<int32_t> observed_tokens_;
};

bool allowedBit(const torch::Tensor& packed_mask, int64_t row, int64_t token) {
    const auto word = static_cast<uint32_t>(packed_mask[row][token / 32].item<int32_t>());
    return ((word >> (token % 32)) & 1u) != 0u;
}

class SpecLogitsVerifyRunnerTest: public DeviceTestBase {};

TEST_F(SpecLogitsVerifyRunnerTest, SparseRowsArePackedMappedAndApplied) {
    constexpr size_t kStreams = 3;
    constexpr int    kStep    = 2;
    constexpr size_t kVocab   = 35;

    auto mask_34 = std::make_shared<StubSpecProcessor>(34, 1);
    auto mask_33 = std::make_shared<StubSpecProcessor>(33, 0);

    SpecLogitsVerifyRunner::LaunchTask task;
    task.total_streams = kStreams;
    task.propose_step  = kStep;
    task.vocab_size    = kVocab;
    // P+1 form carries a leading verify anchor that must not reach processors.
    task.draft_tokens = torch::tensor({{90, 1, 2}, {91, 3, 4}, {92, 5, 6}}, torch::kInt32).to(torch::kCUDA);
    task.active.push_back({mask_34, 1, 3, 101, 10, 4});
    task.active.push_back({mask_33, 1, 4, 101, 10, 4});

    SpecLogitsVerifyRunner runner;
    auto                   result = runner.buildInline(task);
    ASSERT_TRUE(result.has_active_processor);
    ASSERT_NE(result.ready_event, nullptr);
    result.ready_event->synchronize();

    ASSERT_EQ(result.packed_allow_mask_cpu_lifetime.size(0), kStep + 1);
    ASSERT_EQ(result.packed_allow_mask_cpu_lifetime.size(1), 2);
    ASSERT_EQ(result.logits_row_indices_cpu_lifetime.size(0), kStep + 1);
    EXPECT_EQ(result.logits_row_indices_cpu_lifetime[0].item<int32_t>(), 3);
    EXPECT_EQ(result.logits_row_indices_cpu_lifetime[1].item<int32_t>(), 4);
    EXPECT_EQ(result.logits_row_indices_cpu_lifetime[2].item<int32_t>(), 5);
    for (int row = 0; row <= kStep; ++row) {
        EXPECT_FALSE(allowedBit(result.packed_allow_mask_cpu_lifetime, row, 33));
        EXPECT_FALSE(allowedBit(result.packed_allow_mask_cpu_lifetime, row, 34));
        EXPECT_TRUE(allowedBit(result.packed_allow_mask_cpu_lifetime, row, 32));
    }
    EXPECT_EQ((std::vector<int32_t>{3, 4}), mask_34->observedTokens());
    EXPECT_EQ(result.spec_cap_cpu_lifetime[0].item<int32_t>(), kStep);
    EXPECT_EQ(result.spec_cap_cpu_lifetime[1].item<int32_t>(), 0);
    EXPECT_EQ(result.spec_cap_cpu_lifetime[2].item<int32_t>(), kStep);
    EXPECT_EQ(result.applied_processors.size(), 2);

    auto logits = torch::zeros({static_cast<int64_t>(kStreams) * (kStep + 1), static_cast<int64_t>(kVocab)},
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    SpecLogitsVerifyRunner::applyMaskToLogits(logits, result, kVocab);
    result.consumed_event->record(cuda_graph::graphGetCurrentStream());
    auto logits_cpu = logits.cpu();
    const float masked_value = -std::numeric_limits<float>::max();
    for (int64_t row = 0; row < logits_cpu.size(0); ++row) {
        const bool active = row >= 3 && row <= 5;
        EXPECT_EQ(logits_cpu[row][33].item<float>(), active ? masked_value : 0.0f);
        EXPECT_EQ(logits_cpu[row][34].item<float>(), active ? masked_value : 0.0f);
        EXPECT_EQ(logits_cpu[row][32].item<float>(), 0.0f);
    }
}

}  // namespace
}  // namespace rtp_llm
