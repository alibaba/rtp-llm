#include "rtp_llm/cpp/normal_engine/speculative/MtpTargetLogprobs.h"

#include <gtest/gtest.h>
#include <torch/cuda.h>

namespace rtp_llm {
namespace {

constexpr int64_t kRows        = 5;
constexpr int64_t kVocab       = 1031;
constexpr int64_t kPaddedVocab = 1056;

torch::Tensor makeLogits(const torch::Device& device, torch::ScalarType dtype) {
    auto values = torch::arange(kRows * kPaddedVocab, torch::kFloat32).reshape({kRows, kPaddedVocab});
    auto logits = (values.remainder(127) - 63) / 8 + values / 10000;
    // Padding must never affect either normalization or top-k.
    logits.narrow(1, kVocab, kPaddedVocab - kVocab).fill_(1000);
    return logits.to(device, dtype);
}

void expectLogprobs(const MtpTargetLogprobs&    result,
                    const torch::Tensor&        reference,
                    const torch::Tensor&        emitted_ids,
                    const std::vector<int64_t>& rows,
                    int64_t                     top_k) {
    auto indices         = torch::tensor(rows, torch::kInt64);
    auto expected        = reference.index_select(0, indices);
    auto selected_ids    = emitted_ids.cpu().to(torch::kInt64).index_select(0, indices);
    auto expected_tokens = expected.gather(1, selected_ids.unsqueeze(1)).squeeze(1);
    ASSERT_TRUE(result.finalized());
    EXPECT_FALSE(result.retainsFullLmHeadStorage());
    EXPECT_TRUE(torch::allclose(result.token_logprobs.cpu(), expected_tokens, 1e-5, 1e-5));
    ASSERT_EQ(result.top_logprobs.size(0), rows.size());
    ASSERT_EQ(result.top_logprobs.size(1), top_k);
    ASSERT_EQ(result.top_logprob_token_ids.sizes(), result.top_logprobs.sizes());
    if (top_k > 0) {
        auto top_ids   = result.top_logprob_token_ids.cpu().to(torch::kInt64);
        auto top_probs = result.top_logprobs.cpu();
        // Compare probabilities as well as their association with returned IDs;
        // reduced-precision inputs may have valid ties with different top-k IDs.
        EXPECT_TRUE(torch::allclose(top_probs, expected.gather(1, top_ids), 1e-5, 1e-5));
        EXPECT_TRUE(torch::allclose(top_probs, std::get<0>(expected.topk(top_k, -1)), 1e-5, 1e-5));
    }
}

void checkComputeSnapshot(const torch::Device& device, torch::ScalarType dtype) {
    const std::vector<int64_t> all_rows{0, 1, 2, 3, 4};
    for (const auto& rows : std::vector<std::vector<int64_t>>{{}, all_rows, {4, 1, 3}}) {
        for (const int64_t top_k : {0, 7}) {
            SCOPED_TRACE(::testing::Message() << "dtype=" << dtype << " rows=" << rows.size() << " k=" << top_k);
            auto logits    = makeLogits(device, dtype);
            auto reference = logits.cpu().to(torch::kFloat32).narrow(1, 0, kVocab).log_softmax(-1);
            auto emitted   = torch::tensor({0, 1030, 513, 33, 1000}, torch::kInt64).to(device);
            auto result    = computeMtpTargetLogprobs(logits, top_k, kVocab, rows);
            EXPECT_FALSE(result.raw_logits.is_alias_of(logits));
            // Reproduce temperature and token-penalty mutation between capture
            // and finalization without depending on sampler randomness.
            logits.mul_(0.5);
            logits.select(1, 33).sub_(10);
            finalizeMtpTargetLogprobs(result, emitted);
            expectLogprobs(result, reference, emitted, rows.empty() ? all_rows : rows, top_k);
        }
    }
}

void checkSelectedRows(const torch::Device& device, torch::ScalarType dtype) {
    for (const int64_t top_k : {0, 7, 20}) {
        SCOPED_TRACE(::testing::Message() << "dtype=" << dtype << " k=" << top_k);
        auto                       logits    = makeLogits(device, dtype);
        auto                       reference = logits.cpu().to(torch::kFloat32).narrow(1, 0, kVocab).log_softmax(-1);
        auto                       emitted   = torch::tensor({0, 1030, 513, 33, 1000}, torch::kInt32).to(device);
        auto                       result    = captureMtpDecodeTargetLogprobs(logits, top_k, kVocab);
        const std::vector<int64_t> selected_rows{4, 0, 2};
        finalizeSelectedMtpTargetLogprobs(result, emitted, selected_rows);
        expectLogprobs(result, reference, emitted, selected_rows, top_k);
    }
}

TEST(MtpTargetLogprobsTest, ComputeSnapshotSurvivesSamplerMutationCpu) {
    checkComputeSnapshot(torch::Device(torch::kCPU), torch::kFloat32);
}

TEST(MtpTargetLogprobsTest, SelectedRowsMatchLogSoftmaxCpu) {
    checkSelectedRows(torch::Device(torch::kCPU), torch::kFloat32);
}

#if USING_CUDA
TEST(MtpTargetLogprobsTest, ComputeSnapshotSurvivesSamplerMutationCuda) {
    ASSERT_TRUE(torch::cuda::is_available());
    for (auto dtype : {torch::kFloat32, torch::kFloat16, torch::kBFloat16}) {
        checkComputeSnapshot(torch::Device(torch::kCUDA), dtype);
    }
}

TEST(MtpTargetLogprobsTest, SelectedRowsMatchLogSoftmaxCuda) {
    ASSERT_TRUE(torch::cuda::is_available());
    for (auto dtype : {torch::kFloat32, torch::kFloat16, torch::kBFloat16}) {
        checkSelectedRows(torch::Device(torch::kCUDA), dtype);
    }
}
#endif

}  // namespace
}  // namespace rtp_llm
