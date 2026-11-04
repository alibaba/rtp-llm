#include "rtp_llm/models_py/bindings/core/SampleTemperature.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

// Pure-CPU unit tests for the shared sampling-temperature policy
// (SampleTemperature.h). No CUDA, no torch: the header is portable and is the
// single source of truth for both the CUDA greedy sampler (CudaSampleOp.cc) and
// the upstream "the sampler will not temper logits in place this round"
// decisions.
namespace rtp_llm {
namespace {

// The legacy full-vocabulary pass is batchApplyTemperaturePenalty
// (models_py/bindings/common/kernels/sampling_penalty_kernels.cu):
//   inv_temperatures[i] = 1.0f / (temperatures[i] + 1e-6f);
//   logit *= inv_temperatures[batch_idx];
// The fold must use exactly that FP32 factor, otherwise candidate values shift
// relative to the path it replaces.
TEST(SampleTemperatureTest, InvScaleMatchesLegacyKernelFormula) {
    for (const float temperature : {0.05f, 0.3f, 0.7f, 1.0f, 1.5f, 2.0f}) {
        const float expected = 1.0f / (temperature + 1e-6f);
        EXPECT_EQ(sampleTemperatureInvScale(temperature), expected) << "temperature=" << temperature;

        // Cross-check against an independent double-precision evaluation: the
        // two may differ by a few ULPs of FP32, never more.
        const double reference = 1.0 / (static_cast<double>(temperature) + 1e-6);
        EXPECT_NEAR(static_cast<double>(sampleTemperatureInvScale(temperature)), reference, 1e-7 * reference);
    }
}

// T == 1 does not yield exactly 1.0f (the kernel adds 1e-6 before dividing):
// this is why an all-T == 1 batch must stay on the legacy path instead of
// folding a 0.999999 scale into the candidates.
TEST(SampleTemperatureTest, InvScaleIsNotExactlyOneAtUnitTemperature) {
    EXPECT_LT(sampleTemperatureInvScale(1.0f), 1.0f);
    EXPECT_NEAR(sampleTemperatureInvScale(1.0f), 1.0f, 1e-5f);
}

// The fold's core claim: for T > 0 the scaled row has the same ordering, so
// top-k selects the same candidate set, and softmax over those k scaled values
// matches softmax over the tempered full row restricted to them.
TEST(SampleTemperatureTest, CandidateSetAndProbabilitiesMatchTemperedRow) {
    const float              temperature = 0.3f;
    const float              scale       = sampleTemperatureInvScale(temperature);
    const std::vector<float> logits      = {2.5f, -1.0f, 0.0f, 4.25f, 4.2499f, -20.0f, 1.5f, 3.75f, 8.0f, -0.5f};
    const int64_t            top_k       = 4;

    // Legacy: temper first (multiplicative scale, as the kernel does), then top-k.
    std::vector<float> tempered(logits.size());
    std::transform(logits.begin(), logits.end(), tempered.begin(), [scale](float x) { return x * scale; });

    const auto topk_indices = [top_k](const std::vector<float>& values) {
        std::vector<int64_t> indices(values.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(), [&values](int64_t lhs, int64_t rhs) {
            return values[lhs] > values[rhs];
        });
        indices.resize(top_k);
        std::sort(indices.begin(), indices.end());
        return indices;
    };

    const auto legacy_indices = topk_indices(tempered);
    const auto folded_indices = topk_indices(logits);
    EXPECT_EQ(legacy_indices, folded_indices) << "monotone scaling must preserve the candidate set";

    // softmax over the selected values: legacy uses the tempered values, the fold
    // uses raw values scaled by the same factor.
    std::vector<double> legacy_probs;
    std::vector<double> folded_probs;
    double              legacy_max = -std::numeric_limits<double>::infinity();
    double              folded_max = -std::numeric_limits<double>::infinity();
    for (const int64_t index : legacy_indices) {
        legacy_max = std::max<double>(legacy_max, tempered[index]);
    }
    for (const int64_t index : folded_indices) {
        folded_max = std::max<double>(folded_max, static_cast<double>(logits[index] * scale));
    }
    double legacy_sum = 0.0;
    double folded_sum = 0.0;
    for (const int64_t index : legacy_indices) {
        legacy_probs.push_back(std::exp(static_cast<double>(tempered[index]) - legacy_max));
        legacy_sum += legacy_probs.back();
    }
    for (const int64_t index : folded_indices) {
        folded_probs.push_back(std::exp(static_cast<double>(logits[index] * scale) - folded_max));
        folded_sum += folded_probs.back();
    }
    ASSERT_EQ(legacy_probs.size(), folded_probs.size());
    for (size_t i = 0; i < legacy_probs.size(); ++i) {
        EXPECT_NEAR(folded_probs[i] / folded_sum, legacy_probs[i] / legacy_sum, 1e-6);
    }
}

// Gating table of the shared decision. Every gate below is a correctness
// condition (not a tunable switch): each one names a case where folding the
// temperature into the candidate scaling would not be bit-identical to the
// full-vocabulary pass it replaces.
class GreedyTemperatureFoldedForRequestTest: public ::testing::Test {
protected:
    static constexpr int64_t kBatch     = 4;
    static constexpr int     kForceTopk = 4096;

    std::vector<float> temperatures_{0.3f, 0.3f, 0.3f, 0.3f};
};

TEST_F(GreedyTemperatureFoldedForRequestTest, FoldsWithStochasticTemperatureAndFixedWidth) {
    EXPECT_TRUE(
        greedyTemperatureFoldedForRequest(kBatch, temperatures_.data(), kForceTopk, false, false, false, false));
}

TEST_F(GreedyTemperatureFoldedForRequestTest, MixedTemperaturesStillFold) {
    // A row at T == 1 is fine as long as some row is scaled: the legacy kernel
    // tempered every row of the batch, so the fold must scale them all too.
    temperatures_ = {0.3f, 1.0f, 0.7f, 2.0f};
    EXPECT_TRUE(
        greedyTemperatureFoldedForRequest(kBatch, temperatures_.data(), kForceTopk, false, false, false, false));
}

TEST_F(GreedyTemperatureFoldedForRequestTest, AllUnitTemperaturesKeepLegacyPath) {
    temperatures_ = {1.0f, 1.0f, 1.0f, 1.0f};
    EXPECT_FALSE(
        greedyTemperatureFoldedForRequest(kBatch, temperatures_.data(), kForceTopk, false, false, false, false));
}

TEST_F(GreedyTemperatureFoldedForRequestTest, ZeroTemperatureRowKeepsLegacyPath) {
    temperatures_ = {0.3f, 0.0f, 0.3f, 0.3f};
    EXPECT_FALSE(
        greedyTemperatureFoldedForRequest(kBatch, temperatures_.data(), kForceTopk, false, false, false, false));
}

TEST_F(GreedyTemperatureFoldedForRequestTest, DisqualifyingFactsKeepLegacyPath) {
    // No fixed candidate width: the sampler softmaxes the whole row.
    EXPECT_FALSE(greedyTemperatureFoldedForRequest(kBatch, temperatures_.data(), 0, false, false, false, false));
    // A do_sample == false row is restored to untouched logits.
    EXPECT_FALSE(
        greedyTemperatureFoldedForRequest(kBatch, temperatures_.data(), kForceTopk, true, false, false, false));
    // Full-width probability consumers.
    EXPECT_FALSE(
        greedyTemperatureFoldedForRequest(kBatch, temperatures_.data(), kForceTopk, false, true, false, false));
    EXPECT_FALSE(
        greedyTemperatureFoldedForRequest(kBatch, temperatures_.data(), kForceTopk, false, false, true, false));
    // Active logits penalty: rep(x / T) != rep(x) / T.
    EXPECT_FALSE(
        greedyTemperatureFoldedForRequest(kBatch, temperatures_.data(), kForceTopk, false, false, false, true));
    // Degenerate inputs.
    EXPECT_FALSE(greedyTemperatureFoldedForRequest(0, temperatures_.data(), kForceTopk, false, false, false, false));
    EXPECT_FALSE(greedyTemperatureFoldedForRequest(kBatch, nullptr, kForceTopk, false, false, false, false));
}

}  // namespace
}  // namespace rtp_llm
