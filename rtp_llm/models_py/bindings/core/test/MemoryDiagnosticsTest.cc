#include <gtest/gtest.h>

#include "rtp_llm/models_py/bindings/core/DeviceData.h"

namespace rtp_llm {
namespace {

constexpr size_t MiB = 1024 * 1024;
constexpr size_t GiB = 1024 * MiB;

// The deltas are reported separately on purpose: the torch peak growth plus the non-torch growth at
// the end of the forward is the total the KV-cache budget reserves, while the torch *current*
// reading is a diagnostic showing how much the warmup left behind after its teardown.
TEST(MemoryGrowthTest, SeparatesTorchAndNonTorchGrowth) {
    const auto growth = calculateMemoryGrowth(
        /*reserved_baseline_bytes=*/2 * GiB,
        /*reserved_peak_bytes=*/5 * GiB,
        /*reserved_current_bytes=*/3 * GiB,
        /*cuda_used_baseline_bytes=*/3 * GiB,
        /*cuda_used_current_bytes=*/6 * GiB);

    EXPECT_EQ(growth.torch_peak_increase_bytes, 3 * GiB);
    EXPECT_EQ(growth.torch_current_increase_bytes, 1 * GiB);
    EXPECT_EQ(growth.non_torch_increase_bytes, 2 * GiB);
}

TEST(MemoryGrowthTest, NonTorchGrowthIsIndependentOfTheTorchPeak) {
    // A pure non-torch grower (lazily created cuBLAS/NCCL workspace, ~70 MiB in practice) must
    // leave the reserved quantity at zero rather than inflating it.
    const auto growth = calculateMemoryGrowth(
        /*reserved_baseline_bytes=*/2 * GiB,
        /*reserved_peak_bytes=*/2 * GiB,
        /*reserved_current_bytes=*/2 * GiB,
        /*cuda_used_baseline_bytes=*/2 * GiB,
        /*cuda_used_current_bytes=*/2 * GiB + 70 * MiB);

    EXPECT_EQ(growth.torch_peak_increase_bytes, 0u);
    EXPECT_EQ(growth.torch_current_increase_bytes, 0u);
    EXPECT_EQ(growth.non_torch_increase_bytes, 70 * MiB);
}

TEST(MemoryGrowthTest, ClampsCounterRegressionsToZero) {
    const auto growth = calculateMemoryGrowth(
        /*reserved_baseline_bytes=*/4 * GiB,
        /*reserved_peak_bytes=*/3 * GiB,
        /*reserved_current_bytes=*/5 * GiB,
        /*cuda_used_baseline_bytes=*/8 * GiB,
        /*cuda_used_current_bytes=*/7 * GiB);

    EXPECT_EQ(growth.torch_peak_increase_bytes, 0u);
    // current above baseline is real growth even when the peak counter regressed.
    EXPECT_EQ(growth.torch_current_increase_bytes, 1 * GiB);
    EXPECT_EQ(growth.non_torch_increase_bytes, 0u);
}

}  // namespace
}  // namespace rtp_llm
