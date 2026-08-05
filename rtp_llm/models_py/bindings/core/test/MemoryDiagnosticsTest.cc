#include <gtest/gtest.h>

#include <stdexcept>

#include "rtp_llm/models_py/bindings/core/DeviceData.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {
namespace {

constexpr size_t MiB = 1024 * 1024;
constexpr size_t GiB = 1024 * MiB;

TEST(TraceMemoryStateTest, CoversStartupStateTransitionsWithoutGpu) {
    TraceMemoryState state;
    EXPECT_EQ(state.get(), static_cast<int>(TraceMemoryPhase::Pending));
    EXPECT_FALSE(state.isActive());

    state.activate();
    EXPECT_EQ(state.get(), static_cast<int>(TraceMemoryPhase::Active));
    EXPECT_TRUE(state.isActive());

    state.finish();
    EXPECT_EQ(state.get(), static_cast<int>(TraceMemoryPhase::Finished));
    EXPECT_FALSE(state.isActive());
}

TEST(TraceMemoryStateTest, FinishIsSafeBeforeActivationAndAfterFailureCleanup) {
    TraceMemoryState state;
    state.finish();
    state.finish();
    EXPECT_EQ(state.get(), static_cast<int>(TraceMemoryPhase::Finished));
    EXPECT_FALSE(state.isActive());

    state.activate();
    try {
        throw std::runtime_error("simulated warmup failure");
    } catch (const std::runtime_error&) { state.finish(); }
    EXPECT_EQ(state.get(), static_cast<int>(TraceMemoryPhase::Finished));
    EXPECT_FALSE(state.isActive());
}

// The two deltas are reported separately on purpose: the torch peak always feeds the KV-cache
// budget, while the non-torch delta only does so for the share released by the warmup teardown
// (transientNonTorchBytes) -- whatever survives it is already missing from the post-warmup
// available_bytes the budget is computed from.
TEST(MemoryGrowthTest, SeparatesTorchAndNonTorchGrowth) {
    const auto growth = calculateMemoryGrowth(
        /*reserved_baseline_bytes=*/2 * GiB,
        /*reserved_peak_bytes=*/5 * GiB,
        /*reserved_current_bytes=*/3 * GiB,
        /*cuda_used_baseline_bytes=*/3 * GiB,
        /*cuda_used_current_bytes=*/6 * GiB);

    EXPECT_EQ(growth.torch_peak_increase_bytes, 3 * GiB);
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
    EXPECT_EQ(growth.non_torch_increase_bytes, 70 * MiB);
}

TEST(TransientNonTorchTest, ReservesOnlyWhatTheTeardownHandedBack) {
    // PREFILL shape: process-global lazy initialisation, all of it still resident after teardown,
    // so nothing needs reserving -- available_bytes already excludes it.
    EXPECT_EQ(transientNonTorchBytes(/*in_forward=*/70 * MiB, /*post_teardown=*/70 * MiB), 0u);

    // DECODE shape: part of the growth is per-captured-graph driver state released with the
    // executor, so serving re-allocates it and it must be reserved.
    EXPECT_EQ(transientNonTorchBytes(/*in_forward=*/500 * MiB, /*post_teardown=*/70 * MiB), 430 * MiB);
}

TEST(TransientNonTorchTest, ClampsWhenResidentExceedsTheInForwardSample) {
    // Nothing forces the post-teardown sample to be the smaller one (an unrelated process can grow
    // device usage between the two reads); never wrap around into a huge reservation.
    EXPECT_EQ(transientNonTorchBytes(/*in_forward=*/70 * MiB, /*post_teardown=*/128 * MiB), 0u);
}

TEST(MemoryGrowthTest, ClampsCounterRegressionsToZero) {
    const auto growth = calculateMemoryGrowth(
        /*reserved_baseline_bytes=*/4 * GiB,
        /*reserved_peak_bytes=*/3 * GiB,
        /*reserved_current_bytes=*/5 * GiB,
        /*cuda_used_baseline_bytes=*/8 * GiB,
        /*cuda_used_current_bytes=*/7 * GiB);

    EXPECT_EQ(growth.torch_peak_increase_bytes, 0u);
    EXPECT_EQ(growth.non_torch_increase_bytes, 0u);
}

}  // namespace
}  // namespace rtp_llm
