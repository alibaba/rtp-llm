#include <gtest/gtest.h>

#include "rtp_llm/models_py/bindings/core/DeviceData.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {
namespace {

constexpr size_t MiB = 1024 * 1024;
constexpr size_t GiB = 1024 * MiB;

// These two cases cover the TraceMemoryState class in isolation -- NOT the production
// entry points (setTraceMemory/finishTraceMemory/getGpuExecStatus); those are covered on
// the linked ExecOps.cc by trace_memory_ops_test, which needs a GPU.
TEST(TraceMemoryStateTest, PhaseTransitions) {
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

TEST(TraceMemoryStateTest, FinishIsIdempotentAndFinishedIsNotTerminal) {
    TraceMemoryState state;
    state.finish();
    state.finish();
    EXPECT_EQ(state.get(), static_cast<int>(TraceMemoryPhase::Finished));
    EXPECT_FALSE(state.isActive());

    // Finished is not terminal: a second model build re-activates the phase on purpose
    // (see the ExecOps.h comment on activate()), so pin that the transition takes effect
    // and that finishing again afterwards works.
    state.activate();
    EXPECT_TRUE(state.isActive());
    EXPECT_EQ(state.get(), static_cast<int>(TraceMemoryPhase::Active));
    state.finish();
    EXPECT_EQ(state.get(), static_cast<int>(TraceMemoryPhase::Finished));
    EXPECT_FALSE(state.isActive());
}

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
