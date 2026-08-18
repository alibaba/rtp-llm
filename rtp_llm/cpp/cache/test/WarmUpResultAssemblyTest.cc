#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

#include "rtp_llm/cpp/cache/WarmUpResultAssembly.h"

namespace rtp_llm {
namespace {

constexpr size_t MiB = 1024 * 1024;
constexpr size_t GiB = 1024 * MiB;

MemoryStatus makeStatus(size_t torch_peak, size_t torch_current, size_t non_torch, size_t available) {
    // Named assignment: four adjacent size_t fields would make a positional aggregate
    // silently accept any permutation.
    MemoryStatus status;
    status.max_consumed_bytes           = torch_peak;
    status.torch_current_increase_bytes = torch_current;
    status.non_torch_increase_bytes     = non_torch;
    status.available_bytes              = available;
    return status;
}

// The sizing term is the whole growth over the window, and the base predates all of it: serving has
// to allocate the released share again and the retained share is still held, so the pairing does not
// care which is which.
TEST(WarmUpResultAssemblyTest, PairsThePreWarmupPoolWithTheTotalGrowth) {
    // Keep peak and current deliberately different: using current here would undercount by 9 GiB.
    const auto peak = makeStatus(/*torch_peak=*/12 * GiB, /*torch_current=*/3 * GiB, /*non_torch=*/70 * MiB, 0);
    // Post-teardown counters deliberately disagree with the peak sample: only its available field
    // belongs in WarmUpResult, while both growth terms must come from peak.
    const auto post_teardown =
        makeStatus(/*torch_peak=*/0, /*torch_current=*/300 * MiB, /*non_torch=*/128 * MiB, /*available=*/60 * GiB);

    const auto result =
        assembleWarmUpResult(/*pre_warmup_available_bytes=*/60 * GiB + 270 * MiB, peak, post_teardown, true);

    EXPECT_EQ(result.available_bytes_pre_warmup, 60 * GiB + 270 * MiB);
    EXPECT_EQ(result.device_reserved_bytes, 60 * GiB);
    EXPECT_EQ(result.measured_total_growth_bytes, 12 * GiB + 70 * MiB);
    EXPECT_TRUE(result.measurement_trusted);
    // The safety condition the caller checks: what the warmup permanently cost the device (the
    // difference between the two pools) must not exceed what it was measured to grow.
    EXPECT_LE(result.available_bytes_pre_warmup - result.device_reserved_bytes, result.measured_total_growth_bytes);
}

// A zero measurement is not rejected here: MemoryEvaluationHelper is what degrades it to the
// no-warmup formula, and this layer must hand it the 0 to key off.
TEST(WarmUpResultAssemblyTest, ZeroGrowthProducesZeroMeasurement) {
    const auto result = assembleWarmUpResult(
        /*pre_warmup_available_bytes=*/40 * GiB, makeStatus(0, 0, 0, 0), makeStatus(0, 0, 0, 40 * GiB), true);

    EXPECT_EQ(result.measured_total_growth_bytes, 0u);
    EXPECT_EQ(result.available_bytes_pre_warmup, 40 * GiB);
    EXPECT_EQ(result.device_reserved_bytes, 40 * GiB);
}

TEST(WarmUpResultAssemblyTest, RejectsGrowthSumOverflow) {
    constexpr size_t kMax = std::numeric_limits<size_t>::max();
    // Both growth terms are the full counter value: their sum cannot be represented.
    const auto peak          = makeStatus(/*torch_peak=*/kMax, /*torch_current=*/kMax, /*non_torch=*/kMax, 0);
    const auto post_teardown = makeStatus(0, /*torch_current=*/0, /*non_torch=*/0, 0);

    EXPECT_THROW(assembleWarmUpResult(0, peak, post_teardown, true), std::overflow_error);
}

TEST(WarmUpResultAssemblyTest, TrustMustBeDeclaredExplicitly) {
    const auto peak          = makeStatus(1 * GiB, 1 * GiB, 0, 0);
    const auto post_teardown = makeStatus(0, 0, 0, 40 * GiB);

    EXPECT_FALSE(assembleWarmUpResult(41 * GiB, peak, post_teardown, false).measurement_trusted);
    EXPECT_TRUE(assembleWarmUpResult(41 * GiB, peak, post_teardown, true).measurement_trusted);
}

// The shared helper both layers use: normal case is the difference between the two free pools;
// the inverted case (an unrelated process freed memory between the samples) clamps to 0 rather
// than wrapping around into a huge value.
TEST(WarmUpResultAssemblyTest, PoolShrinkBytesIsTheDifferenceClampedAtZero) {
    WarmUpResult result;
    result.available_bytes_pre_warmup = 60 * GiB;
    result.device_reserved_bytes      = 59 * GiB;
    EXPECT_EQ(poolShrinkBytes(result), 1 * GiB);

    result.device_reserved_bytes = 61 * GiB;  // post-teardown pool larger than pre-warmup
    EXPECT_EQ(poolShrinkBytes(result), 0u);

    result.device_reserved_bytes = 60 * GiB;
    EXPECT_EQ(poolShrinkBytes(result), 0u);
}

}  // namespace
}  // namespace rtp_llm
