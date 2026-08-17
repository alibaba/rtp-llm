#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

#include "rtp_llm/cpp/cache/RuntimeMemorySizing.h"

namespace rtp_llm {
namespace {

constexpr size_t MiB = 1024 * 1024;
constexpr size_t GiB = 1024 * MiB;

// The sizing math owns no product defaults, so these cases state their own
// inputs. They happen to match the KVCacheConfig defaults, but nothing here
// asserts that: these are just round numbers that make the expectations easy to
// read. Changing the product default must not change this test.
constexpr double kSafetyRatio   = 0.05;
constexpr size_t kNoWarmupFloor = 2 * GiB;

TEST(RuntimeMemorySizingTest, AddsSafetyHeadroomToWarmupPeak) {
    RuntimeMemorySizingInput input;
    input.has_warmup               = true;
    input.safety_ratio             = kSafetyRatio;
    input.configured_reserve_bytes = 1 * GiB;
    input.warmup_required_bytes    = 6 * GiB;
    input.sampler_required_bytes   = 512 * MiB;
    input.total_gpu_bytes          = 80 * GiB;

    const auto result = calculateRuntimeMemorySizing(input);

    EXPECT_EQ(result.safety_ratio_bytes, 4 * GiB);
    EXPECT_EQ(result.runtime_required_bytes, 10 * GiB);
}

// Forbids merging no_warmup_floor into the warmup branch. The floor (8 GiB)
// strictly dominates every other term AND the expected result (3 GiB), so any
// use of it in this branch -- inside the max() ("never reserve less than the
// floor" is a tempting-looking fix) or additive -- changes the assertion
// value. A traced measurement deliberately replaces the guesswork floors;
// operators needing an absolute floor on this path pass configured_reserve.
TEST(RuntimeMemorySizingTest, WarmupBranchIgnoresNoWarmupFloor) {
    RuntimeMemorySizingInput input;
    input.has_warmup               = true;
    input.safety_ratio             = kSafetyRatio;
    input.configured_reserve_bytes = 1 * GiB;
    input.warmup_required_bytes    = 2 * GiB;
    input.sampler_required_bytes   = 0;
    input.total_gpu_bytes          = 20 * GiB;
    input.no_warmup_floor_bytes    = 8 * GiB;

    const auto result = calculateRuntimeMemorySizing(input);

    // max(1, 2, 0) + 0.05 * 20 = 3 GiB; with the floor merged it would be >= 9 GiB.
    EXPECT_EQ(result.safety_ratio_bytes, 1 * GiB);
    EXPECT_EQ(result.runtime_required_bytes, 3 * GiB);
}

TEST(RuntimeMemorySizingTest, UsesConfiguredOrSamplerFloorDuringWarmup) {
    RuntimeMemorySizingInput input;
    input.has_warmup               = true;
    input.safety_ratio             = kSafetyRatio;
    input.configured_reserve_bytes = 8 * GiB;
    input.warmup_required_bytes    = 2 * GiB;
    input.sampler_required_bytes   = 3 * GiB;
    input.total_gpu_bytes          = 20 * GiB;

    EXPECT_EQ(calculateRuntimeMemorySizing(input).runtime_required_bytes, 9 * GiB);

    input.configured_reserve_bytes = 1 * GiB;
    input.sampler_required_bytes   = 9 * GiB;
    EXPECT_EQ(calculateRuntimeMemorySizing(input).runtime_required_bytes, 10 * GiB);
}

// Pins the warmup branch's floor behaviour on a small GPU with a small measured peak: the
// pre-warmup-feature hard minimum max(2048 MiB, 5% * total) does NOT apply here, so with the
// default configured reserve (1 GiB) the result can legitimately drop below 2048 MiB.
// This is a documented behaviour change (release note + --runtime_mem_safety_ratio help);
// deployments needing the old absolute floor pass it via configured_reserve_bytes
// (--reserver_runtime_mem_mb). If this assertion starts failing because a floor was merged
// into the warmup branch, that is a deliberate semantics change and must be released as such.
TEST(RuntimeMemorySizingTest, WarmupBranchMayReserveLessThanLegacyHardMinimumOnSmallGpus) {
    RuntimeMemorySizingInput input;
    input.has_warmup               = true;
    input.safety_ratio             = kSafetyRatio;
    input.configured_reserve_bytes = 1 * GiB;      // product default reserver_runtime_mem_mb
    input.warmup_required_bytes    = 256 * MiB;    // small measured peak (tiny model)
    input.sampler_required_bytes   = 128 * MiB;
    input.total_gpu_bytes          = 16 * GiB;
    input.no_warmup_floor_bytes    = kNoWarmupFloor;

    const auto result = calculateRuntimeMemorySizing(input);

    // max(1 GiB, 256 MiB, 128 MiB) + 5% * 16 GiB = 1 GiB + 819.2 MiB < legacy 2048 MiB minimum.
    // Literals rather than the production expression: trunc(0.05 * 17179869184) = 858993459.
    EXPECT_EQ(result.safety_ratio_bytes, 858993459u);
    EXPECT_EQ(result.runtime_required_bytes, 1 * GiB + 858993459u);
    EXPECT_LT(result.runtime_required_bytes, kNoWarmupFloor);
    // ... and the case is flagged, which is what makes it greppable in a rollout
    // (MemoryEvaluationHelper turns this into a WARNING naming --reserver_runtime_mem_mb).
    EXPECT_TRUE(result.warmup_below_no_warmup_floor);
}

// The flag must stay off whenever the reserve is not actually undercut, otherwise the
// WARNING it drives becomes noise operators learn to ignore. total_gpu_bytes is 20 GiB
// throughout so the safety term (1 GiB) stays below the 2 GiB floor -- on a large GPU the
// ratio term alone clears the floor and the interesting cases become unreachable.
TEST(RuntimeMemorySizingTest, BelowFloorFlagIsOffWhenTheFloorIsNotUndercut) {
    RuntimeMemorySizingInput input;
    input.has_warmup               = true;
    input.safety_ratio             = kSafetyRatio;
    input.configured_reserve_bytes = 1 * GiB;
    input.warmup_required_bytes    = 6 * GiB;  // large measured peak clears the floor
    input.sampler_required_bytes   = 0;
    input.total_gpu_bytes          = 20 * GiB;  // safety term = 1 GiB
    input.no_warmup_floor_bytes    = kNoWarmupFloor;

    auto result = calculateRuntimeMemorySizing(input);
    EXPECT_EQ(result.safety_ratio_bytes, 1 * GiB);
    EXPECT_EQ(result.runtime_required_bytes, 7 * GiB);
    EXPECT_FALSE(result.warmup_below_no_warmup_floor);

    // Exactly at the floor is not below it: max(1 GiB, 1 GiB, 0) + 1 GiB == 2 GiB.
    input.warmup_required_bytes = 1 * GiB;
    result                      = calculateRuntimeMemorySizing(input);
    EXPECT_EQ(result.runtime_required_bytes, kNoWarmupFloor);
    EXPECT_FALSE(result.warmup_below_no_warmup_floor);

    // One byte below the floor does flag, pinning the boundary from the other side.
    input.configured_reserve_bytes = 1 * GiB - 1;
    input.warmup_required_bytes    = 0;
    result                         = calculateRuntimeMemorySizing(input);
    EXPECT_EQ(result.runtime_required_bytes, kNoWarmupFloor - 1);
    EXPECT_TRUE(result.warmup_below_no_warmup_floor);

    // A zero floor means "no floor configured": nothing to warn about.
    input.no_warmup_floor_bytes = 0;
    result                      = calculateRuntimeMemorySizing(input);
    EXPECT_FALSE(result.warmup_below_no_warmup_floor);

    // The no-warmup branch keeps the floor inside the max(), so it can never undercut it.
    input.has_warmup            = false;
    input.no_warmup_floor_bytes = kNoWarmupFloor;
    result                      = calculateRuntimeMemorySizing(input);
    EXPECT_GE(result.runtime_required_bytes, kNoWarmupFloor);
    EXPECT_FALSE(result.warmup_below_no_warmup_floor);
}

// The no-warmup branch keeps the pre-warmup-feature formula bit-for-bit:
// max(configured, sampler, no_warmup_floor, ratio * total). The ratio term is a
// floor inside the max(), NOT additive headroom -- asserting equality with the
// binding term pins that, because an additive formula would exceed it.
TEST(RuntimeMemorySizingTest, NoWarmupKeepsLegacyMaxSemantics) {
    RuntimeMemorySizingInput input;
    input.has_warmup               = false;
    input.safety_ratio             = kSafetyRatio;
    input.configured_reserve_bytes = 1 * GiB;
    input.sampler_required_bytes   = 512 * MiB;
    input.total_gpu_bytes          = 80 * GiB;
    input.no_warmup_floor_bytes    = kNoWarmupFloor;

    // ratio floor binds: 5% of 80 GiB = 4 GiB, exactly the term, nothing added.
    auto result = calculateRuntimeMemorySizing(input);
    EXPECT_EQ(result.safety_ratio_bytes, 4 * GiB);
    EXPECT_EQ(result.runtime_required_bytes, 4 * GiB);

    // configured binds: the ratio term must not be added on top.
    input.configured_reserve_bytes = 8 * GiB;
    result                         = calculateRuntimeMemorySizing(input);
    EXPECT_EQ(result.runtime_required_bytes, 8 * GiB);
}

TEST(RuntimeMemorySizingTest, NoWarmupHonorsAbsoluteAndSamplerFloors) {
    RuntimeMemorySizingInput input;
    input.has_warmup             = false;
    input.safety_ratio           = kSafetyRatio;
    input.no_warmup_floor_bytes  = kNoWarmupFloor;
    input.total_gpu_bytes        = 20 * GiB;
    input.sampler_required_bytes = 3 * GiB;

    // sampler (3 GiB) > floor (2 GiB) > ratio term (1 GiB).
    EXPECT_EQ(calculateRuntimeMemorySizing(input).runtime_required_bytes, 3 * GiB);

    input.sampler_required_bytes = 0;
    EXPECT_EQ(calculateRuntimeMemorySizing(input).runtime_required_bytes, 2 * GiB);
}

// Locks the lower bound of the accepted safety_ratio range: 0.0 is legal, not a
// rejected value, and it degenerates the warmup path to plain max(...) with no
// headroom. Without this, RejectsInvalidSafetyRatios alone would still pass if
// the guard were tightened to ratio > 0.
TEST(RuntimeMemorySizingTest, AcceptsZeroSafetyRatioAsLowerBound) {
    RuntimeMemorySizingInput input;
    input.has_warmup               = true;
    input.safety_ratio             = 0.0;
    input.configured_reserve_bytes = 1 * GiB;
    input.warmup_required_bytes    = 6 * GiB;
    input.sampler_required_bytes   = 512 * MiB;
    input.total_gpu_bytes          = 80 * GiB;

    const auto result = calculateRuntimeMemorySizing(input);

    EXPECT_EQ(result.safety_ratio_bytes, 0u);
    EXPECT_EQ(result.runtime_required_bytes, 6 * GiB);
}

TEST(RuntimeMemorySizingTest, RejectsInvalidSafetyRatios) {
    RuntimeMemorySizingInput input;
    for (double ratio :
         {-0.01, 1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        SCOPED_TRACE(ratio);
        input.safety_ratio = ratio;
        EXPECT_THROW(calculateRuntimeMemorySizing(input), std::invalid_argument);
    }
}

// Symmetric to AcceptsZeroSafetyRatioAsLowerBound: a high-but-legal ratio near the
// exclusive upper bound is accepted, and the term computes exactly. 0.75 x 80 GiB is
// exactly representable in double (60 GiB), so EXPECT_EQ is safe; switching to a
// non-representable ratio (0.9/0.99) requires a rounding-direction analysis first.
TEST(RuntimeMemorySizingTest, AcceptsHighSafetyRatioBelowOne) {
    RuntimeMemorySizingInput input;
    input.has_warmup               = false;
    input.safety_ratio             = 0.75;
    input.configured_reserve_bytes = 1 * GiB;
    input.total_gpu_bytes          = 80 * GiB;
    input.no_warmup_floor_bytes    = kNoWarmupFloor;

    const auto result = calculateRuntimeMemorySizing(input);

    EXPECT_EQ(result.safety_ratio_bytes, 60 * GiB);
    EXPECT_EQ(result.runtime_required_bytes, 60 * GiB);
}

TEST(RuntimeMemorySizingTest, RejectsSafetyHeadroomAdditionOverflow) {
    RuntimeMemorySizingInput input;
    input.has_warmup               = true;
    input.configured_reserve_bytes = std::numeric_limits<size_t>::max();
    input.total_gpu_bytes          = 100;
    input.safety_ratio             = 0.5;

    EXPECT_THROW(calculateRuntimeMemorySizing(input), std::overflow_error);

    // The no-warmup branch never adds, so the same inputs cannot overflow there.
    input.has_warmup = false;
    EXPECT_EQ(calculateRuntimeMemorySizing(input).runtime_required_bytes, std::numeric_limits<size_t>::max());
}

}  // namespace
}  // namespace rtp_llm
