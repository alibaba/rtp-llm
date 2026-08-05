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
        input.safety_ratio = ratio;
        EXPECT_THROW(calculateRuntimeMemorySizing(input), std::invalid_argument);
    }
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
