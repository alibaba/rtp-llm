#include <gtest/gtest.h>

#include <limits>

#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"
#include "rtp_llm/models_py/bindings/core/DeviceData.h"

namespace rtp_llm {
namespace {

constexpr size_t MiB = 1024 * 1024;
constexpr size_t GiB = 1024 * MiB;

TEST(RuntimeMemorySizingTest, AddsSafetyHeadroomToWarmupPeak) {
    RuntimeMemorySizingInput input;
    input.has_warmup               = true;
    input.configured_reserve_bytes = 1 * GiB;
    input.warmup_required_bytes    = 6 * GiB;
    input.sampler_required_bytes   = 512 * MiB;
    input.total_gpu_bytes          = 80 * GiB;

    const auto result = calculateRuntimeMemorySizing(input);

    EXPECT_EQ(result.safety_headroom_bytes, 4 * GiB);
    EXPECT_EQ(result.runtime_required_bytes, 10 * GiB);
}

TEST(RuntimeMemorySizingTest, UsesConfiguredOrSamplerFloorDuringWarmup) {
    RuntimeMemorySizingInput input;
    input.has_warmup               = true;
    input.configured_reserve_bytes = 8 * GiB;
    input.warmup_required_bytes    = 2 * GiB;
    input.sampler_required_bytes   = 3 * GiB;
    input.total_gpu_bytes          = 20 * GiB;

    EXPECT_EQ(calculateRuntimeMemorySizing(input).runtime_required_bytes, 9 * GiB);

    input.configured_reserve_bytes = 1 * GiB;
    input.sampler_required_bytes   = 9 * GiB;
    EXPECT_EQ(calculateRuntimeMemorySizing(input).runtime_required_bytes, 10 * GiB);
}

TEST(RuntimeMemorySizingTest, PreservesLegacyNoWarmupFormula) {
    RuntimeMemorySizingInput input;
    input.has_warmup               = false;
    input.configured_reserve_bytes = 1 * GiB;
    input.sampler_required_bytes   = 512 * MiB;
    input.total_gpu_bytes          = 80 * GiB;
    input.no_warmup_floor_bytes    = 2 * GiB;

    auto result = calculateRuntimeMemorySizing(input);
    EXPECT_EQ(result.safety_headroom_bytes, 4 * GiB);
    EXPECT_EQ(result.runtime_required_bytes, 4 * GiB);

    input.configured_reserve_bytes = 8 * GiB;
    result                         = calculateRuntimeMemorySizing(input);
    EXPECT_EQ(result.runtime_required_bytes, 8 * GiB);
}

TEST(RuntimeMemorySizingTest, NoWarmupHonorsAbsoluteAndSamplerFloors) {
    RuntimeMemorySizingInput input;
    input.total_gpu_bytes        = 20 * GiB;
    input.sampler_required_bytes = 3 * GiB;

    EXPECT_EQ(calculateRuntimeMemorySizing(input).runtime_required_bytes, 3 * GiB);

    input.sampler_required_bytes = 0;
    EXPECT_EQ(calculateRuntimeMemorySizing(input).runtime_required_bytes, 2 * GiB);
}

TEST(RuntimeMemorySizingTest, RejectsInvalidSafetyRatios) {
    RuntimeMemorySizingInput input;
    for (double ratio :
         {-0.01, 1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        input.safety_ratio = ratio;
        EXPECT_THROW(calculateRuntimeMemorySizing(input), std::invalid_argument);
    }
}

TEST(RuntimeMemorySizingTest, RejectsWarmupAdditionOverflow) {
    RuntimeMemorySizingInput input;
    input.has_warmup               = true;
    input.configured_reserve_bytes = std::numeric_limits<size_t>::max();
    input.total_gpu_bytes          = 100;
    input.safety_ratio             = 0.5;

    EXPECT_THROW(calculateRuntimeMemorySizing(input), std::overflow_error);
}

TEST(MemoryGrowthTest, SeparatesTorchAndNonTorchGrowth) {
    const auto growth = calculateMemoryGrowth(
        /*reserved_baseline_bytes=*/2 * GiB,
        /*reserved_peak_bytes=*/5 * GiB,
        /*reserved_current_bytes=*/3 * GiB,
        /*cuda_used_baseline_bytes=*/3 * GiB,
        /*cuda_used_current_bytes=*/6 * GiB);

    EXPECT_EQ(growth.torch_peak_increase_bytes, 3 * GiB);
    EXPECT_EQ(growth.non_torch_increase_bytes, 2 * GiB);
    EXPECT_EQ(growth.max_consumed_bytes, 5 * GiB);
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
    EXPECT_EQ(growth.max_consumed_bytes, 0u);
}

}  // namespace
}  // namespace rtp_llm
