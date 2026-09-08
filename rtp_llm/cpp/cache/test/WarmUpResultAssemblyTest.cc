#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

#include "rtp_llm/cpp/cache/WarmUpResultAssembly.h"

namespace rtp_llm {
namespace {

constexpr size_t GiB = 1024ULL * 1024 * 1024;

MemoryStatus makeStatus(size_t peak, size_t current, size_t available) {
    MemoryStatus status;
    status.torch_allocated_peak_bytes = peak;
    status.allocated_bytes            = current;
    status.available_bytes            = available;
    return status;
}

TEST(WarmUpResultAssemblyTest, CombinesPersistentGrowthAndTransientTorchHeadroom) {
    const auto result = assembleWarmUpResult(64 * GiB, makeStatus(12 * GiB, 3 * GiB, 62 * GiB), true);

    EXPECT_EQ(result.measured_total_growth_bytes, 11 * GiB);
    EXPECT_TRUE(result.forward_measurement_trusted);
    EXPECT_FALSE(result.cuda_graph_measurement_trusted);
}

TEST(WarmUpResultAssemblyTest, ClampsNegativeGrowthComponentsToZero) {
    const auto result = assembleWarmUpResult(40 * GiB, makeStatus(2 * GiB, 3 * GiB, 41 * GiB), true);
    EXPECT_EQ(result.measured_total_growth_bytes, 0u);
}

TEST(WarmUpResultAssemblyTest, RejectsGrowthSumOverflow) {
    const auto status = makeStatus(1, 0, 0);
    EXPECT_THROW(assembleWarmUpResult(std::numeric_limits<size_t>::max(), status, true), std::overflow_error);
}

}  // namespace
}  // namespace rtp_llm
