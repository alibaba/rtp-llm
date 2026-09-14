#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/WarmUpResultAssembly.h"

namespace rtp_llm {
namespace {

constexpr size_t GiB = 1024ULL * 1024 * 1024;

MemoryStatus makeStatus(size_t peak, size_t current) {
    MemoryStatus status;
    status.torch_allocated_peak_bytes = peak;
    status.allocated_bytes            = current;
    return status;
}

TEST(WarmUpResultAssemblyTest, StoresInitSnapshotAndTransientPeakHeadroom) {
    const auto result = assembleWarmUpResult(64 * GiB, makeStatus(12 * GiB, 3 * GiB), true);

    EXPECT_EQ(result.init_free_memory_bytes, 64 * GiB);
    EXPECT_EQ(result.transient_peak_headroom_bytes, 9 * GiB);
    EXPECT_TRUE(result.forward_measurement_trusted);
    EXPECT_FALSE(result.cuda_graph_measurement_trusted);
}

TEST(WarmUpResultAssemblyTest, ClampsNegativeTransientPeakHeadroomToZero) {
    const auto result = assembleWarmUpResult(40 * GiB, makeStatus(2 * GiB, 3 * GiB), true);
    EXPECT_EQ(result.transient_peak_headroom_bytes, 0u);
}

}  // namespace
}  // namespace rtp_llm
