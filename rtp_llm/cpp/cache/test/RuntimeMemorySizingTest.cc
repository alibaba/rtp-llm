#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

#include "rtp_llm/cpp/cache/RuntimeMemorySizing.h"

namespace rtp_llm {
namespace {

constexpr size_t GiB = 1024ULL * 1024 * 1024;

TEST(RuntimeMemorySizingTest, WarmupAddsCudaGraphAndSafetyToMeasuredRequirement) {
    RuntimeMemorySizingInput input;
    input.has_memory_profile            = true;
    input.configured_reserve_bytes      = 1 * GiB;
    input.transient_peak_headroom_bytes = 6 * GiB;
    input.cuda_graph_memory_bytes       = 2 * GiB;
    input.total_gpu_bytes               = 80 * GiB;
    input.safety_ratio                  = 0.05;

    EXPECT_EQ(calculateRuntimeMemorySizing(input), 12 * GiB);

    // The explicit reserve is a lower bound on headroom from the latest free-memory snapshot.
    input.configured_reserve_bytes = 20 * GiB;
    EXPECT_EQ(calculateRuntimeMemorySizing(input), 20 * GiB);
}

TEST(RuntimeMemorySizingTest, NoWarmupKeepsFixedFloorSemantics) {
    RuntimeMemorySizingInput input;
    input.configured_reserve_bytes = 1 * GiB;
    input.total_gpu_bytes          = 20 * GiB;
    input.safety_ratio             = 0.05;
    input.no_warmup_floor_bytes    = 2 * GiB;

    EXPECT_EQ(calculateRuntimeMemorySizing(input), 2 * GiB);
}

TEST(RuntimeMemorySizingTest, RejectsInvalidSafetyRatios) {
    RuntimeMemorySizingInput input;
    for (double ratio : {-0.01, 1.0, std::numeric_limits<double>::quiet_NaN()}) {
        input.safety_ratio = ratio;
        EXPECT_THROW(calculateRuntimeMemorySizing(input), std::invalid_argument);
    }
}

TEST(RuntimeMemorySizingTest, RejectsWarmupOverflow) {
    RuntimeMemorySizingInput input;
    input.has_memory_profile            = true;
    input.transient_peak_headroom_bytes = std::numeric_limits<size_t>::max();
    input.configured_reserve_bytes      = 1;
    input.total_gpu_bytes               = 2;
    input.safety_ratio                  = 0.5;

    EXPECT_THROW(calculateRuntimeMemorySizing(input), std::overflow_error);
}

TEST(RuntimeMemorySizingTest, RejectsCudaGraphOverflow) {
    RuntimeMemorySizingInput input;
    input.has_memory_profile            = true;
    input.transient_peak_headroom_bytes = std::numeric_limits<size_t>::max();
    input.cuda_graph_memory_bytes       = 1;

    EXPECT_THROW(calculateRuntimeMemorySizing(input), std::overflow_error);
}

}  // namespace
}  // namespace rtp_llm
