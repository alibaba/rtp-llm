#include "rtp_llm/cpp/utils/FallbackLogUtils.h"

#include <atomic>

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

TEST(FallbackLogUtilsTest, LogsFirstAndPowerOfTwoOccurrences) {
    std::atomic<uint64_t> counter{0};
    const bool            expected_should_log[] = {true, true, false, true, false, false, false, true};

    for (uint64_t expected_count = 1; expected_count <= 8; ++expected_count) {
        const auto [count, should_log] = recordRateLimitedFallback(counter);
        EXPECT_EQ(count, expected_count);
        EXPECT_EQ(should_log, expected_should_log[expected_count - 1]);
    }
}

}  // namespace
}  // namespace rtp_llm
