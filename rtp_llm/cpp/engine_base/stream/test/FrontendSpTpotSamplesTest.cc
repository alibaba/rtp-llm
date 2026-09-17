#include "rtp_llm/cpp/engine_base/stream/FrontendSpTpotSamples.h"
#include "gtest/gtest.h"
#include <chrono>

namespace rtp_llm {

TEST(FrontendSpTpotSamplesTest, OldFrameWaitsForItsSampleOnly) {
    FrontendSpTpotSamples samples;
    auto                  first    = samples.begin();
    const auto            boundary = samples.sequence();
    auto                  later    = samples.begin();
    first->complete(12.0);
    auto       reader = std::async(std::launch::async, [&] { return samples.take(boundary); });
    const auto status = reader.wait_for(std::chrono::milliseconds(100));
    later->complete(34.0);
    const auto result = reader.get();
    EXPECT_EQ(status, std::future_status::ready);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], std::make_pair(uint64_t{1}, 12.0));
    const auto remaining = samples.take(samples.sequence());
    ASSERT_EQ(remaining.size(), 1u);
    EXPECT_EQ(remaining[0], std::make_pair(uint64_t{2}, 34.0));
}

TEST(FrontendSpTpotSamplesTest, OwnPendingSampleAndCancellationReleaseReader) {
    FrontendSpTpotSamples samples;
    auto                  pending  = samples.begin();
    const auto            boundary = samples.sequence();
    auto                  reader   = std::async(std::launch::async, [&] { return samples.take(boundary); });
    EXPECT_EQ(reader.wait_for(std::chrono::milliseconds(10)), std::future_status::timeout);
    pending.reset();
    EXPECT_TRUE(reader.get().empty());
    EXPECT_TRUE(samples.take(boundary).empty());
}

TEST(FrontendSpTpotSamplesTest, CoalescedFramesDrainEverySampleOnce) {
    FrontendSpTpotSamples samples;
    auto                  cancelled = samples.begin();
    cancelled.reset();
    auto       second   = samples.begin();
    auto       third    = samples.begin();
    const auto boundary = samples.sequence();
    second->complete(7.5);
    third->complete(2.0);
    EXPECT_TRUE(samples.take(0).empty());
    const auto result = samples.take(boundary);
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0], std::make_pair(uint64_t{2}, 7.5));
    EXPECT_EQ(result[1], std::make_pair(uint64_t{3}, 2.0));
    EXPECT_TRUE(samples.take(boundary).empty());
}
}  // namespace rtp_llm
