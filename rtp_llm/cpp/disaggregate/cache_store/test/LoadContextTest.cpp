#include "gtest/gtest.h"

#include "rtp_llm/cpp/disaggregate/cache_store/LoadContext.h"

#include <limits>

namespace rtp_llm {
namespace {

class TestSyncContext: public SyncContext {
public:
    using SyncContext::makeDeadlineMs;
    using SyncContext::normalizeTimeoutMs;

private:
    bool doCall(const std::shared_ptr<RequestBlockBuffer>&, CacheStoreLoadDeadline) override {
        return true;
    }
};

TEST(LoadContextTest, deadlineCalculationSaturates) {
    constexpr auto kMax = std::numeric_limits<int64_t>::max();

    EXPECT_EQ(kMax, TestSyncContext::makeDeadlineMs(kMax - 5, 10));
    EXPECT_EQ(kMax, TestSyncContext::makeDeadlineMs(0, kMax));
    EXPECT_EQ(123, TestSyncContext::makeDeadlineMs(123, 0));
    EXPECT_EQ(123, TestSyncContext::makeDeadlineMs(123, std::numeric_limits<int64_t>::min()));
}

TEST(LoadContextTest, wireTimeoutDoesNotNarrowOrWrap) {
    constexpr auto kMax = std::numeric_limits<uint32_t>::max();

    EXPECT_EQ(kMax, TestSyncContext::normalizeTimeoutMs(std::numeric_limits<int64_t>::max()));
    EXPECT_EQ(kMax, TestSyncContext::normalizeTimeoutMs(static_cast<int64_t>(kMax) + 1));
    EXPECT_EQ(1, TestSyncContext::normalizeTimeoutMs(1));
    EXPECT_EQ(0, TestSyncContext::normalizeTimeoutMs(0));
    EXPECT_EQ(0, TestSyncContext::normalizeTimeoutMs(-1));
}

TEST(LoadContextTest, remainingTimeoutRoundsUpPositiveBudget) {
    const auto now = CacheStoreLoadClock::time_point(std::chrono::seconds(1));
    uint32_t   remaining_timeout_ms = 0;

    EXPECT_TRUE(getCacheStoreLoadRemainingTimeoutMs(
        now + std::chrono::microseconds(999), now, remaining_timeout_ms));
    EXPECT_EQ(1, remaining_timeout_ms);

    EXPECT_TRUE(getCacheStoreLoadRemainingTimeoutMs(
        now + std::chrono::microseconds(1999), now, remaining_timeout_ms));
    EXPECT_EQ(2, remaining_timeout_ms);

    EXPECT_FALSE(getCacheStoreLoadRemainingTimeoutMs(now, now, remaining_timeout_ms));
    EXPECT_EQ(0, remaining_timeout_ms);
}

TEST(LoadContextTest, terminalDeadlineUsesExactSteadyTimePoint) {
    const auto deadline = CacheStoreLoadClock::time_point(std::chrono::seconds(2));

    EXPECT_FALSE(isCacheStoreLoadDeadlineReached(deadline, deadline - std::chrono::microseconds(1)));
    EXPECT_TRUE(isCacheStoreLoadDeadlineReached(deadline, deadline));
    EXPECT_TRUE(isCacheStoreLoadDeadlineReached(deadline, deadline + std::chrono::microseconds(1)));
}

TEST(LoadContextTest, steadyDeadlineConstructionSaturates) {
    const auto now = CacheStoreLoadClock::time_point(std::chrono::seconds(1));

    EXPECT_EQ(now, makeCacheStoreLoadDeadline(0, now));
    EXPECT_EQ(now, makeCacheStoreLoadDeadline(-1, now));
    EXPECT_EQ(CacheStoreLoadDeadline::max(),
              makeCacheStoreLoadDeadline(std::numeric_limits<int64_t>::max(), now));
}

}  // namespace
}  // namespace rtp_llm
