#include "rtp_llm/cpp/model_rpc/RecentCacheKeyWindow.h"

#include <atomic>
#include <cstdlib>

#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

TEST(RecentCacheKeyWindowTest, HitIsComputedBeforeInsert) {
    std::atomic<int64_t> now{1000};
    RecentCacheKeyWindow window(1000, [&now]() { return now.load(); });

    auto first = window.record({1, 2, 3});
    EXPECT_EQ(first.request_occurrences, 3);
    EXPECT_EQ(first.request_hit_occurrences, 0);
    EXPECT_DOUBLE_EQ(first.request_hit_ratio, 0.0);
    EXPECT_EQ(first.retained_occurrences, 3);
    EXPECT_EQ(first.retained_unique_cache_keys, 3);

    auto second = window.record({2, 3, 4});
    EXPECT_EQ(second.request_occurrences, 3);
    EXPECT_EQ(second.request_hit_occurrences, 2);
    EXPECT_DOUBLE_EQ(second.request_hit_ratio, 2.0 / 3.0);
    EXPECT_EQ(second.retained_occurrences, 6);
    EXPECT_EQ(second.retained_unique_cache_keys, 4);
}

TEST(RecentCacheKeyWindowTest, DuplicateKeysCountByOccurrence) {
    std::atomic<int64_t> now{1000};
    RecentCacheKeyWindow window(1000, [&now]() { return now.load(); });

    auto first = window.record({7, 7, 7});
    EXPECT_EQ(first.request_occurrences, 3);
    EXPECT_EQ(first.request_hit_occurrences, 0);
    EXPECT_EQ(first.retained_occurrences, 3);
    EXPECT_EQ(first.retained_unique_cache_keys, 1);

    auto second = window.record({7, 7});
    EXPECT_EQ(second.request_occurrences, 2);
    EXPECT_EQ(second.request_hit_occurrences, 2);
    EXPECT_EQ(second.retained_occurrences, 5);
    EXPECT_EQ(second.retained_unique_cache_keys, 1);
}

TEST(RecentCacheKeyWindowTest, ExpiredEntriesDecrementCountsAndRemoveZeroKeys) {
    std::atomic<int64_t> now{1000};
    RecentCacheKeyWindow window(1000, [&now]() { return now.load(); });

    window.record({1, 2});
    now = 1500;
    window.record({2, 3});
    auto snapshot = window.snapshot();
    EXPECT_EQ(snapshot.retained_occurrences, 4);
    EXPECT_EQ(snapshot.retained_unique_cache_keys, 3);

    now = 2001;
    snapshot = window.snapshot();
    EXPECT_EQ(snapshot.retained_occurrences, 2);
    EXPECT_EQ(snapshot.retained_unique_cache_keys, 2);

    auto next = window.record({1, 2, 3});
    EXPECT_EQ(next.request_hit_occurrences, 2);
    EXPECT_EQ(next.retained_occurrences, 5);
    EXPECT_EQ(next.retained_unique_cache_keys, 3);
}

TEST(RecentCacheKeyWindowTest, EmptyRequestReportsZeroButKeepsWindow) {
    RecentCacheKeyWindow window(1000, []() { return 1000; });
    window.record({1, 2});
    auto empty = window.record({});
    EXPECT_EQ(empty.request_occurrences, 0);
    EXPECT_EQ(empty.request_hit_occurrences, 0);
    EXPECT_EQ(empty.retained_occurrences, 2);
    EXPECT_EQ(empty.retained_unique_cache_keys, 2);
}

}  // namespace
}  // namespace rtp_llm
