
#include "gtest/gtest.h"

#define private public
#include "rtp_llm/cpp/utils/LRUCache.h"

#include <chrono>
#include <memory>
#include <thread>

using namespace std;

namespace rtp_llm {

class LRUCacheTest: public ::testing::Test {
protected:
};

TEST_F(LRUCacheTest, testSimple) {
    LRUCache<int, std::string> cache(3);

    cache.put(1, "Item1");
    cache.put(2, "Item2");
    cache.put(3, "Item3");
    cache.printCache();
    ASSERT_TRUE(cache.contains(1));
    ASSERT_TRUE(cache.contains(2));
    ASSERT_TRUE(cache.contains(3));

    cache.put(4, "Item4");  // This will remove Item1 as it is the least recently used item
    ASSERT_TRUE(cache.contains(4));
    ASSERT_FALSE(cache.contains(1));
    ASSERT_EQ(std::get<1>(cache.pop()), "Item2");
    cache.printCache();
}

// --------------------------- clear ---------------------------

TEST_F(LRUCacheTest, testClear_OnNonEmptyCache_EmptiesAllItems) {
    LRUCache<int, std::string> cache(3);
    cache.put(1, "A");
    cache.put(2, "B");
    cache.put(3, "C");
    ASSERT_FALSE(cache.empty());
    ASSERT_EQ(cache.size(), 3u);

    auto snap_before    = cache.cacheSnapshot(-1);
    auto version_before = snap_before.version;

    cache.clear();
    ASSERT_TRUE(cache.empty());
    ASSERT_EQ(cache.size(), 0u);
    ASSERT_FALSE(cache.contains(1));
    ASSERT_FALSE(cache.contains(2));
    ASSERT_FALSE(cache.contains(3));

    auto snap_after = cache.cacheSnapshot(-1);
    ASSERT_GT(snap_after.version, version_before);
}

TEST_F(LRUCacheTest, testClear_OnEmptyCache_StaysEmptyAndSafe) {
    LRUCache<int, std::string> cache(2);
    ASSERT_TRUE(cache.empty());
    auto snap_before = cache.cacheSnapshot(-1);
    cache.clear();
    ASSERT_TRUE(cache.empty());
    ASSERT_EQ(cache.size(), 0u);
    auto snap_after = cache.cacheSnapshot(-1);
    // version could increase; just ensure no crash and remains empty
    ASSERT_TRUE(cache.empty());
}

}  // namespace rtp_llm
