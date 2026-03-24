#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <algorithm>
#include <set>
#include "rtp_llm/cpp/cache/BlockCache.h"

namespace rtp_llm {
namespace test {

typedef BlockCache::CacheItem CacheItem;

class BlockCacheTest: public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize before each test case
        cache_ = std::make_unique<BlockCache>();
    }

    void TearDown() override {
        cache_.reset();
    }

    std::unique_ptr<BlockCache> cache_;
};

// ==================== Basic functionality tests ====================

TEST_F(BlockCacheTest, ConstructorTest) {
    // Test constructor
    BlockCache cache1;
    EXPECT_TRUE(cache1.empty());
    EXPECT_EQ(cache1.size(), 0);
}

TEST_F(BlockCacheTest, MatchBasicTest) {
    // 测试put和match的基本功能
    // 空匹配
    auto result0 = cache_->match(1);
    EXPECT_TRUE(isNullBlockIdx(result0.matched_index));

    CacheItem item    = {101, 0, 1, false};
    auto      result1 = cache_->put(item);
    EXPECT_TRUE(result1);

    // Put a duplicate key
    auto result2 = cache_->put(item);
    EXPECT_FALSE(result2);

    auto result3 = cache_->match(101);
    EXPECT_EQ(result3.matched_index, 1);

    auto result4 = cache_->match(102);
    EXPECT_TRUE(isNullBlockIdx(result4.matched_index));
}

TEST_F(BlockCacheTest, PopBasicTest) {
    // Test basic pop functionality
    std::vector<int64_t> cache_keys = {101, 102, 103, 104, 105};
    std::vector<int>     block_ids  = {1, 2, 3, 4, 5};

    CacheItem item1   = {101, 0, 1, false};
    auto      result1 = cache_->put(item1);
    EXPECT_TRUE(result1);
    CacheItem item2   = {102, 0, 2, false};
    auto      result2 = cache_->put(item2);
    EXPECT_TRUE(result2);
    CacheItem item3   = {103, 0, 3, false};
    auto      result3 = cache_->put(item3);
    EXPECT_TRUE(result3);
    CacheItem item4   = {104, 0, 4, false};
    auto      result4 = cache_->put(item4);
    EXPECT_TRUE(result4);
    CacheItem item5   = {105, 0, 5, false};
    auto      result5 = cache_->put(item5);
    EXPECT_TRUE(result5);

    EXPECT_EQ(cache_->size(), 5);

    // The oldest blocks are popped
    auto popped1 = cache_->pop(2);
    EXPECT_EQ(popped1.size(), 2);
    EXPECT_EQ(cache_->size(), 3);
    EXPECT_EQ(popped1[0], 1);
    EXPECT_EQ(popped1[1], 2);

    // The remaining three blocks are popped
    auto popped2 = cache_->pop(3);
    EXPECT_EQ(popped2.size(), 3);
    EXPECT_EQ(cache_->size(), 0);
    EXPECT_EQ(popped2[0], 3);
    EXPECT_EQ(popped2[1], 4);
    EXPECT_EQ(popped2[2], 5);

    // An empty cache cannot pop any items
    auto popped3 = cache_->pop(3);
    EXPECT_EQ(popped3.size(), 0);
    EXPECT_EQ(cache_->size(), 0);

    // 设置resident
    CacheItem item6   = {101, 0, 1, true};
    auto      result6 = cache_->put(item6);
    EXPECT_TRUE(result6);
    EXPECT_EQ(cache_->size(), 1);

    // Resident entries won't be popped
    auto popped4 = cache_->pop(2);
    EXPECT_EQ(popped4.size(), 0);
    EXPECT_EQ(cache_->size(), 1);
}

// ==================== selectAndEvict tests ====================

TEST_F(BlockCacheTest, SelectAndEvictEmptyCache) {
    auto result = cache_->selectAndEvict(5);
    EXPECT_TRUE(result.evicted_keys.empty());
    EXPECT_TRUE(result.evicted_items.empty());
}

TEST_F(BlockCacheTest, SelectAndEvictBasic) {
    // Insert 3 items with different cache_keys, single group
    CacheItem item1 = {101, 0, 1, false};
    CacheItem item2 = {102, 0, 2, false};
    CacheItem item3 = {103, 0, 3, false};
    cache_->put(item1);
    cache_->put(item2);
    cache_->put(item3);
    EXPECT_EQ(cache_->size(), 3);

    // Evict at least 2 blocks — should pick LRU first (101, then 102)
    auto result = cache_->selectAndEvict(2);
    EXPECT_EQ(result.evicted_keys.size(), 2);
    EXPECT_EQ(result.evicted_keys[0], 101);
    EXPECT_EQ(result.evicted_keys[1], 102);

    // Items should be removed from cache
    EXPECT_EQ(cache_->size(), 1);
    EXPECT_TRUE(isNullBlockIdx(cache_->match(101).matched_index));
    EXPECT_TRUE(isNullBlockIdx(cache_->match(102).matched_index));
    EXPECT_EQ(cache_->match(103).matched_index, 3);
}

TEST_F(BlockCacheTest, SelectAndEvictMultipleGroups) {
    // Same cache_key with multiple group_ids (simulating multi-group KV cache)
    CacheItem item1 = {101, 0, 10, false};
    CacheItem item2 = {101, 1, 11, false};
    CacheItem item3 = {102, 0, 20, false};
    CacheItem item4 = {102, 1, 21, false};
    cache_->put(item1);
    cache_->put(item2);
    cache_->put(item3);
    cache_->put(item4);
    EXPECT_EQ(cache_->size(), 4);

    // Evict at least 1 block — should pick LRU cache_key (101) which has 2 items
    auto result = cache_->selectAndEvict(1);
    EXPECT_EQ(result.evicted_keys.size(), 1);
    EXPECT_EQ(result.evicted_keys[0], 101);
    EXPECT_EQ(result.evicted_items[101].size(), 2);

    // Both group items for key 101 should be evicted
    std::set<BlockIdxType> evicted_blocks;
    for (const auto& item : result.evicted_items[101]) {
        evicted_blocks.insert(item.block_index);
    }
    EXPECT_TRUE(evicted_blocks.count(10));
    EXPECT_TRUE(evicted_blocks.count(11));

    // Cache should only have key 102 left
    EXPECT_EQ(cache_->size(), 2);
    EXPECT_TRUE(isNullBlockIdx(cache_->match(101, 0).matched_index));
    EXPECT_TRUE(isNullBlockIdx(cache_->match(101, 1).matched_index));
    EXPECT_EQ(cache_->match(102, 0).matched_index, 20);
    EXPECT_EQ(cache_->match(102, 1).matched_index, 21);
}

TEST_F(BlockCacheTest, SelectAndEvictSkipsResident) {
    // All items are resident — nothing should be evicted
    CacheItem item1 = {101, 0, 1, true};
    CacheItem item2 = {102, 0, 2, true};
    cache_->put(item1);
    cache_->put(item2);
    EXPECT_EQ(cache_->size(), 2);

    auto result = cache_->selectAndEvict(5);
    EXPECT_TRUE(result.evicted_keys.empty());
    EXPECT_EQ(cache_->size(), 2);
}

TEST_F(BlockCacheTest, SelectAndEvictSkipsKeyWithResidentItem) {
    // cache_key 101 has a resident item in group 1 — entire key should be skipped
    CacheItem item1 = {101, 0, 10, false};
    CacheItem item2 = {101, 1, 11, true};
    CacheItem item3 = {102, 0, 20, false};
    cache_->put(item1);
    cache_->put(item2);
    cache_->put(item3);
    EXPECT_EQ(cache_->size(), 3);

    // Should skip key 101 (has resident item) and evict key 102
    auto result = cache_->selectAndEvict(1);
    EXPECT_EQ(result.evicted_keys.size(), 1);
    EXPECT_EQ(result.evicted_keys[0], 102);
    EXPECT_EQ(result.evicted_items[102].size(), 1);
    EXPECT_EQ(result.evicted_items[102][0].block_index, 20);

    // Key 101 items should still be in cache
    EXPECT_EQ(cache_->size(), 2);
    EXPECT_EQ(cache_->match(101, 0).matched_index, 10);
}

TEST_F(BlockCacheTest, SelectAndEvictRequestMoreThanAvailable) {
    CacheItem item1 = {101, 0, 1, false};
    CacheItem item2 = {102, 0, 2, false};
    cache_->put(item1);
    cache_->put(item2);

    // Request more blocks than available — should evict everything possible
    auto result = cache_->selectAndEvict(100);
    EXPECT_EQ(result.evicted_keys.size(), 2);
    EXPECT_EQ(cache_->size(), 0);
}

TEST_F(BlockCacheTest, SelectAndEvictZeroBlocks) {
    CacheItem item1 = {101, 0, 1, false};
    cache_->put(item1);

    // min_blocks=0: the loop selects the first key before checking >= 0, so 1 key is evicted.
    // In practice, callers guard against 0 before calling selectAndEvict.
    auto result = cache_->selectAndEvict(0);
    EXPECT_EQ(result.evicted_keys.size(), 1);
    EXPECT_EQ(cache_->size(), 0);
}

TEST_F(BlockCacheTest, SelectAndEvictLRUOrder) {
    // Insert items, then access some to change LRU order
    CacheItem item1 = {101, 0, 1, false};
    CacheItem item2 = {102, 0, 2, false};
    CacheItem item3 = {103, 0, 3, false};
    cache_->put(item1);  // oldest
    cache_->put(item2);
    cache_->put(item3);  // newest

    // Access key 101 to make it most recently used
    cache_->match(101);

    // Now LRU order should be: 102 (least), 103, 101 (most recent)
    auto result = cache_->selectAndEvict(1);
    EXPECT_EQ(result.evicted_keys.size(), 1);
    EXPECT_EQ(result.evicted_keys[0], 102);
    EXPECT_EQ(cache_->size(), 2);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
