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
        cache_ = std::make_unique<BlockCache>();
    }

    void TearDown() override {
        cache_.reset();
    }

    std::unique_ptr<BlockCache> cache_;
};

// ==================== Basic functionality tests ====================

TEST_F(BlockCacheTest, ConstructorTest) {
    BlockCache cache1;
    EXPECT_TRUE(cache1.empty());
    EXPECT_EQ(cache1.size(), 0);
}

TEST_F(BlockCacheTest, MatchBasicTest) {
    auto result0 = cache_->matchSlot(1);
    EXPECT_TRUE(isNullBlockIdx(result0.matched_index));

    auto result1 = cache_->putSlot(101, 0, 0, 1, false);
    EXPECT_TRUE(result1);

    // Put a duplicate slot
    auto result2 = cache_->putSlot(101, 0, 0, 1, false);
    EXPECT_FALSE(result2);

    auto result3 = cache_->matchSlot(101);
    EXPECT_EQ(result3.matched_index, 1);

    auto result4 = cache_->matchSlot(102);
    EXPECT_TRUE(isNullBlockIdx(result4.matched_index));
}

TEST_F(BlockCacheTest, PopBasicTest) {
    EXPECT_TRUE(cache_->putSlot(101, 0, 0, 1, false));
    EXPECT_TRUE(cache_->putSlot(102, 0, 0, 2, false));
    EXPECT_TRUE(cache_->putSlot(103, 0, 0, 3, false));
    EXPECT_TRUE(cache_->putSlot(104, 0, 0, 4, false));
    EXPECT_TRUE(cache_->putSlot(105, 0, 0, 5, false));

    EXPECT_EQ(cache_->size(), 5);

    // Pop oldest 2 items (each CacheItem has 1 slot → 1 block each)
    auto popped1 = cache_->pop(2);
    EXPECT_EQ(popped1.size(), 2);
    EXPECT_EQ(cache_->size(), 3);
    EXPECT_EQ(popped1[0], 1);
    EXPECT_EQ(popped1[1], 2);

    auto popped2 = cache_->pop(3);
    EXPECT_EQ(popped2.size(), 3);
    EXPECT_EQ(cache_->size(), 0);
    EXPECT_EQ(popped2[0], 3);
    EXPECT_EQ(popped2[1], 4);
    EXPECT_EQ(popped2[2], 5);

    auto popped3 = cache_->pop(3);
    EXPECT_EQ(popped3.size(), 0);
    EXPECT_EQ(cache_->size(), 0);

    // Resident entries won't be popped
    EXPECT_TRUE(cache_->putSlot(101, 0, 0, 1, true));
    EXPECT_EQ(cache_->size(), 1);

    auto popped4 = cache_->pop(2);
    EXPECT_EQ(popped4.size(), 0);
    EXPECT_EQ(cache_->size(), 1);
}

TEST_F(BlockCacheTest, ContainsSlotTest) {
    EXPECT_FALSE(cache_->containsSlot(101, 0, 0));

    cache_->putSlot(101, 0, 0, 1, false);
    EXPECT_TRUE(cache_->containsSlot(101, 0, 0));
    EXPECT_FALSE(cache_->containsSlot(101, 0, 1));
    EXPECT_FALSE(cache_->containsSlot(101, 1, 0));
    EXPECT_FALSE(cache_->containsSlot(102, 0, 0));
}

TEST_F(BlockCacheTest, RemoveItemTest) {
    cache_->putSlot(101, 0, 0, 1, false);
    cache_->putSlot(102, 0, 0, 2, false);
    EXPECT_EQ(cache_->size(), 2);

    auto removed = cache_->removeItem(101);
    ASSERT_TRUE(removed.has_value());
    EXPECT_EQ(removed->cache_key, 101);
    EXPECT_EQ(cache_->size(), 1);
    EXPECT_TRUE(isNullBlockIdx(cache_->matchSlot(101).matched_index));

    auto not_found = cache_->removeItem(999);
    EXPECT_FALSE(not_found.has_value());
}

TEST_F(BlockCacheTest, MultiGroupSlots) {
    // Same cache_key with multiple group_ids
    cache_->putSlot(101, 0, 0, 10, false);
    cache_->putSlot(101, 0, 1, 11, false);

    EXPECT_EQ(cache_->size(), 1);  // one CacheItem for key 101
    EXPECT_EQ(cache_->matchSlot(101, 0, 0).matched_index, 10);
    EXPECT_EQ(cache_->matchSlot(101, 0, 1).matched_index, 11);
    EXPECT_TRUE(cache_->containsSlot(101, 0, 0));
    EXPECT_TRUE(cache_->containsSlot(101, 0, 1));

    // Pop should return both blocks
    auto popped = cache_->pop(1);
    EXPECT_EQ(popped.size(), 2);
    std::set<BlockIdxType> popped_set(popped.begin(), popped.end());
    EXPECT_TRUE(popped_set.count(10));
    EXPECT_TRUE(popped_set.count(11));
    EXPECT_EQ(cache_->size(), 0);
}

// ==================== selectAndEvict tests ====================

TEST_F(BlockCacheTest, SelectAndEvictEmptyCache) {
    auto result = cache_->selectAndEvict(5);
    EXPECT_TRUE(result.evicted_keys.empty());
    EXPECT_TRUE(result.evicted_items.empty());
}

TEST_F(BlockCacheTest, SelectAndEvictBasic) {
    cache_->putSlot(101, 0, 0, 1, false);
    cache_->putSlot(102, 0, 0, 2, false);
    cache_->putSlot(103, 0, 0, 3, false);
    EXPECT_EQ(cache_->size(), 3);

    // Evict at least 2 blocks — should pick LRU first (101, then 102)
    auto result = cache_->selectAndEvict(2);
    EXPECT_EQ(result.evicted_keys.size(), 2);
    EXPECT_EQ(result.evicted_keys[0], 101);
    EXPECT_EQ(result.evicted_keys[1], 102);

    EXPECT_EQ(cache_->size(), 1);
    EXPECT_TRUE(isNullBlockIdx(cache_->matchSlot(101).matched_index));
    EXPECT_TRUE(isNullBlockIdx(cache_->matchSlot(102).matched_index));
    EXPECT_EQ(cache_->matchSlot(103).matched_index, 3);
}

TEST_F(BlockCacheTest, SelectAndEvictMultipleGroups) {
    // Same cache_key with multiple group_ids → one CacheItem with 2 slots
    cache_->putSlot(101, 0, 0, 10, false);
    cache_->putSlot(101, 0, 1, 11, false);
    cache_->putSlot(102, 0, 0, 20, false);
    cache_->putSlot(102, 0, 1, 21, false);
    EXPECT_EQ(cache_->size(), 2);  // 2 CacheItems

    // Evict at least 1 block — picks LRU key (101) which has 2 valid slots
    auto result = cache_->selectAndEvict(1);
    EXPECT_EQ(result.evicted_keys.size(), 1);
    EXPECT_EQ(result.evicted_keys[0], 101);

    const auto& evicted_item = result.evicted_items.at(101);
    EXPECT_EQ(evicted_item.totalValidSlots(), 2);
    EXPECT_EQ(evicted_item.slots[0][0].block_id, 10);
    EXPECT_EQ(evicted_item.slots[0][1].block_id, 11);

    EXPECT_EQ(cache_->size(), 1);
    EXPECT_TRUE(isNullBlockIdx(cache_->matchSlot(101, 0, 0).matched_index));
    EXPECT_TRUE(isNullBlockIdx(cache_->matchSlot(101, 0, 1).matched_index));
    EXPECT_EQ(cache_->matchSlot(102, 0, 0).matched_index, 20);
    EXPECT_EQ(cache_->matchSlot(102, 0, 1).matched_index, 21);
}

TEST_F(BlockCacheTest, SelectAndEvictSkipsResident) {
    cache_->putSlot(101, 0, 0, 1, true);
    cache_->putSlot(102, 0, 0, 2, true);
    EXPECT_EQ(cache_->size(), 2);

    auto result = cache_->selectAndEvict(5);
    EXPECT_TRUE(result.evicted_keys.empty());
    EXPECT_EQ(cache_->size(), 2);
}

TEST_F(BlockCacheTest, SelectAndEvictRequestMoreThanAvailable) {
    cache_->putSlot(101, 0, 0, 1, false);
    cache_->putSlot(102, 0, 0, 2, false);

    auto result = cache_->selectAndEvict(100);
    EXPECT_EQ(result.evicted_keys.size(), 2);
    EXPECT_EQ(cache_->size(), 0);
}

TEST_F(BlockCacheTest, SelectAndEvictZeroBlocks) {
    cache_->putSlot(101, 0, 0, 1, false);

    auto result = cache_->selectAndEvict(0);
    EXPECT_EQ(result.evicted_keys.size(), 1);
    EXPECT_EQ(cache_->size(), 0);
}

TEST_F(BlockCacheTest, SelectAndEvictLRUOrder) {
    cache_->putSlot(101, 0, 0, 1, false);  // oldest
    cache_->putSlot(102, 0, 0, 2, false);
    cache_->putSlot(103, 0, 0, 3, false);  // newest

    // Access key 101 to make it most recently used
    cache_->matchSlot(101);

    // Now LRU order should be: 102 (least), 103, 101 (most recent)
    auto result = cache_->selectAndEvict(1);
    EXPECT_EQ(result.evicted_keys.size(), 1);
    EXPECT_EQ(result.evicted_keys[0], 102);
    EXPECT_EQ(cache_->size(), 2);
}

TEST_F(BlockCacheTest, RegisterModelTest) {
    EXPECT_EQ(cache_->registeredModelNum(), 0);
    cache_->registerModel(0, 2, nullptr);
    EXPECT_EQ(cache_->registeredModelNum(), 1);
    cache_->registerModel(1, 1, nullptr);
    EXPECT_EQ(cache_->registeredModelNum(), 2);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
