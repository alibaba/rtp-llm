// Copyright (c) RTP-LLM

#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/connector/memory/MemoryBlockCache.h"
#include "rtp_llm/cpp/cache/types.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm::test {

TEST(MemoryBlockCacheTest, EmptyState_BasicMethods) {
    MemoryBlockCache cache;
    EXPECT_TRUE(cache.empty());
    EXPECT_EQ(cache.size(), 0u);
    EXPECT_FALSE(cache.contains(42));

    auto mr = cache.match(42);
    EXPECT_TRUE(isNullBlockIdx(mr.matched_index));
    EXPECT_EQ(mr.block_size, 0u);
}

TEST(MemoryBlockCacheTest, PutAndMatch_SucceedsAndUpdatesSize) {
    MemoryBlockCache cache;

    MemoryBlockCache::CacheItem item;
    item.cache_key   = 100;
    item.block_index = 7;
    item.block_size  = 1234;
    item.is_resident = false;

    auto [success, popped_item_opt] = cache.put(item);
    EXPECT_TRUE(success);
    EXPECT_FALSE(popped_item_opt.has_value());
    EXPECT_FALSE(cache.empty());
    EXPECT_EQ(cache.size(), 1u);
    EXPECT_TRUE(cache.contains(100));

    auto mr = cache.match(100);
    EXPECT_EQ(mr.matched_index, 7);
    EXPECT_EQ(mr.block_size, 1234u);
}

TEST(MemoryBlockCacheTest, PutDuplicate_DoesNotOverride_OnlyIncreasesPopularity) {
    MemoryBlockCache            cache;
    MemoryBlockCache::CacheItem item1;
    item1.cache_key                 = 200;
    item1.block_index               = 10;
    item1.block_size                = 4096;
    item1.is_resident               = false;
    auto [success, popped_item_opt] = cache.put(item1);
    EXPECT_TRUE(success);
    EXPECT_FALSE(popped_item_opt.has_value());
    EXPECT_EQ(cache.size(), 1u);

    // Try to put another item with same key but different data; should return false and not change stored value
    MemoryBlockCache::CacheItem item2;
    item2.cache_key                   = 200;
    item2.block_index                 = 11;
    item2.block_size                  = 8192;
    item2.is_resident                 = true;
    auto [success2, popped_item_opt2] = cache.put(item2);
    EXPECT_FALSE(success2);
    EXPECT_FALSE(popped_item_opt2.has_value());
    EXPECT_EQ(cache.size(), 1u);

    auto mr = cache.match(200);
    EXPECT_EQ(mr.matched_index, 10);
    EXPECT_EQ(mr.block_size, 4096u);
}

TEST(MemoryBlockCacheTest, Pop_NonResidentOnly_EvictsAndSkipsResident) {
    MemoryBlockCache cache;
    // Insert 4 items; mark one as resident
    for (int i = 0; i < 4; ++i) {
        MemoryBlockCache::CacheItem item;
        item.cache_key                  = 300 + i;
        item.block_index                = 20 + i;
        item.block_size                 = 1000 + i;
        item.is_resident                = (i == 1);  // key=301 is resident
        auto [success, popped_item_opt] = cache.put(item);
        EXPECT_TRUE(success);
        EXPECT_FALSE(popped_item_opt.has_value());
    }
    EXPECT_EQ(cache.size(), 4u);
    EXPECT_TRUE(cache.contains(301));

    // Pop more than evictable count; resident item should remain
    auto popped = cache.pop(10);
    // Should have evicted 3 non-resident items
    EXPECT_EQ(popped.size(), 3u);
    EXPECT_EQ(cache.size(), 1u);
    EXPECT_TRUE(cache.contains(301));  // resident remains

    // Verify popped indices do not include resident's index (which is 21)
    for (auto idx : popped) {
        EXPECT_NE(idx, 21);
    }
}

TEST(MemoryBlockCacheTest, Pop_LimitedCount) {
    MemoryBlockCache cache;
    std::vector<int> keys;
    keys.reserve(5);
    for (int i = 0; i < 5; ++i) {
        MemoryBlockCache::CacheItem item;
        item.cache_key                  = 400 + i;
        item.block_index                = 30 + i;
        item.block_size                 = 2000 + i;
        item.is_resident                = false;
        auto [success, popped_item_opt] = cache.put(item);
        EXPECT_TRUE(success);
        EXPECT_FALSE(popped_item_opt.has_value());
        keys.push_back(item.cache_key);
    }
    EXPECT_EQ(cache.size(), 5u);

    auto popped = cache.pop(2);
    EXPECT_EQ(popped.size(), 2u);
    EXPECT_EQ(cache.size(), 3u);
    // Popped indices must belong to the previously inserted set
    for (auto idx : popped) {
        bool found = false;
        for (int i = 0; i < 5; ++i) {
            if (idx == 30 + i) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found);
    }
}

TEST(MemoryBlockCacheTest, Match_IncreasesRecency_AffectsPopOrder) {
    MemoryBlockCache cache;
    // Insert 3 non-resident items
    for (int i = 0; i < 3; ++i) {
        MemoryBlockCache::CacheItem item;
        item.cache_key                  = 500 + i;
        item.block_index                = 40 + i;
        item.block_size                 = 3000 + i;
        item.is_resident                = false;
        auto [success, popped_item_opt] = cache.put(item);
        EXPECT_TRUE(success);
        EXPECT_FALSE(popped_item_opt.has_value());
    }
    // Access key=500 to increase its recency
    auto mr = cache.match(500);
    ASSERT_FALSE(isNullBlockIdx(mr.matched_index));

    // Pop one; it should not be the recently matched key's index (40)
    auto popped = cache.pop(1);
    ASSERT_EQ(popped.size(), 1u);
    EXPECT_NE(popped[0], 40);
    EXPECT_EQ(cache.size(), 2u);
}

TEST(MemoryBlockCacheTest, PopNone_WhenAllResident) {
    MemoryBlockCache cache;
    for (int i = 0; i < 2; ++i) {
        MemoryBlockCache::CacheItem item;
        item.cache_key                  = 600 + i;
        item.block_index                = 50 + i;
        item.block_size                 = 4000 + i;
        item.is_resident                = true;
        auto [success, popped_item_opt] = cache.put(item);
        EXPECT_TRUE(success);
        EXPECT_FALSE(popped_item_opt.has_value());
    }
    EXPECT_EQ(cache.size(), 2u);
    auto popped = cache.pop(3);
    EXPECT_TRUE(popped.empty());
    EXPECT_EQ(cache.size(), 2u);
}

}  // namespace rtp_llm::test

int main(int argc, char** argv) {
    rtp_llm::initLogger();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
