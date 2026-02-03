// Copyright (c) RTP-LLM

#include "gtest/gtest.h"

#include <unordered_map>

#include "rtp_llm/cpp/cache/connector/memory/MemoryBlockCache.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/config/StaticConfig.h"
#include "rtp_llm/cpp/utils/Exception.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm::test {

TEST(MemoryBlockCacheTest, match_ReturnNull_WhenKeyNotFoundAndCacheNonEmpty) {
    MemoryBlockCache cache;

    MemoryBlockCache::CacheItem item;
    item.cache_key   = 700;
    item.block_index = 70;
    item.block_size  = 7000;
    item.is_resident = false;
    item.is_big      = true;
    ASSERT_TRUE(cache.put(item).first);

    auto mr = cache.match(999);  // not exist
    EXPECT_TRUE(isNullBlockIdx(mr.matched_index));
    EXPECT_EQ(mr.block_size, 0u);
    EXPECT_FALSE(mr.is_big);
}

TEST(MemoryBlockCacheTest, match_ReturnHit_WhenKeyExistsAndUpdatesRecency) {
    MemoryBlockCache cache;
    // Insert 3 non-resident items
    for (int i = 0; i < 3; ++i) {
        MemoryBlockCache::CacheItem item;
        item.cache_key                  = 500 + i;
        item.block_index                = 40 + i;
        item.block_size                 = 3000 + i;
        item.is_resident                = false;
        item.is_big                     = true;
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

TEST(MemoryBlockCacheTest, contains_ReturnTrue_WhenKeyExists) {
    MemoryBlockCache cache;

    MemoryBlockCache::CacheItem item;
    item.cache_key   = 900;
    item.block_index = 90;
    item.block_size  = 9000;
    item.is_resident = false;
    item.is_big      = true;
    ASSERT_TRUE(cache.put(item).first);

    EXPECT_TRUE(cache.contains(900));
}

TEST(MemoryBlockCacheTest, contains_ReturnFalse_WhenKeyNotFoundAndCacheNonEmpty) {
    MemoryBlockCache cache;

    MemoryBlockCache::CacheItem item;
    item.cache_key   = 901;
    item.block_index = 91;
    item.block_size  = 9001;
    item.is_resident = false;
    item.is_big      = true;
    ASSERT_TRUE(cache.put(item).first);

    EXPECT_FALSE(cache.contains(999));
}

TEST(MemoryBlockCacheTest, contains_ReturnFalse_WhenKeyNotFoundAndCacheEmpty) {
    MemoryBlockCache cache;
    EXPECT_FALSE(cache.contains(42));
}

TEST(MemoryBlockCacheTest, contains_ReturnFalse_WhenContainsDoesNotUpdateRecency) {
    MemoryBlockCache cache;
    // Insert 3 non-resident items (LRU: 1000 is oldest).
    for (int i = 0; i < 3; ++i) {
        MemoryBlockCache::CacheItem item;
        item.cache_key   = 1000 + i;
        item.block_index = 10 + i;
        item.block_size  = 1;
        item.is_resident = false;
        item.is_big      = true;
        ASSERT_TRUE(cache.put(item).first);
    }

    // `contains()` should NOT increase recency. Oldest should still be evicted first.
    ASSERT_TRUE(cache.contains(1000));
    auto popped = cache.pop(1);
    ASSERT_EQ(popped.size(), 1u);
    EXPECT_EQ(popped[0], 10);
}

TEST(MemoryBlockCacheTest, put_ReturnTrue_WhenInsertNewItem) {
    MemoryBlockCache cache;

    MemoryBlockCache::CacheItem item;
    item.cache_key   = 100;
    item.block_index = 7;
    item.block_size  = 1234;
    item.is_resident = false;
    item.is_big      = true;

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

TEST(MemoryBlockCacheTest, put_ReturnThrow_WhenBlockIndexIsNull) {
    // Ensure CHECK throws instead of aborting.
    rtp_llm::StaticConfig::user_ft_core_dump_on_exception = false;

    MemoryBlockCache            cache;
    MemoryBlockCache::CacheItem item;
    item.cache_key   = 123;
    item.block_index = -1;
    item.block_size  = 1;
    item.is_resident = false;

    EXPECT_THROW((void)cache.put(item), rtp_llm::RTPException);
}

TEST(MemoryBlockCacheTest, put_ReturnFalse_WhenCacheKeyDuplicate) {
    MemoryBlockCache            cache;
    MemoryBlockCache::CacheItem item1;
    item1.cache_key                 = 200;
    item1.block_index               = 10;
    item1.block_size                = 4096;
    item1.is_resident               = false;
    item1.is_big                    = true;
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
    item2.is_big                      = true;
    auto [success2, popped_item_opt2] = cache.put(item2);
    EXPECT_FALSE(success2);
    EXPECT_FALSE(popped_item_opt2.has_value());
    EXPECT_EQ(cache.size(), 1u);

    auto mr = cache.match(200);
    EXPECT_EQ(mr.matched_index, 10);
    EXPECT_EQ(mr.block_size, 4096u);
}

TEST(MemoryBlockCacheTest, match_ReturnHit_WhenKeyExistsButIsSmall) {
    MemoryBlockCache cache;

    MemoryBlockCache::CacheItem item;
    item.cache_key   = 777;
    item.block_index = 77;
    item.block_size  = 7777;
    item.is_resident = false;
    item.is_big      = false;  // small (partial KV)
    ASSERT_TRUE(cache.put(item).first);

    auto mr = cache.match(777);
    EXPECT_FALSE(isNullBlockIdx(mr.matched_index));
    EXPECT_EQ(mr.matched_index, 77);
    EXPECT_EQ(mr.block_size, 7777u);
    EXPECT_FALSE(mr.is_big);
}

TEST(MemoryBlockCacheTest, put_UpgradeSmallToBig_ReplacesAndReturnsOldItem) {
    MemoryBlockCache cache;

    MemoryBlockCache::CacheItem small;
    small.cache_key   = 888;
    small.block_index = 80;
    small.block_size  = 1000;
    small.is_resident = false;
    small.is_big      = false;
    {
        auto [ok, popped] = cache.put(small);
        EXPECT_TRUE(ok);
        EXPECT_FALSE(popped.has_value());
    }

    MemoryBlockCache::CacheItem big;
    big.cache_key   = 888;
    big.block_index = 81;
    big.block_size  = 2000;
    big.is_resident = false;
    big.is_big      = true;
    {
        auto [ok, popped] = cache.put(big);
        EXPECT_TRUE(ok);
        ASSERT_TRUE(popped.has_value());
        EXPECT_EQ(popped->cache_key, 888);
        EXPECT_EQ(popped->block_index, 80);
        EXPECT_EQ(popped->block_size, 1000u);
        EXPECT_FALSE(popped->is_big);
    }

    auto mr = cache.match(888);
    EXPECT_EQ(mr.matched_index, 81);
    EXPECT_EQ(mr.block_size, 2000u);
    EXPECT_TRUE(mr.is_big);
}

TEST(MemoryBlockCacheTest, put_ReturnPoppedItem_WhenCacheFull) {
    MemoryBlockCache cache;

    // Shrink capacity to make "full" branch testable.
    cache.lru_cache_ = LRUCache<CacheKeyType, MemoryBlockCache::CacheItem>(1);

    MemoryBlockCache::CacheItem item1;
    item1.cache_key   = 1001;
    item1.block_index = 1;
    item1.block_size  = 111;
    item1.is_resident = false;
    item1.is_big      = true;
    {
        auto [success, popped_item_opt] = cache.put(item1);
        EXPECT_TRUE(success);
        EXPECT_FALSE(popped_item_opt.has_value());
        EXPECT_EQ(cache.size(), 1u);
    }

    MemoryBlockCache::CacheItem item2;
    item2.cache_key   = 1002;
    item2.block_index = 2;
    item2.block_size  = 222;
    item2.is_resident = false;
    item2.is_big      = true;
    {
        auto [success, popped_item_opt] = cache.put(item2);
        EXPECT_TRUE(success);
        ASSERT_TRUE(popped_item_opt.has_value());
        EXPECT_EQ(popped_item_opt->cache_key, 1001);
        EXPECT_EQ(popped_item_opt->block_index, 1);
        EXPECT_EQ(popped_item_opt->block_size, 111u);
        EXPECT_EQ(cache.size(), 1u);

        auto mr1 = cache.match(1001);
        EXPECT_TRUE(isNullBlockIdx(mr1.matched_index));
        EXPECT_EQ(mr1.block_size, 0u);

        auto mr2 = cache.match(1002);
        EXPECT_EQ(mr2.matched_index, 2);
        EXPECT_EQ(mr2.block_size, 222u);
    }
}

TEST(MemoryBlockCacheTest, put_ReturnFalse_WhenCacheFullButPopFailed) {
    MemoryBlockCache cache;

    // Force a degenerate LRU cache with capacity 0 so `full()` is true but `pop()` fails because it's empty.
    cache.lru_cache_ = LRUCache<CacheKeyType, MemoryBlockCache::CacheItem>(0);

    MemoryBlockCache::CacheItem item;
    item.cache_key   = 2001;
    item.block_index = 1;
    item.block_size  = 1;
    item.is_resident = false;
    item.is_big      = true;

    auto [success, popped_item_opt] = cache.put(item);
    EXPECT_FALSE(success);
    EXPECT_FALSE(popped_item_opt.has_value());
    EXPECT_TRUE(cache.empty());
    EXPECT_EQ(cache.size(), 0u);
}

TEST(MemoryBlockCacheTest, pop_ReturnNonResidentBlocks_WhenSkipResident) {
    MemoryBlockCache cache;
    // Insert 4 items; mark one as resident
    for (int i = 0; i < 4; ++i) {
        MemoryBlockCache::CacheItem item;
        item.cache_key                  = 300 + i;
        item.block_index                = 20 + i;
        item.block_size                 = 1000 + i;
        item.is_resident                = (i == 1);  // key=301 is resident
        item.is_big                     = true;
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

TEST(MemoryBlockCacheTest, pop_ReturnEmpty_WhenCacheEmpty) {
    MemoryBlockCache cache;
    auto             popped = cache.pop(1);
    EXPECT_TRUE(popped.empty());
}

TEST(MemoryBlockCacheTest, pop_ReturnThrow_WhenNumsNotPositive) {
    // Ensure CHECK throws instead of aborting.
    rtp_llm::StaticConfig::user_ft_core_dump_on_exception = false;

    MemoryBlockCache cache;
    EXPECT_THROW((void)cache.pop(0), rtp_llm::RTPException);
    EXPECT_THROW((void)cache.pop(-1), rtp_llm::RTPException);
}

TEST(MemoryBlockCacheTest, pop_ReturnLimitedBlocks_WhenNumsLessThanSize) {
    MemoryBlockCache cache;
    std::vector<int> keys;
    keys.reserve(5);
    for (int i = 0; i < 5; ++i) {
        MemoryBlockCache::CacheItem item;
        item.cache_key                  = 400 + i;
        item.block_index                = 30 + i;
        item.block_size                 = 2000 + i;
        item.is_resident                = false;
        item.is_big                     = true;
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

TEST(MemoryBlockCacheTest, pop_ReturnEmpty_WhenAllResident) {
    MemoryBlockCache cache;
    for (int i = 0; i < 2; ++i) {
        MemoryBlockCache::CacheItem item;
        item.cache_key                  = 600 + i;
        item.block_index                = 50 + i;
        item.block_size                 = 4000 + i;
        item.is_resident                = true;
        item.is_big                     = true;
        auto [success, popped_item_opt] = cache.put(item);
        EXPECT_TRUE(success);
        EXPECT_FALSE(popped_item_opt.has_value());
    }
    EXPECT_EQ(cache.size(), 2u);
    auto popped = cache.pop(3);
    EXPECT_TRUE(popped.empty());
    EXPECT_EQ(cache.size(), 2u);
}

TEST(MemoryBlockCacheTest, empty_ReturnTrue_WhenCacheEmpty) {
    MemoryBlockCache cache;
    EXPECT_TRUE(cache.empty());
    EXPECT_EQ(cache.size(), 0u);
    EXPECT_FALSE(cache.contains(42));

    auto mr = cache.match(42);
    EXPECT_TRUE(isNullBlockIdx(mr.matched_index));
    EXPECT_EQ(mr.block_size, 0u);
    EXPECT_FALSE(mr.is_big);
}

}  // namespace rtp_llm::test

int main(int argc, char** argv) {
    rtp_llm::initLogger();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
