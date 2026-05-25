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
    EXPECT_EQ(result1.action, BlockCache::PutResult::Action::INSERTED);

    // Put a duplicate key
    auto result2 = cache_->put(item);
    EXPECT_EQ(result2.action, BlockCache::PutResult::Action::REPLACED);

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
    EXPECT_EQ(result1.action, BlockCache::PutResult::Action::INSERTED);
    CacheItem item2   = {102, 0, 2, false};
    auto      result2 = cache_->put(item2);
    EXPECT_EQ(result2.action, BlockCache::PutResult::Action::INSERTED);
    CacheItem item3   = {103, 0, 3, false};
    auto      result3 = cache_->put(item3);
    EXPECT_EQ(result3.action, BlockCache::PutResult::Action::INSERTED);
    CacheItem item4   = {104, 0, 4, false};
    auto      result4 = cache_->put(item4);
    EXPECT_EQ(result4.action, BlockCache::PutResult::Action::INSERTED);
    CacheItem item5   = {105, 0, 5, false};
    auto      result5 = cache_->put(item5);
    EXPECT_EQ(result5.action, BlockCache::PutResult::Action::INSERTED);

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
    EXPECT_EQ(result6.action, BlockCache::PutResult::Action::INSERTED);
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

// ==================== Epoch sentinel separation ====================
//
// Three-way semantics:
//   current_batch_epoch == NO_EPOCH_FILTER (-1): bypass filter, see everything
//   current_batch_epoch == 0 (GLOBAL_EPOCH):     no batch identity, only global
//   current_batch_epoch >= 1:                    same batch + global

TEST_F(BlockCacheTest, MatchEpochZeroDoesNotSeeBatchLocal) {
    // global entry
    CacheItem global_item = {201, 0, /*block_index=*/1, /*is_resident=*/false, /*epoch=*/0};
    EXPECT_EQ(cache_->put(global_item).action, BlockCache::PutResult::Action::INSERTED);

    // batch-local entry, epoch=5
    CacheItem batch_item = {202, 0, /*block_index=*/2, /*is_resident=*/false, /*epoch=*/5};
    EXPECT_EQ(cache_->put(batch_item).action, BlockCache::PutResult::Action::INSERTED);

    // current_batch_epoch == 0: "no batch identity" — must NOT see batch-local
    EXPECT_EQ(cache_->match(201, /*group_id=*/0, /*current_batch_epoch=*/0).matched_index, 1);
    EXPECT_TRUE(isNullBlockIdx(cache_->match(202, /*group_id=*/0, /*current_batch_epoch=*/0).matched_index));
    EXPECT_TRUE(isNullBlockIdx(cache_->match(202).matched_index));

    // current_batch_epoch == NO_EPOCH_FILTER: see everything
    EXPECT_EQ(cache_->match(201, 0, BlockCache::NO_EPOCH_FILTER).matched_index, 1);
    EXPECT_EQ(cache_->match(202, 0, BlockCache::NO_EPOCH_FILTER).matched_index, 2);

    // same batch (5): see global + same-batch
    EXPECT_EQ(cache_->match(201, 0, /*current_batch_epoch=*/5).matched_index, 1);
    EXPECT_EQ(cache_->match(202, 0, /*current_batch_epoch=*/5).matched_index, 2);

    // different batch (7): only global
    EXPECT_EQ(cache_->match(201, 0, /*current_batch_epoch=*/7).matched_index, 1);
    EXPECT_TRUE(isNullBlockIdx(cache_->match(202, 0, /*current_batch_epoch=*/7).matched_index));
}

TEST_F(BlockCacheTest, PositiveEpochsWithSameKeyDoNotOverwriteEachOther) {
    CacheItem epoch_42 = {301, 0, /*block_index=*/2, /*is_resident=*/false, /*epoch=*/42};
    CacheItem epoch_43 = {301, 0, /*block_index=*/3, /*is_resident=*/false, /*epoch=*/43};

    EXPECT_EQ(cache_->put(epoch_42).action, BlockCache::PutResult::Action::INSERTED);
    EXPECT_EQ(cache_->put(epoch_43).action, BlockCache::PutResult::Action::INSERTED);

    EXPECT_EQ(cache_->match(301, 0, /*current_batch_epoch=*/42).matched_index, 2);
    EXPECT_EQ(cache_->match(301, 0, /*current_batch_epoch=*/43).matched_index, 3);
    EXPECT_TRUE(isNullBlockIdx(cache_->match(301, 0, BlockCache::GLOBAL_EPOCH).matched_index));
    EXPECT_EQ(cache_->size(), 2u);
}

TEST_F(BlockCacheTest, SameEpochSameKeyStillReplaces) {
    CacheItem first  = {302, 0, /*block_index=*/4, /*is_resident=*/false, /*epoch=*/42};
    CacheItem second = {302, 0, /*block_index=*/5, /*is_resident=*/false, /*epoch=*/42};

    EXPECT_EQ(cache_->put(first).action, BlockCache::PutResult::Action::INSERTED);
    auto result = cache_->put(second);

    EXPECT_EQ(result.action, BlockCache::PutResult::Action::REPLACED);
    EXPECT_EQ(result.old_block_index, 4);
    EXPECT_EQ(cache_->match(302, 0, /*current_batch_epoch=*/42).matched_index, 5);
    EXPECT_EQ(cache_->size(), 1u);
}

TEST_F(BlockCacheTest, ContainsOnlyChecksGlobalEpoch) {
    CacheItem batch_item = {303, 0, /*block_index=*/6, /*is_resident=*/false, /*epoch=*/42};
    EXPECT_EQ(cache_->put(batch_item).action, BlockCache::PutResult::Action::INSERTED);

    EXPECT_FALSE(cache_->contains(303, 0));
    EXPECT_TRUE(cache_->containsAnyEpoch(303, 0));

    CacheItem global_item = {303, 0, /*block_index=*/7, /*is_resident=*/false, /*epoch=*/0};
    EXPECT_EQ(cache_->put(global_item).action, BlockCache::PutResult::Action::INSERTED);

    EXPECT_TRUE(cache_->contains(303, 0));
    EXPECT_TRUE(cache_->containsAnyEpoch(303, 0));
}

TEST_F(BlockCacheTest, RemoveOnlyRemovesGlobalEpoch) {
    CacheItem batch_item  = {304, 0, /*block_index=*/8, /*is_resident=*/false, /*epoch=*/42};
    CacheItem global_item = {304, 0, /*block_index=*/9, /*is_resident=*/false, /*epoch=*/0};
    EXPECT_EQ(cache_->put(batch_item).action, BlockCache::PutResult::Action::INSERTED);
    EXPECT_EQ(cache_->put(global_item).action, BlockCache::PutResult::Action::INSERTED);

    auto removed = cache_->remove(304, 0);
    ASSERT_TRUE(removed.has_value());
    EXPECT_EQ(removed->block_index, 9);

    EXPECT_FALSE(cache_->contains(304, 0));
    EXPECT_TRUE(cache_->containsAnyEpoch(304, 0));
    EXPECT_EQ(cache_->match(304, 0, /*current_batch_epoch=*/42).matched_index, 8);
}

// ==================== Resident protection on put() ====================

TEST_F(BlockCacheTest, PutDoesNotDowngradeResidentEntry) {
    // Insert a resident entry first.
    CacheItem resident_item = {101, 0, /*block_index=*/1, /*is_resident=*/true};
    auto      r1            = cache_->put(resident_item);
    EXPECT_EQ(r1.action, BlockCache::PutResult::Action::INSERTED);

    // A subsequent put with is_resident=false (different physical block) must NOT
    // overwrite the resident entry. Old behavior on `main` was a complete no-op
    // (return false). After the PutResult refactor, REPLACE must be guarded so a
    // non-resident put cannot drop the resident bit and swap blocks.
    CacheItem non_resident_item = {101, 0, /*block_index=*/2, /*is_resident=*/false};
    auto      r2                = cache_->put(non_resident_item);
    EXPECT_EQ(r2.action, BlockCache::PutResult::Action::SKIPPED);

    // The resident entry survives, still pointing to its original block.
    auto match = cache_->match(101);
    EXPECT_EQ(match.matched_index, 1);

    // pop() must not evict the resident entry.
    auto popped = cache_->pop(5);
    EXPECT_EQ(popped.size(), 0);
    EXPECT_EQ(cache_->size(), 1);
}

TEST_F(BlockCacheTest, PutAllowsResidentToReplaceNonResident) {
    // Reverse direction: an existing non-resident entry SHOULD be replaceable by
    // a resident put — that's how multi-task prompts get promoted.
    CacheItem non_resident_item = {101, 0, /*block_index=*/1, /*is_resident=*/false};
    EXPECT_EQ(cache_->put(non_resident_item).action, BlockCache::PutResult::Action::INSERTED);

    CacheItem resident_item = {101, 0, /*block_index=*/2, /*is_resident=*/true};
    auto      r2            = cache_->put(resident_item);
    EXPECT_EQ(r2.action, BlockCache::PutResult::Action::REPLACED);
    EXPECT_EQ(r2.old_block_index, 1);

    // After promotion the entry is resident and points to the new block.
    EXPECT_EQ(cache_->match(101).matched_index, 2);
    auto popped = cache_->pop(5);
    EXPECT_EQ(popped.size(), 0);
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

// Tiered eviction (memory cache export) must never select epoch>0 entries:
// memory cache has no epoch concept, so exporting batch-local data would leak
// it into the global memory cache and break batch isolation. epoch>0 entries
// are reclaimed only via BlockCache::pop's Phase 1 (local free, no export).
TEST_F(BlockCacheTest, SelectAndEvictSkipsBatchLocalEntries) {
    CacheItem global1     = {101, 0, 1, false, /*epoch=*/0};
    CacheItem batch_local = {102, 0, 2, false, /*epoch=*/42};
    CacheItem global2     = {103, 0, 3, false, /*epoch=*/0};
    cache_->put(global1);
    cache_->put(batch_local);
    cache_->put(global2);

    auto result = cache_->selectAndEvict(/*min_blocks=*/3);
    // Only the two epoch=0 entries are eligible for tiered eviction.
    EXPECT_EQ(result.evicted_keys.size(), 2u);
    std::set<CacheKeyType> evicted(result.evicted_keys.begin(), result.evicted_keys.end());
    EXPECT_TRUE(evicted.count(101));
    EXPECT_TRUE(evicted.count(103));
    EXPECT_EQ(evicted.count(102), 0u);  // batch-local entry preserved
    // batch-local entry stays in cache for Phase-1 pop to reclaim locally.
    EXPECT_EQ(cache_->size(), 1u);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
