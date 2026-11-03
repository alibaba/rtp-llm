#include "gtest/gtest.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "rtp_llm/cpp/cache/BlockPool.h"

#include "rtp_llm/cpp/cache/connector/memory/MemoryDiskBlockCache.h"

namespace rtp_llm::test {
namespace {

MemoryDiskBlockCache::CacheItem memoryItem(CacheKeyType key, BlockIdxType block, bool complete = true) {
    MemoryDiskBlockCache::CacheItem item;
    item.cache_key    = key;
    item.backing_type = CacheBackingType::MEMORY;
    item.block_index  = block;
    item.disk_slot    = -1;
    item.is_complete  = complete;
    return item;
}

MemoryDiskBlockCache::CacheItem diskItem(CacheKeyType key, int32_t slot, bool complete = true) {
    MemoryDiskBlockCache::CacheItem item;
    item.cache_key    = key;
    item.backing_type = CacheBackingType::DISK;
    item.block_index  = NULL_BLOCK_IDX;
    item.disk_slot    = slot;
    item.is_complete  = complete;
    return item;
}

}  // namespace

TEST(MemoryDiskBlockCacheTest, ContainsAndMatchMemoryAndDisk) {
    MemoryDiskBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(memoryItem(1, 10)).first);
    ASSERT_TRUE(cache.putCommitted(diskItem(2, 20)).first);

    EXPECT_TRUE(cache.contains(1));
    EXPECT_TRUE(cache.contains(2));

    auto mem = cache.match(1);
    EXPECT_EQ(mem.backing_type, CacheBackingType::MEMORY);
    EXPECT_EQ(mem.matched_index, 10);

    auto disk = cache.match(2);
    EXPECT_EQ(disk.backing_type, CacheBackingType::DISK);
    EXPECT_EQ(disk.disk_slot, 20);
}

TEST(MemoryDiskBlockCacheTest, SharedAccessSeqEvictsOldestAcrossBackings) {
    MemoryDiskBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(memoryItem(1, 10)).first);
    ASSERT_TRUE(cache.putCommitted(diskItem(2, 20)).first);

    ASSERT_FALSE(isNullBlockIdx(cache.match(1).matched_index));

    auto evicted = cache.popOldestEvictable();
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 2);
    EXPECT_EQ(evicted->backing_type, CacheBackingType::DISK);
}

TEST(MemoryDiskBlockCacheTest, KindAwareEvictionOnlyPopsRequestedKind) {
    MemoryDiskBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(memoryItem(1, 10, false)).first);
    ASSERT_TRUE(cache.putCommitted(diskItem(2, 20, false)).first);
    ASSERT_TRUE(cache.putCommitted(memoryItem(3, 30, true)).first);

    auto evicted = cache.popOldestEvictable(CacheBlockKind::COMPLETE);
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 3);
    EXPECT_TRUE(evicted->is_complete);

    evicted = cache.popOldestEvictable(CacheBlockKind::INCOMPLETE);
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 1);
    EXPECT_FALSE(evicted->is_complete);
}

TEST(MemoryDiskBlockCacheTest, KindAwareEvictionChoosesOldestAcrossMemoryAndDiskForSameKind) {
    MemoryDiskBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(memoryItem(1, 10, true)).first);
    ASSERT_TRUE(cache.putCommitted(diskItem(2, 20, true)).first);
    ASSERT_FALSE(isNullBlockIdx(cache.match(1).matched_index));

    auto evicted = cache.popOldestEvictable(CacheBlockKind::COMPLETE);
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 2);
    EXPECT_EQ(evicted->backing_type, CacheBackingType::DISK);
}

TEST(MemoryDiskBlockCacheTest, ContainsDoesNotUpdateRecency) {
    MemoryDiskBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(memoryItem(1, 10)).first);
    ASSERT_TRUE(cache.putCommitted(diskItem(2, 20)).first);
    ASSERT_TRUE(cache.contains(1));

    auto evicted = cache.popOldestEvictable();
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 1);
}

TEST(MemoryDiskBlockCacheTest, PartialToCompleteCanUpgradeAcrossBacking) {
    MemoryDiskBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(memoryItem(1, 10, false)).first);

    auto [ok, popped] = cache.putCommitted(diskItem(1, 20, true));
    ASSERT_TRUE(ok);
    ASSERT_TRUE(popped.has_value());
    EXPECT_EQ(popped->backing_type, CacheBackingType::MEMORY);
    EXPECT_EQ(popped->block_index, 10);

    auto match = cache.match(1);
    EXPECT_EQ(match.backing_type, CacheBackingType::DISK);
    EXPECT_EQ(match.disk_slot, 20);
    EXPECT_TRUE(match.is_complete);
}

TEST(MemoryDiskBlockCacheTest, PartialToCompleteDoesNotReplaceInFlightItem) {
    MemoryDiskBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(memoryItem(1, 10, false)).first);

    auto in_flight = cache.matchAndMarkInFlight(1);
    EXPECT_EQ(in_flight.backing_type, CacheBackingType::MEMORY);
    EXPECT_EQ(in_flight.matched_index, 10);

    auto [ok, popped] = cache.putCommitted(diskItem(1, 20, true));
    EXPECT_FALSE(ok);
    EXPECT_FALSE(popped.has_value());

    auto match = cache.match(1);
    EXPECT_EQ(match.backing_type, CacheBackingType::MEMORY);
    EXPECT_EQ(match.matched_index, 10);
    EXPECT_FALSE(match.is_complete);
}

TEST(MemoryDiskBlockCacheTest, InFlightEntryIsNotEvictable) {
    MemoryDiskBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(memoryItem(1, 10)).first);
    ASSERT_TRUE(cache.putCommitted(diskItem(2, 20)).first);
    ASSERT_TRUE(cache.markInFlight(1, CacheBackingType::MEMORY, 10, -1));

    auto evicted = cache.popOldestEvictable();
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 2);

    cache.releaseInFlight(1, CacheBackingType::MEMORY, 10, -1);
    evicted = cache.popOldestEvictable();
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 1);
}

TEST(MemoryDiskBlockCacheTest, MatchAndMarkInFlightPreventsEviction) {
    MemoryDiskBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(memoryItem(1, 10)).first);
    ASSERT_TRUE(cache.putCommitted(diskItem(2, 20)).first);

    auto match = cache.matchAndMarkInFlight(1);
    EXPECT_EQ(match.backing_type, CacheBackingType::MEMORY);
    EXPECT_EQ(match.matched_index, 10);

    auto evicted = cache.popOldestEvictable();
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 2);

    cache.releaseInFlight(1, CacheBackingType::MEMORY, 10, -1);
    evicted = cache.popOldestEvictable();
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->cache_key, 1);
}

TEST(MemoryDiskBlockCacheTest, RemoveIfMatchChecksBackingAndSlot) {
    MemoryDiskBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(diskItem(2, 20)).first);

    EXPECT_FALSE(cache.removeIfMatch(2, CacheBackingType::DISK, NULL_BLOCK_IDX, 21).has_value());
    auto removed = cache.removeIfMatch(2, CacheBackingType::DISK, NULL_BLOCK_IDX, 20);
    ASSERT_TRUE(removed.has_value());
    EXPECT_FALSE(cache.contains(2));
}

}  // namespace rtp_llm::test

namespace rtp_llm::test {

TEST(MemoryDiskBlockCacheTest, CompletedReadCannotConsumeAnotherReadersUnpinnedBacking) {
    for (const auto& item : {memoryItem(1, 10), diskItem(1, 20)}) {
        SCOPED_TRACE(static_cast<int>(item.backing_type));
        MemoryDiskBlockCache cache;
        ASSERT_TRUE(cache.putCommitted(item).first);
        cache.matchAndMarkInFlight(1);  // A's CopyPlan already owns its backing ref.
        cache.matchAndMarkInFlight(1);  // B pauses before backing requestReference.
        EXPECT_FALSE(cache.removeIfMatch(1,
                                         item.backing_type,
                                         item.block_index,
                                         item.disk_slot,
                                         /*completed_read=*/true));
        EXPECT_TRUE(cache.contains(1));
        cache.releaseInFlight(1, item.backing_type, item.block_index, item.disk_slot);  // A's plan is destroyed.
        EXPECT_FALSE(cache.popOldestEvictable());  // B's marker still protects the cache owner.
        // B acquires its backing ref, finishes its copy, then consumes as the
        // sole remaining reader. This preserves the old single-reader behavior.
        EXPECT_TRUE(cache.removeIfMatch(1,
                                        item.backing_type,
                                        item.block_index,
                                        item.disk_slot,
                                        /*completed_read=*/true));
        EXPECT_FALSE(cache.contains(1));
    }
}

TEST(MemoryDiskBlockCacheTest, OrdinaryRemovalNeverConsumesAnInFlightMarker) {
    for (const auto& item : {memoryItem(1, 10), diskItem(1, 20)}) {
        MemoryDiskBlockCache cache;
        ASSERT_TRUE(cache.putCommitted(item).first);
        cache.matchAndMarkInFlight(1);
        EXPECT_FALSE(cache.removeIfMatch(1, item.backing_type, item.block_index, item.disk_slot));
        EXPECT_FALSE(cache.remove(1));
        if (item.backing_type == CacheBackingType::MEMORY) {
            EXPECT_FALSE(cache.removeIfMatch(1, item.block_index));
        }
        cache.releaseInFlight(1, item.backing_type, item.block_index, item.disk_slot);
        EXPECT_TRUE(cache.removeIfMatch(1, item.backing_type, item.block_index, item.disk_slot));
        EXPECT_TRUE(cache.empty());
    }
}

TEST(MemoryDiskBlockCacheTest, CompletedReadRequiresItsOwnMatchingMarker) {
    MemoryDiskBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(memoryItem(1, 10)).first);
    EXPECT_FALSE(cache.removeIfMatch(1, CacheBackingType::MEMORY, 10, -1, /*completed_read=*/true));
    cache.matchAndMarkInFlight(1);
    EXPECT_FALSE(cache.removeIfMatch(1, CacheBackingType::MEMORY, 11, -1, /*completed_read=*/true));
    EXPECT_FALSE(cache.removeIfMatch(1, CacheBackingType::DISK, NULL_BLOCK_IDX, 10, /*completed_read=*/true));
    EXPECT_TRUE(cache.removeIfMatch(1, CacheBackingType::MEMORY, 10, -1, /*completed_read=*/true));
}

TEST(MemoryDiskBlockCacheTest, AbortedSecondReaderReleasesMarkerWithoutPreventingFirstReaderConsume) {
    for (const auto& item : {memoryItem(1, 10), diskItem(1, 20)}) {
        MemoryDiskBlockCache cache;
        ASSERT_TRUE(cache.putCommitted(item).first);
        cache.matchAndMarkInFlight(1);
        cache.matchAndMarkInFlight(1);
        cache.releaseInFlight(1, item.backing_type, item.block_index, item.disk_slot);  // B aborts.
        EXPECT_TRUE(cache.removeIfMatch(1,
                                        item.backing_type,
                                        item.block_index,
                                        item.disk_slot,
                                        /*completed_read=*/true));
        cache.releaseInFlight(1, item.backing_type, item.block_index, item.disk_slot);  // A's plan retires.
        EXPECT_TRUE(cache.empty());
    }
}

namespace {
class MemoryCopyPinTest: public ::testing::Test {
protected:
    void SetUp() override {
        if (const char* value = std::getenv("RTP_LLM_PIN_HOST_BLOCK_POOL")) {
            old_pin_ = value;
        }
        setenv("RTP_LLM_PIN_HOST_BLOCK_POOL", "0", 1);
    }
    void TearDown() override {
        if (old_pin_) {
            setenv("RTP_LLM_PIN_HOST_BLOCK_POOL", old_pin_->c_str(), 1);
        } else {
            unsetenv("RTP_LLM_PIN_HOST_BLOCK_POOL");
        }
    }
    std::optional<std::string> old_pin_;
};

BlockPoolPtr oneBlockHostPool() {
    MemoryLayoutConfig layout;
    layout.layer_num                = 1;
    layout.block_num                = 2;
    layout.dtype                    = TYPE_FP16;
    layout.kv_cache_offset_bytes    = 0;
    layout.kv_scale_offset_bytes    = 2048;
    layout.kv_block_stride_bytes    = 1024;
    layout.k_block_stride_bytes     = 512;
    layout.v_block_stride_bytes     = 512;
    layout.kv_block_pool_size_bytes = 2048;
    layout.kv_scale_pool_size_bytes = 0;
    layout.total_size_bytes         = layout.kv_block_pool_size_bytes;
    BlockPoolConfig config;
    config.block_num        = 2;
    config.total_size_bytes = layout.total_size_bytes;
    config.memory_layouts   = {layout};
    auto pool               = std::make_shared<BlockPool>(config, AllocationType::HOST);
    if (!pool->init()) {
        throw std::runtime_error("host pool init failed");
    }
    return pool;
}
}  // namespace

TEST_F(MemoryCopyPinTest, FirstCompletionCannotFreeOrOverwriteSecondReadersMatchedHostBlock) {
    auto pool   = oneBlockHostPool();
    auto blocks = pool->malloc(1);
    ASSERT_EQ(blocks.size(), 1u);
    const auto block = blocks[0];
    auto*      value = static_cast<uint64_t*>(pool->convertIndexToAddr(0, block).kv_addr);
    ASSERT_NE(value, nullptr);
    *value = 0x123456789abcdef0ULL;
    pool->blockCacheReference(block);
    pool->requestFree(block);  // Initial writer leaves the cache owner.
    MemoryDiskBlockCache cache;
    ASSERT_TRUE(cache.putCommitted(memoryItem(1, block)).first);

    auto first = cache.matchAndMarkInFlight(1);
    pool->requestReference(first.matched_index);   // A's CopyPlan owns one request ref.
    auto second  = cache.matchAndMarkInFlight(1);  // B is paused before taking its ref.
    auto removed = cache.removeIfMatch(1, CacheBackingType::MEMORY, block, -1, /*completed_read=*/true);
    EXPECT_FALSE(removed);
    if (removed) {
        pool->blockCacheFree(removed->block_index);  // Exactly the old read_done callback.
    }
    pool->requestFree(first.matched_index);  // A's CopyPlan destructor.
    cache.releaseInFlight(1, CacheBackingType::MEMORY, block, -1);
    EXPECT_EQ(pool->freeBlocksNum(), 0u);
    auto unrelated = pool->malloc(1);
    EXPECT_TRUE(unrelated.empty());  // Old code can allocate/overwrite B's matched block here.
    if (!unrelated.empty()) {
        *value = 0;
    }

    pool->requestReference(second.matched_index);  // B resumes its real control flow.
    EXPECT_EQ(*value, 0x123456789abcdef0ULL);
    removed = cache.removeIfMatch(1, CacheBackingType::MEMORY, block, -1, /*completed_read=*/true);
    ASSERT_TRUE(removed);
    pool->blockCacheFree(removed->block_index);
    pool->requestFree(second.matched_index);
    cache.releaseInFlight(1, CacheBackingType::MEMORY, block, -1);
    if (!unrelated.empty()) {
        pool->requestFree(unrelated);
    }
    EXPECT_EQ(pool->freeBlocksNum(), 1u);
    EXPECT_TRUE(cache.empty());
}

}  // namespace rtp_llm::test
