// Unit tests for SharedBlockCache unified-path CACHE refcount bookkeeping.
//
// Primary regression coverage: XR-C1 / R03 finding #5 — the
// snapshot-then-resurrect-then-cascade race where a put on the same key with
// the same super_block_id between Phase A (selectAndEvictUnified) and Phase
// B (cascade) used to skip the dec, leaking exactly one CACHE refcount per
// occurrence and never returning S to the SuperBlockFreeList.
//
// All tests run without a BlockPool wired (``init`` is not called), so the
// cascade body only touches the UnifiedRefCounter side — sufficient for
// regression coverage of the leak path.

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/BlockRefCounter.h"
#include "rtp_llm/cpp/cache/SharedBlockCache.h"
#include "rtp_llm/cpp/cache/Types.h"

namespace rtp_llm {
namespace test {

class SharedBlockCacheUnifiedTest: public ::testing::Test {
protected:
    void SetUp() override {
        cache_   = std::make_unique<SharedBlockCache>();
        counter_ = std::make_unique<UnifiedRefCounter>();
        counter_->init(/*num_super_blocks=*/64);

        cache_->setUnifiedRefCounter(counter_.get());
        cache_->setSuperBlockReclaimCallback([this](int S) { reclaimed_.push_back(S); });
        // group_pools_ left empty — the cascade body's per-pool loop is a
        // no-op when there are no pools, so we exercise the unified counter
        // path in isolation.
    }

    void TearDown() override {
        cache_.reset();
        counter_.reset();
        reclaimed_.clear();
    }

    int cacheRef(int S) const {
        return counter_->getRefCount(S, UnifiedRefCounter::Kind::CACHE);
    }

    std::unique_ptr<SharedBlockCache>  cache_;
    std::unique_ptr<UnifiedRefCounter> counter_;
    std::vector<int>                   reclaimed_;
};

// Sanity: a single put + a single evictAndFree drops CACHE to zero and fires
// the reclaim callback (no race involved).
TEST_F(SharedBlockCacheUnifiedTest, SinglePutEvictBalance) {
    constexpr CacheKeyType K = 0xC0FFEE;
    constexpr int          S = 42;

    cache_->putUnified(K, S, /*is_resident=*/true);
    EXPECT_EQ(cacheRef(S), 1);
    EXPECT_FALSE(counter_->isZero(S));

    const size_t cascaded = cache_->evictAndFreeUnified(/*min_super_blocks=*/1);
    EXPECT_EQ(cascaded, 1u);
    EXPECT_EQ(cacheRef(S), 0);
    EXPECT_TRUE(counter_->isZero(S));
    ASSERT_EQ(reclaimed_.size(), 1u);
    EXPECT_EQ(reclaimed_[0], S);
}

// XR-C1 regression: snapshot-then-resurrect-then-cascade on the same (K, S)
// must NOT leak. Drives the race deterministically via the shared cascade
// helper.
//
// Before the fix, evictAndFreeUnified's phase B re-acquired mu_ and skipped
// the dec when ``lru_cache_.contains(K) && current.superBlockId() == S``,
// leaving CACHE(S) at 2 after the snapshot dec was suppressed. With the
// re-residency probe removed, the snapshot dec always lands.
TEST_F(SharedBlockCacheUnifiedTest, SameKeySameSResurrectionDoesNotLeakCache) {
    constexpr CacheKeyType K = 0xBEEF;
    constexpr int          S = 7;

    // T0: original put. CACHE(S)=1, LRU={K→S}.
    cache_->putUnified(K, S, /*is_resident=*/true);
    ASSERT_EQ(cacheRef(S), 1);

    // T1: Phase A only — snapshot drains the LRU but does NOT touch CACHE.
    auto snapshot = cache_->selectAndEvictUnified(/*min_super_blocks=*/1);
    ASSERT_EQ(snapshot.size(), 1u);
    EXPECT_EQ(snapshot[0].cache_key, K);
    EXPECT_EQ(snapshot[0].superBlockId(), S);
    EXPECT_EQ(cacheRef(S), 1);  // selectAndEvictUnified must not dec CACHE
    EXPECT_FALSE(cache_->contains(K));

    // T2 (racing put on same K, same S): re-inserts via the new-entry
    // branch; bumps CACHE independently of T1's snapshot. CACHE(S)=2.
    cache_->putUnified(K, S, /*is_resident=*/true);
    ASSERT_EQ(cacheRef(S), 2);
    ASSERT_TRUE(cache_->contains(K));

    // T1 continues with Phase B on the stale snapshot. Pre-fix this skipped
    // the dec (because LRU contains K at the same S) and left CACHE at 2
    // permanently. Post-fix: unconditional dec → CACHE drops to 1.
    cache_->cascadeUnifiedSnapshot(snapshot, /*cascaded_out=*/nullptr);
    EXPECT_EQ(cacheRef(S), 1) << "snapshot dec must always pair with the original put bump";
    EXPECT_FALSE(counter_->isZero(S));
    EXPECT_TRUE(reclaimed_.empty()) << "S still owned by the resurrected resident entry";

    // Final natural eviction of T2's resident entry drops the remaining
    // CACHE bump, isZero fires, and S returns to the free list.
    const size_t cascaded = cache_->evictAndFreeUnified(/*min_super_blocks=*/1);
    EXPECT_EQ(cascaded, 1u);
    EXPECT_EQ(cacheRef(S), 0);
    EXPECT_TRUE(counter_->isZero(S));
    ASSERT_EQ(reclaimed_.size(), 1u);
    EXPECT_EQ(reclaimed_[0], S);
}

// Variant: the racing put lands at a DIFFERENT S'. The snapshot dec on S
// is unambiguously correct (no possible double-dec on S'); we mostly assert
// the per-S accounting stays sane and both supers reclaim cleanly.
TEST_F(SharedBlockCacheUnifiedTest, SameKeyDifferentSResurrectionKeepsBothBalanced) {
    constexpr CacheKeyType K  = 0xBABE;
    constexpr int          S1 = 11;
    constexpr int          S2 = 13;

    cache_->putUnified(K, S1, /*is_resident=*/true);
    ASSERT_EQ(cacheRef(S1), 1);

    auto snapshot = cache_->selectAndEvictUnified(1);
    ASSERT_EQ(snapshot.size(), 1u);
    ASSERT_EQ(snapshot[0].superBlockId(), S1);

    // Resurrect K at a DIFFERENT super-block S2.
    cache_->putUnified(K, S2, /*is_resident=*/true);
    EXPECT_EQ(cacheRef(S1), 1);  // unchanged
    EXPECT_EQ(cacheRef(S2), 1);

    cache_->cascadeUnifiedSnapshot(snapshot, /*cascaded_out=*/nullptr);
    EXPECT_EQ(cacheRef(S1), 0);
    EXPECT_TRUE(counter_->isZero(S1));
    ASSERT_EQ(reclaimed_.size(), 1u);
    EXPECT_EQ(reclaimed_.back(), S1);

    // Natural eviction of the resident K→S2 entry closes out S2.
    const size_t cascaded = cache_->evictAndFreeUnified(1);
    EXPECT_EQ(cascaded, 1u);
    EXPECT_EQ(cacheRef(S2), 0);
    EXPECT_TRUE(counter_->isZero(S2));
    ASSERT_EQ(reclaimed_.size(), 2u);
    EXPECT_EQ(reclaimed_.back(), S2);
}

// Multiple snapshot entries: cascading a snapshot with two items must dec
// CACHE twice and fire two reclaim callbacks (assuming no resurrection).
TEST_F(SharedBlockCacheUnifiedTest, MultipleSnapshotEntriesCascadeAll) {
    cache_->putUnified(/*K=*/1, /*S=*/3, true);
    cache_->putUnified(/*K=*/2, /*S=*/4, true);
    cache_->putUnified(/*K=*/3, /*S=*/5, true);

    EXPECT_EQ(cacheRef(3), 1);
    EXPECT_EQ(cacheRef(4), 1);
    EXPECT_EQ(cacheRef(5), 1);

    const size_t cascaded = cache_->evictAndFreeUnified(/*min_super_blocks=*/3);
    EXPECT_EQ(cascaded, 3u);
    EXPECT_EQ(cacheRef(3), 0);
    EXPECT_EQ(cacheRef(4), 0);
    EXPECT_EQ(cacheRef(5), 0);
    EXPECT_EQ(reclaimed_.size(), 3u);
}

}  // namespace test
}  // namespace rtp_llm
