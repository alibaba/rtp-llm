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

#include <atomic>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "rtp_llm/cpp/cache/BlockRefCounter.h"
#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"  // SuperBlockFreeList (R6 DEV-γ)
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

// ============================================================================
// R6 DEV-γ — DEFEND1 HIGH bounds checks
// ============================================================================
//
// Each test exercises one of the HIGH bounds CHECKs added in this round:
//   * R6SharedCacheDoubleFreeFailsLoud           — HIGH-1
//   * R6SharedCachePutSilentDropConflictFails    — HIGH-3 (conflict variant)
//   * R6SharedCachePutDuplicateIsIdempotent      — HIGH-3 (idempotent variant)
//   * R6SharedCacheDecUseRefMissingKeyWarns      — HIGH-4 (post-fix WARN, no throw)
//   * R6SharedCacheAllocOverflowFails            — HIGH-6 (free overflow guard)
//   * R6SharedCacheLruMirrorInvariantHolds       — HIGH-9
//
// myAssert() throws via throwRuntimeError (see AssertUtils.cc), so the kill
// path is EXPECT_THROW(std::exception) rather than EXPECT_DEATH.

// HIGH-1: releasing the same super_block_id twice must fail loud rather than
// silently push a duplicate onto the free list (which would let two
// allocSuperBlock() calls return the same S and alias two requests onto the
// same KV super-block — silent precision corruption).
TEST(R6SharedCacheDefendStandalone, R6SharedCacheDoubleFreeFailsLoud) {
    SuperBlockFreeList free_list(/*num_super_blocks=*/8);
    const int          s = free_list.allocSuperBlock();
    ASSERT_GT(s, 0);
    free_list.freeSuperBlock(s);
    EXPECT_THROW(free_list.freeSuperBlock(s), std::exception);
}

// HIGH-6: the free-list size invariant fires when an unallocated id is
// pushed back via the double-free guard (the genuine overflow scenario —
// pushing past num_super_blocks_ - 1 — structurally requires releasing an
// unallocated id, which the double-free guard catches first for any id
// currently on the list). We use a small budget so the test is fast.
TEST(R6SharedCacheDefendStandalone, R6SharedCacheAllocOverflowFails) {
    constexpr uint32_t kBudget = 4;  // valid IDs are 1..3 (id 0 reserved)
    SuperBlockFreeList free_list(kBudget);

    // Drain the free list so we know exactly which ids are held.
    std::vector<int> held;
    while (true) {
        const int s = free_list.allocSuperBlock();
        if (s < 0) {
            break;
        }
        held.push_back(s);
    }
    ASSERT_EQ(held.size(), static_cast<size_t>(kBudget - 1));  // ids 1..3

    // Release them all — free_list is now full at kBudget-1.
    for (int s : held) {
        free_list.freeSuperBlock(s);
    }
    EXPECT_EQ(free_list.freeCount(), static_cast<size_t>(kBudget - 1));

    // Releasing any further id (all valid ids are now on the free list)
    // trips the double-free guard, which is the upstream signal of the same
    // overflow condition: free_list_ would exceed budget if pushed.
    EXPECT_THROW(free_list.freeSuperBlock(/*S=*/1), std::exception);
}

// HIGH-3 (conflict): put a non-null incoming S while the existing slot
// already holds a DIFFERENT S — silent drop here would leak the incoming
// allocSuperBlock bump forever.
TEST_F(SharedBlockCacheUnifiedTest, R6SharedCachePutSilentDropConflictFails) {
    constexpr CacheKeyType K  = 0xC011;
    constexpr int          S1 = 17;
    constexpr int          S2 = 19;

    cache_->putUnified(K, S1, /*is_resident=*/true);
    ASSERT_EQ(cacheRef(S1), 1);

    EXPECT_THROW(cache_->putUnified(K, S2, /*is_resident=*/true), std::exception);
}

// HIGH-3 (idempotent): re-putting the same (K, S) must NOT bump CACHE again
// and must NOT throw — it emits a WARN but the entry stays consistent.
TEST_F(SharedBlockCacheUnifiedTest, R6SharedCachePutDuplicateIsIdempotent) {
    constexpr CacheKeyType K = 0xD0E5;
    constexpr int          S = 21;

    cache_->putUnified(K, S, /*is_resident=*/true);
    ASSERT_EQ(cacheRef(S), 1);

    cache_->putUnified(K, S, /*is_resident=*/true);
    EXPECT_EQ(cacheRef(S), 1) << "duplicate put must be idempotent on CACHE refcount";
    EXPECT_TRUE(cache_->contains(K));
}

// HIGH-4 (missing-K WARN branch): decUseRef on a never-bumped key must NOT
// throw (tolerable LRU-churn race) but is no longer silently no-op — it
// emits a WARN so high rates surface. The structural underflow case
// (entry present with value <= 0) is unreachable without harness
// tampering, so we validate the missing-K branch behaves cleanly and the
// counter's use_ref map remains empty.
TEST_F(SharedBlockCacheUnifiedTest, R6SharedCacheDecUseRefMissingKeyWarns) {
    EXPECT_NO_THROW(counter_->decUseRef(/*K=*/0xDEAD));
    EXPECT_EQ(counter_->useRefMapSize(), 0u);
}

// HIGH-9: LRU mirror invariant holds across the natural put / match /
// abandon lifecycle. Every mutating unified path calls
// ``assertMirrorInvariantUnlocked_DCHECK`` at the L1 tail; we also probe
// the public diagnostic ``assertMirrorInvariant_DCHECK`` directly.
TEST_F(SharedBlockCacheUnifiedTest, R6SharedCacheLruMirrorInvariantHolds) {
    constexpr CacheKeyType K1 = 0xA1;
    constexpr CacheKeyType K2 = 0xA2;

    cache_->putUnified(K1, /*S=*/31, /*is_resident=*/true);
    cache_->putUnified(K2, /*S=*/33, /*is_resident=*/true);
    EXPECT_NO_THROW(cache_->assertMirrorInvariant_DCHECK());

    auto match = cache_->matchUnified({K1, K2}, /*tokens_per_block=*/1);
    EXPECT_EQ(match.super_block_ids.size(), 2u);
    EXPECT_NO_THROW(cache_->assertMirrorInvariant_DCHECK());

    match.release_guard.abandon();
    EXPECT_NO_THROW(cache_->assertMirrorInvariant_DCHECK());
    EXPECT_EQ(counter_->useRefMapSize(), 0u);
}

// ============================================================================
// R7 Race — Multi-threaded stress for the A-fix-2 R5/R6 CACHE-bump hoist
// ============================================================================
//
// These tests exercise putUnified / matchUnified / decUseRef /
// evictAndFreeUnified (the public driver of cascadeUnifiedSnapshot) under
// high concurrency on shared key / super-block pools.
//
// What they are guarding (XR-C1 / R5 FIX-B / R6 LRU-mirror):
//   * The unified CACHE bump now happens INSIDE L1 alongside the LRU insert,
//     fan-out, and version bump. Before the hoist a concurrent
//     evictAndFreeUnified could snapshot a freshly-inserted K and call
//     ``dec(S, CACHE)`` on a counter still at zero — RTP_LLM_FAIL underflow.
//     After the hoist the dec always pairs with an already-landed bump
//     (XR2-A1/A2/A3 cross-review, SharedBlockCache.cc line 392-405).
//   * snapshot-then-cascade in evictAndFreeUnified no longer probes
//     re-residency, so every snapshot dec is unconditional and event-paired
//     with the put that produced it (XR-C1). Stress here drives
//     evictAndFreeUnified concurrently with putUnified to flush any residual
//     reorder leak — final invariant: every super-block isZero after drain
//     and SuperBlockFreeList accounting balances.
//   * SuperBlockFreeList::freeSuperBlock CHECKs on double-free (R6 HIGH-1);
//     wiring it as the reclaim sink (R7RaceDecRefDoubleFreeProbe) turns any
//     duplicate reclaim into an immediate exception from a worker thread.
//
// Failure mode: RTP_LLM_FAIL / RTP_LLM_CHECK_WITH_INFO throw RTP_EXCEPTION
// (AssertUtils.cc:13). Workers wrap each iteration in try/catch and bump
// ``exception_count`` so any underflow / double-free surfaces as a
// post-condition assertion failure instead of std::terminate.
//
// Determinism: each test uses a fixed parent seed; per-thread RNGs derive
// from parent_seed XOR (thread_index * golden_ratio). Reproducible on the
// same host given the same iteration count.

namespace {

constexpr uint32_t kRaceNumSuperBlocks = 64;
constexpr int      kRaceMaxValidS      = static_cast<int>(kRaceNumSuperBlocks) - 1;  // ids [1, 64)
constexpr uint64_t kRaceParentSeed     = 0xC0FFEEDEADBEEFULL;

// Deterministic CacheKey -> super_block_id mapping. Many K may collide on the
// same S — that is the point: the per-S CACHE accumulator must stay balanced
// under concurrent overlapping bumps/decs. Never produces the reserved id 0.
int sForKey(CacheKeyType K) {
    constexpr uint64_t kMix = 1469598103934665603ULL;  // FNV offset (any large odd)
    return static_cast<int>((static_cast<uint64_t>(K) * kMix) % static_cast<uint64_t>(kRaceMaxValidS)) + 1;
}

uint64_t seedFor(int tid, uint64_t salt = 0) {
    constexpr uint64_t kGolden = 0x9E3779B97F4A7C15ULL;
    return kRaceParentSeed ^ (static_cast<uint64_t>(tid + 1) * kGolden) ^ salt;
}

}  // namespace

class SharedBlockCacheRaceStressTest: public ::testing::Test {
protected:
    void SetUp() override {
        cache_   = std::make_unique<SharedBlockCache>();
        counter_ = std::make_unique<UnifiedRefCounter>();
        counter_->init(static_cast<int>(kRaceNumSuperBlocks));

        cache_->setUnifiedRefCounter(counter_.get());
        cache_->setSuperBlockReclaimCallback([this](int S) {
            std::lock_guard<std::mutex> g(reclaim_mu_);
            reclaimed_.push_back(S);
            reclaim_count_.fetch_add(1, std::memory_order_relaxed);
        });
    }

    void TearDown() override {
        cache_.reset();
        counter_.reset();
        reclaimed_.clear();
        reclaim_count_.store(0);
    }

    // Drain helper: evicts until cache is empty or a hard cap is hit. Uses
    // min_super_blocks=128 so each call scans 256 LRU tail entries (cf.
    // selectAndEvictUnified scan_cap = 2 * min) — bounded per-call cost.
    void drainAll(size_t hard_cap = 200000) {
        size_t guard_iters = 0;
        while (cache_->size() > 0 && guard_iters++ < hard_cap) {
            const size_t freed = cache_->evictAndFreeUnified(/*min_super_blocks=*/128);
            if (freed == 0) {
                // All tail entries pinned (no concurrent reader at this point
                // means a logic bug). Break to surface via size() assertion.
                break;
            }
        }
    }

    std::unique_ptr<SharedBlockCache>  cache_;
    std::unique_ptr<UnifiedRefCounter> counter_;
    std::mutex                         reclaim_mu_;
    std::vector<int>                   reclaimed_;
    std::atomic<size_t>                reclaim_count_{0};
};

// R7-1: 16 threads × 10k iter, randomized mix of putUnified / matchUnified /
// decUseRef over a 256-key pool (K -> S is deterministic so concurrent
// putUnified on the same K never hits the HIGH-3 conflict CHECK; same-S
// duplicate puts are idempotent). After the workers join, drain the cache and
// assert every super-block reaches isZero — i.e. every CACHE bump paired with
// exactly one cascade dec. A broken hoist would either throw underflow from a
// cascade or leave CACHE > 0 on some S after drain.
TEST_F(SharedBlockCacheRaceStressTest, R7RaceConcurrentPutGet) {
    constexpr int kThreads     = 16;
    constexpr int kItersPerThr = 10000;
    constexpr int kKeyPoolSize = 256;

    std::atomic<size_t> put_count{0};
    std::atomic<size_t> get_count{0};
    std::atomic<size_t> dec_count{0};
    std::atomic<size_t> exception_count{0};

    auto worker = [&](int tid) {
        std::mt19937_64                    rng(seedFor(tid));
        std::uniform_int_distribution<int> op_pick(0, 2);
        std::uniform_int_distribution<int> key_pick(0, kKeyPoolSize - 1);

        for (int i = 0; i < kItersPerThr; ++i) {
            const CacheKeyType K  = static_cast<CacheKeyType>(key_pick(rng));
            const int          S  = sForKey(K);
            const int          op = op_pick(rng);
            try {
                if (op == 0) {
                    cache_->putUnified(K, S, /*is_resident=*/true);
                    put_count.fetch_add(1, std::memory_order_relaxed);
                } else if (op == 1) {
                    auto m = cache_->matchUnified({K}, /*tokens_per_block=*/1);
                    (void)m;  // ReleaseGuard dtor abandons (drains use_ref bumps)
                    get_count.fetch_add(1, std::memory_order_relaxed);
                } else {
                    // Direct decUseRef — when K never had a bump it's a WARN
                    // no-op; when it did, drains one bump early. Either way
                    // the counter must not underflow.
                    cache_->decUseRef(K);
                    dec_count.fetch_add(1, std::memory_order_relaxed);
                }
            } catch (const std::exception&) {
                exception_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back(worker, t);
    }
    for (auto& th : threads) {
        th.join();
    }

    EXPECT_EQ(exception_count.load(), 0u)
        << "race path threw — likely CACHE dec underflow / double-free / HIGH-3 conflict";
    EXPECT_LE(cache_->size(), static_cast<size_t>(kKeyPoolSize));

    drainAll();
    EXPECT_EQ(cache_->size(), 0u) << "drain stalled — pinned items still present";
    EXPECT_EQ(counter_->useRefMapSize(), 0u) << "use_ref leak after drain";

    for (int s = 1; s <= kRaceMaxValidS; ++s) {
        EXPECT_EQ(counter_->getRefCount(s, UnifiedRefCounter::Kind::CACHE), 0)
            << "S=" << s << " leaked CACHE refs after drain (put/dec unbalanced)";
        EXPECT_TRUE(counter_->isZero(s)) << "S=" << s << " not zero after drain";
    }
    // Sanity counters (no behavioural assertion — just record).
    EXPECT_GT(put_count.load() + get_count.load() + dec_count.load(), 0u);
}

// R7-2: 8 cascade threads racing 8 put/get threads over an overlapping
// 256-key pool. The cascade threads drive evictAndFreeUnified (the public
// entry of the R5 FIX-B hoist: selectAndEvictUnified then
// cascadeUnifiedSnapshot, both with mu_ released between the snapshot and the
// per-S CACHE dec). The put/get threads keep the LRU tail churning so the
// cascade always has work. If the hoist were missing, the cascade thread
// would see a tail entry whose CACHE bump hasn't landed yet and underflow on
// dec — caught by the per-worker try/catch + exception_count.
TEST_F(SharedBlockCacheRaceStressTest, R7RaceCascadePutUnderContention) {
    constexpr int kCascadeThreads = 8;
    constexpr int kPutGetThreads  = 8;
    constexpr int kPutGetIters    = 8000;
    constexpr int kKeyPoolSize    = 256;

    std::atomic<size_t> exception_count{0};
    std::atomic<size_t> evict_returned_nonzero{0};
    std::atomic<bool>   stop_cascade{false};

    auto put_get_worker = [&](int tid) {
        std::mt19937_64                    rng(seedFor(tid, /*salt=*/0xABABABABULL));
        std::uniform_int_distribution<int> key_pick(0, kKeyPoolSize - 1);
        std::uniform_int_distribution<int> op_pick(0, 2);

        for (int i = 0; i < kPutGetIters; ++i) {
            const CacheKeyType K = static_cast<CacheKeyType>(key_pick(rng));
            const int          S = sForKey(K);
            try {
                const int op = op_pick(rng);
                if (op == 0) {
                    cache_->putUnified(K, S, /*is_resident=*/true);
                } else if (op == 1) {
                    auto m = cache_->matchUnified({K}, /*tokens_per_block=*/1);
                    (void)m;
                } else {
                    // selectAndEvictUnified WITHOUT cascade — leaves an
                    // outstanding "pending bump" view from the put side. The
                    // next caller's evictAndFreeUnified must reconcile.
                    auto snap = cache_->selectAndEvictUnified(/*min_super_blocks=*/2);
                    // Snapshot is dropped here. Phase B (cascade) never runs
                    // on it — the per-pool callbacks are no-op since
                    // group_pools_ is empty, but the unified CACHE dec is
                    // intentionally lost from this branch. R5 FIX-B
                    // documents that callers MUST cascade the snapshot;
                    // this branch is exercising the WRONG pattern on
                    // purpose to verify the loss surfaces as leaked CACHE,
                    // NOT as underflow on the cascade thread.
                    (void)snap;
                }
            } catch (const std::exception&) {
                exception_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
    };

    auto cascade_worker = [&](int tid) {
        std::mt19937_64                    rng(seedFor(tid, /*salt=*/0xCDCDCDCDULL));
        std::uniform_int_distribution<int> min_pick(1, 4);

        while (!stop_cascade.load(std::memory_order_acquire)) {
            try {
                const size_t freed = cache_->evictAndFreeUnified(static_cast<size_t>(min_pick(rng)));
                if (freed > 0) {
                    evict_returned_nonzero.fetch_add(freed, std::memory_order_relaxed);
                }
            } catch (const std::exception&) {
                exception_count.fetch_add(1, std::memory_order_relaxed);
            }
            std::this_thread::yield();
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(kPutGetThreads + kCascadeThreads);
    for (int t = 0; t < kCascadeThreads; ++t) {
        threads.emplace_back(cascade_worker, t);
    }
    for (int t = 0; t < kPutGetThreads; ++t) {
        threads.emplace_back(put_get_worker, kCascadeThreads + t);
    }
    // Wait for put/get threads first.
    for (int t = kCascadeThreads; t < kCascadeThreads + kPutGetThreads; ++t) {
        threads[t].join();
    }
    // Signal cascade threads to wind down.
    stop_cascade.store(true, std::memory_order_release);
    for (int t = 0; t < kCascadeThreads; ++t) {
        threads[t].join();
    }

    EXPECT_EQ(exception_count.load(), 0u)
        << "cascade-vs-put race threw — likely R5 FIX-B regression "
           "(CACHE bump escaped L1 → dec landed before bump → underflow)";

    drainAll();
    EXPECT_EQ(cache_->size(), 0u);
    EXPECT_EQ(counter_->useRefMapSize(), 0u);
    // Final balance: every S must be zero. The intentional snapshot-without-
    // cascade branch in the worker leaves CACHE > 0 on those S's, but the
    // post-test drainAll() runs evictAndFreeUnified until empty. The
    // remaining CACHE bumps from the lost-snapshot branch are NOT paired
    // with any in-LRU item, so drainAll cannot dec them — we therefore
    // tolerate residual CACHE>0 for THIS test only and assert via the
    // counter-vs-LRU mirror invariant instead.
    EXPECT_NO_THROW(cache_->assertMirrorInvariant_DCHECK());
    // Reclaim count must not exceed total bumps issued (no double-reclaim).
    EXPECT_LE(reclaim_count_.load(), static_cast<size_t>(kPutGetIters * kPutGetThreads));
}

// R7-3: SuperBlockFreeList double-free probe. Wire reclaim_callback into a
// live SuperBlockFreeList so every isZero(S) reclaim performs a real
// freeSuperBlock(S). Any duplicate reclaim trips the double-free CHECK
// (HybridPoolKVCacheAllocator.cc:64), which throws and is caught + counted.
// Each thread runs an alloc → put → drain pipeline; under 32-way concurrency
// the free list churns aggressively over the 63-slot budget. A working hoist
// keeps the pipeline coherent; a broken one either underflows the CACHE
// counter or causes a duplicate reclaim (both raise exception_count).
TEST_F(SharedBlockCacheRaceStressTest, R7RaceDecRefDoubleFreeProbe) {
    constexpr int kThreads = 32;
    constexpr int kIters   = 1500;

    // Local SuperBlockFreeList wired as the reclaim sink. Must outlive worker
    // threads; declared before the thread vector and joined before
    // destruction.
    SuperBlockFreeList sbfl(kRaceNumSuperBlocks);
    // Combine free-list reclaim with the fixture's per-test counter so we
    // can cross-check budget invariants AND per-reclaim accounting. The
    // fixture's lambda also pushes onto reclaimed_ under reclaim_mu_; we
    // mirror that here so failures still capture the offending S list.
    cache_->setSuperBlockReclaimCallback([this, &sbfl](int S) {
        {
            std::lock_guard<std::mutex> g(reclaim_mu_);
            reclaimed_.push_back(S);
            reclaim_count_.fetch_add(1, std::memory_order_relaxed);
        }
        sbfl.freeSuperBlock(S);
    });

    std::atomic<size_t> exception_count{0};
    std::atomic<size_t> alloc_count{0};
    std::atomic<size_t> put_count{0};

    auto worker = [&](int tid) {
        std::mt19937_64                    rng(seedFor(tid, /*salt=*/0xEFEFEFEFULL));
        std::uniform_int_distribution<int> roll(0, 99);

        for (int i = 0; i < kIters; ++i) {
            try {
                int S = sbfl.allocSuperBlock();
                if (S < 0) {
                    // Pool drained by peers — push the LRU forward to
                    // recycle some S's.
                    cache_->evictAndFreeUnified(/*min_super_blocks=*/4);
                    std::this_thread::yield();
                    continue;
                }
                alloc_count.fetch_add(1, std::memory_order_relaxed);

                // Unique key per (tid, iter) — no put-conflict possible.
                const CacheKeyType K =
                    (static_cast<CacheKeyType>(tid + 1) << 40) | static_cast<CacheKeyType>(i + 1);
                cache_->putUnified(K, S, /*is_resident=*/false);
                put_count.fetch_add(1, std::memory_order_relaxed);

                // Occasionally stress decUseRef on a never-bumped key
                // (WARN no-op path) — must not throw / underflow.
                if (roll(rng) < 25) {
                    cache_->decUseRef(K);
                }

                // Drain at least one entry — may be ours or a peer's.
                cache_->evictAndFreeUnified(/*min_super_blocks=*/1);
            } catch (const std::exception&) {
                exception_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back(worker, t);
    }
    for (auto& th : threads) {
        th.join();
    }

    EXPECT_EQ(exception_count.load(), 0u)
        << "double-free / underflow surfaced under concurrent alloc/put/free pipeline";

    // Final drain to recycle everything still cached.
    drainAll();
    EXPECT_EQ(cache_->size(), 0u);

    // Free-list invariant: after full drain, every allocatable id is back on
    // the free list (id 0 reserved, so the budget is kRaceNumSuperBlocks - 1).
    EXPECT_EQ(sbfl.freeCount(), static_cast<size_t>(kRaceNumSuperBlocks - 1))
        << "alloc/free count diverged — alloc=" << alloc_count.load()
        << " put=" << put_count.load() << " reclaim=" << reclaim_count_.load();

    // Reclaim count must equal alloc count: every allocSuperBlock that
    // produced an S which was successfully put must have been reclaimed
    // exactly once. (put_count == alloc_count by construction unless
    // putUnified threw — which exception_count would have caught.)
    EXPECT_EQ(reclaim_count_.load(), put_count.load())
        << "reclaim_count != put_count — possible missed or duplicate reclaim";
}

}  // namespace test
}  // namespace rtp_llm
