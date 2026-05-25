#pragma once

#include <mutex>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <functional>

#include "rtp_llm/cpp/utils/LRUCache.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/BlockRefCounter.h"  // UnifiedRefCounter (M03-PR3)

namespace rtp_llm {

class SharedBlockCache;  // fwd for ReleaseGuard

class SharedBlockCache {
public:
    // M03-PR2: dual-storage UnifiedCacheItem.
    //   * Legacy path: ``slots`` is the per-group vector indexed by group_id —
    //     unchanged so all current callers keep their semantics.
    //   * Unified path (when ``CacheConfig::super_block_layout.enabled``):
    //     ``slots`` is a length-1 vector with ``slots[0] = super_block_id``;
    //     read via ``superBlockId()``. ``pin_for_swa_tail`` is set by the SWA
    //     reference path (M03-PR4) and skipped during eviction. ``use_ref`` is
    //     bumped under ``mu_`` by ``matchUnified`` for every matched key and
    //     drained by ``ReleaseGuard::abandon()`` / explicit ``decUseRef`` at
    //     match-completion (lifecycle in M03-PR3 will move to UnifiedRefCounter).
    struct UnifiedCacheItem {
        CacheKeyType              cache_key;
        bool                      is_resident      = false;
        bool                      pin_for_swa_tail = false;  // A09 45-6 — SWA tail-veto
        int                       use_ref          = 0;      // matchUnified pin count
        std::vector<BlockIdxType> slots;

        // Unified-path accessor. Returns NULL_BLOCK_IDX if slots is empty.
        int superBlockId() const {
            return slots.empty() ? static_cast<int>(NULL_BLOCK_IDX) : static_cast<int>(slots[0]);
        }
    };

    struct EvictResult {
        std::vector<CacheKeyType>                                   evicted_keys;
        std::unordered_map<CacheKeyType, std::vector<BlockIdxType>> evicted_slots;
    };

    struct MatchResult {
        bool                      found = false;
        std::vector<BlockIdxType> group_blocks;
    };

    // ---- M03-PR2: unified TOCTOU-safe match types ----
    //
    // ReleaseGuard holds ONE entry per ``incUseRef`` bump performed during a
    // ``matchUnified`` walk (Panel-A item 1 / Fix 78 follow-up: a single-key
    // design leaked N-1 use_ref entries on multi-key matches because abandon
    // touched only the last key — the vector form drains every bump).
    //
    // Lifecycle:
    //   * ``commit()``  — the request has materialised; ownership of every
    //     bumped use_ref transfers to the request. Guard becomes dormant.
    //   * ``abandon()`` — caller failed downstream; iterate ``keys_`` in
    //     reverse and call ``decUseRef`` once per entry.
    //   * dtor — if neither commit nor abandon was called, abandon is invoked
    //     as a RAII safety net (early return / exception path).
    //
    // Copy is deleted (would double-decrement on dtor). Move is allowed so
    // the guard can be returned by value inside MatchResultUnified.
    class ReleaseGuard {
    public:
        ReleaseGuard() = default;
        explicit ReleaseGuard(SharedBlockCache* cache): cache_(cache) {}
        ~ReleaseGuard() {
            abandon();
        }

        ReleaseGuard(const ReleaseGuard&)            = delete;
        ReleaseGuard& operator=(const ReleaseGuard&) = delete;

        ReleaseGuard(ReleaseGuard&& other) noexcept:
            cache_(other.cache_), keys_(std::move(other.keys_)) {
            other.cache_ = nullptr;
            other.keys_.clear();
        }
        ReleaseGuard& operator=(ReleaseGuard&& other) noexcept {
            if (this != &other) {
                abandon();
                cache_       = other.cache_;
                keys_        = std::move(other.keys_);
                other.cache_ = nullptr;
                other.keys_.clear();
            }
            return *this;
        }

        // Record one use_ref bump under ``cache_->mu_`` (called by
        // matchUnified). Caller is responsible for the actual increment of
        // the item's use_ref counter — addKey only records the witness.
        void addKey(CacheKeyType k) {
            keys_.push_back(k);
        }

        // Caller promises to manage refs from here on; guard goes dormant.
        void commit() noexcept {
            keys_.clear();
            cache_ = nullptr;
        }

        // Drain every recorded bump. Iterates in reverse for symmetry with
        // bump order. Idempotent — repeat calls see an empty vector.
        void abandon() noexcept;

        size_t pendingBumpCount() const {
            return keys_.size();
        }

    private:
        SharedBlockCache*         cache_ = nullptr;
        std::vector<CacheKeyType> keys_;  // one entry per use_ref bump (Fix 78 follow-up)
    };

    struct MatchResultUnified {
        bool             found        = false;
        size_t           reuse_length = 0;  // matched_count * Tlog, or 0 on SWA tail-veto
        std::vector<int> super_block_ids;   // one S per matched key (in walk order)
        ReleaseGuard     release_guard;     // RAII over every use_ref bump
    };

    using LRUCacheType = LRUCache<CacheKeyType, UnifiedCacheItem>;

public:
    explicit SharedBlockCache(): lru_cache_(kCacheMaxCapacity) {}

    void init(int group_num, const std::vector<BlockPoolPtr>& group_pools);

    void put(CacheKeyType cache_key, const std::vector<BlockIdxType>& group_slots, bool is_resident);

    MatchResult match(CacheKeyType cache_key);

    BlockIdxType matchGroup(CacheKeyType cache_key, int group_id);

    EvictResult selectAndEvict(size_t min_blocks);

    size_t evictAndFree(size_t min_blocks);

    // ---- M03-PR2: unified-path entry points (default-OFF) ----
    //
    // Caller contract:
    //   * These methods are only invoked when
    //     ``CacheConfig::super_block_layout.enabled == true``. The legacy
    //     ``put`` / ``match`` / ``matchGroup`` / ``selectAndEvict`` / ``evictAndFree``
    //     paths above remain the steady-state code for non-unified configs.
    //   * Locking (see §3.0 of M03 plan):
    //       L1 = ``SharedBlockCache::mu_`` (held only inside these methods'
    //            select / snapshot phase).
    //       L1 is ALWAYS RELEASED before crossing into the per-pool
    //       ``BlockPool::blockCacheFree`` (which takes its own L3 lock pair).
    //   * Snapshot-then-cascade pattern (Fix 79) prevents the lock-drop /
    //     put-overflow race. The cascade tail issues one dec per snapshot
    //     entry unconditionally — CACHE refcounts are event-paired (each
    //     ``putUnified`` insertion bumps once; each cascade entry decs
    //     once). XR-C1: an earlier re-residency probe attempted to skip
    //     the dec when a racing put restored the displaced key at the same
    //     S; that skip un-paired the original bump and leaked CACHE / per-
    //     pool refs permanently. The probe has been removed.
    //
    // ``putUnified``:
    //   Inserts ``{cache_key, super_block_id}`` (length-1 slots vector).
    //   When the LRU overflows ``kCacheMaxCapacity``, surfaces the displaced
    //   item via the LRUCache overflow callback, releases ``mu_``, and
    //   unconditionally cascade-frees the displaced super_block_id's
    //   per-pool blocks + decs UnifiedRefCounter CACHE (event-paired with
    //   the original put's bump; see XR-C1 fix notes in .cc).
    void putUnified(CacheKeyType cache_key, int super_block_id, bool is_resident);

    // ``matchUnified``:
    //   Walks ``keys`` in order under ONE acquisition of ``mu_``. For every
    //   key that hits a resident entry, bumps the per-item ``use_ref`` and
    //   records the key in the returned ``ReleaseGuard``. Stops at the first
    //   miss / non-resident entry (truncation distinguishable via
    //   ``super_block_ids.size() < keys.size()``). If ``check_swa_tail`` is
    //   true and the matched tail is not SWA-valid, returns
    //   ``reuse_length=0`` per M03 §5 FULL_HIT vs KV_ONLY_HIT semantics.
    MatchResultUnified matchUnified(const std::vector<CacheKeyType>& keys,
                                    int                              tokens_per_block = 1,
                                    bool                             check_swa_tail   = false);

    // ``selectAndEvictUnified``:
    //   Builds a snapshot of evictable items under ``mu_``, erases them from
    //   the LRU, drops ``mu_``, then returns the snapshot. Items pinned by
    //   ``pin_for_swa_tail`` or with ``use_ref > 0`` are skipped (R10 / Fix 78).
    std::vector<UnifiedCacheItem> selectAndEvictUnified(size_t min_super_blocks);

    // ``evictAndFreeUnified``:
    //   Calls ``selectAndEvictUnified`` (mu_ dropped), then unconditionally
    //   cascade-frees each item's per-pool blocks + decs UnifiedRefCounter
    //   CACHE. CACHE is event-paired with putUnified, not residency-paired
    //   with LRU membership, so the snapshot dec is correct even if a
    //   racing putUnified has re-inserted the same key at the same S
    //   (XR-C1 fix). Returns the number of items actually cascaded.
    size_t evictAndFreeUnified(size_t min_super_blocks);

    // Decrement the per-item ``use_ref``. Called by ``ReleaseGuard::abandon``
    // and by explicit request-side ``decUseRef`` at match-completion. Bumps
    // are emitted by ``matchUnified`` under ``mu_``. Idempotent on missing
    // keys (treated as no-op rather than fail — the LRU may have been
    // evicted-then-re-evicted between bump and abandon under heavy churn).
    void decUseRef(CacheKeyType cache_key);

    // Inspect ``use_ref`` for a key. Returns 0 if absent. Test / diag only.
    int useRefCount(CacheKeyType cache_key) const;

    // Set / clear the SWA tail-pin flag on a resident item. No-op if absent.
    void setSwaTailPin(CacheKeyType cache_key, bool pinned);

    std::optional<UnifiedCacheItem> remove(CacheKeyType cache_key);

    bool contains(CacheKeyType cache_key) const;

    bool empty() const;

    size_t size() const;

    std::vector<CacheKeyType> allCacheKeys() const;

    int64_t version() const;

    // ---- M03-PR3: UnifiedRefCounter wiring (default-OFF) ----
    //
    // Both setters are no-ops on the legacy path — callers MUST only invoke
    // them when ``CacheConfig::super_block_layout.enabled == true``. When
    // ``unified_ref_counter_`` is non-null, the unified entry points
    // (``putUnified`` / ``matchUnified`` / ``evictAndFreeUnified`` /
    // ``decUseRef``) issue the dual-write contract documented on
    // ``UnifiedRefCounter`` (BlockRefCounter.h): every primary bump/dec is
    // paired with the matching legacy per-pool ``BlockPool::*Reference / *Free``
    // call at the same critical section. When ``unified_ref_counter_`` is
    // null the unified entry points fall back to the legacy use_ref-on-item
    // path introduced by PR-2 (preserved as a fallback for tests that
    // construct ``SharedBlockCache`` without an allocator).
    //
    // ``super_block_reclaim_callback_`` is invoked once per fully-reclaimed
    // super-block (``UnifiedRefCounter::isZero(S) == true`` after a CACHE
    // dec) so the owning ``SuperBlockFreeList`` can push S back. The
    // callback runs OUTSIDE the cache ``mu_`` (lock-order §3.0 — L1 must be
    // released before the allocator's L2). When unset, the super-block is
    // not pushed back (legacy fallback; M01 ``unifiedFree`` handles request-
    // side reclaim).
    void setUnifiedRefCounter(UnifiedRefCounter* counter) {
        unified_ref_counter_ = counter;
    }
    void setSuperBlockReclaimCallback(std::function<void(int)> cb) {
        super_block_reclaim_callback_ = std::move(cb);
    }

    // Diagnostic accessor used by tests / metrics (read-only).
    UnifiedRefCounter* unifiedRefCounter() const {
        return unified_ref_counter_;
    }

    // DEFEND1 HIGH-9 (R6 DEV-γ): LRU mirror desync invariant — public
    // diagnostic. Acquires ``mu_`` internally so external callers (tests /
    // post-condition probes) can invoke directly. See
    // ``assertMirrorInvariantUnlocked_CHECK`` for the actual check and the
    // full invariant rationale.
    void assertMirrorInvariant_CHECK() const;

private:
    // XR-C1: shared cascade body for ``putUnified``'s overflow tail and
    // ``evictAndFreeUnified``'s phase B. Drops one CACHE/per-pool ref per
    // snapshot entry unconditionally — event-paired with the put that
    // produced the entry.
    //
    // Pre-condition: ``mu_`` (L1) MUST NOT be held. The body invokes
    // ``UnifiedRefCounter::dec``/``isZero`` (Lcr), ``BlockPool::blockCacheFree``
    // (L3), and ``super_block_reclaim_callback_`` (which targets L2 on
    // ``SuperBlockFreeList``). L1 itself is NOT a leaf — it nests Lcr
    // and L3 in ``putUnified``/``matchUnified`` for atomic bump+insert
    // — but the cascade path runs outside L1 specifically to keep the
    // L1 → L2 edge from forming via the reclaim callback (forbidden by
    // §3.0 lock-order).
    void cascadeUnifiedSnapshot(const std::vector<UnifiedCacheItem>& snapshot, size_t* cascaded_out);

    // DEFEND1 HIGH-9 (R6 DEV-γ → R8 always-on per user directive
    // "不要藏 bug 在 release 后面") — mirror invariant body, called with
    // ``mu_`` already held by the internal put/match/dec/evict paths.
    // R8 upgrade: removed the prior ``#ifndef NDEBUG`` gate. The CHECK now
    // throws std::runtime_error in BOTH debug and release builds when the
    // dual-write contract is bypassed, so the operator sees a hard failure
    // instead of silent cache corruption. The invariant:
    //
    //   ``count(LRU items with use_ref > 0) <= UnifiedRefCounter::useRefMapSize()``
    //
    // Strict equality only holds when no ``remove()`` orphans exist —
    // ``remove()`` evicts items unconditionally and may drop an item whose
    // use_ref > 0, leaving an orphan entry in the counter that decUseRef
    // later tolerates as a no-op. Counter > LRU is benign; the failure mode
    // we catch is LRU > counter, which means someone bumped item.use_ref
    // without paying the counter bump — a silent dual-write bypass.
    void assertMirrorInvariantUnlocked_CHECK() const;

    static const size_t kCacheMaxCapacity = 10000000;

    LRUCacheType       lru_cache_;
    mutable std::mutex mu_;
    int64_t            version_{0};

    int                       group_num_ = 0;
    std::vector<BlockPoolPtr> group_pools_;

    // M03-PR3: wired by HybridPoolKVCacheAllocator::doInit() when
    // ``config_.super_block_layout.enabled``. Owned by the allocator; the
    // pointer stays valid for the lifetime of the cache.
    UnifiedRefCounter*       unified_ref_counter_ = nullptr;
    std::function<void(int)> super_block_reclaim_callback_;
};

using SharedBlockCachePtr = std::shared_ptr<SharedBlockCache>;

}  // namespace rtp_llm
