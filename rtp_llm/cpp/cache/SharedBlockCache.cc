#include "rtp_llm/cpp/cache/SharedBlockCache.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {

void SharedBlockCache::init(int group_num, const std::vector<BlockPoolPtr>& group_pools) {
    std::lock_guard<std::mutex> lock(mu_);
    RTP_LLM_CHECK_WITH_INFO(static_cast<int>(group_pools.size()) == group_num,
                            "group_pools size %zu != group_num %d",
                            group_pools.size(),
                            group_num);
    group_num_   = group_num;
    group_pools_ = group_pools;
}

void SharedBlockCache::put(CacheKeyType cache_key, const std::vector<BlockIdxType>& group_slots, bool is_resident) {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(mu_);

    if (lru_cache_.contains(cache_key)) {
        auto [success, existing_item] = lru_cache_.get(cache_key);
        if (success) {
            bool updated = false;
            for (size_t gid = 0; gid < group_slots.size(); ++gid) {
                if (isNullBlockIdx(group_slots[gid])) {
                    continue;
                }
                if (gid >= existing_item.slots.size()) {
                    existing_item.slots.resize(gid + 1, NULL_BLOCK_IDX);
                }
                if (isNullBlockIdx(existing_item.slots[gid])) {
                    existing_item.slots[gid] = group_slots[gid];
                    updated                  = true;
                    if (static_cast<int>(gid) < group_num_) {
                        group_pools_[gid]->blockCacheReference(group_slots[gid]);
                    }
                }
            }
            if (updated) {
                lru_cache_.put(cache_key, existing_item);
                ++version_;
            }
        }
        return;
    }

    UnifiedCacheItem item;
    item.cache_key   = cache_key;
    item.is_resident = is_resident;
    item.slots       = group_slots;

    lru_cache_.put(cache_key, item);
    ++version_;

    for (int gid = 0; gid < static_cast<int>(group_slots.size()) && gid < group_num_; ++gid) {
        if (!isNullBlockIdx(group_slots[gid])) {
            group_pools_[gid]->blockCacheReference(group_slots[gid]);
        }
    }
}

SharedBlockCache::MatchResult SharedBlockCache::match(CacheKeyType cache_key) {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(mu_);

    auto [success, item] = lru_cache_.get(cache_key);
    if (!success) {
        return {false, {}};
    }
    return {true, item.slots};
}

BlockIdxType SharedBlockCache::matchGroup(CacheKeyType cache_key, int group_id) {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(mu_);

    auto [success, item] = lru_cache_.get(cache_key);
    if (!success) {
        return NULL_BLOCK_IDX;
    }
    if (group_id < 0 || static_cast<size_t>(group_id) >= item.slots.size()) {
        return NULL_BLOCK_IDX;
    }
    return item.slots[group_id];
}

SharedBlockCache::EvictResult SharedBlockCache::selectAndEvict(size_t min_blocks) {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(mu_);

    EvictResult result;
    if (lru_cache_.empty() || min_blocks == 0) {
        return result;
    }

    std::unordered_set<CacheKeyType> resident_keys;
    for (const auto& [key, item] : lru_cache_.items()) {
        if (item.is_resident) {
            resident_keys.insert(item.cache_key);
        }
    }

    std::vector<CacheKeyType> lru_keys;
    for (auto it = lru_cache_.items().rbegin(); it != lru_cache_.items().rend(); ++it) {
        const auto& item = it->second;
        if (item.is_resident || resident_keys.count(item.cache_key)) {
            continue;
        }
        lru_keys.push_back(item.cache_key);
    }

    size_t selected_blocks = 0;
    for (const auto cache_key : lru_keys) {
        UnifiedCacheItem removed_item;
        if (!lru_cache_.remove(cache_key, &removed_item)) {
            continue;
        }

        result.evicted_keys.push_back(cache_key);
        result.evicted_slots[cache_key] = removed_item.slots;

        for (const auto& slot : removed_item.slots) {
            if (!isNullBlockIdx(slot)) {
                selected_blocks++;
            }
        }
        if (selected_blocks >= min_blocks) {
            break;
        }
    }

    return result;
}

size_t SharedBlockCache::evictAndFree(size_t min_blocks) {
    RTP_LLM_PROFILE_FUNCTION();

    auto evict_result = selectAndEvict(min_blocks);
    if (evict_result.evicted_keys.empty()) {
        return 0;
    }

    size_t freed = 0;
    for (size_t i = 0; i < evict_result.evicted_keys.size(); ++i) {
        const auto  cache_key = evict_result.evicted_keys[i];
        const auto& slots     = evict_result.evicted_slots.at(cache_key);

        for (int gid = 0; gid < static_cast<int>(slots.size()) && gid < group_num_; ++gid) {
            if (!isNullBlockIdx(slots[gid])) {
                group_pools_[gid]->blockCacheFree(slots[gid]);
                freed++;
            }
        }
    }
    return freed;
}

std::optional<SharedBlockCache::UnifiedCacheItem> SharedBlockCache::remove(CacheKeyType cache_key) {
    std::lock_guard<std::mutex> lock(mu_);

    UnifiedCacheItem removed_item;
    if (!lru_cache_.remove(cache_key, &removed_item)) {
        return std::nullopt;
    }
    return removed_item;
}

bool SharedBlockCache::contains(CacheKeyType cache_key) const {
    std::lock_guard<std::mutex> lock(mu_);
    return lru_cache_.contains(cache_key);
}

bool SharedBlockCache::empty() const {
    std::lock_guard<std::mutex> lock(mu_);
    return lru_cache_.empty();
}

size_t SharedBlockCache::size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return lru_cache_.size();
}

std::vector<CacheKeyType> SharedBlockCache::allCacheKeys() const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<CacheKeyType>   keys;
    keys.reserve(lru_cache_.size());
    for (const auto& [key, item] : lru_cache_.items()) {
        keys.push_back(key);
    }
    return keys;
}

int64_t SharedBlockCache::version() const {
    std::lock_guard<std::mutex> lock(mu_);
    return version_;
}

// ============================================================================
// M03-PR2: unified-path methods (default-OFF, opt-in via super_block_layout)
// ============================================================================

void SharedBlockCache::ReleaseGuard::abandon() noexcept {
    if (!cache_ || keys_.empty()) {
        keys_.clear();
        cache_ = nullptr;
        return;
    }
    // Drain in reverse for symmetry with bump order. decUseRef takes mu_
    // itself, so the guard MUST NOT be invoked while the caller already
    // holds SharedBlockCache::mu_ (Panel-A item 1 contract).
    for (auto it = keys_.rbegin(); it != keys_.rend(); ++it) {
        cache_->decUseRef(*it);
    }
    keys_.clear();
    cache_ = nullptr;
}

void SharedBlockCache::decUseRef(CacheKeyType cache_key) {
    // M03-PR3 dual-write: when UnifiedRefCounter is wired, delegate the
    // primary state to it. The per-item ``use_ref`` mirror is still
    // maintained (legacy fallback for tests / non-unified paths).
    //
    // R9 race fix (user directive "不要藏 bug 在 release 后面"): both writes
    // MUST happen atomically under L1, in the order (LRU dec → counter dec).
    // The original code did counter dec OUTSIDE L1 first, then LRU dec inside
    // L1, which opened a transient `lru_pinned > counter_pinned` window any
    // concurrent put/match/dec/evict observer could see. Under R7Race
    // contention this triggers the always-on mirror CHECK
    // (assertMirrorInvariantUnlocked_CHECK).
    //
    // Lock order: L1 → Lcr is the established nesting on the put/match
    // sites (see header doc for cascadeUnifiedSnapshot), so taking the
    // counter's mu_ while holding our mu_ here is consistent with that
    // direction. No L1 → Lcr cycle is created.
    //
    // Dec order: LRU FIRST, then counter, so the invariant
    // `lru_pinned ≤ counter_pinned` only ever observes states where the
    // ratio drops monotonically (transiently lru < counter ≡ benign orphan,
    // never lru > counter ≡ bypass).
    std::lock_guard<std::mutex> lock(mu_);
    auto [success, item] = lru_cache_.get(cache_key);
    if (!success) {
        // The item was evicted (e.g. via remove() or a cascade) after the
        // bump landed. Treat as no-op rather than fail — heavy churn can
        // legitimately race the abandon path. The bump is already gone with
        // the item; no leak.
        //
        // Counter still needs the dec in case remove() didn't drop the
        // counter key (decUseRef on UnifiedRefCounter tolerates missing
        // keys per R6 DEV-γ HIGH-4: missing → WARN, not CHECK fail).
        if (unified_ref_counter_) {
            unified_ref_counter_->decUseRef(cache_key);
        }
        return;
    }
    if (item.use_ref <= 0) {
        // Defensive: should not happen if bump/dec are paired correctly. Log
        // and clamp rather than fail because this is dual-storage code and
        // the legacy path may interact via remove().
        RTP_LLM_LOG_WARNING("SharedBlockCache::decUseRef: use_ref already 0 for key %ld", cache_key);
        return;
    }
    --item.use_ref;
    lru_cache_.put(cache_key, item);
    if (unified_ref_counter_) {
        unified_ref_counter_->decUseRef(cache_key);
    }
    // DEFEND1 HIGH-9 (R6 DEV-γ): post-condition mirror invariant.
    assertMirrorInvariantUnlocked_CHECK();
}

int SharedBlockCache::useRefCount(CacheKeyType cache_key) const {
    std::lock_guard<std::mutex> lock(mu_);
    // Use a const-correct lookup: LRUCache::get touches the entry as MRU,
    // which is fine here (test/diag-only API; callers don't rely on
    // non-promotion semantics). The legacy contains() is true-const.
    if (!lru_cache_.contains(cache_key)) {
        return 0;
    }
    auto& mutable_self = const_cast<SharedBlockCache&>(*this);
    auto [success, item] = mutable_self.lru_cache_.get(cache_key);
    return success ? item.use_ref : 0;
}

void SharedBlockCache::setSwaTailPin(CacheKeyType cache_key, bool pinned) {
    std::lock_guard<std::mutex> lock(mu_);
    auto [success, item] = lru_cache_.get(cache_key);
    if (!success) {
        return;
    }
    item.pin_for_swa_tail = pinned;
    lru_cache_.put(cache_key, item);
}

void SharedBlockCache::putUnified(CacheKeyType cache_key, int super_block_id, bool is_resident) {
    RTP_LLM_PROFILE_FUNCTION();

    // FIX-B HIGH-2 (DEFEND-1 #2): hoist the S != 0 invariant to the entry of
    // putUnified so the OFFENDING CALLER is named at the bad put, not far
    // downstream when the K is later displaced and cascadeUnifiedSnapshot
    // fires the deep CHECK (line 539 below).  S == 0 is reserved by
    // SuperBlockFreeList (allocSuperBlock / unifiedMalloc invariant); any
    // caller hard-coding S = 0 (test stub / connector path that forgets to
    // skip slot 0 / unifiedMalloc regression) is structurally a bug.  The
    // cascade-time CHECK at :539 stays as defense in depth.
    RTP_LLM_CHECK_WITH_INFO(super_block_id != 0,
                            "SharedBlockCache::putUnified received reserved super_block_id 0 "
                            "(cache_key=%ld); SuperBlockFreeList reserves slot 0 — caller must "
                            "allocate via SuperBlockFreeList::alloc",
                            static_cast<long>(cache_key));

    // Snapshot any displaced item under mu_; cascade OUTSIDE mu_ to honour
    // the §3.0 lock-order (L1 released before L2 reclaim callback into the
    // SuperBlockFreeList).
    std::vector<UnifiedCacheItem> displaced;
    {
        std::lock_guard<std::mutex> lock(mu_);

        // In-place update path: bump version, refresh slot, do NOT cascade.
        //
        // XR4-1 NIT (R6 DEV-γ): collapse the prior ``contains() + get()`` two-
        // step into a single ``get()`` — LRUCache::get returns {false, default}
        // and does NOT promote on miss (verified at LRUCache.h:153-160), so the
        // semantics are identical and we drop one hash lookup off the hot put
        // path.
        {
            auto [success, existing_item] = lru_cache_.get(cache_key);
            if (success) {
                if (existing_item.slots.empty()) {
                    existing_item.slots.resize(1, NULL_BLOCK_IDX);
                }
                const BlockIdxType incoming = static_cast<BlockIdxType>(super_block_id);
                const BlockIdxType existing = existing_item.slots[0];
                const bool         existing_null = isNullBlockIdx(existing);
                const bool         incoming_null = isNullBlockIdx(incoming);

                if (existing_null && !incoming_null) {
                    existing_item.slots[0]    = incoming;
                    existing_item.is_resident = is_resident;
                    // M03-PR3 dual-write: bump CACHE on UnifiedRefCounter AND
                    // fan out blockCacheReference to EVERY per-pool counter
                    // (bps[p]==1 ⇒ poolBlockId(p, S, 0) == S). PR-2's pool-0-only
                    // bump was the minimal stub; the full migration touches
                    // each pool so per-pool tryFreeBlocks gates stay coherent.
                    for (auto& pool : group_pools_) {
                        if (pool) {
                            pool->blockCacheReference(incoming);
                        }
                    }
                    // XR2-A2: CACHE bump under L1 — see the new-entry branch
                    // below for the full lock-order design note (race-vs-evict
                    // window that motivated the hoist).
                    if (unified_ref_counter_) {
                        unified_ref_counter_->bump(super_block_id, UnifiedRefCounter::Kind::CACHE);
                    }
                    lru_cache_.put(cache_key, existing_item);
                    ++version_;
                } else if (!existing_null && !incoming_null) {
                    // DEFEND1 HIGH-3 / XR4-1 (R6 DEV-γ): both slots populated.
                    //   * Equal incoming S    -> idempotent put; caller's
                    //     allocSuperBlock bump is intentionally NOT re-bumped
                    //     (we already hold one CACHE refcount for this K). Warn
                    //     so a high duplicate-put rate surfaces as an upstream
                    //     bug rather than silently inflating allocation cost.
                    //   * Different incoming  -> structurally a bug: the
                    //     incoming caller allocated S' under the assumption it
                    //     would be inserted, but we already point K at S. If we
                    //     dropped silently, the caller's allocSuperBlock bump
                    //     leaks until process exit (no second cascade pairs
                    //     with it). Fail loud so the offending producer (likely
                    //     an evict-then-realloc race that didn't wait for the
                    //     prior put to drain) is named.
                    RTP_LLM_CHECK_WITH_INFO(
                        existing == incoming,
                        "SharedBlockCache::putUnified conflict K=%ld existing S=%d incoming S=%d "
                        "— would leak incoming bump (caller allocated via SuperBlockFreeList but "
                        "current LRU entry already points at a different S)",
                        static_cast<long>(cache_key),
                        static_cast<int>(existing),
                        super_block_id);
                    RTP_LLM_LOG_WARNING(
                        "SharedBlockCache::putUnified duplicate put for K=%ld S=%d — refcount NOT "
                        "bumped (idempotent put); caller-side allocSuperBlock bump must be released "
                        "or callers will leak one super-block per duplicate put",
                        static_cast<long>(cache_key),
                        super_block_id);
                }
                // existing_null && incoming_null  -> nothing to do
                // !existing_null && incoming_null -> incoming carries no payload, leave existing
                return;
            }
        }

        UnifiedCacheItem item;
        item.cache_key   = cache_key;
        item.is_resident = is_resident;
        item.slots       = {static_cast<BlockIdxType>(super_block_id)};

        lru_cache_.putWithEvictCallback(
            cache_key, item, [&](UnifiedCacheItem&& gone) { displaced.push_back(std::move(gone)); });
        ++version_;

        if (!isNullBlockIdx(static_cast<BlockIdxType>(super_block_id))) {
            for (auto& pool : group_pools_) {
                if (pool) {
                    pool->blockCacheReference(static_cast<BlockIdxType>(super_block_id));
                }
            }
            // XR2-A1/A2/A3 cross-review (REAL_RACE_FIX_REQUIRED): the unified
            // CACHE bump MUST happen under L1, paired with the per-pool fan-out
            // above and the LRU insert at putWithEvictCallback. Pre-fix, the
            // bump was deferred to OUTSIDE L1 — that opened a window where a
            // concurrent evictAndFreeUnified could snapshot the newly-inserted
            // K (gates skip CACHE per `inUse(S)` definition), run cascade, and
            // call `dec(S, CACHE)` on a counter still at zero → underflow
            // RTP_LLM_FAIL crash (or symmetric reclaim-before-bump race
            // returning S to the free list while LRU still points at it). The
            // in-place branch above already calls `bump` under L1 at line 304
            // → L1 → Lcr is an existing edge, no new lock-order risk.
            if (unified_ref_counter_) {
                unified_ref_counter_->bump(super_block_id, UnifiedRefCounter::Kind::CACHE);
            }
        }
        // DEFEND1 HIGH-9 (R6 DEV-γ): post-condition mirror invariant —
        // run BEFORE we drop mu_ so any divergence is attributed to this
        // putUnified rather than a downstream operation.
        assertMirrorInvariantUnlocked_CHECK();
    }

    // Cascade-free OUTSIDE mu_.
    //
    // XR-C1 fix (cache-ref leak race): the earlier implementation re-acquired
    // mu_ and skipped the dec if the displaced key was already resident again
    // at the same S. That was a residency-paired view of CACHE — but the
    // counter is EVENT-PAIRED. The bump that owns this dec was issued by
    // the LRU insertion that produced ``item``. A racing ``putUnified`` for
    // the same K landing at the same S has ALREADY issued its OWN paired
    // bump (line 323-329 + 337-339); skipping our dec would leave that
    // original bump dangling forever (CACHE(S)>=1 with one fewer logical
    // residency than counter ticks, isZero(S) never fires, S never returns
    // to SuperBlockFreeList). Always dec the snapshot — every put has
    // exactly one matching cascade dec, regardless of subsequent re-puts.
    cascadeUnifiedSnapshot(displaced, /*cascaded_out=*/nullptr);
}

SharedBlockCache::MatchResultUnified SharedBlockCache::matchUnified(const std::vector<CacheKeyType>& keys,
                                                                    int                              tokens_per_block,
                                                                    bool                             check_swa_tail) {
    RTP_LLM_PROFILE_FUNCTION();
    MatchResultUnified out;
    out.release_guard = ReleaseGuard(this);

    if (keys.empty()) {
        return out;
    }

    out.super_block_ids.reserve(keys.size());

    {
        std::lock_guard<std::mutex> lock(mu_);
        for (size_t i = 0; i < keys.size(); ++i) {
            if (!lru_cache_.contains(keys[i])) {
                break;
            }
            auto [success, item] = lru_cache_.get(keys[i]);
            if (!success || !item.is_resident) {
                break;
            }
            const int S = item.superBlockId();
            if (S < 0) {
                break;
            }
            // M03-PR3: route the use_ref bump through UnifiedRefCounter when
            // wired. Witness comparison is item.slots vs item.slots (taken
            // under the same mu_); equal by construction, so the bump always
            // lands for the in-walk case. Out-of-walk callers (M03-PR4 SWA
            // tail-read path) supply a stale ``expected_slots`` so the witness
            // gate has bite.
            if (unified_ref_counter_) {
                if (!unified_ref_counter_->incUseRef(keys[i], S, item.slots, item.slots)) {
                    break;
                }
            }
            // Legacy mirror — preserves the per-item gate read by
            // ``selectAndEvictUnified`` on the non-unified-counter fallback
            // path. Under unified mode the gate also consults
            // ``unified_ref_counter_->useRefPinned(S)``.
            ++item.use_ref;
            lru_cache_.put(keys[i], item);  // re-insert (touches MRU + persists use_ref)
            out.super_block_ids.push_back(S);
            out.release_guard.addKey(keys[i]);
        }
        // DEFEND1 HIGH-9 (R6 DEV-γ): post-condition mirror invariant.
        assertMirrorInvariantUnlocked_CHECK();
    }

    out.found = !out.super_block_ids.empty();

    // SWA tail-veto: when the caller is the SWA group, the matched tail must
    // include at least kSwaActiveTailBlocks worth of state. If not, return
    // reuse_length=0 per M03 §5 (KV_ONLY_HIT downgrades). The decision is
    // simplistic in PR-2: if the caller requested check_swa_tail and the
    // matched count is < 2, demote to reuse_length=0 (M03-PR4 will plumb the
    // actual per-stream SWA-window context). Pinned use_refs remain held by
    // the guard so cascade can't drop the underlying blocks.
    if (check_swa_tail && out.super_block_ids.size() < 2) {
        out.reuse_length = 0;
    } else {
        out.reuse_length = out.super_block_ids.size() * static_cast<size_t>(tokens_per_block);
    }

    return out;
}

std::vector<SharedBlockCache::UnifiedCacheItem>
SharedBlockCache::selectAndEvictUnified(size_t min_super_blocks) {
    RTP_LLM_PROFILE_FUNCTION();
    std::vector<UnifiedCacheItem> snapshot;
    if (min_super_blocks == 0) {
        return snapshot;
    }

    std::lock_guard<std::mutex> lock(mu_);
    if (lru_cache_.empty()) {
        return snapshot;
    }

    // Scan LRU tail; skip items that are pin_for_swa_tail OR use_ref>0 (Fix 78
    // / R5). Up to K=2*min_super_blocks tail scan window mirrors §3.3 in the
    // design doc — PR-2 implements the unbiased version (hint_gid lands in a
    // follow-up; under bps[p]==1 the bias is a no-op anyway).
    const size_t              scan_cap = 2 * min_super_blocks;
    std::vector<CacheKeyType> to_evict;
    to_evict.reserve(min_super_blocks);
    size_t scanned = 0;
    for (auto it = lru_cache_.items().rbegin(); it != lru_cache_.items().rend(); ++it) {
        if (scanned >= scan_cap || to_evict.size() >= min_super_blocks) {
            break;
        }
        ++scanned;
        const auto& item = it->second;
        // M03-PR3: composite eviction gate.
        //   * pin_for_swa_tail — SWA reader claim (M03-PR4 plumbing).
        //   * item.use_ref > 0 — legacy mirror; still authoritative on the
        //     non-unified-counter fallback path.
        //   * unified_ref_counter_->inUse(S) — REQUEST/CONNECTOR active OR a
        //     use_ref bump is pinning S via the s_to_keys_ reverse index
        //     (Panel-A item 2 closure). CACHE>0 alone does NOT block — that
        //     is the entire point of LRU eviction.
        if (item.pin_for_swa_tail || item.use_ref > 0) {
            continue;  // R10 / R5: pinned by reader or by in-flight match
        }
        if (item.superBlockId() < 0) {
            continue;  // no payload to cascade
        }
        if (unified_ref_counter_ && unified_ref_counter_->inUse(item.superBlockId())) {
            continue;  // REQUEST/CONNECTOR active or use_ref pinned via counter
        }
        to_evict.push_back(item.cache_key);
    }

    for (auto k : to_evict) {
        UnifiedCacheItem removed;
        if (lru_cache_.remove(k, &removed)) {
            snapshot.push_back(std::move(removed));
        }
    }
    if (!snapshot.empty()) {
        ++version_;
    }
    // DEFEND1 HIGH-9 (R6 DEV-γ): post-condition mirror invariant. Eviction
    // only drops items with item.use_ref == 0 (gated above), so LRU pinned
    // count is unchanged; counter use_ref unchanged. Invariant holds.
    assertMirrorInvariantUnlocked_CHECK();
    return snapshot;
}

size_t SharedBlockCache::evictAndFreeUnified(size_t min_super_blocks) {
    RTP_LLM_PROFILE_FUNCTION();
    // Phase A: select + erase under mu_.
    auto snapshot = selectAndEvictUnified(min_super_blocks);
    if (snapshot.empty()) {
        return 0;
    }

    // Phase B: cascade per-pool unconditionally on the snapshot.
    //
    // XR-C1 fix (cache-ref leak race): the prior implementation re-acquired
    // mu_ per item and skipped both the legacy per-pool blockCacheFree AND
    // the unified CACHE dec when the just-evicted K had been re-inserted at
    // the same S by a racing putUnified. That skip un-paired the ORIGINAL
    // bump (the one issued by the put that produced this snapshot item),
    // permanently leaking exactly one CACHE refcount per occurrence — the
    // per-pool mirror leaked identically. Under same-S concurrent put the
    // racing putUnified has already issued its OWN paired bump for the new
    // residency; that bump pairs with a future cascade of T2's entry, not
    // with this snapshot. CACHE is event-paired, not residency-paired:
    // every put has exactly one matching cascade dec. Always drop the
    // snapshot.
    size_t cascaded = 0;
    cascadeUnifiedSnapshot(snapshot, &cascaded);
    return cascaded;
}

void SharedBlockCache::assertMirrorInvariant_CHECK() const {
    std::lock_guard<std::mutex> lock(mu_);
    assertMirrorInvariantUnlocked_CHECK();
}

void SharedBlockCache::assertMirrorInvariantUnlocked_CHECK() const {
    // DEFEND1 HIGH-9 (R6 DEV-γ → R8 always-on per user directive
    // "不要藏 bug 在 release 后面"): compare the LRU's per-item ``use_ref``
    // mirror against the UnifiedRefCounter's primary ``use_ref_`` map. Both
    // are bumped/deced together under L1 (the dual-write contract); a
    // divergence where LRU > counter means a silent dual-write bypass.
    //
    // This was originally `#ifndef NDEBUG`-gated (release build no-op) so a
    // dual-write bypass would not crash production but would also not be
    // detected.  Upgraded R8 to RTP_LLM_CHECK_WITH_INFO unconditional so the
    // process throws std::runtime_error and the operator notices, instead of
    // silently corrupting the cache. Hot path cost: O(LRU.size()) per call,
    // amortized by being on the put/match/dec/evict slow path (not the get).
    if (!unified_ref_counter_) {
        return;
    }
    size_t lru_pinned = 0;
    for (const auto& [k, item] : lru_cache_.items()) {
        (void)k;
        if (item.use_ref > 0) {
            ++lru_pinned;
        }
    }
    const size_t counter_pinned = unified_ref_counter_->useRefMapSize();
    // counter_pinned >= lru_pinned: counter may hold orphans from remove()
    // (legitimate, decUseRef tolerates). The reverse is a silent bypass.
    RTP_LLM_CHECK_WITH_INFO(lru_pinned <= counter_pinned,
                            "SharedBlockCache LRU mirror desync: LRU has %zu pinned items but "
                            "UnifiedRefCounter only tracks %zu use_ref keys — someone bumped "
                            "item.use_ref without paying the counter bump (dual-write bypass)",
                            lru_pinned,
                            counter_pinned);
}

void SharedBlockCache::cascadeUnifiedSnapshot(const std::vector<UnifiedCacheItem>& snapshot,
                                              size_t*                              cascaded_out) {
    // Pre-condition: mu_ is NOT held. ``blockCacheFree`` takes per-pool L3,
    // ``UnifiedRefCounter::dec``/``isZero`` take the counter's internal mu_,
    // and the reclaim callback (e.g. ``SuperBlockFreeList::push``) takes
    // its own L2. Calling them with L1 held would invert the §3.0
    // lock-order (L1 → L2 / L3 documented; reverse forbidden).
    for (const auto& item : snapshot) {
        const int S = item.superBlockId();
        if (S < 0) {
            continue;
        }
        // R2-20 / R3-17 defense-in-depth: S==0 is reserved by
        // SuperBlockFreeList (allocSuperBlock/unifiedMalloc invariant).
        // Reaching here with S==0 means a caller leaked the reserved id
        // into the LRU — fail loud rather than silently free pool 0.
        RTP_LLM_CHECK_WITH_INFO(S != 0,
                                "cascadeUnifiedSnapshot received reserved super-block id 0 "
                                "(cache_key=%ld); SuperBlockFreeList invariant violated upstream",
                                static_cast<long>(item.cache_key));
        // M03-PR3 dual-write: always drop legacy per-pool block_cache_ref
        // AND unified CACHE counter. The two mirrors track 1:1 because put
        // bumps both; eviction drops both. Per-pool tryFreeBlocks fires on
        // per-pool ref==0; super-block reclaim fires on unified isZero(S).
        for (auto& pool : group_pools_) {
            if (pool) {
                pool->blockCacheFree(static_cast<BlockIdxType>(S));
            }
        }
        if (unified_ref_counter_) {
            unified_ref_counter_->dec(S, UnifiedRefCounter::Kind::CACHE);
            if (unified_ref_counter_->isZero(S) && super_block_reclaim_callback_) {
                super_block_reclaim_callback_(S);
            }
        }
        if (cascaded_out) {
            ++(*cascaded_out);
        }
    }
}

}  // namespace rtp_llm
