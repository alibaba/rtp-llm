#pragma once

#include <array>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheResource.h"  // CacheKeyType, BlockIdxType, BlockIndicesType, NULL_BLOCK_IDX
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

class BlockRefCounter {
public:
    BlockRefCounter() {}
    BlockRefCounter(int block_nums) {
        init(block_nums);
    }

    void init(int block_nums) {
        ref_counter.clear();
        total_block_nums_ = block_nums - 1;
        for (int i = 1; i < block_nums; ++i) {
            ref_counter[i] = 0;
        }
        busy_block_num_ = 0;
    }

    int getRefCounter(int block_index) const {
        return ref_counter.at(block_index);
    }

    void incrementRefCounter(const std::vector<int>& block_indices) {
        for (int index : block_indices) {
            ref_counter[index]++;
            if (ref_counter[index] == 1) {
                busy_block_num_++;
            }
        }
    }

    void decrementRefCounter(const std::vector<int>& block_indices) {
        for (int index : block_indices) {
            if (ref_counter[index] == 0) {
                RTP_LLM_FAIL("block:%d decrease zero ref count.", index);
                return;
            } else {
                ref_counter[index]--;
                if (ref_counter[index] == 0) {
                    busy_block_num_--;
                }
            }
        }
    }

    uint32_t busyBlockNum() const {
        return busy_block_num_;
    }

    uint32_t freeBlockNum() const {
        return total_block_nums_ - busy_block_num_;
    }

private:
    std::unordered_map<int, int> ref_counter;
    uint32_t                     busy_block_num_ = 0;
    uint32_t                     total_block_nums_;
};

// ============================================================================
// M03-PR3: UnifiedRefCounter — super-block scope, 5-counter family
// ============================================================================
//
// Owns reference state PER SUPER-BLOCK for the unified path. Maintains three
// PRIMARY counters (REQUEST, CONNECTOR, CACHE) and two DERIVED aggregates
// (REQ_CON = REQUEST + CONNECTOR; REQ_CACHE = REQUEST + CACHE), updated
// incrementally on every primary transition so availability queries remain
// O(1). PRIMARY counters drive full-reclaim: a super-block is fully
// reclaimable iff REQUEST==0 && CONNECTOR==0 && CACHE==0 && !useRefPinned(S).
// Derived aggregates are diagnostic / probe convenience — never gate freeing
// on their own (M03 §2.2).
//
// O(1) busy aggregate: ``busy_super_block_num_`` mirrors the legacy
// ``BlockRefCounter::busy_block_num_`` — incremented when any primary
// transitions 0 -> positive; decremented when ALL primaries reach 0 from
// positive. Lookup via ``busySuperBlockNum() / freeSuperBlockNum()``.
//
// CONTRACT — DUAL-WRITE DURING TRANSITION (M03 §2.2 / F02 11-1):
//   Every bump/dec on UnifiedRefCounter is paired with a matching call to
//   the LEGACY per-pool ``BlockPool::{request,connector,blockCache}{Reference,Free}``
//   at the CALL SITE. UnifiedRefCounter does NOT fan out to BlockPool itself
//   (avoids an include cycle since BlockPool.h includes this header, and
//   keeps the counter independently testable). PD wire / connector code
//   (M04) still consumes per-pool counters via BlockPool accessors — the
//   dual-write keeps those queries coherent until M03-PR5 cleanup removes
//   the legacy per-pool counters entirely. Callers in HybridPoolKVCacheAllocator
//   (M01-PR2) and SharedBlockCache (M03-PR2) MUST issue both calls in the
//   same critical section to keep the two views aligned.
//
// LOCK ORDER (mirrors M03 §3.0 / M01 §3.7):
//   * UnifiedRefCounter::mu_ is INTERNAL — taken inside each method, released
//     before return. It is a leaf (never crosses to another lock).
//   * Callers MUST NOT hold ``SharedBlockCache::mu_`` when invoking dec / decRange:
//     underflow → RTP_LLM_FAIL aborts the process; the fatal-path logging
//     under L1 would freeze on a nested L1 acquisition.
//   * super_block_allocator_->mu_ (M01 §3.7) is non-recursive std::mutex —
//     hold neither L1 nor this counter's mu_ when calling into the allocator.
//
// UNDERFLOW PROTOCOL (Panel-A item 7):
//   ``dec`` / ``decRange`` invoke RTP_LLM_FAIL on underflow — the process
//   aborts; the call NEVER returns on the underflow path. Callers may
//   therefore assume the returned value (when introduced) is the post-dec
//   count for well-formed calls.
//
// useRef SEMANTICS (Panel-A item 5 / hidden assumption #4):
//   * ONE bump per ``incUseRef`` call (NOT per slot in ``expected_slots``);
//     the vector is purely a TOCTOU witness, not a counter weight.
//   * ``expected_slots`` is byte-compared element-wise against
//     ``current_slots`` under ``mu_``. Mismatch (item was evicted-then-
//     rematerialised on a different super_block_id, or slot layout
//     changed) -> return false; caller treats as a match miss.
//   * Scalar superBlockId equality is NOT sufficient — a benign LRU touch
//     does not change slots, so element-wise gives no false negatives; it
//     correctly rejects every re-allocation that reused the same S.
//
// REVERSE INDEX:
//   ``s_to_keys_`` (unordered_multimap<int, CacheKeyType>) is maintained
//   alongside every {incUseRef, decUseRef} so ``useRefPinned(S)`` is O(1)
//   average. Without this index ``selectAndEvict`` would skip nothing on
//   use_ref-only items and Fix-78's TOCTOU window reopens.
class UnifiedRefCounter {
public:
    enum class Kind : uint8_t {
        REQUEST   = 0,  // request-side ref (stream malloc)
        CONNECTOR = 1,  // connector-side ref (PD / disk / memory tier)
        REQ_CON   = 2,  // derived: REQUEST + CONNECTOR (full-reclaim probe)
        CACHE     = 3,  // SharedBlockCache donate / put
        REQ_CACHE = 4,  // derived: REQUEST + CACHE (diagnostic only)
    };

    UnifiedRefCounter() = default;

    void init(int num_super_blocks) {
        std::lock_guard<std::mutex> lock(mu_);
        RTP_LLM_CHECK_WITH_INFO(num_super_blocks > 0,
                                "UnifiedRefCounter::init num_super_blocks=%d must be > 0",
                                num_super_blocks);
        num_super_blocks_     = static_cast<uint32_t>(num_super_blocks);
        counters_.assign(num_super_blocks_, std::array<int32_t, 5>{0, 0, 0, 0, 0});
        busy_super_block_num_ = 0;
        use_ref_.clear();
        use_ref_S_.clear();
        s_to_keys_.clear();
    }

    void bump(int S, Kind k) {
        std::lock_guard<std::mutex> lock(mu_);
        bumpUnlocked(S, k);
    }

    void dec(int S, Kind k) {
        std::lock_guard<std::mutex> lock(mu_);
        decUnlocked(S, k);
    }

    void bumpRange(const std::vector<int>& S_list, Kind k) {
        std::lock_guard<std::mutex> lock(mu_);
        for (int S : S_list) {
            bumpUnlocked(S, k);
        }
    }

    void decRange(const std::vector<int>& S_list, Kind k) {
        std::lock_guard<std::mutex> lock(mu_);
        for (int S : S_list) {
            decUnlocked(S, k);
        }
    }

    // Predicates — composition (Panel-A item 2):
    //   isZero(S) = REQUEST==0 && CONNECTOR==0 && CACHE==0 && !useRefPinned(S)
    //   inUse(S)  = REQUEST>0  || CONNECTOR>0  || useRefPinned(S)
    // (CACHE>0 alone does NOT make a super-block "in use" — cached blocks are
    //  eligible for eviction; useRefPinned is the active-reader gate.)
    bool isZero(int S) const {
        std::lock_guard<std::mutex> lock(mu_);
        if (!checkIndex(S)) {
            return true;  // out-of-range treated as fully unused; defensive
        }
        const auto& c = counters_[static_cast<size_t>(S)];
        return c[static_cast<size_t>(Kind::REQUEST)] == 0 && c[static_cast<size_t>(Kind::CONNECTOR)] == 0
               && c[static_cast<size_t>(Kind::CACHE)] == 0 && !useRefPinnedUnlocked(S);
    }

    bool inUse(int S) const {
        std::lock_guard<std::mutex> lock(mu_);
        if (!checkIndex(S)) {
            return false;
        }
        const auto& c = counters_[static_cast<size_t>(S)];
        return c[static_cast<size_t>(Kind::REQUEST)] > 0 || c[static_cast<size_t>(Kind::CONNECTOR)] > 0
               || useRefPinnedUnlocked(S);
    }

    bool useRefPinned(int S) const {
        std::lock_guard<std::mutex> lock(mu_);
        return useRefPinnedUnlocked(S);
    }

    int getRefCount(int S, Kind k) const {
        std::lock_guard<std::mutex> lock(mu_);
        if (!checkIndex(S)) {
            return 0;
        }
        return counters_[static_cast<size_t>(S)][static_cast<size_t>(k)];
    }

    size_t busySuperBlockNum() const {
        std::lock_guard<std::mutex> lock(mu_);
        return busy_super_block_num_;
    }

    size_t freeSuperBlockNum() const {
        std::lock_guard<std::mutex> lock(mu_);
        // Reserved id 0 is excluded from the budget (mirrors SuperBlockFreeList).
        const size_t total = num_super_blocks_ > 0 ? static_cast<size_t>(num_super_blocks_ - 1) : 0;
        return total >= busy_super_block_num_ ? (total - busy_super_block_num_) : 0;
    }

    // incUseRef: one bump per call. Returns false on TOCTOU witness mismatch
    // (caller treats as a match miss). When true, the bump is recorded against
    // K and S; ``s_to_keys_`` gets a (S, K) edge so ``useRefPinned(S)`` is O(1).
    bool incUseRef(CacheKeyType                     K,
                   int                              S,
                   const std::vector<BlockIdxType>& expected_slots,
                   const std::vector<BlockIdxType>& current_slots) {
        std::lock_guard<std::mutex> lock(mu_);
        // Element-wise byte equality. Rejects every re-allocation that reused
        // S with a different slot layout — benign LRU touches do not change
        // slots, so no false negatives.
        if (expected_slots.size() != current_slots.size()) {
            return false;
        }
        for (size_t i = 0; i < expected_slots.size(); ++i) {
            if (expected_slots[i] != current_slots[i]) {
                return false;
            }
        }
        ++use_ref_[K];
        // Track S per K. First bump records S; subsequent bumps must agree
        // (same K cannot suddenly map to a different S without an intervening
        // dec — otherwise the reverse index would be incoherent).
        auto it_s = use_ref_S_.find(K);
        if (it_s == use_ref_S_.end()) {
            use_ref_S_.emplace(K, S);
            s_to_keys_.emplace(S, K);
        } else {
            RTP_LLM_CHECK_WITH_INFO(it_s->second == S,
                                    "UnifiedRefCounter::incUseRef K=%ld remapped from S=%d to S=%d without dec",
                                    static_cast<long>(K),
                                    it_s->second,
                                    S);
        }
        return true;
    }

    void decUseRef(CacheKeyType K) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = use_ref_.find(K);
        if (it == use_ref_.end()) {
            // DEFEND1 HIGH-4 (R6 DEV-γ): K missing entirely is a tolerable
            // race — the bump may have been drained by an evict path that
            // wiped K (e.g. SharedBlockCache::remove called between bump and
            // abandon). Warn so a high rate surfaces as a real bug rather
            // than being silently no-op'd as it was before.
            RTP_LLM_LOG_WARNING(
                "UnifiedRefCounter::decUseRef no-op for K=%ld (use_ref already drained) — "
                "tolerable under heavy LRU churn but a high rate indicates a guard-accounting "
                "bug (double-abandon / incUseRef returned false but addKey called anyway)",
                static_cast<long>(K));
            return;
        }
        // DEFEND1 HIGH-4 (R6 DEV-γ): K present with non-positive counter is a
        // structural invariant violation — we ERASE entries at zero (see below),
        // so reaching here with it->second <= 0 means dual-storage corruption
        // or a missed erase. Fail loud rather than silently no-op (the prior
        // silent path hid the guard-accounting bug; the eventual symptom was
        // ``inUse(S)`` returning false too early -> premature eviction ->
        // use-after-free on the tensor pointer).
        RTP_LLM_CHECK_WITH_INFO(it->second > 0,
                                "UnifiedRefCounter::decUseRef underflow for K=%ld "
                                "(counter present but value=%d <= 0; entries should be erased at 0)",
                                static_cast<long>(K),
                                it->second);
        --it->second;
        if (it->second == 0) {
            const int S = use_ref_S_.at(K);
            use_ref_.erase(it);
            use_ref_S_.erase(K);
            // Erase one matching s_to_keys_ edge.
            auto range = s_to_keys_.equal_range(S);
            for (auto eit = range.first; eit != range.second; ++eit) {
                if (eit->second == K) {
                    s_to_keys_.erase(eit);
                    break;
                }
            }
        }
    }

    int useRefCount(CacheKeyType K) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = use_ref_.find(K);
        return it == use_ref_.end() ? 0 : it->second;
    }

    // DEFEND1 HIGH-9 (R6 DEV-γ) — invariant helper. Returns the number of
    // distinct keys currently holding at least one use_ref bump. Used by
    // SharedBlockCache::assertMirrorInvariant_DCHECK to verify that the LRU
    // mirror's per-item ``use_ref`` count matches the counter's view. Cheap
    // (O(1) on unordered_map::size); takes the counter's mu_ so the read is
    // consistent with the most recent bump/dec.
    size_t useRefMapSize() const {
        std::lock_guard<std::mutex> lock(mu_);
        return use_ref_.size();
    }

private:
    bool checkIndex(int S) const {
        return S >= 0 && static_cast<uint32_t>(S) < num_super_blocks_;
    }

    bool anyPrimaryNonZeroUnlocked(int S) const {
        const auto& c = counters_[static_cast<size_t>(S)];
        return c[static_cast<size_t>(Kind::REQUEST)] > 0 || c[static_cast<size_t>(Kind::CONNECTOR)] > 0
               || c[static_cast<size_t>(Kind::CACHE)] > 0;
    }

    bool useRefPinnedUnlocked(int S) const {
        // O(1) average via s_to_keys_ reverse multimap.
        return s_to_keys_.find(S) != s_to_keys_.end();
    }

    // mu_ held.
    void bumpUnlocked(int S, Kind k) {
        RTP_LLM_CHECK_WITH_INFO(checkIndex(S),
                                "UnifiedRefCounter::bump S=%d out of range [0,%u)",
                                S,
                                num_super_blocks_);
        // Only PRIMARY kinds are externally bumpable; derived counters update
        // incrementally as a side-effect. Block REQ_CON / REQ_CACHE callers.
        RTP_LLM_CHECK_WITH_INFO(k == Kind::REQUEST || k == Kind::CONNECTOR || k == Kind::CACHE,
                                "UnifiedRefCounter::bump only accepts PRIMARY Kind (got %u)",
                                static_cast<unsigned>(k));
        const bool was_busy = anyPrimaryNonZeroUnlocked(S);
        auto&      c        = counters_[static_cast<size_t>(S)];
        c[static_cast<size_t>(k)]++;
        if (k == Kind::REQUEST) {
            c[static_cast<size_t>(Kind::REQ_CON)]++;
            c[static_cast<size_t>(Kind::REQ_CACHE)]++;
        } else if (k == Kind::CONNECTOR) {
            c[static_cast<size_t>(Kind::REQ_CON)]++;
        } else /* CACHE */ {
            c[static_cast<size_t>(Kind::REQ_CACHE)]++;
        }
        if (!was_busy) {
            ++busy_super_block_num_;
        }
    }

    // mu_ held; aborts via RTP_LLM_FAIL on underflow.
    void decUnlocked(int S, Kind k) {
        RTP_LLM_CHECK_WITH_INFO(checkIndex(S),
                                "UnifiedRefCounter::dec S=%d out of range [0,%u)",
                                S,
                                num_super_blocks_);
        RTP_LLM_CHECK_WITH_INFO(k == Kind::REQUEST || k == Kind::CONNECTOR || k == Kind::CACHE,
                                "UnifiedRefCounter::dec only accepts PRIMARY Kind (got %u)",
                                static_cast<unsigned>(k));
        auto& c = counters_[static_cast<size_t>(S)];
        if (c[static_cast<size_t>(k)] <= 0) {
            RTP_LLM_FAIL("UnifiedRefCounter::dec underflow S=%d kind=%u (counter already 0)",
                         S,
                         static_cast<unsigned>(k));
            return;  // unreachable; RTP_LLM_FAIL aborts.
        }
        c[static_cast<size_t>(k)]--;
        if (k == Kind::REQUEST) {
            c[static_cast<size_t>(Kind::REQ_CON)]--;
            c[static_cast<size_t>(Kind::REQ_CACHE)]--;
        } else if (k == Kind::CONNECTOR) {
            c[static_cast<size_t>(Kind::REQ_CON)]--;
        } else /* CACHE */ {
            c[static_cast<size_t>(Kind::REQ_CACHE)]--;
        }
        if (!anyPrimaryNonZeroUnlocked(S)) {
            RTP_LLM_CHECK_WITH_INFO(busy_super_block_num_ > 0,
                                    "UnifiedRefCounter::dec busy_super_block_num_ underflow on S=%d",
                                    S);
            --busy_super_block_num_;
        }
    }

    mutable std::mutex                  mu_;
    uint32_t                            num_super_blocks_{0};
    // counters_[S] = {REQUEST, CONNECTOR, REQ_CON, CACHE, REQ_CACHE}
    std::vector<std::array<int32_t, 5>> counters_;
    size_t                              busy_super_block_num_ = 0;

    // use_ref state keyed by CacheKeyType (M03 §2.2).
    std::unordered_map<CacheKeyType, int32_t>  use_ref_;
    std::unordered_map<CacheKeyType, int>      use_ref_S_;   // K -> S (witness consistency check)
    std::unordered_multimap<int, CacheKeyType> s_to_keys_;   // S -> K (O(1) useRefPinned)
};

}  // namespace rtp_llm
