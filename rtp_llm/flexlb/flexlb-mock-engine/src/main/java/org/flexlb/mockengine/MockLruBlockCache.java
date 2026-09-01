package org.flexlb.mockengine;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Set;

/**
 * Block pool with token counting — the mock counterpart of the production
 * C++ BlockPool/KVCacheAllocator admission chain (KV capacity model v2).
 *
 * <p>One pool per engine, sized in BLOCKS: {@code totalBlocks = ceil(totalKvTokens / spb)}.
 * A block is in exactly one of three states:
 * <ul>
 *   <li><b>held</b> — referenced by an in-flight request (prefill batch member,
 *       running decode stream, decode growth blocks). Never counted as available.</li>
 *   <li><b>LRU</b> — a cache-key block with reference count 0 (pure LRU). Freely
 *       evictable, but eviction sacrifices prefix reuse.</li>
 *   <li><b>free</b> — carrying no data, immediately allocatable.</li>
 * </ul>
 *
 * <p>Production-faithful semantics (C++ verified):
 * <ul>
 *   <li>{@code availableBlocks = totalBlocks - held - referencedKeyBlocks} — the
 *       analogue of {@code BlockPool::availableBlocksNum()} ({@code req_con_ref_counter_
 *       freeBlockNum()}): free blocks AND pure-LRU blocks count as available, so
 *       "release != delete" and "free != available" both hold.</li>
 *   <li>Admission gate ({@code KVCacheAllocator::evaluateInitCapacity},
 *       TOTAL_AND_AVAILABLE): {@code need <= available && reserve <= available - need}
 *       with {@code reserve = ceil(reserveRatio x totalBlocks)}.</li>
 *   <li>Allocation coupling: malloc needs FREE blocks; when short, pure-LRU blocks
 *       are evicted tail-first ({@code KVCacheGroup::ensureFreeBlocks} /
 *       {@code evictAndFreeForGroup}); only if that still fails is the request
 *       rejected (LACK_MEM).</li>
 *   <li>Completion hands held blocks over to the LRU (the hash-keyed part becomes
 *       cache entries; keyless blocks — decode growth, empty-bh requests — return
 *       to free, mirroring the hash-channel limit of the mock).</li>
 * </ul>
 *
 * <p>Thread safety: all mutating reads/writes are synchronized on this instance.
 * Callers must not hold this monitor while taking other engine locks; the
 * standard order is {@code decodeQueueLock -> cache} (never the reverse).
 */
final class MockLruBlockCache {

    /** Default reserve watermark ratio (production reserve_block_num_ ≈ 5% of init available). */
    static final double DEFAULT_RESERVE_RATIO = 0.05;

    private final int totalBlocks;
    private final double reserveRatio;
    /**
     * cache key → reference count. ref == 0: pure LRU block (evictable);
     * ref > 0: referenced by in-flight requests (matchable, NOT evictable).
     * Access-ordered so the eldest pure-LRU entry is the eviction victim
     * (production evicts the least-recently-used chain tail).
     */
    private final LinkedHashMap<Long, Integer> blocks;
    /** Blocks held by in-flight requests that carry no cache key (growth/empty-bh). */
    private int heldBlocks;
    private long evictions;

    MockLruBlockCache(int totalBlocks) {
        this(totalBlocks, DEFAULT_RESERVE_RATIO);
    }

    MockLruBlockCache(int totalBlocks, double reserveRatio) {
        this.totalBlocks = Math.max(0, totalBlocks);
        this.reserveRatio = Math.max(0, Math.min(0.5, reserveRatio));
        this.blocks = new LinkedHashMap<>(16, 0.75f, true);
    }

    // ─────────────────────────── match ───────────────────────────

    /**
     * Longest prefix run of {@code keys} present in the cache (hit blocks are
     * later re-referenced by {@link #acquire}). Same behavior as before the
     * block-pool rework: the first miss truncates the run.
     */
    synchronized int prefixHitBlocks(List<Long> keys) {
        return matchPrefix(keys).size();
    }

    /** Prefix match against ALL indexed keys, including ones referenced in-flight. */
    private List<Long> matchPrefix(List<Long> keys) {
        List<Long> hits = new ArrayList<>();
        for (Long key : keys) {
            if (!blocks.containsKey(key)) {
                break;
            }
            hits.add(key);
        }
        return hits;
    }

    // ─────────────────────────── admission ───────────────────────────

    /**
     * Admission + allocation for one request. {@code needBlocks} is the request's
     * total block demand (cache-key count, or ceil(inputLen/spb) for empty-bh
     * requests); prefix-hit blocks are re-referenced instead of re-allocated
     * (production reuseCache reduces need_blocks BEFORE the capacity gate).
     *
     * <p>Gate (TOTAL_AND_AVAILABLE): {@code need <= available} and
     * {@code reserve <= available - need}. Allocation first spends free blocks,
     * then evicts pure-LRU tail blocks ({@code ensureFreeBlocks}); the gate
     * guarantees eviction can always satisfy the request, so failure here means
     * the GATE rejected (LACK_MEM) with no side effects.
     *
     * @return the lease to hand back on admit/release, or {@code null} = LACK_MEM
     */
    synchronized BlockLease acquire(int needBlocks, List<Long> keys) {
        if (needBlocks <= 0) {
            return new BlockLease(List.of(), 0);
        }
        List<Long> hitKeys = matchPrefix(keys);
        int newBlocks = needBlocks - hitKeys.size();
        int avail = availableBlocks();
        if (needBlocks > avail || avail - needBlocks < reserveBlocks()) {
            return null; // LACK_MEM (or reserve watermark) — no state changed
        }
        // Allocation coupling: free first, then evict the LRU tail (each
        // eviction trades prefix reuse for capacity — evictions counter).
        // The gate above guarantees this terminates; the break is defensive
        // (an invariant breach must not hang the enqueue path).
        while (freeBlocks() < newBlocks) {
            if (!evictOne()) {
                break;
            }
        }
        for (Long key : hitKeys) {
            blocks.put(key, blocks.get(key) + 1);
        }
        heldBlocks += newBlocks;
        return new BlockLease(hitKeys, newBlocks);
    }

    /**
     * Grow a running request's allocation by one block (decode per-step growth —
     * production incrMalloc; only free/LRU are consulted, no reserve gate).
     * @return false when the pool is exhausted (caller degrades growth).
     */
    synchronized boolean grow(BlockLease lease) {
        if (freeBlocks() <= 0 && !evictOne()) {
            return false;
        }
        lease.nakedBlocks++;
        heldBlocks++;
        return true;
    }

    // ─────────────────────────── completion / cancel ───────────────────────────

    /**
     * Normal completion: release the lease and hand the request's cache-keyed
     * blocks over to the LRU. Keyless held blocks (growth, empty-bh) return to
     * free — the hash channel carries no key for them, so they cannot become
     * reusable cache entries.
     *
     * @return true when the indexed key set changed (drives cacheVersion bumps)
     */
    synchronized boolean admit(BlockLease lease, List<Long> keys) {
        dereference(lease.hitKeys);
        heldBlocks -= lease.nakedBlocks;
        boolean changed = false;
        int transferred = 0;
        for (Long key : keys) {
            Integer ref = blocks.get(key);
            if (ref == null) {
                blocks.put(key, 0);
                transferred++;
                changed = true;
            } else {
                blocks.put(key, ref); // already indexed — refresh LRU order only
            }
        }
        // Keyed blocks come out of the lease's held allocation (naked == new-key
        // count at acquire time for keyed requests); defensively evict LRU tail
        // blocks for any overshoot (concurrent same-key completions).
        for (int i = lease.nakedBlocks; i < transferred; i++) {
            if (!evictOne()) {
                break;
            }
        }
        return changed;
    }

    /** Cancelled request: release references and return held blocks to free (no LRU handover). */
    synchronized void release(BlockLease lease) {
        dereference(lease.hitKeys);
        heldBlocks -= lease.nakedBlocks;
    }

    /**
     * Admit keys with NO holding lease — direct LRU insertion (unit tests and
     * legacy callers). Capacity comes from free blocks / LRU tail eviction.
     * @return true when the key set changed
     */
    synchronized boolean admit(List<Long> keys) {
        if (totalBlocks == 0 || keys.isEmpty()) {
            return false;
        }
        boolean changed = false;
        for (Long key : keys) {
            // containsKey guard: never clobber a live reference count — a key
            // currently pinned by an in-flight request only gets its LRU order
            // refreshed, exactly like the lease-based admit path.
            if (!blocks.containsKey(key)) {
                blocks.put(key, 0);
                changed = true;
            }
        }
        while (blocks.size() > totalBlocks && evictOne()) {
            // evictOne already counted the eviction
        }
        return changed;
    }

    // ─────────────────────────── forced eviction (/cache_evict) ───────────────────────────

    /**
     * Force-evict the given keys (control-plane POST /cache_evict). Idempotent:
     * keys not present — or currently referenced by an in-flight request
     * (production: a referenced chain cannot be dropped) — are a no-op. Each
     * actual removal counts as an eviction; returns whether the key set changed.
     */
    synchronized boolean evict(List<Long> keys) {
        boolean changed = false;
        for (Long key : keys) {
            Integer ref = blocks.get(key);
            if (ref != null && ref == 0 && blocks.remove(key) != null) {
                evictions++;
                changed = true;
            }
        }
        return changed;
    }

    // ─────────────────────────── observation ───────────────────────────

    /** Total number of LRU evictions (capacity + forced). */
    synchronized long evictions() {
        return evictions;
    }

    synchronized Set<Long> snapshotKeys() {
        return Set.copyOf(blocks.keySet());
    }

    /** Pool size in blocks (== ceil(totalKvTokens / spb)). */
    synchronized int totalBlocks() {
        return totalBlocks;
    }

    /**
     * Available blocks: free + pure-LRU (production availableBlocksNum —
     * the prefix cache counts toward availability; held does not).
     */
    synchronized int availableBlocks() {
        return totalBlocks - heldBlocks - referencedKeyBlocks();
    }

    /** Blocks held by in-flight requests (keyless allocations + growth). */
    synchronized int heldBlocks() {
        return heldBlocks;
    }

    /** Cache-key blocks referenced by in-flight requests. */
    synchronized int referencedKeyBlocks() {
        int referenced = 0;
        for (Integer ref : blocks.values()) {
            if (ref != null && ref > 0) {
                referenced++;
            }
        }
        return referenced;
    }

    /** Indexed cache-key blocks (pure LRU + referenced). */
    synchronized int lruKeyBlocks() {
        return blocks.size();
    }

    /** Fully unallocated blocks. */
    synchronized int freeBlocks() {
        return totalBlocks - heldBlocks - blocks.size();
    }

    /** Reserve watermark in blocks: ceil(reserveRatio x totalBlocks). */
    synchronized int reserveBlocks() {
        return (int) Math.ceil(totalBlocks * reserveRatio);
    }

    // ─────────────────────────── internals ───────────────────────────

    private void dereference(List<Long> hitKeys) {
        for (Long key : hitKeys) {
            Integer ref = blocks.get(key);
            int next = ref == null ? 0 : ref - 1;
            blocks.put(key, Math.max(0, next));
        }
    }

    /**
     * Evict the eldest PURE-LRU entry (ref == 0). Referenced keys are skipped.
     * @return true when a block was freed
     */
    private boolean evictOne() {
        for (java.util.Map.Entry<Long, Integer> entry : blocks.entrySet()) {
            if (entry.getValue() == 0) {
                blocks.remove(entry.getKey());
                evictions++;
                return true;
            }
        }
        return false;
    }

    /**
     * A running request's block lease: the prefix-hit cache keys it references
     * (ref-counted) and the keyless blocks it holds. Mutable in
     * {@link #nakedBlocks} because decode growth extends the allocation
     * mid-flight (production incrMalloc).
     */
    static final class BlockLease {
        final List<Long> hitKeys;
        int nakedBlocks;

        BlockLease(List<Long> hitKeys, int nakedBlocks) {
            this.hitKeys = hitKeys;
            this.nakedBlocks = nakedBlocks;
        }

        /** Total blocks this lease pins (hit references + keyless holds). */
        int totalBlocks() {
            return hitKeys.size() + nakedBlocks;
        }
    }
}
