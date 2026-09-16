package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import static org.flexlb.mockengine.MockEngineTestSupport.batch;
import static org.flexlb.mockengine.MockEngineTestSupport.enqueue;
import static org.flexlb.mockengine.MockEngineTestSupport.inputWithBlockKeys;
import static org.flexlb.mockengine.MockEngineTestSupport.slot;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Block-pool capacity model (v2) unit tests — the mock counterpart of the
 * production C++ admission chain this model mirrors:
 *
 * <ul>
 *   <li>{@code acquire} = {@code KVCacheAllocator::evaluateInitCapacity}
 *       (TOTAL_AND_AVAILABLE gate + 5% reserve watermark) coupled with
 *       {@code KVCacheGroup::ensureFreeBlocks} (free-first, LRU-tail eviction).</li>
 *   <li>{@code admit} = completion handover to the LRU ({@code release != delete},
 *       {@code free != available}: pure-LRU blocks stay available).</li>
 *   <li>{@code grow} = per-step decode growth ({@code incrMalloc}).</li>
 *   <li>Enqueue-batch LACK_MEM = the master-visible synchronous rejection surface
 *       (error code 602 = production {@code MALLOC_FAILED}).</li>
 * </ul>
 *
 * <p>Mandatory paths per the capacity-model acceptance list: "LRU eviction
 * triggered, then allocation succeeds" and "LACK_MEM rejection".
 */
class BlockPoolCapacityTest {

    private static final int SPB = 1024;

    @TempDir
    Path tempDir;

    private ScheduledExecutorService scheduler;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;

    @BeforeEach
    void setUp() {
        scheduler = Executors.newScheduledThreadPool(4, runnable -> {
            Thread thread = new Thread(runnable, "block-pool-test-scheduler");
            thread.setDaemon(true);
            return thread;
        });
        services = new ConcurrentHashMap<>();
    }

    @AfterEach
    void tearDown() throws InterruptedException {
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            service.shutdown();
        }
        scheduler.shutdownNow();
        scheduler.awaitTermination(3, TimeUnit.SECONDS);
    }

    // ─────────────────── pool unit: admission ───────────────────

    @Test
    void acquireAdmitsWhenFreeBlocksSufficient() {
        MockLruBlockCache cache = new MockLruBlockCache(10);
        MockLruBlockCache.BlockLease lease = cache.acquire(3, List.of(1L, 2L, 3L));
        assertNotNull(lease, "10-block pool must admit a 3-block request");
        assertEquals(3, lease.totalBlocks());
        assertEquals(3, cache.heldBlocks(), "fresh keys allocate keyless held blocks");
        assertEquals(7, cache.availableBlocks(), "held blocks are not available");
        // Keys are indexed only on completion (admit) — the master's
        // getCacheStatus key set never sees in-flight keys.
        assertEquals(0, cache.snapshotKeys().size());
        assertEquals(0, cache.prefixHitBlocks(List.of(1L, 2L, 3L)));
    }

    @Test
    void admitHandsKeysToLruAndRestoresAvailability() {
        MockLruBlockCache cache = new MockLruBlockCache(10);
        MockLruBlockCache.BlockLease lease = cache.acquire(3, List.of(1L, 2L, 3L));
        assertTrue(cache.admit(lease, List.of(1L, 2L, 3L)),
                "completion must index the request's keys");
        assertEquals(3, cache.snapshotKeys().size());
        assertEquals(0, cache.heldBlocks());
        // release != delete: parked LRU keys count as available.
        assertEquals(10, cache.availableBlocks());
        assertEquals(3, cache.prefixHitBlocks(List.of(1L, 2L, 3L)));
    }

    @Test
    void reserveWatermarkRejectsMarginalAllocation() {
        // ceil(0.05 x 10) = 1 reserve block.
        MockLruBlockCache cache = new MockLruBlockCache(10, 0.05);
        // need=10 leaves available-need = 0 < reserve=1 → reject.
        assertNull(cache.acquire(10, List.of(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L)),
                "the reserve watermark must reject a pool-filling allocation");
        // need=9 leaves 1 >= reserve=1 → admit.
        assertNotNull(cache.acquire(9, List.of(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L)));
    }

    // ─────────────── mandatory path 1: LRU eviction → allocation succeeds ───────────────

    @Test
    void lruEvictionFreesBlocksForNewAllocation() {
        // No reserve for exact block arithmetic in this test.
        MockLruBlockCache cache = new MockLruBlockCache(5, 0.0);
        // Park 5 keys in the LRU (pool fully warm, nothing running).
        assertTrue(cache.admit(List.of(1L, 2L, 3L, 4L, 5L)));
        assertEquals(5, cache.snapshotKeys().size());
        assertEquals(5, cache.availableBlocks(), "pure-LRU keys are available");

        // 4-block request: free=0, but the LRU tail is evicted to satisfy it —
        // exactly the production ensureFreeBlocks coupling (eviction trades
        // prefix reuse for capacity).
        MockLruBlockCache.BlockLease lease = cache.acquire(4, List.of(10L, 11L, 12L, 13L));
        assertNotNull(lease, "LRU-tail eviction must satisfy the allocation");
        assertTrue(cache.evictions() > 0, "evictions counter must record the sacrifice");
        assertEquals(1, cache.availableBlocks(), "5 blocks - 4 held");
        assertEquals(4, cache.heldBlocks());

        // The eldest parked keys were the victims (LinkedHashMap eviction order).
        assertTrue(cache.snapshotKeys().size() <= 5);
    }

    // ─────────────── mandatory path 2: LACK_MEM rejection ───────────────

    @Test
    void acquireRejectsWithLackMemWhenPoolExhausted() {
        MockLruBlockCache cache = new MockLruBlockCache(5, 0.0);
        // One in-flight request pins every block: held=5, free=0, LRU empty.
        MockLruBlockCache.BlockLease big = cache.acquire(5, List.of(1L, 2L, 3L, 4L, 5L));
        assertNotNull(big);
        assertEquals(0, cache.availableBlocks());

        // No free blocks and no pure-LRU blocks → LACK_MEM, and the pool state
        // must not change (the caller rejects synchronously).
        assertNull(cache.acquire(1, List.of(9L)),
                "exhausted pool must reject with LACK_MEM (null lease)");
        assertEquals(5, cache.heldBlocks(), "rejected request must leave no residue");
        assertEquals(0, cache.evictions(), "nothing was evictable");
        assertEquals(0, cache.snapshotKeys().size());
    }

    @Test
    void reacquiringParkedKeysReferencesInsteadOfReallocating() {
        MockLruBlockCache cache = new MockLruBlockCache(10);
        MockLruBlockCache.BlockLease first = cache.acquire(3, List.of(1L, 2L, 3L));
        assertTrue(cache.admit(first, List.of(1L, 2L, 3L)));
        assertEquals(10, cache.availableBlocks(), "parked LRU keys are available");

        // A second request with the SAME keys: prefix hits are re-referenced,
        // not re-allocated — those blocks leave the available set while the
        // request runs ("cache hotter → less available while in flight").
        MockLruBlockCache.BlockLease second = cache.acquire(3, List.of(1L, 2L, 3L));
        assertNotNull(second);
        assertEquals(3, second.totalBlocks(), "hit references count as lease blocks");
        assertEquals(0, second.nakedBlocks, "no new blocks allocated for a full hit");
        assertEquals(7, cache.availableBlocks(),
                "referenced key blocks must leave the available set");

        // Completion of the second request restores availability (ref back to
        // 0). No NEW keys entered the index — admit returns false (no
        // cacheVersion bump: the master re-pulls nothing).
        assertFalse(cache.admit(second, List.of(1L, 2L, 3L)),
                "re-completing the same keys must not bump the key set");
        assertEquals(10, cache.availableBlocks());
    }

    // ─────────────────── pool unit: growth / cancel / forced eviction ───────────────────

    @Test
    void growExtendsLeaseUntilPoolExhaustion() {
        MockLruBlockCache cache = new MockLruBlockCache(3, 0.0);
        MockLruBlockCache.BlockLease lease = cache.acquire(2, List.of(1L, 2L));
        assertNotNull(lease);

        assertTrue(cache.grow(lease), "one free block remains");
        assertEquals(3, lease.totalBlocks(), "growth extends the lease");
        assertEquals(3, cache.heldBlocks());

        assertFalse(cache.grow(lease), "pool exhausted — growth stalls, no abort");
        assertEquals(3, cache.heldBlocks());
    }

    @Test
    void releaseReturnsBlocksWithoutLruHandover() {
        MockLruBlockCache cache = new MockLruBlockCache(10);
        MockLruBlockCache.BlockLease lease = cache.acquire(3, List.of(1L, 2L, 3L));
        cache.release(lease);
        assertEquals(0, cache.snapshotKeys().size(),
                "a cancelled request leaves no cache entries");
        assertEquals(0, cache.heldBlocks());
        assertEquals(10, cache.availableBlocks());
    }

    @Test
    void forcedEvictionSkipsReferencedKeys() {
        MockLruBlockCache cache = new MockLruBlockCache(10);
        MockLruBlockCache.BlockLease lease = cache.acquire(2, List.of(1L, 2L));
        // While in flight, the keys cannot be force-evicted (production: a
        // referenced chain cannot be dropped by /cache_evict).
        assertFalse(cache.evict(List.of(1L, 2L)));
        cache.admit(lease, List.of(1L, 2L));
        assertTrue(cache.evict(List.of(1L)), "pure-LRU key is evictable");
        assertFalse(cache.snapshotKeys().contains(1L));
    }

    @Test
    void legacyAdmitListNeverClobbersLiveReferenceCounts() {
        MockLruBlockCache cache = new MockLruBlockCache(10);
        // Park keys in the LRU, then re-acquire them so they carry live refs
        // (an in-flight request's hit keys are pinned — not evictable, not
        // available).
        MockLruBlockCache.BlockLease first = cache.acquire(3, List.of(1L, 2L, 3L));
        assertNotNull(first);
        assertTrue(cache.admit(first, List.of(1L, 2L, 3L)));
        MockLruBlockCache.BlockLease second = cache.acquire(3, List.of(1L, 2L, 3L));
        assertNotNull(second);
        assertEquals(7, cache.availableBlocks(), "referenced keys are not available");

        // A legacy no-lease admit while the keys are referenced must not reset
        // their reference counts (would unpin blocks mid-flight). It reports
        // no key-set change (nothing NEW entered the index)...
        assertFalse(cache.admit(List.of(1L, 2L, 3L)));
        // ...and the live refs survive: still pinned, still un-evictable.
        assertFalse(cache.evict(List.of(1L)), "live ref must not be force-evictable");
        assertEquals(7, cache.availableBlocks(), "refs survived the legacy admit");

        cache.release(second);
        assertEquals(10, cache.availableBlocks());
    }

    // ─────────────── FastRpcService surface: synchronous 602 rejection ───────────────

    @Test
    void enqueueBatchRejectsWithMallocFailedWhenPoolExhausted() throws IOException {
        MockPerformanceModel model =
                MockEngineTestSupport.performanceModel(tempDir, "10", 0.1);
        JavaMockEngineCluster.FastRpcService prefill =
                newService(model, 10); // 10-block pool, reserve = 1

        // 11 hash-channel blocks > 10-block pool → synchronous LACK_MEM in the
        // EnqueueBatch ack (the master's EngineRejectedException surface).
        EngineRpcService.GenerateInputPB tooBig = inputWithBlockKeys(
                7L, SPB, List.of(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L, 11L));
        EngineRpcService.EnqueueBatchResponsePB ack =
                enqueue(prefill, batch(1, slot(0, tooBig)));
        assertEquals(1, ack.getErrorsCount(), "the oversized request must be rejected");
        assertEquals(JavaMockEngineCluster.LACK_MEM_ERROR_CODE,
                ack.getErrors(0).getErrorInfo().getErrorCode(),
                "LACK_MEM must surface MALLOC_FAILED (602), never the master's 8431");
        assertTrue(ack.getErrors(0).getErrorInfo().getErrorMessage().contains("LACK_MEM"));
        assertEquals("rejected", prefill.getRequestStates().get(7L));
        assertEquals(0, prefill.getInflightCount(), "rejected request leaves no residue");

        // A request the pool CAN serve still succeeds afterwards — the pool
        // is not poisoned by the rejection.
        EngineRpcService.GenerateInputPB small =
                inputWithBlockKeys(8L, SPB, List.of(1L, 2L));
        EngineRpcService.EnqueueBatchResponsePB ack2 =
                enqueue(prefill, batch(2, slot(0, small)));
        assertEquals(0, ack2.getErrorsCount());
        assertEquals(1, ack2.getSuccessesCount());
    }

    // ─────────────────── helpers ───────────────────

    private JavaMockEngineCluster.FastRpcService newService(
            MockPerformanceModel model, int blocks) {
        int port = 63700 + services.size();
        JavaMockEngineCluster.FastRpcService service =
                new JavaMockEngineCluster.FastRpcService(
                        "prefill",
                        EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL,
                        port,
                        services,
                        scheduler,
                        model,
                        blocks,
                        new JavaMockEngineCluster.ClusterStats());
        services.put(port, service);
        return service;
    }
}
