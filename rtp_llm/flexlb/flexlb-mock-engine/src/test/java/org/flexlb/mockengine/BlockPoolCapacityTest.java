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
import static org.flexlb.mockengine.MockEngineTestSupport.enqueueAndFetch;
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

    @Test
    void flatLruEvictsOnlyNeededBlocksAndDoesNotTouchSurvivingParents() {
        var cache = new MockLruBlockCache(5, 0);
        cache.setPrefixTreeEnabled(false);
        cache.admit(List.of(1L, 2L, 3L, 4L, 5L));
        var first = cache.acquire(1, List.of());
        assertNotNull(first);
        assertEquals(java.util.Set.of(2L, 3L, 4L, 5L), cache.snapshotKeys());
        assertEquals(1, cache.evictions());
        var next = cache.acquire(1, List.of());
        assertNotNull(next);
        assertEquals(java.util.Set.of(3L, 4L, 5L), cache.snapshotKeys());
        cache.release(first);
        cache.release(next);
    }

    @Test
    void flatLruProtectsReferencesAndHonorsCacheReads() {
        var cache = new MockLruBlockCache(5, 0);
        cache.setPrefixTreeEnabled(false);
        cache.admit(List.of(1L, 2L, 3L));
        cache.admit(List.of(4L, 5L));
        assertEquals(1, cache.prefixHitBlocks(List.of(1L)));
        var pinned = cache.acquire(1, List.of(2L));
        assertNotNull(pinned);
        var incoming = cache.acquire(2, List.of());
        assertNotNull(incoming);
        assertEquals(java.util.Set.of(1L, 2L, 5L), cache.snapshotKeys());
        assertEquals(2, cache.evictions());
        cache.release(incoming);
        cache.release(pinned);
        assertEquals(5, cache.availableBlocks());
    }

    @Test
    void treeEvictionRemainsDefault() {
        var cache = new MockLruBlockCache(5, 0);
        assertTrue(cache.prefixTreeEnabled());
        cache.admit(List.of(1L, 2L, 3L, 4L, 5L));
        assertNotNull(cache.acquire(1, List.of()));
        assertEquals(5, cache.evictions());
    }

    @Test
    void cachedHitsMustBeChargedBeforeWatermarkForBothAdmissionPaths() {
        for (boolean decode : new boolean[] {false, true}) {
            var cache = new MockLruBlockCache(6, .05);
            var keys = List.of(1L, 2L, 3L);
            cache.admit(keys);
            var other = cache.acquire(1, List.of());
            assertEquals(5, cache.availableBlocks());
            var rejected = decode ? cache.acquireWithReuseDetailed(5, keys) : cache.acquireDetailed(5, keys);
            assertEquals(MockLruBlockCache.AllocationFailure.RETRYABLE, rejected.failure());
            assertEquals(5, cache.availableBlocks(), "failed admission rolls back newly pinned hits");
            assertEquals(0, cache.referencedKeyBlocks());
            assertEquals(0, cache.evictions());
            cache.release(other);
            var admitted = decode ? cache.acquireWithReuseDetailed(5, keys) : cache.acquireDetailed(5, keys);
            assertNotNull(admitted.lease());
            assertEquals(1, cache.availableBlocks());
            cache.release(admitted.lease());
            assertEquals(6, cache.availableBlocks());
        }
    }

    @Test
    void prefillAdmissionChargesOnlyNewBlocksAndPreservesWatermark() {
        var cache = new MockLruBlockCache(10, .1);
        var keys = List.of(1L, 2L, 3L, 4L, 5L, 6L);
        var seed = cache.acquire(6, keys);
        assertNotNull(seed);
        var owner = cache.retainComputed(seed, keys);
        assertEquals(4, cache.availableBlocks());

        // Six blocks are already pinned by a concurrent request. Only three
        // additional blocks are needed; one block must remain as reserve.
        var incoming = cache.acquireDetailed(9, keys);
        assertNotNull(incoming.lease());
        assertEquals(3, cache.heldBlocks());
        assertEquals(1, cache.availableBlocks());
        assertEquals(0, cache.evictions());

        // Fully shared KV remains admissible at the watermark, but a new
        // physical block cannot consume the last reserved block.
        var shared = cache.acquire(6, keys);
        assertNotNull(shared);
        assertNull(cache.acquire(7, keys));
        cache.release(shared);
        cache.release(incoming.lease());
        assertEquals(6, cache.referencedKeyBlocks());
        cache.release(owner);
        assertEquals(10, cache.availableBlocks());
        assertEquals(0, cache.heldBlocks());
        assertEquals(0, cache.referencedKeyBlocks());
    }

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
                enqueueAndFetch(prefill, batch(1, slot(0, tooBig)));
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
                enqueueAndFetch(prefill, batch(2, slot(0, small)));
        assertEquals(0, ack2.getErrorsCount());
        assertEquals(1, ack2.getSuccessesCount());
    }

    @Test
    void allocationEvictionImmediatelyUpdatesOnlyItsEngineCacheVersion() throws Exception {
        var model = MockEngineTestSupport.performanceModel(tempDir, "10", 0.1);
        var first = newService(model, 5);
        var second = newService(model, 5);
        var field = JavaMockEngineCluster.FastRpcService.class.getDeclaredField("cache");
        field.setAccessible(true);
        var cache = (MockLruBlockCache) field.get(first);
        cache.setPrefixTreeEnabled(false);
        cache.admit(List.of(1L, 2L, 3L, 4L, 5L));
        long before = first.getCacheVersion();
        long peerBefore = second.getCacheVersion();
        // No event log is installed, and the allocating request has not completed.
        var lease = cache.acquire(1, List.of());
        assertNotNull(lease);
        assertTrue(first.getCacheVersion() > before);
        assertEquals(peerBefore, second.getCacheVersion());
        EngineRpcService.CacheStatusPB snapshot = MockEngineTestSupport.unary(observer ->
                first.getCacheStatus(EngineRpcService.CacheVersionPB.newBuilder()
                        .setLatestCacheVersion(before).setNeedCacheKeys(true).build(), observer));
        assertTrue(snapshot.getVersion() > before);
        assertFalse(snapshot.getCacheKeysMap().containsKey(1L));
        cache.release(lease);
    }

    @Test
    void workerStatusReportsConfiguredFifoLimits() throws Exception {
        var model = MockEngineTestSupport.performanceModel(tempDir, "10", 0.1);
        var field = MockPerformanceModel.class.getDeclaredField("prefillBatchPolicy");
        field.setAccessible(true);
        field.set(model, new MockPrefillBatchPolicy(64, 512000, 3145728,
                786432, 8, false, 0, 256, 128));
        var service = newService(model, 5);
        var status = MockEngineTestSupport.workerStatus(service, 0);
        assertEquals(512000L, status.getMaxBatchTokensSize());
        assertEquals(786432L, status.getMaxSeqLen());
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
