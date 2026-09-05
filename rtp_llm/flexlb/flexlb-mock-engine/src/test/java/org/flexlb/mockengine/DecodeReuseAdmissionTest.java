package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.lang.reflect.Field;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import static org.flexlb.mockengine.MockEngineTestSupport.awaitDecodeQuiescence;
import static org.flexlb.mockengine.MockEngineTestSupport.decodeModel;
import static org.flexlb.mockengine.MockEngineTestSupport.scheduleDecodeCompletion;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * KV v2 fix #5 — decode admission reuse deduction (production
 * {@code DecodeRpcServerNew}: {@code reuse_block_size =
 * generate_stream->reuseBlockSize()}; net new allocation =
 * {@code total_blocks − reuse_block_size}; reused blocks are referenced,
 * never re-allocated; the decode engine re-matches against its OWN LRU at
 * hand-off, which is also the decode LRU's read path).
 *
 * <p>Coverage:
 * <ol>
 *   <li><b>Net allocation</b> — decode admission with LRU hits allocates
 *       {@code total − hit_blocks} into the RUNNING layer (held) and pins the
 *       hits as references in the LRU layer (referenced), never as fresh
 *       running blocks.</li>
 *   <li><b>Net-demand gate</b> — the TOTAL_AND_AVAILABLE gate evaluates the
 *       NET demand: a fully-reused request admits even when
 *       {@code total_demand > available} (reuse only pins references;
 *       production reuse reduces need_blocks BEFORE the capacity gate).</li>
 *   <li><b>Positive feedback</b> — the prefix-match read itself refreshes LRU
 *       recency, so matched chains age slower than un-matched ones, and the
 *       same prefix's hit count never decreases across rounds.</li>
 *   <li><b>Cancelled streams never admit</b> — a cancelled decode stream
 *       leaves no LRU entries (production cancel runs free()); the terminal
 *       race where the cancel lands after the step-boundary claim releases
 *       the lease without LRU handover instead of admitting it.</li>
 * </ol>
 */
class DecodeReuseAdmissionTest {

    private static final int BASE_PORT = 63600;
    private static final int SPB = 1024;

    @TempDir
    Path tempDir;

    private ScheduledExecutorService scheduler;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private int nextPortOffset;

    @BeforeEach
    void setUp() {
        scheduler = Executors.newScheduledThreadPool(4, runnable -> {
            Thread thread = new Thread(runnable, "decode-reuse-scheduler");
            thread.setDaemon(true);
            return thread;
        });
        services = new ConcurrentHashMap<>();
        nextPortOffset = 0;
    }

    @AfterEach
    void tearDown() throws InterruptedException {
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            service.shutdown();
        }
        scheduler.shutdownNow();
        scheduler.awaitTermination(3, TimeUnit.SECONDS);
    }

    // ─────────────── pool unit: net allocation / gate / floor ───────────────

    @Test
    void acquireWithReuseDeductsHitsFromNetAllocation() {
        MockLruBlockCache cache = new MockLruBlockCache(10, 0.0);
        // A previous decode completion parked a 3-key prefix in the LRU.
        assertTrue(cache.admit(List.of(1L, 2L, 3L)));

        // Full demand T=5 (ceil(inputLen/spb), keys cover only the 3-block
        // prefix — the trace's hash-channel coverage gap). Reuse hits 3,
        // net-new 2: hits are REFERENCED in the LRU layer, net-new goes to
        // the RUNNING layer.
        MockLruBlockCache.BlockLease lease =
                cache.acquireWithReuse(5, List.of(1L, 2L, 3L, 4L, 5L));
        assertNotNull(lease, "5-block demand against a 10-block pool must admit");
        assertEquals(3, lease.hitKeys.size(), "3 LRU hits are referenced, not re-allocated");
        assertEquals(2, lease.nakedBlocks, "net new allocation = total − hit_blocks");
        assertEquals(2, cache.heldBlocks(), "only the NET part enters the running layer");
        assertEquals(3, cache.referencedKeyBlocks(),
                "hit blocks are pinned as LRU references, not running allocations");
        assertEquals(5, lease.totalBlocks(), "the lease still spans the full demand");
        assertEquals(5, cache.availableBlocks(), "10 − (2 held + 3 referenced)");
    }

    @Test
    void acquireWithReuseGatesOnNetDemandNotTotalDemand() {
        MockLruBlockCache cache = new MockLruBlockCache(10, 0.0);
        // One in-flight request pins 5 blocks: held=5, available=5, free=0.
        assertNotNull(cache.acquire(5, List.of(100L, 101L, 102L, 103L, 104L)));
        // The remaining 5 blocks park a 5-key prefix in the LRU.
        assertTrue(cache.admit(List.of(1L, 2L, 3L, 4L, 5L)));
        assertEquals(5, cache.availableBlocks());

        // totalDemand=6 with all 5 LRU keys hit → netNew=1 ≤ 5 → admits. The
        // OLD total-demand gate (need ≤ available) would compare 6 > 5 and
        // reject — overstating decode pool pressure exactly the way the
        // production net-demand gate does not.
        MockLruBlockCache.BlockLease lease =
                cache.acquireWithReuse(6, List.of(1L, 2L, 3L, 4L, 5L, 6L));
        assertNotNull(lease,
                "the gate must evaluate the NET demand, not the total demand");
        assertEquals(5, lease.hitKeys.size());
        assertEquals(1, lease.nakedBlocks);

        // The pre-reuse prefill-flavoured gate on the same total (6 > 5)
        // rejects — the two calibers genuinely differ.
        assertNull(cache.acquire(6, List.of(1L, 2L, 3L, 4L, 5L, 6L)),
                "prefill-side acquire keeps its total-demand gate");
    }

    @Test
    void acquireWithReuseFloorsNetAllocationAtZeroAndClampsReuse() {
        MockLruBlockCache cache = new MockLruBlockCache(10, 0.0);
        assertTrue(cache.admit(List.of(1L, 2L, 3L, 4L)));
        // Demand 3 fully covered by a longer parked prefix: reuse is clamped
        // to the demand, net allocation floors at 0.
        MockLruBlockCache.BlockLease lease =
                cache.acquireWithReuse(3, List.of(1L, 2L, 3L, 4L, 5L));
        assertNotNull(lease);
        assertEquals(3, lease.hitKeys.size(), "reuse is clamped to the demand");
        assertEquals(0, lease.nakedBlocks, "net allocation floors at zero");
        assertEquals(0, cache.heldBlocks(), "a fully-reused request allocates nothing");
        assertEquals(3, cache.referencedKeyBlocks());

        // Empty keys (empty-bh traffic): no reuse possible, full net demand.
        MockLruBlockCache.BlockLease keyless =
                cache.acquireWithReuse(4, List.of());
        assertNotNull(keyless);
        assertEquals(0, keyless.hitKeys.size());
        assertEquals(4, keyless.nakedBlocks);
        assertEquals(4, cache.heldBlocks());
    }

    @Test
    void prefixMatchReadRefreshesLruRecency() {
        MockLruBlockCache cache = new MockLruBlockCache(3, 0.0);
        assertTrue(cache.admit(List.of(1L, 2L, 3L)));
        // Access order after insertion: 1 (eldest), 2, 3.

        // READ key 1 through the prefix-match path — the read itself must
        // refresh its recency (order becomes 2, 3, 1).
        assertEquals(1, cache.prefixHitBlocks(List.of(1L)));

        // A fresh allocation must evict the ELDEST pure-LRU block, which is
        // now key 2: without the read refresh, key 1 would be the victim.
        assertNotNull(cache.acquire(1, List.of(9L)));
        assertFalse(cache.snapshotKeys().contains(2L),
                "key 2 must be the eviction victim (eldest after the read)");
        assertTrue(cache.snapshotKeys().contains(1L),
                "key 1 must survive — the match read refreshed its recency");
        assertTrue(cache.snapshotKeys().contains(3L));
    }

    /**
     * Completion parks the request's hash keys into the LRU by CAPACITY
     * conservation, not by the lease's net-allocation arithmetic: trace
     * metadata may carry more hash keys than the token demand covers (10
     * tokens → 1 block, 2 keys), and those keys park fine while the pool
     * has room. Genuine over-subscription evicts the pure-LRU tail (eldest
     * first — including just-parked keys when they are the eldest).
     */
    @Test
    void admitParksAllKeysWhenHashKeysExceedTokenDemand() {
        // Keys (2) exceed the token demand (ceil(10/1024) = 1): admission
        // nets 1 block, but completion must park BOTH keys — the pool has room.
        MockLruBlockCache cache = new MockLruBlockCache(10, 0.0);
        MockLruBlockCache.BlockLease lease =
                cache.acquireWithReuse(1, List.of(401L, 402L));
        assertNotNull(lease);
        assertEquals(1, lease.nakedBlocks);
        assertEquals(0, lease.hitKeys.size());
        assertTrue(cache.admit(lease, List.of(401L, 402L)));
        assertEquals(Set.of(401L, 402L), cache.snapshotKeys(),
                "hash keys beyond the token demand still park when the pool has room");

        // Genuine over-subscription: a 3-block pool with 2 parked keys and 1
        // held block cannot index 4 new keys at once — the ELDEST pure-LRU
        // entries are the victims (1, 2, then the just-parked 7), keeping
        // the pool at capacity, never above.
        MockLruBlockCache tiny = new MockLruBlockCache(3, 0.0);
        assertTrue(tiny.admit(List.of(1L, 2L)));
        MockLruBlockCache.BlockLease over = tiny.acquireWithReuse(1, List.of(7L));
        assertNotNull(over);
        assertEquals(1, tiny.heldBlocks());
        assertTrue(tiny.admit(over, List.of(7L, 8L, 9L, 10L)));
        assertEquals(3, tiny.snapshotKeys().size(),
                "the pool must stay at capacity, never above");
        assertTrue(tiny.snapshotKeys().containsAll(List.of(8L, 9L, 10L)),
                "the eldest entries (1, 2, then 7) are the eviction victims");
        assertFalse(tiny.snapshotKeys().contains(1L));
        assertFalse(tiny.snapshotKeys().contains(2L));
        assertFalse(tiny.snapshotKeys().contains(7L));
    }

    // ─────────────── engine level: hand-off re-match against the OWN LRU ───────────────

    /**
     * Full acceptance check #1: decode admission with LRU hits nets
     * {@code total − hit_blocks} into the running layer and references the
     * hits in the LRU layer. The first round runs cold (no hits — full net
     * allocation, and the TOKEN caliber means the uncovered suffix is
     * allocated too), hands its keys to the decode LRU on completion, and the
     * second round re-matches them.
     */
    @Test
    void decodeAdmissionReusesOwnLruHitsNetAllocatesTheRest() throws Exception {
        // Slow steps (10000 × 0.1 = 1000 ms) so the mid-flight decomposition
        // is stable to observe before the stream completes.
        MockPerformanceModel model = decodeModel(tempDir, 10_000.0, null);
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(model, 8);
        List<Long> keys = List.of(1L, 2L, 3L);

        // Round 1 (cold): inputLen 4096 → T=4, keys cover 3, hits 0 → net 4.
        assertTrue(scheduleDecodeCompletion(decode,
                shapeWithKeys(model, 101L, 4096, 26, keys), -1, null));
        awaitRunningStreams(decode, 1);
        MockLruBlockCache cache = cacheOf(decode);
        assertEquals(4, cache.heldBlocks(),
                "cold admission allocates the FULL token demand "
                        + "(ceil(inputLen/spb)), suffix included");
        assertEquals(0, cache.referencedKeyBlocks());
        assertEquals(4 * SPB, decode.getActiveKvTokens());
        awaitDecodeQuiescence(decode, 30_000);
        // Completion handed the 3 hash keys to the decode LRU.
        assertTrue(cache.snapshotKeys().containsAll(keys),
                "completion must park the request's keys in the decode LRU");

        // Round 2 (warm): same keys → 3 hits referenced, net new = 4 − 3 = 1.
        assertTrue(scheduleDecodeCompletion(decode,
                shapeWithKeys(model, 102L, 4096, 26, keys), -1, null));
        awaitRunningStreams(decode, 1);
        assertEquals(3, cache.referencedKeyBlocks(),
                "the 3 LRU hits must be pinned as references");
        assertEquals(1, cache.heldBlocks(),
                "net new allocation = total(4) − hit_blocks(3)");
        assertEquals(4 * SPB, decode.getActiveKvTokens(),
                "occupied = (held + referenced) × spb — full demand either way");

        awaitDecodeQuiescence(decode, 30_000);
        assertEquals(0, decode.getActiveKvTokens(), "KV must net to zero");
        assertEquals(2, decode.getCompletedCount());
        decode.checkLeakDrain(0L);
        assertFalse(decode.isLeakDetected());
    }

    /**
     * Full acceptance check #2 (positive feedback): the decode hit count for
     * a growing shared prefix never decreases across rounds — round k's
     * completion extends the parked prefix, so round k+1 matches at least
     * everything round k matched.
     */
    @Test
    void decodeReusePositiveFeedbackAcrossRounds() throws Exception {
        MockPerformanceModel model = decodeModel(tempDir, 10_000.0, null);
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(model, 8);
        MockLruBlockCache cache = cacheOf(decode);

        // Round 1: prefix [1] parked by completion.
        assertTrue(scheduleDecodeCompletion(decode,
                shapeWithKeys(model, 201L, 1024, 26, List.of(1L)), -1, null));
        awaitDecodeQuiescence(decode, 30_000);
        int hitsRound1 = cache.prefixHitBlocks(List.of(1L, 2L, 3L));

        // Round 2: longer prefix [1, 2] — matches ≥ round 1's matches.
        assertTrue(scheduleDecodeCompletion(decode,
                shapeWithKeys(model, 202L, 2048, 26, List.of(1L, 2L)), -1, null));
        awaitRunningStreams(decode, 1);
        int hitsRound2 = cache.referencedKeyBlocks();
        awaitDecodeQuiescence(decode, 30_000);

        // Round 3: longest prefix [1, 2, 3] — matches ≥ round 2's matches.
        assertTrue(scheduleDecodeCompletion(decode,
                shapeWithKeys(model, 203L, 3072, 26, List.of(1L, 2L, 3L)), -1, null));
        awaitRunningStreams(decode, 1);
        int hitsRound3 = cache.referencedKeyBlocks();
        awaitDecodeQuiescence(decode, 30_000);

        assertTrue(hitsRound2 >= hitsRound1,
                "same-prefix second request must hit at least the first ("
                        + hitsRound1 + " → " + hitsRound2 + ")");
        assertTrue(hitsRound3 >= hitsRound2,
                "the growing prefix must hit monotonically more ("
                        + hitsRound2 + " → " + hitsRound3 + ")");
        assertEquals(0, decode.getActiveKvTokens());
        decode.checkLeakDrain(0L);
        assertFalse(decode.isLeakDetected());
    }

    // ─────────────── engine level: cancelled streams never admit ───────────────

    /**
     * Full acceptance check #3 (cancel wins the terminal race): a cancelled
     * running decode stream leaves NO LRU entries — production cancel runs
     * free() and the request's blocks return to the pool without handover.
     */
    @Test
    void cancelledDecodeStreamLeavesNoLruEntries() throws Exception {
        MockPerformanceModel model = decodeModel(tempDir, 10_000.0, null);
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(model, 8);
        List<Long> keys = List.of(1L, 2L, 3L);

        assertTrue(scheduleDecodeCompletion(decode,
                shapeWithKeys(model, 301L, 4096, 260, keys), -1, null));
        awaitRunningStreams(decode, 1);
        MockLruBlockCache cache = cacheOf(decode);
        assertEquals(4, cache.heldBlocks());

        EngineRpcService.TaskPhase phase = decode.cancel(301L);
        assertNotNull(phase, "the running stream must be found and cancelled");
        awaitDecodeQuiescence(decode, 60_000);

        assertTrue(cache.snapshotKeys().isEmpty(),
                "a cancelled stream must leave no LRU entries (free(), no handover)");
        assertEquals(0, decode.getActiveKvTokens());
        assertEquals(0, cache.heldBlocks());
        assertEquals(0, cache.referencedKeyBlocks());
        decode.checkLeakDrain(0L);
        assertFalse(decode.isLeakDetected());
    }

    /**
     * The other half of the cancel fix — the terminal race where the cancel
     * lands AFTER the step-boundary claim (runningTasks already emptied by
     * claimDecodeTerminalLocked) but BEFORE publishDecodeCompletion: the
     * stream completes its terminal bookkeeping with the cancelled marker
     * armed, and the pre-fix code still admitted its keys to the LRU
     * (mock hit-rate inflation). The fix routes that completion through the
     * release path instead. The window is narrow, so the test storms it with
     * late cancels at randomized offsets and asserts: whenever the cancel
     * actually landed on a live-or-claimed stream (completed count did NOT
     * advance past it), the round's keys are absent from the LRU.
     */
    @Test
    void cancelledAfterTerminalClaimNeverAdmitsKeys() throws Exception {
        // ~26 tokens = 10 steps × (500 × 0.1) ms = 500 ms per request; the
        // late-cancel offsets below sweep the final steps plus the
        // claim/publish window.
        MockPerformanceModel model = decodeModel(tempDir, 500.0, null);
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(model, 8);
        MockLruBlockCache cache = cacheOf(decode);

        int landedCancels = 0;
        for (int i = 0; i < 40; i++) {
            long rid = 1_000L + i;
            List<Long> keys = List.of(rid * 10, rid * 10 + 1, rid * 10 + 2);
            long completedBefore = decode.getCompletedCount();
            assertTrue(scheduleDecodeCompletion(decode,
                    shapeWithKeys(model, rid, 4096, 26, keys), -1, null));
            // Late cancel: 400-540 ms — inside the last steps, sometimes in
            // the claim/publish window, occasionally past it.
            Thread.sleep(400 + (i % 8) * 20);
            decode.cancel(rid);
            awaitDecodeQuiescence(decode, 30_000);

            boolean completedNormally = decode.getCompletedCount() > completedBefore;
            if (!completedNormally) {
                // The cancel landed on a live or just-claimed stream (races A
                // and B): the round's keys must NOT be in the LRU.
                landedCancels++;
                for (Long key : keys) {
                    assertFalse(cache.snapshotKeys().contains(key),
                            "cancel-landed round " + rid
                                    + " must not admit key " + key + " to the LRU");
                }
            }
        }
        // Sanity: the offsets must actually land cancels (otherwise the test
        // would be vacuously green). With 500 ms requests and cancels at
        // 400-540 ms, most rounds land.
        assertTrue(landedCancels >= 10,
                "expected at least 10 landed cancels, got " + landedCancels);
        assertEquals(0, decode.getActiveKvTokens());
        decode.checkLeakDrain(0L);
        assertFalse(decode.isLeakDetected());
    }

    // ─────────────── helpers ───────────────

    private JavaMockEngineCluster.FastRpcService newDecodeService(
            MockPerformanceModel model, int decodeMaxConcurrency) {
        int port = BASE_PORT + nextPortOffset++;
        return MockEngineTestSupport.decodeService(
                model, port, services, scheduler, decodeMaxConcurrency);
    }

    /**
     * Shape carrying hash-channel block keys AND a real output length (the
     * shared inputWithBlockKeys pins maxNewTokens=1; these tests need
     * multi-step streams).
     */
    private static MockPerformanceModel.RequestShape shapeWithKeys(
            MockPerformanceModel model, long requestId, int inputTokens,
            int outputTokens, List<Long> blockKeys) {
        EngineRpcService.GenerateInputPB input =
                MockEngineTestSupport.inputWithBlockKeys(requestId, inputTokens, blockKeys);
        EngineRpcService.GenerateInputPB withOutput = input.toBuilder()
                .setGenerateConfig(input.getGenerateConfig().toBuilder()
                        .setMaxNewTokens(outputTokens)
                        .build())
                .build();
        return model.shape(withOutput, new MockLruBlockCache(100));
    }

    private static MockLruBlockCache cacheOf(
            JavaMockEngineCluster.FastRpcService service) throws Exception {
        Field field = JavaMockEngineCluster.FastRpcService.class.getDeclaredField("cache");
        field.setAccessible(true);
        return (MockLruBlockCache) field.get(service);
    }

    /** Waits until the engine reports {@code expected} running decode streams. */
    private static void awaitRunningStreams(
            JavaMockEngineCluster.FastRpcService service, int expected)
            throws Exception {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        int seen = -1;
        while (System.nanoTime() < deadline) {
            seen = MockEngineTestSupport.activeDecodeRequests(service);
            if (seen == expected) {
                return;
            }
            Thread.sleep(5);
        }
        throw new AssertionError("expected " + expected
                + " running decode streams, last observed " + seen);
    }
}
