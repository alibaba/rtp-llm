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
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import static org.flexlb.mockengine.MockEngineTestSupport.activeDecodeRequests;
import static org.flexlb.mockengine.MockEngineTestSupport.awaitDecodeQuiescence;
import static org.flexlb.mockengine.MockEngineTestSupport.batch;
import static org.flexlb.mockengine.MockEngineTestSupport.enqueue;
import static org.flexlb.mockengine.MockEngineTestSupport.inputWithBlockKeys;
import static org.flexlb.mockengine.MockEngineTestSupport.performanceModel;
import static org.flexlb.mockengine.MockEngineTestSupport.scheduleDecodeCompletion;
import static org.flexlb.mockengine.MockEngineTestSupport.slot;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * P-enqueue decode-KV reservation net-allocation semantics (20260903) —
 * {@code reserveDecodeLease} (92bd024, called at EnqueueBatch Phase 1.6 and
 * on the direct generate_stream path) prices the D-side reservation on the
 * NET demand {@code ceil(inputLen/spb) − own-LRU prefix hits} through the
 * fix-#5 {@code acquireWithReuse} caliber, mirroring the production
 * prepare-stage ALLOCATE: reuse hits are only REFERENCED (pinned, never
 * re-allocated), so the D pool is charged the delta — never the full
 * {@code ceil(input_len/spb)} total.
 *
 * <p>Coverage (the two failure modes a full-total reservation would
 * reintroduce):
 * <ol>
 *   <li><b>Delta allocation</b> — a parked prefix on the D LRU makes the
 *       reservation hold only {@code total − hits} blocks (held) while the
 *       hits pin as references — pool pressure (eviction candidates,
 *       available blocks) stays strictly lower than a full-total charge.</li>
 *   <li><b>Net-demand gate</b> — with the prefix already referenced by an
 *       in-flight stream, a request whose TOTAL demand exceeds the
 *       available blocks still reserves: the gate evaluates
 *       {@code netNew ≤ available}, exactly the production semantics that
 *       keep LACK_MEM from firing early (a full-total gate would reject
 *       with the 8211 surface after the retry window).</li>
 * </ol>
 */
class DecodeReserveNetAllocationTest {

    private static final int SPB = 1024;
    private static final int BASE_PORT = 64200;

    @TempDir
    Path tempDir;

    private ScheduledExecutorService scheduler;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private int nextPortOffset;

    @BeforeEach
    void setUp() {
        scheduler = Executors.newScheduledThreadPool(4, runnable -> {
            Thread thread = new Thread(runnable, "decode-reserve-net-scheduler");
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

    /**
     * Delta allocation: a prior decode completion parks the 3-key prefix
     * [1, 2, 3] in the D LRU; a P-enqueue reservation with a 5-block demand
     * (input 5×spb, keys [1..5], role_addrs → the D engine) must hold only
     * the 2-block delta and pin the 3 hits as references. A full-total
     * reservation would hold 5 — the systematic decode-pool overstatement
     * this test locks out.
     */
    @Test
    void parkedPrefixMakesReservationHoldOnlyTheNetDelta() throws Exception {
        // D model: fast decode steps so the seeding stream completes quickly.
        MockPerformanceModel dModel = performanceModel(tempDir, "10", 1.0, 1.0);
        // P model: slow prefill (5000ms) keeps the reserving request pinned
        // in the running state — the reservation stays live for assertions.
        MockPerformanceModel pModel = performanceModel(tempDir, "5000", 1.0, 1.0);
        JavaMockEngineCluster.FastRpcService prefill = newPrefillService(pModel, 100);
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(dModel, 10);
        MockLruBlockCache cache = cacheOf(decode);

        // Seed: rid 900 (input 3×spb, keys [1,2,3]) completes and parks its
        // 3 keys in the decode LRU (pure-LRU, held back to 0).
        assertTrue(scheduleDecodeCompletion(decode,
                shape(dModel, 900L, 3 * SPB, 26, List.of(1L, 2L, 3L)), -1, null));
        awaitDecodeQuiescence(decode, 30_000);
        assertEquals(0, cache.heldBlocks(), "the seeding stream completed and released its holds");
        assertEquals(3, cache.lruKeyBlocks(), "the 3 prefix keys are parked in the decode LRU");

        // Reserve: rid 901 — 5-block total demand, 3-block prefix overlap,
        // routed to the D engine through role_addrs (Phase 1.6).
        EngineRpcService.EnqueueBatchResponsePB ack =
                enqueue(prefill, batch(1, slot(0, reserveInput(
                        901L, 5 * SPB, List.of(1L, 2L, 3L, 4L, 5L), decode.getGrpcPort()))));
        assertEquals(0, ack.getErrorsCount(), "the net-demand reservation must admit: " + ack);
        assertEquals(1, ack.getSuccessesCount());

        // The reservation holds ONLY the delta; the hits pin as references.
        assertEquals(2, cache.heldBlocks(),
                "held = ceil(il/spb) − own-LRU hits = 5 − 3 (a full-total reservation would hold 5)");
        assertEquals(3, cache.referencedKeyBlocks(),
                "the 3 parked hits are pinned as references, never re-allocated");
        assertEquals(5, cache.availableBlocks(),
                "available = 10 − 2 held − 3 referenced — reuse did not consume fresh capacity");
    }

    /**
     * Net-demand gate (pool-pressure relief): after the prefix [1,2,3] is
     * parked AND referenced by an in-flight decode stream (available drops
     * to 7 of 10), a reservation whose TOTAL demand is 8 blocks — above
     * availability, an outright reject under a full-total gate — still
     * admits on the 5-block net demand and holds exactly those 5 blocks.
     * This is the master-visible decode-pool pressure staying lower than
     * the full-total semantics would report.
     */
    @Test
    void reservationGatePricesNetDemandWhenTotalExceedsAvailability() throws Exception {
        // D model: 20ms steps — the seeding stream finishes in ~0.5s, the
        // in-flight holder (100 tokens) stays running well past assertions.
        MockPerformanceModel dModel = performanceModel(tempDir, "10", 1.0, 20.0);
        MockPerformanceModel pModel = performanceModel(tempDir, "5000", 1.0, 1.0);
        JavaMockEngineCluster.FastRpcService prefill = newPrefillService(pModel, 100);
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(dModel, 10);
        MockLruBlockCache cache = cacheOf(decode);

        // Seed the prefix: rid 800 completes and parks [1,2,3].
        assertTrue(scheduleDecodeCompletion(decode,
                shape(dModel, 800L, 3 * SPB, 26, List.of(1L, 2L, 3L)), -1, null));
        awaitDecodeQuiescence(decode, 30_000);

        // In-flight holder: rid 801 matches the parked prefix (3 hits → 0
        // net) and stays decoding; the hits move to the referenced layer.
        // Input 3×spb − 500 keeps the stream's whole life (input + 100
        // generated tokens) inside its 3 admitted blocks — no mid-flight
        // growth interferes with the held/referenced assertions below.
        assertTrue(scheduleDecodeCompletion(decode,
                shape(dModel, 801L, 3 * SPB - 500, 100, List.of(1L, 2L, 3L)), -1, null));
        awaitRunningDecode(decode, 1);
        assertEquals(3, cache.referencedKeyBlocks(),
                "the in-flight holder pins the parked prefix as references");
        assertEquals(7, cache.availableBlocks(), "10 − 0 held − 3 referenced");

        // Reserve: rid 802 — TOTAL demand 8 blocks (> available 7: a
        // full-total gate rejects after the retry window with the 8211
        // surface), net demand 5 (8 − 3 shared-prefix hits) fits.
        EngineRpcService.EnqueueBatchResponsePB ack =
                enqueue(prefill, batch(2, slot(0, reserveInput(
                        802L, 8 * SPB,
                        List.of(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L),
                        decode.getGrpcPort()))));
        assertEquals(0, ack.getErrorsCount(),
                "total demand 8 > available 7 must still admit on the 5-block NET demand: " + ack);
        assertEquals(1, ack.getSuccessesCount());

        // The reservation holds exactly the net delta; the shared prefix
        // stays referenced (one entry per key, two referencing requests).
        assertEquals(5, cache.heldBlocks(), "held = 8 − 3 hits = 5 net blocks");
        assertEquals(3, cache.referencedKeyBlocks(),
                "the 3 shared-prefix blocks stay referenced — pinned by both requests, counted once");
        assertEquals(2, cache.availableBlocks(), "10 − 5 held − 3 referenced");
    }

    // ─────────────── helpers ───────────────

    /** EnqueueBatch input carrying BOTH the block keys and the decode role_addr. */
    private static EngineRpcService.GenerateInputPB reserveInput(
            long requestId, int inputTokens, List<Long> blockKeys, int decodePort) {
        EngineRpcService.GenerateInputPB base =
                inputWithBlockKeys(requestId, inputTokens, blockKeys);
        return base.toBuilder()
                .setGenerateConfig(base.getGenerateConfig().toBuilder()
                        .addRoleAddrs(EngineRpcService.RoleAddrPB.newBuilder()
                                .setRole(EngineRpcService.RoleAddrPB.RoleType.DECODE)
                                .setRoleStr("DECODE")
                                .setGrpcPort(decodePort)
                                .build())
                        .build())
                .build();
    }

    /**
     * Shape carrying hash-channel block keys and a real output length (the
     * shared inputWithBlockKeys pins maxNewTokens=1; seeding streams need
     * multi-step runs so completion parks their keys in the LRU).
     */
    private static MockPerformanceModel.RequestShape shape(
            MockPerformanceModel model, long requestId, int inputTokens,
            int outputTokens, List<Long> blockKeys) {
        EngineRpcService.GenerateInputPB input =
                inputWithBlockKeys(requestId, inputTokens, blockKeys);
        EngineRpcService.GenerateInputPB withOutput = input.toBuilder()
                .setGenerateConfig(input.getGenerateConfig().toBuilder()
                        .setMaxNewTokens(outputTokens)
                        .build())
                .build();
        return model.shape(withOutput, new MockLruBlockCache(100));
    }

    private JavaMockEngineCluster.FastRpcService newPrefillService(
            MockPerformanceModel model, int blocks) {
        int port = BASE_PORT + nextPortOffset++;
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

    private JavaMockEngineCluster.FastRpcService newDecodeService(
            MockPerformanceModel model, int blocks) {
        int port = BASE_PORT + nextPortOffset++;
        JavaMockEngineCluster.FastRpcService service =
                new JavaMockEngineCluster.FastRpcService(
                        "decode-" + port,
                        "127.0.0.1",
                        "decode",
                        EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE,
                        port,
                        services,
                        scheduler,
                        model,
                        blocks,
                        new JavaMockEngineCluster.ClusterStats(),
                        10_000_000L,
                        8);
        services.put(port, service);
        return service;
    }

    private static MockLruBlockCache cacheOf(
            JavaMockEngineCluster.FastRpcService service) throws Exception {
        Field field = JavaMockEngineCluster.FastRpcService.class.getDeclaredField("cache");
        field.setAccessible(true);
        return (MockLruBlockCache) field.get(service);
    }

    /** Waits until the engine reports {@code expected} running decode streams. */
    private static void awaitRunningDecode(
            JavaMockEngineCluster.FastRpcService service, int expected)
            throws Exception {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        int seen = -1;
        while (System.nanoTime() < deadline) {
            seen = activeDecodeRequests(service);
            if (seen == expected) {
                return;
            }
            Thread.sleep(5);
        }
        throw new AssertionError("expected " + expected
                + " running decode streams, last observed " + seen);
    }
}
