package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.flexlb.mockengine.MockEngineTestSupport.batch;
import static org.flexlb.mockengine.MockEngineTestSupport.enqueue;
import static org.flexlb.mockengine.MockEngineTestSupport.httpGet;
import static org.flexlb.mockengine.MockEngineTestSupport.inputWithBlockKeys;
import static org.flexlb.mockengine.MockEngineTestSupport.performanceModel;
import static org.flexlb.mockengine.MockEngineTestSupport.slot;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Block-pool observability series (KV capacity model v2) the mock reports on
 * {@code /metrics} in BOTH emission modes (per-engine and role-aggregated):
 *
 * <ul>
 *   <li>{@code mock_engine_cache_blocks} / {@code available_blocks} /
 *       {@code held_blocks} / {@code referenced_blocks} — the three-state block
 *       decomposition as per-scrape GAUGES (prefill leases hold keyless blocks
 *       mid-flight; completion hands them to the LRU restoring availability;
 *       decode reuse pins hit keys as references).</li>
 *   <li>{@code mock_engine_lack_mem_rejects_total} — the PERMANENT-family
 *       LACK_MEM surface: prefill requests synchronously rejected with 602
 *       (the enqueue-batch Phase-1.5 P-pool gate) AND decode-side failures
 *       whose demand can NEVER fit (required + reserve > physical total —
 *       the production KVCacheAllocator PERMANENT verdict), whether rejected
 *       by the P-pool gate or by an ALLOCATE retry window exhausted on D.
 *       Distinct from {@code mock_engine_kv_admission_fails_total}, which
 *       counts the RETRYABLE-family DECODE-side REQUEST TERMINAL LACK_MEM
 *       failures (admission / growth / reservation rejects — pool total
 *       sufficient, temporarily short; production-aligned 20260903 — the
 *       former un-pooled degradation era is retired): prefill REJECTS,
 *       decode TERMINATES — the healthy-run contract is both stay 0, only
 *       overload runs light them up, and each on its own role.</li>
 *   <li>{@code mock_engine_decode_reuse_blocks_total} — cumulative counter of
 *       the fix #5 net-demand deduction (acquireWithReuse hit keys against the
 *       engine's OWN LRU): a CUMULATIVE counter, never drained (unlike the
 *       rtp_llm_* window series) — the reuse savings only ever add up.</li>
 * </ul>
 */
class BlockPoolMetricsObservabilityTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static final int SPB = 1024;
    private static final int BASE_PORT = 63900;

    /** {@code metric_name{engine_name=...,role=...,grpc_port="N",...} value} */
    private static final Pattern PER_ENGINE_METRIC_PATTERN = Pattern.compile(
            "(\\w+)\\{engine_name=\"[^\"]+\",role=\"[^\"]+\",grpc_port=\"(\\d+)\",engine_ip=\"[^\"]+\"\\}\\s+(\\d+)");

    /** {@code metric_name{role="..."} value} */
    private static final Pattern ROLE_METRIC_PATTERN = Pattern.compile(
            "(\\w+)\\{role=\"(\\w+)\"\\}\\s+(\\d+)");

    @TempDir
    Path tempDir;

    private ScheduledExecutorService scheduler;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private MockControlServer controlServer;
    private int nextPortOffset;

    @BeforeEach
    void setUp() throws IOException {
        scheduler = Executors.newScheduledThreadPool(4, runnable -> {
            Thread thread = new Thread(runnable, "block-pool-metrics-scheduler");
            thread.setDaemon(true);
            return thread;
        });
        services = new ConcurrentHashMap<>();
        nextPortOffset = 0;
    }

    @AfterEach
    void tearDown() throws InterruptedException {
        if (controlServer != null) {
            controlServer.stop();
            controlServer = null;
        }
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            service.shutdown();
        }
        scheduler.shutdownNow();
        scheduler.awaitTermination(3, TimeUnit.SECONDS);
    }

    /**
     * The prefill LACK_MEM rejection surface: an oversized request (11 blocks
     * vs the 10-block pool) is rejected synchronously with error 602 AND
     * counted by mock_engine_lack_mem_rejects_total in both emission modes —
     * while kv_admission_fails (the DECODE-side terminal-failure counter)
     * stays 0, the two surfaces must never cross-book.
     */
    @Test
    void lackMemRejectCounterCountsPrefillSyncRejections() throws Exception {
        MockPerformanceModel model = performanceModel(tempDir, "10");
        JavaMockEngineCluster.FastRpcService prefill = newPrefillService(model, 10);

        EngineRpcService.GenerateInputPB tooBig = inputWithBlockKeys(
                7L, SPB, List.of(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L, 11L));
        EngineRpcService.EnqueueBatchResponsePB ack =
                enqueue(prefill, batch(1, slot(0, tooBig)));
        assertEquals(1, ack.getErrorsCount(), "the oversized request must be rejected");
        assertEquals(JavaMockEngineCluster.LACK_MEM_ERROR_CODE,
                ack.getErrors(0).getErrorInfo().getErrorCode());

        // Per-engine mode: the rejection is on the prefill engine, and the
        // decode-flavored admission-fail counter stays 0.
        Map<String, Map<Integer, Long>> perEngine =
                parsePerEngineMetrics(httpGet(controlPort(), "/metrics?per_engine=true"));
        int prefillPort = prefill.getGrpcPort();
        assertEquals(1L,
                perEngine.get("mock_engine_lack_mem_rejects_total")
                        .getOrDefault(prefillPort, -1L),
                "one synchronous 602 rejection must be counted");
        assertEquals(0L,
                perEngine.get("mock_engine_kv_admission_fails_total")
                        .getOrDefault(prefillPort, -1L),
                "prefill rejections must NOT enter the decode terminal-failure counter");

        // Role-aggregated mode: prefill bucket carries the reject, the
        // decode bucket is absent (no decode engine in this cluster — the
        // rejection surface is prefill-only), and the pool gauge is role-summed.
        Map<String, Map<String, Long>> byRole =
                parseRoleMetrics(httpGet(controlPort(), "/metrics"));
        assertEquals(1L, byRole.get("mock_engine_lack_mem_rejects_total")
                        .getOrDefault("prefill", -1L),
                "aggregated prefill bucket must carry the 602 rejection");
        assertEquals(0L, byRole.get("mock_engine_kv_admission_fails_total")
                        .getOrDefault("prefill", -1L),
                "aggregated prefill bucket must carry no decode terminal failure");
        assertEquals(null, byRole.get("mock_engine_lack_mem_rejects_total").get("decode"),
                "the decode bucket must carry no prefill rejection");
        assertEquals(null, byRole.get("mock_engine_kv_admission_fails_total").get("decode"),
                "the decode bucket must carry no admission failure");
        assertEquals(10L, byRole.get("mock_engine_cache_blocks").getOrDefault("prefill", -1L),
                "aggregated cache_blocks = the role's pool total");

        // A serviceable request afterwards: the counter holds (no double
        // counting) — the pool is not poisoned by the rejection.
        EngineRpcService.GenerateInputPB small =
                inputWithBlockKeys(8L, SPB, List.of(1L, 2L));
        assertEquals(0, enqueue(prefill, batch(2, slot(0, small))).getErrorsCount());
        Map<String, Map<Integer, Long>> after =
                parsePerEngineMetrics(httpGet(controlPort(), "/metrics?per_engine=true"));
        assertEquals(1L, after.get("mock_engine_lack_mem_rejects_total")
                        .getOrDefault(prefillPort, -1L),
                "an admitted request must not bump the reject counter");
    }

    /**
     * The three-state gauge decomposition over a prefill request's life:
     * mid-flight the lease holds its keyless blocks (held up, available
     * down); completion hands them to the LRU and availability is restored
     * (release != delete — pure-LRU blocks count as available).
     */
    @Test
    void blockPoolGaugesTrackThreeStateDecomposition() throws Exception {
        // Slow prefill (800ms) for a stable mid-flight observation window.
        MockPerformanceModel model = performanceModel(tempDir, "800");
        JavaMockEngineCluster.FastRpcService prefill = newPrefillService(model, 100);

        // in-flight asserted via snapshot before the metrics scrape so the
        // gauge reading cannot race the completion.
        assertEquals(0, enqueue(prefill, batch(3, slot(0,
                inputWithBlockKeys(30L, 3 * SPB, List.of(31L, 32L, 33L)))))
                .getErrorsCount());
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (System.nanoTime() < deadline && prefill.getInflightCount() < 1) {
            Thread.sleep(5);
        }
        assertEquals(1, prefill.getInflightCount(), "the request must be in flight");

        Map<String, Map<Integer, Long>> mid =
                parsePerEngineMetrics(httpGet(controlPort(), "/metrics?per_engine=true"));
        int prefillPort = prefill.getGrpcPort();
        assertEquals(100L, mid.get("mock_engine_cache_blocks").getOrDefault(prefillPort, -1L),
                "the pool size gauge is the configured block count");
        assertEquals(3L, mid.get("mock_engine_held_blocks").getOrDefault(prefillPort, -1L),
                "the in-flight lease holds its 3 keyless blocks");
        assertEquals(0L, mid.get("mock_engine_referenced_blocks")
                        .getOrDefault(prefillPort, -1L),
                "fresh keys carry no references");
        assertEquals(97L, mid.get("mock_engine_available_blocks")
                        .getOrDefault(prefillPort, -1L),
                "held blocks leave the available set");

        while (System.nanoTime() < deadline && prefill.getInflightCount() > 0) {
            Thread.sleep(5);
        }
        assertEquals(0, prefill.getInflightCount(), "the request must have completed");

        Map<String, Map<Integer, Long>> done =
                parsePerEngineMetrics(httpGet(controlPort(), "/metrics?per_engine=true"));
        assertEquals(0L, done.get("mock_engine_held_blocks").getOrDefault(prefillPort, -1L),
                "completion returns the lease");
        assertEquals(0L, done.get("mock_engine_referenced_blocks")
                        .getOrDefault(prefillPort, -1L),
                "no in-flight references remain");
        assertEquals(100L, done.get("mock_engine_available_blocks")
                        .getOrDefault(prefillPort, -1L),
                "parked pure-LRU keys count as available (release != delete)");
    }

    /**
     * The decode reuse counter: cold round parks keys in the decode engine's
     * OWN LRU (no reuse yet); the warm round re-matches them (referenced =
     * hit keys, held = net-new only) and the CUMULATIVE counter picks up the
     * hit blocks — the counter is never drained, so round 3 adds again.
     */
    @Test
    void decodeReuseCounterAccumulatesOwnLruHits() throws Exception {
        MockPerformanceModel model =
                MockEngineTestSupport.decodeModel(tempDir, 10_000.0, null);
        JavaMockEngineCluster.FastRpcService decode =
                MockEngineTestSupport.decodeService(
                        model, BASE_PORT + 50, services, scheduler, 8);
        startControlServer();
        int decodePort = decode.getGrpcPort();
        List<Long> keys = List.of(1L, 2L, 3L);

        // Round 1 (cold): 4096 tokens -> T=4, keys cover 3, hits 0. The
        // completion parks the keys in the decode LRU.
        assertTrue(MockEngineTestSupport.scheduleDecodeCompletion(decode,
                shapeWithKeys(model, 101L, 4096, 26, keys), -1, null));
        MockEngineTestSupport.awaitDecodeQuiescence(decode, 30_000);
        Map<String, Map<Integer, Long>> cold =
                parsePerEngineMetrics(httpGet(controlPort(), "/metrics?per_engine=true"));
        assertEquals(0L, cold.get("mock_engine_decode_reuse_blocks_total")
                        .getOrDefault(decodePort, -1L),
                "a cold round has no reuse to count");
        assertEquals(0L, cold.get("mock_engine_kv_admission_fails_total")
                        .getOrDefault(decodePort, -1L),
                "a healthy warm-up admission never terminates on KV");

        // Round 2 (warm): 3 hits referenced + 1 net-new held. Mid-flight the
        // referenced gauge shows the pinned reuse; on completion the CUMULATIVE
        // counter picks up the 3 hit blocks.
        assertTrue(MockEngineTestSupport.scheduleDecodeCompletion(decode,
                shapeWithKeys(model, 102L, 4096, 26, keys), -1, null));
        long warmDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        while (System.nanoTime() < warmDeadline
                && MockEngineTestSupport.activeDecodeRequests(decode) < 1) {
            Thread.sleep(5);
        }
        Map<String, Map<Integer, Long>> warm =
                parsePerEngineMetrics(httpGet(controlPort(), "/metrics?per_engine=true"));
        assertEquals(3L, warm.get("mock_engine_referenced_blocks")
                        .getOrDefault(decodePort, -1L),
                "the 3 reused keys are pinned as references mid-flight");
        assertEquals(1L, warm.get("mock_engine_held_blocks").getOrDefault(decodePort, -1L),
                "net new allocation = total(4) - hit_blocks(3)");
        MockEngineTestSupport.awaitDecodeQuiescence(decode, 30_000);
        Map<String, Map<Integer, Long>> warmDone =
                parsePerEngineMetrics(httpGet(controlPort(), "/metrics?per_engine=true"));
        assertEquals(3L, warmDone.get("mock_engine_decode_reuse_blocks_total")
                        .getOrDefault(decodePort, -1L),
                "the warm round's 3 hit blocks must be accumulated (cumulative, "
                        + "never drained)");

        // Round 3 (same keys): the counter adds AGAIN — cumulative semantics,
        // unlike the per-window rtp_llm_* drain series.
        assertTrue(MockEngineTestSupport.scheduleDecodeCompletion(decode,
                shapeWithKeys(model, 103L, 4096, 26, keys), -1, null));
        MockEngineTestSupport.awaitDecodeQuiescence(decode, 30_000);
        Map<String, Map<Integer, Long>> third =
                parsePerEngineMetrics(httpGet(controlPort(), "/metrics?per_engine=true"));
        assertEquals(6L, third.get("mock_engine_decode_reuse_blocks_total")
                        .getOrDefault(decodePort, -1L),
                "reuse savings only ever add up (3 + 3)");
    }

    /**
     * Decode KV admission failure is a REQUEST TERMINAL FAILURE
     * (production-aligned, 20260903 — the former un-pooled degradation is
     * retired): an oversized decode request (11 blocks vs a 10-block pool,
     * cold LRU so the net demand alone overflows) cannot provision its lease
     * at hand-off admission. scheduleDecodeCompletion ACCEPTS and TERMINATES
     * the request: the typed terminal carries the master-surface error 8211
     * (DECODE_MALLOC_FAILED — the production P-side closeGrpcStream rewrite;
     * the raw 602 stays in the message text), the engine_events decode_done
     * row carries error_code=8211 with cancelled=false, an error frame closes
     * the client stream, the lifecycle ends "failed", and because 11 +
     * reserve > 10 total the failure classifies PERMANENT — lack_mem_rejects
     * counts 1 on the D engine (kv_admission_fails stays 0) — and every
     * run-start claim (slot / pendingRequests / runningTasks) is rolled back
     * — zero pool residue, no leak.
     */
    @Test
    void decodeAdmissionFailureIsRequestTerminalLackMem() throws Exception {
        MockPerformanceModel model = MockEngineTestSupport.decodeModel(tempDir, 20.0, null);
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(model, 10);
        Path eventsFile = tempDir.resolve("engine_events.jsonl");
        decode.setEngineEventLog(
                JavaMockEngineCluster.EngineEventLog.open(eventsFile.toString()));
        int decodePort = decode.getGrpcPort();

        // 11 blocks demanded vs a 10-block pool, cold D LRU (no reuse): the
        // net demand exceeds availability outright.
        LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> queue =
                new LinkedBlockingQueue<>();
        assertTrue(MockEngineTestSupport.scheduleDecodeCompletion(decode,
                shapeWithKeys(model, 201L, 11 * SPB, 4,
                        List.of(201L, 202L, 203L, 204L, 205L,
                                206L, 207L, 208L, 209L, 210L, 211L)),
                -1, queue),
                "the admission failure still ACCEPTS the request (accepted-and-terminal)");

        // The client stream closes with an error frame: RpcErrorPB has no
        // LACK_MEM enum face, so the frame carries UNKNOWN_ERROR and the
        // numeric 602 contract travels in the message.
        EngineRpcService.GenerateOutputsPB frame = queue.poll(10, TimeUnit.SECONDS);
        assertNotNull(frame, "an error frame must close the stream");
        assertTrue(frame.hasErrorInfo(), "the closing frame is an error frame");
        assertEquals(EngineRpcService.ErrorCodePB.UNKNOWN_ERROR,
                frame.getErrorInfo().getErrorCode());
        assertTrue(frame.getErrorInfo().getErrorMessage().contains("LACK_MEM (602)"),
                "the message must carry the 602 contract: "
                        + frame.getErrorInfo().getErrorMessage());
        assertTrue(frame.getErrorInfo().getErrorMessage().contains("admission"),
                "the message must mark the failure stage: "
                        + frame.getErrorInfo().getErrorMessage());

        // Typed terminal: the master-facing completion carries error 602.
        EngineRpcService.WorkerStatusPB status =
                MockEngineTestSupport.workerStatus(decode, 0);
        assertEquals(1, status.getFinishedTaskListCount(),
                "exactly one typed terminal must be published");
        EngineRpcService.TaskInfoPB terminal = status.getFinishedTaskList(0);
        assertEquals(201L, terminal.getRequestId());
        assertTrue(terminal.hasErrorInfo(), "the terminal must carry error info");
        assertEquals(JavaMockEngineCluster.DECODE_LACK_MEM_ERROR_CODE,
                terminal.getErrorInfo().getErrorCode(),
                "decode terminals carry the master-surface 8211");
        assertTrue(terminal.getErrorInfo().getErrorMessage().contains("602"),
                "the raw 602 travels in the message text: "
                        + terminal.getErrorInfo().getErrorMessage());
        // Lifecycle: the backfilled arrival row ends "failed" (/requests).
        assertEquals("failed", decode.getRequestLifecycleSnapshot()
                        .get("201").get("end_state"),
                "the lifecycle must end failed");

        // engine_events decode_done row: cancelled=false + error_code=8211
        // (the aggregate-side skip key).
        List<JsonNode> rows = readEventRows(eventsFile);
        assertEquals(1, rows.size(), "exactly one decode_done row");
        JsonNode row = rows.get(0);
        assertEquals("decode_done", row.path("event").asText());
        assertEquals(8211L, row.path("error_code").asLong(),
                "the failure row must carry error_code=8211 (master surface)");
        assertEquals(false, row.path("cancelled").asBoolean());

        // D-side accounting + full rollback: 11 + reserve > 10 total →
        // PERMANENT family → lack_mem_rejects 1 / kv_admission_fails 0,
        // zero residue.
        Map<String, Map<Integer, Long>> metrics =
                parsePerEngineMetrics(httpGet(controlPort(), "/metrics?per_engine=true"));
        assertEquals(1L, metrics.get("mock_engine_lack_mem_rejects_total")
                        .getOrDefault(decodePort, -1L),
                "the PERMANENT admission failure counts in lack_mem_rejects");
        assertEquals(0L, metrics.get("mock_engine_kv_admission_fails_total")
                        .getOrDefault(decodePort, -1L),
                "a never-fits failure must NOT count as RETRYABLE");
        assertEquals(0L, metrics.get("mock_engine_held_blocks")
                        .getOrDefault(decodePort, -1L),
                "no lease residue on the pool");
        assertEquals(10L, metrics.get("mock_engine_available_blocks")
                        .getOrDefault(decodePort, -1L),
                "the pool is fully available again");
        assertEquals(0, MockEngineTestSupport.activeDecodeRequests(decode),
                "the run-start slot claim is rolled back");
        assertEquals(0, decode.getRunningCount(), "no running-task residue");
        assertEquals(0, decode.getInflightCount(), "no pending residue");
        assertFalse(decode.isLeakDetected(), "no leak detected");
    }

    /**
     * P-enqueue decode-KV reservation reject (the mock counterpart of
     * production's prepare-stage ALLOCATE rejection): a request that fits
     * the PREFILL pool but overflows the role_addrs-targeted DECODE pool is
     * rejected SYNCHRONOUSLY in the enqueue ack with the master-surface 8211
     * (raw 602 in the message text), the message marks the decode-side
     * allocation, the P lease is released (zero P residue), and because 11 +
     * reserve > 10 total the failure classifies PERMANENT — it counts on the
     * DECODE engine's lack_mem_rejects (never in kv_admission_fails, and the
     * P engine's own counters stay clean — the P pool did not reject), and
     * the D pool keeps no residue.
     */
    @Test
    void enqueueDecodeReservationRejectIsSynchronous8211() throws Exception {
        MockPerformanceModel model = performanceModel(tempDir, "10");
        JavaMockEngineCluster.FastRpcService prefill = newPrefillService(model, 100);
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(model, 10);
        int prefillPort = prefill.getGrpcPort();
        int decodePort = decode.getGrpcPort();

        // 11 blocks fit the 100-block P pool but overflow the 10-block D pool.
        EngineRpcService.GenerateInputPB tooBigForDecode = inputWithDecodeAndKeys(
                301L, 11 * SPB, decodePort, 4,
                List.of(301L, 302L, 303L, 304L, 305L,
                        306L, 307L, 308L, 309L, 310L, 311L));
        EngineRpcService.EnqueueBatchResponsePB ack =
                enqueue(prefill, batch(6, slot(0, tooBigForDecode)));
        assertEquals(1, ack.getErrorsCount(),
                "the D-pool overflow must reject the request synchronously");
        assertEquals(JavaMockEngineCluster.DECODE_LACK_MEM_ERROR_CODE,
                ack.getErrors(0).getErrorInfo().getErrorCode(),
                "decode-side reservation rejects carry the master-surface 8211");
        assertTrue(ack.getErrors(0).getErrorInfo().getErrorMessage().contains("602"),
                "the raw 602 travels in the message text: "
                        + ack.getErrors(0).getErrorInfo().getErrorMessage());
        assertTrue(ack.getErrors(0).getErrorInfo().getErrorMessage()
                        .contains("decode-side KV allocation rejected by D engine port="
                                + decodePort),
                "the error message must mark the decode-side surface: "
                        + ack.getErrors(0).getErrorInfo().getErrorMessage());

        Map<String, Map<Integer, Long>> metrics =
                parsePerEngineMetrics(httpGet(controlPort(), "/metrics?per_engine=true"));
        assertEquals(1L, metrics.get("mock_engine_lack_mem_rejects_total")
                        .getOrDefault(decodePort, -1L),
                "the PERMANENT reservation reject counts on the D engine");
        assertEquals(0L, metrics.get("mock_engine_kv_admission_fails_total")
                        .getOrDefault(decodePort, -1L),
                "a never-fits failure must NOT count as RETRYABLE");
        assertEquals(0L, metrics.get("mock_engine_lack_mem_rejects_total")
                        .getOrDefault(prefillPort, -1L),
                "the P-pool rejection counter stays clean (the P pool did not reject)");
        assertEquals(0L, metrics.get("mock_engine_held_blocks")
                        .getOrDefault(prefillPort, -1L),
                "the P lease is released (no P residue)");
        assertEquals(0L, metrics.get("mock_engine_held_blocks")
                        .getOrDefault(decodePort, -1L),
                "no D lease residue");
        assertEquals(10L, metrics.get("mock_engine_available_blocks")
                        .getOrDefault(decodePort, -1L),
                "the D pool is untouched (the failed acquire changed no state)");
        assertEquals(0, MockEngineTestSupport.activeDecodeRequests(decode),
                "nothing was ever admitted on D");
    }

    /**
     * P-enqueue reservation lifecycle (production-aligned, 20260903):
     * (a) the reservation charges the D pool AT ENQUEUE — the request's
     *     net-new blocks are held while the prefill still executes;
     * (b) hand-off ADOPTS the reservation without re-charging — the decoding
     *     stream runs on the SAME blocks (plus per-step growth), never
     *     doubled;
     * (c) normal completion admits the lease into the LRU (pure-LRU keys
     *     count as available again);
     * (d) a prefill cancel mid-flight releases BOTH the P lease and the D
     *     reservation — the cancel-loop closure leaves no leaked accounting.
     */
    @Test
    void reservationLifecycleAdoptionAndCancelClosure() throws Exception {
        // Slow prefill (800 ms) for a stable mid-flight window; 50 ms decode
        // steps keep the decoding phase observable (26 output tokens ≈ 1.3 s).
        MockPerformanceModel model = performanceModel(tempDir, "800", 1.0, 50.0);
        JavaMockEngineCluster.FastRpcService prefill = newPrefillService(model, 100);
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(model, 20);
        int prefillPort = prefill.getGrpcPort();
        int decodePort = decode.getGrpcPort();

        // (a) Reservation charged at enqueue: 4 blocks (cold D LRU, no reuse).
        EngineRpcService.GenerateInputPB first = inputWithDecodeAndKeys(
                401L, 4 * SPB, decodePort, 26,
                List.of(401L, 402L, 403L, 404L));
        assertEquals(0, enqueue(prefill, batch(7, slot(0, first))).getErrorsCount());
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(20);
        while (System.nanoTime() < deadline && prefill.getInflightCount() < 1) {
            Thread.sleep(5);
        }
        assertEquals(1, prefill.getInflightCount(), "the prefill must be in flight");
        Map<String, Map<Integer, Long>> reserved =
                parsePerEngineMetrics(httpGet(controlPort(), "/metrics?per_engine=true"));
        assertEquals(4L, reserved.get("mock_engine_held_blocks")
                        .getOrDefault(prefillPort, -1L),
                "the P lease holds the request's 4 keyless blocks mid-prefill");
        assertEquals(4L, reserved.get("mock_engine_held_blocks")
                        .getOrDefault(decodePort, -1L),
                "the P-enqueue reservation holds the SAME 4 blocks on the D pool");
        assertEquals(16L, reserved.get("mock_engine_available_blocks")
                        .getOrDefault(decodePort, -1L),
                "reserved blocks leave the D pool's available set");
        assertEquals(0L, reserved.get("mock_engine_kv_admission_fails_total")
                        .getOrDefault(decodePort, -1L),
                "a serviceable reservation is not a failure");

        // (b) Adoption at hand-off: the decoding stream runs on the SAME 4
        // blocks (+ per-step growth) — if adoption re-charged, held would
        // jump to 8+. Quiescence cannot be awaited here (it would race past
        // the observation window), so poll the running window instead.
        while (System.nanoTime() < deadline
                && MockEngineTestSupport.activeDecodeRequests(decode) < 1) {
            Thread.sleep(5);
        }
        Map<String, Map<Integer, Long>> decoding =
                parsePerEngineMetrics(httpGet(controlPort(), "/metrics?per_engine=true"));
        long heldWhileDecoding = decoding.get("mock_engine_held_blocks")
                .getOrDefault(decodePort, -1L);
        assertTrue(heldWhileDecoding >= 4L && heldWhileDecoding <= 5L,
                "adoption must reuse the reserved blocks (held 4..5 with the "
                        + "26-token growth, never doubled 8+), got " + heldWhileDecoding);
        assertEquals(20L - heldWhileDecoding,
                decoding.get("mock_engine_available_blocks")
                        .getOrDefault(decodePort, -1L),
                "the D pool keeps capacity conservation while decoding");

        // (c) Normal completion: the lease admits into the LRU — pure-LRU
        // keys count as available again, the happy path never failed.
        MockEngineTestSupport.awaitDecodeQuiescence(decode, 30_000);
        Map<String, Map<Integer, Long>> done =
                parsePerEngineMetrics(httpGet(controlPort(), "/metrics?per_engine=true"));
        assertEquals(0L, done.get("mock_engine_held_blocks").getOrDefault(decodePort, -1L),
                "completion returns the lease");
        assertEquals(20L, done.get("mock_engine_available_blocks")
                        .getOrDefault(decodePort, -1L),
                "parked pure-LRU keys count as available (admit != delete)");
        assertEquals(0L, done.get("mock_engine_kv_admission_fails_total")
                        .getOrDefault(decodePort, -1L),
                "the happy path never fails");
        assertEquals("completed", decode.getRequestLifecycleSnapshot()
                        .get("401").get("end_state"),
                "the decode request ran to normal completion (not failed)");

        // (d) Cancel closure: a second reservation mid-prefill is fully
        // released — both the P lease and the D-side reservation.
        EngineRpcService.GenerateInputPB second = inputWithDecodeAndKeys(
                402L, 4 * SPB, decodePort, 26,
                List.of(405L, 406L, 407L, 408L));
        assertEquals(0, enqueue(prefill, batch(8, slot(0, second))).getErrorsCount());
        while (System.nanoTime() < deadline && prefill.getInflightCount() < 1) {
            Thread.sleep(5);
        }
        Map<String, Map<Integer, Long>> secondReserved =
                parsePerEngineMetrics(httpGet(controlPort(), "/metrics?per_engine=true"));
        assertEquals(4L, secondReserved.get("mock_engine_held_blocks")
                        .getOrDefault(decodePort, -1L),
                "the second reservation holds its blocks on the D pool");
        prefill.cancel(402L);
        long cancelDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        Map<String, Map<Integer, Long>> cancelled = null;
        while (System.nanoTime() < cancelDeadline) {
            cancelled = parsePerEngineMetrics(
                    httpGet(controlPort(), "/metrics?per_engine=true"));
            if (cancelled.get("mock_engine_held_blocks")
                    .getOrDefault(decodePort, -1L) == 0L) {
                break;
            }
            Thread.sleep(5);
        }
        assertNotNull(cancelled, "metrics must have been scraped");
        assertEquals(0L, cancelled.get("mock_engine_held_blocks")
                        .getOrDefault(decodePort, -1L),
                "the cancel closure releases the reserved D blocks (no leak)");
        assertEquals(0L, cancelled.get("mock_engine_held_blocks")
                        .getOrDefault(prefillPort, -1L),
                "the P lease is released too");
        assertEquals(0, MockEngineTestSupport.activeDecodeRequests(decode),
                "the cancelled request never ran on D");
        assertFalse(prefill.isLeakDetected(), "no leak on P");
        assertFalse(decode.isLeakDetected(), "no leak on D");
    }

    // ────────────────── helpers ──────────────────

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
        startControlServer();
        return service;
    }

    private void startControlServer() {
        if (controlServer == null) {
            try {
                controlServer = new MockControlServer(
                        services, new ConcurrentHashMap<>(), null, null, "127.0.0.1", 0);
                controlServer.start();
            } catch (IOException e) {
                throw new IllegalStateException("control server failed to start", e);
            }
        }
    }

    private int controlPort() {
        assertNotNull(controlServer, "control server must be running");
        return controlServer.getPort();
    }

    /** Shape carrying hash-channel block keys AND a real output length. */
    private static MockPerformanceModel.RequestShape shapeWithKeys(
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

    /**
     * Decode service with a SIZABLE block pool (the shared decodeService
     * helper hardcodes 100 blocks — the failure-path tests need small pools
     * to force admission / reservation rejects).
     */
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
        startControlServer();
        return service;
    }

    /**
     * Input carrying BOTH the decode routing (role_addrs DECODE + grpc
     * port — the reservation's D-locating source, the same source startDecode
     * uses at hand-off) and hash-channel block keys.
     */
    private static EngineRpcService.GenerateInputPB inputWithDecodeAndKeys(
            long requestId, int inputTokens, int decodePort, int outputTokens,
            List<Long> blockKeys) {
        EngineRpcService.GenerateInputPB base =
                inputWithBlockKeys(requestId, inputTokens, blockKeys);
        return base.toBuilder()
                .setGenerateConfig(base.getGenerateConfig().toBuilder()
                        .setMaxNewTokens(outputTokens)
                        .addRoleAddrs(EngineRpcService.RoleAddrPB.newBuilder()
                                .setRole(EngineRpcService.RoleAddrPB.RoleType.DECODE)
                                .setRoleStr("DECODE")
                                .setGrpcPort(decodePort)
                                .build())
                        .build())
                .build();
    }

    /** Parse the engine_events.jsonl rows (blank lines skipped). */
    private static List<JsonNode> readEventRows(Path eventsFile) throws IOException {
        List<JsonNode> rows = new ArrayList<>();
        for (String line : Files.readAllLines(eventsFile, StandardCharsets.UTF_8)) {
            if (!line.isBlank()) {
                rows.add(MAPPER.readTree(line));
            }
        }
        return rows;
    }

    private static Map<String, Map<Integer, Long>> parsePerEngineMetrics(String body) {
        Map<String, Map<Integer, Long>> result = new java.util.HashMap<>();
        Matcher matcher = PER_ENGINE_METRIC_PATTERN.matcher(body);
        while (matcher.find()) {
            result.computeIfAbsent(matcher.group(1), k -> new java.util.HashMap<>())
                    .put(Integer.parseInt(matcher.group(2)), Long.parseLong(matcher.group(3)));
        }
        return result;
    }

    private static Map<String, Map<String, Long>> parseRoleMetrics(String body) {
        Map<String, Map<String, Long>> result = new java.util.HashMap<>();
        Matcher matcher = ROLE_METRIC_PATTERN.matcher(body);
        while (matcher.find()) {
            result.computeIfAbsent(matcher.group(1), k -> new java.util.HashMap<>())
                    .put(matcher.group(2), Long.parseLong(matcher.group(3)));
        }
        return result;
    }
}
