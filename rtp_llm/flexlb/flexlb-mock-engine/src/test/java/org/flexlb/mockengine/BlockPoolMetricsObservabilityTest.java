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
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.flexlb.mockengine.MockEngineTestSupport.batch;
import static org.flexlb.mockengine.MockEngineTestSupport.enqueue;
import static org.flexlb.mockengine.MockEngineTestSupport.httpGet;
import static org.flexlb.mockengine.MockEngineTestSupport.inputWithBlockKeys;
import static org.flexlb.mockengine.MockEngineTestSupport.performanceModel;
import static org.flexlb.mockengine.MockEngineTestSupport.slot;
import static org.junit.jupiter.api.Assertions.assertEquals;
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
 *   <li>{@code mock_engine_lack_mem_rejects_total} — prefill requests
 *       synchronously rejected with LACK_MEM 602 (the enqueue-batch
 *       Phase-1.5 gate). Distinct from {@code mock_engine_kv_admission_fails_total},
 *       which counts DECODE degradations: prefill REJECTS, decode DEGRADES —
 *       the healthy-run contract is both stay 0, only overload runs light them
 *       up, and each on its own role.</li>
 *   <li>{@code mock_engine_decode_reuse_blocks_total} — cumulative counter of
 *       the fix #5 net-demand deduction (acquireWithReuse hit keys against the
 *       engine's OWN LRU): a CUMULATIVE counter, never drained (unlike the
 *       rtp_llm_* window series) — the reuse savings only ever add up.</li>
 * </ul>
 */
class BlockPoolMetricsObservabilityTest {

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
     * while kv_admission_fails (the DECODE degradation counter) stays 0, the
     * two surfaces must never cross-book.
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
                "prefill rejections must NOT enter the decode degradation counter");

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
                "aggregated prefill bucket must carry no decode degradation");
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
                "a healthy warm-up admission never degrades");

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
