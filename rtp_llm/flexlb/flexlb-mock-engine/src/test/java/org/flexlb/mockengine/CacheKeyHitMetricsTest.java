package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
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
 * Key-level cache-hit observability (the production
 * recent_cache_key_hit_count / total_count caliber) the mock reports as two
 * cumulative counters on {@code /metrics} in BOTH emission modes plus the
 * {@code /snapshot} terminal state:
 *
 * <ul>
 *   <li>{@code mock_engine_cache_key_hits_total} — Σ raw prefix-match run
 *       lengths (matched key count) recorded at the prefill admission hit
 *       computation (MockPerformanceModel.shape's prefixHitBlocks call).</li>
 *   <li>{@code mock_engine_cache_keys_requested_total} — Σ request blockKeys
 *       sizes observed at the same point; an empty-bh request adds 0/0 and
 *       never contributes.</li>
 * </ul>
 *
 * <p>Fixture shape (the known-hit structure the report chain test mirrors):
 * a cold request parks keys [k1,k2,k3] in the LRU; the warm re-request hits
 * all 3; a [k1,k2,k9] request hits only the 2-key prefix (the k9 miss
 * truncates the run) — cumulative 5 hits over 9 requested keys.
 */
class CacheKeyHitMetricsTest {

    private static final int SPB = 1024;
    private static final int BASE_PORT = 64100;
    private static final ObjectMapper MAPPER = new ObjectMapper();

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
    void setUp() {
        scheduler = Executors.newScheduledThreadPool(4, runnable -> {
            Thread thread = new Thread(runnable, "cache-key-hit-metrics-scheduler");
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
     * The cumulative pair tracks prefix-match runs across a cold → warm →
     * partial-prefix sequence: 0/3 (cold miss parks nothing yet — the LRU is
     * empty), 3/6 (warm re-request fully matches), 2/9 ([k1,k2,k9] hits the
     * 2-key prefix; the k9 miss truncates the run). The ratio after the third
     * request is 5/9 — the key-level hit rate of the known-hit fixture.
     */
    @Test
    void keyHitCountersAccumulatePrefixMatchRuns() throws Exception {
        MockPerformanceModel model = performanceModel(tempDir, "10");
        JavaMockEngineCluster.FastRpcService prefill = newPrefillService(model, 100);
        int prefillPort = prefill.getGrpcPort();
        List<Long> keys = List.of(1L, 2L, 3L);

        // Round 1 (cold): 0 hits over 3 requested keys. Completion parks the
        // keys in the engine LRU.
        assertEquals(0, enqueue(prefill, batch(1, slot(0,
                inputWithBlockKeys(7L, 3 * SPB, keys)))).getErrorsCount());
        awaitPrefillQuiescence(prefill, 1);
        Map<String, Map<Integer, Long>> cold =
                parsePerEngineMetrics(httpGet(controlPort(), "/metrics?per_engine=true"));
        assertEquals(0L, cold.get("mock_engine_cache_key_hits_total")
                        .getOrDefault(prefillPort, -1L),
                "a cold request against an empty LRU hits nothing");
        assertEquals(3L, cold.get("mock_engine_cache_keys_requested_total")
                        .getOrDefault(prefillPort, -1L),
                "the 3 requested keys are counted even on a full miss");

        // Round 2 (warm): the parked keys all match — 3 more hits over 3 more
        // requested keys (cumulative 3/6).
        assertEquals(0, enqueue(prefill, batch(2, slot(0,
                inputWithBlockKeys(8L, 3 * SPB, keys)))).getErrorsCount());
        awaitPrefillQuiescence(prefill, 2);
        Map<String, Map<Integer, Long>> warm =
                parsePerEngineMetrics(httpGet(controlPort(), "/metrics?per_engine=true"));
        assertEquals(3L, warm.get("mock_engine_cache_key_hits_total")
                        .getOrDefault(prefillPort, -1L),
                "the warm re-request matches all 3 parked keys");
        assertEquals(6L, warm.get("mock_engine_cache_keys_requested_total")
                        .getOrDefault(prefillPort, -1L),
                "requested keys accumulate (3 + 3)");

        // Round 3 (partial prefix): [k1,k2,k9] — k1,k2 hit, the k9 miss
        // truncates the run (cumulative 5/9, the key-level hit rate 5/9).
        assertEquals(0, enqueue(prefill, batch(3, slot(0,
                inputWithBlockKeys(9L, 3 * SPB, List.of(1L, 2L, 90L))))).getErrorsCount());
        awaitPrefillQuiescence(prefill, 3);
        Map<String, Map<Integer, Long>> partial =
                parsePerEngineMetrics(httpGet(controlPort(), "/metrics?per_engine=true"));
        assertEquals(5L, partial.get("mock_engine_cache_key_hits_total")
                        .getOrDefault(prefillPort, -1L),
                "the partial-prefix request adds exactly its 2-key hit run (3 + 2)");
        assertEquals(9L, partial.get("mock_engine_cache_keys_requested_total")
                        .getOrDefault(prefillPort, -1L),
                "requested keys accumulate (3 + 3 + 3)");

        // Role-aggregated mode: the prefill bucket carries the same pair.
        Map<String, Map<String, Long>> byRole =
                parseRoleMetrics(httpGet(controlPort(), "/metrics"));
        assertEquals(5L, byRole.get("mock_engine_cache_key_hits_total")
                        .getOrDefault("prefill", -1L),
                "aggregated prefill bucket carries the cumulative hits");
        assertEquals(9L, byRole.get("mock_engine_cache_keys_requested_total")
                        .getOrDefault("prefill", -1L),
                "aggregated prefill bucket carries the cumulative requested keys");

        // /snapshot terminal state matches the /metrics surface.
        JsonNode engines = MAPPER.readTree(httpGet(controlPort(), "/snapshot"))
                .path("engines");
        JsonNode engineSnap = null;
        for (JsonNode engine : engines) {
            if (engine.path("port").asInt() == prefillPort) {
                engineSnap = engine;
            }
        }
        assertNotNull(engineSnap, "the prefill engine must appear in /snapshot");
        assertEquals(5L, engineSnap.path("cache_key_hits").asLong(-1L),
                "/snapshot cache_key_hits matches the metrics counter");
        assertEquals(9L, engineSnap.path("cache_keys_requested").asLong(-1L),
                "/snapshot cache_keys_requested matches the metrics counter");
    }

    /**
     * An empty-bh request (no block keys) contributes 0/0 by construction —
     * the requested counter must not gain a phantom denominator, the hit
     * counter must not gain a phantom numerator.
     */
    @Test
    void emptyBlockHashRequestContributesNothing() throws Exception {
        MockPerformanceModel model = performanceModel(tempDir, "10");
        JavaMockEngineCluster.FastRpcService prefill = newPrefillService(model, 100);
        int prefillPort = prefill.getGrpcPort();

        assertEquals(0, enqueue(prefill, batch(1, slot(0,
                inputWithBlockKeys(11L, 2 * SPB, List.of())))).getErrorsCount());
        awaitPrefillQuiescence(prefill, 1);
        Map<String, Map<Integer, Long>> perEngine =
                parsePerEngineMetrics(httpGet(controlPort(), "/metrics?per_engine=true"));
        assertEquals(0L, perEngine.get("mock_engine_cache_key_hits_total")
                        .getOrDefault(prefillPort, -1L),
                "an empty-bh request has no keys to hit");
        assertEquals(0L, perEngine.get("mock_engine_cache_keys_requested_total")
                        .getOrDefault(prefillPort, -1L),
                "an empty-bh request must not add a denominator entry (0/0, no contribution)");

        Map<String, Map<String, Long>> byRole =
                parseRoleMetrics(httpGet(controlPort(), "/metrics"));
        assertEquals(0L, byRole.get("mock_engine_cache_key_hits_total")
                        .getOrDefault("prefill", -1L));
        assertEquals(0L, byRole.get("mock_engine_cache_keys_requested_total")
                        .getOrDefault("prefill", -1L));
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

    /** Wait until the engine reports {@code expectedCompleted} completions —
     * the prefill completion callback is what parks the request's keys in
     * the LRU, so the NEXT round's hit assertion must observe it first. */
    private static void awaitPrefillQuiescence(
            JavaMockEngineCluster.FastRpcService service, long expectedCompleted)
            throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        while (System.nanoTime() < deadline
                && service.getCompletedCount() < expectedCompleted) {
            Thread.sleep(5);
        }
        assertEquals(expectedCompleted, service.getCompletedCount(),
                "the prefill request must complete (keys parked in the LRU)");
    }

    private static Map<String, Map<Integer, Long>> parsePerEngineMetrics(String body) {
        Map<String, Map<Integer, Long>> result = new java.util.HashMap<>();
        Matcher matcher = PER_ENGINE_METRIC_PATTERN.matcher(body);
        while (matcher.find()) {
            result.computeIfAbsent(matcher.group(1), k -> new java.util.HashMap<>())
                    .put(Integer.parseInt(matcher.group(2)), Long.parseLong(matcher.group(3)));
        }
        assertTrue(result.containsKey("mock_engine_cache_key_hits_total"),
                "per-engine metrics must carry the key-hit counter");
        return result;
    }

    private static Map<String, Map<String, Long>> parseRoleMetrics(String body) {
        Map<String, Map<String, Long>> result = new java.util.HashMap<>();
        Matcher matcher = ROLE_METRIC_PATTERN.matcher(body);
        while (matcher.find()) {
            result.computeIfAbsent(matcher.group(1), k -> new java.util.HashMap<>())
                    .put(matcher.group(2), Long.parseLong(matcher.group(3)));
        }
        assertTrue(result.containsKey("mock_engine_cache_key_hits_total"),
                "role metrics must carry the key-hit counter");
        return result;
    }
}
