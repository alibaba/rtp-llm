package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.net.http.HttpClient;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import static org.flexlb.mockengine.MockEngineTestSupport.batch;
import static org.flexlb.mockengine.MockEngineTestSupport.enqueue;
import static org.flexlb.mockengine.MockEngineTestSupport.inputWithDecode;
import static org.flexlb.mockengine.MockEngineTestSupport.slot;
import static org.flexlb.mockengine.MockEngineTestSupport.unary;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Phase 2 Python-compatibility tests for {@link MockControlServer}.
 *
 * <p>Verifies the Python control-plane compatibility layer:
 * engine-name addressing, the {@code {"engines": [...]}} snapshot schema with
 * the full Python field set, Python {@code /inject} config format coexisting
 * with the legacy Java format, {@code /set_perf} Python fields taking real
 * effect, absolute-value {@code /set_kv_pressure}, engine-name-keyed
 * {@code /requests}, {@code /health} status, and Python-aligned
 * {@code /metrics} names/labels in both aggregated and per-engine modes.
 */
class PythonCompatControlApiTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final HttpClient HTTP_CLIENT = HttpClient.newHttpClient();
    private static final int BASE_PORT = 63100;

    @TempDir
    Path tempDir;

    private ScheduledExecutorService scheduler;
    private MockControlServer controlServer;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private List<JavaMockEngineCluster.FastRpcService> prefillServices;
    private List<JavaMockEngineCluster.FastRpcService> decodeServices;

    @AfterEach
    void tearDown() throws InterruptedException {
        if (controlServer != null) {
            controlServer.stop();
        }
        if (services != null) {
            for (JavaMockEngineCluster.FastRpcService service : services.values()) {
                service.shutdown();
            }
        }
        if (scheduler != null) {
            scheduler.shutdownNow();
            scheduler.awaitTermination(3, TimeUnit.SECONDS);
        }
    }

    // ──────────── Cluster setup ────────────

    /**
     * Creates engines with the Python naming scheme (prefill-i / decode-i)
     * using the full 12-arg constructor, like JavaMockEngineCluster.startRole.
     */
    private void startCluster(MockPerformanceModel model, int nPrefill, int nDecode) throws Exception {
        startCluster(model, nPrefill, nDecode,
                JavaMockEngineCluster.DEFAULT_TOTAL_KV_TOKENS,
                JavaMockEngineCluster.DEFAULT_DECODE_MAX_CONCURRENCY);
    }

    private void startCluster(MockPerformanceModel model, int nPrefill, int nDecode,
                              long totalKvTokens, int decodeMaxConcurrency) throws Exception {
        scheduler = Executors.newScheduledThreadPool(8, runnable -> {
            Thread thread = new Thread(runnable, "mock-engine-scheduler");
            thread.setDaemon(true);
            return thread;
        });
        services = new ConcurrentHashMap<>();
        prefillServices = new ArrayList<>();
        decodeServices = new ArrayList<>();

        for (int i = 0; i < nPrefill; i++) {
            int port = BASE_PORT + i;
            JavaMockEngineCluster.FastRpcService service = new JavaMockEngineCluster.FastRpcService(
                    "prefill-" + i, "127.0.0.1", "prefill",
                    EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL,
                    port, services, scheduler, model, 100,
                    new JavaMockEngineCluster.ClusterStats(),
                    totalKvTokens, decodeMaxConcurrency);
            services.put(port, service);
            prefillServices.add(service);
        }
        for (int i = 0; i < nDecode; i++) {
            int port = BASE_PORT + nPrefill + i;
            JavaMockEngineCluster.FastRpcService service = new JavaMockEngineCluster.FastRpcService(
                    "decode-" + i, "127.0.0.1", "decode",
                    EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE,
                    port, services, scheduler, model, 100,
                    new JavaMockEngineCluster.ClusterStats(),
                    totalKvTokens, decodeMaxConcurrency);
            services.put(port, service);
            decodeServices.add(service);
        }

        controlServer = new MockControlServer(services, new ConcurrentHashMap<>(),
                null, null, "127.0.0.1", 0);
        controlServer.start();
    }

    private MockPerformanceModel model(String formula, double sleepScale) throws Exception {
        return MockEngineTestSupport.performanceModel(tempDir, formula, sleepScale);
    }

    // ════════════════════════════════════════════════════════════════
    //  Test 1: dual addressing by engine name (inject + stop_engine)
    // ════════════════════════════════════════════════════════════════

    @Test
    void engineNameAddressingForInjectAndStopEngine() throws Exception {
        startCluster(model("10", 0.1), 2, 1);

        // /inject by engine name (Python config format)
        HttpResponse<String> response = httpPostResponse("/inject",
                "{\"engine\":\"prefill-0\",\"config\":{\"enqueue_error\":true}}");
        assertEquals(200, response.statusCode());
        JsonNode json = MAPPER.readTree(response.body());
        assertEquals("ok", json.get("status").asText());
        assertEquals("prefill-0", json.get("engine").asText());
        assertTrue(prefillServices.get(0).getFaultConfig().isFailOnEnqueue());
        assertFalse(prefillServices.get(1).getFaultConfig().isFailOnEnqueue(),
                "only prefill-0 should be affected");

        // /stop_engine by engine name
        response = httpPostResponse("/stop_engine", "{\"engine\":\"decode-0\"}");
        assertEquals(200, response.statusCode());
        json = MAPPER.readTree(response.body());
        assertEquals("stopped", json.get("action").asText());
        assertEquals("decode-0", json.get("engine").asText());
        assertTrue(decodeServices.get(0).isStopped());

        // Stopped state visible in the Python snapshot schema.
        JsonNode snapshot = MAPPER.readTree(httpGet("/snapshot"));
        boolean foundStopped = false;
        for (JsonNode e : snapshot.get("engines")) {
            if ("decode-0".equals(e.get("name").asText())) {
                assertTrue(e.get("stopped").asBoolean());
                foundStopped = true;
            }
        }
        assertTrue(foundStopped);

        // Unknown engine name -> 404
        response = httpPostResponse("/stop_engine", "{\"engine\":\"prefill-99\"}");
        assertEquals(404, response.statusCode());
        assertTrue(MAPPER.readTree(response.body()).get("error").asText().contains("prefill-99"));

        // Neither engine nor port -> 400
        response = httpPostResponse("/clear_inject", "{}");
        assertEquals(400, response.statusCode());
        assertTrue(MAPPER.readTree(response.body()).get("error").asText().contains("engine"));

        // Port addressing still works (backward compatible)
        response = httpPostResponse("/clear_inject",
                "{\"port\":" + prefillServices.get(0).getGrpcPort() + "}");
        assertEquals(200, response.statusCode());
        assertFalse(prefillServices.get(0).getFaultConfig().isFailOnEnqueue());
    }

    // ════════════════════════════════════════════════════════════════
    //  Test 2: Python /inject format replaces config; legacy format kept
    // ════════════════════════════════════════════════════════════════

    @Test
    void injectPythonFormatReplacesConfigAndCoexistsWithLegacyFormat() throws Exception {
        startCluster(model("10", 0.1), 1, 1);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);

        // Seed a legacy fault first.
        httpPost("/inject",
                "{\"port\":" + prefill.getGrpcPort() + ",\"type\":\"fetch_error\",\"enabled\":true}");
        assertTrue(prefill.getFaultConfig().isFetchError());

        // Python format REPLACES the whole config (fetch_error must disappear).
        httpPost("/inject",
                "{\"engine\":\"prefill-0\",\"config\":{\"generate_error\":true,\"no_respond\":true}}");
        FaultInjectionConfig config = prefill.getFaultConfig();
        assertTrue(config.isGenerateError());
        assertTrue(config.isNoRespond());
        assertFalse(config.isFetchError(), "Python /inject replaces the whole config");
        assertFalse(config.isFailOnEnqueue());

        // Missing flags default to false (whole-replacement semantics).
        httpPost("/inject", "{\"engine\":\"prefill-0\",\"config\":{\"enqueue_error\":true}}");
        config = prefill.getFaultConfig();
        assertTrue(config.isFailOnEnqueue());
        assertFalse(config.isGenerateError());
        assertFalse(config.isNoRespond());

        // Legacy Java format still works unchanged.
        httpPost("/inject",
                "{\"port\":" + prefill.getGrpcPort() + ",\"type\":\"no_respond\",\"enabled\":true}");
        assertTrue(prefill.getFaultConfig().isNoRespond());

        // /clear_inject by engine name.
        httpPost("/clear_inject", "{\"engine\":\"prefill-0\"}");
        config = prefill.getFaultConfig();
        assertFalse(config.isFailOnEnqueue());
        assertFalse(config.isNoRespond());
    }

    // ════════════════════════════════════════════════════════════════
    //  Test 3: /snapshot Python schema (engines wrapper + full field set)
    // ════════════════════════════════════════════════════════════════

    @Test
    void snapshotHasPythonSchemaAndFields() throws Exception {
        startCluster(model("10", 0.1), 1, 2);

        JsonNode root = MAPPER.readTree(httpGet("/snapshot"));
        assertTrue(root.has("engines"), "/snapshot must wrap engines in {\"engines\": [...]}");
        assertTrue(root.get("engines").isArray());
        assertEquals(3, root.get("engines").size());

        JsonNode engine = root.get("engines").get(0);
        // Python field set (legacy MockEngineState.snapshot) — names and types.
        assertEquals("prefill-0", engine.get("name").asText());
        assertEquals("prefill", engine.get("role").asText());
        assertEquals("127.0.0.1:" + (BASE_PORT), engine.get("grpc_addr").asText());
        assertTrue(engine.get("http_addr").isTextual());
        assertTrue(engine.get("running").isNumber());
        assertTrue(engine.get("waiting").isNumber());
        assertTrue(engine.get("accepted").isNumber());
        assertTrue(engine.get("completed").isNumber());
        assertTrue(engine.get("cache_keys").isNumber());
        assertTrue(engine.get("cache_evictions").isNumber());
        assertTrue(engine.get("active_kv_tokens").isNumber());
        assertTrue(engine.get("available_kv_tokens").isNumber());
        assertTrue(engine.get("inject_config").isObject());
        assertTrue(engine.get("inject_config").has("enqueue_error"));
        assertTrue(engine.get("inject_config").has("fetch_error"));
        assertTrue(engine.get("inject_config").has("generate_error"));
        assertTrue(engine.get("inject_config").has("no_respond"));
        JsonNode rpcCounts = engine.get("rpc_counts");
        assertTrue(rpcCounts.isObject());
        assertTrue(rpcCounts.has("enqueue_batch"));
        assertTrue(rpcCounts.has("generate_stream"));
        assertTrue(rpcCounts.has("fetch_response"));
        assertTrue(rpcCounts.has("cancel"));
        assertTrue(engine.get("cancelled_count").isNumber());
        assertTrue(engine.get("cancelled_rids").isArray());
        assertTrue(engine.get("request_lifecycle").isObject());
        assertTrue(engine.get("prefill_ms_avg").isNumber());
        assertTrue(engine.get("prefill_ms_p99").isNumber());
        assertTrue(engine.get("prefill_ms_count").isNumber());
        assertTrue(engine.get("decode_ms_avg").isNumber());
        assertTrue(engine.get("decode_ms_p99").isNumber());
        assertTrue(engine.get("decode_ms_count").isNumber());
        assertFalse(engine.get("stopped").asBoolean());
        // Java-only fields retained.
        assertTrue(engine.get("port").isNumber());
        assertTrue(engine.get("inflight").isNumber());
        assertTrue(engine.get("leak_detected").isBoolean());

        assertEquals("decode-0", root.get("engines").get(1).get("name").asText());
        assertEquals("decode", root.get("engines").get(1).get("role").asText());
        assertEquals("decode-1", root.get("engines").get(2).get("name").asText());
    }

    // ════════════════════════════════════════════════════════════════
    //  Test 4: /requests keyed by engine name, /health status ok
    // ════════════════════════════════════════════════════════════════

    @Test
    void requestsKeyedByEngineNameAndHealthStatusOk() throws Exception {
        startCluster(model("10", 0.1), 1, 1);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);

        enqueue(prefill, batch(5000, slot(0,
                inputWithDecode(9001, 10, decodeServices.get(0).getGrpcPort()))));

        JsonNode requests = MAPPER.readTree(httpGet("/requests"));
        assertTrue(requests.isObject(), "/requests must be an object keyed by engine name");
        assertTrue(requests.has("prefill-0"));
        assertTrue(requests.has("decode-0"));
        JsonNode lifecycle = requests.get("prefill-0").get("9001");
        assertNotNull(lifecycle, "request lifecycle should be present for rid 9001");
        assertEquals("enqueue_batch", lifecycle.get("method").asText());

        JsonNode health = MAPPER.readTree(httpGet("/health"));
        assertEquals("ok", health.get("status").asText());
        assertTrue(health.get("healthy").asBoolean());
        assertEquals(2, health.get("engines").asInt());
    }

    // ════════════════════════════════════════════════════════════════
    //  Test 5: /set_perf Python fields take real effect
    // ════════════════════════════════════════════════════════════════

    @Test
    void setPerfPythonFieldsTakeEffect() throws Exception {
        // sleep_scale=1.0, formula "10" -> baseline prefill 10ms, decode 1ms/token.
        MockPerformanceModel model = model("10", 1.0);
        startCluster(model, 1, 1);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);
        MockPerformanceModel perf = prefill.getPerformance();

        MockLruBlockCache probeCache = new MockLruBlockCache(16);
        MockPerformanceModel.RequestShape shape = perf.shape(
                inputWithDecode(1, 10, BASE_PORT + 1), probeCache);

        assertEquals(10L, perf.prefillMs(List.of(shape)), "baseline prefill should be 10ms");
        assertEquals(1L, perf.decodeMs(1, 1), "baseline decode should be 1ms");
        assertEquals(1, prefill.getMaxPrefillConcurrency(), "default max_prefill_concurrency");

        httpPost("/set_perf",
                "{\"engine\":\"prefill-0\",\"prefill_fixed_ms\":50,"
                        + "\"decode_scale\":2.0,\"max_prefill_concurrency\":3}");

        assertEquals(50L, perf.prefillMs(List.of(shape)),
                "prefill_fixed_ms must take effect");
        assertEquals(2L, perf.decodeMs(1, 1),
                "decode_scale must double decode latency");
        assertEquals(3, prefill.getMaxPrefillConcurrency());

        // worker_status available_concurrency mirrors Python:
        // max(0, max_prefill_concurrency - running) == 3 when idle.
        EngineRpcService.WorkerStatusPB status = unary(observer -> prefill.getWorkerStatus(
                EngineRpcService.StatusVersionPB.newBuilder().build(), observer));
        assertEquals(3, status.getAvailableConcurrency(),
                "idle prefill available_concurrency should equal max_prefill_concurrency");
    }

    // ════════════════════════════════════════════════════════════════
    //  Test 6: max_prefill_concurrency lanes run prefills in parallel
    // ════════════════════════════════════════════════════════════════

    @Test
    void maxPrefillConcurrencyLanesRunInParallel() throws Exception {
        // sleep_scale=1.0, formula "200" -> each prefill batch takes ~200ms.
        startCluster(model("200", 1.0), 1, 1);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);
        JavaMockEngineCluster.FastRpcService decode = decodeServices.get(0);

        httpPost("/set_perf", "{\"engine\":\"prefill-0\",\"max_prefill_concurrency\":3}");

        long startNanos = System.nanoTime();
        for (int i = 0; i < 3; i++) {
            enqueue(prefill, batch(6000 + i, slot(0,
                    inputWithDecode(9100 + i, 10, decode.getGrpcPort()))));
        }
        awaitCompleted(decode, 3, 10_000);
        long elapsedMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startNanos);

        // 3 parallel lanes: ~200ms total. Serial (legacy C=1) would be >= 600ms.
        assertTrue(elapsedMs < 500,
                "3 batches of 200ms with concurrency 3 should finish < 500ms, took " + elapsedMs);
        assertTrue(elapsedMs >= 180,
                "prefill sleep should actually happen, took " + elapsedMs);
    }

    // ════════════════════════════════════════════════════════════════
    //  Test 7: /set_kv_pressure absolute semantics
    // ════════════════════════════════════════════════════════════════

    @Test
    void setKvPressureAbsoluteSemantics() throws Exception {
        long totalKvTokens = 100_000L;
        startCluster(model("10", 0.1), 1, 1, totalKvTokens, 132);

        httpPost("/set_kv_pressure", "{\"engine\":\"prefill-0\",\"active_kv_tokens\":5000}");
        JsonNode engine = engineByName("prefill-0");
        assertEquals(5000, engine.get("active_kv_tokens").asLong(),
                "active_kv_tokens should be the absolute value set via /set_kv_pressure");
        assertEquals(95_000, engine.get("available_kv_tokens").asLong());

        // Reset to 0.
        httpPost("/set_kv_pressure", "{\"engine\":\"prefill-0\",\"active_kv_tokens\":0}");
        engine = engineByName("prefill-0");
        assertEquals(0, engine.get("active_kv_tokens").asLong());
        assertEquals(totalKvTokens, engine.get("available_kv_tokens").asLong());
    }

    // ════════════════════════════════════════════════════════════════
    //  Test 8: /set_queue_depth (Python field name, Java real rejection)
    // ════════════════════════════════════════════════════════════════

    @Test
    void setQueueDepthRejectsWhenFull() throws Exception {
        // 200ms prefill keeps the first request pending long enough to hit the limit.
        startCluster(model("200", 1.0), 1, 1);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);

        httpPost("/set_queue_depth", "{\"engine\":\"prefill-0\",\"queue_depth\":1}");

        EngineRpcService.EnqueueBatchResponsePB first = enqueue(prefill, batch(7000, slot(0,
                inputWithDecode(9200, 10, decodeServices.get(0).getGrpcPort()))));
        assertEquals(1, first.getSuccessesCount());

        // Second enqueue while the first is still pending -> rejected (Java semantics;
        // Python only fakes the queue depth display — divergence deferred to Phase 5).
        EngineRpcService.EnqueueBatchResponsePB second = enqueue(prefill, batch(7001, slot(0,
                inputWithDecode(9201, 10, decodeServices.get(0).getGrpcPort()))));
        assertEquals(0, second.getSuccessesCount());
        assertEquals(1, second.getErrorsCount());
        assertTrue(second.getErrors(0).getErrorInfo().getErrorMessage()
                .contains("queue depth"));

        // Clearing the limit restores acceptance.
        httpPost("/set_queue_depth", "{\"engine\":\"prefill-0\",\"queue_depth\":0}");
        EngineRpcService.EnqueueBatchResponsePB third = enqueue(prefill, batch(7002, slot(0,
                inputWithDecode(9202, 10, decodeServices.get(0).getGrpcPort()))));
        assertEquals(1, third.getSuccessesCount());
    }

    // ════════════════════════════════════════════════════════════════
    // Test 9: /metrics — Python names in both modes
    // ════════════════════════════════════════════════════════════════

    @Test
    void metricsDefaultModeContainsPythonNames() throws Exception {
        startCluster(model("10", 0.1), 2, 2);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);
        enqueue(prefill, batch(8000, slot(0,
                inputWithDecode(9300, 10, decodeServices.get(0).getGrpcPort()),
                inputWithDecode(9301, 10, decodeServices.get(1).getGrpcPort()))));
        awaitCompleted(decodeServices.get(0), 1, 10_000);
        awaitCompleted(decodeServices.get(1), 1, 10_000);

        String body = httpGet("/metrics");
        // Python metric names required by the Grafana dashboard.
        for (String metric : new String[]{
                "mock_engine_up", "mock_engine_running", "mock_engine_waiting",
                "mock_engine_accepted_total", "mock_engine_completed_total",
                "mock_engine_cancelled_total", "mock_engine_cache_keys",
                "mock_engine_cache_evictions_total", "mock_engine_active_kv_tokens",
                "mock_engine_available_kv_tokens", "mock_engine_rpc_total",
                "mock_engine_prefill_ms_avg", "mock_engine_decode_ms_p99"}) {
            assertTrue(body.contains(metric), "/metrics should contain " + metric);
        }
        // Aggregated mode uses role-only labels.
        assertTrue(body.contains("mock_engine_accepted_total{role=\"prefill\"} 2"),
                "aggregated prefill accepted should be 2:\n" + body);
        assertTrue(body.contains("mock_engine_completed_total{role=\"decode\"} 2"),
                "aggregated decode completed should be 2");
        assertTrue(body.contains("mock_engine_rpc_total{role=\"prefill\",rpc_method=\"enqueue_batch\"} 1"));
    }

    @Test
    void metricsPerEngineModeUsesEngineLabels() throws Exception {
        startCluster(model("10", 0.1), 1, 1);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);
        enqueue(prefill, batch(8100, slot(0,
                inputWithDecode(9400, 10, decodeServices.get(0).getGrpcPort()))));
        awaitCompleted(decodeServices.get(0), 1, 10_000);

        String body = httpGet("/metrics?per_engine=true");
        String expectedLabels = "engine_name=\"prefill-0\",role=\"prefill\","
                + "grpc_port=\"" + prefill.getGrpcPort() + "\",engine_ip=\"127.0.0.1\"";
        assertTrue(body.contains("mock_engine_up{" + expectedLabels + "} 1"),
                "per-engine up series expected:\n" + body);
        assertTrue(body.contains("mock_engine_accepted_total{" + expectedLabels + "} 1"));
        assertTrue(body.contains("mock_engine_rpc_total{" + expectedLabels
                + ",rpc_method=\"enqueue_batch\"} 1"));
        assertTrue(body.contains("mock_engine_completed_total{engine_name=\"decode-0\","
                + "role=\"decode\",grpc_port=\"" + decodeServices.get(0).getGrpcPort()
                + "\",engine_ip=\"127.0.0.1\"} 1"));
    }

    // ════════════════════════════════════════════════════════════════
    //  Test 10: rpc_counts / cancelled_rids / request_lifecycle
    // ════════════════════════════════════════════════════════════════

    @Test
    void rpcCountsAndCancelledRidsExposed() throws Exception {
        startCluster(model("10", 0.1), 1, 1);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);

        enqueue(prefill, batch(9000, slot(0,
                inputWithDecode(9500, 10, decodeServices.get(0).getGrpcPort()),
                inputWithDecode(9501, 10, decodeServices.get(0).getGrpcPort()))));

        prefill.cancel(9500);

        JsonNode engine = engineByName("prefill-0");
        assertEquals(1, engine.get("rpc_counts").get("enqueue_batch").asLong());
        assertEquals(1, engine.get("rpc_counts").get("cancel").asLong(),
                "cancel(long) is the cancel entry point and counts as a cancel RPC");
        assertEquals(1, engine.get("cancelled_count").asLong());
        JsonNode cancelledRids = engine.get("cancelled_rids");
        assertEquals(1, cancelledRids.size());
        assertEquals(9500, cancelledRids.get(0).asLong());

        JsonNode lifecycle = engine.get("request_lifecycle").get("9500");
        assertNotNull(lifecycle);
        assertEquals("cancelled", lifecycle.get("end_state").asText());
        assertTrue(lifecycle.get("end_ms").asLong() > 0);

        JsonNode runningLifecycle = engine.get("request_lifecycle").get("9501");
        assertNotNull(runningLifecycle);
        assertEquals("enqueue_batch", runningLifecycle.get("method").asText());
    }

    // ──────────── Helpers ────────────

    private JsonNode engineByName(String name) throws Exception {
        JsonNode engines = MAPPER.readTree(httpGet("/snapshot")).get("engines");
        for (JsonNode engine : engines) {
            if (name.equals(engine.get("name").asText())) {
                return engine;
            }
        }
        fail("engine " + name + " not found in /snapshot");
        return null;
    }

    private void awaitCompleted(JavaMockEngineCluster.FastRpcService service,
                                int expected, long timeoutMs) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (System.nanoTime() < deadline) {
            if (service.getCompletedCount() >= expected) {
                return;
            }
            Thread.sleep(5);
        }
        fail("timeout waiting for " + expected + " completions on port "
                + service.getGrpcPort() + ", got " + service.getCompletedCount());
    }

    private String httpGet(String path) throws Exception {
        return MockEngineTestSupport.httpGet(controlServer.getPort(), path);
    }

    private void httpPost(String path, String body) throws Exception {
        HttpResponse<String> response = httpPostResponse(path, body);
        assertEquals(200, response.statusCode(),
                "POST " + path + " failed: " + response.body());
    }

    private HttpResponse<String> httpPostResponse(String path, String body) throws Exception {
        return MockEngineTestSupport.httpPostResponse(controlServer.getPort(), path, body);
    }

}
