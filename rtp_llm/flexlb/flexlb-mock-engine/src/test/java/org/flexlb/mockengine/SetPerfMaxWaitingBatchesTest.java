package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import static org.flexlb.mockengine.MockEngineTestSupport.batch;
import static org.flexlb.mockengine.MockEngineTestSupport.enqueue;
import static org.flexlb.mockengine.MockEngineTestSupport.input;
import static org.flexlb.mockengine.MockEngineTestSupport.slot;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Runtime {@code /set_perf} override tests for the prefill waiting-queue cap
 * ({@code prefill.max_waiting_batches}) — the B3 admission gate. The cap was
 * previously constructor-final (performance JSON only), so tests could not
 * trigger the gate mid-run; this is its runtime control-plane entry, mirroring
 * the existing {@code prefill_fixed_ms} / {@code decode_scale} /
 * {@code max_prefill_concurrency} overrides.
 *
 * <p>Override chain (same priority as {@code prefillMs}): runtime /set_perf
 * value &gt; JSON-configured value. The override value carries the same
 * semantics as the JSON field — 0 = unbounded (production default), &gt; 0 =
 * cap on QUEUED batches (running batches never count). Negative values are
 * neither a cap nor the explicit unbounded 0, so the endpoint answers 400
 * and leaves the live cap untouched.
 */
class SetPerfMaxWaitingBatchesTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final int BASE_PORT = 63250;

    @TempDir
    Path tempDir;

    private ScheduledExecutorService scheduler;
    private MockControlServer controlServer;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;

    @BeforeEach
    void setUp() {
        scheduler = Executors.newScheduledThreadPool(4, runnable -> {
            Thread thread = new Thread(runnable, "setperf-waiting-cap-scheduler");
            thread.setDaemon(true);
            return thread;
        });
        services = new ConcurrentHashMap<>();
    }

    @AfterEach
    void tearDown() throws InterruptedException {
        if (controlServer != null) {
            controlServer.stop();
        }
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            service.shutdown();
        }
        scheduler.shutdownNow();
        scheduler.awaitTermination(3, TimeUnit.SECONDS);
    }

    // ──────────── Test 1: set → accumulate → cap → reject → restore ────────────

    @Test
    void runtimeOverrideGatesEnqueueAndRestores() throws Exception {
        // Constructor leaves the queue unbounded; the cap arrives live via /set_perf.
        JavaMockEngineCluster.FastRpcService prefill = startPrefill(model("200", null));

        // Slow prefill: 5 batches all land before any completion — proof the
        // constructor value (absent = 0 = unbounded) does not gate on its own.
        for (int i = 1; i <= 5; i++) {
            assertEquals(1, enqueue(prefill, batch(1000 + i, slot(0, input(i, 10))))
                    .getSuccessesCount(), "batch " + i + " should be accepted (unbounded)");
        }
        assertEquals(4, prefill.getWaitingCount(), "1 running + 4 queued expected");

        // set → cap arrives exactly at the current queue depth → next batch rejected.
        httpPost("/set_perf", "{\"engine\":\"prefill-0\",\"max_waiting_batches\":4}");
        EngineRpcService.EnqueueBatchResponsePB rejected =
                enqueue(prefill, batch(1006, slot(0, input(6, 10))));
        assertEquals(0, rejected.getSuccessesCount(), "6th batch should not be accepted");
        assertEquals(1, rejected.getErrorsCount(), "6th batch should be rejected");
        String message = rejected.getErrors(0).getErrorInfo().getErrorMessage();
        assertTrue(message.contains("prefill waiting queue full (backpressure)"),
                "rejection must carry the explicit backpressure error, got: " + message);
        assertTrue(message.contains("waiting=4 cap=4"),
                "rejection should report waiting/cap, got: " + message);

        // Raise → acceptance restored.
        httpPost("/set_perf", "{\"engine\":\"prefill-0\",\"max_waiting_batches\":6}");
        assertEquals(1, enqueue(prefill, batch(1007, slot(0, input(7, 10))))
                .getSuccessesCount(), "raised cap should accept again");

        // Clear (0 = unbounded) → acceptance restored.
        httpPost("/set_perf", "{\"engine\":\"prefill-0\",\"max_waiting_batches\":0}");
        assertEquals(1, enqueue(prefill, batch(1008, slot(0, input(8, 10))))
                .getSuccessesCount(), "clearing to 0 should accept again");

        // The 7 admitted batches drain to completion despite the rejection.
        awaitInflightZero(prefill, 5_000);
        assertEquals(7, prefill.getCompletedCount());
        assertEquals(0, prefill.getWaitingCount());
        assertFalse(prefill.isLeakDetected());
    }

    // ──────────── Test 2: override beats the JSON-configured value ────────────

    @Test
    void overrideBeatsConfiguredCap() throws Exception {
        // Constructor cap = 1 (via performance JSON).
        JavaMockEngineCluster.FastRpcService prefill = startPrefill(model("200", 1));

        assertEquals(1, enqueue(prefill, batch(2001, slot(0, input(1, 10))))
                .getSuccessesCount(), "1st batch runs");
        assertEquals(1, enqueue(prefill, batch(2002, slot(0, input(2, 10))))
                .getSuccessesCount(), "2nd batch queues under configured cap 1");
        assertEquals(1, enqueue(prefill, batch(2003, slot(0, input(3, 10))))
                .getErrorsCount(), "3rd batch should hit the configured cap of 1");

        // Runtime override raises the cap above the JSON value.
        httpPost("/set_perf", "{\"engine\":\"prefill-0\",\"max_waiting_batches\":3}");
        assertEquals(1, enqueue(prefill, batch(2004, slot(0, input(4, 10))))
                .getSuccessesCount(), "overridden cap 3 should accept batch 4");
        assertEquals(1, enqueue(prefill, batch(2005, slot(0, input(5, 10))))
                .getSuccessesCount(), "overridden cap 3 should accept batch 5");
        assertEquals(1, enqueue(prefill, batch(2006, slot(0, input(6, 10))))
                .getErrorsCount(), "6th batch should hit the overridden cap of 3");

        awaitInflightZero(prefill, 5_000);
        assertEquals(4, prefill.getCompletedCount(), "1st, 2nd, 4th, 5th should complete");
        assertFalse(prefill.isLeakDetected());
    }

    // ──────────── Test 3: runtime 0 disables a configured cap ────────────

    @Test
    void runtimeZeroDisablesConfiguredCap() throws Exception {
        // Constructor cap = 2, but the runtime override sets 0 = unbounded.
        JavaMockEngineCluster.FastRpcService prefill = startPrefill(model("200", 2));

        httpPost("/set_perf", "{\"engine\":\"prefill-0\",\"max_waiting_batches\":0}");

        for (int i = 1; i <= 8; i++) {
            EngineRpcService.EnqueueBatchResponsePB response =
                    enqueue(prefill, batch(3000 + i, slot(0, input(i, 10))));
            assertEquals(1, response.getSuccessesCount(),
                    "batch " + i + " should be accepted under runtime 0 override");
            assertEquals(0, response.getErrorsCount());
        }

        awaitInflightZero(prefill, 5_000);
        assertEquals(8, prefill.getCompletedCount());
        assertFalse(prefill.isLeakDetected());
    }

    // ──────────── Test 4: negative value → 400, live cap unchanged ────────────

    @Test
    void negativeValueIsRejectedAndKeepsLiveCap() throws Exception {
        JavaMockEngineCluster.FastRpcService prefill = startPrefill(model("200", null));

        httpPost("/set_perf", "{\"engine\":\"prefill-0\",\"max_waiting_batches\":1}");

        HttpResponse<String> response = httpPostResponse("/set_perf",
                "{\"engine\":\"prefill-0\",\"max_waiting_batches\":-1}");
        assertEquals(400, response.statusCode(), "negative max_waiting_batches must be a 400");
        assertTrue(MAPPER.readTree(response.body()).get("error").asText().contains("max_waiting_batches"),
                "error should name the field, got: " + response.body());

        // The failed request must not have relaxed or changed the live cap:
        // with cap still 1, batch 1 runs, batch 2 queues, batch 3 is rejected.
        assertEquals(1, enqueue(prefill, batch(4001, slot(0, input(1, 10))))
                .getSuccessesCount());
        assertEquals(1, enqueue(prefill, batch(4002, slot(0, input(2, 10))))
                .getSuccessesCount());
        assertEquals(1, enqueue(prefill, batch(4003, slot(0, input(3, 10))))
                .getErrorsCount(), "cap must still be 1 after the rejected negative request");

        awaitInflightZero(prefill, 5_000);
        assertEquals(2, prefill.getCompletedCount());
        assertFalse(prefill.isLeakDetected());
    }

    // ──────────── Harness ────────────

    private JavaMockEngineCluster.FastRpcService startPrefill(MockPerformanceModel model) throws Exception {
        JavaMockEngineCluster.FastRpcService service = new JavaMockEngineCluster.FastRpcService(
                "prefill-0", "127.0.0.1", "prefill",
                EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL,
                BASE_PORT, services, scheduler, model, 100,
                new JavaMockEngineCluster.ClusterStats(),
                JavaMockEngineCluster.DEFAULT_TOTAL_KV_TOKENS,
                JavaMockEngineCluster.DEFAULT_DECODE_MAX_CONCURRENCY);
        services.put(BASE_PORT, service);
        controlServer = new MockControlServer(services, new ConcurrentHashMap<>(),
                null, null, "127.0.0.1", 0);
        controlServer.start();
        return service;
    }

    /** @param maxWaitingBatches constructor-time JSON cap; null = absent (default 0). */
    private MockPerformanceModel model(String prefillFormula, Integer maxWaitingBatches)
            throws Exception {
        Map<String, ?> prefillOverrides = maxWaitingBatches == null
                ? Map.of() : Map.of("max_waiting_batches", maxWaitingBatches);
        return MockEngineTestSupport.performanceModel(
                tempDir, prefillFormula, 1.0, 1.0, prefillOverrides, Map.of());
    }

    private static void awaitInflightZero(JavaMockEngineCluster.FastRpcService service,
                                          long timeoutMs) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (System.nanoTime() < deadline) {
            if (service.getInflightCount() == 0) {
                return;
            }
            Thread.sleep(10);
        }
        fail("inflight not zero: inflight=" + service.getInflightCount()
                + " waiting=" + service.getWaitingCount()
                + " running=" + service.getRunningCount());
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
