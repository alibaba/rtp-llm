package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.flexlb.mockengine.MockEngineTestSupport.batch;
import static org.flexlb.mockengine.MockEngineTestSupport.enqueue;
import static org.flexlb.mockengine.MockEngineTestSupport.input;
import static org.flexlb.mockengine.MockEngineTestSupport.inputWithDecode;
import static org.flexlb.mockengine.MockEngineTestSupport.slot;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Inflight leak verification tests for the Java mock engine.
 *
 * <p>Verifies that inflight counts drain to zero across five scenarios:
 * normal completion, mid-flight engine stop, fault injection, cancel mid-flight,
 * and leak detection triggering.
 */
class InflightLeakTest {

    private static final int BASE_PORT = 62000;

    @TempDir
    Path tempDir;

    private MockEngineTestCluster cluster;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private List<JavaMockEngineCluster.FastRpcService> prefillServices;
    private List<JavaMockEngineCluster.FastRpcService> decodeServices;

    @AfterEach
    void tearDown() {
        if (cluster != null) {
            cluster.close();
        }
    }

    // ──────────── Test 1: Normal Completion — No Leak ────────────

    @Test
    void normalCompletionNoLeak() throws Exception {
        MockPerformanceModel model = model("10");
        startCluster(model, 2, 2);

        int n = 100;
        // Batch all requests: 50 per prefill engine, with decode routing
        cluster.enqueueBatch(prefillServices.get(0), 1000, 1, n / 2, decodeServices);
        cluster.enqueueBatch(prefillServices.get(1), 2000, n / 2 + 1, n / 2, decodeServices);

        // Wait for all completions
        cluster.awaitCompleted(n, 10_000);

        // Assert inflight == 0 for all engines
        cluster.assertAllInflightZero();

        // Assert HTTP snapshot shows inflight: 0 and leak_detected: false
        JsonNode snapshot = cluster.snapshot();
        assertEquals(services.size(), snapshot.size());
        for (JsonNode engine : snapshot) {
            assertEquals(0, engine.get("inflight").asInt(),
                    "snapshot inflight should be 0 for port " + engine.get("port").asInt());
            assertFalse(engine.get("leak_detected").asBoolean(),
                    "snapshot leak_detected should be false for port " + engine.get("port").asInt());
        }
    }

    // ──────────── Test 2: Mid-Flight Engine Stop — Inflight Drains ────────────

    @Test
    void midFlightEngineStopDrainsInflight() throws Exception {
        MockPerformanceModel model = model("500"); // 500ms prefill to keep requests in-flight
        startCluster(model, 2, 2);

        int n = 50;
        // Batch 25 requests per prefill engine
        cluster.enqueueBatch(prefillServices.get(0), 2000, 1, n / 2, decodeServices);
        cluster.enqueueBatch(prefillServices.get(1), 2001, n / 2 + 1, n / 2, decodeServices);

        // Wait for requests to be in-flight on prefill engine 0
        cluster.awaitInflight(prefillServices.get(0), 1, 1_000);

        // Stop 1 prefill engine via HTTP /stop_engine
        MockEngineTestSupport.httpPost(cluster.controlPort(), "/stop_engine",
                "{\"port\":" + prefillServices.get(0).getGrpcPort() + "}");
        assertTrue(prefillServices.get(0).isStopped(), "engine should be stopped");

        // Wait for all inflight to drain (scheduled completions still fire after stop)
        cluster.awaitAllInflightZero(5_000);

        // Assert all engines have inflight == 0
        cluster.assertAllInflightZero();
    }

    // ──────────── Test 3: Fault Injection — Enqueue Error, No Leak ────────────

    @Test
    void faultInjectionEnqueueErrorNoLeak() throws Exception {
        MockPerformanceModel model = model("10");
        startCluster(model, 2, 2);

        // Inject enqueue_error on 1 prefill engine via HTTP /inject
        MockEngineTestSupport.httpPost(cluster.controlPort(), "/inject",
                "{\"port\":" + prefillServices.get(0).getGrpcPort()
                        + ",\"type\":\"enqueue_error\",\"enabled\":true}");
        assertTrue(prefillServices.get(0).getFaultConfig().isFailOnEnqueue());

        int n = 50;
        int totalErrors = 0;
        // Enqueue to both prefill engines; the faulted one returns errors
        for (int i = 0; i < n; i++) {
            JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(i % 2);
            int decodePort = decodeServices.get(i % 2).getGrpcPort();
            EngineRpcService.GenerateInputPB input = inputWithDecode(i + 1, 10, decodePort);
            EngineRpcService.EnqueueBatchResponsePB response =
                    enqueue(prefill, batch(3000 + i, slot(0, input)));
            totalErrors += response.getErrorsCount();
        }

        // Clear injection
        MockEngineTestSupport.httpPost(cluster.controlPort(), "/clear_inject",
                "{\"port\":" + prefillServices.get(0).getGrpcPort() + "}");
        assertFalse(prefillServices.get(0).getFaultConfig().isFailOnEnqueue());

        // Wait for non-failed requests to complete
        cluster.awaitCompleted(n - totalErrors, 5_000);

        // Assert all engines have inflight == 0
        cluster.assertAllInflightZero();

        // Assert some errors occurred
        assertTrue(totalErrors > 0, "some requests should have failed on the injected engine");
    }

    // ──────────── Test 4: Cancel Mid-Flight — No Leak ────────────

    @Test
    void cancelMidFlightNoLeak() throws Exception {
        MockPerformanceModel model = model("2000"); // 2s prefill to keep requests in-flight
        startCluster(model, 2, 2);

        int n = 20;
        // Batch all 20 requests on prefill engine 0, with decode routing
        cluster.enqueueBatch(prefillServices.get(0), 4000, 1, n, decodeServices);

        // Wait for requests to be in-flight
        cluster.awaitInflight(prefillServices.get(0), 1, 1_000);

        // Cancel 5 requests via cancel(requestId)
        List<Long> cancelledIds = List.of(1L, 2L, 3L, 4L, 5L);
        for (long requestId : cancelledIds) {
            prefillServices.get(0).cancel(requestId);
        }

        // Wait for all inflight to drain
        cluster.awaitAllInflightZero(5_000);

        // Assert inflight == 0 for all engines
        cluster.assertAllInflightZero();

        // Assert cancelled requests are not in "running" state
        for (long requestId : cancelledIds) {
            for (JavaMockEngineCluster.FastRpcService service : services.values()) {
                String state = service.getRequestStates().get(requestId);
                if (state != null) {
                    assertNotEquals("running", state,
                            "cancelled request " + requestId + " should not be in 'running' state"
                                    + " on port " + service.getGrpcPort());
                }
            }
        }
    }

    // ──────────── Test 5: Leak Detection Trigger ────────────

    @Test
    void leakDetectionTrigger() throws Exception {
        MockPerformanceModel model = model("3000"); // 3s prefill to keep requests in-flight
        startCluster(model, 1, 1);

        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);

        int n = 10;
        // Enqueue 10 requests without decode routing (prefill-only, stays in-flight)
        EngineRpcService.GenerateInputPB[] inputs = new EngineRpcService.GenerateInputPB[n];
        for (int i = 0; i < n; i++) {
            inputs[i] = input(i + 1, 10);
        }
        enqueue(prefill, batch(5000, slot(0, inputs)));

        // Wait for requests to be in-flight
        cluster.awaitInflight(prefill, 1, 1_000);
        assertTrue(prefill.getInflightCount() > 0,
                "prefill should have inflight > 0 after enqueue");

        // During grace window (60s) — should NOT detect leak
        prefill.checkLeakDrain(60_000_000_000L);
        assertFalse(prefill.isLeakDetected(),
                "leak should NOT be detected during grace window");

        // After grace expires (force with 0 grace) — should detect leak
        prefill.checkLeakDrain(0);
        assertTrue(prefill.isLeakDetected(),
                "leak should be detected when inflight > 0 and grace expired");

        // Assert via HTTP snapshot
        JsonNode snapshot = cluster.snapshot();
        for (JsonNode engine : snapshot) {
            if (engine.get("port").asInt() == prefill.getGrpcPort()) {
                assertTrue(engine.get("leak_detected").asBoolean(),
                        "snapshot should show leak_detected=true for prefill engine");
            }
        }
    }

    // ────────── Test 6: Double Scheduling — No Leak ──────────

    @Test
    void doubleSchedulingNoLeak() throws Exception {
        // Prefill 500ms gives a wide window for the second scheduling path to
        // arrive while the first decode completion is still pending.
        MockPerformanceModel model = model("500");
        startCluster(model, 1, 1);

        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);
        JavaMockEngineCluster.FastRpcService decode = decodeServices.get(0);

        // Stretch decode to 1000ms so that the first scheduleDecodeCompletion
        // (from path B) is still in runningTasks when path A's prefill finishes
        // and triggers a second scheduleDecodeCompletion for the same requestId.
        decode.getPerformance().setOverrideDecodeStepMs(1000.0);

        long requestId = 1L;
        int decodePort = decode.getGrpcPort();

        // Path A: enqueueBatch → schedulePrefillCompletion → prefill done
        //         → startDecode → decode.scheduleDecodeCompletion
        EngineRpcService.GenerateInputPB inputA = inputWithDecode(requestId, 10, decodePort);
        enqueue(prefill, batch(7000, slot(0, inputA)));

        // Path B: generateStreamCall on the decode engine with the SAME requestId
        //         → decode.scheduleDecodeCompletion (immediate)
        EngineRpcService.GenerateInputPB inputB = input(requestId, 10);
        generateStream(decode, inputB);

        // Wait for all inflight to drain (prefill 500ms + decode 1000ms + margin)
        cluster.awaitAllInflightZero(5_000);

        // Assert pendingRequests == 0 on every engine (no leak)
        cluster.assertAllInflightZero();

        // Assert runningTasks is empty on every engine
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            assertEquals(0, service.getRunningCount(),
                    "runningTasks should be empty for engine on port " + service.getGrpcPort());
        }

        // Assert no leak was detected
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            assertFalse(service.isLeakDetected(),
                    "leak should not be detected on engine on port " + service.getGrpcPort());
        }
    }

    // ──────────── Cluster setup ────────────

    private void startCluster(MockPerformanceModel model, int nPrefill, int nDecode) throws IOException {
        cluster = MockEngineTestCluster.start(model, BASE_PORT, nPrefill, nDecode);
        services = cluster.services();
        prefillServices = cluster.prefills();
        decodeServices = cluster.decodes();
    }

    // ──────────── Model helper ────────────

    private MockPerformanceModel model(String formula) throws Exception {
        return MockEngineTestSupport.performanceModel(tempDir, formula);
    }

    private static List<EngineRpcService.GenerateOutputsPB> generateStream(
            JavaMockEngineCluster.FastRpcService service,
            EngineRpcService.GenerateInputPB request) {
        List<EngineRpcService.GenerateOutputsPB> outputs = new ArrayList<>();
        CountDownLatch latch = new CountDownLatch(1);
        AtomicReference<Throwable> error = new AtomicReference<>();
        service.generateStreamCall(request, new StreamObserver<>() {
            @Override
            public void onNext(EngineRpcService.GenerateOutputsPB value) {
                outputs.add(value);
            }

            @Override
            public void onError(Throwable throwable) {
                error.set(throwable);
                latch.countDown();
            }

            @Override
            public void onCompleted() {
                latch.countDown();
            }
        });
        try {
            if (!latch.await(10, TimeUnit.SECONDS)) {
                fail("generateStreamCall timeout for requestId " + request.getRequestId());
            }
        } catch (InterruptedException e) {
            fail("interrupted waiting for generateStreamCall");
        }
        if (error.get() != null) {
            throw new AssertionError(error.get());
        }
        return outputs;
    }
}
