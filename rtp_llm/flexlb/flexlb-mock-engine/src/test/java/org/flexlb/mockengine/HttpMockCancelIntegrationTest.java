package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.stub.StreamObserver;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.scheduler.priority.EngineCancelChannel;
import org.flexlb.balance.scheduler.priority.EngineCancelChannel.CancelOutcome;
import org.flexlb.balance.scheduler.priority.EngineCancelChannel.CancelReason;
import org.flexlb.balance.scheduler.priority.HttpMockEngineCancelChannel;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.TaskPhase;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.lang.reflect.Method;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Cross-process 8429 wiring integration test (C1 three-piece set, self-test
 * evidence): a REAL {@link MockControlServer} on a real HTTP port +
 * {@link HttpMockEngineCancelChannel} pointed at it, asserting:
 * <ul>
 *   <li>the /cancel_request endpoint drives the three-branch cancelRequest
 *       contract (found / already_finished / not found) over HTTP,</li>
 *   <li>a cancelled QUEUED decode request (opt-in
 *       {@code decode.report_queued_as_kv_allocated}) reports phase
 *       KV_ALLOCATED — the accepted-layer contract Phase 5 eviction needs,</li>
 *   <li>the CANCELLED terminal surfaces in the next WorkerStatus finished
 *       list (iron rule 4 confirmation source),</li>
 *   <li>unknown engine port → unsupported branch (HTTP 404),</li>
 *   <li>transport failure (dead URL) → failed future, never a synchronous
 *       throw.</li>
 * </ul>
 */
class HttpMockCancelIntegrationTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final int BASE_PORT = 63500;

    @TempDir
    Path tempDir;

    private ScheduledExecutorService scheduler;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private MockControlServer controlServer;
    private JavaMockEngineCluster.FastRpcService decodeService;
    private int nextPortOffset;

    @BeforeEach
    void setUp() {
        scheduler = Executors.newScheduledThreadPool(4, runnable -> {
            Thread thread = new Thread(runnable, "http-cancel-integration-scheduler");
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
        }
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            service.shutdown();
        }
        scheduler.shutdownNow();
        scheduler.awaitTermination(3, TimeUnit.SECONDS);
    }

    // ──────────── found branch over HTTP: queued request, KV_ALLOCATED phase ────────────

    @Test
    void httpCancelOfQueuedRequestReportsKvAllocatedAndSurfacesCancelled() throws Exception {
        startGatedDecodeCluster(true);
        EngineCancelChannel channel = channel();

        // 1 running + 1 queued (KV_ALLOCATED under the opt-in flag).
        assertTrue(invokeScheduleDecodeCompletion(decodeService, shapeOf(1L), -1, null));
        assertTrue(invokeScheduleDecodeCompletion(decodeService, shapeOf(2L), -1, null));

        CancelOutcome outcome = channel
                .cancel(endpoint(decodeService.getGrpcPort()), 2L, CancelReason.PRIORITY_PREEMPTED)
                .get(5, TimeUnit.SECONDS);
        assertTrue(outcome.found(), "queued request must be found over HTTP");
        assertFalse(outcome.alreadyFinished());
        assertFalse(outcome.unsupported());
        assertEquals(TaskPhase.KV_ALLOCATED, outcome.phase(),
                "8429 wiring evidence: HTTP cancel of an accepted (queued) "
                        + "request must carry the KV_ALLOCATED phase");

        // Iron rule 4: release confirmation via the next WorkerStatus report.
        EngineRpcService.WorkerStatusPB status = workerStatus(decodeService, 0);
        boolean cancelledReported = status.getFinishedTaskListList().stream()
                .anyMatch(task -> task.getRequestId() == 2L
                        && task.getErrorInfo().getErrorCode()
                        == EngineRpcService.ErrorCodePB.CANCELLED.getNumber());
        assertTrue(cancelledReported,
                "CANCELLED completion must appear in the next WorkerStatus finished list");
    }

    @Test
    void httpCancelOfRunningRequestReportsRunningPhase() throws Exception {
        startGatedDecodeCluster(true);
        EngineCancelChannel channel = channel();

        assertTrue(invokeScheduleDecodeCompletion(decodeService, shapeOf(11L), -1, null));

        CancelOutcome outcome = channel
                .cancel(endpoint(decodeService.getGrpcPort()), 11L, CancelReason.ADMIN)
                .get(5, TimeUnit.SECONDS);
        assertTrue(outcome.found());
        assertEquals(TaskPhase.RUNNING, outcome.phase(),
                "a truly running request must report the RUNNING phase");
    }

    // ──────────── already_finished / notFound branches over HTTP ────────────

    @Test
    void httpDoubleCancelReportsFinishedBeforeCancel() throws Exception {
        startGatedDecodeCluster(false);
        EngineCancelChannel channel = channel();

        assertTrue(invokeScheduleDecodeCompletion(decodeService, shapeOf(21L), -1, null));
        CancelOutcome first = channel
                .cancel(endpoint(decodeService.getGrpcPort()), 21L, CancelReason.ADMIN)
                .get(5, TimeUnit.SECONDS);
        assertTrue(first.found());

        CancelOutcome second = channel
                .cancel(endpoint(decodeService.getGrpcPort()), 21L, CancelReason.ADMIN)
                .get(5, TimeUnit.SECONDS);
        assertTrue(second.alreadyFinished(), "second cancel must be idempotent");
        assertTrue(second.found(), "finishedBeforeCancel contract is found=true");
        assertNull(second.phase());
    }

    @Test
    void httpCancelUnknownRequestReportsNotFound() throws Exception {
        startGatedDecodeCluster(false);
        EngineCancelChannel channel = channel();

        CancelOutcome outcome = channel
                .cancel(endpoint(decodeService.getGrpcPort()), 424242L,
                        CancelReason.PRIORITY_PREEMPTED)
                .get(5, TimeUnit.SECONDS);
        assertFalse(outcome.found());
        assertFalse(outcome.alreadyFinished());
        assertFalse(outcome.unsupported());
    }

    // ──────────── unsupported branch + transport failure + isSupported ────────────

    @Test
    void httpCancelUnknownEnginePortMapsToUnsupported() throws Exception {
        startGatedDecodeCluster(false);
        EngineCancelChannel channel = channel();

        assertTrue(channel.isSupported(endpoint(decodeService.getGrpcPort())),
                "a configured control URL supports every endpoint");

        CancelOutcome outcome = channel
                .cancel(endpoint(59999), 1L, CancelReason.ADMIN)
                .get(5, TimeUnit.SECONDS);
        assertTrue(outcome.unsupported(), "unknown engine port (HTTP 404) → unsupported branch");
        assertFalse(outcome.found());
    }

    @Test
    void deadControlUrlSurfacesAsFailedFutureNotSynchronousThrow() throws Exception {
        startGatedDecodeCluster(false);
        // Port 1 is never listening — connection refused.
        EngineCancelChannel channel = new HttpMockEngineCancelChannel("http://127.0.0.1:1");

        var future = channel.cancel(endpoint(decodeService.getGrpcPort()), 1L, CancelReason.ADMIN);
        assertNotNull(future, "cancel must never throw synchronously");
        assertThrows(ExecutionException.class, () -> future.get(5, TimeUnit.SECONDS),
                "transport failure must surface as a failed future");
    }

    // ──────────── raw endpoint schema check (curl-equivalent evidence) ────────────

    @Test
    void rawCancelRequestEndpointSchemaAndValidation() throws Exception {
        startGatedDecodeCluster(true);
        String base = "http://127.0.0.1:" + controlServer.getPort();
        HttpClient http = HttpClient.newHttpClient();

        // Missing request_id → 400.
        HttpResponse<String> badRequest = http.send(HttpRequest.newBuilder()
                        .uri(URI.create(base + "/cancel_request"))
                        .POST(HttpRequest.BodyPublishers.ofString(
                                "{\"port\": " + decodeService.getGrpcPort() + "}"))
                        .build(),
                HttpResponse.BodyHandlers.ofString());
        assertEquals(400, badRequest.statusCode());

        // Non-integer request_id → 400 (P2-3: asLong() would coerce to 0 and
        // silently cancel request 0 instead of rejecting the schema bug).
        HttpResponse<String> textualId = http.send(HttpRequest.newBuilder()
                        .uri(URI.create(base + "/cancel_request"))
                        .POST(HttpRequest.BodyPublishers.ofString(
                                "{\"port\": " + decodeService.getGrpcPort()
                                        + ", \"request_id\": \"abc\"}"))
                        .build(),
                HttpResponse.BodyHandlers.ofString());
        assertEquals(400, textualId.statusCode(), "non-numeric string request_id must be a 400");
        HttpResponse<String> fractionalId = http.send(HttpRequest.newBuilder()
                        .uri(URI.create(base + "/cancel_request"))
                        .POST(HttpRequest.BodyPublishers.ofString(
                                "{\"port\": " + decodeService.getGrpcPort()
                                        + ", \"request_id\": 1.5}"))
                        .build(),
                HttpResponse.BodyHandlers.ofString());
        assertEquals(400, fractionalId.statusCode(), "fractional request_id must be a 400");

        // GET → 405.
        HttpResponse<String> wrongMethod = http.send(HttpRequest.newBuilder()
                        .uri(URI.create(base + "/cancel_request"))
                        .GET()
                        .build(),
                HttpResponse.BodyHandlers.ofString());
        assertEquals(405, wrongMethod.statusCode());

        // Engine-name addressing (Python-compat dual addressing) + full schema.
        assertTrue(invokeScheduleDecodeCompletion(decodeService, shapeOf(31L), -1, null));
        assertTrue(invokeScheduleDecodeCompletion(decodeService, shapeOf(32L), -1, null));
        HttpResponse<String> ok = http.send(HttpRequest.newBuilder()
                        .uri(URI.create(base + "/cancel_request"))
                        .POST(HttpRequest.BodyPublishers.ofString(
                                "{\"engine\": \"" + decodeService.getEngineName()
                                        + "\", \"request_id\": 32}"))
                        .build(),
                HttpResponse.BodyHandlers.ofString());
        assertEquals(200, ok.statusCode());
        JsonNode json = MAPPER.readTree(ok.body());
        assertTrue(json.get("found").asBoolean());
        assertEquals("TASK_PHASE_KV_ALLOCATED", json.get("phase").asText(),
                "queued request phase must serialize as the proto enum name");
        assertFalse(json.get("already_finished").asBoolean());
        assertEquals(decodeService.getGrpcPort(), json.get("port").asInt());
    }

    // ──────────── Setup helpers ────────────

    /**
     * One gated decode engine (decodeMaxConcurrency=1, pending queue cap 4)
     * behind a real MockControlServer on an ephemeral port. Long decode step
     * (10s × 0.1 sleep_scale = 1s) keeps requests in flight during asserts.
     */
    private void startGatedDecodeCluster(boolean reportQueuedAsKvAllocated) throws Exception {
        MockPerformanceModel model = model(10_000.0, 4, reportQueuedAsKvAllocated);
        int port = BASE_PORT + nextPortOffset++;
        decodeService = new JavaMockEngineCluster.FastRpcService(
                "decode-0", "127.0.0.1", "decode",
                EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE,
                port, services, scheduler, model, 100,
                new JavaMockEngineCluster.ClusterStats(), 10_000_000L, 1);
        services.put(port, decodeService);
        controlServer = new MockControlServer(services, new ConcurrentHashMap<>(),
                null, null, "127.0.0.1", 0);
        controlServer.start();
    }

    private EngineCancelChannel channel() {
        return new HttpMockEngineCancelChannel("http://127.0.0.1:" + controlServer.getPort());
    }

    private static DecodeEndpoint endpoint(int grpcPort) {
        WorkerStatus status = new WorkerStatus();
        status.setIp("127.0.0.1");
        status.setPort(grpcPort - 2);
        status.setGrpcPort(grpcPort);
        return new DecodeEndpoint(status);
    }

    private MockPerformanceModel model(double decodeStepMs, Integer maxPendingRequests,
                                       boolean reportQueuedAsKvAllocated) throws Exception {
        Path performance = tempDir.resolve("performance-" + System.nanoTime() + ".json");
        Path master = tempDir.resolve("master-" + System.nanoTime() + ".json");
        Map<String, Object> decodeConfig = new LinkedHashMap<>();
        decodeConfig.put("scale", 1.0);
        decodeConfig.put("step_ms_by_batch", List.of(List.of(1, decodeStepMs)));
        if (maxPendingRequests != null) {
            decodeConfig.put("max_pending_requests", maxPendingRequests);
        }
        if (reportQueuedAsKvAllocated) {
            decodeConfig.put("report_queued_as_kv_allocated", true);
        }
        MAPPER.writeValue(performance.toFile(), Map.of(
                "block_size", 1024,
                "sleep_scale", 0.1,
                "jitter_pct", 0.0,
                "prefill", Map.of("scale", 1.0),
                "decode", decodeConfig));
        MAPPER.writeValue(master.toFile(), Map.of(
                "zone_process_setting", Map.of(
                        "process_info", Map.of(
                                "envs", List.of(List.of("PREFILL_TIME_FORMULA", "10"))))));
        return MockPerformanceModel.load(performance.toString(), master.toString());
    }

    private MockPerformanceModel.RequestShape shapeOf(long requestId) throws Exception {
        EngineRpcService.GenerateInputPB.Builder input = EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(requestId)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                        .setMaxNewTokens(1)
                        .build());
        for (int token = 0; token < 8; token++) {
            input.addTokenIds(token);
        }
        return decodeService.getPerformance().shape(input.build(), new MockLruBlockCache(100));
    }

    private static EngineRpcService.WorkerStatusPB workerStatus(
            JavaMockEngineCluster.FastRpcService service, long sinceVersion) {
        AtomicReference<EngineRpcService.WorkerStatusPB> response = new AtomicReference<>();
        AtomicReference<Throwable> error = new AtomicReference<>();
        CountDownLatch latch = new CountDownLatch(1);
        service.getWorkerStatus(
                EngineRpcService.StatusVersionPB.newBuilder()
                        .setLatestFinishedVersion(sinceVersion)
                        .build(),
                new StreamObserver<>() {
                    @Override
                    public void onNext(EngineRpcService.WorkerStatusPB value) {
                        response.set(value);
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
            if (!latch.await(5, TimeUnit.SECONDS)) {
                fail("worker status timeout");
            }
        } catch (InterruptedException e) {
            fail("interrupted waiting for worker status");
        }
        if (error.get() != null) {
            fail(String.valueOf(error.get()));
        }
        assertNotNull(response.get());
        return response.get();
    }

    private static boolean invokeScheduleDecodeCompletion(
            JavaMockEngineCluster.FastRpcService service,
            MockPerformanceModel.RequestShape shape,
            long batchId,
            LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> responseQueue)
            throws Exception {
        Method method = JavaMockEngineCluster.FastRpcService.class.getDeclaredMethod(
                "scheduleDecodeCompletion",
                MockPerformanceModel.RequestShape.class,
                long.class,
                LinkedBlockingQueue.class);
        method.setAccessible(true);
        return (Boolean) method.invoke(service, shape, batchId, responseQueue);
    }
}
