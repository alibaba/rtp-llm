package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.scheduler.EndpointEventProjector;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.eviction.EngineCancelChannel.CancelAck;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import static org.flexlb.mockengine.MockEngineTestSupport.workerStatus;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

/**
 * Cross-process 8429 wiring integration test (C1 three-piece set, self-test
 * evidence): a REAL {@link MockControlServer} on a real HTTP port +
 * {@link HttpMockEngineCancelChannel} pointed at it, asserting:
 * <ul>
 *   <li>a live request and its priority-cancel tombstone return ACCEPTED;
 *       completed-before-cancel, unknown, or wrongly routed Prefill requests
 *       return NOT_FOUND,</li>
 *   <li>a Decode target returns HTTP 501 / a failed channel future, matching
 *       the production UNIMPLEMENTED contract,</li>
 *   <li>the raw /cancel_request JSON still exposes the mock control-plane
 *       detail (found / already_finished / phase) for self-test evidence —
 *       including a queued decode request (opt-in
 *       {@code decode.report_queued_as_kv_allocated}) reporting phase
 *       KV_ALLOCATED — the accepted-layer contract Phase 5 eviction needs,</li>
 *   <li>the CANCELLED terminal surfaces in the next WorkerStatus finished
 *       list (iron rule 4 confirmation source),</li>
 *   <li>unknown engine port → UNSUPPORTED (HTTP 404),</li>
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
    private JavaMockEngineCluster.FastRpcService prefillService;
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

    // ──────────── accepted over HTTP: queued request cancelled, CANCELLED surfaces ────────────

    @Test
    void httpCancelOfQueuedRequestAcceptedAndSurfacesCancelled() throws Exception {
        startGatedDecodeCluster(true);
        EngineCancelChannel channel = channel();

        // 1 running + 1 queued (KV_ALLOCATED under the opt-in flag).
        assertTrue(scheduleOwnedDecode(1L));
        assertTrue(scheduleOwnedDecode(2L));

        CancelAck outcome = channel
                .cancel(target(prefillService.getGrpcPort()), 2L, 5_000)
                .get(5, TimeUnit.SECONDS);
        assertEquals(CancelAck.ACCEPTED, outcome,
                "queued request cancel over HTTP must register the intent (ACCEPTED)");

        // Iron rule 4: release confirmation via the next WorkerStatus report.
        EngineRpcService.WorkerStatusPB status = workerStatus(decodeService, 0);
        boolean cancelledReported = status.getFinishedTaskListList().stream()
                .anyMatch(task -> task.getRequestId() == 2L
                        && task.getErrorInfo().getErrorCode()
                        == EngineRpcService.ErrorCodePB.CANCELLED.getNumber());
        assertTrue(cancelledReported,
                "CANCELLED completion must appear in the next WorkerStatus finished list");
        assertFalse(prefillService.hasDownstreamOwnership(2L));
        assertFalse(decodeService.hasUpstreamOwnership(2L));
        assertTrue(prefillService.hasDownstreamOwnership(1L),
                "the unrelated live request must retain its exact ownership");
    }

    @Test
    void rawHttpCancelOfRunningRequestReportsRunningPhase() throws Exception {
        startGatedDecodeCluster(true);

        assertTrue(scheduleOwnedDecode(11L));

        // The channel outcome is intent-only (ACCEPTED, no phase) — the phase
        // evidence lives in the raw control-plane JSON.
        HttpResponse<String> ok = HttpClient.newHttpClient().send(HttpRequest.newBuilder()
                        .uri(URI.create("http://127.0.0.1:" + controlServer.getPort() + "/cancel_request"))
                        .POST(HttpRequest.BodyPublishers.ofString(
                                "{\"port\": " + prefillService.getGrpcPort() + ", \"request_id\": 11}"))
                        .build(),
                HttpResponse.BodyHandlers.ofString());
        assertEquals(200, ok.statusCode());
        JsonNode json = MAPPER.readTree(ok.body());
        assertTrue(json.get("found").asBoolean());
        assertEquals("TASK_PHASE_RUNNING", json.get("phase").asText(),
                "a truly running request must report the RUNNING phase");
    }

    // ──────────── idempotent tombstone / NOT_FOUND unknown request ────────────

    @Test
    void httpRepeatedPriorityCancelStaysAcceptedAndPublishesOneTerminal() throws Exception {
        startGatedDecodeCluster(false);
        EngineCancelChannel channel = channel();

        assertTrue(scheduleOwnedDecode(21L));
        CancelAck first = channel
                .cancel(target(prefillService.getGrpcPort()), 21L, 5_000)
                .get(5, TimeUnit.SECONDS);
        assertEquals(CancelAck.ACCEPTED, first);

        CancelAck second = channel
                .cancel(target(prefillService.getGrpcPort()), 21L, 5_000)
                .get(5, TimeUnit.SECONDS);
        assertEquals(CancelAck.ACCEPTED, second);
        long terminalCount = workerStatus(prefillService, -1).getFinishedTaskListList().stream()
                .filter(task -> task.getRequestId() == 21L
                        && task.getErrorInfo().getErrorCode() == 8429L
                        && task.getPriorityPreemptionProgress()
                        == EngineRpcService.PriorityPreemptionProgressPB
                        .PRIORITY_PREEMPTION_CANCELED)
                .count();
        assertEquals(1L, terminalCount,
                "a retry must not publish a second CANCELED+8429 terminal");
    }

    @Test
    void httpCancelUnknownRequestIsNotFound() throws Exception {
        startGatedDecodeCluster(false);
        EngineCancelChannel channel = channel();

        CancelAck outcome = channel
                .cancel(target(prefillService.getGrpcPort()), 424242L, 5_000)
                .get(5, TimeUnit.SECONDS);
        assertEquals(CancelAck.NOT_FOUND, outcome);
    }

    @Test
    void httpWrongWorkerDoesNotScanOtherServices() throws Exception {
        startGatedDecodeCluster(false);
        assertTrue(scheduleOwnedDecode(23L));

        int wrongPort = BASE_PORT + nextPortOffset++;
        JavaMockEngineCluster.FastRpcService wrongPrefill =
                new JavaMockEngineCluster.FastRpcService(
                        "prefill-wrong", EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL,
                        wrongPort, services, scheduler, decodeService.getPerformance(), 100,
                        new JavaMockEngineCluster.ClusterStats());
        services.put(wrongPort, wrongPrefill);

        CancelAck outcome = channel().cancel(target(wrongPort), 23L, 5_000)
                .get(5, TimeUnit.SECONDS);

        assertEquals(CancelAck.NOT_FOUND, outcome);
        assertTrue(decodeService.getInflightCount() > 0,
                "the control plane must not find and cancel a request on another worker");
    }

    @Test
    void httpDecodeTargetIsUnimplementedAndDoesNotCancelOwnedRequest() throws Exception {
        startGatedDecodeCluster(false);
        assertTrue(scheduleOwnedDecode(24L));

        var future = channel().cancel(target(decodeService.getGrpcPort()), 24L, 5_000);
        assertThrows(ExecutionException.class,
                () -> future.get(5, TimeUnit.SECONDS),
                "HTTP 501 must surface as the channel FAILED path");
        assertTrue(decodeService.getInflightCount() > 0,
                "a Decode-targeted Cancel must not cancel the request");

        HttpResponse<String> raw = HttpClient.newHttpClient().send(
                HttpRequest.newBuilder()
                        .uri(URI.create("http://127.0.0.1:" + controlServer.getPort()
                                + "/cancel_request"))
                        .POST(HttpRequest.BodyPublishers.ofString(
                                "{\"port\": " + decodeService.getGrpcPort()
                                        + ", \"request_id\": 24}"))
                        .build(),
                HttpResponse.BodyHandlers.ofString());
        assertEquals(501, raw.statusCode());
        assertEquals("UNIMPLEMENTED", MAPPER.readTree(raw.body()).path("status").asText());
    }

    // ──────────── unsupported branch + transport failure + isSupported ────────────

    @Test
    void httpCancelUnknownEnginePortMapsToUnsupported() throws Exception {
        startGatedDecodeCluster(false);
        EngineCancelChannel channel = channel();

        assertTrue(channel.isSupported(endpoint(decodeService.getGrpcPort())),
                "a configured control URL supports every endpoint");

        CancelAck outcome = channel
                .cancel(target(59999), 1L, 5_000)
                .get(5, TimeUnit.SECONDS);
        assertEquals(CancelAck.UNSUPPORTED, outcome,
                "unknown engine port (HTTP 404) → UNSUPPORTED");
    }

    @Test
    void deadControlUrlSurfacesAsFailedFutureNotSynchronousThrow() throws Exception {
        startGatedDecodeCluster(false);
        // Port 1 is never listening — connection refused.
        EngineCancelChannel channel = new HttpMockEngineCancelChannel("http://127.0.0.1:1");

        var future = channel.cancel(target(prefillService.getGrpcPort()), 1L, 5_000);
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
                                "{\"port\": " + prefillService.getGrpcPort() + "}"))
                        .build(),
                HttpResponse.BodyHandlers.ofString());
        assertEquals(400, badRequest.statusCode());

        // Non-integer request_id → 400 (P2-3: asLong() would coerce to 0 and
        // silently cancel request 0 instead of rejecting the schema bug).
        HttpResponse<String> textualId = http.send(HttpRequest.newBuilder()
                        .uri(URI.create(base + "/cancel_request"))
                        .POST(HttpRequest.BodyPublishers.ofString(
                                "{\"port\": " + prefillService.getGrpcPort()
                                        + ", \"request_id\": \"abc\"}"))
                        .build(),
                HttpResponse.BodyHandlers.ofString());
        assertEquals(400, textualId.statusCode(), "non-numeric string request_id must be a 400");
        HttpResponse<String> fractionalId = http.send(HttpRequest.newBuilder()
                        .uri(URI.create(base + "/cancel_request"))
                        .POST(HttpRequest.BodyPublishers.ofString(
                                "{\"port\": " + prefillService.getGrpcPort()
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
        assertTrue(scheduleOwnedDecode(31L));
        assertTrue(scheduleOwnedDecode(32L));
        HttpResponse<String> ok = http.send(HttpRequest.newBuilder()
                        .uri(URI.create(base + "/cancel_request"))
                        .POST(HttpRequest.BodyPublishers.ofString(
                                "{\"engine\": \"" + prefillService.getEngineName()
                                        + "\", \"request_id\": 32}"))
                        .build(),
                HttpResponse.BodyHandlers.ofString());
        assertEquals(200, ok.statusCode());
        JsonNode json = MAPPER.readTree(ok.body());
        assertEquals("ACCEPTED", json.get("status").asText());
        assertTrue(json.get("found").asBoolean());
        assertEquals("TASK_PHASE_KV_ALLOCATED", json.get("phase").asText(),
                "queued request phase must serialize as the proto enum name");
        assertFalse(json.get("already_finished").asBoolean());
        assertEquals(prefillService.getGrpcPort(), json.get("port").asInt());
    }

    // ──────────── Setup helpers ────────────

    /**
     * One gated decode engine (decodeMaxConcurrency=1, pending queue cap 4)
     * behind a real MockControlServer on an ephemeral port. Long decode step
     * (10s × 0.1 sleep_scale = 1s) keeps requests in flight during asserts.
     */
    private void startGatedDecodeCluster(boolean reportQueuedAsKvAllocated) throws Exception {
        MockPerformanceModel model = model(10_000.0, 4, reportQueuedAsKvAllocated);
        int prefillPort = BASE_PORT + nextPortOffset++;
        prefillService = new JavaMockEngineCluster.FastRpcService(
                "prefill-0", "127.0.0.1", "prefill",
                EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL,
                prefillPort, services, scheduler, model, 100,
                new JavaMockEngineCluster.ClusterStats(), 10_000_000L, 1);
        services.put(prefillPort, prefillService);
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

    private boolean scheduleOwnedDecode(long requestId) throws Exception {
        prefillService.registerDecodeOwnership(requestId, decodeService);
        boolean accepted = invokeScheduleDecodeCompletion(
                decodeService, shapeOf(requestId), -1, null);
        if (!accepted) {
            prefillService.clearDecodeOwnership(requestId, decodeService);
        }
        return accepted;
    }

    private EngineCancelChannel channel() {
        return new HttpMockEngineCancelChannel("http://127.0.0.1:" + controlServer.getPort());
    }

    private static CancelTarget target(int grpcPort) {
        return new CancelTarget("127.0.0.1", grpcPort);
    }

    private static DecodeEndpoint endpoint(int grpcPort) {
        WorkerStatus status = WorkerStatus.createDiscovered(
                RoleType.DECODE, "test", "127.0.0.1",
                grpcPort - 2, grpcPort, null);
        return new DecodeEndpoint(status, mock(EndpointEventProjector.class));
    }

    private MockPerformanceModel model(double decodeStepMs, Integer maxPendingRequests,
                                       boolean reportQueuedAsKvAllocated) throws Exception {
        Map<String, Object> decodeConfig = new LinkedHashMap<>();
        if (maxPendingRequests != null) {
            decodeConfig.put("max_pending_requests", maxPendingRequests);
        }
        if (reportQueuedAsKvAllocated) {
            decodeConfig.put("report_queued_as_kv_allocated", true);
        }
        return MockEngineTestSupport.performanceModel(
                tempDir, "10", 0.1, decodeStepMs, Map.of(), decodeConfig);
    }

    private MockPerformanceModel.RequestShape shapeOf(long requestId) throws Exception {
        return MockEngineTestSupport.requestShape(
                decodeService.getPerformance(), requestId, 8);
    }

    private static boolean invokeScheduleDecodeCompletion(
            JavaMockEngineCluster.FastRpcService service,
            MockPerformanceModel.RequestShape shape,
            long batchId,
            LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> responseQueue)
            throws Exception {
        return MockEngineTestSupport.scheduleDecodeCompletion(
                service, shape, batchId, responseQueue);
    }
}
