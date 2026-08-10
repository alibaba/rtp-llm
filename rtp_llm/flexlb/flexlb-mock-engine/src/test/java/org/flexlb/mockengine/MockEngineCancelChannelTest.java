package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.stub.StreamObserver;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.scheduler.priority.EngineCancelChannel;
import org.flexlb.balance.scheduler.priority.EngineCancelChannel.CancelAck;
import org.flexlb.balance.scheduler.priority.EngineCancelChannel.CancelOutcome;
import org.flexlb.balance.scheduler.priority.EngineCancelChannel.CancelReason;
import org.flexlb.balance.scheduler.priority.EngineCancelChannel.CancelTarget;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * {@link MockEngineCancelChannel} contract tests against the in-process mock
 * engine cluster (test wiring only — production Spring contexts keep
 * UnsupportedEngineCancelChannel):
 * <ul>
 *   <li>accepted: any routable cancel (mid-flight, after completion, double
 *       cancel, unknown request id) registers the intent and acks ACCEPTED;
 *       for a mid-flight cancel the CANCELLED completion surfaces in the next
 *       WorkerStatus finished list (iron rule 4 confirmation source),</li>
 *   <li>unsupported: endpoint whose port maps to no mock engine.</li>
 * </ul>
 */
class MockEngineCancelChannelTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final int BASE_PORT = 62500;

    @TempDir
    Path tempDir;

    private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(8);
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private List<JavaMockEngineCluster.FastRpcService> prefillServices;
    private List<JavaMockEngineCluster.FastRpcService> decodeServices;

    @AfterEach
    void tearDown() throws InterruptedException {
        if (services != null) {
            for (JavaMockEngineCluster.FastRpcService service : services.values()) {
                service.shutdown();
            }
        }
        scheduler.shutdownNow();
        scheduler.awaitTermination(3, TimeUnit.SECONDS);
    }

    // ---- accepted: mid-flight cancel drives the mock ----

    @Test
    void cancelMidFlightAcceptedAndCancelledSurfacesInWorkerStatus() throws Exception {
        startCluster(model("500"), 1, 1);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);
        EngineCancelChannel channel = new MockEngineCancelChannel(services);

        int n = 4;
        EngineRpcService.GenerateInputPB[] inputs = new EngineRpcService.GenerateInputPB[n];
        for (int i = 0; i < n; i++) {
            inputs[i] = inputWithDecode(i + 1, 10, decodeServices.get(0).getGrpcPort());
        }
        EngineRpcService.EnqueueBatchResponsePB response =
                enqueue(prefill, batch(9000, slot(0, inputs)));
        assertEquals(n, response.getSuccessesCount());
        awaitInflight(prefill, 1, 1_000);

        CancelOutcome outcome = channel
                .cancel(CancelTarget.of(endpoint(prefill.getGrpcPort()), 0L), 1L, CancelReason.PRIORITY_PREEMPTED)
                .get(2, TimeUnit.SECONDS);
        assertEquals(CancelAck.ACCEPTED, outcome.ack(),
                "mid-flight cancel must register the intent");

        // Iron rule 4: the CANCELLED terminal must surface in WorkerStatus finished list.
        EngineRpcService.WorkerStatusPB status = workerStatus(prefill, 0);
        boolean cancelledReported = status.getFinishedTaskListList().stream()
                .anyMatch(task -> task.getRequestId() == 1L
                        && task.getErrorInfo().getErrorCode()
                        == EngineRpcService.ErrorCodePB.CANCELLED.getNumber());
        assertTrue(cancelledReported,
                "CANCELLED completion for request 1 must appear in WorkerStatus finished list");

        awaitAllInflightZero(10_000);
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            assertFalse(service.isLeakDetected(),
                    "no leak on port " + service.getGrpcPort());
        }
    }

    // ---- accepted: cancel after completion / double cancel (idempotent no-op) ----

    @Test
    void cancelAfterCompletionStillAccepted() throws Exception {
        startCluster(model("10"), 1, 1);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);
        EngineCancelChannel channel = new MockEngineCancelChannel(services);

        enqueue(prefill, batch(9100, slot(0,
                inputWithDecode(11, 10, decodeServices.get(0).getGrpcPort()))));
        awaitAllInflightZero(5_000);

        CancelOutcome outcome = channel
                .cancel(CancelTarget.of(endpoint(prefill.getGrpcPort()), 0L), 11L, CancelReason.USER_CANCELLED)
                .get(2, TimeUnit.SECONDS);
        // Intent registration: the ack carries no terminal info — a cancel
        // landing after completion is still ACCEPTED (engine-side no-op).
        assertEquals(CancelAck.ACCEPTED, outcome.ack());
        // Behavior: the request had already finished; nothing is re-inflight.
        assertEquals(0, prefill.getInflightCount());
    }

    @Test
    void doubleCancelIsIdempotentAndAccepted() throws Exception {
        startCluster(model("500"), 1, 1);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);
        EngineCancelChannel channel = new MockEngineCancelChannel(services);

        enqueue(prefill, batch(9200, slot(0,
                inputWithDecode(21, 10, decodeServices.get(0).getGrpcPort()))));
        awaitInflight(prefill, 1, 1_000);

        CancelOutcome first = channel
                .cancel(CancelTarget.of(endpoint(prefill.getGrpcPort()), 0L), 21L, CancelReason.ADMIN)
                .get(2, TimeUnit.SECONDS);
        assertEquals(CancelAck.ACCEPTED, first.ack());

        CancelOutcome second = channel
                .cancel(CancelTarget.of(endpoint(prefill.getGrpcPort()), 0L), 21L, CancelReason.ADMIN)
                .get(2, TimeUnit.SECONDS);
        assertEquals(CancelAck.ACCEPTED, second.ack(), "second cancel must be idempotent");
        awaitAllInflightZero(10_000);
    }

    // ---- accepted: unknown request id (intent registered, engine no-op) ----

    @Test
    void cancelUnknownRequestStillAccepted() throws Exception {
        startCluster(model("10"), 1, 1);
        EngineCancelChannel channel = new MockEngineCancelChannel(services);

        CancelOutcome outcome = channel
                .cancel(CancelTarget.of(endpoint(prefillServices.get(0).getGrpcPort()), 0L), 424242L,
                        CancelReason.PRIORITY_PREEMPTED)
                .get(2, TimeUnit.SECONDS);
        assertEquals(CancelAck.ACCEPTED, outcome.ack(),
                "unknown request id is still ACCEPTED (intent registered, engine no-op)");
    }

    // ---- unsupported branch + isSupported gate ----

    @Test
    void unknownPortIsUnsupported() throws Exception {
        startCluster(model("10"), 1, 1);
        EngineCancelChannel channel = new MockEngineCancelChannel(services);

        assertTrue(channel.isSupported(endpoint(prefillServices.get(0).getGrpcPort())));
        assertTrue(channel.isSupported(endpoint(decodeServices.get(0).getGrpcPort())));
        assertFalse(channel.isSupported(endpoint(59999)));

        CancelOutcome outcome = channel
                .cancel(CancelTarget.of(endpoint(59999), 0L), 1L, CancelReason.ADMIN)
                .get(2, TimeUnit.SECONDS);
        assertEquals(CancelAck.UNSUPPORTED, outcome.ack());
    }

    // ---- helpers ----

    private static DecodeEndpoint endpoint(int grpcPort) {
        WorkerStatus status = new WorkerStatus();
        status.setIp("127.0.0.1");
        status.setPort(grpcPort - 2);
        status.setGrpcPort(grpcPort);
        return new DecodeEndpoint(status);
    }

    private void startCluster(MockPerformanceModel model, int nPrefill, int nDecode) {
        services = new ConcurrentHashMap<>();
        prefillServices = new ArrayList<>();
        decodeServices = new ArrayList<>();

        for (int i = 0; i < nPrefill; i++) {
            int port = BASE_PORT + i;
            JavaMockEngineCluster.FastRpcService service = new JavaMockEngineCluster.FastRpcService(
                    "prefill", EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL,
                    port, services, scheduler, model, 100,
                    new JavaMockEngineCluster.ClusterStats());
            services.put(port, service);
            prefillServices.add(service);
        }
        for (int i = 0; i < nDecode; i++) {
            int port = BASE_PORT + nPrefill + i;
            JavaMockEngineCluster.FastRpcService service = new JavaMockEngineCluster.FastRpcService(
                    "decode", EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE,
                    port, services, scheduler, model, 100,
                    new JavaMockEngineCluster.ClusterStats());
            services.put(port, service);
            decodeServices.add(service);
        }
    }

    private MockPerformanceModel model(String formula) throws IOException {
        Path performance = tempDir.resolve("performance-" + System.nanoTime() + ".json");
        Path master = tempDir.resolve("master-" + System.nanoTime() + ".json");
        MAPPER.writeValue(performance.toFile(), Map.of(
                "block_size", 1024,
                "sleep_scale", 1.0,
                "jitter_pct", 0.0,
                "prefill", Map.of("scale", 1.0),
                "decode", Map.of("scale", 1.0, "step_ms_by_batch", List.of(List.of(1, 1.0)))));
        MAPPER.writeValue(master.toFile(), Map.of(
                "zone_process_setting", Map.of(
                        "process_info", Map.of(
                                "envs", List.of(List.of("PREFILL_TIME_FORMULA", formula))))));
        return MockPerformanceModel.load(performance.toString(), master.toString());
    }

    private void awaitInflight(JavaMockEngineCluster.FastRpcService service, int min, long timeoutMs)
            throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (System.nanoTime() < deadline) {
            if (service.getInflightCount() >= min) {
                return;
            }
            Thread.sleep(5);
        }
        fail("inflight never reached " + min + " on port " + service.getGrpcPort());
    }

    private void awaitAllInflightZero(long timeoutMs) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (System.nanoTime() < deadline) {
            if (services.values().stream().allMatch(s -> s.getInflightCount() == 0)) {
                return;
            }
            Thread.sleep(10);
        }
        StringBuilder sb = new StringBuilder("inflight not zero: ");
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            sb.append("port=").append(service.getGrpcPort())
                    .append(" inflight=").append(service.getInflightCount()).append(" ");
        }
        fail(sb.toString());
    }

    private static EngineRpcService.WorkerStatusPB workerStatus(
            JavaMockEngineCluster.FastRpcService service, long sinceVersion) {
        return unary(observer -> service.getWorkerStatus(
                EngineRpcService.StatusVersionPB.newBuilder()
                        .setLatestFinishedVersion(sinceVersion)
                        .build(),
                observer));
    }

    private static EngineRpcService.GenerateInputPB inputWithDecode(
            long requestId, int inputTokens, int decodePort) {
        EngineRpcService.GenerateInputPB.Builder input = EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(requestId)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                        .setMaxNewTokens(1)
                        .addRoleAddrs(EngineRpcService.RoleAddrPB.newBuilder()
                                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                                .setGrpcPort(decodePort)
                                .build())
                        .build());
        for (int token = 0; token < inputTokens; token++) {
            input.addTokenIds(token);
        }
        return input.build();
    }

    private static EngineRpcService.EnqueueBatchDpSlotPB slot(
            int dpRank, EngineRpcService.GenerateInputPB... inputs) {
        EngineRpcService.EnqueueBatchDpSlotPB.Builder slot =
                EngineRpcService.EnqueueBatchDpSlotPB.newBuilder().setDpRank(dpRank);
        for (EngineRpcService.GenerateInputPB input : inputs) {
            slot.addRequests(EngineRpcService.EnqueueBatchExternalInputPB.newBuilder()
                    .setInput(input)
                    .build());
        }
        return slot.build();
    }

    private static EngineRpcService.EnqueueBatchRequestPB batch(
            long batchId, EngineRpcService.EnqueueBatchDpSlotPB... slots) {
        return EngineRpcService.EnqueueBatchRequestPB.newBuilder()
                .setBatchId(batchId)
                .addAllDpSlots(List.of(slots))
                .build();
    }

    private static EngineRpcService.EnqueueBatchResponsePB enqueue(
            JavaMockEngineCluster.FastRpcService service,
            EngineRpcService.EnqueueBatchRequestPB request) {
        return unary(observer -> service.enqueueBatch(request, observer));
    }

    private static <T> T unary(Consumer<StreamObserver<T>> invocation) {
        AtomicReference<T> response = new AtomicReference<>();
        AtomicReference<Throwable> error = new AtomicReference<>();
        CountDownLatch latch = new CountDownLatch(1);
        invocation.accept(new StreamObserver<>() {
            @Override
            public void onNext(T value) {
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
                fail("unary response timeout");
            }
        } catch (InterruptedException e) {
            fail("interrupted waiting for unary response");
        }
        if (error.get() != null) {
            throw new AssertionError(error.get());
        }
        assertNotNull(response.get(), "unary response");
        return response.get();
    }
}
