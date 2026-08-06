package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.function.Predicate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Cancel semantics and acceptance instrumentation of the mock engine cluster:
 * exactly-once settlement between Cancel and the scheduled completion,
 * counter release visible through WorkerStatus, cancel-record bookkeeping,
 * and the (default-off) input-capture used for priority pass-through
 * assertions in later stages.
 */
class JavaMockEngineClusterCancelTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final long TOTAL_KV_TOKENS = 6_291_456L;
    private static final int PREFILL_PORT = 61_000;
    private static final int DECODE_PORT = 61_001;

    @TempDir
    Path tempDir;

    private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(2);

    @AfterEach
    void tearDown() throws InterruptedException {
        scheduler.shutdownNow();
        scheduler.awaitTermination(2, TimeUnit.SECONDS);
    }

    @Test
    void cancelRunningPrefillTaskSettlesExactlyOnce() throws Exception {
        JavaMockEngineCluster.FastRpcService prefill = prefillService(model("300", 1.0));

        enqueue(prefill, batch(31, slot(0, input(1, 100), input(2, 100))));
        awaitStatus(prefill, status -> status.getRunningQueryLen() == 2, 1_000);

        cancel(prefill, 1);

        // The cancelled task leaves runningTasks immediately and shows up as
        // finished (executionTimeMs == 0) without waiting for batch completion.
        EngineRpcService.WorkerStatusPB afterCancel = awaitStatus(prefill,
                status -> status.getRunningTaskInfoCount() == 1
                        && status.getFinishedTaskListCount() == 1,
                1_000);
        assertEquals(1, afterCancel.getFinishedTaskList(0).getRequestId());
        assertEquals(0, afterCancel.getFinishedTaskList(0).getExecutionTimeMs());
        assertEquals(2, afterCancel.getRunningTaskInfo(0).getRequestId());

        // Batch completion settles only the surviving task — no duplicate
        // finished record for the cancelled one.
        EngineRpcService.WorkerStatusPB finished = awaitStatus(prefill,
                status -> status.getRunningTaskInfoCount() == 0
                        && status.getFinishedTaskListCount() == 2,
                2_000);
        assertEquals(1, finished.getFinishedTaskListList().stream()
                .filter(task -> task.getRequestId() == 1).count());
        assertEquals(1, finished.getFinishedTaskListList().stream()
                .filter(task -> task.getRequestId() == 2).count());
        assertEquals(300, finished.getFinishedTaskListList().stream()
                .filter(task -> task.getRequestId() == 2)
                .findFirst().orElseThrow().getExecutionTimeMs());

        List<JavaMockEngineCluster.FastRpcService.CancelRecord> records = prefill.cancelRecords();
        assertEquals(1, records.size());
        assertEquals(1, records.get(0).requestId());
        assertTrue(records.get(0).foundRunning());
        assertEquals(JavaMockEngineCluster.FastRpcService.CancelRecord.REASON_UNSPECIFIED,
                records.get(0).reason());
    }

    @Test
    void cancelUnknownRequestIsRecordedIdempotentNoOp() throws Exception {
        JavaMockEngineCluster.FastRpcService prefill = prefillService(model("50", 1.0));

        cancel(prefill, 999);
        cancel(prefill, 999);

        EngineRpcService.WorkerStatusPB status = status(prefill);
        assertEquals(0, status.getRunningTaskInfoCount());
        assertEquals(0, status.getFinishedTaskListCount());

        List<JavaMockEngineCluster.FastRpcService.CancelRecord> records = prefill.cancelRecords();
        assertEquals(2, records.size());
        assertFalse(records.get(0).foundRunning());
        assertFalse(records.get(1).foundRunning());
    }

    @Test
    void cancelDecodeTaskReleasesKvAccountingWithoutDoubleSettle() throws Exception {
        Map<Integer, JavaMockEngineCluster.FastRpcService> services = new ConcurrentHashMap<>();
        MockPerformanceModel model = model("1", 1.0);
        JavaMockEngineCluster.FastRpcService prefill = service(
                services, "prefill", EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL, PREFILL_PORT, model);
        JavaMockEngineCluster.FastRpcService decode = service(
                services, "decode", EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE, DECODE_PORT, model);

        // maxNewTokens=500 → decode runs ~500ms, leaving a cancel window.
        enqueue(prefill, batch(41, slot(0, decodeRoutedInput(7, 100, 500))));
        awaitStatus(decode, status -> status.getRunningQueryLen() == 1, 2_000);
        assertEquals(TOTAL_KV_TOKENS - 100, status(decode).getAvailableKvCache());

        cancel(decode, 7);

        EngineRpcService.WorkerStatusPB afterCancel = awaitStatus(decode,
                status -> status.getRunningTaskInfoCount() == 0
                        && status.getFinishedTaskListCount() == 1,
                1_000);
        assertEquals(TOTAL_KV_TOKENS, afterCancel.getAvailableKvCache());
        assertEquals(7, afterCancel.getFinishedTaskList(0).getRequestId());

        // Let the originally scheduled decode completion window elapse: the
        // claimed task must not settle again (no negative KV accounting, no
        // duplicate finished record).
        Thread.sleep(700);
        EngineRpcService.WorkerStatusPB afterWindow = status(decode);
        assertEquals(TOTAL_KV_TOKENS, afterWindow.getAvailableKvCache());
        assertEquals(0, afterWindow.getRunningTaskInfoCount());
        assertEquals(1, afterWindow.getFinishedTaskListCount());
    }

    @Test
    void inputCaptureIsOffByDefaultAndCapturesGenerateConfigWhenEnabled() throws Exception {
        JavaMockEngineCluster.FastRpcService prefill = prefillService(model("10", 1.0));

        enqueue(prefill, batch(51, slot(0, input(11, 10))));
        assertNull(prefill.capturedInput(11), "capture must be off by default");

        prefill.enableInputCapture();
        enqueue(prefill, batch(52, slot(0, input(12, 10))));

        EngineRpcService.GenerateInputPB captured = prefill.capturedInput(12);
        assertNotNull(captured);
        assertEquals(12, captured.getRequestId());
        assertEquals(1, captured.getGenerateConfig().getMaxNewTokens());

        // Priority pass-through parity guard: the engine GenerateConfigPB has
        // no priority field yet. When the Stage 2 protocol change lands, this
        // assertion flips and the captured value becomes assertable here.
        assertNull(EngineRpcService.GenerateConfigPB.getDescriptor().findFieldByName("priority"),
                "GenerateConfigPB.priority not expected before the Stage 2 proto change");
    }

    // ==================== fixtures ====================

    private JavaMockEngineCluster.FastRpcService prefillService(MockPerformanceModel model) {
        Map<Integer, JavaMockEngineCluster.FastRpcService> services = new ConcurrentHashMap<>();
        return service(services, "prefill",
                EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL, PREFILL_PORT, model);
    }

    private JavaMockEngineCluster.FastRpcService service(
            Map<Integer, JavaMockEngineCluster.FastRpcService> services,
            String roleName,
            EngineRpcService.RoleTypePB roleType,
            int grpcPort,
            MockPerformanceModel model) {
        JavaMockEngineCluster.FastRpcService service = new JavaMockEngineCluster.FastRpcService(
                roleName, roleType, grpcPort, services, scheduler, model, 100,
                new JavaMockEngineCluster.ClusterStats());
        services.put(grpcPort, service);
        return service;
    }

    private MockPerformanceModel model(String formula, double sleepScale) throws Exception {
        Path performance = tempDir.resolve("performance-" + System.nanoTime() + ".json");
        Path master = tempDir.resolve("master-" + System.nanoTime() + ".json");
        MAPPER.writeValue(performance.toFile(), Map.of(
                "block_size", 1024,
                "sleep_scale", sleepScale,
                "prefill", Map.of("scale", 1.0),
                "decode", Map.of("scale", 1.0, "step_ms_by_batch", List.of(List.of(1, 1.0)))));
        MAPPER.writeValue(master.toFile(), Map.of(
                "zone_process_setting", Map.of(
                        "process_info", Map.of(
                                "envs", List.of(List.of("PREFILL_TIME_FORMULA", formula))))));
        return MockPerformanceModel.load(performance.toString(), master.toString());
    }

    private static EngineRpcService.GenerateInputPB input(long requestId, int inputTokens) {
        EngineRpcService.GenerateInputPB.Builder input = EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(requestId)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                        .setMaxNewTokens(1)
                        .build());
        for (int token = 0; token < inputTokens; token++) {
            input.addTokenIds(token);
        }
        return input.build();
    }

    private static EngineRpcService.GenerateInputPB decodeRoutedInput(
            long requestId, int inputTokens, int maxNewTokens) {
        EngineRpcService.GenerateInputPB.Builder input = EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(requestId)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                        .setMaxNewTokens(maxNewTokens)
                        .addRoleAddrs(EngineRpcService.RoleAddrPB.newBuilder()
                                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                                .setIp("127.0.0.1")
                                .setGrpcPort(DECODE_PORT)
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

    private static EngineRpcService.EmptyPB cancel(
            JavaMockEngineCluster.FastRpcService service, long requestId) {
        return unary(observer -> service.cancel(
                EngineRpcService.CancelRequestPB.newBuilder().setRequestId(requestId).build(),
                observer));
    }

    private static EngineRpcService.WorkerStatusPB status(
            JavaMockEngineCluster.FastRpcService service) {
        return unary(observer -> service.getWorkerStatus(
                EngineRpcService.StatusVersionPB.newBuilder()
                        .setLatestFinishedVersion(0)
                        .build(),
                observer));
    }

    private static EngineRpcService.WorkerStatusPB awaitStatus(
            JavaMockEngineCluster.FastRpcService service,
            Predicate<EngineRpcService.WorkerStatusPB> predicate,
            long timeoutMs) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        EngineRpcService.WorkerStatusPB last = null;
        while (System.nanoTime() < deadline) {
            last = status(service);
            if (predicate.test(last)) {
                return last;
            }
            Thread.sleep(5);
        }
        fail("status condition not reached, last status=" + last);
        return last;
    }

    private static <T> T unary(Consumer<StreamObserver<T>> invocation) {
        AtomicReference<T> response = new AtomicReference<>();
        AtomicReference<Throwable> error = new AtomicReference<>();
        invocation.accept(new StreamObserver<>() {
            @Override
            public void onNext(T value) {
                response.set(value);
            }

            @Override
            public void onError(Throwable throwable) {
                error.set(throwable);
            }

            @Override
            public void onCompleted() {
            }
        });
        if (error.get() != null) {
            throw new AssertionError(error.get());
        }
        assertNotNull(response.get(), "unary response");
        return response.get();
    }
}
