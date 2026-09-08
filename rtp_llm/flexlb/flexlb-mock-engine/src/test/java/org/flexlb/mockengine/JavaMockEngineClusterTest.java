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
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

class JavaMockEngineClusterTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @TempDir
    Path tempDir;

    private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(2);

    @AfterEach
    void tearDown() throws InterruptedException {
        scheduler.shutdownNow();
        scheduler.awaitTermination(2, TimeUnit.SECONDS);
    }

    @Test
    void prefillFormulaUsesAllTasksInBatch() throws Exception {
        MockPerformanceModel model = model(
                "10 + 5*batchSize + 0.01*sum(computeTokens)", 1.0);
        MockLruBlockCache cache = new MockLruBlockCache(100);
        MockPerformanceModel.RequestShape first = model.shape(input(1, 100), cache);
        MockPerformanceModel.RequestShape second = model.shape(input(2, 100), cache);

        long singleMs = model.prefillMs(List.of(first));
        long batchMs = model.prefillMs(List.of(first, second));

        assertTrue(batchMs > singleMs,
                "batch duration must be computed from the complete task set");
    }

    @Test
    void ackIsImmediateWhileQueuedBatchTransitionsFromWaitingToRunningToFinished() throws Exception {
        JavaMockEngineCluster.FastRpcService service = service(model("180", 1.0));

        EngineRpcService.EnqueueBatchResponsePB firstAck = enqueue(
                service, batch(11, slot(0, input(1, 100), input(2, 200))));
        EngineRpcService.EnqueueBatchResponsePB secondAck = enqueue(
                service, batch(12, slot(0, input(3, 300), input(4, 400), input(5, 500))));

        assertEquals(2, firstAck.getSuccessesCount());
        assertEquals(3, secondAck.getSuccessesCount());

        EngineRpcService.WorkerStatusPB initial = awaitStatus(service,
                status -> status.getRunningTaskInfoCount() == 5
                        && status.getRunningQueryLen() == 2
                        && status.getWaitingQueryLen() == 3,
                1_000);
        assertEquals(0, initial.getFinishedTaskListCount());
        assertEquals(3, initial.getRunningTaskInfoList().stream()
                .filter(task -> task.getPhase() == EngineRpcService.TaskPhase.TASK_PHASE_RECEIVED)
                .count());
        assertTrue(initial.getRunningTaskInfoList().stream()
                .filter(task -> task.getBatchId() == 12)
                .allMatch(task -> task.getPhase() == EngineRpcService.TaskPhase.TASK_PHASE_RECEIVED));

        EngineRpcService.WorkerStatusPB secondRunning = awaitStatus(service,
                status -> status.getRunningTaskInfoCount() == 3
                        && status.getRunningQueryLen() == 3
                        && status.getWaitingQueryLen() == 0
                        && status.getFinishedTaskListCount() == 2,
                2_000);
        assertTrue(secondRunning.getRunningTaskInfoList().stream()
                .allMatch(task -> task.getBatchId() == 12));

        EngineRpcService.WorkerStatusPB finished = awaitStatus(service,
                status -> status.getRunningTaskInfoCount() == 0
                        && status.getFinishedTaskListCount() == 5,
                2_000);
        assertTrue(finished.getFinishedTaskListList().stream()
                .allMatch(task -> task.getExecutionTimeMs() == 180));
    }

    @Test
    void dpSlotsAreIndependentBatchesWithPerSlotExecutionTime() throws Exception {
        JavaMockEngineCluster.FastRpcService service = service(model("100*batchSize", 1.0));

        enqueue(service, batch(21,
                slot(0, input(1, 100)),
                slot(1, input(2, 100))));

        EngineRpcService.WorkerStatusPB finished = awaitStatus(service,
                status -> status.getFinishedTaskListCount() == 2,
                2_000);
        assertTrue(finished.getFinishedTaskListList().stream()
                .allMatch(task -> task.getExecutionTimeMs() == 100));
        assertEquals(List.of(0L, 1L), finished.getFinishedTaskListList().stream()
                .map(EngineRpcService.TaskInfoPB::getDpRank)
                .sorted()
                .toList());
    }

    private JavaMockEngineCluster.FastRpcService service(MockPerformanceModel model) {
        Map<Integer, JavaMockEngineCluster.FastRpcService> services = new ConcurrentHashMap<>();
        JavaMockEngineCluster.FastRpcService service = new JavaMockEngineCluster.FastRpcService(
                "prefill",
                EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL,
                61_000,
                services,
                scheduler,
                model,
                100,
                new JavaMockEngineCluster.ClusterStats());
        services.put(61_000, service);
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
