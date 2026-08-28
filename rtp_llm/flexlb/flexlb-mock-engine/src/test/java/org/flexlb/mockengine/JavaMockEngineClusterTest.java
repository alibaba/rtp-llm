package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
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
import java.util.function.Predicate;

import static org.flexlb.mockengine.MockEngineTestSupport.batch;
import static org.flexlb.mockengine.MockEngineTestSupport.enqueue;
import static org.flexlb.mockengine.MockEngineTestSupport.input;
import static org.flexlb.mockengine.MockEngineTestSupport.slot;
import static org.flexlb.mockengine.MockEngineTestSupport.workerStatus;
import static org.junit.jupiter.api.Assertions.assertEquals;
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
        return MockEngineTestSupport.performanceModel(tempDir, formula, sleepScale);
    }

    private static EngineRpcService.WorkerStatusPB awaitStatus(
            JavaMockEngineCluster.FastRpcService service,
            Predicate<EngineRpcService.WorkerStatusPB> predicate,
            long timeoutMs) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        EngineRpcService.WorkerStatusPB last = null;
        while (System.nanoTime() < deadline) {
            last = workerStatus(service, 0);
            if (predicate.test(last)) {
                return last;
            }
            Thread.sleep(5);
        }
        fail("status condition not reached, last status=" + last);
        return last;
    }

}
