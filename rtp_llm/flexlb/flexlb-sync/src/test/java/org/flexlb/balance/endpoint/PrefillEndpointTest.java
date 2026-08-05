package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.BatchIdGenerator;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

class PrefillEndpointTest {

    private PrefillEndpoint endpoint;
    private FlexlbConfig config;

    @BeforeEach
    void setUp() {
        WorkerStatus status = new WorkerStatus();
        status.setIp("127.0.0.1");
        status.setPort(8080);
        status.setGrpcPort(8090);
        status.setRole(RoleType.PREFILL);

        config = new FlexlbConfig();
        config.setFlexlbBatchQueueMaxSize(100);
        config.setFlexlbBatchFixedWaitMs(300);
        config.setCostFormula("10 + 0.1*sum(computeTokens) + 5*batchSize");

        endpoint = new PrefillEndpoint(status, config,
                mock(EngineGrpcClient.class), mock(BatchDispatchExecutor.class),
                new BatchIdGenerator("127.0.0.1", 7001), () -> 0,
                mock(BatchSchedulerReporter.class), null);
    }

    @AfterEach
    void tearDown() {
        endpoint.close();
    }

    // ---- batch commit / release ----

    @Test
    void commitBatchIncreasesInflightCount() {
        assertEquals(0, trackedEntryCount());

        BatchItem item = createBatchItem(1L, 500, 200);
        endpoint.commitBatch(1L, 100, List.of(item));

        assertEquals(1, trackedEntryCount());
        assertEquals(1, endpoint.prefillPendingRequestCount());
    }

    @Test
    void releaseBatchDecreasesInflightCount() {
        BatchItem item = createBatchItem(1L, 500, 200);
        endpoint.commitBatch(1L, 100, List.of(item));
        endpoint.releaseBatch(1L);

        assertEquals(0, trackedEntryCount());
    }

    @Test
    void releaseBatchNonExistentDoesNotThrow() {
        endpoint.releaseBatch(999L); // should not throw
        assertEquals(0, trackedEntryCount());
    }

    @Test
    void commitMultipleBatches() {
        BatchItem item1 = createBatchItem(1L, 500, 200);
        BatchItem item2 = createBatchItem(2L, 300, 100);
        BatchItem item3 = createBatchItem(3L, 400, 0);

        endpoint.commitBatch(1L, 100, List.of(item1, item2));
        endpoint.commitBatch(2L, 50, List.of(item3));

        assertEquals(2, trackedEntryCount());
        assertEquals(3, endpoint.prefillPendingRequestCount());
    }

    // ---- repack batch ----

    @Test
    void repackBatchRemovesFailedRequests() {
        BatchItem item1 = createBatchItem(1L, 500, 200);
        BatchItem item2 = createBatchItem(2L, 300, 100);
        endpoint.commitBatch(1L, 100, List.of(item1, item2));

        endpoint.repackBatch(1L, Set.of(2L));
        assertEquals(1, trackedEntryCount());
        assertEquals(1, endpoint.prefillPendingRequestCount());
    }

    @Test
    void repackBatchAllFailedReturnsNull() {
        BatchItem item1 = createBatchItem(1L, 500, 200);
        endpoint.commitBatch(1L, 100, List.of(item1));

        endpoint.repackBatch(1L, Set.of(1L));
        assertEquals(0, trackedEntryCount());
    }

    // ---- calibrate ----

    @Test
    void calibrateRemovesBatchOnSuccess() {
        BatchItem item = createBatchItem(1L, 500, 200);
        endpoint.commitBatch(1L, 100, List.of(item));

        Map<String, TaskInfo> finished = new HashMap<>();
        TaskInfo successTask = new TaskInfo();
        successTask.setRequestId(1L);
        successTask.setBatchId(1L);
        successTask.setErrorCode(0);
        finished.put("1", successTask);

        calibrate(finished, Map.of());

        assertEquals(0, trackedEntryCount());
    }

    @Test
    void calibrateRepacksOnPartialFailure() {
        BatchItem item1 = createBatchItem(1L, 500, 200);
        BatchItem item2 = createBatchItem(2L, 300, 100);
        endpoint.commitBatch(1L, 100, List.of(item1, item2));

        Map<String, TaskInfo> finished = new HashMap<>();
        TaskInfo failedTask = new TaskInfo();
        failedTask.setRequestId(2L);
        failedTask.setBatchId(1L);
        failedTask.setErrorCode(500);
        failedTask.setErrorMessage("engine error");
        finished.put("2", failedTask);

        calibrate(finished, Map.of());

        assertEquals(1, trackedEntryCount());
        assertEquals(1, endpoint.prefillPendingRequestCount());
    }

    @Test
    void calibrateHandlesTaskWithNoBatchId() {
        BatchItem item = createBatchItem(1L, 500, 200);
        endpoint.commitBatch(1L, 100, List.of(item));

        Map<String, TaskInfo> finished = new HashMap<>();
        TaskInfo badTask = new TaskInfo();
        badTask.setRequestId(999L); // non-colliding: won't match batchId=1
        badTask.setBatchId(-1);
        badTask.setErrorCode(0);
        finished.put("1", badTask);

        // should not throw, just log a warning for missing non-batch inflight
        calibrate(finished, Map.of());
        assertEquals(1, trackedEntryCount());
    }

    @Test
    void calibrateDoesNotRemoveBatchWithForeignRequestId() {
        // Commit batch with requestId=100
        BatchItem item = createBatchItem(100L, 500, 200);
        endpoint.commitBatch(1L, 100, List.of(item));
        assertEquals(1, trackedEntryCount());

        // Engine reports success for batchId=1 but with requestId=999 (foreign)
        Map<String, TaskInfo> finished = new HashMap<>();
        TaskInfo foreignTask = new TaskInfo();
        foreignTask.setBatchId(1L);
        foreignTask.setRequestId(999L);
        foreignTask.setErrorCode(0);
        finished.put("999", foreignTask);

        calibrate(finished, new HashMap<>());
        // Batch should NOT be removed — requestId doesn't match
        assertEquals(1, trackedEntryCount());
    }

    @Test
    void calibrateRemovesBatchWithMatchingRequestId() {
        BatchItem item = createBatchItem(100L, 500, 200);
        endpoint.commitBatch(1L, 100, List.of(item));

        Map<String, TaskInfo> finished = new HashMap<>();
        TaskInfo task = new TaskInfo();
        task.setBatchId(1L);
        task.setRequestId(100L);
        task.setErrorCode(0);
        finished.put("100", task);

        calibrate(finished, new HashMap<>());
        assertEquals(0, trackedEntryCount());
    }

    // ---- estimated waiting time ----

    @Test
    void estimatedWaitTimeZeroWhenNoInflight() {
        assertEquals(0, endpoint.prefillEstimatedWaitTimeMs());
    }

    @Test
    void estimatedWaitTimePositiveWithInflight() {
        BatchItem item = createBatchItem(1L, 500, 200);
        endpoint.commitBatch(1L, 5000, List.of(item)); // 5s prediction

        long waitMs = endpoint.prefillEstimatedWaitTimeMs();
        assertTrue(waitMs > 0, "Should have non-zero wait time with inflight batch");
        assertTrue(waitMs <= 5000, "Wait time should not exceed prediction");
    }

    // ---- eviction ----

    @Test
    void evictExpiredBatchesCleansUpStaleEntries() throws InterruptedException {
        BatchItem item = createBatchItem(1L, 500, 200);
        endpoint.commitBatch(1L, 100, List.of(item));

        assertEquals(1, trackedEntryCount());

        // Wait a bit so the batch ages
        Thread.sleep(10);

        int evicted = endpoint.evictExpiredBatches(1); // 1ms TTL — should evict
        assertEquals(1, evicted);
        assertEquals(0, trackedEntryCount());
    }

    @Test
    void evictExpiredBatchesFreshEntriesSurvive() {
        BatchItem item = createBatchItem(1L, 500, 200);
        endpoint.commitBatch(1L, 100, List.of(item));

        int evicted = endpoint.evictExpiredBatches(60_000); // 60s TTL — fresh entry survives
        assertEquals(0, evicted);
        assertEquals(1, trackedEntryCount());
    }

    // ---- prefillPendingRequestCount ----

    @Test
    void pendingRequestCountIncludesBatcherQueue() throws InterruptedException {
        // Initially, batcher queue is empty
        assertEquals(0, endpoint.prefillPendingRequestCount());

        BatchItem item = createBatchItem(1L, 500, 200);
        endpoint.getBatcher().offer(item);

        long deadlineMs = System.currentTimeMillis() + 100;
        while (endpoint.prefillPendingRequestCount() == 0 && System.currentTimeMillis() < deadlineMs) {
            Thread.sleep(1);
        }
        assertTrue(endpoint.prefillPendingRequestCount() > 0, "Pending count should include batcher queue");
    }

    // ---- WorkerEndpoint inherited behavior ----

    @Test
    void onWorkerStatusUpdateUpdatesAliveStatus() {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        WorkerStatus status = new WorkerStatus();
        status.setIp("127.0.0.1");
        status.setPort(8080);
        status.setAlive(true);

        endpoint.onWorkerStatusUpdate(status, response);

        assertTrue(endpoint.getStatus().isAlive());
    }

    // ---- close ----

    @Test
    void closeShutsDownBatcher() {
        assertNotNull(endpoint.getBatcher());
        endpoint.close();
        // After close, offering should fail (batcher is stopped)
        BatchItem item = createBatchItem(1L, 500, 200);
        endpoint.getBatcher().offer(item);
        // Should not throw — batcher handles stopped state
    }

    // ---- two-layer inflight (layer 1: inflightEntries, layer 2: engineTasks) ----

    @Test
    void commitRequestCountsSingleRequest() {
        // Non-batch path tracks the request as a typed single-request entry.
        endpoint.commitRequest(42L, 100);

        assertEquals(1, trackedEntryCount());
        assertEquals(1, endpoint.prefillPendingRequestCount());

        endpoint.releaseBatch(42L);
        assertEquals(0, trackedEntryCount());
        assertEquals(0, endpoint.prefillPendingRequestCount());
    }

    @Test
    void runningReportMigratesToEngineTasksWithoutDoubleCount() {
        BatchItem item = createBatchItem(1L, 500, 200);
        endpoint.commitBatch(1L, 100, List.of(item));

        Map<String, TaskInfo> running = new HashMap<>();
        TaskInfo task = new TaskInfo();
        task.setRequestId(1L);
        task.setBatchId(1L);
        task.setPhase(TaskPhase.PENDING);
        running.put("1", task);
        calibrate(Map.of(), running);

        // Migrated layer 1 -> layer 2: still exactly one tracked entry/request
        assertEquals(1, trackedEntryCount());
        assertEquals(1, endpoint.prefillPendingRequestCount());

        // Finished in a later round removes it from layer 2
        Map<String, TaskInfo> finished = new HashMap<>();
        TaskInfo done = new TaskInfo();
        done.setRequestId(1L);
        done.setBatchId(1L);
        done.setErrorCode(0);
        finished.put("1", done);
        calibrate(finished, Map.of());

        assertEquals(0, trackedEntryCount());
        assertEquals(0, endpoint.prefillPendingRequestCount());
    }

    @Test
    void runningReportWithForeignRequestIdDoesNotMigrate() {
        BatchItem item = createBatchItem(100L, 500, 200);
        endpoint.commitBatch(1L, 100, List.of(item));

        Map<String, TaskInfo> running = new HashMap<>();
        TaskInfo foreign = new TaskInfo();
        foreign.setRequestId(999L); // not a member of local batch 1
        foreign.setBatchId(1L);
        foreign.setPhase(TaskPhase.RUNNING);
        running.put("999", foreign);
        calibrate(Map.of(), running);

        // Stale/foreign report must not migrate or drop the local batch
        assertEquals(1, trackedEntryCount());
        assertEquals(1, endpoint.prefillPendingRequestCount());
    }

    @Test
    void nonBatchRequestMigratesAndFinishesByRequestIdKey() {
        endpoint.commitRequest(7L, 100);

        // Engine reports non-batch tasks with batch_id=-1, keyed by requestId
        Map<String, TaskInfo> running = new HashMap<>();
        TaskInfo task = new TaskInfo();
        task.setRequestId(7L);
        task.setBatchId(-1);
        task.setPhase(TaskPhase.RUNNING);
        running.put("7", task);
        calibrate(Map.of(), running);
        assertEquals(1, trackedEntryCount());
        assertEquals(1, endpoint.prefillPendingRequestCount());

        Map<String, TaskInfo> finished = new HashMap<>();
        TaskInfo done = new TaskInfo();
        done.setRequestId(7L);
        done.setBatchId(-1);
        done.setErrorCode(0);
        finished.put("7", done);
        calibrate(finished, Map.of());
        assertEquals(0, trackedEntryCount());
        assertEquals(0, endpoint.prefillPendingRequestCount());
    }

    @Test
    void nonBatchFinishedBeforeAcceptanceFastPathRemoves() {
        // Cross-round fast path: finished shows up while still in layer 1
        endpoint.commitRequest(7L, 100);

        Map<String, TaskInfo> finished = new HashMap<>();
        TaskInfo done = new TaskInfo();
        done.setRequestId(7L);
        done.setBatchId(-1);
        done.setErrorCode(0);
        finished.put("7", done);
        calibrate(finished, Map.of());

        assertEquals(0, trackedEntryCount());
        assertEquals(0, endpoint.prefillPendingRequestCount());
    }

    @Test
    void staleEngineTaskEvictedAfterMissingRounds() {
        BatchItem item = createBatchItem(1L, 500, 200);
        endpoint.commitBatch(1L, 100, List.of(item));

        Map<String, TaskInfo> running = new HashMap<>();
        TaskInfo task = new TaskInfo();
        task.setRequestId(1L);
        task.setBatchId(1L);
        task.setPhase(TaskPhase.RUNNING);
        running.put("1", task);
        calibrate(Map.of(), running); // round 1: accepted into layer 2

        // Absent from the next STALE_EVICT_ROUNDS (3) reports -> evicted
        calibrate(Map.of(), Map.of()); // round 2
        calibrate(Map.of(), Map.of()); // round 3
        assertEquals(1, trackedEntryCount());
        calibrate(Map.of(), Map.of()); // round 4: 4 - 1 >= 3
        assertEquals(0, trackedEntryCount());
        assertEquals(0, endpoint.prefillPendingRequestCount());
    }

    @Test
    void evictExpiredBatchesCoversEngineTaskLayer() throws InterruptedException {
        BatchItem item = createBatchItem(1L, 500, 200);
        endpoint.commitBatch(1L, 100, List.of(item));

        Map<String, TaskInfo> running = new HashMap<>();
        TaskInfo task = new TaskInfo();
        task.setRequestId(1L);
        task.setBatchId(1L);
        task.setPhase(TaskPhase.RUNNING);
        running.put("1", task);
        calibrate(Map.of(), running); // migrate to layer 2

        Thread.sleep(10);

        // Wall-clock TTL is the backstop when the worker stops reporting
        int evicted = endpoint.evictExpiredBatches(1);
        assertEquals(1, evicted);
        assertEquals(0, trackedEntryCount());
        assertEquals(0, endpoint.prefillPendingRequestCount());
    }

    @Test
    void partialFinishOnEngineTaskLayerShrinksBatch() {
        BatchItem item1 = createBatchItem(1L, 500, 200);
        BatchItem item2 = createBatchItem(2L, 300, 100);
        endpoint.commitBatch(1L, 100, List.of(item1, item2));

        Map<String, TaskInfo> running = new HashMap<>();
        TaskInfo r1 = new TaskInfo();
        r1.setRequestId(1L);
        r1.setBatchId(1L);
        r1.setPhase(TaskPhase.RUNNING);
        running.put("1", r1);
        calibrate(Map.of(), running); // migrate whole batch (any member seen)
        assertEquals(2, endpoint.prefillPendingRequestCount());

        Map<String, TaskInfo> finished = new HashMap<>();
        TaskInfo done = new TaskInfo();
        done.setRequestId(1L);
        done.setBatchId(1L);
        done.setErrorCode(0);
        finished.put("1", done);
        calibrate(finished, Map.of());

        // Only member 1 finished — survivor 2 keeps the entry tracked
        assertEquals(1, trackedEntryCount());
        assertEquals(1, endpoint.prefillPendingRequestCount());
    }

    // ---- new explicit views ----

    @Test
    void newViewsTrackLayerCountsAndPhases() {
        BatchItem item1 = createBatchItem(1L, 500, 200);
        BatchItem item2 = createBatchItem(2L, 300, 100);
        endpoint.commitBatch(1L, 100, List.of(item1, item2));
        endpoint.commitRequest(7L, 50);

        assertEquals(2, endpoint.prefillInflightCount());
        assertEquals(0, endpoint.prefillEngineTaskCount());
        assertEquals(0, endpoint.prefillEngineWaitingCount());
        assertEquals(0, endpoint.prefillEngineRunningCount());

        // batch accepted as WAITING (PENDING maps to WAITING for prefill)
        Map<String, TaskInfo> running = new HashMap<>();
        TaskInfo batchTask = new TaskInfo();
        batchTask.setRequestId(1L);
        batchTask.setBatchId(1L);
        batchTask.setPhase(TaskPhase.PENDING);
        running.put("1", batchTask);
        calibrate(Map.of(), running);

        assertEquals(1, endpoint.prefillInflightCount());
        assertEquals(1, endpoint.prefillEngineTaskCount());
        assertEquals(1, endpoint.prefillEngineWaitingCount());
        assertEquals(0, endpoint.prefillEngineRunningCount());

        // single request accepted as RUNNING
        Map<String, TaskInfo> running2 = new HashMap<>();
        running2.put("1", batchTask);
        TaskInfo reqTask = new TaskInfo();
        reqTask.setRequestId(7L);
        reqTask.setBatchId(-1);
        reqTask.setPhase(TaskPhase.RUNNING);
        running2.put("7", reqTask);
        calibrate(Map.of(), running2);

        assertEquals(0, endpoint.prefillInflightCount());
        assertEquals(2, endpoint.prefillEngineTaskCount());
        assertEquals(1, endpoint.prefillEngineWaitingCount());
        assertEquals(1, endpoint.prefillEngineRunningCount());
    }

    @Test
    void layerMigrationPreservesTrackedEntryCount() {
        BatchItem item = createBatchItem(1L, 500, 200);
        endpoint.commitBatch(1L, 100, List.of(item));
        endpoint.commitRequest(7L, 50);
        assertEquals(2, trackedEntryCount());

        Map<String, TaskInfo> running = new HashMap<>();
        TaskInfo task = new TaskInfo();
        task.setRequestId(1L);
        task.setBatchId(1L);
        task.setPhase(TaskPhase.RUNNING);
        running.put("1", task);
        calibrate(Map.of(), running);

        // migration moves entries between layers without changing the sum
        assertEquals(2, trackedEntryCount());
        assertEquals(1, endpoint.prefillInflightCount());
        assertEquals(1, endpoint.prefillEngineTaskCount());
    }

    @Test
    void estimatedWaitTimeSumsLayerOneAtFullValue() {
        endpoint.commitRequest(7L, 3000);
        endpoint.commitRequest(8L, 2000);

        // not yet accepted — no elapsed-time discount
        assertEquals(5000, endpoint.prefillEstimatedWaitTimeMs());
    }

    @Test
    void estimatedWaitTimeKeepsWaitingTasksAtFullValue() throws InterruptedException {
        BatchItem item = createBatchItem(1L, 500, 200);
        endpoint.commitBatch(1L, 5000, List.of(item));

        Map<String, TaskInfo> running = new HashMap<>();
        TaskInfo task = new TaskInfo();
        task.setRequestId(1L);
        task.setBatchId(1L);
        task.setPhase(TaskPhase.PENDING); // WAITING
        running.put("1", task);
        calibrate(Map.of(), running);

        Thread.sleep(30);
        // queued work spends no predicted time
        assertEquals(5000, endpoint.prefillEstimatedWaitTimeMs());
    }

    @Test
    void estimatedWaitTimeDiscountsRunningRemainder() throws InterruptedException {
        BatchItem item = createBatchItem(1L, 500, 200);
        endpoint.commitBatch(1L, 5000, List.of(item));

        Map<String, TaskInfo> running = new HashMap<>();
        TaskInfo task = new TaskInfo();
        task.setRequestId(1L);
        task.setBatchId(1L);
        task.setPhase(TaskPhase.RUNNING);
        running.put("1", task);
        calibrate(Map.of(), running);

        Thread.sleep(50);

        long estimated = endpoint.prefillEstimatedWaitTimeMs();
        assertTrue(estimated < 5000, "RUNNING task should be discounted by elapsed time, got " + estimated);
        assertTrue(estimated >= 4000, "Discount should only cover elapsed time, got " + estimated);
    }

    @Test
    void batcherQueueSizeViewMatchesBatcher() {
        assertEquals(endpoint.getBatcher().queueSize(), endpoint.prefillBatcherQueueSize());
        assertEquals(0, endpoint.prefillBatcherQueueSize());
    }

    // ---- helpers ----

    private int trackedEntryCount() {
        return endpoint.prefillInflightCount() + endpoint.prefillEngineTaskCount();
    }

    private void calibrate(Map<String, TaskInfo> finished, Map<String, TaskInfo> running) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setFinishedTaskInfo(finished);
        response.setRunningTaskInfo(running);
        endpoint.onWorkerStatusUpdate(endpoint.getStatus(), response);
    }

    private BatchItem createBatchItem(long requestId, long seqLen, long hitCacheLen) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);

        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);

        ServerStatus prefill = new ServerStatus();
        prefill.setRole(RoleType.PREFILL);
        prefill.setServerIp("127.0.0.1");
        prefill.setHttpPort(8080);
        prefill.setGrpcPort(8090);
        DebugInfo debugInfo = new DebugInfo();
        debugInfo.setHitCacheLen(hitCacheLen);
        prefill.setDebugInfo(debugInfo);

        return new BatchItem(ctx, new java.util.concurrent.CompletableFuture<>(), null,
                prefill, null, endpoint, null, System.currentTimeMillis());
    }
}
