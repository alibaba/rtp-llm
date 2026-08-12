package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.BatchDecisionHandler;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.DispatchMeta;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.ScheduleBudget;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
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
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

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

        endpoint = new PrefillEndpoint(status, config, noopHandler(), mock(BatchSchedulerReporter.class));
    }

    @AfterEach
    void tearDown() {
        endpoint.close();
    }

    // ---- batch commit / release ----

    @Test
    void commitBatchIncreasesInflightCount() {
        assertEquals(0, endpoint.getInflightBatchCount());

        BatchItem item = createBatchItem(1L, 500, 200);
        endpoint.commitBatch(1L, 100, List.of(item));

        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.realPendingCount());
    }

    @Test
    void releaseBatchDecreasesInflightCount() {
        BatchItem item = createBatchItem(1L, 500, 200);
        endpoint.commitBatch(1L, 100, List.of(item));
        endpoint.releaseBatch(1L);

        assertEquals(0, endpoint.getInflightBatchCount());
    }

    @Test
    void releaseBatchNonExistentDoesNotThrow() {
        endpoint.releaseBatch(999L); // should not throw
        assertEquals(0, endpoint.getInflightBatchCount());
    }

    @Test
    void commitMultipleBatches() {
        BatchItem item1 = createBatchItem(1L, 500, 200);
        BatchItem item2 = createBatchItem(2L, 300, 100);
        BatchItem item3 = createBatchItem(3L, 400, 0);

        endpoint.commitBatch(1L, 100, List.of(item1, item2));
        endpoint.commitBatch(2L, 50, List.of(item3));

        assertEquals(2, endpoint.getInflightBatchCount());
        assertEquals(3, endpoint.realPendingCount());
    }

    // ---- repack batch ----

    @Test
    void repackBatchRemovesFailedRequests() {
        BatchItem item1 = createBatchItem(1L, 500, 200);
        BatchItem item2 = createBatchItem(2L, 300, 100);
        endpoint.commitBatch(1L, 100, List.of(item1, item2));

        endpoint.repackBatch(1L, Set.of(2L));
        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.realPendingCount());
    }

    @Test
    void repackBatchAllFailedReturnsNull() {
        BatchItem item1 = createBatchItem(1L, 500, 200);
        endpoint.commitBatch(1L, 100, List.of(item1));

        endpoint.repackBatch(1L, Set.of(1L));
        assertEquals(0, endpoint.getInflightBatchCount());
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

        assertEquals(0, endpoint.getInflightBatchCount());
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

        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.realPendingCount());
    }

    @Test
    void calibrateKeepsBatchInflightUntilEveryMemberFinishes() {
        BatchItem item1 = createBatchItem(1L, 500, 200);
        BatchItem item2 = createBatchItem(2L, 300, 100);
        endpoint.commitBatch(1L, 100, List.of(item1, item2));

        TaskInfo finishedFirst = taskInfo(1L, 1L, null, 0, 40);
        TaskInfo runningSecond = taskInfo(2L, 1L, TaskPhase.RUNNING, 0, 0);
        calibrate(Map.of("1", finishedFirst), Map.of("2", runningSecond));

        assertEquals(1, endpoint.getInflightBatchCount(),
                "one finished member must not release the whole batch");
        assertEquals(1, endpoint.realPendingCount());

        TaskInfo finishedSecond = taskInfo(2L, 1L, null, 0, 75);
        calibrate(Map.of("2", finishedSecond), Map.of());

        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.realPendingCount());
    }

    @Test
    void calibrateMixedTerminalMembersKeepsOnlyRunningSurvivor() {
        BatchItem item1 = createBatchItem(1L, 500, 200);
        BatchItem item2 = createBatchItem(2L, 300, 100);
        BatchItem item3 = createBatchItem(3L, 400, 0);
        endpoint.commitBatch(1L, 100, List.of(item1, item2, item3));

        TaskInfo success = taskInfo(1L, 1L, null, 0, 40);
        TaskInfo failure = taskInfo(2L, 1L, null, 500, 50);
        TaskInfo running = taskInfo(3L, 1L, TaskPhase.RUNNING, 0, 0);
        calibrate(Map.of("1", success, "2", failure), Map.of("3", running));

        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.realPendingCount());

        // WorkerStatus may repeat an already observed terminal task.  It must
        // not decrement the survivor count a second time.
        calibrate(Map.of("1", success), Map.of("3", running));
        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.realPendingCount());
    }

    @Test
    void runningObservationRefreshesBatchInactivityTtl() throws InterruptedException {
        BatchItem item = createBatchItem(1L, 500, 200);
        endpoint.commitBatch(1L, 100, List.of(item));

        Thread.sleep(150);
        TaskInfo running = taskInfo(1L, 1L, TaskPhase.RUNNING, 0, 0);
        calibrate(Map.of(), Map.of("1", running));

        assertEquals(0, endpoint.evictExpiredBatches(100),
                "an actively observed long-running batch must not be evicted by creation age");
        assertEquals(1, endpoint.getInflightBatchCount());
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
        assertEquals(1, endpoint.getInflightBatchCount());
    }

    @Test
    void calibrateDoesNotRemoveBatchWithForeignRequestId() {
        // Commit batch with requestId=100
        BatchItem item = createBatchItem(100L, 500, 200);
        endpoint.commitBatch(1L, 100, List.of(item));
        assertEquals(1, endpoint.getInflightBatchCount());

        // Engine reports success for batchId=1 but with requestId=999 (foreign)
        Map<String, TaskInfo> finished = new HashMap<>();
        TaskInfo foreignTask = new TaskInfo();
        foreignTask.setBatchId(1L);
        foreignTask.setRequestId(999L);
        foreignTask.setErrorCode(0);
        finished.put("999", foreignTask);

        calibrate(finished, new HashMap<>());
        // Batch should NOT be removed — requestId doesn't match
        assertEquals(1, endpoint.getInflightBatchCount());
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
        assertEquals(0, endpoint.getInflightBatchCount());
    }

    // ---- estimated waiting time ----

    @Test
    void realWaitTimeMsZeroWhenNoInflight() {
        assertEquals(0, endpoint.realWaitTimeMs());
    }

    @Test
    void realWaitTimeMsPositiveWithInflight() {
        BatchItem item = createBatchItem(1L, 500, 200);
        endpoint.commitBatch(1L, 5000, List.of(item)); // 5s prediction

        long waitMs = endpoint.realWaitTimeMs();
        assertTrue(waitMs > 0, "Should have non-zero wait time with inflight batch");
        assertTrue(waitMs <= 5000, "Wait time should not exceed prediction");
    }

    @Test
    void realWaitTimeMsDecreasesOverTime() throws InterruptedException {
        BatchItem item = createBatchItem(1L, 500, 200);
        endpoint.commitBatch(1L, 5000, List.of(item));

        long waitBefore = endpoint.realWaitTimeMs();

        // Mark the batch as running so elapsed time counts
        Map<String, TaskInfo> running = new HashMap<>();
        TaskInfo runningTask = new TaskInfo();
        runningTask.setRequestId(1L);
        runningTask.setBatchId(1L);
        runningTask.setPhase(TaskPhase.RUNNING);
        running.put("1", runningTask);
        calibrate(Map.of(), running);

        Thread.sleep(50);

        long waitAfter = endpoint.realWaitTimeMs();
        assertTrue(waitAfter <= waitBefore, "Wait time should decrease after progress");
    }

    // ---- eviction ----

    @Test
    void evictExpiredBatchesCleansUpStaleEntries() throws InterruptedException {
        BatchItem item = createBatchItem(1L, 500, 200);
        endpoint.commitBatch(1L, 100, List.of(item));

        assertEquals(1, endpoint.getInflightBatchCount());

        // Wait a bit so the batch ages
        Thread.sleep(10);

        int evicted = endpoint.evictExpiredBatches(1); // 1ms TTL — should evict
        assertEquals(1, evicted);
        assertEquals(0, endpoint.getInflightBatchCount());
    }

    @Test
    void evictExpiredBatchesFreshEntriesSurvive() {
        BatchItem item = createBatchItem(1L, 500, 200);
        endpoint.commitBatch(1L, 100, List.of(item));

        int evicted = endpoint.evictExpiredBatches(60_000); // 60s TTL — fresh entry survives
        assertEquals(0, evicted);
        assertEquals(1, endpoint.getInflightBatchCount());
    }

    // ---- realPendingCount ----

    @Test
    void realPendingCountIncludesBatcherQueue() throws InterruptedException {
        // Initially, batcher queue is empty
        assertEquals(0, endpoint.realPendingCount());

        BatchItem item = createBatchItem(1L, 500, 200);
        endpoint.getBatcher().offer(item);

        long deadlineMs = System.currentTimeMillis() + 100;
        while (endpoint.realPendingCount() == 0 && System.currentTimeMillis() < deadlineMs) {
            Thread.sleep(1);
        }
        assertTrue(endpoint.realPendingCount() > 0, "Pending count should include batcher queue");
    }

    // ---- batch metrics reporting ----

    @Test
    void reportBatchMetricsBucketsQueueLengthByPriority() {
        // Long fixed window so offered items stay queued during the assertions
        PrefillEndpoint slowEndpoint = newFixedWindowEndpoint(60_000);
        try {
            slowEndpoint.getBatcher().offer(createPriorityBatchItem(1L, 70));
            slowEndpoint.getBatcher().offer(createBatchItem(2L, 300, 0)); // legacy: budget=null -> priority 0

            BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);
            slowEndpoint.reportBatchMetrics(reporter);

            // Single-report with priority tag (no global untagged series)
            verify(reporter, never()).reportBatcherQueueDepth(anyString(), anyString(), anyInt());
            verify(reporter).reportBatcherQueueSize("PREFILL", "127.0.0.1", 2);
            // Priority buckets on the same routing.queue.length metric
            verify(reporter).reportBatcherQueueDepthByPriority("PREFILL", "127.0.0.1", 70, 1);
            verify(reporter).reportBatcherQueueDepthByPriority("PREFILL", "127.0.0.1", 0, 1);
        } finally {
            slowEndpoint.close();
        }
    }

    @Test
    void reportBatchMetricsEmitsPriorityZeroFallbackForEmptyQueue() {
        BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);
        endpoint.reportBatchMetrics(reporter);

        verify(reporter, never()).reportBatcherQueueDepth(anyString(), anyString(), anyInt());
        verify(reporter).reportBatcherQueueSize("PREFILL", "127.0.0.1", 0);
        // Empty queue fallback: single priority=0 depth=0 report so tagged panels don't gap
        verify(reporter).reportBatcherQueueDepthByPriority("PREFILL", "127.0.0.1", 0, 0);
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

    // ---- helpers ----

    private PrefillEndpoint newFixedWindowEndpoint(long fixedWaitMs) {
        WorkerStatus status = new WorkerStatus();
        status.setIp("127.0.0.1");
        status.setPort(8080);
        status.setGrpcPort(8090);
        status.setRole(RoleType.PREFILL);

        FlexlbConfig slowConfig = new FlexlbConfig();
        slowConfig.setFlexlbBatchQueueMaxSize(100);
        slowConfig.setFlexlbBatchSizeMax(100);
        slowConfig.setFlexlbBatchAlgorithm("fixed_window");
        slowConfig.setFlexlbBatchFixedWaitMs(fixedWaitMs);
        return new PrefillEndpoint(status, slowConfig, noopHandler(), mock(BatchSchedulerReporter.class));
    }

    private BatchItem createPriorityBatchItem(long requestId, int priority) {
        long now = System.currentTimeMillis();
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(500);
        request.setPriority(priority);

        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setBudget(ScheduleBudget.forDeadline(priority, now, now + 60_000));

        return new BatchItem(ctx, null, null, null, null, null, null, now);
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

        return new BatchItem(ctx, null, null, prefill, null, endpoint, null, System.currentTimeMillis());
    }

    private static TaskInfo taskInfo(long requestId,
                                     long batchId,
                                     TaskPhase phase,
                                     int errorCode,
                                     long executionTimeMs) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setBatchId(batchId);
        task.setPhase(phase);
        task.setErrorCode(errorCode);
        task.setExecutionTimeMs(executionTimeMs);
        return task;
    }

    private static BatchDecisionHandler noopHandler() {
        return new BatchDecisionHandler() {
            @Override public void onExpired(BatchItem head) {}
            @Override public void onBatchReady(List<BatchItem> items, DispatchMeta meta) {}
            @Override public void onOfferFailure(BatchItem item, Throwable error) {}
        };
    }
}
