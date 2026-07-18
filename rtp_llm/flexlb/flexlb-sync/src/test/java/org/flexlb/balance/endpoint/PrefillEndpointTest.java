package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.BatchDecisionHandler;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.DispatchMeta;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
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
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
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
        assertEquals(1, endpoint.getInflightRequestCount());
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
        assertEquals(3, endpoint.getInflightRequestCount());
    }

    // ---- repack batch ----

    @Test
    void repackBatchRemovesFailedRequests() {
        BatchItem item1 = createBatchItem(1L, 500, 200);
        BatchItem item2 = createBatchItem(2L, 300, 100);
        endpoint.commitBatch(1L, 100, List.of(item1, item2));

        BatchInflight result = endpoint.repackBatch(1L, Set.of(2L));
        assertNotNull(result);
        assertEquals(1, result.requests().size());
        assertEquals(1L, result.requests().get(0).requestId());
    }

    @Test
    void repackBatchAllFailedReturnsNull() {
        BatchItem item1 = createBatchItem(1L, 500, 200);
        endpoint.commitBatch(1L, 100, List.of(item1));

        BatchInflight result = endpoint.repackBatch(1L, Set.of(1L));
        assertNull(result);
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

        endpoint.calibrate(finished, Map.of());

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

        endpoint.calibrate(finished, Map.of());

        assertEquals(1, endpoint.getInflightBatchCount());
        BatchInflight remaining = endpoint.getInflightBatches().get(1L);
        assertNotNull(remaining);
        assertEquals(1, remaining.requests().size());
        assertEquals(1L, remaining.requests().get(0).requestId());
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
        endpoint.calibrate(finished, Map.of());
        assertEquals(1, endpoint.getInflightBatchCount());
    }

    @Test
    void calibrateUpdatesProgressAnchorsForRunningBatches() {
        BatchItem item = createBatchItem(1L, 500, 200);
        endpoint.commitBatch(1L, 100, List.of(item));

        Map<String, TaskInfo> running = new HashMap<>();
        TaskInfo runningTask = new TaskInfo();
        runningTask.setRequestId(1L);
        runningTask.setBatchId(1L);
        runningTask.setPhase(TaskPhase.RUNNING);
        running.put("1", runningTask);

        endpoint.calibrate(Map.of(), running);

        BatchInflight batch = endpoint.getInflightBatches().get(1L);
        assertNotNull(batch);
        assertTrue(batch.progressBaseMs() > 0, "Running batch should have its progress anchor updated");
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

        endpoint.calibrate(finished, new HashMap<>());
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

        endpoint.calibrate(finished, new HashMap<>());
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
        endpoint.calibrate(Map.of(), running);

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

    // ==================== inflightRequestCount Counter Tests ====================

    @Test
    void commitBatchIncrementsRequestCount() {
        BatchItem item1 = createBatchItem(1001L, 500, 200);
        BatchItem item2 = createBatchItem(1002L, 300, 100);
        endpoint.commitBatch(1L, 100, List.of(item1, item2));
        assertEquals(2, endpoint.getInflightRequestCount());
    }

    @Test
    void commitBatchDuplicateKeySwapsRequestCount() {
        BatchItem item1 = createBatchItem(1001L, 500, 200);
        BatchItem item2 = createBatchItem(1002L, 300, 100);
        endpoint.commitBatch(1L, 100, List.of(item1, item2));
        assertEquals(2, endpoint.getInflightRequestCount());

        // Commit same batchId with different requests
        BatchItem item3 = createBatchItem(1003L, 400, 0);
        endpoint.commitBatch(1L, 50, List.of(item3));
        // Old (2) subtracted, new (1) added → 1
        assertEquals(1, endpoint.getInflightRequestCount());
    }

    @Test
    void releaseBatchDecrementsRequestCount() {
        BatchItem item1 = createBatchItem(1001L, 500, 200);
        BatchItem item2 = createBatchItem(1002L, 300, 100);
        endpoint.commitBatch(1L, 100, List.of(item1, item2));
        endpoint.releaseBatch(1L);
        assertEquals(0, endpoint.getInflightRequestCount());
    }

    @Test
    void repackBatchAllFailedDecrementsRequestCount() {
        BatchItem item1 = createBatchItem(1001L, 500, 200);
        BatchItem item2 = createBatchItem(1002L, 300, 100);
        endpoint.commitBatch(1L, 100, List.of(item1, item2));
        assertEquals(2, endpoint.getInflightRequestCount());

        endpoint.repackBatch(1L, Set.of(1001L, 1002L));
        assertEquals(0, endpoint.getInflightRequestCount());
    }

    @Test
    void repackBatchPartialFailureDecrementsRequestCount() {
        BatchItem item1 = createBatchItem(1001L, 500, 200);
        BatchItem item2 = createBatchItem(1002L, 300, 100);
        BatchItem item3 = createBatchItem(1003L, 400, 0);
        endpoint.commitBatch(1L, 100, List.of(item1, item2, item3));
        assertEquals(3, endpoint.getInflightRequestCount());

        // Fail item2 → 3 - 1 = 2
        endpoint.repackBatch(1L, Set.of(1002L));
        assertEquals(2, endpoint.getInflightRequestCount());
    }

    @Test
    void calibratePhase1NonBatchDecrementsRequestCount() {
        // Non-batch request: committed with batchId = requestId
        BatchItem item = createBatchItem(1001L, 500, 200);
        endpoint.commitBatch(1001L, 100, List.of(item));
        assertEquals(1, endpoint.getInflightRequestCount());

        // Calibrate with finished non-batch task (batchId = -1)
        Map<String, TaskInfo> finished = new HashMap<>();
        TaskInfo task = new TaskInfo();
        task.setRequestId(1001L);
        task.setBatchId(-1);
        task.setErrorCode(0);
        finished.put("1001", task);
        endpoint.calibrate(finished, Map.of());
        assertEquals(0, endpoint.getInflightRequestCount());
    }

    @Test
    void calibratePhase2BatchSuccessDecrementsRequestCount() {
        BatchItem item1 = createBatchItem(1001L, 500, 200);
        BatchItem item2 = createBatchItem(1002L, 300, 100);
        endpoint.commitBatch(1L, 100, List.of(item1, item2));
        assertEquals(2, endpoint.getInflightRequestCount());

        Map<String, TaskInfo> finished = new HashMap<>();
        TaskInfo successTask = new TaskInfo();
        successTask.setRequestId(1001L);
        successTask.setBatchId(1L);
        successTask.setErrorCode(0);
        finished.put("1001", successTask);
        endpoint.calibrate(finished, Map.of());
        assertEquals(0, endpoint.getInflightRequestCount());
    }

    @Test
    void evictExpiredBatchesDecrementsRequestCount() throws InterruptedException {
        BatchItem item1 = createBatchItem(1001L, 500, 200);
        BatchItem item2 = createBatchItem(1002L, 300, 100);
        endpoint.commitBatch(1L, 100, List.of(item1, item2));
        assertEquals(2, endpoint.getInflightRequestCount());

        Thread.sleep(10);
        endpoint.evictExpiredBatches(1);
        assertEquals(0, endpoint.getInflightRequestCount());
    }

    @Test
    void calibratePhase2ThenPhase3NoDoubleDeduction() {
        BatchItem item1 = createBatchItem(1001L, 500, 200);
        BatchItem item2 = createBatchItem(1002L, 300, 100);
        endpoint.commitBatch(1L, 100, List.of(item1, item2));
        assertEquals(2, endpoint.getInflightRequestCount());

        // One success, one failure for same batchId
        Map<String, TaskInfo> finished = new HashMap<>();
        TaskInfo successTask = new TaskInfo();
        successTask.setRequestId(1001L);
        successTask.setBatchId(1L);
        successTask.setErrorCode(0);
        finished.put("1001", successTask);

        TaskInfo failedTask = new TaskInfo();
        failedTask.setRequestId(1002L);
        failedTask.setBatchId(1L);
        failedTask.setErrorCode(500);
        failedTask.setErrorMessage("engine error");
        finished.put("1002", failedTask);

        endpoint.calibrate(finished, Map.of());
        // Phase 2 removes entire batch (counter -= 2)
        // Phase 3 skips because batchId is in batchesWithSuccess
        assertEquals(0, endpoint.getInflightRequestCount());
    }

    @Test
    void multipleBatchesProgressiveDecrease() {
        BatchItem item1 = createBatchItem(1001L, 500, 200);
        BatchItem item2 = createBatchItem(1002L, 300, 100);
        BatchItem item3 = createBatchItem(1003L, 400, 0);
        BatchItem item4 = createBatchItem(1004L, 600, 50);

        endpoint.commitBatch(1L, 100, List.of(item1, item2));
        endpoint.commitBatch(2L, 50, List.of(item3));
        endpoint.commitBatch(3L, 200, List.of(item4));
        assertEquals(4, endpoint.getInflightRequestCount());

        // Release batch 1
        endpoint.releaseBatch(1L);
        assertEquals(2, endpoint.getInflightRequestCount());

        // Calibrate batch 2 (success)
        Map<String, TaskInfo> finished = new HashMap<>();
        TaskInfo successTask = new TaskInfo();
        successTask.setRequestId(1003L);
        successTask.setBatchId(2L);
        successTask.setErrorCode(0);
        finished.put("1003", successTask);
        endpoint.calibrate(finished, Map.of());
        assertEquals(1, endpoint.getInflightRequestCount());

        // Release batch 3
        endpoint.releaseBatch(3L);
        assertEquals(0, endpoint.getInflightRequestCount());
    }

    @Test
    void emptyMapRequestCountIsZero() {
        assertEquals(0, endpoint.getInflightRequestCount());
    }

    @Test
    void requestCountEquivalentToTraversal() {
        BatchItem item1 = createBatchItem(1001L, 500, 200);
        BatchItem item2 = createBatchItem(1002L, 300, 100);
        BatchItem item3 = createBatchItem(1003L, 400, 0);

        endpoint.commitBatch(1L, 100, List.of(item1, item2));
        endpoint.commitBatch(2L, 50, List.of(item3));

        // Remove batch 1 via release
        endpoint.releaseBatch(1L);

        // Only batch 2 with 1 request remains
        assertEquals(manualRequestCountSum(), endpoint.getInflightRequestCount());
        assertEquals(1, endpoint.getInflightRequestCount());
    }

    // ---- counter test helper ----

    private int manualRequestCountSum() {
        int sum = 0;
        for (BatchInflight batch : endpoint.getInflightBatches().values()) {
            sum += batch.requests().size();
        }
        return sum;
    }

    // ---- helpers ----

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

        return new BatchItem(ctx, null, null, prefill, null, endpoint, null, 0, System.currentTimeMillis());
    }

    private static BatchDecisionHandler noopHandler() {
        return new BatchDecisionHandler() {
            @Override public void onExpired(BatchItem head) {}
            @Override public void onUrgent(BatchItem head, DispatchMeta meta) {}
            @Override public void onBatchReady(List<BatchItem> items, DispatchMeta meta) {}
            @Override public void onOfferFailure(BatchItem item, Throwable error) {}
        };
    }
}
