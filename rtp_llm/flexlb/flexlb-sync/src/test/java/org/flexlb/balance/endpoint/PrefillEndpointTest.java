package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.DecisionGroupHandler;
import org.flexlb.balance.scheduler.DecisionGroupMetadata;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentMatchers;

import java.util.AbstractList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

class PrefillEndpointTest {

    private PrefillEndpoint endpoint;
    private FlexlbConfig config;
    private BatchSchedulerReporter endpointReporter;

    @BeforeEach
    void setUp() {
        WorkerStatus status = new WorkerStatus();
        status.setIp("127.0.0.1");
        status.setPort(8080);
        status.setGrpcPort(8090);
        status.setRole(RoleType.PREFILL);

        config = new FlexlbConfig();
        configureBatch(config, 100, config.batchDispatcher().getMaxRequests(), 300, null);
        setFormula(config, "10 + 0.1*sum(computeTokens) + 5*batchSize");

        endpointReporter = mock(BatchSchedulerReporter.class);
        endpoint = new PrefillEndpoint(status, config, noopHandler(), endpointReporter);
    }

    @AfterEach
    void tearDown() {
        endpoint.close();
    }

    // ---- batch commit / release ----

    @Test
    void commitBatchIncreasesInflightCount() {
        assertEquals(0, endpoint.getInflightBatchCount());

        BatchItem item = createBatchItem("1", 500, 200);
        endpoint.commitBatch(1L, 100, List.of(item));

        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.realPendingCount());
    }

    @Test
    void releaseBatchDecreasesInflightCount() {
        BatchItem item = createBatchItem("1", 500, 200);
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
    void releaseBatchRetainsOnlyProtectedMembers() {
        endpoint.commitBatch(7L, 100, List.of(
                createBatchItem("101", 500, 200),
                createBatchItem("102", 300, 100)));
        assertTrue(endpoint.tryProtectBatchMember(7L, "101"));

        endpoint.releaseBatch(7L);

        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.realPendingCount(),
                "a delivery failure must not reopen capacity owned by an Engine fence");

        endpoint.releaseBatchMemberProtection(7L, "101");
        endpoint.releaseBatch(7L);
        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.realPendingCount());
    }

    @Test
    void commitMultipleBatches() {
        BatchItem item1 = createBatchItem("1", 500, 200);
        BatchItem item2 = createBatchItem("2", 300, 100);
        BatchItem item3 = createBatchItem("3", 400, 0);

        endpoint.commitBatch(1L, 100, List.of(item1, item2));
        endpoint.commitBatch(2L, 50, List.of(item3));

        assertEquals(2, endpoint.getInflightBatchCount());
        assertEquals(3, endpoint.realPendingCount());
    }

    // ---- repack batch ----

    @Test
    void repackBatchRemovesFailedRequests() {
        BatchItem item1 = createBatchItem("1", 500, 200);
        BatchItem item2 = createBatchItem("2", 300, 100);
        endpoint.commitBatch(1L, 100, List.of(item1, item2));

        endpoint.repackBatch(1L, Set.of("2"));
        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.realPendingCount());
    }

    @Test
    void repackBatchAllFailedReturnsNull() {
        BatchItem item1 = createBatchItem("1", 500, 200);
        endpoint.commitBatch(1L, 100, List.of(item1));

        endpoint.repackBatch(1L, Set.of("1"));
        assertEquals(0, endpoint.getInflightBatchCount());
    }

    // ---- calibrate ----

    @Test
    void calibrateRemovesBatchOnSuccess() {
        BatchItem item = createBatchItem("1", 500, 200);
        endpoint.commitBatch(1L, 100, List.of(item));

        Map<String, TaskInfo> finished = new HashMap<>();
        TaskInfo successTask = new TaskInfo();
        successTask.setRequestId("1");
        successTask.setBatchId(1L);
        successTask.setErrorCode(0);
        finished.put("1", successTask);

        calibrate(finished, Map.of());

        assertEquals(0, endpoint.getInflightBatchCount());
    }

    @Test
    void completion_observer_failure_does_not_escape_finished_settlement() {
        endpoint.commitBatch(9L, 100, List.of(createBatchItem("9", 500, 200)));
        doThrow(new IllegalStateException("metrics unavailable"))
                .when(endpointReporter)
                .reportBatchPredictedTimeMs("PREFILL", "127.0.0.1", 100);

        TaskInfo finished = taskInfo("9", 9L, null, 0, 125);
        assertDoesNotThrow(() -> calibrate(Map.of("9", finished), Map.of()));

        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.realPendingCount());
        verify(endpointReporter).reportBatchActualTimeMs("PREFILL", "127.0.0.1", 125);
        verify(endpointReporter).reportBatchPredictGapMs("PREFILL", "127.0.0.1", 25);
    }

    @Test
    void calibrateRepacksOnPartialFailure() {
        BatchItem item1 = createBatchItem("1", 500, 200);
        BatchItem item2 = createBatchItem("2", 300, 100);
        endpoint.commitBatch(1L, 100, List.of(item1, item2));

        Map<String, TaskInfo> finished = new HashMap<>();
        TaskInfo failedTask = new TaskInfo();
        failedTask.setRequestId("2");
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
        BatchItem shortItem = createBatchItem("1", 500, 200);
        BatchItem longItem = createBatchItem("2", 10000, 0);
        endpoint.commitBatch(1L, 2_000, List.of(shortItem, longItem));

        TaskInfo finishedShort = taskInfo("1", 1L, null, 0, 40);
        TaskInfo runningLong = taskInfo("2", 1L, TaskPhase.RUNNING, 0, 0);
        calibrate(Map.of("1", finishedShort), Map.of("2", runningLong));

        assertEquals(1, endpoint.getInflightBatchCount(),
                "one finished member must not release the whole batch");
        assertEquals(1, endpoint.realPendingCount(),
                "the still-running long member must remain in Master accounting");

        TaskInfo finishedLong = taskInfo("2", 1L, null, 0, 1_900);
        calibrate(Map.of("2", finishedLong), Map.of());

        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.realPendingCount());
    }

    @Test
    void calibrateMixedTerminalMembersKeepsOnlyRunningSurvivor() {
        BatchItem succeeded = createBatchItem("1", 500, 200);
        BatchItem failed = createBatchItem("2", 300, 100);
        BatchItem running = createBatchItem("3", 10000, 0);
        endpoint.commitBatch(1L, 2_000, List.of(succeeded, failed, running));

        TaskInfo success = taskInfo("1", 1L, null, 0, 40);
        TaskInfo failure = taskInfo("2", 1L, null, 500, 50);
        TaskInfo runningTask = taskInfo("3", 1L, TaskPhase.RUNNING, 0, 0);
        calibrate(Map.of("1", success, "2", failure), Map.of("3", runningTask));

        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.realPendingCount());

        // WorkerStatus may repeat a terminal observation in adjacent snapshots.
        // Repeating it must not decrement the survivor count again.
        calibrate(Map.of("1", success), Map.of("3", runningTask));
        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.realPendingCount());
    }

    @Test
    void calibrateAllFailuresClearsBatchIdempotentlyWithoutCompletionMetrics() {
        endpoint.commitBatch(1L, 2_000, List.of(
                createBatchItem("1", 500, 200),
                createBatchItem("2", 10000, 0)));

        TaskInfo firstFailure = taskInfo("1", 1L, null, 500, 40);
        TaskInfo secondFailure = taskInfo("2", 1L, null, 501, 50);
        calibrate(Map.of("1", firstFailure, "2", secondFailure), Map.of());

        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.realPendingCount());
        verify(endpointReporter, never()).reportBatchPredictedTimeMs(
                anyString(), anyString(), ArgumentMatchers.anyLong());

        calibrate(Map.of("1", firstFailure, "2", secondFailure), Map.of());
        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.realPendingCount(),
                "repeated failure deltas must not decrement the ledger twice");
    }

    @Test
    void repeatedSuccessfulTerminalReportsCompletionExactlyOnce() {
        endpoint.commitBatch(1L, 100, List.of(createBatchItem("1", 500, 200)));
        TaskInfo success = taskInfo("1", 1L, null, 0, 40);

        calibrate(Map.of("1", success), Map.of());
        calibrate(Map.of("1", success), Map.of());

        assertEquals(0, endpoint.realPendingCount());
        verify(endpointReporter).reportBatchPredictedTimeMs(
                anyString(), anyString(), ArgumentMatchers.anyLong());
        verify(endpointReporter).reportBatchActualTimeMs(
                anyString(), anyString(), ArgumentMatchers.anyLong());
        verify(endpointReporter).reportBatchPredictGapMs(
                anyString(), anyString(), ArgumentMatchers.anyLong());
    }

    @Test
    void batchInflightReanchorsAcrossRunningQueuedRunning() {
        BatchInflight batch = new BatchInflight(5_000,
                List.of(createBatchItem("1", 500, 0)));
        long baseMs = batch.progressBaseMs();

        batch.markRunning(baseMs + 10);
        assertEquals(baseMs + 10, batch.progressBaseMs());
        batch.markQueued(baseMs + 20);
        assertEquals(baseMs + 20, batch.progressBaseMs());
        batch.markRunning(baseMs + 30);
        assertEquals(baseMs + 30, batch.progressBaseMs());
    }

    @Test
    void batchInflightKeepsCreationAgeSeparateFromActivity() {
        BatchInflight batch = new BatchInflight(5_000,
                List.of(createBatchItem("1", 500, 0)));
        long createdAtMs = batch.createdAtMs();

        batch.touch(createdAtMs + 1_000);

        assertEquals(createdAtMs, batch.createdAtMs());
        assertEquals(createdAtMs + 1_000, batch.lastObservedAtMs());
    }

    @Test
    void inflightMaxAgeMetricUsesCreationTimeNotLatestActivity() throws InterruptedException {
        endpoint.commitBatch(1L, 5_000, List.of(createBatchItem("1", 500, 0)));
        Thread.sleep(30);
        calibrate(Map.of(), Map.of(
                "1", taskInfo("1", 1L, TaskPhase.RUNNING, 0, 0)));

        endpoint.reportBatchMetrics(endpointReporter);

        verify(endpointReporter).reportInflightMaxAgeMs(
                anyString(), anyString(), ArgumentMatchers.longThat(age -> age >= 20));
    }

    @Test
    void runningObservationRefreshesBatchInactivityTtl() throws InterruptedException {
        BatchItem longItem = createBatchItem("1", 10000, 0);
        endpoint.commitBatch(1L, 2_000, List.of(longItem));

        Thread.sleep(150);
        TaskInfo running = taskInfo("1", 1L, TaskPhase.RUNNING, 0, 0);
        calibrate(Map.of(), Map.of("1", running));

        assertEquals(0, endpoint.evictExpiredBatches(100),
                "an actively observed long-running batch must not be evicted by creation age");
        assertEquals(1, endpoint.getInflightBatchCount());
    }

    @Test
    void foreignRunningObservationDoesNotRefreshBatchInactivityTtl()
            throws InterruptedException {
        endpoint.commitBatch(1L, 2_000, List.of(createBatchItem("1", 10000, 0)));
        Thread.sleep(10);

        TaskInfo foreign = taskInfo("999", 1L, TaskPhase.RUNNING, 0, 0);
        calibrate(Map.of(), Map.of("999", foreign));

        assertEquals(1, endpoint.evictExpiredBatches(1));
        assertEquals(0, endpoint.getInflightBatchCount());
    }

    @Test
    void partialCompletionKeepsFixedWindowMaxInflightGateClosed() throws Exception {
        FlexlbConfig limitedConfig = new FlexlbConfig();
        configureBatch(limitedConfig, 100, 1, 0, 1);
        setFormula(limitedConfig, "10 + 0.1*sum(computeTokens) + 5*batchSize");

        WorkerStatus status = new WorkerStatus();
        status.setIp("127.0.0.3");
        status.setPort(8080);
        status.setGrpcPort(8090);
        status.setRole(RoleType.PREFILL);

        CountDownLatch dispatched = new CountDownLatch(1);
        DecisionGroupHandler handler = new DecisionGroupHandler() {
            @Override public void onExpired(BatchItem head) {}
            @Override public void onDecisionGroupReady(List<BatchItem> items, DecisionGroupMetadata meta) {
                dispatched.countDown();
            }
            @Override public void onOfferFailure(BatchItem item, Throwable error) {}
            @Override public void onDeliveryFailure(BatchItem item, Throwable error) {}
        };
        PrefillEndpoint limited = new PrefillEndpoint(
                status, limitedConfig, handler, mock(BatchSchedulerReporter.class));
        try {
            limited.commitBatch(700L, 2_000, List.of(
                    createBatchItem(limited, "101", 500, 200),
                    createBatchItem(limited, "102", 10_000, 0)));
            limited.getBatcher().offer(createBatchItem(limited, "103", 500, 0));

            assertFalse(dispatched.await(50, TimeUnit.MILLISECONDS));

            WorkerStatusResponse partial = new WorkerStatusResponse();
            partial.setFinishedTaskInfo(Map.of(
                    "101", taskInfo("101", 700L, null, 0, 40)));
            partial.setRunningTaskInfo(Map.of(
                    "102", taskInfo("102", 700L, TaskPhase.RUNNING, 0, 0)));
            limited.onWorkerStatusUpdate(limited.getStatus(), partial);

            assertEquals(1, limited.getInflightBatchCount());
            assertFalse(dispatched.await(100, TimeUnit.MILLISECONDS),
                    "a short member finishing must not reopen maxInflight=1 while its long sibling runs");

            WorkerStatusResponse complete = new WorkerStatusResponse();
            complete.setFinishedTaskInfo(Map.of(
                    "102", taskInfo("102", 700L, null, 0, 1_900)));
            complete.setRunningTaskInfo(Map.of());
            limited.onWorkerStatusUpdate(limited.getStatus(), complete);

            assertTrue(dispatched.await(2, TimeUnit.SECONDS),
                    "the next batch should dispatch after the final member completes");
        } finally {
            limited.close();
        }
    }

    @Test
    void calibrateHandlesTaskWithNoBatchId() {
        BatchItem item = createBatchItem("1", 500, 200);
        endpoint.commitBatch(1L, 100, List.of(item));

        Map<String, TaskInfo> finished = new HashMap<>();
        TaskInfo badTask = new TaskInfo();
        badTask.setRequestId("999"); // non-colliding: won't match batchId=1
        badTask.setBatchId(-1);
        badTask.setErrorCode(0);
        finished.put("1", badTask);

        // should not throw, just log a warning for missing non-batch inflight
        calibrate(finished, Map.of());
        assertEquals(1, endpoint.getInflightBatchCount());
    }

    @Test
    void calibrateMissingBatchIdRemovesSingleMemberRealBatch() {
        endpoint.commitBatch(700L, 100, List.of(createBatchItem("101", 500, 200)));

        calibrate(Map.of("101", priorityCanceledTask("101", -1L)), Map.of());

        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.realPendingCount());
    }

    @Test
    void calibrateMissingBatchIdStillRemovesOrdinaryNonBatchReservation() {
        endpoint.commitBatch("request-101", 100, List.of());
        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.getInflightRouteRequestCount());

        TaskInfo finished = new TaskInfo();
        finished.setRequestId("request-101");
        finished.setBatchId(-1L);
        finished.setErrorCode(0);
        calibrate(Map.of("request-101", finished), Map.of());

        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.getInflightRouteRequestCount());
        assertEquals(0, endpoint.realPendingCount());
    }

    @Test
    void stringRequestReservationKeepsBatchAccountingAndRollback() {
        endpoint.commitBatch("req-abc-001", 100, List.of());
        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.getInflightRouteRequestCount());
        assertEquals(0, endpoint.getInflightRequestCount());
        endpoint.releaseBatch("req-abc-001");
        assertEquals(0, endpoint.getInflightBatchCount());
    }

    @Test
    void calibrateMissingBatchIdRepacksOnlyMatchingMember() {
        endpoint.commitBatch(700L, 100, List.of(
                createBatchItem("101", 500, 200),
                createBatchItem("102", 300, 100)));

        calibrate(Map.of("101", priorityCanceledTask("101", -1L)), Map.of());

        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.realPendingCount(),
                "missing batch id must retire only the canceled member");

        TaskInfo survivingSuccess = new TaskInfo();
        survivingSuccess.setRequestId("102");
        survivingSuccess.setBatchId(700L);
        survivingSuccess.setErrorCode(0);
        calibrate(Map.of("102", survivingSuccess), Map.of());
        assertEquals(0, endpoint.getInflightBatchCount(),
                "the surviving member must remain associated with the original batch");
    }

    @Test
    void calibrateMissingBatchIdDoesNotTreatRequestIdAsForeignBatchId() {
        // The canceled request id collides with another real batch id. Directly
        // removing key=101 would erase the foreign request (201); member lookup
        // must instead find request 101 in batch 700.
        endpoint.commitBatch(101L, 100, List.of(createBatchItem("201", 500, 200)));
        endpoint.commitBatch(700L, 100, List.of(createBatchItem("101", 300, 100)));

        calibrate(Map.of("101", priorityCanceledTask("101", -1L)), Map.of());

        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.realPendingCount());

        TaskInfo foreignBatchMemberSuccess = new TaskInfo();
        foreignBatchMemberSuccess.setRequestId("201");
        foreignBatchMemberSuccess.setBatchId(101L);
        foreignBatchMemberSuccess.setErrorCode(0);
        calibrate(Map.of("201", foreignBatchMemberSuccess), Map.of());
        assertEquals(0, endpoint.getInflightBatchCount(),
                "the colliding batch-id entry must survive until its own member finishes");
    }

    @Test
    void calibrateMissingBatchIdDoesNotGuessAcrossDuplicateLiveBatches() {
        endpoint.commitBatch(700L, 100, List.of(createBatchItem("101", 500, 200)));
        endpoint.commitBatch(701L, 100, List.of(createBatchItem("101", 300, 100)));

        calibrate(Map.of("101", priorityCanceledTask("101", -1L)), Map.of());

        assertEquals(2, endpoint.getInflightBatchCount(),
                "an ambiguous missing generation must not erase either live batch");
        assertEquals(2, endpoint.realPendingCount());
    }

    @Test
    void calibrateMissingBatchIdPreservesProtectedBatchMember() {
        endpoint.commitBatch(700L, 100, List.of(
                createBatchItem("101", 500, 200),
                createBatchItem("102", 300, 100)));
        endpoint.tryProtectBatchMember(700L, "101");

        calibrate(Map.of("102", priorityCanceledTask("102", -1L)), Map.of());
        assertEquals(1, endpoint.getInflightBatchCount(),
                "a non-protected sibling must be repacked, not erase the protected batch");
        assertEquals(1, endpoint.realPendingCount());

        TaskInfo canceled = priorityCanceledTask("101", -1L);
        calibrate(Map.of("101", canceled), Map.of());
        assertEquals(1, endpoint.getInflightBatchCount(),
                "generic endpoint calibration must not bypass the reconciliation owner");
        assertEquals(1, endpoint.realPendingCount());

        endpoint.releaseBatchMemberProtection(700L, "101");
        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.realPendingCount());
    }

    @Test
    void batchMemberProtectionWinsConcurrentFinishedSettlement() throws Exception {
        BlockingReadList<BatchItem> requests = new BlockingReadList<>(
                List.of(createBatchItem("101", 500, 200)));
        endpoint.commitBatch(700L, 100, requests);
        requests.blockNextRead();

        ExecutorService executor = Executors.newFixedThreadPool(2);
        CountDownLatch settlementStarted = new CountDownLatch(1);
        try {
            CompletableFuture<Boolean> protection =
                    CompletableFuture.supplyAsync(
                            () -> endpoint.tryProtectBatchMember(700L, "101"), executor);
            assertTrue(requests.awaitReadBlocked(),
                    "protection must enter the batch-key critical section");

            TaskInfo finished = taskInfo("101", 700L, null, 0, 10);
            CompletableFuture<Void> settlement = CompletableFuture.runAsync(() -> {
                settlementStarted.countDown();
                calibrate(Map.of("101", finished), Map.of());
            }, executor);
            assertTrue(settlementStarted.await(2, TimeUnit.SECONDS));

            requests.releaseRead();
            assertTrue(protection.get(2, TimeUnit.SECONDS));
            settlement.get(2, TimeUnit.SECONDS);

            assertEquals(1, endpoint.getInflightBatchCount());
            assertEquals(1, endpoint.realPendingCount(),
                    "the terminal racing after protection must be deferred");
            endpoint.releaseBatchMemberProtection(700L, "101");
            assertEquals(0, endpoint.getInflightBatchCount());
            assertEquals(0, endpoint.realPendingCount());
        } finally {
            requests.releaseRead();
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(2, TimeUnit.SECONDS));
        }
    }

    @Test
    void finishedSettlementWinsConcurrentBatchMemberProtection() throws Exception {
        BlockingReadList<BatchItem> requests = new BlockingReadList<>(
                List.of(createBatchItem("101", 500, 200)));
        endpoint.commitBatch(700L, 100, requests);
        requests.blockNextRead();

        ExecutorService executor = Executors.newFixedThreadPool(2);
        CountDownLatch protectionStarted = new CountDownLatch(1);
        try {
            TaskInfo finished = taskInfo("101", 700L, null, 0, 10);
            CompletableFuture<Void> settlement = CompletableFuture.runAsync(
                    () -> calibrate(Map.of("101", finished), Map.of()), executor);
            assertTrue(requests.awaitReadBlocked(),
                    "settlement must enter the batch-key critical section");

            CompletableFuture<Boolean> protection =
                    CompletableFuture.supplyAsync(() -> {
                        protectionStarted.countDown();
                        return endpoint.tryProtectBatchMember(700L, "101");
                    }, executor);
            assertTrue(protectionStarted.await(2, TimeUnit.SECONDS));

            requests.releaseRead();
            settlement.get(2, TimeUnit.SECONDS);
            assertFalse(protection.get(2, TimeUnit.SECONDS));
            assertEquals(0, endpoint.getInflightBatchCount());
            assertEquals(0, endpoint.realPendingCount());
        } finally {
            requests.releaseRead();
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(2, TimeUnit.SECONDS));
        }
    }

    @Test
    void missingBatchIdCleanupUnblocksFixedWindowMaxInflightOne() throws Exception {
        FlexlbConfig limitedConfig = new FlexlbConfig();
        configureBatch(limitedConfig, 100, 1, 0, 1);
        setFormula(limitedConfig, "10 + 0.1*sum(computeTokens) + 5*batchSize");

        WorkerStatus status = new WorkerStatus();
        status.setIp("127.0.0.2");
        status.setPort(8080);
        status.setGrpcPort(8090);
        status.setRole(RoleType.PREFILL);

        CountDownLatch dispatched = new CountDownLatch(1);
        DecisionGroupHandler handler = new DecisionGroupHandler() {
            @Override public void onExpired(BatchItem head) {}
            @Override public void onDecisionGroupReady(List<BatchItem> items, DecisionGroupMetadata meta) {
                dispatched.countDown();
            }
            @Override public void onOfferFailure(BatchItem item, Throwable error) {}
            @Override public void onDeliveryFailure(BatchItem item, Throwable error) {}
        };
        PrefillEndpoint limited = new PrefillEndpoint(
                status, limitedConfig, handler, mock(BatchSchedulerReporter.class));
        try {
            limited.commitBatch(700L, 100,
                    List.of(createBatchItem(limited, "101", 500, 200)));
            limited.getBatcher().offer(createBatchItem(limited, "102", 300, 100));

            assertFalse(dispatched.await(50, TimeUnit.MILLISECONDS),
                    "maxInflight=1 must hold the next batch while the ledger is occupied");

            WorkerStatusResponse response = new WorkerStatusResponse();
            response.setFinishedTaskInfo(Map.of("101", priorityCanceledTask("101", -1L)));
            response.setRunningTaskInfo(Map.of());
            limited.onWorkerStatusUpdate(limited.getStatus(), response);

            assertTrue(dispatched.await(2, TimeUnit.SECONDS),
                    "missing-batch-id terminal must release the slot for the next dispatch");
            assertEquals(0, limited.getInflightBatchCount());
        } finally {
            limited.close();
        }
    }

    @Test
    void calibrateDoesNotRemoveBatchWithForeignRequestId() {
        // Commit batch with requestId=100
        BatchItem item = createBatchItem("100", 500, 200);
        endpoint.commitBatch(1L, 100, List.of(item));
        assertEquals(1, endpoint.getInflightBatchCount());

        // Engine reports success for batchId=1 but with requestId=999 (foreign)
        Map<String, TaskInfo> finished = new HashMap<>();
        TaskInfo foreignTask = new TaskInfo();
        foreignTask.setBatchId(1L);
        foreignTask.setRequestId("999");
        foreignTask.setErrorCode(0);
        finished.put("999", foreignTask);

        calibrate(finished, new HashMap<>());
        // Batch should NOT be removed — requestId doesn't match
        assertEquals(1, endpoint.getInflightBatchCount());
    }

    @Test
    void calibrateRemovesBatchWithMatchingRequestId() {
        BatchItem item = createBatchItem("100", 500, 200);
        endpoint.commitBatch(1L, 100, List.of(item));

        Map<String, TaskInfo> finished = new HashMap<>();
        TaskInfo task = new TaskInfo();
        task.setBatchId(1L);
        task.setRequestId("100");
        task.setErrorCode(0);
        finished.put("100", task);

        calibrate(finished, new HashMap<>());
        assertEquals(0, endpoint.getInflightBatchCount());
    }

    @Test
    void calibrateSuccessOnlyRetiresSiblingWhileBatchMemberReconciles() {
        BatchItem reconciling = createBatchItem("101", 500, 200);
        BatchItem sibling = createBatchItem("102", 300, 100);
        endpoint.commitBatch(7L, 100, List.of(reconciling, sibling));
        endpoint.tryProtectBatchMember(7L, "101");

        TaskInfo siblingSuccess = new TaskInfo();
        siblingSuccess.setBatchId(7L);
        siblingSuccess.setRequestId("102");
        siblingSuccess.setErrorCode(0);
        calibrate(Map.of("102", siblingSuccess), Map.of());

        assertEquals(1, endpoint.getInflightBatchCount(),
                "sibling success must not erase the reconciling batch member");
        assertEquals(1, endpoint.realPendingCount());

        TaskInfo ambiguousMemberSuccess = new TaskInfo();
        ambiguousMemberSuccess.setBatchId(7L);
        ambiguousMemberSuccess.setRequestId("101");
        ambiguousMemberSuccess.setErrorCode(0);
        calibrate(Map.of("101", ambiguousMemberSuccess), Map.of());
        assertEquals(1, endpoint.getInflightBatchCount(),
                "ordinary success is not a reconciliation terminal");
        assertEquals(1, endpoint.realPendingCount());

        endpoint.releaseBatchMemberProtection(7L, "101");
        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.realPendingCount());
    }

    @Test
    void protectedAndSiblingFailuresSettleFromOneWorkerSnapshot() {
        endpoint.commitBatch(7L, 100, List.of(
                createBatchItem("101", 500, 200),
                createBatchItem("102", 300, 100)));
        endpoint.tryProtectBatchMember(7L, "101");

        TaskInfo protectedFailure = taskInfo("101", 7L, null, 500, 40);
        TaskInfo siblingFailure = taskInfo("102", 7L, null, 501, 50);
        calibrate(Map.of("101", protectedFailure, "102", siblingFailure), Map.of());

        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.realPendingCount());

        endpoint.releaseBatchMemberProtection(7L, "101");
        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.realPendingCount());
        verify(endpointReporter, never()).reportBatchPredictedTimeMs(
                anyString(), anyString(), ArgumentMatchers.anyLong());

        endpoint.releaseBatchMemberProtection(7L, "101");
        assertEquals(0, endpoint.realPendingCount());
    }

    // ---- estimated waiting time ----

    @Test
    void realWaitTimeMsZeroWhenNoInflight() {
        assertEquals(0, endpoint.realWaitTimeMs());
    }

    @Test
    void realWaitTimeMsPositiveWithInflight() {
        BatchItem item = createBatchItem("1", 500, 200);
        endpoint.commitBatch(1L, 5000, List.of(item)); // 5s prediction

        long waitMs = endpoint.realWaitTimeMs();
        assertTrue(waitMs > 0, "Should have non-zero wait time with inflight batch");
        assertTrue(waitMs <= 5000, "Wait time should not exceed prediction");
    }

    @Test
    void realWaitTimeMsDecreasesOverTime() throws InterruptedException {
        BatchItem item = createBatchItem("1", 500, 200);
        endpoint.commitBatch(1L, 5000, List.of(item));

        long waitBefore = endpoint.realWaitTimeMs();

        // Mark the batch as running so elapsed time counts
        Map<String, TaskInfo> running = new HashMap<>();
        TaskInfo runningTask = new TaskInfo();
        runningTask.setRequestId("1");
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
        BatchItem item = createBatchItem("1", 500, 200);
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
        BatchItem item = createBatchItem("1", 500, 200);
        endpoint.commitBatch(1L, 100, List.of(item));

        int evicted = endpoint.evictExpiredBatches(60_000); // 60s TTL — fresh entry survives
        assertEquals(0, evicted);
        assertEquals(1, endpoint.getInflightBatchCount());
    }

    @Test
    void evictExpiredBatchesRetainsAckAmbiguousBatchUntilReconciled()
            throws InterruptedException {
        BatchItem item = createBatchItem("1", 500, 200);
        endpoint.commitBatch(1L, 100, List.of(item));
        endpoint.tryProtectBatchMember(1L, item.requestId());
        Thread.sleep(10);

        assertEquals(0, endpoint.evictExpiredBatches(1));
        assertEquals(1, endpoint.getInflightBatchCount());

        endpoint.releaseBatchMemberProtection(1L, item.requestId());
        assertEquals(0, endpoint.evictExpiredBatches(1),
                "authoritative reconciliation settlement refreshes batch activity");
        Thread.sleep(10);
        assertEquals(1, endpoint.evictExpiredBatches(1));
        assertEquals(0, endpoint.getInflightBatchCount());
    }

    // ---- realPendingCount ----

    @Test
    void realPendingCountUnionsEngineTasksWithLocalLedger() {
        endpoint.commitBatch(1L, 100, List.of(
                createBatchItem("101", 500, 0),
                createBatchItem("102", 500, 0)));

        TaskInfo overlapping = taskInfo("102", 1L, TaskPhase.RUNNING, 0, 0);
        TaskInfo untrackedOne = taskInfo("900", 90L, TaskPhase.RUNNING, 0, 0);
        TaskInfo untrackedTwo = taskInfo("901", 91L, TaskPhase.RECEIVED, 0, 0);
        TaskInfo duplicateUntracked = taskInfo("900", 92L, TaskPhase.RUNNING, 0, 0);
        TaskInfo overlayOnly = taskInfo("999", 99L, TaskPhase.PENDING, 0, 0);
        overlayOnly.setPriorityPreemptionProgress(PriorityPreemptionProgress.CANCELING);

        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setFinishedTaskInfo(Map.of());
        response.setRunningTaskInfo(Map.of(
                "102", overlapping,
                "900a", untrackedOne,
                "901", untrackedTwo,
                "900b", duplicateUntracked,
                "999", overlayOnly));
        endpoint.onWorkerStatusUpdate(endpoint.getStatus(), response);

        assertEquals(4, endpoint.realPendingCount(),
                "two local requests plus two unique Engine-only tasks");

        response.setRunningTaskInfo(Map.of());
        endpoint.onWorkerStatusUpdate(endpoint.getStatus(), response);
        assertEquals(2, endpoint.realPendingCount());
    }

    @Test
    void realPendingCountFallsBackToEngineQueryLengthScalars() {
        endpoint.commitBatch(1L, 100, List.of(createBatchItem("101", 500, 0)));

        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setFinishedTaskInfo(Map.of());
        response.setRunningTaskInfo(Map.of());
        response.setWaitingQueryLen(3);
        response.setRunningQueryLen(2);
        endpoint.onWorkerStatusUpdate(endpoint.getStatus(), response);

        assertEquals(5, endpoint.realPendingCount());
    }

    @Test
    void realPendingCountUsesConservativeScalarBoundForPartialTaskDetails() {
        endpoint.commitBatch(1L, 100, List.of(createBatchItem("101", 500, 0)));

        TaskInfo overlapping = taskInfo("101", 1L, TaskPhase.RUNNING, 0, 0);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setFinishedTaskInfo(Map.of());
        response.setRunningTaskInfo(Map.of("101", overlapping));
        response.setWaitingQueryLen(3);
        response.setRunningQueryLen(2);
        endpoint.onWorkerStatusUpdate(endpoint.getStatus(), response);

        assertEquals(5, endpoint.realPendingCount(),
                "scalar active count must cover a partial detail list without double-counting local tasks");
    }

    @Test
    void realPendingCountIncludesBatcherQueue() throws InterruptedException {
        // Initially, batcher queue is empty
        assertEquals(0, endpoint.realPendingCount());

        BatchItem item = createBatchItem("1", 500, 200);
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
            slowEndpoint.getBatcher().offer(createPriorityBatchItem("1", 70));
            slowEndpoint.getBatcher().offer(createBatchItem("2", 300, 0)); // legacy: budget=null -> priority 0

            BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);
            slowEndpoint.reportBatchMetrics(reporter);

            // Single-report with priority tag (no global untagged series)
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
        BatchItem item = createBatchItem("1", 500, 200);
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
        configureBatch(slowConfig, 100, 100, fixedWaitMs, null);
        return new PrefillEndpoint(status, slowConfig, noopHandler(), mock(BatchSchedulerReporter.class));
    }

    private BatchItem createPriorityBatchItem(String requestId, int priority) {
        long now = System.currentTimeMillis();
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(500);
        request.setPriority(priority);

        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setConfig(config);
        ctx.setSchedulingMetadata(SchedulingMetadata.explicit(priority, now + 60_000));

        return new BatchItem(ctx, null, null, null, null, null, null, now);
    }

    private void calibrate(Map<String, TaskInfo> finished, Map<String, TaskInfo> running) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setFinishedTaskInfo(finished);
        response.setRunningTaskInfo(running);
        endpoint.onWorkerStatusUpdate(endpoint.getStatus(), response);
    }

    private BatchItem createBatchItem(String requestId, long seqLen, long hitCacheLen) {
        return createBatchItem(endpoint, requestId, seqLen, hitCacheLen);
    }

    private static BatchItem createBatchItem(PrefillEndpoint owner,
                                             String requestId,
                                             long seqLen,
                                             long hitCacheLen) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);

        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setConfig(new FlexlbConfig());

        ServerStatus prefill = new ServerStatus();
        prefill.setRole(RoleType.PREFILL);
        prefill.setServerIp("127.0.0.1");
        prefill.setHttpPort(8080);
        prefill.setGrpcPort(8090);
        DebugInfo debugInfo = new DebugInfo();
        debugInfo.setHitCacheLen(hitCacheLen);
        prefill.setDebugInfo(debugInfo);

        return new BatchItem(ctx, null, null, prefill, null, owner, null, System.currentTimeMillis());
    }

    private static void configureBatch(
            FlexlbConfig target,
            int maxWaiting,
            int maxRequests,
            long maxCollectionWaitMs,
            Integer maxInflightBatches) {
        target.batchDispatcher().setMaxWaitingRequestsPerPrefillWorker(maxWaiting);
        target.batchDispatcher().setMaxRequests(maxRequests);
        target.batchDispatcher().setMaxCollectionWaitMs(maxCollectionWaitMs);
        target.batchDispatcher().setMaxInflightBatchesPerPrefillWorker(maxInflightBatches);
    }

    private static void setFormula(FlexlbConfig target, String expression) {
        RoutingConfig.FormulaEstimatorConfig estimator =
                (RoutingConfig.FormulaEstimatorConfig) target.getRouter().getRoles()
                        .getPrefill().getExecutionTimeEstimator();
        estimator.setExpression(expression);
    }

    private static TaskInfo priorityCanceledTask(String requestId, long batchId) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setBatchId(batchId);
        task.setErrorCode(8429);
        task.setErrorMessage("priority preempted");
        task.setPriorityPreemptionProgress(PriorityPreemptionProgress.CANCELED);
        return task;
    }

    private static TaskInfo taskInfo(String requestId,
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

    private static DecisionGroupHandler noopHandler() {
        return new DecisionGroupHandler() {
            @Override public void onExpired(BatchItem head) {}
            @Override public void onDecisionGroupReady(List<BatchItem> items, DecisionGroupMetadata meta) {}
            @Override public void onOfferFailure(BatchItem item, Throwable error) {}
            @Override public void onDeliveryFailure(BatchItem item, Throwable error) {}
        };
    }

    /** One-shot read barrier used to hold a real batch-key computation in place. */
    private static final class BlockingReadList<E> extends AbstractList<E> {
        private final List<E> delegate;
        private final AtomicBoolean blockNextRead = new AtomicBoolean();
        private final CountDownLatch readBlocked = new CountDownLatch(1);
        private final CountDownLatch releaseRead = new CountDownLatch(1);

        private BlockingReadList(List<E> delegate) {
            this.delegate = List.copyOf(delegate);
        }

        private void blockNextRead() {
            if (!blockNextRead.compareAndSet(false, true)) {
                throw new IllegalStateException("read barrier is already armed");
            }
        }

        private boolean awaitReadBlocked() throws InterruptedException {
            return readBlocked.await(2, TimeUnit.SECONDS);
        }

        private void releaseRead() {
            releaseRead.countDown();
        }

        @Override
        public E get(int index) {
            if (blockNextRead.compareAndSet(true, false)) {
                readBlocked.countDown();
                try {
                    if (!releaseRead.await(5, TimeUnit.SECONDS)) {
                        throw new AssertionError("timed out waiting to release batch read");
                    }
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    throw new AssertionError("batch read interrupted", interrupted);
                }
            }
            return delegate.get(index);
        }

        @Override
        public int size() {
            return delegate.size();
        }
    }
}
