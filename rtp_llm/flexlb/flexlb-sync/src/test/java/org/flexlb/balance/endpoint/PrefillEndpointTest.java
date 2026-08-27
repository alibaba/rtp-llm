package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.DecisionGroupHandler;
import org.flexlb.balance.scheduler.AdmittedDecisionGroup;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.DecisionGroupMetadata;
import org.flexlb.balance.scheduler.DeliveryCapacityAdmission;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.balance.scheduler.TestCapacityAdmission;
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

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyInt;
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
        configureBatch(config, 100, config.fixedWindowDecision().getMaxRequests(), 300, null);
        setFormula(config, "10 + 0.1*sum(computeTokens) + 5*batchSize");

        endpointReporter = mock(BatchSchedulerReporter.class);
        endpoint = new PrefillEndpoint(
                status,
                config,
                noopHandler(),
                TestCapacityAdmission.alwaysAvailable(),
                endpointReporter);
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
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 1L, 100, List.of(item));

        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.getQueueBatchCapacityUsage());
        assertEquals(1, endpoint.realPendingCount());
    }

    @Test
    void releaseBatchDecreasesInflightCount() {
        BatchItem item = createBatchItem(1L, 500, 200);
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 1L, 100, List.of(item));
        endpoint.releaseBatch(1L);

        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.getQueueBatchCapacityUsage());
    }

    @Test
    void releaseBatchRetainsOnlyProtectedMembers() {
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 7L, 100, List.of(
                createBatchItem(101L, 500, 200),
                createBatchItem(102L, 300, 100)));
        assertTrue(endpoint.tryProtectBatchMember(7L, 101L));

        endpoint.releaseBatch(7L);

        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.getQueueBatchCapacityUsage(),
                "partial repack must retain the registered group slot");
        assertEquals(1, endpoint.realPendingCount(),
                "a delivery failure must not reopen capacity owned by an Engine fence");

        endpoint.releaseBatchMemberProtection(7L, 101L);
        endpoint.releaseBatch(7L);
        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.getQueueBatchCapacityUsage(),
                "the last member releases the registered group slot");
        assertEquals(0, endpoint.realPendingCount());
    }

    @Test
    void commitMultipleBatches() {
        BatchItem item1 = createBatchItem(1L, 500, 200);
        BatchItem item2 = createBatchItem(2L, 300, 100);
        BatchItem item3 = createBatchItem(3L, 400, 0);

        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 1L, 100, List.of(item1, item2));
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 2L, 50, List.of(item3));

        assertEquals(2, endpoint.getInflightBatchCount());
        assertEquals(3, endpoint.realPendingCount());
    }

    // ---- repack batch ----

    @Test
    void repackBatchRemovesFailedRequests() {
        BatchItem item1 = createBatchItem(1L, 500, 200);
        BatchItem item2 = createBatchItem(2L, 300, 100);
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 1L, 100, List.of(item1, item2));

        endpoint.repackBatch(1L, Set.of(2L));
        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.getQueueBatchCapacityUsage());
        assertEquals(1, endpoint.realPendingCount());
    }

    @Test
    void repackBatchAllFailedReturnsNull() {
        BatchItem item1 = createBatchItem(1L, 500, 200);
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 1L, 100, List.of(item1));

        endpoint.repackBatch(1L, Set.of(1L));
        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.getQueueBatchCapacityUsage());
    }

    // ---- calibrate ----

    @Test
    void calibrateRemovesBatchOnSuccess() {
        BatchItem item = createBatchItem(1L, 500, 200);
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 1L, 100, List.of(item));

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
    void completion_observer_failure_does_not_escape_finished_settlement() {
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 9L, 100, List.of(createBatchItem(9L, 500, 200)));
        doThrow(new IllegalStateException("metrics unavailable"))
                .when(endpointReporter)
                .reportBatchPredictedTimeMs("PREFILL", "127.0.0.1", 100);

        TaskInfo finished = taskInfo(9L, 9L, null, 0, 125);
        assertDoesNotThrow(() -> calibrate(Map.of("9", finished), Map.of()));

        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.realPendingCount());
        verify(endpointReporter).reportBatchActualTimeMs("PREFILL", "127.0.0.1", 125);
        verify(endpointReporter).reportBatchPredictGapMs("PREFILL", "127.0.0.1", 25);
    }

    @Test
    void calibrateRepacksOnPartialFailure() {
        BatchItem item1 = createBatchItem(1L, 500, 200);
        BatchItem item2 = createBatchItem(2L, 300, 100);
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 1L, 100, List.of(item1, item2));

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
        BatchItem shortItem = createBatchItem(1L, 500, 200);
        BatchItem longItem = createBatchItem(2L, 10_000, 0);
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 1L, 2_000, List.of(shortItem, longItem));

        TaskInfo finishedShort = taskInfo(1L, 1L, null, 0, 40);
        TaskInfo runningLong = taskInfo(2L, 1L, TaskPhase.RUNNING, 0, 0);
        calibrate(Map.of("1", finishedShort), Map.of("2", runningLong));

        assertEquals(1, endpoint.getInflightBatchCount(),
                "one finished member must not release the whole batch");
        assertEquals(1, endpoint.realPendingCount(),
                "the still-running long member must remain in Master accounting");

        TaskInfo finishedLong = taskInfo(2L, 1L, null, 0, 1_900);
        calibrate(Map.of("2", finishedLong), Map.of());

        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.realPendingCount());
    }

    @Test
    void calibrateMixedTerminalMembersKeepsOnlyRunningSurvivor() {
        BatchItem succeeded = createBatchItem(1L, 500, 200);
        BatchItem failed = createBatchItem(2L, 300, 100);
        BatchItem running = createBatchItem(3L, 10_000, 0);
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 1L, 2_000, List.of(succeeded, failed, running));

        TaskInfo success = taskInfo(1L, 1L, null, 0, 40);
        TaskInfo failure = taskInfo(2L, 1L, null, 500, 50);
        TaskInfo runningTask = taskInfo(3L, 1L, TaskPhase.RUNNING, 0, 0);
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
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 1L, 2_000, List.of(
                createBatchItem(1L, 500, 200),
                createBatchItem(2L, 10_000, 0)));

        TaskInfo firstFailure = taskInfo(1L, 1L, null, 500, 40);
        TaskInfo secondFailure = taskInfo(2L, 1L, null, 501, 50);
        calibrate(Map.of("1", firstFailure, "2", secondFailure), Map.of());

        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.realPendingCount());
        verify(endpointReporter, never()).reportBatchPredictedTimeMs(
                anyString(), anyString(), org.mockito.ArgumentMatchers.anyLong());

        calibrate(Map.of("1", firstFailure, "2", secondFailure), Map.of());
        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.realPendingCount(),
                "repeated failure deltas must not decrement the ledger twice");
    }

    @Test
    void repeatedSuccessfulTerminalReportsCompletionExactlyOnce() {
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 1L, 100, List.of(createBatchItem(1L, 500, 200)));
        TaskInfo success = taskInfo(1L, 1L, null, 0, 40);

        calibrate(Map.of("1", success), Map.of());
        calibrate(Map.of("1", success), Map.of());

        assertEquals(0, endpoint.realPendingCount());
        verify(endpointReporter).reportBatchPredictedTimeMs(
                anyString(), anyString(), org.mockito.ArgumentMatchers.anyLong());
        verify(endpointReporter).reportBatchActualTimeMs(
                anyString(), anyString(), org.mockito.ArgumentMatchers.anyLong());
        verify(endpointReporter).reportBatchPredictGapMs(
                anyString(), anyString(), org.mockito.ArgumentMatchers.anyLong());
    }

    @Test
    void batchInflightReanchorsAcrossRunningQueuedRunning() {
        BatchInflight batch = new BatchInflight(5_000,
                List.of(createBatchItem(1L, 500, 0)), () -> { });
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
                List.of(createBatchItem(1L, 500, 0)), () -> { });
        long createdAtMs = batch.createdAtMs();

        batch.touch(createdAtMs + 1_000);

        assertEquals(createdAtMs, batch.createdAtMs());
        assertEquals(createdAtMs + 1_000, batch.lastObservedAtMs());
    }

    @Test
    void inflightMaxAgeMetricUsesCreationTimeNotLatestActivity() throws InterruptedException {
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 1L, 5_000, List.of(createBatchItem(1L, 500, 0)));
        Thread.sleep(30);
        calibrate(Map.of(), Map.of(
                "1", taskInfo(1L, 1L, TaskPhase.RUNNING, 0, 0)));

        endpoint.reportBatchMetrics(endpointReporter);

        verify(endpointReporter).reportInflightMaxAgeMs(
                anyString(), anyString(), org.mockito.ArgumentMatchers.longThat(age -> age >= 20));
    }

    @Test
    void runningObservationRefreshesBatchInactivityTtl() throws InterruptedException {
        BatchItem longItem = createBatchItem(1L, 10_000, 0);
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 1L, 2_000, List.of(longItem));

        Thread.sleep(150);
        TaskInfo running = taskInfo(1L, 1L, TaskPhase.RUNNING, 0, 0);
        calibrate(Map.of(), Map.of("1", running));

        assertEquals(0, endpoint.evictExpiredBatches(100),
                "an actively observed long-running batch must not be evicted by creation age");
        assertEquals(1, endpoint.getInflightBatchCount());
    }

    @Test
    void foreignRunningObservationDoesNotRefreshBatchInactivityTtl()
            throws InterruptedException {
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 1L, 2_000, List.of(createBatchItem(1L, 10_000, 0)));
        Thread.sleep(10);

        TaskInfo foreign = taskInfo(999L, 1L, TaskPhase.RUNNING, 0, 0);
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
            @Override public void onDecisionGroupAdmitted(
                    AdmittedDecisionGroup group, DecisionGroupMetadata meta) {
                TestCapacityAdmission.complete(group);
                dispatched.countDown();
            }
            @Override public void onOfferFailure(BatchItem item, Throwable error) {}
            @Override public void onDeliveryFailure(BatchItem item, Throwable error) {}
        };
        AtomicReference<PrefillEndpoint> endpointRef = new AtomicReference<>();
        PrefillEndpoint limited = new PrefillEndpoint(
                status,
                limitedConfig,
                handler,
                TestCapacityAdmission.withEndpointBatchCapacity(endpointRef::get),
                mock(BatchSchedulerReporter.class));
        endpointRef.set(limited);
        try {
            TestCapacityAdmission.registerQueueBatchLifecycle(limited, 700L, 2_000, List.of(
                    createBatchItem(limited, limitedConfig, 101L, 500, 200),
                    createBatchItem(limited, limitedConfig, 102L, 10_000, 0)));
            limited.getBatcher().offer(
                    createBatchItem(limited, limitedConfig, 103L, 500, 0));

            assertFalse(dispatched.await(50, TimeUnit.MILLISECONDS));

            WorkerStatusResponse partial = new WorkerStatusResponse();
            partial.setFinishedTaskInfo(Map.of(
                    "101", taskInfo(101L, 700L, null, 0, 40)));
            partial.setRunningTaskInfo(Map.of(
                    "102", taskInfo(102L, 700L, TaskPhase.RUNNING, 0, 0)));
            limited.onWorkerStatusUpdate(limited.getStatus(), partial);

            assertEquals(1, limited.getInflightBatchCount());
            assertFalse(dispatched.await(100, TimeUnit.MILLISECONDS),
                    "a short member finishing must not reopen maxInflight=1 while its long sibling runs");

            WorkerStatusResponse complete = new WorkerStatusResponse();
            complete.setFinishedTaskInfo(Map.of(
                    "102", taskInfo(102L, 700L, null, 0, 1_900)));
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
        BatchItem item = createBatchItem(1L, 500, 200);
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 1L, 100, List.of(item));

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
    void calibrateMissingBatchIdRemovesSingleMemberRealBatch() {
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 700L, 100, List.of(createBatchItem(101L, 500, 200)));

        calibrate(Map.of("101", priorityCanceledTask(101L, -1L)), Map.of());

        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.realPendingCount());
    }

    @Test
    void calibrateMissingBatchIdRemovesDirectRequestLedgerEntry() {
        endpoint.registerDirectRequest(101L, 100);
        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.getIndividuallyTrackedRequestCount());

        TaskInfo finished = new TaskInfo();
        finished.setRequestId(101L);
        finished.setBatchId(-1L);
        finished.setErrorCode(0);
        calibrate(Map.of("101", finished), Map.of());

        assertEquals(0, endpoint.getIndividuallyTrackedRequestCount());
        assertEquals(0, endpoint.realPendingCount());
    }

    @Test
    void calibrateMissingBatchIdRepacksOnlyMatchingMember() {
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 700L, 100, List.of(
                createBatchItem(101L, 500, 200),
                createBatchItem(102L, 300, 100)));

        calibrate(Map.of("101", priorityCanceledTask(101L, -1L)), Map.of());

        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.realPendingCount(),
                "missing batch id must retire only the canceled member");

        TaskInfo survivingSuccess = new TaskInfo();
        survivingSuccess.setRequestId(102L);
        survivingSuccess.setBatchId(700L);
        survivingSuccess.setErrorCode(0);
        calibrate(Map.of("102", survivingSuccess), Map.of());
        assertEquals(0, endpoint.getInflightBatchCount(),
                "the surviving member must remain associated with the original batch");
    }

    @Test
    void calibrateZeroBatchIdRemovesSingleMemberRealBatch() {
        // An engine built on proto3 reports the unset batch_id default 0.
        // It must settle through request-id reconciliation exactly like the
        // -1 priority-cancel sentinel instead of leaking the batch slot into
        // the phantom finishedByBatch[0] bucket until TTL eviction.
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 700L, 100, List.of(createBatchItem(101L, 500, 200)));

        calibrate(Map.of("101", priorityCanceledTask(101L, 0L)), Map.of());

        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.realPendingCount());
    }

    @Test
    void calibrateZeroBatchIdRepacksOnlyMatchingMember() {
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 700L, 100, List.of(
                createBatchItem(101L, 500, 200),
                createBatchItem(102L, 300, 100)));

        calibrate(Map.of("101", priorityCanceledTask(101L, 0L)), Map.of());

        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.realPendingCount(),
                "zero batch id must retire only the matching member");

        TaskInfo survivingSuccess = new TaskInfo();
        survivingSuccess.setRequestId(102L);
        survivingSuccess.setBatchId(700L);
        survivingSuccess.setErrorCode(0);
        calibrate(Map.of("102", survivingSuccess), Map.of());
        assertEquals(0, endpoint.getInflightBatchCount(),
                "the surviving member must remain associated with the original batch");
    }

    @Test
    void calibrateZeroBatchIdWithoutOwnerBatchIsNoOp() {
        BatchItem item = createBatchItem(1L, 500, 200);
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 1L, 100, List.of(item));

        Map<String, TaskInfo> finished = new HashMap<>();
        TaskInfo orphanTask = new TaskInfo();
        orphanTask.setRequestId(999L); // non-colliding: no live batch owns it
        orphanTask.setBatchId(0);
        orphanTask.setErrorCode(0);
        finished.put("999", orphanTask);

        // should not throw, just log a warning for missing non-batch inflight
        calibrate(finished, Map.of());
        assertEquals(1, endpoint.getInflightBatchCount());
    }

    @Test
    void directRequestIdMatchingQueueBatchIdDoesNotOverwriteEitherLifecycle() {
        // DIRECT request 101 and QUEUE batch 101 live in different ledgers.
        // Completing the DIRECT request must not erase QUEUE member 201.
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 101L, 100, List.of(createBatchItem(201L, 500, 200)));
        endpoint.registerDirectRequest(101L, 100);
        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.getQueueBatchCapacityUsage());
        assertEquals(1, endpoint.getIndividuallyTrackedRequestCount());

        calibrate(Map.of("101", priorityCanceledTask(101L, -1L)), Map.of());

        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.getQueueBatchCapacityUsage());
        assertEquals(0, endpoint.getIndividuallyTrackedRequestCount());
        assertEquals(1, endpoint.realPendingCount());

        TaskInfo foreignBatchMemberSuccess = new TaskInfo();
        foreignBatchMemberSuccess.setRequestId(201L);
        foreignBatchMemberSuccess.setBatchId(101L);
        foreignBatchMemberSuccess.setErrorCode(0);
        calibrate(Map.of("201", foreignBatchMemberSuccess), Map.of());
        assertEquals(0, endpoint.getInflightBatchCount(),
                "the matching QUEUE batch id must survive until its own member finishes");
        assertEquals(0, endpoint.getQueueBatchCapacityUsage());
    }

    @Test
    void calibrateMissingBatchIdDoesNotGuessAcrossDuplicateLiveBatches() {
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 700L, 100, List.of(createBatchItem(101L, 500, 200)));
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 701L, 100, List.of(createBatchItem(101L, 300, 100)));

        calibrate(Map.of("101", priorityCanceledTask(101L, -1L)), Map.of());

        assertEquals(2, endpoint.getInflightBatchCount(),
                "an ambiguous missing generation must not erase either live batch");
        assertEquals(2, endpoint.realPendingCount());
    }

    @Test
    void calibrateMissingBatchIdPreservesProtectedBatchMember() {
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 700L, 100, List.of(
                createBatchItem(101L, 500, 200),
                createBatchItem(102L, 300, 100)));
        endpoint.tryProtectBatchMember(700L, 101L);

        calibrate(Map.of("102", priorityCanceledTask(102L, -1L)), Map.of());
        assertEquals(1, endpoint.getInflightBatchCount(),
                "a non-protected sibling must be repacked, not erase the protected batch");
        assertEquals(1, endpoint.realPendingCount());

        TaskInfo canceled = priorityCanceledTask(101L, -1L);
        calibrate(Map.of("101", canceled), Map.of());
        assertEquals(1, endpoint.getInflightBatchCount(),
                "generic endpoint calibration must not bypass the reconciliation owner");
        assertEquals(1, endpoint.realPendingCount());

        endpoint.releaseBatchMemberProtection(700L, 101L);
        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.realPendingCount());
    }

    @Test
    void protectedBatchMemberDefersFinishedSettlementUntilProtectionEnds() {
        TestCapacityAdmission.registerQueueBatchLifecycle(
                endpoint,
                700L,
                100,
                List.of(createBatchItem(101L, 500, 200)));
        assertTrue(endpoint.tryProtectBatchMember(700L, 101L));

        calibrate(Map.of(
                "101", taskInfo(101L, 700L, null, 0, 10)), Map.of());

        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.getQueueBatchCapacityUsage());
        assertEquals(1, endpoint.realPendingCount());

        endpoint.releaseBatchMemberProtection(700L, 101L);
        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.getQueueBatchCapacityUsage());
        assertEquals(0, endpoint.realPendingCount());
    }

    @Test
    void deferredLearningModelUpdateSignalsWithoutAnotherWorkerStatus() {
        PrefillEndpoint learningEndpoint = createLearningEndpoint();
        try {
            // LearningPredictor publishes one model revision per four valid
            // completions. Seed three unchanged samples first.
            for (int sample = 1; sample <= 3; sample++) {
                long batchId = 8_000L + sample;
                long requestId = 9_000L + sample;
                TestCapacityAdmission.registerQueueBatchLifecycle(
                        learningEndpoint,
                        batchId,
                        100L,
                        List.of(createBatchItem(
                                learningEndpoint, requestId, 500L, 200L)));
                reportSuccessfulBatchMember(
                        learningEndpoint, batchId, requestId, 100L + sample);
            }
            assertEquals(0L, learningEndpoint.getPredictor().generation());

            long batchId = 8_004L;
            long requestId = 9_004L;
            TestCapacityAdmission.registerQueueBatchLifecycle(
                    learningEndpoint,
                    batchId,
                    100L,
                    List.of(createBatchItem(
                            learningEndpoint, requestId, 500L, 200L)));
            assertTrue(learningEndpoint.tryProtectBatchMember(batchId, requestId));

            long versionBeforeStatus =
                    learningEndpoint.getBatcher().schedulingInputVersion();
            reportSuccessfulBatchMember(
                    learningEndpoint, batchId, requestId, 104L);
            long versionAfterDeferredStatus =
                    learningEndpoint.getBatcher().schedulingInputVersion();
            assertEquals(versionBeforeStatus + 1L, versionAfterDeferredStatus,
                    "WorkerStatus publishes its own scheduling-input change");
            assertEquals(0L, learningEndpoint.getPredictor().generation(),
                    "the protected terminal has not reached predictor learning");
            assertEquals(1, learningEndpoint.getInflightBatchCount());

            // No WorkerStatus call occurs after this boundary. Releasing the
            // protection applies the cached success and publishes sample four.
            learningEndpoint.releaseBatchMemberProtection(batchId, requestId);

            assertEquals(1L, learningEndpoint.getPredictor().generation());
            assertEquals(versionAfterDeferredStatus + 1L,
                    learningEndpoint.getBatcher().schedulingInputVersion(),
                    "MODEL_UPDATED must wake decisions at the learning boundary");
            assertEquals(0, learningEndpoint.getInflightBatchCount());
        } finally {
            learningEndpoint.close();
        }
    }

    @Test
    void deferredUnchangedLearningAddsNoSignalBeyondWorkerStatus() {
        PrefillEndpoint learningEndpoint = createLearningEndpoint();
        try {
            long batchId = 8_101L;
            long requestId = 9_101L;
            TestCapacityAdmission.registerQueueBatchLifecycle(
                    learningEndpoint,
                    batchId,
                    100L,
                    List.of(createBatchItem(
                            learningEndpoint, requestId, 500L, 200L)));
            assertTrue(learningEndpoint.tryProtectBatchMember(batchId, requestId));

            long versionBeforeStatus =
                    learningEndpoint.getBatcher().schedulingInputVersion();
            reportSuccessfulBatchMember(
                    learningEndpoint, batchId, requestId, 101L);
            long versionAfterDeferredStatus =
                    learningEndpoint.getBatcher().schedulingInputVersion();
            assertEquals(versionBeforeStatus + 1L, versionAfterDeferredStatus);
            assertEquals(0L, learningEndpoint.getPredictor().generation());

            learningEndpoint.releaseBatchMemberProtection(batchId, requestId);

            assertEquals(0L, learningEndpoint.getPredictor().generation(),
                    "the first sample returns MODEL_UNCHANGED");
            assertEquals(versionAfterDeferredStatus,
                    learningEndpoint.getBatcher().schedulingInputVersion(),
                    "MODEL_UNCHANGED must not publish an extra scheduling signal");
            assertEquals(0, learningEndpoint.getInflightBatchCount());
        } finally {
            learningEndpoint.close();
        }
    }

    @Test
    void finishedSettlementRemovesMemberBeforeLateProtection() {
        TestCapacityAdmission.registerQueueBatchLifecycle(
                endpoint,
                700L,
                100,
                List.of(createBatchItem(101L, 500, 200)));

        calibrate(Map.of(
                "101", taskInfo(101L, 700L, null, 0, 10)), Map.of());

        assertFalse(endpoint.tryProtectBatchMember(700L, 101L));
        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.getQueueBatchCapacityUsage());
        assertEquals(0, endpoint.realPendingCount());
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
            @Override public void onDecisionGroupAdmitted(
                    AdmittedDecisionGroup group, DecisionGroupMetadata meta) {
                TestCapacityAdmission.complete(group);
                dispatched.countDown();
            }
            @Override public void onOfferFailure(BatchItem item, Throwable error) {}
            @Override public void onDeliveryFailure(BatchItem item, Throwable error) {}
        };
        AtomicReference<PrefillEndpoint> endpointRef = new AtomicReference<>();
        PrefillEndpoint limited = new PrefillEndpoint(
                status,
                limitedConfig,
                handler,
                TestCapacityAdmission.withEndpointBatchCapacity(endpointRef::get),
                mock(BatchSchedulerReporter.class));
        endpointRef.set(limited);
        try {
            TestCapacityAdmission.registerQueueBatchLifecycle(limited, 700L, 100,
                    List.of(createBatchItem(
                            limited, limitedConfig, 101L, 500, 200)));
            limited.getBatcher().offer(
                    createBatchItem(limited, limitedConfig, 102L, 300, 100));

            assertFalse(dispatched.await(50, TimeUnit.MILLISECONDS),
                    "maxInflight=1 must hold the next batch while the ledger is occupied");

            WorkerStatusResponse response = new WorkerStatusResponse();
            response.setFinishedTaskInfo(Map.of("101", priorityCanceledTask(101L, -1L)));
            response.setRunningTaskInfo(Map.of());
            limited.onWorkerStatusUpdate(limited.getStatus(), response);

            assertTrue(dispatched.await(2, TimeUnit.SECONDS),
                    "missing-batch-id terminal must release the slot for the next dispatch");
            assertEquals(1, limited.getInflightBatchCount(),
                    "the released slot is transferred to the newly dispatched batch");
            assertEquals(1, limited.getQueueBatchCapacityUsage());
        } finally {
            limited.close();
        }
    }

    @Test
    void calibrateDoesNotRemoveBatchWithForeignRequestId() {
        // Commit batch with requestId=100
        BatchItem item = createBatchItem(100L, 500, 200);
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 1L, 100, List.of(item));
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
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 1L, 100, List.of(item));

        Map<String, TaskInfo> finished = new HashMap<>();
        TaskInfo task = new TaskInfo();
        task.setBatchId(1L);
        task.setRequestId(100L);
        task.setErrorCode(0);
        finished.put("100", task);

        calibrate(finished, new HashMap<>());
        assertEquals(0, endpoint.getInflightBatchCount());
    }

    @Test
    void calibrateSuccessOnlyRetiresSiblingWhileBatchMemberReconciles() {
        BatchItem reconciling = createBatchItem(101L, 500, 200);
        BatchItem sibling = createBatchItem(102L, 300, 100);
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 7L, 100, List.of(reconciling, sibling));
        endpoint.tryProtectBatchMember(7L, 101L);

        TaskInfo siblingSuccess = new TaskInfo();
        siblingSuccess.setBatchId(7L);
        siblingSuccess.setRequestId(102L);
        siblingSuccess.setErrorCode(0);
        calibrate(Map.of("102", siblingSuccess), Map.of());

        assertEquals(1, endpoint.getInflightBatchCount(),
                "sibling success must not erase the reconciling batch member");
        assertEquals(1, endpoint.realPendingCount());

        TaskInfo ambiguousMemberSuccess = new TaskInfo();
        ambiguousMemberSuccess.setBatchId(7L);
        ambiguousMemberSuccess.setRequestId(101L);
        ambiguousMemberSuccess.setErrorCode(0);
        calibrate(Map.of("101", ambiguousMemberSuccess), Map.of());
        assertEquals(1, endpoint.getInflightBatchCount(),
                "ordinary success is not a reconciliation terminal");
        assertEquals(1, endpoint.realPendingCount());

        endpoint.releaseBatchMemberProtection(7L, 101L);
        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.realPendingCount());
    }

    @Test
    void protectedAndSiblingFailuresSettleFromOneWorkerSnapshot() {
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 7L, 100, List.of(
                createBatchItem(101L, 500, 200),
                createBatchItem(102L, 300, 100)));
        endpoint.tryProtectBatchMember(7L, 101L);

        TaskInfo protectedFailure = taskInfo(101L, 7L, null, 500, 40);
        TaskInfo siblingFailure = taskInfo(102L, 7L, null, 501, 50);
        calibrate(Map.of("101", protectedFailure, "102", siblingFailure), Map.of());

        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.realPendingCount());

        endpoint.releaseBatchMemberProtection(7L, 101L);
        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.realPendingCount());
        verify(endpointReporter, never()).reportBatchPredictedTimeMs(
                anyString(), anyString(), org.mockito.ArgumentMatchers.anyLong());

        endpoint.releaseBatchMemberProtection(7L, 101L);
        assertEquals(0, endpoint.realPendingCount());
    }

    // ---- estimated waiting time ----

    @Test
    void realWaitTimeMsZeroWhenNoInflight() {
        assertEquals(0, endpoint.realWaitTimeMs());
    }

    @Test
    void realWaitTimeMsPositiveWithInflight() {
        BatchItem item = createBatchItem(1L, 500, 200);
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 1L, 5000, List.of(item)); // 5s prediction

        long waitMs = endpoint.realWaitTimeMs();
        assertTrue(waitMs > 0, "Should have non-zero wait time with inflight batch");
        assertTrue(waitMs <= 5000, "Wait time should not exceed prediction");
    }

    @Test
    void realWaitTimeMsDecreasesOverTime() throws InterruptedException {
        BatchItem item = createBatchItem(1L, 500, 200);
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 1L, 5000, List.of(item));

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
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 1L, 100, List.of(item));

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
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 1L, 100, List.of(item));

        int evicted = endpoint.evictExpiredBatches(60_000); // 60s TTL — fresh entry survives
        assertEquals(0, evicted);
        assertEquals(1, endpoint.getInflightBatchCount());
    }

    @Test
    void evictExpiredBatchesRetainsAckAmbiguousBatchUntilReconciled()
            throws InterruptedException {
        BatchItem item = createBatchItem(1L, 500, 200);
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 1L, 100, List.of(item));
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
    void batchLoadPublicationPreventsLowLoadSnapshotDuringCallbackHandoff()
            throws Exception {
        FlexlbConfig handoffConfig = new FlexlbConfig();
        configureBatch(handoffConfig, 100, 1, 0, null);
        setFormula(handoffConfig, "0");

        WorkerStatus status = new WorkerStatus();
        status.setIp("127.0.0.4");
        status.setPort(8080);
        status.setGrpcPort(8090);
        status.setRole(RoleType.PREFILL);

        long batchId = 8_001L;
        CountDownLatch callbackEntered = new CountDownLatch(1);
        CountDownLatch lifecycleMayBeEstablished = new CountDownLatch(1);
        CountDownLatch lifecycleEstablished = new CountDownLatch(1);
        CountDownLatch callbackMayReturn = new CountDownLatch(1);
        AtomicReference<Throwable> callbackFailure = new AtomicReference<>();
        DecisionGroupHandler handler = new DecisionGroupHandler() {
            @Override public void onExpired(BatchItem head) { }

            @Override
            public void onDecisionGroupAdmitted(
                    AdmittedDecisionGroup group,
                    DecisionGroupMetadata metadata) {
                callbackEntered.countDown();
                try {
                    assertTrue(lifecycleMayBeEstablished.await(2, TimeUnit.SECONDS));
                    for (AdmittedDecisionGroup.AdmittedItem member : group.members()) {
                        assertTrue(member.transferCapacityToEndpointLifecycle());
                    }
                    group.transferBatchCapacityToLifecycle(
                            batchId, 0L, group.requests());
                    for (AdmittedDecisionGroup.AdmittedItem member : group.members()) {
                        assertTrue(member.completeDeliveryHandoff());
                    }
                    lifecycleEstablished.countDown();
                    assertTrue(callbackMayReturn.await(2, TimeUnit.SECONDS));
                } catch (Throwable failure) {
                    callbackFailure.compareAndSet(null, failure);
                }
            }

            @Override public void onOfferFailure(BatchItem item, Throwable error) {
                callbackFailure.compareAndSet(null, error);
            }

            @Override public void onDeliveryFailure(BatchItem item, Throwable error) {
                callbackFailure.compareAndSet(null, error);
            }
        };

        AtomicReference<PrefillEndpoint> handoffEndpointRef = new AtomicReference<>();
        PrefillEndpoint handoffEndpoint = new PrefillEndpoint(
                status,
                handoffConfig,
                handler,
                TestCapacityAdmission.withEndpointBatchCapacity(
                        handoffEndpointRef::get),
                mock(BatchSchedulerReporter.class));
        handoffEndpointRef.set(handoffEndpoint);
        try {
            BatchItem item = createBatchItem(
                    handoffEndpoint, handoffConfig, 8_001L, 128, 0);
            assertTrue(handoffEndpoint.getBatcher().tryOffer(item));
            assertTrue(callbackEntered.await(2, TimeUnit.SECONDS));

            assertEquals(0, handoffEndpoint.getBatcher().queueSize(),
                    "the request must already have left ACTIVE before the callback runs");
            assertEquals(1, handoffEndpoint.getLocallyOwnedRequestCount(),
                    "callback-owned batch members remain visible to endpoint load accounting");
            assertEquals(0, handoffEndpoint.getInflightBatchCount(),
                    "the real BatchInflight has not been registered yet");
            assertEquals(Long.MAX_VALUE, handoffEndpoint.realPendingCount());
            assertEquals(Long.MAX_VALUE, handoffEndpoint.realWaitTimeMs());

            lifecycleMayBeEstablished.countDown();
            assertTrue(lifecycleEstablished.await(2, TimeUnit.SECONDS));
            assertEquals(1, handoffEndpoint.getInflightBatchCount());
            assertEquals(1, handoffEndpoint.getLocallyOwnedRequestCount());
            assertEquals(Long.MAX_VALUE, handoffEndpoint.realPendingCount(),
                    "the transition stays unpublished until the callback returns");
            assertEquals(Long.MAX_VALUE, handoffEndpoint.realWaitTimeMs());

            callbackMayReturn.countDown();
            long exactPending = awaitFinitePendingCount(handoffEndpoint);
            assertEquals(1, exactPending);
            assertEquals(1, handoffEndpoint.getLocallyOwnedRequestCount());
            assertEquals(0, handoffEndpoint.realWaitTimeMs());
            assertTrue(callbackFailure.get() == null,
                    () -> String.valueOf(callbackFailure.get()));

            handoffEndpoint.releaseBatch(batchId);
            assertEquals(0, handoffEndpoint.getLocallyOwnedRequestCount());
            assertEquals(0, handoffEndpoint.realPendingCount());
            assertEquals(0, handoffEndpoint.realWaitTimeMs());
        } finally {
            lifecycleMayBeEstablished.countDown();
            callbackMayReturn.countDown();
            handoffEndpoint.close();
        }
    }

    @Test
    void realPendingCountUnionsEngineTasksWithLocalLedger() {
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 1L, 100, List.of(
                createBatchItem(101L, 500, 0),
                createBatchItem(102L, 500, 0)));

        TaskInfo overlapping = taskInfo(102L, 1L, TaskPhase.RUNNING, 0, 0);
        TaskInfo untrackedOne = taskInfo(900L, 90L, TaskPhase.RUNNING, 0, 0);
        TaskInfo untrackedTwo = taskInfo(901L, 91L, TaskPhase.RECEIVED, 0, 0);
        TaskInfo duplicateUntracked = taskInfo(900L, 92L, TaskPhase.RUNNING, 0, 0);
        TaskInfo overlayOnly = taskInfo(999L, 99L, TaskPhase.PENDING, 0, 0);
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
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 1L, 100, List.of(createBatchItem(101L, 500, 0)));

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
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 1L, 100, List.of(createBatchItem(101L, 500, 0)));

        TaskInfo overlapping = taskInfo(101L, 1L, TaskPhase.RUNNING, 0, 0);
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
    void retirementOwnerCanReenterCloseFromSynchronousShutdownCallback()
            throws Exception {
        FlexlbConfig retirementConfig = new FlexlbConfig();
        configureBatch(retirementConfig, 100, 1, 0, null);
        SchedulingTestConfig.useNonBatchDispatcher(retirementConfig);

        WorkerStatus status = new WorkerStatus();
        status.setIp("127.0.0.5");
        status.setPort(8080);
        status.setGrpcPort(8090);
        status.setRole(RoleType.PREFILL);

        CountDownLatch admissionBlocked = new CountDownLatch(1);
        CountDownLatch reentrantCloseEntered = new CountDownLatch(1);
        CountDownLatch reentrantCloseReturned = new CountDownLatch(1);
        AtomicReference<Throwable> callbackFailure = new AtomicReference<>();
        AtomicReference<PrefillEndpoint> retirementEndpointRef = new AtomicReference<>();
        DecisionGroupHandler handler = new DecisionGroupHandler() {
            @Override public void onExpired(BatchItem head) { }

            @Override
            public void onDecisionGroupAdmitted(
                    AdmittedDecisionGroup group,
                    DecisionGroupMetadata metadata) {
                callbackFailure.compareAndSet(null,
                        new AssertionError("capacity-blocked request was admitted"));
            }

            @Override
            public void onOfferFailure(BatchItem item, Throwable error) {
                reentrantCloseEntered.countDown();
                retirementEndpointRef.get().close();
                reentrantCloseReturned.countDown();
            }

            @Override public void onDeliveryFailure(BatchItem item, Throwable error) {
                callbackFailure.compareAndSet(null, error);
            }
        };
        DeliveryCapacityAdmission blockedAdmission = item -> {
            admissionBlocked.countDown();
            return new DeliveryCapacityAdmission.CapacityUnavailable(
                    DeliveryCapacityAdmission.CapacityResource.PREFILL_REQUEST,
                    () -> false);
        };
        PrefillEndpoint retirementEndpoint = new PrefillEndpoint(
                status,
                retirementConfig,
                handler,
                blockedAdmission,
                mock(BatchSchedulerReporter.class));
        retirementEndpointRef.set(retirementEndpoint);
        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            BatchItem item = createBatchItem(
                    retirementEndpoint, retirementConfig, 8_101L, 128, 0);
            assertTrue(retirementEndpoint.getBatcher().tryOffer(item));
            assertTrue(admissionBlocked.await(2, TimeUnit.SECONDS));

            Future<?> outerClose = executor.submit(retirementEndpoint::close);
            outerClose.get(2, TimeUnit.SECONDS);
            assertTrue(reentrantCloseEntered.await(1, TimeUnit.SECONDS));
            assertTrue(reentrantCloseReturned.await(1, TimeUnit.SECONDS),
                    "retirement-owner reentry must return instead of waiting on itself");
            assertTrue(callbackFailure.get() == null,
                    () -> String.valueOf(callbackFailure.get()));

            PrefillEndpoint.RequestCapacityReservationAcquisition rejected =
                    retirementEndpoint.acquireRequestCapacityReservation(
                            8_102L, 0L, 1);
            assertEquals(
                    PrefillEndpoint.RequestCapacityReservationStatus.ENDPOINT_RETIRED,
                    rejected.status());
        } finally {
            retirementEndpoint.close();
            executor.shutdownNow();
        }
    }

    @Test
    void admittedCallbackCanCloseEndpointBeforeItsHandoffPermitIsReleased()
            throws Exception {
        FlexlbConfig retirementConfig = new FlexlbConfig();
        configureBatch(retirementConfig, 100, 1, 0, null);
        setFormula(retirementConfig, "0");

        WorkerStatus status = new WorkerStatus();
        status.setIp("127.0.0.6");
        status.setPort(8080);
        status.setGrpcPort(8090);
        status.setRole(RoleType.PREFILL);

        CountDownLatch callbackCloseReturned = new CountDownLatch(1);
        CountDownLatch callbackResolved = new CountDownLatch(1);
        CountDownLatch callbackMayReturn = new CountDownLatch(1);
        AtomicReference<Throwable> callbackFailure = new AtomicReference<>();
        AtomicReference<PrefillEndpoint> retirementEndpointRef = new AtomicReference<>();
        DecisionGroupHandler handler = new DecisionGroupHandler() {
            @Override public void onExpired(BatchItem head) { }

            @Override
            public void onDecisionGroupAdmitted(
                    AdmittedDecisionGroup group,
                    DecisionGroupMetadata metadata) {
                boolean restoreInterrupt = false;
                try {
                    retirementEndpointRef.get().close();
                    callbackCloseReturned.countDown();
                    restoreInterrupt = Thread.interrupted();
                    TestCapacityAdmission.complete(group);
                    callbackResolved.countDown();
                    assertTrue(callbackMayReturn.await(2, TimeUnit.SECONDS));
                } catch (Throwable failure) {
                    callbackFailure.compareAndSet(null, failure);
                } finally {
                    if (restoreInterrupt) {
                        Thread.currentThread().interrupt();
                    }
                }
            }

            @Override public void onOfferFailure(BatchItem item, Throwable error) {
                callbackFailure.compareAndSet(null, error);
            }

            @Override public void onDeliveryFailure(BatchItem item, Throwable error) {
                callbackFailure.compareAndSet(null, error);
            }
        };
        PrefillEndpoint retirementEndpoint = new PrefillEndpoint(
                status,
                retirementConfig,
                handler,
                TestCapacityAdmission.withEndpointBatchCapacity(
                        retirementEndpointRef::get),
                mock(BatchSchedulerReporter.class));
        retirementEndpointRef.set(retirementEndpoint);
        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            BatchItem admitted = createBatchItem(
                    retirementEndpoint, retirementConfig, 8_201L, 128, 0);
            assertTrue(retirementEndpoint.getBatcher().tryOffer(admitted));
            assertTrue(callbackCloseReturned.await(2, TimeUnit.SECONDS),
                    "close inside an admitted callback must return synchronously");
            assertTrue(callbackResolved.await(2, TimeUnit.SECONDS));

            Future<?> concurrentClose = executor.submit(retirementEndpoint::close);
            assertThrows(TimeoutException.class,
                    () -> concurrentClose.get(100, TimeUnit.MILLISECONDS),
                    "a non-owner close must wait for the callback's handoff permit");

            callbackMayReturn.countDown();
            concurrentClose.get(2, TimeUnit.SECONDS);
            assertTrue(callbackFailure.get() == null,
                    () -> String.valueOf(callbackFailure.get()));

            PrefillEndpoint.QueueBatchSlotAdmissionFailed rejected = assertInstanceOf(
                    PrefillEndpoint.QueueBatchSlotAdmissionFailed.class,
                    retirementEndpoint.tryReserveQueueBatchSlot(admitted, 1));
            assertInstanceOf(EndpointGenerationRetiredException.class, rejected.cause());
        } finally {
            callbackMayReturn.countDown();
            retirementEndpoint.close();
            executor.shutdownNow();
        }
    }

    @Test
    void closePreservesRegisteredLifecycleAndRejectsNewBatchReservations() {
        BatchItem item = createBatchItem(1L, 500, 200);
        TestCapacityAdmission.registerQueueBatchLifecycle(
                endpoint,
                1L,
                100,
                List.of(item));
        assertEquals(1, endpoint.getQueueBatchCapacityUsage());

        endpoint.close();
        assertEquals(1, endpoint.getInflightBatchCount(),
                "close must not erase an Engine-owned batch lifecycle");
        assertEquals(1, endpoint.getQueueBatchCapacityUsage());
        PrefillEndpoint.QueueBatchSlotAdmissionFailed retired = assertInstanceOf(
                PrefillEndpoint.QueueBatchSlotAdmissionFailed.class,
                endpoint.tryReserveQueueBatchSlot(item, 2));
        assertInstanceOf(EndpointGenerationRetiredException.class, retired.cause(),
                "retirement is a delivery failure, not silent request-ownership loss");

        endpoint.releaseBatch(1L);
        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.getQueueBatchCapacityUsage());
        endpoint.close();
        assertEquals(0, endpoint.getQueueBatchCapacityUsage());
    }

    @Test
    void closeRetiresDirectAccountingAndPreservesCommittedQueueRoute() {
        endpoint.registerDirectRequest(100L, 100);
        assertTrue(TestCapacityAdmission.commitRouteRequest(
                endpoint, 200L, 200, 1));
        assertEquals(2, endpoint.getIndividuallyTrackedRequestCount());
        assertEquals(1, endpoint.getQueueRouteCapacityUsage());

        endpoint.close();

        assertEquals(1, endpoint.getIndividuallyTrackedRequestCount(),
                "the retired generation must discard DIRECT load accounting");
        assertEquals(1, endpoint.getQueueRouteCapacityUsage(),
                "committed QUEUE_ROUTE ownership must outlive endpoint selection");
        assertFalse(endpoint.releaseRequest(100L));
        assertTrue(endpoint.releaseRequest(200L));
        assertEquals(0, endpoint.getIndividuallyTrackedRequestCount());
        assertEquals(0, endpoint.getQueueRouteCapacityUsage());
    }

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
        configureBatch(slowConfig, 100, 100, fixedWaitMs, null);
        return new PrefillEndpoint(
                status,
                slowConfig,
                noopHandler(),
                TestCapacityAdmission.alwaysAvailable(),
                mock(BatchSchedulerReporter.class));
    }

    private BatchItem createPriorityBatchItem(long requestId, int priority) {
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

    private static void reportSuccessfulBatchMember(
            PrefillEndpoint target,
            long batchId,
            long requestId,
            long executionTimeMs) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setFinishedTaskInfo(Map.of(
                Long.toString(requestId),
                taskInfo(requestId, batchId, null, 0, executionTimeMs)));
        response.setRunningTaskInfo(Map.of());
        target.onWorkerStatusUpdate(target.getStatus(), response);
    }

    private static PrefillEndpoint createLearningEndpoint() {
        WorkerStatus status = new WorkerStatus();
        status.setIp("127.0.0.8");
        status.setPort(8080);
        status.setGrpcPort(8090);
        status.setRole(RoleType.PREFILL);

        FlexlbConfig learningConfig = new FlexlbConfig();
        configureBatch(
                learningConfig,
                100,
                learningConfig.fixedWindowDecision().getMaxRequests(),
                300,
                null);
        learningConfig.getRouter().getRoles().getPrefill()
                .setExecutionTimeEstimator(
                        new RoutingConfig.LearningEstimatorConfig());
        return new PrefillEndpoint(
                status,
                learningConfig,
                noopHandler(),
                TestCapacityAdmission.alwaysAvailable(),
                mock(BatchSchedulerReporter.class));
    }

    private static long awaitFinitePendingCount(PrefillEndpoint target) {
        long deadlineNanos = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
        long pending;
        do {
            pending = target.realPendingCount();
            if (pending != Long.MAX_VALUE) {
                return pending;
            }
            Thread.onSpinWait();
        } while (System.nanoTime() < deadlineNanos);
        throw new AssertionError("batch load publication did not close");
    }

    private BatchItem createBatchItem(long requestId, long seqLen, long hitCacheLen) {
        return createBatchItem(endpoint, requestId, seqLen, hitCacheLen);
    }

    private static BatchItem createBatchItem(PrefillEndpoint owner,
                                             long requestId,
                                             long seqLen,
                                             long hitCacheLen) {
        return createBatchItem(
                owner, new FlexlbConfig(), requestId, seqLen, hitCacheLen);
    }

    private static BatchItem createBatchItem(
            PrefillEndpoint owner,
            FlexlbConfig requestConfig,
            long requestId,
            long seqLen,
            long hitCacheLen) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);

        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setConfig(requestConfig);

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
        target.queueScheduler().getCapacity()
                .setMaxWaitingRequestsPerPrefillWorker(maxWaiting);
        target.fixedWindowDecision().setMaxRequests(maxRequests);
        target.fixedWindowDecision().setMaxCollectionWaitMs(maxCollectionWaitMs);
        target.batchDispatcher().setMaxInflightBatchesPerPrefillWorker(maxInflightBatches);
    }

    private static void setFormula(FlexlbConfig target, String expression) {
        RoutingConfig.FormulaEstimatorConfig estimator =
                (RoutingConfig.FormulaEstimatorConfig) target.getRouter().getRoles()
                        .getPrefill().getExecutionTimeEstimator();
        estimator.setExpression(expression);
    }

    private static TaskInfo priorityCanceledTask(long requestId, long batchId) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setBatchId(batchId);
        task.setErrorCode(8429);
        task.setErrorMessage("priority preempted");
        task.setPriorityPreemptionProgress(PriorityPreemptionProgress.CANCELED);
        return task;
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

    private static DecisionGroupHandler noopHandler() {
        return new DecisionGroupHandler() {
            @Override public void onExpired(BatchItem head) {}
            @Override public void onDecisionGroupAdmitted(
                    AdmittedDecisionGroup group, DecisionGroupMetadata meta) {
                TestCapacityAdmission.complete(group);
            }
            @Override public void onOfferFailure(BatchItem item, Throwable error) {}
            @Override public void onDeliveryFailure(BatchItem item, Throwable error) {}
        };
    }

}
