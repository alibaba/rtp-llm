package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.scheduler.RequestRegistry;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.DispatcherConfig;
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
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
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
    private EndpointTestSupport.TestRequestRuntime requestRuntime;

    @BeforeEach
    void setUp() {
        WorkerStatus status = EndpointTestSupport.workerStatus(
                RoleType.PREFILL, "127.0.0.1", 8080, 8090);

        config = new FlexlbConfig();
        configureBatch(config, 100, config.fixedWindowDecision().getMaxRequests(), 300, null);
        setFormula(config, "10 + 0.1*sum(computeTokens) + 5*batchSize");

        endpointReporter = mock(BatchSchedulerReporter.class);
        requestRuntime = EndpointTestSupport.requestRuntime();
        endpoint = new PrefillEndpoint(
                status,
                config,
                EndpointTestSupport.routeStrategy(requestRuntime),
                requestRuntime.events(),
                endpointReporter);
        endpoint.startGeneration();
    }

    @AfterEach
    void tearDown() {
        endpoint.close();
    }

    // ---- batch commit / release ----

    @Test
    void commitBatchIncreasesInflightCount() {
        assertEquals(0, endpoint.getInflightBatchCount());

        ScheduledRequest item = createScheduledRequest(1L, 500, 200);
        registerBatch(endpoint, 1L, 100, List.of(item));

        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1,
                endpoint.captureRouteProjectionInputs().work().batches().size());
        assertEquals(1, endpoint.admissionPendingRequestCount());
    }

    @Test
    void releaseBatchDecreasesInflightCount() {
        ScheduledRequest item = createScheduledRequest(1L, 500, 200);
        registerBatch(endpoint, 1L, 100, List.of(item));
        assertTrue(endpoint.releaseCommittedItem(item));

        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0,
                endpoint.captureRouteProjectionInputs().work().batches().size());
    }

    @Test
    void releaseBatchRetainsOnlyProtectedMembers() {
        ScheduledRequest protectedItem = createScheduledRequest(101L, 500, 200);
        ScheduledRequest sibling = createScheduledRequest(102L, 300, 100);
        registerBatch(endpoint, 7L, 100, List.of(protectedItem, sibling));
        PrefillState.Protection protection =
                endpoint.acquireBatchMemberProtection(7L, protectedItem);
        assertNotNull(protection);

        assertTrue(endpoint.releaseCommittedItem(sibling));
        assertTrue(endpoint.releaseCommittedItem(protectedItem));

        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1,
                endpoint.captureRouteProjectionInputs().work().batches().size(),
                "partial repack must retain the registered group slot");
        assertEquals(1, endpoint.admissionPendingRequestCount(),
                "a delivery failure must not reopen capacity owned by an Engine fence");

        endpoint.releaseEngineFenceProtection(protection);
        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0,
                endpoint.captureRouteProjectionInputs().work().batches().size(),
                "the last member releases the registered group slot");
        assertEquals(0, endpoint.admissionPendingRequestCount());
    }

    @Test
    void commitMultipleBatches() {
        ScheduledRequest item1 = createScheduledRequest(1L, 500, 200);
        ScheduledRequest item2 = createScheduledRequest(2L, 300, 100);
        ScheduledRequest item3 = createScheduledRequest(3L, 400, 0);

        registerBatch(endpoint, 1L, 100, List.of(item1, item2));
        registerBatch(endpoint, 2L, 50, List.of(item3));

        assertEquals(2, endpoint.getInflightBatchCount());
        assertEquals(3, endpoint.admissionPendingRequestCount());
    }

    // ---- repack batch ----

    @Test
    void repackBatchRemovesFailedRequests() {
        ScheduledRequest item1 = createScheduledRequest(1L, 500, 200);
        ScheduledRequest item2 = createScheduledRequest(2L, 300, 100);
        registerBatch(endpoint, 1L, 100, List.of(item1, item2));

        assertTrue(endpoint.releaseCommittedItem(item2));
        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1,
                endpoint.captureRouteProjectionInputs().work().batches().size());
        assertEquals(1, endpoint.admissionPendingRequestCount());
    }

    @Test
    void repackBatchAllFailedReturnsNull() {
        ScheduledRequest item1 = createScheduledRequest(1L, 500, 200);
        registerBatch(endpoint, 1L, 100, List.of(item1));

        assertTrue(endpoint.releaseCommittedItem(item1));
        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0,
                endpoint.captureRouteProjectionInputs().work().batches().size());
    }

    @Test
    void failedRepackPredictionSettlesMembershipAndPublishesUnknownWork() {
        PrefillEndpoint invalidPredictorEndpoint = newEndpointWithFormula("-1");
        try {
            ScheduledRequest survivor = createScheduledRequest(
                    invalidPredictorEndpoint, 1L, 500L, 200L);
            ScheduledRequest failed = createScheduledRequest(
                    invalidPredictorEndpoint, 2L, 300L, 100L);
            registerBatch(
                    invalidPredictorEndpoint,
                    1L,
                    100L,
                    List.of(survivor, failed));

            assertDoesNotThrow(() -> reportSuccessfulBatchMember(
                    invalidPredictorEndpoint, 1L, 2L, 30L));

            WorkSnapshot snapshot = invalidPredictorEndpoint
                    .captureRouteProjectionInputs().work();
            assertEquals(1, invalidPredictorEndpoint.admissionPendingRequestCount(),
                    "membership settlement must not depend on prediction");
            assertEquals(List.of(1L), snapshot.batches().getFirst().requestIds());
            assertTrue(snapshot.batches().getFirst().remainingWorkMs().isEmpty());
            assertTrue(snapshot.hasUnknownWork());
            assertTrue(invalidPredictorEndpoint.getLoadMetric().isEmpty(),
                    "monitoring must omit unknown repacked work");

            reportSuccessfulBatchMember(
                    invalidPredictorEndpoint, 1L, 1L, 40L);
            assertEquals(0, invalidPredictorEndpoint.getInflightBatchCount(),
                    "unknown duration must not block later lifecycle settlement");
            assertEquals(0, invalidPredictorEndpoint.admissionPendingRequestCount());
        } finally {
            invalidPredictorEndpoint.close();
        }
    }

    // ---- calibrate ----

    @Test
    void calibrateRemovesBatchOnSuccess() {
        ScheduledRequest item = createScheduledRequest(1L, 500, 200);
        registerBatch(endpoint, 1L, 100, List.of(item));

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
        registerBatch(endpoint, 9L, 100, List.of(createScheduledRequest(9L, 500, 200)));
        doThrow(new IllegalStateException("metrics unavailable"))
                .when(endpointReporter)
                .reportBatchPredictedTimeMs("PREFILL", "127.0.0.1", 100);

        TaskInfo finished = taskInfo(9L, 9L, null, 0, 125);
        assertDoesNotThrow(() -> calibrate(Map.of("9", finished), Map.of()));

        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.admissionPendingRequestCount());
        verify(endpointReporter).reportBatchActualTimeMs("PREFILL", "127.0.0.1", 125);
        verify(endpointReporter).reportBatchPredictGapMs("PREFILL", "127.0.0.1", 25);
    }

    @Test
    void calibrateRepacksOnPartialFailure() {
        ScheduledRequest item1 = createScheduledRequest(1L, 500, 200);
        ScheduledRequest item2 = createScheduledRequest(2L, 300, 100);
        registerBatch(endpoint, 1L, 100, List.of(item1, item2));

        Map<String, TaskInfo> finished = new HashMap<>();
        TaskInfo failedTask = new TaskInfo();
        failedTask.setRequestId(2L);
        failedTask.setBatchId(1L);
        failedTask.setErrorCode(500);
        failedTask.setErrorMessage("engine error");
        finished.put("2", failedTask);

        calibrate(finished, Map.of());

        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.admissionPendingRequestCount());
    }

    @Test
    void calibrateKeepsBatchInflightUntilEveryMemberFinishes() {
        ScheduledRequest shortItem = createScheduledRequest(1L, 500, 200);
        ScheduledRequest longItem = createScheduledRequest(2L, 10_000, 0);
        registerBatch(endpoint, 1L, 2_000, List.of(shortItem, longItem));

        TaskInfo finishedShort = taskInfo(1L, 1L, null, 0, 40);
        TaskInfo runningLong = taskInfo(2L, 1L, TaskPhase.RUNNING, 0, 0);
        calibrate(Map.of("1", finishedShort), Map.of("2", runningLong));

        assertEquals(1, endpoint.getInflightBatchCount(),
                "one finished member must not release the whole batch");
        assertEquals(1, endpoint.admissionPendingRequestCount(),
                "the still-running long member must remain in Master accounting");

        TaskInfo finishedLong = taskInfo(2L, 1L, null, 0, 1_900);
        calibrate(Map.of("2", finishedLong), Map.of());

        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.admissionPendingRequestCount());
    }

    @Test
    void calibrateMixedTerminalMembersKeepsOnlyRunningSurvivor() {
        ScheduledRequest succeeded = createScheduledRequest(1L, 500, 200);
        ScheduledRequest failed = createScheduledRequest(2L, 300, 100);
        ScheduledRequest running = createScheduledRequest(3L, 10_000, 0);
        registerBatch(endpoint, 1L, 2_000, List.of(succeeded, failed, running));

        TaskInfo success = taskInfo(1L, 1L, null, 0, 40);
        TaskInfo failure = taskInfo(2L, 1L, null, 500, 50);
        TaskInfo runningTask = taskInfo(3L, 1L, TaskPhase.RUNNING, 0, 0);
        calibrate(Map.of("1", success, "2", failure), Map.of("3", runningTask));

        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.admissionPendingRequestCount());

        // WorkerStatus may repeat a terminal observation in adjacent snapshots.
        // Repeating it must not decrement the survivor count again.
        calibrate(Map.of("1", success), Map.of("3", runningTask));
        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.admissionPendingRequestCount());
    }

    @Test
    void calibrateAllFailuresClearsBatchIdempotentlyWithoutCompletionMetrics() {
        registerBatch(endpoint, 1L, 2_000, List.of(
                createScheduledRequest(1L, 500, 200),
                createScheduledRequest(2L, 10_000, 0)));

        TaskInfo firstFailure = taskInfo(1L, 1L, null, 500, 40);
        TaskInfo secondFailure = taskInfo(2L, 1L, null, 501, 50);
        calibrate(Map.of("1", firstFailure, "2", secondFailure), Map.of());

        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.admissionPendingRequestCount());
        verify(endpointReporter, never()).reportBatchPredictedTimeMs(
                anyString(), anyString(), org.mockito.ArgumentMatchers.anyLong());

        calibrate(Map.of("1", firstFailure, "2", secondFailure), Map.of());
        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.admissionPendingRequestCount(),
                "repeated failure deltas must not decrement the ledger twice");
    }

    @Test
    void repeatedSuccessfulTerminalReportsCompletionExactlyOnce() {
        registerBatch(endpoint, 1L, 100, List.of(createScheduledRequest(1L, 500, 200)));
        TaskInfo success = taskInfo(1L, 1L, null, 0, 40);

        calibrate(Map.of("1", success), Map.of());
        calibrate(Map.of("1", success), Map.of());

        assertEquals(0, endpoint.admissionPendingRequestCount());
        verify(endpointReporter).reportBatchPredictedTimeMs(
                anyString(), anyString(), org.mockito.ArgumentMatchers.anyLong());
        verify(endpointReporter).reportBatchActualTimeMs(
                anyString(), anyString(), org.mockito.ArgumentMatchers.anyLong());
        verify(endpointReporter).reportBatchPredictGapMs(
                anyString(), anyString(), org.mockito.ArgumentMatchers.anyLong());
    }

    @Test
    void batchInflightReanchorsAcrossRunningQueuedRunning() {
        registerBatch(endpoint, 1L, 5_000,
                List.of(createScheduledRequest(1L, 500, 0)));
        assertEquals(WorkSnapshot.Phase.COMMITTED,
                onlyBatchWork(endpoint).phase());

        calibrate(Map.of(), Map.of(
                "1", taskInfo(1L, 1L, TaskPhase.RUNNING, 0, 0)));
        assertEquals(WorkSnapshot.Phase.ENGINE_RUNNING,
                onlyBatchWork(endpoint).phase());

        calibrate(Map.of(), Map.of(
                "1", taskInfo(1L, 1L, TaskPhase.PENDING, 0, 0)));
        assertEquals(WorkSnapshot.Phase.ENGINE_QUEUED,
                onlyBatchWork(endpoint).phase());

        calibrate(Map.of(), Map.of(
                "1", taskInfo(1L, 1L, TaskPhase.RUNNING, 0, 0)));
        assertEquals(WorkSnapshot.Phase.ENGINE_RUNNING,
                onlyBatchWork(endpoint).phase());
    }

    @Test
    void batchInflightMaxAgeMeasuresTimeSinceLatestActivity()
            throws InterruptedException {
        // The refactor unified inflight age tracking onto a single
        // last-observation clock. A recent Engine observation both keeps the
        // canonical owner alive (eviction) and resets the reported max age; the
        // age then grows with the time elapsed since that last observation.
        registerBatch(endpoint, 1L, 5_000,
                List.of(createScheduledRequest(1L, 500, 0)));
        Thread.sleep(20);
        calibrate(Map.of(), Map.of(
                "1", taskInfo(1L, 1L, TaskPhase.RUNNING, 0, 0)));

        assertEquals(0, endpoint.evictExpiredBatches(5),
                "recent Engine activity keeps the canonical owner live");
        Thread.sleep(15);
        endpoint.reportBatchMetrics(endpointReporter);
        verify(endpointReporter).reportInflightMaxAgeMs(
                anyString(), anyString(),
                org.mockito.ArgumentMatchers.longThat(age -> age >= 10));
    }

    @Test
    void inflightMaxAgeMetricTracksTimeSinceLatestObservation()
            throws InterruptedException {
        // Reported max age measures staleness (time since the batch was last
        // observed), not wall-clock time since creation: a fresh RUNNING
        // observation resets it, after which it grows with elapsed idle time.
        registerBatch(endpoint, 1L, 5_000, List.of(createScheduledRequest(1L, 500, 0)));
        calibrate(Map.of(), Map.of(
                "1", taskInfo(1L, 1L, TaskPhase.RUNNING, 0, 0)));
        Thread.sleep(30);

        endpoint.reportBatchMetrics(endpointReporter);

        verify(endpointReporter).reportInflightMaxAgeMs(
                anyString(), anyString(), org.mockito.ArgumentMatchers.longThat(age -> age >= 20));
    }

    @Test
    void runningObservationRefreshesBatchInactivityTtl() throws InterruptedException {
        ScheduledRequest longItem = createScheduledRequest(1L, 10_000, 0);
        registerBatch(endpoint, 1L, 2_000, List.of(longItem));

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
        registerBatch(endpoint, 1L, 2_000, List.of(createScheduledRequest(1L, 10_000, 0)));
        Thread.sleep(10);

        TaskInfo foreign = taskInfo(999L, 1L, TaskPhase.RUNNING, 0, 0);
        calibrate(Map.of(), Map.of("999", foreign));

        assertEquals(1, endpoint.evictExpiredBatches(1));
        assertEquals(0, endpoint.getInflightBatchCount());
    }

    @Test
    void partialCompletionKeepsFixedWindowMaxInflightGateClosed() throws Exception {
        ScheduledRequest shortItem = createScheduledRequest(101L, 500, 200);
        ScheduledRequest longItem = createScheduledRequest(102L, 10_000, 0);
        registerBatch(endpoint, 700L, 2_000,
                List.of(shortItem, longItem));
        assertFalse(endpoint.batchAdmissionAvailability(1).isAvailable());

        calibrate(
                Map.of("101", taskInfo(101L, 700L, null, 0, 40)),
                Map.of("102", taskInfo(
                        102L, 700L, TaskPhase.RUNNING, 0, 0)));

        assertEquals(1, endpoint.getInflightBatchCount());
        assertFalse(endpoint.batchAdmissionAvailability(1).isAvailable(),
                "a short member finishing must not reopen maxInflight=1 while its long sibling runs");

        calibrate(
                Map.of("102", taskInfo(102L, 700L, null, 0, 1_900)),
                Map.of());

        assertTrue(endpoint.batchAdmissionAvailability(1).isAvailable(),
                "the final member must reopen the exact batch availability source");
    }

    @Test
    void calibrateHandlesTaskWithNoBatchId() {
        ScheduledRequest item = createScheduledRequest(1L, 500, 200);
        registerBatch(endpoint, 1L, 100, List.of(item));

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
    void calibrateMissingBatchIdDoesNotRetireRealBatchMember() {
        registerBatch(endpoint, 700L, 100, List.of(createScheduledRequest(101L, 500, 200)));

        calibrate(Map.of("101", priorityCanceledTask(101L, -1L)), Map.of());

        // A terminal that carries no valid batch id (batchId <= 0) can no longer
        // be attributed to a committed batch member: the canonical ledger only
        // settles a batch member from a terminal that names the exact batch id.
        // The member therefore stays committed until an exact-batch terminal,
        // protection release, or TTL eviction reconciles it.
        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.admissionPendingRequestCount());
    }

    @Test
    void calibrateMissingBatchIdRemovesDirectRequestLedgerEntry() {
        registerDirect(endpoint, 101L, 100L);
        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.getIndividuallyTrackedRequestCount());

        TaskInfo finished = new TaskInfo();
        finished.setRequestId(101L);
        finished.setBatchId(-1L);
        finished.setErrorCode(0);
        calibrate(Map.of("101", finished), Map.of());

        assertEquals(0, endpoint.getIndividuallyTrackedRequestCount());
        assertEquals(0, endpoint.admissionPendingRequestCount());
    }

    @Test
    void directRegistrationCanRollbackFromAsyncCompletionThread()
            throws Exception {
        PrefillState.DirectRegistration registration =
                EndpointTestSupport.registerDirect(endpoint, 102L, 100L);
        assertEquals(1, endpoint.admissionPendingRequestCount());

        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            executor.submit(registration::close).get(5, TimeUnit.SECONDS);
        } finally {
            executor.shutdownNow();
        }

        assertEquals(0, endpoint.admissionPendingRequestCount());
    }

    @Test
    void calibrateMissingBatchIdLeavesMembersUntilExactBatchTerminal() {
        registerBatch(endpoint, 700L, 100, List.of(
                createScheduledRequest(101L, 500, 200),
                createScheduledRequest(102L, 300, 100)));

        calibrate(Map.of("101", priorityCanceledTask(101L, -1L)), Map.of());

        // A missing-batch-id terminal cannot be attributed to the batch, so no
        // member is retired: both members remain committed.
        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(2, endpoint.admissionPendingRequestCount(),
                "a terminal without a valid batch id retires no batch member");

        TaskInfo survivingSuccess = new TaskInfo();
        survivingSuccess.setRequestId(102L);
        survivingSuccess.setBatchId(700L);
        survivingSuccess.setErrorCode(0);
        calibrate(Map.of("102", survivingSuccess), Map.of());
        // The exact-batch terminal retires only its own member; member 101,
        // whose only terminal named no batch id, stays with the original batch.
        assertEquals(1, endpoint.getInflightBatchCount(),
                "the exact-batch terminal retires only its own member");
        assertEquals(1, endpoint.admissionPendingRequestCount());
    }

    @Test
    void directRequestIdMatchingQueueBatchIdDoesNotOverwriteEitherLifecycle() {
        // DIRECT request 101 and QUEUE batch 101 live in different ledgers.
        // Completing the DIRECT request must not erase QUEUE member 201.
        registerBatch(endpoint, 101L, 100, List.of(createScheduledRequest(201L, 500, 200)));
        registerDirect(endpoint, 101L, 100L);
        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1,
                endpoint.captureRouteProjectionInputs().work().batches().size());
        assertEquals(1, endpoint.getIndividuallyTrackedRequestCount());

        calibrate(Map.of("101", priorityCanceledTask(101L, -1L)), Map.of());

        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1,
                endpoint.captureRouteProjectionInputs().work().batches().size());
        assertEquals(0, endpoint.getIndividuallyTrackedRequestCount());
        assertEquals(1, endpoint.admissionPendingRequestCount());

        TaskInfo foreignBatchMemberSuccess = new TaskInfo();
        foreignBatchMemberSuccess.setRequestId(201L);
        foreignBatchMemberSuccess.setBatchId(101L);
        foreignBatchMemberSuccess.setErrorCode(0);
        calibrate(Map.of("201", foreignBatchMemberSuccess), Map.of());
        assertEquals(0, endpoint.getInflightBatchCount(),
                "the matching QUEUE batch id must survive until its own member finishes");
        assertEquals(0,
                endpoint.captureRouteProjectionInputs().work().batches().size());
    }

    @Test
    void calibrateMissingBatchIdDoesNotGuessAcrossDuplicateLiveBatches() {
        ScheduledRequest first = createScheduledRequest(101L, 500, 200);
        ScheduledRequest reusedRequestId = createScheduledRequest(101L, 300, 100);
        registerBatch(endpoint, 700L, 100, List.of(first));

        PrefillState.ReservationResult<PrefillState.BatchReservation> duplicate =
                endpoint.reserveBatch(reusedRequestId, 701L, 10);

        assertFalse(duplicate.status()
                        == PrefillState.CapacityStatus.ACQUIRED,
                "the canonical ledger rejects ambiguous duplicate live owners");
        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.admissionPendingRequestCount());
    }

    @Test
    void calibrateMissingBatchIdPreservesProtectedBatchMember() {
        ScheduledRequest protectedItem = createScheduledRequest(101L, 500, 200);
        ScheduledRequest sibling = createScheduledRequest(102L, 300, 100);
        registerBatch(endpoint, 700L, 100,
                List.of(protectedItem, sibling));
        PrefillState.Protection protection =
                endpoint.acquireBatchMemberProtection(700L, protectedItem);
        assertNotNull(protection);

        // Missing-batch-id terminals cannot be attributed to the batch, so
        // neither the sibling nor the protected member is retired.
        calibrate(Map.of("102", priorityCanceledTask(102L, -1L)), Map.of());
        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(2, endpoint.admissionPendingRequestCount());

        TaskInfo canceled = priorityCanceledTask(101L, -1L);
        calibrate(Map.of("101", canceled), Map.of());
        assertEquals(1, endpoint.getInflightBatchCount(),
                "generic endpoint calibration must not bypass the exact-batch reducer");
        assertEquals(2, endpoint.admissionPendingRequestCount());

        // The protection never captured a deferred terminal (both missing-batch
        // -id terminals were dropped), so releasing it settles nothing.
        endpoint.releaseEngineFenceProtection(protection);
        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(2, endpoint.admissionPendingRequestCount());
    }

    @Test
    void authoritativeWorkerTerminalSettlesProtectedBatchMemberImmediately() {
        ScheduledRequest protectedItem = createScheduledRequest(101L, 500, 200);
        registerBatch(
                endpoint,
                700L,
                100,
                List.of(protectedItem));
        PrefillState.Protection protection =
                endpoint.acquireBatchMemberProtection(700L, protectedItem);
        assertNotNull(protection);

        calibrate(Map.of(
                "101", taskInfo(101L, 700L, null, 0, 10)), Map.of());

        // A WorkerStatus terminal is an authoritative Engine reducer: it settles
        // the exact-batch member immediately and invalidates the protection.
        // Protection only fences external/TTL cleanup, so it no longer defers a
        // canonical Engine terminal.
        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0,
                endpoint.captureRouteProjectionInputs().work().batches().size());
        assertEquals(0, endpoint.admissionPendingRequestCount());

        // Releasing the already-invalidated protection is a graceful no-op.
        endpoint.releaseEngineFenceProtection(protection);
        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0,
                endpoint.captureRouteProjectionInputs().work().batches().size());
        assertEquals(0, endpoint.admissionPendingRequestCount());
    }

    @Test
    void authoritativeWorkerTerminalAppliesLearningImmediatelyDespiteProtection() {
        PrefillEndpoint learningEndpoint = createLearningEndpoint();
        try {
            PrefillTimePredictor.Evaluator initialEvaluator =
                    learningEndpoint.getPredictor().evaluator();
            // LearningPredictor publishes one model revision per four valid
            // completions. Seed three unchanged samples first.
            for (int sample = 1; sample <= 3; sample++) {
                long batchId = 8_000L + sample;
                long requestId = 9_000L + sample;
                registerBatch(
                        learningEndpoint,
                        batchId,
                        100L,
                        List.of(createScheduledRequest(
                                learningEndpoint, requestId, 500L, 200L)));
                reportSuccessfulBatchMember(
                        learningEndpoint, batchId, requestId, 100L + sample);
            }
            assertSame(initialEvaluator,
                    learningEndpoint.getPredictor().evaluator());

            long batchId = 8_004L;
            long requestId = 9_004L;
            ScheduledRequest protectedItem = createScheduledRequest(
                    learningEndpoint, requestId, 500L, 200L);
            registerBatch(
                    learningEndpoint,
                    batchId,
                    100L,
                    List.of(protectedItem));
            PrefillState.Protection protection =
                    learningEndpoint.acquireBatchMemberProtection(
                            batchId, protectedItem);
            assertNotNull(protection);

            // The WorkerStatus terminal is authoritative: it settles the member
            // and feeds the predictor immediately, without waiting for the
            // protection to end. The fourth valid sample publishes a new model
            // revision at report time.
            reportSuccessfulBatchMember(
                    learningEndpoint, batchId, requestId, 104L);
            assertNotSame(initialEvaluator,
                    learningEndpoint.getPredictor().evaluator(),
                    "the authoritative terminal reaches predictor learning immediately");
            assertEquals(0, learningEndpoint.getInflightBatchCount());

            // Releasing the already-invalidated protection changes nothing more.
            learningEndpoint.releaseEngineFenceProtection(protection);
            assertNotSame(initialEvaluator,
                    learningEndpoint.getPredictor().evaluator());
            assertEquals(0, learningEndpoint.getInflightBatchCount());
        } finally {
            learningEndpoint.close();
        }
    }

    @Test
    void deferredUnchangedLearningAddsNoSignalBeyondWorkerStatus() {
        PrefillEndpoint learningEndpoint = createLearningEndpoint();
        try {
            PrefillTimePredictor.Evaluator initialEvaluator =
                    learningEndpoint.getPredictor().evaluator();
            long batchId = 8_101L;
            long requestId = 9_101L;
            ScheduledRequest protectedItem = createScheduledRequest(
                    learningEndpoint, requestId, 500L, 200L);
            registerBatch(
                    learningEndpoint,
                    batchId,
                    100L,
                    List.of(protectedItem));
            PrefillState.Protection protection =
                    learningEndpoint.acquireBatchMemberProtection(
                            batchId, protectedItem);
            assertNotNull(protection);

            reportSuccessfulBatchMember(
                    learningEndpoint, batchId, requestId, 101L);
            assertSame(initialEvaluator,
                    learningEndpoint.getPredictor().evaluator());

            learningEndpoint.releaseEngineFenceProtection(protection);

            assertSame(initialEvaluator,
                    learningEndpoint.getPredictor().evaluator(),
                    "the first sample returns MODEL_UNCHANGED");
            assertEquals(0, learningEndpoint.getInflightBatchCount());
        } finally {
            learningEndpoint.close();
        }
    }

    @Test
    void finishedSettlementRemovesMemberBeforeLateProtection() {
        ScheduledRequest item = createScheduledRequest(101L, 500, 200);
        registerBatch(
                endpoint,
                700L,
                100,
                List.of(item));

        calibrate(Map.of(
                "101", taskInfo(101L, 700L, null, 0, 10)), Map.of());

        assertTrue(endpoint.acquireBatchMemberProtection(700L, item) == null);
        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0,
                endpoint.captureRouteProjectionInputs().work().batches().size());
        assertEquals(0, endpoint.admissionPendingRequestCount());
    }

    @Test
    void missingBatchIdTerminalDoesNotReleaseFixedWindowSlot() throws Exception {
        registerBatch(endpoint, 700L, 100,
                List.of(createScheduledRequest(101L, 500, 200)));
        assertFalse(endpoint.batchAdmissionAvailability(1).isAvailable(),
                "maxInflight=1 must stay closed while the ledger is occupied");

        calibrate(Map.of(
                "101", priorityCanceledTask(101L, -1L)), Map.of());

        // A terminal without a valid batch id cannot be attributed to the
        // batch member, so the exact fixed-window slot stays occupied.
        assertFalse(endpoint.batchAdmissionAvailability(1).isAvailable(),
                "missing-batch-id terminal cannot release the exact slot");
        assertEquals(1, endpoint.getInflightBatchCount());
    }

    @Test
    void calibrateDoesNotRemoveBatchWithForeignRequestId() {
        // Commit batch with requestId=100
        ScheduledRequest item = createScheduledRequest(100L, 500, 200);
        registerBatch(endpoint, 1L, 100, List.of(item));
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
        ScheduledRequest item = createScheduledRequest(100L, 500, 200);
        registerBatch(endpoint, 1L, 100, List.of(item));

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
        ScheduledRequest reconciling = createScheduledRequest(101L, 500, 200);
        ScheduledRequest sibling = createScheduledRequest(102L, 300, 100);
        registerBatch(endpoint, 7L, 100, List.of(reconciling, sibling));
        PrefillState.Protection protection =
                endpoint.acquireBatchMemberProtection(7L, reconciling);
        assertNotNull(protection);

        TaskInfo siblingSuccess = new TaskInfo();
        siblingSuccess.setBatchId(7L);
        siblingSuccess.setRequestId(102L);
        siblingSuccess.setErrorCode(0);
        calibrate(Map.of("102", siblingSuccess), Map.of());

        assertEquals(1, endpoint.getInflightBatchCount(),
                "sibling success must not erase the reconciling batch member");
        assertEquals(1, endpoint.admissionPendingRequestCount());

        TaskInfo ambiguousMemberSuccess = new TaskInfo();
        ambiguousMemberSuccess.setBatchId(7L);
        ambiguousMemberSuccess.setRequestId(101L);
        ambiguousMemberSuccess.setErrorCode(0);
        calibrate(Map.of("101", ambiguousMemberSuccess), Map.of());
        // The exact-batch success terminal is an authoritative Engine reducer:
        // it settles the protected member immediately and invalidates the
        // protection, emptying the batch.
        assertEquals(0, endpoint.getInflightBatchCount(),
                "an exact-batch terminal settles even a protected member");
        assertEquals(0, endpoint.admissionPendingRequestCount());

        // Releasing the already-invalidated protection is a graceful no-op.
        endpoint.releaseEngineFenceProtection(protection);
        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.admissionPendingRequestCount());
    }

    @Test
    void protectedAndSiblingFailuresSettleFromOneWorkerSnapshot() {
        ScheduledRequest protectedItem = createScheduledRequest(101L, 500, 200);
        ScheduledRequest sibling = createScheduledRequest(102L, 300, 100);
        registerBatch(endpoint, 7L, 100,
                List.of(protectedItem, sibling));
        PrefillState.Protection protection =
                endpoint.acquireBatchMemberProtection(7L, protectedItem);
        assertNotNull(protection);

        TaskInfo protectedFailure = taskInfo(101L, 7L, null, 500, 40);
        TaskInfo siblingFailure = taskInfo(102L, 7L, null, 501, 50);
        calibrate(Map.of("101", protectedFailure, "102", siblingFailure), Map.of());

        // Both exact-batch failures settle from one authoritative worker
        // snapshot: the protected member is not deferred, so the batch empties
        // immediately and the protection is invalidated.
        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.admissionPendingRequestCount());

        endpoint.releaseEngineFenceProtection(protection);
        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.admissionPendingRequestCount());
        verify(endpointReporter, never()).reportBatchPredictedTimeMs(
                anyString(), anyString(), org.mockito.ArgumentMatchers.anyLong());

        // Releasing the already-invalidated protection again is an idempotent
        // no-op rather than a double-free error.
        endpoint.releaseEngineFenceProtection(protection);
        assertEquals(0, endpoint.admissionPendingRequestCount());
    }

    // ---- committed remaining work ----

    @Test
    void committedWorkMetricIsZeroWhenIdle() {
        assertEquals(0L, endpoint.getLoadMetric().orElseThrow());
    }

    @Test
    void committedWorkMetricReflectsInflightPrediction() {
        ScheduledRequest item = createScheduledRequest(1L, 500, 200);
        registerBatch(endpoint, 1L, 5000, List.of(item)); // 5s prediction

        long remainingWorkMs = endpoint.getLoadMetric().orElseThrow();
        assertTrue(remainingWorkMs > 0,
                "inflight work must have a positive remaining duration");
        assertTrue(remainingWorkMs <= 5000,
                "remaining work must not exceed the original prediction");
    }

    @Test
    void runningCommittedWorkMetricDecreasesWithElapsedTime()
            throws InterruptedException {
        ScheduledRequest item = createScheduledRequest(1L, 500, 200);
        registerBatch(endpoint, 1L, 5000, List.of(item));

        long remainingBefore = endpoint.getLoadMetric().orElseThrow();

        // Mark the batch as running so elapsed time counts
        Map<String, TaskInfo> running = new HashMap<>();
        TaskInfo runningTask = new TaskInfo();
        runningTask.setRequestId(1L);
        runningTask.setBatchId(1L);
        runningTask.setPhase(TaskPhase.RUNNING);
        running.put("1", runningTask);
        calibrate(Map.of(), running);

        Thread.sleep(50);

        long remainingAfter = endpoint.getLoadMetric().orElseThrow();
        assertTrue(remainingAfter <= remainingBefore,
                "remaining work must decrease after observed progress");
    }

    // ---- eviction ----

    @Test
    void evictExpiredBatchesCleansUpStaleEntries() throws InterruptedException {
        ScheduledRequest item = createScheduledRequest(1L, 500, 200);
        registerBatch(endpoint, 1L, 100, List.of(item));

        assertEquals(1, endpoint.getInflightBatchCount());

        // Wait a bit so the batch ages
        Thread.sleep(10);

        int evicted = endpoint.evictExpiredBatches(1); // 1ms TTL — should evict
        assertEquals(1, evicted);
        assertEquals(0, endpoint.getInflightBatchCount());
    }

    @Test
    void evictExpiredBatchesFreshEntriesSurvive() {
        ScheduledRequest item = createScheduledRequest(1L, 500, 200);
        registerBatch(endpoint, 1L, 100, List.of(item));

        int evicted = endpoint.evictExpiredBatches(60_000); // 60s TTL — fresh entry survives
        assertEquals(0, evicted);
        assertEquals(1, endpoint.getInflightBatchCount());
    }

    @Test
    void evictExpiredBatchesRetainsAckAmbiguousBatchUntilReconciled()
            throws InterruptedException {
        ScheduledRequest item = createScheduledRequest(1L, 500, 200);
        registerBatch(endpoint, 1L, 100, List.of(item));
        PrefillState.Protection protection =
                endpoint.acquireBatchMemberProtection(1L, item);
        assertNotNull(protection);
        Thread.sleep(10);

        assertEquals(0, endpoint.evictExpiredBatches(1));
        assertEquals(1, endpoint.getInflightBatchCount());

        endpoint.releaseEngineFenceProtection(protection);
        // The protection captured no deferred terminal, so releasing it does
        // not refresh batch activity. The already-aged batch therefore becomes
        // immediately evictable once the fence is gone.
        assertEquals(1, endpoint.evictExpiredBatches(1),
                "releasing an unreconciled protection does not refresh activity");
        assertEquals(0, endpoint.getInflightBatchCount());
    }

    // ---- admissionPendingRequestCount ----

    @Test
    void admissionPendingRequestCountUnionsEngineTasksWithLocalLedger() {
        registerBatch(endpoint, 1L, 100, List.of(
                createScheduledRequest(101L, 500, 0),
                createScheduledRequest(102L, 500, 0)));

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
        EndpointTestSupport.applyStatus(endpoint, response);

        assertEquals(4, endpoint.admissionPendingRequestCount(),
                "two local requests plus two unique Engine-only tasks");

        response.setRunningTaskInfo(Map.of());
        EndpointTestSupport.applyStatus(endpoint, response);
        assertEquals(2, endpoint.admissionPendingRequestCount());
    }

    @Test
    void admissionPendingRequestCountFallsBackToEngineQueryLengthScalars() {
        registerBatch(endpoint, 1L, 100, List.of(createScheduledRequest(101L, 500, 0)));

        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setFinishedTaskInfo(Map.of());
        response.setRunningTaskInfo(Map.of());
        response.setWaitingQueryLen(3);
        response.setRunningQueryLen(2);
        EndpointTestSupport.applyStatus(endpoint, response);

        assertEquals(5, endpoint.admissionPendingRequestCount(),
                "the local prefill batch is unioned with, not added on top of, the "
                        + "scalar Engine work: pending = 1 local + max(0, 5 reported - 1 local)");
    }

    @Test
    void admissionPendingRequestCountUsesConservativeScalarBoundForPartialTaskDetails() {
        registerBatch(endpoint, 1L, 100, List.of(createScheduledRequest(101L, 500, 0)));

        TaskInfo overlapping = taskInfo(101L, 1L, TaskPhase.RUNNING, 0, 0);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setFinishedTaskInfo(Map.of());
        response.setRunningTaskInfo(Map.of("101", overlapping));
        response.setWaitingQueryLen(3);
        response.setRunningQueryLen(2);
        EndpointTestSupport.applyStatus(endpoint, response);

        assertEquals(5, endpoint.admissionPendingRequestCount(),
                "scalar active count must cover a partial detail list without double-counting local tasks");
    }

    @Test
    void admissionPendingRequestCountIncludesBatcherQueue() throws InterruptedException {
        PrefillEndpoint queuedEndpoint = newFixedWindowEndpoint(60_000L);
        try {
            assertEquals(0, queuedEndpoint.admissionPendingRequestCount());
            ScheduledRequest item = createScheduledRequest(
                    queuedEndpoint, 1L, 500L, 200L);
            assertTrue(EndpointTestSupport.offer(queuedEndpoint, item));

            assertEquals(1, queuedEndpoint.admissionPendingRequestCount(),
                    "pending count includes the canonical ACTIVE queue owner");
        } finally {
            queuedEndpoint.close();
        }
    }

    @Test
    void admissionPendingRequestCountConservativelyCoversPublishedWorkBeforeActiveRemoval() {
        PrefillEndpoint handoffEndpoint = newFixedWindowEndpoint(60_000);
        try {
            ScheduledRequest active = createScheduledRequest(handoffEndpoint, 111L, 500L, 0L);
            assertTrue(EndpointTestSupport.offer(handoffEndpoint, active));

            org.flexlb.balance.projection.RouteProjection.Inputs snapshot =
                    handoffEndpoint.captureRouteProjectionInputs();
            assertEquals(1, snapshot.queue().activeItems().size());
            assertEquals(1L, snapshot.pendingRequestCount(),
                    "queue/work/pending are materialized at one ownership boundary");
            assertEquals(111L,
                    snapshot.queue().activeItems().getFirst().requestId());
        } finally {
            handoffEndpoint.close();
        }
    }

    @Test
    void admissionPendingRequestCountCannotMissActiveToCommittedHandoff()
            throws Exception {
        PrefillEndpoint handoffEndpoint = newFixedWindowEndpoint(60_000);
        long requestId = 222L;
        try {
            ScheduledRequest active = createScheduledRequest(
                    handoffEndpoint, requestId, 500L, 0L);
            assertTrue(EndpointTestSupport.offer(handoffEndpoint, active));
            assertTrue(handoffEndpoint.removeQueued(
                    active, "test exact ownership handoff"));
            registerDirect(handoffEndpoint, requestId, 100L);

            assertEquals(1L, handoffEndpoint.admissionPendingRequestCount());
            assertEquals(0, handoffEndpoint.queuedRequestCount());
            assertEquals(1,
                    handoffEndpoint.captureRouteProjectionInputs()
                            .work().requests().size());
        } finally {
            handoffEndpoint.close();
        }
    }

    // ---- batch metrics reporting ----

    @Test
    void reportBatchMetricsBucketsQueueLengthByPriority() {
        // Long fixed window so offered items stay queued during the assertions
        PrefillEndpoint slowEndpoint = newFixedWindowEndpoint(60_000);
        try {
            assertTrue(EndpointTestSupport.offer(
                    slowEndpoint, createPriorityScheduledRequest(slowEndpoint, 1L, 70)));
            assertTrue(EndpointTestSupport.offer(
                    slowEndpoint,
                    createScheduledRequest(slowEndpoint, 2L, 300, 0)));

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
    void applyWorkerStatusResponseUpdatesAliveStatusOnSameGeneration() {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setAlive(true);

        EndpointTestSupport.applyStatus(endpoint, response);

        assertTrue(endpoint.getStatus().pollHealth().reportedAlive());
    }

    // ---- close ----

    @Test
    void retirementOwnerCanReenterCloseFromSynchronousShutdownCallback()
            throws Exception {
        FlexlbConfig retirementConfig = new FlexlbConfig();
        configureBatch(retirementConfig, 100, 1, 0, null);
        retirementConfig.setDispatcher(DispatcherConfig.nonBatch());

        WorkerStatus status = EndpointTestSupport.workerStatus(
                RoleType.PREFILL, "127.0.0.5", 8080, 8090);

        CountDownLatch reentrantCloseEntered = new CountDownLatch(1);
        CountDownLatch reentrantCloseReturned = new CountDownLatch(1);
        AtomicReference<Throwable> callbackFailure = new AtomicReference<>();
        AtomicReference<PrefillEndpoint> retirementEndpointRef = new AtomicReference<>();
        EndpointTestSupport.TestRequestRuntime runtime =
                new EndpointTestSupport.TestRequestRuntime() {
            @Override
            void onQueueOfferFailure(
                    org.flexlb.balance.scheduler.ScheduledRequest item,
                    Throwable error) {
                reentrantCloseEntered.countDown();
                try {
                    retirementEndpointRef.get().close();
                    reentrantCloseReturned.countDown();
                } catch (Throwable failure) {
                    callbackFailure.compareAndSet(null, failure);
                }
            }
        };
        PrefillEndpoint retirementEndpoint = new PrefillEndpoint(
                status,
                retirementConfig,
                EndpointTestSupport.routeStrategy(runtime),
                runtime.events(),
                mock(BatchSchedulerReporter.class));
        retirementEndpointRef.set(retirementEndpoint);
        retirementEndpoint.startGeneration();
        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            ScheduledRequest item = createScheduledRequest(
                    retirementEndpoint, retirementConfig, 8_101L, 128, 0);
            assertTrue(EndpointTestSupport.offer(retirementEndpoint, item));

            Future<?> outerClose = executor.submit(retirementEndpoint::close);
            outerClose.get(2, TimeUnit.SECONDS);
            assertTrue(reentrantCloseEntered.await(1, TimeUnit.SECONDS));
            assertTrue(reentrantCloseReturned.await(1, TimeUnit.SECONDS),
                    "retirement-owner reentry must return instead of waiting on itself");
            assertTrue(callbackFailure.get() == null,
                    () -> String.valueOf(callbackFailure.get()));

            assertEquals(
                    PrefillState.CapacityStatus.ENDPOINT_RETIRED,
                    retirementEndpoint.reserveRoute(
                            createScheduledRequest(
                                    retirementEndpoint,
                                    retirementConfig,
                                    8_102L,
                                    128,
                                    0),
                            0L,
                            1).status());
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

        WorkerStatus status = EndpointTestSupport.workerStatus(
                RoleType.PREFILL, "127.0.0.6", 8080, 8090);

        CountDownLatch callbackResolved = new CountDownLatch(1);
        AtomicReference<Throwable> callbackFailure = new AtomicReference<>();
        AtomicReference<PrefillEndpoint> retirementEndpointRef = new AtomicReference<>();
        EndpointTestSupport.TestRequestRuntime runtime =
                new EndpointTestSupport.TestRequestRuntime() {
            @Override
            void onCompleted(
                    RequestRegistry.DeliveryClaim claim,
                    DeliveryResult completion) {
                try {
                    retirementEndpointRef.get().close();
                } catch (Throwable failure) {
                    callbackFailure.compareAndSet(null, failure);
                } finally {
                    callbackResolved.countDown();
                }
            }
        };
        PrefillEndpoint retirementEndpoint = new PrefillEndpoint(
                status,
                retirementConfig,
                EndpointTestSupport.liveRouteStrategy(runtime),
                runtime.events(),
                mock(BatchSchedulerReporter.class));
        retirementEndpointRef.set(retirementEndpoint);
        retirementEndpoint.startGeneration();
        try {
            DecodeEndpoint decode = mock(DecodeEndpoint.class);
            DecodeEndpoint.ReservationHandle decodeReservation =
                    mock(DecodeEndpoint.ReservationHandle.class);
            DecodeEndpoint.EngineDispatchPermit permit =
                    mock(DecodeEndpoint.EngineDispatchPermit.class);
            org.mockito.Mockito.when(decode.acquireEngineDispatchPermit(
                            org.mockito.Mockito.anyLong(),
                            org.mockito.Mockito.anyLong(),
                            org.mockito.Mockito.anyLong()))
                    .thenReturn(new DecodeEndpoint.EngineDispatchPermitAcquisition(
                            DecodeEndpoint.EngineDispatchPermitAcquireStatus.ACQUIRED,
                            decodeReservation,
                            permit));
            org.mockito.Mockito.when(permit.transferToEngineLifecycle())
                    .thenReturn(
                            DecodeEndpoint.EngineDispatchPermitTransferStatus.TRANSFERRED);
            ScheduledRequest admitted = createScheduledRequest(
                    retirementEndpoint,
                    retirementConfig,
                    8_201L,
                    128,
                    0,
                    decode,
                    decodeReservation);
            assertTrue(EndpointTestSupport.offer(retirementEndpoint, admitted));
            assertTrue(callbackResolved.await(2, TimeUnit.SECONDS));
            assertNull(callbackFailure.get(),
                    "an admitted handoff must defer cleanup instead of self-awaiting");
            retirementEndpoint.awaitRetirement();
        } finally {
            retirementEndpoint.close();
        }
    }

    @Test
    void closePreservesRegisteredLifecycleAndRejectsNewBatchReservations() {
        ScheduledRequest item = createScheduledRequest(1L, 500, 200);
        registerBatch(
                endpoint,
                1L,
                100,
                List.of(item));
        assertEquals(1, endpoint.getInflightBatchCount());

        endpoint.close();
        assertEquals(0, endpoint.getInflightBatchCount());
        EndpointTestSupport.PrefillRetirement retirement =
                requestRuntime.prefillRetirements().stream()
                        .findFirst()
                        .orElseThrow();
        assertTrue(retirement.ownedItems().contains(item),
                "retirement must publish the exact canonical owner");
        assertEquals(PrefillState.CapacityStatus.ENDPOINT_RETIRED,
                endpoint.reserveBatch(item, 2L, 1).status());
        endpoint.close();
    }

    @Test
    void closeRetiresDirectAccountingAndPreservesCommittedQueueRoute() {
        registerDirect(endpoint, 100L, 100L);
        ScheduledRequest route = createScheduledRequest(200L, 200L, 0L);
        // A queue route can only be reserved after its canonical item has been
        // offered into the ACTIVE queue; reserveRoute rejects a non-active
        // identity with REQUEST_NOT_ACTIVE.
        assertTrue(EndpointTestSupport.offer(endpoint, route));
        List<PrefillState.CommittedHandoff> handoffs =
                EndpointTestSupport.commitRoutes(
                        endpoint, 200L, List.of(route));
        handoffs.forEach(PrefillState.CommittedHandoff::close);
        assertEquals(2, endpoint.getIndividuallyTrackedRequestCount());

        endpoint.close();

        assertEquals(0, endpoint.getIndividuallyTrackedRequestCount());
        EndpointTestSupport.PrefillRetirement retirement =
                requestRuntime.prefillRetirements().stream()
                        .findFirst()
                        .orElseThrow();
        assertEquals(List.of(route), retirement.ownedItems(),
                "only the canonical item-bearing route owner crosses retirement");
    }

    @Test
    void closeShutsDownBatcher() {
        endpoint.close();
        ScheduledRequest item = createScheduledRequest(1L, 500, 200);
        assertFalse(EndpointTestSupport.offer(endpoint, item));
        assertEquals(0, endpoint.queuedRequestCount());
    }

    // ---- helpers ----

    private PrefillEndpoint newFixedWindowEndpoint(long fixedWaitMs) {
        WorkerStatus status = EndpointTestSupport.workerStatus(
                RoleType.PREFILL, "127.0.0.1", 8080, 8090);

        FlexlbConfig slowConfig = new FlexlbConfig();
        configureBatch(slowConfig, 100, 100, fixedWaitMs, null);
        EndpointTestSupport.TestRequestRuntime runtime =
                EndpointTestSupport.requestRuntime();
        PrefillEndpoint created = new PrefillEndpoint(
                status,
                slowConfig,
                EndpointTestSupport.routeStrategy(runtime),
                runtime.events(),
                mock(BatchSchedulerReporter.class));
        created.startGeneration();
        return created;
    }

    private static PrefillEndpoint newEndpointWithFormula(String expression) {
        WorkerStatus status = EndpointTestSupport.workerStatus(
                RoleType.PREFILL, "127.0.0.9", 8089, 8099);

        FlexlbConfig endpointConfig = new FlexlbConfig();
        configureBatch(
                endpointConfig,
                100,
                endpointConfig.fixedWindowDecision().getMaxRequests(),
                300L,
                null);
        setFormula(endpointConfig, expression);
        EndpointTestSupport.TestRequestRuntime runtime =
                EndpointTestSupport.requestRuntime();
        PrefillEndpoint created = new PrefillEndpoint(
                status,
                endpointConfig,
                EndpointTestSupport.routeStrategy(runtime),
                runtime.events(),
                mock(BatchSchedulerReporter.class));
        created.startGeneration();
        return created;
    }

    private ScheduledRequest createPriorityScheduledRequest(
            PrefillEndpoint owner, long requestId, int priority) {
        long now = System.currentTimeMillis();
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(500);
        request.setPriority(priority);

        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setConfig(config);
        ctx.setSchedulingMetadata(SchedulingMetadata.explicit(priority, now + 60_000));

        return new ScheduledRequest(
                ctx, null, null, null, null, owner, null, null, now);
    }

    private void calibrate(Map<String, TaskInfo> finished, Map<String, TaskInfo> running) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setFinishedTaskInfo(finished);
        response.setRunningTaskInfo(running);
        EndpointTestSupport.applyStatus(endpoint, response);
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
        EndpointTestSupport.applyStatus(target, response);
    }

    private static PrefillEndpoint createLearningEndpoint() {
        WorkerStatus status = EndpointTestSupport.workerStatus(
                RoleType.PREFILL, "127.0.0.8", 8080, 8090);

        FlexlbConfig learningConfig = new FlexlbConfig();
        configureBatch(
                learningConfig,
                100,
                learningConfig.fixedWindowDecision().getMaxRequests(),
                300,
                null);
        learningConfig.getRouter().getRoles().getPrefill()
                .getExecutionTimeEstimator()
                .setType(RoutingConfig.EstimatorType.LEARNING);
        EndpointTestSupport.TestRequestRuntime runtime =
                EndpointTestSupport.requestRuntime();
        PrefillEndpoint created = new PrefillEndpoint(
                status,
                learningConfig,
                EndpointTestSupport.routeStrategy(runtime),
                runtime.events(),
                mock(BatchSchedulerReporter.class));
        created.startGeneration();
        return created;
    }

    private static WorkSnapshot.BatchWork onlyBatchWork(
            PrefillEndpoint target) {
        List<WorkSnapshot.BatchWork> batches =
                target.captureRouteProjectionInputs().work().batches();
        assertEquals(1, batches.size());
        return batches.get(0);
    }

    private ScheduledRequest createScheduledRequest(long requestId, long seqLen, long hitCacheLen) {
        return createScheduledRequest(endpoint, requestId, seqLen, hitCacheLen);
    }

    private static ScheduledRequest createScheduledRequest(PrefillEndpoint owner,
                                             long requestId,
                                             long seqLen,
                                             long hitCacheLen) {
        return createScheduledRequest(
                owner, new FlexlbConfig(), requestId, seqLen, hitCacheLen);
    }

    private static ScheduledRequest createScheduledRequest(
            PrefillEndpoint owner,
            FlexlbConfig requestConfig,
            long requestId,
            long seqLen,
            long hitCacheLen) {
        return createScheduledRequest(
                owner,
                requestConfig,
                requestId,
                seqLen,
                hitCacheLen,
                null,
                null);
    }

    private static ScheduledRequest createScheduledRequest(
            PrefillEndpoint owner,
            FlexlbConfig requestConfig,
            long requestId,
            long seqLen,
            long hitCacheLen,
            DecodeEndpoint decode,
            DecodeEndpoint.ReservationHandle decodeReservation) {
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

        return new ScheduledRequest(
                ctx,
                null,
                null,
                prefill,
                null,
                owner,
                decode,
                decodeReservation,
                System.currentTimeMillis());
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
        ((org.flexlb.config.DispatcherConfig) target.getDispatcher())
                .setMaxInflightBatchesPerPrefillWorker(maxInflightBatches);
    }

    private static void setFormula(FlexlbConfig target, String expression) {
        target.getRouter().getRoles().getPrefill().getExecutionTimeEstimator()
                .setExpression(expression);
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

    private static void registerBatch(
            PrefillEndpoint target,
            long batchId,
            long predictedMs,
            List<ScheduledRequest> items) {
        for (ScheduledRequest item : items) {
            if (!EndpointTestSupport.offer(target, item)) {
                throw new IllegalStateException(
                        "test item could not be offered to the endpoint queue");
            }
        }
        try (PrefillState.CommittedHandoff ignored =
                     EndpointTestSupport.commitBatch(
                             target, batchId, predictedMs, items)) {
            // Keep canonical ledger ownership; release only the generation pin.
        }
    }

    private static void registerDirect(
            PrefillEndpoint target,
            long requestId,
            long predictedMs) {
        try (PrefillState.DirectRegistration registration =
                     EndpointTestSupport.registerDirect(
                             target, requestId, predictedMs)) {
            registration.commit();
        }
    }

}
