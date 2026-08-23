package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.DecisionGroupHandler;
import org.flexlb.balance.scheduler.AdmittedDecisionGroup;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.DecisionGroupMetadata;
import org.flexlb.balance.scheduler.TestCapacityAdmission;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.flexlb.constant.MetricConstant.INFLIGHT_BATCH_COUNT;
import static org.flexlb.constant.MetricConstant.INFLIGHT_MAX_AGE_MS;
import static org.flexlb.constant.MetricConstant.INFLIGHT_REQUEST_COUNT;

class PrefillRequestLedgerTest {

    private PrefillEndpoint endpoint;
    private BatchSchedulerReporter reporter;
    private RecordingMonitor monitor;

    @BeforeEach
    void setUp() {
        WorkerStatus status = new WorkerStatus();
        status.setIp("127.0.0.41");
        status.setPort(8041);
        status.setGrpcPort(9041);
        status.setRole(RoleType.PREFILL);

        FlexlbConfig config = new FlexlbConfig();
        config.queueScheduler().getCapacity().setMaxWaitingRequestsPerPrefillWorker(128);
        config.fixedWindowDecision().setMaxCollectionWaitMs(60_000);
        ((RoutingConfig.FormulaEstimatorConfig) config.getRouter().getRoles().getPrefill()
                .getExecutionTimeEstimator()).setExpression(
                        "10 + 0.1*sum(computeTokens) + 5*batchSize");

        monitor = new RecordingMonitor();
        reporter = new BatchSchedulerReporter(monitor);
        endpoint = new PrefillEndpoint(
                status,
                config,
                noopHandler(),
                TestCapacityAdmission.alwaysAvailable(),
                reporter);
    }

    @AfterEach
    void tearDown() {
        endpoint.close();
    }

    @Test
    void capacityReservationAcquisitionDistinguishesOwnershipFromCapacity() {
        AtomicInteger capacityNotifications = new AtomicInteger();
        PrefillRequestLedger ledger = new PrefillRequestLedger(
                capacityNotifications::incrementAndGet);

        PrefillRequestLedger.RequestCapacityReservationAcquisition acquired =
                ledger.acquireCapacityReservation(1L, 700, 1);
        assertEquals(
                PrefillRequestLedger.RequestCapacityReservationAcquisition.Status.ACQUIRED,
                acquired.status());
        assertNotNull(acquired.reservation());
        assertEquals(1, ledger.count(),
                "an uncommitted reservation must immediately occupy capacity");
        assertEquals(0, ledger.available(1));

        PrefillRequestLedger.RequestCapacityReservationAcquisition alreadyTracked =
                ledger.acquireCapacityReservation(1L, 9_999, 1);
        assertEquals(
                PrefillRequestLedger.RequestCapacityReservationAcquisition.Status
                        .REQUEST_ALREADY_TRACKED,
                alreadyTracked.status());
        assertNull(alreadyTracked.reservation(),
                "idempotent admission must not grant release authority over an existing entry");

        PrefillRequestLedger.RequestCapacityReservationAcquisition capacityFull =
                ledger.acquireCapacityReservation(2L, 500, 1);
        assertEquals(
                PrefillRequestLedger.RequestCapacityReservationAcquisition.Status.CAPACITY_FULL,
                capacityFull.status());
        assertNull(capacityFull.reservation());
        assertEquals(1, ledger.count());
        assertEquals(0, capacityNotifications.get());

        assertTrue(acquired.reservation().release());
        assertEquals(1, capacityNotifications.get());
        assertEquals(0, ledger.count());
    }

    @Test
    void preparedCapacityReservationTransfersLedgerOwnershipExactlyOnce() {
        PrefillRequestLedger ledger = new PrefillRequestLedger(() -> {});
        PrefillRequestLedger.RequestCapacityReservation reservation =
                ledger.acquireCapacityReservation(11L, 800, 1).reservation();

        assertNotNull(reservation);
        assertTrue(reservation.prepareForDelivery());
        assertFalse(reservation.prepareForDelivery(), "prepare must be idempotent");
        assertFalse(ledger.release(11L),
                "ordinary lifecycle cleanup must not remove a prepared reservation");
        assertFalse(ledger.settle(11L),
                "WorkerStatus must not settle work before delivery owns it");
        reservation.completePreparedDeliveryTransfer();
        assertFalse(reservation.release(),
                "a committed token no longer owns release of the ledger entry");
        assertEquals(1, ledger.count());

        assertTrue(ledger.release(11L),
                "ordinary lifecycle ownership must settle the committed entry");
        assertEquals(0, ledger.count());
    }

    @Test
    void preparedCapacityReservationCanAbortBeforeDeliveryAndReleaseCapacity() {
        AtomicInteger capacityNotifications = new AtomicInteger();
        PrefillRequestLedger ledger = new PrefillRequestLedger(
                capacityNotifications::incrementAndGet);
        PrefillRequestLedger.RequestCapacityReservation reservation =
                ledger.acquireCapacityReservation(12L, 800, 1).reservation();

        assertNotNull(reservation);
        assertTrue(reservation.prepareForDelivery());
        assertFalse(reservation.prepareForDelivery(),
                "a repeated prepare must keep returning false");
        assertEquals(1, ledger.count());
        assertEquals(0, ledger.available(1));

        assertTrue(reservation.abortBeforeDelivery());
        assertFalse(reservation.abortBeforeDelivery(), "abort must be idempotent");
        assertFalse(reservation.release(), "the closed token must not release twice");
        assertEquals(0, ledger.count());
        assertEquals(1, ledger.available(1));
        assertEquals(1, capacityNotifications.get(),
                "aborting the exact committed entry must publish one capacity release");
    }

    @Test
    void endpointRetirementRemovesDirectAccountingButPreservesQueueRouteOwnership() {
        AtomicInteger capacityNotifications = new AtomicInteger();
        PrefillRequestLedger ledger = new PrefillRequestLedger(
                capacityNotifications::incrementAndGet);

        assertTrue(ledger.registerDirectRequest(1L, 100));
        assertTrue(ledger.registerDirectRequest(2L, 200));
        assertTrue(commitQueueRoute(ledger, 3L, 300, 2));
        assertTrue(commitQueueRoute(ledger, 4L, 400, 2));
        assertEquals(4, ledger.count());
        assertEquals(2, ledger.queueRouteCount());

        assertEquals(2, ledger.retireDirectRequests());
        assertEquals(0, ledger.retireDirectRequests(),
                "generation retirement must be idempotent");
        assertEquals(2, ledger.count());
        assertEquals(2, ledger.queueRouteCount(),
                "retirement must not release hard QUEUE_ROUTE capacity");
        assertFalse(ledger.release(1L));
        assertFalse(ledger.release(2L));
        assertEquals(0, capacityNotifications.get());

        assertTrue(ledger.release(3L));
        assertTrue(ledger.release(4L));
        assertEquals(2, capacityNotifications.get());
        assertEquals(0, ledger.count());
    }

    @Test
    void transferredCapacityReservationCannotRemoveNormalLifecycleEntry() {
        AtomicInteger capacityNotifications = new AtomicInteger();
        PrefillRequestLedger ledger = new PrefillRequestLedger(
                capacityNotifications::incrementAndGet);
        PrefillRequestLedger.RequestCapacityReservation reservation =
                ledger.acquireCapacityReservation(13L, 800, 1).reservation();

        assertNotNull(reservation);
        assertTrue(reservation.prepareForDelivery());
        assertFalse(reservation.prepareForDelivery(),
                "a repeated prepare must keep returning false");
        reservation.completePreparedDeliveryTransfer();
        reservation.completePreparedDeliveryTransfer();

        assertFalse(reservation.abortBeforeDelivery(),
                "a transferred token must not compensate normal lifecycle ownership");
        assertFalse(reservation.release(),
                "a transferred token must not release normal lifecycle ownership");
        assertEquals(1, ledger.count());
        assertEquals(0, ledger.available(1));
        assertEquals(0, capacityNotifications.get());

        assertTrue(ledger.release(13L),
                "only the normal lifecycle may settle the transferred entry");
        assertEquals(0, ledger.count());
        assertEquals(1, ledger.available(1));
        assertEquals(1, capacityNotifications.get());
    }

    @Test
    void preparedReservationPinsItsEntryUntilTheTokenResolvesIt() {
        AtomicInteger capacityNotifications = new AtomicInteger();
        PrefillRequestLedger ledger = new PrefillRequestLedger(
                capacityNotifications::incrementAndGet);
        PrefillRequestLedger.RequestCapacityReservation staleReservation =
                ledger.acquireCapacityReservation(14L, 800, 1).reservation();

        assertNotNull(staleReservation);
        assertTrue(staleReservation.prepareForDelivery());
        assertFalse(staleReservation.prepareForDelivery(),
                "a repeated prepare must keep returning false");
        assertFalse(ledger.release(14L),
                "ordinary cleanup cannot break a prepared composite admission");
        assertFalse(ledger.settle(14L),
                "a stale WorkerStatus terminal cannot remove prepared capacity");
        assertTrue(staleReservation.abortBeforeDelivery());

        PrefillRequestLedger.RequestCapacityReservation replacement =
                ledger.acquireCapacityReservation(14L, 1_600, 1).reservation();
        assertNotNull(replacement);

        assertFalse(staleReservation.abortBeforeDelivery(),
                "the old entry token must not remove a replacement with the same request id");
        assertFalse(staleReservation.release());
        assertEquals(1, ledger.count());
        assertEquals(0, ledger.available(1));
        assertEquals(1, capacityNotifications.get(),
                "the stale token must not publish a replacement capacity release");

        assertTrue(replacement.prepareForDelivery());
        assertTrue(replacement.abortBeforeDelivery(),
                "the replacement token must retain compensation authority over its own entry");
        assertEquals(0, ledger.count());
        assertEquals(1, ledger.available(1));
        assertEquals(2, capacityNotifications.get());
    }

    @Test
    void releasedCapacityReservationIsIdempotentAndCannotBeCommitted() {
        AtomicInteger capacityNotifications = new AtomicInteger();
        PrefillRequestLedger ledger = new PrefillRequestLedger(
                capacityNotifications::incrementAndGet);
        PrefillRequestLedger.RequestCapacityReservation reservation =
                ledger.acquireCapacityReservation(21L, 900, 1).reservation();

        assertNotNull(reservation);
        assertTrue(reservation.release());
        assertFalse(reservation.release(), "release must be idempotent");
        assertFalse(reservation.prepareForDelivery(),
                "a released token cannot later be prepared");
        assertEquals(0, ledger.count());
        assertEquals(1, capacityNotifications.get());
    }

    @Test
    void capacityReservationCommitAndReleaseHaveOneWinner() throws Exception {
        PrefillRequestLedger ledger = new PrefillRequestLedger(() -> {});
        PrefillRequestLedger.RequestCapacityReservation reservation =
                ledger.acquireCapacityReservation(25L, 900, 1).reservation();
        assertNotNull(reservation);

        ExecutorService executor = Executors.newFixedThreadPool(2);
        CountDownLatch start = new CountDownLatch(1);
        try {
            Future<Boolean> commit = executor.submit(() -> {
                await(start);
                return reservation.prepareForDelivery();
            });
            Future<Boolean> release = executor.submit(() -> {
                await(start);
                return reservation.release();
            });

            start.countDown();
            boolean committed = commit.get(5, TimeUnit.SECONDS);
            boolean released = release.get(5, TimeUnit.SECONDS);
            assertTrue(committed ^ released,
                    "exactly one terminal reservation operation must succeed");
            assertEquals(committed ? 1 : 0, ledger.count());
            if (committed) {
                assertFalse(ledger.release(25L),
                        "ordinary cleanup cannot overtake a prepared token");
                reservation.completePreparedDeliveryTransfer();
                assertTrue(ledger.release(25L));
            }
            assertEquals(0, ledger.count());
        } finally {
            start.countDown();
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
        }
    }

    @Test
    void staleCapacityReservationCannotReleaseReusedRequestId() {
        AtomicInteger capacityNotifications = new AtomicInteger();
        PrefillRequestLedger ledger = new PrefillRequestLedger(
                capacityNotifications::incrementAndGet);
        PrefillRequestLedger.RequestCapacityReservation staleReservation =
                ledger.acquireCapacityReservation(31L, 1_000, 1).reservation();
        assertNotNull(staleReservation);

        assertTrue(ledger.release(31L));
        PrefillRequestLedger.RequestCapacityReservation replacementReservation =
                ledger.acquireCapacityReservation(31L, 2_000, 1).reservation();
        assertNotNull(replacementReservation);

        assertFalse(staleReservation.release(),
                "the old entry token must not delete a replacement with the same request id");
        assertFalse(staleReservation.prepareForDelivery(),
                "the old entry token must not claim a replacement with the same request id");
        assertEquals(1, ledger.count());
        assertEquals(2_000, ledger.estimate(System.currentTimeMillis()),
                "the replacement entry and its prediction must remain intact");

        assertTrue(replacementReservation.release());
        assertEquals(0, ledger.count());
        assertEquals(2, capacityNotifications.get());
    }

    @Test
    void staleCapacityReservationCannotCommitReusedRequestId() {
        PrefillRequestLedger ledger = new PrefillRequestLedger(() -> {});
        PrefillRequestLedger.RequestCapacityReservation staleReservation =
                ledger.acquireCapacityReservation(41L, 1_000, 1).reservation();
        assertNotNull(staleReservation);

        assertTrue(ledger.release(41L));
        PrefillRequestLedger.RequestCapacityReservation replacementReservation =
                ledger.acquireCapacityReservation(41L, 2_000, 1).reservation();
        assertNotNull(replacementReservation);

        assertFalse(staleReservation.prepareForDelivery(),
                "the old entry token must not claim a replacement with the same request id");
        assertEquals(1, ledger.count());
        assertTrue(replacementReservation.prepareForDelivery());
        replacementReservation.completePreparedDeliveryTransfer();
        assertFalse(replacementReservation.release(),
                "the committed replacement is owned by the ordinary ledger lifecycle");
        assertTrue(ledger.release(41L));
        assertEquals(0, ledger.count());
    }

    @Test
    void duplicateRouteReservationIsRejectedAndReleaseIsIdempotent() {
        BatchItem request = batchItem(101L);

        assertTrue(TestCapacityAdmission.commitRouteRequest(endpoint, request.requestId(), 700, 4));
        assertFalse(TestCapacityAdmission.commitRouteRequest(
                endpoint, request.requestId(), 9_999, 4));
        assertEquals(1, endpoint.getLocallyOwnedRequestCount());
        assertEquals(1, endpoint.getIndividuallyTrackedRequestCount());
        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.realPendingCount());

        assertTrue(endpoint.releaseRequest(101L));
        assertFalse(endpoint.releaseRequest(101L));
        assertEquals(0, endpoint.getLocallyOwnedRequestCount());
        assertEquals(0, endpoint.getIndividuallyTrackedRequestCount());
    }

    @Test
    void routeRequestCapRemainsIndependentFromRealBatchMembers() {
        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 900L, 1_000, List.of(batchItem(1L), batchItem(2L)));

        assertEquals(2, endpoint.availableRequestSlots(2));
        assertTrue(TestCapacityAdmission.commitRouteRequest(endpoint, 3L, 500, 2));
        assertTrue(TestCapacityAdmission.commitRouteRequest(endpoint, 4L, 500, 2));
        assertFalse(TestCapacityAdmission.commitRouteRequest(endpoint, 5L, 500, 2));
        assertEquals(4, endpoint.getLocallyOwnedRequestCount());
        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(2, endpoint.getIndividuallyTrackedRequestCount());

        endpoint.releaseRequest(3L);
        endpoint.releaseRequest(4L);
        endpoint.releaseBatch(900L);
        assertEquals(0, endpoint.getLocallyOwnedRequestCount());
    }

    @Test
    void laterBatchCommitCannotBreakRouteRequestHardCap() {
        assertTrue(TestCapacityAdmission.commitRouteRequest(endpoint, 11L, 500, 2));
        assertTrue(TestCapacityAdmission.commitRouteRequest(endpoint, 12L, 500, 2));

        TestCapacityAdmission.registerQueueBatchLifecycle(endpoint, 901L, 1_000,
                List.of(batchItem(21L), batchItem(22L), batchItem(23L)));

        assertEquals(2, endpoint.getIndividuallyTrackedRequestCount());
        assertEquals(5, endpoint.getLocallyOwnedRequestCount());
        assertEquals(0, endpoint.availableRequestSlots(2));
        assertFalse(TestCapacityAdmission.commitRouteRequest(endpoint, 13L, 500, 2));
    }

    @Test
    void concurrentRequestCommitsNeverExceedHardCap() throws Exception {
        int attempts = 96;
        int cap = 11;
        ExecutorService executor = Executors.newFixedThreadPool(16);
        CountDownLatch start = new CountDownLatch(1);
        CountDownLatch done = new CountDownLatch(attempts);
        AtomicInteger admitted = new AtomicInteger();
        try {
            for (int i = 0; i < attempts; i++) {
                long requestId = 10_000L + i;
                executor.execute(() -> {
                    await(start);
                    if (TestCapacityAdmission.commitRouteRequest(endpoint, requestId, 100, cap)) {
                        admitted.incrementAndGet();
                    }
                    done.countDown();
                });
            }

            start.countDown();
            assertTrue(done.await(5, TimeUnit.SECONDS));

            assertEquals(cap, admitted.get());
            assertEquals(cap, endpoint.getLocallyOwnedRequestCount());
            assertEquals(cap, endpoint.getIndividuallyTrackedRequestCount());
            assertEquals(0, endpoint.availableRequestSlots(cap));
        } finally {
            start.countDown();
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
        }
    }

    @Test
    void repeatedCommitReleaseCyclesLeaveNoAccountingBehind() {
        for (long requestId = 1; requestId <= 5_000; requestId++) {
            assertTrue(TestCapacityAdmission.commitRouteRequest(endpoint, requestId, 10, 1));
            assertTrue(endpoint.releaseRequest(requestId));
        }

        assertEquals(0, endpoint.getLocallyOwnedRequestCount());
        assertEquals(0, endpoint.getIndividuallyTrackedRequestCount());
        assertEquals(0, endpoint.realPendingCount());
    }

    @Test
    void workerStatusSettlesRequestByRequestIdRegardlessOfBatchId() {
        assertTrue(TestCapacityAdmission.commitRouteRequest(endpoint, 77L, 1_000, 4));

        TaskInfo finished = taskInfo(77L, 123_456L, null);
        updateWorkerStatus(Map.of("77", finished), Map.of());
        updateWorkerStatus(Map.of("77", finished), Map.of());

        assertEquals(0, endpoint.getLocallyOwnedRequestCount());
        assertEquals(0, endpoint.getIndividuallyTrackedRequestCount());
    }

    @Test
    void runningObservationRefreshesRequestInactivityTtl() throws Exception {
        assertTrue(TestCapacityAdmission.commitRouteRequest(endpoint, 88L, 1_000, 4));
        Thread.sleep(25);

        TaskInfo running = taskInfo(88L, -1L, TaskPhase.RUNNING);
        updateWorkerStatus(Map.of(), Map.of("88", running));

        assertEquals(0, endpoint.evictExpiredRequests(10));
        assertEquals(1, endpoint.getIndividuallyTrackedRequestCount());
    }

    @Test
    void requestTtlEvictionReleasesCapacityExactlyOnce() throws Exception {
        assertTrue(TestCapacityAdmission.commitRouteRequest(endpoint, 99L, 1_000, 1));
        Thread.sleep(20);

        assertEquals(1, endpoint.evictExpiredInflight(1));
        assertEquals(0, endpoint.evictExpiredInflight(1));
        assertEquals(0, endpoint.getLocallyOwnedRequestCount());
        assertEquals(0, endpoint.getIndividuallyTrackedRequestCount());
        assertTrue(TestCapacityAdmission.commitRouteRequest(endpoint, 100L, 1_000, 1));
    }

    @Test
    void engineFenceProtectionPinsExpiredRequestUntilExplicitEnd() {
        assertTrue(TestCapacityAdmission.commitRouteRequest(endpoint, 109L, 1_000, 1));
        assertTrue(endpoint.beginEngineFenceProtection(109L));
        assertTrue(endpoint.beginEngineFenceProtection(109L),
                "begin is idempotent for the same live request");

        assertEquals(0, endpoint.evictExpiredRequests(-1),
                "even an immediately-expired entry remains owned by the fence");
        assertEquals(1, endpoint.getIndividuallyTrackedRequestCount());

        assertTrue(endpoint.endEngineFenceProtection(109L));
        assertFalse(endpoint.endEngineFenceProtection(109L));
        assertEquals(1, endpoint.evictExpiredRequests(-1));
        assertEquals(0, endpoint.getLocallyOwnedRequestCount());
    }

    @Test
    void releaseAndWorkerTerminalClearEngineFenceProtectionExactlyOnce() {
        assertTrue(TestCapacityAdmission.commitRouteRequest(endpoint, 110L, 1_000, 2));
        assertTrue(endpoint.beginEngineFenceProtection(110L));
        assertTrue(endpoint.releaseRequest(110L));
        assertFalse(endpoint.endEngineFenceProtection(110L));

        assertTrue(TestCapacityAdmission.commitRouteRequest(endpoint, 111L, 1_000, 2));
        assertTrue(endpoint.beginEngineFenceProtection(111L));
        updateWorkerStatus(Map.of(
                "111", taskInfo(111L, -1L, TaskPhase.RUNNING)), Map.of());

        assertFalse(endpoint.endEngineFenceProtection(111L));
        assertEquals(0, endpoint.getLocallyOwnedRequestCount());
        assertEquals(0, endpoint.getIndividuallyTrackedRequestCount());
    }

    @Test
    void pendingCountDoesNotDoubleCountTrackedEngineRequest() {
        assertTrue(TestCapacityAdmission.commitRouteRequest(endpoint, 301L, 1_000, 4));

        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setFinishedTaskInfo(Map.of());
        response.setRunningTaskInfo(Map.of(
                "301", taskInfo(301L, -1L, TaskPhase.RUNNING),
                "302", taskInfo(302L, -1L, TaskPhase.RUNNING)));
        endpoint.onWorkerStatusUpdate(endpoint.getStatus(), response);

        assertEquals(2, endpoint.realPendingCount(),
                "one local request plus one distinct Engine-only request");
    }

    @Test
    void waitEstimateAndMetricsIncludeRequestLedger() throws Exception {
        assertTrue(TestCapacityAdmission.commitRouteRequest(endpoint, 401L, 2_000, 4));
        assertEquals(2_000, endpoint.realWaitTimeMs());
        Thread.sleep(20);

        endpoint.reportBatchMetrics(reporter);

        assertEquals(0, monitor.value(INFLIGHT_BATCH_COUNT));
        assertEquals(1, monitor.value(INFLIGHT_REQUEST_COUNT));
        assertTrue(monitor.value(INFLIGHT_MAX_AGE_MS) >= 10);
    }

    @Test
    void requestWaitAggregateUpdatesWithoutScanningEntryState() throws Exception {
        assertTrue(TestCapacityAdmission.commitRouteRequest(endpoint, 501L, 1_000, 4));
        assertTrue(TestCapacityAdmission.commitRouteRequest(endpoint, 502L, 2_000, 4));
        assertEquals(3_000, endpoint.realWaitTimeMs());

        assertTrue(endpoint.releaseRequest(501L));
        assertEquals(2_000, endpoint.realWaitTimeMs());

        updateWorkerStatus(Map.of(), Map.of(
                "502", taskInfo(502L, -1L, TaskPhase.RUNNING)));
        Thread.sleep(20);
        assertTrue(endpoint.realWaitTimeMs() < 2_000,
                "observed execution progress should reduce the aggregate wait estimate");
    }

    @Test
    void waitSnapshotRetriesAcrossRunningToQueuedTransfer() throws Exception {
        assertTrue(TestCapacityAdmission.commitRouteRequest(endpoint, 503L, 1_000, 4));
        updateWorkerStatus(Map.of(), Map.of(
                "503", taskInfo(503L, -1L, TaskPhase.RUNNING)));

        CountDownLatch queuedContributionRead = new CountDownLatch(1);
        CountDownLatch resumeReader = new CountDownLatch(1);
        AtomicBoolean pauseOnce = new AtomicBoolean();
        endpoint.setWaitSnapshotHookForTest(stage -> {
            if (stage == PrefillEndpoint.WaitSnapshotStage.AFTER_REQUEST_QUEUED_READ
                    && pauseOnce.compareAndSet(false, true)) {
                queuedContributionRead.countDown();
                await(resumeReader);
            }
        });

        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<Long> waitRead = executor.submit(endpoint::realWaitTimeMs);
            assertTrue(queuedContributionRead.await(5, TimeUnit.SECONDS));

            // The reader has observed queued=0. Move the same request from the
            // running aggregate back to its stripe before it reads running=0.
            updateWorkerStatus(Map.of(), Map.of(
                    "503", taskInfo(503L, -1L, TaskPhase.PENDING)));
            resumeReader.countDown();

            assertTrue(waitRead.get(5, TimeUnit.SECONDS) > 0,
                    "the torn zero snapshot must be rejected and retried");
        } finally {
            resumeReader.countDown();
            endpoint.setWaitSnapshotHookForTest(null);
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
        }
    }

    @Test
    void staleCachePublicationCannotOverwriteCommitInvalidation() throws Exception {
        CountDownLatch validatedEmptySnapshot = new CountDownLatch(1);
        CountDownLatch resumePublisher = new CountDownLatch(1);
        AtomicBoolean pauseOnce = new AtomicBoolean();
        endpoint.setWaitSnapshotHookForTest(stage -> {
            if (stage == PrefillEndpoint.WaitSnapshotStage.BEFORE_CACHE_PUBLISH
                    && pauseOnce.compareAndSet(false, true)) {
                validatedEmptySnapshot.countDown();
                await(resumePublisher);
            }
        });

        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<Long> staleRead = executor.submit(endpoint::realWaitTimeMs);
            assertTrue(validatedEmptySnapshot.await(5, TimeUnit.SECONDS));

            assertTrue(TestCapacityAdmission.commitRouteRequest(endpoint, 504L, 1_000, 4));
            resumePublisher.countDown();
            assertTrue(staleRead.get(5, TimeUnit.SECONDS) > 0,
                    "the combined snapshot must reject a late stale cache publication");
            endpoint.setWaitSnapshotHookForTest(null);
            assertTrue(endpoint.realWaitTimeMs() > 0,
                    "a subsequent read must reject the stale zero cache");
        } finally {
            resumePublisher.countDown();
            endpoint.setWaitSnapshotHookForTest(null);
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
        }
    }

    @Test
    void runningWaitDoesNotDependOnRequestIdStripe() throws Exception {
        // 801 and 833 share the low five hash bits; 901 and 902 do not.
        long sameStripeWaitMs = observeRunningPairWait(801L, 833L);
        long differentStripeWaitMs = observeRunningPairWait(901L, 902L);

        assertTrue(Math.abs(sameStripeWaitMs - differentStripeWaitMs) < 100,
                "equivalent running work must not age once per occupied hash stripe: same="
                        + sameStripeWaitMs + ", different=" + differentStripeWaitMs);
    }

    @Test
    void runningServiceCreditStaysWithOldestRequestWhenItFinishesFirst() {
        AtomicLong clock = new AtomicLong();
        PrefillRequestLedger ledger = requestLedger(clock);
        assertTrue(commitQueueRoute(ledger, 1L, 1_000, 0));
        assertEquals(PrefillRequestLedger.ProgressOwnership.QUEUE_ROUTE,
                ledger.observe(1L, true, 0));
        clock.set(900);
        assertTrue(commitQueueRoute(ledger, 2L, 1_000, 0));
        assertEquals(PrefillRequestLedger.ProgressOwnership.QUEUE_ROUTE,
                ledger.observe(2L, true, 900));
        clock.set(950);
        assertTrue(ledger.settle(1L));

        assertEquals(1_000, ledger.estimate(950),
                "the old request's 950ms service credit must not consume newer work");
    }

    @Test
    void removingNewerRunningRequestPreservesOldestRemainder() {
        AtomicLong clock = new AtomicLong();
        PrefillRequestLedger ledger = requestLedger(clock);
        assertTrue(commitQueueRoute(ledger, 1L, 1_000, 0));
        assertEquals(PrefillRequestLedger.ProgressOwnership.QUEUE_ROUTE,
                ledger.observe(1L, true, 0));
        clock.set(900);
        assertTrue(commitQueueRoute(ledger, 2L, 1_000, 0));
        assertEquals(PrefillRequestLedger.ProgressOwnership.QUEUE_ROUTE,
                ledger.observe(2L, true, 900));
        clock.set(950);
        assertTrue(ledger.settle(2L));

        assertEquals(50, ledger.estimate(950),
                "removing non-head work must leave the serviced head remainder");
    }

    @Test
    void runningServiceCreditCanCrossMultipleEntries() {
        AtomicLong clock = new AtomicLong();
        PrefillRequestLedger ledger = requestLedger(clock);
        assertTrue(commitQueueRoute(ledger, 1L, 100, 0));
        assertTrue(commitQueueRoute(ledger, 2L, 200, 0));
        assertTrue(commitQueueRoute(ledger, 3L, 300, 0));
        assertEquals(PrefillRequestLedger.ProgressOwnership.QUEUE_ROUTE,
                ledger.observe(1L, true, 0));
        assertEquals(PrefillRequestLedger.ProgressOwnership.QUEUE_ROUTE,
                ledger.observe(2L, true, 0));
        assertEquals(PrefillRequestLedger.ProgressOwnership.QUEUE_ROUTE,
                ledger.observe(3L, true, 0));
        clock.set(250);
        assertTrue(ledger.settle(3L));

        assertEquals(50, ledger.estimate(250),
                "250ms of service must exhaust the first and consume 150ms of the second");
    }

    @Test
    void exhaustedRunningEntryCanBeRemovedIdempotently() {
        AtomicLong clock = new AtomicLong();
        PrefillRequestLedger ledger = requestLedger(clock);
        assertTrue(commitQueueRoute(ledger, 1L, 100, 0));
        assertTrue(commitQueueRoute(ledger, 2L, 200, 0));
        assertEquals(PrefillRequestLedger.ProgressOwnership.QUEUE_ROUTE,
                ledger.observe(1L, true, 0));
        assertEquals(PrefillRequestLedger.ProgressOwnership.QUEUE_ROUTE,
                ledger.observe(2L, true, 0));
        clock.set(150);
        assertTrue(commitQueueRoute(ledger, 3L, 100, 0));
        assertEquals(PrefillRequestLedger.ProgressOwnership.QUEUE_ROUTE,
                ledger.observe(3L, true, 150));
        assertEquals(250, ledger.estimate(150));

        assertTrue(ledger.settle(1L));
        assertFalse(ledger.settle(1L));

        assertEquals(250, ledger.estimate(150),
                "removing an already-exhausted entry must not unlink another request");
    }

    @Test
    void shuffledTenThousandRunningRemovalsUnlinkExactlyOnce() {
        AtomicLong clock = new AtomicLong();
        PrefillRequestLedger ledger = requestLedger(clock);
        List<Long> requests = new ArrayList<>(10_000);
        for (int i = 0; i < 10_000; i++) {
            long requestId = i + 1L;
            requests.add(requestId);
            assertTrue(commitQueueRoute(ledger, requestId, 1_000_000, 0));
            assertEquals(PrefillRequestLedger.ProgressOwnership.QUEUE_ROUTE,
                    ledger.observe(requestId, true, 0));
        }
        Collections.shuffle(requests, new Random(0x5eedL));

        for (long requestId : requests) {
            assertTrue(ledger.settle(requestId));
            assertFalse(ledger.settle(requestId));
        }

        assertEquals(0, ledger.estimate(0));
        assertEquals(0, ledger.count());
    }

    @Test
    void runningTransitionAndTtlEvictionCannotDoubleRelease() throws Exception {
        ExecutorService executor = Executors.newFixedThreadPool(2);
        try {
            for (long requestId = 20_000; requestId < 20_050; requestId++) {
                assertTrue(TestCapacityAdmission.commitRouteRequest(endpoint, requestId, 1_000, 1));
                Thread.sleep(2);

                CountDownLatch start = new CountDownLatch(1);
                CountDownLatch done = new CountDownLatch(2);
                long id = requestId;
                executor.execute(() -> {
                    await(start);
                    updateWorkerStatus(Map.of(), Map.of(
                            Long.toString(id), taskInfo(id, -1L, TaskPhase.RUNNING)));
                    done.countDown();
                });
                executor.execute(() -> {
                    await(start);
                    endpoint.evictExpiredRequests(1);
                    done.countDown();
                });
                start.countDown();
                assertTrue(done.await(5, TimeUnit.SECONDS));

                int live = endpoint.getIndividuallyTrackedRequestCount();
                assertTrue(live == 0 || live == 1);
                assertEquals(live, endpoint.getLocallyOwnedRequestCount());
                if (live == 1) {
                    assertTrue(endpoint.releaseRequest(id));
                }
                assertEquals(0, endpoint.getIndividuallyTrackedRequestCount());
                assertEquals(0, endpoint.getLocallyOwnedRequestCount());
            }
        } finally {
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
        }
    }

    @Test
    void directRequestPredictionKeepsExistingAgingSemantics() throws Exception {
        endpoint.registerDirectRequest(700L, 1_000);
        Thread.sleep(20);

        long waitMs = endpoint.realWaitTimeMs();
        assertTrue(waitMs > 0 && waitMs < 1_000);
    }

    @Test
    void oldRunningWorkDoesNotConsumeNewQueuedRequestPrediction() throws Exception {
        assertTrue(TestCapacityAdmission.commitRouteRequest(endpoint, 601L, 20, 4));
        updateWorkerStatus(Map.of(), Map.of(
                "601", taskInfo(601L, -1L, TaskPhase.RUNNING)));
        Thread.sleep(40);

        assertTrue(TestCapacityAdmission.commitRouteRequest(endpoint, 602L, 5_000, 4));
        assertTrue(endpoint.realWaitTimeMs() >= 4_900,
                "elapsed time before a queued request arrived must not be charged to it");
    }

    @Test
    void agedDirectRequestDoesNotConsumeNewRequestPrediction() throws Exception {
        endpoint.registerDirectRequest(800L, 20);
        Thread.sleep(40);

        assertTrue(TestCapacityAdmission.commitRouteRequest(endpoint, 701L, 5_000, 4));
        assertTrue(endpoint.realWaitTimeMs() >= 4_900,
                "request progress anchors must remain independent per entry");
    }

    private void updateWorkerStatus(Map<String, TaskInfo> finished,
                                    Map<String, TaskInfo> running) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setFinishedTaskInfo(finished);
        response.setRunningTaskInfo(running);
        endpoint.onWorkerStatusUpdate(endpoint.getStatus(), response);
    }

    private long observeRunningPairWait(long firstRequestId,
                                        long secondRequestId) throws Exception {
        assertTrue(TestCapacityAdmission.commitRouteRequest(endpoint, firstRequestId, 10_000, 4));
        assertTrue(TestCapacityAdmission.commitRouteRequest(endpoint, secondRequestId, 10_000, 4));
        updateWorkerStatus(Map.of(), Map.of(
                Long.toString(firstRequestId),
                taskInfo(firstRequestId, -1L, TaskPhase.RUNNING),
                Long.toString(secondRequestId),
                taskInfo(secondRequestId, -1L, TaskPhase.RUNNING)));
        Thread.sleep(200);
        long waitMs = endpoint.realWaitTimeMs();
        assertTrue(endpoint.releaseRequest(firstRequestId));
        assertTrue(endpoint.releaseRequest(secondRequestId));
        return waitMs;
    }

    private BatchItem batchItem(long requestId) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(512);

        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(new FlexlbConfig());

        ServerStatus prefill = new ServerStatus();
        prefill.setRole(RoleType.PREFILL);
        prefill.setServerIp(endpoint.getIp());
        prefill.setHttpPort(endpoint.getHttpPort());
        prefill.setGrpcPort(endpoint.getGrpcPort());
        DebugInfo debugInfo = new DebugInfo();
        debugInfo.setHitCacheLen(0);
        prefill.setDebugInfo(debugInfo);

        return new BatchItem(context, null, null, prefill, null,
                endpoint, null, System.currentTimeMillis());
    }

    private static TaskInfo taskInfo(long requestId, long batchId, TaskPhase phase) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setBatchId(batchId);
        task.setPhase(phase);
        task.setErrorCode(0);
        return task;
    }

    private static void await(CountDownLatch latch) {
        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private static PrefillRequestLedger requestLedger(AtomicLong clock) {
        return new PrefillRequestLedger(() -> {}, clock::get, ignored -> {});
    }

    private static boolean commitQueueRoute(
            PrefillRequestLedger ledger,
            long requestId,
            long predictedMs,
            int maximumInflightRequests) {
        PrefillRequestLedger.RequestCapacityReservationAcquisition acquisition =
                ledger.acquireCapacityReservation(
                        requestId, predictedMs, maximumInflightRequests);
        if (acquisition.status()
                != PrefillRequestLedger.RequestCapacityReservationAcquisition.Status.ACQUIRED) {
            return false;
        }
        PrefillRequestLedger.RequestCapacityReservation reservation =
                acquisition.reservation();
        if (!reservation.prepareForDelivery()) {
            reservation.release();
            return false;
        }
        reservation.completePreparedDeliveryTransfer();
        return true;
    }

    private static DecisionGroupHandler noopHandler() {
        return new DecisionGroupHandler() {
            @Override public void onExpired(BatchItem head) {}
            @Override public void onDecisionGroupAdmitted(
                    AdmittedDecisionGroup group,
                    DecisionGroupMetadata meta) {
                TestCapacityAdmission.complete(group);
            }
            @Override public void onOfferFailure(BatchItem item, Throwable error) {}
            @Override public void onDeliveryFailure(BatchItem item, Throwable error) {}
        };
    }

    private static final class RecordingMonitor implements FlexMonitor {
        private final Map<String, Double> values = new HashMap<>();

        @Override
        public void register(String metricName, FlexMetricType metricType) {}

        @Override
        public void register(String metricName,
                             FlexMetricType metricType,
                             FlexPriorityType priorityType) {}

        @Override
        public void register(String metricName, FlexMetricType metricType, int statisticsType) {}

        @Override
        public void report(String metricName, double value) {
            values.put(metricName, value);
        }

        @Override
        public void report(String metricName, FlexMetricTags metricsTags, double value) {
            values.put(metricName, value);
        }

        long value(String metricName) {
            return values.getOrDefault(metricName, -1.0).longValue();
        }
    }
}
