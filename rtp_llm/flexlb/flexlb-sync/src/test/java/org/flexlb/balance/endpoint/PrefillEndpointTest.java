package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.scheduler.DeliveryStrategy;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.RequestContext.DeliveryClaim;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.balance.scheduler.AbstractRequestScheduler;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
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
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.flexlb.balance.scheduler.SchedulingTestConfig.freezeInputs;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

class PrefillEndpointTest {

    private PrefillEndpoint endpoint;
    private FlexlbConfig config;
    private DeliveryMetricsReporter endpointReporter;
    private EndpointTestSupport.TestRequestRuntime requestRuntime;

    @BeforeEach
    void setUp() {
        WorkerStatus status = EndpointTestSupport.workerStatus(
                RoleType.PREFILL, "127.0.0.1", 8080, 8090);

        config = org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig();
        configureBatch(config, config.decisionPolicy().getMaxRequests(), 300, null);
        setFormula(config, "10 + 0.1*sum(computeTokens) + 5*batchSize");

        endpointReporter = mock(DeliveryMetricsReporter.class);
        requestRuntime = EndpointTestSupport.requestRuntime();
        endpoint = EndpointTestSupport.prefill(status, config, EndpointTestSupport.routeStrategy(requestRuntime), org.flexlb.balance.scheduler.SchedulerTestSupport.repository(requestRuntime.events()), endpointReporter);
    }

    @AfterEach
    void tearDown() {
        endpoint.close();
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void staleOrExpiredQueuedSelectionDoesNotAcquireGenerationHandoff(boolean expired) {
        var local = org.mockito.Mockito.spy(EndpointTestSupport.unstartedPrefill(config,
                EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.7", 8080, 8090),
                EndpointTestSupport.routeStrategy(requestRuntime), requestRuntime.events()));
        var state = (PrefillState) org.springframework.test.util.ReflectionTestUtils.getField(local, "prefillState");
        var queued = org.mockito.Mockito.spy(createRequestRoute(local, config, 999L, 100L, 0L));
        org.flexlb.balance.scheduler.SchedulerTestSupport.bindOwner(queued.ctx(), requestRuntime.events());
        var selected = expired ? queued : createRequestRoute(local, config, 999L, 100L, 0L);
        long nowMs = System.currentTimeMillis();
        if (expired) { org.mockito.Mockito.doReturn(true).when(queued).requestExpired(nowMs); }
        try {
            state.ownershipLock().lock();
            try {
                assertTrue(state.enqueueActiveLocked(queued, 0L));
                assertNull(local.commitQueuedRoutesLocked(List.of(selected), new long[]{10L}, null, nowMs));
                assertEquals(List.of(queued), state.captureQueue(2).items());
                WorkSnapshot committed = state.committedSnapshot();
                assertFalse(committed.containsRequest(999L));
                assertEquals(0L, committed.totalRemainingWorkMs().orElseThrow());
            } finally { state.ownershipLock().unlock(); }
            org.mockito.Mockito.verify(local, org.mockito.Mockito.never()).tryBeginRouteCommitAdmission();
            try (var admission = local.tryBeginRouteCommitAdmission()) { org.junit.jupiter.api.Assertions.assertNotNull(admission); }
        } finally { local.close(); }
    }

    @Test
    void directProjectionReusesCacheAndTracksWorkWithoutCreatingABatcher() {
        FlexlbConfig directConfig = org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig();
        directConfig.setScheduler(org.flexlb.config.SchedulerConfig.direct());
        directConfig.setDispatcher(DispatcherConfig.nonBatch());
        PrefillEndpoint direct = routeEndpoint(directConfig, "127.0.0.8");
        try {
            assertNull(EndpointTestSupport.batcher(direct));
            var empty = direct.captureRouteProjectionInputs();
            assertFalse(empty.queue().queueScheduling());
            assertSame(empty, direct.captureRouteProjectionInputs());
            RequestRoute item = createRequestRoute(direct, directConfig, 999L, 100, 0);
            {
                var reservation = EndpointTestSupport.reserveUnqueued(direct, item, 100L);
                try (var preparationReservation = EndpointTestSupport.preparation(reservation)) {
                    var reserved = direct.captureRouteProjectionInputs();
                    assertNotSame(empty, reserved);
                    assertSame(reserved, direct.captureRouteProjectionInputs());
                    assertTrue(reserved.work().containsRequest(999L));
                    assertEquals(0, direct.queuedRequestCount());
                    direct.signalRouteReady();
                    var changed = direct.captureRouteProjectionInputs();
                    assertNotSame(reserved, changed);
                    assertSame(changed.work(), reserved.work());
                }
            }
            assertFalse(direct.captureRouteProjectionInputs().work().containsRequest(999L));
            assertEquals(0, direct.admissionSummary(0).occupiedRequests());
        } finally {
            direct.close();
        }
    }

    // ---- batch commit / release ----

    @Test
    void nonBatchPublishOwnsOneExactQueuedIdentityUntilRemoval() {
        FlexlbConfig routeConfig = org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig();
        routeConfig.setDispatcher(DispatcherConfig.nonBatch());
        PrefillEndpoint routeEndpoint = routeEndpoint(routeConfig, "127.0.0.4");
        try {
            assertEquals(0, routeEndpoint.queuedRequestCount());

            RequestRoute queued = createRequestRoute(
                    routeEndpoint, routeConfig, 1L, 500, 200);
            assertTrue(EndpointTestSupport.offer(routeEndpoint, queued));
            assertEquals(1, routeEndpoint.queuedRequestCount());

            assertTrue(routeEndpoint.removeQueued(queued, "test cleanup"));
            assertEquals(0, routeEndpoint.queuedRequestCount());
        } finally {
            routeEndpoint.close();
        }
    }

    private PrefillEndpoint routeEndpoint(FlexlbConfig routeConfig, String ip) {
        PrefillEndpoint routeEndpoint = EndpointTestSupport.prefill(EndpointTestSupport.workerStatus(
                        RoleType.PREFILL, ip, 8080, 8090), routeConfig, EndpointTestSupport.routeStrategy(requestRuntime), org.flexlb.balance.scheduler.SchedulerTestSupport.repository(requestRuntime.events()), endpointReporter);
        return routeEndpoint;
    }

    @Test
    void unchangedFullStatusReusesProjectionInputs() {
        FlexlbConfig directConfig = org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig();
        directConfig.setScheduler(org.flexlb.config.SchedulerConfig.direct());
        directConfig.setDispatcher(DispatcherConfig.nonBatch());
        PrefillEndpoint direct = routeEndpoint(directConfig, "127.0.0.9");
        try {
            var before = direct.captureRouteProjectionInputs();
            WorkerStatusResponse response = new WorkerStatusResponse();
            response.setAlive(true);
            response.setRunningTaskInfo(Map.of());
            response.setFinishedTaskInfo(Map.of());
            EndpointTestSupport.applyStatus(direct, response).run();
            assertSame(before, direct.captureRouteProjectionInputs());
        } finally { direct.close(); }
    }

    @Test
    void initializationRejectsAnotherGenerationBeforeChangingProjection() {
        var before = endpoint.captureRouteProjectionInputs();
        WorkerStatus other = EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.2", 8080, 8090);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setAlive(true);
        response.setRunningTaskInfo(Map.of("999", taskInfo(999L, 0L, TaskPhase.RUNNING, 0, 0L)));
        response.setFinishedTaskInfo(Map.of());
        assertThrows(IllegalArgumentException.class,
                () -> endpoint.initializeFromPreparedStatus(endpoint.getStatus(), other.freezeStatusResponse(response)));
        assertFalse(endpoint.isGenerationRetiringOrRetired());
        assertSame(before, endpoint.captureRouteProjectionInputs());
    }

    @Test
    void commitBatchIncreasesInflightCount() {
        assertEquals(0, endpoint.ownershipStats().batchCount());
        WorkSnapshot empty = endpoint.captureRouteProjectionInputs().work();

        RequestRoute item = createRequestRoute(1L, 500, 200);
        registerBatch(endpoint, 1L, 100, List.of(item));

        assertEquals(1, endpoint.ownershipStats().batchCount());
        WorkSnapshot committed = endpoint.captureRouteProjectionInputs().work();
        assertTrue(committed.containsRequest(1L));
        assertEquals(100L, committed.totalRemainingWorkMs().orElseThrow());
        assertFalse(empty.containsRequest(1L));
        assertEquals(0L, empty.totalRemainingWorkMs().orElseThrow());
        assertEquals(1, endpoint.admissionSummary(0).occupiedRequests());
    }

    @Test
    void releaseBatchDecreasesInflightCount() {
        RequestRoute item = createRequestRoute(1L, 500, 200);
        registerBatch(endpoint, 1L, 100, List.of(item));
        WorkSnapshot committed = endpoint.captureRouteProjectionInputs().work();
        assertTrue(endpoint.releaseRequest(item));

        assertEquals(0, endpoint.ownershipStats().batchCount());
        WorkSnapshot released = endpoint.captureRouteProjectionInputs().work();
        assertFalse(released.containsRequest(1L));
        assertEquals(0L, released.totalRemainingWorkMs().orElseThrow());
        assertTrue(committed.containsRequest(1L));
        assertEquals(100L, committed.totalRemainingWorkMs().orElseThrow());
    }

    @Test
    void expireBatchMemberReleasesOnlyItsExactOwnership() {
        RequestRoute first = createRequestRoute(101L, 500, 200);
        RequestRoute sibling = createRequestRoute(102L, 300, 100);
        registerBatch(endpoint, 7L, 100, List.of(first, sibling));
        WorkSnapshot committed = endpoint.captureRouteProjectionInputs().work();
        assertEquals(2, endpoint.ownershipStats().locallyOwnedRequests());
        assertEquals(1, endpoint.ownershipStats().batchCount());

        assertTrue(endpoint.releaseRequest(first));
        assertFalse(endpoint.releaseRequest(first));
        assertEquals(1, endpoint.ownershipStats().locallyOwnedRequests());
        assertEquals(1, endpoint.ownershipStats().batchCount());
        assertEquals(1, endpoint.admissionSummary(0).occupiedRequests());
        WorkSnapshot remaining = endpoint.captureRouteProjectionInputs().work();
        assertFalse(remaining.containsRequest(101L));
        assertTrue(remaining.containsRequest(102L));
        assertEquals(100L, remaining.totalRemainingWorkMs().orElseThrow());
        assertTrue(committed.containsRequest(101L));
        assertTrue(committed.containsRequest(102L));
        assertEquals(100L, committed.totalRemainingWorkMs().orElseThrow());

        assertTrue(endpoint.releaseRequest(sibling));
        assertFalse(endpoint.releaseRequest(sibling));
        assertEquals(0, endpoint.ownershipStats().locallyOwnedRequests());
        assertEquals(0, endpoint.ownershipStats().batchCount());
        WorkSnapshot released = endpoint.captureRouteProjectionInputs().work();
        assertFalse(released.containsRequest(101L));
        assertFalse(released.containsRequest(102L));
        assertEquals(0L, released.totalRemainingWorkMs().orElseThrow());
        assertTrue(remaining.containsRequest(102L));
        assertEquals(100L, remaining.totalRemainingWorkMs().orElseThrow());
        assertEquals(0, endpoint.admissionSummary(0).occupiedRequests());
    }

    @Test
    void staleExpirationCannotReleaseReusedRequestId() {
        RequestRoute original = createRequestRoute(101L, 500, 200);
        registerBatch(endpoint, 7L, 100, List.of(original));
        assertTrue(endpoint.releaseRequest(original));

        RequestRoute replacement = createRequestRoute(101L, 300, 100);
        registerBatch(endpoint, 8L, 100, List.of(replacement));
        assertFalse(endpoint.releaseRequest(original));
        assertEquals(1, endpoint.ownershipStats().batchCount());
        assertEquals(1, endpoint.admissionSummary(0).occupiedRequests());
        assertTrue(endpoint.releaseRequest(replacement));
        assertEquals(0, endpoint.ownershipStats().batchCount());
    }

    @Test
    void commitMultipleBatches() {
        RequestRoute item1 = createRequestRoute(1L, 500, 200);
        RequestRoute item2 = createRequestRoute(2L, 300, 100);
        RequestRoute item3 = createRequestRoute(3L, 400, 0);

        registerBatch(endpoint, 1L, 100, List.of(item1, item2));
        registerBatch(endpoint, 2L, 50, List.of(item3));

        assertEquals(2, endpoint.ownershipStats().batchCount());
        assertEquals(3, endpoint.admissionSummary(0).occupiedRequests());
    }

    // ---- repack batch ----

    @Test
    void localMemberCleanupPreservesBatchPrediction() {
        RequestRoute item1 = createRequestRoute(1L, 500, 200);
        RequestRoute item2 = createRequestRoute(2L, 300, 100);
        registerBatch(endpoint, 1L, 100, List.of(item1, item2));

        assertTrue(endpoint.releaseRequest(item2));
        assertEquals(1, endpoint.ownershipStats().batchCount());
        assertEquals(1, endpoint.admissionSummary(0).occupiedRequests());
        WorkSnapshot remaining = endpoint.captureRouteProjectionInputs().work();
        assertTrue(remaining.containsRequest(1L));
        assertFalse(remaining.containsRequest(2L));
        assertFalse(remaining.hasUnknownWork());
        assertEquals(100L, remaining.totalRemainingWorkMs().orElseThrow());
    }

    @Test
    void repackBatchAllFailedReturnsNull() {
        RequestRoute item1 = createRequestRoute(1L, 500, 200);
        registerBatch(endpoint, 1L, 100, List.of(item1));

        assertTrue(endpoint.releaseRequest(item1));
        assertEquals(0, endpoint.ownershipStats().batchCount());
        WorkSnapshot released = endpoint.captureRouteProjectionInputs().work();
        assertFalse(released.containsRequest(1L));
        assertEquals(0L, released.totalRemainingWorkMs().orElseThrow());
    }

    @ParameterizedTest
    @CsvSource({"-1", "-0.1", "0/0", "1/0"})
    void invalidRepackPredictionUsesDefaultFormula(String formula) {
        PrefillEndpoint invalidPredictorEndpoint = newEndpointWithFormula(formula);
        try {
            RequestRoute survivor = createRequestRoute(
                    invalidPredictorEndpoint, 1L, 500L, 200L);
            RequestRoute failed = createRequestRoute(
                    invalidPredictorEndpoint, 2L, 300L, 100L);
            registerBatch(
                    invalidPredictorEndpoint,
                    1L,
                    100L,
                    List.of(survivor, failed));

            assertDoesNotThrow(() -> reportRejectedBatchMember(
                    invalidPredictorEndpoint, 1L, 2L));

            WorkSnapshot snapshot = invalidPredictorEndpoint
                    .captureRouteProjectionInputs().work();
            assertEquals(1, invalidPredictorEndpoint.admissionSummary(0).occupiedRequests(),
                    "membership settlement must not depend on prediction");
            assertTrue(snapshot.containsRequest(1L));
            assertFalse(snapshot.containsRequest(2L));
            assertEquals(360L, snapshot.totalRemainingWorkMs().orElseThrow());
            assertFalse(snapshot.hasUnknownWork());
            assertEquals(360L, invalidPredictorEndpoint.getLoadMetric().orElseThrow());

            reportSuccessfulBatchMember(
                    invalidPredictorEndpoint, 1L, 1L, 40L);
            assertEquals(0, invalidPredictorEndpoint.ownershipStats().batchCount(),
                    "fallback prediction must not block later lifecycle settlement");
            assertEquals(0, invalidPredictorEndpoint.admissionSummary(0).occupiedRequests());
        } finally {
            invalidPredictorEndpoint.close();
        }
    }

    @Test
    @org.junit.jupiter.api.Timeout(5)
    void repackReusesPredictionAcrossUnrelatedQueueMutation() {
        RequestRoute survivor = createRequestRoute(1L, 500L, 200L);
        RequestRoute finished = createRequestRoute(2L, 300L, 100L);
        registerBatch(endpoint, 1L, 100L, List.of(survivor, finished));
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        PrefillTimePredictor.Evaluator evaluator = mock(PrefillTimePredictor.Evaluator.class);
        org.mockito.Mockito.when(predictor.evaluator()).thenReturn(evaluator);
        var state = (PrefillState) org.springframework.test.util.ReflectionTestUtils.getField(endpoint, "prefillState");
        AtomicInteger predictions = new AtomicInteger();
        org.mockito.Mockito.when(evaluator.predictBatchMs(org.mockito.ArgumentMatchers.any())).thenAnswer(call -> {
            assertFalse(state.ownershipLock().isHeldByCurrentThread());
            int n = predictions.incrementAndGet();
            assertTrue(n <= 2, "unrelated mutations must not keep retrying prediction");
            assertTrue(EndpointTestSupport.offer(endpoint, createRequestRoute(100L + n, 100L, 0L)));
            return 55.0;
        });
        org.springframework.test.util.ReflectionTestUtils.setField(endpoint, "predictor", predictor);
        reportRejectedBatchMember(endpoint, 1L, 2L);
        assertEquals(1, predictions.get());
        WorkSnapshot remaining = endpoint.captureRouteProjectionInputs().work();
        assertTrue(remaining.containsRequest(1L));
        assertFalse(remaining.containsRequest(2L));
        assertEquals(55L, remaining.totalRemainingWorkMs().orElseThrow());
        assertTrue(endpoint.releaseRequest(survivor));
        assertEquals(0, endpoint.ownershipStats().batchCount());
    }

    @Test
    void rollbackPublishesOnlyActualCapacityReleaseOutsideStateLock() {
        var availability = mock(org.flexlb.balance.scheduler.PlacementAvailability.class);
        org.springframework.test.util.ReflectionTestUtils.setField(endpoint, "placementAvailability", availability);
        var state = (PrefillState) org.springframework.test.util.ReflectionTestUtils.getField(endpoint, "prefillState");
        AtomicInteger notifications = new AtomicInteger();
        org.mockito.Mockito.doAnswer(call -> {
            assertFalse(state.ownershipLock().isHeldByCurrentThread());
            notifications.incrementAndGet();
            return null;
        }).when(availability).changed(org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.any());
        var first = createRequestRoute(90L, 100L, 0L);
        PrefillState.RouteReservation open;
        try (var pin = endpoint.tryPinGeneration()) { open = endpoint.reserveUnqueuedRoute(pin, first, 10L).reservation(); }
        endpoint.rollbackReservation(open);
        endpoint.rollbackReservation(open);
        assertEquals(1, notifications.get());
        assertEquals(0L, endpoint.admissionSummary(0).occupiedRequests());
        var second = createRequestRoute(91L, 100L, 0L);
        PrefillState.RouteReservation consumed;
        try (var pin = endpoint.tryPinGeneration()) { consumed = endpoint.reserveUnqueuedRoute(pin, second, 10L).reservation(); }
        try (var commit = endpoint.tryBeginRouteCommitAdmission();
             var handoff = commit.commit(List.of(second), List.of(consumed))) {
            endpoint.rollbackReservation(consumed);
            assertEquals(1, notifications.get(), "successful admission releases no capacity");
        }
        assertTrue(endpoint.releaseRequest(second));
        assertEquals(2, notifications.get());
    }

    @Test
    void exactRequestReleaseConsumesDirectPreparationAndNotifiesOnlyOnceOutsideStateLock() {
        var availability = mock(org.flexlb.balance.scheduler.PlacementAvailability.class);
        org.springframework.test.util.ReflectionTestUtils.setField(endpoint, "placementAvailability", availability);
        var state = (PrefillState) org.springframework.test.util.ReflectionTestUtils.getField(endpoint, "prefillState");
        AtomicInteger notifications = new AtomicInteger();
        org.mockito.Mockito.doAnswer(call -> {
            assertFalse(state.ownershipLock().isHeldByCurrentThread());
            notifications.incrementAndGet();
            return null;
        }).when(availability).changed(org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.any());
        var first = createRequestRoute(90L, 100L, 0L);
        var preparation = EndpointTestSupport.reserveUnqueued(endpoint, first, 10L);
        assertTrue(endpoint.releaseRequest(first));
        assertFalse(endpoint.releaseRequest(first));
        endpoint.rollbackReservation(preparation);
        assertEquals(1, notifications.get());
        assertEquals(0L, endpoint.admissionSummary(0).occupiedRequests());

        var replacement = createRequestRoute(90L, 100L, 0L);
        var replacementPreparation = EndpointTestSupport.reserveUnqueued(endpoint, replacement, 20L);
        try (var commit = endpoint.tryBeginRouteCommitAdmission();
             var handoff = commit.commit(List.of(replacement), List.of(replacementPreparation))) {
            assertFalse(endpoint.releaseRequest(first));
            assertEquals(1L, endpoint.admissionSummary(0).occupiedRequests());
            assertTrue(endpoint.releaseRequest(replacement));
        }
        endpoint.rollbackReservation(replacementPreparation);
        assertEquals(2, notifications.get());
        assertEquals(0L, endpoint.admissionSummary(0).occupiedRequests());
    }

    @Test
    void throwingRepackPredictorUsesDefaultFormula() throws Exception {
        RequestRoute survivor = createRequestRoute(1L, 500L, 200L);
        RequestRoute finished = createRequestRoute(2L, 300L, 100L);
        registerBatch(endpoint, 1L, 100L, List.of(survivor, finished));
        PrefillTimePredictor failingPredictor = mock(PrefillTimePredictor.class);
        org.mockito.Mockito.when(failingPredictor.evaluator()).thenThrow(new IllegalStateException("broken predictor"));
        var field = PrefillEndpoint.class.getDeclaredField("predictor");
        field.setAccessible(true);
        field.set(endpoint, failingPredictor);

        assertDoesNotThrow(() -> reportRejectedBatchMember(endpoint, 1L, 2L));
        assertEquals(360L, endpoint.captureRouteProjectionInputs().work().totalRemainingWorkMs().orElseThrow());
        assertEquals(1L, endpoint.admissionSummary(0).occupiedRequests());
        assertTrue(endpoint.releaseRequest(survivor));
        assertEquals(0L, endpoint.admissionSummary(0).occupiedRequests());
        assertEquals(0L, endpoint.ownershipStats().batchCount());
    }

    @ParameterizedTest
    @CsvSource({"500,600,15", "500,-1,65", "-1,20,15"})
    void repackNormalizesInvalidFeatures(long seqLen, long hitCache, long expectedMs) {
        RequestRoute survivor = org.mockito.Mockito.spy(createRequestRoute(1L, 500L, 200L));
        RequestRoute finished = createRequestRoute(2L, 300L, 100L);
        var malformed = new java.util.concurrent.atomic.AtomicBoolean();
        org.mockito.Mockito.doAnswer(call -> malformed.get() ? seqLen : call.callRealMethod())
                .when(survivor).seqLen();
        org.mockito.Mockito.doAnswer(call -> malformed.get() ? hitCache : call.callRealMethod())
                .when(survivor).hitCache();
        registerBatch(endpoint, 1L, 100L, List.of(survivor, finished));
        malformed.set(true);

        assertDoesNotThrow(() -> reportRejectedBatchMember(endpoint, 1L, 2L));
        assertEquals(expectedMs, endpoint.captureRouteProjectionInputs().work().totalRemainingWorkMs().orElseThrow());
        assertTrue(endpoint.releaseRequest(survivor));
        assertEquals(0L, endpoint.admissionSummary(0).occupiedRequests());
    }

    // ---- calibrate ----

    @Test
    void calibrateRemovesBatchOnSuccess() {
        RequestRoute item = createRequestRoute(1L, 500, 200);
        registerBatch(endpoint, 1L, 100, List.of(item));

        Map<String, TaskInfo> finished = new HashMap<>();
        TaskInfo successTask = new TaskInfo();
        successTask.setRequestId(1L);
        successTask.setBatchId(1L);
        successTask.setErrorCode(0);
        finished.put("1", successTask);

        calibrate(finished, Map.of());

        assertEquals(0, endpoint.ownershipStats().batchCount());
    }

    @Test
    void completion_observer_failure_does_not_escape_finished_settlement() {
        registerBatch(endpoint, 9L, 100, List.of(createRequestRoute(9L, 500, 200)));
        doThrow(new IllegalStateException("metrics unavailable"))
                .when(endpointReporter)
                .reportBatchCompletion("127.0.0.1", 9L, 100L, 125L);

        TaskInfo finished = taskInfo(9L, 9L, null, 0, 125);
        assertDoesNotThrow(() -> calibrate(Map.of("9", finished), Map.of()));

        assertEquals(0, endpoint.ownershipStats().batchCount());
        assertEquals(0, endpoint.admissionSummary(0).occupiedRequests());
        verify(endpointReporter).reportBatchCompletion("127.0.0.1", 9L, 100L, 125L);
    }

    @Test
    void calibrateRepacksOnPartialFailure() {
        RequestRoute item1 = createRequestRoute(1L, 500, 200);
        RequestRoute item2 = createRequestRoute(2L, 300, 100);
        registerBatch(endpoint, 1L, 100, List.of(item1, item2));

        Map<String, TaskInfo> finished = new HashMap<>();
        TaskInfo failedTask = new TaskInfo();
        failedTask.setRequestId(2L);
        failedTask.setBatchId(1L);
        failedTask.setErrorCode(500);
        failedTask.setErrorMessage("engine error");
        finished.put("2", failedTask);

        calibrate(finished, Map.of());

        assertEquals(1, endpoint.ownershipStats().batchCount());
        assertEquals(1, endpoint.admissionSummary(0).occupiedRequests());
    }

    @Test
    void calibrateKeepsBatchInflightUntilEveryMemberFinishes() {
        RequestRoute shortItem = createRequestRoute(1L, 500, 200);
        RequestRoute longItem = createRequestRoute(2L, 10_000, 0);
        registerBatch(endpoint, 1L, 2_000, List.of(shortItem, longItem));

        TaskInfo finishedShort = taskInfo(1L, 1L, null, 0, 40);
        TaskInfo runningLong = taskInfo(2L, 1L, TaskPhase.RUNNING, 0, 0);
        calibrate(Map.of("1", finishedShort), Map.of("2", runningLong));

        assertEquals(1, endpoint.ownershipStats().batchCount(),
                "one finished member must not release the whole batch");
        assertEquals(1, endpoint.admissionSummary(0).occupiedRequests(),
                "the still-running long member must remain in Master accounting");

        TaskInfo finishedLong = taskInfo(2L, 1L, null, 0, 1_900);
        calibrate(Map.of("2", finishedLong), Map.of());

        assertEquals(0, endpoint.ownershipStats().batchCount());
        assertEquals(0, endpoint.admissionSummary(0).occupiedRequests());
    }

    @Test
    void calibrateMixedTerminalMembersKeepsOnlyRunningSurvivor() {
        RequestRoute succeeded = createRequestRoute(1L, 500, 200);
        RequestRoute failed = createRequestRoute(2L, 300, 100);
        RequestRoute running = createRequestRoute(3L, 10_000, 0);
        registerBatch(endpoint, 1L, 2_000, List.of(succeeded, failed, running));

        TaskInfo success = taskInfo(1L, 1L, null, 0, 40);
        TaskInfo failure = taskInfo(2L, 1L, null, 500, 50);
        TaskInfo runningTask = taskInfo(3L, 1L, TaskPhase.RUNNING, 0, 0);
        calibrate(Map.of("1", success, "2", failure), Map.of("3", runningTask));

        assertEquals(1, endpoint.ownershipStats().batchCount());
        assertEquals(1, endpoint.admissionSummary(0).occupiedRequests());

        // WorkerStatus may repeat a terminal observation in adjacent snapshots.
        // Repeating it must not decrement the survivor count again.
        calibrate(Map.of("1", success), Map.of("3", runningTask));
        assertEquals(1, endpoint.ownershipStats().batchCount());
        assertEquals(1, endpoint.admissionSummary(0).occupiedRequests());
    }

    @Test
    void calibrateAllFailuresClearsBatchIdempotentlyWithoutCompletionMetrics() {
        registerBatch(endpoint, 1L, 2_000, List.of(
                createRequestRoute(1L, 500, 200),
                createRequestRoute(2L, 10_000, 0)));

        TaskInfo firstFailure = taskInfo(1L, 1L, null, 500, 40);
        TaskInfo secondFailure = taskInfo(2L, 1L, null, 501, 50);
        calibrate(Map.of("1", firstFailure, "2", secondFailure), Map.of());

        assertEquals(0, endpoint.ownershipStats().batchCount());
        assertEquals(0, endpoint.admissionSummary(0).occupiedRequests());
        verify(endpointReporter, never()).reportBatchCompletion(
                anyString(), org.mockito.ArgumentMatchers.anyLong(),
                org.mockito.ArgumentMatchers.anyLong(), org.mockito.ArgumentMatchers.anyLong());

        calibrate(Map.of("1", firstFailure, "2", secondFailure), Map.of());
        assertEquals(0, endpoint.ownershipStats().batchCount());
        assertEquals(0, endpoint.admissionSummary(0).occupiedRequests(),
                "repeated failure deltas must not decrement the ledger twice");
    }

    @Test
    void repeatedSuccessfulTerminalReportsCompletionExactlyOnce() {
        registerBatch(endpoint, 1L, 100, List.of(createRequestRoute(1L, 500, 200)));
        TaskInfo success = taskInfo(1L, 1L, null, 0, 40);

        calibrate(Map.of("1", success), Map.of());
        calibrate(Map.of("1", success), Map.of());

        assertEquals(0, endpoint.admissionSummary(0).occupiedRequests());
        verify(endpointReporter).reportBatchCompletion("127.0.0.1", 1L, 100L, 40L);
    }

    @Test
    void batchInflightReanchorsAcrossRunningQueuedRunning() {
        registerBatch(endpoint, 1L, 5_000,
                List.of(createRequestRoute(1L, 500, 0)));
        assertEquals(WorkSnapshot.Phase.COMMITTED,
                batchServicePhase(endpoint, 1L));
        WorkSnapshot committed = endpoint.captureRouteProjectionInputs().work();
        assertEquals(5_000L, committed.totalRemainingWorkMs().orElseThrow());
        assertEquals(5_000L, committed.totalRemainingWorkMsAt(committed.capturedAtMs() + 1_000L).orElseThrow());

        calibrate(Map.of(), Map.of(
                "1", taskInfo(1L, 1L, TaskPhase.RUNNING, 0, 0)));
        assertEquals(WorkSnapshot.Phase.ENGINE_RUNNING,
                batchServicePhase(endpoint, 1L));
        WorkSnapshot running = endpoint.captureRouteProjectionInputs().work();
        long runningWorkMs = running.totalRemainingWorkMs().orElseThrow();
        assertEquals(Math.max(0L, runningWorkMs - 1_000L),
                running.totalRemainingWorkMsAt(running.capturedAtMs() + 1_000L).orElseThrow());

        calibrate(Map.of(), Map.of(
                "1", taskInfo(1L, 1L, TaskPhase.PENDING, 0, 0)));
        assertEquals(WorkSnapshot.Phase.ENGINE_QUEUED,
                batchServicePhase(endpoint, 1L));
        WorkSnapshot queued = endpoint.captureRouteProjectionInputs().work();
        assertEquals(queued.totalRemainingWorkMs().orElseThrow(),
                queued.totalRemainingWorkMsAt(queued.capturedAtMs() + 1_000L).orElseThrow());
        assertEquals(runningWorkMs, running.totalRemainingWorkMs().orElseThrow());
        assertEquals(Math.max(0L, runningWorkMs - 1_000L),
                running.totalRemainingWorkMsAt(running.capturedAtMs() + 1_000L).orElseThrow());

        calibrate(Map.of(), Map.of(
                "1", taskInfo(1L, 1L, TaskPhase.RUNNING, 0, 0)));
        assertEquals(WorkSnapshot.Phase.ENGINE_RUNNING,
                batchServicePhase(endpoint, 1L));
        WorkSnapshot resumed = endpoint.captureRouteProjectionInputs().work();
        long resumedWorkMs = resumed.totalRemainingWorkMs().orElseThrow();
        assertEquals(Math.max(0L, resumedWorkMs - 1_000L),
                resumed.totalRemainingWorkMsAt(resumed.capturedAtMs() + 1_000L).orElseThrow());
        assertEquals(5_000L, committed.totalRemainingWorkMs().orElseThrow());
    }

    @Test
    void batchInflightMaxAgeMeasuresTimeSinceLatestActivity()
            throws InterruptedException {
        // The refactor unified inflight age tracking onto a single
        // last-observation clock. A recent Engine observation both keeps the
        // canonical owner alive (eviction) and resets the reported max age; the
        // age then grows with the time elapsed since that last observation.
        registerBatch(endpoint, 1L, 5_000,
                List.of(createRequestRoute(1L, 500, 0)));
        Thread.sleep(20);
        calibrate(Map.of(), Map.of(
                "1", taskInfo(1L, 1L, TaskPhase.RUNNING, 0, 0)));

        assertEquals(0, endpoint.evictExpiredInflight(5, ignored -> false),
                "recent Engine activity keeps the canonical owner live");
        Thread.sleep(15);
        endpoint.reportBatchMetrics(endpointReporter);
        verify(endpointReporter).reportPrefillInflight(
                eq("127.0.0.1"), org.mockito.ArgumentMatchers.argThat(stats -> stats.maxObservedAgeMs() >= 10));
    }

    @Test
    void inflightMaxAgeMetricTracksTimeSinceLatestObservation()
            throws InterruptedException {
        // Reported max age measures staleness (time since the batch was last
        // observed), not wall-clock time since creation: a fresh RUNNING
        // observation resets it, after which it grows with elapsed idle time.
        registerBatch(endpoint, 1L, 5_000, List.of(createRequestRoute(1L, 500, 0)));
        calibrate(Map.of(), Map.of(
                "1", taskInfo(1L, 1L, TaskPhase.RUNNING, 0, 0)));
        Thread.sleep(30);

        endpoint.reportBatchMetrics(endpointReporter);

        verify(endpointReporter).reportPrefillInflight(
                eq("127.0.0.1"), org.mockito.ArgumentMatchers.argThat(stats -> stats.maxObservedAgeMs() >= 20));
    }

    @Test
    void runningObservationRefreshesBatchInactivityTtl() throws InterruptedException {
        RequestRoute longItem = createRequestRoute(1L, 10_000, 0);
        registerBatch(endpoint, 1L, 2_000, List.of(longItem));

        Thread.sleep(150);
        TaskInfo running = taskInfo(1L, 1L, TaskPhase.RUNNING, 0, 0);
        calibrate(Map.of(), Map.of("1", running));

        assertEquals(0, endpoint.evictExpiredInflight(100, ignored -> false),
                "an actively observed long-running batch must not be evicted by creation age");
        assertEquals(1, endpoint.ownershipStats().batchCount());
    }

    @Test
    void foreignRunningObservationDoesNotRefreshBatchInactivityTtl()
            throws InterruptedException {
        registerBatch(endpoint, 1L, 2_000, List.of(createRequestRoute(1L, 10_000, 0)));
        Thread.sleep(10);

        TaskInfo foreign = taskInfo(999L, 1L, TaskPhase.RUNNING, 0, 0);
        calibrate(Map.of(), Map.of("999", foreign));

        assertEquals(1, endpoint.evictExpiredInflight(1, ignored -> false));
        assertEquals(0, endpoint.ownershipStats().batchCount());
    }

    @Test
    void partialCompletionKeepsFixedWindowMaxInflightGateClosed() throws Exception {
        RequestRoute shortItem = createRequestRoute(101L, 500, 200);
        RequestRoute longItem = createRequestRoute(102L, 10_000, 0);
        registerBatch(endpoint, 700L, 2_000,
                List.of(shortItem, longItem));
        assertFalse(endpoint.batchAdmissionAvailability(1).isAvailable());

        calibrate(
                Map.of("101", taskInfo(101L, 700L, null, 0, 40)),
                Map.of("102", taskInfo(
                        102L, 700L, TaskPhase.RUNNING, 0, 0)));

        assertEquals(1, endpoint.ownershipStats().batchCount());
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
        RequestRoute item = createRequestRoute(1L, 500, 200);
        registerBatch(endpoint, 1L, 100, List.of(item));

        Map<String, TaskInfo> finished = new HashMap<>();
        TaskInfo badTask = new TaskInfo();
        badTask.setRequestId(999L); // non-colliding: won't match batchId=1
        badTask.setBatchId(-1);
        badTask.setErrorCode(0);
        finished.put("1", badTask);

        // should not throw, just log a warning for missing non-batch inflight
        calibrate(finished, Map.of());
        assertEquals(1, endpoint.ownershipStats().batchCount());
    }

    @Test
    void calibrateMissingBatchIdDoesNotRetireRealBatchMember() {
        registerBatch(endpoint, 700L, 100, List.of(createRequestRoute(101L, 500, 200)));

        calibrate(Map.of("101", priorityCanceledTask(101L, -1L)), Map.of());

        assertEquals(1, endpoint.ownershipStats().batchCount());
        assertEquals(1, endpoint.admissionSummary(0).occupiedRequests());
    }

    @Test
    void calibrateMissingBatchIdRemovesDirectRequestLedgerEntry() {
        registerDirect(endpoint, 101L, 100L);
        assertEquals(0, endpoint.ownershipStats().batchCount());
        assertEquals(1, endpoint.ownershipStats().individuallyOwnedRequests());

        TaskInfo finished = new TaskInfo();
        finished.setRequestId(101L);
        finished.setBatchId(-1L);
        finished.setErrorCode(0);
        calibrate(Map.of("101", finished), Map.of());

        assertEquals(0, endpoint.ownershipStats().individuallyOwnedRequests());
        assertEquals(0, endpoint.admissionSummary(0).occupiedRequests());
    }

    @Test
    void directRegistrationCanRollbackFromAsyncCompletionThread()
            throws Exception {
        PrefillState.RouteReservation registration =
                EndpointTestSupport.reserveUnqueued(endpoint, createRequestRoute(102L, 100, 0), 100L);
        assertEquals(1, endpoint.admissionSummary(0).occupiedRequests());

        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            executor.submit(() -> EndpointTestSupport.rollback(registration)).get(5, TimeUnit.SECONDS);
        } finally {
            executor.shutdownNow();
        }

        assertEquals(0, endpoint.admissionSummary(0).occupiedRequests());
    }

    @Test
    void calibrateMissingBatchIdLeavesMembersUntilExactBatchTerminal() {
        registerBatch(endpoint, 700L, 100, List.of(
                createRequestRoute(101L, 500, 200),
                createRequestRoute(102L, 300, 100)));

        calibrate(Map.of("101", priorityCanceledTask(101L, -1L)), Map.of());

        // A missing-batch-id terminal cannot be attributed to the batch, so no
        // member is retired: both members remain committed.
        assertEquals(1, endpoint.ownershipStats().batchCount());
        assertEquals(2, endpoint.admissionSummary(0).occupiedRequests(),
                "a terminal without a valid batch id retires no batch member");

        TaskInfo survivingSuccess = new TaskInfo();
        survivingSuccess.setRequestId(102L);
        survivingSuccess.setBatchId(700L);
        survivingSuccess.setErrorCode(0);
        calibrate(Map.of("102", survivingSuccess), Map.of());
        // The exact-batch terminal retires only its own member; member 101,
        // whose only terminal named no batch id, stays with the original batch.
        assertEquals(1, endpoint.ownershipStats().batchCount(),
                "the exact-batch terminal retires only its own member");
        assertEquals(1, endpoint.admissionSummary(0).occupiedRequests());
    }

    @Test
    void directRequestIdMatchingQueueBatchIdDoesNotOverwriteEitherLifecycle() {
        // DIRECT request 101 and QUEUE batch 101 live in different ledgers.
        // Completing the DIRECT request must not erase QUEUE member 201.
        registerBatch(endpoint, 101L, 100, List.of(createRequestRoute(201L, 500, 200)));
        registerDirect(endpoint, 101L, 100L);
        assertEquals(1, endpoint.ownershipStats().batchCount());
        WorkSnapshot committed = endpoint.captureRouteProjectionInputs().work();
        assertTrue(committed.containsRequest(101L));
        assertTrue(committed.containsRequest(201L));
        assertEquals(200L, committed.totalRemainingWorkMs().orElseThrow());
        assertEquals(1, endpoint.ownershipStats().individuallyOwnedRequests());

        calibrate(Map.of("101", priorityCanceledTask(101L, -1L)), Map.of());

        assertEquals(1, endpoint.ownershipStats().batchCount());
        WorkSnapshot remaining = endpoint.captureRouteProjectionInputs().work();
        assertFalse(remaining.containsRequest(101L));
        assertTrue(remaining.containsRequest(201L));
        assertEquals(100L, remaining.totalRemainingWorkMs().orElseThrow());
        assertTrue(committed.containsRequest(101L));
        assertEquals(200L, committed.totalRemainingWorkMs().orElseThrow());
        assertEquals(0, endpoint.ownershipStats().individuallyOwnedRequests());
        assertEquals(1, endpoint.admissionSummary(0).occupiedRequests());

        TaskInfo foreignBatchMemberSuccess = new TaskInfo();
        foreignBatchMemberSuccess.setRequestId(201L);
        foreignBatchMemberSuccess.setBatchId(101L);
        foreignBatchMemberSuccess.setErrorCode(0);
        calibrate(Map.of("201", foreignBatchMemberSuccess), Map.of());
        assertEquals(0, endpoint.ownershipStats().batchCount(),
                "the matching QUEUE batch id must survive until its own member finishes");
        WorkSnapshot released = endpoint.captureRouteProjectionInputs().work();
        assertFalse(released.containsRequest(101L));
        assertFalse(released.containsRequest(201L));
        assertEquals(0L, released.totalRemainingWorkMs().orElseThrow());
        assertTrue(remaining.containsRequest(201L));
        assertEquals(100L, remaining.totalRemainingWorkMs().orElseThrow());
    }

    @Test
    void calibrateMissingBatchIdDoesNotGuessAcrossDuplicateLiveBatches() {
        RequestRoute first = createRequestRoute(101L, 500, 200);
        RequestRoute reusedRequestId = createRequestRoute(101L, 300, 100);
        registerBatch(endpoint, 700L, 100, List.of(first));

        PrefillState.ReservationResult<PrefillState.BatchReservation> duplicate =
                endpoint.reserveBatch(reusedRequestId, 701L, 10);

        assertFalse(duplicate.status()
                        == PrefillState.CapacityStatus.ACQUIRED,
                "the canonical ledger rejects ambiguous duplicate live owners");
        assertEquals(1, endpoint.ownershipStats().batchCount());
        assertEquals(1, endpoint.admissionSummary(0).occupiedRequests());
    }

    @Test
    void calibrateMissingBatchIdPreservesExactBatchMember() {
        RequestRoute firstItem = createRequestRoute(101L, 500, 200);
        RequestRoute sibling = createRequestRoute(102L, 300, 100);
        registerBatch(endpoint, 700L, 100,
                List.of(firstItem, sibling));

        calibrate(Map.of("102", priorityCanceledTask(102L, -1L)), Map.of());
        assertEquals(1, endpoint.ownershipStats().batchCount());
        assertEquals(2, endpoint.admissionSummary(0).occupiedRequests());

        TaskInfo canceled = priorityCanceledTask(101L, -1L);
        calibrate(Map.of("101", canceled), Map.of());
        assertEquals(1, endpoint.ownershipStats().batchCount(),
                "generic endpoint calibration must not bypass the exact-batch reducer");
        assertEquals(2, endpoint.admissionSummary(0).occupiedRequests());

        assertEquals(1, endpoint.ownershipStats().batchCount());
        assertEquals(2, endpoint.admissionSummary(0).occupiedRequests());
    }

    @Test
    void authoritativeWorkerTerminalSettlesBatchMemberImmediately() {
        RequestRoute firstItem = createRequestRoute(101L, 500, 200);
        registerBatch(
                endpoint,
                700L,
                100,
                List.of(firstItem));
        WorkSnapshot committed = endpoint.captureRouteProjectionInputs().work();

        calibrate(Map.of(
                "101", taskInfo(101L, 700L, null, 0, 10)), Map.of());

        assertEquals(0, endpoint.ownershipStats().batchCount());
        WorkSnapshot completed = endpoint.captureRouteProjectionInputs().work();
        assertFalse(completed.containsRequest(101L));
        assertEquals(0L, completed.totalRemainingWorkMs().orElseThrow());
        assertTrue(committed.containsRequest(101L));
        assertEquals(100L, committed.totalRemainingWorkMs().orElseThrow());
        assertEquals(0, endpoint.admissionSummary(0).occupiedRequests());

        calibrate(Map.of(
                "101", taskInfo(101L, 700L, null, 0, 10)), Map.of());
        assertEquals(0, endpoint.ownershipStats().batchCount());
        assertSame(completed, endpoint.captureRouteProjectionInputs().work());
        assertEquals(0, endpoint.admissionSummary(0).occupiedRequests());
    }

    @Test
    void authoritativeWorkerTerminalAppliesLearningImmediately() {
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
                        List.of(createRequestRoute(
                                learningEndpoint, requestId, 500L, 200L)));
                reportSuccessfulBatchMember(
                        learningEndpoint, batchId, requestId, 100L + sample);
            }
            assertSame(initialEvaluator,
                    learningEndpoint.getPredictor().evaluator());

            long batchId = 8_004L;
            long requestId = 9_004L;
            RequestRoute firstItem = createRequestRoute(
                    learningEndpoint, requestId, 500L, 200L);
            registerBatch(
                    learningEndpoint,
                    batchId,
                    100L,
                    List.of(firstItem));

            reportSuccessfulBatchMember(
                    learningEndpoint, batchId, requestId, 104L);
            assertNotSame(initialEvaluator,
                    learningEndpoint.getPredictor().evaluator(),
                    "the authoritative terminal reaches predictor learning immediately");
            assertEquals(0, learningEndpoint.ownershipStats().batchCount());

            assertNotSame(initialEvaluator,
                    learningEndpoint.getPredictor().evaluator());
            assertEquals(0, learningEndpoint.ownershipStats().batchCount());
        } finally {
            learningEndpoint.close();
        }
    }

    @Test
    void unchangedLearningAddsNoSignalBeyondWorkerStatus() {
        PrefillEndpoint learningEndpoint = createLearningEndpoint();
        try {
            PrefillTimePredictor.Evaluator initialEvaluator =
                    learningEndpoint.getPredictor().evaluator();
            long batchId = 8_101L;
            long requestId = 9_101L;
            RequestRoute firstItem = createRequestRoute(
                    learningEndpoint, requestId, 500L, 200L);
            registerBatch(
                    learningEndpoint,
                    batchId,
                    100L,
                    List.of(firstItem));

            reportSuccessfulBatchMember(
                    learningEndpoint, batchId, requestId, 101L);
            assertSame(initialEvaluator,
                    learningEndpoint.getPredictor().evaluator());

            assertSame(initialEvaluator,
                    learningEndpoint.getPredictor().evaluator(),
                    "the first sample returns MODEL_UNCHANGED");
            assertEquals(0, learningEndpoint.ownershipStats().batchCount());
        } finally {
            learningEndpoint.close();
        }
    }

    @Test
    void finishedSettlementMakesLateExpirationANoOp() {
        RequestRoute item = createRequestRoute(101L, 500, 200);
        registerBatch(
                endpoint,
                700L,
                100,
                List.of(item));

        calibrate(Map.of(
                "101", taskInfo(101L, 700L, null, 0, 10)), Map.of());

        assertFalse(endpoint.releaseRequest(item));
        assertEquals(0, endpoint.ownershipStats().batchCount());
        WorkSnapshot completed = endpoint.captureRouteProjectionInputs().work();
        assertFalse(completed.containsRequest(101L));
        assertEquals(0L, completed.totalRemainingWorkMs().orElseThrow());
        assertEquals(0, endpoint.admissionSummary(0).occupiedRequests());
    }

    @Test
    void missingBatchIdTerminalDoesNotReleaseFixedWindowSlot() throws Exception {
        registerBatch(endpoint, 700L, 100,
                List.of(createRequestRoute(101L, 500, 200)));
        assertFalse(endpoint.batchAdmissionAvailability(1).isAvailable(),
                "maxInflight=1 must stay closed while the ledger is occupied");

        calibrate(Map.of(
                "101", priorityCanceledTask(101L, -1L)), Map.of());

        // A terminal without a valid batch id cannot be attributed to the
        // batch member, so the exact fixed-window slot stays occupied.
        assertFalse(endpoint.batchAdmissionAvailability(1).isAvailable(),
                "missing-batch-id terminal cannot release the exact slot");
        assertEquals(1, endpoint.ownershipStats().batchCount());
    }

    @Test
    void calibrateDoesNotRemoveBatchWithForeignRequestId() {
        // Commit batch with requestId=100
        RequestRoute item = createRequestRoute(100L, 500, 200);
        registerBatch(endpoint, 1L, 100, List.of(item));
        assertEquals(1, endpoint.ownershipStats().batchCount());

        // Engine reports success for batchId=1 but with requestId=999 (foreign)
        Map<String, TaskInfo> finished = new HashMap<>();
        TaskInfo foreignTask = new TaskInfo();
        foreignTask.setBatchId(1L);
        foreignTask.setRequestId(999L);
        foreignTask.setErrorCode(0);
        finished.put("999", foreignTask);

        calibrate(finished, new HashMap<>());
        // Batch should NOT be removed — requestId doesn't match
        assertEquals(1, endpoint.ownershipStats().batchCount());
    }

    @Test
    void calibrateRemovesBatchWithMatchingRequestId() {
        RequestRoute item = createRequestRoute(100L, 500, 200);
        registerBatch(endpoint, 1L, 100, List.of(item));

        Map<String, TaskInfo> finished = new HashMap<>();
        TaskInfo task = new TaskInfo();
        task.setBatchId(1L);
        task.setRequestId(100L);
        task.setErrorCode(0);
        finished.put("100", task);

        calibrate(finished, new HashMap<>());
        assertEquals(0, endpoint.ownershipStats().batchCount());
    }

    @Test
    void calibrateSuccessOnlyRetiresSiblingWhileBatchMemberReconciles() {
        RequestRoute reconciling = createRequestRoute(101L, 500, 200);
        RequestRoute sibling = createRequestRoute(102L, 300, 100);
        registerBatch(endpoint, 7L, 100, List.of(reconciling, sibling));

        TaskInfo siblingSuccess = new TaskInfo();
        siblingSuccess.setBatchId(7L);
        siblingSuccess.setRequestId(102L);
        siblingSuccess.setErrorCode(0);
        calibrate(Map.of("102", siblingSuccess), Map.of());

        assertEquals(1, endpoint.ownershipStats().batchCount(),
                "sibling success must not erase the reconciling batch member");
        assertEquals(1, endpoint.admissionSummary(0).occupiedRequests());

        TaskInfo ambiguousMemberSuccess = new TaskInfo();
        ambiguousMemberSuccess.setBatchId(7L);
        ambiguousMemberSuccess.setRequestId(101L);
        ambiguousMemberSuccess.setErrorCode(0);
        calibrate(Map.of("101", ambiguousMemberSuccess), Map.of());
        assertEquals(0, endpoint.ownershipStats().batchCount(),
                "an exact-batch terminal settles the remaining member");
        assertEquals(0, endpoint.admissionSummary(0).occupiedRequests());

        assertEquals(0, endpoint.ownershipStats().batchCount());
        assertEquals(0, endpoint.admissionSummary(0).occupiedRequests());
    }

    @Test
    void batchMemberFailuresSettleFromOneWorkerSnapshot() {
        RequestRoute firstItem = createRequestRoute(101L, 500, 200);
        RequestRoute sibling = createRequestRoute(102L, 300, 100);
        registerBatch(endpoint, 7L, 100,
                List.of(firstItem, sibling));

        TaskInfo firstFailure = taskInfo(101L, 7L, null, 500, 40);
        TaskInfo siblingFailure = taskInfo(102L, 7L, null, 501, 50);
        calibrate(Map.of("101", firstFailure, "102", siblingFailure), Map.of());

        assertEquals(0, endpoint.ownershipStats().batchCount());
        assertEquals(0, endpoint.admissionSummary(0).occupiedRequests());

        assertEquals(0, endpoint.ownershipStats().batchCount());
        assertEquals(0, endpoint.admissionSummary(0).occupiedRequests());
        verify(endpointReporter, never()).reportBatchCompletion(
                anyString(), org.mockito.ArgumentMatchers.anyLong(),
                org.mockito.ArgumentMatchers.anyLong(), org.mockito.ArgumentMatchers.anyLong());

        assertEquals(0, endpoint.admissionSummary(0).occupiedRequests());
    }

    // ---- committed remaining work ----

    @Test
    void committedWorkMetricIsZeroWhenIdle() {
        assertEquals(0L, endpoint.getLoadMetric().orElseThrow());
    }

    @Test
    void committedWorkMetricReflectsInflightPrediction() {
        RequestRoute item = createRequestRoute(1L, 500, 200);
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
        RequestRoute item = createRequestRoute(1L, 500, 200);
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
        RequestRoute item = createRequestRoute(1L, 500, 200);
        registerBatch(endpoint, 1L, 100, List.of(item));

        assertEquals(1, endpoint.ownershipStats().batchCount());

        // Wait a bit so the batch ages
        Thread.sleep(10);

        int evicted = endpoint.evictExpiredInflight(1, ignored -> false); // 1ms TTL — should evict
        assertEquals(1, evicted);
        assertEquals(0, endpoint.ownershipStats().batchCount());
    }

    @Test
    void evictExpiredBatchesFreshEntriesSurvive() {
        RequestRoute item = createRequestRoute(1L, 500, 200);
        registerBatch(endpoint, 1L, 100, List.of(item));

        int evicted = endpoint.evictExpiredInflight(60_000, ignored -> false); // 60s TTL — fresh entry survives
        assertEquals(0, evicted);
        assertEquals(1, endpoint.ownershipStats().batchCount());
    }

    @Test
    void expireUnobservedBatchDoesNotWaitForAnEngineReply() {
        RequestRoute item = createRequestRoute(1L, 500, 200);
        registerBatch(endpoint, 1L, 100, List.of(item));

        assertTrue(endpoint.releaseRequest(item));
        assertEquals(0, endpoint.ownershipStats().batchCount());
        assertEquals(0, endpoint.admissionSummary(0).occupiedRequests());
        assertFalse(endpoint.releaseRequest(item));
    }

    // ---- outstandingRequestCount ----

    @Test
    void outstandingRequestCountUnionsEngineTasksWithLocalLedger() {
        registerBatch(endpoint, 1L, 100, List.of(
                createRequestRoute(101L, 500, 0),
                createRequestRoute(102L, 500, 0)));

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

        assertEquals(4, endpoint.admissionSummary(0).occupiedRequests(),
                "two local requests plus two unique Engine-only tasks");

        response.setRunningTaskInfo(Map.of());
        EndpointTestSupport.applyStatus(endpoint, response);
        assertEquals(2, endpoint.admissionSummary(0).occupiedRequests());
    }

    @Test
    void outstandingRequestCountFallsBackToEngineQueryLengthScalars() {
        registerBatch(endpoint, 1L, 100, List.of(createRequestRoute(101L, 500, 0)));

        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setFinishedTaskInfo(Map.of());
        response.setRunningTaskInfo(Map.of());
        response.setWaitingQueryLen(3);
        response.setRunningQueryLen(2);
        EndpointTestSupport.applyStatus(endpoint, response);

        assertEquals(6, endpoint.admissionSummary(0).occupiedRequests(),
                "an unseen local shadow cannot prove identity with scalar Engine work");
        assertTrue(endpoint.captureRouteProjectionInputs().work().hasUnknownWork());
        assertTrue(endpoint.captureRouteProjectionInputs().work().totalRemainingWorkMs().isEmpty());
    }

    @Test
    void outstandingRequestCountUsesConservativeScalarBoundForPartialTaskDetails() {
        registerBatch(endpoint, 1L, 100, List.of(createRequestRoute(101L, 500, 0)));

        TaskInfo overlapping = taskInfo(101L, 1L, TaskPhase.RUNNING, 0, 0);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setFinishedTaskInfo(Map.of());
        response.setRunningTaskInfo(Map.of("101", overlapping));
        response.setWaitingQueryLen(3);
        response.setRunningQueryLen(2);
        EndpointTestSupport.applyStatus(endpoint, response);

        assertEquals(5, endpoint.admissionSummary(0).occupiedRequests(),
                "scalar active count must cover a partial detail list without double-counting local tasks");
    }

    @Test
    void outstandingRequestCountIncludesBatcherQueue() throws InterruptedException {
        PrefillEndpoint queuedEndpoint = newFixedWindowEndpoint(60_000L);
        try {
            assertEquals(0, queuedEndpoint.admissionSummary(0).occupiedRequests());
            RequestRoute item = createRequestRoute(
                    queuedEndpoint, 1L, 500L, 200L);
            assertTrue(EndpointTestSupport.offer(queuedEndpoint, item));

            assertEquals(1, queuedEndpoint.admissionSummary(0).occupiedRequests(),
                    "pending count includes the canonical ACTIVE queue owner");
        } finally {
            queuedEndpoint.close();
        }
    }

    @Test
    void projectionCapturesExactPublishedIdentityBeforeActiveRemoval() {
        PrefillEndpoint handoffEndpoint = newFixedWindowEndpoint(60_000);
        try {
            RequestRoute active = createRequestRoute(handoffEndpoint, 111L, 500L, 0L);
            assertTrue(EndpointTestSupport.offer(handoffEndpoint, active));

            org.flexlb.balance.projection.RouteProjection.Inputs snapshot =
                    handoffEndpoint.captureRouteProjectionInputs();
            assertEquals(1, snapshot.queue().activeItems().size());
            assertEquals(111L,
                    snapshot.queue().activeItems().getFirst().requestId());
        } finally {
            handoffEndpoint.close();
        }
    }

    @Test
    void outstandingRequestCountCannotMissActiveToCommittedHandoff()
            throws Exception {
        PrefillEndpoint handoffEndpoint = newFixedWindowEndpoint(60_000);
        long requestId = 222L;
        try {
            RequestRoute active = createRequestRoute(
                    handoffEndpoint, requestId, 500L, 0L);
            assertTrue(EndpointTestSupport.offer(handoffEndpoint, active));
            assertTrue(handoffEndpoint.removeQueued(
                    active, "test exact ownership handoff"));
            registerDirect(handoffEndpoint, requestId, 100L);

            assertEquals(1L, handoffEndpoint.admissionSummary(0).occupiedRequests());
            assertEquals(0, handoffEndpoint.queuedRequestCount());
            WorkSnapshot committed = handoffEndpoint.captureRouteProjectionInputs().work();
            assertTrue(committed.containsRequest(requestId));
            assertEquals(100L, committed.totalRemainingWorkMs().orElseThrow());
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
                    slowEndpoint, createPriorityRequestRoute(slowEndpoint, 1L, 70)));
            assertTrue(EndpointTestSupport.offer(
                    slowEndpoint,
                    createRequestRoute(slowEndpoint, 2L, 300, 0)));

            DeliveryMetricsReporter reporter = mock(DeliveryMetricsReporter.class);
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
        DeliveryMetricsReporter reporter = mock(DeliveryMetricsReporter.class);
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
        FlexlbConfig retirementConfig = org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig();
        configureBatch(retirementConfig, 1, 0, null);
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
                    org.flexlb.balance.scheduler.RequestRoute item,
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
        PrefillEndpoint retirementEndpoint = EndpointTestSupport.prefill(status, retirementConfig, EndpointTestSupport.routeStrategy(runtime), org.flexlb.balance.scheduler.SchedulerTestSupport.repository(runtime.events()), mock(DeliveryMetricsReporter.class));
        retirementEndpointRef.set(retirementEndpoint);
        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            RequestRoute item = createRequestRoute(
                    retirementEndpoint, retirementConfig, 8_101L, 128, 0);
            assertTrue(EndpointTestSupport.offer(retirementEndpoint, item));

            Future<?> outerClose = executor.submit(retirementEndpoint::close);
            outerClose.get(2, TimeUnit.SECONDS);
            assertTrue(reentrantCloseEntered.await(1, TimeUnit.SECONDS));
            assertTrue(reentrantCloseReturned.await(1, TimeUnit.SECONDS),
                    "retirement-owner reentry must return instead of waiting on itself");
            assertTrue(callbackFailure.get() == null,
                    () -> String.valueOf(callbackFailure.get()));

            assertFalse(EndpointTestSupport.offer(retirementEndpoint,
                    createRequestRoute(retirementEndpoint, retirementConfig, 8_102L, 128, 0)));

        } finally {
            retirementEndpoint.close();
            executor.shutdownNow();
        }
    }

    @Test
    void admittedCallbackCanCloseEndpointBeforeItsHandoffPermitIsReleased()
            throws Exception {
        FlexlbConfig retirementConfig = org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig();
        configureBatch(retirementConfig, 1, 0, null);
        retirementConfig.setDispatcher(DispatcherConfig.nonBatch());
        setFormula(retirementConfig, "1");

        WorkerStatus status = EndpointTestSupport.workerStatus(
                RoleType.PREFILL, "127.0.0.6", 8080, 8090);

        CountDownLatch callbackResolved = new CountDownLatch(1);
        AtomicReference<Throwable> callbackFailure = new AtomicReference<>();
        AtomicReference<PrefillEndpoint> retirementEndpointRef = new AtomicReference<>();
        EndpointTestSupport.TestRequestRuntime runtime =
                new EndpointTestSupport.TestRequestRuntime() {
            @Override
            void onCompleted(
                    DeliveryClaim claim,
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
        PrefillEndpoint retirementEndpoint = EndpointTestSupport.prefill(status, retirementConfig, EndpointTestSupport.liveRouteStrategy(runtime), org.flexlb.balance.scheduler.SchedulerTestSupport.repository(runtime.events()), mock(DeliveryMetricsReporter.class));
        retirementEndpointRef.set(retirementEndpoint);
        try {
            DecodeEndpoint decode = mock(DecodeEndpoint.class);
            DecodeResources.ReservationHandle decodeReservation =
                    mock(DecodeResources.ReservationHandle.class);
            org.mockito.Mockito.when(decodeReservation.requestId()).thenReturn(8_201L);
            DecodeEndpoint.EngineDispatchPermit permit =
                    mock(DecodeEndpoint.EngineDispatchPermit.class);
            org.mockito.Mockito.when(decode.acquireDispatchPermit(
                    org.mockito.Mockito.any(DecodeResources.ReservationHandle.class),
                    org.mockito.Mockito.any(DecodeResources.AdmissionCapacity.class)))
                    .thenReturn(new DecodeEndpoint.EngineDispatchPermitAcquisition(
                            DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED,
                            permit));
            org.mockito.Mockito.when(permit.dispatch())
                    .thenReturn(
                            DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED);
            RequestRoute admitted = createRequestRoute(
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
    void zeroCompletionPredictionDoesNotBlockDelivery()
            throws Exception {
        FlexlbConfig routeConfig = org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig();
        configureBatch(routeConfig, 1, 0, null);
        routeConfig.setDispatcher(DispatcherConfig.nonBatch());
        setFormula(routeConfig, "0");
        CountDownLatch firstPreparation = new CountDownLatch(1);
        CountDownLatch delivered = new CountDownLatch(1);
        EndpointTestSupport.TestRequestRuntime runtime = new EndpointTestSupport.TestRequestRuntime() {
            @Override
            void onCompleted(DeliveryClaim claim, DeliveryResult completion) {
                delivered.countDown();
            }
        };
        DeliveryStrategy strategy = org.mockito.Mockito.spy(
                EndpointTestSupport.liveRouteStrategy(runtime));
        org.mockito.Mockito.doAnswer(invocation -> {
            Object transaction = invocation.callRealMethod();
            firstPreparation.countDown();
            return transaction;
        }).when(strategy).prepare(org.mockito.Mockito.anyList(),
                org.mockito.Mockito.any(), org.mockito.Mockito.any());
        PrefillEndpoint routeEndpoint = EndpointTestSupport.prefill(EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.10", 8080, 8090), routeConfig, strategy, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(runtime.events()), endpointReporter);
        try {
            RequestRoute waiting = createRequestRoute(routeEndpoint, routeConfig, 8_301L, 128, 0);
            assertTrue(EndpointTestSupport.offer(routeEndpoint, waiting));
            assertTrue(firstPreparation.await(2, TimeUnit.SECONDS));
            assertTrue(delivered.await(2, TimeUnit.SECONDS),
                    "zero predicted execution time must not require a later status update");
            assertEquals(0, routeEndpoint.queuedRequestCount());
            assertEquals(1, routeEndpoint.admissionSummary(0).occupiedRequests(),
                    "delivery must retain ownership until Engine completion");
        } finally {
            routeEndpoint.close();
        }
    }

    @Test
    void unknownCompletionTimeDoesNotBlockDelivery() throws Exception {
        FlexlbConfig routeConfig = org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig();
        configureBatch(routeConfig, 1, 0, null);
        routeConfig.setDispatcher(DispatcherConfig.nonBatch());
        setFormula(routeConfig, "100");
        CountDownLatch firstPreparation = new CountDownLatch(1);
        CountDownLatch delivered = new CountDownLatch(1);
        AtomicInteger preparationCount = new AtomicInteger();
        EndpointTestSupport.TestRequestRuntime runtime = new EndpointTestSupport.TestRequestRuntime() {
            @Override
            void onCompleted(DeliveryClaim claim, DeliveryResult completion) {
                delivered.countDown();
            }
        };
        DeliveryStrategy strategy = org.mockito.Mockito.spy(
                EndpointTestSupport.liveRouteStrategy(runtime));
        org.mockito.Mockito.doAnswer(invocation -> {
            Object transaction = invocation.callRealMethod();
            preparationCount.incrementAndGet();
            firstPreparation.countDown();
            return transaction;
        }).when(strategy).prepare(org.mockito.Mockito.anyList(),
                org.mockito.Mockito.any(), org.mockito.Mockito.any());
        PrefillEndpoint routeEndpoint = EndpointTestSupport.prefill(EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.11", 8080, 8090), routeConfig, strategy, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(runtime.events()), endpointReporter);
        try {
            TaskInfo unknown = taskInfo(8_402L, -1L, TaskPhase.RUNNING, 0, -1L);
            unknown.setInputLength(128L);
            WorkerStatusResponse status = new WorkerStatusResponse();
            status.setRunningTaskInfo(Map.of("8402", unknown));
            status.setRunningQueryLen(1L);
            EndpointTestSupport.applyStatus(routeEndpoint, status).run();
            RequestRoute waiting = createRequestRoute(routeEndpoint, routeConfig, 8_401L, 128, 0);
            assertTrue(EndpointTestSupport.offer(routeEndpoint, waiting));
            assertTrue(firstPreparation.await(2, TimeUnit.SECONDS));
            assertTrue(delivered.await(2, TimeUnit.SECONDS),
                    "unknown preceding execution time must not become an admission gate");
            assertEquals(0, routeEndpoint.queuedRequestCount());
            assertEquals(2, routeEndpoint.admissionSummary(0).occupiedRequests(),
                    "Engine work and the delivered route retain separate identities");

            for (int repeat = 0; repeat < 20; repeat++) {
                applyHeartbeat(routeEndpoint, status);
            }
            assertEquals(1, preparationCount.get(), "identical heartbeats must not redeliver the route");
            assertEquals(0, routeEndpoint.queuedRequestCount());
        } finally {
            routeEndpoint.close();
        }
    }

    private static void applyHeartbeat(PrefillEndpoint endpoint, WorkerStatusResponse response) {
        WorkerStatus status = endpoint.getStatus();
        Runnable projection;
        status.lock.lock();
        try {
            response.setStatusVersion(status.appliedStatusCursor().statusVersion());
            projection = endpoint.applyStatusHeartbeat(status, status.freezeStatusResponse(response));
        } finally {
            status.lock.unlock();
        }
        projection.run();
    }

    @Test
    void closePreservesRegisteredLifecycleAndRejectsNewBatchReservations() {
        RequestRoute item = createRequestRoute(1L, 500, 200);
        registerBatch(
                endpoint,
                1L,
                100,
                List.of(item));
        assertEquals(1, endpoint.ownershipStats().batchCount());

        endpoint.close();
        assertEquals(0, endpoint.ownershipStats().batchCount());
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
    void retirementReportsEveryBatchAfterSchedulerAndReportingFailures() {
        registerBatch(endpoint, 1L, 100L, List.of(
                createRequestRoute(11L, 500, 0), createRequestRoute(12L, 500, 0)));
        registerBatch(endpoint, 2L, 200L, List.of(
                createRequestRoute(21L, 500, 0), createRequestRoute(22L, 500, 0)));
        calibrate(Map.of("11", taskInfo(11L, 1L, null, 0, 40),
                        "21", taskInfo(21L, 2L, null, 0, 50)),
                Map.of("12", taskInfo(12L, 1L, TaskPhase.RUNNING, 0, 0),
                        "22", taskInfo(22L, 2L, TaskPhase.RUNNING, 0, 0)));
        var schedulerFailure = new IllegalStateException("scheduler retirement failed");
        var firstReportFailure = new AssertionError("first report failed");
        var secondReportFailure = new AssertionError("second report failed");
        doThrow(schedulerFailure).when(requestRuntime.events()).onPrefillGenerationRetired(
                eq(endpoint), org.mockito.ArgumentMatchers.any());
        doThrow(firstReportFailure).when(endpointReporter)
                .reportBatchCompletion(anyString(), eq(1L), eq(100L), eq(40L));
        doThrow(secondReportFailure).when(endpointReporter)
                .reportBatchCompletion(anyString(), eq(2L), eq(200L), eq(50L));

        assertSame(schedulerFailure, assertThrows(IllegalStateException.class, endpoint::closeEndpoint));
        assertEquals(List.of(firstReportFailure, secondReportFailure),
                List.of(schedulerFailure.getSuppressed()));
        assertEquals(0, endpoint.ownershipStats().locallyOwnedRequests());
        assertEquals(0, endpoint.ownershipStats().batchCount());
        verify(endpointReporter).reportBatchCompletion(anyString(), eq(1L), eq(100L), eq(40L));
        verify(endpointReporter).reportBatchCompletion(anyString(), eq(2L), eq(200L), eq(50L));
        assertDoesNotThrow(endpoint::closeEndpoint);
    }

    @Test
    void closeProjectsEveryCommittedRouteAcrossSchedulingModes() {
        registerDirect(endpoint, 100L, 100L);
        RequestRoute route = createRequestRoute(200L, 200L, 0L);
        // A queue route can only be reserved after its canonical item has been
        // offered into the ACTIVE queue; reserveRoute rejects a non-active
        // identity with REQUEST_NOT_ACTIVE.
        assertTrue(EndpointTestSupport.offer(endpoint, route));
        List<PrefillState.CommittedHandoff> handoffs =
                EndpointTestSupport.commitRoutes(
                        endpoint, 200L, List.of(route));
        handoffs.forEach(PrefillState.CommittedHandoff::close);
        assertEquals(2, endpoint.ownershipStats().individuallyOwnedRequests());

        endpoint.close();

        assertEquals(0, endpoint.ownershipStats().individuallyOwnedRequests());
        var retiredItems = requestRuntime.prefillRetirements().stream()
                .flatMap(retirement -> retirement.ownedItems().stream()).toList();
        assertEquals(List.of(100L, 200L), retiredItems.stream()
                .map(RequestRoute::requestId).sorted().toList(),
                "both immediate and queued routes reach their canonical retirement owner");
        assertTrue(retiredItems.contains(route));
    }

    @Test
    void closeShutsDownBatcher() {
        endpoint.close();
        RequestRoute item = createRequestRoute(1L, 500, 200);
        assertFalse(EndpointTestSupport.offer(endpoint, item));
        assertEquals(0, endpoint.queuedRequestCount());
    }

    // ---- helpers ----

    private PrefillEndpoint newFixedWindowEndpoint(long fixedWaitMs) {
        WorkerStatus status = EndpointTestSupport.workerStatus(
                RoleType.PREFILL, "127.0.0.1", 8080, 8090);

        FlexlbConfig slowConfig = org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig();
        configureBatch(slowConfig, 100, fixedWaitMs, null);
        EndpointTestSupport.TestRequestRuntime runtime =
                EndpointTestSupport.requestRuntime();
        PrefillEndpoint created = EndpointTestSupport.prefill(status, slowConfig, EndpointTestSupport.routeStrategy(runtime), org.flexlb.balance.scheduler.SchedulerTestSupport.repository(runtime.events()), mock(DeliveryMetricsReporter.class));
        return created;
    }

    private static PrefillEndpoint newEndpointWithFormula(String expression) {
        WorkerStatus status = EndpointTestSupport.workerStatus(
                RoleType.PREFILL, "127.0.0.9", 8089, 8099);

        FlexlbConfig endpointConfig = org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig();
        configureBatch(
                endpointConfig,
                endpointConfig.decisionPolicy().getMaxRequests(),
                300L,
                null);
        setFormula(endpointConfig, expression);
        EndpointTestSupport.TestRequestRuntime runtime =
                EndpointTestSupport.requestRuntime();
        PrefillEndpoint created = EndpointTestSupport.prefill(status, endpointConfig, EndpointTestSupport.routeStrategy(runtime), org.flexlb.balance.scheduler.SchedulerTestSupport.repository(runtime.events()), mock(DeliveryMetricsReporter.class));
        return created;
    }

    private RequestRoute createPriorityRequestRoute(
            PrefillEndpoint owner, long requestId, int priority) {
        long now = System.currentTimeMillis();
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(500);
        request.setPriority(priority);

        RequestContext ctx = new RequestContext(config);
        ctx.setRequest(request);
        ctx.setSchedulingMetadata(SchedulingMetadata.explicit(priority, now + 60_000));

        return org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(ctx), null, null, null, owner, null, null, now);
    }

    private void calibrate(Map<String, TaskInfo> finished, Map<String, TaskInfo> running) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setFinishedTaskInfo(finished);
        response.setRunningTaskInfo(running);
        EndpointTestSupport.applyStatus(endpoint, response);
    }

    private static void reportRejectedBatchMember(PrefillEndpoint target, long batchId, long requestId) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setFinishedTaskInfo(Map.of(Long.toString(requestId),
                taskInfo(requestId, batchId, null, 500, 0)));
        response.setRunningTaskInfo(Map.of());
        EndpointTestSupport.applyStatus(target, response);
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

        FlexlbConfig learningConfig = org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig();
        configureBatch(
                learningConfig,
                learningConfig.decisionPolicy().getMaxRequests(),
                300,
                null);
        learningConfig.getRouter().getRoles().getPrefill()
                .getExecutionTimeEstimator()
                .setType(RoutingConfig.EstimatorType.LEARNING);
        EndpointTestSupport.TestRequestRuntime runtime =
                EndpointTestSupport.requestRuntime();
        PrefillEndpoint created = EndpointTestSupport.prefill(status, learningConfig, EndpointTestSupport.routeStrategy(runtime), org.flexlb.balance.scheduler.SchedulerTestSupport.repository(runtime.events()), mock(DeliveryMetricsReporter.class));
        return created;
    }

    private static WorkSnapshot.Phase batchServicePhase(
            PrefillEndpoint target, long batchId) {
        PrefillState state = (PrefillState) org.springframework.test.util.ReflectionTestUtils
                .getField(target, "prefillState");
        state.ownershipLock().lock();
        try {
            Map<?, ?> batches = (Map<?, ?>) org.springframework.test.util.ReflectionTestUtils
                    .getField(state, "batches");
            assertEquals(1, batches.size());
            Object batch = batches.get(batchId);
            assertNotNull(batch);
            return (WorkSnapshot.Phase) org.springframework.test.util.ReflectionTestUtils
                    .getField(batch, "servicePhase");
        } finally {
            state.ownershipLock().unlock();
        }
    }

    private RequestRoute createRequestRoute(long requestId, long seqLen, long hitCacheLen) {
        return createRequestRoute(endpoint, requestId, seqLen, hitCacheLen);
    }

    private static RequestRoute createRequestRoute(PrefillEndpoint owner,
                                             long requestId,
                                             long seqLen,
                                             long hitCacheLen) {
        return createRequestRoute(
                owner, org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig(), requestId, seqLen, hitCacheLen);
    }

    private static RequestRoute createRequestRoute(
            PrefillEndpoint owner,
            FlexlbConfig requestConfig,
            long requestId,
            long seqLen,
            long hitCacheLen) {
        return createRequestRoute(
                owner,
                requestConfig,
                requestId,
                seqLen,
                hitCacheLen,
                null,
                null);
    }

    private static RequestRoute createRequestRoute(
            PrefillEndpoint owner,
            FlexlbConfig requestConfig,
            long requestId,
            long seqLen,
            long hitCacheLen,
            DecodeEndpoint decode,
            DecodeResources.ReservationHandle decodeReservation) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);

        RequestContext ctx = new RequestContext(requestConfig);
        ctx.setRequest(request);

        ServerStatus prefill = new ServerStatus();
        prefill.setRole(RoleType.PREFILL);
        prefill.setServerIp("127.0.0.1");
        prefill.setHttpPort(8080);
        prefill.setGrpcPort(8090);
        DebugInfo debugInfo = new DebugInfo();
        debugInfo.setHitCacheLen(hitCacheLen);
        prefill.setDebugInfo(debugInfo);

        return org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(ctx),
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
            int maxRequests,
            long maxCollectionWaitMs,
            Integer maxInflightBatches) {
        target.decisionPolicy().setMaxRequests(maxRequests);
        target.decisionPolicy().setMaxCollectionWaitMs(maxCollectionWaitMs);
        if (maxInflightBatches != null) {
            target.getDispatcher().setMaxInflightPerPrefillWorker(maxInflightBatches);
        }
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
            List<RequestRoute> items) {
        for (RequestRoute item : items) {
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
        EndpointTestSupport.commitUnqueued(target, requestId, predictedMs);
    }

}
