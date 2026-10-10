package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.DecodeResources.CapacityRelease;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.EndpointTestSupport;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import static org.flexlb.balance.scheduler.ContextResult.*;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.master.WorkerStatusTestSupport;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** Prefill status must carry stage completion through the real endpoint and context reducers. */
class PrefillCompletionProjectionTest {
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void successfulPrefillCompletionReleasesTokensAndProjectsHandoffEvidence(boolean decodeAlreadyAccepted)
            throws Exception {
        var config = SchedulingTestConfig.newConfig();
        config.setScheduler(ConfigTestSupport.direct());
        SchedulingTestConfig.useNonBatchDispatcher(config);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var reporter = mock(DeliveryMetricsReporter.class);
        var requestReporter = mock(RequestSchedulerReporter.class);
        var requests = SchedulerTestSupport.create(service, reporter, requestReporter,
                mock(RecentCacheKeyTraceReporter.class));
        var projector = requests;
        var endpoints = new EndpointRegistry(service, SchedulerTestSupport.repository(projector), reporter, DeliveryTestSupport.route(reporter, SchedulingTestConfig.capacity(SchedulingTestConfig.newConfig())), new PlacementAvailability());
        var runtime = new SchedulerRuntime(SchedulerTestSupport.repository(requests), endpoints, reporter, requestReporter, org.mockito.Mockito.mock(DeliveryDispatcher.class), service, org.mockito.Mockito.mock(org.flexlb.balance.eviction.EngineCancelChannel.class));
        try {
            WorkerStatus worker = WorkerStatus.createDiscovered(
                    RoleType.PREFILL, "g1", "127.0.0.1", 8080, 8081, "test");
            PrefillEndpoint prefill;
            worker.lock.lock();
            try {
                prefill = (PrefillEndpoint) endpoints.publishPreparedEndpoint(worker.getIpPort(), worker,
                        worker.prepareNewStatus(WorkerStatusTestSupport.freezeStatusResponse(worker, status(1L, Map.of(), Map.of()))));
            } finally {
                worker.lock.unlock();
            }
            WorkerStatus decodeWorker = WorkerStatus.createDiscovered(
                    RoleType.DECODE, "g1", "127.0.0.2", 8080, 8081, "test");
            DecodeEndpoint decode = EndpointTestSupport.decode(decodeWorker, SchedulerTestSupport.repository(projector));
            applyStatus(decode, decodeStatus(1L, Map.of()));
            DecodeResources.ReservationHandle reservation;
            var capacity = new DecodeResources.AdmissionCapacity(10L, 90L);
            try (var pin = decode.tryPinGeneration()) {
                reservation = decode.tryReserveQueuedRequest(pin, 101L, 16L, 32L, 50, capacity).handle();
                assertNotNull(reservation);
                var acquisition = decode.acquireDispatchPermit(reservation, capacity);
                assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED, acquisition.status());
                assertEquals(DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED,
                        acquisition.handle().dispatch().status());
            }
            var context = RequestProtocolTestSupport.context(config, 101L);
            var future = RequestProtocolTestSupport.register(requests, context);
            context.data().setFuture(future);
            RequestRoute item = SchedulingTestConfig.createRoute(context, new Response(), null, null,
                    prefill, decode, reservation, System.currentTimeMillis());
            AtomicReference<PrefillState.RouteReservation> routeReservation = new AtomicReference<>();
            try (var mutation = requests.claimAdmissionHandle(101L, future); var admissionCompletion1 = RequestProtocolTestSupport.finishOnExit(mutation);
                 var pin = prefill.tryPinGeneration()) {
                assertNotNull(mutation);
                assertNotNull(pin);
                assertTrue((requests.commitRoute(item, RequestProtocolTestSupport.publication(() -> {
                    var registered = prefill.reserveUnqueuedRoute(pin, item, 30_000L);
                    assertEquals(PrefillState.CapacityStatus.ACQUIRED, registered.status());
                    routeReservation.set(registered.reservation());
                    return true;
                })) == org.flexlb.balance.PlacementResult.Status.SUCCESS));
            }
            {
                var claim = RequestProtocolTestSupport.claimRouteWithoutPrediction(requests, item, () -> {
                    try (var handoff = prefill.commit(new PrefillState.CommitPlan(PrefillState.CommitKind.DIRECT, List.of(item), List.of(routeReservation.get()), null, new long[0], null, System.currentTimeMillis()))) {
                        return true;
                    }
                });
                assertNotNull(claim);
                long now = System.currentTimeMillis();
                var work = new WorkSnapshot(now, java.util.List.of(), java.util.List.of(), 0L);
                requests.onDeliveryEvidence(claim, new RouteDelivered(claim,
                        PrefillPredictionBoundary.predictedVisibilityDeadlineMs(work, 30_000L,
                                config.getRequestLifecycle().getDecision().getLifetime(), now), now));
            }
            assertTrue(future.get(2L, TimeUnit.SECONDS).isSuccess());
            assertEquals(1L, prefill.ownership.admissionSummary(0, prefill.requestLimit).occupiedRequests());

            TaskInfo task = new TaskInfo();
            task.setRequestId(101L);
            task.setInputLength(16L);
            task.setPhase(TaskPhase.RUNNING);
            applyStatus(prefill, status(2L, Map.of("101", task), Map.of()));
            SchedulerTestSupport.awaitRequestWork(requests.runtime);
            RequestContext requestContext = requests.requests.findActive(101L);
            synchronized (requestContext) {
                assertTrue(requestContext.snapshot().deadlines().visibility() == null, "running Prefill is positive Engine evidence");
            }
            if (decodeAlreadyAccepted) {
                applyStatus(decode, decodeStatus(2L, Map.of("101", task)));
                SchedulerTestSupport.awaitRequestWork(requests.runtime);
            }

            applyStatus(prefill, status(3L, Map.of(), Map.of("101", task)));
            SchedulerTestSupport.awaitRequestWork(requests.runtime);
            assertEquals(0L, prefill.ownership.admissionSummary(0, prefill.requestLimit).occupiedRequests());
            synchronized (requestContext) {
                assertTrue(requestContext.snapshot().phase() != Phase.CLOSED, "Prefill completion must retain the Decode lifecycle");
                assertTrue(visibilityInstalledOrAbsent(requests, requestContext), "handoff prediction must have no pending timer installation");
            }
            assertEquals(decodeAlreadyAccepted ? 0L : 32L, EndpointTestSupport.expectedReservedKv(decode.resourceSnapshot()),
                    "missing Decode acceptance must retain this request's reservation");
            assertTrue(capacity.evaluate(decode.routingView().dispatchUsage(), 16L, 32L, CapacityRelease.NONE).fits(),
                    "a suspected lost request must not isolate a worker with available capacity");
            try (var pin = decode.tryPinGeneration()) {
                var waiting = decode.tryReserveQueuedRequest(pin, 102L, 16L, 32L, 50, capacity).handle();
                assertNotNull(waiting);
                var acquisition = decode.acquireDispatchPermit(waiting, capacity);
                assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED, acquisition.status());
                assertTrue(acquisition.handle().release());
                decode.release(waiting, DecodeResources.ReleaseReason.LOCAL_ROLLBACK).status();
            }

            applyStatus(prefill, status(4L, Map.of(), Map.of("101", task)));
            SchedulerTestSupport.awaitRequestWork(requests.runtime);
            assertEquals(0L, prefill.ownership.admissionSummary(0, prefill.requestLimit).occupiedRequests());

            var decodeFinished = decodeStatus(3L, Map.of());
            decodeFinished.setFinishedTaskInfo(Map.of("101", task));
            decodeFinished.setLatestFinishedVersion(1L);
            applyStatus(decode, decodeFinished);
            SchedulerTestSupport.awaitRequestWork(requests.runtime);
            assertEquals(0L, EndpointTestSupport.expectedReservedKv(decode.resourceSnapshot()));
            assertEquals(0, decode.routingView().engineCapacityUsed());
            org.junit.jupiter.api.Assertions.assertNull(requests.requests.findActive(101L),
                    "the exact Decode terminal must finish the retained lifecycle before shutdown");
            assertTrue(future.join().isSuccess(), "resource completion cannot replace the published response");
        } finally {
            runtime.shutdown();
        }
    }

    private static boolean visibilityInstalledOrAbsent(AbstractRequestScheduler owner, RequestContext context) {
        var token = context.snapshot().deadlines().visibility();
        if (token == null) { return true; }
        var slots = (java.util.Map<?, ?>) org.springframework.test.util.ReflectionTestUtils.getField(owner.runtime.deadlines(), "slots");
        var slot = slots.get(context);
        if (slot == null) { return false; }
        var handles = (java.util.Map<?, ?>) org.springframework.test.util.ReflectionTestUtils.invokeMethod(slot, "handles");
        var handle = handles.get(ExpirationTimer.Kind.VISIBILITY);
        return handle != null && org.springframework.test.util.ReflectionTestUtils.getField(handle, "token") == token
                && org.springframework.test.util.ReflectionTestUtils.getField(handle, "scheduled") != null;
    }

    private static WorkerStatusResponse status(long version, Map<String, TaskInfo> running,
                                                Map<String, TaskInfo> finished) {
        var response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setAlive(true);
        response.setStatusVersion(version);
        response.setLatestFinishedVersion(finished.isEmpty() ? 0L : 1L);
        response.setRunningTaskInfo(running);
        response.setRunningQueryLen((long) running.size());
        response.setFinishedTaskInfo(finished);
        return response;
    }

    private static WorkerStatusResponse decodeStatus(long version, Map<String, TaskInfo> running) {
        var response = status(version, running, Map.of());
        response.setRole(RoleType.DECODE);
        response.setTotalKvCacheTokens(10_000L);
        response.setAvailableKvCacheTokens(9_000L);
        return response;
    }

    private static void applyStatus(WorkerEndpoint endpoint, WorkerStatusResponse response) {
        WorkerStatus worker = endpoint.getStatus();
        org.flexlb.balance.endpoint.WorkerEndpoint.StatusChange projection;
        worker.lock.lock();
        try {
            projection = endpoint.applyPreparedStatus(worker,
                    worker.prepareNewStatus(WorkerStatusTestSupport.freezeStatusResponse(worker, response)));
        } finally {
            worker.lock.unlock();
        }
        endpoint.publishStatus(projection);
    }
}
