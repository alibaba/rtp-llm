package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.config.ConfigService;
import org.flexlb.config.SchedulerConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
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

/** Prefill status must carry stage completion through the real endpoint and slot reducers. */
class PrefillCompletionProjectionTest {
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void successfulPrefillCompletionReleasesTokensAndProjectsHandoffEvidence(boolean decodeAlreadyAccepted)
            throws Exception {
        var config = SchedulingTestConfig.newConfig();
        config.setScheduler(SchedulerConfig.direct());
        SchedulingTestConfig.useNonBatchDispatcher(config);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var reporter = mock(BatchSchedulerReporter.class);
        var requestReporter = mock(RequestSchedulerReporter.class);
        var requests = new RequestRegistry(service, reporter, requestReporter);
        var projector = new EndpointEventProjector(requests);
        var endpoints = new EndpointRegistry(service, projector, reporter,
                new RouteDeliveryStrategy(requests, new DeliveryMetrics(reporter)), new PlacementAvailability());
        var runtime = new SchedulerRuntime(requests, endpoints, reporter, requestReporter);
        try {
            WorkerStatus worker = WorkerStatus.createDiscovered(
                    RoleType.PREFILL, "g1", "127.0.0.1", 8080, 8081, "test");
            PrefillEndpoint prefill;
            worker.lock.lock();
            try {
                prefill = (PrefillEndpoint) endpoints.publishPreparedEndpoint(worker.getIpPort(), worker,
                        worker.prepareNewStatus(worker.freezeStatusResponse(status(1L, Map.of(), Map.of()))))
                        .endpoint();
            } finally {
                worker.lock.unlock();
            }
            WorkerStatus decodeWorker = WorkerStatus.createDiscovered(
                    RoleType.DECODE, "g1", "127.0.0.2", 8080, 8081, "test");
            DecodeEndpoint decode = new DecodeEndpoint(decodeWorker, projector);
            applyStatus(decode, decodeStatus(1L, Map.of()));
            DecodeEndpoint.ReservationHandle reservation;
            var capacity = new DecodeEndpoint.AdmissionCapacity(10L, 90L);
            try (var pin = decode.tryPinGeneration()) {
                reservation = decode.tryReservePlacementPinned(pin, 101L, 16L, 32L, 50, capacity);
                assertNotNull(reservation);
                var acquisition = decode.acquireEngineDispatchPermit(reservation, capacity);
                assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.ACQUIRED, acquisition.status());
                assertEquals(DecodeEndpoint.EngineDispatchPermitTransferStatus.TRANSFERRED,
                        acquisition.permit().transferToEngineLifecycle());
            }
            var context = RequestLifecycleTestSupport.context(config, 101L);
            var future = requests.register(context);
            ScheduledRequest item = new ScheduledRequest(context, future, new Response(), null, null,
                    prefill, decode, reservation, System.currentTimeMillis());
            AtomicReference<PrefillState.RouteReservation> routeReservation = new AtomicReference<>();
            try (var mutation = requests.claimAdmissionMutation(101L, future);
                 var pin = prefill.tryPinGeneration()) {
                assertNotNull(mutation);
                assertNotNull(pin);
                assertTrue(requests.commitItemForPublication(item, () -> {
                    var registered = prefill.reserveUnqueuedRoute(pin, item, 30_000L);
                    assertEquals(PrefillState.CapacityStatus.ACQUIRED, registered.status());
                    routeReservation.set(registered.reservation());
                    return true;
                }));
            }
            try (var routeCommit = prefill.tryBeginRouteCommitAdmission()) {
                assertNotNull(routeCommit);
                var claim = requests.tryClaimRouteDelivery(item, () -> {
                    try (var handoff = routeCommit.commit(List.of(item), List.of(routeReservation.get()))) {
                        return true;
                    }
                });
                assertNotNull(claim);
                requests.beginRouteDelivery(claim, new WorkSnapshot(System.currentTimeMillis(), java.util.List.of(), java.util.List.of(), 0L), 30_000L);
            }
            assertTrue(future.get(2L, TimeUnit.SECONDS).isSuccess());
            assertEquals(1L, prefill.observedRequestCount());

            TaskInfo task = new TaskInfo();
            task.setRequestId(101L);
            task.setInputLength(16L);
            task.setPhase(TaskPhase.RUNNING);
            applyStatus(prefill, status(2L, Map.of("101", task), Map.of()));
            RequestSlot slot = requests.requestSlot(101L);
            synchronized (slot) {
                assertTrue(slot.decisionDeadlineAtMs().isEmpty(), "running Prefill is positive Engine evidence");
            }
            if (decodeAlreadyAccepted) {
                applyStatus(decode, decodeStatus(2L, Map.of("101", task)));
            }

            applyStatus(prefill, status(3L, Map.of(), Map.of("101", task)));
            assertEquals(0L, prefill.observedRequestCount());
            synchronized (slot) {
                assertTrue(slot.isLiveGeneration(), "Prefill completion must retain the Decode lifecycle");
                assertTrue(slot.decisionDeadlineAtMs().isEmpty());
            }
            assertEquals(decodeAlreadyAccepted ? 0L : 32L, decode.routingView().inflightExpectedKv(),
                    "missing Decode acceptance must retain this request's reservation");
            assertTrue(capacity.evaluate(decode.routingView().dispatchUsage(), 16L, 32L).fits(),
                    "a suspected lost request must not isolate a worker with available capacity");
            try (var pin = decode.tryPinGeneration()) {
                var waiting = decode.tryReservePlacementPinned(pin, 102L, 16L, 32L, 50, capacity);
                assertNotNull(waiting);
                var acquisition = decode.acquireEngineDispatchPermit(waiting, capacity);
                assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.ACQUIRED, acquisition.status());
                assertTrue(acquisition.permit().release());
                decode.releaseReservationExact(waiting);
            }

            applyStatus(prefill, status(4L, Map.of(), Map.of("101", task)));
            assertEquals(0L, prefill.observedRequestCount());
        } finally {
            runtime.shutdown();
        }
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
        Runnable projection;
        worker.lock.lock();
        try {
            projection = endpoint.applyPreparedStatus(worker,
                    worker.prepareNewStatus(worker.freezeStatusResponse(response)));
        } finally {
            worker.lock.unlock();
        }
        projection.run();
    }
}
