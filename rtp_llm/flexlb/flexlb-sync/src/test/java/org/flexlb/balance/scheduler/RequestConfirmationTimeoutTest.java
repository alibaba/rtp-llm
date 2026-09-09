package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.config.ConfigService;
import org.flexlb.config.SchedulerConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** Request TTL reclaims real endpoint capacity locally, without an Engine Cancel channel. */
class RequestConfirmationTimeoutTest {
    private static final long REQUEST_ID = 101L;
    private static final long TIMEOUT_MS = TimeUnit.HOURS.toMillis(1L);

    @ParameterizedTest
    @EnumSource(ConfirmationWait.class)
    void unconfirmedRequestReleasesPrefillAndDecodeCapacityAtTtl(ConfirmationWait waiting)
            throws Exception {
        var config = SchedulingTestConfig.newConfig();
        config.setScheduler(SchedulerConfig.direct());
        SchedulingTestConfig.useNonBatchDispatcher(config);
        config.getDispatcher().setMaxInflightPerPrefillWorker(1);
        config.getRequestLifecycle().getRequest().setTimeoutMs(
                waiting == ConfirmationWait.AUTOMATIC_TIMER ? 300L : TIMEOUT_MS);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var reporter = mock(BatchSchedulerReporter.class);
        var requestReporter = mock(RequestSchedulerReporter.class);
        var requests = new RequestRegistry(service, reporter, requestReporter);
        var projector = new EndpointEventProjector(requests);
        var endpoints = new EndpointRegistry(service, projector, reporter,
                new RouteDeliveryStrategy(requests, new DeliveryMetrics(reporter)), new PlacementAvailability());
        var runtime = new SchedulerRuntime(requests, endpoints, reporter, requestReporter);
        DecodeEndpoint decode = null;
        try {
            WorkerStatus prefillWorker = worker(RoleType.PREFILL, "127.0.0.1");
            PrefillEndpoint prefill;
            prefillWorker.lock.lock();
            try {
                prefill = (PrefillEndpoint) endpoints.publishPreparedEndpoint(prefillWorker.getIpPort(),
                        prefillWorker, prefillWorker.prepareNewStatus(
                                prefillWorker.freezeStatusResponse(status(RoleType.PREFILL)))).endpoint();
            } finally {
                prefillWorker.lock.unlock();
            }
            decode = new DecodeEndpoint(worker(RoleType.DECODE, "127.0.0.2"), projector);
            applyStatus(decode, status(RoleType.DECODE));
            var capacity = new DecodeEndpoint.AdmissionCapacity(1L, 90L);
            DecodeEndpoint.ReservationHandle reservation;
            try (var pin = decode.tryPinGeneration()) {
                reservation = decode.tryReservePlacementPinned(pin, REQUEST_ID, 16L, 32L, 50, capacity);
                assertNotNull(reservation);
                var acquired = decode.acquireEngineDispatchPermit(reservation, capacity);
                assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.ACQUIRED, acquired.status());
                assertEquals(DecodeEndpoint.EngineDispatchPermitTransferStatus.TRANSFERRED,
                        acquired.permit().transferToEngineLifecycle());
            }
            var context = RequestLifecycleTestSupport.context(config, REQUEST_ID);
            var future = requests.register(context);
            RequestSlot slot = requests.requestSlot(REQUEST_ID);
            ServerStatus prefillMetadata = new ServerStatus();
            prefillMetadata.setRole(RoleType.PREFILL);
            prefillMetadata.setServerIp("127.0.0.1");
            prefillMetadata.setGrpcPort(8081);
            ScheduledRequest item = new ScheduledRequest(context, future, new Response(), prefillMetadata,
                    null, prefill, decode, reservation, slot.createdAtMs());
            AtomicReference<PrefillState.RouteReservation> routeReservation = new AtomicReference<>();
            try (var mutation = requests.claimAdmissionMutation(REQUEST_ID, future);
                 var pin = prefill.tryPinGeneration()) {
                assertNotNull(mutation);
                assertNotNull(pin);
                assertTrue(requests.commitItemForPublication(item, () -> {
                    var reserved = prefill.reserveUnqueuedRoute(pin, item, 30_000L);
                    assertEquals(PrefillState.CapacityStatus.ACQUIRED, reserved.status());
                    routeReservation.set(reserved.reservation());
                    return true;
                }));
            }
            RequestRegistry.DeliveryClaim claim;
            try (var routeCommit = prefill.tryBeginRouteCommitAdmission()) {
                assertNotNull(routeCommit);
                claim = requests.tryClaimRouteDelivery(item, () -> {
                    try (var handoff = routeCommit.commit(List.of(item), List.of(routeReservation.get()))) {
                        return true;
                    }
                });
                assertNotNull(claim);
                requests.beginDelivery(claim,
                        new WorkSnapshot(System.currentTimeMillis(), List.of(), List.of(), 0L), 30_000L);
            }
            if (waiting == ConfirmationWait.UNCERTAIN_REPLY) {
                requests.complete(claim, DeliveryResult.uncertain(new IllegalStateException("reply was lost")));
            }
            assertEquals(1, requests.liveRequestCount());
            assertFalse(future.isDone());
            assertEquals(1L, prefill.observedRequestCount());
            assertEquals(16L, decode.routingView().inflightHardKv());
            assertEquals(32L, decode.routingView().inflightExpectedKv());
            assertEquals(1, decode.routingView().engineCapacityUsed());

            if (waiting != ConfirmationWait.AUTOMATIC_TIMER) {
                requests.expireInactiveRequest(slot, slot.createdAtMs() + TIMEOUT_MS);
            }

            // AUTOMATIC_TIMER relies only on ExpirationTimer; no manual expiry entry point runs.
            assertFalse(future.get(2L, TimeUnit.SECONDS).isSuccess());
            assertEquals(RequestState.Phase.TIMED_OUT, requests.getRequestState(REQUEST_ID, 0L).state());
            assertEquals(0, requests.liveRequestCount());
            assertEquals(0L, prefill.observedRequestCount());
            assertEquals(0, prefill.getLocallyOwnedRequestCount());
            assertEquals(0L, decode.routingView().inflightHardKv());
            assertEquals(0L, decode.routingView().inflightExpectedKv());
            assertEquals(0, decode.routingView().engineCapacityUsed());

            // Local expiration restores admission capacity without an Engine Cancel channel.
            try (var pin = decode.tryPinGeneration()) {
                var next = decode.tryReservePlacementPinned(pin, 102L, 16L, 32L, 50, capacity);
                assertNotNull(next);
                var acquired = decode.acquireEngineDispatchPermit(next, capacity);
                assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.ACQUIRED, acquired.status());
                assertTrue(acquired.permit().release());
                decode.releaseReservationExact(next);
            }
            requests.onPrefillFact(prefill, RoleType.PREFILL, PrefillState.WorkerStatusFact.active(item));
            requests.onDecodeFact(decode, DecodeEndpoint.WorkerStatusFact.active(reservation));
            assertEquals(RequestState.Phase.TIMED_OUT, requests.getRequestState(REQUEST_ID, 0L).state());
            assertEquals(0, requests.liveRequestCount());
            assertEquals(0L, prefill.observedRequestCount());
            assertEquals(0, decode.routingView().engineCapacityUsed());
        } finally {
            try {
                if (decode != null) {
                    decode.close();
                }
            } finally {
                runtime.shutdown();
            }
        }
    }

    private enum ConfirmationWait {
        MISSING_REPLY,
        UNCERTAIN_REPLY,
        AUTOMATIC_TIMER
    }

    private static WorkerStatus worker(RoleType role, String ip) {
        return WorkerStatus.createDiscovered(role, "g1", ip, 8080, 8081, "test");
    }

    private static WorkerStatusResponse status(RoleType role) {
        var response = new WorkerStatusResponse();
        response.setRole(role);
        response.setAlive(true);
        response.setStatusVersion(1L);
        response.setLatestFinishedVersion(0L);
        response.setRunningTaskInfo(Map.of());
        response.setRunningQueryLen(0L);
        response.setFinishedTaskInfo(Map.of());
        response.setTotalKvCacheTokens(10_000L);
        response.setAvailableKvCacheTokens(10_000L);
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
