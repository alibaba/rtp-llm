package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointTestSupport;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.RequestContext.DeliveryClaim;
import org.flexlb.config.ConfigService;
import org.flexlb.config.SchedulerConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.flexlb.balance.scheduler.SchedulingTestConfig.freezeInputs;
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
        var reporter = mock(DeliveryMetricsReporter.class);
        var requestReporter = mock(RequestSchedulerReporter.class);
        var requests = org.flexlb.balance.scheduler.SchedulerTestSupport.create(service, reporter, requestReporter,
                mock(RecentCacheKeyTraceReporter.class));
        var projector = requests;
        var endpoints = new EndpointRegistry(service, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(projector), reporter, new RouteDeliveryStrategy(reporter), new PlacementAvailability());
        var runtime = new SchedulerRuntime(org.flexlb.balance.scheduler.SchedulerTestSupport.repository(requests), endpoints, reporter, requestReporter, org.mockito.Mockito.mock(DefaultBatchDispatcher.class), service, org.mockito.Mockito.mock(org.flexlb.service.RecentCacheKeyTraceReporter.class), org.mockito.Mockito.mock(org.flexlb.balance.eviction.EngineCancelChannel.class));
        DecodeEndpoint decode = null;
        try {
            WorkerStatus prefillWorker = worker(RoleType.PREFILL, "127.0.0.1");
            PrefillEndpoint prefill;
            prefillWorker.lock.lock();
            try {
                prefill = (PrefillEndpoint) endpoints.publishPreparedEndpoint(prefillWorker.getIpPort(),
                        prefillWorker, prefillWorker.prepareNewStatus(
                                prefillWorker.freezeStatusResponse(status(RoleType.PREFILL))));
            } finally {
                prefillWorker.lock.unlock();
            }
            decode = EndpointTestSupport.decode(worker(RoleType.DECODE, "127.0.0.2"), org.flexlb.balance.scheduler.SchedulerTestSupport.repository(projector));
            applyStatus(decode, status(RoleType.DECODE));
            var capacity = new DecodeResources.AdmissionCapacity(1L, 90L);
            DecodeResources.ReservationHandle reservation;
            try (var pin = decode.tryPinGeneration()) {
                reservation = decode.tryReserveQueuedRequest(pin, REQUEST_ID, 16L, 32L, 50, capacity);
                assertNotNull(reservation);
                var acquired = decode.acquireDispatchPermit(reservation, capacity);
                assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED, acquired.status());
                assertEquals(DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED,
                        acquired.permit().dispatch());
            }
            var context = RequestProtocolTestSupport.context(config, REQUEST_ID);
            var future = RequestProtocolTestSupport.register(requests, context);
            RequestContext requestContext = requests.findRequestContext(REQUEST_ID);
            ServerStatus prefillMetadata = new ServerStatus();
            prefillMetadata.setRole(RoleType.PREFILL);
            prefillMetadata.setServerIp("127.0.0.1");
            prefillMetadata.setGrpcPort(8081);
            context.setFuture(future);
            RequestRoute item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context), new Response(), prefillMetadata,
                    null, prefill, decode, reservation, requestContext.createdAtMs());
            AtomicReference<PrefillState.RouteReservation> routeReservation = new AtomicReference<>();
            try (var mutation = requests.claimAdmissionHandle(REQUEST_ID, future); var admissionCompletion1 = RequestProtocolTestSupport.finishOnExit(mutation);
                 var pin = prefill.tryPinGeneration()) {
                assertNotNull(mutation);
                assertNotNull(pin);
                assertTrue((requests.commitRoute(item, RequestProtocolTestSupport.publication(() -> {
                    var reserved = prefill.reserveUnqueuedRoute(pin, item, 30_000L);
                    assertEquals(PrefillState.CapacityStatus.ACQUIRED, reserved.status());
                    routeReservation.set(reserved.reservation());
                    return true;
                })) == org.flexlb.balance.PlacementResult.Status.SUCCESS));
            }
            DeliveryClaim claim;
            try (var routeCommit = prefill.tryBeginRouteCommitAdmission()) {
                assertNotNull(routeCommit);
                java.util.function.BooleanSupplier commitPrefill = () -> {
                    try (var handoff = routeCommit.commit(List.of(item), List.of(routeReservation.get()))) {
                        return true;
                    }
                };
                claim = waiting == ConfirmationWait.UNCERTAIN_REPLY
                        ? RequestProtocolTestSupport.claimBatchWithoutPrediction(requests, item, 1L, commitPrefill)
                        : RequestProtocolTestSupport.claimRouteWithoutPrediction(requests, item, commitPrefill);
                assertNotNull(claim);
                requests.setDeliveryPrediction(claim, new WorkSnapshot(System.currentTimeMillis(), List.of(), List.of(), 0L), 30_000L);
            }
            if (waiting == ConfirmationWait.UNCERTAIN_REPLY) {
                org.mockito.Mockito.when(SchedulerTestSupport.cancelChannel(requests).cancel(
                        org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.anyLong(),
                        org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.anyLong()))
                        .thenReturn(java.util.concurrent.CompletableFuture.completedFuture(org.flexlb.balance.eviction.EngineCancelChannel.CancelAck.REQUEST_CLEANED));
                assertTrue(claim.item.ctx().scheduler().tryStartSend(claim));
                claim.item.ctx().scheduler().completeDelivery(claim, DeliveryResult.uncertain(new IllegalStateException("reply was lost")));
            }
            assertEquals(1, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(requests).liveRequestCount());
            assertFalse(future.isDone());
            assertEquals(1L, prefill.admissionSummary(0).occupiedRequests());
            assertEquals(16L, decode.routingView().inflightHardKv());
            assertEquals(32L, EndpointTestSupport.expectedReservedKv(decode.resourceSnapshot()));
            assertEquals(1, decode.routingView().engineCapacityUsed());

            if (waiting != ConfirmationWait.AUTOMATIC_TIMER) {
                RequestProtocolTestSupport.expireInactiveRequest(requests, requestContext,
                        RequestProtocolTestSupport.<Long>inspect(requests, requestContext, "inactivityExpiresAtMsLocked"));
            }

            // AUTOMATIC_TIMER relies only on ExpirationTimer; no manual expiry entry point runs.
            assertFalse(future.get(2L, TimeUnit.SECONDS).isSuccess());
            assertEquals(RequestState.Phase.TIMED_OUT, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(requests).getRequestState(REQUEST_ID, 0L).state());
            SchedulerTestSupport.runtime(requests).continuations().awaitIdle();
            assertEquals(0, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(requests).liveRequestCount());
            assertEquals(0L, prefill.admissionSummary(0).occupiedRequests());
            assertEquals(0, prefill.ownershipStats().locallyOwnedRequests());
            assertEquals(0L, decode.routingView().inflightHardKv());
            assertEquals(0L, EndpointTestSupport.expectedReservedKv(decode.resourceSnapshot()));
            assertEquals(0, decode.routingView().engineCapacityUsed());

            // Admission resumes only after delivery settlement and exact local cleanup.
            try (var pin = decode.tryPinGeneration()) {
                var next = decode.tryReserveQueuedRequest(pin, 102L, 16L, 32L, 50, capacity);
                assertNotNull(next);
                var acquired = decode.acquireDispatchPermit(next, capacity);
                assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED, acquired.status());
                assertTrue(acquired.permit().release());
                decode.release(next, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
            }
            RequestProtocolTestSupport.applyPrefillStatus(requests, prefill, RoleType.PREFILL, PrefillState.PrefillRequestStatus.active(item));
            RequestProtocolTestSupport.applyDecodeStatus(requests, decode, DecodeResources.DecodeRequestStatus.active(reservation));
            assertEquals(RequestState.Phase.TIMED_OUT, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(requests).getRequestState(REQUEST_ID, 0L).state());
            SchedulerTestSupport.runtime(requests).continuations().awaitIdle();
            assertEquals(0, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(requests).liveRequestCount());
            assertEquals(0L, prefill.admissionSummary(0).occupiedRequests());
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
