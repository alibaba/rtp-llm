package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.EndpointTestSupport;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.master.WorkerStatusTestSupport;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.service.TheoryCacheHitReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
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
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Request TTL reclaims real endpoint capacity locally, without an Engine Cancel channel.
 */
class RequestConfirmationTimeoutTest {
    private static final long REQUEST_ID = 101L;
    private static final long TIMEOUT_MS = TimeUnit.HOURS.toMillis(1L);

    @ParameterizedTest
    @EnumSource(ConfirmationWait.class)
    void unconfirmedRequestReleasesPrefillAndDecodeCapacityAtTtl(ConfirmationWait waiting)
            throws Exception {
        var config = SchedulingTestConfig.newConfig();
        config.setScheduler(ConfigTestSupport.direct());
        SchedulingTestConfig.useNonBatchDispatcher(config);
        config.getDispatcher().setMaxInflightPerPrefillWorker(1);
        config.getRequestLifecycle()
                .getRequest()
                .setTimeoutMs(waiting == ConfirmationWait.AUTOMATIC_TIMER ? 300L : TIMEOUT_MS);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var reporter = mock(DeliveryMetricsReporter.class);
        var requestReporter = mock(RequestSchedulerReporter.class);
        var requests =
                SchedulerTestSupport.create(
                        service, reporter, requestReporter, mock(TheoryCacheHitReporter.class));
        var projector = requests;
        var endpoints =
                new EndpointRegistry(
                        service,
                        SchedulerTestSupport.repository(projector),
                        reporter,
                        DeliveryTestSupport.route(reporter),
                        new PlacementAvailability(),
                        org.flexlb.balance.planner.BuiltinGroupPlanner.from(
                                service.loadBalanceConfig()),
                        order ->
                                org.flexlb.balance.endpoint.OrderedPrefillRequestQueue.ordered(
                                        16, order));
        var runtime =
                new SchedulerTestServices(
                        SchedulerTestSupport.repository(requests),
                        endpoints,
                        reporter,
                        requestReporter,
                        org.mockito.Mockito.mock(DeliveryDispatcher.class),
                        service,
                        org.mockito.Mockito.mock(EngineGrpcClient.class));
        DecodeEndpoint decode = null;
        try {
            WorkerStatus prefillEndpoint = worker(RoleType.PREFILL, "127.0.0.1");
            PrefillEndpoint prefill;
            prefillEndpoint.lock.lock();
            try {
                prefill =
                        (PrefillEndpoint)
                                endpoints.publishPreparedEndpoint(
                                        prefillEndpoint.getLogicalIpPort(),
                                        prefillEndpoint,
                                        prefillEndpoint.prepareNewStatus(
                                                WorkerStatusTestSupport.freezeStatusResponse(
                                                        prefillEndpoint,
                                                        status(RoleType.PREFILL))));
            } finally {
                prefillEndpoint.lock.unlock();
            }
            decode =
                    EndpointTestSupport.decode(
                            worker(RoleType.DECODE, "127.0.0.2"),
                            SchedulerTestSupport.repository(projector));
            applyStatus(decode, status(RoleType.DECODE));
            var capacity = new DecodeResources.AdmissionCapacity(1L, 90L);
            DecodeResources.RequestReservation reservation;
            {
                var selectedEndpoint = decode;

                reservation =
                        decode.tryReserveQueuedRequest(
                                        String.valueOf(REQUEST_ID), 16L, 32L, 50, capacity)
                                .value();
                assertNotNull(reservation);
                var acquired = decode.acquireDispatchPermit(reservation, capacity);
                assertEquals(
                        DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED,
                        acquired.status());
                assertEquals(
                        DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED,
                        acquired.value().dispatch().status());
            }
            var context = RequestProtocolTestSupport.context(config, REQUEST_ID);
            var future = RequestProtocolTestSupport.submitForManualPlacement(requests, context);
            RequestContext requestContext =
                    requests.requests.findActive(String.valueOf(REQUEST_ID));
            ServerStatus prefillMetadata = new ServerStatus();
            prefillMetadata.setRole(RoleType.PREFILL);
            prefillMetadata.setServerIp("127.0.0.1");
            prefillMetadata.setGrpcPort(8081);
            context.data().setFuture(future);
            RouteExecution item =
                    SchedulingTestConfig.createRoute(
                            context,
                            new Response(),
                            prefillMetadata,
                            null,
                            prefill,
                            decode,
                            reservation,
                            requestContext.snapshot().createdAtMs());
            AtomicReference<PrefillEndpoint.RequestRecord> routeReservation =
                    new AtomicReference<>();
            {
                var selectedEndpoint = prefill;
                try (var mutation =
                                SchedulerTestSupport.claim(
                                        requests, String.valueOf(REQUEST_ID), future);
                        var admissionCompletion1 =
                                RequestProtocolTestSupport.finishOnExit(mutation)) {
                    assertNotNull(mutation);
                    assertNotNull(selectedEndpoint);
                    assertTrue(
                            (RequestProtocolTestSupport.commitRoute(
                                            requests,
                                            item,
                                            RequestProtocolTestSupport.publication(
                                                    () -> {
                                                        var reserved =
                                                                prefill.reserveUnqueuedRoute(
                                                                        item, 30_000L);
                                                        assertTrue(reserved.accepted());
                                                        routeReservation.set(reserved.value());
                                                        return true;
                                                    }))
                                    == org.flexlb.balance.PlacementResult.Status.SUCCESS));
                }
            }
            Delivery.Member claim;
            {
                java.util.function.BooleanSupplier commitPrefill =
                        () -> {
                            {
                                var handoff =
                                        org.flexlb.balance.endpoint.EndpointTestSupport
                                                .commitDirect(
                                                        prefill, item, routeReservation.get());

                                return true;
                            }
                        };
                claim =
                        waiting == ConfirmationWait.UNCERTAIN_REPLY
                                ? RequestProtocolTestSupport.claimBatchWithoutPrediction(
                                        requests, item, 1L, commitPrefill)
                                : RequestProtocolTestSupport.claimRouteWithoutPrediction(
                                        requests, item, commitPrefill);
                assertNotNull(claim);
                requests.onDeliveryResult(
                        new ContextResult.Prediction(
                                claim,
                                org.flexlb.balance.prediction.PrefillTimePredictor
                                        .predictedVisibilityDeadlineMs(
                                                new WorkSnapshot(
                                                        System.currentTimeMillis(),
                                                        List.of(),
                                                        List.of(),
                                                        0L),
                                                0L,
                                                30_000L,
                                                System.currentTimeMillis()),
                                System.currentTimeMillis()));
            }
            if (waiting == ConfirmationWait.UNCERTAIN_REPLY) {
                org.mockito.Mockito.when(
                                SchedulerTestSupport.engineClient(requests)
                                        .cancelAsync(
                                                org.mockito.ArgumentMatchers.any(),
                                                anyInt(),
                                                org.mockito.ArgumentMatchers.any(),
                                                org.mockito.ArgumentMatchers.anyLong()))
                        .thenReturn(
                                java.util.concurrent.CompletableFuture.completedFuture(
                                        EngineGrpcClient.CancelReply.REQUEST_CLEANED));
                assertTrue(claim.route().ctx().data().getRequestScheduler().tryStartSend(claim));
                requests.onDeliveryResult(
                        new ContextResult.TransportResult(
                                claim,
                                Delivery.Result.uncertain(
                                        new IllegalStateException("reply was lost")),
                                System.currentTimeMillis()));
            }
            assertEquals(1, SchedulerTestSupport.repository(requests).liveRequestCount());
            assertFalse(future.isDone());
            assertEquals(1L, prefill.admissionSummary(0, prefill.requestLimit).occupiedRequests());
            assertEquals(16L, decode.routingView().inflightHardKv());
            assertEquals(32L, EndpointTestSupport.expectedReservedKv(decode.resourceSnapshot()));
            assertEquals(1, decode.routingView().engineCapacityUsed());

            if (waiting != ConfirmationWait.AUTOMATIC_TIMER) {
                RequestProtocolTestSupport.expireInactiveRequest(
                        requests,
                        requestContext,
                        RequestProtocolTestSupport.<Long>inspect(
                                requests, requestContext, "inactivityExpiresAtMsLocked"));
            }

            // AUTOMATIC_TIMER relies only on ExpirationTimer; no manual expiry entry point runs.
            assertFalse(future.get(2L, TimeUnit.SECONDS).isSuccess());
            assertEquals(
                    RequestState.Phase.TIMED_OUT,
                    SchedulerTestSupport.repository(requests)
                            .getRequestState(String.valueOf(REQUEST_ID), 0L)
                            .state());
            SchedulerTestSupport.awaitRequestWork(SchedulerTestSupport.runtime(requests));
            RequestProtocolTestSupport.awaitCondition(
                    () -> SchedulerTestSupport.repository(requests).liveRequestCount() == 0);
            assertEquals(0, SchedulerTestSupport.repository(requests).liveRequestCount());
            assertEquals(0L, prefill.admissionSummary(0, prefill.requestLimit).occupiedRequests());
            assertEquals(0, prefill.stats().locallyOwnedRequests());
            assertEquals(0L, decode.routingView().inflightHardKv());
            assertEquals(0L, EndpointTestSupport.expectedReservedKv(decode.resourceSnapshot()));
            assertEquals(0, decode.routingView().engineCapacityUsed());

            // Admission resumes only after delivery settlement and exact local cleanup.
            {
                var selectedEndpoint = decode;

                var next =
                        decode.tryReserveQueuedRequest(String.valueOf(102L), 16L, 32L, 50, capacity)
                                .value();
                assertNotNull(next);
                var acquired = decode.acquireDispatchPermit(next, capacity);
                assertEquals(
                        DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED,
                        acquired.status());
                assertTrue(acquired.value().release());
                decode.release(next, DecodeResources.ReleaseReason.LOCAL_ROLLBACK).status();
            }
            RequestProtocolTestSupport.applyPrefillStatus(
                    requests,
                    prefill,
                    RoleType.PREFILL,
                    new PrefillEndpoint.PrefillRequestStatus(
                            item, PrefillEndpoint.PrefillRequestStatus.Kind.ACTIVE, 0L));
            RequestProtocolTestSupport.applyDecodeStatus(
                    requests, decode, DecodeResources.DecodeRequestStatus.active(reservation));
            assertEquals(
                    RequestState.Phase.TIMED_OUT,
                    SchedulerTestSupport.repository(requests)
                            .getRequestState(String.valueOf(REQUEST_ID), 0L)
                            .state());
            SchedulerTestSupport.awaitRequestWork(SchedulerTestSupport.runtime(requests));
            RequestProtocolTestSupport.awaitCondition(
                    () -> SchedulerTestSupport.repository(requests).liveRequestCount() == 0);
            assertEquals(0, SchedulerTestSupport.repository(requests).liveRequestCount());
            assertEquals(0L, prefill.admissionSummary(0, prefill.requestLimit).occupiedRequests());
            assertEquals(0, decode.routingView().engineCapacityUsed());
        } finally {
            try {
                if (decode != null) {
                    decode.close();
                }
            } finally {
                SchedulerTestSupport.shutdown(runtime);
            }
        }
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
        org.flexlb.balance.endpoint.WorkerEndpoint.StatusChange projection;
        worker.lock.lock();
        try {
            projection =
                    endpoint.applyPreparedStatus(
                            worker,
                            worker.prepareNewStatus(
                                    WorkerStatusTestSupport.freezeStatusResponse(
                                            worker, response)));
        } finally {
            worker.lock.unlock();
        }
        endpoint.publishStatus(projection);
    }

    private enum ConfirmationWait {
        MISSING_REPLY,
        UNCERTAIN_REPLY,
        AUTOMATIC_TIMER
    }
}
