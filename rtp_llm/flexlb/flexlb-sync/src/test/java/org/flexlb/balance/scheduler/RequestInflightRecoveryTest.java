package org.flexlb.balance.scheduler;

import org.flexlb.balance.RoutingResult;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.master.WorkerStatusTestSupport;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.TheoryCacheHitReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.flexlb.balance.endpoint.EndpointTestSupport.withStateLock;
import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Real endpoint ledgers, status reducers, request ownership and capacity gates.
 * Transport replies are injected; this does not run an Engine or a PrefillQueueScheduler thread.
 *
 * A rejected reply plus NOT_FOUND is deliberately insufficient to release Decode.
 * The red cases supply exact terminal evidence from BOTH endpoints before requiring settlement.
 */
class RequestInflightRecoveryTest {
    @ParameterizedTest(name = "rejectedBeforeTerminal={0}")
    @ValueSource(booleans = {false, true})
    void exactPrefillAndDecodeTerminalsMustSettleAbandonedDelivery(boolean rejectedBeforeTerminal)
            throws Exception {
        try (Fixture f = new Fixture()) {
            f.send(
                    rejectedBeforeTerminal
                            ? Delivery.Result.prefillRejected(
                                    new IllegalStateException("Prefill prepare failed"))
                            : Delivery.Result.delivered());
            if (!rejectedBeforeTerminal) {
                assertTrue(f.route.ctx().data().getFuture().get(2, TimeUnit.SECONDS).isSuccess());
                f.owner.cancel(
                        String.valueOf(Fixture.REQUEST_ID), 0L, CancelReason.CLIENT_CANCELLED);
            } else {
                assertFalse(f.route.ctx().data().getFuture().get(2, TimeUnit.SECONDS).isSuccess());
            }
            f.drain();
            assertTrue(f.owner.repository.isCurrent(f.context));
            assertFalse(f.claim.completion().toCompletableFuture().isDone());

            f.prefillTerminal(rejectedBeforeTerminal ? 42L : 0L);
            f.decodeTerminal();
            f.drain();

            // Establish the safety evidence before asserting the missing lifecycle transition.
            assertEquals(
                    0L, f.prefill.admissionSummary(0, f.prefill.requestLimit).occupiedRequests());

            assertEquals(0L, f.decode.routingView().totalLoad());
            assertFalse(f.decode.ownership(f.reservation).owned());
            assertTrue(f.prefillState.batchCapacityAvailable(1));
            f.assertNextDecodeReservationAccepted();

            assertAll(
                    "Both exact endpoint terminals have released ownership; the sender has exited",
                    () ->
                            assertTrue(
                                    f.claim.completion().toCompletableFuture().isDone(),
                                    "Delivery remains unsettled even though both Engine stages have ended"),
                    () ->
                            assertFalse(
                                    f.owner.repository.isCurrent(f.context),
                                    "An ended request must leave the active repository"));
        }
    }

    @ParameterizedTest(name = "prefillErrorTerminal={0}")
    @ValueSource(booleans = {false, true})
    void rejectedNotFoundAndEmptyFullMustRetainUnprovedOwnership(boolean prefillErrorTerminal)
            throws Exception {
        try (Fixture f = new Fixture()) {
            f.send(
                    Delivery.Result.prefillRejected(
                            new IllegalStateException("Prefill prepare failed")));
            assertFalse(f.route.ctx().data().getFuture().get(2, TimeUnit.SECONDS).isSuccess());
            f.drain();
            if (prefillErrorTerminal) {
                f.prefillTerminal(42L);
            } else {
                f.emptyFull(f.prefill);
            }
            f.emptyFull(f.decode);
            f.drain();

            // Java cannot infer "Decode never started" from these wire replies.
            assertEquals(
                    prefillErrorTerminal ? 0L : 1L,
                    f.prefill.admissionSummary(0, f.prefill.requestLimit).occupiedRequests());
            assertEquals(1L, f.decode.routingView().totalLoad());
            assertTrue(f.decode.ownership(f.reservation).owned());
            assertFalse(f.claim.completion().toCompletableFuture().isDone());
            assertEquals(prefillErrorTerminal, f.prefillState.batchCapacityAvailable(1));
            f.assertNextDecodeReservationRejected();

            assertEquals(
                    0,
                    f.decode.evictExpiredRequests(-1L, f.owner.repository::retainsIdentity),
                    "A live exact request identity protects its uncertain reservation");
            RequestProtocolTestSupport.expireInactiveRequest(f.owner, f.context, Long.MAX_VALUE);
            f.drain();
            assertEquals(1L, f.decode.routingView().totalLoad());
            assertTrue(f.owner.repository.isCurrent(f.context));
            f.assertNextDecodeReservationRejected();
        }
    }

    @Test
    void definiteNotSentMustReleaseBothLedgersAndReopenCapacity() throws Exception {
        try (Fixture f = new Fixture()) {
            f.send(
                    Delivery.Result.notSent(
                            new IllegalStateException("Sender never called Engine")));
            f.drain();
            assertFalse(f.route.ctx().data().getFuture().get(2, TimeUnit.SECONDS).isSuccess());
            f.assertReleased();
        }
    }

    @Test
    void exactRemoteCleanedProofMustReleaseRejectedDeliveryAndReopenCapacity() throws Exception {
        try (Fixture f = new Fixture()) {
            f.send(
                    Delivery.Result.prefillRejected(
                            new IllegalStateException("Prefill prepare failed")));
            f.drain();
            assertEquals(1L, f.decode.routingView().totalLoad());
            f.owner.recordRemoteCleanup(
                    new ContextResult.CleanupAcknowledgement(
                            f.claim,
                            ContextResult.CancelEvidenceKind.CLEANED,
                            System.currentTimeMillis()));
            f.drain();
            f.assertReleased();
        }
    }

    @Test
    void socketResetKeepsUnknownAttemptFencedUntilBothWorkerTerminals() throws Exception {
        try (Fixture f = new Fixture()) {
            f.send(Delivery.Result.uncertain(new IllegalStateException("schedule socket reset")));
            assertFalse(
                    f.route.ctx().data().getFuture().isDone(),
                    "UNKNOWN transport cannot complete the request");
            assertTrue(f.owner.repository.isCurrent(f.context));
            f.assertNextDecodeReservationRejected();

            f.prefillTerminal(0L);
            f.decodeTerminal();
            f.drain();

            f.assertReleased();
            f.assertNextDecodeReservationAccepted();
        }
    }

    @Test
    void retry001SameLogicalRetryCannotReplaceUnknownAttempt() throws Exception {
        try (Fixture f = new Fixture()) {
            f.send(
                    Delivery.Result.uncertain(
                            new IllegalStateException("accepted then socket reset")));
            var duplicate = RequestProtocolTestSupport.submitForManualPlacement(f.owner, f.context);

            assertTrue(
                    duplicate.isDone(), "a retry of the active logical request must be rejected");
            assertFalse(duplicate.join().isSuccess());
            assertTrue(f.owner.repository.isCurrent(f.context));
            f.assertNextDecodeReservationRejected();

            f.prefillTerminal(0L);
            f.decodeTerminal();
            f.drain();
            f.assertReleased();
        }
    }

    @Test
    void orderedPrefixStopsAtBusyDecodeWhileLaterRouteReservesAnOtherwiseDispatchableDecode()
            throws Exception {
        try (Fixture f = new Fixture()) {
            DecodeResources.RequestReservation occupyingD1;
            {
                var selectedEndpoint = f.decode;

                // Both selections fit before the engine concurrency policy falls to one.
                occupyingD1 =
                        f.decode
                                .tryRegisterSelection(
                                        f.decode.routingView(),
                                        String.valueOf(201L),
                                        16L,
                                        32L,
                                        50,
                                        new DecodeResources.AdmissionCapacity(2L, 90L))
                                .value();
            }
            assertNotNull(occupyingD1);
            var occupied = f.decode.acquireDispatchPermit(occupyingD1, f.capacity);
            assertEquals(
                    DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED, occupied.status());
            assertEquals(
                    DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED,
                    occupied.value().dispatch().status());
            try {
                var d2 = (DecodeEndpoint) f.publish(RoleType.DECODE, "127.0.0.3");
                DecodeResources.RequestReservation reservedD2;
                {
                    var selectedEndpoint = d2;

                    reservedD2 =
                            d2.tryRegisterSelection(
                                            d2.routingView(),
                                            String.valueOf(302L),
                                            16L,
                                            32L,
                                            50,
                                            f.capacity)
                                    .value();
                }
                assertNotNull(reservedD2);
                var behindHead = f.enqueue(302L, d2, reservedD2);
                var state = (org.flexlb.balance.endpoint.PrefillEndpoint) f.prefill;
                ReflectionTestUtils.setField(
                        org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(f.owner)
                                .deliveries(),
                        "decodeDeliveryCapacity",
                        f.capacity);
                var worker =
                        org.flexlb.balance.endpoint.EndpointTestSupport.attach(
                                org.flexlb.balance.endpoint.EndpointTestSupport.worker(
                                        "ordered-prefix",
                                        f.prefill,
                                        QueueExecutionSettings.capture(
                                                f.config,
                                                org.flexlb.balance.planner.BuiltinGroupPlanner.from(
                                                        f.config)),
                                        new DeliveryCoordinator(
                                                new org.flexlb.balance.scheduler
                                                        .RequestRepository(),
                                                null,
                                                DeliveryTestSupport.service(f.config),
                                                new DeliveryMetricsReporter(null),
                                                new org.flexlb.balance.scheduler
                                                                .SchedulerConfiguration()
                                                        .decodeAdmissionCapacity(
                                                                DeliveryTestSupport.service(
                                                                        f.config)),
                                                org.mockito.Mockito.mock(
                                                        org.flexlb.balance.scheduler
                                                                .DeliveryExecution.class)),
                                        state,
                                        SchedulerTestSupport.queueDependencies(
                                                org.flexlb.balance.scheduler.SchedulerTestSupport
                                                        .runtime(f.owner))));
                var boundary =
                        worker.processSelection(
                                List.of(f.route, behindHead),
                                "ordered-prefix",
                                mock(PrefillTimePredictor.Evaluator.class),
                                java.util.OptionalLong.empty());
                assertNotNull(boundary, "The ordered prefix cannot skip its blocked head");
                assertEquals(CapacityBoundary.Status.UNAVAILABLE, boundary.status());
                assertEquals(0L, d2.routingView().engineCapacityUsed());
                assertEquals(
                        1L,
                        d2.routingView().totalLoad(),
                        "Queued B already occupies D2 placement capacity");
                var bPermit = d2.acquireDispatchPermit(reservedD2, f.capacity);
                assertEquals(
                        DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED,
                        bPermit.status(),
                        "B itself could dispatch on D2 if ordered-prefix policy allowed reaching it");
                bPermit.value().release();
                {
                    var selectedEndpoint = d2;

                    assertNull(
                            d2.tryRegisterSelection(
                                            d2.routingView(),
                                            String.valueOf(303L),
                                            16L,
                                            32L,
                                            50,
                                            f.capacity)
                                    .value(),
                            "New C cannot reserve D2 while queued B holds its sole placement slot");
                }
            } finally {
                f.decode
                        .release(occupyingD1, DecodeResources.ReleaseReason.REMOTE_CLEANUP)
                        .status();
            }
        }
    }

    private static WorkerStatusResponse status(
            RoleType role, long version, Map<String, TaskInfo> finished) {
        var response = new WorkerStatusResponse();
        response.setRole(role);
        response.setAlive(true);
        response.setStatusVersion(version);
        response.setLatestFinishedVersion(finished.isEmpty() ? 0L : 1L);
        response.setRunningTaskInfo(Map.of());
        response.setRunningQueryLen(0L);
        response.setFinishedTaskInfo(finished);
        response.setTotalKvCacheTokens(10_000L);
        response.setAvailableKvCacheTokens(9_000L);
        return response;
    }

    private static void apply(WorkerEndpoint endpoint, WorkerStatusResponse response) {
        var worker = endpoint.getStatus();
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

    /**
     * Tracks a real P/D registration through controlled transport failures and exact Worker terminal reports.
     */
    private static final class Fixture implements AutoCloseable {
        static final long REQUEST_ID = 101L;
        static final long BATCH_ID = 51L;

        final AbstractRequestScheduler owner;
        final FlexlbConfig config;
        final EndpointRegistry endpoints;
        final PrefillEndpoint prefill;
        final DecodeEndpoint decode;
        final PrefillEndpoint prefillState;
        final DecodeResources.AdmissionCapacity capacity =
                new DecodeResources.AdmissionCapacity(1L, 90L);
        final DecodeResources.RequestReservation reservation;
        final RequestContext context;
        final RouteResult route;

        Delivery.Member claim;

        long prefillVersion = 1L;
        long decodeVersion = 1L;

        Fixture() {
            config = SchedulingTestConfig.batchConfig();
            SchedulingTestConfig.useBatchDispatcher(config);
            var service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            var reporter = mock(DeliveryMetricsReporter.class);
            owner =
                    SchedulerTestSupport.create(
                            service,
                            reporter,
                            mock(RequestSchedulerReporter.class),
                            mock(TheoryCacheHitReporter.class));
            endpoints =
                    new EndpointRegistry(
                            service,
                            owner.repository,
                            reporter,
                            DeliveryTestSupport.route(reporter),
                            new CapacityNotifier(),
                            org.flexlb.balance.planner.BuiltinGroupPlanner.from(
                                    service.loadBalanceConfig()),
                            () -> org.flexlb.balance.endpoint.OrderedPrefillRequestQueue.fifo(16));
            org.flexlb.balance.scheduler.SchedulerTestServices.setDependency(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(owner),
                    "endpoints",
                    endpoints);
            when(SchedulerTestSupport.engineClient(owner)
                            .cancelAsync(any(), anyInt(), any(), anyLong()))
                    .thenReturn(
                            CompletableFuture.completedFuture(
                                    EngineGrpcClient.CancelReply.NOT_FOUND));
            prefill = (PrefillEndpoint) publish(RoleType.PREFILL, "127.0.0.1");
            decode = (DecodeEndpoint) publish(RoleType.DECODE, "127.0.0.2");
            {
                var selectedEndpoint = decode;

                reservation =
                        decode.tryRegisterSelection(
                                        decode.routingView(),
                                        String.valueOf(REQUEST_ID),
                                        16L,
                                        32L,
                                        50,
                                        capacity)
                                .value();
            }
            assertNotNull(reservation);
            context = RequestProtocolTestSupport.context(config, REQUEST_ID);
            var future = RequestProtocolTestSupport.submitForManualPlacement(owner, context);
            context.data().setFuture(future);
            route =
                    RequestExecutionTestSupport.registerPrefillRoute(
                            context, prefill, decode, reservation, 0L, System.currentTimeMillis());
            prefillState = (PrefillEndpoint) prefill;
            prefillState.enableQueue(
                    QueueExecutionSettings.capture(
                            config, org.flexlb.balance.planner.BuiltinGroupPlanner.from(config)));
            var admission = SchedulerTestSupport.claim(owner, String.valueOf(REQUEST_ID), future);
            try (var completion = RequestProtocolTestSupport.finishOnExit(admission)) {
                assertNotNull(admission);
                assertEquals(
                        RoutingResult.Status.SUCCESS,
                        RequestProtocolTestSupport.commitRoute(
                                owner,
                                route,
                                RequestProtocolTestSupport.publication(
                                        () -> {
                                            return withStateLock(
                                                    org.flexlb.balance.endpoint.EndpointTestSupport
                                                            .stateLock(prefillState),
                                                    () -> {
                                                        return org.flexlb.balance.endpoint
                                                                        .EndpointTestSupport
                                                                        .enqueue(
                                                                                prefillState, route)
                                                                == route;
                                                    });
                                        })));
            }
        }

        @Override
        public void close() {
            SchedulerTestServices.scheduler(owner).stop();
            // Inject strong proof only AFTER each test's assertions. This prevents an unrelated
            // shutdown timeout from hiding the missing terminal-settlement assertion.
            if (claim != null && !claim.completion().toCompletableFuture().isDone()) {
                owner.recordRemoteCleanup(
                        new ContextResult.CleanupAcknowledgement(
                                claim,
                                ContextResult.CancelEvidenceKind.CLEANED,
                                System.currentTimeMillis()));
            }
            drain();
            endpoints.close();
            drain();
            SchedulerTestSupport.shutdown(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(owner));
        }

        WorkerEndpoint publish(RoleType role, String ip) {
            var worker = WorkerStatus.createDiscovered(role, "g1", ip, 8080, 8081, "test");
            worker.lock.lock();
            try {
                return endpoints.publishPreparedEndpoint(
                        worker.getLogicalIpPort(),
                        worker,
                        worker.prepareNewStatus(
                                WorkerStatusTestSupport.freezeStatusResponse(
                                        worker, status(role, 1L, Map.of()))));
            } finally {
                worker.lock.unlock();
            }
        }

        RouteResult enqueue(
                long requestId, DecodeEndpoint endpoint, DecodeResources.RequestReservation held) {
            var requestContext = RequestProtocolTestSupport.context(config, requestId);
            var future = RequestProtocolTestSupport.submitForManualPlacement(owner, requestContext);
            requestContext.data().setFuture(future);
            var item =
                    RequestExecutionTestSupport.registerPrefillRoute(
                            requestContext,
                            prefill,
                            endpoint,
                            held,
                            0L,
                            System.currentTimeMillis());
            var admission = SchedulerTestSupport.claim(owner, String.valueOf(requestId), future);
            try (var completion = RequestProtocolTestSupport.finishOnExit(admission)) {
                assertNotNull(admission);
                assertEquals(
                        RoutingResult.Status.SUCCESS,
                        RequestProtocolTestSupport.commitRoute(
                                owner,
                                item,
                                RequestProtocolTestSupport.publication(
                                        () -> {
                                            return withStateLock(
                                                    org.flexlb.balance.endpoint.EndpointTestSupport
                                                            .stateLock(prefillState),
                                                    () -> {
                                                        return org.flexlb.balance.endpoint
                                                                        .EndpointTestSupport
                                                                        .enqueue(prefillState, item)
                                                                == item;
                                                    });
                                        })));
            }
            return item;
        }

        void send(Delivery.Result result) {
            var acquisition = decode.acquireDispatchPermit(reservation, capacity);
            assertEquals(
                    DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED,
                    acquisition.status());
            assertEquals(
                    DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED,
                    acquisition.value().dispatch().status());
            var batch =
                    new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(BATCH_ID, 1);
            assertNotNull(batch);
            var handoff =
                    withStateLock(
                            org.flexlb.balance.endpoint.EndpointTestSupport.stateLock(prefillState),
                            () -> {
                                return prefillState
                                        .dequeueForDelivery(
                                                org.flexlb.balance.scheduler.DeliveryTestSupport
                                                        .queued(
                                                                List.of(route),
                                                                batch,
                                                                new long[] {30_000L}))
                                        .value();
                            });
            assertNotNull(handoff);
            {
                claim =
                        RequestProtocolTestSupport.claimBatchWithoutPrediction(
                                owner, route, BATCH_ID, () -> true);
                assertNotNull(claim);
                assertEquals(
                        ContextResult.SendState.SENDING, route.ctx().snapshot().delivery().send());
                owner.recordDeliveryPrediction(
                        new ContextResult.Prediction(
                                claim,
                                org.flexlb.balance.prediction.PrefillTimePredictor
                                        .predictedVisibilityDeadlineMs(
                                                new WorkSnapshot(
                                                        System.currentTimeMillis(),
                                                        List.of(),
                                                        List.of(),
                                                        0L),
                                                30_000L,
                                                claim.owner()
                                                        .data()
                                                        .getConfig()
                                                        .getRequestLifecycle()
                                                        .getDecision()
                                                        .getLifetime(),
                                                System.currentTimeMillis()),
                                System.currentTimeMillis()));
                owner.onDeliveryResult(
                        new ContextResult.TransportResult(
                                claim, result, System.currentTimeMillis()));
            }
        }

        void prefillTerminal(long errorCode) {
            apply(
                    prefill,
                    status(RoleType.PREFILL, ++prefillVersion, Map.of("101", terminal(errorCode))));
        }

        TaskInfo terminal(long errorCode) {
            var task = new TaskInfo();
            task.setRequestId(String.valueOf(REQUEST_ID));
            task.setBatchId(BATCH_ID);
            task.setInputLength(16L);
            task.setPhase(TaskPhase.RUNNING);
            task.setErrorCode(errorCode);
            return task;
        }

        void decodeTerminal() {
            apply(decode, status(RoleType.DECODE, ++decodeVersion, Map.of("101", terminal(0L))));
        }

        void emptyFull(WorkerEndpoint endpoint) {
            var role = endpoint == prefill ? RoleType.PREFILL : RoleType.DECODE;
            apply(
                    endpoint,
                    status(
                            role,
                            endpoint == prefill ? ++prefillVersion : ++decodeVersion,
                            Map.of()));
        }

        void drain() {
            SchedulerTestSupport.awaitRequestWork(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(owner));
        }

        void assertNextDecodeReservationRejected() {
            {
                var selectedEndpoint = decode;

                assertNull(
                        decode.tryRegisterSelection(
                                        decode.routingView(),
                                        String.valueOf(102L),
                                        16L,
                                        32L,
                                        50,
                                        capacity)
                                .value(),
                        "The retained shadow fills the one-request Decode placement limit");
            }
        }

        void assertReleased() {
            assertEquals(0L, prefill.admissionSummary(0, prefill.requestLimit).occupiedRequests());
            assertEquals(0L, decode.routingView().totalLoad());
            assertFalse(decode.ownership(reservation).owned());
            assertTrue(prefillState.batchCapacityAvailable(1));
            assertTrue(claim.completion().toCompletableFuture().isDone());
            assertFalse(owner.repository.isCurrent(context));
            assertNextDecodeReservationAccepted();
        }

        void assertNextDecodeReservationAccepted() {
            {
                var selectedEndpoint = decode;

                var next =
                        decode.tryRegisterSelection(
                                        decode.routingView(),
                                        String.valueOf(102L),
                                        16L,
                                        32L,
                                        50,
                                        capacity)
                                .value();
                assertNotNull(next, "Released capacity must admit the next request");
                decode.release(next, DecodeResources.ReleaseReason.LOCAL_ROLLBACK).status();
            }
        }
    }
}
