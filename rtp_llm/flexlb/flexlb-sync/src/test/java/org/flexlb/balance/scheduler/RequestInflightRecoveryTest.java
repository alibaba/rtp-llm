package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.master.WorkerStatusTestSupport;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.*;

/**
 * Real endpoint ledgers, status reducers, request ownership and capacity gates.
 * Transport replies are injected; this does not run an Engine or a WorkerBatcher thread.
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
            f.send(rejectedBeforeTerminal
                    ? DeliveryResult.prefillRejected(new IllegalStateException("Prefill prepare failed"))
                    : DeliveryResult.delivered());
            if (!rejectedBeforeTerminal) {
                assertTrue(f.route.future().get(2, TimeUnit.SECONDS).isSuccess());
                f.owner.cancel(Fixture.REQUEST_ID, 0L, CancelReason.CLIENT_CANCELLED);
            } else {
                assertFalse(f.route.future().get(2, TimeUnit.SECONDS).isSuccess());
            }
            f.drain();
            assertTrue(f.owner.requests.isCurrent(f.context));
            assertFalse(f.claim.owner().data().getScheduler().runtime.signals().delivery(f.claim).toCompletableFuture().isDone());

            f.prefillTerminal(rejectedBeforeTerminal ? 42L : 0L);
            f.decodeTerminal();
            f.drain();

            // Establish the safety evidence before asserting the missing lifecycle transition.
            assertEquals(0L, f.prefill.ownership.admissionSummary(0, f.prefill.requestLimit).occupiedRequests());
            assertEquals(0L, f.decode.routingView().totalLoad());
            assertFalse(f.decode.ownership(f.reservation).owned());
            assertTrue(f.prefillState.batchCapacityAvailable(1));
            f.assertNextDecodeReservationAccepted();

            assertAll("Both exact endpoint terminals have released ownership; the sender has exited",
                    () -> assertTrue(f.claim.owner().data().getScheduler().runtime.signals().delivery(f.claim).toCompletableFuture().isDone(),
                            "Delivery remains unsettled even though both Engine stages have ended"),
                    () -> assertFalse(f.owner.requests.isCurrent(f.context),
                            "An ended request must leave the active repository"));
        }
    }

    @ParameterizedTest(name = "prefillErrorTerminal={0}")
    @ValueSource(booleans = {false, true})
    void rejectedNotFoundAndEmptyFullMustRetainUnprovedOwnership(boolean prefillErrorTerminal)
            throws Exception {
        try (Fixture f = new Fixture()) {
            f.send(DeliveryResult.prefillRejected(new IllegalStateException("Prefill prepare failed")));
            assertFalse(f.route.future().get(2, TimeUnit.SECONDS).isSuccess());
            f.drain();
            if (prefillErrorTerminal) { f.prefillTerminal(42L); }
            else { f.emptyFull(f.prefill); }
            f.emptyFull(f.decode);
            f.drain();

            // Java cannot infer "Decode never started" from these wire replies.
            assertEquals(prefillErrorTerminal ? 0L : 1L,
                    f.prefill.ownership.admissionSummary(0, f.prefill.requestLimit).occupiedRequests());
            assertEquals(1L, f.decode.routingView().totalLoad());
            assertTrue(f.decode.ownership(f.reservation).owned());
            assertFalse(f.claim.owner().data().getScheduler().runtime.signals().delivery(f.claim).toCompletableFuture().isDone());
            assertEquals(prefillErrorTerminal, f.prefillState.batchCapacityAvailable(1));
            f.assertNextDecodeReservationRejected();

            assertEquals(0, f.decode.evictExpiredRequests(-1L, f.owner.requests::retainsIdentity),
                    "A live exact request identity protects its uncertain reservation");
            RequestProtocolTestSupport.expireInactiveRequest(f.owner, f.context, Long.MAX_VALUE);
            f.drain();
            assertEquals(1L, f.decode.routingView().totalLoad());
            assertTrue(f.owner.requests.isCurrent(f.context));
            f.assertNextDecodeReservationRejected();
        }
    }

    @Test
    void definiteNotSentMustReleaseBothLedgersAndReopenCapacity() throws Exception {
        try (Fixture f = new Fixture()) {
            f.send(DeliveryResult.notSent(new IllegalStateException("Sender never called Engine")));
            f.drain();
            assertFalse(f.route.future().get(2, TimeUnit.SECONDS).isSuccess());
            f.assertReleased();
        }
    }

    @Test
    void exactRemoteCleanedProofMustReleaseRejectedDeliveryAndReopenCapacity() throws Exception {
        try (Fixture f = new Fixture()) {
            f.send(DeliveryResult.prefillRejected(new IllegalStateException("Prefill prepare failed")));
            f.drain();
            assertEquals(1L, f.decode.routingView().totalLoad());
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.CleanupAcknowledgement(f.claim, ContextResult.CancelEvidenceKind.CLEANED, System.currentTimeMillis()));
            f.drain();
            f.assertReleased();
        }
    }

    @Test
    void orderedPrefixStopsAtBusyDecodeWhileLaterRouteReservesAnOtherwiseDispatchableDecode()
            throws Exception {
        try (Fixture f = new Fixture()) {
            DecodeResources.ReservationHandle occupyingD1;
            try (var pin = f.decode.tryPinGeneration()) {
                occupyingD1 = f.decode.tryReserveQueuedRequest(pin, 201L, 16L, 32L, 50, null).handle();
            }
            assertNotNull(occupyingD1);
            var occupied = f.decode.acquireDispatchPermit(occupyingD1, f.capacity);
            assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED, occupied.status());
            assertEquals(DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED,
                    occupied.handle().dispatch().status());
            try {
                var d2 = (DecodeEndpoint) f.publish(RoleType.DECODE, "127.0.0.3");
                DecodeResources.ReservationHandle reservedD2;
                try (var pin = d2.tryPinGeneration()) {
                    reservedD2 = d2.tryReserveQueuedRequest(pin, 302L, 16L, 32L, 50, f.capacity).handle();
                }
                assertNotNull(reservedD2);
                var behindHead = f.enqueue(302L, d2, reservedD2);
                var state = (org.flexlb.balance.endpoint.PrefillState) ReflectionTestUtils.getField(f.prefill, "ownership");
                ReflectionTestUtils.setField(f.owner.runtime.deliveries(), "decodeDeliveryCapacity", f.capacity);
                var worker = new WorkerBatcher("ordered-prefix", f.prefill, QueueExecutionSettings.capture(f.config),
                        new DeliveryDispatcher(null, DeliveryTestSupport.service(f.config), new DeliveryMetricsReporter(null)), state);
                var boundary = worker.processSelection(List.of(f.route, behindHead), "ordered-prefix",
                        mock(PrefillTimePredictor.Evaluator.class), java.util.OptionalLong.empty());
                assertNotNull(boundary, "The ordered prefix cannot skip its blocked head");
                assertEquals(CapacityBoundary.Status.UNAVAILABLE, boundary.status());
                assertEquals(0L, d2.routingView().engineCapacityUsed());
                assertEquals(1L, d2.routingView().totalLoad(), "Queued B already occupies D2 placement capacity");
                var bPermit = d2.acquireDispatchPermit(reservedD2, f.capacity);
                assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED, bPermit.status(),
                        "B itself could dispatch on D2 if ordered-prefix policy allowed reaching it");
                bPermit.handle().release();
                try (var pin = d2.tryPinGeneration()) {
                    assertNull(d2.tryReserveQueuedRequest(pin, 303L, 16L, 32L, 50, f.capacity).handle(),
                            "New C cannot reserve D2 while queued B holds its sole placement slot");
                }
            } finally {
                f.decode.release(occupyingD1, DecodeResources.ReleaseReason.REMOTE_CLEANUP).status();
            }
        }
    }

    private static final class Fixture implements AutoCloseable {
        static final long REQUEST_ID = 101L;
        static final long BATCH_ID = 51L;
        final AbstractRequestScheduler owner;
        final FlexlbConfig config;
        final EndpointRegistry endpoints;
        final PrefillEndpoint prefill;
        final DecodeEndpoint decode;
        final PrefillState prefillState;
        final DecodeResources.AdmissionCapacity capacity =
                new DecodeResources.AdmissionCapacity(1L, 90L);
        final DecodeResources.ReservationHandle reservation;
        final RequestContext context;
        final RequestRoute route;
        DeliveryClaim claim;
        long prefillVersion = 1L;
        long decodeVersion = 1L;

        Fixture() {
            config = SchedulingTestConfig.batchConfig();
            SchedulingTestConfig.useBatchDispatcher(config);
            var service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            var reporter = mock(DeliveryMetricsReporter.class);
            owner = SchedulerTestSupport.create(service, reporter, mock(RequestSchedulerReporter.class),
                    mock(RecentCacheKeyTraceReporter.class));
            endpoints = new EndpointRegistry(service, owner.requests, reporter,
                    DeliveryTestSupport.route(reporter),
                    new PlacementAvailability());
            ReflectionTestUtils.setField(owner.runtime, "endpoints", endpoints);
            when(SchedulerTestSupport.engineClient(owner).cancelAsync(any(), anyInt(), any(), anyLong()))
                    .thenReturn(CompletableFuture.completedFuture(EngineGrpcClient.CancelReply.NOT_FOUND));
            prefill = (PrefillEndpoint) publish(RoleType.PREFILL, "127.0.0.1");
            decode = (DecodeEndpoint) publish(RoleType.DECODE, "127.0.0.2");
            try (var pin = decode.tryPinGeneration()) {
                reservation = decode.tryReserveQueuedRequest(pin, REQUEST_ID, 16L, 32L, 50, capacity).handle();
            }
            assertNotNull(reservation);
            context = RequestProtocolTestSupport.context(config, REQUEST_ID);
            var future = RequestProtocolTestSupport.register(owner, context);
            context.data().setFuture(future);
            route = SchedulingTestConfig.createRoute(context, new Response(), null, null,
                    prefill, decode, reservation, System.currentTimeMillis());
            prefillState = (PrefillState) ReflectionTestUtils.getField(prefill, "ownership");
            prefillState.lock.lock();
            try { ReflectionTestUtils.invokeMethod(prefillState, "enableQueueLocked",
                    Comparator.comparingLong(RequestRoute::requestId)); }
            finally { prefillState.lock.unlock(); }
            try (var admission = owner.claimAdmissionHandle(REQUEST_ID, future);
                 var completion = RequestProtocolTestSupport.finishOnExit(admission)) {
                assertNotNull(admission);
                assertEquals(PlacementResult.Status.SUCCESS, owner.commitRoute(route,
                        RequestProtocolTestSupport.publication(() -> {
                            prefillState.lock.lock();
                            try { return prefillState.updateQueue(PrefillState.QueueChange.ENQUEUE,
                                    route, Long.MAX_VALUE, System.currentTimeMillis()) == route; }
                            finally { prefillState.lock.unlock(); }
                        })));
            }
        }

        WorkerEndpoint publish(RoleType role, String ip) {
            var worker = WorkerStatus.createDiscovered(role, "g1", ip, 8080, 8081, "test");
            worker.lock.lock();
            try {
                return endpoints.publishPreparedEndpoint(worker.getIpPort(), worker,
                        worker.prepareNewStatus(WorkerStatusTestSupport.freezeStatusResponse(worker,
                                status(role, 1L, Map.of()))));
            } finally { worker.lock.unlock(); }
        }

        RequestRoute enqueue(long requestId, DecodeEndpoint endpoint,
                             DecodeResources.ReservationHandle held) {
            var requestContext = RequestProtocolTestSupport.context(config, requestId);
            var future = RequestProtocolTestSupport.register(owner, requestContext);
            requestContext.data().setFuture(future);
            var item = SchedulingTestConfig.createRoute(requestContext, new Response(), null, null,
                    prefill, endpoint, held, System.currentTimeMillis());
            try (var admission = owner.claimAdmissionHandle(requestId, future);
                 var completion = RequestProtocolTestSupport.finishOnExit(admission)) {
                assertNotNull(admission);
                assertEquals(PlacementResult.Status.SUCCESS, owner.commitRoute(item,
                        RequestProtocolTestSupport.publication(() -> {
                            prefillState.lock.lock();
                            try { return prefillState.updateQueue(PrefillState.QueueChange.ENQUEUE,
                                    item, Long.MAX_VALUE, System.currentTimeMillis()) == item; }
                            finally { prefillState.lock.unlock(); }
                        })));
            }
            return item;
        }

        void send(DeliveryResult result) {
            var acquisition = decode.acquireDispatchPermit(reservation, capacity);
            assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED, acquisition.status());
            assertEquals(DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED,
                    acquisition.handle().dispatch().status());
            var batch = prefill.reserveBatch(route, BATCH_ID, 1).reservation();
            assertNotNull(batch);
            PrefillState.CommittedHandoff handoff;
            prefillState.lock.lock();
            try { handoff = prefillState.commitLocked(new PrefillState.CommitPlan(PrefillState.CommitKind.BATCH, List.of(route), List.of(), batch, new long[]{30_000L}, null, System.currentTimeMillis()), null); }
            finally { prefillState.lock.unlock(); }
            assertNotNull(handoff);
            try (handoff) {
                claim = RequestProtocolTestSupport.claimBatchWithoutPrediction(owner, route, BATCH_ID, () -> true);
                assertNotNull(claim);
                assertTrue(owner.tryStartSend(claim));
                owner.onDeliveryEvidence(claim, new ContextResult.Prediction(claim, org.flexlb.balance.prediction.PrefillPredictionBoundary.predictedVisibilityDeadlineMs(new WorkSnapshot(System.currentTimeMillis(), List.of(), List.of(), 0L), 30_000L, claim.owner().data().getConfig().getRequestLifecycle().getDecision().getLifetime(), System.currentTimeMillis()), System.currentTimeMillis()));
                owner.onDeliveryEvidence(claim, new ContextResult.TransportResult(claim, result, System.currentTimeMillis()));
            }
        }

        void prefillTerminal(long errorCode) {
            apply(prefill, status(RoleType.PREFILL, ++prefillVersion, Map.of("101", terminal(errorCode))));
        }

        void decodeTerminal() {
            apply(decode, status(RoleType.DECODE, ++decodeVersion, Map.of("101", terminal(0L))));
        }

        void emptyFull(WorkerEndpoint endpoint) {
            var role = endpoint == prefill ? RoleType.PREFILL : RoleType.DECODE;
            apply(endpoint, status(role, endpoint == prefill ? ++prefillVersion : ++decodeVersion, Map.of()));
        }

        TaskInfo terminal(long errorCode) {
            var task = new TaskInfo();
            task.setRequestId(REQUEST_ID);
            task.setBatchId(BATCH_ID);
            task.setInputLength(16L);
            task.setPhase(TaskPhase.RUNNING);
            task.setErrorCode(errorCode);
            return task;
        }

        void drain() { SchedulerTestSupport.awaitRequestWork(owner.runtime); }

        void assertNextDecodeReservationRejected() {
            try (var pin = decode.tryPinGeneration()) {
                assertNull(decode.tryReserveQueuedRequest(pin, 102L, 16L, 32L, 50, capacity).handle(),
                        "The retained shadow fills the one-request Decode placement limit");
            }
        }

        void assertNextDecodeReservationAccepted() {
            try (var pin = decode.tryPinGeneration()) {
                var next = decode.tryReserveQueuedRequest(pin, 102L, 16L, 32L, 50, capacity).handle();
                assertNotNull(next, "Released capacity must admit the next request");
                decode.release(next, DecodeResources.ReleaseReason.LOCAL_ROLLBACK).status();
            }
        }

        void assertReleased() {
            assertEquals(0L, prefill.ownership.admissionSummary(0, prefill.requestLimit).occupiedRequests());
            assertEquals(0L, decode.routingView().totalLoad());
            assertFalse(decode.ownership(reservation).owned());
            assertTrue(prefillState.batchCapacityAvailable(1));
            assertTrue(claim.owner().data().getScheduler().runtime.signals().delivery(claim).toCompletableFuture().isDone());
            assertFalse(owner.requests.isCurrent(context));
            assertNextDecodeReservationAccepted();
        }

        @Override
        public void close() {
            // Inject strong proof only AFTER each test's assertions. This prevents an unrelated
            // shutdown timeout from hiding the missing terminal-settlement assertion.
            if (claim != null && !claim.owner().data().getScheduler().runtime.signals().delivery(claim).toCompletableFuture().isDone()) {
                owner.onDeliveryEvidence(claim, new ContextResult.CleanupAcknowledgement(claim, ContextResult.CancelEvidenceKind.CLEANED, System.currentTimeMillis()));
            }
            drain();
            endpoints.close();
            drain();
            owner.runtime.shutdown();
        }
    }

    private static WorkerStatusResponse status(RoleType role, long version, Map<String, TaskInfo> finished) {
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
            projection = endpoint.applyPreparedStatus(worker,
                    worker.prepareNewStatus(WorkerStatusTestSupport.freezeStatusResponse(worker, response)));
        } finally { worker.lock.unlock(); }
        endpoint.publishStatus(projection);
    }
}
