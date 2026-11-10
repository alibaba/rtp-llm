package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.DecodeResources.CapacityRelease;
import org.flexlb.balance.endpoint.DecodeResources.ReservationReleaseResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.EndpointTestSupport;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.dao.loadbalance.Response;
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
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Resource assertions use real endpoint ledgers, not invocations of mocked release methods.
 */
class RequestResourceAccountingTest {
    private static final long ID = 101L;
    private static final long HARD_KV = 160L;
    private static final long EXPECTED_KV = 320L;
    private static final long TOTAL_KV = 10_000L;
    private static final long TTL = TimeUnit.HOURS.toMillis(1);

    @Test
    void schedulerCannotConsumeAnotherEndpointsRealDispatchPermit() throws Exception {
        try (Fixture owner = new Fixture();
                Fixture other = new Fixture()) {
            var permit =
                    other.decode.acquireDispatchPermit(other.reservation, other.capacity).value();
            assertNotNull(permit);
            try {
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                org.flexlb.balance.scheduler.DeliveryTestSupport.transport(
                                                owner.requests.dispatcher)
                                        .commit(
                                                new org.flexlb.balance.scheduler.Delivery.Member(
                                                        owner.item, permit),
                                                Delivery.SendMode.BATCH_ENQUEUE,
                                                1L,
                                                null));
                assertNull(owner.item.ctx().snapshot().delivery());
                assertEquals(ContextResult.Phase.OPEN, owner.item.ctx().snapshot().phase());
                assertNotNull(owner.item.ctx().snapshot().route());
                assertNull(owner.item.ctx().snapshot().delivery());
                assertEquals(1, other.decode.resourceSnapshot().activeDispatchPermits());
                assertEquals(1, other.decode.resourceSnapshot().queuedCount());
                assertFalse(owner.item.ctx().data().getFuture().isDone());
            } finally {
                permit.release();
            }
        }
    }

    @Test
    void partialCleanupFailureStillAttemptsOtherResourcesAndPreservesDecodeSettlement() {
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.batchConfig(), ID);
        var prefill = mock(PrefillEndpoint.class);
        var decode = RequestProtocolTestSupport.decodeEndpoint();
        var reservation = new DecodeResources.RequestReservation(1L, String.valueOf(ID), 1L);
        var item =
                SchedulingTestConfig.createRoute(
                        context,
                        new Response(),
                        null,
                        null,
                        prefill,
                        decode,
                        reservation,
                        System.currentTimeMillis());
        when(decode.release(reservation, DecodeResources.ReleaseReason.NOT_SENT))
                .thenReturn(
                        org.flexlb.balance.endpoint.DecodeResources.ResourceChange.of(
                                DecodeResources.ReservationReleaseResult.RELEASED, false));
        var releaseFailure = new IllegalStateException("Prefill release failed");
        when(prefill.releaseRequest(item)).thenThrow(releaseFailure);
        RequestProtocolTestSupport.closing(item);
        var firstFailure =
                AbstractRequestScheduler.releaseRouteResources(
                        item, DecodeResources.ReleaseReason.NOT_SENT);
        assertFalse(context.snapshot().close().prefillSettled());
        assertTrue(context.snapshot().close().decodeSettled());
        assertSame(releaseFailure, firstFailure);
        org.mockito.Mockito.verify(decode)
                .release(reservation, DecodeResources.ReleaseReason.NOT_SENT);

        org.mockito.Mockito.doReturn(true).when(prefill).releaseRequest(item);
        assertNull(
                AbstractRequestScheduler.releaseRouteResources(
                        item, DecodeResources.ReleaseReason.NOT_SENT));
        assertTrue(context.snapshot().close().prefillSettled());
        assertTrue(context.snapshot().close().decodeSettled());
        org.mockito.Mockito.verify(decode, org.mockito.Mockito.times(1))
                .release(reservation, DecodeResources.ReleaseReason.NOT_SENT);
    }

    @ParameterizedTest
    @EnumSource(LocalEnd.class)
    void localTerminationReclaimsReservationAndPrefillInflight(LocalEnd end) throws Exception {
        try (Fixture f = new Fixture()) {
            f.assertReserved();
            switch (end) {
                case CANCEL ->
                        f.requests.cancel(String.valueOf(ID), 0, CancelReason.CLIENT_CANCELLED);
                case FUTURE_CANCEL -> assertTrue(f.item.ctx().data().getFuture().cancel(false));
                case EXPIRE -> f.expire();
                case SHUTDOWN -> {
                    assertTrue(
                            RequestProtocolTestSupport.closeRegistrationAndAwaitScheduling(
                                    f.requests));
                    f.requests.closeOutstandingAndTerminalize();
                }
            }
            f.assertEmpty();
            f.assertCapacityReusable();
        }
    }

    @Test
    void failedMaterializationBeforeDeliveryReclaimsCommittedRouteAndPreparedPermit()
            throws Exception {
        try (Fixture f = new Fixture()) {
            f.assertReserved();
            var prepared =
                    Delivery.Member.prepare(
                            f.item,
                            SchedulingTestConfig.capacity(SchedulingTestConfig.newConfig()));
            assertTrue(prepared.accepted());
            assertEquals(1, f.decode.resourceSnapshot().activeDispatchPermits());
            try (var member = prepared.value()) {
                var failure = new IllegalStateException("snapshot materialization failed");
                f.requests.recordPreparationFailure(
                        new ContextResult.PreparationFailure(
                                null, f.item, failure.getMessage(), System.currentTimeMillis()));
                f.requests.recordPreparationFailure(
                        new ContextResult.PreparationFailure(
                                null, f.item, failure.getMessage(), System.currentTimeMillis()));
                assertFalse(f.item.ctx().data().getFuture().get(2, TimeUnit.SECONDS).isSuccess());
            }
            f.assertEmpty();
            f.assertCapacityReusable();
        }
    }

    @ParameterizedTest
    @EnumSource(CancelReason.class)
    void admissionRetainsResourcesUntilItsOwnerCloses(CancelReason reason) throws Exception {
        try (Fixture f = new Fixture(true)) {
            f.assertReserved();
            if (reason == CancelReason.CLIENT_CANCELLED) {
                f.requests.cancel(String.valueOf(ID), 0, reason);
            } else {
                f.expire();
            }
            assertFalse(f.item.ctx().data().getFuture().isDone());
            f.assertReserved();
            AbstractRequestScheduler.finishRouting(f.admission);
            f.assertEmpty();
            f.assertCapacityReusable();
        }
    }

    @Test
    void cancellationDuringPreparationAlsoReclaimsTheAcquiredDispatchPermit() throws Exception {
        try (Fixture f = new Fixture()) {
            var member =
                    Delivery.Member.prepare(
                            f.item,
                            SchedulingTestConfig.capacity(SchedulingTestConfig.newConfig()));
            assertTrue(member.accepted());
            assertEquals(1, f.decode.resourceSnapshot().activeDispatchPermits());
            assertEquals(1, f.decode.routingView().engineCapacityUsed());
            f.requests.cancel(String.valueOf(ID), 0, CancelReason.CLIENT_CANCELLED);
            // The transaction's eventual rollback must remain harmless after request cleanup.
            assertNull(org.flexlb.util.Failures.close(member.value()));
            f.assertEmpty();
            f.assertCapacityReusable();
        }
    }

    @ParameterizedTest
    @EnumSource(
            value = Delivery.Result.Status.class,
            names = {"DELIVERED", "UNCERTAIN"})
    void responseOrAmbiguousTransportRetainsResourcesUntilRemoteCleanup(
            Delivery.Result.Status result) throws Exception {
        try (Fixture f = new Fixture()) {
            Delivery.Member claim = f.handoff();
            claim.route()
                    .ctx()
                    .data()
                    .getRequestScheduler()
                    .onDeliveryResult(
                            new ContextResult.TransportResult(
                                    claim,
                                    new Delivery.Result(
                                            result,
                                            result == Delivery.Result.Status.DELIVERED
                                                    ? null
                                                    : new IllegalStateException("reply lost")),
                                    System.currentTimeMillis()));
            if (result == Delivery.Result.Status.DELIVERED) {
                assertTrue(f.item.ctx().data().getFuture().get(2, TimeUnit.SECONDS).isSuccess());
            }
            f.assertHandedOff();
            f.requests.cancel(String.valueOf(ID), 0, CancelReason.CLIENT_CANCELLED);
            f.assertHandedOff();
            f.expire();
            f.assertHandedOff();
            f.proveCleanup();
            f.assertEmpty();
            f.assertCapacityReusable();
        }
    }

    @ParameterizedTest
    @EnumSource(
            value = LocalEnd.class,
            names = {"EXPIRE", "CANCEL"})
    void prefillFailureKeepsConfirmedDecodeTrackedUntilTerminalOrInactivity(LocalEnd ending)
            throws Exception {
        try (Fixture f = new Fixture()) {
            var acquisition = f.decode.acquireDispatchPermit(f.reservation, f.capacity);
            assertEquals(
                    DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED,
                    acquisition.status());
            try (var member = new Delivery.Member(f.item, acquisition.value())) {
                var claim =
                        org.flexlb.balance.scheduler.DeliveryTestSupport.transport(
                                        f.requests.dispatcher)
                                .commit(
                                        new org.flexlb.balance.scheduler.Delivery.Member(
                                                f.item, member.decode()),
                                        Delivery.SendMode.ROUTE_DECISION,
                                        0L,
                                        null);
                assertNotNull(claim);
                f.requests.onDeliveryResult(
                        new ContextResult.RouteDelivered(
                                claim,
                                org.flexlb.balance.prediction.PrefillTimePredictor
                                        .predictedVisibilityDeadlineMs(
                                                new org.flexlb.balance.projection
                                                        .RequestTimeEstimation.WorkSnapshot(
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
            } finally {

            }
            assertTrue(
                    f.item.ctx().data().getFuture().get(2, TimeUnit.SECONDS).isSuccess(),
                    "publish the scheduling response before injecting the later worker failure");
            f.decodeStatus(Map.of("101", task(ID)), Map.of(), TOTAL_KV - HARD_KV);
            TaskInfo failed = task(ID);
            failed.setErrorCode(500L);
            applyStatus(
                    f.prefill,
                    status(RoleType.PREFILL, 2L, Map.of(), Map.of("101", failed), TOTAL_KV));
            SchedulerTestSupport.awaitRequestWork(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(f.requests));
            assertEquals(ContextResult.Phase.ENDING, f.item.ctx().snapshot().phase());
            assertEquals(1, f.decode.resourceSnapshot().runningCount());
            assertEquals(1, SchedulerTestSupport.repository(f.requests).liveRequestCount());
            assertEquals(
                    0,
                    f.prefill.stats().locallyOwnedRequests(),
                    "the Prefill terminal reducer has already released its exact resource owner");
            org.mockito.Mockito.verify(f.prefill, org.mockito.Mockito.never())
                    .releaseRequest(f.item);
            if (ending == LocalEnd.EXPIRE) {
                f.expire();
            } else {
                f.decodeStatus(Map.of(), Map.of("101", task(ID)), TOTAL_KV);
            }
            f.assertEmpty();
            org.mockito.Mockito.verify(f.prefill, org.mockito.Mockito.never())
                    .releaseRequest(f.item);
            assertTrue(
                    f.item.ctx().data().getFuture().join().isSuccess(),
                    "resource completion preserves the published scheduling response");
        }
    }

    @Test
    void localRollbackCannotReleaseAnEngineOwnerWhoseProjectionHasNotArrived() throws Exception {
        try (Fixture f = new Fixture()) {
            var acquisition = f.decode.acquireDispatchPermit(f.reservation, f.capacity);
            assertEquals(
                    DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED,
                    acquisition.status());
            assertEquals(
                    DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED,
                    acquisition.value().dispatch().status());
            // Endpoint ownership can advance before its notification reaches the context.
            f.requests.cancel(String.valueOf(ID), 0, CancelReason.CLIENT_CANCELLED);
            SchedulerTestSupport.awaitRequestWork(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(f.requests));
            assertEquals(1, SchedulerTestSupport.repository(f.requests).liveRequestCount());
            assertEquals(ContextResult.Phase.ENDING, f.item.ctx().snapshot().phase());
            assertEquals(0, f.prefill.stats().locallyOwnedRequests());
            assertEquals(HARD_KV, f.decode.routingView().inflightHardKv());
            assertEquals(1, f.decode.routingView().engineCapacityUsed());
            f.decodeStatus(Map.of(), Map.of("101", task(ID)), TOTAL_KV);
            f.assertEmpty();
            f.assertCapacityReusable();
        }
    }

    @Test
    void definiteDispatchRejectionReclaimsTransferredCapacity() throws Exception {
        try (Fixture f = new Fixture()) {
            Delivery.Member claim = f.handoff();
            f.assertHandedOff();
            claim.route()
                    .ctx()
                    .data()
                    .getRequestScheduler()
                    .onDeliveryResult(
                            new ContextResult.TransportResult(
                                    claim,
                                    Delivery.Result.notSent(
                                            new IllegalStateException("not delivered")),
                                    System.currentTimeMillis()));
            assertFalse(f.item.ctx().data().getFuture().get(2, TimeUnit.SECONDS).isSuccess());
            f.assertEmpty();
            f.assertCapacityReusable();
        }
    }

    @Test
    void runningDecodeWinsAgainstUncertainTransportAndTerminalReclaimsBothEndpoints()
            throws Exception {
        try (Fixture f = new Fixture()) {
            Delivery.Member claim = f.handoff();
            TaskInfo running = task(ID);
            f.decodeStatus(Map.of("101", running), Map.of(), TOTAL_KV - HARD_KV);
            assertEquals(1, f.decode.resourceSnapshot().runningCount());
            assertEquals(1, f.decode.routingView().engineCapacityUsed());
            assertEquals(
                    0,
                    f.decode.routingView().inflightHardKv(),
                    "Engine status replaces the local KV prediction");
            claim.route()
                    .ctx()
                    .data()
                    .getRequestScheduler()
                    .onDeliveryResult(
                            new ContextResult.TransportResult(
                                    claim,
                                    Delivery.Result.uncertain(
                                            new IllegalStateException("late transport failure")),
                                    System.currentTimeMillis()));
            assertTrue(f.item.ctx().data().getFuture().get(2, TimeUnit.SECONDS).isSuccess());
            assertEquals(1, SchedulerTestSupport.repository(f.requests).liveRequestCount());
            assertEquals(1, f.decode.resourceSnapshot().runningCount());
            assertEquals(1, f.prefill.stats().locallyOwnedRequests());
            applyStatus(
                    f.prefill,
                    status(RoleType.PREFILL, 2L, Map.of(), Map.of("101", running), TOTAL_KV));
            f.decodeStatus(Map.of(), Map.of("101", running), TOTAL_KV);
            f.assertEmpty();
            f.assertCapacityReusable();
        }
    }

    @ParameterizedTest
    @CsvSource({
        "RUNNING, RECEIVED",
        "RUNNING, PENDING",
        "KV_ALLOCATED, RECEIVED",
        "KV_ALLOCATED, PENDING"
    })
    void decodePhaseRegressionStillDeliversTerminalWithoutInactivity(
            TaskPhase confirmed, TaskPhase regressed) throws Exception {
        try (Fixture f = new Fixture()) {
            var claim = f.handoff();
            f.requests.onDeliveryResult(
                    new ContextResult.TransportResult(
                            claim, Delivery.Result.delivered(), System.currentTimeMillis()));
            assertTrue(f.item.ctx().data().getFuture().get(2, TimeUnit.SECONDS).isSuccess());
            applyStatus(
                    f.prefill,
                    status(RoleType.PREFILL, 2L, Map.of(), Map.of("101", task(ID)), TOTAL_KV));
            TaskInfo active = task(ID);
            active.setPhase(confirmed);
            f.decodeStatus(Map.of("101", active), Map.of(), TOTAL_KV - HARD_KV);
            TaskInfo stillPresent = task(ID);
            stillPresent.setPhase(regressed);
            f.decodeStatus(Map.of("101", stillPresent), Map.of(), TOTAL_KV);
            assertEquals(
                    RequestState.Phase.ACKNOWLEDGED,
                    SchedulerTestSupport.repository(f.requests)
                            .getRequestState(String.valueOf(ID), 0)
                            .state());
            // No inactivity expiry, sweep, sleep or shutdown is allowed to make this assertion
            // pass.
            f.decodeStatus(Map.of(), Map.of("101", task(ID)), TOTAL_KV);
            f.assertEmpty();
            org.mockito.Mockito.verify(f.prefill, org.mockito.Mockito.never())
                    .releaseRequest(f.item);
            f.assertCapacityReusable();
        }
    }

    @Test
    void prefillCompletionAfterDecodeAcceptanceRemainsSettledAtDecodeTerminal() throws Exception {
        try (Fixture f = new Fixture()) {
            var claim = f.handoff();
            f.requests.onDeliveryResult(
                    new ContextResult.TransportResult(
                            claim, Delivery.Result.delivered(), System.currentTimeMillis()));
            assertTrue(f.item.ctx().data().getFuture().get(2, TimeUnit.SECONDS).isSuccess());
            f.decodeStatus(Map.of("101", task(ID)), Map.of(), TOTAL_KV - HARD_KV);
            assertTrue(f.item.ctx().snapshot().engine().decodeAccepted());
            assertTrue(f.item.ctx().snapshot().deadlines().visibility() == null);
            applyStatus(
                    f.prefill,
                    status(RoleType.PREFILL, 2L, Map.of(), Map.of("101", task(ID)), TOTAL_KV));
            SchedulerTestSupport.awaitRequestWork(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(f.requests));
            assertTrue(
                    f.item.ctx().snapshot().deadlines().visibility() == null,
                    "Prefill completion cannot rearm visibility checks after Decode acceptance");
            assertEquals(0, f.prefill.stats().locallyOwnedRequests());
            assertEquals(1, f.decode.resourceSnapshot().runningCount());
            f.decodeStatus(Map.of(), Map.of("101", task(ID)), TOTAL_KV);
            f.assertEmpty();
            org.mockito.Mockito.verify(f.prefill, org.mockito.Mockito.never())
                    .releaseRequest(f.item);
        }
    }

    @Test
    void inactivityWaitsForRemoteCleanupWithoutInventingPhysicalKvRelease() throws Exception {
        try (Fixture f = new Fixture()) {
            var claim = f.handoff();
            f.requests.onDeliveryResult(
                    new ContextResult.TransportResult(
                            claim, Delivery.Result.delivered(), System.currentTimeMillis()));
            f.decodeStatus(Map.of("101", task(ID)), Map.of(), TOTAL_KV - HARD_KV);
            assertEquals(1, f.decode.resourceSnapshot().runningCount());
            f.expire();
            assertEquals(1, f.decode.resourceSnapshot().runningCount());
            f.proveCleanup();
            f.assertEmpty();
            assertEquals(
                    TOTAL_KV - HARD_KV,
                    f.decode.routingView().realKvAvailable(),
                    "local expiry must not rewrite the last physical Engine KV sample");
            f.assertCapacityReusable();
        }
    }

    @Test
    void lateCallbacksAndOldReservationCannotReleaseReplacementCapacity() throws Exception {
        try (Fixture f = new Fixture()) {
            Delivery.Member oldClaim = f.handoff();
            var oldReservation = f.reservation;
            f.expire();
            f.assertHandedOff();
            oldClaim.route()
                    .ctx()
                    .data()
                    .getRequestScheduler()
                    .onDeliveryResult(
                            new ContextResult.TransportResult(
                                    oldClaim,
                                    Delivery.Result.delivered(),
                                    System.currentTimeMillis()));
            f.proveCleanup();
            f.assertEmpty();
            var replacementRef = new AtomicReference<DecodeResources.RequestReservation>();
            RequestProtocolTestSupport.awaitCondition(
                    () -> {
                        if (replacementRef.get() != null) {
                            return true;
                        }
                        f.decode.evictExpiredRequests(0, ignored -> false);
                        {
                            var selectedEndpoint = f.decode;

                            replacementRef.set(
                                    f.decode
                                            .tryRegisterSelection(
                                                    f.decode.routingView(),
                                                    String.valueOf(ID),
                                                    HARD_KV * 2,
                                                    EXPECTED_KV * 2,
                                                    50,
                                                    f.capacity)
                                            .value());
                        }
                        return replacementRef.get() != null;
                    });
            var replacement = replacementRef.get();
            assertNotEquals(oldReservation.reservationToken(), replacement.reservationToken());
            var archived = oldClaim.owner().snapshot();
            var stale =
                    org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                            oldClaim.owner(),
                            observedRequest ->
                                    observedRequest.recordDeliveryResult(
                                            new ContextResult.TransportResult(
                                                    oldClaim,
                                                    Delivery.Result.delivered(),
                                                    System.currentTimeMillis())));
            assertEquals(RequestChangeAssertions.UpdateKind.STALE, stale.kind());
            assertSame(archived, stale.before());
            assertSame(archived, stale.after());
            f.expire();
            assertNotEquals(
                    ReservationReleaseResult.RELEASED,
                    f.decode
                            .release(oldReservation, DecodeResources.ReleaseReason.EXPIRED)
                            .status());
            assertNotEquals(
                    ReservationReleaseResult.RELEASED,
                    f.decode
                            .release(
                                    oldReservation,
                                    DecodeResources.ReleaseReason.COUNTERPART_FINISHED)
                            .status());
            assertEquals(HARD_KV * 2, f.decode.routingView().inflightHardKv());
            assertEquals(
                    EXPECTED_KV * 2,
                    EndpointTestSupport.expectedReservedKv(f.decode.resourceSnapshot()));
            assertEquals(replacement, EndpointTestSupport.decodeReservation(f.decode, ID));
            f.decode.release(replacement, DecodeResources.ReleaseReason.LOCAL_ROLLBACK).status();
            f.assertEmpty();
        }
    }

    private static TaskInfo task(long id) {
        var task = new TaskInfo();
        task.setRequestId(String.valueOf(id));
        task.setInputLength(HARD_KV);
        task.setPhase(TaskPhase.RUNNING);
        return task;
    }

    private static WorkerStatusResponse status(
            RoleType role,
            long version,
            Map<String, TaskInfo> running,
            Map<String, TaskInfo> finished,
            long freeKv) {
        var response = new WorkerStatusResponse();
        response.setRole(role);
        response.setAlive(true);
        response.setStatusVersion(version);
        response.setLatestFinishedVersion(finished.isEmpty() ? 0L : version);
        response.setRunningTaskInfo(running);
        response.setRunningQueryLen((long) running.size());
        response.setFinishedTaskInfo(finished);
        response.setTotalKvCacheTokens(TOTAL_KV);
        response.setAvailableKvCacheTokens(freeKv);
        return response;
    }

    private static void applyStatus(WorkerEndpoint endpoint, WorkerStatusResponse response) {
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

    private enum LocalEnd {
        CANCEL,
        FUTURE_CANCEL,
        EXPIRE,
        SHUTDOWN
    }

    /**
     * Assembles exact P/D registrations and handoff permits to compare accounting before and after lifecycle events.
     */
    private static final class Fixture implements AutoCloseable {
        final AbstractRequestScheduler requests;
        final SchedulerTestServices runtime;
        final PrefillEndpoint prefill;
        final DecodeEndpoint decode;
        final RouteResult item;
        final DecodeResources.RequestReservation reservation;
        final PrefillEndpoint.RequestRecord routeReservation;
        final WorkSnapshot prefillHandoff;
        final RoutingContext admission;
        final DecodeResources.AdmissionCapacity capacity =
                new DecodeResources.AdmissionCapacity(1, 90);

        long version = 1;

        final java.util.concurrent.CompletableFuture<EngineGrpcClient.CancelReply> cleanup =
                new java.util.concurrent.CompletableFuture<>();

        Fixture() {
            this(false);
        }

        Fixture(boolean keepAdmissionOpen) {
            var config = SchedulingTestConfig.newConfig();
            config.setScheduler(ConfigTestSupport.direct());
            SchedulingTestConfig.useNonBatchDispatcher(config).setMaxInflightPerPrefillWorker(1);
            config.getRequestLifecycle().getRequest().setTimeoutMs(TTL);
            ConfigService service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            var reporter = mock(DeliveryMetricsReporter.class);
            var requestReporter = mock(RequestSchedulerReporter.class);
            requests =
                    SchedulerTestSupport.create(
                            service, reporter, requestReporter, mock(TheoryCacheHitReporter.class));
            var channel = mock(EngineGrpcClient.class);
            when(channel.cancelAsync(any(), anyInt(), any(), anyLong())).thenReturn(cleanup);
            org.flexlb.balance.scheduler.DeliveryTestSupport.setField(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(requests).dispatcher,
                    "grpcClient",
                    channel);
            var projector = requests;
            var endpoints =
                    new EndpointRegistry(
                            service,
                            SchedulerTestSupport.repository(projector),
                            reporter,
                            DeliveryTestSupport.route(reporter),
                            new CapacityNotifier(),
                            org.flexlb.balance.planner.BuiltinGroupPlanner.from(
                                    service.loadBalanceConfig()),
                            () -> org.flexlb.balance.endpoint.OrderedPrefillRequestQueue.fifo(16));
            runtime = org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(requests);
            org.flexlb.balance.scheduler.SchedulerTestServices.setDependency(
                    runtime, "endpoints", endpoints);
            var worker =
                    WorkerStatus.createDiscovered(
                            RoleType.PREFILL, "g", "127.0.0.1", 8080, 8081, "test");
            worker.lock.lock();
            try {
                prefill =
                        org.mockito.Mockito.spy(
                                (PrefillEndpoint)
                                        endpoints.publishPreparedEndpoint(
                                                worker.getLogicalIpPort(),
                                                worker,
                                                worker.prepareNewStatus(
                                                        WorkerStatusTestSupport
                                                                .freezeStatusResponse(
                                                                        worker,
                                                                        status(
                                                                                RoleType.PREFILL,
                                                                                1,
                                                                                Map.of(),
                                                                                Map.of(),
                                                                                TOTAL_KV)))));
            } finally {
                worker.lock.unlock();
            }
            decode =
                    EndpointTestSupport.decode(
                            WorkerStatus.createDiscovered(
                                    RoleType.DECODE, "g", "127.0.0.2", 8080, 8081, "test"),
                            SchedulerTestSupport.repository(projector));
            decodeStatus(Map.of(), Map.of(), TOTAL_KV);
            {
                var selectedEndpoint = decode;

                reservation =
                        decode.tryRegisterSelection(
                                        decode.routingView(),
                                        String.valueOf(ID),
                                        HARD_KV,
                                        EXPECTED_KV,
                                        50,
                                        capacity)
                                .value();
            }
            assertNotNull(reservation);
            var context = RequestProtocolTestSupport.context(config, ID);
            var future = RequestProtocolTestSupport.submitForManualPlacement(requests, context);
            context.data().setFuture(future);
            item =
                    RequestExecutionTestSupport.registerPrefillRoute(
                            context,
                            prefill,
                            decode,
                            reservation,
                            30_000L,
                            System.currentTimeMillis());
            var held = new AtomicReference<PrefillEndpoint.RequestRecord>();
            admission = SchedulerTestSupport.claim(requests, String.valueOf(ID), future);
            assertNotNull(admission);
            {
                var selectedEndpoint = prefill;

                assertTrue(
                        (RequestProtocolTestSupport.commitRoute(
                                        requests,
                                        item,
                                        RequestProtocolTestSupport.publication(
                                                () -> {
                                                    held.set(item.prefillReservation());
                                                    return true;
                                                }))
                                == org.flexlb.balance.RoutingResult.Status.SUCCESS));
            }
            routeReservation = held.get();
            prefillHandoff =
                    org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(prefill, item);
            assertNotNull(prefillHandoff);
            if (!keepAdmissionOpen) {
                AbstractRequestScheduler.finishRouting(admission);
            }
        }

        @Override
        public void close() {
            try {
                AbstractRequestScheduler.finishRouting(admission);

                prefill.rollback(routeReservation);
                SchedulerTestSupport.shutdown(runtime);
            } finally {
                decode.close();
            }
        }

        Delivery.Member handoff() {
            var acquisition = decode.acquireDispatchPermit(reservation, capacity);
            assertEquals(
                    DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED,
                    acquisition.status());
            var member = new Delivery.Member(item, acquisition.value());
            try {
                var claim =
                        org.flexlb.balance.scheduler.DeliveryTestSupport.transport(
                                        requests.dispatcher)
                                .commit(
                                        new org.flexlb.balance.scheduler.Delivery.Member(
                                                item, member.decode()),
                                        Delivery.SendMode.BATCH_ENQUEUE,
                                        1L,
                                        null);
                assertNotNull(claim);
                assertEquals(
                        ContextResult.SendState.SENDING,
                        claim.owner().snapshot().delivery().send());
                return claim;
            } finally {
                member.close();
            }
        }

        void assertHandedOff() {
            assertReserved();
            assertEquals(0, decode.resourceSnapshot().activeDispatchPermits());
            assertEquals(1, decode.routingView().engineCapacityUsed());
            assertFalse(
                    capacity.evaluate(
                                    decode.routingView().dispatchUsage(),
                                    HARD_KV,
                                    EXPECTED_KV,
                                    CapacityRelease.NONE)
                            .fits());
        }

        void assertReserved() {
            assertEquals(1, SchedulerTestSupport.repository(requests).liveRequestCount());
            assertEquals(1, prefill.stats().locallyOwnedRequests());
            assertEquals(HARD_KV, decode.routingView().inflightHardKv());
            assertEquals(
                    EXPECTED_KV, EndpointTestSupport.expectedReservedKv(decode.resourceSnapshot()));
        }

        void assertCapacityReusable() {
            {
                var selectedEndpoint = decode;

                var next =
                        decode.tryRegisterSelection(
                                        decode.routingView(),
                                        String.valueOf(999L),
                                        HARD_KV,
                                        EXPECTED_KV,
                                        50,
                                        capacity)
                                .value();
                assertNotNull(next);
                var permit = decode.acquireDispatchPermit(next, capacity);
                assertEquals(
                        DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED,
                        permit.status());
                assertTrue(permit.value().release());
                decode.release(next, DecodeResources.ReleaseReason.LOCAL_ROLLBACK).status();
            }
            // A second exact request must be able to use the sole Prefill slot too.
            var nextContext =
                    RequestProtocolTestSupport.context(item.ctx().data().getConfig(), 999L);
            nextContext.data().setFuture(new CompletableFuture<>());
            var next =
                    RequestExecutionTestSupport.registerPrefillRoute(
                            nextContext, prefill, null, null, 1L, System.currentTimeMillis());
            {
                var selectedEndpoint = prefill;

                prefill.rollback(next.prefillReservation());
            }
            assertEmpty();
        }

        void assertEmpty() {
            SchedulerTestSupport.awaitRequestWork(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(requests));
            assertEquals(
                    0,
                    SchedulerTestSupport.repository(requests).liveRequestCount(),
                    "request inflight");
            assertEquals(0, prefill.queueDepth(), "Prefill queue membership");
            assertEquals(0, prefill.stats().locallyOwnedRequests(), "Prefill inflight");
            assertEquals(
                    0, prefill.stats().individuallyOwnedRequests(), "Prefill individual lease");
            assertEquals(0, prefill.stats().batchCount(), "Prefill batch occupancy");
            var view = decode.resourceSnapshot();
            assertEquals(0, view.routing().inflightHardKv(), "Decode hard KV reservation");
            assertEquals(
                    0,
                    EndpointTestSupport.expectedReservedKv(view),
                    "Decode expected KV reservation");
            assertEquals(0, view.activeDispatchPermits(), "Decode dispatch permits");
            assertEquals(0, view.queuedCount(), "Decode queued ownership");
            assertEquals(0, view.acceptedCount(), "Decode accepted streams");
            assertEquals(0, view.runningCount(), "Decode running streams");
            assertEquals(0, view.engineCapacityUsed(), "Decode request capacity");
            assertTrue(view.reservedCount() == 0);
            assertTrue(view.confirmedCount() == 0);
        }

        void proveCleanup() throws Exception {
            cleanup.complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
            item.ctx()
                    .snapshot()
                    .delivery()
                    .identity()
                    .completion()
                    .toCompletableFuture()
                    .get(2, TimeUnit.SECONDS);
            SchedulerTestSupport.awaitRequestWork(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(requests));
        }

        void expire() {
            RequestProtocolTestSupport.expireInactiveRequest(
                    requests,
                    requests.repository.findActive(String.valueOf(ID)),
                    System.currentTimeMillis() + TTL + 1);
        }

        void decodeStatus(
                Map<String, TaskInfo> running, Map<String, TaskInfo> finished, long freeKv) {
            applyStatus(decode, status(RoleType.DECODE, version++, running, finished, freeKv));
        }
    }
}
