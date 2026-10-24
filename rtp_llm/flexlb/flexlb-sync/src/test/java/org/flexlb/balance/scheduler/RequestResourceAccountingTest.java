package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources.CapacityRelease;
import org.flexlb.balance.endpoint.DecodeResources.ReservationReleaseResult;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.EndpointTestSupport;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.endpoint.WorkerEndpoint;
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
import org.flexlb.service.RecentCacheKeyTraceReporter;
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

/** Resource assertions use real endpoint ledgers, not invocations of mocked release methods. */
class RequestResourceAccountingTest {
    private static final long ID = 101L;
    private static final long HARD_KV = 160L;
    private static final long EXPECTED_KV = 320L;
    private static final long TOTAL_KV = 10_000L;
    private static final long TTL = TimeUnit.HOURS.toMillis(1);

    @Test
    void schedulerCannotConsumeAnotherEndpointsRealDispatchPermit() throws Exception {
        try (Fixture owner = new Fixture(); Fixture other = new Fixture()) {
            var permit = other.decode.acquireDispatchPermit(other.reservation, other.capacity).handle();
            assertNotNull(permit);
            try {
                assertThrows(IllegalArgumentException.class, () -> owner.requests.claimDelivery(
                        owner.item, DeliveryClaimKind.BATCH_ENQUEUE, 1L, permit));
                assertNull(owner.item.ctx().snapshot().delivery());
                assertEquals(ContextResult.Phase.OPEN, owner.item.ctx().snapshot().phase());
                assertNotNull(owner.item.ctx().snapshot().route());
                assertNull(owner.item.ctx().snapshot().delivery());
                assertEquals(1, other.decode.resourceSnapshot().activeDispatchPermits());
                assertEquals(1, other.decode.resourceSnapshot().queuedCount());
                assertFalse(owner.item.future().isDone());
            } finally { permit.release(); }
        }
    }

    @Test
    void partialCleanupFailureStillAttemptsOtherResourcesAndPreservesDecodeSettlement() {
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.batchConfig(), ID);
        var prefill = mock(PrefillEndpoint.class);
        var decode = RequestProtocolTestSupport.decodeEndpoint();
        var reservation = new DecodeResources.ReservationHandle(1L, ID, 1L);
        var item = SchedulingTestConfig.createRoute(context, new Response(), null, null,
                prefill, decode, reservation, System.currentTimeMillis());
        when(decode.release(reservation, DecodeResources.ReleaseReason.NOT_SENT))
                .thenReturn(org.flexlb.balance.endpoint.DecodeResources.ResourceChange.of(DecodeResources.ReservationReleaseResult.RELEASED, false));
        var releaseFailure = new IllegalStateException("Prefill release failed");
        when(prefill.release(null, item)).thenThrow(releaseFailure);
        var first = AbstractRequestScheduler.releaseResources(new ContextResult.ClosePermit(context, 1L, item, DecodeResources.ReleaseReason.NOT_SENT, false, false, false, false));
        assertFalse(first.prefillSettled());
        assertTrue(first.decodeSettled());
        assertSame(releaseFailure, first.failure());
        org.mockito.Mockito.verify(decode).release(reservation, DecodeResources.ReleaseReason.NOT_SENT);

        org.mockito.Mockito.doReturn(true).when(prefill).release(null, item);
        var retry = AbstractRequestScheduler.releaseResources(new ContextResult.ClosePermit(context, 1L, item, DecodeResources.ReleaseReason.NOT_SENT,
                first.prefillSettled(), first.decodeSettled(), false, false));
        assertTrue(retry.prefillSettled());
        assertTrue(retry.decodeSettled());
        assertNull(retry.failure());
        org.mockito.Mockito.verify(decode, org.mockito.Mockito.times(1))
                .release(reservation, DecodeResources.ReleaseReason.NOT_SENT);
    }

    @ParameterizedTest
    @EnumSource(LocalEnd.class)
    void localTerminationReclaimsReservationAndPrefillInflight(LocalEnd end) throws Exception {
        try (Fixture f = new Fixture()) {
            f.assertReserved();
            switch (end) {
                case CANCEL -> f.requests.cancel(ID, 0, CancelReason.CLIENT_CANCELLED);
                case FUTURE_CANCEL -> assertTrue(f.item.future().cancel(false));
                case EXPIRE -> f.expire();
                case SHUTDOWN -> {
                    assertTrue(RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(f.requests));
                    f.requests.closeOutstandingAndTerminalize();
                }
            }
            f.assertEmpty();
            f.assertCapacityReusable();
        }
    }

    @Test
    void failedMaterializationBeforeDeliveryReclaimsCommittedRouteAndPreparedPermit() throws Exception {
        try (Fixture f = new Fixture()) {
            f.assertReserved();
            var prepared = DeliveryDispatcher.Member.prepare(f.item, SchedulingTestConfig.capacity(SchedulingTestConfig.newConfig()));
            assertTrue(prepared.accepted());
            assertEquals(1, f.decode.resourceSnapshot().activeDispatchPermits());
            try (var member = prepared.value()) {
                var failure = new IllegalStateException("snapshot materialization failed");
                f.requests.onDeliveryEvidence(null, new ContextResult.PreparationFailure(null, f.item, failure.getMessage(), System.currentTimeMillis()));
                f.requests.onDeliveryEvidence(null, new ContextResult.PreparationFailure(null, f.item, failure.getMessage(), System.currentTimeMillis()));
                assertFalse(f.item.future().get(2, TimeUnit.SECONDS).isSuccess());
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
                f.requests.cancel(ID, 0, reason);
            } else {
                f.expire();
            }
            assertFalse(f.item.future().isDone());
            f.assertReserved();
            f.admission.finish();
            f.assertEmpty();
            f.assertCapacityReusable();
        }
    }

    @Test
    void cancellationDuringPreparationAlsoReclaimsTheAcquiredDispatchPermit() throws Exception {
        try (Fixture f = new Fixture()) {
            var member = DeliveryDispatcher.Member.prepare(f.item, SchedulingTestConfig.capacity(SchedulingTestConfig.newConfig()));
            assertTrue(member.accepted());
            assertEquals(1, f.decode.resourceSnapshot().activeDispatchPermits());
            assertEquals(1, f.decode.routingView().engineCapacityUsed());
            f.requests.cancel(ID, 0, CancelReason.CLIENT_CANCELLED);
            // The transaction's eventual rollback must remain harmless after request cleanup.
            assertNull(org.flexlb.util.Failures.close(member.value()));
            f.assertEmpty();
            f.assertCapacityReusable();
        }
    }

    @ParameterizedTest
    @EnumSource(value = DeliveryResult.Status.class, names = {"DELIVERED", "UNCERTAIN"})
    void responseOrAmbiguousTransportRetainsResourcesUntilRemoteCleanup(DeliveryResult.Status result) throws Exception {
        try (Fixture f = new Fixture()) {
            DeliveryClaim claim = f.handoff();
            claim.item().ctx().data().getScheduler().onDeliveryEvidence(claim, new ContextResult.TransportResult(claim, new DeliveryResult(result, result == DeliveryResult.Status.DELIVERED ? null : new IllegalStateException("reply lost")), System.currentTimeMillis()));
            if (result == DeliveryResult.Status.DELIVERED) {
                assertTrue(f.item.future().get(2, TimeUnit.SECONDS).isSuccess());
            }
            f.assertHandedOff();
            f.requests.cancel(ID, 0, CancelReason.CLIENT_CANCELLED);
            f.assertHandedOff();
            f.expire();
            f.assertHandedOff();
            f.proveCleanup();
            f.assertEmpty();
            f.assertCapacityReusable();
        }
    }

    @ParameterizedTest
    @EnumSource(value = LocalEnd.class, names = {"EXPIRE", "CANCEL"})
    void prefillFailureKeepsConfirmedDecodeTrackedUntilTerminalOrInactivity(LocalEnd ending) throws Exception {
        try (Fixture f = new Fixture()) {
            var acquisition = f.decode.acquireDispatchPermit(f.reservation, f.capacity);
            assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED, acquisition.status());
            try (var member = new DeliveryDispatcher.Member(f.item, acquisition.handle())) {
                var claim = f.requests.claimDelivery(f.item, DeliveryClaimKind.ROUTE_DECISION, 0L, member.decode());
                assertNotNull(claim);
                f.requests.onDeliveryEvidence(claim, new ContextResult.RouteDelivered(claim, org.flexlb.balance.prediction.PrefillPredictionBoundary.predictedVisibilityDeadlineMs(new org.flexlb.balance.projection.WorkSnapshot(
                        System.currentTimeMillis(), List.of(), List.of(), 0L), 30_000L, claim.owner().data().getConfig().getRequestLifecycle().getDecision().getLifetime(), System.currentTimeMillis()), System.currentTimeMillis()));
            } finally { f.prefillHandoff.close(); }
            f.decodeStatus(Map.of("101", task(ID)), Map.of(), TOTAL_KV - HARD_KV);
            TaskInfo failed = task(ID);
            failed.setErrorCode(500L);
            applyStatus(f.prefill, status(RoleType.PREFILL, 2L, Map.of(), Map.of("101", failed), TOTAL_KV));
            SchedulerTestSupport.awaitRequestWork(f.requests.runtime);
            assertEquals(ContextResult.Phase.ENDING, f.item.ctx().snapshot().phase());
            assertEquals(1, f.decode.resourceSnapshot().runningCount());
            assertEquals(1, SchedulerTestSupport.repository(f.requests).liveRequestCount());
            assertEquals(0, f.prefill.ownership.stats().locallyOwnedRequests(),
                    "the Prefill terminal reducer has already released its exact resource owner");
            org.mockito.Mockito.verify(f.prefill, org.mockito.Mockito.never()).release(null, f.item);
            if (ending == LocalEnd.EXPIRE) { f.expire(); } else { f.decodeStatus(Map.of(), Map.of("101", task(ID)), TOTAL_KV); }
            f.assertEmpty();
            org.mockito.Mockito.verify(f.prefill, org.mockito.Mockito.never()).release(null, f.item);
            assertTrue(f.item.future().join().isSuccess(), "resource completion preserves the published scheduling response");
        }
    }

    @Test
    void localRollbackCannotReleaseAnEngineOwnerWhoseProjectionHasNotArrived() throws Exception {
        try (Fixture f = new Fixture()) {
            var acquisition = f.decode.acquireDispatchPermit(f.reservation, f.capacity);
            assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED, acquisition.status());
            assertEquals(DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED,
                    acquisition.handle().dispatch().status());
            // Endpoint ownership can advance before its notification reaches the context.
            f.requests.cancel(ID, 0, CancelReason.CLIENT_CANCELLED);
            SchedulerTestSupport.awaitRequestWork(f.requests.runtime);
            assertEquals(1, SchedulerTestSupport.repository(f.requests).liveRequestCount());
            assertEquals(ContextResult.Phase.ENDING, f.item.ctx().snapshot().phase());
            assertEquals(0, f.prefill.ownership.stats().locallyOwnedRequests());
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
            DeliveryClaim claim = f.handoff();
            f.assertHandedOff();
            claim.item().ctx().data().getScheduler().onDeliveryEvidence(claim, new ContextResult.TransportResult(claim, DeliveryResult.notSent(new IllegalStateException("not delivered")), System.currentTimeMillis()));
            assertFalse(f.item.future().get(2, TimeUnit.SECONDS).isSuccess());
            f.assertEmpty();
            f.assertCapacityReusable();
        }
    }

    @Test
    void runningDecodeWinsAgainstUncertainTransportAndTerminalReclaimsBothEndpoints() throws Exception {
        try (Fixture f = new Fixture()) {
            DeliveryClaim claim = f.handoff();
            TaskInfo running = task(ID);
            f.decodeStatus(Map.of("101", running), Map.of(), TOTAL_KV - HARD_KV);
            assertEquals(1, f.decode.resourceSnapshot().runningCount());
            assertEquals(1, f.decode.routingView().engineCapacityUsed());
            assertEquals(0, f.decode.routingView().inflightHardKv(), "Engine status replaces the local KV prediction");
            claim.item().ctx().data().getScheduler().onDeliveryEvidence(claim, new ContextResult.TransportResult(claim, DeliveryResult.uncertain(new IllegalStateException("late transport failure")), System.currentTimeMillis()));
            assertTrue(f.item.future().get(2, TimeUnit.SECONDS).isSuccess());
            assertEquals(1, SchedulerTestSupport.repository(f.requests).liveRequestCount());
            assertEquals(1, f.decode.resourceSnapshot().runningCount());
            assertEquals(1, f.prefill.ownership.stats().locallyOwnedRequests());
            applyStatus(f.prefill, status(RoleType.PREFILL, 2L, Map.of(), Map.of("101", running), TOTAL_KV));
            f.decodeStatus(Map.of(), Map.of("101", running), TOTAL_KV);
            f.assertEmpty();
            f.assertCapacityReusable();
        }
    }

    @ParameterizedTest
    @CsvSource({"RUNNING, RECEIVED", "RUNNING, PENDING", "KV_ALLOCATED, RECEIVED", "KV_ALLOCATED, PENDING"})
    void decodePhaseRegressionStillDeliversTerminalWithoutInactivity(TaskPhase confirmed, TaskPhase regressed)
            throws Exception {
        try (Fixture f = new Fixture()) {
            var claim = f.handoff();
            f.requests.onDeliveryEvidence(claim, new ContextResult.TransportResult(claim, DeliveryResult.delivered(), System.currentTimeMillis()));
            assertTrue(f.item.future().get(2, TimeUnit.SECONDS).isSuccess());
            applyStatus(f.prefill, status(RoleType.PREFILL, 2L, Map.of(), Map.of("101", task(ID)), TOTAL_KV));
            TaskInfo active = task(ID);
            active.setPhase(confirmed);
            f.decodeStatus(Map.of("101", active), Map.of(), TOTAL_KV - HARD_KV);
            TaskInfo stillPresent = task(ID);
            stillPresent.setPhase(regressed);
            f.decodeStatus(Map.of("101", stillPresent), Map.of(), TOTAL_KV);
            assertEquals(RequestState.Phase.ACKNOWLEDGED, SchedulerTestSupport.repository(f.requests).getRequestState(ID, 0).state());
            // No inactivity expiry, sweep, sleep or shutdown is allowed to make this assertion pass.
            f.decodeStatus(Map.of(), Map.of("101", task(ID)), TOTAL_KV);
            f.assertEmpty();
            org.mockito.Mockito.verify(f.prefill, org.mockito.Mockito.never()).release(null, f.item);
            f.assertCapacityReusable();
        }
    }

    @Test
    void prefillCompletionAfterDecodeAcceptanceRemainsSettledAtDecodeTerminal() throws Exception {
        try (Fixture f = new Fixture()) {
            var claim = f.handoff();
            f.requests.onDeliveryEvidence(claim, new ContextResult.TransportResult(claim, DeliveryResult.delivered(), System.currentTimeMillis()));
            assertTrue(f.item.future().get(2, TimeUnit.SECONDS).isSuccess());
            f.decodeStatus(Map.of("101", task(ID)), Map.of(), TOTAL_KV - HARD_KV);
            assertTrue(f.item.ctx().snapshot().engine().decodeAccepted());
            assertTrue(f.item.ctx().snapshot().deadlines().visibility() == null);
            applyStatus(f.prefill, status(RoleType.PREFILL, 2L, Map.of(), Map.of("101", task(ID)), TOTAL_KV));
            SchedulerTestSupport.awaitRequestWork(f.requests.runtime);
            assertTrue(f.item.ctx().snapshot().deadlines().visibility() == null,
                    "Prefill completion cannot rearm visibility checks after Decode acceptance");
            assertEquals(0, f.prefill.ownership.stats().locallyOwnedRequests());
            assertEquals(1, f.decode.resourceSnapshot().runningCount());
            f.decodeStatus(Map.of(), Map.of("101", task(ID)), TOTAL_KV);
            f.assertEmpty();
            org.mockito.Mockito.verify(f.prefill, org.mockito.Mockito.never()).release(null, f.item);
        }
    }

    @Test
    void inactivityWaitsForRemoteCleanupWithoutInventingPhysicalKvRelease() throws Exception {
        try (Fixture f = new Fixture()) {
            var claim = f.handoff();
            f.requests.onDeliveryEvidence(claim, new ContextResult.TransportResult(claim, DeliveryResult.delivered(), System.currentTimeMillis()));
            f.decodeStatus(Map.of("101", task(ID)), Map.of(), TOTAL_KV - HARD_KV);
            assertEquals(1, f.decode.resourceSnapshot().runningCount());
            f.expire();
            assertEquals(1, f.decode.resourceSnapshot().runningCount());
            f.proveCleanup();
            f.assertEmpty();
            assertEquals(TOTAL_KV - HARD_KV, f.decode.routingView().realKvAvailable(),
                    "local expiry must not rewrite the last physical Engine KV sample");
            f.assertCapacityReusable();
        }
    }

    @Test
    void lateCallbacksAndOldReservationCannotReleaseReplacementCapacity() throws Exception {
        try (Fixture f = new Fixture()) {
            DeliveryClaim oldClaim = f.handoff();
            var oldReservation = f.reservation;
            f.expire();
            f.assertHandedOff();
            oldClaim.item().ctx().data().getScheduler().onDeliveryEvidence(oldClaim, new ContextResult.TransportResult(oldClaim, DeliveryResult.delivered(), System.currentTimeMillis()));
            f.proveCleanup();
            f.assertEmpty();
            var replacementRef = new AtomicReference<DecodeResources.ReservationHandle>();
            RequestProtocolTestSupport.awaitCondition(() -> {
                if (replacementRef.get() != null) { return true; }
                f.decode.evictExpiredRequests(0, ignored -> false);
                try (var pin = f.decode.tryPinGeneration()) {
                    replacementRef.set(f.decode.tryReserveQueuedRequest(pin, ID, HARD_KV * 2, EXPECTED_KV * 2, 50, null).handle());
                }
                return replacementRef.get() != null;
            });
            var replacement = replacementRef.get();
            assertNotEquals(oldReservation.reservationToken(), replacement.reservationToken());
            var archived = oldClaim.owner().snapshot();
            var stale = oldClaim.owner().recordDeliveryResult(new ContextResult.TransportResult(oldClaim, DeliveryResult.delivered(), System.currentTimeMillis()));
            assertEquals(ContextResult.UpdateKind.STALE, stale.kind());
            assertSame(archived, stale.before());
            assertSame(archived, stale.after());
            f.expire();
            assertNotEquals(ReservationReleaseResult.RELEASED, f.decode.release(oldReservation, DecodeResources.ReleaseReason.EXPIRED).status());
            assertNotEquals(ReservationReleaseResult.RELEASED, f.decode.release(oldReservation, DecodeResources.ReleaseReason.COUNTERPART_FINISHED).status());
            assertEquals(HARD_KV * 2, f.decode.routingView().inflightHardKv());
            assertEquals(EXPECTED_KV * 2, EndpointTestSupport.expectedReservedKv(f.decode.resourceSnapshot()));
            assertEquals(replacement, EndpointTestSupport.decodeReservation(f.decode, ID));
            f.decode.release(replacement, DecodeResources.ReleaseReason.LOCAL_ROLLBACK).status();
            f.assertEmpty();
        }
    }

    private enum LocalEnd { CANCEL, FUTURE_CANCEL, EXPIRE, SHUTDOWN }

    private static final class Fixture implements AutoCloseable {
        final AbstractRequestScheduler requests;
        final SchedulerRuntime runtime;
        final PrefillEndpoint prefill;
        final DecodeEndpoint decode;
        final RequestRoute item;
        final DecodeResources.ReservationHandle reservation;
        final PrefillState.RouteReservation routeReservation;
        final PrefillState.CommittedHandoff prefillHandoff;
        final AdmissionHandle admission;
        final DecodeResources.AdmissionCapacity capacity = new DecodeResources.AdmissionCapacity(1, 90);
        long version = 1;
        final java.util.concurrent.CompletableFuture<EngineGrpcClient.CancelReply> cleanup = new java.util.concurrent.CompletableFuture<>();

        Fixture() { this(false); }

        Fixture(boolean keepAdmissionOpen) {
            var config = SchedulingTestConfig.newConfig();
            config.setScheduler(ConfigTestSupport.direct());
            SchedulingTestConfig.useNonBatchDispatcher(config).setMaxInflightPerPrefillWorker(1);
            config.getRequestLifecycle().getRequest().setTimeoutMs(TTL);
            ConfigService service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            var reporter = mock(DeliveryMetricsReporter.class);
            var requestReporter = mock(RequestSchedulerReporter.class);
            requests = SchedulerTestSupport.create(service, reporter, requestReporter,
                mock(RecentCacheKeyTraceReporter.class));
            var channel = mock(EngineGrpcClient.class);
            when(channel.cancelAsync(any(), anyInt(), any(), anyLong())).thenReturn(cleanup);
            org.springframework.test.util.ReflectionTestUtils.setField(requests.runtime, "engineClient", channel);
            var projector = requests;
            var endpoints = new EndpointRegistry(service, SchedulerTestSupport.repository(projector), reporter, DeliveryTestSupport.route(reporter), new PlacementAvailability());
            runtime = requests.runtime;
            org.springframework.test.util.ReflectionTestUtils.setField(runtime, "endpoints", endpoints);
            var worker = WorkerStatus.createDiscovered(RoleType.PREFILL, "g", "127.0.0.1", 8080, 8081, "test");
            worker.lock.lock();
            try {
                prefill = org.mockito.Mockito.spy((PrefillEndpoint) endpoints.publishPreparedEndpoint(worker.getIpPort(), worker,
                        worker.prepareNewStatus(WorkerStatusTestSupport.freezeStatusResponse(worker, status(RoleType.PREFILL, 1, Map.of(), Map.of(), TOTAL_KV)))));
            } finally { worker.lock.unlock(); }
            decode = EndpointTestSupport.decode(WorkerStatus.createDiscovered(RoleType.DECODE, "g", "127.0.0.2", 8080, 8081, "test"), SchedulerTestSupport.repository(projector));
            decodeStatus(Map.of(), Map.of(), TOTAL_KV);
            try (var pin = decode.tryPinGeneration()) {
                reservation = decode.tryReserveQueuedRequest(pin, ID, HARD_KV, EXPECTED_KV, 50, null).handle();
            }
            assertNotNull(reservation);
            var context = RequestProtocolTestSupport.context(config, ID);
            var future = RequestProtocolTestSupport.register(requests, context);
            context.data().setFuture(future);
            item = SchedulingTestConfig.createRoute(context, new Response(), null, null, prefill, decode, reservation, System.currentTimeMillis());
            var held = new AtomicReference<PrefillState.RouteReservation>();
            admission = requests.claimAdmissionHandle(ID, future);
            assertNotNull(admission);
            try (var pin = prefill.tryPinGeneration()) {
                assertTrue((requests.commitRoute(item, RequestProtocolTestSupport.publication(() -> {
                    var result = prefill.reserveUnqueuedRoute(pin, item, 30_000);
                    assertEquals(PrefillState.CapacityStatus.ACQUIRED, result.status());
                    held.set(result.reservation());
                    return true;
                })) == org.flexlb.balance.PlacementResult.Status.SUCCESS));
            }
            routeReservation = held.get();
            prefillHandoff = prefill.commit(new PrefillState.CommitPlan(PrefillState.CommitKind.DIRECT, List.of(item), List.of(routeReservation), null, new long[0], null, System.currentTimeMillis()));
            assertNotNull(prefillHandoff);
            if (!keepAdmissionOpen) { admission.finish(); }
        }

        DeliveryClaim handoff() {
            var acquisition = decode.acquireDispatchPermit(reservation, capacity);
            assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED, acquisition.status());
            var member = new DeliveryDispatcher.Member(item, acquisition.handle());
            try {
                var claim = requests.claimDelivery(item, DeliveryClaimKind.BATCH_ENQUEUE, 1L, member.decode());
                assertNotNull(claim);
                assertTrue(claim.item().ctx().data().getScheduler().tryStartSend(claim));
                return claim;
            } finally {
                DeliveryDispatcher.closeCommitted(List.of(member), prefillHandoff);
            }
        }

        void assertReserved() {
            assertEquals(1, SchedulerTestSupport.repository(requests).liveRequestCount());
            assertEquals(1, prefill.ownership.stats().locallyOwnedRequests());
            assertEquals(HARD_KV, decode.routingView().inflightHardKv());
            assertEquals(EXPECTED_KV, EndpointTestSupport.expectedReservedKv(decode.resourceSnapshot()));
        }

        void assertHandedOff() {
            assertReserved();
            assertEquals(0, decode.resourceSnapshot().activeDispatchPermits());
            assertEquals(1, decode.routingView().engineCapacityUsed());
            assertFalse(capacity.evaluate(decode.routingView().dispatchUsage(), HARD_KV, EXPECTED_KV, CapacityRelease.NONE).fits());
        }

        void assertEmpty() {
            SchedulerTestSupport.awaitRequestWork(requests.runtime);
            assertEquals(0, SchedulerTestSupport.repository(requests).liveRequestCount(), "request inflight");
            assertEquals(0, prefill.ownership.progress().queueDepth(), "Prefill queue membership");
            assertEquals(0, prefill.ownership.stats().locallyOwnedRequests(), "Prefill inflight");
            assertEquals(0, prefill.ownership.stats().individuallyOwnedRequests(), "Prefill individual lease");
            assertEquals(0, prefill.ownership.stats().batchCount(), "Prefill batch occupancy");
            var view = decode.resourceSnapshot();
            assertEquals(0, view.routing().inflightHardKv(), "Decode hard KV reservation");
            assertEquals(0, EndpointTestSupport.expectedReservedKv(view), "Decode expected KV reservation");
            assertEquals(0, view.activeDispatchPermits(), "Decode dispatch permits");
            assertEquals(0, view.queuedCount(), "Decode queued ownership");
            assertEquals(0, view.acceptedCount(), "Decode accepted streams");
            assertEquals(0, view.runningCount(), "Decode running streams");
            assertEquals(0, view.engineCapacityUsed(), "Decode request capacity");
            assertTrue(view.reservedCount() == 0);
            assertTrue(view.confirmedCount() == 0);
        }

        void assertCapacityReusable() {
            try (var pin = decode.tryPinGeneration()) {
                var next = decode.tryReserveQueuedRequest(pin, 999L, HARD_KV, EXPECTED_KV, 50, null).handle();
                assertNotNull(next);
                var permit = decode.acquireDispatchPermit(next, capacity);
                assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED, permit.status());
                assertTrue(permit.handle().release());
                decode.release(next, DecodeResources.ReleaseReason.LOCAL_ROLLBACK).status();
            }
            // A second exact request must be able to use the sole Prefill slot too.
            var nextContext = RequestProtocolTestSupport.context(item.ctx().data().getConfig(), 999L);
            nextContext.data().setFuture(new CompletableFuture<>());
            var next = SchedulingTestConfig.createRoute(nextContext,
                    new Response(), null, null, prefill, null, null, System.currentTimeMillis());
            try (var pin = prefill.tryPinGeneration()) {
                var result = prefill.reserveUnqueuedRoute(pin, next, 1);
                assertEquals(PrefillState.CapacityStatus.ACQUIRED, result.status());
                prefill.release(result.reservation(), null);
            }
            assertEmpty();
        }

        void proveCleanup() throws Exception {
            cleanup.complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
            requests.runtime.signals().delivery(item.ctx().snapshot().delivery().identity()).toCompletableFuture().get(2, TimeUnit.SECONDS);
            SchedulerTestSupport.awaitRequestWork(requests.runtime);
        }
        void expire() { RequestProtocolTestSupport.expireInactiveRequest(requests, requests.requests.findActive(ID), System.currentTimeMillis() + TTL + 1); }
        void decodeStatus(Map<String, TaskInfo> running, Map<String, TaskInfo> finished, long freeKv) {
            applyStatus(decode, status(RoleType.DECODE, version++, running, finished, freeKv));
        }
        @Override
        public void close() {
            try {
                admission.finish();
                prefillHandoff.close();
                prefill.release(routeReservation, null);
                runtime.shutdown();
            } finally {
                decode.close();
            }
        }
    }

    private static TaskInfo task(long id) {
        var task = new TaskInfo();
        task.setRequestId(id);
        task.setInputLength(HARD_KV);
        task.setPhase(TaskPhase.RUNNING);
        return task;
    }

    private static WorkerStatusResponse status(RoleType role, long version, Map<String, TaskInfo> running,
                                                Map<String, TaskInfo> finished, long freeKv) {
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
        try { projection = endpoint.applyPreparedStatus(worker, worker.prepareNewStatus(WorkerStatusTestSupport.freezeStatusResponse(worker, response))); }
        finally { worker.lock.unlock(); }
        endpoint.publishStatus(projection);
    }
}
