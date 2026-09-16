package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.RequestSlot.DeliveryClaim;
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
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** Resource assertions use real endpoint ledgers, not invocations of mocked release methods. */
class RequestResourceAccountingTest {
    private static final long ID = 101L;
    private static final long HARD_KV = 160L;
    private static final long EXPECTED_KV = 320L;
    private static final long TOTAL_KV = 10_000L;
    private static final long TTL = TimeUnit.HOURS.toMillis(1);

    @ParameterizedTest
    @EnumSource(LocalEnd.class)
    void localTerminationReclaimsReservationAndPrefillInflight(LocalEnd end) throws Exception {
        try (Fixture f = new Fixture()) {
            f.assertReserved();
            switch (end) {
                case CANCEL -> f.requests.cancelRequest(ID, 0, CancelReason.CLIENT_CANCELLED);
                case FUTURE_CANCEL -> assertTrue(f.item.future().cancel(false));
                case EXPIRE -> f.expire();
                case SHUTDOWN -> {
                    assertTrue(f.requests.closeAdmissionAndAwaitMutations());
                    f.requests.closeOutstandingAndTerminalize();
                }
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
                f.requests.cancelRequest(ID, 0, reason);
            } else {
                f.expire();
            }
            assertFalse(f.item.future().isDone());
            f.assertReserved();
            f.admission.close();
            f.assertEmpty();
            f.assertCapacityReusable();
        }
    }

    @Test
    void cancellationDuringPreparationAlsoReclaimsTheAcquiredDispatchPermit() throws Exception {
        try (Fixture f = new Fixture()) {
            var member = PrefillAdmissionResources.prepareMember(f.item);
            assertTrue(member.accepted());
            assertEquals(1, f.decode.layeredAdmissionView().activeDispatchPermits());
            assertEquals(1, f.decode.routingView().engineCapacityUsed());
            f.requests.cancelRequest(ID, 0, CancelReason.CLIENT_CANCELLED);
            // The transaction's eventual rollback must remain harmless after request cleanup.
            assertNull(PrefillAdmissionResources.rollbackMember(member.value(), null));
            f.assertEmpty();
            f.assertCapacityReusable();
        }
    }

    @ParameterizedTest
    @EnumSource(value = DeliveryResult.Status.class, names = {"DELIVERED", "UNCERTAIN", "TIMED_OUT"})
    void responseOrAmbiguousTransportRetainsResourcesUntilInactivity(DeliveryResult.Status result) throws Exception {
        try (Fixture f = new Fixture()) {
            DeliveryClaim claim = f.handoff();
            claim.complete(new DeliveryResult(result, result == DeliveryResult.Status.DELIVERED ? null : new IllegalStateException("reply lost")));
            if (result == DeliveryResult.Status.DELIVERED) {
                assertTrue(f.item.future().get(2, TimeUnit.SECONDS).isSuccess());
            }
            f.assertHandedOff();
            f.requests.cancelRequest(ID, 0, CancelReason.CLIENT_CANCELLED);
            f.assertHandedOff();
            f.expire();
            f.assertEmpty();
            f.assertCapacityReusable();
        }
    }

    @Test
    void localRollbackCannotReleaseAnEngineOwnerWhoseProjectionHasNotArrived() throws Exception {
        try (Fixture f = new Fixture()) {
            var acquisition = f.decode.acquireEngineDispatchPermit(f.reservation, f.capacity);
            assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.ACQUIRED, acquisition.status());
            assertEquals(DecodeEndpoint.EngineDispatchPermitTransferStatus.TRANSFERRED,
                    acquisition.permit().transferToEngineLifecycle());
            // Endpoint ownership can advance before its notification reaches the slot.
            f.requests.cancelRequest(ID, 0, CancelReason.CLIENT_CANCELLED);
            assertEquals(0, f.requests.liveRequestCount());
            assertEquals(0, f.prefill.getLocallyOwnedRequestCount());
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
            claim.complete(DeliveryResult.notSent(new IllegalStateException("not delivered")));
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
            assertEquals(1, f.decode.layeredAdmissionView().runningCount());
            assertEquals(1, f.decode.routingView().engineCapacityUsed());
            assertEquals(0, f.decode.routingView().inflightHardKv(), "Engine status replaces the local KV prediction");
            claim.complete(DeliveryResult.uncertain(new IllegalStateException("late transport failure")));
            assertTrue(f.item.future().get(2, TimeUnit.SECONDS).isSuccess());
            assertEquals(1, f.requests.liveRequestCount());
            assertEquals(1, f.decode.layeredAdmissionView().runningCount());
            assertEquals(1, f.prefill.getLocallyOwnedRequestCount());
            applyStatus(f.prefill, status(RoleType.PREFILL, 2L, Map.of(), Map.of("101", running), TOTAL_KV));
            f.decodeStatus(Map.of(), Map.of("101", running), TOTAL_KV);
            f.assertEmpty();
            f.assertCapacityReusable();
        }
    }

    @Test
    void inactivityDropsLocalRunningOwnershipWithoutInventingPhysicalKvRelease() throws Exception {
        try (Fixture f = new Fixture()) {
            f.handoff().complete(DeliveryResult.delivered());
            f.decodeStatus(Map.of("101", task(ID)), Map.of(), TOTAL_KV - HARD_KV);
            assertEquals(1, f.decode.layeredAdmissionView().runningCount());
            f.expire();
            f.assertEmpty();
            assertEquals(TOTAL_KV - HARD_KV, f.decode.realKvAvailable(),
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
            f.assertEmpty();
            var replacementRef = new AtomicReference<DecodeEndpoint.ReservationHandle>();
            RequestLifecycleTestSupport.awaitCondition(() -> {
                if (replacementRef.get() != null) { return true; }
                f.decode.evictExpiredRequests(0, ignored -> false);
                try (var pin = f.decode.tryPinGeneration()) {
                    replacementRef.set(f.decode.tryReservePlacementPinned(pin, ID, HARD_KV * 2, EXPECTED_KV * 2, 50));
                }
                return replacementRef.get() != null;
            });
            var replacement = replacementRef.get();
            assertNotEquals(oldReservation.reservationToken(), replacement.reservationToken());
            oldClaim.complete(DeliveryResult.delivered());
            f.expire();
            assertFalse(f.decode.expireReservationExact(oldReservation));
            assertFalse(f.decode.releaseLocalShadowIfExact(oldReservation));
            assertEquals(HARD_KV * 2, f.decode.routingView().inflightHardKv());
            assertEquals(EXPECTED_KV * 2, f.decode.routingView().inflightExpectedKv());
            assertEquals(replacement, f.decode.reservationHandle(ID));
            f.decode.releaseReservationExact(replacement);
            f.assertEmpty();
        }
    }

    private enum LocalEnd { CANCEL, FUTURE_CANCEL, EXPIRE, SHUTDOWN }

    private static final class Fixture implements AutoCloseable {
        final RequestRegistry requests;
        final SchedulerRuntime runtime;
        final PrefillEndpoint prefill;
        final DecodeEndpoint decode;
        final ScheduledRequest item;
        final DecodeEndpoint.ReservationHandle reservation;
        final PrefillState.RouteReservation routeReservation;
        final PrefillState.CommittedHandoff prefillHandoff;
        final RequestSlot.AdmissionHandle admission;
        final DecodeEndpoint.AdmissionCapacity capacity = new DecodeEndpoint.AdmissionCapacity(1, 90);
        long version = 1;

        Fixture() { this(false); }

        Fixture(boolean keepAdmissionOpen) {
            var config = SchedulingTestConfig.newConfig();
            config.setScheduler(SchedulerConfig.direct());
            SchedulingTestConfig.useNonBatchDispatcher(config).setMaxInflightPerPrefillWorker(1);
            config.getRequestLifecycle().getRequest().setTimeoutMs(TTL);
            ConfigService service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            var reporter = mock(BatchSchedulerReporter.class);
            var requestReporter = mock(RequestSchedulerReporter.class);
            requests = new RequestRegistry(service, reporter, requestReporter);
            var projector = new EndpointEventProjector(requests);
            var endpoints = new EndpointRegistry(service, projector, reporter,
                    new RouteDeliveryStrategy(requests, new DeliveryMetrics(reporter)), new PlacementAvailability());
            runtime = new SchedulerRuntime(requests, endpoints, reporter, requestReporter);
            var worker = WorkerStatus.createDiscovered(RoleType.PREFILL, "g", "127.0.0.1", 8080, 8081, "test");
            worker.lock.lock();
            try {
                prefill = (PrefillEndpoint) endpoints.publishPreparedEndpoint(worker.getIpPort(), worker,
                        worker.prepareNewStatus(worker.freezeStatusResponse(status(RoleType.PREFILL, 1, Map.of(), Map.of(), TOTAL_KV)))).endpoint();
            } finally { worker.lock.unlock(); }
            decode = new DecodeEndpoint(WorkerStatus.createDiscovered(RoleType.DECODE, "g", "127.0.0.2", 8080, 8081, "test"), projector);
            decodeStatus(Map.of(), Map.of(), TOTAL_KV);
            try (var pin = decode.tryPinGeneration()) {
                reservation = decode.tryReservePlacementPinned(pin, ID, HARD_KV, EXPECTED_KV, 50);
            }
            assertNotNull(reservation);
            var context = RequestLifecycleTestSupport.context(config, ID);
            var future = requests.register(context);
            item = new ScheduledRequest(context, future, new Response(), null, null, prefill, decode, reservation, System.currentTimeMillis());
            var held = new AtomicReference<PrefillState.RouteReservation>();
            admission = requests.claimAdmissionHandle(ID, future);
            assertNotNull(admission);
            try (var pin = prefill.tryPinGeneration()) {
                assertTrue(requests.commitItemForPublication(item, () -> {
                    var result = prefill.reserveUnqueuedRoute(pin, item, 30_000);
                    assertEquals(PrefillState.CapacityStatus.ACQUIRED, result.status());
                    held.set(result.reservation());
                    return true;
                }));
            }
            routeReservation = held.get();
            try (var commit = prefill.tryBeginRouteCommitAdmission()) {
                assertNotNull(commit);
                prefillHandoff = commit.commit(List.of(item), List.of(routeReservation));
            }
            if (!keepAdmissionOpen) { admission.close(); }
        }

        DeliveryClaim handoff() {
            var acquisition = decode.acquireEngineDispatchPermit(reservation, capacity);
            assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.ACQUIRED, acquisition.status());
            var member = new PrefillAdmissionResources.Member(item, acquisition.permit());
            try (var owner = PrefillAdmissionResources.createCommittedOwner(List.of(member))) {
                owner.bindPrefillHandoff(prefillHandoff);
                var claim = RequestLifecycleTestSupport.claimBatchWithoutPrediction(
                        requests, item, 1L, () -> owner.transferToEndpoint(item));
                assertNotNull(claim);
                return claim;
            }
        }

        void assertReserved() {
            assertEquals(1, requests.liveRequestCount());
            assertEquals(1, prefill.getLocallyOwnedRequestCount());
            assertEquals(HARD_KV, decode.routingView().inflightHardKv());
            assertEquals(EXPECTED_KV, decode.routingView().inflightExpectedKv());
        }

        void assertHandedOff() {
            assertReserved();
            assertEquals(0, decode.layeredAdmissionView().activeDispatchPermits());
            assertEquals(1, decode.routingView().engineCapacityUsed());
            assertFalse(capacity.evaluate(decode.routingView().dispatchUsage(), HARD_KV, EXPECTED_KV).fits());
        }

        void assertEmpty() {
            assertEquals(0, requests.liveRequestCount(), "request inflight");
            assertEquals(0, prefill.queuedRequestCount(), "Prefill queue membership");
            assertEquals(0, prefill.getLocallyOwnedRequestCount(), "Prefill inflight");
            assertEquals(0, prefill.getIndividuallyTrackedRequestCount(), "Prefill individual lease");
            assertEquals(0, prefill.getInflightBatchCount(), "Prefill batch occupancy");
            var view = decode.layeredAdmissionView();
            assertEquals(0, view.routing().inflightHardKv(), "Decode hard KV reservation");
            assertEquals(0, view.routing().inflightExpectedKv(), "Decode expected KV reservation");
            assertEquals(0, view.activeDispatchPermits(), "Decode dispatch permits");
            assertEquals(0, view.queuedCount(), "Decode queued ownership");
            assertEquals(0, view.acceptedCount(), "Decode accepted streams");
            assertEquals(0, view.runningCount(), "Decode running streams");
            assertEquals(0, view.engineCapacityUsed(), "Decode request capacity");
            assertTrue(view.reserved().isEmpty());
            assertTrue(view.confirmed().isEmpty());
        }

        void assertCapacityReusable() {
            try (var pin = decode.tryPinGeneration()) {
                var next = decode.tryReservePlacementPinned(pin, 999L, HARD_KV, EXPECTED_KV, 50);
                assertNotNull(next);
                var permit = decode.acquireEngineDispatchPermit(next, capacity);
                assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.ACQUIRED, permit.status());
                assertTrue(permit.permit().release());
                decode.releaseReservationExact(next);
            }
            // A second exact request must be able to use the sole Prefill slot too.
            var nextContext = RequestLifecycleTestSupport.context(item.ctx().getConfig(), 999L);
            var next = new ScheduledRequest(nextContext, new CompletableFuture<>(),
                    new Response(), null, null, prefill, null, null, System.currentTimeMillis());
            try (var pin = prefill.tryPinGeneration()) {
                var result = prefill.reserveUnqueuedRoute(pin, next, 1);
                assertEquals(PrefillState.CapacityStatus.ACQUIRED, result.status());
                result.reservation().close();
            }
            assertEmpty();
        }

        void expire() { requests.expireInactiveRequest(requests.requestSlot(ID), System.currentTimeMillis() + TTL + 1); }
        void decodeStatus(Map<String, TaskInfo> running, Map<String, TaskInfo> finished, long freeKv) {
            applyStatus(decode, status(RoleType.DECODE, version++, running, finished, freeKv));
        }
        @Override
        public void close() {
            try {
                admission.close();
                prefillHandoff.close();
                routeReservation.close();
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
        Runnable projection;
        worker.lock.lock();
        try { projection = endpoint.applyPreparedStatus(worker, worker.prepareNewStatus(worker.freezeStatusResponse(response))); }
        finally { worker.lock.unlock(); }
        projection.run();
    }
}
