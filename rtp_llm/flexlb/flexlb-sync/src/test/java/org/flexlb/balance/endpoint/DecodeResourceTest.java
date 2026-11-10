package org.flexlb.balance.endpoint;

import org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRequestStatus;
import org.flexlb.balance.endpoint.DecodeResources.DispatchOutcome;
import org.flexlb.balance.endpoint.DecodeResources.ReleaseReason;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.master.WorkerStatusTestSupport;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.enums.TaskPhase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;
import java.util.Map;

import static org.flexlb.balance.endpoint.DecodeResources.EngineDispatchPermitTransferStatus.OWNERSHIP_LOST;
import static org.flexlb.balance.endpoint.DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED;
import static org.flexlb.balance.endpoint.DecodeResources.ReservationReleaseResult.ENGINE_ACCEPTED;
import static org.flexlb.balance.endpoint.DecodeResources.ReservationReleaseResult.RELEASED;
import static org.flexlb.balance.endpoint.DecodeResources.ReservationReleaseResult.STALE;
import static org.flexlb.balance.endpoint.DecodeResources.ReservationReleaseResult.STILL_OWNED;
import static org.flexlb.balance.endpoint.EndpointTestSupport.withStateLock;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Ledger transitions do not invoke the endpoint, scheduler, RPC client or callbacks.
 */
class DecodeResourceTest {
    private static final AdmissionCapacity CAPACITY = new AdmissionCapacity(2, 100);

    @Test
    void dispatchAndReturnRaceResolvesOnePermitWithoutRecreatingCapacity() throws Exception {
        try (var workers = java.util.concurrent.Executors.newFixedThreadPool(2)) {
            for (int run = 0; run < 30; run++) {
                var state = EndpointTestSupport.decodeEndpoint(status());
                var reservation =
                        EndpointTestSupport.reserveQueuedDecode(state, 1L, 100L, 200L, 50, CAPACITY)
                                .value();
                var permit = state.acquireDispatchPermit(reservation, CAPACITY).value();
                var start = new java.util.concurrent.CyclicBarrier(2);
                var sent =
                        workers.submit(
                                () -> {
                                    start.await(2, java.util.concurrent.TimeUnit.SECONDS);
                                    return permit.dispatch().status();
                                });
                var returned =
                        workers.submit(
                                () -> {
                                    start.await(2, java.util.concurrent.TimeUnit.SECONDS);
                                    return permit.release();
                                });
                boolean transferred =
                        sent.get(2, java.util.concurrent.TimeUnit.SECONDS) == TRANSFERRED;
                assertEquals(!transferred, returned.get(2, java.util.concurrent.TimeUnit.SECONDS));
                assertEquals(0, state.resourceSnapshot().activeDispatchPermits());
                assertEquals(transferred ? 1 : 0, state.routingView().engineCapacityUsed());
                assertFalse(permit.release(), "late return cannot release a newer resource owner");
                if (transferred) {
                    assertThrows(
                            IllegalStateException.class,
                            () -> state.release(reservation, ReleaseReason.LOCAL_ROLLBACK));
                } else {
                    assertEquals(
                            RELEASED,
                            state.release(reservation, ReleaseReason.LOCAL_ROLLBACK).status());
                }
            }
        }
    }

    @Test
    void failedRetirementSnapshotDoesNotInvalidateTheLiveDispatchPermit() {
        WorkerStatus status = org.mockito.Mockito.spy(status());
        long generation = status.getGenerationId();
        DecodeEndpoint state = EndpointTestSupport.decodeEndpoint(status);
        var reservation =
                EndpointTestSupport.reserveQueuedDecode(state, 1L, 100L, 200L, 50, CAPACITY)
                        .value();
        var permit = state.acquireDispatchPermit(reservation, CAPACITY).value();
        long version = state.placementVersion();
        IllegalStateException failure = new IllegalStateException("snapshot unavailable");
        org.mockito.Mockito.doThrow(failure).when(status).getGenerationId();
        assertSame(failure, assertThrows(IllegalStateException.class, state::retire));
        org.mockito.Mockito.doReturn(generation).when(status).getGenerationId();
        assertEquals(version, state.placementVersion());
        assertEquals(1, state.resourceSnapshot().activeDispatchPermits());
        assertTrue(
                state.dispatch(permit, DispatchOutcome.ABANDONED).capacityReleased(),
                "failed retirement must not make the live permit look already retired");
        assertEquals(0, state.resourceSnapshot().activeDispatchPermits());

        var replacement = state.acquireDispatchPermit(reservation, CAPACITY).value();
        assertEquals(java.util.List.of(reservation), state.retire());
        assertFalse(state.ownership(reservation).owned());
        assertEquals(0, state.resourceSnapshot().activeDispatchPermits());
        assertEquals(TRANSFERRED, state.dispatch(replacement, DispatchOutcome.ABANDONED).status());
        assertTrue(state.retire().isEmpty());
    }

    @Test
    void returnedPermitCannotDispatchOrReleaseItsReplacement() {
        WorkerStatus status = status();
        DecodeEndpoint state = EndpointTestSupport.decodeEndpoint(status);
        var reservation =
                EndpointTestSupport.reserveQueuedDecode(state, 1, 100, 200, 50, CAPACITY).value();
        var first = state.acquireDispatchPermit(reservation, CAPACITY).value();
        assertTrue(state.dispatch(first, DispatchOutcome.ABANDONED).capacityReleased());
        assertTrue(EndpointTestSupport.isQueued(state.resourceSnapshot(), String.valueOf(1)));

        var replacement = state.acquireDispatchPermit(reservation, CAPACITY).value();
        assertEquals(OWNERSHIP_LOST, state.dispatch(first, DispatchOutcome.ENGINE_OWNED).status());
        assertEquals(OWNERSHIP_LOST, state.dispatch(first, DispatchOutcome.ABANDONED).status());
        assertEquals(1, state.resourceSnapshot().activeDispatchPermits());
        assertEquals(
                TRANSFERRED, state.dispatch(replacement, DispatchOutcome.ENGINE_OWNED).status());
        assertEquals(0, state.resourceSnapshot().activeDispatchPermits());
        assertEquals(1, state.routingView().engineLoad());
        assertThrows(
                IllegalStateException.class,
                () -> state.release(reservation, ReleaseReason.LOCAL_ROLLBACK).status());
        assertEquals(
                STILL_OWNED,
                state.release(reservation, ReleaseReason.COUNTERPART_FINISHED).status());
        assertEquals(RELEASED, state.release(reservation, ReleaseReason.EXPIRED).status());
        assertFalse(state.ownership(reservation).owned());
        assertEquals(0, state.routingView().engineLoad());
    }

    @ParameterizedTest
    @CsvSource({
        "MASTER_QUEUED_NOT_DISPATCHED, LOCAL_ROLLBACK",
        "MASTER_QUEUED_NOT_DISPATCHED, COUNTERPART_FINISHED",
        "MASTER_QUEUED_NOT_DISPATCHED, NOT_SENT",
        "MASTER_QUEUED_NOT_DISPATCHED, UNSENT_ROUTE_ROLLBACK",
        "ENGINE_MAY_HAVE_SEEN, LOCAL_ROLLBACK",
        "ENGINE_MAY_HAVE_SEEN, COUNTERPART_FINISHED",
        "ENGINE_MAY_HAVE_SEEN, NOT_SENT",
        "ENGINE_MAY_HAVE_SEEN, UNSENT_ROUTE_ROLLBACK",
        "ACCEPTED_NOT_RUNNING, LOCAL_ROLLBACK",
        "ACCEPTED_NOT_RUNNING, COUNTERPART_FINISHED",
        "ACCEPTED_NOT_RUNNING, NOT_SENT",
        "ACCEPTED_NOT_RUNNING, UNSENT_ROUTE_ROLLBACK",
        "RUNNING, LOCAL_ROLLBACK",
        "RUNNING, COUNTERPART_FINISHED",
        "RUNNING, NOT_SENT",
        "RUNNING, UNSENT_ROUTE_ROLLBACK"
    })
    void releaseReasonPreservesEngineOwnershipAndSettlesOnlyItsExactReservation(
            DecodeTaskPhase phase, ReleaseReason reason) {
        WorkerStatus status = status();
        DecodeEndpoint state = EndpointTestSupport.decodeEndpoint(status);
        var reservation =
                EndpointTestSupport.reserveQueuedDecode(state, 1, 100, 200, 50, CAPACITY).value();
        var permit = state.acquireDispatchPermit(reservation, CAPACITY).value();
        if (phase == DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN) {
            state.dispatch(permit, DispatchOutcome.ENGINE_OWNED).status();
        } else if (phase.isEngineConfirmed()) {
            calibrate(
                    state,
                    status,
                    Map.of(
                            "1",
                            task(
                                    1,
                                    phase == DecodeTaskPhase.RUNNING
                                            ? TaskPhase.RUNNING
                                            : TaskPhase.KV_ALLOCATED)),
                    Map.of());
        }
        var sibling =
                EndpointTestSupport.reserveQueuedDecode(state, 2, 70, 90, 30, CAPACITY).value();
        long version = state.placementVersion();
        var stale =
                new DecodeResources.RequestReservation(
                        reservation.endpointGenerationId(),
                        String.valueOf(1),
                        reservation.reservationToken() + 100);
        var foreign =
                new DecodeResources.RequestReservation(
                        reservation.endpointGenerationId() + 1,
                        String.valueOf(1),
                        reservation.reservationToken());
        assertEquals(STALE, state.release(stale, reason).status());
        assertEquals(STALE, state.release(foreign, reason).status());
        assertEquals(version, state.placementVersion());

        boolean released =
                phase == DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED
                        || phase == DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN
                                && (reason == ReleaseReason.NOT_SENT
                                        || reason == ReleaseReason.UNSENT_ROUTE_ROLLBACK);
        if (released) {
            assertEquals(RELEASED, state.release(reservation, reason).status());
            assertEquals(STALE, state.release(reservation, reason).status());
            assertFalse(state.ownership(reservation).owned());
            assertEquals(70L, state.routingView().inflightHardKv());
            assertEquals(90L, EndpointTestSupport.expectedReservedKv(state.resourceSnapshot()));
            assertEquals(0, state.resourceSnapshot().activeDispatchPermits());
            assertEquals(1, state.routingView().totalLoad());
            assertEquals(
                    OWNERSHIP_LOST, state.dispatch(permit, DispatchOutcome.ABANDONED).status());
        } else {
            if (reason == ReleaseReason.LOCAL_ROLLBACK) {
                assertThrows(
                        IllegalStateException.class,
                        () -> state.release(reservation, reason).status());
            } else {
                assertEquals(
                        reason == ReleaseReason.NOT_SENT
                                        || reason == ReleaseReason.UNSENT_ROUTE_ROLLBACK
                                ? ENGINE_ACCEPTED
                                : STILL_OWNED,
                        state.release(reservation, reason).status());
            }
            assertTrue(state.ownership(reservation).owned());
            assertEquals(version, state.placementVersion());
            assertEquals(2, state.routingView().totalLoad());
        }
        assertTrue(state.ownership(sibling).owned());
        assertEquals(1, state.resourceSnapshot().queuedCount());
    }

    @Test
    void calibrationConvertsTheSameReservationWithoutDoubleChargingAndSettlesTerminal() {
        WorkerStatus status = status();
        DecodeEndpoint state = EndpointTestSupport.decodeEndpoint(status);
        var reservation =
                EndpointTestSupport.reserveQueuedDecode(state, 2, 100, 200, 50, CAPACITY).value();
        var permit = state.acquireDispatchPermit(reservation, CAPACITY).value();
        TaskInfo task = new TaskInfo();
        task.setRequestId(String.valueOf(2L));
        task.setPhase(TaskPhase.KV_ALLOCATED);
        var accepted = calibrate(state, status, Map.of("2", task), Map.of());
        assertEquals(reservation, accepted.getFirst().reservation());
        assertTrue(state.ownership(reservation).confirmed());
        assertEquals(1, state.routingView().engineCapacityUsed());
        assertEquals(0, state.resourceSnapshot().activeDispatchPermits());
        assertTrue(state.resourceSnapshot().reservedCount() == 0);
        assertEquals(TRANSFERRED, state.dispatch(permit, DispatchOutcome.ENGINE_OWNED).status());

        var finished = calibrate(state, status, Map.of(), Map.of("2", task));
        assertEquals(DecodeResources.DecodeRequestStatus.Kind.TERMINAL, finished.getFirst().kind());
        assertEquals(reservation, finished.getFirst().reservation());
        assertFalse(state.ownership(reservation).owned());
        assertEquals(0, state.routingView().engineCapacityUsed());
        // The unified public entry returns this permit's original decision. It cannot recreate
        // resource ownership after an independently observed terminal result.
        assertEquals(TRANSFERRED, state.dispatch(permit, DispatchOutcome.ENGINE_OWNED).status());
        assertFalse(state.ownership(reservation).owned());
        assertEquals(0, state.resourceSnapshot().activeDispatchPermits());
    }

    @ParameterizedTest
    @CsvSource({
        "MASTER_QUEUED_NOT_DISPATCHED, false", "MASTER_QUEUED_NOT_DISPATCHED, true",
        "ENGINE_MAY_HAVE_SEEN, false", "ENGINE_MAY_HAVE_SEEN, true",
        "ACCEPTED_NOT_RUNNING, false", "ACCEPTED_NOT_RUNNING, true",
        "RUNNING, false", "RUNNING, true"
    })
    void terminalAndExpirationReleaseOnlyTheirOwnerAcrossResourcePhases(
            DecodeTaskPhase phase, boolean expired) {
        WorkerStatus status = status();
        DecodeEndpoint state = EndpointTestSupport.decodeEndpoint(status);
        var reservation =
                EndpointTestSupport.reserveQueuedDecode(state, 1, 100, 200, 50, CAPACITY).value();
        var permit = state.acquireDispatchPermit(reservation, CAPACITY).value();
        if (phase == DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN) {
            state.dispatch(permit, DispatchOutcome.ENGINE_OWNED).status();
        } else if (phase.isEngineConfirmed()) {
            calibrate(
                    state,
                    status,
                    Map.of(
                            "1",
                            task(
                                    1,
                                    phase == DecodeTaskPhase.RUNNING
                                            ? TaskPhase.RUNNING
                                            : TaskPhase.KV_ALLOCATED)),
                    Map.of());
        }
        var sibling =
                EndpointTestSupport.reserveQueuedDecode(state, 2, 70, 90, 30, CAPACITY).value();
        assertNotNull(sibling);
        var stale =
                new DecodeResources.RequestReservation(
                        reservation.endpointGenerationId(),
                        String.valueOf(1),
                        reservation.reservationToken() + 100);
        assertEquals(STALE, state.release(stale, ReleaseReason.EXPIRED).status());
        assertTrue(state.ownership(reservation).owned());

        for (int i = 0; i < 2; i++) {
            if (expired) {
                assertEquals(
                        i == 0 ? RELEASED : STALE,
                        state.release(reservation, ReleaseReason.EXPIRED).status());
            } else {
                calibrate(state, status, Map.of(), Map.of("1", task(1, TaskPhase.RUNNING)));
            }
            assertFalse(state.ownership(reservation).owned());
            assertTrue(state.ownership(sibling).owned());
            assertEquals(0, state.resourceSnapshot().activeDispatchPermits());
            assertEquals(1, state.resourceSnapshot().queuedCount());
            assertEquals(70L, state.routingView().inflightHardKv());
            assertEquals(90L, EndpointTestSupport.expectedReservedKv(state.resourceSnapshot()));
            assertEquals(1, state.routingView().totalLoad());
            assertEquals(
                    OWNERSHIP_LOST, state.dispatch(permit, DispatchOutcome.ABANDONED).status());
        }
        assertEquals(RELEASED, state.release(sibling, ReleaseReason.LOCAL_ROLLBACK).status());
        assertEquals(0, state.routingView().totalLoad());
        assertEquals(0, state.routingView().inflightHardKv());
        assertEquals(0, EndpointTestSupport.expectedReservedKv(state.resourceSnapshot()));
    }

    @ParameterizedTest
    @ValueSource(strings = {"EXPIRED", "COUNTERPART_FINISHED", "NOT_SENT", "FULL_ABSENCE"})
    void terminalHistoryFencesLateReportsUntilRetentionExpires(String settlement) {
        WorkerStatus status = status();
        DecodeEndpoint state = EndpointTestSupport.decodeEndpoint(status);
        var reservation =
                EndpointTestSupport.reserveQueuedDecode(state, 1, 100, 200, 50, CAPACITY).value();
        if (settlement.equals("FULL_ABSENCE")) {
            calibrate(state, status, Map.of("1", task(1, TaskPhase.RUNNING)), Map.of());
            calibrate(state, status, Map.of(), Map.of());
        } else {
            assertEquals(
                    RELEASED,
                    state.release(reservation, ReleaseReason.valueOf(settlement)).status());
        }
        assertFalse(state.ownership(reservation).owned());
        assertEquals(STALE, state.release(reservation, ReleaseReason.EXPIRED).status());
        assertFalse(
                EndpointTestSupport.evictDecode(state, Long.MAX_VALUE, ignored -> false)
                        .capacityReleased());
        calibrate(state, status, Map.of(), Map.of("1", task(1, TaskPhase.RUNNING)));
        var late = calibrate(state, status, Map.of("1", task(1, TaskPhase.RUNNING)), Map.of());
        assertTrue(late.isEmpty());
        assertEquals(0, state.routingView().totalLoad());

        assertTrue(
                EndpointTestSupport.evictDecode(state, -1L, ignored -> false).capacityReleased());
        calibrate(state, status, Map.of("1", task(1, TaskPhase.RUNNING)), Map.of());
        assertEquals(1, state.routingView().totalLoad());
        assertFalse(
                state.ownership(reservation).owned(),
                "expired history must not restore the old token");
    }

    @Test
    void initializationRejectsAnotherGenerationBeforeChangingResources() {
        WorkerStatus owner = status();
        DecodeEndpoint state = EndpointTestSupport.decodeEndpoint(owner);
        var incoming = response();
        incoming.setRunningTaskInfo(Map.of("1", task(1, TaskPhase.RUNNING)));
        long version = state.placementVersion();
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        state.initializeFromPreparedStatus(
                                state.getStatus(),
                                WorkerStatusTestSupport.freezeStatusResponse(status(), incoming)));
        assertEquals(version, state.placementVersion());
        assertEquals(0, state.routingView().totalLoad());
    }

    @Test
    void shadowSweepRechecksPhaseAfterRetentionCallbackConfirmsTheRequest() {
        WorkerStatus status = status();
        DecodeEndpoint state = EndpointTestSupport.decodeEndpoint(status);
        var reservation =
                EndpointTestSupport.reserveQueuedDecode(state, 1, 100, 200, 50, CAPACITY).value();
        state.acquireDispatchPermit(reservation, CAPACITY);

        var cleanup =
                EndpointTestSupport.evictDecode(
                        state,
                        -1L,
                        requestId -> {
                            calibrate(
                                    state,
                                    status,
                                    Map.of("1", task(1, TaskPhase.RUNNING)),
                                    Map.of());
                            return false;
                        });

        assertEquals(0, cleanup.expiredReservations());
        assertFalse(cleanup.capacityReleased());
        assertTrue(state.ownership(reservation).confirmed());
        assertTrue(state.ownership(reservation).owned());
        assertEquals(1, state.routingView().engineCapacityUsed());
        assertEquals(0, state.resourceSnapshot().activeDispatchPermits());
        assertEquals(0, state.resourceSnapshot().queuedCount());
        assertEquals(0, state.routingView().inflightHardKv());
    }

    @Test
    void endpointRejectsAnotherEndpointsPermitWithoutConsumingIt() {
        DecodeEndpoint owner =
                EndpointTestSupport.decode(
                        status(),
                        org.flexlb.balance.scheduler.SchedulerTestSupport.repository(
                                EndpointTestSupport.noopEventSink()));
        DecodeEndpoint other =
                EndpointTestSupport.decode(
                        status(),
                        org.flexlb.balance.scheduler.SchedulerTestSupport.repository(
                                EndpointTestSupport.noopEventSink()));
        try {
            DecodeResources.RequestReservation reservation;
            {
                var selectedEndpoint = owner;

                reservation =
                        org.flexlb.balance.scheduler.SelectionTestSupport.registerQueuedDecode(
                                        owner, String.valueOf(3), 100, 200, 50)
                                .value();
            }
            var permit = owner.acquireDispatchPermit(reservation, CAPACITY).value();
            assertThrows(
                    IllegalArgumentException.class,
                    () -> other.dispatch(permit, DispatchOutcome.ENGINE_OWNED).status());
            assertEquals(1, owner.resourceSnapshot().activeDispatchPermits());
            assertEquals(
                    TRANSFERRED, owner.dispatch(permit, DispatchOutcome.ENGINE_OWNED).status());
            assertEquals(
                    TRANSFERRED, owner.dispatch(permit, DispatchOutcome.ENGINE_OWNED).status());
        } finally {
            owner.close();
            other.close();
        }
    }

    @ParameterizedTest
    @EnumSource(
            value = TaskPhase.class,
            names = {"RECEIVED", "PENDING"})
    void regressedActivePhaseRetainsIdentityAcrossRepeatedReportsAndResumption(
            TaskPhase regressed) {
        WorkerStatus status = status();
        DecodeEndpoint state = EndpointTestSupport.decodeEndpoint(status);
        var reservation =
                EndpointTestSupport.reserveQueuedDecode(state, 10, 100, 200, 50, CAPACITY).value();
        calibrate(state, status, Map.of("10", task(10, TaskPhase.RUNNING)), Map.of());
        for (int i = 0; i < 2; i++) {
            var observation = calibrate(state, status, Map.of("10", task(10, regressed)), Map.of());
            assertEquals(1, observation.size(), "each live report must renew the exact request");
            assertEquals(reservation, observation.getFirst().reservation());
            assertTrue(
                    state.ownership(reservation).owned(),
                    "live snapshot membership must preserve ownership");
        }
        var resumed = calibrate(state, status, Map.of("10", task(10, TaskPhase.RUNNING)), Map.of());
        assertEquals(reservation, resumed.getFirst().reservation());
        assertEquals(1, state.routingView().engineCapacityUsed());
        var terminal =
                calibrate(state, status, Map.of(), Map.of("10", task(10, TaskPhase.RECEIVED)));
        assertEquals(reservation, terminal.getFirst().reservation());
        assertEquals(DecodeResources.DecodeRequestStatus.Kind.TERMINAL, terminal.getFirst().kind());
        assertFalse(state.ownership(reservation).owned());
        assertEquals(0, state.routingView().engineCapacityUsed());
    }

    @Test
    void expiryOfRegressedRequestDoesNotSubtractAnotherRunningRequestsSlot() {
        WorkerStatus status = status();
        DecodeEndpoint state = EndpointTestSupport.decodeEndpoint(status);
        var regressed =
                EndpointTestSupport.reserveQueuedDecode(state, 10, 100, 200, 50, CAPACITY).value();
        var running =
                EndpointTestSupport.reserveQueuedDecode(state, 11, 100, 200, 50, CAPACITY).value();
        calibrate(
                state,
                status,
                Map.of("10", task(10, TaskPhase.RUNNING), "11", task(11, TaskPhase.RUNNING)),
                Map.of());
        calibrate(
                state,
                status,
                Map.of("10", task(10, TaskPhase.RECEIVED), "11", task(11, TaskPhase.RUNNING)),
                Map.of());
        // These are local Engine ownership slots, not physical GPU running concurrency.
        assertEquals(2, state.routingView().engineCapacityUsed());
        assertEquals(RELEASED, state.release(regressed, ReleaseReason.EXPIRED).status());
        assertEquals(1, state.routingView().engineCapacityUsed());
        assertTrue(state.ownership(running).owned());
        assertFalse(state.ownership(regressed).owned());
    }

    @Test
    void regressedClaimKeepsOneHoldAndExplicitFinishedWinsOverActiveSnapshot() {
        WorkerStatus status = status();
        DecodeEndpoint state = EndpointTestSupport.decodeEndpoint(status);
        var victim =
                EndpointTestSupport.reserveQueuedDecode(state, 10, 100, 200, 50, CAPACITY).value();
        calibrate(state, status, Map.of("10", task(10, TaskPhase.RUNNING)), Map.of());
        assertEquals(
                DecodeResources.PreemptionBeginResult.SUCCESS,
                state.beginPreemption(
                        1,
                        java.util.List.of(victim),
                        String.valueOf(11),
                        100,
                        200,
                        80,
                        new AdmissionCapacity(1, 100)));
        assertTrue(EndpointTestSupport.handoffPreemption(state, 1));
        for (int i = 0; i < 2; i++) {
            calibrate(state, status, Map.of("10", task(10, TaskPhase.RECEIVED)), Map.of());
            assertEquals(
                    2,
                    state.routingView().engineCapacityUsed(),
                    "one victim hold plus one incoming shadow");
            assertEquals(
                    9_800L,
                    state.routingView().realKvAvailable(),
                    "hard KV hold must not be lost or duplicated");
        }
        var terminal =
                calibrate(
                        state,
                        status,
                        Map.of("10", task(10, TaskPhase.RECEIVED)),
                        Map.of("10", task(10, TaskPhase.RECEIVED)));
        assertEquals(1, terminal.size());
        assertEquals(DecodeResources.DecodeRequestStatus.Kind.TERMINAL, terminal.getFirst().kind());
        assertEquals(victim, terminal.getFirst().reservation());
        assertFalse(state.ownership(victim).owned());
        var incoming = state.finishPreemption(1, true);
        assertNotNull(incoming);
        assertTrue(state.ownership(incoming).withdrawable());
        assertEquals(0, state.routingView().engineCapacityUsed());
        assertEquals(9_900L, state.routingView().realKvAvailable());
        assertEquals(
                RELEASED,
                state.release(
                                EndpointTestSupport.decodeReservation(state, 11),
                                ReleaseReason.LOCAL_ROLLBACK)
                        .status());
        assertEquals(0, state.routingView().engineCapacityUsed());
    }

    @Test
    void topo002StatusRegressionAndTerminalRecoveryKeepGenerationOwnership() {
        WorkerStatus status = status();
        DecodeEndpoint state = EndpointTestSupport.decodeEndpoint(status);
        var reservation =
                EndpointTestSupport.reserveQueuedDecode(state, 77L, 100L, 200L, 50, CAPACITY)
                        .value();
        var permit = state.acquireDispatchPermit(reservation, CAPACITY).value();
        assertEquals(TRANSFERRED, state.dispatch(permit, DispatchOutcome.ENGINE_OWNED).status());

        calibrate(state, status, Map.of("77", task(77L, TaskPhase.RUNNING)), Map.of());
        assertEquals(1, state.routingView().totalLoad());
        calibrate(state, status, Map.of("77", task(77L, TaskPhase.RECEIVED)), Map.of());
        assertTrue(state.ownership(reservation).owned());
        assertEquals(1, state.routingView().totalLoad());

        var terminal =
                calibrate(state, status, Map.of(), Map.of("77", task(77L, TaskPhase.RUNNING)));
        assertEquals(DecodeRequestStatus.Kind.TERMINAL, terminal.getFirst().kind());
        assertFalse(state.ownership(reservation).owned());
        assertEquals(0, state.routingView().totalLoad());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void repeatedAllocationPreservesFirstKvAndExactIdentity(boolean locallyReserved) {
        WorkerStatus status = status();
        DecodeEndpoint state = EndpointTestSupport.decodeEndpoint(status);
        var reservation =
                locallyReserved
                        ? EndpointTestSupport.reserveQueuedDecode(state, 10, 200, 400, 50, CAPACITY)
                                .value()
                        : null;
        var first =
                calibrate(state, status, Map.of("10", task(10, TaskPhase.KV_ALLOCATED)), Map.of());
        TaskInfo running = task(10, TaskPhase.RUNNING);
        running.setInputLength(900L);
        var repeated = calibrate(state, status, Map.of("10", running), Map.of());
        var owner = state.resourceSnapshot().requests().get(String.valueOf(10L));
        assertEquals(100L, owner.kvTokens());
        assertEquals(100L, owner.expectedKvTokens());
        assertEquals(DecodeTaskPhase.RUNNING, owner.phase());
        assertEquals(0, state.resourceSnapshot().reservedCount());
        assertEquals(0, state.resourceSnapshot().queuedCount());
        assertEquals(1, state.routingView().engineCapacityUsed());
        if (locallyReserved) {
            assertEquals(reservation, first.getFirst().reservation());
            assertEquals(reservation, repeated.getFirst().reservation());
            assertTrue(state.ownership(reservation).confirmed());
        } else {
            assertEquals(0L, owner.reservationToken());
            assertTrue(first.isEmpty());
            assertTrue(repeated.isEmpty());
        }
    }

    @ParameterizedTest
    @EnumSource(
            value = TaskPhase.class,
            names = {"RECEIVED", "PENDING"})
    void allocationProofInDuplicateReportsClearsOnlyTheVictimsSyntheticKvHold(TaskPhase regressed) {
        WorkerStatus status = status();
        DecodeEndpoint state = EndpointTestSupport.decodeEndpoint(status);
        var victim =
                EndpointTestSupport.reserveQueuedDecode(state, 10, 100, 200, 50, CAPACITY).value();
        calibrate(state, status, Map.of("10", task(10, TaskPhase.RUNNING)), Map.of());
        assertEquals(
                DecodeResources.PreemptionBeginResult.SUCCESS,
                state.beginPreemption(
                        1,
                        java.util.List.of(victim),
                        String.valueOf(11),
                        100,
                        200,
                        80,
                        new AdmissionCapacity(1, 100)));
        assertTrue(EndpointTestSupport.handoffPreemption(state, 1));
        calibrate(state, status, Map.of("10", task(10, regressed)), Map.of());
        assertEquals(9_800L, state.routingView().realKvAvailable());
        var observed =
                calibrate(
                        state,
                        status,
                        Map.of(
                                "allocated",
                                task(10, TaskPhase.KV_ALLOCATED),
                                "regressed",
                                task(10, regressed)),
                        Map.of());
        assertEquals(2, observed.size());
        assertTrue(observed.stream().allMatch(event -> event.reservation().equals(victim)));
        assertEquals(2, state.routingView().engineCapacityUsed());
        assertEquals(
                9_900L,
                state.routingView().realKvAvailable(),
                "any allocation evidence in the full snapshot removes the synthetic hold");
        calibrate(state, status, Map.of(), Map.of("10", task(10, TaskPhase.RUNNING)));
        var incoming = state.finishPreemption(1, true);
        assertNotNull(incoming);
        assertEquals(RELEASED, state.release(incoming, ReleaseReason.LOCAL_ROLLBACK).status());
        assertEquals(0, state.routingView().totalLoad());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void orphanIncomingExpirySettlesItsAttemptWithoutReleasingOutboundVictims(boolean handedOff) {
        WorkerStatus status = status();
        DecodeEndpoint state = EndpointTestSupport.decodeEndpoint(status);
        var victim =
                EndpointTestSupport.reserveQueuedDecode(state, 10L, 100L, 200L, 50, CAPACITY)
                        .value();
        calibrate(state, status, Map.of("10", task(10L, TaskPhase.RUNNING)), Map.of());
        assertEquals(
                DecodeResources.PreemptionBeginResult.SUCCESS,
                state.beginPreemption(
                        1L,
                        List.of(victim),
                        String.valueOf(11L),
                        100L,
                        200L,
                        80,
                        new AdmissionCapacity(1L, 100L)));
        var incoming = EndpointTestSupport.decodeReservation(state, 11L);
        if (handedOff) {
            assertTrue(
                    state.updatePreemption(1L, DecodeResources.PreemptionUpdate.handedOff(victim))
                            .status()
                            .applied());
        }
        calibrate(state, status, Map.of(), Map.of());
        var candidate = state.cleanupCandidates().get("11");

        var cleanup = state.expireReservations(-1L, Map.of("11", candidate));
        assertEquals(1, cleanup.expiredReservations());
        assertFalse(state.ownership(incoming).owned());
        assertNull(state.finishPreemption(1L, true));
        assertNull(state.finishPreemption(1L, false));
        assertEquals(
                handedOff,
                state.resourceSnapshot()
                        .requests()
                        .get(String.valueOf(10L))
                        .claimedForPreemption());
        assertEquals(handedOff ? 100L : 0L, 10_000L - state.routingView().realKvAvailable());

        var replacement =
                EndpointTestSupport.reserveQueuedDecode(state, 11L, 70L, 90L, 80, CAPACITY).value();
        assertNotNull(replacement, "TTL cannot leave an incoming attempt blocking the request ID");
        assertEquals(
                handedOff,
                state.updatePreemption(1L, DecodeResources.PreemptionUpdate.canceled(victim))
                        .status()
                        .applied());
        assertTrue(
                state.ownership(replacement).withdrawable(),
                "old victim settlement cannot release the new incoming owner");
    }

    @Test
    void confirmedIncomingCannotCommitAndAbortPreservesEngineOwnership() {
        WorkerStatus status = status();
        DecodeEndpoint state = EndpointTestSupport.decodeEndpoint(status);
        var victim =
                EndpointTestSupport.reserveQueuedDecode(state, 10L, 100L, 200L, 50, CAPACITY)
                        .value();
        calibrate(state, status, Map.of("10", task(10L, TaskPhase.RUNNING)), Map.of());
        assertEquals(
                DecodeResources.PreemptionBeginResult.SUCCESS,
                state.beginPreemption(
                        1L,
                        List.of(victim),
                        String.valueOf(11L),
                        100L,
                        200L,
                        80,
                        new AdmissionCapacity(1L, 100L)));
        var incoming = EndpointTestSupport.decodeReservation(state, 11L);
        calibrate(
                state,
                status,
                Map.of("10", task(10L, TaskPhase.RUNNING), "11", task(11L, TaskPhase.RUNNING)),
                Map.of());
        assertTrue(
                state.updatePreemption(1L, DecodeResources.PreemptionUpdate.canceled(victim))
                        .status()
                        .applied());

        assertNull(state.finishPreemption(1L, true));
        assertEquals(incoming, state.finishPreemption(1L, false));
        assertTrue(state.ownership(incoming).confirmed());
        assertEquals(1, state.routingView().engineLoad());
        assertEquals(0, state.stats().inflight());
        assertThrows(
                IllegalStateException.class,
                () -> state.release(incoming, ReleaseReason.LOCAL_ROLLBACK).status());
        assertEquals(RELEASED, state.release(incoming, ReleaseReason.EXPIRED).status());
        assertEquals(0, state.routingView().totalLoad());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void confirmedIncomingFullAbsenceAbortsOnlyItsAttemptAndFencesOldMessages(boolean handedOff) {
        WorkerStatus status = status();
        DecodeEndpoint state = EndpointTestSupport.decodeEndpoint(status);
        var victim =
                EndpointTestSupport.reserveQueuedDecode(state, 10L, 100L, 200L, 50, CAPACITY)
                        .value();
        calibrate(state, status, Map.of("10", task(10L, TaskPhase.RUNNING)), Map.of());
        assertEquals(
                DecodeResources.PreemptionBeginResult.SUCCESS,
                state.beginPreemption(
                        1L,
                        List.of(victim),
                        String.valueOf(11L),
                        100L,
                        200L,
                        80,
                        new AdmissionCapacity(1L, 100L)));
        var incoming = EndpointTestSupport.decodeReservation(state, 11L);
        if (handedOff) {
            assertTrue(
                    state.updatePreemption(1L, DecodeResources.PreemptionUpdate.handedOff(victim))
                            .status()
                            .applied());
        }
        calibrate(
                state,
                status,
                Map.of("10", task(10L, TaskPhase.RUNNING), "11", task(11L, TaskPhase.RUNNING)),
                Map.of());
        assertTrue(state.ownership(incoming).confirmed());

        long version = state.placementVersion();
        calibrate(state, status, Map.of("10", task(10L, TaskPhase.RUNNING)), Map.of());
        assertEquals(version + 1L, state.placementVersion());
        assertFalse(state.ownership(incoming).owned());
        assertNull(
                state.finishPreemption(1L, true),
                "FULL absence must end the incoming admission attempt");
        assertNull(state.finishPreemption(1L, false));
        assertEquals(
                handedOff,
                state.resourceSnapshot()
                        .requests()
                        .get(String.valueOf(10L))
                        .claimedForPreemption());
        assertEquals(1, state.routingView().totalLoad());

        calibrate(state, status, Map.of(), Map.of());
        assertEquals(handedOff, state.ownership(victim).owned());
        assertEquals(handedOff ? 100L : 0L, 10_000L - state.routingView().realKvAvailable());
        // Expire history without touching any live owner, then admit a new exact token for the same
        // ID.
        state.expireReservations(-1L, Map.of());
        var replacement =
                EndpointTestSupport.reserveQueuedDecode(state, 11L, 70L, 90L, 80, CAPACITY).value();
        assertNotNull(replacement);
        assertTrue(replacement.reservationToken() != incoming.reservationToken());
        assertNull(state.finishPreemption(1L, true));
        assertNull(state.finishPreemption(1L, false));
        assertEquals(STALE, state.release(incoming, ReleaseReason.EXPIRED).status());
        assertEquals(
                handedOff,
                state.updatePreemption(1L, DecodeResources.PreemptionUpdate.canceled(victim))
                        .status()
                        .applied());
        assertTrue(state.ownership(replacement).withdrawable());
        assertEquals(1, state.routingView().totalLoad());
    }

    @Test
    void failedAttemptInstallationRollsBackWithoutCreatingATerminalHistoryFence() throws Exception {
        WorkerStatus status = status();
        DecodeEndpoint state = EndpointTestSupport.decodeEndpoint(status);
        var victim =
                EndpointTestSupport.reserveQueuedDecode(state, 10L, 100L, 200L, 50, CAPACITY)
                        .value();
        calibrate(state, status, Map.of("10", task(10L, TaskPhase.RUNNING)), Map.of());
        IllegalStateException failure = new IllegalStateException("attempt installation failed");
        var attempts = DecodeEndpoint.class.getDeclaredField("preemptionAttempts");
        attempts.setAccessible(true);
        attempts.set(
                state,
                new java.util.HashMap<Long, Object>() {
                    @Override
                    public Object put(Long token, Object attempt) {
                        throw failure;
                    }
                });

        assertSame(
                failure,
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                state.beginPreemption(
                                        1L,
                                        List.of(victim),
                                        String.valueOf(11L),
                                        100L,
                                        200L,
                                        80,
                                        new AdmissionCapacity(1L, 100L))));
        assertEquals(1, state.routingView().totalLoad());
        assertFalse(
                state.resourceSnapshot()
                        .requests()
                        .get(String.valueOf(10L))
                        .claimedForPreemption());
        assertNotNull(
                EndpointTestSupport.reserveQueuedDecode(state, 11L, 100L, 200L, 80, CAPACITY)
                        .value(),
                "an unpublished attempt failure must allow immediate retry of its incoming request ID");
    }

    @Test
    void handleConstructionFailureDoesNotPublishAnUnreachableReservation() {
        WorkerStatus status = org.mockito.Mockito.spy(status());
        long generation = status.getGenerationId();
        DecodeEndpoint state = EndpointTestSupport.decodeEndpoint(status);
        IllegalStateException failure = new IllegalStateException("generation unavailable");
        org.mockito.Mockito.doThrow(failure).when(status).getGenerationId();
        assertSame(
                failure,
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                EndpointTestSupport.reserveQueuedDecode(
                                                state, 1L, 100L, 200L, 50, null)
                                        .value()));
        org.mockito.Mockito.doReturn(generation).when(status).getGenerationId();

        assertEquals(0, state.stats().inflight());
        assertEquals(0, state.routingView().inflightHardKv());
        assertTrue(state.resourceSnapshot().requests().isEmpty());
        assertNotNull(
                EndpointTestSupport.reserveQueuedDecode(state, 1L, 100L, 200L, 50, CAPACITY)
                        .value());
    }

    @Test
    void fullObservationPublishesOneRevisionForMultipleOwnershipChanges() {
        WorkerStatus status = status();
        DecodeEndpoint state = EndpointTestSupport.decodeEndpoint(status);
        var first =
                EndpointTestSupport.reserveQueuedDecode(state, 1L, 100L, 200L, 50, CAPACITY)
                        .value();
        var second =
                EndpointTestSupport.reserveQueuedDecode(state, 2L, 70L, 90L, 50, CAPACITY).value();
        var previous = state.routingView();
        long revision = state.placementVersion();
        calibrate(
                state,
                status,
                Map.of("1", task(1L, TaskPhase.RUNNING), "2", task(2L, TaskPhase.KV_ALLOCATED)),
                Map.of());
        assertEquals(revision + 1L, state.placementVersion());
        assertTrue(state.ownership(first).confirmed());
        assertTrue(state.ownership(second).confirmed());
        assertEquals(0, state.stats().inflight());
        assertEquals(2, state.routingView().engineLoad());
        assertTrue(previous != state.routingView());

        revision = state.placementVersion();
        calibrate(
                state,
                status,
                Map.of(),
                Map.of("1", task(1L, TaskPhase.RUNNING), "2", task(2L, TaskPhase.RUNNING)));
        assertEquals(revision + 1L, state.placementVersion());
        assertEquals(0, state.routingView().totalLoad());
    }

    private static TaskInfo task(long requestId, TaskPhase phase) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(String.valueOf(requestId));
        task.setInputLength(100L);
        task.setPhase(phase);
        return task;
    }

    private static WorkerStatus status() {
        WorkerStatus status =
                EndpointTestSupport.workerStatus(RoleType.DECODE, "127.0.0.1", 8000, 8001);
        EndpointTestSupport.publishStatus(status, response());
        return status;
    }

    private static WorkerStatusResponse response() {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.DECODE);
        response.setAlive(true);
        response.setTotalKvCacheTokens(10_000L);
        response.setAvailableKvCacheTokens(10_000L);
        response.setRunningTaskInfo(Map.of());
        response.setFinishedTaskInfo(Map.of());
        return response;
    }

    private static List<DecodeRequestStatus> calibrate(
            DecodeEndpoint state,
            WorkerStatus status,
            Map<String, TaskInfo> running,
            Map<String, TaskInfo> finished) {
        WorkerStatusResponse response = response();
        response.setRunningTaskInfo(running);
        response.setFinishedTaskInfo(finished);
        response.setStatusVersion(status.appliedStatusCursor().statusVersion() + 1);
        response.setLatestFinishedVersion(
                status.appliedStatusCursor().latestFinishedTaskVersion() + 1);
        status.lock.lock();
        try {
            var prepared =
                    status.prepareNewStatus(
                            WorkerStatusTestSupport.freezeStatusResponse(status, response));
            return withStateLock(
                    state.stateLock(),
                    () -> {
                        var result = state.calibrateLocked(prepared.observation());
                        status.publishReducedStatus(prepared);
                        return result;
                    });
        } finally {
            status.lock.unlock();
        }
    }
}
