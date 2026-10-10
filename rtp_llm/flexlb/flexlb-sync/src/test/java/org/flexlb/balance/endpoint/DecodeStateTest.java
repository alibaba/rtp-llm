package org.flexlb.balance.endpoint;

import org.flexlb.dao.master.WorkerStatusTestSupport;

import org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRequestStatus;
import org.flexlb.balance.endpoint.DecodeResources.DispatchOutcome;
import org.flexlb.balance.endpoint.DecodeResources.ReleaseReason;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
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
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The resource ledger can run without an Endpoint, scheduler, RPC client or callback. */
class DecodeStateTest {
    private static final AdmissionCapacity CAPACITY = new AdmissionCapacity(2, 100);

    @Test
    void failedRetirementSnapshotDoesNotInvalidateTheLiveDispatchPermit() {
        WorkerStatus status = org.mockito.Mockito.spy(status());
        long generation = status.getGenerationId();
        DecodeState state = new DecodeState(status);
        var reservation = state.tryReserveQueuedRequest(1L, 100L, 200L, 50, CAPACITY).handle();
        var permit = state.acquireDispatchPermit(reservation, CAPACITY).handle();
        long version = state.placementVersion();
        IllegalStateException failure = new IllegalStateException("snapshot unavailable");
        org.mockito.Mockito.doThrow(failure).when(status).getGenerationId();
        assertSame(failure, assertThrows(IllegalStateException.class, state::retire));
        org.mockito.Mockito.doReturn(generation).when(status).getGenerationId();
        assertEquals(version, state.placementVersion());
        assertEquals(1, state.resourceSnapshot().activeDispatchPermits());
        assertTrue(state.dispatch(permit, DispatchOutcome.ABANDONED).capacityReleased(),
                "failed retirement must not make the live permit look already retired");
        assertEquals(0, state.resourceSnapshot().activeDispatchPermits());

        var replacement = state.acquireDispatchPermit(reservation, CAPACITY).handle();
        assertEquals(java.util.List.of(reservation), state.retire());
        assertFalse(state.ownership(reservation).owned());
        assertEquals(0, state.resourceSnapshot().activeDispatchPermits());
        assertEquals(TRANSFERRED, state.dispatch(replacement, DispatchOutcome.ABANDONED).status());
        assertTrue(state.retire().isEmpty());
    }

    @Test
    void returnedPermitCannotDispatchOrReleaseItsReplacement() {
        WorkerStatus status = status();
        DecodeState state = new DecodeState(status);
        var reservation = state.tryReserveQueuedRequest(1, 100, 200, 50, CAPACITY).handle();
        var first = state.acquireDispatchPermit(reservation, CAPACITY).handle();
        assertTrue(state.dispatch(first, DispatchOutcome.ABANDONED).capacityReleased());
        assertTrue(EndpointTestSupport.isQueued(state.resourceSnapshot(), 1));

        var replacement = state.acquireDispatchPermit(reservation, CAPACITY).handle();
        assertEquals(OWNERSHIP_LOST, state.dispatch(first, DispatchOutcome.ENGINE_OWNED).status());
        assertEquals(OWNERSHIP_LOST, state.dispatch(first, DispatchOutcome.ABANDONED).status());
        assertEquals(1, state.resourceSnapshot().activeDispatchPermits());
        assertEquals(TRANSFERRED, state.dispatch(replacement, DispatchOutcome.ENGINE_OWNED).status());
        assertEquals(0, state.resourceSnapshot().activeDispatchPermits());
        assertEquals(1, state.routingView("decode").engineLoad());
        assertThrows(IllegalStateException.class, () -> state.release(reservation, ReleaseReason.LOCAL_ROLLBACK).status());
        assertEquals(STILL_OWNED, state.release(reservation, ReleaseReason.COUNTERPART_FINISHED).status());
        assertEquals(RELEASED, state.release(reservation, ReleaseReason.EXPIRED).status());
        assertFalse(state.ownership(reservation).owned());
        assertEquals(0, state.routingView("decode").engineLoad());
    }

    @ParameterizedTest
    @CsvSource({
            "MASTER_QUEUED_NOT_DISPATCHED, LOCAL_ROLLBACK",
            "MASTER_QUEUED_NOT_DISPATCHED, COUNTERPART_FINISHED",
            "MASTER_QUEUED_NOT_DISPATCHED, NOT_SENT",
            "ENGINE_MAY_HAVE_SEEN, LOCAL_ROLLBACK",
            "ENGINE_MAY_HAVE_SEEN, COUNTERPART_FINISHED",
            "ENGINE_MAY_HAVE_SEEN, NOT_SENT",
            "ACCEPTED_NOT_RUNNING, LOCAL_ROLLBACK",
            "ACCEPTED_NOT_RUNNING, COUNTERPART_FINISHED",
            "ACCEPTED_NOT_RUNNING, NOT_SENT",
            "RUNNING, LOCAL_ROLLBACK",
            "RUNNING, COUNTERPART_FINISHED",
            "RUNNING, NOT_SENT"
    })
    void releaseReasonPreservesEngineOwnershipAndSettlesOnlyItsExactReservation(
            DecodeTaskPhase phase, ReleaseReason reason) {
        WorkerStatus status = status();
        DecodeState state = new DecodeState(status);
        var reservation = state.tryReserveQueuedRequest(1, 100, 200, 50, CAPACITY).handle();
        var permit = state.acquireDispatchPermit(reservation, CAPACITY).handle();
        if (phase == DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN) {
            state.dispatch(permit, DispatchOutcome.ENGINE_OWNED).status();
        } else if (phase.isEngineConfirmed()) {
            calibrate(state, status, Map.of("1", task(1,
                    phase == DecodeTaskPhase.RUNNING ? TaskPhase.RUNNING : TaskPhase.KV_ALLOCATED)), Map.of());
        }
        var sibling = state.tryReserveQueuedRequest(2, 70, 90, 30, CAPACITY).handle();
        long version = state.placementVersion();
        var stale = new DecodeResources.ReservationHandle(
                reservation.endpointGenerationId(), 1, reservation.reservationToken() + 100);
        var foreign = new DecodeResources.ReservationHandle(
                reservation.endpointGenerationId() + 1, 1, reservation.reservationToken());
        assertEquals(STALE, state.release(stale, reason).status());
        assertEquals(STALE, state.release(foreign, reason).status());
        assertEquals(version, state.placementVersion());

        boolean released = phase == DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED
                || phase == DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN && reason == ReleaseReason.NOT_SENT;
        if (released) {
            assertEquals(RELEASED, state.release(reservation, reason).status());
            assertEquals(STALE, state.release(reservation, reason).status());
            assertFalse(state.ownership(reservation).owned());
            assertEquals(70L, state.routingView("decode").inflightHardKv());
            assertEquals(90L, EndpointTestSupport.expectedReservedKv(state.resourceSnapshot()));
            assertEquals(0, state.resourceSnapshot().activeDispatchPermits());
            assertEquals(1, state.routingView("decode").totalLoad());
            assertEquals(OWNERSHIP_LOST, state.dispatch(permit, DispatchOutcome.ABANDONED).status());
        } else {
            if (reason == ReleaseReason.LOCAL_ROLLBACK) {
                assertThrows(IllegalStateException.class, () -> state.release(reservation, reason).status());
            } else {
                assertEquals(reason == ReleaseReason.NOT_SENT ? ENGINE_ACCEPTED : STILL_OWNED,
                        state.release(reservation, reason).status());
            }
            assertTrue(state.ownership(reservation).owned());
            assertEquals(version, state.placementVersion());
            assertEquals(2, state.routingView("decode").totalLoad());
        }
        assertTrue(state.ownership(sibling).owned());
        assertEquals(1, state.resourceSnapshot().queuedCount());
    }

    @Test
    void calibrationConvertsTheSameReservationWithoutDoubleChargingAndSettlesTerminal() {
        WorkerStatus status = status();
        DecodeState state = new DecodeState(status);
        var reservation = state.tryReserveQueuedRequest(2, 100, 200, 50, CAPACITY).handle();
        var permit = state.acquireDispatchPermit(reservation, CAPACITY).handle();
        TaskInfo task = new TaskInfo();
        task.setRequestId(2L);
        task.setPhase(TaskPhase.KV_ALLOCATED);
        var accepted = calibrate(state, status, Map.of("2", task), Map.of());
        assertEquals(reservation, accepted.getFirst().reservation());
        assertTrue(state.ownership(reservation).confirmed());
        assertEquals(1, state.routingView("decode").engineCapacityUsed());
        assertEquals(0, state.resourceSnapshot().activeDispatchPermits());
        assertTrue(state.resourceSnapshot().reservedCount() == 0);
        assertEquals(TRANSFERRED, state.dispatch(permit, DispatchOutcome.ENGINE_OWNED).status());

        var finished = calibrate(state, status, Map.of(), Map.of("2", task));
        assertEquals(DecodeResources.DecodeRequestStatus.Kind.TERMINAL, finished.getFirst().kind());
        assertEquals(reservation, finished.getFirst().reservation());
        assertFalse(state.ownership(reservation).owned());
        assertEquals(0, state.routingView("decode").engineCapacityUsed());
        assertEquals(OWNERSHIP_LOST, state.dispatch(permit, DispatchOutcome.ENGINE_OWNED).status());
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
        DecodeState state = new DecodeState(status);
        var reservation = state.tryReserveQueuedRequest(1, 100, 200, 50, CAPACITY).handle();
        var permit = state.acquireDispatchPermit(reservation, CAPACITY).handle();
        if (phase == DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN) {
            state.dispatch(permit, DispatchOutcome.ENGINE_OWNED).status();
        } else if (phase.isEngineConfirmed()) {
            calibrate(state, status, Map.of("1", task(1,
                    phase == DecodeTaskPhase.RUNNING ? TaskPhase.RUNNING : TaskPhase.KV_ALLOCATED)), Map.of());
        }
        var sibling = state.tryReserveQueuedRequest(2, 70, 90, 30, CAPACITY).handle();
        assertNotNull(sibling);
        var stale = new DecodeResources.ReservationHandle(
                reservation.endpointGenerationId(), 1, reservation.reservationToken() + 100);
        assertEquals(STALE, state.release(stale, ReleaseReason.EXPIRED).status());
        assertTrue(state.ownership(reservation).owned());

        for (int i = 0; i < 2; i++) {
            if (expired) {
                assertEquals(i == 0 ? RELEASED : STALE, state.release(reservation, ReleaseReason.EXPIRED).status());
            } else {
                calibrate(state, status, Map.of(), Map.of("1", task(1, TaskPhase.RUNNING)));
            }
            assertFalse(state.ownership(reservation).owned());
            assertTrue(state.ownership(sibling).owned());
            assertEquals(0, state.resourceSnapshot().activeDispatchPermits());
            assertEquals(1, state.resourceSnapshot().queuedCount());
            assertEquals(70L, state.routingView("decode").inflightHardKv());
            assertEquals(90L, EndpointTestSupport.expectedReservedKv(state.resourceSnapshot()));
            assertEquals(1, state.routingView("decode").totalLoad());
            assertEquals(OWNERSHIP_LOST, state.dispatch(permit, DispatchOutcome.ABANDONED).status());
        }
        assertEquals(RELEASED, state.release(sibling, ReleaseReason.LOCAL_ROLLBACK).status());
        assertEquals(0, state.routingView("decode").totalLoad());
        assertEquals(0, state.routingView("decode").inflightHardKv());
        assertEquals(0, EndpointTestSupport.expectedReservedKv(state.resourceSnapshot()));
    }

    @ParameterizedTest
    @ValueSource(strings = {"EXPIRED", "COUNTERPART_FINISHED", "NOT_SENT", "FULL_ABSENCE"})
    void terminalHistoryFencesLateReportsUntilRetentionExpires(String settlement) {
        WorkerStatus status = status();
        DecodeState state = new DecodeState(status);
        var reservation = state.tryReserveQueuedRequest(1, 100, 200, 50, CAPACITY).handle();
        if (settlement.equals("FULL_ABSENCE")) {
            calibrate(state, status, Map.of("1", task(1, TaskPhase.RUNNING)), Map.of());
            calibrate(state, status, Map.of(), Map.of());
        } else {
            assertEquals(RELEASED, state.release(reservation, ReleaseReason.valueOf(settlement)).status());
        }
        assertFalse(state.ownership(reservation).owned());
        assertEquals(STALE, state.release(reservation, ReleaseReason.EXPIRED).status());
        assertFalse(EndpointTestSupport.evictDecode(state, Long.MAX_VALUE, ignored -> false).capacityReleased());
        calibrate(state, status, Map.of(), Map.of("1", task(1, TaskPhase.RUNNING)));
        var late = calibrate(state, status, Map.of("1", task(1, TaskPhase.RUNNING)), Map.of());
        assertTrue(late.isEmpty());
        assertEquals(0, state.routingView("decode").totalLoad());

        assertTrue(EndpointTestSupport.evictDecode(state, -1L, ignored -> false).capacityReleased());
        calibrate(state, status, Map.of("1", task(1, TaskPhase.RUNNING)), Map.of());
        assertEquals(1, state.routingView("decode").totalLoad());
        assertFalse(state.ownership(reservation).owned(), "expired history must not restore the old token");
    }

    @Test
    void initializationRejectsAnotherGenerationBeforeChangingResources() {
        WorkerStatus owner = status();
        DecodeState state = new DecodeState(owner);
        var incoming = response();
        incoming.setRunningTaskInfo(Map.of("1", task(1, TaskPhase.RUNNING)));
        long version = state.placementVersion();
        assertThrows(IllegalArgumentException.class, () -> state.initialize(WorkerStatusTestSupport.freezeStatusResponse(status(), incoming)));
        assertEquals(version, state.placementVersion());
        assertEquals(0, state.routingView("decode").totalLoad());
    }

    @Test
    void shadowSweepRechecksPhaseAfterRetentionCallbackConfirmsTheRequest() {
        WorkerStatus status = status();
        DecodeState state = new DecodeState(status);
        var reservation = state.tryReserveQueuedRequest(1, 100, 200, 50, CAPACITY).handle();
        state.acquireDispatchPermit(reservation, CAPACITY);

        var cleanup = EndpointTestSupport.evictDecode(state, -1L, requestId -> {
            calibrate(state, status, Map.of("1", task(1, TaskPhase.RUNNING)), Map.of());
            return false;
        });

        assertEquals(0, cleanup.expiredReservations());
        assertFalse(cleanup.capacityReleased());
        assertTrue(state.ownership(reservation).confirmed());
        assertTrue(state.ownership(reservation).owned());
        assertEquals(1, state.routingView("decode").engineCapacityUsed());
        assertEquals(0, state.resourceSnapshot().activeDispatchPermits());
        assertEquals(0, state.resourceSnapshot().queuedCount());
        assertEquals(0, state.routingView("decode").inflightHardKv());
    }

    @Test
    void endpointRejectsAnotherEndpointsPermitWithoutConsumingIt() {
        DecodeEndpoint owner = EndpointTestSupport.decode(status(), org.flexlb.balance.scheduler.SchedulerTestSupport.repository(EndpointTestSupport.noopEventSink()));
        DecodeEndpoint other = EndpointTestSupport.decode(status(), org.flexlb.balance.scheduler.SchedulerTestSupport.repository(EndpointTestSupport.noopEventSink()));
        try {
            DecodeResources.ReservationHandle reservation;
            try (var pin = owner.tryPinGeneration()) {
                reservation = owner.tryReserveQueuedRequest(pin, 3, 100, 200, 50, null).handle();
            }
            var permit = owner.acquireDispatchPermit(reservation, CAPACITY).handle();
            assertThrows(IllegalArgumentException.class, () -> other.dispatch(permit, DispatchOutcome.ENGINE_OWNED).status());
            assertEquals(1, owner.resourceSnapshot().activeDispatchPermits());
            assertEquals(TRANSFERRED, owner.dispatch(permit, DispatchOutcome.ENGINE_OWNED).status());
            assertEquals(TRANSFERRED, owner.dispatch(permit, DispatchOutcome.ENGINE_OWNED).status());
        } finally {
            owner.close();
            other.close();
        }
    }

    @ParameterizedTest
    @EnumSource(value = TaskPhase.class, names = {"RECEIVED", "PENDING"})
    void regressedActivePhaseRetainsIdentityAcrossRepeatedReportsAndResumption(TaskPhase regressed) {
        WorkerStatus status = status();
        DecodeState state = new DecodeState(status);
        var reservation = state.tryReserveQueuedRequest(10, 100, 200, 50, CAPACITY).handle();
        calibrate(state, status, Map.of("10", task(10, TaskPhase.RUNNING)), Map.of());
        for (int i = 0; i < 2; i++) {
            var observation = calibrate(state, status, Map.of("10", task(10, regressed)), Map.of());
            assertEquals(1, observation.size(), "each live report must renew the exact request");
            assertEquals(reservation, observation.getFirst().reservation());
            assertTrue(state.ownership(reservation).owned(), "live snapshot membership must preserve ownership");
        }
        var resumed = calibrate(state, status, Map.of("10", task(10, TaskPhase.RUNNING)), Map.of());
        assertEquals(reservation, resumed.getFirst().reservation());
        assertEquals(1, state.routingView("decode").engineCapacityUsed());
        var terminal = calibrate(state, status, Map.of(), Map.of("10", task(10, TaskPhase.RECEIVED)));
        assertEquals(reservation, terminal.getFirst().reservation());
        assertEquals(DecodeResources.DecodeRequestStatus.Kind.TERMINAL, terminal.getFirst().kind());
        assertFalse(state.ownership(reservation).owned());
        assertEquals(0, state.routingView("decode").engineCapacityUsed());
    }

    @Test
    void expiryOfRegressedRequestDoesNotSubtractAnotherRunningRequestsSlot() {
        WorkerStatus status = status();
        DecodeState state = new DecodeState(status);
        var regressed = state.tryReserveQueuedRequest(10, 100, 200, 50, CAPACITY).handle();
        var running = state.tryReserveQueuedRequest(11, 100, 200, 50, CAPACITY).handle();
        calibrate(state, status, Map.of("10", task(10, TaskPhase.RUNNING), "11", task(11, TaskPhase.RUNNING)), Map.of());
        calibrate(state, status, Map.of("10", task(10, TaskPhase.RECEIVED), "11", task(11, TaskPhase.RUNNING)), Map.of());
        // These are local Engine ownership slots, not physical GPU running concurrency.
        assertEquals(2, state.routingView("decode").engineCapacityUsed());
        assertEquals(RELEASED, state.release(regressed, ReleaseReason.EXPIRED).status());
        assertEquals(1, state.routingView("decode").engineCapacityUsed());
        assertTrue(state.ownership(running).owned());
        assertFalse(state.ownership(regressed).owned());
    }

    @Test
    void regressedClaimKeepsOneHoldAndExplicitFinishedWinsOverActiveSnapshot() {
        WorkerStatus status = status();
        DecodeState state = new DecodeState(status);
        var victim = state.tryReserveQueuedRequest(10, 100, 200, 50, CAPACITY).handle();
        calibrate(state, status, Map.of("10", task(10, TaskPhase.RUNNING)), Map.of());
        assertEquals(DecodeResources.PreemptionBeginResult.SUCCESS, state.beginPreemption(
                1, java.util.List.of(victim), 11, 100, 200, 80, new AdmissionCapacity(1, 100)));
        assertTrue(EndpointTestSupport.handoffPreemption(state, 1));
        for (int i = 0; i < 2; i++) {
            calibrate(state, status, Map.of("10", task(10, TaskPhase.RECEIVED)), Map.of());
            assertEquals(2, state.routingView("decode").engineCapacityUsed(), "one victim hold plus one incoming shadow");
            assertEquals(9_800L, state.routingView("decode").realKvAvailable(), "hard KV hold must not be lost or duplicated");
        }
        var terminal = calibrate(state, status, Map.of("10", task(10, TaskPhase.RECEIVED)),
                Map.of("10", task(10, TaskPhase.RECEIVED)));
        assertEquals(1, terminal.size());
        assertEquals(DecodeResources.DecodeRequestStatus.Kind.TERMINAL, terminal.getFirst().kind());
        assertEquals(victim, terminal.getFirst().reservation());
        assertFalse(state.ownership(victim).owned());
        assertNotNull(state.finishPreemption(1, true));
        assertEquals(1, state.routingView("decode").engineCapacityUsed());
        assertEquals(9_900L, state.routingView("decode").realKvAvailable());
        assertEquals(RELEASED, state.release(EndpointTestSupport.decodeReservation(state, 11), ReleaseReason.LOCAL_ROLLBACK).status());
        assertEquals(0, state.routingView("decode").engineCapacityUsed());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void repeatedAllocationPreservesFirstKvAndExactIdentity(boolean locallyReserved) {
        WorkerStatus status = status();
        DecodeState state = new DecodeState(status);
        var reservation = locallyReserved
                ? state.tryReserveQueuedRequest(10, 200, 400, 50, CAPACITY).handle() : null;
        var first = calibrate(state, status, Map.of("10", task(10, TaskPhase.KV_ALLOCATED)), Map.of());
        TaskInfo running = task(10, TaskPhase.RUNNING);
        running.setInputLength(900L);
        var repeated = calibrate(state, status, Map.of("10", running), Map.of());
        var owner = state.resourceSnapshot().requests().get(10L);
        assertEquals(100L, owner.kvTokens());
        assertEquals(100L, owner.expectedKvTokens());
        assertEquals(DecodeTaskPhase.RUNNING, owner.phase());
        assertEquals(0, state.resourceSnapshot().reservedCount());
        assertEquals(0, state.resourceSnapshot().queuedCount());
        assertEquals(1, state.routingView("decode").engineCapacityUsed());
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
    @EnumSource(value = TaskPhase.class, names = {"RECEIVED", "PENDING"})
    void allocationProofInDuplicateReportsClearsOnlyTheVictimsSyntheticKvHold(TaskPhase regressed) {
        WorkerStatus status = status();
        DecodeState state = new DecodeState(status);
        var victim = state.tryReserveQueuedRequest(10, 100, 200, 50, CAPACITY).handle();
        calibrate(state, status, Map.of("10", task(10, TaskPhase.RUNNING)), Map.of());
        assertEquals(DecodeResources.PreemptionBeginResult.SUCCESS, state.beginPreemption(
                1, java.util.List.of(victim), 11, 100, 200, 80, new AdmissionCapacity(1, 100)));
        assertTrue(EndpointTestSupport.handoffPreemption(state, 1));
        calibrate(state, status, Map.of("10", task(10, regressed)), Map.of());
        assertEquals(9_800L, state.routingView("decode").realKvAvailable());
        var observed = calibrate(state, status, Map.of("allocated", task(10, TaskPhase.KV_ALLOCATED),
                "regressed", task(10, regressed)), Map.of());
        assertEquals(2, observed.size());
        assertTrue(observed.stream().allMatch(event -> event.reservation().equals(victim)));
        assertEquals(2, state.routingView("decode").engineCapacityUsed());
        assertEquals(9_900L, state.routingView("decode").realKvAvailable(),
                "any allocation evidence in the full snapshot removes the synthetic hold");
        calibrate(state, status, Map.of(), Map.of("10", task(10, TaskPhase.RUNNING)));
        var incoming = state.finishPreemption(1, true);
        assertNotNull(incoming);
        assertEquals(RELEASED, state.release(incoming, ReleaseReason.LOCAL_ROLLBACK).status());
        assertEquals(0, state.routingView("decode").totalLoad());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void orphanIncomingExpirySettlesItsAttemptWithoutReleasingOutboundVictims(boolean handedOff) {
        WorkerStatus status = status();
        DecodeState state = new DecodeState(status);
        var victim = state.tryReserveQueuedRequest(10L, 100L, 200L, 50, CAPACITY).handle();
        calibrate(state, status, Map.of("10", task(10L, TaskPhase.RUNNING)), Map.of());
        assertEquals(DecodeResources.PreemptionBeginResult.SUCCESS, state.beginPreemption(
                1L, List.of(victim), 11L, 100L, 200L, 80, new AdmissionCapacity(1L, 100L)));
        var incoming = EndpointTestSupport.decodeReservation(state, 11L);
        if (handedOff) { assertTrue(state.updatePreemption(1L, DecodeResources.PreemptionUpdate.handedOff(victim)).status().applied()); }
        calibrate(state, status, Map.of(), Map.of());
        var candidate = state.cleanupCandidates().get(11L);

        var cleanup = state.evictExpiredRequests(-1L, Map.of(11L, candidate));
        assertEquals(1, cleanup.expiredReservations());
        assertFalse(state.ownership(incoming).owned());
        assertNull(state.finishPreemption(1L, true));
        assertNull(state.finishPreemption(1L, false));
        assertEquals(handedOff, state.resourceSnapshot().requests().get(10L).claimedForPreemption());
        assertEquals(handedOff ? 100L : 0L, 10_000L - state.routingView("decode").realKvAvailable());

        var replacement = state.tryReserveQueuedRequest(11L, 70L, 90L, 80, CAPACITY).handle();
        assertNotNull(replacement, "TTL cannot leave an incoming attempt blocking the request ID");
        assertEquals(handedOff, state.updatePreemption(1L, DecodeResources.PreemptionUpdate.canceled(victim)).status().applied());
        assertTrue(state.ownership(replacement).withdrawable(), "old victim settlement cannot release the new incoming owner");
    }

    @Test
    void confirmedIncomingCannotCommitAndAbortPreservesEngineOwnership() {
        WorkerStatus status = status();
        DecodeState state = new DecodeState(status);
        var victim = state.tryReserveQueuedRequest(10L, 100L, 200L, 50, CAPACITY).handle();
        calibrate(state, status, Map.of("10", task(10L, TaskPhase.RUNNING)), Map.of());
        assertEquals(DecodeResources.PreemptionBeginResult.SUCCESS, state.beginPreemption(
                1L, List.of(victim), 11L, 100L, 200L, 80, new AdmissionCapacity(1L, 100L)));
        var incoming = EndpointTestSupport.decodeReservation(state, 11L);
        calibrate(state, status, Map.of("10", task(10L, TaskPhase.RUNNING),
                "11", task(11L, TaskPhase.RUNNING)), Map.of());
        assertTrue(state.updatePreemption(1L, DecodeResources.PreemptionUpdate.canceled(victim)).status().applied());

        assertNull(state.finishPreemption(1L, true));
        assertEquals(incoming, state.finishPreemption(1L, false));
        assertTrue(state.ownership(incoming).confirmed());
        assertEquals(1, state.routingView("decode").engineLoad());
        assertEquals(0, state.stats().inflight());
        assertThrows(IllegalStateException.class, () -> state.release(incoming, ReleaseReason.LOCAL_ROLLBACK).status());
        assertEquals(RELEASED, state.release(incoming, ReleaseReason.EXPIRED).status());
        assertEquals(0, state.routingView("decode").totalLoad());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void confirmedIncomingFullAbsenceAbortsOnlyItsAttemptAndFencesOldMessages(boolean handedOff) {
        WorkerStatus status = status();
        DecodeState state = new DecodeState(status);
        var victim = state.tryReserveQueuedRequest(10L, 100L, 200L, 50, CAPACITY).handle();
        calibrate(state, status, Map.of("10", task(10L, TaskPhase.RUNNING)), Map.of());
        assertEquals(DecodeResources.PreemptionBeginResult.SUCCESS, state.beginPreemption(
                1L, List.of(victim), 11L, 100L, 200L, 80, new AdmissionCapacity(1L, 100L)));
        var incoming = EndpointTestSupport.decodeReservation(state, 11L);
        if (handedOff) { assertTrue(state.updatePreemption(1L, DecodeResources.PreemptionUpdate.handedOff(victim)).status().applied()); }
        calibrate(state, status, Map.of("10", task(10L, TaskPhase.RUNNING),
                "11", task(11L, TaskPhase.RUNNING)), Map.of());
        assertTrue(state.ownership(incoming).confirmed());

        long version = state.placementVersion();
        calibrate(state, status, Map.of("10", task(10L, TaskPhase.RUNNING)), Map.of());
        assertEquals(version + 1L, state.placementVersion());
        assertFalse(state.ownership(incoming).owned());
        assertNull(state.finishPreemption(1L, true), "FULL absence must end the incoming admission attempt");
        assertNull(state.finishPreemption(1L, false));
        assertEquals(handedOff, state.resourceSnapshot().requests().get(10L).claimedForPreemption());
        assertEquals(1, state.routingView("decode").totalLoad());

        calibrate(state, status, Map.of(), Map.of());
        assertEquals(handedOff, state.ownership(victim).owned());
        assertEquals(handedOff ? 100L : 0L, 10_000L - state.routingView("decode").realKvAvailable());
        // Expire history without touching any live owner, then admit a new exact token for the same ID.
        state.evictExpiredRequests(-1L, Map.of());
        var replacement = state.tryReserveQueuedRequest(11L, 70L, 90L, 80, CAPACITY).handle();
        assertNotNull(replacement);
        assertTrue(replacement.reservationToken() != incoming.reservationToken());
        assertNull(state.finishPreemption(1L, true));
        assertNull(state.finishPreemption(1L, false));
        assertEquals(STALE, state.release(incoming, ReleaseReason.EXPIRED).status());
        assertEquals(handedOff, state.updatePreemption(1L, DecodeResources.PreemptionUpdate.canceled(victim)).status().applied());
        assertTrue(state.ownership(replacement).withdrawable());
        assertEquals(1, state.routingView("decode").totalLoad());
    }

    @Test
    void failedAttemptInstallationRollsBackWithoutCreatingATerminalHistoryFence() throws Exception {
        WorkerStatus status = status();
        DecodeState state = new DecodeState(status);
        var victim = state.tryReserveQueuedRequest(10L, 100L, 200L, 50, CAPACITY).handle();
        calibrate(state, status, Map.of("10", task(10L, TaskPhase.RUNNING)), Map.of());
        IllegalStateException failure = new IllegalStateException("attempt installation failed");
        var attempts = DecodeState.class.getDeclaredField("preemptionAttempts");
        attempts.setAccessible(true);
        attempts.set(state, new java.util.HashMap<Long, Object>() {
            @Override public Object put(Long token, Object attempt) { throw failure; }
        });

        assertSame(failure, assertThrows(IllegalStateException.class, () -> state.beginPreemption(
                1L, List.of(victim), 11L, 100L, 200L, 80, new AdmissionCapacity(1L, 100L))));
        assertEquals(1, state.routingView("decode").totalLoad());
        assertFalse(state.resourceSnapshot().requests().get(10L).claimedForPreemption());
        assertNotNull(state.tryReserveQueuedRequest(11L, 100L, 200L, 80, CAPACITY).handle(),
                "an unpublished attempt failure must allow immediate retry of its incoming request ID");
    }

    @Test
    void handleConstructionFailureDoesNotPublishAnUnreachableReservation() {
        WorkerStatus status = org.mockito.Mockito.spy(status());
        long generation = status.getGenerationId();
        DecodeState state = new DecodeState(status);
        IllegalStateException failure = new IllegalStateException("generation unavailable");
        org.mockito.Mockito.doThrow(failure).when(status).getGenerationId();
        assertSame(failure, assertThrows(IllegalStateException.class,
                () -> state.tryReserveQueuedRequest(1L, 100L, 200L, 50, null).handle()));
        org.mockito.Mockito.doReturn(generation).when(status).getGenerationId();

        assertEquals(0, state.stats().inflight());
        assertEquals(0, state.routingView("decode").inflightHardKv());
        assertTrue(state.resourceSnapshot().requests().isEmpty());
        assertNotNull(state.tryReserveQueuedRequest(1L, 100L, 200L, 50, CAPACITY).handle());
    }

    @Test
    void fullObservationPublishesOneRevisionForMultipleOwnershipChanges() {
        WorkerStatus status = status();
        DecodeState state = new DecodeState(status);
        var first = state.tryReserveQueuedRequest(1L, 100L, 200L, 50, CAPACITY).handle();
        var second = state.tryReserveQueuedRequest(2L, 70L, 90L, 50, CAPACITY).handle();
        var previous = state.routingView("decode");
        long revision = state.placementVersion();
        calibrate(state, status, Map.of("1", task(1L, TaskPhase.RUNNING),
                "2", task(2L, TaskPhase.KV_ALLOCATED)), Map.of());
        assertEquals(revision + 1L, state.placementVersion());
        assertTrue(state.ownership(first).confirmed());
        assertTrue(state.ownership(second).confirmed());
        assertEquals(0, state.stats().inflight());
        assertEquals(2, state.routingView("decode").engineLoad());
        assertTrue(previous != state.routingView("decode"));

        revision = state.placementVersion();
        calibrate(state, status, Map.of(), Map.of("1", task(1L, TaskPhase.RUNNING),
                "2", task(2L, TaskPhase.RUNNING)));
        assertEquals(revision + 1L, state.placementVersion());
        assertEquals(0, state.routingView("decode").totalLoad());
    }

    private static TaskInfo task(long requestId, TaskPhase phase) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setInputLength(100L);
        task.setPhase(phase);
        return task;
    }

    private static WorkerStatus status() {
        WorkerStatus status = EndpointTestSupport.workerStatus(RoleType.DECODE, "127.0.0.1", 8000, 8001);
        EndpointTestSupport.publishStatus(status, response());
        return status;
    }

    private static List<DecodeRequestStatus> calibrate(DecodeState state, WorkerStatus status,
                                                           Map<String, TaskInfo> running,
                                                           Map<String, TaskInfo> finished) {
        WorkerStatusResponse response = response();
        response.setRunningTaskInfo(running);
        response.setFinishedTaskInfo(finished);
        response.setStatusVersion(status.appliedStatusCursor().statusVersion() + 1);
        response.setLatestFinishedVersion(status.appliedStatusCursor().latestFinishedTaskVersion() + 1);
        status.lock.lock();
        try {
            var prepared = status.prepareNewStatus(WorkerStatusTestSupport.freezeStatusResponse(status, response));
            state.ownershipLock().lock();
            try {
                var result = state.calibrateLocked(prepared.observation());
                status.publishPreparedStatus(prepared);
                return result;
            } finally { state.ownershipLock().unlock(); }
        } finally {
            status.lock.unlock();
        }
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
}
