package org.flexlb.balance.endpoint;

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
        var reservation = state.tryReserveQueuedRequest(1L, 100L, 200L, 50, CAPACITY);
        var permit = state.acquireDispatchPermit(reservation, CAPACITY).permit();
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

        var replacement = state.acquireDispatchPermit(reservation, CAPACITY).permit();
        assertEquals(java.util.List.of(reservation), state.retire());
        assertFalse(state.hasOwnedResources(reservation));
        assertEquals(0, state.resourceSnapshot().activeDispatchPermits());
        assertEquals(TRANSFERRED, state.dispatch(replacement, DispatchOutcome.ABANDONED).status());
        assertTrue(state.retire().isEmpty());
    }

    @Test
    void returnedPermitCannotDispatchOrReleaseItsReplacement() {
        WorkerStatus status = status();
        DecodeState state = new DecodeState(status);
        var reservation = state.tryReserveQueuedRequest(1, 100, 200, 50, CAPACITY);
        var first = state.acquireDispatchPermit(reservation, CAPACITY).permit();
        assertTrue(state.dispatch(first, DispatchOutcome.ABANDONED).capacityReleased());
        assertTrue(EndpointTestSupport.isQueued(state.resourceSnapshot(), 1));

        var replacement = state.acquireDispatchPermit(reservation, CAPACITY).permit();
        assertEquals(OWNERSHIP_LOST, state.dispatch(first, DispatchOutcome.ENGINE_OWNED).status());
        assertEquals(OWNERSHIP_LOST, state.dispatch(first, DispatchOutcome.ABANDONED).status());
        assertEquals(1, state.resourceSnapshot().activeDispatchPermits());
        assertEquals(TRANSFERRED, state.dispatch(replacement, DispatchOutcome.ENGINE_OWNED).status());
        assertEquals(0, state.resourceSnapshot().activeDispatchPermits());
        assertEquals(1, state.routingView().engineLoad());
        assertThrows(IllegalStateException.class, () -> state.release(reservation, ReleaseReason.LOCAL_ROLLBACK));
        assertEquals(STILL_OWNED, state.release(reservation, ReleaseReason.COUNTERPART_FINISHED));
        assertEquals(RELEASED, state.release(reservation, ReleaseReason.EXPIRED));
        assertFalse(state.hasOwnedResources(reservation));
        assertEquals(0, state.routingView().engineLoad());
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
        var reservation = state.tryReserveQueuedRequest(1, 100, 200, 50, CAPACITY);
        var permit = state.acquireDispatchPermit(reservation, CAPACITY).permit();
        if (phase == DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN) {
            state.dispatch(permit, DispatchOutcome.ENGINE_OWNED);
        } else if (phase.isEngineConfirmed()) {
            calibrate(state, status, Map.of("1", task(1,
                    phase == DecodeTaskPhase.RUNNING ? TaskPhase.RUNNING : TaskPhase.KV_ALLOCATED)), Map.of());
        }
        var sibling = state.tryReserveQueuedRequest(2, 70, 90, 30, CAPACITY);
        long version = state.placementVersion();
        var stale = new DecodeResources.ReservationHandle(
                reservation.endpointGenerationId(), 1, reservation.reservationToken() + 100);
        var foreign = new DecodeResources.ReservationHandle(
                reservation.endpointGenerationId() + 1, 1, reservation.reservationToken());
        assertEquals(STALE, state.release(stale, reason));
        assertEquals(STALE, state.release(foreign, reason));
        assertEquals(version, state.placementVersion());

        boolean released = phase == DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED
                || phase == DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN && reason == ReleaseReason.NOT_SENT;
        if (released) {
            assertEquals(RELEASED, state.release(reservation, reason));
            assertEquals(STALE, state.release(reservation, reason));
            assertFalse(state.hasOwnedResources(reservation));
            assertEquals(70L, state.routingView().inflightHardKv());
            assertEquals(90L, EndpointTestSupport.expectedReservedKv(state.resourceSnapshot()));
            assertEquals(0, state.resourceSnapshot().activeDispatchPermits());
            assertEquals(1, state.routingView().totalLoad());
            assertEquals(OWNERSHIP_LOST, state.dispatch(permit, DispatchOutcome.ABANDONED).status());
        } else {
            if (reason == ReleaseReason.LOCAL_ROLLBACK) {
                assertThrows(IllegalStateException.class, () -> state.release(reservation, reason));
            } else {
                assertEquals(reason == ReleaseReason.NOT_SENT ? ENGINE_ACCEPTED : STILL_OWNED,
                        state.release(reservation, reason));
            }
            assertTrue(state.hasOwnedResources(reservation));
            assertEquals(version, state.placementVersion());
            assertEquals(2, state.routingView().totalLoad());
        }
        assertTrue(state.hasOwnedResources(sibling));
        assertEquals(1, state.resourceSnapshot().queuedCount());
    }

    @Test
    void calibrationConvertsTheSameReservationWithoutDoubleChargingAndSettlesTerminal() {
        WorkerStatus status = status();
        DecodeState state = new DecodeState(status);
        var reservation = state.tryReserveQueuedRequest(2, 100, 200, 50, CAPACITY);
        var permit = state.acquireDispatchPermit(reservation, CAPACITY).permit();
        TaskInfo task = new TaskInfo();
        task.setRequestId(2L);
        task.setPhase(TaskPhase.KV_ALLOCATED);
        var accepted = calibrate(state, status, Map.of("2", task), Map.of());
        assertEquals(reservation, accepted.getFirst().reservation());
        assertTrue(state.isAcceptedByEngine(reservation));
        assertEquals(1, state.routingView().engineCapacityUsed());
        assertEquals(0, state.resourceSnapshot().activeDispatchPermits());
        assertTrue(state.resourceSnapshot().reservedCount() == 0);
        assertEquals(TRANSFERRED, state.dispatch(permit, DispatchOutcome.ENGINE_OWNED).status());

        var finished = calibrate(state, status, Map.of(), Map.of("2", task));
        assertEquals(DecodeResources.DecodeRequestStatus.Kind.TERMINAL, finished.getFirst().kind());
        assertEquals(reservation, finished.getFirst().reservation());
        assertFalse(state.hasOwnedResources(reservation));
        assertEquals(0, state.routingView().engineCapacityUsed());
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
        var reservation = state.tryReserveQueuedRequest(1, 100, 200, 50, CAPACITY);
        var permit = state.acquireDispatchPermit(reservation, CAPACITY).permit();
        if (phase == DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN) {
            state.dispatch(permit, DispatchOutcome.ENGINE_OWNED);
        } else if (phase.isEngineConfirmed()) {
            calibrate(state, status, Map.of("1", task(1,
                    phase == DecodeTaskPhase.RUNNING ? TaskPhase.RUNNING : TaskPhase.KV_ALLOCATED)), Map.of());
        }
        var sibling = state.tryReserveQueuedRequest(2, 70, 90, 30, CAPACITY);
        assertNotNull(sibling);
        var stale = new DecodeResources.ReservationHandle(
                reservation.endpointGenerationId(), 1, reservation.reservationToken() + 100);
        assertEquals(STALE, state.release(stale, ReleaseReason.EXPIRED));
        assertTrue(state.hasOwnedResources(reservation));

        for (int i = 0; i < 2; i++) {
            if (expired) {
                assertEquals(i == 0 ? RELEASED : STALE, state.release(reservation, ReleaseReason.EXPIRED));
            } else {
                calibrate(state, status, Map.of(), Map.of("1", task(1, TaskPhase.RUNNING)));
            }
            assertFalse(state.hasOwnedResources(reservation));
            assertTrue(state.hasOwnedResources(sibling));
            assertEquals(0, state.resourceSnapshot().activeDispatchPermits());
            assertEquals(1, state.resourceSnapshot().queuedCount());
            assertEquals(70L, state.routingView().inflightHardKv());
            assertEquals(90L, EndpointTestSupport.expectedReservedKv(state.resourceSnapshot()));
            assertEquals(1, state.routingView().totalLoad());
            assertEquals(OWNERSHIP_LOST, state.dispatch(permit, DispatchOutcome.ABANDONED).status());
        }
        assertEquals(RELEASED, state.release(sibling, ReleaseReason.LOCAL_ROLLBACK));
        assertEquals(0, state.routingView().totalLoad());
        assertEquals(0, state.routingView().inflightHardKv());
        assertEquals(0, EndpointTestSupport.expectedReservedKv(state.resourceSnapshot()));
    }

    @ParameterizedTest
    @ValueSource(strings = {"EXPIRED", "COUNTERPART_FINISHED", "NOT_SENT", "FULL_ABSENCE"})
    void terminalHistoryFencesLateReportsUntilRetentionExpires(String settlement) {
        WorkerStatus status = status();
        DecodeState state = new DecodeState(status);
        var reservation = state.tryReserveQueuedRequest(1, 100, 200, 50, CAPACITY);
        if (settlement.equals("FULL_ABSENCE")) {
            calibrate(state, status, Map.of("1", task(1, TaskPhase.RUNNING)), Map.of());
            calibrate(state, status, Map.of(), Map.of());
        } else {
            assertEquals(RELEASED, state.release(reservation, ReleaseReason.valueOf(settlement)));
        }
        assertFalse(state.hasOwnedResources(reservation));
        assertEquals(STALE, state.release(reservation, ReleaseReason.EXPIRED));
        assertFalse(EndpointTestSupport.evictDecode(state, Long.MAX_VALUE, ignored -> false).capacityReleased());
        calibrate(state, status, Map.of(), Map.of("1", task(1, TaskPhase.RUNNING)));
        var late = calibrate(state, status, Map.of("1", task(1, TaskPhase.RUNNING)), Map.of());
        assertTrue(late.isEmpty());
        assertEquals(0, state.routingView().totalLoad());

        assertTrue(EndpointTestSupport.evictDecode(state, -1L, ignored -> false).capacityReleased());
        calibrate(state, status, Map.of("1", task(1, TaskPhase.RUNNING)), Map.of());
        assertEquals(1, state.routingView().totalLoad());
        assertFalse(state.hasOwnedResources(reservation), "expired history must not restore the old token");
    }

    @Test
    void initializationRejectsAnotherGenerationBeforeChangingResources() {
        WorkerStatus owner = status();
        DecodeState state = new DecodeState(owner);
        var incoming = response();
        incoming.setRunningTaskInfo(Map.of("1", task(1, TaskPhase.RUNNING)));
        long version = state.placementVersion();
        assertThrows(IllegalArgumentException.class, () -> state.initialize(status().freezeStatusResponse(incoming)));
        assertEquals(version, state.placementVersion());
        assertEquals(0, state.routingView().totalLoad());
    }

    @Test
    void shadowSweepRechecksPhaseAfterRetentionCallbackConfirmsTheRequest() {
        WorkerStatus status = status();
        DecodeState state = new DecodeState(status);
        var reservation = state.tryReserveQueuedRequest(1, 100, 200, 50, CAPACITY);
        state.acquireDispatchPermit(reservation, CAPACITY);

        var cleanup = EndpointTestSupport.evictDecode(state, -1L, requestId -> {
            calibrate(state, status, Map.of("1", task(1, TaskPhase.RUNNING)), Map.of());
            return false;
        });

        assertEquals(0, cleanup.expiredReservations());
        assertFalse(cleanup.capacityReleased());
        assertTrue(state.isAcceptedByEngine(reservation));
        assertTrue(state.hasOwnedResources(reservation));
        assertEquals(1, state.routingView().engineCapacityUsed());
        assertEquals(0, state.resourceSnapshot().activeDispatchPermits());
        assertEquals(0, state.resourceSnapshot().queuedCount());
        assertEquals(0, state.routingView().inflightHardKv());
    }

    @Test
    void endpointRejectsAnotherEndpointsPermitWithoutConsumingIt() {
        DecodeEndpoint owner = EndpointTestSupport.decode(status(), org.flexlb.balance.scheduler.SchedulerTestSupport.repository(EndpointTestSupport.noopEventSink()));
        DecodeEndpoint other = EndpointTestSupport.decode(status(), org.flexlb.balance.scheduler.SchedulerTestSupport.repository(EndpointTestSupport.noopEventSink()));
        try {
            DecodeResources.ReservationHandle reservation;
            try (var pin = owner.tryPinGeneration()) {
                reservation = owner.tryReserveQueuedRequest(pin, 3, 100, 200, 50, null);
            }
            var permit = owner.acquireDispatchPermit(reservation, CAPACITY).permit();
            assertThrows(IllegalArgumentException.class, () -> other.dispatch(permit, DispatchOutcome.ENGINE_OWNED));
            assertEquals(1, owner.resourceSnapshot().activeDispatchPermits());
            assertEquals(TRANSFERRED, owner.dispatch(permit, DispatchOutcome.ENGINE_OWNED));
            assertEquals(TRANSFERRED, owner.dispatch(permit, DispatchOutcome.ENGINE_OWNED));
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
        var reservation = state.tryReserveQueuedRequest(10, 100, 200, 50, CAPACITY);
        calibrate(state, status, Map.of("10", task(10, TaskPhase.RUNNING)), Map.of());
        for (int i = 0; i < 2; i++) {
            var observation = calibrate(state, status, Map.of("10", task(10, regressed)), Map.of());
            assertEquals(1, observation.size(), "each live report must renew the exact request");
            assertEquals(reservation, observation.getFirst().reservation());
            assertTrue(state.hasOwnedResources(reservation), "live snapshot membership must preserve ownership");
        }
        var resumed = calibrate(state, status, Map.of("10", task(10, TaskPhase.RUNNING)), Map.of());
        assertEquals(reservation, resumed.getFirst().reservation());
        assertEquals(1, state.routingView().engineCapacityUsed());
        var terminal = calibrate(state, status, Map.of(), Map.of("10", task(10, TaskPhase.RECEIVED)));
        assertEquals(reservation, terminal.getFirst().reservation());
        assertEquals(DecodeResources.DecodeRequestStatus.Kind.TERMINAL, terminal.getFirst().kind());
        assertFalse(state.hasOwnedResources(reservation));
        assertEquals(0, state.routingView().engineCapacityUsed());
    }

    @Test
    void expiryOfRegressedRequestDoesNotSubtractAnotherRunningRequestsSlot() {
        WorkerStatus status = status();
        DecodeState state = new DecodeState(status);
        var regressed = state.tryReserveQueuedRequest(10, 100, 200, 50, CAPACITY);
        var running = state.tryReserveQueuedRequest(11, 100, 200, 50, CAPACITY);
        calibrate(state, status, Map.of("10", task(10, TaskPhase.RUNNING), "11", task(11, TaskPhase.RUNNING)), Map.of());
        calibrate(state, status, Map.of("10", task(10, TaskPhase.RECEIVED), "11", task(11, TaskPhase.RUNNING)), Map.of());
        // These are local Engine ownership slots, not physical GPU running concurrency.
        assertEquals(2, state.routingView().engineCapacityUsed());
        assertEquals(RELEASED, state.release(regressed, ReleaseReason.EXPIRED));
        assertEquals(1, state.routingView().engineCapacityUsed());
        assertTrue(state.hasOwnedResources(running));
        assertFalse(state.hasOwnedResources(regressed));
    }

    @Test
    void regressedClaimKeepsOneHoldAndExplicitFinishedWinsOverActiveSnapshot() {
        WorkerStatus status = status();
        DecodeState state = new DecodeState(status);
        var victim = state.tryReserveQueuedRequest(10, 100, 200, 50, CAPACITY);
        calibrate(state, status, Map.of("10", task(10, TaskPhase.RUNNING)), Map.of());
        assertEquals(DecodeResources.PreemptionBeginResult.SUCCESS, state.beginPreemption(
                1, java.util.List.of(victim), 11, 100, 200, 80, new AdmissionCapacity(1, 100)));
        assertTrue(EndpointTestSupport.handoffPreemption(state, 1));
        for (int i = 0; i < 2; i++) {
            calibrate(state, status, Map.of("10", task(10, TaskPhase.RECEIVED)), Map.of());
            assertEquals(2, state.routingView().engineCapacityUsed(), "one victim hold plus one incoming shadow");
            assertEquals(9_800L, state.routingView().realKvAvailable(), "hard KV hold must not be lost or duplicated");
        }
        var terminal = calibrate(state, status, Map.of("10", task(10, TaskPhase.RECEIVED)),
                Map.of("10", task(10, TaskPhase.RECEIVED)));
        assertEquals(1, terminal.size());
        assertEquals(DecodeResources.DecodeRequestStatus.Kind.TERMINAL, terminal.getFirst().kind());
        assertEquals(victim, terminal.getFirst().reservation());
        assertFalse(state.hasOwnedResources(victim));
        assertNotNull(state.finishPreemption(1, true));
        assertEquals(1, state.routingView().engineCapacityUsed());
        assertEquals(9_900L, state.routingView().realKvAvailable());
        assertEquals(RELEASED, state.release(EndpointTestSupport.decodeReservation(state, 11), ReleaseReason.LOCAL_ROLLBACK));
        assertEquals(0, state.routingView().engineCapacityUsed());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void repeatedAllocationPreservesFirstKvAndExactIdentity(boolean locallyReserved) {
        WorkerStatus status = status();
        DecodeState state = new DecodeState(status);
        var reservation = locallyReserved
                ? state.tryReserveQueuedRequest(10, 200, 400, 50, CAPACITY) : null;
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
        assertEquals(1, state.routingView().engineCapacityUsed());
        if (locallyReserved) {
            assertEquals(reservation, first.getFirst().reservation());
            assertEquals(reservation, repeated.getFirst().reservation());
            assertTrue(state.isAcceptedByEngine(reservation));
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
        var victim = state.tryReserveQueuedRequest(10, 100, 200, 50, CAPACITY);
        calibrate(state, status, Map.of("10", task(10, TaskPhase.RUNNING)), Map.of());
        assertEquals(DecodeResources.PreemptionBeginResult.SUCCESS, state.beginPreemption(
                1, java.util.List.of(victim), 11, 100, 200, 80, new AdmissionCapacity(1, 100)));
        assertTrue(EndpointTestSupport.handoffPreemption(state, 1));
        calibrate(state, status, Map.of("10", task(10, regressed)), Map.of());
        assertEquals(9_800L, state.routingView().realKvAvailable());
        var observed = calibrate(state, status, Map.of("allocated", task(10, TaskPhase.KV_ALLOCATED),
                "regressed", task(10, regressed)), Map.of());
        assertEquals(2, observed.size());
        assertTrue(observed.stream().allMatch(event -> event.reservation().equals(victim)));
        assertEquals(2, state.routingView().engineCapacityUsed());
        assertEquals(9_900L, state.routingView().realKvAvailable(),
                "any allocation evidence in the full snapshot removes the synthetic hold");
        calibrate(state, status, Map.of(), Map.of("10", task(10, TaskPhase.RUNNING)));
        var incoming = state.finishPreemption(1, true);
        assertNotNull(incoming);
        assertEquals(RELEASED, state.release(incoming, ReleaseReason.LOCAL_ROLLBACK));
        assertEquals(0, state.routingView().totalLoad());
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
            var prepared = status.prepareNewStatus(status.freezeStatusResponse(response));
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
