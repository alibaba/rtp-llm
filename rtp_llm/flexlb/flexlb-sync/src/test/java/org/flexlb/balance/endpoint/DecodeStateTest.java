package org.flexlb.balance.endpoint;

import org.flexlb.balance.endpoint.DecodeEndpoint.AdmissionCapacity;
import org.flexlb.balance.endpoint.DecodeEndpoint.DispatchOutcome;
import org.flexlb.balance.endpoint.DecodeEndpoint.ReleaseReason;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.Map;

import static org.flexlb.balance.endpoint.DecodeEndpoint.EngineDispatchPermitTransferStatus.OWNERSHIP_LOST;
import static org.flexlb.balance.endpoint.DecodeEndpoint.EngineDispatchPermitTransferStatus.TRANSFERRED;
import static org.flexlb.balance.endpoint.DecodeEndpoint.ReservationReleaseResult.RELEASED;
import static org.flexlb.balance.endpoint.DecodeEndpoint.ReservationReleaseResult.STILL_OWNED;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The resource ledger can run without an Endpoint, scheduler, RPC client or callback. */
class DecodeStateTest {
    private static final AdmissionCapacity CAPACITY = new AdmissionCapacity(2, 100);

    @Test
    void returnedPermitCannotDispatchOrReleaseItsReplacement() {
        WorkerStatus status = status();
        DecodeState state = new DecodeState(status);
        var reservation = state.reserve(1, 100, 200, 50, true, CAPACITY);
        var first = state.acquireDispatchPermit(reservation, CAPACITY).permit();
        assertTrue(state.dispatch(first, DispatchOutcome.ABANDONED).capacityReleased());
        assertTrue(state.resourceSnapshot().isQueued(1));

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

    @Test
    void calibrationConvertsTheSameReservationWithoutDoubleChargingAndSettlesTerminal() {
        WorkerStatus status = status();
        DecodeState state = new DecodeState(status);
        var reservation = state.reserve(2, 100, 200, 50, true, CAPACITY);
        var permit = state.acquireDispatchPermit(reservation, CAPACITY).permit();
        TaskInfo task = new TaskInfo();
        task.setRequestId("2");
        task.setPhase(TaskPhase.KV_ALLOCATED);
        var accepted = calibrate(state, status, Map.of("2", task), Map.of());
        assertEquals(reservation, accepted.facts().getFirst().reservation());
        assertTrue(state.isAcceptedByEngine(reservation));
        assertEquals(1, state.routingView().engineCapacityUsed());
        assertEquals(0, state.resourceSnapshot().activeDispatchPermits());
        assertTrue(state.resourceSnapshot().reserved().isEmpty());
        assertEquals(TRANSFERRED, state.dispatch(permit, DispatchOutcome.ENGINE_OWNED).status());

        var finished = calibrate(state, status, Map.of(), Map.of("2", task));
        assertEquals(DecodeEndpoint.WorkerStatusFact.Kind.TERMINAL, finished.facts().getFirst().kind());
        assertEquals(reservation, finished.facts().getFirst().reservation());
        assertFalse(state.hasOwnedResources(reservation));
        assertEquals(0, state.routingView().engineCapacityUsed());
        assertEquals(OWNERSHIP_LOST, state.dispatch(permit, DispatchOutcome.ENGINE_OWNED).status());
    }

    @Test
    void endpointRejectsAnotherEndpointsPermitWithoutConsumingIt() {
        DecodeEndpoint owner = new DecodeEndpoint(status(), EndpointTestSupport.noopEventSink());
        DecodeEndpoint other = new DecodeEndpoint(status(), EndpointTestSupport.noopEventSink());
        try {
            DecodeEndpoint.ReservationHandle reservation;
            try (var pin = owner.tryPinGeneration()) {
                reservation = owner.reserve(pin, 3, 100, 200, 50);
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
        var reservation = state.reserve(10, 100, 200, 50, true, CAPACITY);
        calibrate(state, status, Map.of("10", task(10, TaskPhase.RUNNING)), Map.of());
        for (int i = 0; i < 2; i++) {
            var observation = calibrate(state, status, Map.of("10", task(10, regressed)), Map.of());
            assertEquals(1, observation.facts().size(), "each live report must renew the exact request");
            assertEquals(reservation, observation.facts().getFirst().reservation());
            assertTrue(state.hasOwnedResources(reservation), "live snapshot membership must preserve ownership");
        }
        var resumed = calibrate(state, status, Map.of("10", task(10, TaskPhase.RUNNING)), Map.of());
        assertEquals(reservation, resumed.facts().getFirst().reservation());
        assertEquals(1, state.routingView().engineCapacityUsed());
        var terminal = calibrate(state, status, Map.of(), Map.of("10", task(10, TaskPhase.RECEIVED)));
        assertEquals(reservation, terminal.facts().getFirst().reservation());
        assertEquals(DecodeEndpoint.WorkerStatusFact.Kind.TERMINAL, terminal.facts().getFirst().kind());
        assertFalse(state.hasOwnedResources(reservation));
        assertEquals(0, state.routingView().engineCapacityUsed());
    }

    @Test
    void expiryOfRegressedRequestDoesNotSubtractAnotherRunningRequestsSlot() {
        WorkerStatus status = status();
        DecodeState state = new DecodeState(status);
        var regressed = state.reserve(10, 100, 200, 50, true, CAPACITY);
        var running = state.reserve(11, 100, 200, 50, true, CAPACITY);
        calibrate(state, status, Map.of("10", task(10, TaskPhase.RUNNING), "11", task(11, TaskPhase.RUNNING)), Map.of());
        calibrate(state, status, Map.of("10", task(10, TaskPhase.RECEIVED), "11", task(11, TaskPhase.RUNNING)), Map.of());
        // These are local Engine ownership slots, not physical GPU running concurrency.
        assertEquals(2, state.routingView().engineCapacityUsed());
        assertEquals(RELEASED, state.release(regressed, ReleaseReason.EXPIRED));
        assertEquals(1, state.routingView().engineCapacityUsed());
        assertTrue(state.hasOwnedResources(running));
        assertFalse(state.hasOwnedResources(regressed));
    }

    @ParameterizedTest
    @EnumSource(value = PreemptionCancelPhase.class,
            names = {"CANCEL_REQUESTED", "NOT_FOUND_STALE", "CANCEL_UNKNOWN"})
    void regressedClaimKeepsOneHoldAndExplicitFinishedWinsOverActiveSnapshot(PreemptionCancelPhase reply) {
        WorkerStatus status = status();
        DecodeState state = new DecodeState(status);
        var victim = state.reserve(10, 100, 200, 50, true, CAPACITY);
        calibrate(state, status, Map.of("10", task(10, TaskPhase.RUNNING)), Map.of());
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS, state.beginPreemption(
                1, java.util.List.of(victim), "11", 100, 200, 80, new AdmissionCapacity(1, 100)));
        assertTrue(state.updatePreemption(1, DecodeEndpoint.PreemptionUpdate.cancelSending()));
        assertTrue(state.updatePreemption(1, DecodeEndpoint.PreemptionUpdate.cancelReply(10, reply)));
        for (int i = 0; i < 2; i++) {
            calibrate(state, status, Map.of("10", task(10, TaskPhase.RECEIVED)), Map.of());
            assertEquals(2, state.routingView().engineCapacityUsed(), "one victim hold plus one incoming shadow");
            assertEquals(9_800L, state.routingView().realKvAvailable(), "hard KV hold must not be lost or duplicated");
        }
        var terminal = calibrate(state, status, Map.of("10", task(10, TaskPhase.RECEIVED)),
                Map.of("10", task(10, TaskPhase.RECEIVED)));
        assertEquals(1, terminal.facts().size());
        assertEquals(DecodeEndpoint.WorkerStatusFact.Kind.TERMINAL, terminal.facts().getFirst().kind());
        assertEquals(victim, terminal.facts().getFirst().reservation());
        assertFalse(state.hasOwnedResources(victim));
        assertTrue(state.finishPreemption(1, DecodeEndpoint.PreemptionDecision.COMMIT));
        assertEquals(1, state.routingView().engineCapacityUsed());
        assertEquals(9_900L, state.routingView().realKvAvailable());
        assertEquals(RELEASED, state.release(state.reservationHandle("11"), ReleaseReason.LOCAL_ROLLBACK));
        assertEquals(0, state.routingView().engineCapacityUsed());
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

    private static DecodeState.CalibrationResult calibrate(DecodeState state, WorkerStatus status,
                                                           Map<String, TaskInfo> running,
                                                           Map<String, TaskInfo> finished) {
        WorkerStatusResponse response = response();
        response.setRunningTaskInfo(running);
        response.setFinishedTaskInfo(finished);
        response.setStatusVersion(status.appliedStatusCursor().statusVersion() + 1);
        response.setLatestFinishedVersion(status.appliedStatusCursor().latestFinishedTaskVersion() + 1);
        status.lock.lock();
        try {
            return state.calibrate(status.prepareNewStatus(status.freezeStatusResponse(response)));
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
