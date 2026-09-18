package org.flexlb.balance.endpoint;

import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.flexlb.balance.endpoint.DecodeEndpoint.EngineDispatchPermitAcquireStatus.ACQUIRED;
import static org.flexlb.balance.endpoint.DecodeEndpoint.EngineDispatchPermitTransferStatus.TRANSFERRED;
import static org.flexlb.balance.endpoint.DecodeEndpoint.PreemptionBeginResult.SUCCESS;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ReturnedPreemptionTest {
    private DecodeEndpoint endpoint;
    private DecodeEndpoint.ReservationHandle victim;

    @BeforeEach
    void setUp() {
        WorkerStatus status = WorkerStatus.createDiscovered(
                RoleType.DECODE, "g1", "10.0.0.1", 8080, 8081, null, null, 1, 2);
        endpoint = new DecodeEndpoint(status, EndpointTestSupport.noopEventSink());
        update(Map.of(), Map.of(), 10_000);
        try (var pin = endpoint.tryPinGeneration()) {
            victim = endpoint.reservePinned(pin, "victim-0001", 128, 256, 30);
        }
        update(Map.of(victim.requestId(), task(victim.requestId())), Map.of(), 9_872);
    }

    @AfterEach
    void close() {
        endpoint.close();
    }

    @Test
    void instructionsCanBeDeliveredBeforeCapacityIsReleased() {
        prepare();
        try (var pin = endpoint.tryPinGeneration()) {
            assertTrue(endpoint.markQueuedExact(pin, endpoint.reservationHandle("incoming")));
        }
        assertTrue(endpoint.isEngineDispatchPermitAvailable("incoming", 1, 100));
        var acquired = endpoint.acquireEngineDispatchPermit("incoming", 1, 100);
        assertEquals(ACQUIRED, acquired.status());
        assertEquals(TRANSFERRED, acquired.permit().transferToEngineLifecycle());
        assertEquals(2, endpoint.routingView().totalLoad());
        assertTrue(claimed());
        update(Map.of(), Map.of(victim.requestId(), task(victim.requestId())), 10_000);
        assertEquals(1, endpoint.routingView().totalLoad());
        assertEquals(9_872, endpoint.realKvAvailable());
        assertFalse(claimed());
        update(Map.of(), Map.of(victim.requestId(), task(victim.requestId())), 10_000);
        assertEquals(1, endpoint.routingView().totalLoad());
    }

    @Test
    void unpublishedRollbackReopensVictimAndReleasesIncomingExactlyOnce() {
        prepare();
        var incoming = endpoint.reservationHandle("incoming");
        endpoint.releaseReservationExact(incoming);
        endpoint.releaseReservationExact(incoming);
        assertEquals(1, endpoint.routingView().totalLoad());
        assertEquals(9_872, endpoint.realKvAvailable());
        assertFalse(claimed());
        prepare();
        assertTrue(claimed());
    }

    @Test
    void missingVictimRemainsChargedAfterIncomingTerminal() {
        prepare();
        dispatch();
        update(Map.of(), Map.of(), 10_000);
        assertEquals(9_744, endpoint.realKvAvailable());
        update(Map.of(), Map.of("incoming", task("incoming")), 10_000);
        assertEquals(1, endpoint.routingView().totalLoad());
        assertEquals(9_872, endpoint.realKvAvailable());
        assertTrue(claimed());
        update(Map.of(), Map.of(victim.requestId(), task(victim.requestId())), 10_000);
        assertEquals(0, endpoint.routingView().totalLoad());
        assertEquals(10_000, endpoint.realKvAvailable());
    }

    @Test
    void terminalBeforeDeliveryStillAllowsRollback() {
        prepare();
        update(Map.of(), Map.of(victim.requestId(), task(victim.requestId())), 10_000);
        endpoint.releaseReservationExact(endpoint.reservationHandle("incoming"));
        assertEquals(0, endpoint.routingView().totalLoad());
        assertEquals(10_000, endpoint.realKvAvailable());
    }

    @Test
    void missingVictimRollbackReleasesSyntheticCapacity() {
        prepare();
        update(Map.of(), Map.of(), 10_000);
        endpoint.releaseReservationExact(endpoint.reservationHandle("incoming"));
        assertEquals(0, endpoint.routingView().totalLoad());
        assertEquals(10_000, endpoint.realKvAvailable());
    }

    @Test
    void rollbackRetainsOverlappingGenericFence() {
        prepare();
        var fence = endpoint.beginEngineFenceProtection(victim);
        update(Map.of(), Map.of(), 10_000);
        endpoint.releaseReservationExact(endpoint.reservationHandle("incoming"));
        assertEquals(1, endpoint.routingView().totalLoad());
        assertEquals(9_872, endpoint.realKvAvailable());
        fence.settleAuthoritativeTerminal();
        assertEquals(0, endpoint.routingView().totalLoad());
        assertEquals(10_000, endpoint.realKvAvailable());
    }

    @Test
    void publishedVictimAlsoSettlesThroughAnIndependentCancellationFence() {
        prepare();
        dispatch();
        var fence = endpoint.beginEngineFenceProtection(victim);
        update(Map.of(), Map.of(), 10_000);
        fence.settleAuthoritativeTerminal();
        fence.settleAuthoritativeTerminal();
        assertFalse(claimed());
        assertEquals(1, endpoint.routingView().totalLoad());
        assertEquals(9_872, endpoint.realKvAvailable());
    }

    @Test
    void exactVictimCannotBeClaimedTwiceOrMovedBetweenEngines() {
        prepare();
        assertEquals(DecodeEndpoint.PreemptionBeginResult.VICTIM_ALREADY_CLAIMED,
                endpoint.beginReturnedPreemption(2, List.of(victim), "next", 128, 256, 80,
                        new DecodeEndpoint.AdmissionCapacity(1, 100)));
        DecodeEndpoint sibling = new DecodeEndpoint(WorkerStatus.createDiscovered(
                RoleType.DECODE, "g1", "10.0.0.1", 8080, 8081, null, null, 0, 2),
                EndpointTestSupport.noopEventSink());
        try {
            assertEquals(DecodeEndpoint.PreemptionBeginResult.VICTIM_GONE,
                    sibling.beginReturnedPreemption(3, List.of(victim), "next", 128, 256, 80,
                            new DecodeEndpoint.AdmissionCapacity(1, 100)));
        } finally {
            sibling.close();
        }
    }

    private void prepare() {
        assertEquals(SUCCESS, endpoint.beginReturnedPreemption(1, List.of(victim),
                "incoming", 128, 256, 70, new DecodeEndpoint.AdmissionCapacity(1, 100)));
    }

    private void dispatch() {
        try (var pin = endpoint.tryPinGeneration()) {
            assertTrue(endpoint.markQueuedExact(pin, endpoint.reservationHandle("incoming")));
        }
        assertEquals(TRANSFERRED, endpoint.acquireEngineDispatchPermit("incoming", 1, 100)
                .permit().transferToEngineLifecycle());
    }

    private boolean claimed() {
        return endpoint.layeredAdmissionView().confirmed().stream()
                .anyMatch(task -> task.requestId().equals(victim.requestId()) && task.claimedForPreemption());
    }

    private void update(Map<String, TaskInfo> running, Map<String, TaskInfo> finished, long available) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(running);
        response.setFinishedTaskInfo(finished);
        response.setAvailableKvCacheTokens(available);
        response.setTotalKvCacheTokens(10_000L);
        EndpointTestSupport.applyStatus(endpoint, response).run();
    }

    private static TaskInfo task(String id) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(id);
        task.setPhase(TaskPhase.RUNNING);
        task.setInputLength(128);
        task.setErrorCode(8429);
        return task;
    }
}
