package org.flexlb.balance.endpoint;

import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Exact local lease expiration must not wait for Engine reconciliation. */
class DecodeRequestExpirationTest {

    private WorkerStatus status;
    private DecodeEndpoint endpoint;
    private final Map<Long, DecodeEndpoint.ReservationHandle> reservations =
            new HashMap<>();

    @BeforeEach
    void setUp() {
        status = EndpointTestSupport.workerStatus(
                RoleType.DECODE, "10.0.0.8", 8080, 8081);
        endpoint = new DecodeEndpoint(
                status, EndpointTestSupport.noopEventSink());
        updateStatus(Map.of(), Map.of(), 10_000);
    }

    @Test
    void expireUnobservedShadowReleasesAllAccountingAndBlocksStaleStatus() {
        DecodeEndpoint.ReservationHandle reservation = reserve(1L, 500, 700, 0);
        markQueued(1L);

        assertTrue(endpoint.expireReservationExact(reservation));
        assertFalse(endpoint.expireReservationExact(reservation));
        assertEquals(0, endpoint.getInflightCount());
        assertEquals(0, endpoint.routingView().totalLoad());
        assertEquals(0, endpoint.routingView().inflightHardKv());
        assertEquals(0, endpoint.routingView().inflightExpectedKv());
        assertEquals(0, endpoint.layeredAdmissionView().queuedCount());

        updateStatus(Map.of("1", task(1L, TaskPhase.RUNNING, 500)), Map.of(), 9_500);
        assertFalse(isConfirmed(1L), "late status cannot recreate expired local ownership");
        assertEquals(9_500, endpoint.realKvAvailable(), "physical KV remains Engine-reported");
    }

    @Test
    void expirationReleasesDispatchPermitAndMakesLateReleaseHarmless() {
        DecodeEndpoint.ReservationHandle reservation = reserve(1L, 500, 700, 0);
        markQueued(1L);
        DecodeEndpoint.EngineDispatchPermitAcquisition acquired =
                endpoint.acquireEngineDispatchPermit(reservation,
                        new DecodeEndpoint.AdmissionCapacity(1, 100L));
        assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.ACQUIRED, acquired.status());
        assertEquals(1, endpoint.layeredAdmissionView().activeDispatchPermits());

        assertTrue(endpoint.expireReservationExact(reservation));
        assertFalse(acquired.permit().release());
        assertEquals(0, endpoint.layeredAdmissionView().activeDispatchPermits());
        assertEquals(0, endpoint.routingView().totalLoad());
        assertEquals(10_000, endpoint.realKvAvailable());
    }

    @Test
    void sentButUnobservedRequestExpiresWithoutAnyEngineReply() {
        DecodeEndpoint.ReservationHandle reservation = reserve(1L, 500, 700, 0);
        markQueued(1L);
        DecodeEndpoint.EngineDispatchPermitAcquisition acquired =
                endpoint.acquireEngineDispatchPermit(reservation,
                        new DecodeEndpoint.AdmissionCapacity(1, 100L));
        assertEquals(DecodeEndpoint.EngineDispatchPermitTransferStatus.TRANSFERRED,
                acquired.permit().transferToEngineLifecycle());
        assertEquals(1, endpoint.routingView().engineLoad());

        assertTrue(endpoint.expireReservationExact(reservation));
        assertEquals(0, endpoint.routingView().engineLoad());
        assertEquals(0, endpoint.routingView().inflightHardKv());
        assertEquals(0, endpoint.routingView().inflightExpectedKv());
        assertFalse(endpoint.expireReservationExact(reservation));
    }

    @Test
    void confirmedLeaseExpirationPreservesPhysicalKvSample() {
        DecodeEndpoint.ReservationHandle reservation = reserve(1L, 500, 700, 0);
        updateStatus(Map.of("1", task(1L, TaskPhase.RUNNING, 500)), Map.of(), 9_500);
        assertEquals(1, confirmedCount());

        assertTrue(endpoint.expireReservationExact(reservation));
        assertFalse(endpoint.expireReservationExact(reservation));
        assertEquals(0, confirmedCount());
        assertEquals(0, endpoint.routingView().totalLoad());
        assertEquals(9_500, endpoint.realKvAvailable());
        assertEquals(500, endpoint.routingView().realKvUsed());
    }

    @Test
    void staleEndpointAndReservationTokensCannotExpireNewOwnership() {
        DecodeEndpoint.ReservationHandle original = reserve(1L, 500, 700, 0);
        assertFalse(endpoint.expireReservationExact(new DecodeEndpoint.ReservationHandle(
                original.endpointGenerationId() + 1L, 1L, original.reservationToken())));
        assertFalse(endpoint.expireReservationExact(new DecodeEndpoint.ReservationHandle(
                original.endpointGenerationId(), 1L, original.reservationToken() + 1L)));
        assertEquals(1, endpoint.getInflightCount());

        assertTrue(endpoint.expireReservationExact(original));
        endpoint.evictExpiredRequests(-1L, requestId -> false);
        DecodeEndpoint.ReservationHandle replacement = reserve(1L, 300, 450, 0);
        assertFalse(endpoint.expireReservationExact(original));
        assertEquals(300, endpoint.routingView().inflightHardKv());
        assertTrue(endpoint.expireReservationExact(replacement));
    }

    @Test
    void missingConfirmedPriorityOwnerExpiresSyntheticHoldExactlyOnce() {
        DecodeEndpoint.ReservationHandle victim = reserve(1L, 500, 700, 30);
        updateStatus(Map.of("1", task(1L, TaskPhase.RUNNING, 500)), Map.of(), 9_500);
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                beginPreemption(101L, List.of(1L), 9L, 100, 120, 70));
        assertTrue(endpoint.markPriorityCancelInFlight(101L));
        assertTrue(endpoint.recordPriorityCancelPhase(
                101L, 1L, PreemptionCancelPhase.NOT_FOUND_STALE));
        updateStatus(Map.of(), Map.of(), 10_000);
        endpoint.abortPriorityPreemption(101L);
        assertEquals(1, endpoint.routingView().totalLoad());
        assertEquals(9_500, endpoint.realKvAvailable());

        assertTrue(endpoint.expireReservationExact(victim));
        assertFalse(endpoint.expireReservationExact(victim));
        assertFalse(endpoint.settlePriorityCanceled(101L, victim));
        assertFalse(endpoint.reconcilePriorityVictimFinished(101L, victim));
        assertEquals(0, endpoint.routingView().totalLoad());
        assertEquals(10_000, endpoint.realKvAvailable());
        assertEquals(0, endpoint.routingView().realKvUsed());
    }

    @Test
    void incomingLeaseExpirationAbortsOnlyItsAdmissionAttempt() {
        DecodeEndpoint.ReservationHandle victim = reserve(1L, 500, 700, 30);
        updateStatus(Map.of("1", task(1L, TaskPhase.RUNNING, 500)), Map.of(), 9_500);
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                beginPreemption(101L, List.of(1L), 9L, 100, 120, 70));
        DecodeEndpoint.ReservationHandle incoming = endpoint.reservationHandle(9L);
        assertTrue(endpoint.expireReservationExact(incoming));
        assertFalse(endpoint.expireReservationExact(incoming));
        endpoint.abortPriorityPreemption(101L);
        assertEquals(1, endpoint.routingView().totalLoad());
        assertEquals(0, endpoint.getInflightCount());
        assertTrue(endpoint.expireReservationExact(victim));
    }

    @Test
    void ordinaryFinishedRequestsDoNotPopulateRetainedTombstones() {
        int requestCount = 10_000;
        Map<String, TaskInfo> finished = new HashMap<>(requestCount);
        for (long requestId = 1; requestId <= requestCount; requestId++) {
            reserve(requestId, 1, 1, 0);
            finished.put(Long.toString(requestId),
                    task(requestId, TaskPhase.PENDING, 1));
        }

        updateStatus(Map.of(), finished, 10_000);

        assertEquals(0, endpoint.getInflightCount());
        updateStatus(Map.of("1", task(1L, TaskPhase.RUNNING, 1)), Map.of(), 9_999);
        assertTrue(isConfirmed(1L),
                "a later fresh active observation is not blocked by an ordinary completion");
    }

    @Test
    void priorityNotFoundOrdinaryFinishedDoesNotRetainGenerationFence() {
        reserve(1L, 500, 700, 30);
        updateStatus(Map.of("1", task(1L, TaskPhase.RUNNING, 500)), Map.of(), 9_500);
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                beginPreemption(101L, List.of(1L),
                        9L, 100, 120, 70));
        assertTrue(endpoint.markPriorityCancelInFlight(101L));
        assertTrue(endpoint.recordPriorityCancelPhase(
                101L, 1L, PreemptionCancelPhase.NOT_FOUND_STALE));
        endpoint.abortPriorityPreemption(101L);

        assertTrue(endpoint.reconcilePriorityVictimFinished(
                101L, reservations.get(1L)));

        updateStatus(Map.of("1", task(1L, TaskPhase.RUNNING, 500)), Map.of(), 9_500);
        assertTrue(isConfirmed(1L));
    }

    private void updateStatus(Map<String, TaskInfo> running,
                              Map<String, TaskInfo> finished,
                              long availableKvCacheTokens) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(running);
        response.setFinishedTaskInfo(finished);
        response.setAvailableKvCacheTokens(availableKvCacheTokens);
        response.setTotalKvCacheTokens(10_000L);
        EndpointTestSupport.applyStatus(endpoint, response);
    }

    private DecodeEndpoint.ReservationHandle reserve(
            long requestId,
            long hardKv,
            long expectedKv,
            int priority) {
        try (WorkerEndpoint.GenerationPin pin = endpoint.tryPinGeneration()) {
            assertTrue(pin != null);
            DecodeEndpoint.ReservationHandle reservation =
                    endpoint.reservePinned(
                            pin, requestId, hardKv, expectedKv, priority);
            reservations.put(requestId, reservation);
            return reservation;
        }
    }

    private void markQueued(long requestId) {
        try (WorkerEndpoint.GenerationPin pin = endpoint.tryPinGeneration()) {
            assertTrue(pin != null);
            assertTrue(endpoint.markQueuedExact(pin, reservations.get(requestId)));
        }
    }

    private boolean isConfirmed(long requestId) {
        return endpoint.layeredAdmissionView().confirmed().stream()
                .anyMatch(view -> view.requestId() == requestId);
    }

    private int confirmedCount() {
        return endpoint.layeredAdmissionView().confirmed().size();
    }

    private DecodeEndpoint.PreemptionBeginResult beginPreemption(
            long attemptToken,
            List<Long> victimIds,
            long incomingRequestId,
            long hardKv,
            long expectedKv,
            int priority) {
        return endpoint.beginPriorityPreemption(
                attemptToken,
                victimIds.stream().map(reservations::get).toList(),
                incomingRequestId,
                hardKv,
                expectedKv,
                priority,
                new DecodeEndpoint.AdmissionCapacity(
                        Math.max(1, endpoint.routingView().totalLoad()), 100));
    }

    private static TaskInfo task(long requestId, TaskPhase phase, long inputLength) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setPhase(phase);
        task.setInputLength(inputLength);
        task.setErrorCode(0);
        return task;
    }
}
