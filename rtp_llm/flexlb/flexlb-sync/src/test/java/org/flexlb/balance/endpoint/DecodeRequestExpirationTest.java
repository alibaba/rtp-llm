package org.flexlb.balance.endpoint;

import org.flexlb.balance.endpoint.DecodeResources.ReservationReleaseResult;
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

import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Exact local lease expiration must not wait for Engine reconciliation. */
class DecodeRequestExpirationTest {

    private WorkerStatus status;
    private DecodeEndpoint endpoint;
    private final Map<Long, DecodeResources.ReservationHandle> reservations =
            new HashMap<>();

    @BeforeEach
    void setUp() {
        status = EndpointTestSupport.workerStatus(
                RoleType.DECODE, "10.0.0.8", 8080, 8081);
        endpoint = EndpointTestSupport.decode(status, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(EndpointTestSupport.noopEventSink()));
        updateStatus(Map.of(), Map.of(), 10_000);
    }

    @Test
    void expireUnobservedShadowReleasesAllAccountingAndBlocksStaleStatus() {
        DecodeResources.ReservationHandle reservation = reserve(1L, 500, 700, 0);
        markQueued(1L);

        assertEquals(ReservationReleaseResult.RELEASED, endpoint.release(reservation, DecodeResources.ReleaseReason.EXPIRED));
        assertNotEquals(ReservationReleaseResult.RELEASED, endpoint.release(reservation, DecodeResources.ReleaseReason.EXPIRED));
        assertEquals(0, endpoint.resourceSnapshot().reservedCount());
        assertEquals(0, endpoint.routingView().totalLoad());
        assertEquals(0, endpoint.routingView().inflightHardKv());
        assertEquals(0, EndpointTestSupport.expectedReservedKv(endpoint.resourceSnapshot()));
        assertEquals(0, endpoint.resourceSnapshot().queuedCount());

        updateStatus(Map.of("1", task(1L, TaskPhase.RUNNING, 500)), Map.of(), 9_500);
        assertFalse(isConfirmed(1L), "late status cannot recreate expired local ownership");
        assertEquals(9_500, endpoint.routingView().realKvAvailable(), "physical KV remains Engine-reported");
    }

    @Test
    void expirationReleasesDispatchPermitAndMakesLateReleaseHarmless() {
        DecodeResources.ReservationHandle reservation = reserve(1L, 500, 700, 0);
        markQueued(1L);
        DecodeEndpoint.EngineDispatchPermitAcquisition acquired =
                endpoint.acquireDispatchPermit(reservation, new DecodeResources.AdmissionCapacity(1, 100L));
        assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED, acquired.status());
        assertEquals(1, endpoint.resourceSnapshot().activeDispatchPermits());

        assertEquals(ReservationReleaseResult.RELEASED, endpoint.release(reservation, DecodeResources.ReleaseReason.EXPIRED));
        assertFalse(acquired.permit().release());
        assertEquals(0, endpoint.resourceSnapshot().activeDispatchPermits());
        assertEquals(0, endpoint.routingView().totalLoad());
        assertEquals(10_000, endpoint.routingView().realKvAvailable());
    }

    @Test
    void sentButUnobservedRequestExpiresWithoutAnyEngineReply() {
        DecodeResources.ReservationHandle reservation = reserve(1L, 500, 700, 0);
        markQueued(1L);
        DecodeEndpoint.EngineDispatchPermitAcquisition acquired =
                endpoint.acquireDispatchPermit(reservation, new DecodeResources.AdmissionCapacity(1, 100L));
        assertEquals(DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED,
                acquired.permit().dispatch());
        assertEquals(1, endpoint.routingView().engineLoad());

        assertEquals(ReservationReleaseResult.RELEASED, endpoint.release(reservation, DecodeResources.ReleaseReason.EXPIRED));
        assertEquals(0, endpoint.routingView().engineLoad());
        assertEquals(0, endpoint.routingView().inflightHardKv());
        assertEquals(0, EndpointTestSupport.expectedReservedKv(endpoint.resourceSnapshot()));
        assertNotEquals(ReservationReleaseResult.RELEASED, endpoint.release(reservation, DecodeResources.ReleaseReason.EXPIRED));
    }

    @Test
    void confirmedLeaseExpirationPreservesPhysicalKvSample() {
        DecodeResources.ReservationHandle reservation = reserve(1L, 500, 700, 0);
        updateStatus(Map.of("1", task(1L, TaskPhase.RUNNING, 500)), Map.of(), 9_500);
        assertEquals(1, confirmedCount());

        assertEquals(ReservationReleaseResult.RELEASED, endpoint.release(reservation, DecodeResources.ReleaseReason.EXPIRED));
        assertNotEquals(ReservationReleaseResult.RELEASED, endpoint.release(reservation, DecodeResources.ReleaseReason.EXPIRED));
        assertEquals(0, confirmedCount());
        assertEquals(0, endpoint.routingView().totalLoad());
        assertEquals(9_500, endpoint.routingView().realKvAvailable());
        assertEquals(500, endpoint.routingView().realKvUsed());
    }

    @Test
    void staleEndpointAndReservationTokensCannotExpireNewOwnership() {
        DecodeResources.ReservationHandle original = reserve(1L, 500, 700, 0);
        assertNotEquals(ReservationReleaseResult.RELEASED, endpoint.release(new DecodeResources.ReservationHandle(
                original.endpointGenerationId() + 1L, 1L, original.reservationToken()), DecodeResources.ReleaseReason.EXPIRED));
        assertNotEquals(ReservationReleaseResult.RELEASED, endpoint.release(new DecodeResources.ReservationHandle(
                original.endpointGenerationId(), 1L, original.reservationToken() + 1L), DecodeResources.ReleaseReason.EXPIRED));
        assertEquals(1, endpoint.resourceSnapshot().reservedCount());

        assertEquals(ReservationReleaseResult.RELEASED, endpoint.release(original, DecodeResources.ReleaseReason.EXPIRED));
        endpoint.evictExpiredRequests(-1L, requestId -> false);
        DecodeResources.ReservationHandle replacement = reserve(1L, 300, 450, 0);
        assertNotEquals(ReservationReleaseResult.RELEASED, endpoint.release(original, DecodeResources.ReleaseReason.EXPIRED));
        assertEquals(300, endpoint.routingView().inflightHardKv());
        assertEquals(ReservationReleaseResult.RELEASED, endpoint.release(replacement, DecodeResources.ReleaseReason.EXPIRED));
    }

    @Test
    void missingConfirmedPriorityOwnerExpiresSyntheticHoldExactlyOnce() {
        DecodeResources.ReservationHandle victim = reserve(1L, 500, 700, 30);
        updateStatus(Map.of("1", task(1L, TaskPhase.RUNNING, 500)), Map.of(), 9_500);
        assertEquals(DecodeResources.PreemptionBeginResult.SUCCESS,
                beginPreemption(101L, List.of(1L), 9L, 100, 120, 70));
        assertTrue(EndpointTestSupport.handoffPreemption(endpoint, 101L));
        updateStatus(Map.of(), Map.of(), 10_000);
        endpoint.abortPreemption(101L);
        assertEquals(1, endpoint.routingView().totalLoad());
        assertEquals(9_500, endpoint.routingView().realKvAvailable());

        assertEquals(ReservationReleaseResult.RELEASED, endpoint.release(victim, DecodeResources.ReleaseReason.EXPIRED));
        assertNotEquals(ReservationReleaseResult.RELEASED, endpoint.release(victim, DecodeResources.ReleaseReason.EXPIRED));
        assertFalse(endpoint.updatePreemption(101L, DecodeResources.PreemptionUpdate.canceled(victim)));
        assertFalse(endpoint.updatePreemption(101L, DecodeResources.PreemptionUpdate.finished(victim)));
        assertEquals(0, endpoint.routingView().totalLoad());
        assertEquals(10_000, endpoint.routingView().realKvAvailable());
        assertEquals(0, endpoint.routingView().realKvUsed());
    }

    @Test
    void incomingLeaseExpirationAbortsOnlyItsAdmissionAttempt() {
        DecodeResources.ReservationHandle victim = reserve(1L, 500, 700, 30);
        updateStatus(Map.of("1", task(1L, TaskPhase.RUNNING, 500)), Map.of(), 9_500);
        assertEquals(DecodeResources.PreemptionBeginResult.SUCCESS,
                beginPreemption(101L, List.of(1L), 9L, 100, 120, 70));
        DecodeResources.ReservationHandle incoming = EndpointTestSupport.decodeReservation(endpoint, 9L);
        assertEquals(ReservationReleaseResult.RELEASED, endpoint.release(incoming, DecodeResources.ReleaseReason.EXPIRED));
        assertNotEquals(ReservationReleaseResult.RELEASED, endpoint.release(incoming, DecodeResources.ReleaseReason.EXPIRED));
        endpoint.abortPreemption(101L);
        assertEquals(1, endpoint.routingView().totalLoad());
        assertEquals(0, endpoint.resourceSnapshot().reservedCount());
        assertEquals(ReservationReleaseResult.RELEASED, endpoint.release(victim, DecodeResources.ReleaseReason.EXPIRED));
    }

    @Test
    void ordinaryFinishedRequestsDoNotPopulateRetainedTerminalRecords() {
        int requestCount = 10_000;
        Map<String, TaskInfo> finished = new HashMap<>(requestCount);
        for (long requestId = 1; requestId <= requestCount; requestId++) {
            reserve(requestId, 1, 1, 0);
            finished.put(Long.toString(requestId),
                    task(requestId, TaskPhase.PENDING, 1));
        }

        updateStatus(Map.of(), finished, 10_000);

        assertEquals(0, endpoint.resourceSnapshot().reservedCount());
        updateStatus(Map.of("1", task(1L, TaskPhase.RUNNING, 1)), Map.of(), 9_999);
        assertTrue(isConfirmed(1L),
                "a later fresh active observation is not blocked by an ordinary completion");
    }

    @Test
    void priorityNotFoundOrdinaryFinishedDoesNotRetainGenerationFence() {
        reserve(1L, 500, 700, 30);
        updateStatus(Map.of("1", task(1L, TaskPhase.RUNNING, 500)), Map.of(), 9_500);
        assertEquals(DecodeResources.PreemptionBeginResult.SUCCESS,
                beginPreemption(101L, List.of(1L),
                        9L, 100, 120, 70));
        assertTrue(EndpointTestSupport.handoffPreemption(endpoint, 101L));
        endpoint.abortPreemption(101L);

        assertTrue(endpoint.updatePreemption(101L, DecodeResources.PreemptionUpdate.finished(reservations.get(1L))));

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

    private DecodeResources.ReservationHandle reserve(
            long requestId,
            long hardKv,
            long expectedKv,
            int priority) {
        try (WorkerEndpoint.GenerationPin pin = endpoint.tryPinGeneration()) {
            assertTrue(pin != null);
            DecodeResources.ReservationHandle reservation =
                    EndpointTestSupport.reserveUnqueuedDecode(endpoint, pin, requestId, hardKv, expectedKv, priority);
            reservations.put(requestId, reservation);
            return reservation;
        }
    }

    private void markQueued(long requestId) {
        try (WorkerEndpoint.GenerationPin pin = endpoint.tryPinGeneration()) {
            assertTrue(pin != null);
            assertTrue(endpoint.markQueued(pin, reservations.get(requestId)));
        }
    }

    private boolean isConfirmed(long requestId) {
        return endpoint.resourceSnapshot().requests().values().stream()
                .filter(request -> request.phase().isEngineConfirmed())
                .anyMatch(view -> view.requestId() == requestId);
    }

    private int confirmedCount() {
        return endpoint.resourceSnapshot().confirmedCount();
    }

    private DecodeResources.PreemptionBeginResult beginPreemption(
            long attemptToken,
            List<Long> victimIds,
            long incomingRequestId,
            long hardKv,
            long expectedKv,
            int priority) {
        return endpoint.beginPreemption(attemptToken, victimIds.stream().map(reservations::get).toList(), incomingRequestId, hardKv, expectedKv, priority, new DecodeResources.AdmissionCapacity(
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
