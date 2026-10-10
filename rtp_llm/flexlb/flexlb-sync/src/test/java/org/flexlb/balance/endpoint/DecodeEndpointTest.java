package org.flexlb.balance.endpoint;

import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.enums.DecodeTaskPhase;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.flexlb.balance.endpoint.DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

class DecodeEndpointTest {

    private WorkerStatus status;
    private DecodeEndpoint endpoint;
    private final Map<Long, DecodeResources.ReservationHandle> reservations =
            new HashMap<>();

    @BeforeEach
    void setUp() {
        status = EndpointTestSupport.workerStatus(
                RoleType.DECODE, "10.0.0.1", 8080, 8081);
        endpoint = EndpointTestSupport.decode(status, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(EndpointTestSupport.noopEventSink()));
    }

    @Test
    void dispatchPermitIdentifiesTheExactEndpointGenerationAndReservation() {
        var reservation = reserve(100L, 500L, 500L);
        markQueued(100L);
        var permit = acquirePermit(100L);
        try {
            assertTrue(permit.belongsTo(endpoint, reservation));
            assertFalse(permit.belongsTo(org.mockito.Mockito.mock(DecodeEndpoint.class), reservation));
            assertFalse(permit.belongsTo(endpoint, new DecodeResources.ReservationHandle(
                    reservation.endpointGenerationId() + 1, 100L, reservation.reservationToken())));
            assertFalse(permit.belongsTo(endpoint, new DecodeResources.ReservationHandle(
                    reservation.endpointGenerationId(), 101L, reservation.reservationToken())));
            assertFalse(permit.belongsTo(endpoint, new DecodeResources.ReservationHandle(
                    reservation.endpointGenerationId(), 100L, reservation.reservationToken() + 1)));
            assertEquals(1, endpoint.resourceSnapshot().activeDispatchPermits());
        } finally {
            permit.release();
            endpoint.release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        }
        assertEquals(0, endpoint.resourceSnapshot().activeDispatchPermits());
    }

    @ParameterizedTest
    @EnumSource(TaskPhase.class)
    void activeRequestStatusesPreserveExactOwnershipWhileLedgerKeepsResourcePhase(TaskPhase phase) {
        var sink = EndpointTestSupport.noopEventSink();
        endpoint = EndpointTestSupport.decode(status, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(sink));
        var reservation = reserve(100L, 500L, 500L);
        TaskInfo active = task(100L);
        active.setPhase(phase);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(Map.of("100", active));
        response.setTotalKvCacheTokens(10000L);
        response.setAvailableKvCacheTokens(10000L);

        EndpointTestSupport.applyStatus(endpoint, response).run();
        boolean allocated = phase == TaskPhase.KV_ALLOCATED || phase == TaskPhase.RUNNING;
        var requestStatus = allocated ? DecodeResources.DecodeRequestStatus.allocated(reservation)
                : DecodeResources.DecodeRequestStatus.active(reservation);
        verify(sink).onDecodeStatus(org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.eq(endpoint), org.mockito.ArgumentMatchers.eq(requestStatus));
        assertEquals(allocated, endpoint.isAcceptedByEngine(reservation));
        assertEquals(allocated ? 0 : 1, endpoint.resourceSnapshot().reservedCount());
        assertEquals(phase == TaskPhase.RUNNING ? DecodeTaskPhase.RUNNING
                        : allocated ? DecodeTaskPhase.ACCEPTED_NOT_RUNNING : DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN,
                endpoint.resourceSnapshot().requests().get(100L).phase());

        endpoint.applyStatusHeartbeat(status, status.freezeStatusResponse(response)).run();
        verify(sink, times(2)).onDecodeStatus(org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.eq(endpoint), org.mockito.ArgumentMatchers.eq(requestStatus));
        assertEquals(allocated, endpoint.isAcceptedByEngine(reservation));

        WorkerStatusResponse finished = new WorkerStatusResponse();
        finished.setFinishedTaskInfo(Map.of("100", task(100L)));
        EndpointTestSupport.applyStatus(endpoint, finished).run();
        verify(sink).onDecodeStatus(org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.eq(endpoint), org.mockito.ArgumentMatchers.eq(DecodeResources.DecodeRequestStatus.terminal(reservation, 0L)));
        assertFalse(endpoint.resourceSnapshot().requests().containsKey(100L));
        assertEquals(0, endpoint.resourceSnapshot().reservedCount());
    }

    @Test
    void reserve_updatesSnapshotAndInflight() {
        updateStatus(null, null, 10000);
        reserve(100L, 500, 500);
        assertEquals(1, endpoint.resourceSnapshot().reservedCount());
        assertEquals(9500, endpoint.routingView().realKvAvailable());
    }

    @Test
    void release_decrementsInflight() {
        reserve(100L, 500, 500);
        reserve(101L, 300, 300);
        release(100L);

        assertEquals(1, endpoint.resourceSnapshot().reservedCount());
    }

    @Test
    void release_unknownRequestId_noEffect() {
        reserve(100L, 500, 500);
        release(999L);
        assertEquals(1, endpoint.resourceSnapshot().reservedCount());
    }

    @Test
    void release_neverGoesNegative() {
        reserve(100L, 100, 100);
        release(100L);
        release(100L);
        assertEquals(0, endpoint.resourceSnapshot().reservedCount());
        assertEquals(0, endpoint.routingView().realKvAvailable());
    }

    @Test
    void calibrate_kvAllocatedReleasesFromInflight() {
        reserve(100L, 500, 500);

        TaskInfo running = task(100L);
        running.setPhase(TaskPhase.KV_ALLOCATED);
        updateStatus(Map.of("100", running), null, 10000);

        assertEquals(0, endpoint.resourceSnapshot().reservedCount());
        assertEquals(10000, endpoint.routingView().realKvAvailable());
    }

    @Test
    void calibrate_finishedFailureReleasesFromInflight() {
        reserve(100L, 500, 500);

        TaskInfo failed = task(100L);
        failed.setErrorCode(1);
        failed.setErrorMessage("timeout");
        updateStatus(null, Map.of("100", failed), 10000);

        assertEquals(0, endpoint.resourceSnapshot().reservedCount());
    }

    @Test
    void calibrate_finishedSuccessReleasesIfStillPresent() {
        reserve(100L, 500, 500);

        TaskInfo success = task(100L);
        success.setErrorCode(0);
        updateStatus(null, Map.of("100", success), 10000);

        assertEquals(0, endpoint.resourceSnapshot().reservedCount());
    }

    @Test
    void calibrate_updatesReportedKvAvailable() {
        reserve(100L, 500, 500);
        updateStatus(null, null, 10000);

        assertEquals(9500, endpoint.routingView().realKvAvailable());
    }

    @Test
    void availableKvTokens_accountsForReservations() {
        updateStatus(null, null, 10000);

        reserve(100L, 3000, 3000);
        reserve(101L, 2000, 2000);

        assertEquals(5000, endpoint.routingView().realKvAvailable());
    }

    @Test
    void ipPort_format() {
        assertEquals("10.0.0.1:8080", endpoint.ipPort());
    }

    // ==================== PR-C: getEngineLoad O(1) + queuedPhaseCount drift ===========

    @Test
    void getEngineLoad_o1_tracks_markQueued_and_markDispatched() {
        updateStatus(null, null, 10000);
        reserve(1L, 100, 100);
        reserve(2L, 200, 200);
        reserve(3L, 300, 300);
        // No queued phase: engineLoad == totalLoad == inflight(3)
        assertEquals(3, endpoint.routingView().engineLoad());
        assertEquals(endpoint.routingView().totalLoad(), endpoint.routingView().engineLoad());

        // mark req 1,2 as queued → engine-facing load drops to 1
        markQueued(1L);
        markQueued(2L);
        assertEquals(1, endpoint.routingView().engineLoad());

        var queued = endpoint.resourceSnapshot().requests().get(1L);
        assertTrue(queued.phase().isMasterQueued());
        assertEquals(DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED, queued.phase());
        assertFalse(endpoint.isAcceptedByEngine(reservations.get(1L)));

        // Commit req 1's pre-delivery permit → back to engine load 2.
        assertEquals(TRANSFERRED, acquirePermit(1L).dispatch());
        assertEquals(2, endpoint.routingView().engineLoad());
        var dispatched = endpoint.resourceSnapshot().requests().get(1L);
        assertFalse(dispatched.phase().isMasterQueued());
        assertEquals(DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN, dispatched.phase());
        assertTrue(queued.phase().isMasterQueued(), "dispatch cannot change the previously captured phase");

        // release req 2 (was queued) → inflight=2, queued=0
        release(2L);
        assertEquals(2, endpoint.routingView().engineLoad());

        // req 1 was transferred to the engine lifecycle. Once dispatched to the
        // engine, local shadow rollback is no longer permitted: only an
        // authoritative engine terminal may settle it. A stray local release
        // must fail closed rather than double-release the engine-owned slot.
        assertThrows(IllegalStateException.class, () -> release(1L));
        assertEquals(2, endpoint.routingView().engineLoad());

        // release req 3 (plain inflight, never dispatched) → inflight=1 (req 1
        // remains engine-owned).
        release(3L);
        assertEquals(1, endpoint.routingView().engineLoad());

        // authoritative engine terminal for req 1 settles the last slot → 0.
        updateStatus(null, Map.of("1", task(1L)), 10000);
        assertEquals(0, endpoint.routingView().engineLoad());
    }

    @Test
    void getEngineLoad_calibrate_prunes_queued_phase_count() {
        reserve(1L, 100, 100);
        reserve(2L, 100, 100);
        markQueued(1L);
        markQueued(2L);
        assertEquals(0, endpoint.routingView().engineLoad()); // both queued

        // calibrate: req 1 confirmed → removed from inflight + queued
        TaskInfo running = task(1L);
        running.setPhase(TaskPhase.KV_ALLOCATED);
        updateStatus(Map.of("1", running), null, 10000);

        var confirmed = endpoint.resourceSnapshot().requests().get(1L);
        assertEquals(1L, confirmed.requestId());
        assertEquals(DecodeTaskPhase.ACCEPTED_NOT_RUNNING, confirmed.phase());
        assertFalse(confirmed.phase().isMasterQueued());
        assertTrue(endpoint.isAcceptedByEngine(reservations.get(1L)));
        assertTrue(endpoint.resourceSnapshot().requests().get(2L).phase().isMasterQueued());

        // req 2 still queued, inflight=1 (req2), confirmed=1 (req1)
        // engineLoad = confirmed(1) + max(0, inflight(1) - queued(1)) = 1
        assertEquals(1, endpoint.routingView().engineLoad());
    }

    @Test
    void getEngineLoad_idempotent_markQueued_does_not_double_count() {
        reserve(1L, 100, 100);
        markQueued(1L);
        markQueued(1L); // idempotent: add returns false
        assertEquals(0, endpoint.routingView().engineLoad());
    }

    @Test
    void getEngineLoad_idempotentPermitCommit_doesNotOverDecrement() {
        reserve(1L, 100, 100);
        markQueued(1L);
        DecodeEndpoint.EngineDispatchPermit permit = acquirePermit(1L);
        assertEquals(TRANSFERRED, permit.dispatch());
        assertEquals(TRANSFERRED, permit.dispatch());
        assertEquals(1, endpoint.routingView().engineLoad());
    }

    @Test
    void getEngineLoad_clamps_negative_drift_to_zero() throws Exception {
        reserve(1L, 100, 100);
        setQueuedPhaseCount(-5);
        // inflight=1, queued clamped from -5 to 0 → engineLoad = 0 + max(0,1-0) = 1
        assertEquals(1, endpoint.routingView().engineLoad());
    }

    @Test
    void getEngineLoad_clamps_overflow_drift_to_inflight() throws Exception {
        reserve(1L, 100, 100);
        reserve(2L, 100, 100);
        setQueuedPhaseCount(100);
        // inflight=2, queued clamped from 100 to 2 → engineLoad = 0 + max(0,2-2) = 0
        assertEquals(0, endpoint.routingView().engineLoad());
    }

    // ==================== P1-6: evictExpiredRequests counter (PR-C) ====================

    /** Expiration removes the queued phase and all incremental counters together. */
    @Test
    void evictExpiredRequests_prunesQueuedPhase_andRestoresEngineLoad() throws InterruptedException {
        updateStatus(null, null, 10000);
        reserve(100L, 500, 500);
        reserve(101L, 500, 500);
        reserve(102L, 500, 500);
        markQueued(100L);
        markQueued(101L);
        markQueued(102L);

        assertEquals(3, endpoint.resourceSnapshot().reservedCount());
        assertEquals(0, endpoint.routingView().engineLoad());
        assertEquals(3, endpoint.routingView().totalLoad());
        assertEquals(3, endpoint.resourceSnapshot().queuedCount());

        Thread.sleep(20);
        int evicted = endpoint.evictExpiredRequests(5, requestId -> false);

        assertEquals(3, evicted);
        assertEquals(0, endpoint.resourceSnapshot().reservedCount());
        assertTrue(endpoint.resourceSnapshot().queuedCount() == 0);
        assertEquals(0, endpoint.routingView().engineLoad());
        assertEquals(0, endpoint.routingView().totalLoad());
        assertEquals(0, endpoint.routingView().inflightHardKv());
        assertEquals(0, EndpointTestSupport.expectedReservedKv(endpoint.resourceSnapshot()));
        assertEquals(0, endpoint.routingView().dispatchUsage().expectedKvUsed());
    }

    /** Directly mutate the private counter to simulate drift. */
    private void setQueuedPhaseCount(int value) throws Exception {
        java.lang.reflect.Field owner = DecodeEndpoint.class.getDeclaredField("state");
        owner.setAccessible(true);
        java.lang.reflect.Field usage = DecodeState.class.getDeclaredField("queuedUsage");
        usage.setAccessible(true);
        Object counter = usage.get(owner.get(endpoint));
        java.lang.reflect.Field requests = counter.getClass().getDeclaredField("requests");
        requests.setAccessible(true);
        requests.setInt(counter, value);
    }

    private void updateStatus(Map<String, TaskInfo> running, Map<String, TaskInfo> finished,
                              long availableKvCacheTokens) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(running);
        response.setFinishedTaskInfo(finished);
        response.setAvailableKvCacheTokens(availableKvCacheTokens);
        response.setTotalKvCacheTokens(availableKvCacheTokens);
        EndpointTestSupport.applyStatus(endpoint, response);
    }

    private DecodeResources.ReservationHandle reserve(
            long requestId, long hardKv, long expectedKv) {
        try (WorkerEndpoint.GenerationPin pin = endpoint.tryPinGeneration()) {
            assertNotNull(pin);
            DecodeResources.ReservationHandle reservation =
                    EndpointTestSupport.reserveUnqueuedDecode(endpoint, pin, requestId, hardKv, expectedKv, 0);
            reservations.put(requestId, reservation);
            return reservation;
        }
    }

    private void release(long requestId) {
        DecodeResources.ReservationHandle reservation =
                reservations.get(requestId);
        if (reservation != null) {
            endpoint.release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        }
    }

    private void markQueued(long requestId) {
        try (WorkerEndpoint.GenerationPin pin = endpoint.tryPinGeneration()) {
            assertNotNull(pin);
            assertTrue(endpoint.markQueued(pin, reservations.get(requestId)));
        }
    }

    private TaskInfo task(long requestId) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        return task;
    }

    private DecodeEndpoint.EngineDispatchPermit acquirePermit(long requestId) {
        DecodeEndpoint.EngineDispatchPermitAcquisition acquisition =
                endpoint.acquireDispatchPermit(reservations.get(requestId), new DecodeResources.AdmissionCapacity(0, 100L));
        assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED,
                acquisition.status());
        assertNotNull(acquisition.permit());
        return acquisition.permit();
    }
}
