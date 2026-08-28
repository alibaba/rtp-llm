package org.flexlb.balance.endpoint;

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

/** Deterministic ownership and cleanup tests for Decode EngineFence accounting. */
class DecodeEngineFenceProtectionTest {

    private WorkerStatus status;
    private DecodeEndpoint endpoint;
    private final Map<Long, DecodeEndpoint.ReservationHandle> reservations =
            new HashMap<>();
    private final Map<Long, DecodeEndpoint.EngineFenceLease> fences =
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
    void shadowOlderThanTtlRemainsUntilFenceEnds() {
        reserve(1L, 500, 700, 0);
        assertTrue(beginFence(1L));
        assertTrue(beginFence(1L),
                "begin is idempotent for one live accounting generation");

        assertEquals(0, endpoint.evictExpiredRequests(
                -1, requestId -> false));
        assertEquals(1, endpoint.getInflightCount());
        assertEquals(500, endpoint.inflightHardKvReserved());

        assertTrue(closeFence(1L));
        assertFalse(closeFence(1L));
        assertEquals(1, endpoint.evictExpiredRequests(
                -1, requestId -> false));
        assertEquals(0, endpoint.getInflightCount());
        assertEquals(0, endpoint.inflightHardKvReserved());
    }

    @Test
    void confirmedMetadataOlderThanTtlRemainsUntilFenceEnds() {
        reserve(1L, 500, 700, 0);
        assertTrue(beginFence(1L));
        updateStatus(Map.of("1", task(1L, TaskPhase.RUNNING, 500)), Map.of(), 9_500);

        endpoint.evictExpiredRequests(-1, requestId -> false);
        assertTrue(isConfirmed(1L));

        assertTrue(closeFence(1L));
        endpoint.evictExpiredRequests(-1, requestId -> false);
        assertFalse(isConfirmed(1L));
    }

    @Test
    void releaseAndOrdinaryTerminalClearProtectionExactlyOnce() {
        reserve(1L, 500, 700, 0);
        assertTrue(beginFence(1L));
        assertTrue(closeFence(1L));
        release(1L);
        release(1L);
        assertFalse(closeFence(1L));
        assertEquals(0, endpoint.getInflightCount());

        reserve(2L, 300, 450, 0);
        assertTrue(beginFence(2L));
        updateStatus(Map.of(), Map.of("2", task(2L, TaskPhase.PENDING, 300)), 10_000);

        // The ordinary finished terminal is authoritative: it already removed
        // the exact engine-fence protection (see assertions below). The test
        // still holds the lease handle, so closing it now is an idempotent
        // no-op on the endpoint and simply reports that the handle was live.
        assertTrue(closeFence(2L));
        assertEquals(0, endpoint.getInflightCount());
        assertEquals(0, endpoint.getTotalLoad());
        assertEquals(10_000, endpoint.realKvAvailable());
        updateStatus(Map.of("2", task(2L, TaskPhase.RUNNING, 300)), Map.of(), 9_700);
        assertFalse(isConfirmed(2L),
                "ordinary finished uses the same stale-status fence");
        endpoint.evictExpiredRequests(-1, requestId -> false);
        updateStatus(Map.of("2", task(2L, TaskPhase.RUNNING, 300)), Map.of(), 9_700);
        assertTrue(isConfirmed(2L));
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
        assertTrue(endpoint.markPriorityCancelNotFound(101L, 1L));
        endpoint.abortPriorityPreemption(101L);

        assertTrue(endpoint.reconcilePriorityVictimFinished(
                101L, reservations.get(1L)));

        updateStatus(Map.of("1", task(1L, TaskPhase.RUNNING, 500)), Map.of(), 9_500);
        assertTrue(isConfirmed(1L));
    }

    @Test
    void missingConfirmedRequestTransfersToOneSyntheticSlotAndKvOwner() {
        moveProtectedRequestToMissingConfirmed(1L, 500, 700);

        assertEquals(1, confirmedCount());
        assertEquals(1, endpoint.getTotalLoad());
        assertTrue(isConfirmed(1L));
        assertEquals(9_500, endpoint.realKvAvailable());
        assertEquals(700, endpoint.realKvUsed());

        // Repeated absence must not add another synthetic slot or KV hold.
        updateStatus(Map.of(), Map.of(), 10_000);
        assertEquals(1, confirmedCount());
        assertEquals(9_500, endpoint.realKvAvailable());
        assertEquals(700, endpoint.realKvUsed());

        assertTrue(closeFence(1L));
        // Closing the fence releases the synthetic slot and its KV hold, so the
        // engine-facing load and KV accounting drop back to idle. The confirmed
        // tombstone metadata for the missing request is not part of the fence,
        // so it survives here and is only pruned by TTL eviction below.
        assertEquals(1, confirmedCount());
        assertEquals(0, endpoint.getTotalLoad());
        assertEquals(10_000, endpoint.realKvAvailable());
        assertEquals(0, endpoint.realKvUsed());
        assertFalse(closeFence(1L));

        endpoint.evictExpiredRequests(-1, requestId -> false);
        assertFalse(isConfirmed(1L));
    }

    @Test
    void freshActiveObservationReturnsSyntheticOwnershipToEngine() {
        moveProtectedRequestToMissingConfirmed(1L, 500, 700);

        updateStatus(Map.of("1", task(1L, TaskPhase.RUNNING, 500)), Map.of(), 9_500);
        assertEquals(1, confirmedCount());
        assertEquals(1, endpoint.getTotalLoad());
        assertEquals(9_500, endpoint.realKvAvailable(),
                "reported engine KV replaces, rather than stacks with, the synthetic hold");

        assertTrue(closeFence(1L));
        assertEquals(1, endpoint.getTotalLoad(),
                "clearing the fence must not release a freshly confirmed engine owner");

        updateStatus(Map.of(), Map.of("1", task(1L, TaskPhase.RUNNING, 500)), 10_000);
        assertEquals(0, endpoint.getTotalLoad());
        assertEquals(10_000, endpoint.realKvAvailable());
    }

    @Test
    void authoritativeTerminalReleasesSyntheticOwnerAndProtection() {
        moveProtectedRequestToMissingConfirmed(1L, 500, 700);

        updateStatus(Map.of(), Map.of("1", task(1L, TaskPhase.RUNNING, 500)), 10_000);

        assertEquals(0, confirmedCount());
        assertEquals(0, endpoint.getTotalLoad());
        assertEquals(10_000, endpoint.realKvAvailable());
        assertEquals(0, endpoint.realKvUsed());
        assertFalse(isConfirmed(1L));
        // The authoritative terminal already released the synthetic owner and
        // removed the exact protection. The lease handle is still held by the
        // test, so closing it is an idempotent no-op that only reports that the
        // handle itself was live.
        assertTrue(closeFence(1L));
    }

    @Test
    void tombstonedShadowAtomicallyClearsAccountingAndFencesStaleStatus() {
        reserve(1L, 500, 700, 0);
        markQueued(1L);
        assertTrue(beginFence(1L));

        assertTrue(settleTombstoned(1L));
        assertFalse(settleTombstoned(1L),
                "duplicate settlement must not mutate counters twice");
        assertEquals(0, endpoint.getInflightCount());
        assertEquals(0, endpoint.getTotalLoad());
        assertEquals(0, endpoint.inflightHardKvReserved());
        assertEquals(0, endpoint.routingView().inflightExpectedKv());
        assertTrue(endpoint.layeredAdmissionView().queued().isEmpty());
        assertFalse(closeFence(1L));

        updateStatus(Map.of("1", task(1L, TaskPhase.RUNNING, 500)), Map.of(), 9_500);
        assertFalse(isConfirmed(1L),
                "a delayed active sample cannot resurrect a tombstoned request");
        assertEquals(0, confirmedCount());

        endpoint.evictExpiredRequests(-1, requestId -> false);
        updateStatus(Map.of("1", task(1L, TaskPhase.RUNNING, 500)), Map.of(), 9_500);
        assertTrue(isConfirmed(1L),
                "the settled fence is bounded by the configured endpoint TTL");
    }

    @Test
    void tombstonedConfirmedOwnerClearsTrackedSlotAndGenericFence() {
        reserve(1L, 500, 700, 0);
        updateStatus(Map.of("1", task(1L, TaskPhase.RUNNING, 500)), Map.of(), 9_500);
        assertTrue(beginFence(1L));

        assertTrue(settleTombstoned(1L));
        assertFalse(isConfirmed(1L));
        assertEquals(0, confirmedCount());
        assertEquals(0, endpoint.getTotalLoad());
        assertEquals(9_500, endpoint.realKvAvailable(),
                "local settlement must not rewrite the last engine KV sample");
        assertFalse(closeFence(1L));

        updateStatus(Map.of(), Map.of(), 10_000);
        assertEquals(10_000, endpoint.realKvAvailable());
    }

    @Test
    void tombstonedSyntheticOwnerReleasesSlotAndKvExactlyOnce() {
        moveProtectedRequestToMissingConfirmed(1L, 500, 700);
        assertEquals(1, confirmedCount());
        assertEquals(9_500, endpoint.realKvAvailable());
        assertEquals(700, endpoint.realKvUsed());

        assertTrue(settleTombstoned(1L));
        assertFalse(isConfirmed(1L));
        assertEquals(0, confirmedCount());
        assertEquals(0, endpoint.getTotalLoad());
        assertEquals(10_000, endpoint.realKvAvailable());
        assertEquals(0, endpoint.realKvUsed());
        assertFalse(settleTombstoned(1L));
    }

    @Test
    void genericAndPriorityFenceOwnersFormOneAccountingUnion() {
        reserve(1L, 500, 700, 30);
        updateStatus(Map.of("1", task(1L, TaskPhase.RUNNING, 500)), Map.of(), 9_500);
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                beginPreemption(101L, List.of(1L),
                        9L, 100, 120, 70));
        assertTrue(endpoint.markPriorityCancelInFlight(101L));
        assertTrue(endpoint.markPriorityCancelNotFound(101L, 1L));

        updateStatus(Map.of(), Map.of(), 10_000);
        assertTrue(endpoint.transferPriorityNotFoundClaimToEngineFence(101L, 1L));
        endpoint.abortPriorityPreemption(101L);
        assertEquals(1, endpoint.getTotalLoad());
        assertEquals(9_500, endpoint.realKvAvailable());

        assertTrue(beginFence(1L));
        assertTrue(closeFence(1L));
        assertEquals(1, endpoint.getTotalLoad(),
                "ending generic ownership must not release the priority token owner");
        assertEquals(9_500, endpoint.realKvAvailable());

        assertTrue(beginFence(1L));
        assertTrue(endpoint.settleEngineFenceClaim(
                101L, reservations.get(1L)));
        // Settling the priority token owner is the authoritative terminal: it
        // releases the confirmed slot and its held KV. The overlapping generic
        // fence created just above never installed an independent synthetic
        // slot (the priority claim already owned the accounting), so there is no
        // union hold left for it to inherit; the load and KV return to idle.
        assertEquals(0, endpoint.getTotalLoad(),
                "priority settlement releases the confirmed slot; the overlapping "
                        + "generic fence held no independent synthetic slot");
        assertEquals(10_000, endpoint.realKvAvailable());

        assertTrue(settleTombstoned(1L));
        assertEquals(0, endpoint.getTotalLoad());
        assertEquals(10_000, endpoint.realKvAvailable());
        assertFalse(closeFence(1L));
        assertFalse(endpoint.settleEngineFenceClaim(
                101L, reservations.get(1L)));
    }

    private void moveProtectedRequestToMissingConfirmed(long requestId,
                                                        long hardKvTokens,
                                                        long expectedKvTokens) {
        reserve(requestId, hardKvTokens, expectedKvTokens, 0);
        assertTrue(beginFence(requestId));
        updateStatus(Map.of(Long.toString(requestId),
                task(requestId, TaskPhase.RUNNING, hardKvTokens)), Map.of(),
                10_000 - hardKvTokens);
        updateStatus(Map.of(), Map.of(), 10_000);
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
            assertTrue(endpoint.markQueuedExact(
                    pin, reservations.get(requestId))
                    != DecodeEndpoint.MarkQueuedResult.NOT_OWNED);
        }
    }

    private void release(long requestId) {
        DecodeEndpoint.ReservationHandle reservation =
                reservations.get(requestId);
        if (reservation != null) {
            endpoint.rollbackExact(reservation);
        }
    }

    private boolean beginFence(long requestId) {
        DecodeEndpoint.EngineFenceLease lease =
                endpoint.beginEngineFenceProtection(
                        reservations.get(requestId));
        if (lease == null) {
            return false;
        }
        fences.put(requestId, lease);
        return true;
    }

    private boolean closeFence(long requestId) {
        DecodeEndpoint.EngineFenceLease lease = fences.remove(requestId);
        if (lease == null) {
            return false;
        }
        lease.close();
        return true;
    }

    private boolean settleTombstoned(long requestId) {
        DecodeEndpoint.EngineFenceLease lease = fences.remove(requestId);
        if (lease == null) {
            return false;
        }
        endpoint.settleAuthoritativeTerminal(
                lease.authoritativeTerminalProof());
        return true;
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
                        Math.max(1, endpoint.getTotalLoad()), 0));
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
