package org.flexlb.balance.endpoint;

import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
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

    @BeforeEach
    void setUp() {
        status = new WorkerStatus();
        status.setIp("10.0.0.8");
        status.setPort(8080);
        status.setGrpcPort(8081);
        status.getTotalKvCacheTokens().set(10_000);
        endpoint = new DecodeEndpoint(status);
        updateStatus(Map.of(), Map.of(), 10_000);
    }

    @Test
    void shadowOlderThanTtlRemainsUntilFenceEnds() {
        endpoint.reserve(1L, 500, 700);
        assertTrue(endpoint.beginEngineFenceProtection(1L));
        assertTrue(endpoint.beginEngineFenceProtection(1L),
                "begin is idempotent for one live accounting generation");

        assertEquals(0, endpoint.evictExpiredRequests(-1));
        assertEquals(1, endpoint.getInflightCount());
        assertEquals(500, endpoint.inflightHardKvReserved());

        assertTrue(endpoint.endEngineFenceProtection(1L));
        assertFalse(endpoint.endEngineFenceProtection(1L));
        assertEquals(1, endpoint.evictExpiredRequests(-1));
        assertEquals(0, endpoint.getInflightCount());
        assertEquals(0, endpoint.inflightHardKvReserved());
    }

    @Test
    void confirmedMetadataOlderThanTtlRemainsUntilFenceEnds() {
        endpoint.reserve(1L, 500, 700);
        assertTrue(endpoint.beginEngineFenceProtection(1L));
        updateStatus(Map.of("1", task(1L, TaskPhase.RUNNING, 500)), Map.of(), 9_500);

        endpoint.evictExpiredRequests(-1);
        assertTrue(endpoint.isConfirmedTracked(1L));

        assertTrue(endpoint.endEngineFenceProtection(1L));
        endpoint.evictExpiredRequests(-1);
        assertFalse(endpoint.isConfirmedTracked(1L));
    }

    @Test
    void releaseAndOrdinaryTerminalClearProtectionExactlyOnce() {
        endpoint.reserve(1L, 500, 700);
        assertTrue(endpoint.beginEngineFenceProtection(1L));
        endpoint.release(1L);
        endpoint.release(1L);
        assertFalse(endpoint.endEngineFenceProtection(1L));
        assertEquals(0, endpoint.getInflightCount());

        endpoint.reserve(2L, 300, 450);
        assertTrue(endpoint.beginEngineFenceProtection(2L));
        updateStatus(Map.of(), Map.of("2", task(2L, TaskPhase.PENDING, 300)), 10_000);

        assertFalse(endpoint.endEngineFenceProtection(2L));
        assertEquals(0, endpoint.getInflightCount());
        assertEquals(0, endpoint.getTotalLoad());
        assertEquals(10_000, endpoint.realKvAvailable());
        assertEquals(1, endpoint.settledTombstoneCountForTest(),
                "a protected generation retains exactly one stale-status fence");

        updateStatus(Map.of("2", task(2L, TaskPhase.RUNNING, 300)), Map.of(), 9_700);
        assertFalse(endpoint.isConfirmedTracked(2L),
                "ordinary finished uses the same stale-status fence");
        endpoint.evictExpiredRequests(-1);
        assertEquals(0, endpoint.settledTombstoneCountForTest());
        updateStatus(Map.of("2", task(2L, TaskPhase.RUNNING, 300)), Map.of(), 9_700);
        assertTrue(endpoint.isConfirmedTracked(2L));
    }

    @Test
    void ordinaryFinishedRequestsDoNotPopulateRetainedTombstones() {
        int requestCount = 10_000;
        Map<String, TaskInfo> finished = new HashMap<>(requestCount);
        for (long requestId = 1; requestId <= requestCount; requestId++) {
            endpoint.reserve(requestId, 1, 1);
            finished.put(Long.toString(requestId),
                    task(requestId, TaskPhase.PENDING, 1));
        }

        updateStatus(Map.of(), finished, 10_000);

        assertEquals(0, endpoint.getInflightCount());
        assertEquals(0, endpoint.settledTombstoneCountForTest(),
                "ordinary throughput must not be retained for the endpoint TTL");

        updateStatus(Map.of("1", task(1L, TaskPhase.RUNNING, 1)), Map.of(), 9_999);
        assertTrue(endpoint.isConfirmedTracked(1L),
                "a later fresh active observation is not blocked by an ordinary completion");
    }

    @Test
    void priorityNotFoundOrdinaryFinishedDoesNotRetainGenerationFence() {
        endpoint.reserve(1L, 500, 700, 30);
        updateStatus(Map.of("1", task(1L, TaskPhase.RUNNING, 500)), Map.of(), 9_500);
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                endpoint.beginPriorityPreemption(101L, List.of(1L),
                        9L, 100, 120, 70,
                        endpoint.admissionVersion(), true));
        assertTrue(endpoint.markPriorityCancelInFlight(101L));
        assertTrue(endpoint.markPriorityCancelNotFound(101L, 1L));
        endpoint.abortPriorityPreemption(101L);

        assertTrue(endpoint.reconcilePriorityVictimFinished(1L));
        assertEquals(0, endpoint.settledTombstoneCountForTest(),
                "an ordinary Decode terminal is not an Engine tombstone proof");

        updateStatus(Map.of("1", task(1L, TaskPhase.RUNNING, 500)), Map.of(), 9_500);
        assertTrue(endpoint.isConfirmedTracked(1L));
    }

    @Test
    void missingConfirmedRequestTransfersToOneSyntheticSlotAndKvOwner() {
        moveProtectedRequestToMissingConfirmed(1L, 500, 700);

        assertEquals(1, endpoint.getConfirmedRunningCount());
        assertEquals(1, endpoint.getTotalLoad());
        assertTrue(endpoint.isConfirmedTracked(1L));
        assertEquals(9_500, endpoint.realKvAvailable());
        assertEquals(700, endpoint.realKvUsed());

        // Repeated absence must not add another synthetic slot or KV hold.
        updateStatus(Map.of(), Map.of(), 10_000);
        assertEquals(1, endpoint.getConfirmedRunningCount());
        assertEquals(9_500, endpoint.realKvAvailable());
        assertEquals(700, endpoint.realKvUsed());

        assertTrue(endpoint.endEngineFenceProtection(1L));
        assertEquals(0, endpoint.getConfirmedRunningCount());
        assertEquals(0, endpoint.getTotalLoad());
        assertEquals(10_000, endpoint.realKvAvailable());
        assertEquals(0, endpoint.realKvUsed());
        assertFalse(endpoint.endEngineFenceProtection(1L));

        endpoint.evictExpiredRequests(-1);
        assertFalse(endpoint.isConfirmedTracked(1L));
    }

    @Test
    void freshActiveObservationReturnsSyntheticOwnershipToEngine() {
        moveProtectedRequestToMissingConfirmed(1L, 500, 700);

        updateStatus(Map.of("1", task(1L, TaskPhase.RUNNING, 500)), Map.of(), 9_500);
        assertEquals(1, endpoint.getConfirmedRunningCount());
        assertEquals(1, endpoint.getTotalLoad());
        assertEquals(9_500, endpoint.realKvAvailable(),
                "reported engine KV replaces, rather than stacks with, the synthetic hold");

        assertTrue(endpoint.endEngineFenceProtection(1L));
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

        assertEquals(0, endpoint.getConfirmedRunningCount());
        assertEquals(0, endpoint.getTotalLoad());
        assertEquals(10_000, endpoint.realKvAvailable());
        assertEquals(0, endpoint.realKvUsed());
        assertFalse(endpoint.isConfirmedTracked(1L));
        assertFalse(endpoint.endEngineFenceProtection(1L));
    }

    @Test
    void tombstonedShadowAtomicallyClearsAccountingAndFencesStaleStatus() {
        endpoint.reserve(1L, 500, 700);
        endpoint.markQueuedPhase(1L);
        assertTrue(endpoint.beginEngineFenceProtection(1L));

        assertTrue(endpoint.settleTombstonedRequest(1L));
        assertFalse(endpoint.settleTombstonedRequest(1L),
                "duplicate settlement must not mutate counters twice");
        assertEquals(1, endpoint.settledTombstoneCountForTest());
        assertEquals(0, endpoint.getInflightCount());
        assertEquals(0, endpoint.getTotalLoad());
        assertEquals(0, endpoint.inflightHardKvReserved());
        assertEquals(0, endpoint.inflightExpectedKvReserved());
        assertTrue(endpoint.layeredAdmissionView().queued().isEmpty());
        assertFalse(endpoint.endEngineFenceProtection(1L));

        updateStatus(Map.of("1", task(1L, TaskPhase.RUNNING, 500)), Map.of(), 9_500);
        assertFalse(endpoint.isConfirmedTracked(1L),
                "a delayed active sample cannot resurrect a tombstoned request");
        assertEquals(0, endpoint.getConfirmedRunningCount());

        endpoint.evictExpiredRequests(-1);
        assertEquals(0, endpoint.settledTombstoneCountForTest());
        updateStatus(Map.of("1", task(1L, TaskPhase.RUNNING, 500)), Map.of(), 9_500);
        assertTrue(endpoint.isConfirmedTracked(1L),
                "the settled fence is bounded by the configured endpoint TTL");
    }

    @Test
    void tombstonedConfirmedOwnerClearsTrackedSlotAndGenericFence() {
        endpoint.reserve(1L, 500, 700);
        updateStatus(Map.of("1", task(1L, TaskPhase.RUNNING, 500)), Map.of(), 9_500);
        assertTrue(endpoint.beginEngineFenceProtection(1L));

        assertTrue(endpoint.settleTombstonedRequest(1L));
        assertFalse(endpoint.isConfirmedTracked(1L));
        assertEquals(0, endpoint.getConfirmedRunningCount());
        assertEquals(0, endpoint.getTotalLoad());
        assertEquals(9_500, endpoint.realKvAvailable(),
                "local settlement must not rewrite the last engine KV sample");
        assertFalse(endpoint.endEngineFenceProtection(1L));

        updateStatus(Map.of(), Map.of(), 10_000);
        assertEquals(10_000, endpoint.realKvAvailable());
    }

    @Test
    void tombstonedSyntheticOwnerReleasesSlotAndKvExactlyOnce() {
        moveProtectedRequestToMissingConfirmed(1L, 500, 700);
        assertEquals(1, endpoint.getConfirmedRunningCount());
        assertEquals(9_500, endpoint.realKvAvailable());
        assertEquals(700, endpoint.realKvUsed());

        assertTrue(endpoint.settleTombstonedRequest(1L));
        assertFalse(endpoint.isConfirmedTracked(1L));
        assertEquals(0, endpoint.getConfirmedRunningCount());
        assertEquals(0, endpoint.getTotalLoad());
        assertEquals(10_000, endpoint.realKvAvailable());
        assertEquals(0, endpoint.realKvUsed());
        assertFalse(endpoint.settleTombstonedRequest(1L));
    }

    @Test
    void genericAndPriorityFenceOwnersFormOneAccountingUnion() {
        endpoint.reserve(1L, 500, 700, 30);
        updateStatus(Map.of("1", task(1L, TaskPhase.RUNNING, 500)), Map.of(), 9_500);
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                endpoint.beginPriorityPreemption(101L, List.of(1L),
                        9L, 100, 120, 70,
                        endpoint.admissionVersion(), true));
        assertTrue(endpoint.markPriorityCancelInFlight(101L));
        assertTrue(endpoint.markPriorityCancelNotFound(101L, 1L));

        updateStatus(Map.of(), Map.of(), 10_000);
        assertTrue(endpoint.transferPriorityNotFoundClaimToEngineFence(101L, 1L));
        endpoint.abortPriorityPreemption(101L);
        assertEquals(1, endpoint.getTotalLoad());
        assertEquals(9_500, endpoint.realKvAvailable());

        assertTrue(endpoint.beginEngineFenceProtection(1L));
        assertTrue(endpoint.endEngineFenceProtection(1L));
        assertEquals(1, endpoint.getTotalLoad(),
                "ending generic ownership must not release the priority token owner");
        assertEquals(9_500, endpoint.realKvAvailable());

        assertTrue(endpoint.beginEngineFenceProtection(1L));
        assertTrue(endpoint.settleEngineFenceClaim(101L, 1L));
        assertEquals(1, endpoint.getTotalLoad(),
                "priority settlement transfers the union hold to the generic owner");
        assertEquals(9_500, endpoint.realKvAvailable());

        assertTrue(endpoint.settleTombstonedRequest(1L));
        assertEquals(0, endpoint.getTotalLoad());
        assertEquals(10_000, endpoint.realKvAvailable());
        assertFalse(endpoint.endEngineFenceProtection(1L));
        assertFalse(endpoint.settleEngineFenceClaim(101L, 1L));
    }

    private void moveProtectedRequestToMissingConfirmed(long requestId,
                                                        long hardKvTokens,
                                                        long expectedKvTokens) {
        endpoint.reserve(requestId, hardKvTokens, expectedKvTokens);
        assertTrue(endpoint.beginEngineFenceProtection(requestId));
        updateStatus(Map.of(Long.toString(requestId),
                task(requestId, TaskPhase.RUNNING, hardKvTokens)), Map.of(),
                10_000 - hardKvTokens);
        updateStatus(Map.of(), Map.of(), 10_000);
    }

    private void updateStatus(Map<String, TaskInfo> running,
                              Map<String, TaskInfo> finished,
                              long availableKvCacheTokens) {
        status.getAvailableKvCacheTokens().set(availableKvCacheTokens);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(running);
        response.setFinishedTaskInfo(finished);
        endpoint.onWorkerStatusUpdate(status, response);
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
