package org.flexlb.balance.endpoint;

import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.enums.TaskPhase;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.stream.LongStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Phase 4 tests for the decode admission state of {@link DecodeEndpoint}:
 * priority-carrying reservations, the admission version, the atomic
 * release-victims-and-reserve-incoming commit (all-or-nothing, design doc
 * 11.5/17.2), and the reserved-only view after calibrate (10.1).
 */
class DecodeEndpointAdmissionTest {

    private WorkerStatus status;
    private DecodeEndpoint endpoint;

    @BeforeEach
    void setUp() {
        status = new WorkerStatus();
        status.setIp("10.0.0.1");
        status.setPort(8080);
        status.setGrpcPort(8081);
        endpoint = new DecodeEndpoint(status);
    }

    // ==================== realKvAvailable = reported - hard reservations ====================

    @Test
    void realKvAvailable_subtractsHardNotExpectedReservations() {
        updateStatus(null, null, 10_000);
        endpoint.reserve(1L, 500, 600, 70, 123L);

        // Hard (500), not expected (600), is subtracted from the report.
        assertEquals(9_500, endpoint.realKvAvailable());
        assertEquals(500, endpoint.inflightHardKvReserved());
        assertEquals(600, endpoint.inflightExpectedKvReserved());

        RequestInflight entry = endpoint.reservedView().get(1L);
        assertEquals(70, entry.priority());
        assertEquals(123L, entry.deadlineMs());
        assertEquals(DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN, entry.phase());
    }

    // ==================== reserve / release bump the admission version ====================

    @Test
    void reserveAndRelease_bumpVersion_andReverseShadowAccounting() {
        long v0 = endpoint.admissionVersion();

        endpoint.reserve(1L, 500, 600, 30, 0);
        assertEquals(v0 + 1, endpoint.admissionVersion());
        assertEquals(1, endpoint.getTotalLoad());

        endpoint.release(1L);
        assertEquals(v0 + 2, endpoint.admissionVersion());
        assertEquals(0, endpoint.getTotalLoad());
        assertEquals(0, endpoint.inflightHardKvReserved());
        assertEquals(0, endpoint.inflightExpectedKvReserved());
    }

    // ==================== atomic release+reserve: success ====================

    @Test
    void tryReleaseVictimsAndReserveIncoming_success_appliesAtomically() {
        endpoint.reserve(1L, 100, 110, 30, 1_000);
        endpoint.reserve(2L, 200, 220, 40, 2_000);
        endpoint.markQueuedPhase(1L);
        endpoint.markQueuedPhase(2L);
        long version = endpoint.admissionVersion();

        DecodeEndpoint.ReleaseReserveResult result = endpoint.tryReleaseVictimsAndReserveIncoming(
                List.of(1L, 2L), 9L, 700, 708, 70, 3_000, version);

        assertEquals(DecodeEndpoint.ReleaseReserveResult.SUCCESS, result);
        assertFalse(endpoint.reservedView().containsKey(1L));
        assertFalse(endpoint.reservedView().containsKey(2L));
        assertEquals(70, endpoint.reservedView().get(9L).priority());
        assertEquals(1, endpoint.getTotalLoad());
        assertEquals(700, endpoint.inflightHardKvReserved());
        assertTrue(endpoint.admissionVersion() > version);
    }

    // ==================== atomic release+reserve: validation failures apply nothing ====================

    @Test
    void tryReleaseVictimsAndReserveIncoming_versionMismatch_appliesNothing() {
        endpoint.reserve(1L, 100, 110, 30, 1_000);
        long staleVersion = endpoint.admissionVersion() - 1;

        DecodeEndpoint.ReleaseReserveResult result = endpoint.tryReleaseVictimsAndReserveIncoming(
                List.of(1L), 9L, 700, 708, 70, 3_000, staleVersion);

        assertEquals(DecodeEndpoint.ReleaseReserveResult.VERSION_MISMATCH, result);
        assertTrue(endpoint.reservedView().containsKey(1L));
        assertFalse(endpoint.reservedView().containsKey(9L));
        assertEquals(100, endpoint.inflightHardKvReserved());
        assertEquals(staleVersion + 1, endpoint.admissionVersion());
    }

    @Test
    void tryReleaseVictimsAndReserveIncoming_victimGone_appliesNothing() {
        endpoint.reserve(1L, 100, 110, 30, 1_000);
        endpoint.markQueuedPhase(1L);
        long version = endpoint.admissionVersion();

        DecodeEndpoint.ReleaseReserveResult result = endpoint.tryReleaseVictimsAndReserveIncoming(
                List.of(1L, 42L), 9L, 700, 708, 70, 3_000, version);

        assertEquals(DecodeEndpoint.ReleaseReserveResult.VICTIM_GONE, result);
        assertTrue(endpoint.reservedView().containsKey(1L));
        assertFalse(endpoint.reservedView().containsKey(9L));
        assertEquals(100, endpoint.inflightHardKvReserved());
        assertEquals(version, endpoint.admissionVersion());
    }

    @Test
    void dispatchAndLocalEvictionHaveOneAdmissionLockWinner() {
        // Eviction wins: it removes the reservation, so the batch item must be
        // skipped before startDispatch / gRPC publication.
        endpoint.reserve(1L, 100, 110, 30, 1_000);
        endpoint.markQueuedPhase(1L);
        assertTrue(endpoint.releaseIfHeld(1L));
        assertFalse(endpoint.tryMarkEngineMayHaveSeen(1L));

        // Dispatch wins: the queued bit is atomically cleared while the
        // reservation stays held; a later local-eviction attempt cannot touch it.
        endpoint.reserve(2L, 100, 110, 30, 1_000);
        endpoint.markQueuedPhase(2L);
        assertTrue(endpoint.tryMarkEngineMayHaveSeen(2L));
        assertFalse(endpoint.releaseIfHeld(2L));
        assertTrue(endpoint.reservedView().containsKey(2L));

        // Legacy paths never set the queued bit but still own a reservation.
        endpoint.reserve(3L, 100, 110, 30, 1_000);
        assertTrue(endpoint.tryMarkEngineMayHaveSeen(3L));
    }

    @Test
    void batchDispatchClaim_stopsAtConfiguredEngineFacingLimit() {
        // Four legacy/non-queued reservations already face the Engine.
        for (long requestId = 100; requestId < 104; requestId++) {
            endpoint.reserve(requestId, 100, 110, 30, 1_000);
        }
        // A single Prefill batch may contain many reservations which are all
        // deliberately invisible to getEngineLoad while still queued.
        for (long requestId = 1; requestId <= 20; requestId++) {
            endpoint.reserve(requestId, 100, 110, 50, 2_000);
            endpoint.markQueuedPhase(requestId);
        }

        List<DecodeEndpoint.DispatchClaimResult> results = LongStream.rangeClosed(1, 20)
                .mapToObj(requestId -> endpoint.tryClaimEngineDispatch(requestId, 5))
                .toList();

        assertEquals(DecodeEndpoint.DispatchClaimResult.CLAIMED, results.getFirst());
        assertTrue(results.subList(1, results.size()).stream()
                .allMatch(result -> result == DecodeEndpoint.DispatchClaimResult.CAPACITY_FULL));
        assertEquals(5, endpoint.getEngineLoad());
        assertEquals(19, endpoint.layeredAdmissionView().queued().size(),
                "capacity-blocked reservations must remain queued");
    }

    @Test
    void batchDispatchClaim_preservesUnlimitedAndLegacySemantics() {
        endpoint.reserve(1L, 100, 110, 30, 1_000);
        endpoint.reserve(2L, 100, 110, 30, 1_000);
        endpoint.markQueuedPhase(1L);
        endpoint.markQueuedPhase(2L);

        assertEquals(DecodeEndpoint.DispatchClaimResult.CLAIMED,
                endpoint.tryClaimEngineDispatch(1L, 0));
        assertEquals(DecodeEndpoint.DispatchClaimResult.CLAIMED,
                endpoint.tryClaimEngineDispatch(2L, 0));

        // A legacy reservation was already charged to engine-facing load;
        // retrying the claim must be idempotent even when a finite limit is full.
        endpoint.reserve(3L, 100, 110, 30, 1_000);
        assertEquals(DecodeEndpoint.DispatchClaimResult.CLAIMED,
                endpoint.tryClaimEngineDispatch(3L, 1));
    }

    @Test
    void batchDispatchClaim_reportsNotOwnedAfterReleaseOrPreemptionClaim() {
        endpoint.reserve(1L, 100, 110, 30, 1_000);
        endpoint.markQueuedPhase(1L);
        endpoint.release(1L);
        assertEquals(DecodeEndpoint.DispatchClaimResult.NOT_OWNED,
                endpoint.tryClaimEngineDispatch(1L, 5));

        endpoint.reserve(2L, 100, 110, 30, 1_000);
        long version = endpoint.admissionVersion();
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                endpoint.beginPriorityPreemption(
                        101L, List.of(2L), 9L, 100, 110,
                        70, 2_000, version, true));
        assertEquals(DecodeEndpoint.DispatchClaimResult.NOT_OWNED,
                endpoint.tryClaimEngineDispatch(2L, 5));
    }

    // ==================== 10.1: confirmed requests leave the reserved view ====================

    @Test
    void calibrate_movesConfirmedOutOfReservedView() {
        endpoint.reserve(1L, 500, 508, 30, 1_000);

        TaskInfo running = new TaskInfo();
        running.setRequestId(1L);
        running.setPhase(TaskPhase.KV_ALLOCATED);
        updateStatus(Map.of("1", running), null, 10_000);

        // Confirmed by the engine: no longer a reserved (evictable) entry,
        // but still counted in the total load via confirmedRunningCount.
        assertTrue(endpoint.reservedView().isEmpty());
        assertEquals(0, endpoint.getInflightCount());
        assertEquals(1, endpoint.getTotalLoad());
        assertEquals(0, endpoint.inflightHardKvReserved());
    }

    @Test
    void lightweightActivityRefreshRenewsConfirmedTtlWithoutInvalidatingAdmissionSnapshot()
            throws InterruptedException {
        TaskInfo running = new TaskInfo();
        running.setRequestId(1L);
        running.setPhase(TaskPhase.RUNNING);
        updateStatus(Map.of("1", running), null, 10_000);
        long admissionVersion = endpoint.admissionVersion();
        DecodeTaskPhase phase = endpoint.layeredAdmissionView().confirmed().get(0).phase();
        Thread.sleep(150);
        running.setPhase(TaskPhase.KV_ALLOCATED);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(Map.of("1", running));

        endpoint.refreshWorkerStatusActivity(status, response);

        assertEquals(admissionVersion, endpoint.admissionVersion(),
                "heartbeat-only refresh must not invalidate Auto-TPM admission snapshots");
        assertEquals(phase, endpoint.layeredAdmissionView().confirmed().get(0).phase(),
                "equal-version refresh must not mutate the versioned phase view");
        endpoint.evictExpiredRequests(100);
        assertTrue(endpoint.isConfirmedTracked(1L),
                "the refreshed confirmed task must survive inactivity eviction");
        assertEquals(admissionVersion, endpoint.admissionVersion());
        assertEquals(1, endpoint.getTotalLoad());
    }

    // ==================== helpers ====================

    private void updateStatus(Map<String, TaskInfo> running, Map<String, TaskInfo> finished,
                              long availableKvCacheTokens) {
        status.getAvailableKvCacheTokens().set(availableKvCacheTokens);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(running);
        response.setFinishedTaskInfo(finished);
        endpoint.applyWorkerStatusResponse(status, response);
    }
}
