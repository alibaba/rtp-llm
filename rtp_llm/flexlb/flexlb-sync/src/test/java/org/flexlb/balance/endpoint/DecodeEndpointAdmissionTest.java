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
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
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
        endpoint.reserve("1", 500, 600, 70);

        // Hard (500), not expected (600), is subtracted from the report.
        assertEquals(9_500, endpoint.realKvAvailable());
        assertEquals(500, endpoint.inflightHardKvReserved());
        assertEquals(600, endpoint.inflightExpectedKvReserved());

        RequestInflight entry = endpoint.reservedView().get("1");
        assertEquals(70, entry.priority());
        assertEquals(DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN, entry.phase());
    }

    // ==================== reserve / release bump the admission version ====================

    @Test
    void reserveAndRelease_bumpVersion_andReverseShadowAccounting() {
        long v0 = endpoint.admissionVersion();

        endpoint.reserve("1", 500, 600, 30);
        assertEquals(v0 + 1, endpoint.admissionVersion());
        assertEquals(1, endpoint.getTotalLoad());

        endpoint.release("1");
        assertEquals(v0 + 2, endpoint.admissionVersion());
        assertEquals(0, endpoint.getTotalLoad());
        assertEquals(0, endpoint.inflightHardKvReserved());
        assertEquals(0, endpoint.inflightExpectedKvReserved());
    }

    @Test
    void conditionalOrphanReleasePreservesReplacementReservation() {
        long requestId = 2L;
        endpoint.reserve(String.valueOf(requestId), 100, 110, 30);
        RequestInflight staleSnapshot = endpoint.reservedView().get(String.valueOf(requestId));

        endpoint.reserve(String.valueOf(requestId), 200, 220, 70);
        RequestInflight replacement = endpoint.reservedView().get(String.valueOf(requestId));
        assertNotSame(staleSnapshot, replacement);

        assertFalse(endpoint.releaseReservationIfCurrent(
                String.valueOf(requestId), staleSnapshot));
        assertSame(replacement, endpoint.reservedView().get(String.valueOf(requestId)));
        assertEquals(200, endpoint.inflightHardKvReserved());
        assertEquals(220, endpoint.inflightExpectedKvReserved());

        assertTrue(endpoint.releaseReservationIfCurrent(
                String.valueOf(requestId), replacement));
        assertFalse(endpoint.reservedView().containsKey(String.valueOf(requestId)));
        assertEquals(0, endpoint.inflightHardKvReserved());
        assertEquals(0, endpoint.inflightExpectedKvReserved());
    }

    // ==================== atomic release+reserve: success ====================

    @Test
    void tryReleaseVictimsAndReserveIncoming_success_appliesAtomically() {
        endpoint.reserve("1", 100, 110, 30);
        endpoint.reserve("2", 200, 220, 40);
        endpoint.markQueuedPhase("1");
        endpoint.markQueuedPhase("2");
        long version = endpoint.admissionVersion();

        DecodeEndpoint.ReleaseReserveResult result = endpoint.tryReleaseVictimsAndReserveIncoming(
                List.of("1", "2"), "9", 700, 708, 70, version);

        assertEquals(DecodeEndpoint.ReleaseReserveResult.SUCCESS, result);
        assertFalse(endpoint.reservedView().containsKey("1"));
        assertFalse(endpoint.reservedView().containsKey("2"));
        assertEquals(70, endpoint.reservedView().get("9").priority());
        assertEquals(1, endpoint.getTotalLoad());
        assertEquals(700, endpoint.inflightHardKvReserved());
        assertTrue(endpoint.admissionVersion() > version);
    }

    // ==================== atomic release+reserve: validation failures apply nothing ====================

    @Test
    void tryReleaseVictimsAndReserveIncoming_versionMismatch_appliesNothing() {
        endpoint.reserve("1", 100, 110, 30);
        long staleVersion = endpoint.admissionVersion() - 1;

        DecodeEndpoint.ReleaseReserveResult result = endpoint.tryReleaseVictimsAndReserveIncoming(
                List.of("1"), "9", 700, 708, 70, staleVersion);

        assertEquals(DecodeEndpoint.ReleaseReserveResult.VERSION_MISMATCH, result);
        assertTrue(endpoint.reservedView().containsKey("1"));
        assertFalse(endpoint.reservedView().containsKey("9"));
        assertEquals(100, endpoint.inflightHardKvReserved());
        assertEquals(staleVersion + 1, endpoint.admissionVersion());
    }

    @Test
    void tryReleaseVictimsAndReserveIncoming_victimGone_appliesNothing() {
        endpoint.reserve("1", 100, 110, 30);
        endpoint.markQueuedPhase("1");
        long version = endpoint.admissionVersion();

        DecodeEndpoint.ReleaseReserveResult result = endpoint.tryReleaseVictimsAndReserveIncoming(
                List.of("1", "42"), "9", 700, 708, 70, version);

        assertEquals(DecodeEndpoint.ReleaseReserveResult.VICTIM_GONE, result);
        assertTrue(endpoint.reservedView().containsKey("1"));
        assertFalse(endpoint.reservedView().containsKey("9"));
        assertEquals(100, endpoint.inflightHardKvReserved());
        assertEquals(version, endpoint.admissionVersion());
    }

    @Test
    void dispatchAndLocalEvictionHaveOneAdmissionLockWinner() {
        // Eviction wins: it removes the reservation, so the batch item must be
        // skipped before the engine-dispatch claim / gRPC publication.
        endpoint.reserve("1", 100, 110, 30);
        endpoint.markQueuedPhase("1");
        assertTrue(endpoint.releaseIfHeld("1"));
        assertFalse(endpoint.tryMarkEngineMayHaveSeen("1"));

        // Dispatch wins: the queued bit is atomically cleared while the
        // reservation stays held; a later local-eviction attempt cannot touch it.
        endpoint.reserve("2", 100, 110, 30);
        endpoint.markQueuedPhase("2");
        assertTrue(endpoint.tryMarkEngineMayHaveSeen("2"));
        assertFalse(endpoint.releaseIfHeld("2"));
        assertTrue(endpoint.reservedView().containsKey("2"));

        // Legacy paths never set the queued bit but still own a reservation.
        endpoint.reserve("3", 100, 110, 30);
        assertTrue(endpoint.tryMarkEngineMayHaveSeen("3"));
    }

    @Test
    void batchDispatchClaim_stopsAtConfiguredEngineFacingLimit() {
        // Four legacy/non-queued reservations already face the Engine.
        for (long requestId = 100; requestId < 104; requestId++) {
            endpoint.reserve(String.valueOf(requestId), 100, 110, 30);
        }
        // A single Prefill batch may contain many reservations which are all
        // deliberately invisible to getEngineLoad while still queued.
        for (long requestId = 1; requestId <= 20; requestId++) {
            endpoint.reserve(String.valueOf(requestId), 100, 110, 50);
            endpoint.markQueuedPhase(String.valueOf(requestId));
        }

        List<DecodeEndpoint.DispatchClaimResult> results = LongStream.rangeClosed(1, 20)
                .mapToObj(requestId -> endpoint.tryClaimEngineDispatch(String.valueOf(requestId), 5))
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
        endpoint.reserve("1", 100, 110, 30);
        endpoint.reserve("2", 100, 110, 30);
        endpoint.markQueuedPhase("1");
        endpoint.markQueuedPhase("2");

        assertEquals(DecodeEndpoint.DispatchClaimResult.CLAIMED,
                endpoint.tryClaimEngineDispatch("1", 0));
        assertEquals(DecodeEndpoint.DispatchClaimResult.CLAIMED,
                endpoint.tryClaimEngineDispatch("2", 0));

        // A legacy reservation was already charged to engine-facing load;
        // retrying the claim must be idempotent even when a finite limit is full.
        endpoint.reserve("3", 100, 110, 30);
        assertEquals(DecodeEndpoint.DispatchClaimResult.CLAIMED,
                endpoint.tryClaimEngineDispatch("3", 1));
    }

    @Test
    void batchDispatchClaim_reportsNotOwnedAfterReleaseOrPreemptionClaim() {
        endpoint.reserve("1", 100, 110, 30);
        endpoint.markQueuedPhase("1");
        endpoint.release("1");
        assertEquals(DecodeEndpoint.DispatchClaimResult.NOT_OWNED,
                endpoint.tryClaimEngineDispatch("1", 5));

        endpoint.reserve("2", 100, 110, 30);
        long version = endpoint.admissionVersion();
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                endpoint.beginPriorityPreemption(
                        101L, List.of("2"), "9", 100, 110,
                        70, version, true));
        assertEquals(DecodeEndpoint.DispatchClaimResult.NOT_OWNED,
                endpoint.tryClaimEngineDispatch("2", 5));
    }

    // ==================== 10.1: confirmed requests leave the reserved view ====================

    @Test
    void calibrate_movesConfirmedOutOfReservedView() {
        endpoint.reserve("1", 500, 508, 30);

        TaskInfo running = new TaskInfo();
        running.setRequestId("1");
        running.setPhase(TaskPhase.KV_ALLOCATED);
        updateStatus(Map.of("1", running), null, 10_000);

        // Confirmed by the engine: no longer a reserved (evictable) entry,
        // but still counted in the total load via confirmedRunningCount.
        assertTrue(endpoint.reservedView().isEmpty());
        assertEquals(0, endpoint.getInflightCount());
        assertEquals(1, endpoint.getTotalLoad());
        assertEquals(0, endpoint.inflightHardKvReserved());
    }

    // ==================== helpers ====================

    private void updateStatus(Map<String, TaskInfo> running, Map<String, TaskInfo> finished,
                              long availableKvCacheTokens) {
        status.getAvailableKvCacheTokens().set(availableKvCacheTokens);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(running);
        response.setFinishedTaskInfo(finished);
        endpoint.onWorkerStatusUpdate(status, response);
    }
}
