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
        assertEquals(DecodeTaskPhase.RESERVED_NOT_ACCEPTED, entry.phase());
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
        long version = endpoint.admissionVersion();

        DecodeEndpoint.ReleaseReserveResult result = endpoint.tryReleaseVictimsAndReserveIncoming(
                List.of(1L, 42L), 9L, 700, 708, 70, 3_000, version);

        assertEquals(DecodeEndpoint.ReleaseReserveResult.VICTIM_GONE, result);
        assertTrue(endpoint.reservedView().containsKey(1L));
        assertFalse(endpoint.reservedView().containsKey(9L));
        assertEquals(100, endpoint.inflightHardKvReserved());
        assertEquals(version, endpoint.admissionVersion());
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
