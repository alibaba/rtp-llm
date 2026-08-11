package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.priority.DecodeEndpointSnapshot;
import org.flexlb.balance.scheduler.priority.DecodeRequestSnapshot;
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
 * Phase 5 tests for the decode layered view: calibrate splits confirmed
 * requests into accepted / running layers with priority-deadline inheritance,
 * the registry follows the WorkerStatus reports (retain / finished / TTL),
 * the shadow accounting invariants stay byte-for-byte Phase 4, and
 * {@code tryBeginAcceptedEviction} is all-or-nothing and never reserves the
 * incoming request (iron rule 4).
 */
class DecodeEndpointLayeredViewTest {

    private WorkerStatus status;
    private DecodeEndpoint endpoint;

    @BeforeEach
    void setUp() {
        status = new WorkerStatus();
        status.setIp("10.0.0.1");
        status.setPort(8080);
        status.setGrpcPort(8081);
        status.getTotalKvCacheTokens().set(20_000);
        endpoint = new DecodeEndpoint(status);
    }

    // ==================== calibrate: layer split + inheritance ====================

    @Test
    void calibrate_splitsConfirmedIntoAcceptedAndRunningLayers() {
        endpoint.reserve(1L, 500, 508, 30, 1_000);
        endpoint.reserve(2L, 500, 508, 40, 2_000);

        TaskInfo accepted = runningTask(1L, TaskPhase.KV_ALLOCATED, 256);
        TaskInfo running = runningTask(2L, TaskPhase.RUNNING, 512);
        updateStatus(Map.of("1", accepted, "2", running), null, 10_000);

        assertEquals(1, endpoint.getAcceptedLayerCount());
        assertEquals(1, endpoint.getRunningLayerCount());
        assertEquals(2, endpoint.getConfirmedRunningCount());
        assertEquals(0, endpoint.getInflightCount());
        assertTrue(endpoint.isConfirmedTracked(1L));
        assertTrue(endpoint.isConfirmedTracked(2L));

        // Layered view inherits priority/deadline from the shadow entry
        // removed this round; KV is the reported inputLength estimate.
        DecodeEndpoint.ConfirmedTaskView acceptedView = confirmedView(1L);
        assertEquals(DecodeTaskPhase.ACCEPTED_NOT_RUNNING, acceptedView.phase());
        assertEquals(30, acceptedView.priority());
        assertEquals(1_000, acceptedView.deadlineMs());
        assertEquals(256, acceptedView.kvTokens());
        assertFalse(acceptedView.cancelRequested());

        DecodeEndpoint.ConfirmedTaskView runningView = confirmedView(2L);
        assertEquals(DecodeTaskPhase.RUNNING, runningView.phase());
        assertEquals(40, runningView.priority());
    }

    @Test
    void calibrate_unknownConfirmedFallsBackToNoPriority() {
        // Report precedes any reserve: no shadow entry to inherit from.
        // Task40: the fallback is the no-priority sentinel (0), which keeps
        // untracked engine tasks out of every eviction candidate set.
        updateStatus(Map.of("9", runningTask(9L, TaskPhase.KV_ALLOCATED, 64)), null, 10_000);

        DecodeEndpoint.ConfirmedTaskView view = confirmedView(9L);
        assertEquals(0, view.priority());
        assertEquals(0, view.deadlineMs());
        assertEquals(64, view.kvTokens());
    }

    @Test
    void calibrate_promotesAcceptedToRunningOnRefresh() {
        endpoint.reserve(1L, 500, 508, 30, 1_000);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.KV_ALLOCATED, 256)), null, 10_000);
        assertEquals(1, endpoint.getAcceptedLayerCount());

        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 256)), null, 10_000);

        assertEquals(0, endpoint.getAcceptedLayerCount());
        assertEquals(1, endpoint.getRunningLayerCount());
        // Identity fields stay from first sight.
        assertEquals(30, confirmedView(1L).priority());
    }

    // ==================== calibrate: registry follows the reports ====================

    @Test
    void calibrate_dropsEntriesNoLongerReported() {
        endpoint.reserve(1L, 500, 508, 30, 1_000);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.KV_ALLOCATED, 256)), null, 10_000);
        assertTrue(endpoint.isConfirmedTracked(1L));

        // Next report no longer lists the request as confirmed — this is the
        // release-confirmation signal the accepted-eviction wait polls for.
        updateStatus(Map.of(), null, 10_000);

        assertFalse(endpoint.isConfirmedTracked(1L));
        assertEquals(0, endpoint.getAcceptedLayerCount());
        assertEquals(0, endpoint.getConfirmedRunningCount());
    }

    @Test
    void calibrate_finishedRemovesTrackedEntry() {
        endpoint.reserve(1L, 500, 508, 30, 1_000);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 256)), null, 10_000);
        assertTrue(endpoint.isConfirmedTracked(1L));

        // Same round lists it both running and finished: finished wins.
        TaskInfo finished = runningTask(1L, TaskPhase.RUNNING, 256);
        finished.setErrorCode(0);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 256)),
                Map.of("1", finished), 10_000);

        assertFalse(endpoint.isConfirmedTracked(1L));
    }

    @Test
    void evictExpiredRequests_purgesStaleTrackedEntries() throws InterruptedException {
        endpoint.reserve(1L, 500, 508, 30, 1_000);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.KV_ALLOCATED, 256)), null, 10_000);
        assertTrue(endpoint.isConfirmedTracked(1L));

        Thread.sleep(5);
        long versionBefore = endpoint.admissionVersion();
        endpoint.evictExpiredRequests(1);

        assertFalse(endpoint.isConfirmedTracked(1L));
        assertTrue(endpoint.admissionVersion() > versionBefore);
    }

    // ==================== accounting invariants unchanged (iron rule 5) ====================

    @Test
    void accounting_invariantsStayPhase4Equivalent() {
        endpoint.reserve(1L, 500, 508, 30, 1_000);
        endpoint.reserve(2L, 300, 308, 40, 2_000);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.KV_ALLOCATED, 256)), null, 10_000);

        // realKvAvailable = reportedKvAvailable - remaining shadow hard KV.
        assertEquals(10_000 - 300, endpoint.realKvAvailable());
        assertEquals(300, endpoint.inflightHardKvReserved());
        // totalLoad = confirmedRunningCount + reserved inflight count.
        assertEquals(2, endpoint.getTotalLoad());
        assertEquals(1, endpoint.getInflightCount());
    }

    // ==================== tryBeginAcceptedEviction ====================

    @Test
    void beginAcceptedEviction_marksCancelAndReleasesReserved_neverReservesIncoming() {
        endpoint.reserve(1L, 500, 508, 30, 1_000);
        endpoint.reserve(2L, 400, 408, 30, 2_000);
        updateStatus(Map.of("2", runningTask(2L, TaskPhase.KV_ALLOCATED, 256)), null, 10_000);
        long version = endpoint.admissionVersion();

        DecodeEndpoint.ReleaseReserveResult result =
                endpoint.tryBeginAcceptedEviction(List.of(1L), List.of(2L), version);

        assertEquals(DecodeEndpoint.ReleaseReserveResult.SUCCESS, result);
        // Reserved victim released, accepted victim marked but still tracked.
        assertFalse(endpoint.reservedView().containsKey(1L));
        assertTrue(endpoint.isConfirmedTracked(2L));
        assertTrue(confirmedView(2L).cancelRequested());
        // Iron rule 4: the incoming is NOT reserved by begin.
        assertEquals(0, endpoint.getInflightCount());
        assertEquals(0, endpoint.inflightHardKvReserved());
        assertTrue(endpoint.admissionVersion() > version);
    }

    @Test
    void beginAcceptedEviction_versionMismatch_appliesNothing() {
        endpoint.reserve(1L, 500, 508, 30, 1_000);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.KV_ALLOCATED, 256)), null, 10_000);

        DecodeEndpoint.ReleaseReserveResult result = endpoint.tryBeginAcceptedEviction(
                List.of(), List.of(1L), endpoint.admissionVersion() - 1);

        assertEquals(DecodeEndpoint.ReleaseReserveResult.VERSION_MISMATCH, result);
        assertFalse(confirmedView(1L).cancelRequested());
    }

    @Test
    void beginAcceptedEviction_victimGone_isAllOrNothing() {
        endpoint.reserve(1L, 500, 508, 30, 1_000);
        endpoint.reserve(2L, 400, 408, 30, 2_000);
        updateStatus(Map.of("2", runningTask(2L, TaskPhase.KV_ALLOCATED, 256)), null, 10_000);
        long version = endpoint.admissionVersion();

        // Unknown accepted victim: nothing applied — the reserved victim survives.
        assertEquals(DecodeEndpoint.ReleaseReserveResult.VICTIM_GONE,
                endpoint.tryBeginAcceptedEviction(List.of(1L), List.of(999L), version));
        assertTrue(endpoint.reservedView().containsKey(1L));
        assertFalse(confirmedView(2L).cancelRequested());
        assertEquals(version, endpoint.admissionVersion());
    }

    @Test
    void beginAcceptedEviction_rejectsRunningAndAlreadyCancelledVictims() {
        endpoint.reserve(1L, 500, 508, 30, 1_000);
        endpoint.reserve(2L, 400, 408, 30, 2_000);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 256),
                "2", runningTask(2L, TaskPhase.KV_ALLOCATED, 256)), null, 10_000);

        // RUNNING-layer entry is never a valid accepted victim.
        assertEquals(DecodeEndpoint.ReleaseReserveResult.VICTIM_GONE,
                endpoint.tryBeginAcceptedEviction(List.of(), List.of(1L),
                        endpoint.admissionVersion()));

        // A pending cancel dedups a second eviction claiming the same victim.
        assertEquals(DecodeEndpoint.ReleaseReserveResult.SUCCESS,
                endpoint.tryBeginAcceptedEviction(List.of(), List.of(2L),
                        endpoint.admissionVersion()));
        assertEquals(DecodeEndpoint.ReleaseReserveResult.VICTIM_GONE,
                endpoint.tryBeginAcceptedEviction(List.of(), List.of(2L),
                        endpoint.admissionVersion()));
    }

    // ==================== snapshot capture: layered lists ====================

    @Test
    void snapshotCapture_splitsLayersAndExcludesCancelRequested() {
        endpoint.reserve(1L, 500, 508, 30, 1_000);
        endpoint.reserve(2L, 400, 408, 30, 2_000);
        endpoint.reserve(3L, 300, 308, 30, 3_000);
        updateStatus(Map.of("2", runningTask(2L, TaskPhase.KV_ALLOCATED, 256),
                "3", runningTask(3L, TaskPhase.RUNNING, 512)), null, 10_000);

        DecodeEndpointSnapshot snapshot = DecodeEndpointSnapshot.capture(endpoint, 4);
        assertEquals(List.of(1L), ids(snapshot.reserved()));
        assertEquals(List.of(2L), ids(snapshot.accepted()));
        assertEquals(List.of(3L), ids(snapshot.running()));
        DecodeRequestSnapshot accepted = snapshot.accepted().get(0);
        assertEquals(DecodeTaskPhase.ACCEPTED_NOT_RUNNING, accepted.phase());
        assertEquals(256, accepted.kvTokens());

        // A cancel-requested entry is claimed by an in-flight eviction and
        // must not be offered to planning again.
        endpoint.tryBeginAcceptedEviction(List.of(), List.of(2L), endpoint.admissionVersion());
        DecodeEndpointSnapshot after = DecodeEndpointSnapshot.capture(endpoint, 4);
        assertTrue(after.accepted().isEmpty());
        assertEquals(List.of(3L), ids(after.running()));
    }

    // ==================== helpers ====================

    private static List<Long> ids(List<DecodeRequestSnapshot> entries) {
        return entries.stream().map(DecodeRequestSnapshot::requestId).toList();
    }

    private DecodeEndpoint.ConfirmedTaskView confirmedView(long requestId) {
        return endpoint.layeredAdmissionView().confirmed().stream()
                .filter(view -> view.requestId() == requestId)
                .findFirst()
                .orElseThrow(() -> new AssertionError("request " + requestId + " not tracked"));
    }

    private void updateStatus(Map<String, TaskInfo> running, Map<String, TaskInfo> finished,
                              long availableKvCacheTokens) {
        status.getAvailableKvCacheTokens().set(availableKvCacheTokens);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(running);
        response.setFinishedTaskInfo(finished);
        endpoint.onWorkerStatusUpdate(status, response);
    }

    private static TaskInfo runningTask(long requestId, TaskPhase phase, long inputLength) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setPhase(phase);
        task.setInputLength(inputLength);
        return task;
    }
}
