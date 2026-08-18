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
 * the token-fenced weak-ACK transaction is all-or-nothing, retains victim
 * accounting until typed CANCELED, and provisionally reserves the incoming.
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
        assertFalse(acceptedView.claimedForPreemption());

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
        assertFalse(view.priorityKnown());
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
        assertEquals(0, endpoint.getConfirmedRunningCount(),
                "tracked TTL removal must release the published confirmed slot");
        assertEquals(0, endpoint.getTotalLoad());
        assertTrue(endpoint.admissionVersion() > versionBefore);
    }

    @Test
    void evictExpiredRequests_boundsPriorityCanceledTombstones() throws InterruptedException {
        endpoint.reserve(1L, 500, 508, 30, 1_000);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 256)), null, 10_000);
        long version = endpoint.admissionVersion();
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                endpoint.beginPriorityPreemption(101L, List.of(1L),
                        9L, 128, 136, 70, 2_000, version, true));
        assertTrue(endpoint.markPriorityCancelInFlight(101L));
        assertTrue(endpoint.markPriorityCancelAccepted(101L, 1L));
        assertTrue(endpoint.settlePriorityCanceled(101L, 1L));
        assertTrue(endpoint.commitPriorityPreemption(101L));

        // A delayed Decode report cannot resurrect a recently canceled victim.
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 256)), null, 10_000);
        assertFalse(endpoint.isConfirmedTracked(1L));

        Thread.sleep(5);
        endpoint.evictExpiredRequests(1);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 256)), null, 10_000);
        assertTrue(endpoint.isConfirmedTracked(1L),
                "the cancel fence follows the configured terminal retention TTL");
    }

    @Test
    void priorityTombstoneIsAuthoritativeWithoutAcceptedOrWorkerCanceled() {
        endpoint.reserve(1L, 500, 508, 30, 1_000);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 256)), null, 10_000);
        long version = endpoint.admissionVersion();
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                endpoint.beginPriorityPreemption(102L, List.of(1L),
                        9L, 128, 136, 70, 2_000, version, true));
        assertTrue(endpoint.markPriorityCancelInFlight(102L));

        assertTrue(endpoint.settlePriorityTombstoned(102L, 1L));
        assertTrue(endpoint.commitPriorityPreemption(102L));

        assertFalse(endpoint.isConfirmedTracked(1L));
        assertTrue(endpoint.reservedView().containsKey(9L));
        assertEquals(1, endpoint.getTotalLoad());
        // The same late Decode sample rejected by typed-CANCELED fencing must
        // also be rejected after the stronger absent+tombstone proof.
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 256)), null, 10_000);
        assertFalse(endpoint.isConfirmedTracked(1L));
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

    // ==================== token-fenced weak-ACK preemption ====================

    @Test
    void beginPriorityPreemption_claimsVictimAndProvisionallyReservesIncoming() {
        endpoint.reserve(2L, 400, 408, 30, 2_000);
        updateStatus(Map.of("2", runningTask(2L, TaskPhase.KV_ALLOCATED, 256)), null, 10_000);
        long version = endpoint.admissionVersion();

        DecodeEndpoint.PreemptionBeginResult result = endpoint.beginPriorityPreemption(
                101L, List.of(2L), 9L, 700, 708, 70, 3_000,
                version, true);

        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS, result);
        // Weak ACK boundary: victim accounting is untouched and the incoming
        // reservation is provisional until typed Prefill CANCELED settles it.
        assertTrue(endpoint.isConfirmedTracked(2L));
        assertTrue(confirmedView(2L).claimedForPreemption());
        assertTrue(endpoint.reservedView().containsKey(9L));
        assertEquals(700, endpoint.inflightHardKvReserved());
        assertTrue(endpoint.markPriorityCancelInFlight(101L));
        assertTrue(endpoint.markPriorityCancelAccepted(101L, 2L));
        assertTrue(endpoint.isConfirmedTracked(2L));
        assertTrue(endpoint.admissionVersion() > version);
    }

    @Test
    void beginPriorityPreemption_versionMismatch_appliesNothing() {
        endpoint.reserve(1L, 500, 508, 30, 1_000);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.KV_ALLOCATED, 256)), null, 10_000);

        DecodeEndpoint.PreemptionBeginResult result = endpoint.beginPriorityPreemption(
                101L, List.of(1L), 9L, 700, 708, 70, 3_000,
                endpoint.admissionVersion() - 1, true);

        assertEquals(DecodeEndpoint.PreemptionBeginResult.VERSION_MISMATCH, result);
        assertFalse(confirmedView(1L).claimedForPreemption());
        assertFalse(endpoint.reservedView().containsKey(9L));
    }

    @Test
    void beginPriorityPreemption_victimGone_isAllOrNothing() {
        endpoint.reserve(2L, 400, 408, 30, 2_000);
        updateStatus(Map.of("2", runningTask(2L, TaskPhase.KV_ALLOCATED, 256)), null, 10_000);
        long version = endpoint.admissionVersion();

        assertEquals(DecodeEndpoint.PreemptionBeginResult.VICTIM_GONE,
                endpoint.beginPriorityPreemption(101L, List.of(2L, 999L),
                        9L, 700, 708, 70, 3_000, version, true));
        assertFalse(confirmedView(2L).claimedForPreemption());
        assertFalse(endpoint.reservedView().containsKey(9L));
        assertEquals(version, endpoint.admissionVersion());
    }

    @Test
    void beginPriorityPreemption_acceptsRunningAndRejectsAlreadyClaimedVictims() {
        endpoint.reserve(1L, 500, 508, 30, 1_000);
        endpoint.reserve(2L, 400, 408, 30, 2_000);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 256),
                "2", runningTask(2L, TaskPhase.KV_ALLOCATED, 256)), null, 10_000);

        // RUNNING is engine-owned too and follows the same cancel path.
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                endpoint.beginPriorityPreemption(101L, List.of(1L),
                        9L, 700, 708, 70, 3_000,
                        endpoint.admissionVersion(), true));

        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                endpoint.beginPriorityPreemption(102L, List.of(2L),
                        10L, 700, 708, 70, 3_000,
                        endpoint.admissionVersion(), true));
        assertEquals(DecodeEndpoint.PreemptionBeginResult.VICTIM_ALREADY_CLAIMED,
                endpoint.beginPriorityPreemption(103L, List.of(2L),
                        11L, 700, 708, 70, 3_000,
                endpoint.admissionVersion(), true));
    }

    @Test
    void ttlEvictionCannotReleaseClaimedEngineVisibleShadow() throws Exception {
        endpoint.reserve(1L, 500, 508, 30, 1_000);
        // Keep a wide age gap so the provisional incoming reservation cannot
        // become TTL-eligible merely because this test runs on a loaded JVM.
        Thread.sleep(150);
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                endpoint.beginPriorityPreemption(101L, List.of(1L),
                        9L, 700, 708, 70, 3_000,
                        endpoint.admissionVersion(), true));
        assertTrue(endpoint.markPriorityCancelInFlight(101L));

        assertEquals(0, endpoint.evictExpiredRequests(100));
        assertTrue(endpoint.reservedView().containsKey(1L),
                "generic TTL cleanup must not deduct a claimed victim");
        assertEquals(1_200, endpoint.inflightHardKvReserved(),
                "victim and provisional incoming remain fully charged");

        endpoint.abortPriorityPreemption(101L);
        assertEquals(1, endpoint.evictExpiredRequests(100));
        assertFalse(endpoint.reservedView().containsKey(1L));
        assertEquals(0, endpoint.inflightHardKvReserved());
    }

    @Test
    void activeAfterNotFoundReleasesSyntheticHeldKvWithClaim() {
        endpoint.reserve(1L, 500, 508, 30, 1_000);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 500)), null, 10_000);
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                endpoint.beginPriorityPreemption(101L, List.of(1L),
                        9L, 700, 708, 70, 3_000,
                        endpoint.admissionVersion(), true));
        assertTrue(endpoint.markPriorityCancelInFlight(101L));
        assertTrue(endpoint.markPriorityCancelNotFound(101L, 1L));
        endpoint.abortPriorityPreemption(101L);

        // Decode disappears while NOT_FOUND is being reconciled. Its KV is
        // conservatively held until the original Prefill reports it active.
        updateStatus(Map.of(), null, 10_000);
        assertEquals(9_500, endpoint.realKvAvailable());

        assertTrue(endpoint.reconcilePriorityVictimActive(1L));
        assertEquals(10_000, endpoint.realKvAvailable(),
                "active reconciliation must release held KV before dropping the claim");
        assertFalse(confirmedView(1L).claimedForPreemption());
    }

    @Test
    void notFoundTransferRetainsSyntheticKvUntilExactEngineFenceSettlement() {
        endpoint.reserve(1L, 500, 508, 30, 1_000);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 500)), null, 10_000);
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                endpoint.beginPriorityPreemption(104L, List.of(1L),
                        9L, 700, 708, 70, 3_000,
                        endpoint.admissionVersion(), true));
        assertTrue(endpoint.markPriorityCancelInFlight(104L));
        assertTrue(endpoint.markPriorityCancelNotFound(104L, 1L));

        // Decode disappearance moves the victim's 500-token charge into a
        // synthetic hold. The 700-token provisional incoming reservation is
        // still independently owned by the live preemption attempt.
        updateStatus(Map.of(), null, 10_000);
        assertEquals(700, endpoint.inflightHardKvReserved());
        assertEquals(2, endpoint.getTotalLoad(),
                "victim and provisional incoming must both remain charged before abort");
        assertEquals(8_800, endpoint.realKvAvailable());
        assertTrue(endpoint.transferPriorityNotFoundClaimToEngineFence(104L, 1L));
        endpoint.abortPriorityPreemption(104L);

        assertEquals(0, endpoint.inflightHardKvReserved(),
                "aborting the attempt releases only its provisional incoming reservation");
        assertEquals(9_500, endpoint.realKvAvailable(),
                "control-owner transfer must not release the synthetic KV hold");
        assertEquals(1, endpoint.getTotalLoad(),
                "the disappeared confirmed victim remains a synthetic slot");

        assertTrue(endpoint.settleEngineFenceClaim(104L, 1L));
        assertEquals(10_000, endpoint.realKvAvailable());
        assertEquals(0, endpoint.getTotalLoad());
        assertFalse(endpoint.settleEngineFenceClaim(104L, 1L),
                "the exact fence generation settles accounting at most once");
        assertEquals(10_000, endpoint.realKvAvailable());
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
        endpoint.beginPriorityPreemption(101L, List.of(2L),
                20L, 64, 72, 70, 4_000, endpoint.admissionVersion(), true);
        endpoint.beginPriorityPreemption(102L, List.of(3L),
                30L, 64, 72, 70, 4_000, endpoint.admissionVersion(), true);
        DecodeEndpointSnapshot after = DecodeEndpointSnapshot.capture(endpoint, 4);
        assertTrue(after.accepted().isEmpty());
        assertTrue(after.running().isEmpty());
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
