package org.flexlb.balance.endpoint;

import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.enums.TaskPhase;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DecodeEndpointTest {

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

    @Test
    void reserve_updatesSnapshotAndInflight() {
        updateStatus(null, null, 10000);
        endpoint.reserve(100L, 500, 500);
        assertEquals(1, endpoint.getInflightCount());
        assertEquals(9500, endpoint.realKvAvailable());
    }

    @Test
    void release_decrementsInflight() {
        endpoint.reserve(100L, 500, 500);
        endpoint.reserve(101L, 300, 300);
        endpoint.release(100L);

        assertEquals(1, endpoint.getInflightCount());
    }

    @Test
    void release_unknownRequestId_noEffect() {
        endpoint.reserve(100L, 500, 500);
        endpoint.release(999L);
        assertEquals(1, endpoint.getInflightCount());
    }

    @Test
    void release_neverGoesNegative() {
        endpoint.reserve(100L, 100, 100);
        endpoint.release(100L);
        endpoint.release(100L);
        assertEquals(0, endpoint.getInflightCount());
        assertEquals(0, endpoint.realKvAvailable());
    }

    @Test
    void calibrate_kvAllocatedReleasesFromInflight() {
        endpoint.reserve(100L, 500, 500);

        TaskInfo running = task(100L);
        running.setPhase(TaskPhase.KV_ALLOCATED);
        updateStatus(Map.of("100", running), null, 10000);

        assertEquals(0, endpoint.getInflightCount());
        assertEquals(10000, endpoint.realKvAvailable());
    }

    @Test
    void calibrate_finishedFailureReleasesFromInflight() {
        endpoint.reserve(100L, 500, 500);

        TaskInfo failed = task(100L);
        failed.setErrorCode(1);
        failed.setErrorMessage("timeout");
        updateStatus(null, Map.of("100", failed), 10000);

        assertEquals(0, endpoint.getInflightCount());
    }

    @Test
    void calibrate_finishedSuccessReleasesIfStillPresent() {
        endpoint.reserve(100L, 500, 500);

        TaskInfo success = task(100L);
        success.setErrorCode(0);
        updateStatus(null, Map.of("100", success), 10000);

        assertEquals(0, endpoint.getInflightCount());
    }

    @Test
    void calibrate_updatesReportedKvAvailable() {
        endpoint.reserve(100L, 500, 500);
        updateStatus(null, null, 10000);

        assertEquals(9500, endpoint.realKvAvailable());
    }

    @Test
    void availableKvTokens_accountsForReservations() {
        updateStatus(null, null, 10000);

        endpoint.reserve(100L, 3000, 3000);
        endpoint.reserve(101L, 2000, 2000);

        assertEquals(5000, endpoint.realKvAvailable());
    }

    @Test
    void ipPort_format() {
        assertEquals("10.0.0.1:8080", endpoint.ipPort());
    }

    // ==================== PR-C: getEngineLoad O(1) + queuedPhaseCount drift ===========

    @Test
    void getEngineLoad_o1_tracks_markQueued_and_markDispatched() {
        updateStatus(null, null, 10000);
        endpoint.reserve(1L, 100, 100);
        endpoint.reserve(2L, 200, 200);
        endpoint.reserve(3L, 300, 300);
        // No queued phase: engineLoad == totalLoad == inflight(3)
        assertEquals(3, endpoint.getEngineLoad());
        assertEquals(endpoint.getTotalLoad(), endpoint.getEngineLoad());

        // mark req 1,2 as queued → engine-facing load drops to 1
        endpoint.markQueuedPhase(1L);
        endpoint.markQueuedPhase(2L);
        assertEquals(1, endpoint.getEngineLoad());

        // dispatch req 1 → back to engine load 2
        assertTrue(endpoint.tryMarkEngineMayHaveSeen(1L));
        assertEquals(2, endpoint.getEngineLoad());

        // release req 2 (was queued) → inflight=2, queued=0
        endpoint.release(2L);
        assertEquals(2, endpoint.getEngineLoad());

        // release req 1 → inflight=1
        endpoint.release(1L);
        assertEquals(1, endpoint.getEngineLoad());

        // release req 3 → 0
        endpoint.release(3L);
        assertEquals(0, endpoint.getEngineLoad());
    }

    @Test
    void getEngineLoad_calibrate_prunes_queued_phase_count() {
        endpoint.reserve(1L, 100, 100);
        endpoint.reserve(2L, 100, 100);
        endpoint.markQueuedPhase(1L);
        endpoint.markQueuedPhase(2L);
        assertEquals(0, endpoint.getEngineLoad()); // both queued

        // calibrate: req 1 confirmed → removed from inflight + queued
        TaskInfo running = task(1L);
        running.setPhase(TaskPhase.KV_ALLOCATED);
        updateStatus(Map.of("1", running), null, 10000);

        // req 2 still queued, inflight=1 (req2), confirmed=1 (req1)
        // engineLoad = confirmed(1) + max(0, inflight(1) - queued(1)) = 1
        assertEquals(1, endpoint.getEngineLoad());
    }

    @Test
    void getEngineLoad_idempotent_markQueued_does_not_double_count() {
        endpoint.reserve(1L, 100, 100);
        endpoint.markQueuedPhase(1L);
        endpoint.markQueuedPhase(1L); // idempotent: add returns false
        assertEquals(0, endpoint.getEngineLoad());
    }

    @Test
    void getEngineLoad_idempotent_markDispatched_does_not_over_decrement() {
        endpoint.reserve(1L, 100, 100);
        endpoint.markQueuedPhase(1L);
        assertTrue(endpoint.tryMarkEngineMayHaveSeen(1L));
        assertTrue(endpoint.tryMarkEngineMayHaveSeen(1L)); // already engine-visible
        assertEquals(1, endpoint.getEngineLoad());
    }

    @Test
    void getEngineLoad_clamps_negative_drift_to_zero() throws Exception {
        endpoint.reserve(1L, 100, 100);
        setQueuedPhaseCount(-5);
        // inflight=1, queued clamped from -5 to 0 → engineLoad = 0 + max(0,1-0) = 1
        assertEquals(1, endpoint.getEngineLoad());
    }

    @Test
    void getEngineLoad_clamps_overflow_drift_to_inflight() throws Exception {
        endpoint.reserve(1L, 100, 100);
        endpoint.reserve(2L, 100, 100);
        setQueuedPhaseCount(100);
        // inflight=2, queued clamped from 100 to 2 → engineLoad = 0 + max(0,2-2) = 0
        assertEquals(0, endpoint.getEngineLoad());
    }

    // ==================== P1-6: evictExpiredRequests counter (PR-C) ====================

    /**
     * evictExpiredRequests must prune queuedPhase entries that are no longer in
     * inflightRequests and decrement queuedPhaseCount accordingly, so that
     * getEngineLoad() returns 0 after all entries are evicted.
     */
    @Test
    void evictExpiredRequests_prunesQueuedPhase_andRestoresEngineLoad() throws InterruptedException {
        updateStatus(null, null, 10000);
        endpoint.reserve(100L, 500, 500);
        endpoint.reserve(101L, 500, 500);
        endpoint.reserve(102L, 500, 500);
        endpoint.markQueuedPhase(100L);
        endpoint.markQueuedPhase(101L);
        endpoint.markQueuedPhase(102L);

        assertEquals(3, endpoint.getInflightCount());
        assertEquals(0, endpoint.getEngineLoad());
        assertEquals(3, endpoint.getTotalLoad());
        assertEquals(3, endpoint.layeredAdmissionView().queued().size());

        Thread.sleep(20);
        int evicted = endpoint.evictExpiredRequests(5);

        assertEquals(3, evicted);
        assertEquals(0, endpoint.getInflightCount());
        assertTrue(endpoint.layeredAdmissionView().queued().isEmpty());
        assertEquals(0, endpoint.getEngineLoad());
        assertEquals(0, endpoint.getTotalLoad());
    }

    // ==================== hard age cap (zombie preemption claims) ====================

    @Test
    void hardAgeCap_evictsClaimExemptedEntry_andReleasesCounters() throws InterruptedException {
        updateStatus(null, null, 10_000);
        endpoint.reserve(100L, 500, 500, 5, 0);
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                endpoint.beginPriorityPreemption(1L, java.util.List.of(100L), 200L,
                        100, 100, 10, 0, 0, false));
        Thread.sleep(20);

        // Regular TTL pass: the claim exempts the victim; only the unclaimed
        // incoming reservation expires.
        assertEquals(1, endpoint.evictExpiredRequests(5));
        assertEquals(1, endpoint.getInflightCount());

        // The hard cap force-releases the zombie claim and its accounting.
        assertEquals(1, endpoint.evictExpiredRequests(60_000, 5, requestId -> false));
        assertEquals(0, endpoint.getInflightCount());
        assertEquals(0, endpoint.getTotalLoad());
        assertEquals(10_000, endpoint.realKvAvailable());
    }

    @Test
    void hardAgeCap_skipsSchedulerOwnedRequests() throws InterruptedException {
        updateStatus(null, null, 10_000);
        endpoint.reserve(100L, 500, 500, 5, 0);
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                endpoint.beginPriorityPreemption(1L, java.util.List.of(100L), 200L,
                        100, 100, 10, 0, 0, false));
        Thread.sleep(20);

        assertEquals(1, endpoint.evictExpiredRequests(60_000, 5, requestId -> requestId == 100L),
                "scheduler-owned victim survives the cap; the incoming reservation is evicted");
        assertEquals(1, endpoint.getInflightCount());

        assertEquals(1, endpoint.evictExpiredRequests(60_000, 5, requestId -> false),
                "once the scheduler releases ownership the cap applies");
        assertEquals(0, endpoint.getInflightCount());
    }

    @Test
    void hardAgeCap_disabled_keepsClaimExemption() throws InterruptedException {
        updateStatus(null, null, 10_000);
        endpoint.reserve(100L, 500, 500, 5, 0);
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                endpoint.beginPriorityPreemption(1L, java.util.List.of(100L), 200L,
                        100, 100, 10, 0, 0, false));
        Thread.sleep(20);

        assertEquals(1, endpoint.evictExpiredRequests(5, 0, requestId -> false),
                "cap disabled: only the regular TTL pass runs");
        assertEquals(1, endpoint.getInflightCount(),
                "the claimed victim stays pinned exactly as before");
    }

    // ==================== reason-split eviction breakdown ====================

    @Test
    void evictExpiredRequestsByReason_splitsTtlAndHardAgeCapExits() throws InterruptedException {
        updateStatus(null, null, 10_000);
        endpoint.reserve(100L, 500, 500, 5, 0);
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                endpoint.beginPriorityPreemption(1L, java.util.List.of(100L), 200L,
                        100, 100, 10, 0, 0, false));
        Thread.sleep(20);

        // Regular TTL pass: the claim exempts the victim; only the incoming
        // reservation expires → reason=ttl bucket.
        EvictionBreakdown ttlPass =
                endpoint.evictExpiredRequestsByReason(5, 0, requestId -> false);
        assertEquals(0, ttlPass.allTerminal());
        assertEquals(0, ttlPass.ageCapped());
        assertEquals(0, ttlPass.hardAgeCap());
        assertEquals(1, ttlPass.ttl());
        assertEquals(1, ttlPass.total());
        assertEquals(1, endpoint.getInflightCount());

        // The hard cap force-releases the zombie claim → reason=hard_age_cap
        // bucket (60s TTL keeps the ttl leg out).
        EvictionBreakdown cappedPass =
                endpoint.evictExpiredRequestsByReason(60_000, 5, requestId -> false);
        assertEquals(0, cappedPass.allTerminal());
        assertEquals(0, cappedPass.ageCapped());
        assertEquals(1, cappedPass.hardAgeCap());
        assertEquals(0, cappedPass.ttl());
        assertEquals(1, cappedPass.total());
        assertEquals(0, endpoint.getInflightCount());
    }

    /** Directly mutate the private counter to simulate drift. */
    private void setQueuedPhaseCount(int value) throws Exception {
        java.lang.reflect.Field f = DecodeEndpoint.class.getDeclaredField("queuedPhaseCount");
        f.setAccessible(true);
        java.util.concurrent.atomic.AtomicInteger counter =
                (java.util.concurrent.atomic.AtomicInteger) f.get(endpoint);
        counter.set(value);
    }

    private void updateStatus(Map<String, TaskInfo> running, Map<String, TaskInfo> finished,
                              long availableKvCacheTokens) {
        status.getAvailableKvCacheTokens().set(availableKvCacheTokens);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(running);
        response.setFinishedTaskInfo(finished);
        endpoint.onWorkerStatusUpdate(status, response);
    }

    private TaskInfo task(long requestId) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        return task;
    }
}
