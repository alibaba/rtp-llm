package org.flexlb.balance.endpoint;

import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.enums.TaskPhase;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

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
        assertEquals(1, endpoint.decodeInflightCount());
        assertEquals(9500, endpoint.decodeRealKvAvailable());
    }

    @Test
    void release_decrementsInflight() {
        endpoint.reserve(100L, 500, 500);
        endpoint.reserve(101L, 300, 300);
        endpoint.release(100L);

        assertEquals(1, endpoint.decodeInflightCount());
    }

    @Test
    void release_unknownRequestId_noEffect() {
        endpoint.reserve(100L, 500, 500);
        endpoint.release(999L);
        assertEquals(1, endpoint.decodeInflightCount());
    }

    @Test
    void release_neverGoesNegative() {
        endpoint.reserve(100L, 100, 100);
        endpoint.release(100L);
        endpoint.release(100L);
        assertEquals(0, endpoint.decodeInflightCount());
        assertEquals(0, endpoint.decodeRealKvAvailable());
    }

    @Test
    void calibrate_kvAllocatedReleasesFromInflight() {
        endpoint.reserve(100L, 500, 500);

        TaskInfo running = task(100L);
        running.setPhase(TaskPhase.KV_ALLOCATED);
        updateStatus(Map.of("100", running), null, 10000);

        assertEquals(0, endpoint.decodeInflightCount());
        assertEquals(10000, endpoint.decodeRealKvAvailable());
    }

    @Test
    void calibrate_finishedFailureReleasesFromInflight() {
        endpoint.reserve(100L, 500, 500);

        TaskInfo failed = task(100L);
        failed.setErrorCode(1);
        failed.setErrorMessage("timeout");
        updateStatus(null, Map.of("100", failed), 10000);

        assertEquals(0, endpoint.decodeInflightCount());
    }

    @Test
    void calibrate_finishedSuccessReleasesIfStillPresent() {
        endpoint.reserve(100L, 500, 500);

        TaskInfo success = task(100L);
        success.setErrorCode(0);
        updateStatus(null, Map.of("100", success), 10000);

        assertEquals(0, endpoint.decodeInflightCount());
    }

    @Test
    void calibrate_updatesReportedKvAvailable() {
        endpoint.reserve(100L, 500, 500);
        updateStatus(null, null, 10000);

        assertEquals(9500, endpoint.decodeRealKvAvailable());
    }

    @Test
    void availableKvTokens_accountsForReservations() {
        updateStatus(null, null, 10000);

        endpoint.reserve(100L, 3000, 3000);
        endpoint.reserve(101L, 2000, 2000);

        assertEquals(5000, endpoint.decodeRealKvAvailable());
    }

    @Test
    void ipPort_format() {
        assertEquals("10.0.0.1:8080", endpoint.ipPort());
    }

    // ---- two-layer migration & phase mapping (layer 2: engineTasks) ----

    @Test
    void calibrate_kvAllocatedMigratesToEngineTasksAsLoading() {
        endpoint.reserve(100L, 500, 500);

        TaskInfo running = task(100L);
        running.setPhase(TaskPhase.KV_ALLOCATED);
        updateStatus(Map.of("100", running), null, 10000);

        assertEquals(0, endpoint.decodeInflightCount());
        assertEquals(1, endpoint.decodeEngineTaskCount());
        assertEquals(EngineTaskPhase.LOADING, endpoint.engineTaskPhase(100L));
        // layer-1 KV reservation released on acceptance
        assertEquals(0, endpoint.decodeInflightHardKvReserved());
        assertEquals(10000, endpoint.decodeRealKvAvailable());
    }

    @Test
    void calibrate_runningMigratesToEngineTasksAsRunning() {
        endpoint.reserve(100L, 500, 500);

        TaskInfo running = task(100L);
        running.setPhase(TaskPhase.RUNNING);
        updateStatus(Map.of("100", running), null, 10000);

        assertEquals(0, endpoint.decodeInflightCount());
        assertEquals(1, endpoint.decodeEngineTaskCount());
        assertEquals(EngineTaskPhase.RUNNING, endpoint.engineTaskPhase(100L));
    }

    @Test
    void calibrate_pendingAndReceivedMigrateAsWaiting() {
        // unified acceptance boundary: any reported phase means the engine has
        // taken ownership — PENDING/RECEIVED migrate to layer 2 as WAITING
        endpoint.reserve(100L, 500, 500);

        TaskInfo pending = task(100L);
        pending.setPhase(TaskPhase.PENDING);
        updateStatus(Map.of("100", pending), null, 10000);

        assertEquals(0, endpoint.decodeInflightCount());
        assertEquals(1, endpoint.decodeEngineTaskCount());
        assertEquals(EngineTaskPhase.WAITING, endpoint.engineTaskPhase(100L));
        // layer-1 KV reservation released on acceptance
        assertEquals(0, endpoint.decodeInflightHardKvReserved());
        assertEquals(10000, endpoint.decodeRealKvAvailable());
        // decodeTotalLoad unchanged: still one request across both layers
        assertEquals(1, endpoint.decodeTotalLoad());

        endpoint.reserve(101L, 300, 300);
        TaskInfo received = task(101L);
        received.setPhase(TaskPhase.RECEIVED);
        updateStatus(Map.of("101", received), null, 10000);

        assertEquals(0, endpoint.decodeInflightCount());
        assertEquals(2, endpoint.decodeEngineTaskCount());
        assertEquals(EngineTaskPhase.WAITING, endpoint.engineTaskPhase(101L));
    }

    @Test
    void calibrate_phaseTransitionLoadingToRunning() {
        endpoint.reserve(100L, 500, 500);

        TaskInfo loading = task(100L);
        loading.setPhase(TaskPhase.KV_ALLOCATED);
        updateStatus(Map.of("100", loading), null, 10000);
        assertEquals(EngineTaskPhase.LOADING, endpoint.engineTaskPhase(100L));

        TaskInfo running = task(100L);
        running.setPhase(TaskPhase.RUNNING);
        updateStatus(Map.of("100", running), null, 10000);

        // phase refreshed in place — no duplicate task
        assertEquals(EngineTaskPhase.RUNNING, endpoint.engineTaskPhase(100L));
        assertEquals(1, endpoint.decodeEngineTaskCount());
        assertEquals(0, endpoint.decodeInflightCount());
    }

    @Test
    void calibrate_finishedRemovesEngineTask() {
        endpoint.reserve(100L, 500, 500);

        TaskInfo running = task(100L);
        running.setPhase(TaskPhase.RUNNING);
        updateStatus(Map.of("100", running), null, 10000);
        assertEquals(1, endpoint.decodeTotalLoad());

        TaskInfo finished = task(100L);
        finished.setErrorCode(0);
        updateStatus(null, Map.of("100", finished), 10000);

        assertEquals(0, endpoint.decodeEngineTaskCount());
        assertEquals(0, endpoint.decodeTotalLoad());
    }

    @Test
    void calibrate_staleEngineTaskEvictedAfterMissingRounds() {
        endpoint.reserve(100L, 500, 500);

        TaskInfo running = task(100L);
        running.setPhase(TaskPhase.RUNNING);
        updateStatus(Map.of("100", running), null, 10000); // round 1: accepted

        // Absent from the next STALE_EVICT_ROUNDS (3) reports -> evicted
        updateStatus(null, null, 10000); // round 2
        updateStatus(null, null, 10000); // round 3
        assertEquals(1, endpoint.decodeEngineTaskCount());
        updateStatus(null, null, 10000); // round 4: 4 - 1 >= 3
        assertEquals(0, endpoint.decodeEngineTaskCount());
        assertEquals(0, endpoint.decodeTotalLoad());
    }

    @Test
    void calibrate_reportKeepsStaleCounterFresh() {
        endpoint.reserve(100L, 500, 500);

        TaskInfo running = task(100L);
        running.setPhase(TaskPhase.RUNNING);
        updateStatus(Map.of("100", running), null, 10000); // round 1

        // continuously reported task survives any number of rounds
        for (int i = 0; i < 5; i++) {
            TaskInfo again = task(100L);
            again.setPhase(TaskPhase.RUNNING);
            updateStatus(Map.of("100", again), null, 10000);
        }
        assertEquals(1, endpoint.decodeEngineTaskCount());
    }

    @Test
    void calibrate_untrackedAcceptedTaskStillCounted() {
        // legacy confirmedRunningCount counted every reported accepted task,
        // reserved locally or not — decodeTotalLoad keeps that coverage
        TaskInfo foreign = task(999L);
        foreign.setPhase(TaskPhase.RUNNING);
        updateStatus(Map.of("999", foreign), null, 10000);

        assertEquals(1, endpoint.decodeEngineTaskCount());
        assertEquals(1, endpoint.decodeTotalLoad());
        // no local reservation — KV counters untouched
        assertEquals(0, endpoint.decodeInflightHardKvReserved());
        assertEquals(0, endpoint.decodeInflightExpectedKvReserved());
    }

    @Test
    void release_onlyTouchesInflightLayer() {
        endpoint.reserve(100L, 500, 500);

        TaskInfo running = task(100L);
        running.setPhase(TaskPhase.RUNNING);
        updateStatus(Map.of("100", running), null, 10000);

        endpoint.release(100L);
        // engineTasks mirror engine reports; release must not drop them
        assertEquals(1, endpoint.decodeEngineTaskCount());
        assertEquals(0, endpoint.decodeInflightCount());
    }

    @Test
    void evictExpiredRequests_sparesEngineTasks() throws InterruptedException {
        endpoint.reserve(100L, 500, 500);
        endpoint.reserve(101L, 300, 300);

        TaskInfo running = task(100L);
        running.setPhase(TaskPhase.RUNNING);
        updateStatus(Map.of("100", running), null, 10000);

        Thread.sleep(10);
        // layer-1 TTL backstop only covers layer 1 — long-running decode
        // tasks (layer 2) must survive it
        int evicted = endpoint.evictExpiredRequests(1);
        assertEquals(1, evicted);
        assertEquals(0, endpoint.decodeInflightCount());
        assertEquals(1, endpoint.decodeEngineTaskCount());
        assertEquals(0, endpoint.decodeInflightHardKvReserved());
    }

    @Test
    void evictExpiredEngineTasks_ttlBackstopWhenRoundsStall() throws InterruptedException {
        endpoint.reserve(100L, 500, 500);

        TaskInfo running = task(100L);
        running.setPhase(TaskPhase.RUNNING);
        updateStatus(Map.of("100", running), null, 10000); // accepted into layer 2

        // Worker stops reporting entirely: no further onWorkerStatusUpdate,
        // so calibrate rounds never advance and stale-round eviction cannot
        // fire — only the wall-clock TTL backstop can reclaim the task.
        Thread.sleep(10);
        assertEquals(0, endpoint.evictExpiredEngineTasks(60_000)); // fresh: kept
        assertEquals(1, endpoint.decodeEngineTaskCount());

        int evicted = endpoint.evictExpiredEngineTasks(1);
        assertEquals(1, evicted);
        assertEquals(0, endpoint.decodeEngineTaskCount());
        assertEquals(0, endpoint.decodeTotalLoad());
    }

    // ---- new-view absolute values across layers ----

    @Test
    void newViewsReportAbsoluteValuesAfterAcceptance() {
        updateStatus(null, null, 10000);
        status.getTotalKvCacheTokens().set(20000);
        endpoint.reserve(100L, 500, 800);
        endpoint.reserve(101L, 300, 400);

        TaskInfo running = task(100L);
        running.setPhase(TaskPhase.KV_ALLOCATED);
        updateStatus(Map.of("100", running), null, 10000);

        // 1 accepted (layer 2) + 1 still inflight (layer 1)
        assertEquals(2, endpoint.decodeTotalLoad());
        assertEquals(1, endpoint.decodeInflightCount());
        assertEquals(300, endpoint.decodeInflightHardKvReserved());
        assertEquals(400, endpoint.decodeInflightExpectedKvReserved());
        assertEquals(20000, endpoint.decodeKvTotal());
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
