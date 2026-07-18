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
    void duplicateReservationReplacesBothCounters() {
        endpoint.reserve(100L, 500, 800);
        endpoint.reserve(100L, 300, 600);

        assertEquals(300, endpoint.inflightHardKvReserved());
        assertEquals(600, endpoint.inflightKvReserved());
    }

    @Test
    void evictionDecrementsBothCounters() throws InterruptedException {
        endpoint.reserve(100L, 500, 800);
        Thread.sleep(10);

        endpoint.evictExpiredRequests(1);

        assertEquals(0, endpoint.inflightHardKvReserved());
        assertEquals(0, endpoint.inflightKvReserved());
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
    void inflightKvReserved_returnsExpectedKvTokens() {
        updateStatus(null, null, 10000);

        // kvTokens=1000, expectedKvTokens=5000 (prompt + generation)
        endpoint.reserve(100L, 1000, 5000);
        // kvTokens=2000, expectedKvTokens=3000
        endpoint.reserve(101L, 2000, 3000);

        // inflightKvReserved sums expectedKvTokens (conservative estimate)
        assertEquals(8000, endpoint.inflightKvReserved());
    }

    @Test
    void inflightHardKvReserved_returnsKvTokens() {
        updateStatus(null, null, 10000);

        endpoint.reserve(100L, 1000, 5000);
        endpoint.reserve(101L, 2000, 3000);

        // inflightHardKvReserved sums kvTokens (hard demand = prompt only)
        assertEquals(3000, endpoint.inflightHardKvReserved());
    }

    @Test
    void realKvAvailable_usesHardKvNotExpectedKv() {
        updateStatus(null, null, 10000);

        // kvTokens=1000 (prompt), expectedKvTokens=5000 (prompt + generation)
        endpoint.reserve(100L, 1000, 5000);

        // realKvAvailable uses inflightHardKvReserved (1000), not inflightKvReserved (5000)
        assertEquals(9000, endpoint.realKvAvailable());
    }

    @Test
    void realKvUsed_usesExpectedKvForScoring() {
        updateStatus(null, null, 10000);
        // totalKv=0 → reportedUsed=0; realKvUsed = 0 + inflightKvReserved
        // But totalKv is 0 by default, so reportedUsed=0
        endpoint.reserve(100L, 1000, 5000);

        // realKvUsed = reportedUsed(0) + inflightKvReserved(5000) = 5000
        assertEquals(5000, endpoint.realKvUsed());
    }

    @Test
    void ipPort_format() {
        assertEquals("10.0.0.1:8080", endpoint.ipPort());
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
