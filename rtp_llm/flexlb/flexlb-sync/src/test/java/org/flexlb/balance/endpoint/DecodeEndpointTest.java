package org.flexlb.balance.endpoint;

import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.TaskPhase;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

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
        endpoint.calibrate(null, null, 10000);
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
        endpoint.calibrate(Map.of("100", running), null, 10000);

        assertEquals(0, endpoint.getInflightCount());
        assertEquals(10000, endpoint.realKvAvailable());
    }

    @Test
    void calibrate_finishedFailureReleasesFromInflight() {
        endpoint.reserve(100L, 500, 500);

        TaskInfo failed = task(100L);
        failed.setErrorCode(1);
        failed.setErrorMessage("timeout");
        endpoint.calibrate(null, Map.of("100", failed), 10000);

        assertEquals(0, endpoint.getInflightCount());
    }

    @Test
    void calibrate_finishedSuccessReleasesIfStillPresent() {
        endpoint.reserve(100L, 500, 500);

        TaskInfo success = task(100L);
        success.setErrorCode(0);
        endpoint.calibrate(null, Map.of("100", success), 10000);

        assertEquals(0, endpoint.getInflightCount());
    }

    @Test
    void calibrate_updatesReportedKvAvailable() {
        endpoint.reserve(100L, 500, 500);
        endpoint.calibrate(null, null, 10000);

        assertEquals(9500, endpoint.realKvAvailable());
    }

    @Test
    void availableKvTokens_accountsForReservations() {
        endpoint.calibrate(null, null, 10000);

        endpoint.reserve(100L, 3000, 3000);
        endpoint.reserve(101L, 2000, 2000);

        assertEquals(5000, endpoint.realKvAvailable());
    }

    @Test
    void inflightKvReserved_returnsExpectedKvTokens() {
        endpoint.calibrate(null, null, 10000);

        // kvTokens=1000, expectedKvTokens=5000 (prompt + generation)
        endpoint.reserve(100L, 1000, 5000);
        // kvTokens=2000, expectedKvTokens=3000
        endpoint.reserve(101L, 2000, 3000);

        // inflightKvReserved sums expectedKvTokens (conservative estimate)
        assertEquals(8000, endpoint.inflightKvReserved());
    }

    @Test
    void inflightHardKvReserved_returnsKvTokens() {
        endpoint.calibrate(null, null, 10000);

        endpoint.reserve(100L, 1000, 5000);
        endpoint.reserve(101L, 2000, 3000);

        // inflightHardKvReserved sums kvTokens (hard demand = prompt only)
        assertEquals(3000, endpoint.inflightHardKvReserved());
    }

    @Test
    void realKvAvailable_usesHardKvNotExpectedKv() {
        endpoint.calibrate(null, null, 10000);

        // kvTokens=1000 (prompt), expectedKvTokens=5000 (prompt + generation)
        endpoint.reserve(100L, 1000, 5000);

        // realKvAvailable uses inflightHardKvReserved (1000), not inflightKvReserved (5000)
        assertEquals(9000, endpoint.realKvAvailable());
    }

    @Test
    void realKvUsed_usesExpectedKvForScoring() {
        endpoint.calibrate(null, null, 10000);
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

    // ==================== KV Counter Incremental Maintenance Tests ====================

    @Test
    void reserveIncrementsKvCounters() {
        endpoint.reserve(1001L, 500, 800);
        assertEquals(500, endpoint.inflightHardKvReserved());
        assertEquals(800, endpoint.inflightKvReserved());
    }

    @Test
    void reserveDuplicateKeySwapsKvCounters() {
        endpoint.reserve(1001L, 500, 800);
        endpoint.reserve(1001L, 300, 600);
        // Old values subtracted, new values added
        assertEquals(300, endpoint.inflightHardKvReserved());
        assertEquals(600, endpoint.inflightKvReserved());
    }

    @Test
    void releaseDecrementsKvCounters() {
        endpoint.reserve(1001L, 500, 800);
        endpoint.release(1001L);
        assertEquals(0, endpoint.inflightHardKvReserved());
        assertEquals(0, endpoint.inflightKvReserved());
    }

    @Test
    void calibratePhase1DecrementsKvCounters() {
        endpoint.reserve(1001L, 500, 800);
        TaskInfo running = task(1001L);
        running.setPhase(TaskPhase.KV_ALLOCATED);
        endpoint.calibrate(Map.of("1001", running), null, 10000);
        assertEquals(0, endpoint.inflightHardKvReserved());
        assertEquals(0, endpoint.inflightKvReserved());
    }

    @Test
    void calibratePhase2DecrementsKvCounters() {
        endpoint.reserve(1001L, 500, 800);
        TaskInfo failed = task(1001L);
        failed.setErrorCode(1);
        failed.setErrorMessage("timeout");
        endpoint.calibrate(null, Map.of("1001", failed), 10000);
        assertEquals(0, endpoint.inflightHardKvReserved());
        assertEquals(0, endpoint.inflightKvReserved());
    }

    @Test
    void calibratePhase3DecrementsKvCounters() {
        endpoint.reserve(1001L, 500, 800);
        TaskInfo success = task(1001L);
        success.setErrorCode(0);
        endpoint.calibrate(null, Map.of("1001", success), 10000);
        assertEquals(0, endpoint.inflightHardKvReserved());
        assertEquals(0, endpoint.inflightKvReserved());
    }

    @Test
    void evictExpiredDecrementsKvCounters() throws InterruptedException {
        endpoint.reserve(1001L, 500, 800);
        Thread.sleep(10);
        endpoint.evictExpiredRequests(1);
        assertEquals(0, endpoint.inflightHardKvReserved());
        assertEquals(0, endpoint.inflightKvReserved());
    }

    @Test
    void calibratePhase1ThenPhase2And3NoDoubleDeduction() {
        endpoint.reserve(1001L, 500, 800);
        // Phase 1: KV_ALLOCATED removes from inflight
        TaskInfo running = task(1001L);
        running.setPhase(TaskPhase.KV_ALLOCATED);
        // Phase 2: finished error tries to remove same requestId
        TaskInfo failed = task(1001L);
        failed.setErrorCode(1);
        failed.setErrorMessage("timeout");
        // Phase 3: finished success tries to remove same requestId
        TaskInfo success = task(1001L);
        success.setErrorCode(0);
        endpoint.calibrate(Map.of("running-1001", running),
                Map.of("err-1001", failed, "succ-1001", success), 10000);
        // Counters should be 0, not negative (no double deduction)
        assertEquals(0, endpoint.inflightHardKvReserved());
        assertEquals(0, endpoint.inflightKvReserved());
    }

    @Test
    void releaseThenCalibrateNoDoubleDeduction() {
        endpoint.reserve(1001L, 500, 800);
        endpoint.release(1001L);
        // Calibrate tries to remove same requestId — already gone
        TaskInfo success = task(1001L);
        success.setErrorCode(0);
        endpoint.calibrate(null, Map.of("1001", success), 10000);
        // Counters should still be 0, not negative
        assertEquals(0, endpoint.inflightHardKvReserved());
        assertEquals(0, endpoint.inflightKvReserved());
    }

    @Test
    void multipleRequestsProgressiveDecrease() {
        endpoint.reserve(1001L, 100, 200);
        endpoint.reserve(1002L, 200, 400);
        endpoint.reserve(1003L, 300, 600);
        // Total: hard=600, expected=1200
        assertEquals(600, endpoint.inflightHardKvReserved());
        assertEquals(1200, endpoint.inflightKvReserved());

        // Release first
        endpoint.release(1001L);
        assertEquals(500, endpoint.inflightHardKvReserved());
        assertEquals(1000, endpoint.inflightKvReserved());

        // Calibrate second (KV_ALLOCATED)
        TaskInfo running = task(1002L);
        running.setPhase(TaskPhase.KV_ALLOCATED);
        endpoint.calibrate(Map.of("1002", running), null, 10000);
        assertEquals(300, endpoint.inflightHardKvReserved());
        assertEquals(600, endpoint.inflightKvReserved());

        // Release third
        endpoint.release(1003L);
        assertEquals(0, endpoint.inflightHardKvReserved());
        assertEquals(0, endpoint.inflightKvReserved());
    }

    @Test
    void emptyMapCountersAreZero() {
        assertEquals(0, endpoint.inflightHardKvReserved());
        assertEquals(0, endpoint.inflightKvReserved());
    }

    @Test
    void hardAndExpectedKvTrackIndependently() {
        // kvTokens=1000 (prompt only), expectedKvTokens=5000 (prompt + generation)
        endpoint.reserve(1001L, 1000, 5000);
        endpoint.reserve(1002L, 2000, 3000);
        // Hard KV: 1000 + 2000 = 3000
        assertEquals(3000, endpoint.inflightHardKvReserved());
        // Expected KV: 5000 + 3000 = 8000
        assertEquals(8000, endpoint.inflightKvReserved());
        // Verify they differ — proving independent tracking
        assertNotEquals(endpoint.inflightHardKvReserved(), endpoint.inflightKvReserved());
    }

    @Test
    void countersEquivalentToTraversal() {
        // Mix of operations
        endpoint.reserve(1001L, 100, 200);
        endpoint.reserve(1002L, 300, 600);
        endpoint.reserve(1003L, 500, 1000);

        // Remove one via release
        endpoint.release(1002L);

        // Remove one via calibrate Phase 1
        TaskInfo running = task(1003L);
        running.setPhase(TaskPhase.RUNNING);
        endpoint.calibrate(Map.of("1003", running), null, 10000);

        // Only 1001 remains: hard=100, expected=200
        // Verify counter values equal manual traversal sums
        assertEquals(manualHardKvSum(), endpoint.inflightHardKvReserved());
        assertEquals(manualExpectedKvSum(), endpoint.inflightKvReserved());
        assertEquals(100, endpoint.inflightHardKvReserved());
        assertEquals(200, endpoint.inflightKvReserved());
    }

    // ---- counter test helpers ----

    private long manualHardKvSum() {
        long sum = 0;
        for (RequestInflight ri : endpoint.getInflightRequests().values()) {
            sum += ri.kvTokens();
        }
        return sum;
    }

    private long manualExpectedKvSum() {
        long sum = 0;
        for (RequestInflight ri : endpoint.getInflightRequests().values()) {
            sum += ri.expectedKvTokens();
        }
        return sum;
    }

    private TaskInfo task(long requestId) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        return task;
    }
}
