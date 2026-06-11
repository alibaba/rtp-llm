package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.BatchDecisionHandler;
import org.flexlb.balance.strategy.BatchRequest;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.domain.worker.WorkerStatusResponse;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class WorkerEndpointTest {

    private WorkerStatus status;
    private PrefillEndpoint endpoint;

    @BeforeEach
    void setUp() {
        status = new WorkerStatus();
        status.setIp("10.0.0.1");
        status.setPort(8080);
        FlexlbConfig config = new FlexlbConfig();
        config.setCostAlpha0(0);
        config.setCostAlpha1(1);
        BatchDecisionHandler handler = Mockito.mock(BatchDecisionHandler.class);
        endpoint = new PrefillEndpoint("10.0.0.1", 8080, 8081, status, config, handler);
    }

    @Test
    void commitBatch_incrementsEstimate() {
        endpoint.commitBatch(1L, 500, List.of(new BatchRequest(100L, 1000, 0)));
        assertEquals(500, endpoint.getEstimatedWaitingTimeMs());

        endpoint.commitBatch(2L, 300, List.of(new BatchRequest(101L, 500, 0)));
        assertEquals(800, endpoint.getEstimatedWaitingTimeMs());
    }

    @Test
    void releaseBatch_decrementsEstimate() {
        endpoint.commitBatch(1L, 500, List.of(new BatchRequest(100L, 1000, 0)));
        endpoint.commitBatch(2L, 300, List.of(new BatchRequest(101L, 500, 0)));

        endpoint.releaseBatch(1L);
        assertEquals(300, endpoint.getEstimatedWaitingTimeMs());
    }

    @Test
    void releaseBatch_unknownBatchId_noEffect() {
        endpoint.commitBatch(1L, 500, List.of(new BatchRequest(100L, 1000, 0)));
        endpoint.releaseBatch(999L);
        assertEquals(500, endpoint.getEstimatedWaitingTimeMs());
    }

    @Test
    void releaseBatch_neverGoesNegative() {
        endpoint.commitBatch(1L, 100, List.of(new BatchRequest(100L, 1000, 0)));
        endpoint.releaseBatch(1L);
        endpoint.releaseBatch(1L);
        assertEquals(0, endpoint.getEstimatedWaitingTimeMs());
    }

    @Test
    void calibrate_noInflight_resetsToZero() {
        endpoint.commitBatch(1L, 500, List.of(new BatchRequest(100L, 1000, 0)));

        TaskInfo finished = task(100L, 1000, 0, 1L);
        finished.setErrorCode(0);
        endpoint.calibrate(Map.of("100", finished), null);

        assertEquals(0, endpoint.getEstimatedWaitingTimeMs());
        assertTrue(endpoint.getInflightBatches().isEmpty());
    }

    @Test
    void calibrate_finishedBatch_removedFromInflight() {
        endpoint.commitBatch(5L, 9999, List.of(
                new BatchRequest(100L, 1000, 0), new BatchRequest(101L, 2000, 0)));

        TaskInfo t1 = task(100L, 1000, 0, 5L);
        t1.setErrorCode(0);
        TaskInfo t2 = task(101L, 2000, 0, 5L);
        t2.setErrorCode(0);
        endpoint.calibrate(Map.of("100", t1, "101", t2), null);

        assertEquals(0, endpoint.getEstimatedWaitingTimeMs());
        assertFalse(endpoint.getInflightBatches().containsKey(5L));
    }

    @Test
    void calibrate_partialBatchFailure_repacks() {
        endpoint.commitBatch(5L, 9999, List.of(
                new BatchRequest(100L, 1000, 0), new BatchRequest(101L, 2000, 0)));

        TaskInfo failed = task(100L, 1000, 0, 5L);
        failed.setErrorCode(1);
        failed.setErrorMessage("timeout");
        TaskInfo success = task(101L, 2000, 0, 5L);
        success.setErrorCode(0);
        endpoint.calibrate(Map.of("100", failed, "101", success), null);

        assertFalse(endpoint.getInflightBatches().containsKey(5L));
        assertEquals(0, endpoint.getEstimatedWaitingTimeMs());
    }

    @Test
    void calibrate_inflightUnconfirmedBatchesSurvive() {
        endpoint.commitBatch(5L, 1000, List.of(new BatchRequest(100L, 500, 0)));
        endpoint.commitBatch(7L, 2000, List.of(new BatchRequest(200L, 1000, 0)));

        TaskInfo finished = task(100L, 500, 0, 5L);
        finished.setErrorCode(0);
        endpoint.calibrate(Map.of("100", finished), null);

        assertFalse(endpoint.getInflightBatches().containsKey(5L));
        assertTrue(endpoint.getInflightBatches().containsKey(7L));
        assertEquals(2000, endpoint.getEstimatedWaitingTimeMs());
    }

    @Test
    void repackBatch_removesFailedRequests() {
        endpoint.commitBatch(5L, 9999, List.of(
                new BatchRequest(100L, 1000, 0),
                new BatchRequest(101L, 2000, 0),
                new BatchRequest(102L, 3000, 0)));

        BatchInflight repacked = endpoint.repackBatch(5L, java.util.Set.of(101L));

        assertNotNull(repacked);
        assertEquals(2, repacked.requests().size());
        assertTrue(repacked.requests().stream().anyMatch(r -> r.requestId() == 100L));
        assertTrue(repacked.requests().stream().anyMatch(r -> r.requestId() == 102L));
        assertFalse(repacked.requests().stream().anyMatch(r -> r.requestId() == 101L));
    }

    @Test
    void repackBatch_allFailed_removesBatch() {
        endpoint.commitBatch(5L, 500, List.of(new BatchRequest(100L, 1000, 0)));

        BatchInflight repacked = endpoint.repackBatch(5L, java.util.Set.of(100L));

        assertNull(repacked);
        assertFalse(endpoint.getInflightBatches().containsKey(5L));
        assertEquals(0, endpoint.getEstimatedWaitingTimeMs());
    }

    @Test
    void ipPort_format() {
        assertEquals("10.0.0.1:8080", endpoint.ipPort());
    }

    // ==================== getStatus() immutable snapshot ====================

    @Test
    void getStatus_returns_immutable_snapshot() {
        status.setAlive(true);
        status.setAvailableConcurrency(42L);
        status.setDpRank(3);

        WorkerStatus snapshot = endpoint.getStatus();
        assertTrue(snapshot.isAlive());
        assertEquals(42L, (long) snapshot.getAvailableConcurrency());
        assertEquals(3L, snapshot.getDpRank());

        // Mutate the snapshot — must not affect the original
        snapshot.setAlive(false);
        snapshot.setAvailableConcurrency(999L);
        snapshot.setDpRank(99);

        // Original endpoint state unchanged
        assertTrue(endpoint.isAlive());
        assertEquals(42L, (long) endpoint.getAvailableConcurrency());
        assertEquals(3L, endpoint.getDpRank());
    }

    @Test
    void getStatus_snapshot_atomic_fields_are_independent() {
        status.getAvailableKvCacheTokens().set(5000L);
        status.getStatusVersion().set(10L);

        WorkerStatus snapshot = endpoint.getStatus();
        snapshot.getAvailableKvCacheTokens().set(0L);
        snapshot.getStatusVersion().set(999L);

        // Internal AtomicLong values unchanged
        assertEquals(5000L, endpoint.getAvailableKvCacheTokens().get());
        assertEquals(10L, endpoint.getStatusVersion().get());
    }

    // ==================== updateFromGrpcResponse ====================

    @Test
    void updateFromGrpcResponse_with_null_is_noop() {
        status.setAlive(true);
        status.setAvailableConcurrency(10L);

        endpoint.updateFromGrpcResponse(null);

        assertTrue(endpoint.isAlive());
        assertEquals(10L, (long) endpoint.getAvailableConcurrency());
    }

    @Test
    void updateFromGrpcResponse_is_idempotent() {
        WorkerStatusResponse resp = new WorkerStatusResponse();
        resp.setRole("DECODE");
        resp.setAlive(true);
        resp.setAvailableConcurrency(8L);
        resp.setStepLatencyMs(25.0);
        resp.setIterateCount(100L);
        resp.setDpSize(4);
        resp.setTpSize(2);
        resp.setDpRank(1);
        resp.setAvailableKvCacheTokens(10000L);
        resp.setStatusVersion(5L);
        resp.setLatestFinishedVersion(3L);

        // First call
        endpoint.updateFromGrpcResponse(resp);

        assertEquals("DECODE", endpoint.getRole());
        assertTrue(endpoint.isAlive());
        assertEquals(8L, (long) endpoint.getAvailableConcurrency());
        assertEquals(25.0, endpoint.getStepLatencyMs(), 0.001);
        assertEquals(100L, endpoint.getIterateCount());
        assertEquals(4L, endpoint.getDpSize());
        assertEquals(2L, endpoint.getTpSize());
        assertEquals(1L, endpoint.getDpRank());
        assertEquals(10000L, endpoint.getAvailableKvCacheTokens().get());
        assertEquals(5L, endpoint.getStatusVersion().get());
        assertEquals(3L, endpoint.getLatestFinishedTaskVersion().get());

        // Second call with same values — state unchanged
        endpoint.updateFromGrpcResponse(resp);

        assertEquals("DECODE", endpoint.getRole());
        assertTrue(endpoint.isAlive());
        assertEquals(8L, (long) endpoint.getAvailableConcurrency());
        assertEquals(25.0, endpoint.getStepLatencyMs(), 0.001);
        assertEquals(100L, endpoint.getIterateCount());
        assertEquals(4L, endpoint.getDpSize());
        assertEquals(2L, endpoint.getTpSize());
        assertEquals(1L, endpoint.getDpRank());
        assertEquals(10000L, endpoint.getAvailableKvCacheTokens().get());
        assertEquals(5L, endpoint.getStatusVersion().get());
        assertEquals(3L, endpoint.getLatestFinishedTaskVersion().get());
    }

    @Test
    void updateFromGrpcResponse_overwrites_previous_state() {
        WorkerStatusResponse first = new WorkerStatusResponse();
        first.setRole("PREFILL");
        first.setAlive(true);
        first.setAvailableConcurrency(5L);
        first.setDpRank(0);
        first.setAvailableKvCacheTokens(5000L);
        first.setStatusVersion(1L);
        first.setLatestFinishedVersion(1L);

        endpoint.updateFromGrpcResponse(first);
        assertEquals("PREFILL", endpoint.getRole());
        assertEquals(0L, endpoint.getDpRank());

        WorkerStatusResponse second = new WorkerStatusResponse();
        second.setRole("DECODE");
        second.setAlive(false);
        second.setAvailableConcurrency(0L);
        second.setDpRank(2);
        second.setAvailableKvCacheTokens(2000L);
        second.setStatusVersion(2L);
        second.setLatestFinishedVersion(2L);

        endpoint.updateFromGrpcResponse(second);

        assertEquals("DECODE", endpoint.getRole());
        assertFalse(endpoint.isAlive());
        assertEquals(0L, (long) endpoint.getAvailableConcurrency());
        assertEquals(2L, endpoint.getDpRank());
        assertEquals(2000L, endpoint.getAvailableKvCacheTokens().get());
        assertEquals(2L, endpoint.getStatusVersion().get());
        assertEquals(2L, endpoint.getLatestFinishedTaskVersion().get());
    }

    private TaskInfo task(long requestId, long inputLength, long prefixLength, long batchId) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setInputLength(inputLength);
        task.setPrefixLength(prefixLength);
        task.setBatchId(batchId);
        return task;
    }
}
