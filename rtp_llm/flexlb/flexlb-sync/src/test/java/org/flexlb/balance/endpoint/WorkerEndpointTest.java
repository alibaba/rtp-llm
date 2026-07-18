package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.BatchDecisionHandler;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
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
        status.setGrpcPort(8081);
        FlexlbConfig config = new FlexlbConfig();
        config.setCostFormula("sum(computeTokens)");
        BatchDecisionHandler handler = Mockito.mock(BatchDecisionHandler.class);
        endpoint = new PrefillEndpoint(status, config, handler, Mockito.mock(BatchSchedulerReporter.class));
    }

    @Test
    void commitBatch_incrementsEstimate() {
        endpoint.commitBatch(1L, 500, List.of(new BatchItem(ctx(100L, 1000), null, null, null, null, null, null, 0, 0)));
        assertEquals(500, endpoint.realWaitTimeMs());

        endpoint.commitBatch(2L, 300, List.of(new BatchItem(ctx(101L, 500), null, null, null, null, null, null, 0, 0)));
        assertEquals(800, endpoint.realWaitTimeMs());
    }

    @Test
    void releaseBatch_decrementsEstimate() {
        endpoint.commitBatch(1L, 500, List.of(new BatchItem(ctx(100L, 1000), null, null, null, null, null, null, 0, 0)));
        endpoint.commitBatch(2L, 300, List.of(new BatchItem(ctx(101L, 500), null, null, null, null, null, null, 0, 0)));

        endpoint.releaseBatch(1L);
        assertEquals(300, endpoint.realWaitTimeMs());
    }

    @Test
    void releaseBatch_unknownBatchId_noEffect() {
        endpoint.commitBatch(1L, 500, List.of(new BatchItem(ctx(100L, 1000), null, null, null, null, null, null, 0, 0)));
        endpoint.releaseBatch(999L);
        assertEquals(500, endpoint.realWaitTimeMs());
    }

    @Test
    void releaseBatch_neverGoesNegative() {
        endpoint.commitBatch(1L, 100, List.of(new BatchItem(ctx(100L, 1000), null, null, null, null, null, null, 0, 0)));
        endpoint.releaseBatch(1L);
        endpoint.releaseBatch(1L);
        assertEquals(0, endpoint.realWaitTimeMs());
    }

    @Test
    void calibrate_noInflight_resetsToZero() {
        endpoint.commitBatch(1L, 500, List.of(new BatchItem(ctx(100L, 1000), null, null, null, null, null, null, 0, 0)));

        TaskInfo finished = task(100L, 1000, 0, 1L);
        finished.setErrorCode(0);
        endpoint.calibrate(Map.of("100", finished), null);

        assertEquals(0, endpoint.realWaitTimeMs());
        assertTrue(endpoint.getInflightBatches().isEmpty());
    }

    @Test
    void calibrate_finishedBatch_removedFromInflight() {
        endpoint.commitBatch(5L, 9999, List.of(
                new BatchItem(ctx(100L, 1000), null, null, null, null, null, null, 0, 0), new BatchItem(ctx(101L, 2000), null, null, null, null, null, null, 0, 0)));

        TaskInfo t1 = task(100L, 1000, 0, 5L);
        t1.setErrorCode(0);
        TaskInfo t2 = task(101L, 2000, 0, 5L);
        t2.setErrorCode(0);
        endpoint.calibrate(Map.of("100", t1, "101", t2), null);

        assertEquals(0, endpoint.realWaitTimeMs());
        assertFalse(endpoint.getInflightBatches().containsKey(5L));
    }

    @Test
    void calibrate_partialBatchFailure_repacks() {
        endpoint.commitBatch(5L, 9999, List.of(
                new BatchItem(ctx(100L, 1000), null, null, null, null, null, null, 0, 0), new BatchItem(ctx(101L, 2000), null, null, null, null, null, null, 0, 0)));

        TaskInfo failed = task(100L, 1000, 0, 5L);
        failed.setErrorCode(1);
        failed.setErrorMessage("timeout");
        TaskInfo success = task(101L, 2000, 0, 5L);
        success.setErrorCode(0);
        endpoint.calibrate(Map.of("100", failed, "101", success), null);

        assertFalse(endpoint.getInflightBatches().containsKey(5L));
        assertEquals(0, endpoint.realWaitTimeMs());
    }

    @Test
    void calibrate_inflightUnconfirmedBatchesSurvive() {
        endpoint.commitBatch(5L, 1000, List.of(new BatchItem(ctx(100L, 500), null, null, null, null, null, null, 0, 0)));
        endpoint.commitBatch(7L, 2000, List.of(new BatchItem(ctx(200L, 1000), null, null, null, null, null, null, 0, 0)));

        TaskInfo finished = task(100L, 500, 0, 5L);
        finished.setErrorCode(0);
        endpoint.calibrate(Map.of("100", finished), null);

        assertFalse(endpoint.getInflightBatches().containsKey(5L));
        assertTrue(endpoint.getInflightBatches().containsKey(7L));
        // realWaitTimeMs = predictMs - elapsedMs; allow small timing delta
        assertTrue(Math.abs(endpoint.realWaitTimeMs() - 2000) < 50,
                "Expected ~2000ms but got " + endpoint.realWaitTimeMs());
    }

    @Test
    void repackBatch_removesFailedRequests() {
        endpoint.commitBatch(5L, 9999, List.of(
                new BatchItem(ctx(100L, 1000), null, null, null, null, null, null, 0, 0),
                new BatchItem(ctx(101L, 2000), null, null, null, null, null, null, 0, 0),
                new BatchItem(ctx(102L, 3000), null, null, null, null, null, null, 0, 0)));
        BatchInflight repacked = endpoint.repackBatch(5L, java.util.Set.of(101L));

        assertNotNull(repacked);
        assertEquals(2, repacked.requests().size());
        assertTrue(repacked.requests().stream().anyMatch(r -> r.requestId() == 100L));
        assertTrue(repacked.requests().stream().anyMatch(r -> r.requestId() == 102L));
        assertFalse(repacked.requests().stream().anyMatch(r -> r.requestId() == 101L));
    }

    @Test
    void repackBatch_allFailed_removesBatch() {
        endpoint.commitBatch(5L, 500, List.of(new BatchItem(ctx(100L, 1000), null, null, null, null, null, null, 0, 0)));

        BatchInflight repacked = endpoint.repackBatch(5L, java.util.Set.of(100L));

        assertNull(repacked);
        assertFalse(endpoint.getInflightBatches().containsKey(5L));
        assertEquals(0, endpoint.realWaitTimeMs());
    }

    @Test
    void ipPort_format() {
        assertEquals("10.0.0.1:8080", endpoint.ipPort());
    }

    // ==================== getStatus() returns live reference ====================

    @Test
    void getStatus_returns_live_reference() {
        status.setAlive(true);
        status.setAvailableConcurrency(42L);
        status.setDpRank(3);

        WorkerStatus liveStatus = endpoint.getStatus();
        assertSame(status, liveStatus);
        assertTrue(liveStatus.isAlive());
        assertEquals(42L, (long) liveStatus.getAvailableConcurrency());
        assertEquals(3L, liveStatus.getDpRank());
    }

    // ==================== WorkerStatus.updateFromResponse ====================

    @Test
    void updateFromResponse_applies_all_engine_fields() {
        WorkerStatusResponse resp = new WorkerStatusResponse();
        resp.setRole(RoleType.DECODE);
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

        status.updateFromResponse(resp);

        assertEquals(RoleType.DECODE, status.getRole());
        assertTrue(status.isAlive());
        assertEquals(8L, (long) status.getAvailableConcurrency());
        assertEquals(25.0, status.getStepLatencyMs(), 0.001);
        assertEquals(100L, status.getIterateCount());
        assertEquals(4L, status.getDpSize());
        assertEquals(2L, status.getTpSize());
        assertEquals(1L, status.getDpRank());
        assertEquals(10000L, status.getAvailableKvCacheTokens().get());
        assertEquals(5L, status.getStatusVersion().get());
        // latestFinishedTaskVersion is intentionally NOT set by updateFromResponse();
        // it is advanced only after calibrate processes finished tasks
        assertEquals(-1L, status.getLatestFinishedTaskVersion().get());
    }

    @Test
    void updateFromResponse_null_is_noop() {
        status.setAlive(true);
        status.setAvailableConcurrency(10L);

        status.updateFromResponse(null);

        assertTrue(status.isAlive());
        assertEquals(10L, (long) status.getAvailableConcurrency());
    }

    // ==================== onWorkerStatusUpdate ====================

    @Test
    void onWorkerStatusUpdate_replaces_status_reference() {
        WorkerStatusResponse resp = new WorkerStatusResponse();
        WorkerStatus newStatus = new WorkerStatus();
        newStatus.setSite("site-a");
        newStatus.setGroup("group-b");
        newStatus.setAlive(true);

        assertNotSame(newStatus, endpoint.getStatus());

        endpoint.onWorkerStatusUpdate(newStatus, resp);

        assertSame(newStatus, endpoint.getStatus());
        assertEquals("site-a", endpoint.getStatus().getSite());
        assertEquals("group-b", endpoint.getStatus().getGroup());
    }

    @Test
    void onWorkerStatusUpdate_calibrates_prefill() {
        WorkerStatusResponse resp = new WorkerStatusResponse();
        resp.setFinishedTaskInfo(Map.of("100", task(100L, 1000, 0, 1L)));

        // PrefillEndpoint calibrates even when runningTaskInfo is null
        endpoint.onWorkerStatusUpdate(status, resp);
        // No exception = calibrate handled null gracefully
    }

    @Test
    void onWorkerStatusUpdate_preserves_engine_state_from_ws() {
        WorkerStatusResponse resp = new WorkerStatusResponse();
        WorkerStatus ws = new WorkerStatus();
        ws.setSite("site-x");
        ws.setGroup("group-x");
        ws.setDpRank(5);
        ws.setAlive(true);

        endpoint.onWorkerStatusUpdate(ws, resp);

        assertEquals("site-x", endpoint.getStatus().getSite());
        assertEquals("group-x", endpoint.getStatus().getGroup());
        assertEquals(5L, endpoint.getStatus().getDpRank());
        assertTrue(endpoint.getStatus().isAlive());
    }

    private BalanceContext ctx(long requestId, long seqLen) {
        Request req = new Request();
        req.setRequestId(requestId);
        req.setSeqLen(seqLen);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(req);
        return ctx;
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
