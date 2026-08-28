package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class WorkerEndpointTest {

    private WorkerStatus status;
    private PrefillEndpoint endpoint;

    @BeforeEach
    void setUp() {
        status = EndpointTestSupport.workerStatus(
                RoleType.PREFILL,
                "group-x",
                "10.0.0.1",
                8080,
                8081,
                "site-x");
        FlexlbConfig config = new FlexlbConfig();
        ((RoutingConfig.FormulaEstimatorConfig) config.getRouter().getRoles().getPrefill()
                .getExecutionTimeEstimator()).setExpression("sum(computeTokens)");
        EndpointTestSupport.TestRequestRuntime requestRuntime =
                EndpointTestSupport.requestRuntime();
        endpoint = new PrefillEndpoint(
                status,
                config,
                EndpointTestSupport.routeStrategy(requestRuntime),
                requestRuntime,
                requestRuntime,
                Mockito.mock(BatchSchedulerReporter.class));
        endpoint.startGeneration();
    }

    @AfterEach
    void tearDown() {
        endpoint.close();
    }

    @Test
    void commitBatch_increasesCommittedWork() {
        registerBatch(1L, 500,
                item(100L, 1000));
        assertCommittedWorkNear(500);

        registerBatch(2L, 300,
                item(101L, 500));
        assertCommittedWorkNear(800);
    }

    @Test
    void releaseBatch_decreasesCommittedWork() {
        ScheduledRequest first = item(100L, 1000);
        ScheduledRequest second = item(101L, 500);
        registerBatch(1L, 500, first);
        registerBatch(2L, 300, second);

        assertTrue(endpoint.releaseCommittedItem(first));
        assertCommittedWorkNear(300);
    }

    @Test
    void releaseBatch_unknownBatchId_noEffect() {
        ScheduledRequest committed = item(100L, 1000);
        registerBatch(1L, 500, committed);
        ScheduledRequest unknown = item(999L, 1000);
        assertTrue(!endpoint.releaseCommittedItem(unknown));
        assertCommittedWorkNear(500);
    }

    @Test
    void releaseBatch_neverGoesNegative() {
        ScheduledRequest item = item(100L, 1000);
        registerBatch(1L, 100, item);
        assertTrue(endpoint.releaseCommittedItem(item));
        assertTrue(!endpoint.releaseCommittedItem(item));
        assertEquals(0, endpoint.getLoadMetric().orElseThrow());
    }

    private void assertCommittedWorkNear(long expectedMs) {
        long actualMs = endpoint.getLoadMetric().orElseThrow();
        assertTrue(actualMs <= expectedMs && actualMs >= expectedMs - 50,
                "Expected committed work near " + expectedMs + "ms but got " + actualMs + "ms");
    }

    @Test
    void calibrate_noInflight_resetsToZero() {
        registerBatch(1L, 500,
                item(100L, 1000));

        TaskInfo finished = task(100L, 1000, 0, 1L);
        finished.setErrorCode(0);
        calibrate(Map.of("100", finished), null);

        assertEquals(0, endpoint.getLoadMetric().orElseThrow());
        assertEquals(0, endpoint.getInflightBatchCount());
    }

    @Test
    void calibrate_finishedBatch_removedFromInflight() {
        registerBatch(5L, 9999,
                item(100L, 1000), item(101L, 2000));

        TaskInfo t1 = task(100L, 1000, 0, 5L);
        t1.setErrorCode(0);
        TaskInfo t2 = task(101L, 2000, 0, 5L);
        t2.setErrorCode(0);
        calibrate(Map.of("100", t1, "101", t2), null);

        assertEquals(0, endpoint.getLoadMetric().orElseThrow());
        assertEquals(0, endpoint.getInflightBatchCount());
    }

    @Test
    void calibrate_partialBatchFailure_repacks() {
        registerBatch(5L, 9999,
                item(100L, 1000), item(101L, 2000));

        TaskInfo failed = task(100L, 1000, 0, 5L);
        failed.setErrorCode(1);
        failed.setErrorMessage("timeout");
        TaskInfo success = task(101L, 2000, 0, 5L);
        success.setErrorCode(0);
        calibrate(Map.of("100", failed, "101", success), null);

        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.getLoadMetric().orElseThrow());
    }

    @Test
    void calibrate_inflightUnconfirmedBatchesSurvive() {
        registerBatch(5L, 1000,
                item(100L, 500));
        registerBatch(7L, 2000,
                item(200L, 1000));

        TaskInfo finished = task(100L, 500, 0, 5L);
        finished.setErrorCode(0);
        calibrate(Map.of("100", finished), null);

        assertEquals(1, endpoint.getInflightBatchCount());
        // Remaining work is predicted duration minus elapsed running time.
        assertTrue(Math.abs(endpoint.getLoadMetric().orElseThrow() - 2000) < 50,
                "Expected ~2000ms but got " + endpoint.getLoadMetric().orElseThrow());
    }

    @Test
    void repackBatch_removesFailedRequests() {
        ScheduledRequest first = item(100L, 1000);
        ScheduledRequest failed = item(101L, 2000);
        ScheduledRequest third = item(102L, 3000);
        registerBatch(5L, 9999, first, failed, third);
        assertTrue(endpoint.releaseCommittedItem(failed));

        assertEquals(2, endpoint.admissionPendingRequestCount());
    }

    @Test
    void repackBatch_allFailed_removesBatch() {
        ScheduledRequest item = item(100L, 1000);
        registerBatch(5L, 500, item);
        assertTrue(endpoint.releaseCommittedItem(item));

        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.getLoadMetric().orElseThrow());
    }

    @Test
    void ipPort_format() {
        assertEquals("10.0.0.1:8080", endpoint.ipPort());
    }

    @Test
    void generationPinCanBeReleasedByCompletionThread() {
        WorkerEndpoint.GenerationPin pin = endpoint.tryPinGeneration();

        CompletableFuture.runAsync(pin::close).join();

        assertThrows(IllegalArgumentException.class,
                () -> endpoint.requirePinnedGeneration(pin));
    }

    // ==================== getStatus() returns live reference ====================

    @Test
    void getStatus_returns_live_reference() {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setAlive(true);
        response.setAvailableConcurrency(42L);
        response.setDpRank(3);
        EndpointTestSupport.applyStatus(endpoint, response);

        WorkerStatus liveStatus = endpoint.getStatus();
        assertSame(status, liveStatus);
        assertTrue(liveStatus.pollHealth().reportedAlive());
        assertEquals(42L, (long) liveStatus.getAvailableConcurrency());
        assertEquals(3L, liveStatus.getDpRank());
    }

    // ==================== WorkerStatus response transaction ====================

    @Test
    void responseFieldsAndAppliedCursorsHaveSeparateCommitBoundaries() {
        WorkerStatusResponse resp = new WorkerStatusResponse();
        resp.setRole(RoleType.DECODE);
        resp.setAlive(true);
        resp.setAvailableConcurrency(8L);
        resp.setStepLatencyMs(25.0);
        resp.setIterateCount(100L);
        resp.setDpSize(4);
        resp.setTpSize(2);
        resp.setDpRank(1);
        resp.setMaxSeqLen(131072L);
        resp.setMaxBatchTokensSize(262144L);
        resp.setAvailableKvCacheTokens(10000L);
        resp.setStatusVersion(5L);
        resp.setLatestFinishedVersion(3L);

        WorkerStatus.PreparedStatus prepared;
        status.lock.lock();
        try {
            prepared = status.prepareNewStatus(
                    status.freezeStatusResponse(resp));
        } finally {
            status.lock.unlock();
        }

        assertEquals(RoleType.PREFILL, status.getRole());
        assertEquals(-1L, status.appliedStatusCursor().statusVersion());
        assertEquals(-1L,
                status.appliedStatusCursor().latestFinishedTaskVersion());

        status.lock.lock();
        try {
            status.publishPreparedStatus(prepared);
            status.recordSuccessfulPoll(true);
        } finally {
            status.lock.unlock();
        }
        assertEquals(RoleType.DECODE, status.getRole());
        assertTrue(status.pollHealth().reportedAlive());
        assertEquals(8L, (long) status.getAvailableConcurrency());
        assertEquals(25.0, status.getStepLatencyMs(), 0.001);
        assertEquals(100L, status.getIterateCount());
        assertEquals(4L, status.getDpSize());
        assertEquals(2L, status.getTpSize());
        assertEquals(1L, status.getDpRank());
        assertEquals(131072L, status.getMaxSeqLen());
        assertEquals(262144L, status.getMaxBatchTokensSize());
        assertEquals(10000L, status.getAvailableKvCacheTokens());
        assertEquals(5L, status.appliedStatusCursor().statusVersion());
        assertEquals(3L,
                status.appliedStatusCursor().latestFinishedTaskVersion());
    }

    @Test
    void nullStatusResponseIsRejectedBeforeMutation() {
        WorkerStatus.CommittedWorkerStatus before =
                status.committedWorkerStatus();
        assertThrows(NullPointerException.class,
                () -> status.freezeStatusResponse(null));
        assertSame(before, status.committedWorkerStatus());
    }

    // ==================== onWorkerStatusUpdate ====================

    @Test
    void foreignStatusGenerationCannotRebindEndpoint() {
        WorkerStatusResponse resp = new WorkerStatusResponse();
        resp.setRole(RoleType.PREFILL);
        resp.setStatusVersion(1L);
        resp.setLatestFinishedVersion(0L);
        WorkerStatus newStatus = EndpointTestSupport.workerStatus(
                RoleType.PREFILL, "group-b", "10.0.0.2",
                8082, 8083, "site-a");
        newStatus.lock.lock();
        try {
            WorkerStatus.PreparedStatus prepared = newStatus.prepareNewStatus(
                    newStatus.freezeStatusResponse(resp));
            assertThrows(IllegalArgumentException.class,
                    () -> endpoint.applyPreparedStatus(newStatus, prepared));
        } finally {
            newStatus.lock.unlock();
        }
        assertSame(status, endpoint.getStatus());
    }

    @Test
    void onWorkerStatusUpdate_calibrates_prefill() {
        WorkerStatusResponse resp = new WorkerStatusResponse();
        resp.setFinishedTaskInfo(Map.of("100", task(100L, 1000, 0, 1L)));

        // PrefillEndpoint calibrates even when runningTaskInfo is null
        EndpointTestSupport.applyStatus(endpoint, resp);
        // No exception = calibrate handled null gracefully
    }

    @Test
    void initializeFromAppliedStatusPreservesGenerationFields() {
        WorkerStatusResponse resp = new WorkerStatusResponse();
        resp.setDpRank(5);
        resp.setAlive(true);

        EndpointTestSupport.applyStatus(endpoint, resp);

        assertEquals("site-x", endpoint.getStatus().getSite());
        assertEquals("group-x", endpoint.getStatus().getGroup());
        assertEquals(5L, endpoint.getStatus().getDpRank());
        assertTrue(endpoint.getStatus().pollHealth().reportedAlive());
    }

    private void calibrate(Map<String, TaskInfo> finished, Map<String, TaskInfo> running) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setFinishedTaskInfo(finished);
        response.setRunningTaskInfo(running);
        EndpointTestSupport.applyStatus(endpoint, response);
    }

    private void registerBatch(
            long batchId,
            long predictedMs,
            ScheduledRequest... items) {
        for (ScheduledRequest item : items) {
            if (!EndpointTestSupport.offer(endpoint, item)) {
                throw new IllegalStateException(
                        "test item could not be offered to endpoint queue");
            }
        }
        try (PrefillState.CommittedHandoff ignored =
                     EndpointTestSupport.commitBatch(
                             endpoint, batchId, predictedMs, List.of(items))) {
            // Closing transfers only the generation handoff. The ledger keeps
            // the exact committed item identities until status/terminal facts.
        }
    }

    private ScheduledRequest item(long requestId, long seqLen) {
        return new ScheduledRequest(
                ctx(requestId, seqLen),
                null,
                null,
                null,
                null,
                endpoint,
                null,
                null,
                0L);
    }

    private BalanceContext ctx(long requestId, long seqLen) {
        Request req = new Request();
        req.setRequestId(requestId);
        req.setSeqLen(seqLen);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(req);
        ctx.setConfig(new FlexlbConfig());
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
