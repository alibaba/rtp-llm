package org.flexlb.worker.endpoint;

import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.balance.schedule.RequestResult;
import org.flexlb.balance.schedule.SchedulingTestConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.status.TaskInfo;
import org.flexlb.worker.status.WorkerStatus;
import org.flexlb.worker.status.WorkerStatusResponse;
import org.flexlb.worker.status.WorkerStatusTestSupport;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Checks worker identity and status commit boundaries.
 */
class WorkerEndpointTest {
    private WorkerStatus status;
    private PrefillEndpoint endpoint;

    @BeforeEach
    void setUp() {
        status =
                EndpointTestSupport.workerStatus(
                        RoleType.PREFILL, "group-x", "10.0.0.1", 8080, 8081, "site-x");
        endpoint =
                new PrefillEndpoint(
                        status,
                        SchedulingTestConfig.newConfig(),
                        new org.flexlb.worker.endpoint.CapacityNotifier());
    }

    @AfterEach
    void tearDown() {
        org.flexlb.worker.endpoint.EndpointTestSupport.closeAndWait(endpoint);
    }

    @Test
    void ipPort_format() {
        assertEquals("10.0.0.1:8080@0", endpoint.ipPort());
    }

    @Test
    void endpointIdentityRemainsStableAfterRetirement() {
        WorkerEndpoint selectedEndpoint = endpoint;
        assertSame(status, selectedEndpoint.getStatus());
        org.flexlb.worker.endpoint.EndpointTestSupport.closeAndWait(endpoint);
        assertTrue(selectedEndpoint.hasStoppedAcceptingRequests());
        assertSame(status, selectedEndpoint.getStatus());
    }

    // ==================== getStatus() returns live reference ====================

    @Test
    void getStatus_returns_live_reference() {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setAlive(true);
        response.setAvailableConcurrency(42L);
        response.setDpRank(3);
        EndpointTestSupport.applyStatus(endpoint, response).run();

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
            prepared =
                    status.prepareStatus(
                            WorkerStatusTestSupport.freezeStatusResponse(status, resp));
        } finally {
            status.lock.unlock();
        }

        assertEquals(RoleType.PREFILL, status.getRole());
        assertEquals(-1L, status.appliedStatusCursor().statusVersion());
        assertEquals(-1L, status.appliedStatusCursor().latestFinishedTaskVersion());

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
        assertEquals(10000L, status.committedEngineObservation().availableKvCacheTokens());
        assertEquals(5L, status.appliedStatusCursor().statusVersion());
        assertEquals(3L, status.appliedStatusCursor().latestFinishedTaskVersion());
    }

    // ==================== onWorkerStatusUpdate ====================

    @Test
    void foreignStatusGenerationCannotRebindEndpoint() {
        WorkerStatusResponse resp = new WorkerStatusResponse();
        resp.setRole(RoleType.PREFILL);
        resp.setStatusVersion(1L);
        resp.setLatestFinishedVersion(0L);
        WorkerStatus newStatus =
                EndpointTestSupport.workerStatus(
                        RoleType.PREFILL, "group-b", "10.0.0.2", 8082, 8083, "site-a");
        newStatus.lock.lock();
        try {
            WorkerStatus.PreparedStatus prepared =
                    newStatus.prepareStatus(
                            WorkerStatusTestSupport.freezeStatusResponse(newStatus, resp));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> endpoint.applyWorkerStatus(newStatus, prepared));
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
        EndpointTestSupport.applyStatus(endpoint, resp).run();
        // No exception = calibrate handled null gracefully
    }

    @Test
    void initializeFromAppliedStatusPreservesGenerationFields() {
        WorkerStatusResponse resp = new WorkerStatusResponse();
        resp.setDpRank(5);
        resp.setAlive(true);

        EndpointTestSupport.applyStatus(endpoint, resp).run();

        assertEquals("site-x", endpoint.getStatus().getSite());
        assertEquals("group-x", endpoint.getStatus().getGroup());
        assertEquals(5L, endpoint.getStatus().getDpRank());
        assertTrue(endpoint.getStatus().pollHealth().reportedAlive());
    }

    private void calibrate(Map<String, TaskInfo> finished, Map<String, TaskInfo> running) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setFinishedTaskInfo(finished);
        response.setRunningTaskInfo(running);
        EndpointTestSupport.applyStatus(endpoint, response).run();
    }

    private RequestResult item(long requestId, long seqLen) {
        return org.flexlb.balance.schedule.RequestExecutionTestSupport.mockProtocolRoute(
                SchedulingTestConfig.createRoute(
                        ctx(requestId, seqLen), null, null, null, endpoint, null, null, 0L));
    }

    private RequestContext ctx(long requestId, long seqLen) {
        Request req = Request.builder().requestId(String.valueOf(requestId)).seqLen(seqLen).build();
        RequestContext ctx = new RequestContext(SchedulingTestConfig.newConfig());
        ctx.setRequest(req);
        return ctx;
    }

    private TaskInfo task(long requestId, long inputLength, long prefixLength, long batchId) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(String.valueOf(requestId));
        task.setInputLength(inputLength);
        task.setPrefixLength(prefixLength);
        task.setBatchId(batchId);
        return task;
    }
}
