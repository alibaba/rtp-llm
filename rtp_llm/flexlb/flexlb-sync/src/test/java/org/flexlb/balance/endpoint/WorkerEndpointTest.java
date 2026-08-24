package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.BatchIdGenerator;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link WorkerEndpoint} 基类语义与端点状态引用行为测试。
 *
 * <p>Prefill 侧的等待估算与条目计数已迁移至状态账本（StateLedger per-EP
 * 视图）：本类不再覆盖本地 inflight 记账（commit/release/calibrate/repack
 * 已随旧路径移除），账本语义由 ledger 相关测试覆盖。</p>
 */
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
        endpoint = new PrefillEndpoint(status, config,
                Mockito.mock(EngineGrpcClient.class), Mockito.mock(BatchDispatchExecutor.class),
                new BatchIdGenerator("127.0.0.1", 7001), () -> 0,
                Mockito.mock(BatchSchedulerReporter.class), null);
    }

    @AfterEach
    void tearDown() {
        endpoint.close();
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
        resp.setMaxSeqLen(131072L);
        resp.setMaxBatchTokensSize(262144L);
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
        assertEquals(131072L, status.getMaxSeqLen());
        assertEquals(262144L, status.getMaxBatchTokensSize());
        assertEquals(10000L, status.getAvailableKvCacheTokens().get());
        assertEquals(5L, status.getStatusVersion().get());
        // latestFinishedTaskVersion is intentionally NOT set by updateFromResponse();
        // it is advanced only after the status-check runner consumed finished tasks
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
    void onWorkerStatusUpdate_handles_finished_tasks_gracefully() {
        // shadowBridge=null（退化模式）：状态 tick 只替换 status 引用并跳过
        // 账本计数刷新——携带 finishedTaskInfo 的报文不触发任何本地结算。
        WorkerStatusResponse resp = new WorkerStatusResponse();
        TaskInfo finished = new TaskInfo();
        finished.setRequestId(100L);
        finished.setErrorCode(0L);
        resp.setFinishedTaskInfo(Map.of("100", finished));

        endpoint.onWorkerStatusUpdate(status, resp);
        // No exception = null bridge handled gracefully
        assertEquals(0, endpoint.prefillActiveRequestCount(),
                "退化模式无记账源：读点恒零");
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
}
