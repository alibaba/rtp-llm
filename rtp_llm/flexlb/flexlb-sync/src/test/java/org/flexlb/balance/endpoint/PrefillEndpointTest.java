package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.BatchIdGenerator;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.sync.shadow.StateShadowBridge;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

/**
 * {@link PrefillEndpoint} 账本口径测试：条目计数、等待估算与引擎侧相位
 * 读点走 StateLedger per-EP 视图（真 bridge 装配，janitor 手动模式）。
 *
 * <p>旧两层 inflight 记账（layer-1 批次条目 + layer-2 engineWork 镜像、
 * calibrate 状态机、repack / TTL 驱逐）已随旧路径移除：终局由 ledger
 * 单出口 settle（引擎 finished / 本地取消），陈旧/兜底清理由 LedgerJanitor
 * 承担——其语义由 flexlb-state 的 LedgerJanitor 测试覆盖，本类不再重复。</p>
 *
 * <p>时序约定（生产链路对齐）：{@code submit} 开账（BatchScheduler.submit
 * 的 {@code onPrefillSubmit} 挂点：register + onQueued）→ {@code commitBatch}
 * / {@code commitRequest} 派发绑定（onDispatching + 分摊预测 + onDispatched
 * 入端点桶）→ 引擎报文经事件泵（{@code bridge.observeWorkerStatus}）驱动
 * 相位与终局。读点走 per-EP 计数缓存，由引擎状态 tick 刷新。</p>
 */
class PrefillEndpointTest {

    private PrefillEndpoint endpoint;
    private FlexlbConfig config;
    private StateShadowBridge bridge;
    /** 报文级版本屏障：跨报严格单调递增（引擎契约）。 */
    private long statusVersion;

    @BeforeEach
    void setUp() {
        WorkerStatus status = new WorkerStatus();
        status.setIp("127.0.0.1");
        status.setPort(8080);
        status.setGrpcPort(8090);
        status.setRole(RoleType.PREFILL);

        config = new FlexlbConfig();
        config.setFlexlbBatchQueueMaxSize(100);
        config.setFlexlbBatchFixedWaitMs(300);
        config.setCostFormula("10 + 0.1*sum(computeTokens) + 5*batchSize");
        config.setFlexlbStateV2ShadowEnabled(true);
        bridge = StateShadowBridge.create(config, null, false);

        endpoint = new PrefillEndpoint(status, config,
                mock(EngineGrpcClient.class), mock(BatchDispatchExecutor.class),
                new BatchIdGenerator("127.0.0.1", 7001), () -> 0,
                mock(BatchSchedulerReporter.class), bridge);
    }

    @AfterEach
    void tearDown() {
        endpoint.close();
        bridge.close();
    }

    // ---- dispatch commit (ledger attach points) ----

    @Test
    void commitBatchCountsEachMemberRequest() {
        assertEquals(0, endpoint.prefillActiveRequestCount());

        submit(1L);
        submit(2L);
        endpoint.commitBatch(1L, 100, List.of(
                createBatchItem(1L, 500, 200),
                createBatchItem(2L, 300, 100)));
        refreshCounters();

        assertEquals(2, endpoint.prefillActiveRequestCount());
        assertEquals(2, endpoint.prefillPendingRequestCount());
    }

    @Test
    void commitMultipleBatches() {
        submit(1L);
        submit(2L);
        submit(3L);
        endpoint.commitBatch(1L, 100, List.of(
                createBatchItem(1L, 500, 200),
                createBatchItem(2L, 300, 100)));
        endpoint.commitBatch(2L, 50, List.of(createBatchItem(3L, 400, 0)));
        refreshCounters();

        assertEquals(3, endpoint.prefillActiveRequestCount());
        assertEquals(3, endpoint.prefillPendingRequestCount());
    }

    @Test
    void commitWithoutRegistrationIsDroppedByLedger() {
        // 开账前置：未 register 的条目 dispatch 挂点为防御 no-op——不入端点桶
        endpoint.commitBatch(1L, 100, List.of(createBatchItem(1L, 500, 200)));
        refreshCounters();

        assertEquals(0, endpoint.prefillActiveRequestCount());
    }

    @Test
    void commitRequestCountsSingleRequest() {
        submit(42L);
        endpoint.commitRequest(42L, 100);
        refreshCounters();

        assertEquals(1, endpoint.prefillActiveRequestCount());
        assertEquals(1, endpoint.prefillPendingRequestCount());

        // 引擎 finished 终局 → 条目移除（ledger 单出口 settle）
        pumpEngineReport(null, Map.of("42", finishedTask(42L, -1L, 0)));

        assertEquals(0, endpoint.prefillActiveRequestCount());
        assertEquals(0, endpoint.prefillPendingRequestCount());
    }

    // ---- engine acceptance & phase read points ----

    @Test
    void engineAcceptanceMovesEntryToEngineOwned() {
        submit(1L);
        endpoint.commitBatch(1L, 100, List.of(createBatchItem(1L, 500, 200)));
        refreshCounters();
        assertEquals(1, endpoint.prefillActiveRequestCount());
        assertEquals(0, endpoint.prefillEngineOwnedCount());

        // PENDING 观察 → 保守接收位（P_RECEIVED）：引擎已见但未进等待窗口
        pumpEngineReport(Map.of("1", runningTask(1L, 1L, TaskPhase.PENDING)), null);

        assertEquals(1, endpoint.prefillActiveRequestCount());
        assertEquals(1, endpoint.prefillEngineOwnedCount());
        assertEquals(0, endpoint.prefillEngineWaitingCount(),
                "P_RECEIVED 是接收位——等待窗口从 KV 装载起算");
        assertEquals(0, endpoint.prefillEngineRunningCount());

        // KV_ALLOCATED 观察 → 装载完毕的等待窗口
        pumpEngineReport(Map.of("1", runningTask(1L, 1L, TaskPhase.KV_ALLOCATED)), null);

        assertEquals(1, endpoint.prefillEngineWaitingCount());
        assertEquals(0, endpoint.prefillEngineRunningCount());
    }

    @Test
    void enginePhaseSplitReadPoints() {
        submit(1L);
        submit(2L);
        endpoint.commitBatch(1L, 100, List.of(
                createBatchItem(1L, 500, 200),
                createBatchItem(2L, 300, 100)));

        TaskInfo waiting = runningTask(1L, 1L, TaskPhase.KV_ALLOCATED);
        TaskInfo running = runningTask(2L, 1L, TaskPhase.RUNNING);
        pumpEngineReport(Map.of("1", waiting, "2", running), null);

        assertEquals(2, endpoint.prefillEngineOwnedCount());
        assertEquals(1, endpoint.prefillEngineWaitingCount());
        assertEquals(1, endpoint.prefillEngineRunningCount());
        assertEquals(2, endpoint.prefillActiveRequestCount());
    }

    @Test
    void engineFinishRemovesEntry() {
        submit(1L);
        endpoint.commitBatch(1L, 100, List.of(createBatchItem(1L, 500, 200)));
        pumpEngineReport(Map.of("1", runningTask(1L, 1L, TaskPhase.RUNNING)), null);
        assertEquals(1, endpoint.prefillActiveRequestCount());

        pumpEngineReport(null, Map.of("1", finishedTask(1L, 1L, 0)));

        assertEquals(0, endpoint.prefillActiveRequestCount());
        assertEquals(0, endpoint.prefillEngineOwnedCount());
        assertEquals(0, endpoint.prefillPendingRequestCount());
    }

    @Test
    void partialFinishShrinksBatch() {
        submit(1L);
        submit(2L);
        endpoint.commitBatch(1L, 100, List.of(
                createBatchItem(1L, 500, 200),
                createBatchItem(2L, 300, 100)));
        pumpEngineReport(Map.of("1", runningTask(1L, 1L, TaskPhase.RUNNING)), null);

        // 成员 1 终局——成员 2 仍活跃（请求级口径：批次成员独立结算）
        pumpEngineReport(null, Map.of("1", finishedTask(1L, 1L, 0)));

        assertEquals(1, endpoint.prefillActiveRequestCount());
        assertEquals(1, endpoint.prefillPendingRequestCount());
    }

    @Test
    void unregisteredEngineReportIsNotAdopted() {
        // 影子开账语义：正常 observe 模式只认本地已开账条目——外来请求
        // 只计 unknown 事件、不收养（收养仅 rebuild 重放路径）。
        pumpEngineReport(Map.of("999", runningTask(999L, -1L, TaskPhase.RUNNING)), null);

        assertEquals(0, endpoint.prefillActiveRequestCount());
        assertEquals(0, endpoint.prefillEngineOwnedCount());
    }

    // ---- estimated waiting time ----

    @Test
    void estimatedWaitTimeZeroWhenNoInflight() {
        refreshCounters();
        assertEquals(0, endpoint.prefillEstimatedWaitTimeMs());
    }

    @Test
    void estimatedWaitTimeSumsPerRequestShares() {
        submit(7L);
        endpoint.commitRequest(7L, 3000);
        submit(8L);
        endpoint.commitRequest(8L, 2000);
        refreshCounters();

        assertEquals(5000, endpoint.prefillEstimatedWaitTimeMs());
    }

    @Test
    void estimatedWaitTimeSplitsBatchPredictionAcrossMembers() {
        submit(1L);
        submit(2L);
        // 批次预测 100ms 分摊到 2 个成员——各 50ms
        endpoint.commitBatch(1L, 100, List.of(
                createBatchItem(1L, 500, 200),
                createBatchItem(2L, 300, 100)));
        refreshCounters();

        assertEquals(100, endpoint.prefillEstimatedWaitTimeMs());
    }

    @Test
    void estimatedWaitTimeKeepsEngineRunningEntriesAtFullValue() {
        // 保守高估（拒绝偏向）：执行中条目不打折——新到请求仍按完整
        // 分摊预测排队估算（旧路径按 elapsed 折扣的口径已随旧账本移除）。
        submit(1L);
        endpoint.commitBatch(1L, 5000, List.of(createBatchItem(1L, 500, 200)));
        pumpEngineReport(Map.of("1", runningTask(1L, 1L, TaskPhase.RUNNING)), null);

        assertEquals(5000, endpoint.prefillEstimatedWaitTimeMs());
    }

    // ---- prefillPendingRequestCount ----

    @Test
    void pendingRequestCountIncludesBatcherQueue() throws InterruptedException {
        // Initially, batcher queue is empty
        assertEquals(0, endpoint.prefillPendingRequestCount());

        BatchItem item = createBatchItem(1L, 500, 200);
        endpoint.getBatcher().offer(item);

        long deadlineMs = System.currentTimeMillis() + 100;
        while (endpoint.prefillPendingRequestCount() == 0 && System.currentTimeMillis() < deadlineMs) {
            Thread.sleep(1);
        }
        assertTrue(endpoint.prefillPendingRequestCount() > 0, "Pending count should include batcher queue");
    }

    // ---- WorkerEndpoint inherited behavior ----

    @Test
    void onWorkerStatusUpdateUpdatesAliveStatus() {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        WorkerStatus status = new WorkerStatus();
        status.setIp("127.0.0.1");
        status.setPort(8080);
        status.setAlive(true);

        endpoint.onWorkerStatusUpdate(status, response);

        assertTrue(endpoint.getStatus().isAlive());
    }

    // ---- close ----

    @Test
    void closeShutsDownBatcher() {
        assertNotNull(endpoint.getBatcher());
        endpoint.close();
        // After close, offering should fail (batcher is stopped)
        BatchItem item = createBatchItem(1L, 500, 200);
        endpoint.getBatcher().offer(item);
        // Should not throw — batcher handles stopped state
    }

    // ---- batcher queue view ----

    @Test
    void batcherQueueSizeViewMatchesBatcher() {
        assertEquals(endpoint.getBatcher().queueSize(), endpoint.prefillBatcherQueueSize());
        assertEquals(0, endpoint.prefillBatcherQueueSize());
    }

    // ==================== helpers ====================

    /** 开账（BatchScheduler.submit 的 onPrefillSubmit 挂点：register + onQueued）。 */
    private void submit(long requestId) {
        bridge.onPrefillSubmit(requestId);
    }

    /** 刷新 per-EP 账本计数缓存（引擎状态 tick；测试显式触发）。 */
    private void refreshCounters() {
        endpoint.onWorkerStatusUpdate(endpoint.getStatus(), null);
    }

    /**
     * 引擎状态报文 tick：先泵入账本（相位迁移 / 终局裁决），再触发端点
     * 计数缓存刷新（生产链路由 Runner 的 versionAdvanced 分支驱动）。
     */
    private void pumpEngineReport(Map<String, TaskInfo> running, Map<String, TaskInfo> finished) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setStatusVersion(++statusVersion);
        response.setRole(RoleType.PREFILL);
        response.setRunningTaskInfo(running);
        response.setFinishedTaskInfo(finished);
        // 上报完整性：detailCount 与明细数一致（引擎契约字段）
        response.setRunningDetailCount(running == null ? 0L : running.size());
        bridge.observeWorkerStatus(response, RoleType.PREFILL, endpoint.ipPort());
        refreshCounters();
    }

    private TaskInfo runningTask(long requestId, long batchId, TaskPhase phase) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setBatchId(batchId);
        task.setPhase(phase);
        return task;
    }

    private TaskInfo finishedTask(long requestId, long batchId, long errorCode) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setBatchId(batchId);
        task.setErrorCode(errorCode);
        task.setEndTimeMs(System.currentTimeMillis());
        return task;
    }

    private BatchItem createBatchItem(long requestId, long seqLen, long hitCacheLen) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);

        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);

        ServerStatus prefill = new ServerStatus();
        prefill.setRole(RoleType.PREFILL);
        prefill.setServerIp("127.0.0.1");
        prefill.setHttpPort(8080);
        prefill.setGrpcPort(8090);
        DebugInfo debugInfo = new DebugInfo();
        debugInfo.setHitCacheLen(hitCacheLen);
        prefill.setDebugInfo(debugInfo);

        return new BatchItem(ctx, new java.util.concurrent.CompletableFuture<>(), null,
                prefill, null, endpoint, null, System.currentTimeMillis());
    }
}
