package org.flexlb.balance.endpoint;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.sync.shadow.StateShadowBridge;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * {@link DecodeEndpoint} 账本口径测试：KV 预占与全部调度读点走 StateLedger
 * per-EP 视图（真 bridge 装配，janitor 手动模式）。
 *
 * <p>旧两层 inflight 记账（本地 layer-1 预占 + engineWork layer-2 镜像、
 * calibrate 状态机、stale-round / TTL 驱逐）已随旧路径移除：引擎接管与终局
 * 由事件泵（{@code bridge.observeWorkerStatus}）驱动，陈旧/兜底清理由
 * LedgerJanitor 承担——其语义由 flexlb-state 的 LedgerJanitor 测试覆盖，
 * 本类不再重复。</p>
 *
 * <p>时序约定：reserve/release 直入账本（无本地镜像）；读点走 per-EP 计数
 * 缓存，由引擎状态 tick（{@code onWorkerStatusUpdate}）刷新——测试需显式
 * 调 {@link #refreshCounters()}。</p>
 */
class DecodeEndpointTest {

    private WorkerStatus status;
    private StateShadowBridge bridge;
    private DecodeEndpoint endpoint;
    /** 报文级版本屏障：跨报严格单调递增（引擎契约）。 */
    private long statusVersion;

    @BeforeEach
    void setUp() {
        status = new WorkerStatus();
        status.setIp("10.0.0.1");
        status.setPort(8080);
        status.setGrpcPort(8081);
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbStateV2ShadowEnabled(true);
        bridge = StateShadowBridge.create(config, null, false);
        endpoint = new DecodeEndpoint(status, config, bridge);
    }

    @AfterEach
    void tearDown() {
        bridge.close();
    }

    // ---- KV reservation accounting (ledger per-EP view) ----

    @Test
    void reserve_accountsUnconfirmedReservation() {
        updateStatus(null, null, 10000);
        endpoint.reserve(100L, 500, 500);
        refreshCounters();

        assertEquals(1, endpoint.decodeInflightCount());
        assertEquals(500, endpoint.decodeInflightHardKvReserved());
        assertEquals(500, endpoint.decodeInflightExpectedKvReserved());
        assertEquals(9500, endpoint.decodeRealKvAvailable());
    }

    @Test
    void reserve_withoutEngineTickKeepsZeroCounters() {
        // 计数缓存由引擎状态 tick 刷新——未 tick 时读点全零（低估退化，不阻断调度）
        endpoint.reserve(100L, 500, 500);

        assertEquals(0, endpoint.decodeInflightCount());
        assertEquals(0, endpoint.decodeTotalLoad());
    }

    @Test
    void availableKvTokens_accountsForReservations() {
        updateStatus(null, null, 10000);
        endpoint.reserve(100L, 3000, 3000);
        endpoint.reserve(101L, 2000, 2000);
        refreshCounters();

        assertEquals(5000, endpoint.decodeRealKvAvailable());
        assertEquals(2, endpoint.decodeInflightCount());
    }

    @Test
    void ipPort_format() {
        assertEquals("10.0.0.1:8080", endpoint.ipPort());
    }

    // ---- release (pre-terminal abandonment) ----

    @Test
    void release_retiresLedgerReservation() {
        endpoint.reserve(100L, 500, 500);
        endpoint.reserve(101L, 300, 300);
        endpoint.release(100L);
        refreshCounters();

        assertEquals(1, endpoint.decodeInflightCount());
    }

    @Test
    void release_unknownRequestId_noEffect() {
        endpoint.reserve(100L, 500, 500);
        endpoint.release(999L);
        refreshCounters();

        assertEquals(1, endpoint.decodeInflightCount());
    }

    @Test
    void release_isIdempotentAndNeverGoesNegative() {
        updateStatus(null, null, 10000);
        endpoint.reserve(100L, 100, 100);
        endpoint.release(100L);
        endpoint.release(100L);
        refreshCounters();

        assertEquals(0, endpoint.decodeInflightCount());
        assertEquals(10000, endpoint.decodeRealKvAvailable());
    }

    // ---- engine acceptance: unconfirmed reservation hands over to engine facts ----

    @Test
    void engineAcceptanceReleasesUnconfirmedReservation() {
        endpoint.reserve(100L, 500, 500);

        TaskInfo loading = task(100L);
        loading.setPhase(TaskPhase.KV_ALLOCATED);
        updateStatus(Map.of("100", loading), null, 10000);

        // 确认临界：未确认预占双轨撤账，引擎事实接管
        assertEquals(0, endpoint.decodeInflightCount());
        assertEquals(0, endpoint.decodeInflightHardKvReserved());
        assertEquals(10000, endpoint.decodeRealKvAvailable());
    }

    @Test
    void engineTickWithoutObservationKeepsUnconfirmedAccount() {
        // 空报文（无 running 明细）不产生观察事件——条目保持未确认
        endpoint.reserve(100L, 500, 500);
        updateStatus(null, null, 10000);

        assertEquals(1, endpoint.decodeInflightCount());
        assertEquals(9500, endpoint.decodeRealKvAvailable());
    }

    @Test
    void engineFinishedFailureSettlesEntry() {
        endpoint.reserve(100L, 500, 500);

        TaskInfo failed = task(100L);
        failed.setErrorCode(1);
        failed.setErrorMessage("timeout");
        updateStatus(null, Map.of("100", failed), 10000);

        assertEquals(0, endpoint.decodeInflightCount());
        assertEquals(0, endpoint.decodeTotalLoad());
    }

    @Test
    void engineFinishedSuccessSettlesEntry() {
        endpoint.reserve(100L, 500, 500);

        TaskInfo success = task(100L);
        success.setErrorCode(0);
        updateStatus(null, Map.of("100", success), 10000);

        assertEquals(0, endpoint.decodeInflightCount());
        assertEquals(0, endpoint.decodeTotalLoad());
    }

    // ---- engine phase mapping (ledger phase population) ----

    @Test
    void engineKvAllocatedLandsInLoadingPhase() {
        endpoint.reserve(100L, 500, 500);

        TaskInfo loading = task(100L);
        loading.setPhase(TaskPhase.KV_ALLOCATED);
        updateStatus(Map.of("100", loading), null, 10000);

        assertEquals(0, endpoint.decodeInflightCount());
        assertEquals(1, endpoint.decodeEngineWorkCount());
        assertEquals(1, endpoint.decodeEngineLoadingCount());
        assertEquals(0, endpoint.decodeEngineWaitingCount());
        assertEquals(0, endpoint.decodeEngineRunningCount());
        // 引擎接管后未确认预占撤账，可用 KV 恢复引擎上报值
        assertEquals(0, endpoint.decodeInflightHardKvReserved());
        assertEquals(10000, endpoint.decodeRealKvAvailable());
    }

    @Test
    void engineRunningLandsInRunningPhase() {
        endpoint.reserve(100L, 500, 500);

        TaskInfo running = task(100L);
        running.setPhase(TaskPhase.RUNNING);
        updateStatus(Map.of("100", running), null, 10000);

        assertEquals(0, endpoint.decodeInflightCount());
        assertEquals(1, endpoint.decodeEngineWorkCount());
        assertEquals(1, endpoint.decodeEngineRunningCount());
    }

    @Test
    void enginePendingAndReceivedLandInWaitingPhase() {
        // 保守观察位：PENDING / RECEIVED 均映射至派发等待窗口（DISPATCHED 相位）。
        // 预占撤账临界是引擎加载（D_LOADING）——等待窗口条目仍持有影子预占。
        endpoint.reserve(100L, 500, 500);

        TaskInfo pending = task(100L);
        pending.setPhase(TaskPhase.PENDING);
        updateStatus(Map.of("100", pending), null, 10000);

        assertEquals(1, endpoint.decodeInflightCount(),
                "DISPATCHED 仍在未确认窗口（影子预占到 D_LOADING 才撤）");
        assertEquals(1, endpoint.decodeEngineWorkCount(), "引擎已见条目计 engineOwned");
        assertEquals(1, endpoint.decodeEngineWaitingCount());
        assertEquals(500L, endpoint.decodeInflightHardKvReserved());
        assertEquals(9500L, endpoint.decodeRealKvAvailable());
        // 全相位活跃：仍是同一条目，无重复计数
        assertEquals(1, endpoint.decodeTotalLoad());

        endpoint.reserve(101L, 300, 300);
        TaskInfo received = task(101L);
        received.setPhase(TaskPhase.RECEIVED);
        updateStatus(Map.of("101", received), null, 10000);

        assertEquals(2, endpoint.decodeInflightCount(), "两条目均未达加载临界");
        assertEquals(2, endpoint.decodeEngineWorkCount());
        assertEquals(2, endpoint.decodeEngineWaitingCount());
    }

    @Test
    void enginePhaseAdvancesInPlace() {
        endpoint.reserve(100L, 500, 500);

        TaskInfo loading = task(100L);
        loading.setPhase(TaskPhase.KV_ALLOCATED);
        updateStatus(Map.of("100", loading), null, 10000);
        assertEquals(1, endpoint.decodeEngineLoadingCount());

        TaskInfo running = task(100L);
        running.setPhase(TaskPhase.RUNNING);
        updateStatus(Map.of("100", running), null, 10000);

        // 相位就地推进——不产生重复条目
        assertEquals(1, endpoint.decodeEngineRunningCount());
        assertEquals(1, endpoint.decodeEngineWorkCount());
        assertEquals(0, endpoint.decodeInflightCount());
    }

    @Test
    void engineFinishRemovesEntryAfterAcceptance() {
        endpoint.reserve(100L, 500, 500);

        TaskInfo running = task(100L);
        running.setPhase(TaskPhase.RUNNING);
        updateStatus(Map.of("100", running), null, 10000);
        assertEquals(1, endpoint.decodeTotalLoad());

        TaskInfo finished = task(100L);
        finished.setErrorCode(0);
        updateStatus(null, Map.of("100", finished), 10000);

        assertEquals(0, endpoint.decodeEngineWorkCount());
        assertEquals(0, endpoint.decodeTotalLoad());
    }

    @Test
    void unregisteredEngineReportIsNotAdopted() {
        // 影子开账语义：正常 observe 模式只认本地已开账条目——引擎上报的
        // 外来请求只计 unknown 事件、不收养（收养仅 rebuild 重放路径）。
        TaskInfo foreign = task(999L);
        foreign.setPhase(TaskPhase.RUNNING);
        updateStatus(Map.of("999", foreign), null, 10000);

        assertEquals(0, endpoint.decodeEngineWorkCount());
        assertEquals(0, endpoint.decodeTotalLoad());
        assertEquals(0, endpoint.decodeInflightHardKvReserved());
        assertEquals(0, endpoint.decodeInflightExpectedKvReserved());
    }

    // ---- release after engine acceptance ----

    @Test
    void release_retiresEntryEvenAfterEngineAcceptance() {
        // 主动放弃（pre-terminal abandonment）：确认后释放撤引擎事实账并移除条目
        endpoint.reserve(100L, 500, 500);

        TaskInfo running = task(100L);
        running.setPhase(TaskPhase.RUNNING);
        updateStatus(Map.of("100", running), null, 10000);
        assertEquals(1, endpoint.decodeEngineWorkCount());

        endpoint.release(100L);
        refreshCounters();

        assertEquals(0, endpoint.decodeEngineWorkCount());
        assertEquals(0, endpoint.decodeTotalLoad());
    }

    // ---- absolute-value views across acceptance ----

    @Test
    void newViewsReportAbsoluteValuesAfterAcceptance() {
        updateStatus(null, null, 10000);
        status.getTotalKvCacheTokens().set(20000);
        endpoint.reserve(100L, 500, 800);
        endpoint.reserve(101L, 300, 400);

        // 100 被引擎接管，101 仍未确认（updateStatus 自带缓存刷新）
        TaskInfo running = task(100L);
        running.setPhase(TaskPhase.KV_ALLOCATED);
        updateStatus(Map.of("100", running), null, 10000);

        assertEquals(2, endpoint.decodeTotalLoad());
        assertEquals(1, endpoint.decodeInflightCount());
        assertEquals(300, endpoint.decodeInflightHardKvReserved());
        assertEquals(400, endpoint.decodeInflightExpectedKvReserved());
        assertEquals(20000, endpoint.decodeKvTotal());
    }

    // ---- metrics wiring ----

    @Test
    void reportBatchMetrics_reportsHardAndExpectedKvSeparately() {
        updateStatus(null, null, 10000);
        endpoint.reserve(100L, 500, 900);
        refreshCounters();

        BatchSchedulerReporter reporter = Mockito.mock(BatchSchedulerReporter.class);
        endpoint.reportBatchMetrics(reporter);

        Mockito.verify(reporter).reportInflightRequestCount("DECODE", "10.0.0.1", 1);
        Mockito.verify(reporter).reportDecodeTotalLoad("10.0.0.1", 1);
        Mockito.verify(reporter).reportDecodeInflightKvReserved("10.0.0.1", 900L);
        Mockito.verify(reporter).reportDecodeInflightKvReservedHard("10.0.0.1", 500L);
        // Phase-split: no engine observation yet — all zero
        Mockito.verify(reporter).reportDecodeEngineWaitingCount("10.0.0.1", 0);
        Mockito.verify(reporter).reportDecodeEngineLoadingCount("10.0.0.1", 0);
        Mockito.verify(reporter).reportDecodeEngineRunningCount("10.0.0.1", 0);
        // Layer breakdown: 1 unconfirmed reservation, 0 engine-owned
        Mockito.verify(reporter).reportDecodeInflightRequestsCount("10.0.0.1", 1);
        Mockito.verify(reporter).reportDecodeEngineWorkCount("10.0.0.1", 0);
    }

    @Test
    void reportBatchMetrics_reportsPhaseSplitAndLayerCounts() {
        updateStatus(null, null, 10000);
        endpoint.reserve(100L, 500, 900);
        endpoint.reserve(101L, 300, 400);
        endpoint.reserve(102L, 200, 300);

        // 引擎接管三个请求，观察相位各异：PENDING→派发等待、KV_ALLOCATED→装载、RUNNING→执行
        TaskInfo waiting = task(100L);
        waiting.setPhase(TaskPhase.PENDING);
        TaskInfo loading = task(101L);
        loading.setPhase(TaskPhase.KV_ALLOCATED);
        TaskInfo running = task(102L);
        running.setPhase(TaskPhase.RUNNING);
        updateStatus(Map.of("100", waiting, "101", loading, "102", running), null, 10000);

        // PENDING 条目仍在未确认窗口；KV_ALLOCATED / RUNNING 条目已达撤账临界
        assertEquals(1, endpoint.decodeInflightCount());
        assertEquals(3, endpoint.decodeEngineWorkCount());
        assertEquals(1, endpoint.decodeEngineWaitingCount());
        assertEquals(1, endpoint.decodeEngineLoadingCount());
        assertEquals(1, endpoint.decodeEngineRunningCount());

        BatchSchedulerReporter reporter = Mockito.mock(BatchSchedulerReporter.class);
        endpoint.reportBatchMetrics(reporter);

        Mockito.verify(reporter).reportDecodeEngineWaitingCount("10.0.0.1", 1);
        Mockito.verify(reporter).reportDecodeEngineLoadingCount("10.0.0.1", 1);
        Mockito.verify(reporter).reportDecodeEngineRunningCount("10.0.0.1", 1);
        Mockito.verify(reporter).reportDecodeInflightRequestsCount("10.0.0.1", 1);
        Mockito.verify(reporter).reportDecodeEngineWorkCount("10.0.0.1", 3);
    }

    // ==================== helpers ====================

    /**
     * 引擎状态报文 tick：先泵入账本（相位迁移 / 终局裁决），再触发端点
     * 计数缓存刷新（生产链路由 Runner 的 versionAdvanced 分支驱动）。
     */
    private void updateStatus(Map<String, TaskInfo> running, Map<String, TaskInfo> finished,
                              long availableKvCacheTokens) {
        status.getAvailableKvCacheTokens().set(availableKvCacheTokens);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setStatusVersion(++statusVersion);
        response.setRole(RoleType.DECODE);
        response.setRunningTaskInfo(running);
        response.setFinishedTaskInfo(finished);
        // 上报完整性：detailCount 与明细数一致（引擎契约字段）
        response.setRunningDetailCount(running == null ? 0L : running.size());
        bridge.observeWorkerStatus(response, RoleType.DECODE, endpoint.ipPort());
        endpoint.onWorkerStatusUpdate(status, response);
    }

    /** 刷新 per-EP 账本计数缓存（reserve/release 后显式触发——读点不按需聚合）。 */
    private void refreshCounters() {
        endpoint.onWorkerStatusUpdate(status, null);
    }

    private TaskInfo task(long requestId) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        return task;
    }
}
