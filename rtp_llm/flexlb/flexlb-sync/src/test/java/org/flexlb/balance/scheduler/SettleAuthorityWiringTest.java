package org.flexlb.balance.scheduler;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.constant.MetricConstant;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.route.RoleType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.service.monitor.FlexlbMetricHelper;
import org.flexlb.sync.shadow.StateShadowBridge;
import org.junit.jupiter.api.Test;

import java.util.Map;

import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * 终态结算换权（settle authority）装配分流验证：AbstractScheduler.register
 * 的 whenComplete 按账本启用状态分流——关 = 退化模式（metric 直报，旧四值
 * 口径连续）；开 = 权威单出口（COMPLETED 挂 pending 表由 ledger 终局消费，
 * metric 生产点迁移）。客户端 future 语义两侧不变。
 *
 * <p>结算/读取开关收束后（旧路径移除），authority 恒等于账本开关——分流
 * 只剩 enabled / disabled 两态。</p>
 */
class SettleAuthorityWiringTest {

    /** 最小调度器：直接驱动基类 register 的 whenComplete 分流。 */
    private static final class WiringScheduler extends AbstractScheduler {
        WiringScheduler(FlexlbMetricHelper helper, StateShadowBridge bridge) {
            super(helper, bridge);
        }

        @Override
        public CompletableFuture<Response> submit(BalanceContext ctx) {
            return new CompletableFuture<>();
        }
    }

    /** 账本关（退化模式）：metric 直报（ACK 时点，旧四值口径连续）。 */
    @Test
    void metricReportedDirectlyWhenLedgerDisabled() {
        Fixture fx = new Fixture(false);

        CompletableFuture<Response> future = new CompletableFuture<>();
        fx.scheduler.register(context(60L), future);
        future.complete(successResponse());

        verify(fx.monitor).report(eq(MetricConstant.REQUEST_SUCCESS_QPS), any(FlexMetricTags.class), eq(1.0));
        assertEquals(0, fx.bridge.pendingTerminalMetricCount(), "退化模式不得使用 pending 表");
        fx.tearDown();
    }

    /** 账本开：ACK 时直报静默，metric 挂 pending 由引擎终局消费。 */
    @Test
    void metricParkedThenConsumedWhenLedgerEnabled() {
        Fixture fx = new Fixture(true);
        long requestId = 61L;

        CompletableFuture<Response> future = new CompletableFuture<>();
        fx.scheduler.register(context(requestId), future);
        // 开账前置（正常 BATCH 流程由 BatchScheduler.submit 调用：P 开账 + D 预占）
        fx.bridge.onPrefillSubmit(requestId);
        fx.bridge.onDecodeReserve(requestId, 128L, 136L, RoleType.DECODE, "10.0.0.61:9000");

        future.complete(successResponse());

        verify(fx.monitor, never()).report(
                eq(MetricConstant.REQUEST_SUCCESS_QPS), any(FlexMetricTags.class), eq(1.0));
        assertEquals(1, fx.bridge.pendingTerminalMetricCount(), "ACK 后 metric 应挂 pending 表");

        // 引擎 D 侧 finished 终局 → pending 消费（单点生产出口）
        fx.bridge.observeWorkerStatus(finishedResponse(requestId), RoleType.DECODE, "10.0.0.61:9000");

        verify(fx.monitor).report(eq(MetricConstant.REQUEST_SUCCESS_QPS), any(FlexMetricTags.class), eq(1.0));
        assertEquals(0, fx.bridge.pendingTerminalMetricCount());
        fx.tearDown();
    }

    /** 账本开但 ledger 未覆盖（开账缺失）：退回旧语义 ACK 即报，不进 pending 生命周期。 */
    @Test
    void ackReportedImmediatelyWhenLedgerDoesNotCover() {
        Fixture fx = new Fixture(true);

        CompletableFuture<Response> future = new CompletableFuture<>();
        fx.scheduler.register(context(62L), future);
        // 不调 onPrefillSubmit——ledger 未覆盖（开账异常被吞的罕见场景）

        future.complete(successResponse());

        verify(fx.monitor).report(eq(MetricConstant.REQUEST_SUCCESS_QPS), any(FlexMetricTags.class), eq(1.0));
        assertEquals(0, fx.bridge.pendingTerminalMetricCount(),
                "ledger 未覆盖时不得挂 pending（无终局方会消费——泄漏）");
        fx.tearDown();
    }

    /** 账本开：FAILED 路由失败——双侧主动 settle，metric 即时出口。 */
    @Test
    void failedSettledAndReportedImmediatelyWhenLedgerEnabled() {
        Fixture fx = new Fixture(true);
        long requestId = 63L;

        CompletableFuture<Response> future = new CompletableFuture<>();
        fx.scheduler.register(context(requestId), future);
        fx.bridge.onPrefillSubmit(requestId);

        future.complete(Response.error(
                org.flexlb.dao.loadbalance.StrategyErrorType.WORKER_EXECUTION_FAILED, "boom"));

        verify(fx.monitor).report(eq(MetricConstant.REQUEST_FAILURE_QPS), any(FlexMetricTags.class), eq(1.0));
        assertEquals(0, fx.bridge.pendingTerminalMetricCount(), "FAILED 即时出口——不挂 pending");
        assertEquals(org.flexlb.state.TerminalState.FAILED,
                fx.bridge.ledger().terminalOutcomeOf(requestId, org.flexlb.state.spi.StateRole.PREFILL)
                        .orElseThrow().state());
        fx.tearDown();
    }

    // ==================== fixture ====================

    private static final class Fixture {
        final FlexMonitor monitor = mock(FlexMonitor.class);
        final StateShadowBridge bridge;
        final WiringScheduler scheduler;

        Fixture(boolean ledgerEnabled) {
            FlexlbConfig config = new FlexlbConfig();
            config.setFlexlbStateV2ShadowEnabled(ledgerEnabled);
            bridge = StateShadowBridge.create(config, monitor, false);
            scheduler = new WiringScheduler(
                    new FlexlbMetricHelper(monitor, MetricConstant.PATH_BATCH), bridge);
        }

        void tearDown() {
            bridge.close();
        }
    }

    private static Response successResponse() {
        Response response = new Response();
        response.setSuccess(true);
        return response;
    }

    /** D 侧 finished(success) 报文（单请求）。 */
    private static WorkerStatusResponse finishedResponse(long requestId) {
        TaskInfo finished = new TaskInfo();
        finished.setRequestId(requestId);
        finished.setErrorCode(0L);
        finished.setEndTimeMs(1L);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setStatusVersion(3L);
        response.setRole(RoleType.DECODE);
        response.setRunningDetailCount(0L);
        response.setFinishedTaskInfo(Map.of(String.valueOf(requestId), finished));
        return response;
    }

    private static BalanceContext context(long requestId) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(128);
        request.setMaxNewTokens(8);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setConfig(new FlexlbConfig());
        return ctx;
    }
}
