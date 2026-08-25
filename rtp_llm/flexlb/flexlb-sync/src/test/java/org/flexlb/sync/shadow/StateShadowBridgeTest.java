package org.flexlb.sync.shadow;

import org.flexlb.balance.scheduler.TerminalReason;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.constant.MetricConstant;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.state.DecodeEndpointCounters;
import org.flexlb.state.LedgerJanitorConfig;
import org.flexlb.state.TerminalState;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

/**
 * 影子开关矩阵：关 = 装配返回 DISABLED 单例、所有入口零执行（ledger 不存在）；
 * 开 = 正常泵入（event 计数递增、账本入账）；影子异常绝不外抛（catch-all）。
 */
class StateShadowBridgeTest {

    @Test
    void shouldReturnDisabledSingleton_whenSwitchOff() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbStateV2ShadowEnabled(false);

        StateShadowBridge bridge = StateShadowBridge.create(config, null);

        assertSame(StateShadowBridge.DISABLED, bridge, "开关关必须返回 DISABLED 单例");
        assertFalse(bridge.isEnabled());
        assertNull(bridge.ledger(), "DISABLED 不持有 ledger（零执行）");
        assertNull(bridge.diffCollector(), "DISABLED 不持有 diff collector（零执行）");
    }

    @Test
    void shouldBeNoOpForAllEntryPoints_whenDisabled() {
        StateShadowBridge bridge = StateShadowBridge.DISABLED;

        // 所有入口调用必须无异常、零副作用（主路径不受影响）
        bridge.observeWorkerStatus(null, RoleType.DECODE, "10.0.0.1:9000");
        bridge.onPrefillSubmit(1L);
        bridge.onDecodeReserve(1L, 100L, 200L, RoleType.DECODE, "10.0.0.1:9000");
        bridge.onLocalCancelRequested(1L);
        bridge.onOldTerminal(1L, "COMPLETED");

        // 喂 null response 也不抛（短路在第一行，不触达 translator）
        bridge.observeWorkerStatus(null, null, null);
    }

    @Test
    void shouldCreateEnabledBridge_whenSwitchOn() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbStateV2ShadowEnabled(true);

        // autoStartJanitor=false：不建调度线程（测试钩子；线程启停专项见 janitorScheduler* 测试）
        StateShadowBridge bridge = StateShadowBridge.create(config, null, false);

        assertTrue(bridge.isEnabled());
        assertNotNull(bridge.ledger());
        assertNotNull(bridge.diffCollector());
        assertNotNull(bridge.janitor(), "shadow 开时 janitor 必须挂载");
    }

    @Test
    void shouldPumpObservationIntoLedger_whenEnabled() {
        StateShadowBridge bridge = enabledBridge();

        // 开账前置：D 侧 reserve（正常 observe 不收养未开条目）
        bridge.onDecodeReserve(42L, 100L, 200L, RoleType.DECODE, "10.0.0.2:9000");

        TaskInfo finished = new TaskInfo();
        finished.setRequestId(42L);
        finished.setErrorCode(0L);
        finished.setEndTimeMs(1L);

        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setStatusVersion(3L);
        response.setRole(RoleType.DECODE);
        response.setRunningDetailCount(0L);
        response.setFinishedTaskInfo(Map.of("42", finished));

        bridge.observeWorkerStatus(response, RoleType.DECODE, "10.0.0.2:9000");

        // 泵入计数 + 影子终态对账：D 侧 finished(success) → COMPLETED 墓碑 + diff new 侧记录
        assertEquals(1L, bridge.diffCollector().eventCount());
        assertEquals(0L, bridge.diffCollector().errorCount());
        assertTrue(bridge.ledger().terminalOutcomeOf(42L, org.flexlb.state.spi.StateRole.DECODE).isPresent(),
                "D 侧 finished 应在影子账本产生 COMPLETED 墓碑");
        assertEquals(TerminalState.COMPLETED,
                bridge.ledger().terminalOutcomeOf(42L, org.flexlb.state.spi.StateRole.DECODE).get().state());
        assertEquals(1, bridge.diffCollector().pendingNew(),
                "旧侧终态未达 → new 侧入窗等待");
    }

    @Test
    void shouldSettleCancelledOnBothSides_whenOldPathTerminalIsCancel() {
        StateShadowBridge bridge = enabledBridge();

        // 开账前置：P 侧 submit + D 侧 reserve（两侧条目均存在后 cancel 双清才双侧留墓碑）
        bridge.onPrefillSubmit(7L);
        bridge.onDecodeReserve(7L, 100L, 200L, RoleType.DECODE, "10.0.0.3:9000");

        // 旧路径终态 CANCELLED → 影子双清 + diff 旧侧记录
        bridge.onOldTerminal(7L, "CANCELLED");

        assertTrue(bridge.ledger().terminalOutcomeOf(7L, org.flexlb.state.spi.StateRole.DECODE).isPresent(),
                "旧路径 CANCELLED 应触发影子 D 侧 settle");
        assertTrue(bridge.ledger().terminalOutcomeOf(7L, org.flexlb.state.spi.StateRole.PREFILL).isPresent(),
                "旧路径 CANCELLED 应触发影子 P 侧双清");
        assertEquals(TerminalState.CANCELLED,
                bridge.ledger().terminalOutcomeOf(7L, org.flexlb.state.spi.StateRole.DECODE).get().state());
    }

    /**
     * 本地取消 reason 分流（reason 完备性 sync 侧闭环）：任一侧条目引擎已见 →
     * CANCELLED_IMPLICIT（已见后取消，无 ack 推定成立）；两侧均未达引擎 →
     * CANCELLED_NEVER_ARRIVED（未派发即取消，无需引擎证据）。
     * 产出点：StateShadowBridge#localCancelReason（onOldTerminal /
     * settleBothSidesAuthoritatively 共用）。
     */
    @Test
    void cancelledReasonSplitsByEngineSeen() {
        StateShadowBridge bridge = enabledBridge();

        // 分流一：两侧均未达引擎（无任何引擎上报）→ 未达取消
        bridge.onPrefillSubmit(60L);
        bridge.onDecodeReserve(60L, 100L, 200L, RoleType.DECODE, "10.0.0.20:9000");
        bridge.onOldTerminal(60L, "CANCELLED");

        assertEquals(org.flexlb.state.TerminalReason.CANCELLED_NEVER_ARRIVED,
                bridge.ledger().terminalOutcomeOf(60L, org.flexlb.state.spi.StateRole.PREFILL)
                        .orElseThrow().reason(),
                "从未到达引擎的取消 → CANCELLED_NEVER_ARRIVED");
        assertEquals(org.flexlb.state.TerminalReason.CANCELLED_NEVER_ARRIVED,
                bridge.ledger().terminalOutcomeOf(60L, org.flexlb.state.spi.StateRole.DECODE)
                        .orElseThrow().reason());

        // 分流二：引擎已见（running 上报观察到该请求）→ 隐式取消
        bridge.onPrefillSubmit(61L);
        bridge.onDecodeReserve(61L, 100L, 200L, RoleType.DECODE, "10.0.0.21:9000");
        bridge.observeWorkerStatus(decodeRunningResponse(61L, 1L), RoleType.DECODE, "10.0.0.21:9000");
        assertTrue(bridge.ledger().decode().get(61L).isPresent()
                        && bridge.ledger().decode().get(61L).orElseThrow().engineOwned(),
                "前置：running 上报后 D 条目应 engineOwned");
        bridge.onOldTerminal(61L, "CANCELLED");

        assertEquals(org.flexlb.state.TerminalReason.CANCELLED_IMPLICIT,
                bridge.ledger().terminalOutcomeOf(61L, org.flexlb.state.spi.StateRole.DECODE)
                        .orElseThrow().reason(),
                "引擎已见的取消 → CANCELLED_IMPLICIT");
        assertEquals(org.flexlb.state.TerminalReason.CANCELLED_IMPLICIT,
                bridge.ledger().terminalOutcomeOf(61L, org.flexlb.state.spi.StateRole.PREFILL)
                        .orElseThrow().reason(),
                "cancel 双清沿用同一 outcome——两侧 reason 一致");
    }

    @Test
    void shouldRegisterPrefillAndReserveDecode_whenLocalLifecyclePointsCalled() {
        StateShadowBridge bridge = enabledBridge();

        // 本地 P 侧 submit：register + onQueued
        bridge.onPrefillSubmit(11L);
        assertTrue(bridge.ledger().prefill().get(11L).isPresent(), "P 侧 submit 应入账");

        // 本地 D 侧 reserve（binding 惰性注册，不依赖事件泵先到）
        bridge.onDecodeReserve(11L, 100L, 200L, RoleType.DECODE, "10.0.0.4:9000");
        assertTrue(bridge.ledger().decode().get(11L).isPresent(), "D 侧 reserve 应入账");
    }

    @Test
    void shouldNeverThrow_whenShadowPipelineExplodes() {
        StateShadowBridge bridge = enabledBridge();

        // 构造触发翻译异常的输入：WorkerStatusResponse mock 使 getRunningTaskInfo 抛出
        WorkerStatusResponse exploding = new WorkerStatusResponse() {
            @Override
            public Map<String, TaskInfo> getRunningTaskInfo() {
                throw new IllegalStateException("boom");
            }
        };
        exploding.setStatusVersion(1L);
        exploding.setRole(RoleType.DECODE);

        // catch-all：异常吞掉、计 error，绝不外抛
        bridge.observeWorkerStatus(exploding, RoleType.DECODE, "10.0.0.5:9000");

        assertEquals(1L, bridge.diffCollector().errorCount());
        assertEquals(0L, bridge.diffCollector().eventCount());
    }

    private static StateShadowBridge enabledBridge() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbStateV2ShadowEnabled(true);
        // autoStartJanitor=false——不创建调度线程（确定性 + 线程卫生）
        return StateShadowBridge.create(config, null, false);
    }

    // ==================== master 重启重建（首报收养）====================

    /** running 明细 TaskInfo 构造 helper。 */
    private static TaskInfo runningTask(long requestId, TaskPhase phase, long kvTokens) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setPhase(phase);
        task.setKvTokens(kvTokens);
        return task;
    }

    private static WorkerStatusResponse statusResponse(long statusVersion, RoleType role,
                                                        Map<String, TaskInfo> running,
                                                        Map<String, TaskInfo> finished) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setStatusVersion(statusVersion);
        response.setRole(role);
        response.setRunningDetailCount(running == null ? 0L : (long) running.size());
        if (running != null) {
            response.setRunningTaskInfo(running);
        }
        if (finished != null) {
            response.setFinishedTaskInfo(finished);
        }
        return response;
    }

    /**
     * master 重启后每端点首份报文：未开账 running 按引擎事实收养入账（恢复
     * 重启前丢失的 inflight 计数——引擎上报是唯一事实源）；同端点后续报文
     * 正常 observe（未知 running 不再收养）。
     */
    @Test
    void shouldAdoptUnknownRunningOnFirstReport_thenObserveNormally() {
        StateShadowBridge bridge = enabledBridge();

        // 首报（master 重启语义，无本地开账前置）：running 42 → 收养
        bridge.observeWorkerStatus(
                statusResponse(1L, RoleType.DECODE,
                        Map.of("42", runningTask(42L, TaskPhase.RUNNING, 512L)), null),
                RoleType.DECODE, "10.0.0.9:9000");

        var adopted = bridge.ledger().decode().get(42L).orElseThrow();
        assertTrue(adopted.engineOwned(), "首报未知 running 应按引擎事实收养");
        assertEquals(0L, bridge.diffCollector().errorCount());

        // 次报（同端点）：未知 running 43 不再收养（正常 observe 只计 unknown）
        bridge.observeWorkerStatus(
                statusResponse(2L, RoleType.DECODE,
                        Map.of("43", runningTask(43L, TaskPhase.RUNNING, 0L)), null),
                RoleType.DECODE, "10.0.0.9:9000");
        assertTrue(bridge.ledger().decode().get(43L).isEmpty(), "次报未知 running 不收养");

        // 不同端点（新 ipPort）各自独立首报：另一端点的首报同样收养
        bridge.observeWorkerStatus(
                statusResponse(1L, RoleType.DECODE,
                        Map.of("77", runningTask(77L, TaskPhase.RUNNING, 0L)), null),
                RoleType.DECODE, "10.0.0.11:9000");
        assertTrue(bridge.ledger().decode().get(77L).isPresent(), "新端点首报同样收养");
        assertEquals(0L, bridge.diffCollector().errorCount());
    }

    /**
     * 收养条目经引擎 finished 正常终局（重启恢复的 inflight 走完生命周期）：
     * 墓碑落账 + 新侧终态入 diff 对账窗口（重启场景无旧侧配对是预期形态，
     * 不算 shadow.error）。
     */
    @Test
    void adoptedEntrySettlesViaEngineFinishedWithoutShadowError() {
        StateShadowBridge bridge = enabledBridge();

        // 首报收养 42
        bridge.observeWorkerStatus(
                statusResponse(1L, RoleType.DECODE,
                        Map.of("42", runningTask(42L, TaskPhase.RUNNING, 512L)), null),
                RoleType.DECODE, "10.0.0.10:9000");

        // 次报：引擎 finished 42 → 终局墓碑 + 新侧终态入对账窗口
        TaskInfo finished = new TaskInfo();
        finished.setRequestId(42L);
        finished.setErrorCode(0L);
        finished.setEndTimeMs(1L);
        bridge.observeWorkerStatus(
                statusResponse(2L, RoleType.DECODE, null, Map.of("42", finished)),
                RoleType.DECODE, "10.0.0.10:9000");

        assertTrue(bridge.ledger().terminalOutcomeOf(42L, org.flexlb.state.spi.StateRole.DECODE).isPresent(),
                "收养条目 finished 后应终局落墓碑");
        assertEquals(TerminalState.COMPLETED,
                bridge.ledger().terminalOutcomeOf(42L, org.flexlb.state.spi.StateRole.DECODE).get().state());
        assertEquals(1, bridge.diffCollector().pendingNew(),
                "重启收养条目无旧侧配对 → 新侧终态入窗等待（预期形态，非 error）");
        assertEquals(0L, bridge.diffCollector().errorCount());
    }

    // ==================== 清理层装配（janitor 挂载/调度/配置传播）====================

    /** 关 = janitor 零挂载零调度（DISABLED 单例不创建任何线程）。 */
    @Test
    void janitorNotMountedWhenShadowDisabled() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbStateV2ShadowEnabled(false);

        StateShadowBridge bridge = StateShadowBridge.create(config, null);

        assertSame(StateShadowBridge.DISABLED, bridge);
        assertNull(bridge.janitor(), "关态不挂载 janitor（零执行铁律）");
        bridge.runJanitorOnce(); // no-op 不抛
    }

    /** autoStartJanitor=false：janitor 挂载但无调度线程（可注入测试钩子验证）。 */
    @Test
    void janitorMountedWithoutSchedulerThreadWhenAutoStartFalse() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbStateV2ShadowEnabled(true);

        boolean threadBefore = janitorThreadAlive();
        StateShadowBridge bridge = StateShadowBridge.create(config, null, false);

        assertNotNull(bridge.janitor(), "janitor 仍挂载（手动驱动）");
        assertEquals(threadBefore, janitorThreadAlive(), "autoStart=false 不得创建调度线程");
        bridge.runJanitorOnce(); // 手动驱动一 tick，无异常
        assertTrue(bridge.isEnabled());
    }

    /** 开 + autoStart：调度线程（flexlb-state-janitor）启动；close() 停止且幂等。 */
    @Test
    void janitorSchedulerThreadStartsAndStopsWithClose() throws Exception {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbStateV2ShadowEnabled(true);
        config.setFlexlbStateV2JanitorIntervalMs(20L); // 高频驱动（验证多 tick 无异常）

        StateShadowBridge bridge = StateShadowBridge.create(config, null, true);
        try {
            assertTrue(waitFor(() -> janitorThreadAlive(), 2_000L),
                    "shadow 开 + autoStart → janitor 调度线程应启动");
            bridge.runJanitorOnce(); // 与调度线程双驱动（单调度线程不重入，安全）
        } finally {
            bridge.close();
        }
        assertTrue(waitFor(() -> !janitorThreadAlive(), 2_000L), "close() 应停止 janitor 调度线程");
        bridge.close(); // 幂等
    }

    /** 清理层参数从 FlexlbConfig 传播到 LedgerJanitorConfig（与影子开关同装配模式）。 */
    @Test
    void janitorConfigWiredFromFlexlbConfig() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbStateV2ShadowEnabled(true);
        config.setFlexlbStateV2StaleRounds(5);
        config.setFlexlbStateV2TtlMs(1_234L);
        config.setFlexlbStateV2HardCapMs(5_678L);

        StateShadowBridge bridge = StateShadowBridge.create(config, null, false);

        LedgerJanitorConfig jc = bridge.janitor().config();
        assertEquals(5, jc.staleRounds());
        assertEquals(1_234L, jc.ttlMs());
        assertEquals(5_678L, jc.hardCapMs());
        assertEquals(LedgerJanitorConfig.DEFAULT_SCAN_BUDGET_PER_TICK, jc.scanBudgetPerTick());
    }

    /** janitor 胜者结算须进影子 diff 对账窗口（否则旧侧终态后到会误报 missing_on_new）。 */
    @Test
    void janitorWinnerSettleEntersShadowDiffWindow() throws Exception {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbStateV2ShadowEnabled(true);
        config.setFlexlbStateV2TtlMs(50L); // 小 TTL（hardCap 默认 900s ≫ 50）
        StateShadowBridge bridge = StateShadowBridge.create(config, null, false);

        // D 侧开账（createdAt = 真实时钟）
        bridge.onDecodeReserve(42L, 100L, 200L, RoleType.DECODE, "10.0.0.9:9000");
        assertTrue(bridge.ledger().decode().get(42L).isPresent());

        bridge.runJanitorOnce(); // 未到期 → 无事
        assertTrue(bridge.ledger().decode().get(42L).isPresent());

        Thread.sleep(80L); // createdAt + 80 > ttl=50
        bridge.runJanitorOnce(); // janitor TTL 胜者结算

        assertTrue(bridge.ledger().decode().get(42L).isEmpty(), "TTL 到期应被 janitor 结算");
        assertEquals(1, bridge.diffCollector().pendingNew(),
                "janitor 胜者结算须进 diff 对账窗口（settleListener 补全）");
        assertEquals(TerminalState.SLO_TIMEOUT,
                bridge.ledger().terminalOutcomeOf(42L, org.flexlb.state.spi.StateRole.DECODE)
                        .orElseThrow().state());
    }

    // ==================== 结算权威单出口（onOldTerminalAuthority 权威入口）====================

    /** 开关收束：settle 开关退化为 no-op——shadow 关时不再 fail-fast，装配仍返回 DISABLED。 */
    @Test
    void settleSwitchIsIgnoredWhenShadowOff() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbStateV2ShadowEnabled(false);
        config.setFlexlbStateV2SettleEnabled(true);

        StateShadowBridge bridge = StateShadowBridge.create(config, null, false);

        assertSame(StateShadowBridge.DISABLED, bridge,
                "旧路径移除后 settle 开关不再独立生效——shadow 关即 DISABLED");
        assertFalse(bridge.isSettleAuthority(), "DISABLED 恒非权威");
    }

    /** 开关收束：结算权威恒等于账本启用位（settle 开关不再独立选路）。 */
    @Test
    void settleAuthorityMatrix() {
        FlexlbConfig settleOff = new FlexlbConfig();
        settleOff.setFlexlbStateV2ShadowEnabled(true);
        settleOff.setFlexlbStateV2SettleEnabled(false);
        assertTrue(StateShadowBridge.create(settleOff, null, false).isSettleAuthority(),
                "settle 开关已收束——shadow 开即权威结算");
        assertFalse(StateShadowBridge.DISABLED.isSettleAuthority(), "DISABLED 恒非权威");
        assertTrue(authorityBridge().isSettleAuthority(), "shadow 开 = 权威结算");
    }

    /** 开关收束：settle 开关不再拦截权威入口——COMPLETED 恒挂 pending 表等 ledger 终局。 */
    @Test
    void authorityEntryPointRunsRegardlessOfSettleSwitch() {
        StateShadowBridge bridge = enabledBridge(); // shadow 开、settle 关
        bridge.onPrefillSubmit(31L);

        bridge.onOldTerminalAuthority(31L, "COMPLETED",
                new StateShadowBridge.TerminalMetricContext(TerminalReason.COMPLETED, "PREFILL", "10.0.0.1"));

        assertEquals(1, bridge.pendingTerminalMetricCount(),
                "权威入口恒执行——settle 开关不再拦截");
        assertTrue(bridge.ledger().prefill().get(31L).isPresent(),
                "COMPLETED 不提前结算 ledger 条目（等引擎终局）");
    }

    /** COMPLETED（ACK）：不提前 settle——两侧条目保留，metric 挂 pending 表等 ledger 终局。 */
    @Test
    void completedAckParksMetricWithoutSettlingLedger() {
        StateShadowBridge bridge = authorityBridge();
        bridge.onPrefillSubmit(32L);
        bridge.onDecodeReserve(32L, 100L, 200L, RoleType.DECODE, "10.0.0.6:9000");

        bridge.onOldTerminalAuthority(32L, "COMPLETED",
                new StateShadowBridge.TerminalMetricContext(TerminalReason.COMPLETED, "PREFILL", "10.0.0.6"));

        assertEquals(1, bridge.pendingTerminalMetricCount(), "ACK 后 metric 应挂 pending 表");
        assertTrue(bridge.ledger().prefill().get(32L).isPresent(),
                "ACK 不得提前结算 P 侧（引擎执行相位保留）");
        assertTrue(bridge.ledger().decode().get(32L).isPresent(),
                "ACK 不得提前结算 D 侧（KV 计费移交保留）");
    }

    /** COMPLETED 挂 pending 后由引擎 D 侧 finished 终局消费（单点生产）。 */
    @Test
    void pendingMetricConsumedAtEngineTerminalExit() {
        StateShadowBridge bridge = authorityBridge();
        bridge.onPrefillSubmit(33L);
        bridge.onDecodeReserve(33L, 100L, 200L, RoleType.DECODE, "10.0.0.7:9000");
        bridge.onOldTerminalAuthority(33L, "COMPLETED",
                new StateShadowBridge.TerminalMetricContext(TerminalReason.COMPLETED, "PREFILL", "10.0.0.7"));
        assertEquals(1, bridge.pendingTerminalMetricCount());

        bridge.observeWorkerStatus(finishedResponse(33L), RoleType.DECODE, "10.0.0.7:9000");

        assertEquals(0, bridge.pendingTerminalMetricCount(), "引擎终局出口应消费挂起 metric");
        assertTrue(bridge.ledger().terminalOutcomeOf(33L, org.flexlb.state.spi.StateRole.DECODE).isPresent());
    }

    /** 乱序兜底：引擎 finished 早于 ACK 到达时，挂 pending 后自查立即消费。 */
    @Test
    void pendingMetricSelfConsumedWhenEngineTerminalPrecedesAck() {
        StateShadowBridge bridge = authorityBridge();
        bridge.onPrefillSubmit(34L);
        bridge.onDecodeReserve(34L, 100L, 200L, RoleType.DECODE, "10.0.0.8:9000");
        bridge.observeWorkerStatus(finishedResponse(34L), RoleType.DECODE, "10.0.0.8:9000");
        assertTrue(bridge.ledger().terminalOutcomeOf(34L, org.flexlb.state.spi.StateRole.DECODE).isPresent(),
                "前置：引擎终局先到（墓碑已生成）");

        bridge.onOldTerminalAuthority(34L, "COMPLETED",
                new StateShadowBridge.TerminalMetricContext(TerminalReason.COMPLETED, "PREFILL", "10.0.0.8"));

        assertEquals(0, bridge.pendingTerminalMetricCount(), "墓碑已在——挂 pending 后自查应立即消费");
    }

    /** FAILED：master 已判死——双侧主动 settle + metric 即时出口（不挂 pending）。 */
    @Test
    void failedSettlesBothSidesImmediately() {
        StateShadowBridge bridge = authorityBridge();
        bridge.onPrefillSubmit(35L);
        bridge.onDecodeReserve(35L, 100L, 200L, RoleType.DECODE, "10.0.0.9:9000");

        bridge.onOldTerminalAuthority(35L, "FAILED",
                new StateShadowBridge.TerminalMetricContext(TerminalReason.FAILED, "PREFILL", "10.0.0.9"));

        assertEquals(TerminalState.FAILED,
                bridge.ledger().terminalOutcomeOf(35L, org.flexlb.state.spi.StateRole.PREFILL)
                        .orElseThrow().state(), "FAILED 应主动结算 P 侧");
        assertEquals(TerminalState.FAILED,
                bridge.ledger().terminalOutcomeOf(35L, org.flexlb.state.spi.StateRole.DECODE)
                        .orElseThrow().state(), "FAILED 应主动结算 D 侧");
        assertEquals(0, bridge.pendingTerminalMetricCount(), "FAILED metric 即时出口——不挂 pending");
    }

    /** CANCELLED：双侧主动 settle 为 CANCELLED（本地取消）。 */
    @Test
    void cancelledSettlesBothSidesImmediately() {
        StateShadowBridge bridge = authorityBridge();
        bridge.onPrefillSubmit(36L);
        bridge.onDecodeReserve(36L, 100L, 200L, RoleType.DECODE, "10.0.0.10:9000");

        bridge.onOldTerminalAuthority(36L, "CANCELLED",
                new StateShadowBridge.TerminalMetricContext(TerminalReason.CANCELLED, "PREFILL", "10.0.0.10"));

        assertEquals(TerminalState.CANCELLED,
                bridge.ledger().terminalOutcomeOf(36L, org.flexlb.state.spi.StateRole.PREFILL)
                        .orElseThrow().state());
        assertEquals(TerminalState.CANCELLED,
                bridge.ledger().terminalOutcomeOf(36L, org.flexlb.state.spi.StateRole.DECODE)
                        .orElseThrow().state());
    }

    /** TIMED_OUT：双侧主动 settle 为 SLO_TIMEOUT（存活时间上限，reason 完备性 sync 侧闭环——
     * 产出点 settleBothSidesAuthoritatively 旧路径超时通道）。 */
    @Test
    void timedOutSettlesBothSidesImmediately() {
        StateShadowBridge bridge = authorityBridge();
        bridge.onPrefillSubmit(37L);

        bridge.onOldTerminalAuthority(37L, "TIMED_OUT",
                new StateShadowBridge.TerminalMetricContext(TerminalReason.TIMED_OUT, "PREFILL", "unknown"));

        assertEquals(TerminalState.SLO_TIMEOUT,
                bridge.ledger().terminalOutcomeOf(37L, org.flexlb.state.spi.StateRole.PREFILL)
                        .orElseThrow().state());
        assertEquals(org.flexlb.state.TerminalReason.SLO_BUDGET_EXHAUSTED,
                bridge.ledger().terminalOutcomeOf(37L, org.flexlb.state.spi.StateRole.PREFILL)
                        .orElseThrow().reason(),
                "旧路径 TIMED_OUT → SLO_BUDGET_EXHAUSTED");
    }

    /** 引擎事件丢失场景：COMPLETED 挂 pending 后由 janitor TTL 胜者结算消费。 */
    @Test
    void pendingMetricConsumedByJanitorWhenEngineEventsLost() throws Exception {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbStateV2ShadowEnabled(true);
        config.setFlexlbStateV2SettleEnabled(true);
        config.setFlexlbStateV2TtlMs(50L);
        StateShadowBridge bridge = StateShadowBridge.create(config, null, false);

        bridge.onPrefillSubmit(38L);
        bridge.onDecodeReserve(38L, 100L, 200L, RoleType.DECODE, "10.0.0.11:9000");
        bridge.onOldTerminalAuthority(38L, "COMPLETED",
                new StateShadowBridge.TerminalMetricContext(TerminalReason.COMPLETED, "PREFILL", "10.0.0.11"));
        assertEquals(1, bridge.pendingTerminalMetricCount());

        Thread.sleep(80L); // createdAt + 80 > ttl=50
        bridge.runJanitorOnce(); // janitor TTL 胜者结算 → listener → pending 消费

        assertTrue(bridge.ledger().decode().get(38L).isEmpty(), "TTL 到期应被 janitor 结算");
        assertEquals(0, bridge.pendingTerminalMetricCount(),
                "janitor 胜者结算出口应消费挂起 metric（引擎事件丢失不吞 metric）");
    }

    /** metric 即时出口（FAILED）：经统一 helper 上报 REQUEST_FAILURE_QPS（与旧出口同口径）。 */
    @Test
    void failedMetricReportedThroughUnifiedHelper() {
        FlexMonitor monitor = mock(FlexMonitor.class);
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbStateV2ShadowEnabled(true);
        config.setFlexlbStateV2SettleEnabled(true);
        StateShadowBridge bridge = StateShadowBridge.create(config, monitor, false);

        bridge.onOldTerminalAuthority(39L, "FAILED",
                new StateShadowBridge.TerminalMetricContext(TerminalReason.FAILED, "DECODE", "10.0.0.12"));

        verify(monitor).report(eq(MetricConstant.REQUEST_FAILURE_QPS), any(FlexMetricTags.class), eq(1.0));
    }

    /** 权威结算的 metric 上下文消费（COMPLETED）：引擎终局出口上报 REQUEST_SUCCESS_QPS。 */
    @Test
    void completedMetricReportedAtEngineTerminalExit() {
        FlexMonitor monitor = mock(FlexMonitor.class);
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbStateV2ShadowEnabled(true);
        config.setFlexlbStateV2SettleEnabled(true);
        StateShadowBridge bridge = StateShadowBridge.create(config, monitor, false);

        bridge.onPrefillSubmit(40L);
        bridge.onDecodeReserve(40L, 100L, 200L, RoleType.DECODE, "10.0.0.13:9000");
        bridge.onOldTerminalAuthority(40L, "COMPLETED",
                new StateShadowBridge.TerminalMetricContext(TerminalReason.COMPLETED, "PREFILL", "10.0.0.13"));
        verify(monitor, org.mockito.Mockito.never()).report(
                eq(MetricConstant.REQUEST_SUCCESS_QPS), any(FlexMetricTags.class), eq(1.0));

        bridge.observeWorkerStatus(finishedResponse(40L), RoleType.DECODE, "10.0.0.13:9000");

        verify(monitor).report(eq(MetricConstant.REQUEST_SUCCESS_QPS), any(FlexMetricTags.class), eq(1.0));
    }

    // ==================== 调度读点与 EP 记账（账本门面）====================

    /** 开关收束：read 开关退化为 no-op——不再要求 settle 前置，任意组合不拒启。 */
    @Test
    void readSwitchIsIgnoredWithoutFormerSettlePrerequisite() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbStateV2ShadowEnabled(true);
        config.setFlexlbStateV2ReadEnabled(true);
        config.setFlexlbStateV2SettleEnabled(false);

        StateShadowBridge bridge = StateShadowBridge.create(config, null, false);

        assertTrue(bridge.isReadAuthority(),
                "read 开关已收束——shadow 开即读权威（settle 不再是前置）");
    }

    /** 开关收束：读权威恒等于账本启用位（read/settle 开关不再独立选路）。 */
    @Test
    void readAuthorityMatrix() {
        FlexlbConfig readOff = new FlexlbConfig();
        readOff.setFlexlbStateV2ShadowEnabled(true);
        readOff.setFlexlbStateV2ReadEnabled(false);
        assertTrue(StateShadowBridge.create(readOff, null, false).isReadAuthority(),
                "read 开关已收束——shadow 开即读权威");
        assertTrue(authorityBridge().isReadAuthority(), "shadow 开即读权威");
        assertFalse(StateShadowBridge.DISABLED.isReadAuthority(), "DISABLED 恒非读权威");
        assertTrue(readAuthorityBridge().isReadAuthority(), "shadow 开即读权威");
    }

    /** D 侧权威预占入账 + per-EP 计数读数（读取换权的调度读点数据源）。 */
    @Test
    void decodeReserveAuthorityFeedsEndpointCounters() {
        StateShadowBridge bridge = readAuthorityBridge();
        String ipPort = "10.0.0.21:9000";
        int endpointId = ipPort.hashCode();

        bridge.decodeReserveAuthority(51L, 100L, 200L, ipPort);

        assertTrue(bridge.ledger().decode().get(51L).isPresent(), "权威预占应入账 ledger");
        DecodeEndpointCounters counters = bridge.decodeEndpointCounters(endpointId);
        assertEquals(1, counters.activeTotal(), "per-EP 计数应含该预占条目");
        assertEquals(1, counters.unconfirmedCount(), "RESERVED 相位条目计 unconfirmed");
        assertEquals(200L, counters.unconfirmedExpectedKv(), "未确认预占 expected KV 口径");
        assertEquals(100L, counters.unconfirmedSeqKv(), "未确认预占 seqLen 口径");
        assertEquals(0, counters.engineOwnedCount(), "无引擎观察时非 engineOwned");
    }

    /** 权威释放撤账：release 后条目移除、per-EP 计数归零（幂等）。 */
    @Test
    void decodeReleaseAuthorityDropsEntryAndCounters() {
        StateShadowBridge bridge = readAuthorityBridge();
        String ipPort = "10.0.0.22:9000";
        int endpointId = ipPort.hashCode();
        bridge.decodeReserveAuthority(52L, 100L, 200L, ipPort);
        assertEquals(1, bridge.decodeEndpointCounters(endpointId).activeTotal());

        bridge.decodeReleaseAuthority(52L);

        assertTrue(bridge.ledger().decode().get(52L).isEmpty(), "权威释放应移除条目");
        assertEquals(0, bridge.decodeEndpointCounters(endpointId).activeTotal(), "释放后 per-EP 计数归零");
        bridge.decodeReleaseAuthority(52L); // 幂等：重复释放不抛
    }

    /** 开关收束：read 开关不再拦截记账入口——shadow 开时权威预占/释放照常入账。 */
    @Test
    void decodeAuthorityEntryPointsRunRegardlessOfReadSwitch() {
        StateShadowBridge bridge = authorityBridge(); // shadow 开、read/settle 关
        String ipPort = "10.0.0.23:9000";

        bridge.decodeReserveAuthority(53L, 100L, 200L, ipPort);
        assertTrue(bridge.ledger().decode().get(53L).isPresent(),
                "权威预占恒入账——read 开关不再拦截");

        bridge.decodeReleaseAuthority(53L);
        assertTrue(bridge.ledger().decode().get(53L).isEmpty(), "权威释放照常移除条目");
        assertEquals(0, bridge.decodeEndpointCounters(ipPort.hashCode()).activeTotal(),
                "释放后 per-EP 计数归零");
    }

    /**
     * P 条目派发绑定挂点：dispatch 后条目绑定端点世代，引擎事件可推进相位；
     * 未绑定条目恒被世代屏障拒绝（对照证明 dispatch 挂点补齐了绑定缺口）。
     */
    @Test
    void prefillDispatchBindsEndpointGenerationForEngineEvents() {
        StateShadowBridge bridge = enabledBridge();
        String ipPort = "10.0.0.24:9000";

        // 对照：未 dispatch 的条目（UNBOUND）引擎事件被世代屏障拒绝
        bridge.onPrefillSubmit(55L);
        bridge.observeWorkerStatus(prefillRunningResponse(55L, 2L), RoleType.PREFILL, ipPort);
        assertEquals(2, bridge.ledger().prefill().get(55L).orElseThrow().phaseOrdinal(),
                "UNBOUND 条目引擎事件被世代屏障拒绝——相位停留在 QUEUED");

        // 主验证：dispatch 挂点绑定后引擎事件推进相位
        bridge.onPrefillSubmit(54L);
        bridge.onPrefillDispatched(54L, 900L, ipPort, 0L);
        assertEquals(4, bridge.ledger().prefill().get(54L).orElseThrow().phaseOrdinal(),
                "dispatch 挂点应推进到 DISPATCHED 并绑定世代");

        bridge.observeWorkerStatus(prefillRunningResponse(54L, 3L), RoleType.PREFILL, ipPort);
        assertTrue(bridge.ledger().prefill().get(54L).orElseThrow().phaseOrdinal() >= 5,
                "绑定后引擎事件应推进相位（P_RECEIVED 及以上）");
    }

    /** P 侧 per-EP 计数：仅已派发条目进入端点索引（排队窗口由 batcher 队列覆盖）。 */
    @Test
    void prefillEndpointCountersReflectDispatchedEntries() {
        StateShadowBridge bridge = readAuthorityBridge();
        String ipPort = "10.0.0.25:9000";
        bridge.onPrefillSubmit(56L);

        assertEquals(0, bridge.prefillEndpointCounters(ipPort.hashCode()).activeTotal(),
                "未派发条目不在端点索引（排队/攒批窗口由 batcher 覆盖）");

        bridge.onPrefillDispatched(56L, -1L, ipPort, 0L);

        assertEquals(1, bridge.prefillEndpointCounters(ipPort.hashCode()).activeTotal(),
                "派发后条目进入端点索引");
    }

    // ==================== 新侧终态恰好一次（P 先终局防双重记录）====================

    /**
     * 双重终态防重（真机轮根因回归）：P 引擎 finished 先终局时不得记录新侧终态
     * （P 完成 ≠ 请求完成，D 条目活跃时请求级终态等 D 终局）；D 终局记录恰好一次，
     * 与旧侧终态配对后窗口清空——否则第二次记录永久滞留窗口，高频终态下窗口
     * 满载后 diff 淘汰扫描退化为热路径灾难。
     */
    @Test
    void newSideTerminalRecordedExactlyOnceWhenPrefillFinishesBeforeDecode() {
        StateShadowBridge bridge = authorityBridge();
        String ipPortP = "10.0.0.31:9000";
        String ipPortD = "10.0.0.32:9000";

        // 开账：P submit+dispatch 绑定；D reserve（两阶段请求的全生命周期起点）
        bridge.onPrefillSubmit(60L);
        bridge.onPrefillDispatched(60L, -1L, ipPortP, 0L);
        bridge.onDecodeReserve(60L, 100L, 200L, RoleType.DECODE, ipPortD);

        // P 引擎 finished 先终局：P 墓碑落地，但 D 条目活跃——不得记录新侧终态
        bridge.observeWorkerStatus(prefillFinishedResponse(60L, 5L), RoleType.PREFILL, ipPortP);
        assertTrue(bridge.ledger().terminalOutcomeOf(60L, org.flexlb.state.spi.StateRole.PREFILL).isPresent(),
                "P 侧 finished 应终局 P 账");
        assertTrue(bridge.ledger().decode().get(60L).isPresent(), "D 条目仍活跃（decode 未完成）");
        assertEquals(0, bridge.diffCollector().pendingNew(),
                "P 先终局不得记录请求级终态（否则 D 终局时双重记录、第二次永久滞留窗口）");

        // D 引擎 finished 后终局：记录恰好一次
        bridge.observeWorkerStatus(finishedResponse(60L), RoleType.DECODE, ipPortD);
        assertEquals(1, bridge.diffCollector().pendingNew(), "D 终局记录恰好一次");

        // 旧侧终态到达 → 配对双清，窗口无残留
        bridge.onOldTerminal(60L, "COMPLETED");
        assertEquals(1L, bridge.diffCollector().matchedCount());
        assertEquals(0, bridge.diffCollector().pendingNew(), "配对后窗口清空——零滞留");
    }

    /**
     * P 兜底语义保留：D 侧从未开账（纯 P 阶段失败/取消）时，P 终局即请求级终态
     * ——防重守卫不得误伤此场景（否则该族请求的新侧终态永久缺失）。
     */
    @Test
    void prefillOnlyTerminalRecordedWhenDecodeSideNeverOpened() {
        StateShadowBridge bridge = enabledBridge();
        String ipPortP = "10.0.0.33:9000";

        bridge.onPrefillSubmit(61L);
        bridge.onPrefillDispatched(61L, -1L, ipPortP, 0L);

        bridge.observeWorkerStatus(prefillFinishedResponse(61L, 5L), RoleType.PREFILL, ipPortP);

        assertEquals(1, bridge.diffCollector().pendingNew(),
                "D 侧无条目时 P 终局即请求级终态（P 兜底仍生效）");

        bridge.onOldTerminal(61L, "COMPLETED");
        assertEquals(1L, bridge.diffCollector().matchedCount(), "P 兜底记录与旧侧终态配对");
        assertEquals(0, bridge.diffCollector().pendingNew());
    }

    private static StateShadowBridge readAuthorityBridge() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbStateV2ShadowEnabled(true);
        config.setFlexlbStateV2SettleEnabled(true);
        config.setFlexlbStateV2ReadEnabled(true);
        return StateShadowBridge.create(config, null, false);
    }

    /** P 侧 running 报文（单请求，phase 未报 → PENDING 保守倒推；完整明细）。 */
    private static WorkerStatusResponse prefillRunningResponse(long requestId, long statusVersion) {
        TaskInfo running = new TaskInfo();
        running.setRequestId(requestId);
        running.setBatchId(-1L);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setStatusVersion(statusVersion);
        response.setRole(RoleType.PREFILL);
        response.setRunningDetailCount(1L);
        response.setRunningTaskInfo(Map.of(String.valueOf(requestId), running));
        return response;
    }

    private static StateShadowBridge authorityBridge() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbStateV2ShadowEnabled(true);
        config.setFlexlbStateV2SettleEnabled(true);
        return StateShadowBridge.create(config, null, false);
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

    /** D 侧 running 报文（单请求，上报完整——引擎已见观察）。 */
    private static WorkerStatusResponse decodeRunningResponse(long requestId, long statusVersion) {
        TaskInfo running = new TaskInfo();
        running.setRequestId(requestId);
        running.setPhase(TaskPhase.RUNNING);
        running.setBatchId(-1L);
        running.setKvTokens(512L);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setStatusVersion(statusVersion);
        response.setRole(RoleType.DECODE);
        response.setRunningDetailCount(1L);
        response.setRunningTaskInfo(Map.of(String.valueOf(requestId), running));
        return response;
    }

    /** P 侧 finished(success) 报文（单请求，散请求 batchId=-1）。 */
    private static WorkerStatusResponse prefillFinishedResponse(long requestId, long statusVersion) {
        TaskInfo finished = new TaskInfo();
        finished.setRequestId(requestId);
        finished.setErrorCode(0L);
        finished.setEndTimeMs(1L);
        finished.setBatchId(-1L);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setStatusVersion(statusVersion);
        response.setRole(RoleType.PREFILL);
        response.setRunningDetailCount(0L);
        response.setFinishedTaskInfo(Map.of(String.valueOf(requestId), finished));
        return response;
    }

    private static boolean janitorThreadAlive() {
        return Thread.getAllStackTraces().keySet().stream()
                .anyMatch(t -> "flexlb-state-janitor".equals(t.getName()) && t.isAlive());
    }

    private static boolean waitFor(java.util.function.BooleanSupplier cond, long timeoutMs)
            throws InterruptedException {
        long deadline = System.currentTimeMillis() + timeoutMs;
        while (System.currentTimeMillis() < deadline) {
            if (cond.getAsBoolean()) {
                return true;
            }
            Thread.sleep(10L);
        }
        return cond.getAsBoolean();
    }
}
