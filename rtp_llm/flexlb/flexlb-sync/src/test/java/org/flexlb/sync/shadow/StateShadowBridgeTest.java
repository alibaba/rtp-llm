package org.flexlb.sync.shadow;

import org.flexlb.balance.scheduler.TerminalReason;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.constant.MetricConstant;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
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
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

/**
 * G1 影子开关矩阵：关 = 装配返回 DISABLED 单例、所有入口零执行（ledger 不存在）；
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
        assertNotNull(bridge.janitor(), "M4：shadow 开时 janitor 必须挂载");
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
        // M4：autoStartJanitor=false——不创建调度线程（确定性 + 线程卫生）
        return StateShadowBridge.create(config, null, false);
    }

    // ==================== M4 清理层装配（janitor 挂载/调度/配置传播）====================

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

    /** 清理层参数从 FlexlbConfig 传播到 LedgerJanitorConfig（与 M3 影子开关同装配模式）。 */
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

    // ==================== G3：结算换权开关矩阵（onOldTerminalAuthority 权威单出口）====================

    /** G3 开 ⇒ shadow 开是硬前置：settle 开而 shadow 关必须拒启（fail-fast）。 */
    @Test
    void failFastWhenSettleAuthorityWithoutShadow() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbStateV2ShadowEnabled(false);
        config.setFlexlbStateV2SettleEnabled(true);

        assertThrows(IllegalStateException.class,
                () -> StateShadowBridge.create(config, null, false),
                "结算换权依赖影子链路在跑——settle 开而 shadow 关必须 fail-fast 拒启");
    }

    /** 开关矩阵：settle 开（shadow 开）= 权威；settle 关 / DISABLED = 旧影子语义。 */
    @Test
    void settleAuthorityMatrix() {
        FlexlbConfig settleOff = new FlexlbConfig();
        settleOff.setFlexlbStateV2ShadowEnabled(true);
        assertFalse(StateShadowBridge.create(settleOff, null, false).isSettleAuthority(),
                "settle 关（shadow 开）不得进入权威结算");
        assertFalse(StateShadowBridge.DISABLED.isSettleAuthority(), "DISABLED 恒非权威");
        assertTrue(authorityBridge().isSettleAuthority(), "settle 开 + shadow 开 = 权威结算");
    }

    /** G3 关时权威入口零执行（旧路径 onOldTerminal 语义不变）。 */
    @Test
    void authorityEntryPointNoOpWhenSettleDisabled() {
        StateShadowBridge bridge = enabledBridge(); // shadow 开、settle 关
        bridge.onPrefillSubmit(31L);

        bridge.onOldTerminalAuthority(31L, "COMPLETED",
                new StateShadowBridge.TerminalMetricContext(TerminalReason.COMPLETED, "PREFILL", "10.0.0.1"));

        assertEquals(0, bridge.pendingTerminalMetricCount(), "settle 关时权威入口必须零执行");
        assertTrue(bridge.ledger().prefill().get(31L).isPresent(), "settle 关时不得提前结算 ledger 条目");
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

    /** TIMED_OUT：双侧主动 settle 为 SLO_TIMEOUT（存活时间上限）。 */
    @Test
    void timedOutSettlesBothSidesImmediately() {
        StateShadowBridge bridge = authorityBridge();
        bridge.onPrefillSubmit(37L);

        bridge.onOldTerminalAuthority(37L, "TIMED_OUT",
                new StateShadowBridge.TerminalMetricContext(TerminalReason.TIMED_OUT, "PREFILL", "unknown"));

        assertEquals(TerminalState.SLO_TIMEOUT,
                bridge.ledger().terminalOutcomeOf(37L, org.flexlb.state.spi.StateRole.PREFILL)
                        .orElseThrow().state());
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

    // ==================== G4：读取换权开关矩阵（调度读点与 EP 记账切门面）====================

    /** G4 开 ⇒ settle 开是硬前置：read 开而 settle 关必须拒启（fail-fast）。 */
    @Test
    void failFastWhenReadAuthorityWithoutSettle() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbStateV2ShadowEnabled(true);
        config.setFlexlbStateV2ReadEnabled(true);

        assertThrows(IllegalStateException.class,
                () -> StateShadowBridge.create(config, null, false),
                "读取换权依赖结算单出口——read 开而 settle 关必须 fail-fast 拒启");
    }

    /** 开关矩阵：read 开（shadow+settle 开）= 读权威；read 关 / settle 关 / DISABLED = 旧读点。 */
    @Test
    void readAuthorityMatrix() {
        FlexlbConfig readOff = new FlexlbConfig();
        readOff.setFlexlbStateV2ShadowEnabled(true);
        readOff.setFlexlbStateV2SettleEnabled(true);
        assertFalse(StateShadowBridge.create(readOff, null, false).isReadAuthority(),
                "read 关（settle 开）不得进入读权威");
        assertFalse(authorityBridge().isReadAuthority(), "settle 权威桥未开 read 时非读权威");
        assertFalse(StateShadowBridge.DISABLED.isReadAuthority(), "DISABLED 恒非读权威");
        assertTrue(readAuthorityBridge().isReadAuthority(), "三开 = 读权威");
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

    /** read 关时记账/计数权威入口零执行（读点走旧双层 map，返回全零视图）。 */
    @Test
    void readAuthorityEntryPointsNoOpWhenReadDisabled() {
        StateShadowBridge bridge = authorityBridge(); // shadow+settle 开、read 关
        String ipPort = "10.0.0.23:9000";

        bridge.decodeReserveAuthority(53L, 100L, 200L, ipPort);
        bridge.decodeReleaseAuthority(53L);

        assertTrue(bridge.ledger().decode().get(53L).isEmpty(), "read 关时权威预占零执行");
        assertEquals(0, bridge.decodeEndpointCounters(ipPort.hashCode()).activeTotal(),
                "read 关时计数恒全零视图");
        assertEquals(0, bridge.prefillEndpointCounters(ipPort.hashCode()).activeTotal(),
                "read 关时 P 侧计数恒全零视图");
    }

    /**
     * P 条目派发绑定挂点：dispatch 后条目绑定端点世代，引擎事件可推进相位；
     * 未绑定条目恒被世代屏障拒绝（对照证明挂点补齐了 M3 遗留的绑定缺口）。
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
        bridge.onPrefillDispatched(54L, 900L, ipPort);
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

        bridge.onPrefillDispatched(56L, -1L, ipPort);

        assertEquals(1, bridge.prefillEndpointCounters(ipPort.hashCode()).activeTotal(),
                "派发后条目进入端点索引");
    }

    // ==================== G4 回归：新侧终态恰好一次（P 先终局防双重记录）====================

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
        bridge.onPrefillDispatched(60L, -1L, ipPortP);
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
        bridge.onPrefillDispatched(61L, -1L, ipPortP);

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
