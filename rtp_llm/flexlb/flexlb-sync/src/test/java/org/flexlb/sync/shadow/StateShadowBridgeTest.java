package org.flexlb.sync.shadow;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.state.TerminalState;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

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

        StateShadowBridge bridge = StateShadowBridge.create(config, null);

        assertTrue(bridge.isEnabled());
        assertNotNull(bridge.ledger());
        assertNotNull(bridge.diffCollector());
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
        return StateShadowBridge.create(config, null);
    }
}
