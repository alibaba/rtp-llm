package org.flexlb.sync.shadow;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * flexlbStateV2DebugTransitionLog 开关矩阵（默认关——日志量红线）：
 * 日志器级别统一放行 DEBUG（隔离级别变量），开关是唯一变量。
 * 关 = 相位转换零 debug 日志；开 = 每次转换一行
 * requestId/from/to/version/reason 完备字段。
 */
class StateShadowBridgeDebugTransitionLogTest {

    private Logger syncLogger;
    private Level previousLevel;
    private ListAppender<ILoggingEvent> appender;

    @BeforeEach
    void attachAppender() {
        syncLogger = (Logger) LoggerFactory.getLogger("syncLogger");
        previousLevel = syncLogger.getLevel();
        appender = new ListAppender<>();
        appender.start();
        syncLogger.addAppender(appender);
        syncLogger.setLevel(Level.DEBUG);
    }

    @AfterEach
    void detachAppender() {
        syncLogger.detachAppender(appender);
        syncLogger.setLevel(previousLevel);
    }

    /** 默认关：驱动开账与引擎观察转换，不得产生任何 [state-ledger] transition 日志。 */
    @Test
    void noTransitionLogsWhenSwitchIsDefaultOff() {
        StateShadowBridge bridge = bridgeWithDebugFlag(false);
        driveTransitions(bridge, 1L, "10.0.0.1:9000");

        assertTrue(transitionLogs().isEmpty(),
                "默认关：相位转换不得产生 debug 日志（日志量红线）");
    }

    /** 开：同一驱动必须产生转换日志，且字段完备（requestId/from/to/version/reason）。 */
    @Test
    void transitionLogsEmittedWithCompleteFieldsWhenSwitchOn() {
        StateShadowBridge bridge = bridgeWithDebugFlag(true);
        driveTransitions(bridge, 2L, "10.0.0.2:9000");

        List<String> logs = transitionLogs();
        assertFalse(logs.isEmpty(), "开关开：相位转换必须打 debug 日志");
        // submit 路径的第一条转换：INIT → QUEUED（register 创建后入队）
        String first = logs.get(0);
        assertTrue(first.contains("[state-ledger] transition"), "日志行格式契约");
        assertTrue(first.contains("requestId=2"), "字段契约：requestId");
        assertTrue(first.contains("from=INIT"), "字段契约：from（初始相位）");
        assertTrue(first.contains("to=QUEUED"), "字段契约：to（入队相位）");
        assertTrue(first.contains("version="), "字段契约：version");
        assertTrue(first.contains("reason="), "字段契约：reason");
    }

    /** 驱动一次完整转换链：P 开账入队（INIT→QUEUED）+ D 开账 + 引擎 RUNNING 观察（D 侧格上升）。 */
    private static void driveTransitions(StateShadowBridge bridge, long requestId, String ipPort) {
        bridge.onPrefillSubmit(requestId);
        bridge.onDecodeReserve(requestId, 100L, 200L, RoleType.DECODE, ipPort);
        bridge.observeWorkerStatus(decodeRunningResponse(requestId, 1L), RoleType.DECODE, ipPort);
    }

    private List<String> transitionLogs() {
        return appender.list.stream()
                .map(ILoggingEvent::getFormattedMessage)
                .filter(message -> message.contains("[state-ledger] transition"))
                .toList();
    }

    private static StateShadowBridge bridgeWithDebugFlag(boolean debugTransitionLog) {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbStateV2ShadowEnabled(true);
        config.setFlexlbStateV2DebugTransitionLog(debugTransitionLog);
        // autoStartJanitor=false：不建调度线程（日志断言不被异步 tick 干扰）
        return StateShadowBridge.create(config, null, false);
    }

    /** D 侧 RUNNING 明细报文（单请求，statusVersion=1）。 */
    private static WorkerStatusResponse decodeRunningResponse(long requestId, long statusVersion) {
        TaskInfo running = new TaskInfo();
        running.setRequestId(requestId);
        running.setBatchId(-1L);
        running.setPhase(org.flexlb.enums.TaskPhase.RUNNING);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setStatusVersion(statusVersion);
        response.setRole(RoleType.DECODE);
        response.setRunningDetailCount(1L);
        response.setRunningTaskInfo(Map.of(String.valueOf(requestId), running));
        return response;
    }
}
