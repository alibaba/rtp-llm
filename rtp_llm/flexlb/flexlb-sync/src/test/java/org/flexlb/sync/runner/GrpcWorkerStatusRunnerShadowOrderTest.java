package org.flexlb.sync.runner;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.state.spi.StateRole;
import org.flexlb.sync.shadow.StateShadowBridge;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.when;

/**
 * G1 影子挂载点顺序测试：影子消费（StateShadowBridge.observeWorkerStatus）必须
 * 发生在 latestFinishedVersion 水位推进之前（相位事件顺序与旧路径一致）。
 *
 * <p>断言手段：水位 AtomicLong 捕获桩——{@code set(newValue)} 被调用的时刻检查
 * 影子账本墓碑是否已可见（影子消费完成的可观察效果）。若挂载顺序错误
 * （水位先推、影子后消费），捕获点检查失败。</p>
 */
class GrpcWorkerStatusRunnerShadowOrderTest {

    private static final long REQUEST_ID = 42L;

    @Test
    void shouldObserveShadow_beforeLatestFinishedVersionAdvances() {
        StateShadowBridge bridge = enabledBridge();

        // 开账前置：D 侧本地 reserve（正常 observe 不收养未开条目；与 runner 同 ipPort
        // → 同一端点世代，后续 finished 事件 binding 匹配）
        bridge.onDecodeReserve(REQUEST_ID, 100L, 200L, RoleType.DECODE, "127.0.0.1:8080");

        // 真实引擎报文：DECODE 侧 v200，携带 finished(42, success)，水位 200
        EngineRpcService.TaskInfoPB finishedTask = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(REQUEST_ID)
                .setEndTimeMs(123456L)
                .build();
        EngineRpcService.WorkerStatusPB pb = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                .setStatusVersion(200L)
                .setLatestFinishedVersion(200L)
                .setRunningDetailCount(0)
                .setAlive(true)
                .addFinishedTaskList(finishedTask)
                .build();

        EngineGrpcService engineGrpcService = Mockito.mock(EngineGrpcService.class);
        when(engineGrpcService.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(),
                Mockito.any(RoleType.class))).thenReturn(CompletableFuture.completedFuture(pb));

        // 水位 getter 调用计数拦截（AtomicLong.set 为 final 无法 override，但
        // WorkerStatus.getLatestFinishedTaskVersion() 可 stub）：
        // 调用序列 = #1 run() 发请求带水位 → #2 水位比较 → #3 水位 set 前的 getter。
        // 挂载正确（影子在 step 3.5、水位推进在 step 4）⇒ #2/#3 时刻影子墓碑已可见。
        AtomicLong realWatermark = new AtomicLong(100L);
        AtomicInteger watermarkGetterCalls = new AtomicInteger();
        AtomicBoolean shadowVisibleAtWatermarkAdvance = new AtomicBoolean(true);
        AtomicBoolean watermarkAdvanced = new AtomicBoolean(false);
        AtomicBoolean statusCheckInProgress = new AtomicBoolean(false);

        WorkerStatus workerStatus = Mockito.mock(WorkerStatus.class);
        when(workerStatus.getStatusVersion()).thenReturn(new AtomicLong(100L));
        when(workerStatus.getLatestFinishedTaskVersion()).thenAnswer(inv -> {
            int n = watermarkGetterCalls.incrementAndGet();
            if (n >= 2) {
                // 水位推进区（比较/set）：此刻影子消费必须已完成（墓碑可见）
                shadowVisibleAtWatermarkAdvance.set(
                        bridge.ledger().terminalOutcomeOf(REQUEST_ID, StateRole.DECODE).isPresent());
                watermarkAdvanced.set(true);
            }
            return realWatermark;
        });
        when(workerStatus.getStatusCheckInProgress()).thenReturn(statusCheckInProgress);
        when(workerStatus.getTotalKvCacheTokens()).thenReturn(new AtomicLong(0L));
        when(workerStatus.getAvailableKvCacheTokens()).thenReturn(new AtomicLong(0L));

        EngineHealthReporter engineHealthReporter = Mockito.mock(EngineHealthReporter.class);

        GrpcWorkerStatusRunner runner = new GrpcWorkerStatusRunner(
                "test-model", "127.0.0.1:8080", "site", RoleType.DECODE, "group",
                workerStatus, Map.of("127.0.0.1:8080", workerStatus),
                engineHealthReporter, engineGrpcService, 20L,
                null, Runnable::run, bridge);
        runner.run();

        // 水位推进发生（versionAdvanced: 100 < 200）
        assertTrue(watermarkAdvanced.get(), "latestFinishedVersion 应从 100 推进到 200");
        assertEquals(200L, realWatermark.get());
        assertTrue(watermarkGetterCalls.get() >= 3, "水位 getter 应被调用 ≥3 次（发请求/比较/set）");
        // 核心顺序断言：水位推进时刻，影子已消费 finished（墓碑可见）
        assertTrue(shadowVisibleAtWatermarkAdvance.get(),
                "影子消费必须发生在水位推进之前（挂载点 step 3 < step 4）");
        // 影子链路自身健康：泵入 1 次、零异常
        assertEquals(1L, bridge.diffCollector().eventCount());
        assertEquals(0L, bridge.diffCollector().errorCount());
        // 旧路径行为不变：水位值最终为 200
        assertEquals(200L, workerStatus.getLatestFinishedTaskVersion().get());
    }

    @Test
    void shouldSkipShadow_whenVersionNotAdvanced() {
        StateShadowBridge bridge = enabledBridge();

        EngineRpcService.WorkerStatusPB pb = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                .setStatusVersion(100L) // == 本地版本 → 不推进
                .setLatestFinishedVersion(100L)
                .setAlive(true)
                .build();

        EngineGrpcService engineGrpcService = Mockito.mock(EngineGrpcService.class);
        when(engineGrpcService.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(),
                Mockito.any(RoleType.class))).thenReturn(CompletableFuture.completedFuture(pb));

        AtomicLong watermark = new AtomicLong(100L);
        AtomicBoolean statusCheckInProgress = new AtomicBoolean(false);
        WorkerStatus workerStatus = Mockito.mock(WorkerStatus.class);
        when(workerStatus.getStatusVersion()).thenReturn(new AtomicLong(100L));
        when(workerStatus.getLatestFinishedTaskVersion()).thenReturn(watermark);
        when(workerStatus.getStatusCheckInProgress()).thenReturn(statusCheckInProgress);
        when(workerStatus.getTotalKvCacheTokens()).thenReturn(new AtomicLong(0L));
        when(workerStatus.getAvailableKvCacheTokens()).thenReturn(new AtomicLong(0L));
        when(workerStatus.isAlive()).thenReturn(true);

        EngineHealthReporter engineHealthReporter = Mockito.mock(EngineHealthReporter.class);

        GrpcWorkerStatusRunner runner = new GrpcWorkerStatusRunner(
                "test-model", "127.0.0.1:8080", "site", RoleType.DECODE, "group",
                workerStatus, Map.of("127.0.0.1:8080", workerStatus),
                engineHealthReporter, engineGrpcService, 20L,
                null, Runnable::run, bridge);
        runner.run();

        // versionNotAdvanced 分支不触发影子消费（影子与旧路径同一 versionAdvanced 门）
        assertEquals(0L, bridge.diffCollector().eventCount());
        assertEquals(100L, watermark.get(), "版本未推进 → 水位不动");
    }

    private static StateShadowBridge enabledBridge() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbStateV2ShadowEnabled(true);
        return StateShadowBridge.create(config, null, false);
    }
}
