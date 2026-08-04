package org.flexlb.service.monitor;

import io.micrometer.core.instrument.util.NamedThreadFactory;
import io.netty.channel.EventLoopGroup;
import org.flexlb.cache.domain.CacheHitComparisonResult;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.constant.ZkMasterEvent;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.engine.grpc.client.EngineGrpcClient;
import org.flexlb.engine.grpc.config.GrpcCallbackThreadPoolExecutor;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.sync.synchronizer.RejectionCountingThreadPoolExecutor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import reactor.netty.resources.LoopResources;

import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.doubleThat;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class EngineHealthReporterTest {

    private final FlexMonitor monitor = mock(FlexMonitor.class);
    private final CacheMetricsReporter cacheMetricsReporter = mock(CacheMetricsReporter.class);
    private final EngineGrpcClient engineGrpcClient = mock(EngineGrpcClient.class);
    private final LoopResources loopResources = mock(LoopResources.class);

    private EngineHealthReporter reporter;

    @BeforeEach
    void setUp() {
        when(loopResources.onServer(true)).thenReturn(mock(EventLoopGroup.class));
        when(loopResources.onServerSelect(true)).thenReturn(mock(EventLoopGroup.class));
        when(engineGrpcClient.getEventLoopGroup()).thenReturn(mock(EventLoopGroup.class));
        reporter = new EngineHealthReporter(monitor, cacheMetricsReporter, engineGrpcClient, loopResources);
    }

    @Test
    void shouldRegisterCacheHitComparisonMetrics() {
        reporter.init();

        verify(monitor).register("app.cache.hit.comparison.predicted.tokens", FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).register("app.cache.hit.comparison.actual.tokens", FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).register("app.cache.hit.comparison.delta.tokens", FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).register("app.cache.hit.comparison.local.standby.predicted.tokens",
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).register("app.cache.hit.comparison.local.standby.delta.tokens",
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).register("app.cache.hit.comparison.predicted.ratio",
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).register("app.cache.hit.comparison.actual.ratio",
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).register("app.cache.hit.comparison.local.standby.predicted.ratio",
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
    }

    @Test
    void shouldReportZkMasterEventTime() {
        long beforeReport = System.currentTimeMillis();

        reporter.reportPrefillBalanceMasterEvent(ZkMasterEvent.MASTER_TAKE_LEADERSHIP);

        long afterReport = System.currentTimeMillis();
        verify(monitor).report(
                eq("app.engine.zk.master.event"),
                eq(FlexMetricTags.of("event", ZkMasterEvent.MASTER_TAKE_LEADERSHIP.name())),
                doubleThat(value -> value >= beforeReport && value <= afterReport));
    }

    @Test
    void shouldReportGrpcExecutorCapacityAndRejectedTaskCount() throws InterruptedException {
        GrpcCallbackThreadPoolExecutor callbackExecutor = new GrpcCallbackThreadPoolExecutor(
                1, 1, 1, TimeUnit.MINUTES, new ArrayBlockingQueue<>(1),
                new NamedThreadFactory("grpc-callback-test"));
        CountDownLatch taskStarted = new CountDownLatch(1);
        CountDownLatch releaseTask = new CountDownLatch(1);
        try {
            callbackExecutor.execute(() -> {
                taskStarted.countDown();
                try {
                    releaseTask.await();
                } catch (InterruptedException interruptedException) {
                    Thread.currentThread().interrupt();
                }
            });
            assertTrue(taskStarted.await(1, TimeUnit.SECONDS));
            callbackExecutor.execute(() -> { });
            assertThrows(RejectedExecutionException.class, () -> callbackExecutor.execute(() -> { }));

            reporter.reportThreadPoolInfo("app.engine.balancing.thread.pool.info", "gRpcExecutor", callbackExecutor);

            verify(monitor).report(
                    "app.engine.balancing.thread.pool.info",
                    FlexMetricTags.of("threadPool", "gRpcExecutor", "type", "maximumPoolSize"),
                    1.0);
            verify(monitor).report(
                    "app.engine.balancing.thread.pool.info",
                    FlexMetricTags.of("threadPool", "gRpcExecutor", "type", "largestPoolSize"),
                    1.0);
            verify(monitor).report(
                    "app.engine.balancing.thread.pool.info",
                    FlexMetricTags.of("threadPool", "gRpcExecutor", "type", "rejectedTaskTotal"),
                    1.0);
        } finally {
            releaseTask.countDown();
            callbackExecutor.shutdownNow();
        }
    }

    @Test
    void shouldReportSynchronizationExecutorRejectedTaskCount() throws InterruptedException {
        RejectionCountingThreadPoolExecutor executor = new RejectionCountingThreadPoolExecutor(
                1, 1, 1, TimeUnit.MINUTES, new ArrayBlockingQueue<>(1),
                new NamedThreadFactory("engine-sync-test"), new ThreadPoolExecutor.AbortPolicy());
        CountDownLatch taskStarted = new CountDownLatch(1);
        CountDownLatch releaseTask = new CountDownLatch(1);
        try {
            executor.execute(() -> {
                taskStarted.countDown();
                try {
                    releaseTask.await();
                } catch (InterruptedException interruptedException) {
                    Thread.currentThread().interrupt();
                }
            });
            assertTrue(taskStarted.await(1, TimeUnit.SECONDS));
            executor.execute(() -> { });
            assertThrows(RejectedExecutionException.class, () -> executor.execute(() -> { }));

            reporter.reportThreadPoolInfo("app.engine.balancing.thread.pool.info", "engineSyncExecutor", executor);

            verify(monitor).report(
                    "app.engine.balancing.thread.pool.info",
                    FlexMetricTags.of("threadPool", "engineSyncExecutor", "type", "rejectedTaskTotal"),
                    1.0);
        } finally {
            releaseTask.countDown();
            executor.shutdownNow();
        }
    }

    @Test
    void shouldReportWorkerTaskCounts() {
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp("10.0.0.1");
        workerStatus.setRole("PREFILL");

        reporter.reportStatusCheckerSuccess("test-model", workerStatus, 2, 3, 4);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "engineIp", "10.0.0.1",
                "role", "PREFILL");
        verify(monitor).report("app.engine.health.check.waiting.task.info.size", expectedTags, 2.0);
        verify(monitor).report("app.engine.health.check.running.task.info.size", expectedTags, 3.0);
        verify(monitor).report("app.engine.health.check.finished.task.list.size", expectedTags, 4.0);
    }

    @Test
    void shouldReportCacheCapacityMetricsFromSharedWorkerStatus() {
        WorkerStatus workerStatus = workerStatusWithCacheStatus();
        workerStatus.updateKvCacheTokens(200, 800);

        reporter.reportStatusCheckerSuccess("test-model", workerStatus, 0, 0, 0);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "model", "test-model",
                "engineIp", "10.0.0.1",
                "role", "PREFILL");
        verify(monitor).report("app.cache.block.size", expectedTags, 64.0);
        verify(monitor).report("app.cache.used.kv.cache.tokens", expectedTags, 200.0);
        verify(monitor).report("app.cache.available.kv.cache.tokens", expectedTags, 800.0);
        verify(monitor).report("app.cache.total.kv.cache.tokens", expectedTags, 1000.0);
        verify(monitor).report("app.cache.used.kv.cache.ratio", expectedTags, 20.0);
        verify(monitor, never()).report(eq("app.cache.key.size"), any(FlexMetricTags.class), anyDouble());
    }

    @Test
    void shouldNotReportCacheCapacityMetricsWithoutCacheStatus() {
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp("10.0.0.1");
        workerStatus.setRole("PREFILL");

        reporter.reportStatusCheckerSuccess("test-model", workerStatus, 0, 0, 0);

        verify(monitor, never()).report(eq("app.cache.block.size"), any(FlexMetricTags.class), anyDouble());
        verify(monitor, never()).report(eq("app.cache.used.kv.cache.tokens"), any(FlexMetricTags.class), anyDouble());
        verify(monitor, never()).report(eq("app.cache.available.kv.cache.tokens"), any(FlexMetricTags.class), anyDouble());
        verify(monitor, never()).report(eq("app.cache.total.kv.cache.tokens"), any(FlexMetricTags.class), anyDouble());
        verify(monitor, never()).report(eq("app.cache.used.kv.cache.ratio"), any(FlexMetricTags.class), anyDouble());
    }

    @Test
    void shouldKeepCacheKeyMetricOnCacheStatusCheckerPath() {
        WorkerStatus workerStatus = workerStatusWithCacheStatus();

        reporter.reportCacheStatusCheckerSuccess("test-model", workerStatus);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "model", "test-model",
                "engineIp", "10.0.0.1",
                "role", "PREFILL");
        verify(monitor).report("app.cache.key.size", expectedTags, 7.0);
        verify(monitor, never()).report(eq("app.cache.block.size"), any(FlexMetricTags.class), anyDouble());
    }

    @Test
    void shouldReportCacheHitComparisonTokenMetricsWithStableDimensions() {
        CacheHitComparisonResult comparison = new CacheHitComparisonResult(
                "cache_hit_comparison", "request-1", "KVCM", "PREFILL", "test-group", "10.0.0.1", 8080,
                "running", 200, 64, 4096, 100, 80, true, 120, 20, 40);

        reporter.reportCacheHitComparisonMetrics("test-model", comparison);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "model", "test-model",
                "engineIp", "10.0.0.1",
                "role", "PREFILL",
                "group", "test-group",
                "taskState", "running",
                "cacheMatchSource", "KVCM");
        verify(monitor).report("app.cache.hit.comparison.predicted.tokens", expectedTags, 100.0);
        verify(monitor).report("app.cache.hit.comparison.actual.tokens", expectedTags, 120.0);
        verify(monitor).report("app.cache.hit.comparison.delta.tokens", expectedTags, 20.0);
        verify(monitor).report("app.cache.hit.comparison.local.standby.predicted.tokens", expectedTags, 80.0);
        verify(monitor).report("app.cache.hit.comparison.local.standby.delta.tokens", expectedTags, 40.0);
        verify(monitor).report("app.cache.hit.comparison.predicted.ratio", expectedTags, 0.5);
        verify(monitor).report("app.cache.hit.comparison.actual.ratio", expectedTags, 0.6);
        verify(monitor).report("app.cache.hit.comparison.local.standby.predicted.ratio", expectedTags, 0.4);
        assertEquals(Map.of(
                "model", "test-model",
                "engineIp", "10.0.0.1",
                "role", "PREFILL",
                "group", "test-group",
                "taskState", "running",
                "cacheMatchSource", "KVCM"), expectedTags.getTags());
    }

    @Test
    void shouldNotReportLocalStandbyMetricsWhenPredictionIsUnavailable() {
        CacheHitComparisonResult comparison = new CacheHitComparisonResult(
                "cache_hit_comparison", "request-1", "LOCAL_SYNC", "PREFILL", "test-group", "10.0.0.1", 8080,
                "running", 200, 64, 0, 100, 0, false, 120, 20, 0);

        reporter.reportCacheHitComparisonMetrics("test-model", comparison);

        verify(monitor, never()).report(
                org.mockito.ArgumentMatchers.eq("app.cache.hit.comparison.local.standby.predicted.tokens"),
                org.mockito.ArgumentMatchers.any(FlexMetricTags.class),
                org.mockito.ArgumentMatchers.anyDouble());
        verify(monitor, never()).report(
                org.mockito.ArgumentMatchers.eq("app.cache.hit.comparison.local.standby.delta.tokens"),
                org.mockito.ArgumentMatchers.any(FlexMetricTags.class),
                org.mockito.ArgumentMatchers.anyDouble());
        verify(monitor, never()).report(
                org.mockito.ArgumentMatchers.eq("app.cache.hit.comparison.local.standby.predicted.ratio"),
                org.mockito.ArgumentMatchers.any(FlexMetricTags.class),
                org.mockito.ArgumentMatchers.anyDouble());
    }

    @Test
    void shouldNotReportRatiosWithoutInputTokens() {
        CacheHitComparisonResult comparison = new CacheHitComparisonResult(
                "cache_hit_comparison", "request-1", "KVCM", "PREFILL", "test-group", "10.0.0.1", 8080,
                "running", 0, 64, 4096, 100, 80, true, 120, 20, 40);

        reporter.reportCacheHitComparisonMetrics("test-model", comparison);

        verify(monitor, never()).report(
                org.mockito.ArgumentMatchers.eq("app.cache.hit.comparison.predicted.ratio"),
                org.mockito.ArgumentMatchers.any(FlexMetricTags.class),
                org.mockito.ArgumentMatchers.anyDouble());
        verify(monitor, never()).report(
                org.mockito.ArgumentMatchers.eq("app.cache.hit.comparison.actual.ratio"),
                org.mockito.ArgumentMatchers.any(FlexMetricTags.class),
                org.mockito.ArgumentMatchers.anyDouble());
        verify(monitor, never()).report(
                org.mockito.ArgumentMatchers.eq("app.cache.hit.comparison.local.standby.predicted.ratio"),
                org.mockito.ArgumentMatchers.any(FlexMetricTags.class),
                org.mockito.ArgumentMatchers.anyDouble());
    }

    private WorkerStatus workerStatusWithCacheStatus() {
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp("10.0.0.1");
        workerStatus.setRole("PREFILL");
        workerStatus.setCacheStatus(CacheStatus.builder()
                .blockSize(64)
                .cacheKeySize(7)
                .build());
        return workerStatus;
    }
}
