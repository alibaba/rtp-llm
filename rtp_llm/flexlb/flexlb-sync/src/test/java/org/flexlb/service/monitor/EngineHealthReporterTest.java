package org.flexlb.service.monitor;

import io.netty.channel.EventLoopGroup;
import org.flexlb.cache.domain.CacheHitComparisonResult;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.constant.ZkMasterEvent;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.engine.grpc.client.EngineGrpcClient;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import reactor.netty.resources.LoopResources;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.mockito.ArgumentMatchers.doubleThat;
import static org.mockito.ArgumentMatchers.eq;

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
    void shouldReportCacheCapacityMetricsFromSharedWorkerStatus() {
        WorkerStatus workerStatus = workerStatusWithCacheStatus();
        workerStatus.updateKvCacheTokens(200, 800);

        reporter.reportStatusCheckerSuccess("test-model", workerStatus, 0, 0);

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

        reporter.reportStatusCheckerSuccess("test-model", workerStatus, 0, 0);

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
