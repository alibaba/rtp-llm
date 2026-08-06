package org.flexlb.service.monitor;

import io.netty.channel.EventLoopGroup;
import org.flexlb.cache.domain.CacheHitComparisonResult;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.constant.ZkMasterEvent;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.TaskStateEnum;
import org.flexlb.dao.route.LocalStandbyConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.client.EngineGrpcClient;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexStatisticsType;
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
    private final CacheMatchConfiguration cacheMatchConfiguration = mock(CacheMatchConfiguration.class);
    private final EngineGrpcClient engineGrpcClient = mock(EngineGrpcClient.class);
    private final LoopResources loopResources = mock(LoopResources.class);
    private final LocalStandbyConfig localStandbyConfig = new LocalStandbyConfig();

    private EngineHealthReporter reporter;

    @BeforeEach
    void setUp() {
        when(loopResources.onServer(true)).thenReturn(mock(EventLoopGroup.class));
        when(loopResources.onServerSelect(true)).thenReturn(mock(EventLoopGroup.class));
        when(engineGrpcClient.getEventLoopGroup()).thenReturn(mock(EventLoopGroup.class));
        when(cacheMatchConfiguration.isLocalStandbyEnabled()).thenReturn(true);
        when(cacheMatchConfiguration.getLocalStandbyConfig()).thenReturn(localStandbyConfig);
        reporter = new EngineHealthReporter(
                monitor, cacheMetricsReporter, cacheMatchConfiguration, engineGrpcClient, loopResources);
    }

    @Test
    void shouldRegisterCacheHitComparisonMetrics() {
        reporter.init();

        verify(monitor).register("app.engine.health.check.in.transit.task.size",
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).register("app.cache.hit.comparison.predicted.tokens", FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).register("app.cache.hit.comparison.actual.tokens", FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).register("app.cache.hit.comparison.delta.tokens", FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).register("app.cache.hit.comparison.kvcm.local.delta.tokens",
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).register("app.cache.hit.comparison.kvcm.p2p.total.match.delta.tokens",
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).register("app.cache.hit.comparison.kvcm.effective.delta.tokens",
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
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
        verify(monitor).register("app.cache.local.standby.block.size", FlexMetricType.GAUGE);
    }

    @Test
    void shouldRegisterMasterDecisionToWaitingConfirmationMetric() {
        reporter.init();

        verify(monitor).register("app.engine.worker.status.observed.decision.to.waiting.ms",
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
    }

    @Test
    void shouldRegisterRequestPayloadMetrics() {
        reporter.init();

        verify(monitor).register("app.request.input.ids.count",
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        verify(monitor).register("app.request.body.bytes",
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
    }

    @Test
    void shouldReportRequestPayloadMetricsWithoutResponse() {
        BalanceContext context = new BalanceContext();
        context.setSuccess(false);
        context.setInputIdsCount(512L);
        context.setRequestBodyBytes(5_242_881L);

        reporter.reportRequestPayload(context);

        FlexMetricTags expectedTags = FlexMetricTags.of("success", "false");
        verify(monitor).report("app.request.input.ids.count", expectedTags, 512.0);
        verify(monitor).report("app.request.body.bytes", expectedTags, 5_242_881.0);
    }

    @Test
    void shouldSkipUnknownRequestPayloadMetrics() {
        reporter.reportRequestPayload(new BalanceContext());

        verify(monitor, never()).report(eq("app.request.input.ids.count"), any(FlexMetricTags.class), anyDouble());
        verify(monitor, never()).report(eq("app.request.body.bytes"), any(FlexMetricTags.class), anyDouble());
    }

    @Test
    void shouldReportMasterDecisionToWaitingConfirmationLatency() {
        reporter.reportFlexlbObservedMasterDecisionToWaitingConfirmationLatency(
                "test-model", "10.0.0.1", "PREFILL", "test-group", 53);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "model", "test-model",
                "engineIp", "10.0.0.1",
                "role", "PREFILL",
                "group", "test-group");
        verify(monitor).report("app.engine.worker.status.observed.decision.to.waiting.ms",
                expectedTags, 53.0);
    }

    @Test
    void shouldRegisterWaitingToRunningMetric() {
        reporter.init();

        verify(monitor).register("app.engine.worker.status.observed.waiting.to.running.ms",
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
    }

    @Test
    void shouldReportWaitingToRunningLatency() {
        reporter.reportFlexlbObservedWaitingToRunningLatency(
                "test-model", "10.0.0.1", "PREFILL", "test-group", 42);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "model", "test-model",
                "engineIp", "10.0.0.1",
                "role", "PREFILL",
                "group", "test-group");
        verify(monitor).report("app.engine.worker.status.observed.waiting.to.running.ms",
                expectedTags, 42.0);
    }

    @Test
    void shouldRegisterEngineObservedWaitingToRunningMetric() {
        reporter.init();

        verify(monitor).register("app.engine.worker.status.engine.waiting.to.running.ms",
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
    }

    @Test
    void shouldReportEngineObservedWaitingToRunningLatency() {
        reporter.reportEngineObservedWaitingToRunningLatency(
                "test-model", "10.0.0.1", "PREFILL", "test-group", 42);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "model", "test-model",
                "engineIp", "10.0.0.1",
                "role", "PREFILL",
                "group", "test-group");
        verify(monitor).report("app.engine.worker.status.engine.waiting.to.running.ms",
                expectedTags, 42.0);
    }

    @Test
    void shouldRegisterEngineObservedReceivedToWaitingMetric() {
        reporter.init();

        verify(monitor).register("app.engine.worker.status.engine.received.to.waiting.ms",
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
    }

    @Test
    void shouldReportEngineObservedReceivedToWaitingLatency() {
        reporter.reportEngineObservedReceivedToWaitingLatency(
                "test-model", "10.0.0.1", "PREFILL", "test-group", 42);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "model", "test-model",
                "engineIp", "10.0.0.1",
                "role", "PREFILL",
                "group", "test-group");
        verify(monitor).report("app.engine.worker.status.engine.received.to.waiting.ms",
                expectedTags, 42.0);
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
    void shouldReportWorkerTaskCounts() {
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp("10.0.0.1");
        workerStatus.setRole("PREFILL");
        workerStatus.putLocalTask("in-transit", new TaskInfo());
        TaskInfo confirmedTask = new TaskInfo();
        workerStatus.putLocalTask("confirmed", confirmedTask);
        confirmedTask.updateTaskState(TaskStateEnum.CONFIRMED);

        reporter.reportStatusCheckerSuccess("test-model", workerStatus, 2, 3, 4);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "engineIp", "10.0.0.1",
                "role", "PREFILL");
        verify(monitor).report("app.engine.health.check.waiting.task.info.size", expectedTags, 2.0);
        verify(monitor).report("app.engine.health.check.running.task.info.size", expectedTags, 3.0);
        verify(monitor).report("app.engine.health.check.finished.task.list.size", expectedTags, 4.0);

        FlexMetricTags expectedLocalTaskTags = FlexMetricTags.of(
                "model", "test-model",
                "code", "0",
                "engineIp", "10.0.0.1",
                "role", "PREFILL");
        verify(monitor).report("app.engine.health.check.in.transit.task.size", expectedLocalTaskTags, 1.0);
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
        verify(monitor).report("app.cache.local.standby.block.size", expectedTags, 64.0);
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
        verify(monitor, never()).report(eq("app.cache.local.standby.block.size"),
                any(FlexMetricTags.class), anyDouble());
        verify(monitor, never()).report(eq("app.cache.used.kv.cache.tokens"), any(FlexMetricTags.class), anyDouble());
        verify(monitor, never()).report(eq("app.cache.available.kv.cache.tokens"), any(FlexMetricTags.class), anyDouble());
        verify(monitor, never()).report(eq("app.cache.total.kv.cache.tokens"), any(FlexMetricTags.class), anyDouble());
        verify(monitor, never()).report(eq("app.cache.used.kv.cache.ratio"), any(FlexMetricTags.class), anyDouble());
    }

    @Test
    void shouldReportConfiguredLocalStandbyBlockSize() {
        localStandbyConfig.setBlockSize(4096);
        WorkerStatus workerStatus = workerStatusWithCacheStatus();

        reporter.reportStatusCheckerSuccess("test-model", workerStatus, 0, 0, 0);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "model", "test-model",
                "engineIp", "10.0.0.1",
                "role", "PREFILL");
        verify(monitor).report("app.cache.local.standby.block.size", expectedTags, 4096.0);
    }

    @Test
    void shouldNotReportLocalStandbyBlockSizeWhenStandbyIsDisabled() {
        when(cacheMatchConfiguration.isLocalStandbyEnabled()).thenReturn(false);
        WorkerStatus workerStatus = workerStatusWithCacheStatus();

        reporter.reportStatusCheckerSuccess("test-model", workerStatus, 0, 0, 0);

        verify(monitor, never()).report(eq("app.cache.local.standby.block.size"),
                any(FlexMetricTags.class), anyDouble());
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
                "cache_hit_comparison", "request-1", "KVCM", "PREFILL", "test-group", "10.0.0.1",
                "running", 200,
                new CacheHitComparisonResult.Actual(120),
                new CacheHitComparisonResult.HitComparison(100, 20),
                new CacheHitComparisonResult.HitComparison(80, 40),
                null);

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
    void shouldReportSelectedKvcmP2pMatchDetails() {
        reporter.reportKvcmSelectedMatch(RoleType.PREFILL, "10.0.0.1", 40, 80, 100, 60, true);

        verify(cacheMetricsReporter).reportKvcmSelectedMatch(
                RoleType.PREFILL, "10.0.0.1", 40, 80, 100, 60);
    }

    @Test
    void shouldSkipSelectedKvcmP2pMetricsWhenDetailsAreUnavailable() {
        reporter.reportKvcmSelectedMatch(RoleType.PREFILL, "10.0.0.1", 0, 0, 0, 0, false);

        verify(cacheMetricsReporter, never()).reportKvcmSelectedMatch(
                org.mockito.ArgumentMatchers.any(),
                org.mockito.ArgumentMatchers.anyString(),
                org.mockito.ArgumentMatchers.anyLong(),
                org.mockito.ArgumentMatchers.anyLong(),
                org.mockito.ArgumentMatchers.anyLong(),
                org.mockito.ArgumentMatchers.anyLong());
    }

    @Test
    void shouldNotReportLocalStandbyMetricsWhenPredictionIsUnavailable() {
        CacheHitComparisonResult comparison = new CacheHitComparisonResult(
                "cache_hit_comparison", "request-1", "LOCAL_SYNC", "PREFILL", "test-group", "10.0.0.1",
                "running", 200,
                new CacheHitComparisonResult.Actual(120),
                new CacheHitComparisonResult.HitComparison(100, 20),
                null,
                null);

        reporter.reportCacheHitComparisonMetrics("test-model", comparison);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "model", "test-model",
                "engineIp", "10.0.0.1",
                "role", "PREFILL",
                "group", "test-group",
                "taskState", "running",
                "cacheMatchSource", "LOCAL_SYNC");
        verify(monitor).report("app.cache.hit.comparison.predicted.tokens", expectedTags, 100.0);
        verify(monitor).report("app.cache.hit.comparison.actual.tokens", expectedTags, 120.0);
        verify(monitor).report("app.cache.hit.comparison.delta.tokens", expectedTags, 20.0);
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
        verify(monitor, never()).report(
                org.mockito.ArgumentMatchers.eq("app.cache.hit.comparison.kvcm.local.delta.tokens"),
                org.mockito.ArgumentMatchers.any(FlexMetricTags.class),
                org.mockito.ArgumentMatchers.anyDouble());
        verify(monitor, never()).report(
                org.mockito.ArgumentMatchers.eq("app.cache.hit.comparison.kvcm.p2p.total.match.delta.tokens"),
                org.mockito.ArgumentMatchers.any(FlexMetricTags.class),
                org.mockito.ArgumentMatchers.anyDouble());
        verify(monitor, never()).report(
                org.mockito.ArgumentMatchers.eq("app.cache.hit.comparison.kvcm.effective.delta.tokens"),
                org.mockito.ArgumentMatchers.any(FlexMetricTags.class),
                org.mockito.ArgumentMatchers.anyDouble());
    }

    @Test
    void shouldReportKvcmLocalP2pAndEffectiveDeltasWhenAvailable() {
        CacheHitComparisonResult comparison = new CacheHitComparisonResult(
                "cache_hit_comparison", "request-1", "KVCM", "PREFILL", "test-group", "10.0.0.1",
                "running", 200,
                new CacheHitComparisonResult.Actual(120),
                new CacheHitComparisonResult.HitComparison(60, 60),
                null,
                new CacheHitComparisonResult.KvcmDetails(80, 20));

        reporter.reportCacheHitComparisonMetrics("test-model", comparison);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "model", "test-model",
                "engineIp", "10.0.0.1",
                "role", "PREFILL",
                "group", "test-group",
                "taskState", "running",
                "cacheMatchSource", "KVCM");
        verify(monitor).report("app.cache.hit.comparison.kvcm.local.delta.tokens", expectedTags, 80.0);
        verify(monitor).report("app.cache.hit.comparison.kvcm.p2p.total.match.delta.tokens", expectedTags, 20.0);
        verify(monitor).report("app.cache.hit.comparison.kvcm.effective.delta.tokens", expectedTags, 60.0);
    }

    @Test
    void shouldNotReportRatiosWithoutInputTokens() {
        CacheHitComparisonResult comparison = new CacheHitComparisonResult(
                "cache_hit_comparison", "request-1", "KVCM", "PREFILL", "test-group", "10.0.0.1",
                "running", 0,
                new CacheHitComparisonResult.Actual(120),
                new CacheHitComparisonResult.HitComparison(100, 20),
                new CacheHitComparisonResult.HitComparison(80, 40),
                null);

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
