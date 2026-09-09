package org.flexlb.service.monitor;

import io.netty.channel.EventLoopGroup;
import org.flexlb.cache.domain.CacheHitComparisonResult;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.LocalStandbyConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.constant.ZkMasterEvent;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerIdentity;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.FlexStatisticsType;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import reactor.netty.resources.LoopResources;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
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
    private final CacheMatchConfiguration cacheMatchConfiguration = mock(CacheMatchConfiguration.class);
    private final EngineGrpcClient engineGrpcClient = mock(EngineGrpcClient.class);
    private final LoopResources loopResources = mock(LoopResources.class);
    private final WorkerDirectory workerDirectory = mock(WorkerDirectory.class);
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
                monitor, cacheMetricsReporter, cacheMatchConfiguration,
                engineGrpcClient, loopResources, workerDirectory);
    }

    @Test
    void reportsStepSamplesWithEngineIdentityAndPhase() {
        reporter.init();
        WorkerStatus worker = WorkerStatus.createDiscovered(
                RoleType.PDFUSION, "test-group", "10.0.0.1", 8080, 8081, null, null, 1, 2);
        for (int prefillRequests : new int[]{2, 0}) {
            var step = new WorkerStatus.StepMetrics(42, 1700000000000L, 16000,
                    prefillRequests, prefillRequests > 0 ? 15000 : 0, 32000, 0.5);
            reporter.reportWorkerStepMetrics("test-model", worker, step);
            var tags = FlexMetricTags.of("model", "test-model", "engineIp", "10.0.0.1:8080@1",
                    "role", "PDFUSION", "group", "test-group", "phase", prefillRequests > 0 ? "prefill" : "decode");
            verify(monitor).report("app.engine.worker.step.total.scheduled.tokens", tags, 16000.0);
            verify(monitor).report("app.engine.worker.step.prefill.request.count", tags, (double) prefillRequests);
            verify(monitor).report("app.engine.worker.step.prefill.tokens", tags, prefillRequests > 0 ? 15000.0 : 0.0);
            verify(monitor).report("app.engine.worker.step.token.budget", tags, 32000.0);
            verify(monitor).report("app.engine.worker.step.budget.fill.ratio", tags, 0.5);
        }
        verify(monitor).register("app.engine.worker.step.budget.fill.ratio",
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
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
    void shouldRegisterStatusCheckFailureMetrics() {
        reporter.init();

        verify(monitor).register("app.engine.health.check.fail.total",
                FlexMetricType.COUNTER, FlexPriorityType.PRECISE);
        verify(monitor).register("app.engine.health.check.fail.rt",
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
    }

    @Test
    void shouldReportStatusCheckFailureCountAndLatencyWithSameTags() {
        BalanceStatusEnum failure = BalanceStatusEnum.WORKER_STATUS_GRPC_TIMEOUT;
        FlexMetricTags expectedTags = FlexMetricTags.of(
                "model", "test-model",
                "code", String.valueOf(failure.getCode()),
                "engineIp", "10.0.0.1:8080@0",
                "role", RoleType.PREFILL.getCode());

        reporter.reportStatusCheckerFail("test-model", failure, "10.0.0.1:8080@0", RoleType.PREFILL);
        reporter.reportStatusCheckFailureLatency(
                "test-model", failure, "10.0.0.1:8080@0", RoleType.PREFILL, 201_234);

        verify(monitor).report("app.engine.health.check.fail", expectedTags, 1.0);
        verify(monitor).report("app.engine.health.check.fail.total", expectedTags, 1.0);
        verify(monitor).report("app.engine.health.check.fail.rt", expectedTags, 201_234.0);
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
        verify(monitor).register("app.request.message.bytes",
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        verify(monitor).register("app.request.body.bytes",
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
    }

    @Test
    void shouldReportRequestPayloadMetricsWithoutResponse() {
        BalanceContext context = new BalanceContext();
        context.setSuccess(false);
        context.setInputIdsCount(512L);
        context.setRequestMessageBytes(8192L);
        context.setRequestBodyBytes(5_242_881L);

        reporter.reportRequestPayload(context);

        FlexMetricTags expectedTags = FlexMetricTags.of("success", "false");
        verify(monitor).report("app.request.input.ids.count", expectedTags, 512.0);
        verify(monitor).report("app.request.body.bytes", expectedTags, 5_242_881.0);
        verify(monitor).report("app.request.message.bytes", expectedTags, 8192.0);
    }

    @Test
    void shouldReportSelectedEngineWithConfiguredStrategy() {
        ServerStatus serverStatus = new ServerStatus();
        serverStatus.setRole(RoleType.PREFILL);
        serverStatus.setServerIp("10.0.0.1");
        serverStatus.setHttpPort(8080);
        Response response = new Response();
        response.setSuccess(true);
        response.setCode(200);
        response.setServerStatus(List.of(serverStatus));
        FlexlbConfig config = new FlexlbConfig();
        config.getRouter().getRoles().getPrefill().getCandidateChoice()
                .setType(RoutingConfig.CandidateChoiceType.LEAST_RECENTLY_USED_IN_POOL);
        BalanceContext context = new BalanceContext();
        context.setConfig(config);
        context.setResponse(response);

        reporter.reportBalancingService(context);

        verify(monitor).report("app.engine.balancing.master.select.detail", FlexMetricTags.of(
                "role", "PREFILL",
                "strategy", "LEAST_RECENTLY_USED_IN_POOL",
                "engineIp", "10.0.0.1:8080",
                "success", "true",
                "code", "200"), 1.0);
    }

    @Test
    void shouldReportCacheAffinityDecisionBySelectedEngine() {
        reporter.reportCacheAffinityDecision(RoleType.PREFILL, "10.0.0.1:8080@0", "CACHE_LEADER");

        verify(cacheMetricsReporter).reportCacheAffinityDecision(
                RoleType.PREFILL, "10.0.0.1:8080@0", "CACHE_LEADER");
    }

    @Test
    void shouldSkipUnknownRequestPayloadMetrics() {
        reporter.reportRequestPayload(new BalanceContext());

        verify(monitor, never()).report(eq("app.request.input.ids.count"), any(FlexMetricTags.class), anyDouble());
        verify(monitor, never()).report(eq("app.request.message.bytes"), any(FlexMetricTags.class), anyDouble());
        verify(monitor, never()).report(eq("app.request.body.bytes"), any(FlexMetricTags.class), anyDouble());
    }

    @Test
    void shouldReportMasterDecisionToWaitingConfirmationLatency() {
        reporter.reportFlexlbObservedMasterDecisionToWaitingConfirmationLatency(
                "test-model", "10.0.0.1:8080@0", "PREFILL", "test-group", 53);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "model", "test-model",
                "engineIp", "10.0.0.1:8080@0",
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
                "test-model", "10.0.0.1:8080@0", "PREFILL", "test-group", 42);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "model", "test-model",
                "engineIp", "10.0.0.1:8080@0",
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
                "test-model", "10.0.0.1:8080@0", "PREFILL", "test-group", 42);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "model", "test-model",
                "engineIp", "10.0.0.1:8080@0",
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
                "test-model", "10.0.0.1:8080@0", "PREFILL", "test-group", 42);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "model", "test-model",
                "engineIp", "10.0.0.1:8080@0",
                "role", "PREFILL",
                "group", "test-group");
        verify(monitor).report("app.engine.worker.status.engine.received.to.waiting.ms",
                expectedTags, 42.0);
    }

    @Test
    void reportsCommittedFinishedTaskTelemetryForFusionWorker() {
        WorkerStatus worker = WorkerStatus.createDiscovered(
                RoleType.PDFUSION, "test-group", "10.0.0.1", 8080, 8081, null, null, 0, 1);
        var telemetry = new WorkerStatus.TaskTelemetry(true, 900, 1000, 1100,
                1200, 1600, 200, 1900, 512, 256, 1, 3, 3, 128, 256);
        var task = new WorkerStatus.TaskObservation("request", 768, 300, 1000, 400,
                1, 2000, 0, 0, null, 0, null, 300, null, telemetry);
        reporter.reportFinishedWorkerTask("test-model", worker, task);
        FlexMetricTags tags = FlexMetricTags.of("model", "test-model", "engineIp", "10.0.0.1:8080",
                "role", "PDFUSION", "group", "test-group");
        verify(monitor).report("app.engine.worker.status.input.queue.wait.ms", tags, 100.0);
        verify(monitor).report("app.engine.worker.status.running.to.first.token.ms", tags, 300.0);
        verify(monitor).report("app.engine.worker.status.engine.waiting.to.running.ms", tags, 400.0);
        verify(monitor).report("app.engine.worker.status.engine.received.to.waiting.ms", tags, 300.0);
    }

    @Test
    void shouldReportPrefillWorkerStatusTaskMetrics() {
        TaskInfo task = new TaskInfo();
        task.setInputQueueEnqueueTimeMs(1000);
        task.setInputQueueDrainTimeMs(1100);
        task.setWaitingEnteredTimeMs(1200);
        task.setRunningEnteredTimeMs(1600);
        task.setRemoteKvWaitMs(200);
        task.setFirstTokenTimeMs(1900);
        task.setHbmLocalMatchTokens(512);
        task.setRemoteKvAddedMatchTokens(256);
        task.setPrefillStepCount(3);
        task.setPrefillNonfinalChunkTokensMin(128);
        task.setPrefillNonfinalChunkTokensMax(256);

        reporter.reportPrefillWorkerStatusTask(
                "test-model", "10.0.0.1:8080@0", "PREFILL", "test-group", task);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "model", "test-model",
                "engineIp", "10.0.0.1:8080@0",
                "role", "PREFILL",
                "group", "test-group");
        verify(monitor).report("app.engine.worker.status.input.queue.wait.ms", expectedTags, 100.0);
        verify(monitor).report("app.engine.worker.status.scheduler.to.running.ms", expectedTags, 400.0);
        verify(monitor).report("app.engine.worker.status.scheduler.wait.ms", expectedTags, 200.0);
        verify(monitor).report("app.engine.worker.status.remote.kv.wait.ms", expectedTags, 200.0);
        verify(monitor).report("app.engine.worker.status.running.to.first.token.ms", expectedTags, 300.0);
        verify(monitor).report("app.engine.worker.status.hbm.local.match.tokens", expectedTags, 512.0);
        verify(monitor).report("app.engine.worker.status.remote.kv.added.match.tokens", expectedTags, 256.0);
        verify(monitor).report("app.engine.worker.status.prefill.step.count", expectedTags, 3.0);
        verify(monitor).report("app.engine.worker.status.prefill.nonfinal.chunk.tokens.min", expectedTags, 128.0);
        verify(monitor).report("app.engine.worker.status.prefill.nonfinal.chunk.tokens.max", expectedTags, 256.0);
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
        WorkerStatus workerStatus = workerStatus("10.0.0.1", RoleType.PREFILL);

        reporter.reportStatusCheckerSuccess("test-model", workerStatus, null, 3, 4);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "model", "test-model",
                "engineIp", "10.0.0.1:8080",
                "role", "PREFILL");
        verify(monitor).report("app.engine.health.check.running.task.info.size", expectedTags, 3.0);
        verify(monitor).report("app.engine.health.check.finished.task.list.size", expectedTags, 4.0);
    }

    @Test
    void shouldKeepLogicalMetricIdentityForMultiEngineWorker() {
        WorkerStatus workerStatus = WorkerStatus.createDiscovered(
                RoleType.PREFILL, null, "10.0.0.1", 8080, 8081,
                null, null, 1, 2);

        reporter.reportStatusCheckerSuccess("test-model", workerStatus, null, 3, 4);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "model", "test-model",
                "engineIp", "10.0.0.1:8080@1",
                "role", "PREFILL");
        verify(monitor).report("app.engine.health.check.running.task.info.size", expectedTags, 3.0);
    }

    @Test
    void shouldReportCacheCapacityMetricsFromSharedWorkerStatus() {
        WorkerStatus workerStatus = workerStatusWithCacheStatus();

        reporter.reportStatusCheckerSuccess("test-model", workerStatus, null, 0, 0);
        reporter.reportCacheStatusCheckerSuccess("test-model", workerStatus, 0L);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "model", "test-model",
                "engineIp", "10.0.0.1:8080",
                "role", "PREFILL");
        verify(monitor).report("app.cache.block.size",
                FlexMetricTags.of("model", "test-model", "role", "PREFILL"), 64.0);
        verify(monitor).report("app.cache.local.standby.block.size", expectedTags, 64.0);
        verify(monitor).report("app.cache.used.kv.cache.tokens", expectedTags, 200.0);
        verify(monitor).report("app.cache.available.kv.cache.tokens", expectedTags, 800.0);
        verify(monitor).report("app.cache.total.kv.cache.tokens",
                expectedTags, 1000.0);
        verify(monitor).report("app.cache.used.kv.cache.ratio", expectedTags, 20.0);
        verify(monitor).report("app.cache.key.size", expectedTags, 7.0);
    }

    @Test
    void shouldReportCacheStatusFailuresWithMetricWorkerIdentity() {
        WorkerStatus workerStatus = workerStatus("10.0.0.1", RoleType.PREFILL);
        BalanceStatusEnum failure = BalanceStatusEnum.CACHE_SERVICE_UNAVAILABLE;

        reporter.reportCacheStatusCheckerFail("test-model", workerStatus, failure);

        verify(monitor).report("app.cache.status.check.fail", FlexMetricTags.of(
                "model", "test-model",
                "engineIp", "10.0.0.1:8080",
                "code", String.valueOf(failure.getCode()),
                "role", "PREFILL"), 1.0);
    }

    @Test
    void shouldNotReportCacheCapacityMetricsWithoutCacheStatus() {
        WorkerStatus workerStatus = workerStatus("10.0.0.1", RoleType.PREFILL);

        reporter.reportCacheStatusCheckerSuccess("test-model", workerStatus, 0L);

        verify(monitor, never()).report(eq("app.cache.block.size"), any(FlexMetricTags.class), anyDouble());
        verify(monitor, never()).report(eq("app.cache.local.standby.block.size"),
                any(FlexMetricTags.class), anyDouble());
        verify(monitor, never()).report(eq("app.cache.used.kv.cache.ratio"), any(FlexMetricTags.class), anyDouble());
    }

    @Test
    void shouldReportConfiguredLocalStandbyBlockSize() {
        localStandbyConfig.setBlockSize(4096);
        WorkerStatus workerStatus = workerStatusWithCacheStatus();

        reporter.reportCacheStatusCheckerSuccess("test-model", workerStatus, 0L);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "model", "test-model",
                "engineIp", "10.0.0.1:8080",
                "role", "PREFILL");
        verify(monitor).report("app.cache.local.standby.block.size", expectedTags, 4096.0);
    }

    @Test
    void shouldNotReportLocalStandbyBlockSizeWhenStandbyIsDisabled() {
        when(cacheMatchConfiguration.isLocalStandbyEnabled()).thenReturn(false);
        WorkerStatus workerStatus = workerStatusWithCacheStatus();

        reporter.reportCacheStatusCheckerSuccess("test-model", workerStatus, 0L);

        verify(monitor, never()).report(eq("app.cache.local.standby.block.size"),
                any(FlexMetricTags.class), anyDouble());
    }

    @Test
    void shouldKeepCacheKeyMetricOnCacheStatusCheckerPath() {
        WorkerStatus workerStatus = workerStatusWithCacheStatus();

        reporter.reportCacheStatusCheckerSuccess("test-model", workerStatus, 0L);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "model", "test-model",
                "engineIp", "10.0.0.1:8080",
                "role", "PREFILL");
        verify(monitor).report("app.cache.key.size", expectedTags, 7.0);
    }

    @Test
    void shouldReportCacheHitComparisonTokenMetricsWithStableDimensions() {
        CacheHitComparisonResult comparison = new CacheHitComparisonResult(
                "cache_hit_comparison", "request-1", "KVCM", "PREFILL", "test-group",
                new WorkerIdentity("10.0.0.1", 8080, 0),
                "running", 200,
                new CacheHitComparisonResult.Actual(120),
                new CacheHitComparisonResult.HitComparison(100, 20),
                new CacheHitComparisonResult.HitComparison(80, 40),
                null);

        reporter.reportCacheHitComparisonMetrics("test-model", comparison);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "model", "test-model",
                "engineIp", "10.0.0.1:8080@0",
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
                "engineIp", "10.0.0.1:8080@0",
                "role", "PREFILL",
                "group", "test-group",
                "taskState", "running",
                "cacheMatchSource", "KVCM"), expectedTags.getTags());
    }

    @Test
    void shouldReportSelectedKvcmP2pMatchDetails() {
        reporter.reportKvcmSelectedMatch(RoleType.PREFILL, "10.0.0.1:8080@0", 40, 80, 100, true);

        verify(cacheMetricsReporter).reportKvcmSelectedMatch(
                RoleType.PREFILL, "10.0.0.1:8080@0", 40, 80, 100);
    }

    @Test
    void shouldSkipSelectedKvcmP2pMetricsWhenDetailsAreUnavailable() {
        reporter.reportKvcmSelectedMatch(RoleType.PREFILL, "10.0.0.1:8080@0", 0, 0, 0, false);

        verify(cacheMetricsReporter, never()).reportKvcmSelectedMatch(
                org.mockito.ArgumentMatchers.any(),
                org.mockito.ArgumentMatchers.anyString(),
                org.mockito.ArgumentMatchers.anyLong(),
                org.mockito.ArgumentMatchers.anyLong(),
                org.mockito.ArgumentMatchers.anyLong());
    }

    @Test
    void shouldNotReportLocalStandbyMetricsWhenPredictionIsUnavailable() {
        CacheHitComparisonResult comparison = new CacheHitComparisonResult(
                "cache_hit_comparison", "request-1", "LOCAL_SYNC", "PREFILL", "test-group",
                new WorkerIdentity("10.0.0.1", 8080, 0),
                "running", 200,
                new CacheHitComparisonResult.Actual(120),
                new CacheHitComparisonResult.HitComparison(100, 20),
                null,
                null);

        reporter.reportCacheHitComparisonMetrics("test-model", comparison);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "model", "test-model",
                "engineIp", "10.0.0.1:8080@0",
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
    }

    @Test
    void shouldReportKvcmLocalAndP2pDeltasWhenAvailable() {
        CacheHitComparisonResult comparison = new CacheHitComparisonResult(
                "cache_hit_comparison", "request-1", "KVCM", "PREFILL", "test-group",
                new WorkerIdentity("10.0.0.1", 8080, 0),
                "running", 200,
                new CacheHitComparisonResult.Actual(120),
                new CacheHitComparisonResult.HitComparison(60, 60),
                null,
                new CacheHitComparisonResult.KvcmDetails(
                        new CacheHitComparisonResult.HitComparison(40, 80),
                        new CacheHitComparisonResult.HitComparison(100, 20)));

        reporter.reportCacheHitComparisonMetrics("test-model", comparison);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "model", "test-model",
                "engineIp", "10.0.0.1:8080@0",
                "role", "PREFILL",
                "group", "test-group",
                "taskState", "running",
                "cacheMatchSource", "KVCM");
        verify(monitor).report("app.cache.hit.comparison.kvcm.local.delta.tokens", expectedTags, 80.0);
        verify(monitor).report("app.cache.hit.comparison.kvcm.p2p.total.match.delta.tokens", expectedTags, 20.0);
    }

    @Test
    void shouldNotReportRatiosWithoutInputTokens() {
        CacheHitComparisonResult comparison = new CacheHitComparisonResult(
                "cache_hit_comparison", "request-1", "KVCM", "PREFILL", "test-group",
                new WorkerIdentity("10.0.0.1", 8080, 0),
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
        return workerStatus("10.0.0.1", RoleType.PREFILL, 800L, 1000L,
                CacheStatus.builder()
                .blockSize(64)
                .cacheKeySize(7)
                .build());
    }

    private WorkerStatus workerStatus(String ip, RoleType role) {
        return workerStatus(ip, role, 0L, 0L, null);
    }

    private WorkerStatus workerStatus(
            String ip,
            RoleType role,
            long availableKvCacheTokens,
            long totalKvCacheTokens,
            CacheStatus cacheStatus) {
        WorkerStatus workerStatus = WorkerStatus.createDiscovered(
                role, null, ip, 8080, 8081, "test-site");
        if (cacheStatus != null) {
            workerStatus.publishCacheStatus(cacheStatus);
        }
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(role);
        response.setAlive(true);
        response.setStatusVersion(1L);
        response.setLatestFinishedVersion(0L);
        response.setRunningTaskInfo(Map.of());
        response.setFinishedTaskInfo(Map.of());
        response.setAvailableKvCacheTokens(availableKvCacheTokens);
        response.setTotalKvCacheTokens(totalKvCacheTokens);

        workerStatus.lock.lock();
        try {
            WorkerStatus.PreparedStatus prepared = workerStatus.prepareNewStatus(
                    workerStatus.freezeStatusResponse(response));
            workerStatus.publishPreparedStatus(prepared);
            workerStatus.recordSuccessfulPoll(true);
        } finally {
            workerStatus.lock.unlock();
        }
        return workerStatus;
    }
}
