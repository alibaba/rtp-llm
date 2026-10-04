package org.flexlb.service.monitor;

import io.netty.channel.EventLoopGroup;
import org.flexlb.balance.endpoint.EncoderEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.cache.domain.CacheHitComparisonResult;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.constant.MetricConstant;
import org.flexlb.constant.ZkMasterEvent;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerIdentity;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.client.EngineGrpcClient;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.FlexStatisticsType;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.TaskScheduler;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.scheduling.annotation.ScheduledAnnotationBeanPostProcessor;
import org.springframework.scheduling.support.ScheduledMethodRunnable;
import reactor.netty.resources.LoopResources;

import java.util.List;
import java.util.Map;
import java.util.OptionalLong;

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
    private final EngineGrpcClient engineGrpcClient = mock(EngineGrpcClient.class);
    private final LoopResources loopResources = mock(LoopResources.class);
    private final WorkerDirectory workerDirectory = mock(WorkerDirectory.class);

    private EngineHealthReporter reporter;

    @BeforeEach
    void setUp() {
        when(loopResources.onServer(true)).thenReturn(mock(EventLoopGroup.class));
        when(loopResources.onServerSelect(true)).thenReturn(mock(EventLoopGroup.class));
        when(engineGrpcClient.getEventLoopGroup()).thenReturn(mock(EventLoopGroup.class));
        reporter = new EngineHealthReporter(
                monitor, cacheMetricsReporter, engineGrpcClient,
                loopResources, workerDirectory);
    }

    @Test
    void reportsStepSamplesWithEngineIdentityAndPhase() {
        reporter.init();
        WorkerStatus worker = WorkerStatus.createDiscovered(
                RoleType.PDFUSION, "test-group", "10.0.0.1", 8080, 8081, null, null, 1, 2);
        for (int prefillRequests : new int[]{2, 0}) {
            var step = new WorkerStatus.StepMetrics(42, 1700000000000L, 16000,
                    prefillRequests, prefillRequests > 0 ? 15000 : 0, 32000, 0.5);
            reporter.reportWorkerStepMetrics(worker, step);
            var tags = FlexMetricTags.of("engineIp", "10.0.0.1:8080@1",
                    "role", "PDFUSION", "group", "test-group", "phase", prefillRequests > 0 ? "prefill" : "decode");
            verify(monitor).report(MetricConstant.ENGINE_WORKER_STEP_TOTAL_SCHEDULED_TOKENS, tags, 16000.0);
            verify(monitor).report(MetricConstant.ENGINE_WORKER_STEP_PREFILL_REQUEST_COUNT, tags, (double) prefillRequests);
            verify(monitor).report(MetricConstant.ENGINE_WORKER_STEP_PREFILL_TOKENS, tags, prefillRequests > 0 ? 15000.0 : 0.0);
            verify(monitor).report(MetricConstant.ENGINE_WORKER_STEP_TOKEN_BUDGET, tags, 32000.0);
            verify(monitor).report(MetricConstant.ENGINE_WORKER_STEP_BUDGET_FILL_RATIO, tags, 0.5);
        }
        verify(monitor).register(MetricConstant.ENGINE_WORKER_STEP_BUDGET_FILL_RATIO,
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
    }

    @Test
    void shouldRegisterCacheHitComparisonMetrics() {
        reporter.init();

        verify(monitor).register(MetricConstant.CACHE_HIT_COMPARISON_INPUT_TOKENS, FlexMetricType.COUNTER, FlexPriorityType.PRECISE);
        verify(monitor).register(MetricConstant.CACHE_HIT_COMPARISON_ACTUAL_TOKENS, FlexMetricType.COUNTER, FlexPriorityType.PRECISE);
        verify(monitor).register(MetricConstant.CACHE_HIT_COMPARISON_KVCM_PREDICTED_TOKENS,
                FlexMetricType.COUNTER, FlexPriorityType.PRECISE);
        verify(monitor).register(MetricConstant.CACHE_HIT_COMPARISON_DELTA_TOKENS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).register(MetricConstant.CACHE_HIT_COMPARISON_KVCM_LOCAL_DELTA_TOKENS,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).register(MetricConstant.CACHE_HIT_COMPARISON_KVCM_GLOBAL_MATCH_DELTA_TOKENS,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).register(MetricConstant.CACHE_HIT_COMPARISON_LOCAL_STANDBY_DELTA_TOKENS,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).register(MetricConstant.CACHE_HIT_COMPARISON_LOCAL_STANDBY_DELTA_RATIO,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).register(MetricConstant.CACHE_HIT_COMPARISON_ACTUAL_RATIO,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
    }

    @Test
    void shouldRegisterStatusCheckFailureMetrics() {
        reporter.init();

        verify(monitor).register(MetricConstant.ENGINE_STATUS_CHECK_FAIL_TOTAL,
                FlexMetricType.COUNTER, FlexPriorityType.PRECISE);
        verify(monitor).register(MetricConstant.ENGINE_STATUS_CHECK_FAIL_RT,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
    }

    @Test
    void shouldReportStatusCheckFailureCountAndLatencyWithSameTags() {
        BalanceStatusEnum failure = BalanceStatusEnum.WORKER_STATUS_GRPC_TIMEOUT;
        FlexMetricTags expectedTags = FlexMetricTags.of(
                "code", String.valueOf(failure.getCode()),
                "engineIp", "10.0.0.1:8080@0",
                "role", RoleType.PREFILL.getCode());

        reporter.reportStatusCheckerFail(failure, "10.0.0.1:8080@0", RoleType.PREFILL);
        reporter.reportStatusCheckFailureLatency(
                failure, "10.0.0.1:8080@0", RoleType.PREFILL, 201_234);

        verify(monitor).report(MetricConstant.ENGINE_STATUS_CHECK_FAIL, expectedTags, 1.0);
        verify(monitor).report(MetricConstant.ENGINE_STATUS_CHECK_FAIL_TOTAL, expectedTags, 1.0);
        verify(monitor).report(MetricConstant.ENGINE_STATUS_CHECK_FAIL_RT, expectedTags, 201_234.0);
    }

    @Test
    void statusCheckFailuresWithoutWorkerIdentityUseSameTagsAndCounters() {
        BalanceStatusEnum failure = BalanceStatusEnum.SERVICE_DISCOVERY_ERROR;

        reporter.reportStatusCheckerFail(failure, null);

        FlexMetricTags tags = FlexMetricTags.of("code", String.valueOf(failure.getCode()),
                "engineIp", "", "role", "");
        verify(monitor).report(MetricConstant.ENGINE_STATUS_CHECK_FAIL, tags, 1.0);
        verify(monitor).report(MetricConstant.ENGINE_STATUS_CHECK_FAIL_TOTAL, tags, 1.0);
    }

    @Test
    void lifecycleMetricsRetainMissingLabelsAsEmptyValues() {
        reporter.reportFlexlbObservedWaitingToRunningLatency(null, null, null, 42L);

        FlexMetricTags tags = FlexMetricTags.of("engineIp", "", "role", "", "group", "");
        verify(monitor).report(MetricConstant.ENGINE_WORKER_STATUS_FLEXLB_OBSERVED_WAITING_TO_RUNNING_MS,
                tags, 42.0);
    }

    @Test
    void encoderStatusWithGenericEndpointKeepsCommonCapacityMetrics() {
        WorkerStatus worker = workerStatus("10.0.0.1", RoleType.ENCODER, 800, 1000, null, 3, 4);
        WorkerEndpoint endpoint = mock(WorkerEndpoint.class);
        when(endpoint.getLoadMetric()).thenReturn(OptionalLong.empty());

        reporter.reportStatusCheckerSuccess(worker, endpoint, 3, 1);

        FlexMetricTags tags = FlexMetricTags.of("engineIp", "10.0.0.1:8080", "role", "ENCODER");
        verify(monitor).report(MetricConstant.ENCODER_PENDING_REQUEST_COUNT, tags, 0.0);
        verify(monitor).report(MetricConstant.ENCODER_SELECTION_LOAD, tags, 7.0);
        verify(monitor).report(MetricConstant.ENCODER_UNCACHED_TOKEN_LOAD, tags, 0.0);
        verify(monitor).report(MetricConstant.CACHE_TOTAL_KV_CACHE_TOKENS, tags, 1000.0);
    }

    @Test
    void shouldRegisterMasterDecisionToWaitingConfirmationMetric() {
        reporter.init();

        verify(monitor).register(MetricConstant.ENGINE_WORKER_STATUS_FLEXLB_OBSERVED_MASTER_DECISION_TO_WAITING_CONFIRM_MS,
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
    }

    @Test
    void shouldRegisterRequestPayloadMetrics() {
        reporter.init();

        verify(monitor).register(MetricConstant.REQUEST_BLOCK_SIZE, FlexMetricType.GAUGE);
        verify(monitor).register(MetricConstant.REQUEST_SEQ_LEN,
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        verify(monitor).register(MetricConstant.REQUEST_MESSAGE_BYTES,
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
    }

    @Test
    void shouldReportRequestPayloadMetricsWithoutResponse() {
        BalanceContext context = new BalanceContext();
        context.setSuccess(false);
        Request request = new Request();
        request.setSeqLen(512L);
        request.setCacheKeyBlockSize(1024L);
        request.setBlockSize(2048L);
        context.setRequest(request);
        context.setRequestMessageBytes(8192L);

        reporter.reportRequestPayload(context);

        FlexMetricTags expectedTags = FlexMetricTags.of("success", "false");
        verify(monitor).report(MetricConstant.REQUEST_SEQ_LEN, expectedTags, 512.0);
        verify(monitor).report(MetricConstant.REQUEST_BLOCK_SIZE, expectedTags, 1024.0);
        verify(monitor).report(MetricConstant.REQUEST_MESSAGE_BYTES, expectedTags, 8192.0);
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
        BalanceContext context = new BalanceContext(new FlexlbConfig());
        context.recordSelectionReason(
                RoleType.PREFILL, "LEAST_RECENTLY_USED_IN_POOL");
        context.setResponse(response);

        reporter.reportBalancingService(context);

        verify(monitor).report(MetricConstant.ENGINE_BALANCING_MASTER_SELECT_DETAIL, FlexMetricTags.of(
                "role", "PREFILL",
                "reason", "LEAST_RECENTLY_USED_IN_POOL",
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

        verify(monitor, never()).report(eq(MetricConstant.REQUEST_BLOCK_SIZE), any(FlexMetricTags.class), anyDouble());
        verify(monitor, never()).report(eq(MetricConstant.REQUEST_SEQ_LEN), any(FlexMetricTags.class), anyDouble());
        verify(monitor, never()).report(eq(MetricConstant.REQUEST_MESSAGE_BYTES), any(FlexMetricTags.class), anyDouble());
        verify(monitor, never()).report(eq("app.request.body.bytes"), any(FlexMetricTags.class), anyDouble());
    }

    @Test
    void shouldReportMasterDecisionToWaitingConfirmationLatency() {
        reporter.reportFlexlbObservedMasterDecisionToWaitingConfirmationLatency(
                "10.0.0.1:8080@0", "PREFILL", "test-group", 53);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "engineIp", "10.0.0.1:8080@0",
                "role", "PREFILL",
                "group", "test-group");
        verify(monitor).report(MetricConstant.ENGINE_WORKER_STATUS_FLEXLB_OBSERVED_MASTER_DECISION_TO_WAITING_CONFIRM_MS,
                expectedTags, 53.0);
    }

    @Test
    void shouldRegisterWaitingToRunningMetric() {
        reporter.init();

        verify(monitor).register(MetricConstant.ENGINE_WORKER_STATUS_FLEXLB_OBSERVED_WAITING_TO_RUNNING_MS,
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
    }

    @Test
    void shouldReportWaitingToRunningLatency() {
        reporter.reportFlexlbObservedWaitingToRunningLatency(
                "10.0.0.1:8080@0", "PREFILL", "test-group", 42);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "engineIp", "10.0.0.1:8080@0",
                "role", "PREFILL",
                "group", "test-group");
        verify(monitor).report(MetricConstant.ENGINE_WORKER_STATUS_FLEXLB_OBSERVED_WAITING_TO_RUNNING_MS,
                expectedTags, 42.0);
    }

    @Test
    void shouldRegisterEngineObservedWaitingToRunningMetric() {
        reporter.init();

        verify(monitor).register(MetricConstant.ENGINE_WORKER_STATUS_ENGINE_OBSERVED_WAITING_TO_RUNNING_MS,
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
    }

    @Test
    void shouldReportEngineObservedWaitingToRunningLatency() {
        reporter.reportEngineObservedWaitingToRunningLatency(
                "10.0.0.1:8080@0", "PREFILL", "test-group", 42);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "engineIp", "10.0.0.1:8080@0",
                "role", "PREFILL",
                "group", "test-group");
        verify(monitor).report(MetricConstant.ENGINE_WORKER_STATUS_ENGINE_OBSERVED_WAITING_TO_RUNNING_MS,
                expectedTags, 42.0);
    }

    @Test
    void shouldRegisterEngineObservedReceivedToWaitingMetric() {
        reporter.init();

        verify(monitor).register(MetricConstant.ENGINE_WORKER_STATUS_ENGINE_OBSERVED_RECEIVED_TO_WAITING_MS,
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
    }

    @Test
    void shouldReportEngineObservedReceivedToWaitingLatency() {
        reporter.reportEngineObservedReceivedToWaitingLatency(
                "10.0.0.1:8080@0", "PREFILL", "test-group", 42);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "engineIp", "10.0.0.1:8080@0",
                "role", "PREFILL",
                "group", "test-group");
        verify(monitor).report(MetricConstant.ENGINE_WORKER_STATUS_ENGINE_OBSERVED_RECEIVED_TO_WAITING_MS,
                expectedTags, 42.0);
    }

    @Test
    void shouldReportPrefillWorkerStatusTaskMetrics() {
        reporter.init();
        WorkerStatus.TaskTelemetry task = new WorkerStatus.TaskTelemetry(
                true, 900, 1000, 1100, 1200, 1600, 200, 1900,
                512, 256, 1, 3, 3, 128, 256);

        reporter.reportPrefillWorkerStatusTask(
                "10.0.0.1:8080@0", "PREFILL", "test-group", task);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "engineIp", "10.0.0.1:8080@0",
                "role", "PREFILL",
                "group", "test-group");
        verify(monitor).report(MetricConstant.ENGINE_WORKER_STATUS_INPUT_QUEUE_WAIT_MS, expectedTags, 100.0);
        verify(monitor).report(MetricConstant.ENGINE_WORKER_STATUS_SCHEDULER_TO_RUNNING_MS, expectedTags, 400.0);
        verify(monitor).report(MetricConstant.ENGINE_WORKER_STATUS_ENGINE_OBSERVED_RECEIVED_TO_WAITING_MS, expectedTags, 300.0);
        verify(monitor).report(MetricConstant.ENGINE_WORKER_STATUS_ENGINE_OBSERVED_WAITING_TO_RUNNING_MS, expectedTags, 400.0);
        verify(monitor).report(MetricConstant.ENGINE_WORKER_STATUS_SCHEDULER_WAIT_MS, expectedTags, 200.0);
        verify(monitor).report(MetricConstant.ENGINE_WORKER_STATUS_REMOTE_KV_WAIT_MS, expectedTags, 200.0);
        verify(monitor).report(MetricConstant.ENGINE_WORKER_STATUS_RUNNING_TO_FIRST_TOKEN_MS, expectedTags, 300.0);
        verify(monitor).report(MetricConstant.ENGINE_WORKER_STATUS_HBM_LOCAL_MATCH_TOKENS, expectedTags, 512.0);
        verify(monitor).report(MetricConstant.ENGINE_WORKER_STATUS_REMOTE_KV_ADDED_MATCH_TOKENS, expectedTags, 256.0);
        verify(monitor).report(MetricConstant.ENGINE_WORKER_STATUS_PREFILL_STEP_COUNT, expectedTags, 3.0);
        verify(monitor).report(MetricConstant.ENGINE_WORKER_STATUS_PREFILL_NONFINAL_CHUNK_TOKENS_MIN, expectedTags, 128.0);
        verify(monitor).report(MetricConstant.ENGINE_WORKER_STATUS_PREFILL_NONFINAL_CHUNK_TOKENS_MAX, expectedTags, 256.0);
        verify(monitor).register(MetricConstant.ENGINE_WORKER_STATUS_PREFILL_NONFINAL_CHUNK_TOKENS_MIN,
                FlexMetricType.GAUGE, FlexPriorityType.TRIVIAL);
        verify(monitor).register(MetricConstant.ENGINE_WORKER_STATUS_PREFILL_NONFINAL_CHUNK_TOKENS_MAX,
                FlexMetricType.GAUGE, FlexPriorityType.TRIVIAL);
    }

    @Test
    void shouldExcludeRequestsWithoutNonfinalChunksFromChunkMetrics() {
        WorkerStatus.TaskTelemetry task = new WorkerStatus.TaskTelemetry(
                true, 900, 1000, 1100, 1200, 1600, 0, 1900,
                0, 0, 1, 1, 1, 0, 0);

        reporter.reportPrefillWorkerStatusTask(
                "10.0.0.1:8080@0", "PREFILL", "test-group", task);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "engineIp", "10.0.0.1:8080@0",
                "role", "PREFILL",
                "group", "test-group");
        verify(monitor).report(MetricConstant.ENGINE_WORKER_STATUS_PREFILL_STEP_COUNT, expectedTags, 1.0);
        verify(monitor).report(MetricConstant.ENGINE_WORKER_STATUS_HBM_LOCAL_MATCH_TOKENS, expectedTags, 0.0);
        verify(monitor, never()).report(
                eq(MetricConstant.ENGINE_WORKER_STATUS_PREFILL_NONFINAL_CHUNK_TOKENS_MIN), any(), anyDouble());
        verify(monitor, never()).report(
                eq(MetricConstant.ENGINE_WORKER_STATUS_PREFILL_NONFINAL_CHUNK_TOKENS_MAX), any(), anyDouble());
    }

    @Test
    void shouldReportZkMasterEventTime() {
        long beforeReport = System.currentTimeMillis();

        reporter.reportPrefillBalanceMasterEvent(ZkMasterEvent.MASTER_TAKE_LEADERSHIP);

        long afterReport = System.currentTimeMillis();
        verify(monitor).report(
                eq(MetricConstant.ZK_MASTER_EVENT),
                eq(FlexMetricTags.of("event", ZkMasterEvent.MASTER_TAKE_LEADERSHIP.name())),
                doubleThat(value -> value >= beforeReport && value <= afterReport));
    }

    @Test
    void shouldReportWorkerTaskCounts() {
        WorkerStatus workerStatus = workerStatus("10.0.0.1", RoleType.PREFILL);

        reporter.reportStatusCheckerSuccess(workerStatus, null, 3, 4);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "engineIp", "10.0.0.1:8080",
                "role", "PREFILL");
        verify(monitor).report(MetricConstant.ENGINE_RUNNING_TASK_INFO_SIZE, expectedTags, 3.0);
        verify(monitor).report(MetricConstant.ENGINE_FINISHED_TASK_LIST_SIZE, expectedTags, 4.0);
    }

    @Test
    void reportsEncoderSelectionInputsFromWorkerStatus() {
        reporter.init();
        WorkerStatus workerStatus = workerStatus("10.0.0.1", RoleType.ENCODER, 800, 1000, null, 3, 4);
        EncoderEndpoint endpoint = mock(EncoderEndpoint.class);
        when(endpoint.pendingEncoderRequestCount()).thenReturn(2);
        when(endpoint.inflightUncachedTokenEstimate()).thenReturn(640L);
        when(endpoint.getLoadMetric()).thenReturn(OptionalLong.empty());

        reporter.reportStatusCheckerSuccess(workerStatus, endpoint, 3, 1);

        FlexMetricTags tags = FlexMetricTags.of(
                "engineIp", "10.0.0.1:8080", "role", "ENCODER");
        verify(monitor).register(MetricConstant.ENCODER_PENDING_REQUEST_COUNT,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).register(MetricConstant.ENCODER_SELECTION_LOAD,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).register(MetricConstant.ENCODER_UNCACHED_TOKEN_LOAD,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).report(MetricConstant.ENCODER_PENDING_REQUEST_COUNT, tags, 2.0);
        verify(monitor).report(MetricConstant.ENCODER_SELECTION_LOAD, tags, 9.0);
        verify(monitor).report(MetricConstant.ENCODER_UNCACHED_TOKEN_LOAD, tags, 640.0);
        verify(monitor).report(MetricConstant.CACHE_AVAILABLE_KV_CACHE_TOKENS, tags, 800.0);
    }

    @Test
    void shouldKeepLogicalMetricIdentityForMultiEngineWorker() {
        WorkerStatus workerStatus = WorkerStatus.createDiscovered(
                RoleType.PREFILL, null, "10.0.0.1", 8080, 8081,
                null, null, 1, 2);

        reporter.reportStatusCheckerSuccess(workerStatus, null, 3, 4);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "engineIp", "10.0.0.1:8080@1",
                "role", "PREFILL");
        verify(monitor).report(MetricConstant.ENGINE_RUNNING_TASK_INFO_SIZE, expectedTags, 3.0);
    }

    @Test
    void shouldReportCacheCapacityMetricsFromWorkerStatusWithoutCacheStatusPoll() {
        WorkerStatus workerStatus = workerStatus("10.0.0.1", RoleType.PREFILL, 800L, 1000L, null);

        reporter.reportStatusCheckerSuccess(workerStatus, null, 0, 0);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "engineIp", "10.0.0.1:8080",
                "role", "PREFILL");
        verify(monitor).report(MetricConstant.CACHE_USED_KV_CACHE_TOKENS, expectedTags, 200.0);
        verify(monitor).report(MetricConstant.CACHE_AVAILABLE_KV_CACHE_TOKENS, expectedTags, 800.0);
        verify(monitor).report(MetricConstant.CACHE_TOTAL_KV_CACHE_TOKENS, expectedTags, 1000.0);
        verify(monitor).report(MetricConstant.CACHE_USED_KV_CACHE_RATIO, expectedTags, 20.0);
    }

    @Test
    void shouldReportCacheStatusFailuresWithMetricWorkerIdentity() {
        WorkerStatus workerStatus = workerStatus("10.0.0.1", RoleType.PREFILL);
        BalanceStatusEnum failure = BalanceStatusEnum.CACHE_SERVICE_UNAVAILABLE;

        reporter.reportCacheStatusCheckerFail(workerStatus, failure);

        verify(monitor).report(MetricConstant.CACHE_STATUS_CHECK_FAIL, FlexMetricTags.of(
                "engineIp", "10.0.0.1:8080",
                "code", String.valueOf(failure.getCode()),
                "role", "PREFILL"), 1.0);
    }

    @Test
    void shouldNotReportCacheCapacityMetricsWithoutCacheStatus() {
        WorkerStatus workerStatus = workerStatus("10.0.0.1", RoleType.PREFILL);

        reporter.reportCacheStatusCheckerSuccess(workerStatus, 0L);

        verify(monitor, never()).report(eq(MetricConstant.CACHE_BLOCK_SIZE), any(FlexMetricTags.class), anyDouble());
        verify(monitor, never()).report(eq(MetricConstant.CACHE_LOCAL_STANDBY_BLOCK_SIZE),
                any(FlexMetricTags.class), anyDouble());
        verify(monitor, never()).report(eq(MetricConstant.CACHE_USED_KV_CACHE_RATIO), any(FlexMetricTags.class), anyDouble());
    }

    @Test
    void shouldNotReportLocalStandbyBlockSizeFromWorkerStatus() {
        WorkerStatus workerStatus = workerStatus("10.0.0.1", RoleType.PREFILL);

        reporter.reportStatusCheckerSuccess(workerStatus, null, 0, 0);

        verify(monitor, never()).report(eq(MetricConstant.CACHE_LOCAL_STANDBY_BLOCK_SIZE),
                any(FlexMetricTags.class), anyDouble());
    }

    @Test
    void shouldReportOneWorkerBlockSizePerRole() {
        WorkerStatus prefill = workerStatusWithCacheStatus();
        WorkerStatus decode = workerStatus("10.0.0.2", RoleType.DECODE, 800L, 1000L,
                CacheStatus.builder().blockSize(128).build());
        when(workerDirectory.getWorkerStatuses(RoleType.PREFILL, null))
                .thenReturn(List.of(WorkerStatus.createDiscovered(
                        RoleType.PREFILL, null, "10.0.0.3", 8080, 8081, "test-site"), prefill, prefill));
        when(workerDirectory.getWorkerStatuses(RoleType.DECODE, null)).thenReturn(List.of(decode));

        org.springframework.test.util.ReflectionTestUtils.invokeMethod(reporter, "reportWorkerBlockSizes");

        verify(monitor).report(MetricConstant.CACHE_BLOCK_SIZE, FlexMetricTags.of("role", "PREFILL"), 64.0);
        verify(monitor).report(MetricConstant.CACHE_BLOCK_SIZE, FlexMetricTags.of("role", "DECODE"), 128.0);
    }

    @Test
    void springRegistersWorkerBlockSizeReportingTask() {
        when(workerDirectory.getWorkerStatuses(RoleType.PREFILL, null))
                .thenReturn(List.of(workerStatusWithCacheStatus()));
        when(workerDirectory.getWorkerStatuses(RoleType.DECODE, null)).thenReturn(List.of());

        try (AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext()) {
            context.register(SchedulingConfiguration.class);
            context.registerBean(EngineHealthReporter.class, () -> reporter);
            context.refresh();
            Runnable task = context.getBean(ScheduledAnnotationBeanPostProcessor.class).getScheduledTasks().stream()
                    .map(scheduled -> scheduled.getTask().getRunnable())
                    .filter(runnable -> runnable instanceof ScheduledMethodRunnable method
                            && method.getMethod().getName().equals("reportWorkerBlockSizes"))
                    .findFirst().orElseThrow();

            task.run();

            verify(monitor).report(MetricConstant.CACHE_BLOCK_SIZE,
                    FlexMetricTags.of("role", "PREFILL"), 64.0);
        }
    }

    @Configuration
    @EnableScheduling
    static class SchedulingConfiguration {
        @Bean
        TaskScheduler taskScheduler() {
            return mock(TaskScheduler.class);
        }
    }

    @Test
    void shouldSkipWorkerBlockSizeBeforeStatusIsAvailable() {
        when(workerDirectory.getWorkerStatuses(RoleType.PREFILL, null))
                .thenReturn(List.of(WorkerStatus.createDiscovered(
                        RoleType.PREFILL, null, "10.0.0.1", 8080, 8081, "test-site")));
        when(workerDirectory.getWorkerStatuses(RoleType.DECODE, null))
                .thenReturn(List.of(WorkerStatus.createDiscovered(
                        RoleType.DECODE, null, "10.0.0.2", 8080, 8081, "test-site")));

        org.springframework.test.util.ReflectionTestUtils.invokeMethod(reporter, "reportWorkerBlockSizes");

        verify(monitor, never()).report(eq(MetricConstant.CACHE_BLOCK_SIZE), any(FlexMetricTags.class), anyDouble());
    }

    @Test
    void shouldNotReportLocalStandbyBlockSizeFromCacheStatus() {
        WorkerStatus workerStatus = workerStatusWithCacheStatus();

        reporter.reportCacheStatusCheckerSuccess(workerStatus, 0L);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "engineIp", "10.0.0.1:8080",
                "role", "PREFILL");
        verify(monitor, never()).report(eq(MetricConstant.CACHE_LOCAL_STANDBY_BLOCK_SIZE),
                any(FlexMetricTags.class), anyDouble());
    }

    @Test
    void shouldNotReportLocalStandbyBlockSizeWhenStandbyIsDisabled() {
        WorkerStatus workerStatus = workerStatusWithCacheStatus();

        reporter.reportCacheStatusCheckerSuccess(workerStatus, 0L);

        verify(monitor, never()).report(eq(MetricConstant.CACHE_LOCAL_STANDBY_BLOCK_SIZE),
                any(FlexMetricTags.class), anyDouble());
    }

    @Test
    void shouldKeepCacheKeyMetricOnCacheStatusCheckerPath() {
        WorkerStatus workerStatus = workerStatusWithCacheStatus();

        reporter.reportCacheStatusCheckerSuccess(workerStatus, 0L);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "engineIp", "10.0.0.1:8080",
                "role", "PREFILL");
        verify(monitor).report(MetricConstant.CACHE_KEY_SIZE, expectedTags, 7.0);
    }

    @Test
    void shouldReportActualCacheHitsWithoutPredictions() {
        CacheHitComparisonResult comparison = new CacheHitComparisonResult(
                "cache_hit_comparison", "request-1", "KVCM", "PREFILL", "test-group",
                new WorkerIdentity("10.0.0.1", 8080, 0), "running", 200, 120,
                null, null, null);

        reporter.reportCacheHitComparisonMetrics(comparison);

        verify(monitor).report(eq(MetricConstant.CACHE_HIT_COMPARISON_INPUT_TOKENS),
                any(FlexMetricTags.class), eq(200.0));
        verify(monitor).report(eq(MetricConstant.CACHE_HIT_COMPARISON_ACTUAL_TOKENS),
                any(FlexMetricTags.class), eq(120.0));
        verify(monitor, never()).report(eq(MetricConstant.CACHE_HIT_COMPARISON_DELTA_TOKENS),
                any(FlexMetricTags.class), anyDouble());
    }

    @Test
    void shouldReportCacheHitComparisonTokenMetricsWithStableDimensions() {
        CacheHitComparisonResult comparison = new CacheHitComparisonResult(
                "cache_hit_comparison", "request-1", "KVCM", "PREFILL", "test-group",
                new WorkerIdentity("10.0.0.1", 8080, 0),
                "running", 200,
                120,
                new CacheHitComparisonResult.CachePrediction(100, -1, -1),
                null,
                new CacheHitComparisonResult.CachePrediction(80, -1, -1));

        reporter.reportCacheHitComparisonMetrics(comparison);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "engineIp", "10.0.0.1:8080@0",
                "role", "PREFILL",
                "group", "test-group",
                "taskState", "running",
                "cacheMatchSource", "KVCM");
        verify(monitor).report(MetricConstant.CACHE_HIT_COMPARISON_ACTUAL_TOKENS, expectedTags, 120.0);
        verify(monitor).report(MetricConstant.CACHE_HIT_COMPARISON_KVCM_PREDICTED_TOKENS, expectedTags, 100.0);
        verify(monitor).report(MetricConstant.CACHE_HIT_COMPARISON_DELTA_TOKENS, expectedTags, 20.0);
        verify(monitor).report(MetricConstant.CACHE_HIT_COMPARISON_LOCAL_STANDBY_DELTA_TOKENS, expectedTags, 40.0);
        verify(monitor).report(MetricConstant.CACHE_HIT_COMPARISON_LOCAL_STANDBY_DELTA_RATIO, expectedTags, 0.2);
        verify(monitor).report(MetricConstant.CACHE_HIT_COMPARISON_ACTUAL_RATIO, expectedTags, 0.6);
        verify(monitor).report(MetricConstant.CACHE_HIT_COMPARISON_INPUT_TOKENS, expectedTags, 200.0);
        assertEquals(Map.of(
                "engineIp", "10.0.0.1:8080@0",
                "role", "PREFILL",
                "group", "test-group",
                "taskState", "running",
                "cacheMatchSource", "KVCM"), expectedTags.getTags());
    }

    @Test
    void shouldReportSelectedKvcmGlobalMatchDetails() {
        reporter.reportKvcmSelectedMatch(RoleType.PREFILL, "10.0.0.1:8080@0", 40, 100, 200);

        verify(cacheMetricsReporter).reportKvcmSelectedMatch(
                RoleType.PREFILL, "10.0.0.1:8080@0", 40, 100, 200);
    }

    @Test
    void shouldNotReportLocalStandbyMetricsWhenPredictionIsUnavailable() {
        CacheHitComparisonResult comparison = new CacheHitComparisonResult(
                "cache_hit_comparison", "request-1", "LOCAL_SYNC", "PREFILL", "test-group",
                new WorkerIdentity("10.0.0.1", 8080, 0),
                "running", 200,
                120,
                null,
                new CacheHitComparisonResult.CachePrediction(100, -1, -1),
                null);

        reporter.reportCacheHitComparisonMetrics(comparison);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "engineIp", "10.0.0.1:8080@0",
                "role", "PREFILL",
                "group", "test-group",
                "taskState", "running",
                "cacheMatchSource", "LOCAL_SYNC");
        verify(monitor).report(MetricConstant.CACHE_HIT_COMPARISON_ACTUAL_TOKENS, expectedTags, 120.0);
        verify(monitor, never()).report(
                org.mockito.ArgumentMatchers.eq(MetricConstant.CACHE_HIT_COMPARISON_KVCM_PREDICTED_TOKENS),
                org.mockito.ArgumentMatchers.any(FlexMetricTags.class),
                org.mockito.ArgumentMatchers.anyDouble());
        verify(monitor).report(MetricConstant.CACHE_HIT_COMPARISON_DELTA_TOKENS, expectedTags, 20.0);
        verify(monitor, never()).report(
                org.mockito.ArgumentMatchers.eq(MetricConstant.CACHE_HIT_COMPARISON_LOCAL_STANDBY_DELTA_TOKENS),
                org.mockito.ArgumentMatchers.any(FlexMetricTags.class),
                org.mockito.ArgumentMatchers.anyDouble());
        verify(monitor, never()).report(
                org.mockito.ArgumentMatchers.eq(MetricConstant.CACHE_HIT_COMPARISON_KVCM_LOCAL_DELTA_TOKENS),
                org.mockito.ArgumentMatchers.any(FlexMetricTags.class),
                org.mockito.ArgumentMatchers.anyDouble());
        verify(monitor, never()).report(
                org.mockito.ArgumentMatchers.eq(MetricConstant.CACHE_HIT_COMPARISON_KVCM_GLOBAL_MATCH_DELTA_TOKENS),
                org.mockito.ArgumentMatchers.any(FlexMetricTags.class),
                org.mockito.ArgumentMatchers.anyDouble());
    }

    @Test
    void shouldReportKvcmLocalAndGlobalDeltasWhenAvailable() {
        CacheHitComparisonResult comparison = new CacheHitComparisonResult(
                "cache_hit_comparison", "request-1", "KVCM", "PREFILL", "test-group",
                new WorkerIdentity("10.0.0.1", 8080, 0),
                "running", 200,
                120,
                new CacheHitComparisonResult.CachePrediction(60, 40, 100),
                null,
                null);

        reporter.reportCacheHitComparisonMetrics(comparison);

        FlexMetricTags expectedTags = FlexMetricTags.of(
                "engineIp", "10.0.0.1:8080@0",
                "role", "PREFILL",
                "group", "test-group",
                "taskState", "running",
                "cacheMatchSource", "KVCM");
        verify(monitor).report(MetricConstant.CACHE_HIT_COMPARISON_KVCM_LOCAL_DELTA_TOKENS, expectedTags, 80.0);
        verify(monitor).report(MetricConstant.CACHE_HIT_COMPARISON_KVCM_GLOBAL_MATCH_DELTA_TOKENS, expectedTags, 20.0);
    }

    @Test
    void shouldNotReportRatiosWithoutInputTokens() {
        CacheHitComparisonResult comparison = new CacheHitComparisonResult(
                "cache_hit_comparison", "request-1", "KVCM", "PREFILL", "test-group",
                new WorkerIdentity("10.0.0.1", 8080, 0),
                "running", 0,
                120,
                new CacheHitComparisonResult.CachePrediction(100, -1, -1),
                null,
                new CacheHitComparisonResult.CachePrediction(80, -1, -1));

        reporter.reportCacheHitComparisonMetrics(comparison);

        verify(monitor).report(eq(MetricConstant.CACHE_HIT_COMPARISON_DELTA_TOKENS),
                any(FlexMetricTags.class), eq(20.0));
        verify(monitor, never()).report(
                org.mockito.ArgumentMatchers.eq(MetricConstant.CACHE_HIT_COMPARISON_ACTUAL_RATIO),
                org.mockito.ArgumentMatchers.any(FlexMetricTags.class),
                org.mockito.ArgumentMatchers.anyDouble());
        verify(monitor, never()).report(
                org.mockito.ArgumentMatchers.eq(MetricConstant.CACHE_HIT_COMPARISON_INPUT_TOKENS),
                org.mockito.ArgumentMatchers.any(FlexMetricTags.class),
                org.mockito.ArgumentMatchers.anyDouble());
        verify(monitor, never()).report(
                org.mockito.ArgumentMatchers.eq(MetricConstant.CACHE_HIT_COMPARISON_ACTUAL_TOKENS),
                org.mockito.ArgumentMatchers.any(FlexMetricTags.class),
                org.mockito.ArgumentMatchers.anyDouble());
        verify(monitor, never()).report(
                org.mockito.ArgumentMatchers.eq(MetricConstant.CACHE_HIT_COMPARISON_KVCM_PREDICTED_TOKENS),
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
        return workerStatus(ip, role, availableKvCacheTokens, totalKvCacheTokens,
                cacheStatus, 0L, 0L);
    }

    private WorkerStatus workerStatus(String ip,
                                      RoleType role,
                                      long availableKvCacheTokens,
                                      long totalKvCacheTokens,
                                      CacheStatus cacheStatus,
                                      long runningQueryLen,
                                      long waitingQueryLen) {
        WorkerStatus workerStatus = WorkerStatus.createDiscovered(
                role, null, ip, 8080, 8081, "test-site");
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(role);
        response.setCacheStatus(cacheStatus);
        response.setAlive(true);
        response.setStatusVersion(1L);
        response.setLatestFinishedVersion(0L);
        response.setRunningTaskInfo(Map.of());
        response.setFinishedTaskInfo(Map.of());
        response.setAvailableKvCacheTokens(availableKvCacheTokens);
        response.setTotalKvCacheTokens(totalKvCacheTokens);
        response.setRunningQueryLen(runningQueryLen);
        response.setWaitingQueryLen(waitingQueryLen);

        workerStatus.lock.lock();
        try {
            if (cacheStatus != null) {
                workerStatus.publishCacheStatus(cacheStatus);
            }
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
