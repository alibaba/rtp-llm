package org.flexlb.service.monitor;

import io.netty.channel.EventLoopGroup;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.EndpointTrace.Stage;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.GrpcFailure;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.jupiter.MockitoExtension;
import reactor.netty.resources.LoopResources;

import static org.flexlb.constant.MetricConstant.CACHE_AVAILABLE_KV_CACHE_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_BLOCK_SIZE;
import static org.flexlb.constant.MetricConstant.CACHE_KEY_SIZE;
import static org.flexlb.constant.MetricConstant.CACHE_STATUS_CHECK_FAIL;
import static org.flexlb.constant.MetricConstant.CACHE_STATUS_CHECK_SUCCESS_PERIOD;
import static org.flexlb.constant.MetricConstant.CACHE_TOTAL_KV_CACHE_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_USED_KV_CACHE_RATIO;
import static org.flexlb.constant.MetricConstant.CACHE_USED_KV_CACHE_TOKENS;
import static org.flexlb.constant.MetricConstant.ENGINE_FINISHED_TASK_LIST_SIZE;
import static org.flexlb.constant.MetricConstant.ENGINE_RUNNING_TASK_INFO_SIZE;
import static org.flexlb.constant.MetricConstant.ENGINE_STATUS_CHECK_FAIL;
import static org.flexlb.constant.MetricConstant.PREFILL_SELECTED_ESTIMATED_TTFT_MS;
import static org.flexlb.constant.MetricConstant.PREFILL_SELECTED_EXECUTION_TIME_MS;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

/**
 * Pins the metric tag shapes reported for engine status, cache poll, failure,
 * and selection callbacks, including the decode KV capacity reported at
 * status cadence.
 */
@ExtendWith(MockitoExtension.class)
class EngineHealthReporterSelectionMetricTest {

    @Mock
    private FlexMonitor monitor;
    @Mock
    private CacheMetricsReporter cacheMetricsReporter;
    @Mock
    private EngineGrpcClient engineGrpcClient;
    @Mock
    private LoopResources loopResources;
    @Mock
    private EventLoopGroup serverWorker;
    @Mock
    private EventLoopGroup serverSelector;
    @Mock
    private EventLoopGroup grpcEventLoop;
    @Mock
    private WorkerDirectory workerDirectory;
    @Mock
    private StatusFailureReporter statusFailureReporter;

    private EngineHealthReporter reporter;

    @BeforeEach
    void setUp() {
        when(loopResources.onServer(true)).thenReturn(serverWorker);
        when(loopResources.onServerSelect(true)).thenReturn(serverSelector);
        when(engineGrpcClient.getEventLoopGroup()).thenReturn(grpcEventLoop);
        reporter = new EngineHealthReporter(
                monitor, cacheMetricsReporter, engineGrpcClient, loopResources,
                workerDirectory, statusFailureReporter);
    }

    @Test
    void registersSelectedPrefillEstimateMetricsAsGauges() {
        reporter.init();

        verify(monitor).register(PREFILL_SELECTED_ESTIMATED_TTFT_MS,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).register(PREFILL_SELECTED_EXECUTION_TIME_MS,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
    }

    @Test
    void reportsSelectedPrefillEstimatesWithDeliveryMode() {
        reporter.reportPrefillSelectedEstimates(
                RoleType.PREFILL, "10.0.0.1", "NON_BATCH", 1_250L, 400L);

        FlexMetricTags tags = FlexMetricTags.of(
                "engineIp", "10.0.0.1",
                "role", "PREFILL",
                "delivery_mode", "NON_BATCH");
        verify(monitor).report(PREFILL_SELECTED_ESTIMATED_TTFT_MS, tags, 1_250.0);
        verify(monitor).report(PREFILL_SELECTED_EXECUTION_TIME_MS, tags, 400.0);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void reportsCoarseAndDetailedFailuresTogether(boolean cache) {
        GrpcFailure.Analysis failure = GrpcFailure.analyze(new IllegalStateException("contract"));
        BalanceStatusEnum status = cache ? BalanceStatusEnum.CACHE_UPDATE_FAILED
                : BalanceStatusEnum.MASTER_STATUS_PROCESSING_FAILED;

        reportFailure(cache, status, failure);

        verify(monitor).report(cache ? CACHE_STATUS_CHECK_FAIL : ENGINE_STATUS_CHECK_FAIL,
                FlexMetricTags.of("model", "test-model", "code", String.valueOf(status.getCode()),
                        "role", RoleType.PREFILL.getCode()), 1.0);
        verify(statusFailureReporter).report(RoleType.PREFILL, cache ? "cache_status" : "worker_status",
                Stage.STATUS_PROJECTION_FAILED, "STATUS_PROJECTION_FAILED", failure);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void coarseExportFailureDoesNotSuppressDetailedFailure(boolean cache) {
        GrpcFailure.Analysis failure = GrpcFailure.analyze(new IllegalStateException("contract"));
        BalanceStatusEnum status = cache ? BalanceStatusEnum.CACHE_UPDATE_FAILED
                : BalanceStatusEnum.MASTER_STATUS_PROCESSING_FAILED;
        var exportFailure = new IllegalStateException("metric unavailable");
        FlexMetricTags tags = FlexMetricTags.of(
                "model", "test-model", "code", String.valueOf(status.getCode()),
                "role", RoleType.PREFILL.getCode());
        doThrow(exportFailure).when(monitor)
                .report(cache ? CACHE_STATUS_CHECK_FAIL : ENGINE_STATUS_CHECK_FAIL, tags, 1.0);

        assertSame(exportFailure,
                assertThrows(IllegalStateException.class, () -> reportFailure(cache, status, failure)));

        verify(statusFailureReporter).report(RoleType.PREFILL, cache ? "cache_status" : "worker_status",
                Stage.STATUS_PROJECTION_FAILED, "STATUS_PROJECTION_FAILED", failure);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void cacheMetricsKeepEngineAndRoleDimensions(boolean populated) {
        WorkerStatus worker = Mockito.mock(WorkerStatus.class);
        WorkerStatus.TopologySnapshot topology =
                Mockito.mock(WorkerStatus.TopologySnapshot.class);
        WorkerStatus.EngineObservation observation =
                Mockito.mock(WorkerStatus.EngineObservation.class);
        when(worker.topologySnapshot()).thenReturn(topology);
        when(worker.committedEngineObservation()).thenReturn(observation);
        when(topology.ip()).thenReturn("10.0.0.1");
        when(observation.role()).thenReturn(RoleType.PREFILL);
        if (populated) {
            CacheStatus cache = Mockito.mock(CacheStatus.class);
            when(worker.getCacheStatus()).thenReturn(cache);
            when(cache.getBlockSize()).thenReturn(16L);
            when(cache.getCacheKeySize()).thenReturn(7L);
            when(observation.totalKvCacheTokens()).thenReturn(100L);
            when(observation.availableKvCacheTokens()).thenReturn(40L);
        }

        reporter.reportCacheStatusCheckerSuccess("model", worker, populated ? 20L : 0L);

        FlexMetricTags engine = FlexMetricTags.of(
                "model", "model", "engineIp", "10.0.0.1", "role", "PREFILL");
        FlexMetricTags role = FlexMetricTags.of("model", "model", "role", "PREFILL");
        if (populated) {
            verify(monitor).report(CACHE_STATUS_CHECK_SUCCESS_PERIOD, engine, 20.0);
            verify(monitor).report(CACHE_BLOCK_SIZE, role, 16.0);
            verify(monitor).report(CACHE_KEY_SIZE, engine, 7.0);
            verify(monitor).report(CACHE_USED_KV_CACHE_RATIO, engine, 60.0);
        }
        verify(monitor).report(CACHE_USED_KV_CACHE_TOKENS, engine, populated ? 60.0 : 0.0);
        verify(monitor).report(CACHE_AVAILABLE_KV_CACHE_TOKENS, engine, populated ? 40.0 : 0.0);
        verify(monitor).report(CACHE_TOTAL_KV_CACHE_TOKENS, role, populated ? 100.0 : 0.0);
        verifyNoMoreInteractions(monitor);
    }

    @Test
    void decodeStatusReportsFreshKvCapacityWithoutCacheMetadata() {
        WorkerStatus worker = Mockito.mock(WorkerStatus.class);
        WorkerStatus.TopologySnapshot topology =
                Mockito.mock(WorkerStatus.TopologySnapshot.class);
        WorkerStatus.EngineObservation observation =
                Mockito.mock(WorkerStatus.EngineObservation.class);
        when(worker.topologySnapshot()).thenReturn(topology);
        when(worker.committedEngineObservation()).thenReturn(observation);
        when(worker.pollHealth()).thenReturn(Mockito.mock(WorkerStatus.PollHealth.class));
        when(topology.ip()).thenReturn("10.0.0.1");
        when(observation.role()).thenReturn(RoleType.DECODE);
        when(observation.availableConcurrency()).thenReturn(null);
        when(observation.totalKvCacheTokens()).thenReturn(100L);
        when(observation.availableKvCacheTokens()).thenReturn(40L, 20L);

        reporter.reportStatusCheckerSuccess("model", worker, null, 1, 0);
        reporter.reportStatusCheckerSuccess("model", worker, null, 2, 1);

        FlexMetricTags engine = FlexMetricTags.of(
                "model", "model", "engineIp", "10.0.0.1", "role", "DECODE");
        FlexMetricTags role = FlexMetricTags.of("model", "model", "role", "DECODE");
        verify(monitor).report(CACHE_AVAILABLE_KV_CACHE_TOKENS, engine, 40.0);
        verify(monitor).report(CACHE_AVAILABLE_KV_CACHE_TOKENS, engine, 20.0);
        verify(monitor).report(CACHE_USED_KV_CACHE_TOKENS, engine, 60.0);
        verify(monitor).report(CACHE_USED_KV_CACHE_TOKENS, engine, 80.0);
        verify(monitor, Mockito.times(2)).report(CACHE_TOTAL_KV_CACHE_TOKENS, role, 100.0);
        verify(monitor).report(CACHE_USED_KV_CACHE_RATIO, engine, 60.0);
        verify(monitor).report(CACHE_USED_KV_CACHE_RATIO, engine, 80.0);
    }

    @Test
    void prefillStatusSuccessReportsNoKvCapacity() {
        WorkerStatus worker = Mockito.mock(WorkerStatus.class);
        WorkerStatus.TopologySnapshot topology =
                Mockito.mock(WorkerStatus.TopologySnapshot.class);
        WorkerStatus.EngineObservation observation =
                Mockito.mock(WorkerStatus.EngineObservation.class);
        when(worker.topologySnapshot()).thenReturn(topology);
        when(worker.committedEngineObservation()).thenReturn(observation);
        when(worker.pollHealth()).thenReturn(Mockito.mock(WorkerStatus.PollHealth.class));
        when(topology.ip()).thenReturn("10.0.0.1");
        when(observation.role()).thenReturn(RoleType.PREFILL);
        when(observation.availableConcurrency()).thenReturn(null);

        reporter.reportStatusCheckerSuccess("model", worker, null, 1, 0);

        FlexMetricTags engine = FlexMetricTags.of(
                "model", "model", "engineIp", "10.0.0.1", "role", "PREFILL");
        verify(monitor).report(ENGINE_RUNNING_TASK_INFO_SIZE, engine, 1);
        verify(monitor).report(ENGINE_FINISHED_TASK_LIST_SIZE, engine, 0);
        verifyNoMoreInteractions(monitor);
    }

    private void reportFailure(boolean cache, BalanceStatusEnum status, GrpcFailure.Analysis failure) {
        if (cache) {
            reporter.reportCacheStatusCheckerFail("test-model", status, RoleType.PREFILL,
                    Stage.STATUS_PROJECTION_FAILED, "STATUS_PROJECTION_FAILED", failure);
        } else {
            reporter.reportStatusCheckerFail("test-model", status, RoleType.PREFILL,
                    Stage.STATUS_PROJECTION_FAILED, "STATUS_PROJECTION_FAILED", failure);
        }
    }
}
