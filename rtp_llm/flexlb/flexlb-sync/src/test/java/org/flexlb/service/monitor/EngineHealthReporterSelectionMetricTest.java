package org.flexlb.service.monitor;

import io.netty.channel.EventLoopGroup;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.master.EndpointTrace.Stage;
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
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import reactor.netty.resources.LoopResources;

import static org.flexlb.constant.MetricConstant.PREFILL_SELECTED_ESTIMATED_TTFT_MS;
import static org.flexlb.constant.MetricConstant.PREFILL_SELECTED_EXECUTION_TIME_MS;
import static org.flexlb.constant.MetricConstant.CACHE_STATUS_CHECK_FAIL;
import static org.flexlb.constant.MetricConstant.ENGINE_STATUS_CHECK_FAIL;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

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

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
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

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
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
