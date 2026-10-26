package org.flexlb.service.monitor;

import io.netty.channel.EventLoopGroup;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.client.EngineGrpcClient;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import reactor.netty.resources.LoopResources;

import static org.flexlb.constant.MetricConstant.CACHE_KEY_SIZE;
import static org.flexlb.constant.MetricConstant.CACHE_STATUS_CHECK_SUCCESS_PERIOD;
import static org.flexlb.constant.MetricConstant.PREFILL_SELECTED_ESTIMATED_TTFT_MS;
import static org.flexlb.constant.MetricConstant.PREFILL_SELECTED_EXECUTION_TIME_MS;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class EngineHealthReporterSelectionMetricTest {

    @Mock private FlexMonitor monitor;
    @Mock private EngineGrpcClient engineGrpcClient;
    @Mock private LoopResources loopResources;
    @Mock private EventLoopGroup serverWorker;
    @Mock private EventLoopGroup serverSelector;
    @Mock private EventLoopGroup grpcEventLoop;
    @Mock private EndpointRegistry workerDirectory;

    private EngineHealthReporter reporter;

    @BeforeEach
    void setUp() {
        when(loopResources.onServer(true)).thenReturn(serverWorker);
        when(loopResources.onServerSelect(true)).thenReturn(serverSelector);
        when(engineGrpcClient.getEventLoopGroup()).thenReturn(grpcEventLoop);
        reporter =
                new EngineHealthReporter(
                        monitor,
                        mock(CacheMetricsReporter.class),
                        engineGrpcClient,
                        loopResources,
                        workerDirectory);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void metricFailureCannotInterruptHealthMetricInitializationOrReporting(boolean error) {
        Throwable failure =
                error
                        ? new AssertionError("monitor failed")
                        : new IllegalStateException("monitor failed");
        doThrow(failure)
                .when(monitor)
                .register(
                        PREFILL_SELECTED_ESTIMATED_TTFT_MS,
                        FlexMetricType.GAUGE,
                        FlexPriorityType.PRECISE);
        assertDoesNotThrow(reporter::init);
        verify(monitor)
                .register(
                        PREFILL_SELECTED_EXECUTION_TIME_MS,
                        FlexMetricType.GAUGE,
                        FlexPriorityType.PRECISE);
        var tags =
                FlexMetricTags.of(
                        "engineIp", "10.0.0.1", "role", "PREFILL", "delivery_mode", "NON_BATCH");
        doThrow(failure).when(monitor).report(PREFILL_SELECTED_ESTIMATED_TTFT_MS, tags, 1250);
        assertDoesNotThrow(
                () ->
                        reporter.reportPrefillSelectedEstimates(
                                RoleType.PREFILL, "10.0.0.1", "NON_BATCH", 1250, 400));
        verify(monitor).report(PREFILL_SELECTED_EXECUTION_TIME_MS, tags, 400);
    }

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    void cacheMetricsKeepEngineAndRoleDimensions(boolean populated) {
        WorkerStatus worker = mock(WorkerStatus.class);
        org.mockito.Mockito.lenient().when(worker.getMetricIpPort()).thenReturn("10.0.0.1:8080@2");
        org.mockito.Mockito.lenient().when(worker.getRole()).thenReturn(RoleType.PREFILL);
        if (populated) {
            CacheStatus cache = mock(CacheStatus.class);
            when(worker.getCacheStatus()).thenReturn(cache);
            when(cache.getCacheKeySize()).thenReturn(7L);
        }
        reporter.reportCacheStatusCheckerSuccess(worker, populated ? 20L : 0L);
        FlexMetricTags engine = FlexMetricTags.of("engineIp", "10.0.0.1:8080@2", "role", "PREFILL");
        if (populated) {
            verify(monitor).report(CACHE_STATUS_CHECK_SUCCESS_PERIOD, engine, 20.0);
            verify(monitor).report(CACHE_KEY_SIZE, engine, 7.0);
        }
        verifyNoMoreInteractions(monitor);
    }

    @Test
    void registersSelectedPrefillEstimateMetricsAsGauges() {
        reporter.init();

        verify(monitor)
                .register(
                        PREFILL_SELECTED_ESTIMATED_TTFT_MS,
                        FlexMetricType.GAUGE,
                        FlexPriorityType.PRECISE);
        verify(monitor)
                .register(
                        PREFILL_SELECTED_EXECUTION_TIME_MS,
                        FlexMetricType.GAUGE,
                        FlexPriorityType.PRECISE);
    }

    @Test
    void reportsSelectedPrefillEstimatesWithDeliveryMode() {
        reporter.reportPrefillSelectedEstimates(
                RoleType.PREFILL, "10.0.0.1:8080@0", "NON_BATCH", 1_250L, 400L);

        FlexMetricTags tags =
                FlexMetricTags.of(
                        "engineIp", "10.0.0.1:8080@0",
                        "role", "PREFILL",
                        "delivery_mode", "NON_BATCH");
        verify(monitor).report(PREFILL_SELECTED_ESTIMATED_TTFT_MS, tags, 1_250.0);
        verify(monitor).report(PREFILL_SELECTED_EXECUTION_TIME_MS, tags, 400.0);
    }
}
