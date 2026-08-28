package org.flexlb.service.monitor;

import io.netty.channel.EventLoopGroup;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
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

    private EngineHealthReporter reporter;

    @BeforeEach
    void setUp() {
        when(loopResources.onServer(true)).thenReturn(serverWorker);
        when(loopResources.onServerSelect(true)).thenReturn(serverSelector);
        when(engineGrpcClient.getEventLoopGroup()).thenReturn(grpcEventLoop);
        reporter = new EngineHealthReporter(
                monitor, cacheMetricsReporter, engineGrpcClient, loopResources,
                workerDirectory);
    }

    @Test
    void registersSelectedPrefillEstimateMetricsAsTimers() {
        reporter.init();

        verify(monitor).register(PREFILL_SELECTED_ESTIMATED_TTFT_MS,
                FlexMetricType.TIMER, FlexPriorityType.PRECISE);
        verify(monitor).register(PREFILL_SELECTED_EXECUTION_TIME_MS,
                FlexMetricType.TIMER, FlexPriorityType.PRECISE);
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
}
