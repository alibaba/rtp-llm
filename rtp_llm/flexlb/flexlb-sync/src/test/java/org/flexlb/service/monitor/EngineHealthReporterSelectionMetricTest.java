package org.flexlb.service.monitor;

import io.netty.channel.EventLoopGroup;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import reactor.netty.resources.LoopResources;

import static org.flexlb.constant.MetricConstant.PREFILL_SELECTED_ESTIMATED_TTFT_MS;
import static org.flexlb.constant.MetricConstant.PREFILL_SELECTED_EXECUTION_TIME_MS;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class EngineHealthReporterSelectionMetricTest {

    private FlexMonitor monitor;
    private CacheMetricsReporter cacheMetricsReporter;
    private EngineHealthReporter reporter;

    @BeforeEach
    void setUp() {
        monitor = mock(FlexMonitor.class);
        cacheMetricsReporter = mock(CacheMetricsReporter.class);
        EngineGrpcClient engineGrpcClient = mock(EngineGrpcClient.class);
        LoopResources loopResources = mock(LoopResources.class);
        when(loopResources.onServer(true)).thenReturn(mock(EventLoopGroup.class));
        when(loopResources.onServerSelect(true)).thenReturn(mock(EventLoopGroup.class));
        when(engineGrpcClient.getEventLoopGroup()).thenReturn(mock(EventLoopGroup.class));
        reporter = new EngineHealthReporter(
                monitor, cacheMetricsReporter, engineGrpcClient, loopResources);
    }

    @Test
    void registersAndReportsSelectedPrefillEstimateMetrics() {
        reporter.init();

        verify(monitor).register(PREFILL_SELECTED_ESTIMATED_TTFT_MS,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).register(PREFILL_SELECTED_EXECUTION_TIME_MS,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

        reporter.reportPrefillSelectedEstimates(
                RoleType.PREFILL, "10.0.0.1", "NON_BATCH", 1_250L, 400L);
        FlexMetricTags tags = FlexMetricTags.of(
                "engineIp", "10.0.0.1",
                "role", "PREFILL",
                "delivery_mode", "NON_BATCH");
        verify(monitor).report(PREFILL_SELECTED_ESTIMATED_TTFT_MS, tags, 1_250.0);
        verify(monitor).report(PREFILL_SELECTED_EXECUTION_TIME_MS, tags, 400.0);
    }

    @Test
    void delegatesCacheAffinityDecisionMetrics() {
        reporter.reportCacheAffinityDecision(
                RoleType.PREFILL, "10.0.0.1", "CACHE_LEADER");

        verify(cacheMetricsReporter).reportCacheAffinityDecision(
                RoleType.PREFILL, "10.0.0.1", "CACHE_LEADER");
    }
}
