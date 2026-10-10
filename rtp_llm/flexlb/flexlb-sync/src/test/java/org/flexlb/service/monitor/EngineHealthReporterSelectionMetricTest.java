package org.flexlb.service.monitor;

import io.netty.channel.EventLoopGroup;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.CacheStatus;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import reactor.netty.resources.LoopResources;

import static org.flexlb.constant.MetricConstant.PREFILL_SELECTED_ESTIMATED_TTFT_MS;
import static org.flexlb.constant.MetricConstant.PREFILL_SELECTED_EXECUTION_TIME_MS;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.flexlb.constant.MetricConstant.CACHE_STATUS_CHECK_SUCCESS_PERIOD;
import static org.flexlb.constant.MetricConstant.CACHE_BLOCK_SIZE;
import static org.flexlb.constant.MetricConstant.CACHE_KEY_SIZE;
import static org.flexlb.constant.MetricConstant.CACHE_USED_KV_CACHE_RATIO;
import static org.flexlb.constant.MetricConstant.CACHE_USED_KV_CACHE_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_AVAILABLE_KV_CACHE_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_TOTAL_KV_CACHE_TOKENS;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class EngineHealthReporterSelectionMetricTest {

    @Mock
    private FlexMonitor monitor;
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
    private EndpointRegistry workerDirectory;

    private EngineHealthReporter reporter;

    @BeforeEach
    void setUp() {
        when(loopResources.onServer(true)).thenReturn(serverWorker);
        when(loopResources.onServerSelect(true)).thenReturn(serverSelector);
        when(engineGrpcClient.getEventLoopGroup()).thenReturn(grpcEventLoop);
        reporter = new EngineHealthReporter(
                monitor, engineGrpcClient, loopResources,
                workerDirectory);
    }

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    void cacheMetricsKeepEngineAndRoleDimensions(boolean populated) {
        WorkerStatus worker = mock(WorkerStatus.class);
        var topology = mock(WorkerStatus.TopologySnapshot.class);
        var observation = mock(WorkerStatus.EngineObservation.class);
        when(worker.topologySnapshot()).thenReturn(topology);
        when(worker.committedEngineObservation()).thenReturn(observation);
        when(topology.ip()).thenReturn("10.0.0.1");
        when(observation.role()).thenReturn(RoleType.PREFILL);
        if (populated) {
            CacheStatus cache = mock(CacheStatus.class);
            when(worker.getCacheStatus()).thenReturn(cache);
            when(cache.getBlockSize()).thenReturn(16L);
            when(cache.getCacheKeySize()).thenReturn(7L);
            when(observation.totalKvCacheTokens()).thenReturn(100L);
            when(observation.availableKvCacheTokens()).thenReturn(40L);
        }
        reporter.reportCacheStatusCheckerSuccess("model", worker, populated ? 20L : 0L);
        FlexMetricTags engine = FlexMetricTags.of("model", "model", "engineIp", "10.0.0.1", "role", "PREFILL");
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
}
