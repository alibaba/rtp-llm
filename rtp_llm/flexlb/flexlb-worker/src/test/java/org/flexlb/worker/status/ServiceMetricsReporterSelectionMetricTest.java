package org.flexlb.worker.status;

import org.flexlb.config.service.RoleType;
import org.flexlb.util.metrics.FlexMetricTags;
import org.flexlb.util.metrics.FlexMetricType;
import org.flexlb.util.metrics.FlexMonitor;
import org.flexlb.util.metrics.FlexPriorityType;
import org.flexlb.worker.endpoint.EndpointStats;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.flexlb.util.metrics.MetricConstant.CACHE_AVAILABLE_KV_CACHE_TOKENS;
import static org.flexlb.util.metrics.MetricConstant.CACHE_KEY_SIZE;
import static org.flexlb.util.metrics.MetricConstant.CACHE_STATUS_CHECK_SUCCESS_PERIOD;
import static org.flexlb.util.metrics.MetricConstant.CACHE_TOTAL_KV_CACHE_TOKENS;
import static org.flexlb.util.metrics.MetricConstant.CACHE_USED_KV_CACHE_RATIO;
import static org.flexlb.util.metrics.MetricConstant.CACHE_USED_KV_CACHE_TOKENS;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Checks cache metrics retain engine and role dimensions.
 */
@ExtendWith(MockitoExtension.class)
class ServiceMetricsReporterSelectionMetricTest {

    @Mock private FlexMonitor monitor;
    @Mock private EndpointStats workerDirectory;

    private ServiceMetricsReporter reporter;

    @BeforeEach
    void setUp() {
        reporter = new ServiceMetricsReporter(monitor, workerDirectory);
        verify(monitor)
                .register(
                        org.flexlb.util.metrics.MetricConstant.GRACEFUL_LIFECYCLE_EVENT,
                        FlexMetricType.GAUGE,
                        FlexPriorityType.PRECISE);
    }

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    void cacheMetricsKeepEngineAndRoleDimensions(boolean populated) {
        var worker = mock(WorkerStatus.class);
        var observation = mock(WorkerStatus.EngineObservation.class);
        when(worker.getIp()).thenReturn("10.0.0.1");
        when(worker.getRole()).thenReturn(RoleType.PREFILL);
        when(worker.committedEngineObservation()).thenReturn(observation);
        when(worker.pollHealth()).thenReturn(mock(WorkerStatus.PollHealth.class));
        if (populated) {
            when(observation.totalKvCacheTokens()).thenReturn(100L);
            when(observation.availableKvCacheTokens()).thenReturn(40L);
            when(worker.getCacheStatus())
                    .thenReturn(
                            org.flexlb.worker.cache.CacheStatus.builder().cacheKeySize(7L).build());
        }
        reporter.reportStatusCheckerSuccess(worker, java.util.OptionalLong.empty(), 0, 0);
        reporter.reportCacheStatusCheckerSuccess(worker, populated ? 20L : 0L);
        FlexMetricTags engine = FlexMetricTags.of("engineIp", "10.0.0.1", "role", "PREFILL");
        if (populated) {
            verify(monitor).report(CACHE_STATUS_CHECK_SUCCESS_PERIOD, engine, 20.0);
            verify(monitor).report(CACHE_KEY_SIZE, engine, 7.0);
            verify(monitor).report(CACHE_USED_KV_CACHE_RATIO, engine, 60.0);
            verify(monitor).report(CACHE_USED_KV_CACHE_TOKENS, engine, 60.0);
            verify(monitor).report(CACHE_AVAILABLE_KV_CACHE_TOKENS, engine, 40.0);
            verify(monitor).report(CACHE_TOTAL_KV_CACHE_TOKENS, engine, 100.0);
        } else {
            verify(monitor, org.mockito.Mockito.never())
                    .report(
                            org.mockito.ArgumentMatchers.eq(CACHE_TOTAL_KV_CACHE_TOKENS),
                            org.mockito.ArgumentMatchers.any(),
                            org.mockito.ArgumentMatchers.anyDouble());
        }
    }
}
