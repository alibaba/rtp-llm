package org.flexlb.cache.telemetry;

import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.flexlb.constant.MetricConstant.CACHE_UPDATE_ENGINE_BLOCK_CACHE_RT;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
class CacheMetricsReporterTest {

    @Mock
    private FlexMonitor monitor;

    @InjectMocks
    private CacheMetricsReporter reporter;

    @Test
    void reportsCacheUpdateLatencyWithIndexedEngineIp() {
        reporter.reportUpdateEngineBlockCacheRT("10.0.0.8@1", "PREFILL", 0L, "1");

        verify(monitor).report(
                eq(CACHE_UPDATE_ENGINE_BLOCK_CACHE_RT),
                eq(FlexMetricTags.of("engineIp", "10.0.0.8@1", "role", "PREFILL", "success", "1")),
                anyDouble());
    }
}
