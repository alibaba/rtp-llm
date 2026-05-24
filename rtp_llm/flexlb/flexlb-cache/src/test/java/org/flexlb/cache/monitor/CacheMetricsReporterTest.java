package org.flexlb.cache.monitor;

import org.flexlb.enums.FlexMetricType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.lang.reflect.Field;

import static org.flexlb.constant.MetricConstant.CACHE_RECENT_KEY_EMPTY_REQUEST_COUNT;
import static org.flexlb.constant.MetricConstant.CACHE_RECENT_KEY_HIT_COUNT;
import static org.flexlb.constant.MetricConstant.CACHE_RECENT_KEY_HIT_RATIO;
import static org.flexlb.constant.MetricConstant.CACHE_RECENT_KEY_REQUEST_COUNT;
import static org.flexlb.constant.MetricConstant.CACHE_RECENT_KEY_TOTAL_COUNT;
import static org.junit.jupiter.api.Assertions.fail;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
class CacheMetricsReporterTest {

    @Mock
    private FlexMonitor monitor;

    private CacheMetricsReporter reporter;

    @BeforeEach
    void setUp() {
        reporter = new CacheMetricsReporter();
        try {
            Field monitorField = CacheMetricsReporter.class.getDeclaredField("monitor");
            monitorField.setAccessible(true);
            monitorField.set(reporter, monitor);
        } catch (Exception e) {
            fail("Failed to inject monitor: " + e.getMessage());
        }
    }

    @Test
    void should_register_recent_cache_key_metrics_as_visible_series() {
        reporter.init();

        verify(monitor).register(CACHE_RECENT_KEY_HIT_COUNT, FlexMetricType.GAUGE);
        verify(monitor).register(CACHE_RECENT_KEY_TOTAL_COUNT, FlexMetricType.GAUGE);
        verify(monitor).register(CACHE_RECENT_KEY_HIT_RATIO, FlexMetricType.GAUGE);
        verify(monitor).register(CACHE_RECENT_KEY_REQUEST_COUNT, FlexMetricType.QPS);
        verify(monitor).register(CACHE_RECENT_KEY_EMPTY_REQUEST_COUNT, FlexMetricType.QPS);
    }

    @Test
    void should_report_zero_hit_request_as_visible_data_point() {
        reporter.reportRecentCacheKeyHitMetrics(1800000L, 0L, 3L, 0.0);

        FlexMetricTags tags = FlexMetricTags.of("timeWindowMs", "1800000");
        verify(monitor).report(CACHE_RECENT_KEY_REQUEST_COUNT, tags, 1.0);
        verify(monitor).report(CACHE_RECENT_KEY_HIT_COUNT, tags, 0L);
        verify(monitor).report(CACHE_RECENT_KEY_TOTAL_COUNT, tags, 3L);
        verify(monitor).report(CACHE_RECENT_KEY_HIT_RATIO, tags, 0.0);
        verify(monitor, never()).report(eq(CACHE_RECENT_KEY_EMPTY_REQUEST_COUNT), any(FlexMetricTags.class), any(Double.class));
    }

    @Test
    void should_report_empty_cache_key_request_for_diagnosis() {
        reporter.reportRecentCacheKeyHitMetrics(1800000L, 0L, 0L, 0.0);

        FlexMetricTags tags = FlexMetricTags.of("timeWindowMs", "1800000");
        verify(monitor).report(CACHE_RECENT_KEY_REQUEST_COUNT, tags, 1.0);
        verify(monitor).report(CACHE_RECENT_KEY_EMPTY_REQUEST_COUNT, tags, 1.0);
        verify(monitor).report(CACHE_RECENT_KEY_HIT_COUNT, tags, 0L);
        verify(monitor).report(CACHE_RECENT_KEY_TOTAL_COUNT, tags, 0L);
        verify(monitor).report(CACHE_RECENT_KEY_HIT_RATIO, tags, 0.0);
    }
}
