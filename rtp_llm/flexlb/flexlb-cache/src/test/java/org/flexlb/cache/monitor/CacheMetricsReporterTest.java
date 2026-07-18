package org.flexlb.cache.monitor;

import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.lang.reflect.Field;

import static org.flexlb.constant.MetricConstant.CACHE_RECENT_KEY_HIT_COUNT;
import static org.flexlb.constant.MetricConstant.CACHE_RECENT_KEY_TOTAL_COUNT;
import static org.flexlb.constant.MetricConstant.CACHE_ROUTING_CANDIDATE_MATCH_HIT_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_ROUTING_CANDIDATE_MATCH_TOTAL_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_ROUTING_SELECTED_MATCH_HIT_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_ROUTING_SELECTED_MATCH_TOTAL_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_THEORY_HIT_COUNT;
import static org.flexlb.constant.MetricConstant.CACHE_THEORY_HIT_RATIO;
import static org.flexlb.constant.MetricConstant.CACHE_THEORY_TOTAL_COUNT;
import static org.junit.jupiter.api.Assertions.fail;
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
        verify(monitor).register(CACHE_THEORY_HIT_COUNT, FlexMetricType.GAUGE);
        verify(monitor).register(CACHE_THEORY_TOTAL_COUNT, FlexMetricType.GAUGE);
        verify(monitor).register(CACHE_THEORY_HIT_RATIO, FlexMetricType.GAUGE);
        verify(monitor).register(CACHE_ROUTING_CANDIDATE_MATCH_HIT_TOKENS, FlexMetricType.GAUGE);
        verify(monitor).register(CACHE_ROUTING_CANDIDATE_MATCH_TOTAL_TOKENS, FlexMetricType.GAUGE);
        verify(monitor).register(CACHE_ROUTING_SELECTED_MATCH_HIT_TOKENS, FlexMetricType.GAUGE);
        verify(monitor).register(CACHE_ROUTING_SELECTED_MATCH_TOTAL_TOKENS, FlexMetricType.GAUGE);
    }

    @Test
    void should_report_zero_hit_token_request_as_visible_data_point() {
        reporter.reportRecentCacheKeyHitMetrics(1800000L, 0L, 300L);

        FlexMetricTags tags = FlexMetricTags.of("timeWindowMs", "1800000");
        verify(monitor).report(CACHE_RECENT_KEY_HIT_COUNT, tags, 0L);
        verify(monitor).report(CACHE_RECENT_KEY_TOTAL_COUNT, tags, 300L);
    }

    @Test
    void should_skip_empty_token_request() {
        reporter.reportRecentCacheKeyHitMetrics(1800000L, 0L, 0L);

        FlexMetricTags tags = FlexMetricTags.of("timeWindowMs", "1800000");
        verify(monitor, never()).report(CACHE_RECENT_KEY_HIT_COUNT, tags, 0L);
        verify(monitor, never()).report(CACHE_RECENT_KEY_TOTAL_COUNT, tags, 0L);
    }

    @Test
    void should_report_theory_cache_hit_metrics() {
        CacheHitTheoryStats stats = new CacheHitTheoryStats(() -> 0L);
        CacheHitTheoryStats.Snapshot snapshot = stats.record(2L, 4L, 0L);

        reporter.reportTheoryCacheHitMetrics(snapshot);

        FlexMetricTags allTags = FlexMetricTags.of("window", "all", "windowMs", "0");
        verify(monitor).report(CACHE_THEORY_HIT_COUNT, allTags, 2L);
        verify(monitor).report(CACHE_THEORY_TOTAL_COUNT, allTags, 4L);
        verify(monitor).report(CACHE_THEORY_HIT_RATIO, allTags, 0.5D);
    }

    @Test
    void should_report_routing_cache_match_token_metrics() {
        reporter.reportRoutingCandidateCacheMatchMetrics(RoleType.PREFILL, "127.0.0.1", "127.0.0.1:8080", 256L, 1024L);
        reporter.reportRoutingSelectedCacheMatchMetrics(RoleType.PREFILL, "127.0.0.2", "127.0.0.2:8080", 128L, 1024L);

        FlexMetricTags tags = FlexMetricTags.ofEngine(
                "127.0.0.1", "127.0.0.1:8080",
                "role", RoleType.PREFILL.name());
        verify(monitor).report(CACHE_ROUTING_CANDIDATE_MATCH_HIT_TOKENS, tags, 256L);
        verify(monitor).report(CACHE_ROUTING_CANDIDATE_MATCH_TOTAL_TOKENS, tags, 1024L);

        FlexMetricTags selectedTags = FlexMetricTags.ofEngine(
                "127.0.0.2", "127.0.0.2:8080",
                "role", RoleType.PREFILL.name());
        verify(monitor).report(CACHE_ROUTING_SELECTED_MATCH_HIT_TOKENS, selectedTags, 128L);
        verify(monitor).report(CACHE_ROUTING_SELECTED_MATCH_TOTAL_TOKENS, selectedTags, 1024L);
    }
}
