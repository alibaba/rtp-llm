package org.flexlb.cache.monitor;

import org.flexlb.cache.core.RecentCacheKeyWindow;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.lang.reflect.Field;

import static org.flexlb.constant.MetricConstant.CACHE_AFFINITY_DECISION;
import static org.flexlb.constant.MetricConstant.CACHE_CLUSTER_RECENT_HISTORY_COVERAGE_MS;
import static org.flexlb.constant.MetricConstant.CACHE_CLUSTER_RECENT_HISTORY_READY;
import static org.flexlb.constant.MetricConstant.CACHE_CLUSTER_RECENT_HIT_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_CLUSTER_RECENT_INPUT_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_ENGINE_LOCAL_BYTES;
import static org.flexlb.constant.MetricConstant.CACHE_ENGINE_LOCAL_COUNT;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_COMPARISON_KVCM_GLOBAL_MATCH_DELTA_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_KVCM_SELECTED_GLOBAL_MATCH_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_KVCM_SELECTED_INPUT_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_KVCM_SELECTED_LOCAL_MATCH_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_LOCAL_STANDBY_BLOCK_SIZE;
import static org.flexlb.constant.MetricConstant.CACHE_LOCAL_STANDBY_CAPACITY_REJECTED_QPS;
import static org.flexlb.constant.MetricConstant.CACHE_LOCAL_STANDBY_MAPPING_COUNT;
import static org.flexlb.constant.MetricConstant.CACHE_RECENT_KEY_HIT_COUNT;
import static org.flexlb.constant.MetricConstant.CACHE_RECENT_KEY_TOTAL_COUNT;
import static org.flexlb.constant.MetricConstant.CACHE_ROUTING_CANDIDATE_MAX_HIT_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_ROUTING_SELECTED_MATCH_HIT_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_ROUTING_SELECTED_MATCH_TOTAL_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_THEORY_HIT_COUNT;
import static org.flexlb.constant.MetricConstant.CACHE_THEORY_HIT_RATIO;
import static org.flexlb.constant.MetricConstant.CACHE_THEORY_TOTAL_COUNT;
import static org.junit.jupiter.api.Assertions.fail;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

/**
 * Verifies the metric series, tags, and values emitted by the cache reporter.
 */
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
    void reportsKvcmGlobalCountersAndOnlyComparesAvailablePredictions() {
        reporter.init();
        verify(monitor).register(CACHE_KVCM_SELECTED_GLOBAL_MATCH_TOKENS, FlexMetricType.COUNTER);
        verify(monitor).register(CACHE_HIT_COMPARISON_KVCM_GLOBAL_MATCH_DELTA_TOKENS,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        String address = "127.0.0.1:8080";
        FlexMetricTags selected = FlexMetricTags.of("role", "PREFILL", "engineIp", address);
        reporter.reportKvcmSelectedMatch(RoleType.PREFILL, address, 16L, 48L, 64L);
        verify(monitor).report(CACHE_KVCM_SELECTED_LOCAL_MATCH_TOKENS, selected, 16L);
        verify(monitor).report(CACHE_KVCM_SELECTED_GLOBAL_MATCH_TOKENS, selected, 48L);
        verify(monitor).report(CACHE_KVCM_SELECTED_INPUT_TOKENS, selected, 64L);

        FlexMetricTags comparison = FlexMetricTags.of("role", "PREFILL", "engineIp", address,
                "group", "group", "taskState", "FINISHED", "cacheMatchSource", "KVCM");
        reporter.reportCacheHitComparison(RoleType.PREFILL, address, "group", "FINISHED", 64L, 32L, 16L, -1L, null);
        verify(monitor, never()).report(eq(CACHE_HIT_COMPARISON_KVCM_GLOBAL_MATCH_DELTA_TOKENS),
                eq(comparison), anyDouble());
        reporter.reportCacheHitComparison(RoleType.PREFILL, address, "group", "FINISHED", 64L, 32L, 16L, 48L, null);
        verify(monitor).report(CACHE_HIT_COMPARISON_KVCM_GLOBAL_MATCH_DELTA_TOKENS, comparison, -16L);
    }

    @Test
    void should_register_and_report_standby_capacity_metrics() {
        reporter.init();
        verify(monitor).register(CACHE_LOCAL_STANDBY_BLOCK_SIZE, FlexMetricType.GAUGE);
        verify(monitor).register(CACHE_LOCAL_STANDBY_CAPACITY_REJECTED_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        verify(monitor).register(CACHE_LOCAL_STANDBY_MAPPING_COUNT, FlexMetricType.GAUGE);

        reporter.reportLocalStandbyBlockSize(RoleType.PREFILL, 16L);
        reporter.reportLocalStandbyCapacityRejected();
        reporter.reportLocalStandbyMappingCount(2L);
        verify(monitor).report(CACHE_LOCAL_STANDBY_BLOCK_SIZE, FlexMetricTags.of("role", "PREFILL"), 16L);
        verify(monitor).report(CACHE_LOCAL_STANDBY_CAPACITY_REJECTED_QPS, 1.0);
        verify(monitor).report(CACHE_LOCAL_STANDBY_MAPPING_COUNT, 2L);
    }

    @Test
    void should_register_recent_cache_key_metrics_as_visible_series() {
        reporter.init();

        verify(monitor).register(CACHE_RECENT_KEY_HIT_COUNT, FlexMetricType.GAUGE);
        verify(monitor).register(CACHE_RECENT_KEY_TOTAL_COUNT, FlexMetricType.GAUGE);
        verify(monitor).register(CACHE_CLUSTER_RECENT_HIT_TOKENS, FlexMetricType.QPS);
        verify(monitor).register(CACHE_CLUSTER_RECENT_INPUT_TOKENS, FlexMetricType.QPS);
        verify(monitor).register(CACHE_CLUSTER_RECENT_HISTORY_COVERAGE_MS, FlexMetricType.GAUGE);
        verify(monitor).register(CACHE_CLUSTER_RECENT_HISTORY_READY, FlexMetricType.GAUGE);
        verify(monitor).register(CACHE_THEORY_HIT_COUNT, FlexMetricType.GAUGE);
        verify(monitor).register(CACHE_THEORY_TOTAL_COUNT, FlexMetricType.GAUGE);
        verify(monitor).register(CACHE_THEORY_HIT_RATIO, FlexMetricType.GAUGE);
        verify(monitor).register(CACHE_ROUTING_SELECTED_MATCH_HIT_TOKENS, FlexMetricType.QPS);
        verify(monitor).register(CACHE_ROUTING_SELECTED_MATCH_TOTAL_TOKENS, FlexMetricType.QPS);
        verify(monitor).register(CACHE_ROUTING_CANDIDATE_MAX_HIT_TOKENS, FlexMetricType.QPS);
        verify(monitor).register(CACHE_AFFINITY_DECISION, FlexMetricType.QPS);
    }

    @Test
    void should_not_publish_a_cluster_ratio_from_partial_history() {
        RecentCacheKeyWindow.Snapshot snapshot = mock(RecentCacheKeyWindow.Snapshot.class);
        when(snapshot.getTimeWindowMs()).thenReturn(1800000L);
        when(snapshot.getHistoryCoverageMs()).thenReturn(1000L);
        reporter.reportClusterRecentCacheKeyHitMetrics(snapshot, 0L, 216513L);
        FlexMetricTags tags = FlexMetricTags.of("timeWindowMs", "1800000", "isMaster", "true");
        verify(monitor).report(CACHE_CLUSTER_RECENT_HISTORY_COVERAGE_MS, tags, 1000L);
        verify(monitor).report(CACHE_CLUSTER_RECENT_HISTORY_READY, tags, 0L);
        verifyNoMoreInteractions(monitor);
    }

    @Test
    void should_report_additive_cluster_token_counts_including_cold_requests() {
        RecentCacheKeyWindow.Snapshot snapshot = mock(RecentCacheKeyWindow.Snapshot.class);
        when(snapshot.getTimeWindowMs()).thenReturn(1800000L);
        when(snapshot.getHistoryCoverageMs()).thenReturn(1800000L);
        when(snapshot.isHistoryComplete()).thenReturn(true);
        reporter.reportClusterRecentCacheKeyHitMetrics(snapshot, 0L, 300L);
        reporter.reportClusterRecentCacheKeyHitMetrics(snapshot, 216064L, 216513L);
        FlexMetricTags tags = FlexMetricTags.of("timeWindowMs", "1800000", "isMaster", "true");
        verify(monitor).report(CACHE_CLUSTER_RECENT_HIT_TOKENS, tags, 0L);
        verify(monitor).report(CACHE_CLUSTER_RECENT_INPUT_TOKENS, tags, 300L);
        verify(monitor).report(CACHE_CLUSTER_RECENT_HIT_TOKENS, tags, 216064L);
        verify(monitor).report(CACHE_CLUSTER_RECENT_INPUT_TOKENS, tags, 216513L);
    }

    @Test
    void should_report_cache_affinity_decision() {
        reporter.reportCacheAffinityDecision(
                RoleType.PREFILL, "10.0.0.1", "CACHE_LEADER");

        FlexMetricTags tags = FlexMetricTags.of(
                "role", "PREFILL",
                "engineIp", "10.0.0.1",
                "decision", "CACHE_LEADER");
        verify(monitor).report(CACHE_AFFINITY_DECISION, tags, 1.0);
    }

    @Test
    void should_report_engine_local_metrics_without_engine_ip_port() {
        reporter.reportEngineLocalMetrics("10.0.0.1", "PREFILL", 2);

        FlexMetricTags tags = FlexMetricTags.of("engineIp", "10.0.0.1", "role", "PREFILL");
        verify(monitor).report(CACHE_ENGINE_LOCAL_COUNT, tags, 2);
        verify(monitor).report(CACHE_ENGINE_LOCAL_BYTES, tags, 272L);
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
        reporter.reportRoutingSelectedCacheMatchMetrics(RoleType.PREFILL, 128L, 1024L);
        reporter.reportRoutingCandidateMaxCacheMatchMetrics(RoleType.PREFILL, 256L);

        FlexMetricTags roleTags = FlexMetricTags.of("role", RoleType.PREFILL.name());
        verify(monitor).report(CACHE_ROUTING_SELECTED_MATCH_HIT_TOKENS, roleTags, 128L);
        verify(monitor).report(CACHE_ROUTING_SELECTED_MATCH_TOTAL_TOKENS, roleTags, 1024L);
        verify(monitor).report(CACHE_ROUTING_CANDIDATE_MAX_HIT_TOKENS, roleTags, 256L);
    }
}
