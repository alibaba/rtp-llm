package org.flexlb.cache.monitor;

import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.lang.reflect.Field;

import static org.flexlb.constant.MetricConstant.CACHE_AFFINITY_DECISION;
import static org.flexlb.constant.MetricConstant.CACHE_ENGINE_LOCAL_BYTES;
import static org.flexlb.constant.MetricConstant.CACHE_ENGINE_LOCAL_COUNT;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_COUNT;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_RATIO;
import static org.flexlb.constant.MetricConstant.CACHE_RECENT_KEY_HIT_COUNT;
import static org.flexlb.constant.MetricConstant.CACHE_RECENT_KEY_TOTAL_COUNT;
import static org.flexlb.constant.MetricConstant.CACHE_REQUEST_TOTAL;
import static org.flexlb.constant.MetricConstant.CACHE_ROUTING_CANDIDATE_MAX_HIT_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_ROUTING_SELECTED_MATCH_HIT_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_ROUTING_SELECTED_MATCH_TOTAL_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_THEORY_HIT_COUNT;
import static org.flexlb.constant.MetricConstant.CACHE_THEORY_HIT_RATIO;
import static org.flexlb.constant.MetricConstant.CACHE_THEORY_TOTAL_COUNT;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.fail;
import static org.mockito.Mockito.clearInvocations;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;

@ExtendWith(MockitoExtension.class)
class CacheMetricsReporterTest {

    @Mock private FlexMonitor monitor;

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

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void brokenRegistrationAndReportDoNotStopOtherCacheMetrics(boolean error) {
        Throwable failure =
                error
                        ? new AssertionError("monitor failed")
                        : new IllegalStateException("monitor failed");
        doThrow(failure).when(monitor).register(CACHE_ENGINE_LOCAL_COUNT, FlexMetricType.GAUGE);
        assertDoesNotThrow(reporter::init);
        verify(monitor).register(CACHE_ENGINE_LOCAL_BYTES, FlexMetricType.GAUGE);
        var tags =
                FlexMetricTags.of("engineIp", "10.0.0.1", "enginePort", "8080", "role", "PREFILL");
        doThrow(failure).when(monitor).report(CACHE_ENGINE_LOCAL_COUNT, tags, 2);
        assertDoesNotThrow(() -> reporter.reportEngineLocalMetrics("10.0.0.1:8080", "PREFILL", 2));
        verify(monitor).report(CACHE_ENGINE_LOCAL_BYTES, tags, 272L);
    }

    @Test
    void should_register_recent_cache_key_metrics_as_visible_series() {
        reporter.init();

        verify(monitor).register(CACHE_RECENT_KEY_HIT_COUNT, FlexMetricType.GAUGE);
        verify(monitor).register(CACHE_RECENT_KEY_TOTAL_COUNT, FlexMetricType.GAUGE);
        verify(monitor).register(CACHE_THEORY_HIT_COUNT, FlexMetricType.GAUGE);
        verify(monitor).register(CACHE_THEORY_TOTAL_COUNT, FlexMetricType.GAUGE);
        verify(monitor).register(CACHE_THEORY_HIT_RATIO, FlexMetricType.GAUGE);
        verify(monitor).register(CACHE_ROUTING_SELECTED_MATCH_HIT_TOKENS, FlexMetricType.QPS);
        verify(monitor).register(CACHE_ROUTING_SELECTED_MATCH_TOTAL_TOKENS, FlexMetricType.QPS);
        verify(monitor).register(CACHE_ROUTING_CANDIDATE_MAX_HIT_TOKENS, FlexMetricType.QPS);
        verify(monitor).register(CACHE_AFFINITY_DECISION, FlexMetricType.QPS);
    }

    @Test
    void should_report_cache_affinity_decision() {
        reporter.reportCacheAffinityDecision(RoleType.PREFILL, "10.0.0.1", "CACHE_LEADER");

        FlexMetricTags tags =
                FlexMetricTags.of(
                        "role", "PREFILL",
                        "engineIp", "10.0.0.1",
                        "decision", "CACHE_LEADER");
        verify(monitor).report(CACHE_AFFINITY_DECISION, tags, 1.0);
    }

    @ParameterizedTest
    @CsvSource({"10.0.0.1:8080,10.0.0.1", "::1:8080,::1", "[::1]:8080,::1"})
    void should_report_engine_local_metrics_with_exact_worker_address(String address, String ip) {
        reporter.reportEngineLocalMetrics(address, "PREFILL", 2);

        FlexMetricTags tags =
                FlexMetricTags.of("engineIp", ip, "enginePort", "8080", "role", "PREFILL");
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
        reporter.reportSelectionCache(RoleType.PREFILL, 1024L, 64L, 128L, 256L);

        FlexMetricTags roleTags = FlexMetricTags.of("role", RoleType.PREFILL.name());
        verify(monitor).report(CACHE_ROUTING_SELECTED_MATCH_HIT_TOKENS, roleTags, 128L);
        verify(monitor).report(CACHE_ROUTING_SELECTED_MATCH_TOTAL_TOKENS, roleTags, 1024L);
        verify(monitor).report(CACHE_ROUTING_CANDIDATE_MAX_HIT_TOKENS, roleTags, 256L);
    }

    @ParameterizedTest
    @ValueSource(longs = {-1L, 0L, 1024L})
    void selectionReportsEffectiveAndMatchedTokensWithTheOriginalDenominator(long tokens) {
        reporter.reportSelectionCache(RoleType.PREFILL, tokens, 64L, 128L, 256L);

        var tags = FlexMetricTags.of("role", "PREFILL");
        verify(monitor).report(CACHE_HIT_COUNT, tags, 64L);
        verify(monitor).report(CACHE_HIT_RATIO, tags, tokens > 0L ? 64.0 / tokens : 0.0);
        verify(monitor).report(CACHE_REQUEST_TOTAL, tags, 1.0);
        if (tokens > 0L) {
            verify(monitor).report(CACHE_ROUTING_SELECTED_MATCH_HIT_TOKENS, tags, 128L);
            verify(monitor).report(CACHE_ROUTING_SELECTED_MATCH_TOTAL_TOKENS, tags, tokens);
        }
        verify(monitor).report(CACHE_ROUTING_CANDIDATE_MAX_HIT_TOKENS, tags, 256L);
        verifyNoMoreInteractions(monitor);
    }

    @ParameterizedTest
    @ValueSource(
            strings = {
                CACHE_HIT_COUNT,
                CACHE_ROUTING_SELECTED_MATCH_HIT_TOKENS,
                CACHE_ROUTING_CANDIDATE_MAX_HIT_TOKENS
            })
    void failedMetricDoesNotPreventTheOtherSelectionMetrics(String failedMetric) {
        reporter.init();
        clearInvocations(monitor);
        var tags = FlexMetricTags.of("role", "PREFILL");
        doThrow(new IllegalStateException("monitor failed"))
                .when(monitor)
                .report(failedMetric, tags, 0.0);

        assertDoesNotThrow(
                () -> reporter.reportSelectionCache(RoleType.PREFILL, 1024L, 0L, 0L, 0L));
        verify(monitor).report(CACHE_HIT_COUNT, tags, 0.0);
        verify(monitor).report(CACHE_HIT_RATIO, tags, 0.0);
        verify(monitor).report(CACHE_REQUEST_TOTAL, tags, 1.0);
        verify(monitor).report(CACHE_ROUTING_SELECTED_MATCH_HIT_TOKENS, tags, 0.0);
        verify(monitor).report(CACHE_ROUTING_SELECTED_MATCH_TOTAL_TOKENS, tags, 1024L);
        verify(monitor).report(CACHE_ROUTING_CANDIDATE_MAX_HIT_TOKENS, tags, 0.0);
        verifyNoMoreInteractions(monitor);
    }
}
