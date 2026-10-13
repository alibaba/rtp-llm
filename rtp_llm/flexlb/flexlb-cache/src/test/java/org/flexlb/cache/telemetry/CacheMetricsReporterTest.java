package org.flexlb.cache.telemetry;

import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.MicrometerFlexMonitor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.lang.reflect.Field;

import static org.flexlb.constant.MetricConstant.CACHE_AFFINITY_DECISION;
import static org.flexlb.constant.MetricConstant.CACHE_ENGINE_LOCAL_BYTES;
import static org.flexlb.constant.MetricConstant.CACHE_ENGINE_LOCAL_COUNT;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_COUNT;
import static org.flexlb.constant.MetricConstant.CACHE_INPUT_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_KVCM_PREDICTED_RATIO;
import static org.flexlb.constant.MetricConstant.CACHE_KVCM_PREDICTED_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_KVCM_SELECTED_GLOBAL_MATCH_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_KVCM_SELECTED_INPUT_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_KVCM_SELECTED_LOCAL_MATCH_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_LOCAL_STANDBY_PREDICTED_RATIO;
import static org.flexlb.constant.MetricConstant.CACHE_LOCAL_STANDBY_PREDICTED_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_ROUTING_CANDIDATE_MAX_HIT_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_ROUTING_SELECTED_MATCH_HIT_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_THEORY_HIT_COUNT;
import static org.flexlb.constant.MetricConstant.CACHE_THEORY_HIT_RATIO;
import static org.flexlb.constant.MetricConstant.CACHE_THEORY_TOTAL_COUNT;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;

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
    void should_register_theory_cache_hit_metrics_as_visible_series() {
        reporter.init();

        verify(monitor).register(CACHE_THEORY_HIT_COUNT, FlexMetricType.COUNTER);
        verify(monitor).register(CACHE_THEORY_TOTAL_COUNT, FlexMetricType.COUNTER);
        verify(monitor).register(CACHE_THEORY_HIT_RATIO, FlexMetricType.GAUGE);
        verify(monitor).register(CACHE_ROUTING_SELECTED_MATCH_HIT_TOKENS, FlexMetricType.QPS);
        verify(monitor).register(CACHE_ROUTING_CANDIDATE_MAX_HIT_TOKENS, FlexMetricType.QPS);
        verify(monitor).register(CACHE_AFFINITY_DECISION, FlexMetricType.QPS);
    }

    @Test
    void selectedKvcmMatchWithoutRoleDoesNotReport() {
        reporter.reportKvcmSelectedMatch(null, "10.0.0.1:8080", 20L, 30L, 100L);

        verifyNoInteractions(monitor);
    }

    @Test
    void localStandbyBlockSizeWithoutRoleDoesNotReport() {
        reporter.reportLocalStandbyBlockSize(null, 4096L);

        verifyNoInteractions(monitor);
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
    void should_report_engine_local_metrics_with_logical_worker_address() {
        reporter.reportEngineLocalMetrics("10.0.0.1:8080@0", "PREFILL", 2);

        FlexMetricTags tags = FlexMetricTags.of("engineIp", "10.0.0.1:8080@0", "role", "PREFILL");
        verify(monitor).report(CACHE_ENGINE_LOCAL_COUNT, tags, 2);
        verify(monitor).report(CACHE_ENGINE_LOCAL_BYTES, tags, 272L);
    }

    @Test
    void should_report_zero_hit_token_request_as_visible_data_point() {
        reporter.reportTheoryCacheHitMetrics(0L, 300L);

        FlexMetricTags tags = FlexMetricTags.of();
        verify(monitor).report(CACHE_THEORY_HIT_COUNT, tags, 0L);
        verify(monitor).report(CACHE_THEORY_TOTAL_COUNT, tags, 300L);
        verify(monitor).report(CACHE_THEORY_HIT_RATIO, tags, 0.0D);
    }

    @Test
    void should_report_cache_hit_and_input_tokens_as_counters_with_same_worker_tags() {
        reporter.init();
        reporter.reportCacheHitMetrics(RoleType.PREFILL, "10.0.0.1:8080@0", 20L, 100L, 0.2);
        reporter.reportCacheHitMetrics(RoleType.PREFILL, "10.0.0.1:8080@0", 0L, 200L, 0.0);

        FlexMetricTags tags = FlexMetricTags.of("role", "PREFILL", "engineIp", "10.0.0.1:8080@0");
        verify(monitor).register(CACHE_HIT_COUNT, FlexMetricType.COUNTER);
        verify(monitor).register(CACHE_INPUT_TOKENS, FlexMetricType.COUNTER);
        verify(monitor).report(CACHE_HIT_COUNT, tags, 20L);
        verify(monitor).report(CACHE_HIT_COUNT, tags, 0L);
        verify(monitor).report(CACHE_INPUT_TOKENS, tags, 100L);
        verify(monitor).report(CACHE_INPUT_TOKENS, tags, 200L);
    }

    @Test
    void zero_predictions_have_data_points_for_both_sources() {
        reporter.init();
        reporter.reportKvcmPrediction(RoleType.PREFILL, "10.0.0.1:8080@0", 0, 100);
        reporter.reportLocalStandbyPrediction(RoleType.PREFILL, "10.0.0.1:8080@0", 0, 100);

        FlexMetricTags tags = FlexMetricTags.of("role", "PREFILL", "engineIp", "10.0.0.1:8080@0");
        verify(monitor).register(CACHE_KVCM_PREDICTED_RATIO, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).register(CACHE_LOCAL_STANDBY_PREDICTED_RATIO,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).report(CACHE_KVCM_PREDICTED_TOKENS, tags, 0L);
        verify(monitor).report(CACHE_KVCM_PREDICTED_RATIO, tags, 0.0);
        verify(monitor).report(CACHE_LOCAL_STANDBY_PREDICTED_TOKENS, tags, 0L);
        verify(monitor).report(CACHE_LOCAL_STANDBY_PREDICTED_RATIO, tags, 0.0);
    }

    @Test
    void token_counters_accumulate_across_requests_instead_of_averaging_request_ratios() throws Exception {
        SimpleMeterRegistry registry = new SimpleMeterRegistry();
        try {
            Field monitorField = CacheMetricsReporter.class.getDeclaredField("monitor");
            monitorField.setAccessible(true);
            monitorField.set(reporter, new MicrometerFlexMonitor(registry));
            reporter.init();

            reporter.reportCacheHitMetrics(RoleType.PREFILL, "10.0.0.1:8080", 20L, 100L, 0.2);
            reporter.reportCacheHitMetrics(RoleType.PREFILL, "10.0.0.1:8080", 0L, 200L, 0.0);

            double hitTokens = registry.get("flexlb.app.cache.hit.count")
                    .tags("role", "PREFILL", "engineIp", "10.0.0.1:8080").counter().count();
            double inputTokens = registry.get("flexlb.app.cache.input.tokens")
                    .tags("role", "PREFILL", "engineIp", "10.0.0.1:8080").counter().count();
            assertEquals(20.0, hitTokens);
            assertEquals(300.0, inputTokens);
            assertEquals(20.0 / 300.0, hitTokens / inputTokens);
        } finally {
            registry.close();
        }
    }

    @Test
    void should_skip_empty_token_request() {
        reporter.reportTheoryCacheHitMetrics(0L, 0L);

        FlexMetricTags tags = FlexMetricTags.of();
        verify(monitor, never()).report(CACHE_THEORY_HIT_COUNT, tags, 0L);
        verify(monitor, never()).report(CACHE_THEORY_TOTAL_COUNT, tags, 0L);
    }

    @Test
    void should_report_theory_cache_hit_metrics() {
        reporter.init();
        reporter.reportTheoryCacheHitMetrics(2L, 4L);
        reporter.reportTheoryCacheHitMetrics(0L, 16L);

        FlexMetricTags allTags = FlexMetricTags.of();
        verify(monitor).report(CACHE_THEORY_HIT_COUNT, allTags, 2L);
        verify(monitor).report(CACHE_THEORY_HIT_COUNT, allTags, 0L);
        verify(monitor).report(CACHE_THEORY_TOTAL_COUNT, allTags, 4L);
        verify(monitor).report(CACHE_THEORY_TOTAL_COUNT, allTags, 16L);
        verify(monitor).report(CACHE_THEORY_HIT_RATIO, allTags, 0.5D);
        verify(monitor).report(CACHE_THEORY_HIT_RATIO, allTags, 0.0D);
    }

    @Test
    void should_report_kvcm_selected_match_with_its_input_tokens() {
        reporter.init();
        reporter.reportKvcmSelectedMatch(RoleType.PREFILL, "10.0.0.1:8080", 64L, 128L, 256L);
        reporter.reportKvcmSelectedMatch(RoleType.PREFILL, "10.0.0.1:8080", 0L, 0L, 512L);

        FlexMetricTags tags = FlexMetricTags.of("role", "PREFILL", "engineIp", "10.0.0.1:8080");
        verify(monitor).register(CACHE_KVCM_SELECTED_LOCAL_MATCH_TOKENS, FlexMetricType.COUNTER);
        verify(monitor).register(CACHE_KVCM_SELECTED_GLOBAL_MATCH_TOKENS, FlexMetricType.COUNTER);
        verify(monitor).register(CACHE_KVCM_SELECTED_INPUT_TOKENS, FlexMetricType.COUNTER);
        verify(monitor).report(CACHE_KVCM_SELECTED_LOCAL_MATCH_TOKENS, tags, 64L);
        verify(monitor).report(CACHE_KVCM_SELECTED_GLOBAL_MATCH_TOKENS, tags, 128L);
        verify(monitor).report(CACHE_KVCM_SELECTED_INPUT_TOKENS, tags, 256L);
        verify(monitor).report(CACHE_KVCM_SELECTED_INPUT_TOKENS, tags, 512L);
    }

    @Test
    void should_report_routing_cache_match_token_metrics() {
        reporter.reportRoutingSelectedCacheMatchMetrics(RoleType.PREFILL, 128L);
        reporter.reportRoutingCandidateMaxCacheMatchMetrics(RoleType.PREFILL, 256L);

        FlexMetricTags roleTags = FlexMetricTags.of("role", RoleType.PREFILL.name());
        verify(monitor).report(CACHE_ROUTING_SELECTED_MATCH_HIT_TOKENS, roleTags, 128L);
        verify(monitor).report(CACHE_ROUTING_CANDIDATE_MAX_HIT_TOKENS, roleTags, 256L);
    }
}
