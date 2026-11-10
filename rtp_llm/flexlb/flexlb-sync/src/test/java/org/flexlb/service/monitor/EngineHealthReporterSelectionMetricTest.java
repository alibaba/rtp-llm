package org.flexlb.service.monitor;

import org.flexlb.balance.endpoint.EndpointStats;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.flexlb.constant.MetricConstant.CACHE_AVAILABLE_KV_CACHE_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_KEY_SIZE;
import static org.flexlb.constant.MetricConstant.CACHE_STATUS_CHECK_SUCCESS_PERIOD;
import static org.flexlb.constant.MetricConstant.CACHE_TOTAL_KV_CACHE_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_USED_KV_CACHE_RATIO;
import static org.flexlb.constant.MetricConstant.CACHE_USED_KV_CACHE_TOKENS;
import static org.flexlb.constant.MetricConstant.PREFILL_SELECTED_ESTIMATED_TTFT_MS;
import static org.flexlb.constant.MetricConstant.PREFILL_SELECTED_EXECUTION_TIME_MS;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Checks selected Prefill estimate metrics and delivery-mode tags.
 */
@ExtendWith(MockitoExtension.class)
class EngineHealthReporterSelectionMetricTest {

    @Mock private FlexMonitor monitor;
    @Mock private EndpointStats workerDirectory;

    private EngineHealthReporter reporter;

    @BeforeEach
    void setUp() {
        reporter = new EngineHealthReporter(monitor, workerDirectory);
        verify(monitor)
                .register(
                        org.flexlb.constant.MetricConstant.GRACEFUL_LIFECYCLE_EVENT,
                        FlexMetricType.GAUGE,
                        FlexPriorityType.PRECISE);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void metricFailureCannotInterruptHealthMetricInitializationOrReporting(boolean error) {
        Throwable failure =
                error
                        ? new AssertionError("monitor failed")
                        : new IllegalStateException("monitor failed");
        doThrow(failure)
                .when(monitor)
                .register(
                        PREFILL_SELECTED_ESTIMATED_TTFT_MS,
                        FlexMetricType.GAUGE,
                        FlexPriorityType.PRECISE);
        assertDoesNotThrow(reporter::init);
        verify(monitor)
                .register(
                        PREFILL_SELECTED_EXECUTION_TIME_MS,
                        FlexMetricType.GAUGE,
                        FlexPriorityType.PRECISE);
        var tags =
                FlexMetricTags.of(
                        "engineIp", "10.0.0.1", "role", "PREFILL", "delivery_mode", "NON_BATCH");
        doThrow(failure).when(monitor).report(PREFILL_SELECTED_ESTIMATED_TTFT_MS, tags, 1250);
        assertDoesNotThrow(
                () ->
                        reporter.reportPrefillSelectedEstimates(
                                RoleType.PREFILL, "10.0.0.1", "NON_BATCH", 1250, 400));
        verify(monitor).report(PREFILL_SELECTED_EXECUTION_TIME_MS, tags, 400);
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
        when(observation.role()).thenReturn(RoleType.PREFILL);
        if (populated) {
            when(observation.totalKvCacheTokens()).thenReturn(100L);
            when(observation.availableKvCacheTokens()).thenReturn(40L);
            when(worker.getCacheStatus())
                    .thenReturn(
                            org.flexlb.dao.master.CacheStatus.builder().cacheKeySize(7L).build());
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

    @Test
    void registersSelectedPrefillEstimateMetricsAsGauges() {
        reporter.init();

        verify(monitor)
                .register(
                        PREFILL_SELECTED_ESTIMATED_TTFT_MS,
                        FlexMetricType.GAUGE,
                        FlexPriorityType.PRECISE);
        verify(monitor)
                .register(
                        PREFILL_SELECTED_EXECUTION_TIME_MS,
                        FlexMetricType.GAUGE,
                        FlexPriorityType.PRECISE);
    }

    @Test
    void reportsSelectedPrefillEstimatesWithDeliveryMode() {
        reporter.reportPrefillSelectedEstimates(
                RoleType.PREFILL, "10.0.0.1", "NON_BATCH", 1_250L, 400L);

        FlexMetricTags tags =
                FlexMetricTags.of(
                        "engineIp", "10.0.0.1",
                        "role", "PREFILL",
                        "delivery_mode", "NON_BATCH");
        verify(monitor).report(PREFILL_SELECTED_ESTIMATED_TTFT_MS, tags, 1_250.0);
        verify(monitor).report(PREFILL_SELECTED_EXECUTION_TIME_MS, tags, 400.0);
    }
}
