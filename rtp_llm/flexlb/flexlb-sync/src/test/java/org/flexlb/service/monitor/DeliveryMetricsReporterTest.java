package org.flexlb.service.monitor;

import io.micrometer.core.instrument.Meter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.config.MeterFilter;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
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
import org.springframework.context.annotation.AnnotationConfigApplicationContext;

import static org.flexlb.constant.MetricConstant.BATCHER_QUEUE_SIZE;
import static org.flexlb.constant.MetricConstant.DECODE_INFLIGHT_HARD_KV_RESERVED_TOKENS;
import static org.flexlb.constant.MetricConstant.DECODE_INFLIGHT_KV_RESERVED_TOKENS;
import static org.flexlb.constant.MetricConstant.DECODE_TOTAL_LOAD;
import static org.flexlb.constant.MetricConstant.DISPATCH_ACK_TIME_MS;
import static org.flexlb.constant.MetricConstant.DISPATCH_EXECUTOR_ACTIVE_THREADS;
import static org.flexlb.constant.MetricConstant.DISPATCH_EXECUTOR_COMPLETED_TASKS;
import static org.flexlb.constant.MetricConstant.DISPATCH_EXECUTOR_POOL_SIZE;
import static org.flexlb.constant.MetricConstant.DISPATCH_EXECUTOR_QUEUE_SIZE;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_DISPATCH_REASON;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_SELECT_DETAIL;
import static org.flexlb.constant.MetricConstant.INFLIGHT_MAX_AGE_MS;
import static org.flexlb.constant.MetricConstant.INFLIGHT_REQUEST_COUNT;
import static org.flexlb.constant.MetricConstant.INFLIGHT_TTL_EXPIRED_QPS;
import static org.flexlb.constant.MetricConstant.ROUTE_SUBMIT_TIME_MS;
import static org.flexlb.constant.MetricConstant.ROUTING_QUEUE_LENGTH;
import static org.flexlb.constant.MetricConstant.ROUTING_QUEUE_WAIT_TIME_MS;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class DeliveryMetricsReporterTest {

    @Mock private FlexMonitor monitor;

    private DeliveryMetricsReporter reporter;

    @BeforeEach
    void setUp() {
        reporter = new DeliveryMetricsReporter(monitor);
    }

    @Test
    void scheduledReportReadsStatsAndKeepsEndpointFailuresIndependent() {
        var endpoints = mock(org.flexlb.balance.endpoint.EndpointRegistry.class);
        var prefill = mock(org.flexlb.balance.endpoint.PrefillEndpoint.class);
        var decode = mock(org.flexlb.balance.endpoint.DecodeEndpoint.class);
        var requests = mock(RequestSchedulerReporter.class);
        when(endpoints.endpointSnapshot(org.flexlb.dao.route.RoleType.PREFILL))
                .thenReturn(java.util.Map.of("prefill", prefill));
        when(endpoints.endpointSnapshot(org.flexlb.dao.route.RoleType.DECODE))
                .thenReturn(java.util.Map.of("decode", decode));
        when(prefill.queueDepth()).thenReturn(7);
        doThrow(new IllegalStateException("isolated metric failure"))
                .when(prefill)
                .reportBatchMetrics(reporter);
        try (var context = new AnnotationConfigApplicationContext()) {
            context.registerBean(
                    org.flexlb.balance.scheduler.RequestRepository.class,
                    org.flexlb.balance.scheduler.RequestRepository::new);
            context.registerBean(
                    org.flexlb.balance.endpoint.EndpointRegistry.class, () -> endpoints);
            context.registerBean(RequestSchedulerReporter.class, () -> requests);
            context.registerBean(DeliveryMetricsReporter.class, () -> reporter);
            context.refresh();
            assertDoesNotThrow(reporter::report);
            verify(requests).reportPrefillQueueDepth("prefill", 7);
            verify(decode).reportBatchMetrics(reporter);
            verify(decode).reportAdmissionMetrics(requests);
        }
    }

    @Test
    void executorMetricsReadLiveValuesAndKeepCompletedTasksAsACounter() {
        var registry = new SimpleMeterRegistry();
        try {
            var stats = mock(org.flexlb.metric.ExecutorStats.class);
            when(stats.activeThreads()).thenReturn(2);
            when(stats.queuedTasks()).thenReturn(3);
            when(stats.poolSize()).thenReturn(4);
            when(stats.completedTasks()).thenReturn(7L, 9L);
            reporter =
                    new DeliveryMetricsReporter(
                            new org.flexlb.metric.MicrometerFlexMonitor(registry));
            reporter.registerExecutor(stats);
            assertEquals(
                    2.0,
                    registry.get("flexlb." + DISPATCH_EXECUTOR_ACTIVE_THREADS).gauge().value());
            assertEquals(
                    3.0, registry.get("flexlb." + DISPATCH_EXECUTOR_QUEUE_SIZE).gauge().value());
            assertEquals(
                    4.0, registry.get("flexlb." + DISPATCH_EXECUTOR_POOL_SIZE).gauge().value());
            var completed =
                    registry.get("flexlb." + DISPATCH_EXECUTOR_COMPLETED_TASKS).functionCounter();
            assertEquals(7.0, completed.count());
            assertEquals(9.0, completed.count());
        } finally {
            registry.close();
        }
    }

    @Test
    void failingExecutorGaugeRegistrationDoesNotSuppressTheOtherMetrics() {
        var registry = new SimpleMeterRegistry();
        try {
            registry.config()
                    .meterFilter(
                            new MeterFilter() {
                                @Override
                                public Meter.Id map(Meter.Id id) {
                                    if (id.getName().endsWith(DISPATCH_EXECUTOR_ACTIVE_THREADS)) {
                                        throw new IllegalStateException(
                                                "backend rejected this metric");
                                    }
                                    return id;
                                }
                            });
            reporter =
                    new DeliveryMetricsReporter(
                            new org.flexlb.metric.MicrometerFlexMonitor(registry));
            assertDoesNotThrow(
                    () -> reporter.registerExecutor(mock(org.flexlb.metric.ExecutorStats.class)));
            assertEquals(3, registry.getMeters().size());
        } finally {
            registry.close();
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    void executorBindingUsesTheConfiguredMonitor(boolean withBackend) {
        var registry = new SimpleMeterRegistry();
        try (var context = new AnnotationConfigApplicationContext()) {
            context.registerBean(
                    FlexMonitor.class,
                    () ->
                            withBackend
                                    ? new org.flexlb.metric.MicrometerFlexMonitor(registry)
                                    : monitor);
            if (withBackend) {
                context.registerBean(MeterRegistry.class, () -> registry);
            }
            context.register(DeliveryMetricsReporter.class);
            context.registerBean(
                    "executorBinding",
                    Object.class,
                    () -> {
                        context.getBean(DeliveryMetricsReporter.class)
                                .registerExecutor(mock(org.flexlb.metric.ExecutorStats.class));
                        return new Object();
                    });
            context.refresh();
            assertEquals(withBackend ? 4 : 0, registry.getMeters().size());
        } finally {
            registry.close();
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void metricFailureCannotInterruptDeliveryInitializationOrReporting(boolean error) {
        Throwable failure =
                error
                        ? new AssertionError("monitor failed")
                        : new IllegalStateException("monitor failed");
        doThrow(failure)
                .when(monitor)
                .register(ROUTING_QUEUE_LENGTH, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        assertDoesNotThrow(reporter::init);
        verify(monitor)
                .register(
                        ROUTING_QUEUE_WAIT_TIME_MS, FlexMetricType.TIMER, FlexPriorityType.PRECISE);
        var tags = FlexMetricTags.ofEngine("10.0.0.1", "role", "PREFILL");
        doThrow(failure).when(monitor).prepare(DISPATCH_ACK_TIME_MS, tags);
        assertDoesNotThrow(() -> reporter.prepareEndpointMetrics("PREFILL", "10.0.0.1"));
        verify(monitor).prepare(ROUTE_SUBMIT_TIME_MS, tags);
        doThrow(failure)
                .when(monitor)
                .report(
                        INFLIGHT_REQUEST_COUNT,
                        FlexMetricTags.ofEngine("10.0.0.1", "role", "DECODE"),
                        2);
        assertDoesNotThrow(() -> reporter.reportDecodeInflight("10.0.0.1", 2, 3, 4, 5, 6));
        verify(monitor)
                .report(
                        DECODE_TOTAL_LOAD,
                        FlexMetricTags.ofEngine("10.0.0.1", "role", "DECODE"),
                        3);
    }

    @ParameterizedTest
    @CsvSource({
        "DISPATCH_ACK, app.flexlb.dispatch.ack.time.ms",
        "ROUTE_SUBMIT, app.flexlb.route.submit.time.ms",
        "ACK_TO_RESPONSE, app.flexlb.ack.to.response.time.ms"
    })
    void latencySeriesRetainTheirPublishedNamesAndTags(
            DeliveryMetricsReporter.Latency latency, String metric) {
        reporter.reportLatency(latency, "PREFILL", "10.0.0.1", 17L);

        verify(monitor)
                .report(metric, FlexMetricTags.of("role", "PREFILL", "engineIp", "10.0.0.1"), 17.0);
        verifyNoMoreInteractions(monitor);
    }

    @Test
    void should_register_dispatch_reason_metric_on_init() {
        reporter.init();

        verify(monitor)
                .register(
                        ENGINE_BALANCING_MASTER_DISPATCH_REASON,
                        FlexMetricType.QPS,
                        FlexPriorityType.PRECISE);
        verify(monitor, never()).register(eq(ENGINE_BALANCING_MASTER_SELECT_DETAIL), any());
    }

    @Test
    void should_report_batcher_queue_depth_by_priority_on_routing_queue_length_with_priority_tag() {
        reporter.reportBatcherQueueDepthByPriority("PREFILL", "10.0.0.1", 70, 3);

        FlexMetricTags tags =
                FlexMetricTags.of(
                        "type", "batchQueue",
                        "role", "PREFILL",
                        "engineIp", "10.0.0.1",
                        "priority", "70");
        verify(monitor).report(ROUTING_QUEUE_LENGTH, tags, 3.0);
        // Priority buckets never leak into the independent global series
        verify(monitor, never()).report(eq(BATCHER_QUEUE_SIZE), any(), anyDouble());
    }

    @Test
    void should_report_dispatch_reason_with_correct_tags() {
        reporter.reportDispatchReason("PREFILL", "10.0.0.1", "batch_full");

        FlexMetricTags tags =
                FlexMetricTags.of(
                        "role", "PREFILL",
                        "engineIp", "10.0.0.1",
                        "reason", "batch_full");
        verify(monitor).report(ENGINE_BALANCING_MASTER_DISPATCH_REASON, tags, 1.0);
    }

    @Test
    void should_not_report_dispatch_reason_to_select_detail_metric() {
        reporter.reportDispatchReason("PREFILL", "10.0.0.1", "batch_full");

        verify(monitor, never())
                .report(eq(ENGINE_BALANCING_MASTER_SELECT_DETAIL), any(), anyDouble());
    }

    @Test
    void should_prepare_all_fixed_window_endpoint_metrics() {
        reporter.prepareEndpointMetrics("PREFILL", "10.0.0.1");

        FlexMetricTags endpointTags =
                FlexMetricTags.of(
                        "role", "PREFILL",
                        "engineIp", "10.0.0.1");
        verify(monitor).prepare(DISPATCH_ACK_TIME_MS, endpointTags);
        verify(monitor).prepare(ROUTE_SUBMIT_TIME_MS, endpointTags);
        verify(monitor).prepare(ROUTING_QUEUE_WAIT_TIME_MS, endpointTags);
        for (String reason :
                new String[] {"batch_full", "fixed_window_timeout", "predicted_execution_cap"}) {
            FlexMetricTags reasonTags =
                    FlexMetricTags.of(
                            "role", "PREFILL",
                            "engineIp", "10.0.0.1",
                            "reason", reason);
            verify(monitor).prepare(ENGINE_BALANCING_MASTER_DISPATCH_REASON, reasonTags);
        }
    }

    @Test
    void should_not_prepare_prefill_batch_metrics_for_decode_endpoint() {
        reporter.prepareEndpointMetrics("DECODE", "10.0.0.2");

        verify(monitor, never()).prepare(any(), any());
    }

    @Test
    void should_register_inflight_max_age_and_ttl_expired_metrics_on_init() {
        reporter.init();

        verify(monitor)
                .register(INFLIGHT_MAX_AGE_MS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor)
                .register(INFLIGHT_TTL_EXPIRED_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        verify(monitor)
                .register(
                        DECODE_INFLIGHT_HARD_KV_RESERVED_TOKENS,
                        FlexMetricType.GAUGE,
                        FlexPriorityType.PRECISE);
    }

    @Test
    void should_report_inflight_max_age_with_role_and_engine_tags() {
        reporter.reportDecodeInflight("10.0.0.1", 0, 0, 0, 0, 42_000L);

        FlexMetricTags tags =
                FlexMetricTags.of(
                        "engineIp", "10.0.0.1",
                        "role", "DECODE");
        verify(monitor).report(INFLIGHT_MAX_AGE_MS, tags, 42_000.0);
    }

    @Test
    void prefillSnapshotKeepsTotalOwnershipDistinctFromIndividualOwnership() {
        reporter.reportPrefillInflight("10.0.0.1", 2, 7, 42L);

        var tags = FlexMetricTags.ofEngine("10.0.0.1", "role", "PREFILL");
        verify(monitor).report(org.flexlb.constant.MetricConstant.INFLIGHT_BATCH_COUNT, tags, 2.0);
        verify(monitor).report(INFLIGHT_REQUEST_COUNT, tags, 7.0);
        verify(monitor).report(INFLIGHT_MAX_AGE_MS, tags, 42.0);
        verifyNoMoreInteractions(monitor);
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(strings = {"none", "predicted", "actual", "gap"})
    void completionMetricFailureDoesNotSuppressOtherMetrics(String failedMetric) {
        var tags = FlexMetricTags.ofEngine("10.0.0.1", "role", "PREFILL");
        String predicted = org.flexlb.constant.MetricConstant.BATCH_PREDICTED_TIME_MS;
        String actual = org.flexlb.constant.MetricConstant.BATCH_ACTUAL_TIME_MS;
        String gap = org.flexlb.constant.MetricConstant.BATCH_PREDICT_GAP_MS;
        if (!"none".equals(failedMetric)) {
            org.mockito.Mockito.doThrow(new IllegalStateException("metrics unavailable"))
                    .when(monitor)
                    .report(
                            eq(
                                    switch (failedMetric) {
                                        case "predicted" -> predicted;
                                        case "actual" -> actual;
                                        default -> gap;
                                    }),
                            eq(tags),
                            anyDouble());
        }

        org.junit.jupiter.api.Assertions.assertDoesNotThrow(
                () -> reporter.reportBatchCompletion("10.0.0.1", 100L, 75L));

        verify(monitor).report(predicted, tags, 100.0);
        verify(monitor).report(actual, tags, 75.0);
        verify(monitor).report(gap, tags, -25.0);
        verifyNoMoreInteractions(monitor);
    }

    @Test
    void should_report_scheduler_inflight_max_age_with_scheduler_role() {
        reporter.reportSchedulerInflight(7, 15_000L);

        FlexMetricTags tags =
                FlexMetricTags.of(
                        "engineIp", "scheduler",
                        "role", "SCHEDULER");
        verify(monitor).report(INFLIGHT_MAX_AGE_MS, tags, 15_000.0);
        verify(monitor)
                .report(
                        org.flexlb.constant.MetricConstant.SCHEDULER_INFLIGHT_SIZE,
                        FlexMetricTags.ofEngine("scheduler", "role", "PREFILL"),
                        7.0);
        verifyNoMoreInteractions(monitor);
    }

    @Test
    void should_report_endpoint_inflight_ttl_expired_with_reason_bucket() {
        reporter.reportEndpointInflightTtlExpired("PREFILL", "10.0.0.1", "ttl", 2);

        FlexMetricTags tags =
                FlexMetricTags.of(
                        "engineIp", "10.0.0.1",
                        "role", "PREFILL",
                        "reason", "ttl");
        verify(monitor).report(INFLIGHT_TTL_EXPIRED_QPS, tags, 2.0);
    }

    @Test
    void should_report_endpoint_inflight_ttl_expired_for_decode_endpoint() {
        reporter.reportEndpointInflightTtlExpired("DECODE", "10.0.0.2", "ttl", 1);

        FlexMetricTags tags =
                FlexMetricTags.of(
                        "engineIp", "10.0.0.2",
                        "role", "DECODE",
                        "reason", "ttl");
        verify(monitor).report(INFLIGHT_TTL_EXPIRED_QPS, tags, 1.0);
    }

    @Test
    void should_report_decode_snapshot_with_distinct_expected_and_hard_kv() {
        reporter.reportDecodeInflight("10.0.0.2", 2, 5, 9_216L, 8_192L, 42L);

        FlexMetricTags tags = FlexMetricTags.ofEngine("10.0.0.2", "role", "DECODE");
        verify(monitor).report(INFLIGHT_REQUEST_COUNT, tags, 2.0);
        verify(monitor).report(DECODE_TOTAL_LOAD, tags, 5.0);
        verify(monitor).report(DECODE_INFLIGHT_KV_RESERVED_TOKENS, tags, 9_216.0);
        verify(monitor).report(DECODE_INFLIGHT_HARD_KV_RESERVED_TOKENS, tags, 8_192.0);
        verify(monitor).report(INFLIGHT_MAX_AGE_MS, tags, 42.0);
        verifyNoMoreInteractions(monitor);
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(longs = {0L, 71L})
    void deliveryKeepsRouteAndBatchMetricSchemasWithLargeTokenTotals(long batchId) {
        if (batchId == 0L) {
            reporter.reportBatcherQueueSize("PREFILL", "10.0.0.1", 3);
        } else {
            reporter.reportBatchDelivery(
                    "PREFILL",
                    "10.0.0.1",
                    batchId,
                    "batch_full",
                    3,
                    2,
                    Long.MAX_VALUE,
                    Long.MAX_VALUE,
                    -1L);
        }
        reporter.reportBatchWaitTimeMs("PREFILL", "10.0.0.1", 0L, 17);
        reporter.reportBatchWaitTimeMs("PREFILL", "10.0.0.1", 0L, 17);

        var tags = FlexMetricTags.ofEngine("10.0.0.1", "role", "PREFILL");
        var priorityTags = FlexMetricTags.ofEngine("10.0.0.1", "role", "PREFILL", "priority", "17");
        verify(monitor).report(BATCHER_QUEUE_SIZE, tags, 3.0);
        verify(monitor, org.mockito.Mockito.times(2))
                .report(ROUTING_QUEUE_WAIT_TIME_MS, priorityTags, 0.0);
        if (batchId != 0L) {
            var reasonTags =
                    FlexMetricTags.ofEngine("10.0.0.1", "role", "PREFILL", "reason", "batch_full");
            verify(monitor).report(ENGINE_BALANCING_MASTER_DISPATCH_REASON, reasonTags, 1.0);
            verify(monitor)
                    .report(
                            org.flexlb.constant.MetricConstant.CACHE_HIT_COUNT,
                            tags,
                            (double) Long.MAX_VALUE);
            verify(monitor).report(org.flexlb.constant.MetricConstant.CACHE_HIT_RATIO, tags, 1.0);
            verify(monitor)
                    .report(org.flexlb.constant.MetricConstant.CACHE_REQUEST_TOTAL, tags, 1.0);
            verify(monitor)
                    .report(
                            org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_BATCH_SIZE,
                            reasonTags,
                            2.0);
            verify(monitor)
                    .report(
                            org.flexlb.constant.MetricConstant
                                    .ENGINE_BALANCING_MASTER_BATCH_TOTAL_TOKENS,
                            reasonTags,
                            (double) Long.MAX_VALUE);
            verify(monitor)
                    .report(org.flexlb.constant.MetricConstant.BATCH_PREDICTED_TIME_MS, tags, 0.0);
        }
        verifyNoMoreInteractions(monitor);
    }

    @Test
    void zeroTokenBatchSkipsCacheSeriesButReportsBatchShapeAndPrediction() {
        reporter.reportBatchDelivery("PREFILL", "10.0.0.1", 71L, "batch_full", 0, 1, 0L, 0L, 7L);

        verify(monitor, never())
                .report(eq(org.flexlb.constant.MetricConstant.CACHE_HIT_COUNT), any(), anyDouble());
        verify(monitor, never())
                .report(eq(org.flexlb.constant.MetricConstant.CACHE_HIT_RATIO), any(), anyDouble());
        verify(monitor, never())
                .report(
                        eq(org.flexlb.constant.MetricConstant.CACHE_REQUEST_TOTAL),
                        any(),
                        anyDouble());
        var reasonTags =
                FlexMetricTags.ofEngine("10.0.0.1", "role", "PREFILL", "reason", "batch_full");
        verify(monitor)
                .report(
                        org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_BATCH_SIZE,
                        reasonTags,
                        1.0);
        verify(monitor)
                .report(
                        org.flexlb.constant.MetricConstant
                                .ENGINE_BALANCING_MASTER_BATCH_TOTAL_TOKENS,
                        reasonTags,
                        0.0);
        verify(monitor)
                .report(
                        org.flexlb.constant.MetricConstant.BATCH_PREDICTED_TIME_MS,
                        FlexMetricTags.ofEngine("10.0.0.1", "role", "PREFILL"),
                        7.0);
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(longs = {0L, 71L})
    void deliveryReportingFailureDoesNotEscapeAfterCommit(long batchId) {
        org.mockito.Mockito.doAnswer(
                        call -> {
                            if (BATCHER_QUEUE_SIZE.equals(call.getArgument(0))) {
                                throw new AssertionError("reporting failed");
                            }
                            return null;
                        })
                .when(monitor)
                .report(org.mockito.ArgumentMatchers.anyString(), any(), anyDouble());
        org.junit.jupiter.api.Assertions.assertDoesNotThrow(
                () -> {
                    if (batchId == 0L) {
                        reporter.reportBatcherQueueSize("PREFILL", "10.0.0.1", 0);
                    } else {
                        reporter.reportBatchDelivery(
                                "PREFILL", "10.0.0.1", batchId, "batch_full", 0, 1, 0L, 0L, 0L);
                    }
                });
        verify(monitor).report(eq(BATCHER_QUEUE_SIZE), any(), eq(0.0));
    }
}
