package org.flexlb.service.monitor;

import io.micrometer.prometheus.PrometheusConfig;
import io.micrometer.prometheus.PrometheusMeterRegistry;
import org.flexlb.balance.scheduler.PrefillQueueFlow;
import org.flexlb.config.WhitelistMetricsFilterConfig;
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

import java.util.List;

import static org.flexlb.constant.MetricConstant.BATCHER_QUEUE_SIZE;
import static org.flexlb.constant.MetricConstant.DECODE_INFLIGHT_HARD_KV_RESERVED_TOKENS;
import static org.flexlb.constant.MetricConstant.DISPATCH_ACK_TIME_MS;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_DISPATCH_REASON;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_SELECT_DETAIL;
import static org.flexlb.constant.MetricConstant.INFLIGHT_MAX_AGE_MS;
import static org.flexlb.constant.MetricConstant.INFLIGHT_TTL_EXPIRED_QPS;
import static org.flexlb.constant.MetricConstant.PREFILL_LOCAL_QUEUE_REQUESTS;
import static org.flexlb.constant.MetricConstant.PREFILL_QUEUED_COMPUTE_TOKENS;
import static org.flexlb.constant.MetricConstant.PREFILL_QUEUED_COMPUTE_TOKENS_BY_STAGE;
import static org.flexlb.constant.MetricConstant.PREFILL_QUEUE_FLOW_COMPUTE_TOKENS;
import static org.flexlb.constant.MetricConstant.PREFILL_QUEUE_FLOW_REQUESTS;
import static org.flexlb.constant.MetricConstant.ROUTE_SUBMIT_TIME_MS;
import static org.flexlb.constant.MetricConstant.ROUTING_QUEUE_LENGTH;
import static org.flexlb.constant.MetricConstant.ROUTING_QUEUE_WAIT_TIME_MS;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
class BatchSchedulerReporterTest {

    @Mock
    private FlexMonitor monitor;

    private BatchSchedulerReporter reporter;

    @BeforeEach
    void setUp() {
        reporter = new BatchSchedulerReporter(monitor);
    }

    @Test
    void should_register_dispatch_reason_metric_on_init() {
        reporter.init();

        verify(monitor).register(ENGINE_BALANCING_MASTER_DISPATCH_REASON, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        verify(monitor, never()).register(eq(ENGINE_BALANCING_MASTER_SELECT_DETAIL), any(), any());
    }

    @Test
    void should_report_batcher_queue_depth_by_priority_on_routing_queue_length_with_priority_tag() {
        reporter.reportBatcherQueueDepthByPriority("PREFILL", "10.0.0.1", 70, 3);

        FlexMetricTags tags = FlexMetricTags.of(
                "type", "batchQueue",
                "role", "PREFILL",
                "engineIp", "10.0.0.1",
                "priority", "70");
        verify(monitor).report(ROUTING_QUEUE_LENGTH, tags, 3.0);
        // Priority buckets never leak into the independent global series
        verify(monitor, never()).report(eq(BATCHER_QUEUE_SIZE), any(), anyDouble());
    }

    @Test
    void should_keep_global_batcher_queue_depth_series_untagged_by_priority() {
        reporter.reportBatcherQueueDepth("PREFILL", "10.0.0.1", 5);

        FlexMetricTags tags = FlexMetricTags.of(
                "type", "batchQueue",
                "role", "PREFILL",
                "engineIp", "10.0.0.1");
        verify(monitor).report(ROUTING_QUEUE_LENGTH, tags, 5.0);
    }

    @Test
    void should_report_dispatch_reason_with_correct_tags() {
        reporter.reportDispatchReason("PREFILL", "10.0.0.1", "batch_full");

        FlexMetricTags tags = FlexMetricTags.of(
                "role", "PREFILL",
                "engineIp", "10.0.0.1",
                "reason", "batch_full");
        verify(monitor).report(ENGINE_BALANCING_MASTER_DISPATCH_REASON, tags, 1.0);
    }

    @Test
    void should_not_report_dispatch_reason_to_select_detail_metric() {
        reporter.reportDispatchReason("PREFILL", "10.0.0.1", "batch_full");

        verify(monitor, never()).report(eq(ENGINE_BALANCING_MASTER_SELECT_DETAIL), any(), anyDouble());
    }

    @Test
    void should_prepare_all_fixed_window_endpoint_metrics() {
        reporter.prepareEndpointMetrics("PREFILL", "10.0.0.1");

        FlexMetricTags endpointTags = FlexMetricTags.of(
                "role", "PREFILL",
                "engineIp", "10.0.0.1");
        verify(monitor).prepare(DISPATCH_ACK_TIME_MS, endpointTags);
        verify(monitor).prepare(ROUTE_SUBMIT_TIME_MS, endpointTags);
        verify(monitor).prepare(ROUTING_QUEUE_WAIT_TIME_MS, endpointTags);
        for (String reason : new String[]{"batch_full", "fixed_window_timeout", "predicted_execution_cap"}) {
            FlexMetricTags reasonTags = FlexMetricTags.of(
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

        verify(monitor).register(INFLIGHT_MAX_AGE_MS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).register(INFLIGHT_TTL_EXPIRED_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        verify(monitor).register(DECODE_INFLIGHT_HARD_KV_RESERVED_TOKENS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).register(PREFILL_QUEUED_COMPUTE_TOKENS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).register(PREFILL_QUEUED_COMPUTE_TOKENS_BY_STAGE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).register(PREFILL_LOCAL_QUEUE_REQUESTS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).register(PREFILL_QUEUE_FLOW_REQUESTS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        verify(monitor).register(PREFILL_QUEUE_FLOW_COMPUTE_TOKENS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
    }

    @Test
    void reportsQueueFlowAsIncrementsTaggedByEndpointAndEvent() {
        var flow = new PrefillQueueFlow.Snapshot(List.of(
                new PrefillQueueFlow.Entry(PrefillQueueFlow.Event.ENQUEUE, 2, 80),
                new PrefillQueueFlow.Entry(PrefillQueueFlow.Event.EXPIRED, 1, 30)));

        reporter.reportPrefillQueueFlow(RoleType.PREFILL, "p", flow);

        var enqueued = FlexMetricTags.ofEngine("p", "role", "PREFILL", "event", "ENQUEUE");
        var expired = FlexMetricTags.ofEngine("p", "role", "PREFILL", "event", "EXPIRED");
        verify(monitor).report(PREFILL_QUEUE_FLOW_REQUESTS, enqueued, 2.0);
        verify(monitor).report(PREFILL_QUEUE_FLOW_COMPUTE_TOKENS, enqueued, 80.0);
        verify(monitor).report(PREFILL_QUEUE_FLOW_REQUESTS, expired, 1.0);
        verify(monitor).report(PREFILL_QUEUE_FLOW_COMPUTE_TOKENS, expired, 30.0);
    }

    @Test
    void prometheusExposesPerWorkerQueuedComputeAndUnknownValues() {
        String series = "flexlb_app_flexlb_prefill_queued_compute_tokens";
        var whitelist = new WhitelistMetricsFilterConfig(series);
        var registry = new PrometheusMeterRegistry(PrometheusConfig.DEFAULT);
        try {
            whitelist.init();
            registry.config().meterFilter(whitelist.whitelistMetricsFilter());
            var exporter = new BatchSchedulerReporter(new MicrometerFlexMonitor(registry));
            exporter.init();
            exporter.reportPrefillQueuedComputeTokens(RoleType.PREFILL, "p", 12);
            exporter.reportPrefillQueuedComputeTokens(RoleType.PDFUSION, "f", Double.NaN);
            exporter.reportPrefillQueuedComputeTokensByStage(RoleType.PREFILL, "p", 5, 0, Double.NaN);

            String scrape = registry.scrape();
            assertTrue(scrape.lines().anyMatch(line -> line.startsWith(series + "{")
                    && line.contains("engineIp=\"p\"") && line.contains("role=\"PREFILL\"") && line.endsWith(" 12.0")));
            assertTrue(scrape.lines().anyMatch(line -> line.startsWith(series + "{")
                    && line.contains("engineIp=\"f\"") && line.contains("role=\"PDFUSION\"") && line.endsWith(" NaN")));
            assertTrue(scrape.lines().anyMatch(line -> line.startsWith(series + "_by_stage{")
                    && line.contains("engineIp=\"p\"") && line.contains("stage=\"MASTER_QUEUED\"")
                    && line.endsWith(" 5.0")));
            assertTrue(scrape.lines().anyMatch(line -> line.startsWith(series + "_by_stage{")
                    && line.contains("stage=\"DISPATCHED_UNOBSERVED\"") && line.endsWith(" 0.0")));
            assertTrue(scrape.lines().anyMatch(line -> line.startsWith(series + "_by_stage{")
                    && line.contains("stage=\"WORKER_WAITING\"") && line.endsWith(" NaN")));
        } finally {
            registry.close();
            MicrometerFlexMonitor.setAllowedMetrics(null);
        }
    }

    @Test
    void should_report_inflight_max_age_with_role_and_engine_tags() {
        reporter.reportInflightMaxAgeMs("DECODE", "10.0.0.1", 42_000L);

        FlexMetricTags tags = FlexMetricTags.of(
                "engineIp", "10.0.0.1",
                "role", "DECODE");
        verify(monitor).report(INFLIGHT_MAX_AGE_MS, tags, 42_000.0);
    }

    @Test
    void should_report_scheduler_inflight_max_age_with_scheduler_role() {
        reporter.reportSchedulerInflightMaxAgeMs(15_000L);

        FlexMetricTags tags = FlexMetricTags.of(
                "engineIp", "scheduler",
                "role", "SCHEDULER");
        verify(monitor).report(INFLIGHT_MAX_AGE_MS, tags, 15_000.0);
    }

    @Test
    void should_report_endpoint_inflight_ttl_expired_with_reason_bucket() {
        reporter.reportEndpointInflightTtlExpired("PREFILL", "10.0.0.1", "ttl", 2);

        FlexMetricTags tags = FlexMetricTags.of(
                "engineIp", "10.0.0.1",
                "role", "PREFILL",
                "reason", "ttl");
        verify(monitor).report(INFLIGHT_TTL_EXPIRED_QPS, tags, 2.0);
    }

    @Test
    void should_report_endpoint_inflight_ttl_expired_for_decode_endpoint() {
        reporter.reportEndpointInflightTtlExpired("DECODE", "10.0.0.2", "ttl", 1);

        FlexMetricTags tags = FlexMetricTags.of(
                "engineIp", "10.0.0.2",
                "role", "DECODE",
                "reason", "ttl");
        verify(monitor).report(INFLIGHT_TTL_EXPIRED_QPS, tags, 1.0);
    }

    @Test
    void should_report_decode_inflight_hard_kv_reserved_with_decode_role() {
        reporter.reportDecodeInflightHardKvReserved("10.0.0.2", 8_192L);

        FlexMetricTags tags = FlexMetricTags.of(
                "engineIp", "10.0.0.2",
                "role", "DECODE");
        verify(monitor).report(DECODE_INFLIGHT_HARD_KV_RESERVED_TOKENS, tags, 8_192.0);
    }
}
