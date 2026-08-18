package org.flexlb.service.monitor;

import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.flexlb.constant.MetricConstant.BATCH_INFLIGHT_AGE_CAPPED_QPS;
import static org.flexlb.constant.MetricConstant.BATCH_INFLIGHT_FROZEN_AUDIT_QPS;
import static org.flexlb.constant.MetricConstant.BATCHER_QUEUE_ENTER_QPS;
import static org.flexlb.constant.MetricConstant.BATCHER_QUEUE_LEAVE_QPS;
import static org.flexlb.constant.MetricConstant.BATCHER_QUEUE_SIZE;
import static org.flexlb.constant.MetricConstant.DECODE_INFLIGHT_HARD_KV_RESERVED_TOKENS;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_DISPATCH_REASON;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_SELECT_DETAIL;
import static org.flexlb.constant.MetricConstant.DISPATCH_ACK_TIME_MS;
import static org.flexlb.constant.MetricConstant.DISPATCH_RECONCILIATION_EVENT_QPS;
import static org.flexlb.constant.MetricConstant.DISPATCH_RECONCILIATION_FENCE_SIZE;
import static org.flexlb.constant.MetricConstant.INFLIGHT_MAX_AGE_MS;
import static org.flexlb.constant.MetricConstant.INFLIGHT_TTL_EXPIRED_QPS;
import static org.flexlb.constant.MetricConstant.ROUTE_SUBMIT_TIME_MS;
import static org.flexlb.constant.MetricConstant.ROUTING_QUEUE_LENGTH;
import static org.flexlb.constant.MetricConstant.ROUTING_QUEUE_WAIT_TIME_MS;
import static org.flexlb.constant.MetricConstant.SCHEDULER_INFLIGHT_MAX_AGE_MS;
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
        for (String reason : new String[]{"batch_full", "fixed_window_timeout", "predict_threshold"}) {
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
    }

    @Test
    void should_register_batch_inflight_age_capped_metric_on_init() {
        reporter.init();

        verify(monitor).register(BATCH_INFLIGHT_AGE_CAPPED_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
    }

    @Test
    void should_report_batch_inflight_age_capped_with_engine_and_role_tags() {
        reporter.reportBatchInflightAgeCapped("PREFILL", "10.0.0.1", 2);

        FlexMetricTags tags = FlexMetricTags.of(
                "engineIp", "10.0.0.1",
                "role", "PREFILL");
        verify(monitor).report(BATCH_INFLIGHT_AGE_CAPPED_QPS, tags, 2.0);
    }

    @Test
    void should_register_batch_inflight_frozen_audit_metric_on_init() {
        reporter.init();

        verify(monitor).register(BATCH_INFLIGHT_FROZEN_AUDIT_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
    }

    @Test
    void should_report_batch_inflight_frozen_audit_with_engine_and_role_tags() {
        reporter.reportBatchInflightFrozenAudit("PDFUSION", "10.0.0.2", 3);

        FlexMetricTags tags = FlexMetricTags.of(
                "engineIp", "10.0.0.2",
                "role", "PDFUSION");
        verify(monitor).report(BATCH_INFLIGHT_FROZEN_AUDIT_QPS, tags, 3.0);
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
    void should_register_dispatch_reconciliation_metrics_on_init() {
        reporter.init();

        verify(monitor).register(DISPATCH_RECONCILIATION_EVENT_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        verify(monitor).register(DISPATCH_RECONCILIATION_FENCE_SIZE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
    }

    @Test
    void should_report_scheduler_inflight_ttl_expired_with_scheduler_role_and_reason() {
        reporter.reportSchedulerInflightTtlExpired("hard_age_cap", 3);

        FlexMetricTags tags = FlexMetricTags.of(
                "engineIp", "scheduler",
                "role", "SCHEDULER",
                "reason", "hard_age_cap");
        verify(monitor).report(INFLIGHT_TTL_EXPIRED_QPS, tags, 3.0);
    }

    @Test
    void should_report_endpoint_inflight_ttl_expired_with_engine_tags() {
        reporter.reportEndpointInflightTtlExpired("DECODE", "10.0.0.2", "ttl", 2);

        FlexMetricTags tags = FlexMetricTags.of(
                "engineIp", "10.0.0.2",
                "role", "DECODE",
                "reason", "ttl");
        verify(monitor).report(INFLIGHT_TTL_EXPIRED_QPS, tags, 2.0);
    }

    @Test
    void should_report_scheduler_max_age_on_both_legacy_and_unified_series() {
        reporter.reportSchedulerInflightMaxAgeMs(12_000L);

        FlexMetricTags legacyTags = FlexMetricTags.of(
                "role", "PREFILL",
                "engineIp", "scheduler");
        verify(monitor).report(SCHEDULER_INFLIGHT_MAX_AGE_MS, legacyTags, 12_000.0);
        FlexMetricTags unifiedTags = FlexMetricTags.of(
                "engineIp", "scheduler",
                "role", "SCHEDULER");
        verify(monitor).report(INFLIGHT_MAX_AGE_MS, unifiedTags, 12_000.0);
    }

    @Test
    void should_report_dispatch_reconciliation_event_with_event_and_reason_tags() {
        reporter.reportDispatchReconciliationEvent("forced_terminal", "failure_cap");

        FlexMetricTags tags = FlexMetricTags.of(
                "role", "SCHEDULER",
                "event", "forced_terminal",
                "reason", "failure_cap");
        verify(monitor).report(DISPATCH_RECONCILIATION_EVENT_QPS, tags, 1.0);
    }

    @Test
    void should_report_dispatch_reconciliation_fence_size_gauge() {
        reporter.reportDispatchReconciliationFenceSize(4);

        FlexMetricTags tags = FlexMetricTags.of(
                "role", "SCHEDULER",
                "engineIp", "scheduler");
        verify(monitor).report(DISPATCH_RECONCILIATION_FENCE_SIZE, tags, 4.0);
    }

    @Test
    void should_report_decode_inflight_hard_kv_reserved_with_decode_role() {
        reporter.reportDecodeInflightHardKvReserved("10.0.0.2", 8_192L);

        FlexMetricTags tags = FlexMetricTags.of(
                "engineIp", "10.0.0.2",
                "role", "DECODE");
        verify(monitor).report(DECODE_INFLIGHT_HARD_KV_RESERVED_TOKENS, tags, 8_192.0);
    }

    @Test
    void should_register_batcher_queue_enter_and_leave_metrics_on_init() {
        reporter.init();

        verify(monitor).register(BATCHER_QUEUE_ENTER_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        verify(monitor).register(BATCHER_QUEUE_LEAVE_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
    }

    @Test
    void should_report_batcher_queue_enter_with_engine_and_role_tags() {
        reporter.reportBatcherQueueEnter("PREFILL", "10.0.0.1");

        FlexMetricTags tags = FlexMetricTags.of(
                "engineIp", "10.0.0.1",
                "role", "PREFILL");
        verify(monitor).report(BATCHER_QUEUE_ENTER_QPS, tags, 1.0);
    }

    @Test
    void should_report_batcher_queue_leave_with_reason_tag_and_count() {
        reporter.reportBatcherQueueLeave("PREFILL", "10.0.0.1", "dispatched", 4);

        FlexMetricTags tags = FlexMetricTags.of(
                "engineIp", "10.0.0.1",
                "role", "PREFILL",
                "reason", "dispatched");
        verify(monitor).report(BATCHER_QUEUE_LEAVE_QPS, tags, 4.0);
    }
}
