package org.flexlb.service.monitor;

import org.flexlb.service.monitor.RequestSchedulerReporter.CancelEvent;
import org.flexlb.service.monitor.RequestSchedulerReporter.EvictionEvent;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.flexlb.constant.MetricConstant.AUTO_TPM_CANCEL_CONFIRM_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_CANCEL_QPS;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_CANCEL_REQUEST_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_CANCEL_TIMEOUT_COUNT;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.inOrder;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_VICTIM_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_PRIORITY_PREEMPT_COUNT;

/**
 * Cancel metric contract of {@link RequestSchedulerReporter}: the cancel
 * QPS metric is registered as QPS (TIMER has KMonitor adapter compatibility
 * issues) and every cancel-family report carries the priority tag.
 */
@ExtendWith(MockitoExtension.class)
class RequestSchedulerReporterTest {

    @Mock
    private FlexMonitor monitor;

    private RequestSchedulerReporter reporter;

    @BeforeEach
    void setUp() {
        reporter = new RequestSchedulerReporter(monitor);
    }

    @ParameterizedTest
    @CsvSource({
            "PLAN, auto_tpm.eviction_plan.count, feasible",
            "PLAN, auto_tpm.eviction_plan.count, infeasible",
            "COMMIT, auto_tpm.eviction_commit.count, success",
            "COMMIT, auto_tpm.eviction_commit.count, conflict"
    })
    void evictionOutcomesPreserveTheirMetricAndTagSchema(EvictionEvent event, String metric, String result) {
        reporter.reportEviction(event, 70, "decode_kv_full", result);

        var ordered = inOrder(monitor);
        ordered.verify(monitor).report(metric,
                FlexMetricTags.of("priority", "70", "case", "decode_kv_full", "result", result), 1.0);
        ordered.verifyNoMoreInteractions();
    }

    @Test
    void victimEventPreservesBothMetricSchemasAndTheirOrder() {
        reporter.reportVictim(20, 70, "decode_reserved", "decode_kv_full");

        var ordered = inOrder(monitor);
        ordered.verify(monitor).report(AUTO_TPM_VICTIM_COUNT,
                FlexMetricTags.of("victim_priority", "20", "incoming_priority", "70",
                        "stage", "decode_reserved", "case", "decode_kv_full"), 1.0);
        ordered.verify(monitor).report(AUTO_TPM_PRIORITY_PREEMPT_COUNT,
                FlexMetricTags.of("stage", "decode_reserved"), 1.0);
        ordered.verifyNoMoreInteractions();
    }

    @Test
    void should_register_cancel_qps_metric_as_qps_on_init() {
        reporter.init();

        verify(monitor).register(AUTO_TPM_CANCEL_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        verify(monitor).register(AUTO_TPM_CANCEL_REQUEST_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        verify(monitor).register(AUTO_TPM_CANCEL_CONFIRM_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        verify(monitor).register(AUTO_TPM_CANCEL_TIMEOUT_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
    }

    @Test
    void should_report_cancel_qps_with_priority_and_reason_tags() {
        reporter.reportCancel(30, "PRIORITY_PREEMPTED");

        verify(monitor).report(AUTO_TPM_CANCEL_QPS,
                FlexMetricTags.of("priority", "30", "reason", "PRIORITY_PREEMPTED"), 1.0);
    }

    @Test
    void should_report_user_cancel_qps_with_zero_priority_when_not_inflight() {
        reporter.reportCancel(0, "USER_CANCELLED");

        verify(monitor).report(AUTO_TPM_CANCEL_QPS,
                FlexMetricTags.of("priority", "0", "reason", "USER_CANCELLED"), 1.0);
    }

    @Test
    void should_report_cancel_request_with_priority_tag() {
        reporter.reportEngineCancel(CancelEvent.REQUEST, "10.0.0.2:8081", 30);

        verify(monitor).report(AUTO_TPM_CANCEL_REQUEST_COUNT,
                FlexMetricTags.of("endpoint", "10.0.0.2:8081", "priority", "30"), 1.0);
    }

    @Test
    void should_report_cancel_confirm_with_priority_tag() {
        reporter.reportEngineCancel(CancelEvent.CONFIRM, "10.0.0.2:8081", 30);

        verify(monitor).report(AUTO_TPM_CANCEL_CONFIRM_COUNT,
                FlexMetricTags.of("endpoint", "10.0.0.2:8081", "priority", "30"), 1.0);
    }

    @Test
    void should_report_cancel_timeout_with_incoming_priority_tag() {
        reporter.reportEngineCancel(CancelEvent.TIMEOUT, "10.0.0.2:8081", 70);

        verify(monitor).report(AUTO_TPM_CANCEL_TIMEOUT_COUNT,
                FlexMetricTags.of("endpoint", "10.0.0.2:8081", "priority", "70"), 1.0);
    }
}
