package org.flexlb.service.monitor;

import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.service.monitor.RequestSchedulerReporter.CancelEvent;
import org.flexlb.service.monitor.RequestSchedulerReporter.PreemptionEvent;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.flexlb.constant.MetricConstant.AUTO_TPM_CANCEL_CONFIRM_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_CANCEL_QPS;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_CANCEL_REQUEST_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_CANCEL_TIMEOUT_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_PRIORITY_PREEMPT_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_VICTIM_COUNT;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.verify;

/**
 * Cancel metric contract of {@link RequestSchedulerReporter}: the cancel
 * QPS metric is registered as QPS (TIMER has KMonitor adapter compatibility
 * issues) and every cancel-family report carries the priority tag.
 */
@ExtendWith(MockitoExtension.class)
class RequestSchedulerReporterTest {

    @Mock private FlexMonitor monitor;

    private RequestSchedulerReporter reporter;

    @BeforeEach
    void setUp() {
        reporter = new RequestSchedulerReporter(monitor);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void committedVictimStatisticsDistinguishQueueWithdrawalAndEngineCancellation(
            boolean cancelled) {
        var observed = org.mockito.Mockito.spy(reporter);
        var requestsToPreempt =
                java.util.List.of(
                        new org.flexlb.balance.endpoint.DecodeResources.DecodeRequestView(
                                String.valueOf(1L),
                                20,
                                100L,
                                120L,
                                org.flexlb.enums.DecodeTaskPhase.RUNNING,
                                true,
                                1L,
                                true),
                        new org.flexlb.balance.endpoint.DecodeResources.DecodeRequestView(
                                String.valueOf(2L),
                                30,
                                200L,
                                220L,
                                org.flexlb.enums.DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN,
                                true,
                                2L,
                                true));
        observed.reportPreemptedRequests(
                String.valueOf(3L), 70, "decode", "capacity", requestsToPreempt, cancelled);
        verify(observed)
                .reportPreemptedRequest(
                        20, 70, cancelled ? "decode_running" : "decode_reserved", "capacity");
        verify(observed)
                .reportPreemptedRequest(
                        30, 70, cancelled ? "decode_cancel" : "decode_reserved", "capacity");
        verify(observed)
                .reportPreemptedKvTokens(
                        20, cancelled ? "decode_running" : "decode_reserved", 100L);
        verify(observed)
                .reportPreemptedKvTokens(30, cancelled ? "decode_cancel" : "decode_reserved", 200L);
        verify(observed).reportPreemption(PreemptionEvent.COMMIT, 70, "capacity", "success");
        for (var requestToPreempt : requestsToPreempt) {
            verify(observed, org.mockito.Mockito.times(cancelled ? 1 : 0))
                    .reportEngineCancel(CancelEvent.CONFIRM, "decode", requestToPreempt.priority());
        }
        observed.reportPreemptionCancelRequests("decode", requestsToPreempt);
        for (var requestToPreempt : requestsToPreempt) {
            verify(observed)
                    .reportEngineCancel(CancelEvent.REQUEST, "decode", requestToPreempt.priority());
            verify(observed).reportCancel(requestToPreempt.priority(), "PRIORITY_PREEMPTED");
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void metricFailureCannotInterruptSchedulingOrSuppressNextMetric(boolean error) {
        Throwable failure =
                error
                        ? new AssertionError("monitor failed")
                        : new IllegalStateException("monitor failed");
        doThrow(failure)
                .when(monitor)
                .register(AUTO_TPM_CANCEL_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        assertDoesNotThrow(reporter::init);
        verify(monitor)
                .register(
                        AUTO_TPM_CANCEL_REQUEST_COUNT,
                        FlexMetricType.QPS,
                        FlexPriorityType.PRECISE);
        var tags =
                FlexMetricTags.of(
                        "victim_priority",
                        "20",
                        "incoming_priority",
                        "70",
                        "stage",
                        "decode_reserved",
                        "case",
                        "decode_kv_full");
        doThrow(failure).when(monitor).report(AUTO_TPM_VICTIM_COUNT, tags, 1.0);
        assertDoesNotThrow(
                () -> reporter.reportPreemptedRequest(20, 70, "decode_reserved", "decode_kv_full"));
        verify(monitor)
                .report(
                        AUTO_TPM_PRIORITY_PREEMPT_COUNT,
                        FlexMetricTags.of("stage", "decode_reserved"),
                        1.0);
    }

    @ParameterizedTest
    @CsvSource({
        "PLAN, auto_tpm.eviction_plan.count, feasible",
        "PLAN, auto_tpm.eviction_plan.count, infeasible",
        "COMMIT, auto_tpm.eviction_commit.count, success",
        "COMMIT, auto_tpm.eviction_commit.count, conflict"
    })
    void evictionOutcomesPreserveTheirMetricAndTagSchema(
            PreemptionEvent event, String metric, String result) {
        reporter.reportPreemption(event, 70, "decode_kv_full", result);

        var ordered = inOrder(monitor);
        ordered.verify(monitor)
                .report(
                        metric,
                        FlexMetricTags.of(
                                "priority", "70", "case", "decode_kv_full", "result", result),
                        1.0);
        ordered.verifyNoMoreInteractions();
    }

    @Test
    void victimEventPreservesBothMetricSchemasAndTheirOrder() {
        reporter.reportPreemptedRequest(20, 70, "decode_reserved", "decode_kv_full");

        var ordered = inOrder(monitor);
        ordered.verify(monitor)
                .report(
                        AUTO_TPM_VICTIM_COUNT,
                        FlexMetricTags.of(
                                "victim_priority",
                                "20",
                                "incoming_priority",
                                "70",
                                "stage",
                                "decode_reserved",
                                "case",
                                "decode_kv_full"),
                        1.0);
        ordered.verify(monitor)
                .report(
                        AUTO_TPM_PRIORITY_PREEMPT_COUNT,
                        FlexMetricTags.of("stage", "decode_reserved"),
                        1.0);
        ordered.verifyNoMoreInteractions();
    }

    @Test
    void should_register_cancel_qps_metric_as_qps_on_init() {
        reporter.init();

        verify(monitor).register(AUTO_TPM_CANCEL_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        verify(monitor)
                .register(
                        AUTO_TPM_CANCEL_REQUEST_COUNT,
                        FlexMetricType.QPS,
                        FlexPriorityType.PRECISE);
        verify(monitor)
                .register(
                        AUTO_TPM_CANCEL_CONFIRM_COUNT,
                        FlexMetricType.QPS,
                        FlexPriorityType.PRECISE);
        verify(monitor)
                .register(
                        AUTO_TPM_CANCEL_TIMEOUT_COUNT,
                        FlexMetricType.QPS,
                        FlexPriorityType.PRECISE);
    }

    @Test
    void should_report_cancel_qps_with_priority_and_reason_tags() {
        reporter.reportCancel(30, "PRIORITY_PREEMPTED");

        verify(monitor)
                .report(
                        AUTO_TPM_CANCEL_QPS,
                        FlexMetricTags.of("priority", "30", "reason", "PRIORITY_PREEMPTED"),
                        1.0);
    }

    @Test
    void should_report_user_cancel_qps_with_zero_priority_when_not_inflight() {
        reporter.reportCancel(0, "USER_CANCELLED");

        verify(monitor)
                .report(
                        AUTO_TPM_CANCEL_QPS,
                        FlexMetricTags.of("priority", "0", "reason", "USER_CANCELLED"),
                        1.0);
    }

    @Test
    void should_report_cancel_request_with_priority_tag() {
        reporter.reportEngineCancel(CancelEvent.REQUEST, "10.0.0.2:8081", 30);

        verify(monitor)
                .report(
                        AUTO_TPM_CANCEL_REQUEST_COUNT,
                        FlexMetricTags.of("endpoint", "10.0.0.2:8081", "priority", "30"),
                        1.0);
    }

    @Test
    void should_report_cancel_confirm_with_priority_tag() {
        reporter.reportEngineCancel(CancelEvent.CONFIRM, "10.0.0.2:8081", 30);

        verify(monitor)
                .report(
                        AUTO_TPM_CANCEL_CONFIRM_COUNT,
                        FlexMetricTags.of("endpoint", "10.0.0.2:8081", "priority", "30"),
                        1.0);
    }

    @Test
    void should_report_cancel_timeout_with_incoming_priority_tag() {
        reporter.reportEngineCancel(CancelEvent.TIMEOUT, "10.0.0.2:8081", 70);

        verify(monitor)
                .report(
                        AUTO_TPM_CANCEL_TIMEOUT_COUNT,
                        FlexMetricTags.of("endpoint", "10.0.0.2:8081", "priority", "70"),
                        1.0);
    }
}
