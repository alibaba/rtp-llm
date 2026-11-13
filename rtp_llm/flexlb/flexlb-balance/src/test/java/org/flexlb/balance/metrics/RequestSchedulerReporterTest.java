package org.flexlb.balance.metrics;

import org.flexlb.balance.metrics.RequestSchedulerReporter.CancelEvent;
import org.flexlb.balance.metrics.RequestSchedulerReporter.PreemptionEvent;
import org.flexlb.util.metrics.FlexMetricTags;
import org.flexlb.util.metrics.FlexMetricType;
import org.flexlb.util.metrics.FlexMonitor;
import org.flexlb.util.metrics.FlexPriorityType;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.flexlb.util.metrics.MetricConstant.AUTO_TPM_CANCEL_CONFIRM_COUNT;
import static org.flexlb.util.metrics.MetricConstant.AUTO_TPM_CANCEL_QPS;
import static org.flexlb.util.metrics.MetricConstant.AUTO_TPM_CANCEL_REQUEST_COUNT;
import static org.flexlb.util.metrics.MetricConstant.AUTO_TPM_CANCEL_TIMEOUT_COUNT;
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

    @org.junit.jupiter.api.Test
    void reportsCancellationForEachPreemptedRequest() {
        var observed = org.mockito.Mockito.spy(reporter);
        var requestsToPreempt =
                java.util.List.of(
                        new org.flexlb.worker.endpoint.DecodeResources.DecodeRequestView(
                                String.valueOf(1L),
                                20,
                                100L,
                                120L,
                                org.flexlb.worker.status.DecodeTaskPhase.RUNNING,
                                new DecodeEndpoint.RequestRegistration(1L, String.valueOf(1L))),
                        new org.flexlb.worker.endpoint.DecodeResources.DecodeRequestView(
                                String.valueOf(2L),
                                30,
                                200L,
                                220L,
                                org.flexlb.worker.status.DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN,
                                new DecodeEndpoint.RequestRegistration(1L, String.valueOf(2L))));
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
        var tags = FlexMetricTags.of("endpoint", "decode", "priority", "20");
        doThrow(failure).when(monitor).report(AUTO_TPM_CANCEL_REQUEST_COUNT, tags, 1.0);
        assertDoesNotThrow(() -> reporter.reportEngineCancel(CancelEvent.REQUEST, "decode", 20));
        assertDoesNotThrow(() -> reporter.reportCancel(20, "PRIORITY_PREEMPTED"));
        verify(monitor)
                .report(
                        AUTO_TPM_CANCEL_QPS,
                        FlexMetricTags.of("priority", "20", "reason", "PRIORITY_PREEMPTED"),
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
