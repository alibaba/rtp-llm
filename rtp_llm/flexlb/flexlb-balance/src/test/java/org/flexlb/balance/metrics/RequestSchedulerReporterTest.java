package org.flexlb.balance.metrics;

import org.flexlb.balance.metrics.RequestSchedulerReporter.PreemptionEvent;
import org.flexlb.util.metrics.FlexMetricTags;
import org.flexlb.util.metrics.FlexMetricType;
import org.flexlb.util.metrics.FlexMonitor;
import org.flexlb.util.metrics.FlexPriorityType;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.flexlb.util.metrics.MetricConstant.AUTO_TPM_CANCEL_CONFIRM_COUNT;
import static org.flexlb.util.metrics.MetricConstant.AUTO_TPM_CANCEL_QPS;
import static org.flexlb.util.metrics.MetricConstant.AUTO_TPM_CANCEL_REQUEST_COUNT;
import static org.flexlb.util.metrics.MetricConstant.AUTO_TPM_CANCEL_TIMEOUT_COUNT;
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
}
