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

import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_DISPATCH_REASON;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_SELECT_DETAIL;
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
    void should_report_dispatch_reason_with_correct_tags() {
        reporter.reportDispatchReason("PREFILL", "10.0.0.1", "10.0.0.1:8080", "batch_full");

        FlexMetricTags tags = FlexMetricTags.of(
                "role", "PREFILL",
                "engineIp", "10.0.0.1",
                "engineIpPort", "10.0.0.1:8080",
                "reason", "batch_full");
        verify(monitor).report(ENGINE_BALANCING_MASTER_DISPATCH_REASON, tags, 1.0);
    }

    @Test
    void should_not_report_dispatch_reason_to_select_detail_metric() {
        reporter.reportDispatchReason("PREFILL", "10.0.0.1", "10.0.0.1:8080", "batch_full");

        verify(monitor, never()).report(eq(ENGINE_BALANCING_MASTER_SELECT_DETAIL), any(), anyDouble());
    }
}
