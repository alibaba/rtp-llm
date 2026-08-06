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

import static org.flexlb.constant.MetricConstant.AUTO_TPM_EXPIRED_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_QUEUE_DEPTH;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_QUEUE_WAIT_TIME_MS;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_REQUEST_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_SCHEDULE_TO_ACK_TIME_MS;
import static org.mockito.Mockito.verify;

/**
 * Tests for the Auto-TPM Queue MVP metrics of {@link BatchSchedulerReporter}.
 */
@ExtendWith(MockitoExtension.class)
class BatchSchedulerReporterAutoTpmTest {

    @Mock
    private FlexMonitor monitor;

    private BatchSchedulerReporter reporter;

    @BeforeEach
    void setUp() {
        reporter = new BatchSchedulerReporter(monitor);
    }

    @Test
    void should_register_all_auto_tpm_metrics_on_init() {
        reporter.init();

        verify(monitor).register(AUTO_TPM_REQUEST_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        verify(monitor).register(AUTO_TPM_QUEUE_WAIT_TIME_MS, FlexMetricType.TIMER, FlexPriorityType.PRECISE);
        verify(monitor).register(AUTO_TPM_SCHEDULE_TO_ACK_TIME_MS, FlexMetricType.TIMER, FlexPriorityType.PRECISE);
        verify(monitor).register(AUTO_TPM_EXPIRED_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        verify(monitor).register(AUTO_TPM_QUEUE_DEPTH, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
    }

    @Test
    void should_report_request_count_with_priority_tag() {
        reporter.reportAutoTpmRequestCount(70);

        FlexMetricTags tags = FlexMetricTags.of("priority", "70");
        verify(monitor).report(AUTO_TPM_REQUEST_COUNT, tags, 1.0);
    }

    @Test
    void should_report_queue_wait_time_with_priority_and_engine_tags() {
        reporter.reportAutoTpmQueueWaitTimeMs(50, "10.0.0.1", 123L);

        FlexMetricTags tags = FlexMetricTags.ofEngine("10.0.0.1", "priority", "50");
        verify(monitor).report(AUTO_TPM_QUEUE_WAIT_TIME_MS, tags, 123L);
    }

    @Test
    void should_report_schedule_to_ack_time_with_priority_tag() {
        reporter.reportAutoTpmScheduleToAckTimeMs(30, 456L);

        FlexMetricTags tags = FlexMetricTags.of("priority", "30");
        verify(monitor).report(AUTO_TPM_SCHEDULE_TO_ACK_TIME_MS, tags, 456L);
    }

    @Test
    void should_report_expired_count_with_priority_tag() {
        reporter.reportAutoTpmExpiredCount(30);

        FlexMetricTags tags = FlexMetricTags.of("priority", "30");
        verify(monitor).report(AUTO_TPM_EXPIRED_COUNT, tags, 1.0);
    }

    @Test
    void should_report_queue_depth_with_priority_and_engine_tags() {
        reporter.reportAutoTpmQueueDepth(60, "10.0.0.2", 7);

        FlexMetricTags tags = FlexMetricTags.ofEngine("10.0.0.2", "priority", "60");
        verify(monitor).report(AUTO_TPM_QUEUE_DEPTH, tags, 7);
    }
}
