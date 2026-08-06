package org.flexlb.service.monitor;

import org.flexlb.constant.MetricConstant;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

/**
 * D12 guard: the 0 sentinel (no priority carried) never emits
 * priority-tagged auto_tpm metrics — every priority-dimension report
 * method drops the sample centrally when priority &lt;= 0.
 */
class FlexlbMetricHelperZeroPriorityTest {

    private FlexMonitor monitor;
    private FlexlbMetricHelper helper;

    @BeforeEach
    void setUp() {
        monitor = mock(FlexMonitor.class);
        helper = new FlexlbMetricHelper(monitor, MetricConstant.PATH_BATCH);
    }

    @Test
    void zeroPriority_neverReported_onAnyPriorityDimensionMetric() {
        helper.reportAutoTpmRequestCount(0);
        helper.reportAutoTpmScheduleLatency(0, "success", 12L);
        helper.reportAutoTpmNormalPlacement(0);
        helper.reportAutoTpmQueueReject(0, 70);
        helper.reportAutoTpmQueueReject(30, 0);
        helper.reportAutoTpmRunningCancel(0, 70, "success");
        helper.reportAutoTpmRunningCancel(30, 0, "success");
        helper.reportAutoTpmTtft(0, 12L);
        helper.reportAutoTpmDeadlineMiss(0);

        verify(monitor, never()).report(anyString(), any(), anyDouble());
    }

    @Test
    void positivePriority_stillReported() {
        helper.reportAutoTpmRequestCount(50);
        helper.reportAutoTpmScheduleLatency(50, "success", 12L);
        helper.reportAutoTpmNormalPlacement(50);
        helper.reportAutoTpmQueueReject(30, 70);
        helper.reportAutoTpmRunningCancel(30, 70, "success");
        helper.reportAutoTpmTtft(50, 12L);
        helper.reportAutoTpmDeadlineMiss(50);

        verify(monitor, times(7)).report(anyString(), any(), anyDouble());
    }

    // ---- D10: ttft_ms / deadline_miss.count shape (name, tags, value) ----

    @Test
    void ttft_reportedWithPriorityAndPathTags() {
        helper.reportAutoTpmTtft(50, 123L);

        ArgumentCaptor<FlexMetricTags> tags = ArgumentCaptor.forClass(FlexMetricTags.class);
        verify(monitor).report(eq(MetricConstant.AUTO_TPM_TTFT_MS), tags.capture(), eq(123.0));
        assertEquals("50", tags.getValue().getTags().get(MetricConstant.TAG_PRIORITY));
        assertEquals(MetricConstant.PATH_BATCH, tags.getValue().getTags().get(MetricConstant.TAG_PATH));
    }

    @Test
    void deadlineMiss_reportedWithPriorityAndPathTags() {
        helper.reportAutoTpmDeadlineMiss(70);

        ArgumentCaptor<FlexMetricTags> tags = ArgumentCaptor.forClass(FlexMetricTags.class);
        verify(monitor).report(eq(MetricConstant.AUTO_TPM_DEADLINE_MISS_COUNT), tags.capture(), eq(1.0));
        assertEquals("70", tags.getValue().getTags().get(MetricConstant.TAG_PRIORITY));
        assertEquals(MetricConstant.PATH_BATCH, tags.getValue().getTags().get(MetricConstant.TAG_PATH));
    }
}
