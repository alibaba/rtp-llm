package org.flexlb.service.monitor;

import org.flexlb.constant.MetricConstant;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.anyString;
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

        verify(monitor, never()).report(anyString(), any(), anyDouble());
    }

    @Test
    void positivePriority_stillReported() {
        helper.reportAutoTpmRequestCount(50);
        helper.reportAutoTpmScheduleLatency(50, "success", 12L);
        helper.reportAutoTpmNormalPlacement(50);
        helper.reportAutoTpmQueueReject(30, 70);
        helper.reportAutoTpmRunningCancel(30, 70, "success");

        verify(monitor, times(5)).report(anyString(), any(), anyDouble());
    }
}
