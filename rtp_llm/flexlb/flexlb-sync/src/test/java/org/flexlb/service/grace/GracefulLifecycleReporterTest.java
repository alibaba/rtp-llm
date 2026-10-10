package org.flexlb.service.grace;

import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.flexlb.constant.MetricConstant.GRACEFUL_LIFECYCLE_EVENT;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;

class GracefulLifecycleReporterTest {
    @ParameterizedTest
    @CsvSource({
            "HEALTH_CHECK_OFFLINE, health_check_offline",
            "ZK_NODE_OFFLINE, zk_node_offline",
            "SHUTDOWN_COMPLETE, shutdown_complete",
            "ZK_NODE_ONLINE, zk_node_online",
            "WARMER_COMPLETE, warmer_complete",
            "ONLINE_COMPLETE, online_complete"
    })
    void preservesPublishedDurationTags(GracefulLifecycleReporter.Event event, String type) {
        var monitor = mock(FlexMonitor.class);
        var reporter = new GracefulLifecycleReporter(monitor);
        reporter.reportDuration(event, 17L);
        verify(monitor).register(GRACEFUL_LIFECYCLE_EVENT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).report(GRACEFUL_LIFECYCLE_EVENT,
                FlexMetricTags.of("type", type, "duration_ms", "17"), 1);
        verifyNoMoreInteractions(monitor);
    }

    @Test
    void processReadinessHasNoDurationTag() {
        var monitor = mock(FlexMonitor.class);
        new GracefulLifecycleReporter(monitor).reportProcessOk();
        verify(monitor).register(GRACEFUL_LIFECYCLE_EVENT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).report(GRACEFUL_LIFECYCLE_EVENT, FlexMetricTags.of("type", "process_ok"), 1);
        verifyNoMoreInteractions(monitor);
    }
}
