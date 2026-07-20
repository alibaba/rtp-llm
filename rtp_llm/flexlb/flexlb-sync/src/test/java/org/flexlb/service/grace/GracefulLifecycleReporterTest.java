package org.flexlb.service.grace;

import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class GracefulLifecycleReporterTest {

    @Test
    void processOkRegistersLifecycleMetricAndReportsTags() {
        CapturingFlexMonitor monitor = new CapturingFlexMonitor();
        GracefulLifecycleReporter reporter = new GracefulLifecycleReporter(monitor);

        reporter.reportProcessOk();

        assertEquals(FlexMetricType.GAUGE, monitor.registeredMetricType);
        assertEquals(FlexPriorityType.PRECISE, monitor.registeredPriorityType);
        assertEquals(
                FlexMetricTags.of("type", "process_ok", "duration_ms", "0"),
                monitor.reportedTags);
    }

    private static final class CapturingFlexMonitor implements FlexMonitor {
        private FlexMetricType registeredMetricType;
        private FlexPriorityType registeredPriorityType;
        private FlexMetricTags reportedTags;

        @Override
        public void register(String metricName, FlexMetricType metricType) {
        }

        @Override
        public void register(String metricName, FlexMetricType metricType, FlexPriorityType priorityType) {
            registeredMetricType = metricType;
            registeredPriorityType = priorityType;
        }

        @Override
        public void register(String metricName, FlexMetricType metricType, int statisticsType) {
        }

        @Override
        public void report(String metricName, double value) {
        }

        @Override
        public void report(String metricName, FlexMetricTags metricsTags, double value) {
            reportedTags = metricsTags;
        }
    }
}
