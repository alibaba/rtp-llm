package org.flexlb.service.monitor;

import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.flexlb.constant.MetricConstant.ROUTING_ROUTE_ATTEMPT_EXECUTION_TIME_MS;
import static org.mockito.Mockito.eq;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
class RoutingQueueReporterTest {

    @Mock
    private FlexMonitor monitor;

    @Test
    void registersAndReportsSingleRouteAttemptTime() {
        RoutingQueueReporter reporter = new RoutingQueueReporter(monitor);

        reporter.init();
        reporter.reportRouteAttemptExecutionMetric(37L);

        verify(monitor).register(
                eq(ROUTING_ROUTE_ATTEMPT_EXECUTION_TIME_MS),
                eq(FlexMetricType.GAUGE),
                eq(FlexPriorityType.PRECISE));
        verify(monitor).report(
                eq(ROUTING_ROUTE_ATTEMPT_EXECUTION_TIME_MS),
                eq(FlexMetricTags.of()),
                eq(37.0));
    }
}
