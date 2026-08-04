package org.flexlb.service.monitor;

import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.Test;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

class RoutingQueueReporterTest {

    @Test
    void reportsCancellationAndRollbackSelectionMetrics() {
        FlexMonitor monitor = mock(FlexMonitor.class);
        RoutingQueueReporter reporter = new RoutingQueueReporter(monitor);

        reporter.init();
        reporter.reportRoutingCancelled();
        reporter.reportRoutingRollback("response_rollback", 2);

        verify(monitor).register("app.routing.cancel.qps", FlexMetricType.QPS, FlexPriorityType.PRECISE);
        verify(monitor).register("app.routing.rollback.qps", FlexMetricType.QPS, FlexPriorityType.PRECISE);
        verify(monitor).register("app.routing.rollback.worker.qps", FlexMetricType.QPS, FlexPriorityType.PRECISE);
        verify(monitor).report("app.routing.cancel.qps", FlexMetricTags.of(), 1.0);
        verify(monitor).report(
                "app.routing.rollback.qps", FlexMetricTags.of("reason", "response_rollback"), 1.0);
        verify(monitor).report(
                "app.routing.rollback.worker.qps", FlexMetricTags.of("reason", "response_rollback"), 2.0);
    }
}
