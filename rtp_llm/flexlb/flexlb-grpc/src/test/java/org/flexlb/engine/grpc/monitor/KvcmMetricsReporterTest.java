package org.flexlb.engine.grpc.monitor;

import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.Test;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

class KvcmMetricsReporterTest {

    private final FlexMonitor monitor = mock(FlexMonitor.class);
    private final KvcmMetricsReporter reporter = new KvcmMetricsReporter(monitor);

    @Test
    void reportsFinalQueryFailureQps() {
        reporter.init();
        reporter.reportQueryFailure();

        verify(monitor).register("app.cache.kvcm.query.failure.qps", FlexMetricType.QPS, FlexPriorityType.PRECISE);
        verify(monitor).report("app.cache.kvcm.query.failure.qps", FlexMetricTags.of(), 1.0);
    }
}
