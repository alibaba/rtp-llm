package org.flexlb.cache.kvcm;

import org.flexlb.util.metrics.FlexMetricTags;
import org.flexlb.util.metrics.FlexMetricType;
import org.flexlb.util.metrics.FlexMonitor;
import org.flexlb.util.metrics.FlexPriorityType;
import org.junit.jupiter.api.Test;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

/**
 * Checks KVCM metric registration and query observations.
 */
class KvcmMetricsReporterTest {

    private final FlexMonitor monitor = mock(FlexMonitor.class);
    private final KvcmMetricsReporter reporter = new KvcmMetricsReporter(monitor);

    @Test
    void reportsFinalQueryFailureQps() {
        reporter.init();
        reporter.reportQueryFailure();

        verify(monitor)
                .register(
                        "app.cache.kvcm.query.failure.qps",
                        FlexMetricType.QPS,
                        FlexPriorityType.PRECISE);
        verify(monitor).report("app.cache.kvcm.query.failure.qps", FlexMetricTags.of(), 1.0);
    }
}
