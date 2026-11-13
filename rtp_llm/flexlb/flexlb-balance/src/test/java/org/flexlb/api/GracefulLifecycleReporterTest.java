package org.flexlb.api;

import org.flexlb.util.metrics.FlexMetricTags;
import org.flexlb.util.metrics.FlexMetricType;
import org.flexlb.util.metrics.FlexMonitor;
import org.flexlb.util.metrics.FlexPriorityType;
import org.flexlb.worker.status.ServiceMetricsReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import static org.flexlb.util.metrics.MetricConstant.GRACEFUL_LIFECYCLE_EVENT;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;

/**
 * Checks that monitoring failures cannot interrupt application lifecycle reporting.
 */
class GracefulLifecycleReporterTest {
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void monitoringFailureCannotInterruptLifecycle(boolean error) {
        var monitor = mock(FlexMonitor.class);
        Throwable failure =
                error
                        ? new AssertionError("monitor failed")
                        : new IllegalStateException("monitor failed");
        doThrow(failure)
                .when(monitor)
                .register(GRACEFUL_LIFECYCLE_EVENT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        var reporter =
                assertDoesNotThrow(
                        () ->
                                new ServiceMetricsReporter(
                                        monitor,
                                        org.mockito.Mockito.mock(
                                                org.flexlb.worker.endpoint.EndpointStats.class)));
        var tags = FlexMetricTags.of("type", "shutdown_complete", "duration_ms", "17");
        doThrow(failure).when(monitor).report(GRACEFUL_LIFECYCLE_EVENT, tags, 1);
        assertDoesNotThrow(
                () -> reporter.reportDuration(ServiceMetricsReporter.Event.SHUTDOWN_COMPLETE, 17));
        reporter.reportProcessOk();
        verify(monitor)
                .report(GRACEFUL_LIFECYCLE_EVENT, FlexMetricTags.of("type", "process_ok"), 1);
    }

    @ParameterizedTest
    @CsvSource({
        "HEALTH_CHECK_OFFLINE, health_check_offline",
        "ZK_NODE_OFFLINE, zk_node_offline",
        "SHUTDOWN_COMPLETE, shutdown_complete",
        "ZK_NODE_ONLINE, zk_node_online",
        "WARMER_COMPLETE, warmer_complete",
        "ONLINE_COMPLETE, online_complete"
    })
    void preservesPublishedDurationTags(ServiceMetricsReporter.Event event, String type) {
        var monitor = mock(FlexMonitor.class);
        var reporter =
                new ServiceMetricsReporter(
                        monitor,
                        org.mockito.Mockito.mock(org.flexlb.worker.endpoint.EndpointStats.class));
        reporter.reportDuration(event, 17L);
        verify(monitor)
                .register(GRACEFUL_LIFECYCLE_EVENT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor)
                .report(
                        GRACEFUL_LIFECYCLE_EVENT,
                        FlexMetricTags.of("type", type, "duration_ms", "17"),
                        1);
        verifyNoMoreInteractions(monitor);
    }

    @Test
    void processReadinessHasNoDurationTag() {
        var monitor = mock(FlexMonitor.class);
        new ServiceMetricsReporter(
                        monitor,
                        org.mockito.Mockito.mock(org.flexlb.worker.endpoint.EndpointStats.class))
                .reportProcessOk();
        verify(monitor)
                .register(GRACEFUL_LIFECYCLE_EVENT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor)
                .report(GRACEFUL_LIFECYCLE_EVENT, FlexMetricTags.of("type", "process_ok"), 1);
        verifyNoMoreInteractions(monitor);
    }
}
