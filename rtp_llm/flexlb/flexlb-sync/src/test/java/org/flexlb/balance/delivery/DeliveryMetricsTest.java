package org.flexlb.balance.delivery;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.List;

import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class DeliveryMetricsTest {

    @ParameterizedTest
    @CsvSource({"true,true", "true,false", "false,true", "false,false"})
    void missingWorkerRoleDoesNotDiscardDeliveryMetrics(boolean routeDelivery, boolean hasStatus) {
        var monitor = mock(org.flexlb.metric.FlexMonitor.class);
        DeliveryMetricsReporter reporter = new DeliveryMetricsReporter(monitor);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        if (hasStatus) {
            WorkerStatus status = mock(WorkerStatus.class);
            when(status.getMetricIpPort()).thenReturn("10.0.0.1:8080");
            when(endpoint.getStatus()).thenReturn(status);
        }
        RequestRoute item = mock(RequestRoute.class);
        when(item.prefillEp()).thenReturn(endpoint);
        when(item.priority()).thenReturn(50);
        if (routeDelivery) {
            when(item.ctx()).thenReturn(mock(org.flexlb.balance.scheduler.RequestContext.class));
        }
        reporter.reportDelivery(routeDelivery ? 0L : 1L, "deadline", 3, List.of(item), 100L);

        String engineIp = hasStatus ? "10.0.0.1:8080" : "";
        var tags = org.flexlb.metric.FlexMetricTags.ofEngine(engineIp, "role", "PREFILL");
        verify(monitor).report(org.flexlb.constant.MetricConstant.BATCHER_QUEUE_SIZE, tags, 3.0);
        verify(monitor)
                .report(
                        eq(org.flexlb.constant.MetricConstant.ROUTING_QUEUE_WAIT_TIME_MS),
                        org.mockito.ArgumentMatchers.any(),
                        org.mockito.ArgumentMatchers.anyDouble());
    }
}
