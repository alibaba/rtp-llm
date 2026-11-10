package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;

import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Verifies both delivery modes report metrics when worker status or its role is unavailable.
 */
class DeliveryMetricsTest {
    @ParameterizedTest
    @CsvSource({"true,true", "true,false", "false,true", "false,false"})
    void missingWorkerRoleDoesNotDiscardDeliveryMetrics(boolean routeDelivery, boolean hasStatus) {
        var reporter = mock(DeliveryMetricsReporter.class);
        var endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getIp()).thenReturn("10.0.0.1");
        when(endpoint.queueDepth()).thenReturn(3);
        if (hasStatus) when(endpoint.getStatus()).thenReturn(mock(WorkerStatus.class));
        var route = mock(RouteResult.class);
        when(route.prefillEndpoint()).thenReturn(endpoint);
        when(route.priority()).thenReturn(50);
        var member = mock(Delivery.Member.class);
        when(member.route()).thenReturn(route);
        var requests = mock(RequestRepository.class);
        if (routeDelivery) {
            ReflectionTestUtils.invokeMethod(
                    new DeliveryDispatcher(requests, null, null, null, reporter),
                    "reportRoutes",
                    List.of(member));
            verify(reporter).reportBatcherQueueSize("PREFILL", "10.0.0.1", 3);
        } else {
            var batch = mock(Delivery.class);
            ReflectionTestUtils.setField(batch, "batchId", 1L);
            ReflectionTestUtils.setField(batch, "reason", "deadline");
            ReflectionTestUtils.invokeMethod(
                    new DeliveryDispatcher(requests, null, null, null, reporter),
                    "reportDelivery",
                    batch,
                    List.of(member),
                    100L);
            verify(reporter)
                    .reportBatchDelivery("PREFILL", "10.0.0.1", 1, "deadline", 3, 1, 0, 0, 100);
        }
        verify(reporter).reportBatchWaitTimeMs(eq("PREFILL"), eq("10.0.0.1"), anyLong(), eq(50));
    }
}
