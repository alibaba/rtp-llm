package org.flexlb.balance.delivery;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.Delivery;
import org.flexlb.balance.scheduler.DeliveryDispatcher;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;

import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.CALLS_REAL_METHODS;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class DeliveryMetricsTest {
    @ParameterizedTest
    @CsvSource({"true,true", "true,false", "false,true", "false,false"})
    void missingWorkerRoleDoesNotDiscardDeliveryMetrics(boolean routeDelivery, boolean hasStatus) {
        var reporter = mock(DeliveryMetricsReporter.class);
        var endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getIp()).thenReturn("10.0.0.1");
        if (hasStatus) when(endpoint.getStatus()).thenReturn(mock(WorkerStatus.class));
        var route = mock(RequestRoute.class);
        when(route.prefillEp()).thenReturn(endpoint);
        when(route.priority()).thenReturn(50);
        var member = mock(Delivery.Member.class);
        when(member.route()).thenReturn(route);
        var dispatcher = mock(DeliveryDispatcher.class, CALLS_REAL_METHODS);
        ReflectionTestUtils.setField(dispatcher, "telemetry", reporter);
        if (routeDelivery) {
            ReflectionTestUtils.invokeMethod(
                    dispatcher, "reportRoutes", new Delivery.Member[] {member}, 1, 3);
            verify(reporter).reportBatcherQueueSize("PREFILL", "10.0.0.1", 3);
        } else {
            var batch = mock(Delivery.class);
            when(batch.batchId()).thenReturn(1L);
            when(batch.reason()).thenReturn("deadline");
            when(batch.remainingQueueDepth()).thenReturn(3);
            ReflectionTestUtils.invokeMethod(
                    dispatcher, "reportDelivery", batch, List.of(member), 100L);
            verify(reporter)
                    .reportBatchDelivery("PREFILL", "10.0.0.1", 1, "deadline", 3, 1, 0, 0, 100);
        }
        verify(reporter).reportBatchWaitTimeMs(eq("PREFILL"), eq("10.0.0.1"), anyLong(), eq(50));
    }
}
