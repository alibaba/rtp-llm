package org.flexlb.balance.delivery;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.List;

import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class DeliveryMetricsTest {

    @ParameterizedTest
    @CsvSource({"true,true", "true,false", "false,true", "false,false"})
    void missingWorkerRoleDoesNotDiscardDeliveryMetrics(boolean routeDelivery, boolean hasStatus) {
        BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        if (hasStatus) {
            WorkerStatus status = mock(WorkerStatus.class);
            when(status.getMetricIpPort()).thenReturn("10.0.0.1:8080");
            when(endpoint.getStatus()).thenReturn(status);
        }
        ScheduledRequest item = mock(ScheduledRequest.class);
        when(item.prefillEp()).thenReturn(endpoint);
        when(item.priority()).thenReturn(50);
        if (routeDelivery) {
            when(item.ctx()).thenReturn(mock(BalanceContext.class));
        }
        DeliveryMetrics metrics = new DeliveryMetrics(reporter);

        if (routeDelivery) {
            metrics.routesDelivered(3, List.of(item));
        } else {
            metrics.batchDispatched(1L, "deadline", 3, List.of(item), 100L);
        }

        String engineIp = hasStatus ? "10.0.0.1:8080" : "";
        verify(reporter).reportBatcherQueueSize("PREFILL", engineIp, 3);
        verify(reporter).reportBatchWaitTimeMs(eq("PREFILL"), eq(engineIp), anyLong(), eq(50));
    }
}
