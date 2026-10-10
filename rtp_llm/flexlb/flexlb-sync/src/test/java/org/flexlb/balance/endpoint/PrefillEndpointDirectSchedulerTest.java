package org.flexlb.balance.endpoint;

import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.SchedulerConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;

class PrefillEndpointDirectSchedulerTest {

    @ParameterizedTest
    @EnumSource(value = RoleType.class, names = {"PREFILL", "PDFUSION"})
    void directRequestLimitIsAtomicAndReusableAfterRollback(RoleType role) throws Exception {
        FlexlbConfig config = org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig();
        config.setScheduler(SchedulerConfig.direct());
        config.setDispatcher(DispatcherConfig.nonBatch());
        config.getDispatcher().setMaxInflightPerPrefillWorker(4);
        var runtime = EndpointTestSupport.requestRuntime();
        PrefillEndpoint endpoint = EndpointTestSupport.prefill(EndpointTestSupport.workerStatus(role, "127.0.0.82", 8082, 9082), config, EndpointTestSupport.routeStrategy(runtime), org.flexlb.balance.scheduler.SchedulerTestSupport.repository(runtime.events()), mock(DeliveryMetricsReporter.class));
        try (var executor = Executors.newFixedThreadPool(8)) {
            List<Future<PrefillState.ReservationResult<PrefillState.RouteReservation>>> futures =
                    new ArrayList<>();
            for (long id = 1; id <= 64; id++) {
                long requestId = id;
                futures.add(executor.submit(() -> {
                    try (var pin = endpoint.tryPinGeneration()) {
                        return endpoint.reserveUnqueuedRoute(pin, item(requestId), 10L);
                    }
                }));
            }
            List<PrefillState.RouteReservation> owned = new ArrayList<>();
            for (var future : futures) {
                var result = future.get();
                if (result.status() == PrefillState.CapacityStatus.ACQUIRED) {
                    owned.add(result.reservation());
                } else {
                    assertEquals(PrefillState.CapacityStatus.CAPACITY_FULL, result.status());
                }
            }
            assertEquals(4, owned.size());
            assertEquals(4, endpoint.admissionSummary(0).occupiedRequests());
            try (var pin = endpoint.tryPinGeneration()) {
                var full = endpoint.reserveUnqueuedRoute(pin, item(100L), 10L);
                assertEquals(PrefillState.CapacityStatus.CAPACITY_FULL, full.status(),
                        "four exact owners already consume the four-request limit");
            }
            owned.forEach(endpoint::rollbackReservation);
            assertEquals(0, endpoint.admissionSummary(0).occupiedRequests());
            try (var pin = endpoint.tryPinGeneration()) {
                var result = endpoint.reserveUnqueuedRoute(pin, item(65L), 10L);
                assertEquals(PrefillState.CapacityStatus.ACQUIRED, result.status());
                endpoint.rollbackReservation(result.reservation());
                endpoint.rollbackReservation(result.reservation());
            }
            assertEquals(0, endpoint.admissionSummary(0).occupiedRequests());
        } finally {
            endpoint.close();
        }
    }

    @Test
    void directSchedulerCanConstructPrefillEndpoint() {
        FlexlbConfig config = org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig();
        config.setScheduler(SchedulerConfig.direct());
        config.setDispatcher(DispatcherConfig.nonBatch());

        EndpointTestSupport.TestRequestRuntime requestRuntime =
                EndpointTestSupport.requestRuntime();
        PrefillEndpoint endpoint = assertDoesNotThrow(() -> {
            PrefillEndpoint created = EndpointTestSupport.prefill(workerStatus(), config, EndpointTestSupport.routeStrategy(requestRuntime), org.flexlb.balance.scheduler.SchedulerTestSupport.repository(requestRuntime.events()), mock(DeliveryMetricsReporter.class));
            return created;
        });
        endpoint.close();
    }

    private static org.flexlb.balance.scheduler.RequestRoute item(long requestId) {
        var item = mock(org.flexlb.balance.scheduler.RequestRoute.class);
        org.mockito.Mockito.when(item.requestId()).thenReturn(requestId);
        org.mockito.Mockito.when(item.seqLen()).thenReturn(128L);
        return item;
    }

    private static WorkerStatus workerStatus() {
        WorkerStatus status = EndpointTestSupport.workerStatus(
                RoleType.PREFILL, "127.0.0.81", 8081, 9081);
        return status;
    }
}
