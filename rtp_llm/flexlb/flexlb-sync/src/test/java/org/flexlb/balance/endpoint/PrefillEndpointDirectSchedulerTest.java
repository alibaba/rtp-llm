package org.flexlb.balance.endpoint;

import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.SchedulerConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
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
        PrefillEndpoint endpoint = new PrefillEndpoint(
                EndpointTestSupport.workerStatus(role, "127.0.0.82", 8082, 9082),
                config, EndpointTestSupport.routeStrategy(runtime), runtime.events(),
                mock(BatchSchedulerReporter.class));
        endpoint.startGeneration();
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
            assertEquals(4, endpoint.observedRequestCount());
            try (var pin = endpoint.tryPinGeneration()) {
                var full = endpoint.reserveUnqueuedRoute(pin, item(100L), 10L);
                assertEquals(PrefillState.CapacityStatus.CAPACITY_FULL, full.status(),
                        "four exact owners already consume the four-request limit");
            }
            owned.forEach(PrefillState.RouteReservation::close);
            assertEquals(0, endpoint.observedRequestCount());
            try (var pin = endpoint.tryPinGeneration()) {
                var result = endpoint.reserveUnqueuedRoute(pin, item(65L), 10L);
                assertEquals(PrefillState.CapacityStatus.ACQUIRED, result.status());
                result.reservation().close();
                result.reservation().close();
            }
            assertEquals(0, endpoint.observedRequestCount());
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
            PrefillEndpoint created = new PrefillEndpoint(
                    workerStatus(),
                    config,
                    EndpointTestSupport.routeStrategy(requestRuntime),
                    requestRuntime.events(),
                    mock(BatchSchedulerReporter.class));
            created.startGeneration();
            return created;
        });
        endpoint.close();
    }

    private static org.flexlb.balance.scheduler.ScheduledRequest item(long requestId) {
        var item = mock(org.flexlb.balance.scheduler.ScheduledRequest.class);
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
