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
import java.util.concurrent.atomic.AtomicReference;

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
        PrefillEndpoint endpoint = EndpointTestSupport.standalonePrefillEndpoint(
                EndpointTestSupport.workerStatus(role, "127.0.0.82", 8082, 9082),
                () -> config, EndpointTestSupport.routeStrategy(runtime), runtime.events(),
                mock(BatchSchedulerReporter.class),
                new org.flexlb.balance.scheduler.PlacementAvailability());
        endpoint.startGeneration();
        try (var executor = Executors.newFixedThreadPool(8)) {
            List<Future<PrefillState.ReservationResult<PrefillState.RouteReservation>>> futures =
                    new ArrayList<>();
            for (long id = 1; id <= 64; id++) {
                long requestId = id;
                futures.add(executor.submit(() -> {
                    try (var pin = endpoint.tryPinGeneration()) {
                        return endpoint.reserveUnqueuedRoute(pin, item(requestId, config), 10L);
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
                var full = endpoint.reserveUnqueuedRoute(pin, item(100L, config), 10L);
                assertEquals(PrefillState.CapacityStatus.CAPACITY_FULL, full.status(),
                        "four exact owners already consume the four-request limit");
            }
            owned.forEach(PrefillState.RouteReservation::close);
            assertEquals(0, endpoint.observedRequestCount());
            try (var pin = endpoint.tryPinGeneration()) {
                var result = endpoint.reserveUnqueuedRoute(pin, item(65L, config), 10L);
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
            PrefillEndpoint created = EndpointTestSupport.standalonePrefillEndpoint(
                    workerStatus(),
                    () -> config,
                    EndpointTestSupport.routeStrategy(requestRuntime),
                    requestRuntime.events(),
                    mock(BatchSchedulerReporter.class),
                new org.flexlb.balance.scheduler.PlacementAvailability());
            created.startGeneration();
            return created;
        });
        endpoint.close();
    }

    @Test
    void directLimitFollowsRequestSnapshotAfterConfigUpdate() {
        FlexlbConfig initial = org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig();
        initial.setScheduler(SchedulerConfig.direct());
        initial.setDispatcher(DispatcherConfig.nonBatch());
        initial.getDispatcher().setMaxInflightPerPrefillWorker(1);
        AtomicReference<FlexlbConfig> current = new AtomicReference<>(initial);
        var runtime = EndpointTestSupport.requestRuntime();
        PrefillEndpoint endpoint = EndpointTestSupport.standalonePrefillEndpoint(workerStatus(), current::get,
                EndpointTestSupport.routeStrategy(runtime), runtime.events(),
                mock(BatchSchedulerReporter.class), new org.flexlb.balance.scheduler.PlacementAvailability());
        endpoint.startGeneration();
        try {
            var first = reserve(endpoint, item(1L, initial));
            assertEquals(PrefillState.CapacityStatus.ACQUIRED, first.status());
            assertEquals(PrefillState.CapacityStatus.CAPACITY_FULL, reserve(endpoint, item(2L, initial)).status());

            FlexlbConfig increased = org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig();
            increased.setScheduler(SchedulerConfig.direct());
            increased.setDispatcher(DispatcherConfig.nonBatch());
            increased.getDispatcher().setMaxInflightPerPrefillWorker(2);
            current.set(increased);
            var second = reserve(endpoint, item(3L, increased));
            assertEquals(PrefillState.CapacityStatus.ACQUIRED, second.status());
            assertEquals(PrefillState.CapacityStatus.CAPACITY_FULL, reserve(endpoint, item(4L, increased)).status());
            first.reservation().close();
            second.reservation().close();
        } finally {
            endpoint.close();
        }
    }

    private static PrefillState.ReservationResult<PrefillState.RouteReservation> reserve(PrefillEndpoint endpoint,
                                                                                         org.flexlb.balance.scheduler.ScheduledRequest item) {
        try (var pin = endpoint.tryPinGeneration()) {
            return endpoint.reserveUnqueuedRoute(pin, item, 10L);
        }
    }

    private static org.flexlb.balance.scheduler.ScheduledRequest item(long requestId, FlexlbConfig config) {
        var item = mock(org.flexlb.balance.scheduler.ScheduledRequest.class);
        org.mockito.Mockito.when(item.requestId()).thenReturn(Long.toString(requestId));
        org.mockito.Mockito.when(item.seqLen()).thenReturn(128L);
        org.mockito.Mockito.when(item.ctx()).thenReturn(new org.flexlb.dao.BalanceContext(config));
        return item;
    }

    private static WorkerStatus workerStatus() {
        WorkerStatus status = EndpointTestSupport.workerStatus(
                RoleType.PREFILL, "127.0.0.81", 8081, 9081);
        return status;
    }
}
