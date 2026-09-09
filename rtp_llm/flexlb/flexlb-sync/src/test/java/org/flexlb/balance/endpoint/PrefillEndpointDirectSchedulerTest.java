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
    void directCapacityIsAtomicAndReusableAfterRollback(RoleType role) throws Exception {
        FlexlbConfig config = new FlexlbConfig();
        config.setScheduler(SchedulerConfig.direct());
        config.setDispatcher(DispatcherConfig.nonBatch());
        config.getDispatcher().setMaxInflightRequestsPerPrefillWorker(4);
        var runtime = EndpointTestSupport.requestRuntime();
        PrefillEndpoint endpoint = new PrefillEndpoint(
                EndpointTestSupport.workerStatus(role, "127.0.0.82", 8082, 9082),
                () -> config, EndpointTestSupport.routeStrategy(runtime), runtime.events(),
                mock(BatchSchedulerReporter.class),
                new org.flexlb.balance.scheduler.PlacementAvailability());
        endpoint.startGeneration();
        try (var executor = Executors.newFixedThreadPool(8)) {
            List<Future<PrefillState.ReservationResult<PrefillState.DirectRegistration>>> futures =
                    new ArrayList<>();
            for (long id = 1; id <= 64; id++) {
                String requestId = Long.toString(id);
                futures.add(executor.submit(() -> {
                    try (var pin = endpoint.tryPinGeneration()) {
                        return endpoint.registerDirectRequest(pin, requestId, 10L);
                    }
                }));
            }
            List<PrefillState.DirectRegistration> owned = new ArrayList<>();
            for (var future : futures) {
                var result = future.get();
                if (result.status() == PrefillState.CapacityStatus.ACQUIRED) {
                    owned.add(result.reservation());
                } else {
                    assertEquals(PrefillState.CapacityStatus.CAPACITY_FULL, result.status());
                }
            }
            assertEquals(4, owned.size());
            assertEquals(4, endpoint.admissionPendingRequestCount());
            owned.forEach(PrefillState.DirectRegistration::close);
            assertEquals(0, endpoint.admissionPendingRequestCount());
            try (var pin = endpoint.tryPinGeneration()) {
                var result = endpoint.registerDirectRequest(pin, "65", 10L);
                assertEquals(PrefillState.CapacityStatus.ACQUIRED, result.status());
                result.reservation().close();
                result.reservation().close();
            }
            assertEquals(0, endpoint.admissionPendingRequestCount());
        } finally {
            endpoint.close();
        }
    }

    @Test
    void directSchedulerCanConstructPrefillEndpoint() {
        FlexlbConfig config = new FlexlbConfig();
        config.setScheduler(SchedulerConfig.direct());
        config.setDispatcher(DispatcherConfig.nonBatch());

        EndpointTestSupport.TestRequestRuntime requestRuntime =
                EndpointTestSupport.requestRuntime();
        PrefillEndpoint endpoint = assertDoesNotThrow(() -> {
            PrefillEndpoint created = new PrefillEndpoint(
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

    private static WorkerStatus workerStatus() {
        WorkerStatus status = EndpointTestSupport.workerStatus(
                RoleType.PREFILL, "127.0.0.81", 8081, 9081);
        return status;
    }
}
