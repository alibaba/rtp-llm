package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.config.FlexlbConfig;
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
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

/**
 * Checks direct Prefill admission enforces a shared limit atomically and returns capacity after rollback.
 */
class PrefillEndpointDirectSchedulerTest {

    @ParameterizedTest
    @EnumSource(
            value = RoleType.class,
            names = {"PREFILL", "PDFUSION"})
    void directRequestLimitIsAtomicAndReusableAfterRollback(RoleType role) throws Exception {
        FlexlbConfig config = org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig();
        config.setScheduler(ConfigTestSupport.direct());
        config.setDispatcher(ConfigTestSupport.nonBatch());
        config.getDispatcher().setMaxInflightPerPrefillWorker(4);
        var runtime = EndpointTestSupport.requestRuntime();
        PrefillEndpoint endpoint =
                EndpointTestSupport.prefill(
                        EndpointTestSupport.workerStatus(role, "127.0.0.82", 8082, 9082),
                        config,
                        EndpointTestSupport.routeStrategy(runtime),
                        org.flexlb.balance.scheduler.SchedulerTestSupport.repository(
                                runtime.events()),
                        mock(DeliveryMetricsReporter.class));
        try (var executor = Executors.newFixedThreadPool(8)) {
            List<Future<CapacityBoundary.Attempt<PrefillEndpoint.RequestRecord>>> futures =
                    new ArrayList<>();
            for (long id = 1; id <= 64; id++) {
                long requestId = id;
                futures.add(
                        executor.submit(
                                () -> {
                                    {
                                        var selectedEndpoint = endpoint;

                                        return org.flexlb.balance.scheduler.SelectionTestSupport
                                                .registerPrefill(endpoint, item(requestId), 10L);
                                    }
                                }));
            }
            List<PrefillEndpoint.RequestRecord> owned = new ArrayList<>();
            for (var future : futures) {
                var result = future.get();
                if (result.accepted()) {
                    owned.add(result.value());
                } else {
                    assertEquals(
                            CapacityBoundary.Resource.PREFILL_REQUEST,
                            result.boundary().resource());
                }
            }
            assertEquals(4, owned.size());
            assertEquals(4, endpoint.admissionSummary(0, endpoint.requestLimit).occupiedRequests());
            {
                var selectedEndpoint = endpoint;

                var full =
                        org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                                endpoint, item(100L), 10L);
                assertEquals(
                        CapacityBoundary.Resource.PREFILL_REQUEST,
                        full.boundary().resource(),
                        "four exact owners already consume the four-request limit");
            }
            owned.forEach(reservation -> endpoint.rollback(reservation));
            assertEquals(0, endpoint.admissionSummary(0, endpoint.requestLimit).occupiedRequests());
            {
                var selectedEndpoint = endpoint;

                var result =
                        org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                                endpoint, item(65L), 10L);
                assertTrue(result.accepted());
                endpoint.rollback(result.value());
                endpoint.rollback(result.value());
            }
            assertEquals(0, endpoint.admissionSummary(0, endpoint.requestLimit).occupiedRequests());
        } finally {
            endpoint.close();
        }
    }

    @Test
    void directSchedulerCanConstructPrefillEndpoint() {
        FlexlbConfig config = org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig();
        config.setScheduler(ConfigTestSupport.direct());
        config.setDispatcher(ConfigTestSupport.nonBatch());

        EndpointTestSupport.TestRequestRuntime requestRuntime =
                EndpointTestSupport.requestRuntime();
        PrefillEndpoint endpoint =
                assertDoesNotThrow(
                        () -> {
                            PrefillEndpoint created =
                                    EndpointTestSupport.prefill(
                                            workerStatus(),
                                            config,
                                            EndpointTestSupport.routeStrategy(requestRuntime),
                                            org.flexlb.balance.scheduler.SchedulerTestSupport
                                                    .repository(requestRuntime.events()),
                                            mock(DeliveryMetricsReporter.class));
                            return created;
                        });
        endpoint.close();
    }

    private static org.flexlb.balance.scheduler.RouteResult item(long requestId) {
        var item = mock(org.flexlb.balance.scheduler.RouteResult.class);
        org.mockito.Mockito.when(item.requestId()).thenReturn(String.valueOf(requestId));
        org.mockito.Mockito.when(item.seqLen()).thenReturn(128L);
        return item;
    }

    private static WorkerStatus workerStatus() {
        WorkerStatus status =
                EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.81", 8081, 9081);
        return status;
    }
}
