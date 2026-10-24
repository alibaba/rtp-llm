package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.flexlb.enums.PrefillQueueEnqueueResult.ACCEPTED;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class DecodeCapacityBypassTest {
    @Test
    @Timeout(value = 15, unit = TimeUnit.SECONDS)
    void realNonBatchDeliveryCommitsTailAndKeepsBlockedHeadReservations() throws Exception {
        FlexlbConfig config = SchedulingTestConfig.newConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.useSingleDecision(config);
        SchedulingTestConfig.useNonBatchDispatcher(config);
        config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(1L);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var reporter = mock(BatchSchedulerReporter.class);
        var requestReporter = mock(RequestSchedulerReporter.class);
        var requests = new RequestRegistry(service, reporter, requestReporter);
        var projector = new EndpointEventProjector(requests);
        var queueCapacity = new QueueCapacity();
        var endpoints = new EndpointRegistry(service, projector, reporter,
                new RouteDeliveryStrategy(requests, new DeliveryMetrics(reporter)),
                new PlacementAvailability(), queueCapacity);
        var runtime = new SchedulerRuntime(requests, endpoints, reporter, requestReporter,
                service, queueCapacity, new ModelMetaConfig());
        DecodeEndpoint blocked = new DecodeEndpoint(worker(RoleType.DECODE, "127.0.0.2"), projector);
        DecodeEndpoint ready = new DecodeEndpoint(worker(RoleType.DECODE, "127.0.0.3"), projector);
        try {
            WorkerStatus worker = worker(RoleType.PREFILL, "127.0.0.1");
            PrefillEndpoint prefill;
            worker.lock.lock();
            try {
                prefill = (PrefillEndpoint) endpoints.publishPreparedEndpoint(worker.getIpPort(), worker,
                        worker.prepareNewStatus(worker.freezeStatusResponse(status(RoleType.PREFILL)))).endpoint();
            } finally {
                worker.lock.unlock();
            }
            observe(blocked);
            observe(ready);
            var capacity = new DecodeEndpoint.AdmissionCapacity(1L, 100L);
            DecodeEndpoint.ReservationHandle occupying;
            try (var pin = blocked.tryPinGeneration()) {
                occupying = blocked.reserve(pin, "occupying", 16L, 16L, 50, capacity);
            }
            var permit = blocked.acquireDispatchPermit(occupying, capacity);
            assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.ACQUIRED, permit.status());
            ScheduledRequest head = enqueue(config, requests, prefill, blocked, "head");
            ScheduledRequest tail = enqueue(config, requests, prefill, ready, "tail");
            tail.future().get(5, TimeUnit.SECONDS);
            assertFalse(head.future().isDone());
            assertEquals(List.of(head), prefill.captureQueueSnapshot().items());
            assertNotNull(head.publishedRouteReservation());
            assertNotNull(blocked.reservationHandle("head"));
            assertEquals(1, ready.routingView().engineCapacityUsed());
            assertTrue(permit.permit().release());
            blocked.release(occupying, DecodeEndpoint.ReleaseReason.LOCAL_ROLLBACK);
            head.future().get(5, TimeUnit.SECONDS);
            assertEquals(0, prefill.queuedRequestCount());
            assertEquals(1, blocked.routingView().engineCapacityUsed());
        } finally {
            runtime.shutdown();
            blocked.close();
            ready.close();
        }
    }

    private static ScheduledRequest enqueue(FlexlbConfig config, RequestRegistry requests, PrefillEndpoint prefill,
                                            DecodeEndpoint decode, String requestId) {
        var context = RequestLifecycleTestSupport.context(config, requestId);
        var future = requests.register(context);
        DecodeEndpoint.ReservationHandle reservation;
        try (var pin = decode.tryPinGeneration()) {
            reservation = decode.reserve(pin, requestId, 16L, 16L, 50);
        }
        var item = new ScheduledRequest(context, future, new Response(), metadata(prefill), metadata(decode),
                prefill, decode, reservation, System.currentTimeMillis());
        try (var admission = requests.claimAdmissionHandle(requestId, future);
             var pin = prefill.tryPinGeneration()) {
            assertNotNull(admission);
            assertTrue(requests.commitItemForPublication(item, () -> prefill.offerPinnedResult(pin, item) == ACCEPTED));
        }
        return item;
    }

    private static ServerStatus metadata(WorkerEndpoint endpoint) {
        var metadata = new ServerStatus();
        metadata.setRole(endpoint.getStatus().getRole());
        metadata.setServerIp(endpoint.getStatus().getIp());
        metadata.setGrpcPort(8081);
        return metadata;
    }

    private static WorkerStatus worker(RoleType role, String ip) {
        return WorkerStatus.createDiscovered(role, "g1", ip, 8080, 8081, "test");
    }

    private static WorkerStatusResponse status(RoleType role) {
        var response = new WorkerStatusResponse();
        response.setRole(role);
        response.setAlive(true);
        response.setStatusVersion(1L);
        response.setLatestFinishedVersion(0L);
        response.setRunningTaskInfo(Map.of());
        response.setRunningQueryLen(0L);
        response.setFinishedTaskInfo(Map.of());
        response.setTotalKvCacheTokens(10_000L);
        response.setAvailableKvCacheTokens(10_000L);
        return response;
    }

    private static void observe(WorkerEndpoint endpoint) {
        WorkerStatus worker = endpoint.getStatus();
        Runnable projection;
        worker.lock.lock();
        try {
            projection = endpoint.applyPreparedStatus(worker,
                    worker.prepareNewStatus(worker.freezeStatusResponse(status(worker.getRole()))));
        } finally {
            worker.lock.unlock();
        }
        projection.run();
    }
}
