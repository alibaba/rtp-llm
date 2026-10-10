package org.flexlb.balance.strategy;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.balance.scheduler.DeliveryStrategy;
import org.flexlb.balance.scheduler.DeliveryTransaction;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.scheduler.AbstractRequestScheduler;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.RouteDeliveryStrategy;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.mockito.Mockito;
import java.util.List;
import java.util.Map;

/** Package-local fixtures built only from the frozen endpoint-facing ports. */
final class StrategyTestSupport {

    private StrategyTestSupport() {
    }

    static FlexlbConfig config() {
        FlexlbConfig config = new FlexlbConfig();
        config.getRequestLifecycle().getRequest().setTimeoutMs(60_000L);
        config.getRequestLifecycle().getDecision().setLifetime(2.0);
        return config;
    }

    static EndpointRegistry endpointRegistry(ConfigService configService) {
        TestRequestRuntime runtime = new TestRequestRuntime();
        DeliveryStrategy delivery = parkedDelivery(runtime.requests());
        return new EndpointRegistry(configService, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(runtime.events()), Mockito.mock(DeliveryMetricsReporter.class), delivery, new PlacementAvailability());
    }

    static WorkerStatus workerStatus(
            RoleType role,
            String group,
            String ip,
            int httpPort,
            int grpcPort,
            boolean alive,
            long availableKv,
            long totalKv) {
        WorkerStatus status = WorkerStatus.createDiscovered(
                role, group, ip, httpPort, grpcPort, "test-site");
        WorkerStatusResponse response = response(
                role, alive, availableKv, totalKv,
                Math.max(1L, status.appliedStatusCursor().statusVersion() + 1L));
        publish(status, response);
        return status;
    }

    static WorkerStatusResponse response(
            RoleType role,
            boolean alive,
            long availableKv,
            long totalKv,
            long statusVersion) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(role);
        response.setAlive(alive);
        response.setAvailableKvCacheTokens(availableKv);
        response.setTotalKvCacheTokens(totalKv);
        response.setStatusVersion(statusVersion);
        response.setLatestFinishedVersion(0L);
        return response;
    }

    static void publish(
            WorkerStatus status, WorkerStatusResponse response) {
        status.lock.lock();
        try {
            WorkerStatus.PreparedStatus prepared = status.prepareNewStatus(
                    status.freezeStatusResponse(response));
            status.publishPreparedStatus(prepared);
            status.recordSuccessfulPoll(response.isAlive());
        } finally {
            status.lock.unlock();
        }
    }

    static WorkerEndpoint publishEndpoint(
            EndpointRegistry registry,
            RoleType role,
            String address,
            WorkerStatus source) {
        WorkerStatus status = WorkerStatus.createDiscovered(
                role,
                source.getGroup(),
                source.getIp(),
                source.getPort(),
                source.getGrpcPort(),
                source.getSite());
        WorkerStatus.StatusObservation observation =
                status.bindStatusObservation(
                        source.committedEngineObservation(),
                        true,
                        1L,
                        0L,
                        Map.of());
        status.lock.lock();
        try {
            WorkerStatus.PreparedStatus prepared =
                    status.prepareNewStatus(observation);
            return registry.publishPreparedEndpoint(
                    address, status, prepared);
        } finally {
            status.lock.unlock();
        }
    }

    static boolean offer(
            PrefillEndpoint endpoint, RequestRoute exactItem) {
        org.flexlb.balance.scheduler.SchedulerTestSupport.bindEndpointOwner(endpoint, exactItem);
        try (WorkerEndpoint.GenerationPin pin =
                     endpoint.tryPinGeneration()) {
            return pin != null && endpoint.offerPinned(pin, exactItem, org.flexlb.balance.scheduler.QueueExecutionSettings.capture(exactItem.ctx().getConfig()));
        }
    }

    static PrefillState.CommittedHandoff commitBatch(
            PrefillEndpoint endpoint,
            long batchId,
            long predictedMs,
            List<? extends RequestRoute> exactItems) {
        if (exactItems.isEmpty()) {
            throw new IllegalArgumentException(
                    "committed batch requires at least one item");
        }
        List<RequestRoute> items = List.copyOf(exactItems);
        PrefillState.ReservationResult<PrefillState.BatchReservation> result =
                endpoint.reserveBatch(
                        items.getFirst(), batchId, Integer.MAX_VALUE);
        if (result.status() != PrefillState.CapacityStatus.ACQUIRED) {
            throw new IllegalStateException(
                    "batch reservation rejected: " + result.status());
        }
        {
            PrefillState.BatchReservation reservation =
                     result.reservation();
            try (var preparationReservation = org.flexlb.balance.endpoint.EndpointTestSupport.preparation(reservation)) {
                PrefillState state = (PrefillState) org.springframework.test.util.ReflectionTestUtils.getField(endpoint, "prefillState");
                state.ownershipLock().lock();
                try {
                    return reservation.commitLocked(items, predictedMs, null, System.currentTimeMillis());
                } finally {
                    state.ownershipLock().unlock();
                }
            }
        }
    }

    static void setCacheStatus(
            WorkerStatus status,
            long blockSize,
            long availableKv) {
        CacheStatus cache = new CacheStatus();
        cache.setBlockSize(blockSize);
        cache.setAvailableKvCache(availableKv);
        WorkerStatusResponse response = response(
                status.getRole(), status.pollHealth().reportedAlive(),
                status.getAvailableKvCacheTokens(),
                status.getTotalKvCacheTokens(),
                Math.max(1L, status.appliedStatusCursor().statusVersion() + 1L));
        response.setCacheStatus(cache);
        publish(status, response);
    }

    private static final class TestRequestRuntime {
        private final AbstractRequestScheduler requests =
                Mockito.mock(AbstractRequestScheduler.class);
        private final AbstractRequestScheduler events =
                Mockito.mock(AbstractRequestScheduler.class);

        AbstractRequestScheduler requests() {
            return requests;
        }

        AbstractRequestScheduler events() {
            return events;
        }
    }

    /** Keep selection-owned queues parked without replacing admission internals. */
    private static DeliveryStrategy parkedDelivery(AbstractRequestScheduler requests) {
        RouteDeliveryStrategy route = new RouteDeliveryStrategy(NOOP_METRICS);
        DeliveryStrategy delivery = Mockito.mock(DeliveryStrategy.class);
        Mockito.when(delivery.projectionPolicy())
                .thenReturn(route.projectionPolicy());
        Mockito.when(delivery.newGroupPredictor(Mockito.any()))
                .thenAnswer(invocation -> route.newGroupPredictor(invocation.getArgument(0)));
        Mockito.when(delivery.prepare(
                        Mockito.anyList(), Mockito.any(), Mockito.any()))
                .thenAnswer(invocation -> {
                    List<RequestRoute> candidates = invocation.getArgument(0);
                    DeliveryTransaction transaction =
                            Mockito.mock(DeliveryTransaction.class);
                    Mockito.when(transaction.items()).thenReturn(List.of());
                    Mockito.when(transaction.blockedItem())
                            .thenReturn(candidates.getFirst());
                    Mockito.when(transaction.blockedResult())
                            .thenReturn(PARKED_BOUNDARY);
                    return transaction;
                });
        return delivery;
    }

    private static final CapacityBoundary PARKED_BOUNDARY =
            CapacityBoundary.unavailable(
                    new CapacityBoundary.Availability() {
                        @Override
                        public boolean isAvailable() {
                            return false;
                        }

                        @Override
                        public void addListener(Runnable listener) {
                        }

                        @Override
                        public void removeListener(Runnable listener) {
                        }
                    },
                    new RouteProjection.AdmissionBlockSemantics(
                            "test delivery parked",
                            RouteProjection.AfterProbeAdmission.BLOCKED,
                            "test delivery parked",
                            RoleType.PREFILL));

    private static final DeliveryMetricsReporter NOOP_METRICS =
            Mockito.mock(DeliveryMetricsReporter.class);
}
