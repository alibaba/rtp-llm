package org.flexlb.balance.strategy;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.scheduler.AbstractRequestScheduler;
import org.flexlb.balance.scheduler.DeliveryDispatcher;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.master.WorkerStatusTestSupport;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.mockito.Mockito;

import java.util.List;
import java.util.Map;

/** Package-local fixtures built only from the frozen endpoint-facing ports. */
final class StrategyTestSupport {

    private StrategyTestSupport() {
    }

    static RequestRoute scope(org.flexlb.dao.loadbalance.Request request) {
        var context = new org.flexlb.balance.scheduler.RequestContext(config());
        context.data().setRequest(request);
        return new RequestRoute(context);
    }

    static org.flexlb.balance.PlacementResult<RequestRoute, RoleType> select(
            org.flexlb.dao.loadbalance.Request request,
            java.util.function.Function<RequestRoute, org.flexlb.balance.PlacementResult<org.flexlb.dao.loadbalance.ServerStatus, RoleType>> action) {
        RequestRoute route = scope(request);
        try {
            var result = action.apply(route);
            if (result.status() == org.flexlb.balance.PlacementResult.Status.SUCCESS) {
                return org.flexlb.balance.PlacementResult.success(route);
            }
            route.close();
            return new org.flexlb.balance.PlacementResult<>(result.status(), null, result.failure(), result.blocker(), result.diagnostics());
        } catch (RuntimeException | Error failure) {
            org.flexlb.util.Failures.append(failure, org.flexlb.util.Failures.close(route));
            throw failure;
        }
    }

    static org.flexlb.balance.PlacementResult<RequestRoute, RoleType> select(DecodeSelector selector,
            org.flexlb.dao.loadbalance.Request request, org.flexlb.dao.SchedulingMetadata metadata, String group) {
        return select(request, route -> selector.select(request, metadata, group, route));
    }

    static org.flexlb.balance.PlacementResult<RequestRoute, RoleType> select(CostBasedPrefillStrategy selector,
            org.flexlb.dao.loadbalance.Request request, org.flexlb.dao.SchedulingMetadata metadata, RoleType role, String group) {
        return select(request, route -> selector.select(request, metadata, role, group, route));
    }

    static RequestRoute select(VitWorkerSelector selector, long id, RoleType role, String group) {
        RequestRoute route = scope(org.flexlb.dao.loadbalance.Request.builder().requestId(id).build());
        try {
            if (selector.select(id, role, group, route) != null) { return route; }
            route.close();
            return null;
        } catch (RuntimeException | Error failure) {
            org.flexlb.util.Failures.append(failure, org.flexlb.util.Failures.close(route));
            throw failure;
        }
    }

    static org.flexlb.dao.loadbalance.ServerStatus vit(RequestRoute route) {
        return org.flexlb.dao.loadbalance.ServerStatus.copyOf(
                (org.flexlb.dao.loadbalance.ServerStatus) org.springframework.test.util.ReflectionTestUtils.getField(route, "vit"));
    }

    static WorkerEndpoint.GenerationPin decodePin(RequestRoute route) {
        return org.springframework.test.util.ReflectionTestUtils.invokeMethod(route, "decodePin");
    }

    static FlexlbConfig config() {
        FlexlbConfig config = new FlexlbConfig();
        config.getRequestLifecycle().getRequest().setTimeoutMs(60_000L);
        config.getRequestLifecycle().getDecision().setLifetime(2.0);
        return config;
    }

    static EndpointRegistry endpointRegistry(ConfigService configService) {
        var requests = new org.flexlb.balance.scheduler.RequestRepository();
        DeliveryDispatcher delivery = parkedDelivery();
        return new EndpointRegistry(configService, requests, Mockito.mock(DeliveryMetricsReporter.class), delivery, new PlacementAvailability());
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
                    WorkerStatusTestSupport.freezeStatusResponse(status, response));
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
            return pin != null && endpoint.offerPinned(pin, exactItem, org.flexlb.balance.scheduler.QueueExecutionSettings.capture(exactItem.ctx().data().getConfig()));
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
                PrefillState state = (PrefillState) org.springframework.test.util.ReflectionTestUtils.getField(endpoint, "ownership");
                state.lock.lock();
                try {
                    return state.commitLocked(new PrefillState.CommitPlan(PrefillState.CommitKind.BATCH, items, List.of(), reservation, new long[]{predictedMs}, null, System.currentTimeMillis()), null);
                } finally {
                    state.lock.unlock();
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

    /** Keep selection-owned queues parked without replacing admission internals. */
    private static DeliveryDispatcher parkedDelivery() {
        return org.flexlb.balance.scheduler.DeliveryTestSupport.parked(PARKED_BOUNDARY);
    }

    private static final CapacityBoundary PARKED_BOUNDARY =
            CapacityBoundary.unavailable(
                    CapacityBoundary.Resource.PREFILL_BATCH,
                    new RouteProjection.AdmissionBlockSemantics(
                            "test delivery parked",
                            RouteProjection.AfterProbeAdmission.BLOCKED,
                            "test delivery parked",
                            RoleType.PREFILL));

    private static final DeliveryMetricsReporter NOOP_METRICS =
            Mockito.mock(DeliveryMetricsReporter.class);
}
