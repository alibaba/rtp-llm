package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.DeliveryDispatcher;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.RouteExecution;
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

import static org.flexlb.balance.endpoint.EndpointTestSupport.withStateLock;

/**
 * Package-local fixtures built only from the frozen endpoint-facing ports.
 */
final class StrategyTestSupport {
    /**
     * Exercise the real routing owner together with its policy, including exact selectedEndpoint handoff.
     */
    static final class PrefillRoutingFixture {
        final CostBasedPrefillStrategy policy;
        final org.flexlb.balance.scheduler.RequestRouteSelector routing;

        PrefillRoutingFixture(
                EndpointRegistry registry,
                org.flexlb.cache.match.CacheAwareService cache,
                org.flexlb.service.monitor.EngineHealthReporter health,
                org.flexlb.cache.telemetry.CacheMetricsReporter metrics,
                ConfigService config) {
            policy = new CostBasedPrefillStrategy(cache, config);
            routing =
                    new org.flexlb.balance.scheduler.RequestRouteSelector(
                            registry,
                            policy,
                            Mockito.mock(DecodeSelector.class),
                            Mockito.mock(VitWorkerSelector.class),
                            org.mockito.Mockito.mock(
                                    org.flexlb.balance.strategy.EncoderSelectionStrategy.class),
                            org.mockito.Mockito.mock(
                                    org.flexlb.cache.match.CacheAwareService.class),
                            null,
                            health,
                            metrics,
                            config,
                            Mockito.mock(org.flexlb.config.ModelMetaConfig.class));
        }

        org.flexlb.balance.PlacementResult<
                        org.flexlb.dao.loadbalance.ServerStatus,
                        org.flexlb.balance.scheduler.PlacementKey>
                select(
                        org.flexlb.dao.loadbalance.Request request,
                        org.flexlb.dao.SchedulingMetadata metadata,
                        RoleType role,
                        String group,
                        Selection route) {
            return org.springframework.test.util.ReflectionTestUtils.invokeMethod(
                    routing, "selectPrefill", request, metadata, role, group, route.builder());
        }

        int selectBestCandidate(
                PrefillCandidateSet candidates,
                long minimum,
                RoleType role,
                String group,
                long tokens) {
            return policy.selectBestCandidate(candidates, minimum, role, group, tokens);
        }
    }

    static final class DecodeRoutingFixture {
        final org.flexlb.balance.scheduler.RequestRouteSelector routing;

        DecodeRoutingFixture(
                EndpointRegistry registry,
                ConfigService config,
                org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity capacity) {
            routing =
                    new org.flexlb.balance.scheduler.RequestRouteSelector(
                            registry,
                            Mockito.mock(CostBasedPrefillStrategy.class),
                            new DecodeSelector(config, capacity),
                            Mockito.mock(VitWorkerSelector.class),
                            org.mockito.Mockito.mock(
                                    org.flexlb.balance.strategy.EncoderSelectionStrategy.class),
                            org.mockito.Mockito.mock(
                                    org.flexlb.cache.match.CacheAwareService.class),
                            null,
                            Mockito.mock(org.flexlb.service.monitor.EngineHealthReporter.class),
                            Mockito.mock(org.flexlb.cache.telemetry.CacheMetricsReporter.class),
                            config,
                            Mockito.mock(org.flexlb.config.ModelMetaConfig.class));
        }

        org.flexlb.balance.PlacementResult<org.flexlb.dao.loadbalance.ServerStatus, RoleType>
                select(
                        org.flexlb.dao.loadbalance.Request request,
                        org.flexlb.dao.SchedulingMetadata metadata,
                        String group,
                        Selection route) {
            org.flexlb.balance.PlacementResult<
                            org.flexlb.dao.loadbalance.ServerStatus,
                            org.flexlb.balance.scheduler.PlacementKey>
                    result =
                            org.springframework.test.util.ReflectionTestUtils.invokeMethod(
                                    routing,
                                    "selectDecode",
                                    request,
                                    metadata,
                                    group,
                                    route.builder());
            return new org.flexlb.balance.PlacementResult<>(
                    result.status(),
                    result.value(),
                    result.failure(),
                    result.blocker() == null ? null : result.blocker().role(),
                    result.diagnostics(),
                    result.selection() == null
                            ? null
                            : org.flexlb.balance.PlacementResult.success(
                                    result.selection().value()));
        }
    }

    private StrategyTestSupport() {}

    static Selection scope(org.flexlb.dao.loadbalance.Request request) {
        var context = new org.flexlb.balance.scheduler.RequestContext(config());
        context.data().setRequest(request);
        return new Selection(
                new org.flexlb.balance.scheduler.RouteResult.Builder(context),
                request.getRequestId());
    }

    static <B> org.flexlb.balance.PlacementResult<Selection, B> select(
            org.flexlb.dao.loadbalance.Request request,
            java.util.function.Function<
                            Selection,
                            org.flexlb.balance.PlacementResult<
                                    org.flexlb.dao.loadbalance.ServerStatus, B>>
                    action) {
        var route = scope(request);
        var result = action.apply(route);
        return result.status() == PlacementResult.Status.SUCCESS
                ? PlacementResult.success(route)
                : result.asFailure();
    }

    static org.flexlb.balance.PlacementResult<Selection, RoleType> select(
            DecodeRoutingFixture selector,
            org.flexlb.dao.loadbalance.Request request,
            org.flexlb.dao.SchedulingMetadata metadata,
            String group) {
        return select(request, route -> selector.select(request, metadata, group, route));
    }

    static org.flexlb.balance.PlacementResult<Selection, org.flexlb.balance.scheduler.PlacementKey>
            select(
                    PrefillRoutingFixture selector,
                    org.flexlb.dao.loadbalance.Request request,
                    org.flexlb.dao.SchedulingMetadata metadata,
                    RoleType role,
                    String group) {
        return select(request, route -> selector.select(request, metadata, role, group, route));
    }

    static Selection select(
            EndpointRegistry endpoints,
            VitSelectionStrategy selector,
            long id,
            RoleType role,
            String group) {
        Selection route =
                scope(
                        org.flexlb.dao.loadbalance.Request.builder()
                                .requestId(String.valueOf(id))
                                .build());
        com.google.common.base.Preconditions.checkArgument(role == RoleType.VIT);
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig()).thenReturn(config());
        var router =
                new org.flexlb.balance.scheduler.RequestRouteSelector(
                        endpoints,
                        Mockito.mock(PrefillSelectionStrategy.class),
                        Mockito.mock(DecodeSelectionStrategy.class),
                        selector,
                        org.mockito.Mockito.mock(
                                org.flexlb.balance.strategy.EncoderSelectionStrategy.class),
                        org.mockito.Mockito.mock(org.flexlb.cache.match.CacheAwareService.class),
                        null,
                        Mockito.mock(org.flexlb.service.monitor.EngineHealthReporter.class),
                        Mockito.mock(org.flexlb.cache.telemetry.CacheMetricsReporter.class),
                        configService,
                        Mockito.mock(org.flexlb.config.ModelMetaConfig.class));
        PlacementResult<?, ?> result =
                org.springframework.test.util.ReflectionTestUtils.invokeMethod(
                        router, "selectVit", String.valueOf(id), group, route.builder());
        return result.status() == PlacementResult.Status.SUCCESS ? route : null;
    }

    static org.flexlb.dao.loadbalance.ServerStatus vit(Selection route) {
        org.flexlb.balance.scheduler.RoleSelection selected =
                org.springframework.test.util.ReflectionTestUtils.invokeMethod(
                        route.builder(), "selection", RoleType.VIT);
        return selected == null ? null : selected.toServerStatus(route.requestId());
    }

    static WorkerEndpoint decodeEndpoint(Selection route) {
        return route.role(RoleType.DECODE).endpoint();
    }

    record Selection(org.flexlb.balance.scheduler.RouteResult.Builder builder, String requestId) {
        org.flexlb.balance.scheduler.RoleSelection role(RoleType role) {
            return org.springframework.test.util.ReflectionTestUtils.invokeMethod(
                    builder, "selection", role);
        }

        long prefillWorkMs() {
            return ((org.flexlb.balance.scheduler.RoleSelection.Prefill) role(RoleType.PREFILL))
                    .workMs();
        }

        org.flexlb.dao.loadbalance.ServerStatus prefill() {
            return status(RoleType.PREFILL);
        }

        org.flexlb.dao.loadbalance.ServerStatus decode() {
            return status(RoleType.DECODE);
        }

        private org.flexlb.dao.loadbalance.ServerStatus status(RoleType role) {
            var selected = role(role);
            return selected == null ? null : selected.toServerStatus(requestId);
        }
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
        return new EndpointRegistry(
                configService,
                requests,
                Mockito.mock(DeliveryMetricsReporter.class),
                delivery,
                new PlacementAvailability(),
                org.flexlb.balance.planner.BuiltinGroupPlanner.from(
                        configService.loadBalanceConfig()),
                order -> org.flexlb.balance.endpoint.OrderedPrefillRequestQueue.ordered(16, order));
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
        WorkerStatus status =
                WorkerStatus.createDiscovered(role, group, ip, httpPort, grpcPort, "test-site");
        WorkerStatusResponse response =
                response(
                        role,
                        alive,
                        availableKv,
                        totalKv,
                        Math.max(1L, status.appliedStatusCursor().statusVersion() + 1L));
        publish(status, response);
        return status;
    }

    static WorkerStatusResponse response(
            RoleType role, boolean alive, long availableKv, long totalKv, long statusVersion) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(role);
        response.setAlive(alive);
        response.setAvailableKvCacheTokens(availableKv);
        response.setTotalKvCacheTokens(totalKv);
        response.setStatusVersion(statusVersion);
        response.setLatestFinishedVersion(0L);
        return response;
    }

    static void publish(WorkerStatus status, WorkerStatusResponse response) {
        status.lock.lock();
        try {
            WorkerStatus.PreparedStatus prepared =
                    status.prepareNewStatus(
                            WorkerStatusTestSupport.freezeStatusResponse(status, response));
            status.publishPreparedStatus(prepared);
            status.recordSuccessfulPoll(response.isAlive());
        } finally {
            status.lock.unlock();
        }
    }

    static WorkerEndpoint publishEndpoint(
            EndpointRegistry registry, RoleType role, String address, WorkerStatus source) {
        WorkerStatus status =
                WorkerStatus.createDiscovered(
                        role,
                        source.getGroup(),
                        source.getIp(),
                        source.getPort(),
                        source.getGrpcPort(),
                        source.getSite());
        WorkerStatus.StatusObservation observation =
                status.bindStatusObservation(
                        source.committedEngineObservation(), true, 1L, 0L, Map.of());
        status.lock.lock();
        try {
            WorkerStatus.PreparedStatus prepared = status.prepareNewStatus(observation);
            return registry.publishPreparedEndpoint(address, status, prepared);
        } finally {
            status.lock.unlock();
        }
    }

    static boolean offer(PrefillEndpoint endpoint, RouteExecution exactItem) {
        org.flexlb.balance.scheduler.SchedulerTestSupport.bindEndpointOwner(endpoint, exactItem);
        {
            WorkerEndpoint selectedEndpoint = endpoint;

            return selectedEndpoint != null
                    && endpoint.enqueueRoute(
                            exactItem,
                            org.flexlb.balance.scheduler.QueueExecutionSettings.capture(
                                    exactItem.ctx().data().getConfig(),
                                    org.flexlb.balance.planner.BuiltinGroupPlanner.from(
                                            exactItem.ctx().data().getConfig())));
        }
    }

    static org.flexlb.balance.scheduler.Delivery commitBatch(
            PrefillEndpoint endpoint,
            long batchId,
            long predictedMs,
            List<? extends RouteExecution> exactItems) {
        if (exactItems.isEmpty()) {
            throw new IllegalArgumentException("committed batch requires at least one item");
        }
        List<RouteExecution> items = List.copyOf(exactItems);
        CapacityBoundary.Attempt<PrefillEndpoint.BatchRecord> result =
                endpoint.reserveBatch(batchId, Integer.MAX_VALUE);
        if (!result.accepted()) {
            throw new IllegalStateException("batch reservation rejected: " + result.boundary());
        }
        {
            PrefillEndpoint.BatchRecord reservation = result.value();
            try (var preparationReservation =
                    org.flexlb.balance.endpoint.EndpointTestSupport.preparation(reservation)) {
                PrefillEndpoint state = (PrefillEndpoint) endpoint;
                return withStateLock(
                        org.flexlb.balance.endpoint.EndpointTestSupport.stateLock(state),
                        () -> {
                            return state.commitDelivery(
                                    org.flexlb.balance.scheduler.DeliveryTestSupport.queued(
                                            items,
                                            reservation,
                                            new long[] {predictedMs},
                                            null,
                                            System.currentTimeMillis()));
                        });
            }
        }
    }

    static void setCacheStatus(WorkerStatus status, long blockSize, long availableKv) {
        CacheStatus cache = new CacheStatus();
        cache.setBlockSize(blockSize);
        cache.setAvailableKvCache(availableKv);
        WorkerStatusResponse response =
                response(
                        status.getRole(),
                        status.pollHealth().reportedAlive(),
                        status.getAvailableKvCacheTokens(),
                        status.getTotalKvCacheTokens(),
                        Math.max(1L, status.appliedStatusCursor().statusVersion() + 1L));
        response.setCacheStatus(cache);
        publish(status, response);
    }

    /**
     * Keep selection-owned queues parked without replacing admission internals.
     */
    static DeliveryDispatcher parkedDelivery() {
        return org.flexlb.balance.scheduler.DeliveryTestSupport.parked(PARKED_BOUNDARY);
    }

    private static final CapacityBoundary PARKED_BOUNDARY =
            CapacityBoundary.unavailable(CapacityBoundary.Resource.PREFILL_BATCH);

    private static final DeliveryMetricsReporter NOOP_METRICS =
            Mockito.mock(DeliveryMetricsReporter.class);
}
