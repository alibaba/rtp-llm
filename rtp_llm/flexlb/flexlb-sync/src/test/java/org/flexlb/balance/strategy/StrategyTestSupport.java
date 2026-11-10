package org.flexlb.balance.strategy;

import org.flexlb.balance.RoutingResult;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot;
import org.flexlb.balance.scheduler.CapacityNotifier;
import org.flexlb.balance.scheduler.DeliveryCoordinator;
import org.flexlb.balance.scheduler.RouteResult;
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

/**
 * Package-local fixtures built only from the frozen endpoint-facing ports.
 */
final class StrategyTestSupport {
    private static final CapacityBoundary PARKED_BOUNDARY =
            CapacityBoundary.unavailable(CapacityBoundary.Resource.PREFILL_BATCH);
    private static final DeliveryMetricsReporter NOOP_METRICS =
            Mockito.mock(DeliveryMetricsReporter.class);

    private StrategyTestSupport() {}

    static <B> org.flexlb.balance.RoutingResult<Selection, B> select(
            org.flexlb.dao.loadbalance.Request request,
            java.util.function.Function<
                            Selection,
                            org.flexlb.balance.RoutingResult<
                                    org.flexlb.balance.scheduler.RoleSelection, B>>
                    action) {
        var route = scope(request);
        var result = action.apply(route);
        return result.status() == RoutingResult.Status.SUCCESS
                ? RoutingResult.success(route)
                : result.asFailure();
    }

    static Selection scope(org.flexlb.dao.loadbalance.Request request) {
        var context = new org.flexlb.balance.scheduler.RequestContext(config());
        context.data().setRequest(request);
        context.data()
                .setSchedulingMetadata(
                        org.flexlb.dao.SchedulingMetadata.explicit(50, Long.MAX_VALUE));
        return new Selection(context, new java.util.ArrayList<>());
    }

    static FlexlbConfig config() {
        FlexlbConfig config = new FlexlbConfig();
        config.getRequestLifecycle().getRequest().setTimeoutMs(60_000L);
        config.getRequestLifecycle().getDecision().setLifetime(2.0);
        return config;
    }

    static org.flexlb.balance.RoutingResult<Selection, RoleType> select(
            DecodeRoutingFixture selector,
            org.flexlb.dao.loadbalance.Request request,
            org.flexlb.dao.SchedulingMetadata metadata,
            String group) {
        return select(request, route -> selector.select(request, metadata, group, route));
    }

    static org.flexlb.balance.RoutingResult<Selection, org.flexlb.balance.scheduler.CapacityKey>
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
                        Mockito.mock(
                                PrefillSelectionStrategy.class,
                                org.mockito.Mockito.CALLS_REAL_METHODS),
                        Mockito.mock(
                                DecodeSelectionStrategy.class,
                                org.mockito.Mockito.CALLS_REAL_METHODS),
                        selector,
                        org.mockito.Mockito.mock(
                                org.flexlb.balance.strategy.EncoderSelectionStrategy.class),
                        org.mockito.Mockito.mock(org.flexlb.cache.match.CacheAwareService.class),
                        null,
                        Mockito.mock(org.flexlb.service.monitor.EngineHealthReporter.class),
                        Mockito.mock(org.flexlb.cache.telemetry.CacheMetricsReporter.class),
                        configService,
                        Mockito.mock(org.flexlb.config.ModelMetaConfig.class));
        RoutingResult<?, ?> result =
                org.springframework.test.util.ReflectionTestUtils.invokeMethod(
                        router, "selectVit", String.valueOf(id), group, route.roles());
        return result.status() == RoutingResult.Status.SUCCESS ? route : null;
    }

    static org.flexlb.dao.loadbalance.ServerStatus vit(Selection route) {
        var selected = route.role(RoleType.VIT);
        return selected == null ? null : selected.toServerStatus(route.requestId());
    }

    static WorkerEndpoint decodeEndpoint(Selection route) {
        return route.role(RoleType.DECODE).endpoint();
    }

    static EndpointRegistry endpointRegistry(ConfigService configService) {
        var requests = new org.flexlb.balance.scheduler.RequestRepository();
        DeliveryCoordinator delivery = parkedDelivery();
        return new EndpointRegistry(
                configService,
                requests,
                Mockito.mock(DeliveryMetricsReporter.class),
                delivery,
                new CapacityNotifier(),
                org.flexlb.balance.planner.BuiltinGroupPlanner.from(
                        configService.loadBalanceConfig()),
                () -> org.flexlb.balance.endpoint.OrderedPrefillRequestQueue.fifo(16));
    }

    /**
     * Keep selection-owned queues parked without replacing admission internals.
     */
    static DeliveryCoordinator parkedDelivery() {
        return org.flexlb.balance.scheduler.DeliveryTestSupport.parked(PARKED_BOUNDARY);
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

    static boolean offer(PrefillEndpoint endpoint, RouteResult exactItem) {
        org.flexlb.balance.scheduler.SchedulerTestSupport.bindEndpointOwner(endpoint, exactItem);
        {
            WorkerEndpoint selectedEndpoint = endpoint;

            return selectedEndpoint != null
                    && org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                                    endpoint, exactItem, exactItem.prefillWorkMs(), true)
                            .accepted()
                    && endpoint.enqueueRoute(
                            exactItem,
                            org.flexlb.balance.scheduler.QueueExecutionSettings.capture(
                                    exactItem.ctx().data().getConfig(),
                                    org.flexlb.balance.planner.BuiltinGroupPlanner.from(
                                            exactItem.ctx().data().getConfig())));
        }
    }

    static WorkSnapshot commitBatch(
            PrefillEndpoint endpoint,
            long batchId,
            long predictedMs,
            List<? extends RouteResult> exactItems) {
        if (exactItems.isEmpty()) {
            throw new IllegalArgumentException("committed batch requires at least one item");
        }
        List<RouteResult> items = List.copyOf(exactItems);
        var delivery =
                org.flexlb.balance.scheduler.DeliveryTestSupport.queued(
                        items,
                        new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(
                                batchId, Integer.MAX_VALUE),
                        new long[] {predictedMs});
        return endpoint.dequeueForDelivery(delivery).value();
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
            var model = Mockito.mock(org.flexlb.config.ModelMetaConfig.class);
            Mockito.when(model.requiredRoles())
                    .thenReturn(List.of(RoleType.PREFILL, RoleType.PDFUSION));
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
                            model);
        }

        org.flexlb.balance.RoutingResult<
                        org.flexlb.balance.scheduler.RoleSelection,
                        org.flexlb.balance.scheduler.CapacityKey>
                select(
                        org.flexlb.dao.loadbalance.Request request,
                        org.flexlb.dao.SchedulingMetadata metadata,
                        RoleType role,
                        String group,
                        Selection route) {
            var ctx = route.context();
            ctx.data().setSchedulingMetadata(metadata);
            ctx.data().setRequestedRoles(List.of(role));
            var result = routing.select(ctx, group);
            if (result.status() != RoutingResult.Status.SUCCESS) return result.asFailure();
            var execution = result.value();
            route.roles().addAll(execution.roles());
            route.cleanup.add(
                    () -> execution.prefillEndpoint().rollback(execution.prefillReservation()));
            return RoutingResult.success(execution.role(role));
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

    /**
     * Runs the Decode policy against real endpoint views and registration callbacks, recording acquired resources for test cleanup.
     */
    static final class DecodeRoutingFixture {
        final EndpointRegistry registry;
        final DecodeSelector policy;
        final org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity capacity;

        DecodeRoutingFixture(
                EndpointRegistry registry,
                ConfigService config,
                org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity capacity) {
            this.registry = registry;
            this.capacity = capacity;
            policy = new DecodeSelector(config, capacity);
        }

        org.flexlb.balance.RoutingResult<org.flexlb.balance.scheduler.RoleSelection, RoleType>
                select(
                        org.flexlb.dao.loadbalance.Request request,
                        org.flexlb.dao.SchedulingMetadata metadata,
                        String group,
                        Selection route) {
            var inputs =
                    new java.util.ArrayList<
                            org.flexlb.balance.endpoint.DecodeResources.DecodeRoutingView>();
            registry.decodeRoutingSnapshot(group, inputs);
            var result =
                    policy.selectAndRegister(
                            request,
                            metadata,
                            group,
                            List.copyOf(inputs),
                            view -> {
                                var endpoint = registry.selectDecodeEndpoint(view);
                                return endpoint == null
                                        ? null
                                        : ((org.flexlb.balance.endpoint.DecodeEndpoint) endpoint)
                                                .admissionSummary();
                            },
                            view -> {
                                var endpoint = registry.selectDecodeEndpoint(view);
                                return endpoint == null
                                        ? null
                                        : ((org.flexlb.balance.endpoint.DecodeEndpoint) endpoint)
                                                .routingView();
                            },
                            view -> {
                                var endpoint =
                                        (org.flexlb.balance.endpoint.DecodeEndpoint)
                                                registry.selectDecodeEndpoint(view);
                                if (endpoint == null) return RoutingResult.Registration.UNAVAILABLE;
                                var acquired =
                                        endpoint.tryRegisterSelection(
                                                view,
                                                request.getRequestId(),
                                                request.getHardKvTokens(),
                                                request.getExpectedKvTokens(),
                                                metadata.priority(),
                                                capacity);
                                return switch (acquired.status()) {
                                    case VERSION_CONFLICT, ALREADY_OWNED ->
                                            RoutingResult.Registration.CONFLICT;
                                    case ENDPOINT_RETIRED -> RoutingResult.Registration.UNAVAILABLE;
                                    case CAPACITY_FULL -> RoutingResult.Registration.BLOCKED;
                                    case ACQUIRED -> {
                                        var status =
                                                org.flexlb.dao.loadbalance.ServerStatus.forWorker(
                                                        RoleType.DECODE,
                                                        request.getRequestId(),
                                                        view.topology(),
                                                        view.workerStatus().fields());
                                        route.roles()
                                                .add(
                                                        org.flexlb.balance.scheduler
                                                                .SelectionTestSupport
                                                                .registeredDecode(
                                                                        endpoint,
                                                                        status,
                                                                        acquired.value()));
                                        route.cleanup.add(
                                                () ->
                                                        endpoint.release(
                                                                acquired.value(),
                                                                org.flexlb.balance.endpoint
                                                                        .DecodeResources
                                                                        .ReleaseReason
                                                                        .LOCAL_ROLLBACK));
                                        yield RoutingResult.Registration.REGISTERED;
                                    }
                                };
                            });
            return new RoutingResult<>(
                    result.status(),
                    result.status() == RoutingResult.Status.SUCCESS
                            ? route.role(RoleType.DECODE)
                            : null,
                    result.failure(),
                    result.blocker() == null ? null : result.blocker().role(),
                    result.diagnostics());
        }
    }

    /**
     * Holds selected roles for strategy assertions and releases the resources acquired by the fixture when closed.
     */
    static final class Selection implements AutoCloseable {
        private final org.flexlb.balance.scheduler.RequestContext context;
        private final java.util.List<org.flexlb.balance.scheduler.RoleSelection> roles;
        private final java.util.List<Runnable> cleanup = new java.util.ArrayList<>();

        Selection(
                org.flexlb.balance.scheduler.RequestContext context,
                java.util.List<org.flexlb.balance.scheduler.RoleSelection> roles) {
            this.context = context;
            this.roles = roles;
        }

        public void close() {
            for (var release : cleanup) release.run();
            cleanup.clear();
        }

        org.flexlb.balance.scheduler.RequestContext context() {
            return context;
        }

        java.util.List<org.flexlb.balance.scheduler.RoleSelection> roles() {
            return roles;
        }

        String requestId() {
            return context.data().getRequestId();
        }

        long prefillWorkMs() {
            return ((org.flexlb.balance.scheduler.RoleSelection.Prefill) role(RoleType.PREFILL))
                    .workMs();
        }

        org.flexlb.balance.scheduler.RoleSelection role(RoleType role) {
            return roles.stream()
                    .filter(
                            r ->
                                    r.role() == role
                                            || role.supportsPrefill() && r.role().supportsPrefill())
                    .findFirst()
                    .orElse(null);
        }

        org.flexlb.dao.loadbalance.ServerStatus prefill() {
            return status(RoleType.PREFILL);
        }

        org.flexlb.dao.loadbalance.ServerStatus decode() {
            return status(RoleType.DECODE);
        }

        private org.flexlb.dao.loadbalance.ServerStatus status(RoleType role) {
            var selected = role(role);
            return selected == null ? null : selected.toServerStatus(requestId());
        }
    }
}
