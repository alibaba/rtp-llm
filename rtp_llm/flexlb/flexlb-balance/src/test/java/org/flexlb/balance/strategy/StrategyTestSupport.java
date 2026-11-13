package org.flexlb.balance.strategy;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.schedule.EngineCancellation;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.cache.CacheStatus;
import org.flexlb.worker.endpoint.CapacityNotifier;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.flexlb.worker.status.WorkerStatus;
import org.flexlb.worker.status.WorkerStatusResponse;
import org.flexlb.worker.status.WorkerStatusTestSupport;
import org.mockito.Mockito;

import java.util.List;
import java.util.Map;

/**
 * Package-local fixtures built only from the frozen endpoint-facing ports.
 */
final class StrategyTestSupport {

    private static final DeliveryMetricsReporter NOOP_METRICS =
            Mockito.mock(DeliveryMetricsReporter.class);

    private StrategyTestSupport() {}

    static <B> org.flexlb.balance.strategy.RoutingResult<Selection, B> select(
            org.flexlb.worker.endpoint.Request request,
            java.util.function.Function<
                            Selection,
                            org.flexlb.balance.strategy.RoutingResult<
                                    org.flexlb.balance.strategy.RoleSelection, B>>
                    action) {
        var route = scope(request);
        var result = action.apply(route);
        return result.status() == RoutingResult.Status.SUCCESS
                ? RoutingResult.success(route)
                : result.asFailure();
    }

    static Selection scope(org.flexlb.worker.endpoint.Request request) {
        var context = new org.flexlb.balance.schedule.RequestContext(config());
        context.setRequest(request);
        context.setSchedulingMetadata(
                org.flexlb.balance.schedule.SchedulingMetadata.explicit(50, Long.MAX_VALUE));
        return new Selection(context, new java.util.ArrayList<>());
    }

    static FlexlbConfig config() {
        FlexlbConfig config = new FlexlbConfig();
        config.getRequestLifecycle().getRequest().setTimeoutMs(60_000L);
        return config;
    }

    static org.flexlb.balance.strategy.RoutingResult<Selection, RoleType> select(
            DecodeRoutingFixture selector,
            org.flexlb.worker.endpoint.Request request,
            org.flexlb.balance.schedule.SchedulingMetadata metadata,
            String group) {
        return select(request, route -> selector.select(request, metadata, group, route));
    }

    static org.flexlb.balance.strategy.RoutingResult<
                    Selection, org.flexlb.worker.endpoint.CapacityKey>
            select(
                    PrefillRoutingFixture selector,
                    org.flexlb.worker.endpoint.Request request,
                    org.flexlb.balance.schedule.SchedulingMetadata metadata,
                    RoleType role,
                    String group) {
        return select(request, route -> selector.select(request, metadata, role, group, route));
    }

    static Selection select(
            EndpointRegistry endpoints,
            VitStrategy selector,
            long id,
            RoleType role,
            String group) {
        Selection route =
                scope(
                        org.flexlb.worker.endpoint.Request.builder()
                                .requestId(String.valueOf(id))
                                .build());
        com.google.common.base.Preconditions.checkArgument(role == RoleType.VIT);
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig()).thenReturn(config());
        var router =
                new org.flexlb.balance.strategy.RequestRouteSelector(
                        endpoints,
                        Mockito.mock(CostBasedPrefillStrategy.class),
                        Mockito.mock(DecodeStrategy.class),
                        selector,
                        org.mockito.Mockito.mock(EncoderStrategy.class),
                        Mockito.mock(org.flexlb.config.service.ModelMetaConfig.class));
        RoutingResult<?, ?> result = selector.select(route.context(), group);
        if (result.status() == RoutingResult.Status.SUCCESS)
            route.roles().add((org.flexlb.balance.strategy.RoleSelection) result.value());
        return result.status() == RoutingResult.Status.SUCCESS ? route : null;
    }

    static org.flexlb.worker.status.ServerStatus vit(Selection route) {
        var selected = route.role(RoleType.VIT);
        return selected == null ? null : selected.toServerStatus(route.requestId());
    }

    static WorkerEndpoint decodeEndpoint(Selection route) {
        return route.role(RoleType.DECODE).endpoint();
    }

    static EndpointRegistry endpointRegistry(ConfigService configService) {
        var requests = new org.flexlb.balance.schedule.RequestRepository();
        EngineCancellation delivery = parkedDelivery();
        return new EndpointRegistry(
                configService,
                requests,
                Mockito.mock(DeliveryMetricsReporter.class),
                new CapacityNotifier(),
                org.mockito.Mockito.mock(org.flexlb.cache.match.CacheAwareService.class));
    }

    /**
     * Keep selection-owned queues parked without replacing admission internals.
     */
    static EngineCancellation parkedDelivery() {
        return org.flexlb.balance.schedule.DeliveryTestSupport.route(NOOP_METRICS);
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
                    status.prepareStatus(
                            WorkerStatusTestSupport.freezeStatusResponse(status, response));
            status.publishPreparedStatus(prepared);
            status.recordSuccessfulPoll(response.isAlive());
        } finally {
            status.lock.unlock();
        }
    }

    static WorkerEndpoint publishEndpoint(EndpointRegistry registry, WorkerStatus source) {
        RoleType role = source.getRole();
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
            WorkerStatus.PreparedStatus prepared = status.prepareStatus(observation);
            return registry.updateEndpoint(status.getLogicalIpPort(), status, prepared).endpoint();
        } finally {
            status.lock.unlock();
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
                        status.committedEngineObservation().availableKvCacheTokens(),
                        status.committedEngineObservation().totalKvCacheTokens(),
                        Math.max(1L, status.appliedStatusCursor().statusVersion() + 1L));
        response.setCacheStatus(cache);
        publish(status, response);
    }

    /**
     * Exercise the real routing owner together with its policy, including exact selectedEndpoint handoff.
     */
    static final class PrefillRoutingFixture {
        final CostBasedPrefillStrategy policy;
        final org.flexlb.balance.strategy.RequestRouteSelector routing;

        PrefillRoutingFixture(
                EndpointRegistry registry,
                org.flexlb.cache.match.CacheAwareService cache,
                org.flexlb.worker.status.ServiceMetricsReporter health,
                org.flexlb.cache.metrics.CacheMetricsReporter metrics,
                ConfigService config) {
            policy = new CostBasedPrefillStrategy(config, registry, cache, metrics);
            var model = Mockito.mock(org.flexlb.config.service.ModelMetaConfig.class);
            Mockito.when(model.requiredRoles())
                    .thenReturn(List.of(RoleType.PREFILL, RoleType.PDFUSION));
            routing =
                    new org.flexlb.balance.strategy.RequestRouteSelector(
                            registry,
                            policy,
                            Mockito.mock(DecodeStrategy.class),
                            Mockito.mock(VitStrategy.class),
                            org.mockito.Mockito.mock(EncoderStrategy.class),
                            model);
        }

        org.flexlb.balance.strategy.RoutingResult<
                        org.flexlb.balance.strategy.RoleSelection,
                        org.flexlb.worker.endpoint.CapacityKey>
                select(
                        org.flexlb.worker.endpoint.Request request,
                        org.flexlb.balance.schedule.SchedulingMetadata metadata,
                        RoleType role,
                        String group,
                        Selection route) {
            var ctx = route.context();
            ctx.setSchedulingMetadata(metadata);
            ctx.setRequestedRoles(List.of(role));
            var result = routing.select(ctx, group);
            if (result.status() != RoutingResult.Status.SUCCESS)
                return new RoutingResult<>(
                        result.status(),
                        null,
                        result.failure(),
                        result.blocker(),
                        result.diagnostics());
            var execution = result;
            route.roles().addAll(execution.roles());
            route.cleanup.add(
                    () ->
                            execution
                                    .prefillEndpoint()
                                    .rollbackSelection(execution.prefillRegistration()));
            return RoutingResult.success(execution.role(role));
        }
    }

    /**
     * Runs the Decode policy against real endpoint views and registration callbacks, recording acquired resources for test cleanup.
     */
    static final class DecodeRoutingFixture {
        final EndpointRegistry registry;
        final DecodeStrategy policy;
        final org.flexlb.config.FlexlbConfig configuration;
        final org.flexlb.worker.endpoint.DecodeResources.AdmissionCapacity capacity;

        DecodeRoutingFixture(
                EndpointRegistry registry,
                ConfigService config,
                org.flexlb.worker.endpoint.DecodeResources.AdmissionCapacity capacity) {
            this.registry = registry;
            this.capacity = capacity;
            policy = new DecodeStrategy(config, capacity, registry);
            configuration = config.loadBalanceConfig();
        }

        org.flexlb.balance.strategy.RoutingResult<
                        org.flexlb.balance.strategy.RoleSelection, RoleType>
                select(
                        org.flexlb.worker.endpoint.Request request,
                        org.flexlb.balance.schedule.SchedulingMetadata metadata,
                        String group,
                        Selection route) {
            var model = Mockito.mock(org.flexlb.config.service.ModelMetaConfig.class);
            Mockito.when(model.requiredRoles()).thenReturn(List.of(RoleType.DECODE));
            var metrics = Mockito.mock(org.flexlb.cache.metrics.CacheMetricsReporter.class);
            var configService = Mockito.mock(ConfigService.class);
            Mockito.when(configService.loadBalanceConfig()).thenReturn(configuration);
            var routing =
                    new org.flexlb.balance.strategy.RequestRouteSelector(
                            registry,
                            Mockito.mock(CostBasedPrefillStrategy.class),
                            policy,
                            new VitStrategy(
                                    org.mockito.Mockito.mock(
                                            org.flexlb.cache.VitCacheDirectory.class),
                                    registry),
                            Mockito.mock(EncoderStrategy.class),
                            model);
            var context = route.context();
            context.setSchedulingMetadata(metadata);
            context.setRequestedRoles(List.of(RoleType.DECODE));
            RoutingResult<
                            org.flexlb.balance.strategy.RoleSelection,
                            org.flexlb.worker.endpoint.CapacityKey>
                    result = policy.select(context, group);
            if (result.status() == RoutingResult.Status.SUCCESS) {
                route.roles().add(result.value());
                var endpoint = (DecodeEndpoint) result.value().endpoint();
                var registration =
                        endpoint.resourceSnapshot()
                                .requests()
                                .get(request.getRequestId())
                                .reservation();
                route.cleanup.add(() -> endpoint.releaseRequest(registration));
            }
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
        private final org.flexlb.balance.schedule.RequestContext context;
        private final java.util.List<org.flexlb.balance.strategy.RoleSelection> roles;
        private final java.util.List<Runnable> cleanup = new java.util.ArrayList<>();

        Selection(
                org.flexlb.balance.schedule.RequestContext context,
                java.util.List<org.flexlb.balance.strategy.RoleSelection> roles) {
            this.context = context;
            this.roles = roles;
        }

        public void close() {
            for (var release : cleanup) release.run();
            cleanup.clear();
        }

        org.flexlb.balance.schedule.RequestContext context() {
            return context;
        }

        java.util.List<org.flexlb.balance.strategy.RoleSelection> roles() {
            return roles;
        }

        String requestId() {
            return context.getRequestId();
        }

        long computeTokens() {
            return ((org.flexlb.balance.strategy.RoleSelection) role(RoleType.PREFILL))
                    .computeTokens();
        }

        org.flexlb.balance.strategy.RoleSelection role(RoleType role) {
            return roles.stream()
                    .filter(
                            r ->
                                    r.role() == role
                                            || role.supportsPrefill() && r.role().supportsPrefill())
                    .findFirst()
                    .orElse(null);
        }

        org.flexlb.worker.status.ServerStatus prefill() {
            return status(RoleType.PREFILL);
        }

        org.flexlb.worker.status.ServerStatus decode() {
            return status(RoleType.DECODE);
        }

        private org.flexlb.worker.status.ServerStatus status(RoleType role) {
            var selected = role(role);
            return selected == null ? null : selected.toServerStatus(requestId());
        }
    }
}
