package org.flexlb.balance.strategy;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
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

import static org.flexlb.worker.status.WorkerStatusTestSupport.createDiscovered;

/**
 * Package-local fixtures built only from the frozen endpoint-facing ports.
 */
final class StrategyTestSupport {

    private StrategyTestSupport() {}

    static <B> org.flexlb.balance.strategy.SelectionResult<Selection, B> select(
            org.flexlb.balance.request.Request request,
            java.util.function.Function<
                            Selection,
                            org.flexlb.balance.strategy.SelectionResult<
                                    org.flexlb.balance.strategy.RoleSelection, B>>
                    action) {
        var route = scope(request);
        var result = action.apply(route);
        return result.status() == SelectionResult.Status.SUCCESS
                ? SelectionResult.success(route)
                : result.asFailure();
    }

    static Selection scope(org.flexlb.balance.request.Request request) {
        var context = new org.flexlb.balance.schedule.RequestContext(config());
        context.setRequest(request);
        context.setSchedulingMetadata(
                org.flexlb.balance.schedule.SchedulingMetadata.of(50, null, Long.MAX_VALUE, 50));
        return new Selection(context, new java.util.ArrayList<>());
    }

    static FlexlbConfig config() {
        FlexlbConfig config = new FlexlbConfig();
        config.getRequestLifecycle().setInactivityTimeoutMs(60_000L);
        return config;
    }

    static org.flexlb.balance.strategy.SelectionResult<Selection, RoleType> select(
            DecodeRoutingFixture selector,
            org.flexlb.balance.request.Request request,
            org.flexlb.balance.schedule.SchedulingMetadata metadata,
            String group) {
        return select(request, route -> selector.select(request, metadata, group, route));
    }

    static org.flexlb.balance.strategy.SelectionResult<
                    Selection, org.flexlb.worker.endpoint.CapacityKey>
            select(
                    PrefillRoutingFixture selector,
                    org.flexlb.balance.request.Request request,
                    org.flexlb.balance.schedule.SchedulingMetadata metadata,
                    RoleType role,
                    String group) {
        return select(request, route -> selector.select(request, metadata, role, group, route));
    }

    static WorkerEndpoint decodeEndpoint(Selection route) {
        return route.role(RoleType.DECODE).endpoint();
    }

    static EndpointRegistry endpointRegistry(ConfigService configService) {
        return new EndpointRegistry(
                configService,
                Mockito.mock(DeliveryMetricsReporter.class),
                new CapacityNotifier(),
                org.mockito.Mockito.mock(org.flexlb.cache.match.CacheAwareService.class));
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
        WorkerStatus status = createDiscovered(role, group, ip, httpPort, grpcPort, "test-site");
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
            status.publishReducedStatus(prepared);
            org.flexlb.worker.status.WorkerStatusTestSupport.recordSuccessfulPoll(
                    status, response.isAlive());
        } finally {
            status.lock.unlock();
        }
    }

    static WorkerEndpoint publishEndpoint(EndpointRegistry registry, WorkerStatus source) {
        RoleType role = source.getRole();
        WorkerStatus status =
                createDiscovered(
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

        org.flexlb.balance.strategy.SelectionResult<
                        org.flexlb.balance.strategy.RoleSelection,
                        org.flexlb.worker.endpoint.CapacityKey>
                select(
                        org.flexlb.balance.request.Request request,
                        org.flexlb.balance.schedule.SchedulingMetadata metadata,
                        RoleType role,
                        String group,
                        Selection route) {
            var ctx = route.context();
            ctx.setSchedulingMetadata(metadata);
            ctx.setRequestedRoles(List.of(role));
            var result = routing.selectAndRegister(ctx, group);
            if (result.status() != SelectionResult.Status.SUCCESS)
                return new SelectionResult<>(
                        result.status(),
                        null,
                        result.failure(),
                        result.blocker(),
                        result.diagnostics());
            var execution = result.value();
            route.roles().addAll(execution.roles());
            route.cleanup.add(
                    () ->
                            execution
                                    .prefillEndpoint()
                                    .rollbackSelection(execution.prefillRegistration()));
            return SelectionResult.success(execution.role(role));
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

        org.flexlb.balance.strategy.SelectionResult<
                        org.flexlb.balance.strategy.RoleSelection, RoleType>
                select(
                        org.flexlb.balance.request.Request request,
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
                                            org.flexlb.cache.VitWorkerCacheManager.class),
                                    registry),
                            Mockito.mock(EncoderStrategy.class),
                            model);
            var context = route.context();
            context.setSchedulingMetadata(metadata);
            context.setRequestedRoles(List.of(RoleType.DECODE));
            SelectionResult<
                            org.flexlb.balance.strategy.RoleSelection,
                            org.flexlb.worker.endpoint.CapacityKey>
                    result = policy.selectAndRegister(context, group);
            if (result.status() == SelectionResult.Status.SUCCESS) {
                route.roles().add(result.value());
                var endpoint = (DecodeEndpoint) result.value().endpoint();
                var registration =
                        endpoint.resourceSnapshot()
                                .requests()
                                .get(request.getRequestId())
                                .reservation();
                route.cleanup.add(() -> endpoint.releaseRequest(registration));
            }
            return new SelectionResult<>(
                    result.status(),
                    result.status() == SelectionResult.Status.SUCCESS
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
