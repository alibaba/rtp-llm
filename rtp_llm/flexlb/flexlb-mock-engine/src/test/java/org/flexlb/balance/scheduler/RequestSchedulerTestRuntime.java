package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.DecodePreemption;
import org.flexlb.balance.eviction.DefaultDecodePreemption;
import org.flexlb.balance.eviction.PriorityDecodePreemptionPolicy;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.VitWorkerSelector;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.master.WorkerStatusTestSupport;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;

import java.time.Duration;
import java.util.List;
import java.util.Objects;

/**
 * Test-only composition root for mock-engine end-to-end scheduler tests.
 */
public final class RequestSchedulerTestRuntime implements AutoCloseable {
    private final RequestRepository requests = new RequestRepository();
    private final PlacementAvailability placementAvailability = new PlacementAvailability();
    private final BindingRouter router;
    private final EndpointRegistry registry;
    private final DecodePreemption decodePreemption;
    private final AbstractRequestScheduler scheduler;
    private final AbstractRequestScheduler entry;
    private final SchedulerTestServices runtime;

    private volatile boolean deliveryPaused;

    private final java.util.Set<Runnable> deliveryListeners =
            new java.util.concurrent.CopyOnWriteArraySet<>();

    public RequestSchedulerTestRuntime(
            ConfigService configService,
            DeliveryDispatcher delivery,
            DeliveryMetricsReporter deliveryReporter,
            RequestSchedulerReporter requestReporter,
            EngineGrpcClient engineClient) {
        Objects.requireNonNull(delivery, "delivery");
        var capacity = new PlacementConfiguration().decodeAdmissionCapacity(configService);
        this.registry =
                new EndpointRegistry(
                        configService,
                        requests,
                        deliveryReporter,
                        delivery,
                        placementAvailability,
                        org.flexlb.balance.planner.BuiltinGroupPlanner.from(
                                configService.loadBalanceConfig()),
                        order ->
                                org.flexlb.balance.endpoint.OrderedPrefillRequestQueue.ordered(
                                        16, order));
        this.router = new BindingRouter(registry, configService, capacity);
        this.runtime =
                org.mockito.Mockito.spy(
                        new SchedulerTestServices(
                                requests,
                                registry,
                                deliveryReporter,
                                requestReporter,
                                delivery,
                                configService,
                                engineClient));
        var facilities = org.mockito.Mockito.spy(runtime.deliveries());
        org.springframework.test.util.ReflectionTestUtils.setField(
                runtime, "deliveries", facilities);
        org.springframework.test.util.ReflectionTestUtils.setField(
                delivery, "execution", facilities);
        // Keep transport and Dispatcher real; inject capacity pressure at its infrastructure owner.
        org.mockito.Mockito.doAnswer(
                        invocation ->
                                deliveryPaused
                                        ? CapacityBoundary.Attempt.rejected(
                                                CapacityBoundary.unavailable(
                                                        CapacityBoundary.Resource
                                                                .DISPATCH_EXECUTOR))
                                        : invocation.callRealMethod())
                .when(facilities)
                .tryReserveDelivery();
        org.mockito.Mockito.doAnswer(
                        invocation -> !deliveryPaused && (boolean) invocation.callRealMethod())
                .when(facilities)
                .deliveryCapacityAvailable();
        org.mockito.Mockito.doAnswer(
                        invocation -> {
                            Runnable wake = invocation.getArgument(0);
                            deliveryListeners.add(wake);
                            Runnable remove = (Runnable) invocation.callRealMethod();
                            return (Runnable)
                                    () -> {
                                        deliveryListeners.remove(wake);
                                        remove.run();
                                    };
                        })
                .when(facilities)
                .watchDeliveryCapacity(org.mockito.ArgumentMatchers.any());
        this.decodePreemption =
                new DefaultDecodePreemption(
                        delivery,
                        requests,
                        runtime.tasks(),
                        runtime.taskTimer(),
                        requestReporter,
                        configService,
                        capacity,
                        new PriorityDecodePreemptionPolicy());
        this.scheduler =
                SchedulerTestServices.create(
                        runtime.newLifecycle(
                                configService.loadBalanceConfig(),
                                capacity,
                                org.mockito.Mockito.mock(
                                        org.flexlb.service.TheoryCacheHitReporter.class,
                                        org.mockito.Mockito.withSettings().stubOnly())),
                        configService.loadBalanceConfig(),
                        router,
                        decodePreemption,
                        placementAvailability,
                        org.flexlb.balance.planner.BuiltinGroupPlanner.from(
                                configService.loadBalanceConfig()),
                        new OrderedRequestQueue(
                                configService.loadBalanceConfig().isPriorityOrdering()));
        this.entry = SchedulerTestServices.lifecycle(scheduler);
        runtime.registerScheduler(scheduler);
        scheduler.start();
    }

    /** Pause only new delivery admission; accepted batches and remote ACKs still execute normally. */
    public void pauseDelivery() {
        deliveryPaused = true;
    }

    public void resumeDelivery() {
        deliveryPaused = false;
        deliveryListeners.forEach(Runnable::run);
    }

    public RequestRepository requestRegistry() {
        return requests;
    }

    public AbstractRequestScheduler entry() {
        return entry;
    }

    public AbstractRequestScheduler scheduler() {
        return scheduler;
    }

    public EndpointRegistry endpointRegistry() {
        return registry;
    }

    public PlacementAvailability placementAvailability() {
        return placementAvailability;
    }

    public void bindRouter(RequestRouteSelector exactRouter) {
        router.bind(exactRouter);
    }

    /** Translate fixture response metadata into an exact queue admission. */
    public PlacementResult<RouteResult, PlacementKey> routeResult(
            RequestContext context, Response response) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(response, "response");
        if (!response.isSuccess()) {
            return PlacementResult.rejected(response);
        }
        if (response.getServerStatus() == null) {
            throw new IllegalArgumentException("successful fixture route has no worker metadata");
        }

        var roles = new java.util.ArrayList<RoleSelection>();
        for (ServerStatus status : response.getServerStatus()) {
            if (status == null) {
                continue;
            }
            String address = status.getLogicalIpPort();
            var selectedEndpoint = registry.selectEndpoint(status.getRole(), address);
            if (selectedEndpoint == null) {
                throw new IllegalStateException(
                        "fixture route references an unpublished endpoint: " + address);
            }
            roles.add(
                    SelectionTestSupport.metadata(
                            selectedEndpoint,
                            status,
                            status.getRole().supportsPrefill()
                                    ? Math.max(0L, status.getPrefillTime())
                                    : -1L));
        }
        return SchedulerTestServices.registerSelection(context, roles);
    }

    /** Apply a newer worker status or refresh liveness from a same-version heartbeat. */
    public void applyStatus(WorkerStatus status, WorkerStatusResponse response) {
        Objects.requireNonNull(status, "status");
        Objects.requireNonNull(response, "response");
        RoleType role = Objects.requireNonNull(response.getRole(), "response role");
        WorkerEndpoint endpoint = registry.get(role, status.getLogicalIpPort(), status);
        if (endpoint == null) {
            throw new IllegalStateException(
                    "status generation has no published endpoint: "
                            + status.getLogicalIpPort()
                            + "#"
                            + status.getGenerationId());
        }

        org.flexlb.balance.endpoint.WorkerEndpoint.StatusChange projection;
        status.lock.lock();
        try {
            long responseVersion =
                    Objects.requireNonNull(response.getStatusVersion(), "response status version");
            long committedVersion = status.appliedStatusCursor().statusVersion();
            if (responseVersion < committedVersion) {
                throw new IllegalArgumentException(
                        "worker status version regressed: committed="
                                + committedVersion
                                + ", response="
                                + responseVersion);
            }
            WorkerStatus.StatusObservation observation =
                    WorkerStatusTestSupport.freezeStatusResponse(status, response);
            if (responseVersion == committedVersion) {
                projection = endpoint.applyStatusHeartbeat(status, observation);
            } else {
                WorkerStatus.PreparedStatus prepared = status.prepareNewStatus(observation);
                projection = endpoint.applyPreparedStatus(status, prepared);
            }
        } finally {
            status.lock.unlock();
        }
        endpoint.publishStatus(projection);
    }

    @Override
    public void close() {
        decodePreemption.stopAdmission();
        // The harness stops worker feedback before closing, so queued work cannot drain.
        runtime.shutdown(ShutdownMode.IMMEDIATE, Duration.ofSeconds(30L));
    }

    private static ModelMetaConfig emptyModelMeta() {
        ModelMetaConfig meta = org.mockito.Mockito.mock(ModelMetaConfig.class);
        org.mockito.Mockito.when(meta.requiredRoles()).thenReturn(List.of());
        return meta;
    }

    /**
     * Binds one router after the runtime is assembled, then delegates selection and blocked-endpoint queries.
     */
    private static final class BindingRouter extends RequestRouteSelector {
        private RequestRouteSelector delegate;

        private BindingRouter(
                EndpointRegistry workers,
                ConfigService configService,
                org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity capacity) {
            // Real constructor dependencies keep Mockito instrumentation out of
            // the selector classes exercised by the bound production router.
            super(
                    workers,
                    new CostBasedPrefillStrategy(
                            org.mockito.Mockito.mock(
                                    org.flexlb.cache.match.CacheAwareService.class),
                            configService),
                    new DecodeSelector(configService, capacity),
                    new VitWorkerSelector(),
                    new org.flexlb.balance.strategy.EncoderStrategy(),
                    org.mockito.Mockito.mock(org.flexlb.cache.match.CacheAwareService.class),
                    null,
                    org.mockito.Mockito.mock(org.flexlb.service.monitor.EngineHealthReporter.class),
                    org.mockito.Mockito.mock(CacheMetricsReporter.class),
                    configService,
                    emptyModelMeta());
        }

        @Override
        public PlacementResult<RouteResult, PlacementKey> select(
                RequestContext context, String policyGroup) {
            return requireBound().select(context, policyGroup);
        }

        @Override
        WorkerEndpoint blockedEndpoint(PlacementKey key) {
            return requireBound().blockedEndpoint(key);
        }

        @Override
        PlacementResult<RouteResult, PlacementKey> select(
                RequestContext context,
                String group,
                org.flexlb.balance.endpoint.DecodeEndpoint endpoint,
                org.flexlb.balance.endpoint.DecodeResources.RequestReservation reservation) {
            return requireBound().select(context, group, endpoint, reservation);
        }

        private synchronized void bind(RequestRouteSelector exactRouter) {
            Objects.requireNonNull(exactRouter, "exactRouter");
            if (delegate != null) {
                throw new IllegalStateException("test router was already bound");
            }
            delegate = exactRouter;
        }

        private synchronized RequestRouteSelector requireBound() {
            if (delegate == null) {
                throw new IllegalStateException("test router is not bound");
            }
            return delegate;
        }
    }
}
