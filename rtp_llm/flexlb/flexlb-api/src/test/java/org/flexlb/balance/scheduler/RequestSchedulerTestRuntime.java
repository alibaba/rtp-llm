package org.flexlb.balance.scheduler;

import org.flexlb.balance.RoutingResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.preemption.DefaultDecodePreemption;
import org.flexlb.balance.preemption.PriorityDecodePreemptionPolicy;
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
import java.util.concurrent.CompletableFuture;

/**
 * Test-only composition root for API integration fixtures.
 *
 * <p>The production scheduler deliberately exposes only its facade. This
 * fixture lives in the scheduler package so tests can assemble the same
 * package-private lifecycle owners without reopening production constructors.
 */
public final class RequestSchedulerTestRuntime implements AutoCloseable {
    private final RequestRepository requests = new RequestRepository();
    private final CapacityNotifier capacityNotifier = new CapacityNotifier();
    private final EndpointRegistry registry;
    private final AbstractRequestScheduler scheduler;
    private final AbstractRequestScheduler entry;
    private final SchedulerTestServices runtime;
    private final BindingRouter router;

    public RequestSchedulerTestRuntime(
            ConfigService configService,
            EngineGrpcClient grpcClient,
            Runnable preparationBarrier,
            DeliveryMetricsReporter deliveryReporter,
            RequestSchedulerReporter requestReporter) {
        this(
                configService,
                new DeliveryCoordinator(
                        new org.flexlb.balance.scheduler.RequestRepository(),
                        grpcClient,
                        configService,
                        deliveryReporter,
                        new org.flexlb.balance.scheduler.SchedulerConfiguration()
                                .decodeAdmissionCapacity(configService),
                        org.mockito.Mockito.mock(DeliveryExecution.class)),
                deliveryReporter,
                requestReporter,
                grpcClient,
                preparationBarrier);
    }

    public RequestSchedulerTestRuntime(
            ConfigService configService,
            DeliveryCoordinator deliveryStrategy,
            DeliveryMetricsReporter deliveryReporter,
            RequestSchedulerReporter requestReporter,
            EngineGrpcClient engineClient) {
        this(
                configService,
                deliveryStrategy,
                deliveryReporter,
                requestReporter,
                engineClient,
                null);
    }

    private RequestSchedulerTestRuntime(
            ConfigService configService,
            DeliveryCoordinator deliveryStrategy,
            DeliveryMetricsReporter deliveryReporter,
            RequestSchedulerReporter requestReporter,
            EngineGrpcClient engineClient,
            Runnable preparationBarrier) {
        ConfigService runtimeConfig =
                DeliveryDispatcherTestFactory.runtimeConfigFor(deliveryStrategy, configService);
        var capacity = new SchedulerConfiguration().decodeAdmissionCapacity(configService);
        this.registry =
                new EndpointRegistry(
                        configService,
                        requests,
                        deliveryReporter,
                        deliveryStrategy,
                        capacityNotifier,
                        org.flexlb.balance.planner.BuiltinGroupPlanner.from(
                                configService.loadBalanceConfig()));
        this.router = new BindingRouter(registry, configService, capacity);
        var assembledRuntime =
                new SchedulerTestServices(
                        requests,
                        registry,
                        deliveryReporter,
                        requestReporter,
                        deliveryStrategy,
                        runtimeConfig,
                        engineClient);
        // Intercept the real batch-admission facility while all request policy remains production
        // code.
        this.runtime = assembledRuntime;
        if (preparationBarrier != null) {
            var deliveryExecution = org.mockito.Mockito.spy(runtime.deliveries());
            SchedulerTestServices.setDependency(runtime, "deliveries", deliveryExecution);
            org.mockito.Mockito.doAnswer(
                            invocation -> {
                                preparationBarrier.run();
                                return invocation.callRealMethod();
                            })
                    .when(deliveryExecution)
                    .tryReserveDelivery();
        }
        var eviction =
                new DefaultDecodePreemption(
                        deliveryStrategy,
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
                                new org.flexlb.service.TheoryCacheHitReporter()),
                        configService.loadBalanceConfig(),
                        router,
                        eviction,
                        capacityNotifier,
                        org.flexlb.balance.planner.BuiltinGroupPlanner.from(
                                configService.loadBalanceConfig()),
                        new OrderedRequestQueue(
                                configService.loadBalanceConfig().isPriorityOrdering()));
        entry = SchedulerTestServices.lifecycle(scheduler);
        runtime.registerScheduler(scheduler);
        scheduler.start();
    }

    public AbstractRequestScheduler entry() {
        return entry;
    }

    public RequestRepository requestRegistry() {
        return requests;
    }

    public AbstractRequestScheduler scheduler() {
        return scheduler;
    }

    public EndpointRegistry endpointRegistry() {
        return registry;
    }

    public CapacityNotifier capacityNotifier() {
        return capacityNotifier;
    }

    /** Return the canonical request future retained by the exact test context. */
    public CompletableFuture<Response> requestFuture(long requestId) {
        RequestContext requestContext = requests.findActive(String.valueOf(requestId));
        return requestContext == null ? null : requestContext.data().getFuture();
    }

    /** Return the exact item currently owned by a fixture request context. */
    public RouteResult activeRoute(long requestId) {
        RequestContext requestContext = requests.findActive(String.valueOf(requestId));
        if (requestContext == null) {
            return null;
        }
        synchronized (requestContext) {
            return requestContext.snapshot().route();
        }
    }

    /** Bind the fixture's router exactly once, after endpoints are published. */
    public void bindRouter(RequestRouteSelector exactRouter) {
        router.bind(exactRouter);
    }

    /** Run the same expiration maintenance edge that Spring schedules. */
    public void maintainExpiration() {
        runtime.maintainExpiration();
    }

    /** Apply and project one exact worker-status transaction. */
    public void applyStatus(WorkerStatus status, WorkerStatusResponse response) {
        Objects.requireNonNull(status, "status");
        Objects.requireNonNull(response, "response");
        applyStatus(status, WorkerStatusTestSupport.freezeStatusResponse(status, response));
    }

    /** Apply an immutable protocol observation without a mutable DTO hop. */
    public void applyStatus(WorkerStatus status, WorkerStatus.StatusObservation observation) {
        Objects.requireNonNull(status, "status");
        Objects.requireNonNull(observation, "observation");
        RoleType role = observation.role();
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
                    Objects.requireNonNull(observation.statusVersion(), "response status version");
            long committedVersion = status.appliedStatusCursor().statusVersion();
            if (responseVersion < committedVersion) {
                throw new IllegalArgumentException(
                        "worker status version regressed: committed="
                                + committedVersion
                                + ", response="
                                + responseVersion);
            }
            projection = endpoint.applyWorkerStatus(status, status.prepareStatus(observation));
        } finally {
            status.lock.unlock();
        }
        endpoint.publishStatus(projection);
    }

    /**
     * Convert a fixture's successful route metadata into the exact pinned
     * queue-admission capability consumed by {@link AbstractRequestScheduler}.
     */
    public RoutingResult<RouteResult, CapacityKey> routeResult(
            RequestContext context, Response response) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(response, "response");
        if (!response.isSuccess() || response.getServerStatus() == null) {
            throw new IllegalArgumentException(
                    "queue admission requires a successful route response");
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

    @Override
    public void close() {
        runtime.shutdown(ShutdownMode.IMMEDIATE, Duration.ZERO);
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
        public RoutingResult<RouteResult, CapacityKey> select(
                RequestContext context, String policyGroup) {
            return requireBound().select(context, policyGroup);
        }

        @Override
        WorkerEndpoint blockedEndpoint(CapacityKey key) {
            return requireBound().blockedEndpoint(key);
        }

        @Override
        RoutingResult<RouteResult, CapacityKey> select(
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
