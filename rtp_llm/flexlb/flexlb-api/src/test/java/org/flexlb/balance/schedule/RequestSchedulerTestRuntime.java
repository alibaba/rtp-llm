package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.queue.OrderedRequestQueue;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeStrategy;
import org.flexlb.balance.strategy.EncoderStrategy;
import org.flexlb.balance.strategy.RequestRouteSelector;
import org.flexlb.balance.strategy.RoleSelection;
import org.flexlb.balance.strategy.VitStrategy;
import org.flexlb.config.ConfigService;
import org.flexlb.config.service.ModelMetaConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.CapacityKey;
import org.flexlb.worker.endpoint.CapacityNotifier;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.Response;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.flexlb.worker.rpc.EngineGrpcClient;
import org.flexlb.worker.status.ServerStatus;
import org.flexlb.worker.status.WorkerStatus;
import org.flexlb.worker.status.WorkerStatusResponse;
import org.flexlb.worker.status.WorkerStatusTestSupport;

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
            EngineCancellation deliveryStrategy,
            DeliveryMetricsReporter deliveryReporter,
            RequestSchedulerReporter requestReporter,
            EngineGrpcClient engineClient) {
        ConfigService runtimeConfig = configService;
        var capacity = new SchedulerConfiguration().decodeAdmissionCapacity(configService);
        this.registry =
                new EndpointRegistry(
                        configService,
                        requests,
                        deliveryReporter,
                        capacityNotifier,
                        org.mockito.Mockito.mock(org.flexlb.cache.match.CacheAwareService.class));
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
        this.runtime = assembledRuntime;
        var eviction =
                new DefaultDecodePreemption(
                        requests,
                        requestReporter,
                        configService,
                        capacity,
                        new PriorityDecodePreemptionPolicy());
        this.scheduler =
                SchedulerTestServices.create(
                        runtime.newLifecycle(
                                configService.loadBalanceConfig(),
                                capacity,
                                new org.flexlb.balance.requestlog.TheoryCacheHitReporter()),
                        configService.loadBalanceConfig(),
                        router,
                        eviction,
                        capacityNotifier,
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

    /**
     * Return the canonical request future retained by the exact test context.
     */
    public CompletableFuture<Response> requestFuture(long requestId) {
        RequestContext requestContext = requests.findActive(String.valueOf(requestId));
        return requestContext == null ? null : requestContext.getFuture();
    }

    /**
     * Return the exact item currently owned by a fixture request context.
     */
    public RequestResult activeRoute(long requestId) {
        RequestContext requestContext = requests.findActive(String.valueOf(requestId));
        if (requestContext == null) {
            return null;
        }
        synchronized (requestContext) {
            return requestContext.snapshot().route();
        }
    }

    /**
     * Bind the fixture's router exactly once, after endpoints are published.
     */
    public void bindRouter(RequestRouteSelector exactRouter) {
        router.bind(exactRouter);
    }

    /**
     * Run the same expiration maintenance edge that Spring schedules.
     */
    public void maintainExpiration() {
        runtime.maintainExpiration();
    }

    /**
     * Apply and project one exact worker-status transaction.
     */
    public void applyStatus(WorkerStatus status, WorkerStatusResponse response) {
        Objects.requireNonNull(status, "status");
        Objects.requireNonNull(response, "response");
        applyStatus(status, WorkerStatusTestSupport.freezeStatusResponse(status, response));
    }

    /**
     * Apply an immutable protocol observation without a mutable DTO hop.
     */
    public void applyStatus(WorkerStatus status, WorkerStatus.StatusObservation observation) {
        Objects.requireNonNull(status, "status");
        Objects.requireNonNull(observation, "observation");
        RoleType role = observation.role();
        WorkerEndpoint endpoint = registry.getEndpoint(role, status.getLogicalIpPort());
        if (endpoint == null) {
            throw new IllegalStateException(
                    "status generation has no published endpoint: "
                            + status.getLogicalIpPort()
                            + "#"
                            + status.getGenerationId());
        }

        WorkerEndpoint.StatusChange projection;
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
    public RequestResult routeResult(RequestContext context, Response response) {
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
            var selectedEndpoint = registry.getEndpoint(status.getRole(), address);
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
                org.flexlb.worker.endpoint.DecodeResources.AdmissionCapacity capacity) {
            // Real constructor dependencies keep Mockito instrumentation out of
            // the selector classes exercised by the bound production router.
            super(
                    workers,
                    new CostBasedPrefillStrategy(
                            configService,
                            workers,
                            org.mockito.Mockito.mock(
                                    org.flexlb.cache.match.CacheAwareService.class),
                            org.mockito.Mockito.mock(
                                    org.flexlb.cache.metrics.CacheMetricsReporter.class)),
                    new DecodeStrategy(configService, capacity, workers),
                    new VitStrategy(
                            org.mockito.Mockito.mock(org.flexlb.cache.VitCacheDirectory.class),
                            workers),
                    new EncoderStrategy(configService, workers),
                    emptyModelMeta());
        }

        @Override
        public RequestResult select(RequestContext context, String policyGroup) {
            return requireBound().select(context, policyGroup);
        }

        @Override
        public WorkerEndpoint blockedEndpoint(CapacityKey key) {
            return requireBound().blockedEndpoint(key);
        }

        @Override
        public RequestResult select(
                RequestContext context,
                String group,
                DecodeEndpoint endpoint,
                DecodeEndpoint.RequestRegistration reservation) {
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
