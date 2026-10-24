package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.DecodeCapacityAcquirer;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.VitWorkerSelector;
import org.flexlb.cache.monitor.CacheMetricsReporter;
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

import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

import static org.mockito.ArgumentMatchers.anyInt;

/**
 * Test-only composition root for API integration fixtures.
 *
 * <p>The production scheduler deliberately exposes only its facade. This
 * fixture lives in the scheduler package so tests can assemble the same
 * package-private lifecycle owners without reopening production constructors.
 */
public final class RequestSchedulerTestRuntime implements AutoCloseable {

    private final RequestRepository requests = new RequestRepository();
    private final PlacementAvailability placementAvailability =
            new PlacementAvailability();
    private final EndpointRegistry registry;
    private final AbstractRequestScheduler scheduler;
    private final SchedulerRuntime runtime;
    private final BindingRouter router;

    public RequestSchedulerTestRuntime(ConfigService configService, EngineGrpcClient grpcClient,
            Runnable preparationBarrier, DeliveryMetricsReporter deliveryReporter,
            RequestSchedulerReporter requestReporter) {
        this(configService, new DeliveryDispatcher(grpcClient, configService, deliveryReporter),
                deliveryReporter, requestReporter, localDeliveryCleanup(), preparationBarrier);
    }

    /** In-memory delivery fixtures own no remote resources; model their explicit cleanup response. */
    private static EngineGrpcClient localDeliveryCleanup() {
        var channel = org.mockito.Mockito.mock(EngineGrpcClient.class);
        org.mockito.Mockito.when(channel.cancelAsync(org.mockito.ArgumentMatchers.any(), anyInt(),
                org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.anyLong()))
                .thenReturn(CompletableFuture.completedFuture(EngineGrpcClient.CancelReply.REQUEST_CLEANED));
        return channel;
    }

    public RequestSchedulerTestRuntime(ConfigService configService, DeliveryDispatcher deliveryStrategy,
            DeliveryMetricsReporter deliveryReporter, RequestSchedulerReporter requestReporter,
            EngineGrpcClient engineClient) {
        this(configService, deliveryStrategy, deliveryReporter, requestReporter, engineClient, null);
    }

    private RequestSchedulerTestRuntime(ConfigService configService, DeliveryDispatcher deliveryStrategy,
            DeliveryMetricsReporter deliveryReporter, RequestSchedulerReporter requestReporter,
            EngineGrpcClient engineClient, Runnable preparationBarrier) {
        ConfigService runtimeConfig = DeliveryDispatcherTestFactory.runtimeConfigFor(deliveryStrategy, configService);
        var capacity = new PlacementConfiguration().decodeAdmissionCapacity(configService);
        this.registry = new EndpointRegistry(configService, requests, deliveryReporter, deliveryStrategy, placementAvailability);
        this.router = new BindingRouter(registry, configService, capacity);
        var assembledRuntime = new SchedulerRuntime(requests, registry, deliveryReporter, requestReporter,
                deliveryStrategy, runtimeConfig, engineClient);
        // Intercept the real batch-admission facility while all request policy remains production code.
        this.runtime = assembledRuntime;
        if (preparationBarrier != null) {
            var deliveryExecution = org.mockito.Mockito.spy(runtime.deliveries());
            org.springframework.test.util.ReflectionTestUtils.setField(runtime, "deliveries", deliveryExecution);
            org.mockito.Mockito.doAnswer(invocation -> {
                preparationBarrier.run();
                return invocation.callRealMethod();
            }).when(deliveryExecution).tryReserveDelivery();
        }
        var eviction = new DecodeCapacityAcquirer(deliveryStrategy, requests, runtime, requestReporter, configService, capacity);
        this.scheduler = PlacementConfiguration.create(runtime, configService.loadBalanceConfig(),
                router, deliveryReporter, eviction, placementAvailability, capacity, new org.flexlb.service.RecentCacheKeyTraceReporter());
        runtime.registerScheduler(scheduler);
    }

    public RequestRepository requestRegistry() {
        return requests;
    }

    public RequestScheduler scheduler() {
        return scheduler;
    }

    public EndpointRegistry endpointRegistry() {
        return registry;
    }

    public PlacementAvailability placementAvailability() {
        return placementAvailability;
    }

    /** Return the canonical request future retained by the exact test context. */
    public CompletableFuture<Response> requestFuture(long requestId) {
        RequestContext requestContext = requests.findActive(requestId);
        return requestContext == null ? null : requestContext.data().getFuture();
    }

    /** Return the exact item currently owned by a fixture request context. */
    public RequestRoute activeRoute(long requestId) {
        RequestContext requestContext = requests.findActive(requestId);
        if (requestContext == null) {
            return null;
        }
        synchronized (requestContext) {
            return requestContext.snapshot().route();
        }
    }

    /** Bind the fixture's router exactly once, after endpoints are published. */
    public void bindRouter(RequestWorkerSelector exactRouter) {
        router.bind(exactRouter);
    }

    /** Run the same expiration maintenance edge that Spring schedules. */
    public void maintainExpiration() {
        runtime.maintainExpiration();
    }

    /** Apply and project one exact worker-status transaction. */
    public void applyStatus(
            WorkerStatus status, WorkerStatusResponse response) {
        Objects.requireNonNull(status, "status");
        Objects.requireNonNull(response, "response");
        applyStatus(status, WorkerStatusTestSupport.freezeStatusResponse(status, response));
    }

    /** Apply an immutable protocol observation without a mutable DTO hop. */
    public void applyStatus(
            WorkerStatus status,
            WorkerStatus.StatusObservation observation) {
        Objects.requireNonNull(status, "status");
        Objects.requireNonNull(observation, "observation");
        RoleType role = observation.role();
        WorkerEndpoint endpoint = registry.get(
                role, status.getIpPort(), status);
        if (endpoint == null) {
            throw new IllegalStateException(
                    "status generation has no published endpoint: "
                            + status.getIpPort() + "#" + status.getGenerationId());
        }

        org.flexlb.balance.endpoint.WorkerEndpoint.StatusChange projection;
        status.lock.lock();
        try {
            long responseVersion = Objects.requireNonNull(
                    observation.statusVersion(), "response status version");
            long committedVersion = status.appliedStatusCursor().statusVersion();
            if (responseVersion < committedVersion) {
                throw new IllegalArgumentException(
                        "worker status version regressed: committed="
                                + committedVersion + ", response=" + responseVersion);
            }
            if (responseVersion == committedVersion) {
                projection = endpoint.applyStatusHeartbeat(
                        status, observation);
            } else {
                WorkerStatus.PreparedStatus prepared =
                        status.prepareNewStatus(observation);
                projection = endpoint.applyPreparedStatus(status, prepared);
            }
        } finally {
            status.lock.unlock();
        }
        endpoint.publishStatus(projection);
    }

    /**
     * Convert a fixture's successful route metadata into the exact pinned
     * queue-admission capability consumed by {@link AbstractRequestScheduler}.
     */
    public PlacementResult<RequestRoute, PlacementKey> admittedRoute(
            RequestContext context, Response response) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(response, "response");
        if (!response.isSuccess() || response.getServerStatus() == null) {
            throw new IllegalArgumentException(
                    "queue admission requires a successful route response");
        }

        RequestRoute route = new RequestRoute(context);
        try {
            for (ServerStatus status : response.getServerStatus()) {
                if (status == null) { continue; }
                String address = status.getServerIp() + ":" + status.getHttpPort();
                var pin = registry.capture(status.getRole(), address);
                if (pin == null) { throw new IllegalStateException("fixture route references an unpublished endpoint: " + address); }
                long version = pin.endpoint() instanceof PrefillEndpoint prefill
                        ? prefill.ownership.progress().ownershipVersion()
                        : pin.endpoint() instanceof DecodeEndpoint decode ? decode.placementVersion() : 0L;
                route.add(pin, status, status.getRole().supportsPrefill() ? Math.max(0L, status.getPrefillTime()) : -1L, version);
            }
            return PlacementResult.success(route.prepare());
        } catch (RuntimeException | Error failure) {
            org.flexlb.util.Failures.append(failure, org.flexlb.util.Failures.close(route));
            throw failure;
        }
    }



    @Override
    public void close() {
        runtime.shutdown();
    }

    private static final class BindingRouter extends RequestWorkerSelector {
        private RequestWorkerSelector delegate;

        private BindingRouter(EndpointRegistry workers, ConfigService configService,
                              org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity capacity) {
            // Real constructor dependencies keep Mockito instrumentation out of
            // the selector classes exercised by the bound production router.
            super(new CostBasedPrefillStrategy(workers,
                            org.mockito.Mockito.mock(org.flexlb.cache.service.CacheAwareService.class),
                            org.mockito.Mockito.mock(org.flexlb.service.monitor.EngineHealthReporter.class), org.mockito.Mockito.mock(CacheMetricsReporter.class), configService),
                    new DecodeSelector(workers, configService, capacity),
                    new VitWorkerSelector(workers),
                    emptyModelMeta());
        }

        private synchronized void bind(RequestWorkerSelector exactRouter) {
            Objects.requireNonNull(exactRouter, "exactRouter");
            if (delegate != null) {
                throw new IllegalStateException("test router was already bound");
            }
            delegate = exactRouter;
        }

        @Override
        public PlacementResult<RequestRoute, PlacementKey> select(
                RequestContext context, String policyGroup) {
            return requireBound().select(context, policyGroup);
        }

        private synchronized RequestWorkerSelector requireBound() {
            if (delegate == null) {
                throw new IllegalStateException("test router is not bound");
            }
            return delegate;
        }
    }

    private static ModelMetaConfig emptyModelMeta() {
        ModelMetaConfig meta = org.mockito.Mockito.mock(ModelMetaConfig.class);
        org.mockito.Mockito.when(meta.requiredRoles()).thenReturn(List.of());
        return meta;
    }
}
