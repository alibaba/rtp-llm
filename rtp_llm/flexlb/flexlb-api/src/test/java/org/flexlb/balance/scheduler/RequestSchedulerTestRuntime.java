package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.DecodeCapacityAcquirer;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.VitWorkerSelector;
import org.flexlb.balance.strategy.WorkerAssignment;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;

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

    public RequestSchedulerTestRuntime(
            ConfigService configService,
            Supplier<CapacityBoundary.Attempt<
                    DefaultBatchDispatcher.PreparedSubmission>>
                    prepareBatchSubmission,
            DeliveryMetricsReporter deliveryReporter,
            RequestSchedulerReporter requestReporter) {
        this(configService, prepareBatchSubmission, deliveryReporter, requestReporter,
                localDeliveryCleanup());
    }

    /** In-memory delivery fixtures own no remote resources; model their explicit cleanup response. */
    private static org.flexlb.balance.eviction.EngineCancelChannel localDeliveryCleanup() {
        var channel = org.mockito.Mockito.mock(org.flexlb.balance.eviction.EngineCancelChannel.class);
        org.mockito.Mockito.when(channel.cancel(org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.anyLong(),
                org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.anyLong()))
                .thenReturn(CompletableFuture.completedFuture(org.flexlb.balance.eviction.EngineCancelChannel.CancelAck.REQUEST_CLEANED));
        return channel;
    }

    public RequestSchedulerTestRuntime(ConfigService configService,
            Supplier<CapacityBoundary.Attempt<DefaultBatchDispatcher.PreparedSubmission>> prepareBatchSubmission,
            DeliveryMetricsReporter deliveryReporter, RequestSchedulerReporter requestReporter,
            org.flexlb.balance.eviction.EngineCancelChannel cancelChannel) {
        DispatcherConfig dispatcher = Objects.requireNonNull(
                configService.loadBalanceConfig().getDispatcher(),
                "dispatcher");
        DeliveryStrategy deliveryStrategy = switch (dispatcher.getType()) {
            case BATCH -> {
                AtomicLong batchIds = new AtomicLong();
                yield new BatchDeliveryStrategy(Objects.requireNonNull(
                                prepareBatchSubmission,
                                "prepareBatchSubmission"), batchIds::incrementAndGet, deliveryReporter);
            }
            case NON_BATCH ->
                    new RouteDeliveryStrategy(deliveryReporter);
        };
        this.registry = new EndpointRegistry(configService, requests, deliveryReporter, deliveryStrategy, placementAvailability);
        this.router = new BindingRouter(registry);
        this.runtime = new SchedulerRuntime(requests, registry, deliveryReporter, requestReporter,
                org.mockito.Mockito.mock(DefaultBatchDispatcher.class), configService,
                new org.flexlb.service.RecentCacheKeyTraceReporter(), cancelChannel);
        var eviction = new DecodeCapacityAcquirer(cancelChannel, requests, runtime, requestReporter);
        this.scheduler = PlacementConfiguration.create(runtime, configService.loadBalanceConfig(),
                router, deliveryReporter, eviction, placementAvailability);
        runtime.initializeScheduler(scheduler);
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
        return requestContext == null ? null : requestContext.future();
    }

    /** Return the exact item currently owned by a fixture request context. */
    public RequestRoute activeRoute(long requestId) {
        RequestContext requestContext = requests.findActive(requestId);
        if (requestContext == null) {
            return null;
        }
        synchronized (requestContext) {
            return requestContext.activeRoute();
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
        applyStatus(status, status.freezeStatusResponse(response));
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

        Runnable projection;
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
        projection.run();
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

        List<WorkerAssignment> selections = new ArrayList<>();
        try {
            for (ServerStatus status : response.getServerStatus()) {
                if (status == null) {
                    continue;
                }
                String address = status.getServerIp() + ":" + status.getHttpPort();
                WorkerEndpoint.GenerationPin pin = registry.capture(
                        status.getRole(), address);
                if (pin == null) {
                    throw new IllegalStateException(
                            "fixture route references an unpublished endpoint: "
                                    + status.getRole() + " " + address);
                }
                WorkerAssignment selected = select(pin, status);
                selections.add(selected);
            }
            var result = PlacementResult.<RequestRoute, PlacementKey>success(
                    RequestRoute.prepare(context, selections));
            selections.clear();
            return result;
        } finally {
            for (WorkerAssignment selection : selections) {
                selection.close();
            }
        }
    }

    private static WorkerAssignment select(
            WorkerEndpoint.GenerationPin pin, ServerStatus status) {
        try {
            return switch (status.getRole()) {
                case PREFILL, PDFUSION -> WorkerAssignment.prefill(
                        pin, status, Math.max(0L, status.getPrefillTime()),
                        ((PrefillEndpoint) pin.endpoint()).placementVersion());
                case DECODE -> WorkerAssignment.decode(
                        pin, status, ((DecodeEndpoint) pin.endpoint()).placementVersion());
                case VIT -> WorkerAssignment.stateless(pin, status);
                case FRONTEND -> throw new IllegalArgumentException(
                        "FRONTEND cannot be a worker route");
            };
        } catch (RuntimeException | Error failure) {
            pin.close();
            throw failure;
        }
    }

    @Override
    public void close() {
        runtime.shutdown();
    }

    private static final class BindingRouter extends RequestWorkerSelector {
        private RequestWorkerSelector delegate;

        private BindingRouter(EndpointRegistry workers) {
            // Real constructor dependencies keep Mockito instrumentation out of
            // the selector classes exercised by the bound production router.
            super(new CostBasedPrefillStrategy(workers,
                            org.mockito.Mockito.mock(org.flexlb.cache.service.CacheAwareService.class),
                            org.mockito.Mockito.mock(org.flexlb.service.monitor.EngineHealthReporter.class), org.mockito.Mockito.mock(CacheMetricsReporter.class)),
                    new DecodeSelector(workers),
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
