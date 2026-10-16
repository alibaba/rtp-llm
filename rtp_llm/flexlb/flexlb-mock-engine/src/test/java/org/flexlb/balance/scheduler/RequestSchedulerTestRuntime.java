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

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** Test-only composition root for mock-engine end-to-end scheduler tests. */
public final class RequestSchedulerTestRuntime implements AutoCloseable {

    private final RequestRepository requests = new RequestRepository();
    private final PlacementAvailability placementAvailability =
            new PlacementAvailability();
    private final BindingRouter router;
    private final EndpointRegistry registry;
    private final DecodeCapacityAcquirer decodeCapacity;
    private final AbstractRequestScheduler scheduler;
    private final SchedulerRuntime runtime;
    private volatile boolean deliveryPaused;
    private final java.util.Set<Runnable> deliveryListeners = new java.util.concurrent.CopyOnWriteArraySet<>();

    /** Pause only new delivery admission; accepted batches and remote ACKs still execute normally. */
    public void pauseDelivery() { deliveryPaused = true; }

    public void resumeDelivery() {
        deliveryPaused = false;
        deliveryListeners.forEach(Runnable::run);
    }

    public RequestSchedulerTestRuntime(
            ConfigService configService,
            DeliveryDispatcher delivery,
            DeliveryMetricsReporter deliveryReporter,
            RequestSchedulerReporter requestReporter,
            EngineGrpcClient engineClient) {
        Objects.requireNonNull(delivery, "delivery");
        var capacity = new PlacementConfiguration().decodeAdmissionCapacity(configService);
        this.registry = new EndpointRegistry(configService, requests, deliveryReporter, delivery, placementAvailability);
        this.router = new BindingRouter(registry, configService, capacity);
        this.runtime = org.mockito.Mockito.spy(new SchedulerRuntime(requests, registry, deliveryReporter, requestReporter,
                delivery, configService, engineClient));
        var facilities = org.mockito.Mockito.spy(runtime.deliveries());
        org.springframework.test.util.ReflectionTestUtils.setField(runtime, "deliveries", facilities);
        // Keep transport and Dispatcher real; inject capacity pressure at its infrastructure owner.
        org.mockito.Mockito.doAnswer(invocation -> deliveryPaused
                ? CapacityBoundary.Attempt.rejected(CapacityBoundary.deliveryUnavailable(CapacityBoundary.Resource.DISPATCH_EXECUTOR))
                : invocation.callRealMethod()).when(facilities).tryReserveDelivery();
        org.mockito.Mockito.doAnswer(invocation -> !deliveryPaused && (boolean) invocation.callRealMethod())
                .when(facilities).deliveryCapacityAvailable();
        org.mockito.Mockito.doAnswer(invocation -> {
            Runnable wake = invocation.getArgument(0);
            deliveryListeners.add(wake);
            Runnable remove = (Runnable) invocation.callRealMethod();
            return (Runnable) () -> { deliveryListeners.remove(wake); remove.run(); };
        }).when(facilities).watchDeliveryCapacity(org.mockito.ArgumentMatchers.any());
        this.decodeCapacity = new DecodeCapacityAcquirer(engineClient, requests, runtime, requestReporter, configService, capacity);
        this.scheduler = PlacementConfiguration.create(runtime, configService.loadBalanceConfig(),
                router, deliveryReporter, decodeCapacity, placementAvailability, capacity, new org.flexlb.service.RecentCacheKeyTraceReporter());
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

    public void bindRouter(RequestWorkerSelector exactRouter) {
        router.bind(exactRouter);
    }

    /** Translate fixture response metadata into an exact queue admission. */
    public PlacementResult<RequestRoute, PlacementKey> routeResult(
            RequestContext context, Response response) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(response, "response");
        if (!response.isSuccess()) {
            return PlacementResult.rejected(response);
        }
        if (response.getServerStatus() == null) {
            throw new IllegalArgumentException(
                    "successful fixture route has no worker metadata");
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

    /** Apply a newer worker status or refresh liveness from a same-version heartbeat. */
    public void applyStatus(
            WorkerStatus status, WorkerStatusResponse response) {
        Objects.requireNonNull(status, "status");
        Objects.requireNonNull(response, "response");
        RoleType role = Objects.requireNonNull(response.getRole(), "response role");
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
                    response.getStatusVersion(), "response status version");
            long committedVersion = status.appliedStatusCursor().statusVersion();
            if (responseVersion < committedVersion) {
                throw new IllegalArgumentException(
                        "worker status version regressed: committed="
                                + committedVersion + ", response=" + responseVersion);
            }
            WorkerStatus.StatusObservation observation =
                    WorkerStatusTestSupport.freezeStatusResponse(status, response);
            if (responseVersion == committedVersion) {
                projection = endpoint.applyStatusHeartbeat(status, observation);
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

    private static WorkerAssignment select(
            WorkerEndpoint.GenerationPin pin, ServerStatus status) {
        try {
            return switch (status.getRole()) {
                case PREFILL, PDFUSION -> WorkerAssignment.prefill(
                        pin, status, Math.max(0L, status.getPrefillTime()),
                        ((PrefillEndpoint) pin.endpoint()).ownership.progress().ownershipVersion());
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
        decodeCapacity.shutdown();
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
