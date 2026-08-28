package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.eviction.EvictionManager;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.balance.strategy.CostBasedDecodeStrategy;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.RandomStrategy;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
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

    private final RequestRegistry lifecycle;
    private final EndpointEventProjector endpointEvents;
    private final PlacementAvailability placementAvailability =
            new PlacementAvailability();
    private final EndpointRegistry registry;
    private final RequestScheduler scheduler;
    private final SchedulerRuntime runtime;
    private final BindingRouter router = new BindingRouter();

    public RequestSchedulerTestRuntime(
            ConfigService configService,
            Supplier<CapacityBoundary.Attempt<
                    BatchDeliveryStrategy.PreparedSubmission>>
                    prepareBatchSubmission,
            BatchSchedulerReporter batchReporter,
            RequestSchedulerReporter requestReporter,
            EngineCancelChannel cancelChannel) {
        this.lifecycle = new RequestRegistry(
                configService, batchReporter, requestReporter, cancelChannel);
        this.endpointEvents = new EndpointEventProjector(lifecycle);
        DispatcherConfig dispatcher = Objects.requireNonNull(
                configService.loadBalanceConfig().getDispatcher(),
                "dispatcher");
        DeliveryStrategy deliveryStrategy = switch (dispatcher.getType()) {
            case BATCH -> {
                AtomicLong batchIds = new AtomicLong();
                yield new BatchDeliveryStrategy(
                        Objects.requireNonNull(
                                prepareBatchSubmission,
                                "prepareBatchSubmission"),
                        batchIds::incrementAndGet,
                        lifecycle,
                        new DeliveryMetrics(batchReporter));
            }
            case NON_BATCH ->
                    new RouteDeliveryStrategy(
                            lifecycle,
                            new DeliveryMetrics(batchReporter));
        };
        this.registry = new EndpointRegistry(
                configService,
                endpointEvents,
                batchReporter,
                deliveryStrategy,
                placementAvailability);
        this.scheduler = new RequestScheduler(
                configService,
                router,
                registry,
                batchReporter,
                org.mockito.Mockito.mock(EvictionManager.class),
                lifecycle,
                placementAvailability);
        this.runtime = new SchedulerRuntime(
                lifecycle, registry, batchReporter, requestReporter, scheduler);
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

    /** Return the canonical request future retained by the exact test slot. */
    public CompletableFuture<Response> requestFuture(long requestId) {
        RequestSlot slot = lifecycle.requestSlot(requestId);
        return slot == null ? null : slot.future();
    }

    /** Bind the fixture's router exactly once, after endpoints are published. */
    public void bindRouter(DefaultRouter exactRouter) {
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
                return;
            }
            WorkerStatus.PreparedStatus prepared =
                    status.prepareNewStatus(observation);
            projection = endpoint.applyPreparedStatus(status, prepared);
        } finally {
            status.lock.unlock();
        }
        projection.run();
    }

    /**
     * Convert a fixture's successful route metadata into the exact pinned
     * queue-admission capability consumed by {@link RequestScheduler}.
     */
    public PlacementResult<QueueRouteAdmission, PlacementKey> admittedRoute(
            BalanceContext context, Response response) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(response, "response");
        if (!response.isSuccess() || response.getServerStatus() == null) {
            throw new IllegalArgumentException(
                    "queue admission requires a successful route response");
        }

        List<SelectedRole> selections = new ArrayList<>();
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
                SelectedRole selected = select(pin, status);
                selections.add(selected);
            }
            return PlacementResult.success(
                    QueueRouteAdmission.prepare(context, selections, response));
        } finally {
            for (SelectedRole selection : selections) {
                selection.close();
            }
        }
    }

    private static SelectedRole select(
            WorkerEndpoint.GenerationPin pin, ServerStatus status) {
        try {
            return switch (status.getRole()) {
                case PREFILL, PDFUSION -> SelectedRole.prefill(
                        pin, status, Math.max(0L, status.getPrefillTime()));
                case DECODE -> SelectedRole.decode(
                        pin,
                        status,
                        ((DecodeEndpoint) pin.endpoint()).realKvTotal());
                case VIT -> SelectedRole.stateless(pin, status);
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

    private static final class BindingRouter extends DefaultRouter {
        private DefaultRouter delegate;

        private BindingRouter() {
            super(
                    org.mockito.Mockito.mock(CostBasedPrefillStrategy.class),
                    org.mockito.Mockito.mock(CostBasedDecodeStrategy.class),
                    org.mockito.Mockito.mock(RandomStrategy.class),
                    org.mockito.Mockito.mock(ConfigService.class),
                    emptyModelMeta(),
                    new PlacementAvailability());
        }

        private synchronized void bind(DefaultRouter exactRouter) {
            Objects.requireNonNull(exactRouter, "exactRouter");
            if (delegate != null) {
                throw new IllegalStateException("test router was already bound");
            }
            delegate = exactRouter;
        }

        @Override
        public Response routeDirect(BalanceContext context) {
            return requireBound().routeDirect(context);
        }

        @Override
        public PlacementResult<QueueRouteAdmission, PlacementKey> routeForQueue(
                BalanceContext context) {
            return requireBound().routeForQueue(context);
        }

        private synchronized DefaultRouter requireBound() {
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
