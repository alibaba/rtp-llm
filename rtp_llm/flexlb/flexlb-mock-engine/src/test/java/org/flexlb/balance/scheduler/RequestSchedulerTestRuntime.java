package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.DecodePreemptionCoordinator;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.eviction.EvictionManager;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.CostBasedDecodeStrategy;
import org.flexlb.balance.strategy.RandomStrategy;
import org.flexlb.balance.strategy.SelectedRole;
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
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;

/** Test-only composition root for mock-engine end-to-end scheduler tests. */
public final class RequestSchedulerTestRuntime implements AutoCloseable {

    private final RequestRegistry lifecycle;
    private final EndpointEventProjector endpointEvents;
    private final PlacementAvailability placementAvailability =
            new PlacementAvailability();
    private final BindingRouter router = new BindingRouter();
    private final EndpointRegistry registry;
    private final EvictionManager evictionManager;
    private final RequestScheduler scheduler;
    private final SchedulerRuntime runtime;

    public RequestSchedulerTestRuntime(
            ConfigService configService,
            Supplier<CapacityBoundary.Attempt<
                    BatchDeliveryStrategy.PreparedSubmission>>
                    prepareBatchSubmission,
            BatchSchedulerReporter batchReporter,
            RequestSchedulerReporter requestReporter,
            EngineCancelChannel cancelChannel,
            CostBasedPrefillStrategy evictionPrefillSelector) {
        Objects.requireNonNull(
                prepareBatchSubmission, "prepareBatchSubmission");
        this.lifecycle = new RequestRegistry(
                configService, batchReporter, requestReporter, cancelChannel);
        this.endpointEvents = new EndpointEventProjector(lifecycle);
        AtomicLong batchIds = new AtomicLong();
        this.registry = new EndpointRegistry(
                configService,
                endpointEvents,
                batchReporter,
                new BatchDeliveryStrategy(
                        prepareBatchSubmission,
                        batchIds::incrementAndGet,
                        lifecycle,
                        new DeliveryMetrics(batchReporter)),
                placementAvailability);
        EvictionPlacement placement = new EvictionPlacement(
                router,
                evictionPrefillSelector,
                lifecycle,
                batchReporter);
        this.evictionManager = new EvictionManager(
                registry,
                requestReporter,
                cancelChannel,
                new DecodePreemptionCoordinator(cancelChannel, lifecycle),
                lifecycle,
                placement);
        this.scheduler = new RequestScheduler(
                configService,
                router,
                registry,
                batchReporter,
                evictionManager,
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

    public void bindRouter(DefaultRouter exactRouter) {
        router.bind(exactRouter);
    }

    /** Translate fixture response metadata into an exact queue admission. */
    public PlacementResult<QueueRouteAdmission, PlacementKey> routeResult(
            BalanceContext context, Response response) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(response, "response");
        if (!response.isSuccess()) {
            return PlacementResult.rejected(response);
        }
        if (response.getServerStatus() == null) {
            throw new IllegalArgumentException(
                    "successful fixture route has no worker metadata");
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

    /** Apply and project one exact, strictly newer worker-status response. */
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

        Runnable projection;
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
            if (responseVersion == committedVersion) {
                return;
            }
            WorkerStatus.StatusObservation observation =
                    status.freezeStatusResponse(response);
            WorkerStatus.PreparedStatus prepared =
                    status.prepareNewStatus(observation);
            projection = endpoint.applyPreparedStatus(status, prepared);
        } finally {
            status.lock.unlock();
        }
        projection.run();
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
        evictionManager.shutdown();
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
