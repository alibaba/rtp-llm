package org.flexlb.balance.scheduler;

import org.flexlb.balance.admission.AdmissionFallback;
import org.flexlb.balance.delivery.BatchDeliveryStrategy;
import org.flexlb.balance.delivery.BatchSubmissionPort;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.delivery.RouteDeliveryStrategy;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointEvent;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.EndpointStatusReduction;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.config.BatchDispatcherConfig;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.NonBatchDispatcherConfig;
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

/**
 * Test-only composition root for API integration fixtures.
 *
 * <p>The production scheduler deliberately exposes only its facade. This
 * fixture lives in the scheduler package so tests can assemble the same
 * package-private lifecycle owners without reopening production constructors.
 */
public final class RequestSchedulerTestRuntime implements AutoCloseable {

    private final RequestLifecycleCoordinator lifecycle;
    private final EndpointRegistry registry;
    private final RequestScheduler scheduler;
    private final RequestExpirationOrchestrator expiration;
    private final RequestShutdownOrchestrator shutdown;
    private final BindingRouter router = new BindingRouter();

    public RequestSchedulerTestRuntime(
            ConfigService configService,
            BatchSubmissionPort batchSubmission,
            BatchSchedulerReporter batchReporter,
            RequestSchedulerReporter requestReporter,
            EngineCancelChannel cancelChannel) {
        this.lifecycle = new RequestLifecycleCoordinator(
                configService, batchReporter, requestReporter, cancelChannel);
        DispatcherConfig dispatcher = Objects.requireNonNull(
                configService.loadBalanceConfig().getDispatcher(),
                "dispatcher");
        DeliveryStrategy deliveryStrategy = switch (dispatcher) {
            case BatchDispatcherConfig ignored -> {
                AtomicLong batchIds = new AtomicLong();
                yield new BatchDeliveryStrategy(
                        Objects.requireNonNull(
                                batchSubmission, "batchSubmission"),
                        new BatchPrefillAdmission(batchIds::incrementAndGet),
                        lifecycle,
                        new DeliveryTelemetryAdapter(batchReporter));
            }
            case NonBatchDispatcherConfig ignored ->
                    new RouteDeliveryStrategy(
                            new RoutePrefillAdmission(),
                            lifecycle,
                            new DeliveryTelemetryAdapter(batchReporter));
        };
        this.registry = new EndpointRegistry(
                configService,
                lifecycle,
                batchReporter,
                deliveryStrategy,
                new WorkerBatcherFactory());
        AdmissionFallback noPriorityTakeover = (context, future) -> false;
        this.scheduler = new RequestScheduler(
                configService,
                router,
                registry,
                batchReporter,
                noPriorityTakeover,
                lifecycle);
        this.expiration = new RequestExpirationOrchestrator(
                lifecycle, registry);
        this.shutdown = new RequestShutdownOrchestrator(
                lifecycle, registry);
    }

    public RequestScheduler scheduler() {
        return scheduler;
    }

    public EndpointRegistry endpointRegistry() {
        return registry;
    }

    /** Bind the fixture's router exactly once, after endpoints are published. */
    public void bindRouter(Router exactRouter) {
        router.bind(exactRouter);
    }

    /** Run the same expiration maintenance edge that Spring schedules. */
    public void maintainExpiration() {
        expiration.maintainExpiration();
    }

    /** Apply and project one exact worker-status transaction. */
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

        EndpointStatusReduction reduction;
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
            reduction = endpoint.applyPreparedStatus(status, prepared);
        } finally {
            status.lock.unlock();
        }
        lifecycle.onEndpointEvent(new EndpointEvent.StatusReduced(reduction));
    }

    /**
     * Convert a fixture's successful route metadata into the exact pinned
     * queue-admission capability consumed by {@link RequestScheduler}.
     */
    public QueueRoutingResult admittedRoute(
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
            return new QueueRoutingResult.Admitted(
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
        shutdown.shutdown();
    }

    private static final class BindingRouter implements Router {
        private Router delegate;

        private synchronized void bind(Router exactRouter) {
            Objects.requireNonNull(exactRouter, "exactRouter");
            if (delegate != null) {
                throw new IllegalStateException(
                        "test router was already bound");
            }
            delegate = exactRouter;
        }

        @Override
        public Response routeDirect(BalanceContext context) {
            return requireBound().routeDirect(context);
        }

        @Override
        public QueueRoutingResult routeForQueue(BalanceContext context) {
            return requireBound().routeForQueue(context);
        }

        private synchronized Router requireBound() {
            if (delegate == null) {
                throw new IllegalStateException(
                        "test router is not bound");
            }
            return delegate;
        }
    }
}
