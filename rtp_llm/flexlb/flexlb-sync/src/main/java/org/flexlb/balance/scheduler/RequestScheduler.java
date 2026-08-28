package org.flexlb.balance.scheduler;

import org.flexlb.balance.admission.AdmissionFailure;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.eviction.EvictionManager;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

/**
 * Public request-routing facade.
 *
 * <p>The facade owns no request lifecycle state. Endpoint callbacks, exact
 * request generations, delivery claims, fences, deadlines and publication are
 * canonicalized by {@link RequestLifecycleCoordinator}.
 */
@Component
public final class RequestScheduler {

    private final ConfigService configService;
    private final Router router;
    private final EndpointRegistry endpointRegistry;
    private final BatchSchedulerReporter reporter;
    private final EvictionManager evictionManager;
    private final RequestLifecycleCoordinator lifecycle;
    private final PendingPlacementCoordinator pendingPlacement;

    RequestScheduler(
            ConfigService configService,
            Router router,
            EndpointRegistry endpointRegistry,
            BatchSchedulerReporter reporter,
            EvictionManager evictionManager,
            RequestLifecycleCoordinator lifecycle) {
        this(configService, router, endpointRegistry, reporter,
                evictionManager, lifecycle, new PlacementAvailability());
    }

    @Autowired
    RequestScheduler(
            ConfigService configService,
            Router router,
            EndpointRegistry endpointRegistry,
            BatchSchedulerReporter reporter,
            EvictionManager evictionManager,
            RequestLifecycleCoordinator lifecycle,
            PlacementAvailability placementAvailability) {
        this.configService = Objects.requireNonNull(
                configService, "configService");
        this.router = Objects.requireNonNull(router, "router");
        this.endpointRegistry = Objects.requireNonNull(
                endpointRegistry, "endpointRegistry");
        this.reporter = Objects.requireNonNull(reporter, "reporter");
        this.evictionManager = Objects.requireNonNull(
                evictionManager, "evictionManager");
        this.lifecycle = Objects.requireNonNull(lifecycle, "lifecycle");
        this.pendingPlacement = new PendingPlacementCoordinator(
                Objects.requireNonNull(
                        placementAvailability, "placementAvailability"));
    }

    /** Route one request and hand its exact generation to the lifecycle owner. */
    public CompletableFuture<Response> submit(BalanceContext context) {
        if (context == null || context.getRequest() == null) {
            return CompletableFuture.completedFuture(error(
                    StrategyErrorType.INVALID_REQUEST, null));
        }

        FlexlbConfig activeConfig;
        try {
            activeConfig = configService.loadBalanceConfig();
        } catch (Throwable failure) {
            return CompletableFuture.completedFuture(error(
                    StrategyErrorType.BATCH_DISPATCH_FAILED,
                    "Failed to load scheduler configuration: "
                            + failure.getMessage()));
        }
        int maxOutstanding = activeConfig.queueScheduler()
                .getCapacity().getMaxOutstandingRequestsGlobal();
        CompletableFuture<Response> future =
                lifecycle.register(context, maxOutstanding);
        if (future.isDone()) {
            return future;
        }

        boolean priorityOrdering = activeConfig.isPriorityOrdering();
        PlacementRequest placement = new PlacementRequest(
                context, future, priorityOrdering);
        long availabilitySequence = pendingPlacement.availabilitySequence();
        PendingPlacementCoordinator.AttemptResult initial =
                placement.attempt();
        if (initial == PendingPlacementCoordinator.AttemptResult
                .Finished.INSTANCE) {
            return future;
        }
        PendingPlacementCoordinator.Handle handle = pendingPlacement.park(
                placement,
                (PendingPlacementCoordinator.AttemptResult.Blocked) initial,
                availabilitySequence);
        future.whenComplete((ignored, failure) -> handle.close());
        return future;
    }

    /** One fresh full-selection and exact ownership attempt. */
    private PendingPlacementCoordinator.AttemptResult attemptPlacement(
            BalanceContext context,
            CompletableFuture<Response> future,
            boolean priorityOrdering) {
        if (future.isDone()) {
            return finished();
        }
        RequestLifecycleCoordinator.AdmissionScope mutation =
                lifecycle.beginAdmission(context.getRequestId(), future);
        if (mutation == null) {
            return finished();
        }

        QueueRoutingResult.Blocked blocked = null;
        try (mutation) {
            QueueRoutingResult routing = router.routeForQueue(context);
            if (routing instanceof QueueRoutingResult.Rejected rejected) {
                mutation.close();
                future.complete(rejected.response());
                return finished();
            }
            if (routing instanceof QueueRoutingResult.Blocked unavailable) {
                blocked = unavailable;
            } else {
                QueueRouteAdmission admission =
                        ((QueueRoutingResult.Admitted) routing).admission();
                try (admission) {
                    QueueRouteAdmission.PublishResult publication =
                            admission.tryPublish(context, future, lifecycle);
                    if (publication instanceof
                            QueueRouteAdmission.PublishResult.Published published) {
                        reportRouteSubmitted(context, published.item());
                        return finished();
                    }
                    if (publication instanceof
                            QueueRouteAdmission.PublishResult.Blocked unavailable) {
                        blocked = new QueueRoutingResult.Blocked(
                                unavailable.blocker(),
                                unavailable.scope());
                    } else if (publication == QueueRouteAdmission.PublishResult
                            .AcceptanceLimitReached.INSTANCE) {
                        mutation.close();
                        completeAcceptanceLimit(context, future);
                        return finished();
                    } else {
                        return finished();
                    }
                }
            }
        }

        if (future.isDone() || !lifecycle.isAdmissionOpen(
                context.getRequestId(), future)) {
            return finished();
        }
        if (priorityOrdering
                && evictionManager.tryAdmit(context, future)) {
            return finished();
        }
        return new PendingPlacementCoordinator.AttemptResult.Blocked(
                blocked.blocker(), blocked.scope());
    }

    private void completeAcceptanceLimit(
            BalanceContext context,
            CompletableFuture<Response> future) {
        AdmissionFailure failure = AdmissionFailure.resourceExhausted();
        int limit = context.getConfig().queueScheduler().getLifecycle()
                .getMaxDeliveredNotAcceptedRequestsGlobal();
        future.complete(RequestLifecycleCoordinator
                .buildAdmissionErrorResponse(
                        failure,
                        "post-success backpressure: active_admissions="
                                + lifecycle.decodeAcceptanceCount()
                                + " limit=" + limit));
    }

    private static PendingPlacementCoordinator.AttemptResult finished() {
        return PendingPlacementCoordinator.AttemptResult.Finished.INSTANCE;
    }

    private final class PlacementRequest
            implements PendingPlacementCoordinator.Work {
        private final BalanceContext context;
        private final CompletableFuture<Response> future;
        private final boolean priorityOrdering;

        private PlacementRequest(
                BalanceContext context,
                CompletableFuture<Response> future,
                boolean priorityOrdering) {
            this.context = context;
            this.future = future;
            this.priorityOrdering = priorityOrdering;
        }

        @Override
        public int priority() {
            return context.getPriority();
        }

        @Override
        public boolean priorityOrdering() {
            return priorityOrdering;
        }

        @Override
        public boolean done() {
            return future.isDone();
        }

        @Override
        public PendingPlacementCoordinator.AttemptResult attempt() {
            try {
                return attemptPlacement(
                        context, future, priorityOrdering);
            } catch (Throwable failure) {
                Logger.error(
                        "Placement transaction failed: request_id={}",
                        context.getRequestId(), failure);
                fail(failure);
                return finished();
            }
        }

        @Override
        public void fail(Throwable failure) {
            future.complete(error(
                    StrategyErrorType.BATCH_DISPATCH_FAILED,
                    "Placement failed: " + failure.getMessage()));
        }
    }

    private void reportRouteSubmitted(
            BalanceContext context, BatchItem item) {
        try {
            reporter.reportRouteSubmitTimeMs(
                    RoleType.PREFILL.name(),
                    item.prefillEp().getIp(),
                    System.currentTimeMillis() - context.getStartTime());
        } catch (RuntimeException telemetryFailure) {
            Logger.warn(
                    "Failed to record route-submit telemetry: request_id={}",
                    context.getRequestId(), telemetryFailure);
        }
    }

    public RequestLifecycleSnapshot cancelRequest(
            long requestId,
            long expectedBatchId,
            CancelReason reason) {
        return lifecycle.cancelRequest(requestId, expectedBatchId, reason);
    }

    public int getInflightSize() {
        return lifecycle.getInflightSize();
    }

    public int getQueuedRequestCount() {
        long queued = pendingPlacement.size();
        for (PrefillEndpoint endpoint
                : endpointRegistry.snapshotPrefillEndpoints().values()) {
            queued += endpoint.queuedRequestCount();
            if (queued >= Integer.MAX_VALUE) {
                return Integer.MAX_VALUE;
            }
        }
        return (int) queued;
    }

    public List<RequestLifecycleSnapshot> snapshotActiveRequests() {
        return lifecycle.snapshotActiveRequests();
    }

    public RequestLifecycleSnapshot getRequestState(
            long requestId, long expectedBatchId) {
        return lifecycle.getRequestState(requestId, expectedBatchId);
    }

    public boolean ownsRequestGeneration(long requestId) {
        return lifecycle.ownsRequestGeneration(requestId);
    }

    public void closePlacement() {
        pendingPlacement.close();
    }

    private static Response error(
            StrategyErrorType type, String detail) {
        return RequestLifecycleCoordinator.buildErrorResponse(type, detail);
    }
}
