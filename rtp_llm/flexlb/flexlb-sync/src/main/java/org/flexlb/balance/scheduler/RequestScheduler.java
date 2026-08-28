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
 * canonicalized by {@link RequestRegistry}.
 */
@Component
public final class RequestScheduler {

    private final ConfigService configService;
    private final Router router;
    private final EndpointRegistry endpointRegistry;
    private final BatchSchedulerReporter reporter;
    private final EvictionManager evictionManager;
    private final RequestRegistry lifecycle;
    private final PlacementWaitRegistry placementWaiters;

    RequestScheduler(
            ConfigService configService,
            Router router,
            EndpointRegistry endpointRegistry,
            BatchSchedulerReporter reporter,
            EvictionManager evictionManager,
            RequestRegistry lifecycle) {
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
            RequestRegistry lifecycle,
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
        this.placementWaiters = new PlacementWaitRegistry(
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
        PlacementWaitRegistry.PlacementOrder order =
                placementWaiters.newOrder(
                        context.getPriority(), priorityOrdering);
        PlacementRequest placement = new PlacementRequest(
                context, future, priorityOrdering, order);
        long availabilitySequence = placementWaiters.availabilitySequence();
        PlacementWaitRegistry.AttemptResult initial =
                placement.attempt();
        if (initial == PlacementWaitRegistry.AttemptResult
                .Finished.INSTANCE) {
            return future;
        }
        PlacementWaitRegistry.Handle handle = placementWaiters.park(
                placement,
                order,
                (PlacementWaitRegistry.AttemptResult.Blocked) initial,
                availabilitySequence);
        future.whenComplete((ignored, failure) -> handle.close());
        return future;
    }

    /** One fresh full-selection and exact ownership attempt. */
    private PlacementWaitRegistry.AttemptResult attemptPlacement(
            BalanceContext context,
            CompletableFuture<Response> future,
            boolean priorityOrdering,
            PlacementWaitRegistry.PlacementOrder order) {
        if (future.isDone()
                || context.requestExpired(System.currentTimeMillis())) {
            return finished();
        }
        RequestRegistry.AdmissionScope mutation =
                lifecycle.beginAdmission(context.getRequestId(), future);
        if (mutation == null) {
            return finished();
        }

        PlacementKey blocked = null;
        try (mutation) {
            QueueRoutingResult routing = router.routeForQueue(context);
            if (routing.status() == QueueRoutingResult.Status.REJECTED) {
                mutation.close();
                future.complete(routing.response());
                return finished();
            }
            if (routing.status() == QueueRoutingResult.Status.BLOCKED) {
                blocked = routing.blocker();
            } else {
                QueueRouteAdmission admission = routing.admission();
                try (admission) {
                    PlacementKey predecessor =
                            placementWaiters.blockingPredecessor(
                                    order,
                                    admission.prefillPlacementKey(),
                                    admission.decodePlacementKey());
                    if (predecessor != null) {
                        return new PlacementWaitRegistry.AttemptResult
                                .Blocked(predecessor);
                    }
                    QueueRouteAdmission.PublishResult publication =
                            admission.tryPublish(context, future, lifecycle);
                    if (publication instanceof
                            QueueRouteAdmission.PublishResult.Published published) {
                        reportRouteSubmitted(context, published.item());
                        return finished();
                    }
                    if (publication instanceof
                            QueueRouteAdmission.PublishResult.Blocked unavailable) {
                        blocked = unavailable.blocker();
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
        PlacementKey predecessor = placementWaiters.blockingPredecessor(
                order, blocked);
        if (predecessor != null) {
            return new PlacementWaitRegistry.AttemptResult.Blocked(
                    predecessor);
        }
        if (priorityOrdering
                && evictionManager.tryAdmit(context, future)) {
            return finished();
        }
        return new PlacementWaitRegistry.AttemptResult.Blocked(
                blocked);
    }

    private void completeAcceptanceLimit(
            BalanceContext context,
            CompletableFuture<Response> future) {
        AdmissionFailure failure = AdmissionFailure.resourceExhausted();
        int limit = context.getConfig().queueScheduler().getLifecycle()
                .getMaxDeliveredNotAcceptedRequestsGlobal();
        future.complete(RequestRegistry
                .buildAdmissionErrorResponse(
                        failure,
                        "post-success backpressure: active_admissions="
                                + lifecycle.decodeAcceptanceCount()
                                + " limit=" + limit));
    }

    private static PlacementWaitRegistry.AttemptResult finished() {
        return PlacementWaitRegistry.AttemptResult.Finished.INSTANCE;
    }

    private final class PlacementRequest
            implements PlacementWaitRegistry.Work {
        private final BalanceContext context;
        private final CompletableFuture<Response> future;
        private final boolean priorityOrdering;
        private final PlacementWaitRegistry.PlacementOrder order;

        private PlacementRequest(
                BalanceContext context,
                CompletableFuture<Response> future,
                boolean priorityOrdering,
                PlacementWaitRegistry.PlacementOrder order) {
            this.context = context;
            this.future = future;
            this.priorityOrdering = priorityOrdering;
            this.order = order;
        }

        @Override
        public boolean done() {
            return future.isDone()
                    || context.requestExpired(System.currentTimeMillis());
        }

        @Override
        public PlacementWaitRegistry.AttemptResult attempt() {
            try {
                return attemptPlacement(
                        context, future, priorityOrdering, order);
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
            BalanceContext context, ScheduledRequest item) {
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

    public RequestState.Snapshot cancelRequest(
            long requestId,
            long expectedBatchId,
            CancelReason reason) {
        return lifecycle.cancelRequest(requestId, expectedBatchId, reason);
    }

    public int getInflightSize() {
        return lifecycle.getInflightSize();
    }

    public int getQueuedRequestCount() {
        long queued = placementWaiters.size();
        for (PrefillEndpoint endpoint
                : endpointRegistry.snapshotPrefillEndpoints().values()) {
            queued += endpoint.queuedRequestCount();
            if (queued >= Integer.MAX_VALUE) {
                return Integer.MAX_VALUE;
            }
        }
        return (int) queued;
    }

    public List<RequestState.Snapshot> snapshotActiveRequests() {
        return lifecycle.snapshotActiveRequests();
    }

    public RequestState.Snapshot getRequestState(
            long requestId, long expectedBatchId) {
        return lifecycle.getRequestState(requestId, expectedBatchId);
    }

    public boolean ownsRequestGeneration(long requestId) {
        return lifecycle.ownsRequestGeneration(requestId);
    }

    public void closePlacement() {
        placementWaiters.close();
    }

    private static Response error(
            StrategyErrorType type, String detail) {
        return RequestRegistry.buildErrorResponse(type, detail);
    }
}
