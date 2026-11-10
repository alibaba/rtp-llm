package org.flexlb.balance.scheduler;

import org.flexlb.balance.RoutingResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.TheoryCacheHitReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;

import java.util.Objects;

import static com.google.common.base.Preconditions.checkState;

/**
 * Accepts DIRECT requests and completes one immediate endpoint selection and handoff.
 */
public class DirectRequestScheduler extends AbstractRequestScheduler {
    private final RequestRouteSelector router;
    private final java.util.concurrent.ConcurrentLinkedQueue<RequestContext> pending =
            new java.util.concurrent.ConcurrentLinkedQueue<>();
    private final java.util.concurrent.atomic.AtomicInteger acceptedWork =
            new java.util.concurrent.atomic.AtomicInteger();

    private volatile boolean started;
    private volatile boolean stopped;

    private final java.util.concurrent.atomic.AtomicBoolean scheduling =
            new java.util.concurrent.atomic.AtomicBoolean();

    protected DirectRequestScheduler(
            RequestRouteSelector router,
            RequestRepository requests,
            RequestTaskPool tasks,
            ExpirationTimer deadlines,
            DeliveryCoordinator dispatcher,
            ResponseCompletionExecutor responseCompletions,
            DeliveryMetricsReporter deliveryReporter,
            TheoryCacheHitReporter cacheTrace,
            EndpointRegistry endpoints) {
        super(
                requests,
                tasks,
                deadlines,
                dispatcher,
                responseCompletions,
                deliveryReporter,
                cacheTrace,
                endpoints);
        this.router = Objects.requireNonNull(router, "router");
    }

    @Override
    public void start() {
        if (stopped) return;
        started = true;
        drainPendingRequests();
    }

    @Override
    public void onEndpointChanged(WorkerEndpoint endpoint) {}

    @Override
    public void close() {
        stop();
        com.google.common.base.Preconditions.checkState(
                awaitStopped(Long.MAX_VALUE), "scheduler cannot wait for its own work");
    }

    @Override
    public void stop() {
        stopped = true;
        pending.clear();
    }

    @Override
    public boolean awaitStopped(long deadlineNanos) {
        if (acceptedWork.get() != 0 && taskPool.isExecutingOnCurrentThread()) return false;
        while (scheduling.get() || acceptedWork.get() != 0) {
            if (deadlineNanos != Long.MAX_VALUE && System.nanoTime() >= deadlineNanos) return false;
            java.util.concurrent.locks.LockSupport.parkNanos(100_000L);
        }
        return true;
    }

    @Override
    public void scheduleRequest(RequestContext context) {
        if (!checkRequestMode(context, false)) return;
        if (stopped) {
            cancel(context, 0L, CancelReason.SHUTDOWN);
            return;
        }
        pending.add(context);
        if (stopped) pending.remove(context);
        else if (started) drainPendingRequests();
    }

    public RoutingResult<RouteResult, CapacityKey> selectAndPublish(
            RequestContext context,
            java.util.function.Supplier<RoutingResult<RouteResult, CapacityKey>> selection) {
        var routingContext = tryBeginRouteSelection(context);
        return routingContext == null
                ? RoutingResult.closed()
                : selectAndPublish(routingContext, selection);
    }

    @Override
    protected StrategyErrorType schedulingTimeoutError() {
        return StrategyErrorType.BATCH_SLO_EXPIRED;
    }

    @Override
    protected void removeRequest(RequestContext request) {
        pending.remove(request);
    }

    RoutingResult<RouteResult, CapacityKey> selectAndPublish(
            RoutingContext routingContext,
            java.util.function.Supplier<RoutingResult<RouteResult, CapacityKey>> selection) {
        checkState(
                routingContext.requestContext().snapshot().routingContext() == routingContext,
                "routing requires its exact active admission");
        RouteResult route = null;
        RoutingResult<RouteResult, CapacityKey> result = null;
        ContextResult.RouteDelivered delivered = null;
        Throwable failure = null;
        boolean routeAccepted = false;
        try {
            result = Objects.requireNonNull(selection.get(), "routing result");
            checkState(
                    result.value() == null
                            || result.value().ctx() == routingContext.requestContext(),
                    "route belongs to another request");
            route = result.value();
            routingContext
                    .requestContext()
                    .recordRoutingDiagnostics(routingContext, result.diagnostics());
            if (routingContext.requestContext().data().getFuture().isDone()
                    || !routingContext.requestContext().canPublishRoute(routingContext)) {
                result = RoutingResult.closed();
            }
            if (result.status() == RoutingResult.Status.SUCCESS) {
                result = route.validateEndpoints(result);
            }
            if (result.status() == RoutingResult.Status.SUCCESS) {
                var publication = dispatcher.dispatchSelectedRoute(route);
                if (publication.status() == RoutingResult.Status.SUCCESS) {
                    routeAccepted = true;
                    delivered = publication.value();
                } else {
                    result = publication.asFailure();
                }
            }
        } catch (RuntimeException | Error problem) {
            failure = problem;
            throw problem;
        } finally {
            Throwable cleanup;
            try {
                cleanup = finishUnpublishedRoute(routeAccepted ? null : route, routingContext);
            } catch (RuntimeException | Error completionFailure) {
                cleanup = completionFailure;
            }
            if (cleanup != null) recordRequestFailure(cleanup);
            if (failure != null) Failures.append(failure, cleanup);
            else if (delivered == null) Failures.rethrow(cleanup, "placement cleanup failed");
        }
        if (delivered != null) onDeliveryResult(delivered);
        result =
                route != null && result.status() == RoutingResult.Status.BLOCKED
                        ? RoutingResult.rejected(
                                result.failure() == null
                                        ? Response.buildErrorResponse(
                                                result.blocker().role().getErrorType(),
                                                "Selected endpoint rejected route publication: "
                                                        + result.blocker())
                                        : result.failure(),
                                result.diagnostics())
                        : result;
        return result;
    }

    private void drainPendingRequests() {
        do {
            if (stopped || !scheduling.compareAndSet(false, true)) return;
            try {
                RequestContext request;
                while (!stopped && (request = pending.poll()) != null) {
                    submitSelection(request);
                }
            } finally {
                scheduling.set(false);
            }
        } while (started && !pending.isEmpty());
    }

    private void submitSelection(RequestContext request) {
        acceptedWork.incrementAndGet();
        try {
            taskPool.execute(
                    request,
                    () -> {
                        try {
                            dispatchRegistered(request);
                        } finally {
                            acceptedWork.decrementAndGet();
                        }
                    });
        } catch (RuntimeException | Error failure) {
            acceptedWork.decrementAndGet();
            throw failure;
        }
    }

    private void dispatchRegistered(RequestContext context) {
        Response failure = Response.error(StrategyErrorType.DISPATCH_FAILED);
        try {
            var committed = selectAndCommit(context);
            failure =
                    switch (committed.status()) {
                        case SUCCESS -> null;
                        case REJECTED -> committed.failure();
                        case CLOSED -> Response.error(StrategyErrorType.REQUEST_CANCELLED);
                        case RETRY ->
                                committed.failure() != null
                                        ? committed.failure()
                                        : Response.buildErrorResponse(
                                                StrategyErrorType.SCHEDULER_PLAN_CONFLICT,
                                                "Selection requires retry; no detailed failure was recorded");
                        case BLOCKED ->
                                committed.failure() != null
                                        ? committed.failure()
                                        : Response.buildErrorResponse(
                                                committed.blocker().role().getErrorType(),
                                                "No endpoint admitted this request: "
                                                        + committed.blocker());
                    };
        } catch (RuntimeException problem) {
            failure =
                    Response.buildErrorResponse(
                            StrategyErrorType.DISPATCH_FAILED,
                            "Selection or publication failed: "
                                    + problem.getClass().getSimpleName()
                                    + ": "
                                    + problem.getMessage());
            Logger.warn(
                    "DIRECT admission failed: request_id={}",
                    context.data().getRequestId(),
                    problem);
        } catch (Error fatal) {
            recordRequestFailure(fatal);
            throw fatal;
        } finally {
            if (failure != null) publish(context, failure, false);
        }
    }

    /**
     * Finish admission before publishing the route response.
     */
    private RoutingResult<RouteResult, CapacityKey> selectAndCommit(RequestContext context) {
        var group = router.resolvePolicyGroup(context);
        // Selection and publication share one retry budget. Never replay a delivered route.
        for (int attempt = 0; ; attempt++) {
            var result = selectAndPublish(context, () -> router.select(context, group));
            if (result.status() == RoutingResult.Status.RETRY && attempt == 0) continue;
            return result;
        }
    }
}
