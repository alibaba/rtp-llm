package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.requestlog.TheoryCacheHitReporter;
import org.flexlb.balance.strategy.RequestRouteSelector;
import org.flexlb.balance.strategy.RoutingResult;
import org.flexlb.util.Logger;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.Response;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.flexlb.worker.endpoint.WorkerEndpoint;

import java.util.Objects;

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
            EngineCancellation engineCancellation,
            ResponseCompletionExecutor responseCompletions,
            DeliveryMetricsReporter deliveryReporter,
            TheoryCacheHitReporter cacheTrace,
            EndpointRegistry endpoints) {
        super(
                requests,
                tasks,
                deadlines,
                engineCancellation,
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
            cancel(context, CancelReason.SHUTDOWN);
            return;
        }
        pending.add(context);
        if (stopped) pending.remove(context);
        else if (started) drainPendingRequests();
    }

    public RequestResult selectAndPublish(
            RequestContext context, java.util.function.Supplier<RequestResult> selection) {
        return planRoute(context, selection, false);
    }

    @Override
    protected StrategyErrorType schedulingTimeoutError() {
        return StrategyErrorType.SCHEDULING_DEADLINE_EXCEEDED;
    }

    @Override
    protected void removeRequest(RequestContext request) {
        pending.remove(request);
    }

    @Override
    protected RequestResult selectedRouteResult(RequestResult result) {
        if (result.status() != RoutingResult.Status.BLOCKED) return result;
        return RequestResult.rejected(
                result.failure() == null
                        ? Response.buildErrorResponse(
                                org.flexlb.worker.endpoint.StrategyErrorType.forRole(
                                        result.blocker().role()),
                                "Selected endpoint rejected route publication: " + result.blocker())
                        : result.failure(),
                result.diagnostics());
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
                                                org.flexlb.worker.endpoint.StrategyErrorType
                                                        .forRole(committed.blocker().role()),
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
            Logger.warn("DIRECT admission failed: request_id={}", context.getRequestId(), problem);
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
    private RequestResult selectAndCommit(RequestContext context) {
        var group = router.resolvePolicyGroup(context);
        // Selection and publication share one retry budget. Never replay a delivered route.
        for (int attempt = 0; ; attempt++) {
            var result = selectAndPublish(context, () -> router.select(context, group));
            if (result.status() == RoutingResult.Status.RETRY && attempt == 0) continue;
            return result;
        }
    }
}
