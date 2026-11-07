package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.TheoryCacheHitReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.util.Logger;

import java.util.Objects;

/** Accepts DIRECT requests and completes one immediate endpoint selection and handoff. */
public final class DirectRequestScheduler extends AbstractRequestScheduler {

    private final RequestRouteSelector router;
    private final java.util.concurrent.ConcurrentLinkedQueue<RequestContext> pending =
            new java.util.concurrent.ConcurrentLinkedQueue<>();

    DirectRequestScheduler(
            RequestRouteSelector router,
            RequestRepository requests,
            RequestTaskPool tasks,
            ExpirationTimer deadlines,
            DeliveryDispatcher dispatcher,
            ResponseCompletionExecutor responseCompletions,
            DeliveryMetricsReporter deliveryReporter,
            TheoryCacheHitReporter cacheTrace) {
        super(
                requests,
                tasks,
                deadlines,
                dispatcher,
                responseCompletions,
                deliveryReporter,
                cacheTrace);
        this.router = Objects.requireNonNull(router, "router");
    }

    private final java.util.concurrent.atomic.AtomicInteger acceptedWork =
            new java.util.concurrent.atomic.AtomicInteger();
    private volatile boolean started;
    private volatile boolean stopped;
    private final java.util.concurrent.atomic.AtomicBoolean scheduling =
            new java.util.concurrent.atomic.AtomicBoolean();

    @Override
    public void start() {
        if (stopped) return;
        started = true;
        schedule();
    }

    @Override
    public void schedule() {
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
        try (var task = tasks.registerRequestWork(request)) {
            acceptedWork.incrementAndGet();
            try {
                task.submit(
                        () -> {
                            try {
                                dispatchRegistered(request);
                            } finally {
                                acceptedWork.decrementAndGet();
                            }
                        });
            } catch (RuntimeException | Error failure) {
                if (task.canRetrySubmission()) acceptedWork.decrementAndGet();
                throw failure;
            }
        }
    }

    @Override
    public void stop() {
        stopped = true;
        pending.clear();
    }

    @Override
    public boolean awaitStopped(long deadlineNanos) {
        if (acceptedWork.get() != 0 && tasks.isExecutingOnCurrentThread()) return false;
        while (scheduling.get() || acceptedWork.get() != 0) {
            if (deadlineNanos != Long.MAX_VALUE && System.nanoTime() >= deadlineNanos) return false;
            java.util.concurrent.locks.LockSupport.parkNanos(100_000L);
        }
        return true;
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
    protected void removeRequest(RequestContext request) {
        pending.remove(request);
    }

    @Override
    public void onRequest(RequestContext context) {
        if (!checkRequestMode(context, false)) return;
        Objects.requireNonNull(context, "context");
        if (stopped) {
            cancel(context, 0L, CancelReason.SHUTDOWN);
            return;
        }
        pending.add(context);
        if (stopped) pending.remove(context);
        else if (started) schedule();
    }

    private void dispatchRegistered(RequestContext context) {
        PlacementOperation operation = claimPlacement(context);
        if (operation == null) return;
        Response failure = Response.error(StrategyErrorType.DISPATCH_FAILED);
        try {
            var committed = selectAndCommit(context, operation);
            failure =
                    switch (committed.status()) {
                        case SUCCESS -> null;
                        case REJECTED -> committed.failure();
                        case CLOSED -> Response.error(StrategyErrorType.REQUEST_CANCELLED);
                        case BLOCKED, ENDPOINT_CHANGED ->
                                committed.failure() != null
                                        ? committed.failure()
                                        : Response.error(committed.blocker().role().getErrorType());
                    };
        } catch (RuntimeException problem) {
            Logger.warn(
                    "DIRECT admission failed: request_id={}",
                    context.data().getRequestId(),
                    problem);
        } finally {
            if (failure != null) publish(context, failure, false);
        }
    }

    /**
     * Finish admission before publishing the route response.
     */
    private PlacementResult<RouteExecution, PlacementKey> selectAndCommit(
            RequestContext context, PlacementOperation operation) {
        try {
            var group = router.resolvePolicyGroup(context);
            // Selection and admission share one retry budget. Never replay a delivered route.
            for (int attempt = 0; ; attempt++) {
                var result = prepareSelectedRoute(context, router.select(context, group));
                context.data().setSchedulingDiagnostics(result.diagnostics());
                if (result.status() == PlacementResult.Status.SUCCESS) {
                    var publishing = operation;
                    operation = null;
                    result = publishPlacement(result.value(), publishing, null);
                    if (result.status() == PlacementResult.Status.BLOCKED)
                        return PlacementResult.rejected(
                                Response.error(StrategyErrorType.RESOURCE_EXHAUSTED));
                }
                if (result.status() == PlacementResult.Status.ENDPOINT_CHANGED && attempt == 0) {
                    if (operation == null) {
                        operation = claimPlacement(context);
                        if (operation == null) return PlacementResult.closed();
                    }
                    continue;
                }
                return result;
            }
        } finally {
            if (operation != null) AbstractRequestScheduler.finishPlacement(operation);
        }
    }
}
