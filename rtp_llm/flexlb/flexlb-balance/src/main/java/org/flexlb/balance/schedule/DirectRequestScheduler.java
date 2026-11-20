package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.request.Response;
import org.flexlb.balance.requestlog.TheoryCacheHitReporter;
import org.flexlb.balance.strategy.RequestRouteSelector;
import org.flexlb.balance.strategy.SelectionResult;
import org.flexlb.util.Logger;
import org.flexlb.worker.endpoint.CapacityKey;
import org.flexlb.worker.endpoint.StrategyErrorType;

import java.util.LinkedHashSet;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Accepts DIRECT requests and runs endpoint selection and handoff on a bounded number of
 * per-request virtual threads. Pending requests stay with the scheduler until a slot opens.
 */
public class DirectRequestScheduler extends AbstractRequestScheduler {
    private static final int MAX_SELECTIONS =
            Math.clamp(Runtime.getRuntime().availableProcessors(), 2, 8);

    private final RequestRouteSelector router;
    private final ExecutorService selections =
            Executors.newThreadPerTaskExecutor(
                    Thread.ofVirtual().name("flexlb-direct-", 1).factory());
    private final ThreadLocal<Boolean> selecting = new ThreadLocal<>();
    // The set keeps FIFO order and lets cleanup remove a request directly by its context.
    private final LinkedHashSet<RequestContext> pending = new LinkedHashSet<>();
    private final java.util.concurrent.atomic.AtomicInteger acceptedWork =
            new java.util.concurrent.atomic.AtomicInteger();

    private volatile boolean started;
    private volatile boolean stopped;

    private final java.util.concurrent.atomic.AtomicBoolean scheduling =
            new java.util.concurrent.atomic.AtomicBoolean();

    protected DirectRequestScheduler(
            RequestRouteSelector router,
            RequestRepository requests,
            ExecutorService requestWork,
            ExpirationTimer deadlines,
            EngineRequestCanceller engineRequestCanceller,
            ResponseCompletionExecutor responseCompletions,
            DeliveryMetricsReporter deliveryReporter,
            TheoryCacheHitReporter cacheTrace) {
        super(
                requests,
                requestWork,
                deadlines,
                engineRequestCanceller,
                responseCompletions,
                cacheTrace);
        this.router = Objects.requireNonNull(router, "router");
    }

    @Override
    public void start() {
        if (stopped) return;
        started = true;
        drainPendingRequests();
    }

    @Override
    public void close() {
        stop();
        com.google.common.base.Preconditions.checkState(
                awaitStopped(Long.MAX_VALUE), "scheduler cannot wait for its own work");
        selections.close();
    }

    @Override
    public void stop() {
        synchronized (pending) {
            stopped = true;
            pending.clear();
        }
    }

    @Override
    public boolean awaitStopped(long deadlineNanos) {
        if (Boolean.TRUE.equals(selecting.get())
                || acceptedWork.get() != 0 && isExecutingRequestWork()) return false;
        while (scheduling.get()
                || acceptedWork.get() != 0
                || started && !stopped && hasPendingRequests()) {
            if (deadlineNanos != Long.MAX_VALUE && System.nanoTime() >= deadlineNanos) return false;
            java.util.concurrent.locks.LockSupport.parkNanos(100_000L);
        }
        return true;
    }

    @Override
    public void scheduleRequest(RequestContext context) {
        if (!checkRequestMode(context, false)) return;
        boolean accepting;
        synchronized (pending) {
            if (!context.canBeginRouteSelection()) return;
            accepting = !stopped;
            if (accepting) pending.add(context);
        }
        if (!accepting) {
            cancel(context, CancelReason.SHUTDOWN);
        } else if (started) {
            drainPendingRequests();
        }
    }

    public SelectionResult<RequestRoute, CapacityKey> selectAndPublish(
            RequestContext context,
            java.util.function.Supplier<SelectionResult<RequestRoute, CapacityKey>> selection) {
        var attempt = tryBeginRouteSelection(context);
        if (attempt == null) return SelectionResult.closed();
        var result = executeRouteSelection(attempt, selection);
        if (Objects.equals(context.currentSelectionAttempt(), attempt))
            finishSelectionWithoutCommit(attempt, null);
        return result;
    }

    @Override
    protected boolean isExecutingRequestWork() {
        return Boolean.TRUE.equals(selecting.get()) || super.isExecutingRequestWork();
    }

    @Override
    protected StrategyErrorType schedulingTimeoutError() {
        return StrategyErrorType.SCHEDULING_DEADLINE_EXCEEDED;
    }

    @Override
    protected void removeRequest(RequestContext request) {
        synchronized (pending) {
            pending.remove(request);
        }
    }

    @Override
    protected SelectionResult<RequestRoute, CapacityKey> selectedRouteResult(
            SelectionResult<RequestRoute, CapacityKey> result) {
        if (result.status() != SelectionResult.Status.BLOCKED) return result;
        return SelectionResult.rejected(
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
                while (!stopped
                        && acceptedWork.get() < MAX_SELECTIONS
                        && (request = pollPendingRequest()) != null) {
                    submitSelection(request);
                }
            } finally {
                scheduling.set(false);
            }
        } while (started && acceptedWork.get() < MAX_SELECTIONS && hasPendingRequests());
    }

    private RequestContext pollPendingRequest() {
        synchronized (pending) {
            return pending.isEmpty() ? null : pending.removeFirst();
        }
    }

    private void submitSelection(RequestContext request) {
        acceptedWork.incrementAndGet();
        try {
            selections.execute(
                    () -> {
                        selecting.set(true);
                        try {
                            dispatchRegistered(request);
                        } catch (RuntimeException | Error failure) {
                            recordRequestFailure(failure);
                            throw failure;
                        } finally {
                            acceptedWork.decrementAndGet();
                            try {
                                drainPendingRequests();
                            } finally {
                                selecting.remove();
                            }
                        }
                    });
        } catch (RuntimeException | Error failure) {
            acceptedWork.decrementAndGet();
            recordRequestFailure(failure);
            publishResponse(request, Response.error(StrategyErrorType.DISPATCH_FAILED), false);
            if (failure instanceof Error fatal) throw fatal;
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
        } finally {
            if (failure != null) publishResponse(context, failure, false);
        }
    }

    /**
     * Finish admission before publishing the route response.
     */
    private SelectionResult<RequestRoute, CapacityKey> selectAndCommit(RequestContext context) {
        var group = router.resolvePolicyGroup(context);
        // Selection and publication share one retry budget. Never replay a delivered route.
        for (int attempt = 0; ; attempt++) {
            var result = selectAndPublish(context, () -> router.selectAndRegister(context, group));
            if (result.status() == SelectionResult.Status.RETRY && attempt == 0) continue;
            return result;
        }
    }

    private boolean hasPendingRequests() {
        synchronized (pending) {
            return !pending.isEmpty();
        }
    }
}
