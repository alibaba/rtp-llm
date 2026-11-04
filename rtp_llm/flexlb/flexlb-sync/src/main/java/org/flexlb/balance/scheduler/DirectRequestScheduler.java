package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.util.Logger;

import java.util.Objects;

/** Accepts DIRECT requests and completes one immediate endpoint selection and handoff. */
public final class DirectRequestScheduler extends AbstractRequestScheduler {

    private final RequestWorkerSelector router;
    private final java.util.concurrent.ConcurrentLinkedQueue<RequestContext> pending =
            new java.util.concurrent.ConcurrentLinkedQueue<>();

    DirectRequestScheduler(RequestWorkerSelector router, RequestLifecycle lifecycle) {
        super(lifecycle);
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
        try (var task = lifecycle.tasks.registerRequestWork(request)) {
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
        if (acceptedWork.get() != 0 && lifecycle.tasks.isExecutingOnCurrentThread()) return false;
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
    public void onRequestChanged(RequestContext request) {
        pending.remove(request);
    }

    @Override
    public void onRequest(RequestContext context) {
        if (!checkRequestMode(context, false)) return;
        Objects.requireNonNull(context, "context");
        if (stopped) {
            lifecycle.cancel(context, 0L, CancelReason.SHUTDOWN);
            return;
        }
        pending.add(context);
        if (stopped) pending.remove(context);
        else if (started) schedule();
    }

    private void dispatchRegistered(RequestContext context) {
        AdmissionHandle operation = lifecycle.claimAdmissionHandle(context);
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
            if (failure != null) lifecycle.publish(context, failure, false);
        }
    }

    /** Admission and generation pins finish before response callbacks can retire an endpoint. */
    private PlacementResult<RequestRoute, PlacementKey> selectAndCommit(
            RequestContext context, AdmissionHandle operation) {
        try {
            var group = router.resolvePolicyGroup(context);
            var selection = router.select(context, group);
            if (selection.status() == PlacementResult.Status.ENDPOINT_CHANGED) {
                selection = router.select(context, group);
            }
            context.data().setSchedulingDiagnostics(selection.diagnostics());
            if (selection.status() != PlacementResult.Status.SUCCESS) return selection.asFailure();
            var committed = lifecycle.dispatcher.submitRoute(selection.value(), operation);
            return committed.status() == PlacementResult.Status.BLOCKED
                    ? PlacementResult.rejected(Response.error(StrategyErrorType.RESOURCE_EXHAUSTED))
                    : committed;
        } finally {
            operation.finish();
        }
    }
}
