package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.preemption.PreemptionRequestResult;
import org.flexlb.balance.scheduler.ResponseCompletionExecutor.Registration;
import org.flexlb.balance.scheduler.ResponseCompletionExecutor.RequestFuture;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.TheoryCacheHitReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.telemetry.FlexlbTrace;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;

import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

import static com.google.common.base.Preconditions.checkState;
import static org.flexlb.balance.scheduler.ContextResult.Cancellation;
import static org.flexlb.balance.scheduler.ContextResult.CancellationFacts;
import static org.flexlb.balance.scheduler.ContextResult.DeliveryEvidence;
import static org.flexlb.balance.scheduler.ContextResult.EndEvidence;
import static org.flexlb.balance.scheduler.ContextResult.EndKind;
import static org.flexlb.balance.scheduler.ContextResult.Phase;
import static org.flexlb.balance.scheduler.ContextResult.Prediction;
import static org.flexlb.balance.scheduler.ContextResult.PreparationFailure;
import static org.flexlb.balance.scheduler.ContextResult.ResponseCandidate;
import static org.flexlb.balance.scheduler.ContextResult.ResponseKind;
import static org.flexlb.balance.scheduler.ContextResult.RouteDelivered;
import static org.flexlb.balance.scheduler.ContextResult.RoutingCompletion;
import static org.flexlb.balance.scheduler.ContextResult.SendState;
import static org.flexlb.balance.scheduler.ContextResult.Snapshot;
import static org.flexlb.balance.scheduler.ContextResult.WorkerRequestEvent;
import static org.flexlb.balance.scheduler.ContextResult.WorkerRequestEventType;
import static org.flexlb.balance.scheduler.ResponseCompletionExecutor.RequestFuture.Completion.RESPONSE;
import static org.flexlb.util.Failures.recover;

/**
 * Registers requests and coordinates cancellation, deadlines, responses and resource cleanup.
 * Subclasses own route selection, publication and retry policy; RequestContext serializes request facts.
 * Cleanup drains accepted work before the shared endpoints and executors close.
 */
public abstract class AbstractRequestScheduler implements RequestScheduler {
    protected final RequestRepository repository;
    protected final RequestTaskPool taskPool;
    protected final ExpirationTimer expirationTimer;
    protected final DeliveryCoordinator dispatcher;
    protected final ResponseCompletionExecutor responseCompletions;
    protected final DeliveryMetricsReporter deliveryReporter;
    protected final TheoryCacheHitReporter cacheTrace;
    protected final EndpointRegistry endpoints;
    private final Object shutdownLock = new Object();
    private final RequestCloseQueue closeQueue = new RequestCloseQueue();

    private Throwable requestProcessingFailure;

    protected AbstractRequestScheduler(
            RequestRepository repository,
            RequestTaskPool taskPool,
            ExpirationTimer deadlines,
            DeliveryCoordinator dispatcher,
            ResponseCompletionExecutor responseCompletions,
            DeliveryMetricsReporter deliveryReporter,
            TheoryCacheHitReporter cacheTrace,
            EndpointRegistry endpoints) {
        this.repository = Objects.requireNonNull(repository);
        this.taskPool = Objects.requireNonNull(taskPool);
        this.expirationTimer = Objects.requireNonNull(deadlines);

        this.dispatcher = Objects.requireNonNull(dispatcher);
        this.responseCompletions = Objects.requireNonNull(responseCompletions);
        this.deliveryReporter = Objects.requireNonNull(deliveryReporter);
        this.cacheTrace = Objects.requireNonNull(cacheTrace);
        this.endpoints = Objects.requireNonNull(endpoints);
    }

    public CompletableFuture<Response> submit(RequestContext context) {
        return submit(context, () -> {});
    }

    public CompletableFuture<Response> submit(RequestContext ctx, Runnable onRegistered) {
        Objects.requireNonNull(onRegistered, "onRegistered");
        try {
            if (ctx == null || ctx.data().getRequest() == null) {
                return CompletableFuture.completedFuture(
                        Response.error(StrategyErrorType.INVALID_REQUEST));
            }
            var data = ctx.data();
            var config = data.getConfig();
            if (repository.isStopping() || repository.isClosed()) {
                return CompletableFuture.completedFuture(
                        Response.buildErrorResponse(
                                StrategyErrorType.SCHEDULER_UNAVAILABLE,
                                "request scheduler is shutting down and no longer accepts new requests"));
            }
            if (data.getRequestScheduler() != null) {
                return CompletableFuture.completedFuture(
                        Response.buildErrorResponse(
                                StrategyErrorType.INVALID_REQUEST,
                                "request context is already registered"));
            }
            if (data.getSchedulingMetadata() == null) {
                return CompletableFuture.completedFuture(
                        Response.buildErrorResponse(
                                StrategyErrorType.INVALID_REQUEST, "missing scheduling metadata"));
            }
            long expires = data.getRequestExpiresAtMs();
            if (expires <= 0L || System.currentTimeMillis() >= expires) {
                var rejected =
                        CompletableFuture.completedFuture(
                                Response.buildErrorResponse(
                                        schedulingTimeoutError(),
                                        "request scheduling deadline has expired before placement"));
                data.setFuture(rejected);
                return rejected;
            }
            var inputFailure = dispatcher.validateRequestInput(data);
            if (inputFailure != null) return CompletableFuture.completedFuture(inputFailure);
            var future =
                    new RequestFuture(
                            (completion, response, failure) ->
                                    tryCompleteFromExternal(
                                            ctx, completion, response, failure, true));
            var result =
                    repository.register(
                            ctx,
                            this,
                            future,
                            config.getRequestLifecycle().getRequest().getTimeoutMs(),
                            schedulingTimeoutError());
            if (result == RequestRepository.RegistrationResult.CLOSED) {
                return CompletableFuture.completedFuture(
                        Response.buildErrorResponse(
                                StrategyErrorType.SCHEDULER_UNAVAILABLE,
                                "request scheduler is shutting down and no longer accepts new requests"));
            }
            if (result != RequestRepository.RegistrationResult.REGISTERED) {
                return CompletableFuture.completedFuture(
                        Response.buildErrorResponse(
                                StrategyErrorType.INVALID_REQUEST,
                                result == RequestRepository.RegistrationResult.DUPLICATE_ID
                                        ? "duplicate request_id: " + data.getRequestId()
                                        : "request registration rejected: " + result));
            }
            dispatcher.releaseUnusedInput(data);
            data.setEnqueueTime(System.currentTimeMillis());
            FlexlbTrace.setScheduleAttribute(
                    data.getTraceContext(),
                    FlexlbTrace.SCHEDULE_MODE,
                    config.deliveryType(data.getPhase())
                                    == org.flexlb.config.DispatcherConfig.Type.BATCH
                            ? "BATCH"
                            : config.getScheduler().getType().name());
            future.whenComplete(
                    (response, failure) -> {
                        try {
                            if (failure == null && response != null && response.isSuccess()) {
                                cacheTrace.report(ctx);
                            }
                        } catch (RuntimeException error) {
                            Logger.warn("Route tracing failed", error);
                        } finally {
                            ctx.data().releaseSchedulingInput();
                        }
                    });
            expirationTimer.update(ctx, ctx.snapshot().deadlines());
            onRegistered.run();
            if (repository.isClosed()) {
                cancel(ctx, 0L, CancelReason.SHUTDOWN);
            } else if (System.currentTimeMillis() >= expires) {
                cancel(ctx, 0L, CancelReason.DEADLINE_EXCEEDED);
            }
            if (!future.isDone()) scheduleRequest(ctx);
            else closeRequest(ctx, null);
            return future;
        } catch (RuntimeException failure) {
            Response response =
                    Response.buildErrorResponse(
                            StrategyErrorType.DISPATCH_FAILED, failure.getMessage());
            if (ctx != null && ctx.data().getRequestScheduler() == this) {
                publish(ctx, response, false);
                return ctx.data().getFuture();
            }
            return CompletableFuture.completedFuture(response);
        }
    }

    public RequestState cancel(String id, long batchId, CancelReason reason) {
        return cancel(id, batchId, reason, org.flexlb.dao.route.RequestPhase.GENERATION);
    }

    public RequestState cancel(
            String id, long batchId, CancelReason reason, org.flexlb.dao.route.RequestPhase phase) {
        Objects.requireNonNull(reason);
        var ctx = repository.findActive(id, phase);
        if (ctx != null) {
            if (ctx.data().getRequestScheduler() != this) {
                return null;
            }
            var result = cancel(ctx, batchId, reason);
            if (result != null) {
                return result;
            }
        }
        var terminal = repository.findTerminal(id, phase);
        return terminal != null
                        && terminal.owner() == this
                        && terminal.state().matchesBatch(batchId)
                ? terminal.state()
                : null;
    }

    /**
     * Cancellation freezes its first cause, then directs the owner of the real resource.
     */
    public RequestState cancel(RequestContext ctx, long batchId, CancelReason reason) {
        if (ctx == null || ctx.data().getRequestScheduler() != this || !repository.isCurrent(ctx)) {
            return null;
        }

        boolean changed;
        synchronized (ctx) {
            var state = ctx.snapshot();
            if (state.phase() == Phase.CLOSED) {
                return state.terminalRecord();
            }
            String message = reason.getMessage();
            Map<String, Object> evidence = describeWait(ctx, reason, message);
            var error =
                    reason == CancelReason.DEADLINE_EXCEEDED
                            ? schedulingTimeoutError()
                            : StrategyErrorType.REQUEST_CANCELLED;
            changed =
                    ctx.requestCancel(
                            new Cancellation(
                                    batchId,
                                    reason,
                                    false,
                                    System.currentTimeMillis(),
                                    error,
                                    evidence == null
                                            ? message
                                            : String.valueOf(evidence.get("cause"))));
            if (changed && evidence != null) {
                ctx.data().setSchedulingDiagnostics(evidence);
            }
        }
        Throwable failure = null;
        if (changed) {
            failure = Failures.run(failure, () -> closeRequest(ctx, null));
        }
        var deadlines = ctx.snapshot().deadlines();
        if (ctx.snapshot().phase() == Phase.OPEN) {
            failure = Failures.run(failure, () -> expirationTimer.update(ctx, deadlines));
        }
        if (failure != null) {
            recordRequestFailure(failure);
        }
        Failures.rethrow(failure, "cancellation effects failed");
        var state = RequestState.from(ctx.snapshot());
        return state != null && state.matchesBatch(batchId) ? state : null;
    }

    public synchronized void recordRequestFailure(Throwable failure) {
        requestProcessingFailure = Failures.append(requestProcessingFailure, failure);
    }

    /**
     * Local submission failure uses the same arbitration as external Future completion.
     */
    public boolean publish(RequestContext ctx, Response response, boolean synchronous) {
        return tryCompleteFromExternal(
                ctx, RequestFuture.Completion.RESPONSE, response, null, synchronous);
    }

    public boolean isAdmissionOpen(RequestContext ctx) {
        if (repository.isClosed() || ctx == null || ctx.data().getRequestScheduler() != this) {
            return false;
        }
        synchronized (ctx) {
            var state = ctx.snapshot();
            return repository.isCurrent(ctx)
                    && state.phase() == Phase.OPEN
                    && state.cancellation() == null;
        }
    }

    /**
     * Check that this selecting task may still publish a route; no candidate is stored in Context.
     */
    public RoutingContext currentRoutingContext(RouteResult route) {
        var ctx = route.ctx();
        synchronized (ctx) {
            var state = ctx.snapshot();
            if (repository.isClosed()
                    || !repository.isCurrent(ctx)
                    || ctx.data().getRequestScheduler() != this
                    || state.routingContext() == null) return null;
            var routingContext = state.routingContext();
            return ctx.canPublishRoute(routingContext) ? routingContext : null;
        }
    }

    public static void finishRouting(RoutingContext routingContext) {
        completeRouting(
                routingContext,
                routingContext.routeToWithdraw() == null
                        ? RoutingCompletion.FINISHED
                        : RoutingCompletion.RESTORED);
    }

    public static void completeRouting(RoutingContext routingContext, RoutingCompletion result) {
        com.google.common.base.Preconditions.checkArgument(
                result == RoutingCompletion.FINISHED
                        || result == RoutingCompletion.RESTORED
                        || result == RoutingCompletion.WITHDRAWN,
                "invalid placement completion");
        routingContext
                .requestContext()
                .data()
                .getRequestScheduler()
                .completeRouting(routingContext, result, null);
    }

    /**
     * Record an exact delivery result, then settle or continue the request outside its lock.
     */
    public void onDeliveryResult(DeliveryEvidence result) {
        if (result instanceof RouteDelivered delivered) {
            recordPredictedDelivery(
                    delivered.delivery(),
                    delivered.nowMs(),
                    () -> delivered.delivery().owner().recordDeliveryResult(delivered));
        } else if (result instanceof ContextResult.TransportResult transport) {
            var ctx = transport.delivery().owner();
            if (!isCurrentRequest(ctx)) return;
            var failure = ctx.unsentDeliveryFailure(transport);
            if (failure != null && onRouteFailure(failure, transport)) return;
            recordRequestChange(ctx, () -> ctx.recordDeliveryResult(transport));
        } else if (result instanceof ContextResult.DeliveryNotStarted rejected) {
            onDeliveryNotStarted(rejected);
        } else throw new IllegalArgumentException("expected delivery completion");
    }

    public boolean isCurrentRequest(RequestContext ctx) {
        return ctx.data().getRequestScheduler() == this && repository.isCurrent(ctx);
    }

    /**
     * All unsent route failures share the same withdrawal and cleanup, independent of the failing role.
     */
    public boolean onRouteFailure(
            RouteResult route, ContextResult.RouteFailureReason reason, String detail) {
        return onRouteFailure(new ContextResult.RouteFailure(route, reason, detail), null);
    }

    public void recordRemoteCleanup(ContextResult.CleanupAcknowledgement cleanup) {
        var ctx = cleanup.delivery().owner();
        recordRequestChange(ctx, () -> ctx.recordRemoteCleanup(cleanup));
    }

    public static void failRouting(RoutingContext routingContext, Response failure) {
        Objects.requireNonNull(failure, "failure");
        com.google.common.base.Preconditions.checkArgument(
                !failure.isSuccess(), "placement failure requires an error response");
        routingContext
                .requestContext()
                .data()
                .getRequestScheduler()
                .completeRouting(routingContext, RoutingCompletion.FAILED, failure);
    }

    public void onRequestStateChanged(RequestContext ctx) {
        if (!isCurrentRequest(ctx)) return;
        try {
            processPendingRequestActions(ctx);
        } catch (RuntimeException | Error failure) {
            recordRequestFailure(failure);
        }
    }

    /**
     * Advance on the status caller; user callbacks and BATCH cleanup keep their own executors.
     */
    public void onWorkerRequestEvent(RequestContext ctx, WorkerRequestEvent event) {
        if (ctx == null || ctx.data().getRequestScheduler() != this) {
            return;
        }
        try {
            Snapshot previousState;
            Snapshot updatedState;
            synchronized (ctx) {
                previousState = ctx.snapshot();
                updatedState = ctx.recordWorkerRequestEvent(event);
            }
            if (updatedState == null || updatedState == previousState) return;
            if (!needsFollowUpActions(previousState, updatedState)) return;
            processPendingRequestActions(ctx);
        } catch (Throwable failure) {
            recordRequestFailure(failure);
            Logger.error("Worker request event processing failed", failure);
        }
    }

    public void onEndpointRetired(RequestContext ctx, ContextResult.EndpointRetirement retired) {
        if (ctx == null || ctx.data().getRequestScheduler() != this) return;
        if (ctx.isCurrentEndpointRetirement(retired)
                && retired.route() != null
                && onRouteFailure(
                        retired.route(),
                        ContextResult.RouteFailureReason.ENDPOINT_UNAVAILABLE,
                        retired.detail())) return;
        try {
            Snapshot before;
            Snapshot after;
            synchronized (ctx) {
                before = ctx.snapshot();
                after = ctx.recordEndpointRetirement(retired);
            }
            if (after == null || after == before) return;
            if (!needsFollowUpActions(before, after)) return;
            processPendingRequestActions(ctx);
        } catch (Throwable failure) {
            recordRequestFailure(failure);
            Logger.error("Endpoint retirement processing failed", failure);
        }
    }

    public void rejectRoute(RouteResult route, Throwable cause) {
        var ctx = route.ctx();
        if (ctx.data().getRequestScheduler() != this || !repository.isCurrent(ctx)) {
            return;
        }
        if (onRouteFailure(
                route,
                route.selectedEndpointsAvailable()
                        ? ContextResult.RouteFailureReason.PREPARATION_FAILED
                        : ContextResult.RouteFailureReason.ENDPOINT_UNAVAILABLE,
                cause == null ? "Endpoint queue closed" : cause.getMessage())) return;
        closeRequest(
                ctx,
                EndEvidence.failure(
                        route,
                        EndKind.FAILURE,
                        StrategyErrorType.DISPATCH_FAILED,
                        "Worker scheduling queue rejected request: "
                                + (cause == null
                                        ? "endpoint publication failed"
                                        : cause.getMessage())));
    }

    public void onDeadline(ExpirationTimer.Token exact, long nowMs) {
        var ctx = exact.owner();
        if (ctx.data().getRequestScheduler() != this || !repository.isCurrent(ctx)) {
            return;
        }

        Snapshot after;
        synchronized (ctx) {
            after = recordDeadline(ctx, exact, nowMs, null);
        }
        if (after == null) {
            return;
        }
        taskPool.execute(
                ctx,
                () -> {
                    var current = ctx.snapshot();
                    if (nowMs < exact.atMs()) {
                        expirationTimer.update(ctx, current.deadlines());
                        return;
                    }
                    processPendingRequestActions(ctx);
                });
    }

    /**
     * Drain accepted responses until the caller's absolute nanoTime deadline, then close scheduling.
     */
    public final boolean shutdown(long deadlineNanos) {
        checkShutdownThread();
        synchronized (shutdownLock) {
            if (repository.isClosed()) return repository.liveRequestCount() == 0;
            repository.stopAccepting();
            boolean drained = false;
            Throwable failure = null;
            try {
                drained = repository.awaitResponses(deadlineNanos);
            } catch (Throwable cause) {
                failure = cause;
            }
            if (!repository.closeRegistration()) return drained;
            failure = Failures.run(failure, this::close);
            var owners = new LinkedHashSet<AbstractRequestScheduler>();
            owners.add(this);
            var active = repository.snapshotActive();
            for (var context : active) {
                owners.add(context.data().getRequestScheduler());
                failure =
                        Failures.run(
                                failure,
                                () ->
                                        context.data()
                                                .getRequestScheduler()
                                                .cancel(context, 0L, CancelReason.SHUTDOWN));
            }
            failure = Failures.run(failure, () -> dispatcher.shutdown(active));
            failure = Failures.run(failure, taskPool::awaitIdle);
            for (var owner : owners) failure = Failures.run(failure, owner.closeQueue::awaitIdle);
            failure = Failures.run(failure, endpoints::close);
            failure = Failures.run(failure, () -> dispatcher.close(active));
            failure = Failures.run(failure, expirationTimer::close);
            failure = Failures.run(failure, taskPool::awaitIdle);
            for (var owner : owners)
                failure = Failures.run(failure, owner::closeOutstandingAndTerminalize);
            failure = Failures.run(failure, this::closeExecutors);
            for (var owner : owners) failure = Failures.run(failure, owner::throwIfRequestFailed);
            if (repository.liveRequestCount() != 0)
                failure =
                        Failures.append(
                                failure,
                                new IllegalStateException(
                                        "unsettled requests at shutdown: "
                                                + repository.liveRequestCount()));
            Failures.rethrow(failure, "scheduling shutdown failed");
            return drained;
        }
    }

    /**
     * Reject reentrant shutdown before an application lifecycle acquires its shutdown monitor.
     */
    public void checkShutdownThread() {
        checkState(
                !taskPool.isExecutingOnCurrentThread()
                        && !dispatcher.isStoppingOnCurrentThread()
                        && !closeQueue.isExecutingOnCurrentThread()
                        && !responseCompletions.isCompletingOnCurrentThread(),
                "scheduler shutdown cannot wait for its own request work or response callback");
    }

    protected RoutingContext tryBeginRouteSelection(RequestContext ctx) {
        if (repository.isClosed() || ctx == null || ctx.data().getRequestScheduler() != this) {
            return null;
        }
        synchronized (ctx) {
            if (!repository.isCurrent(ctx)) {
                return null;
            }
            return ctx.beginRouteSelection();
        }
    }

    protected abstract StrategyErrorType schedulingTimeoutError();

    protected Map<String, Object> describeWait(
            RequestContext request, CancelReason reason, String message) {
        if (reason != CancelReason.DEADLINE_EXCEEDED) return null;
        var state = request.snapshot();
        String stage =
                state.delivery() != null
                        ? "DELIVERY"
                        : state.routingContext() != null
                                ? "PLACEMENT"
                                : state.route() == null
                                        ? "WAITING_FOR_PLACEMENT"
                                        : "BEFORE_DELIVERY";
        return completeWaitDiagnostics(message, describeWaitAtStage(request, state, stage));
    }

    protected final Map<String, Object> completeWaitDiagnostics(
            String message, Map<String, Object> details) {
        details.put("cause", message + "; " + details);
        return java.util.Collections.unmodifiableMap(details);
    }

    protected final Map<String, Object> describeWaitAtStage(
            RequestContext request, Snapshot state, String stage) {
        Map<String, Object> details = new java.util.LinkedHashMap<>();
        details.put("stage", stage);
        details.put("capturedAtMs", System.currentTimeMillis());
        if (state.routingContext() != null) {
            details.put("routingPhase", state.routingPhase().name());
        }
        if (state.delivery() == null && state.route() == null) {
            var observation = request.data().getSchedulingDiagnostics();
            if (observation != null) {
                details.put("lastPlacementObservation", observation);
            }
        }
        return details;
    }

    protected final boolean checkRequestMode(RequestContext context, boolean queued) {
        if (context.data().getConfig().isQueue() == queued) return true;
        publish(
                context,
                Response.buildErrorResponse(
                        StrategyErrorType.DISPATCH_FAILED,
                        "request configuration does not match scheduler mode"),
                true);
        return false;
    }

    /**
     * Remove this exact request from the implementation's scheduling queues.
     */
    protected abstract void removeRequest(RequestContext request);

    protected boolean onRouteFailure(
            ContextResult.RouteFailure failure, DeliveryEvidence evidence) {
        return false;
    }

    /**
     * Select one effect owner after recording facts; that owner also handles its failures.
     */
    protected void processPendingRequestActions(RequestContext ctx) {
        var state = ctx.snapshot();
        Throwable failure = null;
        if (state.preemption() != null) {
            try {
                org.flexlb.balance.preemption.PreemptionCompletion.reconcile(
                        ctx, this::recordRequestFailure);
            } catch (RuntimeException | Error failed) {
                failure = failed;
            }
            state = ctx.snapshot();
        }
        try {
            if (state.cancellation() != null
                    && state.pendingEnd() == null
                    && state.phase() == Phase.OPEN) {
                scheduleClose(ctx);
            } else if (state.pendingEnd() != null || state.phase() == Phase.ENDING) {
                closeRequest(ctx, null);
            } else if (state.phase() == Phase.OPEN) {
                confirmDelivery(ctx);
            }
            var current = ctx.snapshot();
            if (current.phase() == Phase.OPEN) {
                recover(
                        this::recordRequestFailure,
                        () -> expirationTimer.update(ctx, current.deadlines()));
            }
        } catch (RuntimeException | Error failed) {
            failure = Failures.append(failure, failed);
        }
        Failures.rethrow(failure, "request continuation failed");
    }

    protected int encoderRequestLimit() {
        return Integer.MAX_VALUE;
    }

    void recordDeliveryPrediction(Prediction prediction) {
        recordPredictedDelivery(
                prediction.delivery(),
                prediction.nowMs(),
                () -> prediction.delivery().owner().recordDeliveryPrediction(prediction));
    }

    void recordPreparationFailure(PreparationFailure failure) {
        var ctx = failure.delivery() == null ? failure.route().ctx() : failure.delivery().owner();
        if (!isCurrentRequest(ctx)) return;
        var routeFailure = ctx.unsentDeliveryFailure(failure);
        if (routeFailure != null && onRouteFailure(routeFailure, failure)) return;
        recordRequestChange(ctx, () -> ctx.recordPreparationFailure(failure));
    }

    synchronized void throwIfRequestFailed() {
        Failures.rethrow(requestProcessingFailure, "request processing failed");
    }

    /**
     * End placement. Return resource cleanup failures; admission completion failures propagate.
     */
    static Throwable finishUnpublishedRoute(RouteResult route, RoutingContext routingContext) {
        Throwable failure =
                route == null
                        ? null
                        : releaseRouteResources(
                                route, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        boolean rollbackFailed = failure != null;
        if (routingContext != null) {
            if (rollbackFailed) route.ctx().retainRouteForCleanup(routingContext, route);
            try {
                if (rollbackFailed)
                    AbstractRequestScheduler.failRouting(
                            routingContext,
                            Response.buildErrorResponse(
                                    StrategyErrorType.DISPATCH_FAILED,
                                    "placement rollback failed"));
                else AbstractRequestScheduler.finishRouting(routingContext);
            } catch (RuntimeException | Error completion) {
                Failures.append(completion, failure);
                throw completion;
            }
        }
        return failure;
    }

    /**
     * One resource-release implementation for admission rollback, route withdrawal and request end.
     * Rollback checks every acquired resource; request end retains successful release progress.
     */
    static Throwable releaseRouteResources(
            RouteResult route, DecodeResources.ReleaseReason reason) {
        var ctx = route.ctx();
        var state = ctx.snapshot();
        boolean rollback =
                reason == DecodeResources.ReleaseReason.LOCAL_ROLLBACK
                        || reason == DecodeResources.ReleaseReason.UNSENT_ROUTE_ROLLBACK;
        boolean prefillAndEncoderReleased =
                !rollback && state.route() == route && state.close().prefillSettled();
        boolean decodeReleased =
                !rollback && state.route() == route && state.close().decodeSettled();
        Throwable failure = null;
        if (!prefillAndEncoderReleased) {
            if (route.encoderEndpoint() != null)
                failure = Failures.run(failure, () -> route.encoderEndpoint().release(route));
            if (route.prefillEndpoint() != null)
                failure =
                        Failures.run(failure, () -> route.prefillEndpoint().releaseRequest(route));
            prefillAndEncoderReleased = failure == null;
        }
        try {
            if (!decodeReleased) {
                decodeReleased =
                        RoleSelection.Decode.release(
                                route.decodeEndpoint(), route.decodeReservation(), reason);
            }
        } catch (Throwable problem) {
            failure = Failures.append(failure, problem);
        }
        ctx.recordResourcesReleased(
                route, prefillAndEncoderReleased, decodeReleased, System.currentTimeMillis());
        return failure;
    }

    static void preparationFailed(RouteResult route, String detail) {
        route.ctx()
                .data()
                .getRequestScheduler()
                .recordPreparationFailure(
                        new PreparationFailure(null, route, detail, System.currentTimeMillis()));
    }

    void completeRouting(
            RoutingContext routingContext, RoutingCompletion result, Response failure) {
        var ctx = routingContext.requestContext();
        checkState(ctx.data().getRequestScheduler() == this, "foreign routing context");
        checkState(
                routingContext.routeToWithdraw() == null, "scheduler cannot remove a queued route");
        if (!ctx.beginRoutingCompletion(routingContext)) return;
        var after = ctx.completeRouting(routingContext, result, System.currentTimeMillis());
        checkState(after != null, "routing completion lost its request context");
        finishRoutingRequest(ctx, after, result, failure, null);
    }

    /**
     * Applies one end-reason precedence after routing, then closes, requeues or resumes the request.
     */
    final void finishRoutingRequest(
            RequestContext ctx,
            Snapshot after,
            RoutingCompletion result,
            Response failure,
            Throwable problem) {
        var end = after.pendingEnd();
        if (failure != null
                && (end == null
                        || end.kind() == EndKind.SHUTDOWN && after.cancellation() == null)) {
            end = EndEvidence.publicationFailure(after.route(), failure);
        }
        if (end != null) {
            var actual = end;
            problem = Failures.run(problem, () -> closeRequest(ctx, actual));
        } else if (result == RoutingCompletion.WITHDRAWN
                && after.phase() == Phase.OPEN
                && !ctx.data().getFuture().isDone()) {
            try {
                if (System.currentTimeMillis() >= ctx.data().getRequestExpiresAtMs()) {
                    cancel(ctx, 0L, CancelReason.DEADLINE_EXCEEDED);
                } else {
                    scheduleRequest(ctx);
                }
                expirationTimer.update(ctx, ctx.snapshot().deadlines());
            } catch (Throwable requeueFailure) {
                problem = Failures.append(problem, requeueFailure);
                var withdrawn = EndEvidence.withdrawnFailure("queued route requeue failed");
                problem = Failures.run(problem, () -> closeRequest(ctx, withdrawn));
            }
        } else {
            problem = Failures.run(problem, () -> processPendingRequestActions(ctx));
        }
        if (problem != null) recordRequestFailure(problem);
        Failures.rethrow(problem, "routing completion failed");
    }

    /**
     * Own closing through completion or a recorded wait, including recovery after an effect fails.
     */
    void closeRequest(RequestContext ctx, EndEvidence evidence) {
        if (ctx.data().getRequestScheduler() != this || !repository.isCurrent(ctx)) {
            return;
        }
        Registration publication = null;
        try {
            synchronized (ctx) {
                ResponseCandidate candidate = null;
                var before = ctx.snapshot();
                if (before.phase() == Phase.CLOSED
                        || evidence == null
                                && before.phase() != Phase.ENDING
                                && before.pendingEnd() == null
                        || evidence != null
                                && evidence.route() != null
                                && before.route() != evidence.route()) {
                    return;
                }
                EndKind ending =
                        before.pendingEnd() != null
                                ? before.pendingEnd().kind()
                                : evidence != null ? evidence.kind() : before.outcome().kind();
                if (before.outcome() == null
                        && ctx.readyToClose(ending)
                        && before.selectedResponse() == null
                        && !ctx.data().getFuture().isDone()) {
                    candidate =
                            new ResponseCandidate(
                                    ctx, ResponseKind.TERMINAL, System.currentTimeMillis());
                    publication = reserveResponse(ctx, candidate);
                }
                var after = ctx.beginClose(evidence, System.currentTimeMillis());
                if (publication != null && after.outcome() != null) {
                    var outcome = after.outcome();
                    if (outcome.kind() == EndKind.DELIVERY_FAILURE) {
                        ctx.selectResponse(candidate);
                    }
                    boolean batched =
                            after.delivery() != null
                                    && after.delivery().identity().kind()
                                            == Delivery.SendMode.BATCH_ENQUEUE;
                    publication.prepare(RESPONSE, response(outcome, before.route(), batched), null);
                } else if (publication != null) {
                    publication.close();
                    publication = null;
                }
            }
        } catch (RuntimeException | Error failed) {
            if (publication != null) publication.failIfUnsubmitted(failed);
            Failures.run(failed, () -> scheduleClose(ctx));
            throw failed;
        }
        submitClose(ctx, publication);
    }

    void scheduleClose(RequestContext ctx) {
        if (Thread.holdsLock(ctx)) {
            throw new IllegalStateException(
                    "request closing requires the Context lock to be released");
        }
        var state = ctx.snapshot();
        if (state.phase() == Phase.OPEN
                && state.pendingEnd() == null
                && state.cancellation() == null) return;

        submitClose(ctx, null);
    }

    final boolean tryCompleteFromExternal(
            RequestContext ctx,
            RequestFuture.Completion completion,
            Response response,
            Throwable failure,
            boolean synchronous) {
        checkState(
                !Thread.holdsLock(ctx), "Future callbacks require the Context lock to be released");
        if (!isCurrentRequest(ctx)) return false;
        Registration publication;
        synchronized (ctx) {
            var before = ctx.snapshot();
            boolean cancelled = completion == RequestFuture.Completion.CANCELLATION;
            if (before.phase() != Phase.OPEN
                    || before.selectedResponse() != null
                    || ctx.data().getFuture().isDone()
                    || before.cancellation() != null
                            && (!cancelled
                                    || before.cancellation().reason()
                                            == CancelReason.DEADLINE_EXCEEDED)) return false;
            boolean success = completion == RESPONSE && (response == null || response.isSuccess());
            if (success
                    || before.delivery() != null
                    || before.engine().decodeAccepted()
                    || before.preemption() != null
                    || before.routingContext() != null && !cancelled) return false;
            var candidate =
                    new ResponseCandidate(
                            ctx,
                            cancelled ? ResponseKind.CANCELLATION : ResponseKind.TERMINAL,
                            System.currentTimeMillis());
            publication = reserveResponse(ctx, candidate);
            try {
                if (cancelled
                        && before.cancellation() == null
                        && !ctx.requestCancel(
                                new Cancellation(
                                        0L,
                                        CancelReason.CLIENT_CANCELLED,
                                        true,
                                        System.currentTimeMillis(),
                                        StrategyErrorType.REQUEST_CANCELLED,
                                        CancelReason.CLIENT_CANCELLED.getMessage()))) {
                    publication.close();
                    return false;
                }
                if (!cancelled
                        && ctx.beginClose(
                                                EndEvidence.externalFailure(
                                                        before.route(), response, failure),
                                                System.currentTimeMillis())
                                        .outcome()
                                == null) {
                    publication.close();
                    return false;
                }
                ctx.selectResponse(candidate);
                publication.prepare(
                        failure != null ? RequestFuture.Completion.FAILURE : completion,
                        response,
                        failure);
            } catch (RuntimeException | Error failed) {
                publication.close();
                throw failed;
            }
        }
        var cleanup = Failures.run(null, () -> closeRequest(ctx, null));
        if (cleanup != null) recordRequestFailure(cleanup);
        return publication.publish(synchronous);
    }

    void closeOutstandingAndTerminalize() {
        Throwable failure = null;
        for (var ctx : repository.snapshotActive()) {
            if (ctx.data().getRequestScheduler() == this) {
                failure =
                        Failures.run(
                                failure,
                                () ->
                                        closeRequest(
                                                ctx,
                                                EndEvidence.failure(
                                                        ctx.snapshot().route(),
                                                        EndKind.SHUTDOWN,
                                                        StrategyErrorType.DISPATCH_FAILED,
                                                        "request scheduler is shutting down")));
            }
        }
        Failures.rethrow(failure, "request shutdown finalization failed");
    }

    void closeExecutors() {
        Throwable failure = Failures.run(null, closeQueue::awaitIdle);
        failure = Failures.run(failure, taskPool::awaitIdle);
        failure = Failures.run(failure, dispatcher::awaitStops);
        failure = Failures.run(failure, taskPool::close);
        failure = Failures.run(failure, responseCompletions::close);
        Failures.rethrow(failure, "request executors failed to close");
    }

    private void onDeliveryNotStarted(ContextResult.DeliveryNotStarted rejected) {
        var member = rejected.delivery();
        var ctx = member.owner();
        if (!isCurrentRequest(ctx)) return;
        synchronized (ctx) {
            var state = ctx.snapshot();
            if (rejected.selecting() == null
                    ? state.route() != member.route() || state.delivery() != null
                    : state.routingContext() != rejected.selecting()) return;
            if (rejected.deadline() != null) {
                var detail =
                        rejected.deadline().kind() == ExpirationTimer.Kind.INACTIVITY
                                ? "REQUEST_INACTIVE: no matching Engine request status before inactivity timeout"
                                : "request scheduling deadline exceeded before delivery";
                recordDeadline(ctx, rejected.deadline(), rejected.nowMs(), detail);
            }
        }
        if (rejected.selecting() == null && rejected.endpointUnavailable()) {
            onRouteFailure(
                    member.route(),
                    ContextResult.RouteFailureReason.ENDPOINT_UNAVAILABLE,
                    "Selected Endpoint unavailable during delivery commit");
        }
        if (ctx.snapshot().pendingEnd() != null) closeRequest(ctx, null);
        else if (System.currentTimeMillis() >= ctx.data().getRequestExpiresAtMs())
            cancel(ctx, 0L, CancelReason.DEADLINE_EXCEEDED);
    }

    /**
     * All timeout entry points record the same exact token and first cancellation cause.
     */
    private Snapshot recordDeadline(
            RequestContext ctx, ExpirationTimer.Token token, long nowMs, String detail) {
        var state = ctx.snapshot();
        boolean inactivity = token.kind() == ExpirationTimer.Kind.INACTIVITY;
        boolean expires =
                inactivity
                        ? nowMs >= state.inactivityExpiresAtMs()
                        : token.kind() == ExpirationTimer.Kind.REQUEST
                                && (state.delivery() == null || state.routingContext() != null);
        CancellationFacts diagnosis = null;
        if (token.owner() == ctx
                && state.deadlines().token(token.kind()) == token
                && nowMs >= token.atMs()
                && expires
                && state.phase() == Phase.OPEN
                && state.cancellation() == null) {
            if (detail == null) {
                detail =
                        inactivity
                                ? "REQUEST_INACTIVE: no matching Engine request status before inactivity timeout"
                                : state.routingContext() != null
                                        ? "request scheduling deadline exceeded during admission"
                                        : CancelReason.DEADLINE_EXCEEDED.getMessage();
            }
            var evidence = describeWait(ctx, CancelReason.DEADLINE_EXCEEDED, detail);
            diagnosis =
                    new CancellationFacts(
                            CancelReason.DEADLINE_EXCEEDED,
                            schedulingTimeoutError(),
                            evidence == null ? detail : String.valueOf(evidence.get("cause")));
            if (evidence != null) ctx.data().setSchedulingDiagnostics(evidence);
        }
        return ctx.recordDeadline(token, nowMs, diagnosis);
    }

    private void recordPredictedDelivery(
            Delivery.Member claim, long nowMs, java.util.function.Supplier<Snapshot> record) {
        var ctx = claim.owner();
        if (!isCurrentRequest(ctx)) return;
        var decode = claim.route().decodeEndpoint();
        var observed = ctx.snapshot();
        boolean confirmed =
                observed.delivery() != null
                        && observed.delivery().identity() == claim
                        && observed.route() == claim.route()
                        && !observed.engine().decodeAccepted()
                        && decode != null
                        && decode.ownership(claim.route().decodeReservation()).confirmed();
        recordRequestChange(
                ctx,
                () -> {
                    var before = ctx.snapshot();
                    if (confirmed
                            && before.delivery() != null
                            && before.delivery().identity() == claim
                            && before.route() == claim.route()
                            && !before.engine().decodeAccepted()) {
                        ctx.recordWorkerRequestEvent(
                                new WorkerRequestEvent(
                                        claim.route(),
                                        claim.route().decodeReservation(),
                                        RoleType.DECODE,
                                        WorkerRequestEventType.ACTIVE,
                                        0L,
                                        null,
                                        nowMs));
                    }
                    return record.get();
                });
    }

    private void recordRequestChange(
            RequestContext ctx, java.util.function.Supplier<Snapshot> record) {
        if (!isCurrentRequest(ctx)) return;
        Snapshot before;
        Snapshot after;
        synchronized (ctx) {
            before = ctx.snapshot();
            after = record.get();
        }
        if (after != null && after != before) onRequestStateChanged(ctx);
    }

    /**
     * Confirm delivery or start settlement; deadline synchronization follows the resulting state.
     */
    private void confirmDelivery(RequestContext ctx) {
        Registration publication = null;
        try {
            synchronized (ctx) {
                var state = ctx.snapshot();
                var delivery = state.delivery();
                if (!state.canConfirmDelivery() || delivery.acknowledged()) return;
                if (System.currentTimeMillis() >= state.inactivityExpiresAtMs()) {
                    String detail =
                            "REQUEST_INACTIVE: no matching Engine request status before inactivity"
                                    + " timeout";
                    if (state.deadlines().inactivity() != null) {
                        recordDeadline(
                                ctx,
                                state.deadlines().inactivity(),
                                System.currentTimeMillis(),
                                detail);
                    }
                } else {
                    if (state.preemption() == null || state.preemption().finished()) {
                        var candidate =
                                new ResponseCandidate(
                                        ctx, ResponseKind.ACK, System.currentTimeMillis());
                        publication = reserveResponse(ctx, candidate);
                    }
                    var acknowledged = ctx.confirmDelivery(System.currentTimeMillis());
                    if (publication != null && !acknowledged) {
                        publication.close();
                        return;
                    }
                    if (publication != null) {
                        ctx.data().discardGenerateInput();
                        if (delivery.identity().kind() == Delivery.SendMode.BATCH_ENQUEUE
                                && delivery.send() == SendState.DELIVERED) {
                            ctx.data().acknowledge(System.currentTimeMillis(), System.nanoTime());
                        }
                    }
                }
            }
            if (publication == null) {
                var state = ctx.snapshot();
                if (state.pendingEnd() != null) {
                    closeRequest(ctx, state.pendingEnd());
                } else {
                    recover(
                            this::recordRequestFailure,
                            () -> expirationTimer.update(ctx, ctx.snapshot().deadlines()));
                }
                return;
            }
            Failures.rethrow(
                    recover(
                            this::recordRequestFailure,
                            () -> expirationTimer.update(ctx, ctx.snapshot().deadlines())),
                    "delivery deadline installation failed");
            var claim = ctx.snapshot().delivery().identity();
            var response =
                    claim.route().successResponse(claim.kind() == Delivery.SendMode.BATCH_ENQUEUE);
            publication.prepare(RESPONSE, response, null);
        } catch (RuntimeException | Error failed) {
            if (publication != null) publication.failIfUnsubmitted(failed);
            throw failed;
        }
        publication.publish(false);
    }

    private Registration reserveResponse(RequestContext ctx, ResponseCandidate candidate) {
        return new Registration(
                responseCompletions,
                ctx,
                (RequestFuture) ctx.data().getFuture(),
                () -> {
                    if (candidate.kind() == ResponseKind.ACK && ctx.data().getAckAtMs() > 0L) {
                        var delivery = ctx.snapshot().delivery();
                        try {
                            deliveryReporter.reportLatency(
                                    DeliveryMetricsReporter.Latency.DISPATCH_ACK,
                                    RoleType.PREFILL.name(),
                                    delivery.identity().route().encoderEndpoint() != null
                                            ? delivery.identity().route().encoderEndpoint().getIp()
                                            : delivery.identity().route().prefillEndpoint().getIp(),
                                    Math.max(0L, ctx.data().getAckAtMs() - delivery.startedAtMs()));
                        } catch (Throwable ignored) {
                        }
                    }
                    return ctx.selectResponse(candidate);
                });
    }

    private static boolean needsFollowUpActions(Snapshot before, Snapshot after) {
        return after.preemption() != null
                || after.pendingEnd() != null
                || after.phase() != Phase.OPEN
                || after.canConfirmDelivery() && !after.delivery().acknowledged()
                || before.deadlines() != after.deadlines();
    }

    /**
     * Hand off one complete close operation; rejected work retains its unfinished cleanup state.
     */
    private void submitClose(RequestContext ctx, Registration publication) {
        var state = ctx.snapshot();
        boolean local = releaseBeforeResponse(state);
        boolean inactivity =
                state.outcome() != null && state.outcome().kind() == EndKind.INACTIVITY;
        Throwable queueFailure = !inactivity ? Failures.run(null, () -> removeRequest(ctx)) : null;
        if (publication != null && !local && !inactivity) {
            var ready = publication;
            queueFailure = Failures.run(queueFailure, () -> ready.publish(false));
            publication = null;
        }
        var responseOwner = publication;
        try {
            closeQueue.submit(
                    ctx,
                    () -> {
                        try {
                            runClose(ctx, responseOwner);
                        } catch (RuntimeException | Error failure) {
                            recordRequestFailure(failure);
                            throw failure;
                        }
                    },
                    local && !inactivity ? Runnable::run : task -> taskPool.execute(ctx, task),
                    responseOwner == null,
                    failure -> {
                        recordRequestFailure(failure);
                        if (responseOwner != null) responseOwner.failIfUnsubmitted(failure);
                    });
        } catch (RuntimeException | Error failure) {
            Failures.append(failure, queueFailure);
            throw failure;
        }
        if (queueFailure != null) recordRequestFailure(queueFailure);
        Failures.rethrow(queueFailure, "request queue removal failed");
    }

    private static boolean releaseBeforeResponse(Snapshot state) {
        return (state.outcome() == null || state.outcome().kind() != EndKind.RETIREMENT)
                && (state.delivery() == null
                        || state.delivery().identity().kind() != Delivery.SendMode.BATCH_ENQUEUE);
    }

    private void runClose(RequestContext ctx, Registration publication) {
        var state = ctx.snapshot();
        Throwable failure =
                state.outcome() != null && state.outcome().kind() == EndKind.INACTIVITY
                        ? Failures.run(null, () -> removeRequest(ctx))
                        : null;
        boolean releaseFirst = releaseBeforeResponse(state);
        if (releaseFirst) failure = Failures.run(failure, () -> cleanupRequest(ctx));
        if (publication != null) failure = Failures.run(failure, () -> publication.publish(false));
        if (!releaseFirst) failure = Failures.run(failure, () -> cleanupRequest(ctx));
        Failures.rethrow(failure, "request close failed");
    }

    private void cleanupRequest(RequestContext ctx) {
        Throwable failure = null;
        for (int pass = 0; pass < 3 && ctx.snapshot().phase() == Phase.ENDING; pass++) {
            Throwable problem =
                    Failures.run(
                            null, () -> expirationTimer.update(ctx, ctx.snapshot().deadlines()));
            problem = Failures.append(problem, requestDeliveryStop(ctx));
            var reason = ctx.releaseReason();
            if (reason != null) {
                var route = ctx.snapshot().route();
                if (route == null) {
                    ctx.recordResourcesReleased(null, true, true, System.currentTimeMillis());
                } else {
                    problem = Failures.append(problem, releaseRouteResources(route, reason));
                }
            }
            if (problem == null) {
                problem = completeRequestEndNotifications(ctx);
            }
            failure = Failures.append(failure, problem);
            if (problem == null) break;
        }
        var after = ctx.snapshot();
        var delivery = after.delivery();
        if (delivery != null
                && after.deliveryReleasable()
                && after.close().prefillSettled()
                && after.close().decodeSettled()) {
            failure = Failures.run(failure, () -> delivery.identity().complete());
        }
        if (after.terminalRecord() != null) {
            var terminal = after.terminalRecord();
            failure = Failures.run(failure, () -> repository.archive(ctx, terminal));
            failure = Failures.run(failure, () -> expirationTimer.remove(ctx));
        }
        Failures.rethrow(failure, "request closing failed");
    }

    private Throwable requestDeliveryStop(RequestContext ctx) {
        var state = ctx.snapshot();
        if (state.phase() != Phase.ENDING || state.close().notificationsAttempted()) return null;
        var delivery = state.delivery();
        if (delivery != null && (delivery.abandonment() != null || state.cancellation() != null)) {
            var failure = Failures.run(null, () -> dispatcher.stop(delivery.identity()));
            if (failure != null) return failure;
        }
        ctx.recordCloseNotifications(true, false, System.currentTimeMillis());
        return null;
    }

    private Throwable completeRequestEndNotifications(RequestContext ctx) {
        var state = ctx.snapshot();
        if (state.phase() != Phase.ENDING || state.close().notificationsDone()) return null;
        var preemption = state.preemption();
        Throwable failure = null;
        if (preemption != null && preemption.finished()) {
            failure =
                    Failures.run(
                            null,
                            () ->
                                    preemption
                                            .identity()
                                            .complete(
                                                    new PreemptionRequestResult(
                                                            ctx.data().getRequestId(),
                                                            PreemptionRequestResult.Outcome
                                                                    .REQUEST_END)));
        }
        ctx.recordCloseNotifications(
                true,
                failure == null && (preemption == null || preemption.finished()),
                System.currentTimeMillis());
        return failure;
    }

    private static Response response(
            ContextResult.End outcome, RouteResult route, boolean batched) {
        if (outcome.phase() == RequestState.Phase.COMPLETED && route != null) {
            return route.successResponse(batched);
        }
        return outcome.errorType() == StrategyErrorType.RESOURCE_EXHAUSTED
                ? Response.error(
                        outcome.errorType(),
                        AdmissionRejectReason.RESOURCE_EXHAUSTED,
                        outcome.detail())
                : Response.buildErrorResponse(outcome.errorType(), outcome.detail());
    }
}
