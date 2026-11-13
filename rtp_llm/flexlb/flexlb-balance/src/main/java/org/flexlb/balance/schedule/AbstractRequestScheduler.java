package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.requestlog.TheoryCacheHitReporter;
import org.flexlb.balance.schedule.RequestContext.SelectionAttempt;
import org.flexlb.balance.schedule.ResponseCompletionExecutor.Registration;
import org.flexlb.balance.schedule.ResponseCompletionExecutor.RequestFuture;
import org.flexlb.balance.strategy.RoleSelection;
import org.flexlb.balance.strategy.RoutingResult;
import org.flexlb.balance.trace.FlexlbTrace;
import org.flexlb.config.service.RoleType;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;
import org.flexlb.worker.endpoint.AdmissionRejectReason;
import org.flexlb.worker.endpoint.CapacityKey;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.Response;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.flexlb.worker.rpc.EngineGrpcClient;

import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

import static com.google.common.base.Preconditions.checkState;
import static org.flexlb.balance.schedule.RequestContext.EndKind;
import static org.flexlb.balance.schedule.RequestContext.Phase;
import static org.flexlb.balance.schedule.RequestContext.RequestEndEvent;
import static org.flexlb.balance.schedule.RequestContext.ResponseKind;
import static org.flexlb.balance.schedule.RequestContext.Snapshot;
import static org.flexlb.balance.schedule.ResponseCompletionExecutor.RequestFuture.Completion.RESPONSE;
import static org.flexlb.util.Failures.recover;

/**
 * Registers requests and coordinates cancellation, deadlines, responses and resource cleanup.
 * Runs shared route selection, publication and cleanup; subclasses supply waiting and rejection policies.
 * Cleanup drains accepted work before the shared endpoints and executors close.
 */
public abstract class AbstractRequestScheduler
        implements RequestScheduler, org.flexlb.worker.endpoint.WorkerEventSink {
    protected final RequestRepository repository;
    protected final RequestTaskPool taskPool;
    private final ExpirationTimer expirationTimer;
    private final EngineCancellation engineCancellation;
    private final ResponseCompletionExecutor responseCompletions;
    private final DeliveryMetricsReporter deliveryReporter;
    private final TheoryCacheHitReporter cacheTrace;
    private final EndpointRegistry endpoints;
    private final Object shutdownLock = new Object();
    private final RequestCloseQueue closeQueue = new RequestCloseQueue();

    private Throwable requestProcessingFailure;

    protected AbstractRequestScheduler(
            RequestRepository repository,
            RequestTaskPool taskPool,
            ExpirationTimer deadlines,
            EngineCancellation engineCancellation,
            ResponseCompletionExecutor responseCompletions,
            DeliveryMetricsReporter deliveryReporter,
            TheoryCacheHitReporter cacheTrace,
            EndpointRegistry endpoints) {
        this.repository = Objects.requireNonNull(repository);
        this.taskPool = Objects.requireNonNull(taskPool);
        this.expirationTimer = Objects.requireNonNull(deadlines);

        this.engineCancellation = Objects.requireNonNull(engineCancellation);
        this.responseCompletions = Objects.requireNonNull(responseCompletions);
        this.deliveryReporter = Objects.requireNonNull(deliveryReporter);
        this.cacheTrace = Objects.requireNonNull(cacheTrace);
        this.endpoints = Objects.requireNonNull(endpoints);
    }

    @Override
    public void onWorkerRequestEvent(
            org.flexlb.worker.endpoint.WorkerRequest request,
            org.flexlb.worker.endpoint.WorkerRequestEvent event) {
        onRequestEvent(
                (RequestContext) request,
                new RequestContext.RequestEvent(
                        (RequestResult) event.route(),
                        event.reservation(),
                        event.role(),
                        event.endpoint(),
                        event.type(),
                        event.errorCode(),
                        event.detail(),
                        event.nowMs()));
    }

    /**
     * Advance on the status caller; user callbacks use their response executor.
     */
    public void onRequestEvent(RequestContext ctx, RequestContext.RequestEvent event) {
        if (ctx == null || ctx.getRequestScheduler() != this) return;
        try {
            if (ctx.recordEvent(event)) {
                try {
                    if (ctx.snapshot().phase() == Phase.CLOSED) {
                        engineCancellation.forget(event.route());
                    }
                } finally {
                    processPendingRequestActions(ctx);
                }
            }
        } catch (Throwable failure) {
            recordRequestFailure(failure);
            Logger.error("Request event processing failed: type={}", event.type(), failure);
        }
    }

    public synchronized void recordRequestFailure(Throwable failure) {
        requestProcessingFailure = Failures.append(requestProcessingFailure, failure);
    }

    public CompletableFuture<Response> submit(RequestContext context) {
        return submit(context, () -> {});
    }

    public CompletableFuture<Response> submit(RequestContext ctx, Runnable onRegistered) {
        Objects.requireNonNull(onRegistered, "onRegistered");
        try {
            if (ctx == null || ctx.getRequest() == null) {
                return CompletableFuture.completedFuture(
                        Response.error(StrategyErrorType.INVALID_REQUEST));
            }
            var data = ctx;
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
            FlexlbTrace.setScheduleAttribute(
                    data.getTraceContext(),
                    FlexlbTrace.SCHEDULE_MODE,
                    config.getScheduler().getType().name());
            future.whenComplete(
                    (response, failure) -> {
                        try {
                            if (failure == null && response != null && response.isSuccess()) {
                                cacheTrace.report(ctx);
                            }
                        } catch (RuntimeException error) {
                            Logger.warn("Route tracing failed", error);
                        } finally {
                            ctx.releaseSchedulingInput();
                        }
                    });
            expirationTimer.update(ctx, ctx.snapshot().deadlines());
            onRegistered.run();
            if (repository.isClosed()) {
                cancel(ctx, CancelReason.SHUTDOWN);
            } else if (System.currentTimeMillis() >= expires) {
                cancel(ctx, CancelReason.DEADLINE_EXCEEDED);
            }
            if (!future.isDone()) scheduleRequest(ctx);
            else closeRequest(ctx, null);
            return future;
        } catch (RuntimeException failure) {
            Response response =
                    Response.buildErrorResponse(
                            StrategyErrorType.DISPATCH_FAILED, failure.getMessage());
            if (ctx != null && ctx.getRequestScheduler() == this) {
                publish(ctx, response, false);
                return ctx.getFuture();
            }
            return CompletableFuture.completedFuture(response);
        }
    }

    public RequestResult cancel(String id, CancelReason reason) {
        return cancel(id, reason, org.flexlb.balance.schedule.RequestPhase.GENERATION);
    }

    public RequestResult cancel(
            String id, CancelReason reason, org.flexlb.balance.schedule.RequestPhase phase) {
        Objects.requireNonNull(reason);
        var ctx = repository.findActive(id, phase);
        if (ctx != null) {
            if (ctx.getRequestScheduler() != this) {
                return null;
            }
            var result = cancel(ctx, reason);
            if (result != null) {
                return result;
            }
        }
        var terminal = repository.findTerminal(id, phase);
        return terminal != null && terminal.owner() == this ? terminal : null;
    }

    /**
     * Records the first cancellation cause and submits cleanup without waiting for the engine reply.
     */
    public RequestResult cancel(RequestContext ctx, CancelReason reason) {
        if (ctx == null || ctx.getRequestScheduler() != this || !repository.isCurrent(ctx)) {
            return null;
        }

        boolean changed;
        Registration publication = null;
        Throwable failure = null;
        Throwable responsePreparationFailure = null;
        synchronized (ctx) {
            var state = ctx.snapshot();
            if (state.phase() == Phase.CLOSED) {
                return state.requestRecordForArchive();
            }
            String message = reason.getMessage();
            Map<String, Object> evidence = describeWait(ctx, reason, message);
            var error =
                    reason == CancelReason.DEADLINE_EXCEEDED
                            ? schedulingTimeoutError()
                            : reason == CancelReason.PRIORITY_PREEMPTED
                                    ? StrategyErrorType.PRIORITY_PREEMPTED
                                    : StrategyErrorType.REQUEST_CANCELLED;
            RequestResult candidate = null;
            if (state.selectionId() == 0L
                    && state.responseForFuture() == null
                    && !ctx.getFuture().isDone()) {
                candidate =
                        new RequestResult(ctx, ResponseKind.TERMINAL, System.currentTimeMillis());
                try {
                    publication = reserveResponse(ctx, candidate);
                } catch (RuntimeException | Error rejected) {
                    failure = rejected;
                    candidate = null;
                }
            }
            changed =
                    ctx.requestCancel(
                            reason,
                            false,
                            System.currentTimeMillis(),
                            error,
                            evidence == null ? message : String.valueOf(evidence.get("cause")),
                            candidate);
            if (changed && evidence != null) {
                ctx.setSchedulingDiagnostics(evidence);
            }
            if (publication != null) {
                var after = ctx.snapshot();
                if (after.responseForFuture() == candidate) {
                    try {
                        publication.prepare(
                                RESPONSE, response(after.requestEndResult(), after.route()), null);
                    } catch (RuntimeException | Error rejected) {
                        responsePreparationFailure = rejected;
                        failure = Failures.append(failure, rejected);
                    }
                } else {
                    publication.close();
                    publication = null;
                }
            }
        }
        if (responsePreparationFailure != null) {
            var rejectedPublication = publication;
            var rejected = responsePreparationFailure;
            failure = Failures.run(failure, () -> rejectedPublication.failIfUnsubmitted(rejected));
            publication = null;
        }
        if (changed && ctx.snapshot().requestEndResult() != null) {
            var responsePublication = publication;
            failure = Failures.run(failure, () -> scheduleRequestCleanup(ctx, responsePublication));
        }
        var deadlines = ctx.snapshot().deadlines();
        if (ctx.snapshot().phase() == Phase.OPEN) {
            failure = Failures.run(failure, () -> expirationTimer.update(ctx, deadlines));
        }
        if (failure != null) {
            recordRequestFailure(failure);
        }
        Failures.rethrow(failure, "cancellation effects failed");
        var state = RequestResult.from(ctx.snapshot());
        return state;
    }

    /**
     * Local submission failure uses the same arbitration as external Future completion.
     */
    public boolean publish(RequestContext ctx, Response response, boolean synchronous) {
        return tryCompleteFromExternal(
                ctx, RequestFuture.Completion.RESPONSE, response, null, synchronous);
    }

    public boolean isAdmissionOpen(RequestContext ctx) {
        if (repository.isClosed() || ctx == null || ctx.getRequestScheduler() != this) {
            return false;
        }
        synchronized (ctx) {
            var state = ctx.snapshot();
            return repository.isCurrent(ctx)
                    && state.phase() == Phase.OPEN
                    && state.cancellationReason() == null;
        }
    }

    /**
     * Checks that this selection may still commit its acquired route.
     */
    public SelectionAttempt currentSelectionAttempt(RequestResult route) {
        var ctx = route.ctx();
        synchronized (ctx) {
            var state = ctx.snapshot();
            if (repository.isClosed()
                    || !repository.isCurrent(ctx)
                    || ctx.getRequestScheduler() != this
                    || state.selectionId() == 0L) return null;
            var routingContext = ctx.currentSelectionAttempt();
            return ctx.canPublishRoute(routingContext) ? routingContext : null;
        }
    }

    /**
     * Ends selection without publishing a route; the caller has already attempted resource rollback.
     * A null failure leaves retry decisions to the scheduling policy.
     */
    public void finishSelectionWithoutCommit(SelectionAttempt routingContext, Response failure) {
        com.google.common.base.Preconditions.checkArgument(
                failure == null || !failure.isSuccess(),
                "selection failure requires an error response");
        var ctx = routingContext.requestContext();
        checkState(ctx.getRequestScheduler() == this, "foreign selection");
        var after =
                ctx.finishSelectionWithoutCommit(
                        routingContext, failure, System.currentTimeMillis());
        if (after != null) processPendingRequestActions(ctx);
    }

    public void onDeadline(ExpirationTimer.Token exact, long nowMs) {
        var ctx = exact.owner();
        if (ctx.getRequestScheduler() != this || !repository.isCurrent(ctx)) {
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
                owners.add(context.getRequestScheduler());
                failure =
                        Failures.run(
                                failure,
                                () ->
                                        context.getRequestScheduler()
                                                .cancel(context, CancelReason.SHUTDOWN));
            }
            failure = Failures.run(failure, () -> engineCancellation.awaitStops());
            failure = Failures.run(failure, taskPool::awaitIdle);
            for (var owner : owners) failure = Failures.run(failure, owner.closeQueue::awaitIdle);
            failure = Failures.run(failure, endpoints::close);
            failure = Failures.run(failure, () -> engineCancellation.awaitStops());
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
                        && !engineCancellation.isStoppingOnCurrentThread()
                        && !closeQueue.isExecutingOnCurrentThread()
                        && !responseCompletions.isCompletingOnCurrentThread(),
                "scheduler shutdown cannot wait for its own request work or response callback");
    }

    /**
     * Starts one selection attempt; publication and rollback share the same completion path.
     */
    protected final RequestResult planRoute(
            RequestContext context,
            java.util.function.Supplier<RequestResult> selection,
            boolean allowDecodePreemption) {
        var operation = tryBeginRouteSelection(context);
        return operation == null
                ? RequestResult.closed()
                : executeRouteSelection(operation, selection, allowDecodePreemption);
    }

    protected SelectionAttempt tryBeginRouteSelection(RequestContext ctx) {
        if (repository.isClosed() || ctx == null || ctx.getRequestScheduler() != this) {
            return null;
        }
        synchronized (ctx) {
            if (!repository.isCurrent(ctx)) {
                return null;
            }
            return ctx.beginRouteSelection();
        }
    }

    /**
     * Executes or resumes selection. A started preemption keeps the attempt until its continuation resumes.
     */
    protected final RequestResult executeRouteSelection(
            SelectionAttempt routingContext,
            java.util.function.Supplier<RequestResult> selection,
            boolean allowDecodePreemption) {
        var request = routingContext.requestContext();
        checkState(
                java.util.Objects.equals(request.currentSelectionAttempt(), routingContext),
                "routing requires its exact active admission");
        RequestResult route = null;
        RequestResult result = null;
        Throwable failure = null;
        boolean routeAccepted = false;
        Runnable continuation = null;
        try {
            result = Objects.requireNonNull(selection.get(), "routing result");
            checkState(
                    result.status() != RoutingResult.Status.SUCCESS || result.ctx() == request,
                    "route belongs to another request");
            route = result.status() == RoutingResult.Status.SUCCESS ? result : null;
            if (route != null)
                routingContext.requestContext().registerSelectedRoute(routingContext, route);
            request.recordRoutingDiagnostics(routingContext, result.diagnostics());
            if (request.getFuture().isDone()
                    || !request.canPublishRoute(routingContext)
                    || routeSelectionStopped()) {
                result = RequestResult.closed();
            } else if (result.status() == RoutingResult.Status.BLOCKED && allowDecodePreemption) {
                continuation = startOrWaitForDecodePreemption(routingContext, result.blocker());
            }
            if (result.status() == RoutingResult.Status.SUCCESS) {
                result = route.validateEndpoints(result);
            }
            if (result.status() == RoutingResult.Status.SUCCESS) {
                var publication = publishSelectedRoute(route);
                result = publication;
                routeAccepted = result.status() == RoutingResult.Status.SUCCESS;
            }
        } catch (RuntimeException | Error problem) {
            failure = problem;
            throw problem;
        } finally {
            if (continuation == null && !routeAccepted) {
                Throwable cleanup = finishUnpublishedRoute(route, routingContext);
                if (cleanup != null) recordRequestFailure(cleanup);
                if (failure != null) Failures.append(failure, cleanup);
                else Failures.rethrow(cleanup, "placement cleanup failed");
            }
        }
        if (continuation != null) return result.waitingForPreemption(continuation);
        return route == null ? result : selectedRouteResult(result);
    }

    protected boolean routeSelectionStopped() {
        return false;
    }

    protected Runnable startOrWaitForDecodePreemption(
            SelectionAttempt operation, CapacityKey blocker) {
        return null;
    }

    protected RequestResult publishSelectedRoute(RequestResult route) {
        route.ctx().recordRouteCommitStart(System.nanoTime());
        if (route.encoderEndpoint() != null) return commitEncoderRoute(route);
        var selecting = currentSelectionAttempt(route);
        if (selecting == null) return route.validateEndpoints(RequestResult.closed());
        if (!route.prefillEndpoint().commitSelection(route))
            return route.validateEndpoints(
                    RequestResult.rejected(
                            Response.buildErrorResponse(
                                    StrategyErrorType.DISPATCH_FAILED,
                                    "Selected Prefill registration was not valid for commit")));
        if (!commitRoute(route, selecting)) return route.validateEndpoints(RequestResult.closed());
        return route;
    }

    protected RequestResult selectedRouteResult(RequestResult result) {
        return result;
    }

    protected abstract StrategyErrorType schedulingTimeoutError();

    protected final boolean checkRequestMode(RequestContext context, boolean queued) {
        if (context.getConfig().isQueue() == queued) return true;
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

    /**
     * Select one effect owner after recording facts; that owner also handles its failures.
     */
    protected void processPendingRequestActions(RequestContext ctx) {
        var state = ctx.snapshot();
        Throwable failure = null;
        try {
            if (state.phase() != Phase.OPEN) {
                closeRequest(ctx, null);
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

    void onRequestStateChanged(RequestContext ctx) {
        if (!isCurrentRequest(ctx)) return;
        try {
            processPendingRequestActions(ctx);
        } catch (RuntimeException | Error failure) {
            recordRequestFailure(failure);
        }
    }

    boolean isCurrentRequest(RequestContext ctx) {
        return ctx.getRequestScheduler() == this && repository.isCurrent(ctx);
    }

    synchronized void throwIfRequestFailed() {
        Failures.rethrow(requestProcessingFailure, "request processing failed");
    }

    /**
     * Roll back unpublished resources and finish selection, returning any cleanup or completion failure.
     */
    static Throwable finishUnpublishedRoute(RequestResult route, SelectionAttempt routingContext) {
        Throwable failure = null;
        try {
            if (route == null && routingContext != null) {
                var context = routingContext.requestContext();
                synchronized (context) {
                    if (Objects.equals(context.currentSelectionAttempt(), routingContext)) {
                        var state = context.snapshot();
                        if (state.committedRoute() == null) route = state.route();
                    }
                }
            }
            failure =
                    route == null
                            ? null
                            : releaseRouteResources(
                                    route, WorkerEndpoint.ReleaseReason.LOCAL_ROLLBACK, false);
            boolean rollbackFailed = failure != null;
            if (routingContext != null) {
                if (rollbackFailed)
                    routingContext
                            .requestContext()
                            .getRequestScheduler()
                            .finishSelectionWithoutCommit(
                                    routingContext,
                                    Response.buildErrorResponse(
                                            StrategyErrorType.DISPATCH_FAILED,
                                            "placement rollback failed"));
                else
                    routingContext
                            .requestContext()
                            .getRequestScheduler()
                            .finishSelectionWithoutCommit(routingContext, null);
            }
            return failure;
        } catch (RuntimeException | Error completion) {
            return Failures.append(completion, failure);
        }
    }

    /**
     * One resource-release implementation for admission rollback, route withdrawal and request end.
     * Rollback checks every acquired resource; request end retains successful release progress.
     */
    static Throwable releaseRouteResources(
            RequestResult route,
            WorkerEndpoint.ReleaseReason reason,
            boolean notificationsSubmitted) {
        var ctx = route.ctx();
        var state = ctx.snapshot();
        boolean rollback = reason == WorkerEndpoint.ReleaseReason.LOCAL_ROLLBACK;
        boolean mayReleaseDecode =
                reason != WorkerEndpoint.ReleaseReason.COUNTERPART_FINISHED
                        || state.committedRoute() == null;
        boolean prefillAndEncoderReleased =
                !rollback && state.route() == route && state.prefillCleanupCompleted();
        boolean decodeReleased =
                !rollback && state.route() == route && state.decodeCleanupCompleted();
        Throwable failure = null;
        if (!prefillAndEncoderReleased) {
            if (route.encoderEndpoint() != null)
                failure =
                        Failures.run(failure, () -> route.encoderEndpoint().releaseRequest(route));
            if (route.prefillEndpoint() != null)
                failure =
                        Failures.run(failure, () -> route.prefillEndpoint().releaseRequest(route));
            prefillAndEncoderReleased = failure == null;
        }
        try {
            if (!decodeReleased && mayReleaseDecode) {
                decodeReleased =
                        RoleSelection.releaseDecode(
                                route.decodeEndpoint(), route.decodeRegistration(), reason);
            }
        } catch (Throwable problem) {
            failure = Failures.append(failure, problem);
        }
        ctx.recordCleanupProgress(
                route,
                prefillAndEncoderReleased,
                decodeReleased,
                notificationsSubmitted && failure == null,
                System.currentTimeMillis());
        return failure;
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
                    || before.responseForFuture() != null
                    || ctx.getFuture().isDone()
                    || before.cancellationReason() != null
                            && (!cancelled
                                    || before.cancellationReason()
                                            == CancelReason.DEADLINE_EXCEEDED)) return false;
            boolean success = completion == RESPONSE && (response == null || response.isSuccess());
            if (success
                    || before.committedRoute() != null
                    || before.decodeAccepted()
                    || before.selectionId() != 0L && !cancelled) return false;
            var candidate =
                    new RequestResult(
                            ctx,
                            cancelled ? ResponseKind.CANCELLATION : ResponseKind.TERMINAL,
                            System.currentTimeMillis());
            publication = reserveResponse(ctx, candidate);
            try {
                if (cancelled
                        && before.cancellationReason() == null
                        && !ctx.requestCancel(
                                CancelReason.CLIENT_CANCELLED,
                                true,
                                System.currentTimeMillis(),
                                StrategyErrorType.REQUEST_CANCELLED,
                                CancelReason.CLIENT_CANCELLED.getMessage(),
                                candidate)) {
                    publication.close();
                    return false;
                }
                if (!cancelled
                        && ctx.endRequest(
                                                RequestEndEvent.externalFailure(
                                                        before.route(), response, failure),
                                                System.currentTimeMillis(),
                                                candidate)
                                        .requestEndResult()
                                == null) {
                    publication.close();
                    return false;
                }
                if (cancelled) ctx.endRequest(null, System.currentTimeMillis(), candidate);
                if (ctx.snapshot().responseForFuture() != candidate) {
                    publication.close();
                    return false;
                }
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

    /**
     * Own closing through completion or a recorded wait, including recovery after an effect fails.
     */
    void closeRequest(RequestContext ctx, RequestEndEvent evidence) {
        if (Thread.holdsLock(ctx)) {
            throw new IllegalStateException(
                    "request closing requires the Context lock to be released");
        }
        if (ctx.getRequestScheduler() != this || !repository.isCurrent(ctx)) {
            return;
        }
        Registration publication = null;
        try {
            synchronized (ctx) {
                RequestResult candidate = null;
                var before = ctx.snapshot();
                if (evidence == null && before.phase() == Phase.OPEN
                        || evidence != null
                                && evidence.route() != null
                                && before.route() != evidence.route()) {
                    return;
                }
                if (before.selectionId() == 0L
                        && before.responseForFuture() == null
                        && !ctx.getFuture().isDone()) {
                    candidate =
                            new RequestResult(
                                    ctx, ResponseKind.TERMINAL, System.currentTimeMillis());
                    publication = reserveResponse(ctx, candidate);
                }
                var after = ctx.endRequest(evidence, System.currentTimeMillis(), candidate);
                if (after.requestEndResult() == null) {
                    if (publication != null) publication.close();
                    return;
                }
                if (publication != null) {
                    if (after.responseForFuture() != candidate) {
                        publication.close();
                        publication = null;
                    } else {
                        var requestEndResult = after.requestEndResult();
                        publication.prepare(
                                RESPONSE, response(requestEndResult, before.route()), null);
                    }
                }
            }
        } catch (RuntimeException | Error failed) {
            ctx.endRequest(evidence, System.currentTimeMillis(), null);
            if (publication != null) publication.failIfUnsubmitted(failed);
            Failures.run(failed, () -> scheduleRequestCleanup(ctx, null));
            throw failed;
        }
        scheduleRequestCleanup(ctx, publication);
    }

    void closeOutstandingAndTerminalize() {
        Throwable failure = null;
        for (var ctx :
                repository.snapshotActive(request -> request.getRequestScheduler() == this)) {
            failure =
                    Failures.run(
                            failure,
                            () ->
                                    closeRequest(
                                            ctx,
                                            RequestEndEvent.failure(
                                                    ctx.snapshot().route(),
                                                    EndKind.SHUTDOWN,
                                                    StrategyErrorType.DISPATCH_FAILED,
                                                    "request scheduler is shutting down")));
        }
        Failures.rethrow(failure, "request shutdown finalization failed");
    }

    void closeExecutors() {
        Throwable failure = Failures.run(null, closeQueue::awaitIdle);
        failure = Failures.run(failure, taskPool::awaitIdle);
        failure = Failures.run(failure, engineCancellation::awaitStops);
        failure = Failures.run(failure, taskPool::close);
        failure = Failures.run(failure, responseCompletions::close);
        Failures.rethrow(failure, "request executors failed to close");
    }

    /**
     * Applies a cancellation reply only to the original active request and committed route.
     * A request may already be archived when an outstanding RPC replies.
     */
    void onEngineCancelReply(RequestResult route, EngineGrpcClient.CancelReply reply) {
        var ctx = route.ctx();
        if (!isCurrentRequest(ctx)) return;
        var type =
                switch (reply) {
                    case REQUEST_CLEANED ->
                            org.flexlb.worker.endpoint.RequestEventType.REMOTE_CLEANED;
                    case REQUEST_FENCED ->
                            org.flexlb.worker.endpoint.RequestEventType.REMOTE_FENCED;
                    default -> null;
                };
        if (type == null) return;
        onRequestEvent(
                ctx,
                new RequestContext.RequestEvent(
                        route, null, null, null, type, 0L, null, System.currentTimeMillis()));
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
                                && (state.committedRoute() == null || state.selectionId() != 0L);
        String diagnosis = null;
        if (token.owner() == ctx
                && state.deadlines().token(token.kind()) == token
                && nowMs >= token.atMs()
                && expires
                && state.phase() == Phase.OPEN
                && state.cancellationReason() == null) {
            if (detail == null) {
                detail =
                        inactivity
                                ? "REQUEST_INACTIVE: no matching Engine request status before inactivity timeout"
                                : state.selectionId() != 0L
                                        ? "request scheduling deadline exceeded during admission"
                                        : CancelReason.DEADLINE_EXCEEDED.getMessage();
            }
            var evidence = describeWait(ctx, CancelReason.DEADLINE_EXCEEDED, detail);
            diagnosis = evidence == null ? detail : String.valueOf(evidence.get("cause"));
            if (evidence != null) ctx.setSchedulingDiagnostics(evidence);
        }
        return ctx.recordDeadline(token, nowMs, diagnosis);
    }

    private Map<String, Object> describeWait(
            RequestContext request, CancelReason reason, String message) {
        if (reason != CancelReason.DEADLINE_EXCEEDED) return null;
        var state = request.snapshot();
        String stage =
                state.committedRoute() != null
                        ? "DELIVERY"
                        : state.selectionId() != 0L
                                ? "PLACEMENT"
                                : state.route() == null
                                        ? "WAITING_FOR_PLACEMENT"
                                        : "BEFORE_DELIVERY";
        return completeWaitDiagnostics(message, describeWaitAtStage(request, state, stage));
    }

    private final Map<String, Object> completeWaitDiagnostics(
            String message, Map<String, Object> details) {
        details.put("cause", message + "; " + details);
        return java.util.Collections.unmodifiableMap(details);
    }

    private final Map<String, Object> describeWaitAtStage(
            RequestContext request, Snapshot state, String stage) {
        Map<String, Object> details = new java.util.LinkedHashMap<>();
        details.put("stage", stage);
        details.put("capturedAtMs", System.currentTimeMillis());
        if (state.selectionId() != 0L) {
            details.put("routingPhase", state.routingPhase().name());
        }
        if (state.committedRoute() == null && state.route() == null) {
            var observation = request.getSchedulingDiagnostics();
            if (observation != null) {
                details.put("lastPlacementObservation", observation);
            }
        }
        return details;
    }

    /**
     * Hand off one complete close operation; rejected work retains its unfinished cleanup state.
     */
    private void scheduleRequestCleanup(RequestContext ctx, Registration publication) {
        var state = ctx.snapshot();
        boolean responseBeforeCleanup =
                state.requestEndResult() != null
                        && state.requestEndResult().endKind() == EndKind.ENDPOINT_CLOSED;
        boolean inactivity =
                state.requestEndResult() != null
                        && state.requestEndResult().endKind() == EndKind.INACTIVITY;
        Throwable queueFailure = !inactivity ? Failures.run(null, () -> removeRequest(ctx)) : null;
        if (publication != null && responseBeforeCleanup && !inactivity) {
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
                            Throwable failure =
                                    inactivity
                                            ? Failures.run(null, () -> removeRequest(ctx))
                                            : null;
                            failure = Failures.run(failure, () -> cleanupRequest(ctx));
                            if (responseOwner != null) {
                                failure = Failures.run(failure, () -> responseOwner.publish(false));
                            }
                            Failures.rethrow(failure, "request close failed");
                        } catch (RuntimeException | Error failure) {
                            recordRequestFailure(failure);
                            throw failure;
                        }
                    },
                    !responseBeforeCleanup && !inactivity
                            ? Runnable::run
                            : task -> taskPool.execute(task),
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

    private void cleanupRequest(RequestContext ctx) {
        var delivery = ctx.snapshot().committedRoute();
        Throwable failure = null;
        for (int pass = 0; pass < 3 && ctx.snapshot().phase() == Phase.ENDING; pass++) {
            Throwable problem =
                    Failures.run(
                            null, () -> expirationTimer.update(ctx, ctx.snapshot().deadlines()));
            problem = Failures.append(problem, submitEngineCancellation(ctx));
            var reason = ctx.releaseReason();
            var route = ctx.snapshot().route();
            if (reason != null) {
                if (route == null) {
                    ctx.recordCleanupProgress(
                            null, true, true, problem == null, System.currentTimeMillis());
                } else {
                    problem =
                            Failures.append(
                                    problem, releaseRouteResources(route, reason, problem == null));
                }
            } else if (problem == null) {
                ctx.recordCleanupProgress(route, false, false, true, System.currentTimeMillis());
            }
            failure = Failures.append(failure, problem);
            if (problem == null) break;
        }
        var after = ctx.snapshot();
        if (delivery != null && after.prefillCleanupCompleted() && after.decodeCleanupCompleted()) {
            failure = Failures.run(failure, () -> engineCancellation.forget(delivery));
        }
        if (after.requestRecordForArchive() != null) {
            var terminal = after.requestRecordForArchive();
            failure = Failures.run(failure, () -> repository.archive(ctx, terminal));
            failure = Failures.run(failure, () -> expirationTimer.remove(ctx));
        }
        Failures.rethrow(failure, "request closing failed");
    }

    private Throwable submitEngineCancellation(RequestContext ctx) {
        var state = ctx.snapshot();
        if (state.phase() != Phase.ENDING || state.notificationsDone()) return null;
        var delivery = state.committedRoute();
        if (delivery != null && state.cancellationReason() != null) {
            var failure = Failures.run(null, () -> engineCancellation.stop(delivery));
            if (failure != null) return failure;
        }
        return null;
    }

    private static Response response(RequestResult requestEndResult, RequestResult route) {
        if (requestEndResult.state() == RequestResult.State.COMPLETED && route != null) {
            return route.successResponse();
        }
        return requestEndResult.errorType() == StrategyErrorType.RESOURCE_EXHAUSTED
                ? Response.error(
                        requestEndResult.errorType(),
                        AdmissionRejectReason.RESOURCE_EXHAUSTED,
                        requestEndResult.detail())
                : Response.buildErrorResponse(
                        requestEndResult.errorType(), requestEndResult.detail());
    }

    private RequestResult commitEncoderRoute(RequestResult route) {
        var context = route.ctx();

        var routingContext = route.ctx().getRequestScheduler().currentSelectionAttempt(route);
        if (routingContext == null) return RequestResult.closed();
        if (!route.encoderEndpoint().commitSelection(route))
            return route.validateEndpoints(
                    RequestResult.blocked(
                            new CapacityKey(
                                    RoleType.ENCODER,
                                    route.encoderEndpoint().getStatus().getGroup(),
                                    route.encoderEndpoint().ipPort())));
        if (!route.selectedEndpointsAvailable())
            return route.validateEndpoints(RequestResult.closed());
        var claim = commitRoute(route, routingContext);
        if (!claim) return RequestResult.closed();
        return route;
    }

    /**
     * Reserves response execution, commits once, then invokes the response outside the request lock.
     */
    private boolean commitRoute(RequestResult route, SelectionAttempt attempt) {
        var ctx = route.ctx();
        Registration publication = null;
        boolean committed = false;
        try {
            synchronized (ctx) {
                if (!repository.isCurrent(ctx)
                        || ctx.getFuture().isDone()
                        || !route.selectedEndpointsAvailable()
                        || route.decodeEndpoint() != null
                                && !route.decodeEndpoint()
                                        .registrationState(route.decodeRegistration())
                                        .registered()) return false;
                long nowMs = System.currentTimeMillis();
                var candidate = new RequestResult(ctx, ResponseKind.ACK, nowMs);
                publication = reserveResponse(ctx, candidate);
                committed = ctx.tryCommitRoute(attempt, route, candidate, nowMs);
                if (!committed) return false;
                publication.prepare(RESPONSE, route.successResponse(), null);
            }
            publication.publish(false);
            processPendingRequestActions(ctx);
            return true;
        } catch (RuntimeException | Error failure) {
            if (committed) {
                if (publication != null) publication.failIfUnsubmitted(failure);
                recordRequestFailure(failure);
                // The committed route is owned by normal request cleanup, never admission rollback.
                var cleanupFailure =
                        Failures.run(
                                null,
                                () ->
                                        closeRequest(
                                                ctx,
                                                RequestEndEvent.failure(
                                                        route,
                                                        EndKind.FAILURE,
                                                        StrategyErrorType.DISPATCH_FAILED,
                                                        "route response failed")));
                if (cleanupFailure != null) recordRequestFailure(cleanupFailure);
                return true;
            }
            throw failure;
        } finally {
            if (publication != null) publication.close();
        }
    }

    private Registration reserveResponse(RequestContext ctx, RequestResult candidate) {
        return new Registration(
                responseCompletions,
                ctx,
                (RequestFuture) ctx.getFuture(),
                () -> {
                    return ctx.snapshot().responseForFuture() == candidate;
                });
    }
}
