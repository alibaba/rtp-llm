package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.request.Response;
import org.flexlb.balance.requestlog.TheoryCacheHitReporter;
import org.flexlb.balance.schedule.RequestContext.SelectionAttempt;
import org.flexlb.balance.schedule.ResponseCompletionExecutor.Registration;
import org.flexlb.balance.schedule.ResponseCompletionExecutor.RequestFuture;
import org.flexlb.balance.strategy.SelectionResult;
import org.flexlb.balance.trace.FlexlbTrace;
import org.flexlb.config.service.RoleType;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;
import org.flexlb.worker.endpoint.AdmissionRejectReason;
import org.flexlb.worker.endpoint.CapacityKey;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.flexlb.worker.rpc.EngineGrpcClient;

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
 * Stops and cleans up its own requests; the application closes shared resources.
 */
public abstract class AbstractRequestScheduler
        implements RequestScheduler, org.flexlb.worker.endpoint.WorkerEventSink {
    protected final RequestRepository repository;
    protected final RequestTaskPool taskPool;
    private final ExpirationTimer expirationTimer;
    private final EngineCancellation engineCancellation;
    private final ResponseCompletionExecutor responseCompletions;
    private final TheoryCacheHitReporter cacheTrace;
    private final RequestCloseQueue closeQueue = new RequestCloseQueue();

    private Throwable requestProcessingFailure;

    protected AbstractRequestScheduler(
            RequestRepository repository,
            RequestTaskPool taskPool,
            ExpirationTimer deadlines,
            EngineCancellation engineCancellation,
            ResponseCompletionExecutor responseCompletions,
            DeliveryMetricsReporter deliveryReporter,
            TheoryCacheHitReporter cacheTrace) {
        this.repository = Objects.requireNonNull(repository);
        this.taskPool = Objects.requireNonNull(taskPool);
        this.expirationTimer = Objects.requireNonNull(deadlines);

        this.engineCancellation = Objects.requireNonNull(engineCancellation);
        this.responseCompletions = Objects.requireNonNull(responseCompletions);
        this.cacheTrace = Objects.requireNonNull(cacheTrace);
    }

    @Override
    public void onWorkerRequestEvent(
            org.flexlb.worker.endpoint.WorkerRequest request,
            org.flexlb.worker.endpoint.WorkerRequestEvent event) {
        onRequestEvent(
                (RequestContext) request,
                new RequestContext.RequestEvent(
                        (RequestRoute) event.route(),
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

    public CompletableFuture<Response> submit(RequestContext request, Runnable onRegistered) {
        Objects.requireNonNull(onRegistered, "onRegistered");
        try {
            Response rejection = admissionRejection(request);
            if (rejection != null) return CompletableFuture.completedFuture(rejection);
            if (request.getRequestExpiresAtMs() <= 0
                    || System.currentTimeMillis() >= request.getRequestExpiresAtMs()) {
                var expired =
                        CompletableFuture.completedFuture(
                                Response.buildErrorResponse(
                                        schedulingTimeoutError(),
                                        "request scheduling deadline has expired before selection"));
                request.setFuture(expired);
                return expired;
            }
            var config = request.getConfig();
            var future =
                    new RequestFuture(
                            (kind, response, failure) ->
                                    tryCompleteFromExternal(
                                            request, kind, response, failure, true));
            var registration =
                    repository.register(
                            request,
                            this,
                            future,
                            config.getRequestLifecycle().getRequest().getTimeoutMs(),
                            schedulingTimeoutError());
            if (registration != RequestRepository.RegistrationResult.REGISTERED) {
                var error =
                        registration == RequestRepository.RegistrationResult.CLOSED
                                ? StrategyErrorType.SCHEDULER_UNAVAILABLE
                                : StrategyErrorType.INVALID_REQUEST;
                String detail =
                        registration == RequestRepository.RegistrationResult.DUPLICATE_ID
                                ? "duplicate request_id: " + request.getRequestId()
                                : "request registration rejected: " + registration;
                return CompletableFuture.completedFuture(
                        Response.buildErrorResponse(error, detail));
            }
            FlexlbTrace.setScheduleAttribute(
                    request.getTraceContext(),
                    FlexlbTrace.SCHEDULE_MODE,
                    config.getScheduler().getType().name());
            future.whenComplete(
                    (response, failure) -> {
                        try {
                            if (failure == null && response != null && response.isSuccess())
                                cacheTrace.report(request);
                        } catch (RuntimeException error) {
                            Logger.warn("Route tracing failed", error);
                        } finally {
                            request.releaseSchedulingInput();
                        }
                    });
            expirationTimer.update(request, request.snapshot().deadlines());
            onRegistered.run();
            if (repository.isClosed()) cancel(request, CancelReason.SHUTDOWN);
            else if (System.currentTimeMillis() >= request.getRequestExpiresAtMs())
                cancel(request, CancelReason.DEADLINE_EXCEEDED);
            if (future.isDone()) closeRequest(request, null);
            else scheduleRequest(request);
            return future;
        } catch (RuntimeException failure) {
            var response =
                    Response.buildErrorResponse(
                            StrategyErrorType.DISPATCH_FAILED, failure.getMessage());
            if (request != null && request.getRequestScheduler() == this) {
                publish(request, response, false);
                return request.getFuture();
            }
            return CompletableFuture.completedFuture(response);
        }
    }

    public RequestState cancel(String id, CancelReason reason) {
        return cancel(id, reason, org.flexlb.balance.schedule.RequestPhase.GENERATION);
    }

    public RequestState cancel(
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
        return repository.findTerminal(id, phase, this);
    }

    /**
     * Records cancellation on the request. Response selection and resource cleanup use the same
     * closing path as worker completion; an executing selection finishes its own rollback first.
     */
    public RequestState cancel(RequestContext ctx, CancelReason reason) {
        Objects.requireNonNull(reason);
        if (ctx == null || !isCurrentRequest(ctx)) return null;
        synchronized (ctx) {
            if (ctx.snapshot().phase() == Phase.CLOSED)
                return ctx.snapshot().requestRecordForArchive();
            var diagnostics = describeWait(ctx, reason, reason.getMessage());
            var error =
                    switch (reason) {
                        case DEADLINE_EXCEEDED -> schedulingTimeoutError();
                        case PRIORITY_PREEMPTED -> StrategyErrorType.PRIORITY_PREEMPTED;
                        default -> StrategyErrorType.REQUEST_CANCELLED;
                    };
            boolean changed =
                    ctx.requestCancel(
                            reason,
                            false,
                            System.currentTimeMillis(),
                            error,
                            diagnostics == null
                                    ? reason.getMessage()
                                    : String.valueOf(diagnostics.get("cause")),
                            null);
            if (changed && diagnostics != null) ctx.setSchedulingDiagnostics(diagnostics);
        }
        try {
            processPendingRequestActions(ctx);
        } catch (RuntimeException | Error failure) {
            recordRequestFailure(failure);
            throw failure;
        }
        return RequestState.from(ctx.snapshot());
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
    public SelectionAttempt currentSelectionAttempt(RequestRoute route) {
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
     * Stops this scheduler and cancels only requests registered with it.
     * Shared admission and execution resources remain owned by the application.
     */
    public final void stopAndCancelRequests() {
        Throwable failure = Failures.run(null, this::close);
        for (var context :
                repository.snapshotActive(request -> request.getRequestScheduler() == this)) {
            failure = Failures.run(failure, () -> cancel(context, CancelReason.SHUTDOWN));
        }
        Failures.rethrow(failure, "scheduler request cancellation failed");
    }

    public final void awaitRequestCleanup() {
        closeQueue.awaitIdle();
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

    public synchronized void throwIfRequestFailed() {
        Failures.rethrow(requestProcessingFailure, "request processing failed");
    }

    /**
     * Archives this scheduler's remaining requests after endpoints have released their records.
     */
    public final void finishRequestCleanup() {
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

    /**
     * Direct scheduling owns one selection through its final result; queued scheduling may retain
     * a blocked selection while preemption waits for resources.
     */
    protected final SelectionResult<RequestRoute, CapacityKey> planRoute(
            RequestContext request,
            java.util.function.Supplier<SelectionResult<RequestRoute, CapacityKey>> select) {
        var exact = tryBeginRouteSelection(request);
        if (exact == null) return SelectionResult.closed();
        var result = executeRouteSelection(exact, select);
        if (Objects.equals(request.currentSelectionAttempt(), exact))
            finishSelectionWithoutCommit(exact, null);
        return result;
    }

    protected SelectionAttempt tryBeginRouteSelection(RequestContext request) {
        if (repository.isClosed() || request == null || request.getRequestScheduler() != this)
            return null;
        synchronized (request) {
            return repository.isCurrent(request) ? request.beginRouteSelection() : null;
        }
    }

    /**
     * Associates acquired registrations with the request before publishing. Failed publication rolls
     * them back; a pure capacity miss leaves the selection available to the queue's preemption path.
     */
    protected final SelectionResult<RequestRoute, CapacityKey> executeRouteSelection(
            SelectionAttempt exact,
            java.util.function.Supplier<SelectionResult<RequestRoute, CapacityKey>> select) {
        var request = exact.requestContext();
        checkState(
                Objects.equals(request.currentSelectionAttempt(), exact),
                "routing requires its current selection");
        RequestRoute acquired = null;
        SelectionResult<RequestRoute, CapacityKey> result = null;
        Throwable failure = null;
        boolean committed = false;
        try {
            result = Objects.requireNonNull(select.get(), "routing result");
            if (result.status() == SelectionResult.Status.SUCCESS) {
                checkState(result.value().ctx() == request, "route belongs to another request");
                acquired = result.value();
                request.registerSelectedRoute(exact, acquired);
            }
            request.recordRoutingDiagnostics(exact, result.diagnostics());
            if (request.isResponseDone()
                    || !request.canPublishRoute(exact)
                    || routeSelectionStopped()) result = SelectionResult.closed();
            else if (acquired != null) {
                result = acquired.validateEndpoints(SelectionResult.success(acquired));
                if (result.status() == SelectionResult.Status.SUCCESS)
                    result = publishSelectedRoute(acquired);
                committed = result.status() == SelectionResult.Status.SUCCESS;
            }
            return acquired == null ? result : selectedRouteResult(result);
        } catch (RuntimeException | Error problem) {
            failure = problem;
            throw problem;
        } finally {
            if (!committed) {
                boolean waitingForCapacity =
                        result != null && result.status() == SelectionResult.Status.BLOCKED;
                Throwable cleanup;
                if (waitingForCapacity && acquired != null) {
                    cleanup = releaseRouteResources(acquired, ReleaseReason.LOCAL_ROLLBACK, false);
                    if (cleanup != null) {
                        var rollbackError =
                                Response.buildErrorResponse(
                                        StrategyErrorType.DISPATCH_FAILED,
                                        "selection rollback failed");
                        cleanup =
                                Failures.run(
                                        cleanup,
                                        () -> finishSelectionWithoutCommit(exact, rollbackError));
                    }
                } else {
                    cleanup = waitingForCapacity ? null : finishUnpublishedRoute(acquired, exact);
                }
                if (cleanup != null) recordRequestFailure(cleanup);
                if (failure != null) Failures.append(failure, cleanup);
                else Failures.rethrow(cleanup, "selection rollback failed");
            }
        }
    }

    protected boolean routeSelectionStopped() {
        return false;
    }

    protected SelectionResult<RequestRoute, CapacityKey> publishSelectedRoute(RequestRoute route) {
        var request = route.ctx();
        request.recordRouteCommitStart(System.nanoTime());
        var exact = currentSelectionAttempt(route);
        if (exact == null) return route.validateEndpoints(SelectionResult.closed());
        if (route.encoderEndpoint() != null) {
            if (!route.encoderEndpoint().commitSelection(route))
                return route.validateEndpoints(
                        SelectionResult.blocked(
                                new CapacityKey(
                                        RoleType.ENCODER,
                                        route.encoderEndpoint().getStatus().getGroup(),
                                        route.encoderEndpoint().ipPort())));
        } else if (!route.prefillEndpoint().commitSelection(route)) {
            return route.validateEndpoints(
                    SelectionResult.rejected(
                            Response.buildErrorResponse(
                                    StrategyErrorType.DISPATCH_FAILED,
                                    "Selected Prefill registration was not valid for commit")));
        }
        return commitRoute(route, exact)
                ? SelectionResult.success(route)
                : route.validateEndpoints(SelectionResult.closed());
    }

    protected SelectionResult<RequestRoute, CapacityKey> selectedRouteResult(
            SelectionResult<RequestRoute, CapacityKey> result) {
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
     * Open requests retain their timers; ending requests use the single response-and-cleanup path.
     */
    protected void processPendingRequestActions(RequestContext request) {
        var state = request.snapshot();
        if (state.phase() == Phase.OPEN) {
            recover(
                    this::recordRequestFailure,
                    () -> expirationTimer.update(request, state.deadlines()));
        } else {
            closeRequest(request, null);
        }
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

    /**
     * Roll back unpublished resources and finish selection, returning any cleanup or completion failure.
     */
    static Throwable finishUnpublishedRoute(RequestRoute route, SelectionAttempt routingContext) {
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
                            : releaseRouteResources(route, ReleaseReason.LOCAL_ROLLBACK, false);
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
            RequestRoute route, ReleaseReason reason, boolean notificationsSubmitted) {
        var ctx = route.ctx();
        var state = ctx.snapshot();
        boolean rollback = reason == ReleaseReason.LOCAL_ROLLBACK;
        boolean mayReleaseDecode =
                reason != ReleaseReason.COUNTERPART_FINISHED || state.committedRoute() == null;
        boolean prefillAndEncoderReleased =
                !rollback && state.route() == route && state.prefillCleanupCompleted();
        boolean decodeReleased =
                !rollback && state.route() == route && state.decodeCleanupCompleted();
        Throwable failure = null;
        if (!prefillAndEncoderReleased) {
            if (route.encoderEndpoint() != null)
                failure =
                        Failures.run(
                                failure,
                                () ->
                                        route.encoderEndpoint()
                                                .releaseRequest(route.encoderRegistration()));
            if (route.prefillEndpoint() != null)
                failure =
                        Failures.run(
                                failure,
                                () ->
                                        route.prefillEndpoint()
                                                .releaseRequest(route.prefillRegistration()));
            prefillAndEncoderReleased = failure == null;
        }
        try {
            if (!decodeReleased && mayReleaseDecode) {
                var endpoint = route.decodeEndpoint();
                var registration = route.decodeRegistration();
                if (endpoint == null || registration == null) {
                    decodeReleased = true;
                } else if (rollback) {
                    endpoint.rollbackSelection(registration);
                    decodeReleased = true;
                } else if (reason == ReleaseReason.COUNTERPART_FINISHED) {
                    decodeReleased = endpoint.releaseUnconfirmedRequest(registration);
                } else {
                    endpoint.releaseRequest(registration);
                    decodeReleased = true;
                }
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
                    reserveResponse(
                            ctx, cancelled ? ResponseKind.CANCELLATION : ResponseKind.TERMINAL);
            publication = candidate;
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
                                        .endState()
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
     * Chooses one ending response under the request lock, then submits response and resource cleanup
     * outside it. A selection still in progress keeps the ending pending until rollback completes.
     */
    void closeRequest(RequestContext request, RequestEndEvent ending) {
        checkState(
                !Thread.holdsLock(request),
                "request closing requires the Context lock to be released");
        if (!isCurrentRequest(request)) return;
        Registration publication = null;
        try {
            synchronized (request) {
                var before = request.snapshot();
                if (ending == null && before.phase() == Phase.OPEN
                        || ending != null
                                && ending.route() != null
                                && ending.route() != before.route()) return;
                Registration responseCandidate = null;
                if (before.selectionId() == 0
                        && before.responseForFuture() == null
                        && !request.isResponseDone()) {
                    responseCandidate = reserveResponse(request, ResponseKind.TERMINAL);
                    publication = responseCandidate;
                }
                var after =
                        request.endRequest(ending, System.currentTimeMillis(), responseCandidate);
                if (after.endState() == null) {
                    if (publication != null) publication.close();
                    return;
                }
                if (publication != null) {
                    if (after.responseForFuture() == responseCandidate) {
                        publication.prepare(RESPONSE, response(after, before.route()), null);
                    } else {
                        publication.close();
                        publication = null;
                    }
                }
            }
        } catch (RuntimeException | Error failure) {
            request.endRequest(ending, System.currentTimeMillis(), null);
            if (publication != null) publication.failIfUnsubmitted(failure);
            Failures.run(failure, () -> scheduleRequestCleanup(request, null));
            throw failure;
        }
        scheduleRequestCleanup(request, publication);
    }

    /**
     * Applies a cancellation reply only to the original active request and committed route.
     * A request may already be archived when an outstanding RPC replies.
     */
    void onEngineCancelReply(RequestRoute route, EngineGrpcClient.CancelReply reply) {
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
        Map<String, Object> details = new java.util.LinkedHashMap<>();
        String stage;
        if (state.committedRoute() != null) stage = "DELIVERY";
        else if (state.selectionId() != 0) stage = "PLACEMENT";
        else stage = state.route() == null ? "WAITING_FOR_PLACEMENT" : "BEFORE_DELIVERY";
        details.put("stage", stage);
        details.put("capturedAtMs", System.currentTimeMillis());
        if (state.selectionId() != 0) details.put("routingPhase", state.routingPhase().name());
        if (state.committedRoute() == null
                && state.route() == null
                && request.getSchedulingDiagnostics() != null)
            details.put("lastPlacementObservation", request.getSchedulingDiagnostics());
        details.put("cause", message + "; " + details);
        return java.util.Collections.unmodifiableMap(details);
    }

    /**
     * Hand off one complete close operation; rejected work retains its unfinished cleanup state.
     */
    private void scheduleRequestCleanup(RequestContext ctx, Registration publication) {
        var state = ctx.snapshot();
        boolean responseBeforeCleanup =
                state.endState() != null && state.endKind() == EndKind.ENDPOINT_CLOSED;
        boolean inactivity = state.endState() != null && state.endKind() == EndKind.INACTIVITY;
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
            var state = ctx.snapshot();
            var reason = releaseReason(state);
            var route = state.route();
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

    private static ReleaseReason releaseReason(RequestContext.Snapshot state) {
        if (state.phase() != Phase.ENDING || state.selectionId() != 0L) return null;
        if (state.committedRoute() != null
                && state.prefillWorkerReleaseConfirmed()
                && state.decodeWorkerReleaseConfirmed()) return ReleaseReason.REMOTE_CLEANUP;
        // Inactivity expiry drops local bookkeeping; confirmed capacity still follows Worker
        // reports.
        return state.inactivityExpired()
                ? ReleaseReason.EXPIRED
                : ReleaseReason.COUNTERPART_FINISHED;
    }

    private static Response response(RequestContext.Snapshot ending, RequestRoute route) {
        if (ending.endState() == RequestState.State.COMPLETED && route != null) {
            return route.successResponse();
        }
        return ending.endError() == StrategyErrorType.RESOURCE_EXHAUSTED
                ? Response.error(
                        ending.endError(),
                        AdmissionRejectReason.RESOURCE_EXHAUSTED,
                        ending.endDetail())
                : Response.buildErrorResponse(ending.endError(), ending.endDetail());
    }

    /**
     * Reserves response execution, commits once, then invokes the response outside the request lock.
     */
    private boolean commitRoute(RequestRoute route, SelectionAttempt attempt) {
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
                var candidate = reserveResponse(ctx, ResponseKind.ACK);
                publication = candidate;
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

    private Registration reserveResponse(RequestContext ctx, ResponseKind kind) {
        return new Registration(responseCompletions, ctx, (RequestFuture) ctx.getFuture(), kind);
    }

    private Response admissionRejection(RequestContext request) {
        if (request == null || request.getRequest() == null)
            return Response.error(StrategyErrorType.INVALID_REQUEST);
        if (repository.isStopping() || repository.isClosed())
            return Response.buildErrorResponse(
                    StrategyErrorType.SCHEDULER_UNAVAILABLE,
                    "request scheduler is shutting down and no longer accepts new requests");
        if (request.getRequestScheduler() != null)
            return Response.buildErrorResponse(
                    StrategyErrorType.INVALID_REQUEST, "request context is already registered");
        if (request.getSchedulingMetadata() == null)
            return Response.buildErrorResponse(
                    StrategyErrorType.INVALID_REQUEST, "missing scheduling metadata");
        return null;
    }

    enum ReleaseReason {
        COUNTERPART_FINISHED,
        LOCAL_ROLLBACK,
        NOT_SENT,
        REMOTE_CLEANUP,
        EXPIRED
    }
}
