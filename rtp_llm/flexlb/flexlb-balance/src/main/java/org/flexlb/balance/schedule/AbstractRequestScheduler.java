package org.flexlb.balance.schedule;

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
import org.flexlb.worker.endpoint.WorkerEndpoint.CommitStatus;
import org.flexlb.worker.endpoint.WorkerEventSink;
import org.flexlb.worker.endpoint.WorkerRequestEvent;
import org.flexlb.worker.rpc.EngineGrpcClient;

import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.StampedLock;

import static com.google.common.base.Preconditions.checkState;
import static org.flexlb.balance.schedule.RequestContext.EndCause;
import static org.flexlb.balance.schedule.RequestContext.Phase;
import static org.flexlb.balance.schedule.RequestContext.RequestEndEvent;
import static org.flexlb.balance.schedule.RequestContext.ResponseKind;
import static org.flexlb.balance.schedule.RequestContext.Snapshot;
import static org.flexlb.balance.schedule.ResponseCompletionExecutor.RequestFuture.Completion.RESPONSE;
import static org.flexlb.util.Failures.recover;

/**
 * Registers requests and coordinates cancellation, deadlines, responses and resource cleanup.
 * Runs shared route selection, publication and cleanup; subclasses supply waiting and rejection policies.
 * Tracks its deadline and cleanup work for shutdown; the application closes the shared executor.
 */
public abstract class AbstractRequestScheduler implements RequestScheduler {
    protected final RequestRepository repository;
    protected final ExecutorService requestWork;
    private final ExpirationTimer expirationTimer;
    private final EngineRequestCanceller engineRequestCanceller;
    private final ResponseCompletionExecutor responseCompletions;
    private final TheoryCacheHitReporter cacheTrace;
    private final StampedLock requestWorkLock = new StampedLock();
    private final ThreadLocal<Boolean> executingRequestWork = new ThreadLocal<>();

    private Throwable requestProcessingFailure;

    protected AbstractRequestScheduler(
            RequestRepository repository,
            ExecutorService requestWork,
            ExpirationTimer deadlines,
            EngineRequestCanceller engineRequestCanceller,
            ResponseCompletionExecutor responseCompletions,
            TheoryCacheHitReporter cacheTrace) {
        this.repository = Objects.requireNonNull(repository);
        this.requestWork = Objects.requireNonNull(requestWork);
        this.expirationTimer = Objects.requireNonNull(deadlines);

        this.engineRequestCanceller = Objects.requireNonNull(engineRequestCanceller);
        this.responseCompletions = Objects.requireNonNull(responseCompletions);
        this.cacheTrace = Objects.requireNonNull(cacheTrace);
    }

    public void onWorkerRequestEvent(RequestRoute route, WorkerRequestEvent event) {
        var ctx = route.ctx();
        if (ctx.getRequestScheduler() != this) return;
        try {
            if (handleWorkerEvent(route, event)) {
                var phase = ctx.snapshot().phase();
                if (phase == Phase.OPEN) {
                    syncDeadlineTimers(ctx);
                } else {
                    try {
                        if (phase == Phase.CLOSED) engineRequestCanceller.remove(route);
                    } finally {
                        closeRequest(ctx, null);
                    }
                }
            }
        } catch (Throwable failure) {
            recordRequestFailure(failure);
            Logger.error("Worker request event processing failed", failure);
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
                            config.getRequestLifecycle().getInactivityTimeoutMs(),
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
                publishResponse(request, response, false);
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
            var diagnostics = buildTimeoutDiagnostics(ctx, reason, reason.getMessage());
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
            if (ctx.snapshot().phase() == Phase.OPEN) {
                syncDeadlineTimers(ctx);
            } else {
                closeRequest(ctx, null);
            }
        } catch (RuntimeException | Error failure) {
            recordRequestFailure(failure);
            throw failure;
        }
        return RequestState.from(ctx.snapshot());
    }

    /**
     * Local submission failure uses the same arbitration as external Future completion.
     */
    public boolean publishResponse(RequestContext ctx, Response response, boolean synchronous) {
        return tryCompleteFromExternal(
                ctx, RequestFuture.Completion.RESPONSE, response, null, synchronous);
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
        if (after == null) return;
        if (after.phase() == Phase.OPEN) {
            syncDeadlineTimers(ctx);
        } else {
            closeRequest(ctx, null);
        }
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
        submitRequestWork(
                acquireRequestWorkLock(),
                () -> {
                    if (nowMs < exact.atMs()) {
                        expirationTimer.update(ctx, ctx.snapshot().deadlines());
                    } else if (ctx.snapshot().phase() == Phase.OPEN) {
                        syncDeadlineTimers(ctx);
                    } else {
                        closeRequest(ctx, null);
                    }
                },
                this::recordRequestFailure);
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

    /**
     * Waits for accepted deadline processing and both synchronous and asynchronous cleanup.
     */
    public final void awaitRequestWork() {
        checkState(
                !Boolean.TRUE.equals(executingRequestWork.get()),
                "request work cannot wait for itself");
        long stamp = requestWorkLock.writeLock();
        requestWorkLock.unlockWrite(stamp);
    }

    /**
     * Reject reentrant shutdown before an application lifecycle acquires its shutdown monitor.
     */
    public void checkShutdownThread() {
        checkState(
                !isExecutingRequestWork()
                        && !engineRequestCanceller.isStoppingOnCurrentThread()
                        && !responseCompletions.isCompletingOnCurrentThread(),
                "scheduler shutdown cannot wait for its own request work or response callback");
    }

    public synchronized void throwIfRequestFailed() {
        Failures.rethrow(requestProcessingFailure, "request processing failed");
    }

    /**
     * Archives this scheduler's remaining requests after endpoints have released their records.
     */
    public final void closeRemainingRequestsOnShutdown() {
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
                                                    EndCause.SCHEDULER_SHUTDOWN,
                                                    StrategyErrorType.DISPATCH_FAILED,
                                                    "request scheduler is shutting down")));
        }
        Failures.rethrow(failure, "request shutdown finalization failed");
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
                    cleanup = rollbackSelectedRouteResources(acquired);
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
        if (exact == null) return SelectionResult.closed();
        WorkerEventSink events = event -> onWorkerRequestEvent(route, event);
        if (route.decodeEndpoint() != null) {
            var decodeStatus =
                    route.decodeEndpoint().confirmSelection(route.decodeRegistration(), events);
            if (decodeStatus == CommitStatus.UNAVAILABLE)
                return SelectionResult.retry(Response.error(StrategyErrorType.NO_DECODE_WORKER));
            if (decodeStatus == CommitStatus.INVALID_REGISTRATION) return SelectionResult.closed();
        }
        var endpoint =
                route.encoderEndpoint() != null ? route.encoderEndpoint() : route.prefillEndpoint();
        var status =
                route.encoderEndpoint() != null
                        ? route.encoderEndpoint()
                                .confirmSelection(route.encoderRegistration(), events)
                        : route.prefillEndpoint()
                                .confirmSelection(
                                        route.prefillRegistration(),
                                        events,
                                        request.getPlacementSequence());
        if (request.isResponseDone() || request.snapshot().cancellationReason() != null)
            return SelectionResult.closed();
        if (status == CommitStatus.UNAVAILABLE)
            return SelectionResult.retry(
                    Response.error(StrategyErrorType.forRole(endpoint.getStatus().getRole())));
        if (status == CommitStatus.INVALID_REGISTRATION)
            return SelectionResult.rejected(
                    Response.buildErrorResponse(
                            StrategyErrorType.DISPATCH_FAILED,
                            "Selected "
                                    + (route.encoderEndpoint() != null ? "Encoder" : "Prefill")
                                    + " registration was not valid for commit"));
        return commitRoute(route, exact);
    }

    protected SelectionResult<RequestRoute, CapacityKey> selectedRouteResult(
            SelectionResult<RequestRoute, CapacityKey> result) {
        return result;
    }

    protected boolean isExecutingRequestWork() {
        return Boolean.TRUE.equals(executingRequestWork.get());
    }

    protected abstract StrategyErrorType schedulingTimeoutError();

    protected final boolean checkRequestMode(RequestContext context, boolean queued) {
        if (context.getConfig().isQueue() == queued) return true;
        publishResponse(
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

    void syncDeadlineTimers(RequestContext request) {
        recover(
                this::recordRequestFailure,
                () -> expirationTimer.update(request, request.snapshot().deadlines()));
    }

    /**
     * Interprets role feedback under the request monitor, then uses the same ending path as local
     * failures. Endpoint closure before execution invalidates selection until rollback completes.
     */
    static boolean handleWorkerEvent(RequestRoute route, WorkerRequestEvent event) {
        var ctx = route.ctx();
        synchronized (ctx) {
            var before = ctx.snapshot();
            if (!ctx.updateWorkerStatus(route, event)) return false;
            var role = event.role();
            boolean finalRole =
                    role == RoleType.DECODE
                            || role == RoleType.PDFUSION
                            || role == RoleType.ENCODER;
            EndCause cause = null;
            boolean invalidatedSelection = false;
            if (before.endState() == null) {
                switch (event.type()) {
                    case COMPLETED -> {
                        if (finalRole) cause = EndCause.WORKER_COMPLETED;
                    }
                    case FAILED -> cause = EndCause.WORKER_FAILED;
                    case PRIORITY_CANCELLED -> cause = EndCause.PRIORITY_PREEMPTED;
                    case ENDPOINT_CLOSED -> {
                        if (finalRole
                                || before.routeCommittedAtMs() < 0 && !before.decodeAccepted()) {
                            if (before.isSelectingRoute()
                                    && before.routeCommittedAtMs() < 0
                                    && !before.decodeAccepted()) {
                                ctx.invalidateSelection(event.detail());
                                invalidatedSelection = true;
                            } else cause = EndCause.ENDPOINT_CLOSED;
                        }
                    }
                    default -> {}
                }
            }
            if (cause != null) {
                String detail = event.detail();
                if (cause == EndCause.WORKER_FAILED && (detail == null || detail.isBlank()))
                    detail = "worker error code " + event.errorCode();
                ctx.endRequest(
                        new RequestEndEvent(route, cause, null, detail), event.nowMs(), null);
            }
            var after = ctx.snapshot();
            return before.deadlines() != after.deadlines()
                    || invalidatedSelection
                    || cause != null
                    || after.endState() != null && !before.equals(after);
        }
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
            failure = route == null ? null : rollbackSelectedRouteResources(route);
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

    boolean isCurrentRequest(RequestContext ctx) {
        return ctx.getRequestScheduler() == this && repository.isCurrent(ctx);
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
        if (reply != EngineGrpcClient.CancelReply.REQUEST_CLEANED
                && reply != EngineGrpcClient.CancelReply.REQUEST_FENCED) return;
        try {
            if (ctx.recordRemoteCleanup(
                    route,
                    reply == EngineGrpcClient.CancelReply.REQUEST_CLEANED,
                    System.currentTimeMillis())) {
                try {
                    if (ctx.snapshot().phase() == Phase.CLOSED)
                        engineRequestCanceller.remove(route);
                } finally {
                    closeRequest(ctx, null);
                }
            }
        } catch (Throwable failure) {
            recordRequestFailure(failure);
            Logger.error("Remote cleanup reply processing failed", failure);
        }
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
            var evidence = buildTimeoutDiagnostics(ctx, CancelReason.DEADLINE_EXCEEDED, detail);
            diagnosis = evidence == null ? detail : String.valueOf(evidence.get("cause"));
            if (evidence != null) ctx.setSchedulingDiagnostics(evidence);
        }
        return ctx.recordDeadline(token, nowMs, diagnosis);
    }

    private Map<String, Object> buildTimeoutDiagnostics(
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
        if (state.isSelectingRoute()) details.put("routingPhase", "SELECTING");
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
                state.endState() != null && state.endCause() == EndCause.ENDPOINT_CLOSED;
        boolean inactivity =
                state.endState() != null
                        && state.endCause() != null
                        && state.endCause().isTimeout();
        Throwable queueFailure = !inactivity ? Failures.run(null, () -> removeRequest(ctx)) : null;
        if (publication != null && responseBeforeCleanup && !inactivity) {
            var ready = publication;
            queueFailure = Failures.run(queueFailure, () -> ready.publish(false));
            publication = null;
        }
        long stamp = acquireRequestWorkLock();
        if (ctx.requestCleanup(publication)) {
            Runnable cleanup =
                    () -> {
                        Throwable failure = null;
                        do {
                            var cause = ctx.snapshot().endCause();
                            if (cause != null && cause.isTimeout()) {
                                failure = Failures.run(failure, () -> removeRequest(ctx));
                            }
                            failure = Failures.run(failure, () -> cleanupRequest(ctx));
                            var response = ctx.takeCleanupResponse();
                            if (response != null) {
                                failure = Failures.run(failure, () -> response.publish(false));
                            }
                        } while (!ctx.finishCleanupPass());
                        if (failure != null) recordRequestFailure(failure);
                        Failures.rethrow(failure, "request close failed");
                    };
            try {
                if (responseBeforeCleanup || inactivity) {
                    submitRequestWork(
                            stamp,
                            cleanup,
                            failure -> {
                                var response = ctx.rejectCleanup();
                                recordRequestFailure(failure);
                                if (response != null) response.failIfUnsubmitted(failure);
                            });
                } else {
                    runRequestWork(stamp, cleanup);
                }
            } catch (RuntimeException | Error failure) {
                Failures.append(failure, queueFailure);
                throw failure;
            }
        } else {
            requestWorkLock.unlockRead(stamp);
        }
        if (queueFailure != null) recordRequestFailure(queueFailure);
        Failures.rethrow(queueFailure, "request queue removal failed");
    }

    /**
     * Acquire before handoff so shutdown also waits for work that has yet to start.
     * A running task can start another cleanup while shutdown is waiting for both.
     */
    private long acquireRequestWorkLock() {
        long stamp = requestWorkLock.tryReadLock();
        return stamp != 0L ? stamp : requestWorkLock.readLock();
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
            var route = state.route();
            if (state.phase() == Phase.ENDING && state.selectionId() == 0L) {
                boolean prefillAndEncoderReleased =
                        route == null || state.prefillCleanupCompleted();
                boolean decodeReleased = route == null || state.decodeCleanupCompleted();
                Throwable releaseFailure = null;
                if (!prefillAndEncoderReleased) {
                    if (route.encoderEndpoint() != null)
                        releaseFailure =
                                Failures.run(
                                        releaseFailure,
                                        () ->
                                                route.encoderEndpoint()
                                                        .releaseRequest(
                                                                route.encoderRegistration()));
                    boolean mayReleasePrefill =
                            state.committedRoute() == null || state.prefillWorkerReleaseConfirmed();
                    if (route.prefillEndpoint() != null && mayReleasePrefill)
                        releaseFailure =
                                Failures.run(
                                        releaseFailure,
                                        () ->
                                                route.prefillEndpoint()
                                                        .releaseRequest(
                                                                route.prefillRegistration()));
                    prefillAndEncoderReleased =
                            releaseFailure == null
                                    && (route.prefillEndpoint() == null || mayReleasePrefill);
                }
                boolean remoteReleaseConfirmed =
                        state.committedRoute() != null
                                && state.prefillWorkerReleaseConfirmed()
                                && state.decodeWorkerReleaseConfirmed();
                try {
                    if (!decodeReleased) {
                        var endpoint = route.decodeEndpoint();
                        var registration = route.decodeRegistration();
                        if (remoteReleaseConfirmed) {
                            // Remove local registration only; reported Worker occupancy remains
                            // authoritative.
                            if (endpoint != null && registration != null)
                                endpoint.releaseRequest(registration);
                            decodeReleased = true;
                        } else if (state.committedRoute() == null) {
                            decodeReleased =
                                    endpoint == null
                                            || registration == null
                                            || endpoint.releaseUnconfirmedRequest(registration);
                        }
                    }
                } catch (Throwable cleanupFailure) {
                    releaseFailure = Failures.append(releaseFailure, cleanupFailure);
                }
                problem = Failures.append(problem, releaseFailure);
                ctx.recordCleanupProgress(
                        route,
                        prefillAndEncoderReleased,
                        decodeReleased,
                        problem == null,
                        System.currentTimeMillis());
            } else if (problem == null) {
                ctx.recordCleanupProgress(route, false, false, true, System.currentTimeMillis());
            }
            failure = Failures.append(failure, problem);
            if (problem == null) break;
        }
        var after = ctx.snapshot();
        if (delivery != null && after.prefillCleanupCompleted() && after.decodeCleanupCompleted()) {
            failure = Failures.run(failure, () -> engineRequestCanceller.remove(delivery));
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
        if (delivery != null
                && delivery.prefillEndpoint() != null
                && state.cancellationReason() != null) {
            var failure =
                    Failures.run(
                            null,
                            () ->
                                    engineRequestCanceller.cancelRequest(
                                            delivery,
                                            this::onEngineCancelReply,
                                            this::recordRequestFailure));
            if (failure != null) return failure;
        }
        return null;
    }

    private void submitRequestWork(
            long stamp, Runnable work, java.util.function.Consumer<Throwable> rejected) {
        var startedOrRejected = new AtomicBoolean();
        try {
            requestWork.execute(
                    () -> {
                        if (!startedOrRejected.compareAndSet(false, true)) return;
                        try {
                            runRequestWork(stamp, work);
                        } catch (RuntimeException | Error failure) {
                            Logger.error("Request work failed", failure);
                        }
                    });
        } catch (RuntimeException | Error failure) {
            if (startedOrRejected.compareAndSet(false, true)) {
                try {
                    Failures.run(failure, () -> rejected.accept(failure));
                } finally {
                    requestWorkLock.unlockRead(stamp);
                }
            }
            throw failure;
        }
    }

    private void runRequestWork(long stamp, Runnable work) {
        var previous = executingRequestWork.get();
        executingRequestWork.set(true);
        try {
            work.run();
        } finally {
            if (previous == null) executingRequestWork.remove();
            else executingRequestWork.set(previous);
            requestWorkLock.unlockRead(stamp);
        }
    }

    /**
     * Rolls back each registration acquired by one unpublished selection, without ending selection.
     * Decode rollback rejects a registration already confirmed by the Worker.
     */
    private static Throwable rollbackSelectedRouteResources(RequestRoute route) {
        Throwable failure = null;
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
        boolean prefillAndEncoderReleased = failure == null;
        boolean decodeReleased = false;
        try {
            if (route.decodeEndpoint() != null && route.decodeRegistration() != null)
                route.decodeEndpoint().rollbackSelection(route.decodeRegistration());
            decodeReleased = true;
        } catch (Throwable problem) {
            failure = Failures.append(failure, problem);
        }
        route.ctx()
                .recordCleanupProgress(
                        route,
                        prefillAndEncoderReleased,
                        decodeReleased,
                        false,
                        System.currentTimeMillis());
        return failure;
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
    private SelectionResult<RequestRoute, CapacityKey> commitRoute(
            RequestRoute route, SelectionAttempt attempt) {
        var ctx = route.ctx();
        Registration publication = null;
        boolean committed = false;
        try {
            long visibilityAtMs = route.predictedVisibilityAtMs(System.currentTimeMillis());
            synchronized (ctx) {
                if (!repository.isCurrent(ctx)
                        || ctx.isResponseDone()
                        || ctx.snapshot().cancellationReason() != null)
                    return SelectionResult.closed();
                var unavailable = RequestRoute.firstUnavailableEndpoint(route.roles());
                if (unavailable != null)
                    return SelectionResult.retry(
                            Response.error(
                                    StrategyErrorType.forRole(unavailable.getStatus().getRole())));
                if (route.decodeEndpoint() != null
                        && !route.decodeEndpoint()
                                .registrationState(route.decodeRegistration())
                                .registered()) return SelectionResult.closed();
                long nowMs = System.currentTimeMillis();
                var candidate = reserveResponse(ctx, ResponseKind.ACK);
                publication = candidate;
                committed = ctx.tryCommitRoute(attempt, route, candidate, nowMs, visibilityAtMs);
                if (!committed) return SelectionResult.closed();
                publication.prepare(RESPONSE, route.successResponse(), null);
            }
            publication.publish(false);
            if (ctx.snapshot().phase() == Phase.OPEN) {
                syncDeadlineTimers(ctx);
            } else {
                closeRequest(ctx, null);
            }
            return SelectionResult.success(route);
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
                                                        EndCause.ROUTE_PUBLICATION_FAILED,
                                                        StrategyErrorType.DISPATCH_FAILED,
                                                        "route response failed")));
                if (cleanupFailure != null) recordRequestFailure(cleanupFailure);
                return SelectionResult.success(route);
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
}
