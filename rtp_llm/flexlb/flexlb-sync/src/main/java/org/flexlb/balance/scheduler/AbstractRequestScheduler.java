package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.balance.preemption.VictimResolution;
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
import java.util.Optional;
import java.util.concurrent.CompletableFuture;

import static com.google.common.base.Preconditions.checkState;
import static org.flexlb.balance.scheduler.ContextResult.Cancellation;
import static org.flexlb.balance.scheduler.ContextResult.CancellationFacts;
import static org.flexlb.balance.scheduler.ContextResult.CleanupAcknowledgement;
import static org.flexlb.balance.scheduler.ContextResult.DeliveryEvidence;
import static org.flexlb.balance.scheduler.ContextResult.EndEvidence;
import static org.flexlb.balance.scheduler.ContextResult.EndKind;
import static org.flexlb.balance.scheduler.ContextResult.Phase;
import static org.flexlb.balance.scheduler.ContextResult.Prediction;
import static org.flexlb.balance.scheduler.ContextResult.PreemptionReport;
import static org.flexlb.balance.scheduler.ContextResult.PreparationFailure;
import static org.flexlb.balance.scheduler.ContextResult.PublicationResult;
import static org.flexlb.balance.scheduler.ContextResult.ResponseCandidate;
import static org.flexlb.balance.scheduler.ContextResult.ResponseKind;
import static org.flexlb.balance.scheduler.ContextResult.RouteDelivered;
import static org.flexlb.balance.scheduler.ContextResult.SendState;
import static org.flexlb.balance.scheduler.ContextResult.Snapshot;
import static org.flexlb.balance.scheduler.ContextResult.WorkerRequestEvent;
import static org.flexlb.balance.scheduler.ContextResult.WorkerRequestEventType;
import static org.flexlb.balance.scheduler.ResponseCompletionExecutor.RequestFuture.Completion.RESPONSE;
import static org.flexlb.dao.loadbalance.Response.buildErrorResponse;
import static org.flexlb.util.Failures.recover;

/**
 * Registers, cancels and completes requests using the same scheduler instance that chooses work.
 * Subclasses own scheduling triggers, request selection and retry policy; RequestContext records
 * the facts needed to serialize cancellation, publication and resource release.
 * Closing reserves one counted job before freezing the outcome. The request's close lane
 * orders notifications, response handoff and resource release, including late cleanup signals;
 * inactivity never blocks the timer thread, and local-only cancellation settles before responding.
 */
public abstract class AbstractRequestScheduler implements RequestScheduler {
    protected final RequestRepository repository;
    protected final RequestTaskPool taskPool;
    protected final ExpirationTimer expirationTimer;
    protected final DeliveryDispatcher dispatcher;
    protected final ResponseCompletionExecutor responseCompletions;
    protected final DeliveryMetricsReporter deliveryReporter;
    protected final TheoryCacheHitReporter cacheTrace;
    protected final EndpointRegistry endpoints;
    private final Object shutdownLock = new Object();

    protected AbstractRequestScheduler(
            RequestRepository repository,
            RequestTaskPool taskPool,
            ExpirationTimer deadlines,
            DeliveryDispatcher dispatcher,
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
                    config.getDispatcher().requiresGenerateInput()
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
            if (!future.isDone()) onRequest(ctx);
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
        try (var accepted = taskPool.tryRegisterRequestWork(ctx)) {
            if (accepted == null) {
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
                failure = Failures.run(failure, () -> reconcilePreemption(ctx));
                failure = Failures.run(failure, () -> closeRequest(ctx, null));
            }
            var deadlines = ctx.snapshot().deadlines();
            if (ctx.snapshot().phase() == Phase.OPEN
                    && expirationTimer.needsUpdate(ctx, deadlines)) {
                failure = Failures.run(failure, () -> expirationTimer.update(ctx, deadlines));
            }
            if (failure != null) {
                taskPool.recordFailure(failure);
            }
            Failures.rethrow(failure, "cancellation effects failed");
            var state = RequestState.from(ctx.snapshot());
            return state != null && state.matchesBatch(batchId) ? state : null;
        }
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

    public PlacementOperation claimPlacement(RequestContext ctx) {
        if (repository.isClosed() || ctx == null || ctx.data().getRequestScheduler() != this) {
            return null;
        }
        synchronized (ctx) {
            return repository.isCurrent(ctx) ? ctx.beginPlacement(null) : null;
        }
    }

    /**
     * Check that this selecting task may still publish a route; no candidate is stored in Context.
     */
    public PlacementOperation placementFor(RouteResult route) {
        var ctx = route.ctx();
        synchronized (ctx) {
            var state = ctx.snapshot();
            if (repository.isClosed()
                    || !repository.isCurrent(ctx)
                    || ctx.data().getRequestScheduler() != this
                    || state.placement() == null) return null;
            var operation = state.placement();
            return ctx.canPublishRoute(operation) ? operation : null;
        }
    }

    /**
     * Publish one selected route and finish its placement before any response callback.
     * Every unpublished result releases its registrations before the scheduler can retry or wait.
     */
    public PlacementResult<RouteResult, PlacementKey> publishPlacement(
            RouteResult route, PlacementOperation admission, QueueExecutionSettings queue) {
        var placement = route.ctx().snapshot().placement();
        checkState(
                placement != null && placement == admission && admission.owner() == route.ctx(),
                "route requires its exact active admission");
        PlacementResult<RouteResult, PlacementKey> result = null;
        ContextResult.RouteDelivered delivered = null;
        Throwable failure = null;
        try {
            result = route.validateEndpoints(PlacementResult.success(route));
            if (result.status() == PlacementResult.Status.SUCCESS) {
                if (queue != null && route.encoderEndpoint() == null) {
                    result = enqueueRoute(route, queue);
                } else {
                    var publication = commitDirectRoute(route);
                    if (publication.status() == PlacementResult.Status.SUCCESS) {
                        delivered = publication.value();
                    } else {
                        result = publication.asFailure();
                        if (result.status() == PlacementResult.Status.BLOCKED
                                || result.status() == PlacementResult.Status.REJECTED
                                || result.status() == PlacementResult.Status.CLOSED
                                        && !route.ctx().data().getFuture().isDone()
                                        && route.ctx().snapshot().cancellation() == null)
                            result = route.validateEndpoints(result);
                    }
                }
            }
        } catch (RuntimeException | Error problem) {
            failure = problem;
            throw problem;
        } finally {
            boolean discarded = false;
            Throwable cleanup = null;
            if (delivered == null
                    && result != null
                    && result.status() == PlacementResult.Status.RETRY
                    && route.ctx().snapshot().route() == route) {
                cleanup =
                        releaseRouteResources(route, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
                if (cleanup == null) {
                    route.ctx().discardRoute(admission, route);
                    discarded = true;
                }
            }
            var unreleased = discarded ? null : route;
            cleanup =
                    Failures.run(
                            cleanup,
                            () ->
                                    Failures.rethrow(
                                            finishPlacement(unreleased, admission),
                                            "placement cleanup failed"));
            if (cleanup != null) taskPool.recordFailure(cleanup);
            if (failure != null) Failures.append(failure, cleanup);
            else if (delivered == null) Failures.rethrow(cleanup, "placement cleanup failed");
        }
        if (delivered != null) onDeliveryResult(delivered);
        return result;
    }

    /**
     * End placement. Return resource cleanup failures; admission completion failures propagate.
     */
    public static Throwable finishPlacement(RouteResult route, PlacementOperation admission) {
        var state = route == null ? null : route.ctx().snapshot();
        boolean published = state != null && state.route() == route;
        Throwable failure =
                route == null || published
                        ? null
                        : releaseRouteResources(
                                route, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        boolean rollbackFailed = failure != null;
        if (admission != null) {
            if (rollbackFailed) route.ctx().retainRouteForCleanup(admission, route);
            try {
                if (rollbackFailed)
                    AbstractRequestScheduler.failPlacement(
                            admission,
                            Response.buildErrorResponse(
                                    StrategyErrorType.DISPATCH_FAILED,
                                    "placement rollback failed"));
                else AbstractRequestScheduler.finishPlacement(admission);
            } catch (RuntimeException | Error completion) {
                Failures.append(completion, failure);
                throw completion;
            }
        }
        return failure;
    }

    public static void finishPlacement(PlacementOperation operation) {
        completePlacement(
                operation,
                operation.withdrawingRoute() == null
                        ? PublicationResult.PUBLISHED
                        : PublicationResult.RESTORED);
    }

    public static void completePlacement(PlacementOperation operation, PublicationResult result) {
        com.google.common.base.Preconditions.checkArgument(
                result == PublicationResult.PUBLISHED
                        || result == PublicationResult.RESTORED
                        || result == PublicationResult.WITHDRAWN,
                "invalid placement completion");
        operation.owner().data().getRequestScheduler().completePlacement(operation, result, null);
    }

    /**
     * Record an exact delivery result, then settle or continue the request outside its lock.
     */
    public void onDeliveryResult(DeliveryEvidence evidence) {
        Objects.requireNonNull(evidence, "evidence");
        var claim = evidence.delivery();
        var ctx = claim == null ? ((PreparationFailure) evidence).route().ctx() : claim.owner();
        if (ctx.data().getRequestScheduler() != this || !repository.isCurrent(ctx)) {
            return;
        }
        var routeFailure = ctx.unsentDeliveryFailure(evidence);
        if (routeFailure != null && onRouteFailure(routeFailure, evidence)) return;
        try (var accepted = taskPool.registerRequestWork(ctx)) {
            var observed = ctx.snapshot();
            boolean decodeConfirmed =
                    (evidence instanceof Prediction || evidence instanceof RouteDelivered)
                            && observed.delivery() != null
                            && observed.delivery().identity() == claim
                            && observed.route() == claim.route()
                            && !observed.engine().decodeAccepted()
                            && claim.route().decodeEndpoint() != null
                            && claim.route()
                                    .decodeEndpoint()
                                    .ownership(claim.route().decodeReservation())
                                    .confirmed();
            Snapshot after;
            Snapshot before;
            synchronized (ctx) {
                before = ctx.snapshot();
                if (decodeConfirmed
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
                                    evidence.nowMs()));
                }
                after = ctx.recordDeliveryResult(evidence);
            }
            if (after == null || after == before) {
                return;
            }
            try {
                advanceRequest(ctx);
            } catch (RuntimeException | Error failed) {
                // The accepted operation's owner has already recovered its remaining effects.
                taskPool.recordFailure(failed);
            }
        }
    }

    /**
     * All unsent route failures share the same withdrawal and cleanup, independent of the failing role.
     */
    public boolean onRouteFailure(
            RouteResult route, ContextResult.RouteFailureReason reason, String detail) {
        return onRouteFailure(new ContextResult.RouteFailure(route, reason, detail), null);
    }

    public static void failPlacement(PlacementOperation operation, Response failure) {
        Objects.requireNonNull(failure, "failure");
        com.google.common.base.Preconditions.checkArgument(
                !failure.isSuccess(), "placement failure requires an error response");
        operation
                .owner()
                .data()
                .getRequestScheduler()
                .completePlacement(operation, PublicationResult.FAILED, failure);
    }

    public Optional<PreemptionRegistration> tryClaim(
            DecodeResources.RequestReservation reservation, long attempt) {
        var ctx = repository.findActive(reservation.requestId());
        if (ctx == null || ctx.data().getRequestScheduler() != this) {
            return Optional.empty();
        }
        synchronized (ctx) {
            var route = ctx.snapshot().route();
            if (route == null) {
                return Optional.empty();
            }
            var prefill = route.role(RoleType.PREFILL);
            var target = new CancelTarget(prefill.ip(), prefill.grpcPort());
            var claim = ctx.beginPreemption(reservation, attempt, target);
            return Optional.ofNullable(claim);
        }
    }

    /**
     * Claim an unstarted request for preemption; finish or restore the returned handle.
     */
    public PlacementOperation beginPreemption(
            DecodeEndpoint endpoint,
            DecodeResources.RequestReservation victim,
            int incomingPriority) {
        if (repository.isClosed()) {
            return null;
        }
        RequestContext context = repository.findActive(victim.requestId());
        if (context == null
                || !allowsPreemption()
                || context.data().getRequestScheduler() != this) {
            return null;
        }
        synchronized (context) {
            RouteResult item = context.snapshot().route();
            // Context serializes the exact queue claim with delivery permit transfer.
            if (item == null
                    || item.priority() >= incomingPriority
                    || item.decodeEndpoint() != endpoint
                    || !Objects.equals(item.decodeReservation(), victim)
                    || !repository.isCurrent(context)
                    || context.data().getFuture().isDone()
                    || item.requestExpired(System.currentTimeMillis())
                    || !endpoint.ownership(victim).withdrawable()) {
                return null;
            }
            return context.beginPlacement(item);
        }
    }

    public boolean onPreemption(PreemptionRegistration attempt, PreemptionEvent event) {
        var ctx = attempt.owner();
        try (var accepted = taskPool.registerRequestWork(ctx)) {
            boolean applied = false, recorded = false, committed = false;
            Snapshot before;
            DecodeEndpoint endpoint = null;
            synchronized (ctx) {
                if (!repository.isCurrent(ctx) || ctx.data().getRequestScheduler() != this) {
                    return false;
                }
                before = ctx.snapshot();
                if (before.preemption() != null
                        && before.preemption().identity() == attempt
                        && (event.endpoint() == null
                                || before.route().decodeEndpoint() == event.endpoint()
                                        && Objects.equals(
                                                before.route().decodeReservation(),
                                                event.reservation()))) {
                    endpoint = event.endpoint();
                }
            }
            // Decode owns its ledger lock. Keep the resource mutation outside the request lock;
            // the second Context section below commits it only if this exact attempt survived.
            if (endpoint != null) {
                var target = endpoint;
                var mutation =
                        recover(
                                taskPool::recordFailure,
                                () ->
                                        target.updatePreemption(
                                                attempt.attemptToken(),
                                                DecodeResources.PreemptionUpdate.fenced(
                                                        event.reservation())));
                committed = mutation != null && mutation.status().applied();
            }
            synchronized (ctx) {
                var current = ctx.snapshot();
                if (current.preemption() != null
                        && current.preemption().identity() == attempt
                        && (endpoint == null
                                || current.route().decodeEndpoint() == endpoint
                                        && Objects.equals(
                                                current.route().decodeReservation(),
                                                event.reservation()))) {
                    var terminal =
                            !committed
                                    ? null
                                    : new EndEvidence(
                                            before.route(),
                                            EndKind.PRIORITY,
                                            null,
                                            false,
                                            0L,
                                            StrategyErrorType.PRIORITY_PREEMPTED,
                                            event.detail(),
                                            false,
                                            true);
                    recorded =
                            ctx.completePreemption(
                                    attempt,
                                    new PreemptionReport(
                                            event.endpoint() != null && !committed
                                                    ? PreemptionCancelPhase.CANCEL_UNKNOWN
                                                    : event.phase(),
                                            committed,
                                            committed,
                                            event.release(),
                                            terminal),
                                    System.currentTimeMillis());
                    applied = recorded && (endpoint == null || committed);
                }
                // A stale attempt must not discard a current delivery's cleanup proof.
                if (event.cleanup() != null) {
                    var previous = ctx.snapshot();
                    var next = ctx.recordDeliveryResult(event.cleanup());
                    recorded |= next != null && next != previous;
                }
            }
            if (committed) {
                var committedEndpoint = endpoint;
                recover(
                        taskPool::recordFailure,
                        () -> committedEndpoint.publishCapacity(DecodeEndpoint.CapacityEdge.BOTH));
            }
            if (recorded) {
                try {
                    advanceRequest(ctx);
                } catch (RuntimeException | Error failed) {
                    taskPool.recordFailure(failed);
                }
            }
            return applied;
        }
    }

    /**
     * Advance on the status caller; user callbacks and BATCH cleanup keep their own executors.
     */
    public void onWorkerRequestEvent(RequestContext ctx, WorkerRequestEvent evidence) {
        if (ctx == null || ctx.data().getRequestScheduler() != this) {
            return;
        }
        try (var accepted = taskPool.tryRegisterRequestWork(ctx)) {
            if (accepted == null) {
                return;
            }
            Snapshot before;
            Snapshot after;
            synchronized (ctx) {
                before = ctx.snapshot();
                after = ctx.recordWorkerRequestEvent(evidence);
            }
            if (after == null || after == before) return;
            if (!needsFollowUpActions(before, after)) return;
            accepted.runInline(() -> advanceRequest(ctx));
        } catch (Throwable failure) {
            taskPool.recordFailure(failure);
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
        try (var accepted = taskPool.tryRegisterRequestWork(ctx)) {
            if (accepted == null) return;
            Snapshot before;
            Snapshot after;
            synchronized (ctx) {
                before = ctx.snapshot();
                after = ctx.recordEndpointRetirement(retired);
            }
            if (after == null || after == before) return;
            if (!needsFollowUpActions(before, after)) return;
            accepted.runInline(() -> advanceRequest(ctx));
        } catch (Throwable failure) {
            taskPool.recordFailure(failure);
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
        try (var accepted = taskPool.tryRegisterRequestWork(ctx)) {
            if (accepted == null) {
                return;
            }
            Snapshot after;
            synchronized (ctx) {
                after = recordDeadline(ctx, exact, nowMs, null);
            }
            if (after == null) {
                return;
            }
            accepted.submit(
                    () -> {
                        var current = ctx.snapshot();
                        if (nowMs < exact.atMs()) {
                            expirationTimer.update(ctx, current.deadlines());
                            return;
                        }
                        advanceRequest(ctx);
                    });
        }
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
            failure = Failures.run(failure, endpoints::close);
            failure = Failures.run(failure, () -> dispatcher.close(active));
            failure = Failures.run(failure, expirationTimer::close);
            failure = Failures.run(failure, taskPool::awaitIdle);
            for (var owner : owners)
                failure = Failures.run(failure, owner::closeOutstandingAndTerminalize);
            failure = Failures.run(failure, this::closeExecutors);
            failure = Failures.run(failure, taskPool::throwIfFailed);
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
                        && !responseCompletions.isCompletingOnCurrentThread(),
                "scheduler shutdown cannot wait for its own request work or response callback");
    }

    protected abstract StrategyErrorType schedulingTimeoutError();

    protected int encoderRequestLimit() {
        return Integer.MAX_VALUE;
    }

    protected boolean allowsPreemption() {
        return false;
    }

    protected Map<String, Object> describeWait(
            RequestContext request, CancelReason reason, String message) {
        return null;
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
     * Keep the selected registrations local until publication. Every exit completes this attempt.
     */
    protected PlacementResult<RouteResult, PlacementKey> selectAndPublish(
            PlacementOperation operation,
            java.util.function.Supplier<PlacementResult<RouteResult, PlacementKey>> selection,
            QueueExecutionSettings queue) {
        RouteResult route = null;
        Throwable failure = null;
        try {
            var result = Objects.requireNonNull(selection.get(), "placement result");
            route = result.value();
            operation.owner().recordPlacementDiagnostics(operation, result.diagnostics());
            if (result.status() != PlacementResult.Status.SUCCESS) return result;
            var publishing = operation;
            operation = null;
            var publication = publishPlacement(route, publishing, queue);
            return queue == null && publication.status() == PlacementResult.Status.BLOCKED
                    ? PlacementResult.rejected(Response.error(StrategyErrorType.RESOURCE_EXHAUSTED))
                    : publication;
        } catch (RuntimeException | Error problem) {
            failure = problem;
            throw problem;
        } finally {
            finishSelection(route, operation, failure);
        }
    }

    protected void finishSelection(
            RouteResult route, PlacementOperation operation, Throwable primaryFailure) {
        if (operation == null) return;
        var cleanup =
                Failures.run(
                        null,
                        () ->
                                Failures.rethrow(
                                        finishPlacement(route, operation),
                                        "placement cleanup failed"));
        if (cleanup != null) taskPool.recordFailure(cleanup);
        if (primaryFailure != null) Failures.append(primaryFailure, cleanup);
        else Failures.rethrow(cleanup, "placement cleanup failed");
    }

    /**
     * Remove this exact request from the implementation's scheduling queues.
     */
    protected abstract void removeRequest(RequestContext request);

    /**
     * Publish the already registered route into its Prefill queue.
     * Exact queue membership determines publication even if notification fails.
     */
    PlacementResult<RouteResult, PlacementKey> enqueueRoute(
            RouteResult route, QueueExecutionSettings settings) {
        var context = route.ctx();
        checkState(
                route.decodeEndpoint() == null || route.decodeReservation() != null,
                "selected route requires Decode registration");
        context.data().setRouteSubmittedNanos(System.nanoTime());
        Throwable failure = null;
        boolean published;
        var operation = placementFor(route);
        if (operation == null) return PlacementResult.closed();
        if (!route.selectedEndpointsAvailable())
            return route.validateEndpoints(PlacementResult.closed());
        try {
            route.prefillEndpoint().enqueueRoute(route, settings);
        } catch (Throwable problem) {
            failure = problem;
        }
        // A queue implementation can throw after insertion. Retain that physical publication.
        published = route.prefillEndpoint().isQueued(route);
        boolean accepted = !published || context.publishRoute(operation, route);
        if (published && accepted) {
            failure =
                    Failures.run(
                            failure,
                            () -> route.prefillEndpoint().signal(PrefillEndpoint.Signal.INPUT));
            try {
                deliveryReporter.reportLatency(
                        DeliveryMetricsReporter.Latency.ROUTE_SUBMIT,
                        route.prefillEndpoint().getStatus().getRole().name(),
                        route.prefillEndpoint().getIp(),
                        System.currentTimeMillis() - context.data().getStartTime());
            } catch (Throwable telemetryFailure) {
                Logger.warn("Failed to record route-submit telemetry", telemetryFailure);
            }
        }
        Failures.rethrow(failure, "route publication failed");
        if (!accepted) return PlacementResult.closed();
        return published
                ? PlacementResult.success(route)
                : route.validateEndpoints(
                        PlacementResult.blocked(route.placementKey(RoleType.PREFILL)));
    }

    PlacementResult<ContextResult.RouteDelivered, PlacementKey> commitDirectRoute(
            RouteResult route) {
        var context = route.ctx();
        if (route.encoderEndpoint() != null) return commitEncoderRoute(route);
        checkState(
                route.decodeEndpoint() == null || route.decodeReservation() != null,
                "selected route requires Decode registration");
        Delivery.Member member = null;
        boolean delivered = false;
        Throwable failure = null;
        try {
            var attempt = Delivery.Member.prepare(route, dispatcher.decodeAdmissionCapacity);
            if (!attempt.accepted()) {
                if (attempt.boundary().unavailable())
                    return PlacementResult.blocked(route.placementKey(RoleType.DECODE));
                if (attempt.boundary().status() == CapacityBoundary.Status.FAILED)
                    throw new IllegalStateException(
                            "route admission failed", attempt.boundary().cause());
                return PlacementResult.rejected(
                        Response.error(StrategyErrorType.RESOURCE_EXHAUSTED));
            }
            member = attempt.value();
            org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot committed;
            var operation = placementFor(route);
            if (operation == null) return PlacementResult.closed();
            committed = route.prefillEndpoint().commitDirect(route);
            if (committed == null)
                return route.validateEndpoints(
                        PlacementResult.rejected(
                                Response.error(StrategyErrorType.RESOURCE_EXHAUSTED)));
            if (!context.publishRoute(operation, route)) return PlacementResult.closed();
            var preceding = committed.calculate();
            if (!route.selectedEndpointsAvailable())
                return route.validateEndpoints(PlacementResult.closed());
            var claim = commitDelivery(member, Delivery.SendMode.ROUTE_DECISION, 0L);
            if (claim == null) return PlacementResult.closed();
            var result =
                    PlacementResult.<ContextResult.RouteDelivered, PlacementKey>success(
                            DeliveryDispatcher.routeDelivered(
                                    claim, preceding, route.prefillWorkMs()));
            delivered = true;
            return result;
        } catch (RuntimeException | Error problem) {
            failure = problem;
            throw problem;
        } finally {
            var cleanup = member == null ? null : Failures.close(member);
            if (failure != null) Failures.append(failure, cleanup);
            else if (delivered) {
                if (cleanup != null) taskPool.recordFailure(cleanup);
            } else Failures.rethrow(cleanup, "DIRECT preparation cleanup failed");
        }
    }

    /**
     * Commit one delivery under the request lock: validate, transfer Decode capacity, then
     * record the send boundary. BATCH payloads must be complete before this call; there is
     * no separately published claim waiting for a later begin-send transition.
     */
    Delivery.Member commitDelivery(Delivery.Member member, Delivery.SendMode kind, long batchId) {
        var route = member.route();
        var ctx = member.owner();
        var permit = member.decode();
        DecodeResources.ResourceChange<DecodeResources.EngineDispatchPermitTransferStatus>
                transfer = null;
        boolean transferred = false;
        try {
            synchronized (ctx) {
                if (!repository.isCurrent(ctx)
                        || ctx.data().getRequestScheduler() != this
                        || ctx.data().getFuture().isDone()) return null;
                long nowMs = System.currentTimeMillis();
                var before = ctx.snapshot();
                boolean inactive = nowMs >= before.inactivityExpiresAtMs();
                boolean expired = nowMs >= ctx.data().getRequestExpiresAtMs();
                if (before.preparedDelivery()
                        && before.cancellation() == null
                        && (inactive || expired)) {
                    String detail =
                            inactive
                                    ? "REQUEST_INACTIVE: no matching Engine request status before inactivity timeout"
                                    : "request scheduling deadline exceeded before delivery";
                    var deadline =
                            inactive
                                    ? before.deadlines().inactivity()
                                    : before.deadlines().request();
                    if (deadline != null) recordDeadline(ctx, deadline, nowMs, detail);
                }
                if (ctx.validateDelivery(route, nowMs) && route.selectedEndpointsAvailable()) {
                    member.initialize(kind, batchId);
                    if (permit != null) {
                        com.google.common.base.Preconditions.checkArgument(
                                permit.belongsTo(route.decodeEndpoint(), route.decodeReservation()),
                                "foreign Decode dispatch permit");
                        transfer = permit.dispatch();
                        checkState(
                                transfer.status()
                                        == DecodeResources.EngineDispatchPermitTransferStatus
                                                .TRANSFERRED,
                                "Decode dispatch ownership lost");
                    }
                    transferred = true;
                    ctx.recordDeliveryStarted(member, nowMs);
                    return member;
                }
            }
        } catch (RuntimeException | Error failure) {
            if (transferred) {
                var recorded = ctx.snapshot().delivery();
                if (recorded != null && recorded.identity() == member) {
                    if (member.kind() == Delivery.SendMode.BATCH_ENQUEUE)
                        onDeliveryResult(
                                new ContextResult.TransportResult(
                                        member,
                                        Delivery.Result.notSent(failure),
                                        System.currentTimeMillis()));
                    else
                        onDeliveryResult(
                                new PreparationFailure(
                                        member,
                                        route,
                                        failure.getMessage(),
                                        System.currentTimeMillis()));
                } else preparationFailed(route, failure.getMessage());
            }
            throw failure;
        } finally {
            if (transfer != null && transfer.capacityReleased()) {
                org.flexlb.util.Failures.recover(
                        taskPool::recordFailure,
                        () ->
                                route.decodeEndpoint()
                                        .publishCapacity(DecodeEndpoint.CapacityEdge.DISPATCH));
            }
        }
        if (!route.selectedEndpointsAvailable()) {
            onRouteFailure(
                    route,
                    ContextResult.RouteFailureReason.ENDPOINT_UNAVAILABLE,
                    "Selected Endpoint unavailable during delivery commit");
        }
        if (ctx.snapshot().pendingEnd() != null) closeRequest(ctx, null);
        else if (System.currentTimeMillis() >= ctx.data().getRequestExpiresAtMs())
            cancel(ctx, 0L, CancelReason.DEADLINE_EXCEEDED);
        return null;
    }

    static void preparationFailed(RouteResult route, String detail) {
        route.ctx()
                .data()
                .getRequestScheduler()
                .onDeliveryResult(
                        new PreparationFailure(null, route, detail, System.currentTimeMillis()));
    }

    /**
     * Own closing through completion or a recorded wait, including recovery after an effect fails.
     */
    void closeRequest(RequestContext ctx, EndEvidence evidence) {
        if (ctx.data().getRequestScheduler() != this || !repository.isCurrent(ctx)) {
            return;
        }
        RequestTaskPool.WorkRegistration execution = null;
        Registration publication = null;
        ResponseCandidate candidate = null;
        Response response = null;
        boolean handedOff = false;
        try {
            synchronized (ctx) {
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
                execution = taskPool.registerRequestWork(ctx);
                if (before.outcome() == null
                        && ctx.readyToClose(ending)
                        && before.selectedResponse() == null
                        && !ctx.data().getFuture().isDone()) {
                    publication =
                            Registration.reserve(responseCompletions, ctx, taskPool::recordFailure);
                }
                var after = ctx.beginClose(evidence, System.currentTimeMillis());
                if (publication != null && after.outcome() != null) {
                    var outcome = after.outcome();
                    candidate =
                            new ResponseCandidate(
                                    ctx, ResponseKind.TERMINAL, System.currentTimeMillis());
                    if (outcome.kind() == EndKind.DELIVERY_FAILURE) {
                        ctx.selectResponse(candidate);
                    }
                    boolean batched =
                            after.delivery() != null
                                    && after.delivery().identity().kind()
                                            == Delivery.SendMode.BATCH_ENQUEUE;
                    response = response(outcome, before.route(), batched);
                } else if (publication != null) {
                    publication.close();
                    publication = null;
                }
            }
            handedOff = true;
            submitClose(ctx, execution, publication, candidate, response);
        } catch (RuntimeException | Error failed) {
            if (!handedOff) {
                try {
                    publicationFailed(ctx, publication, candidate, failed);
                    Failures.run(failed, () -> scheduleClose(ctx));
                } finally {
                    if (publication != null) publication.close();
                }
            }
            throw failed;
        } finally {
            if (execution != null) execution.close();
        }
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
        try (var work = taskPool.registerRequestWork(ctx)) {
            submitClose(ctx, work, null, null, null);
        }
    }

    void completePlacement(PlacementOperation handle, PublicationResult result, Response failure) {
        var ctx = handle.owner();
        checkState(ctx.data().getRequestScheduler() == this, "foreign placement operation");
        if (!ctx.beginPlacementCompletion(handle)) return;
        var previous = handle.withdrawingRoute();
        var routeFailure = ctx.snapshot().routeFailure();
        var withdrawingDelivery = previous == null ? null : ctx.snapshot().delivery();
        Throwable problem = null;
        if (result == PublicationResult.WITHDRAWN) {
            try {
                checkState(previous != null, "withdrawal requires an exact old route");
                if (routeFailure == null) {
                    checkState(
                            previous.prefillEndpoint().removeQueued(previous),
                            "withdrawn route is no longer queued: %s",
                            ctx.data().getRequestId());
                }
                var delivery = ctx.snapshot().delivery();
                if (delivery != null && delivery.identity().route() == previous)
                    delivery.identity().close();
                Failures.rethrow(
                        releaseRouteResources(
                                previous, DecodeResources.ReleaseReason.UNSENT_ROUTE_ROLLBACK),
                        "route withdrawal cleanup failed");
            } catch (Throwable detachFailure) {
                problem = detachFailure;
                result = PublicationResult.FAILED;
                failure =
                        buildErrorResponse(
                                StrategyErrorType.DISPATCH_FAILED,
                                "queued route withdrawal failed");
            }
        }
        Snapshot after;
        synchronized (ctx) {
            after = ctx.completePlacement(handle, result, System.currentTimeMillis());
        }
        checkState(after != null, "placement completion lost its exact request ownership");
        if (result == PublicationResult.WITHDRAWN && withdrawingDelivery != null)
            withdrawingDelivery.identity().complete();
        if (previous != null && after.route() == previous && failure == null) {
            problem =
                    Failures.run(
                            problem,
                            () -> previous.prefillEndpoint().signal(PrefillEndpoint.Signal.INPUT));
        }
        EndEvidence end = after.pendingEnd();
        if ((end == null
                        || failure != null
                                && end.kind() == EndKind.SHUTDOWN
                                && after.cancellation() == null)
                && failure != null) {
            end = EndEvidence.publicationFailure(after.route(), failure);
        }
        if (end != null) {
            var actual = end;
            problem = Failures.run(problem, () -> closeRequest(ctx, actual));
        } else if (result == PublicationResult.WITHDRAWN
                && after.phase() == Phase.OPEN
                && !ctx.data().getFuture().isDone()) {
            try {
                if (System.currentTimeMillis() >= ctx.data().getRequestExpiresAtMs()) {
                    cancel(ctx, 0L, CancelReason.DEADLINE_EXCEEDED);
                } else {
                    onRequest(ctx);
                }
                expirationTimer.update(ctx, ctx.snapshot().deadlines());
            } catch (Throwable requeueFailure) {
                problem = Failures.append(problem, requeueFailure);
                var withdrawn = EndEvidence.withdrawnFailure("queued route requeue failed");
                problem = Failures.run(problem, () -> closeRequest(ctx, withdrawn));
            }
        } else {
            problem = Failures.run(problem, () -> advanceRequest(ctx));
        }
        if (problem != null) {
            taskPool.recordFailure(problem);
        }
        Failures.rethrow(problem, "placement completion failed");
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

    final boolean tryCompleteFromExternal(
            RequestContext ctx,
            RequestFuture.Completion completion,
            Response response,
            Throwable failure,
            boolean synchronous) {
        if (Thread.holdsLock(ctx)) {
            throw new IllegalStateException(
                    "Future callbacks require the Context lock to be released");
        }
        if (ctx.data().getRequestScheduler() != this || !repository.isCurrent(ctx)) return false;
        try (var accepted = taskPool.tryRegisterRequestWork(ctx)) {
            if (accepted == null) return false;
            Registration publication = null;
            try {
                ResponseCandidate candidate;
                synchronized (ctx) {
                    var before = ctx.snapshot();
                    boolean cancelled = completion == RequestFuture.Completion.CANCELLATION;
                    if (before.phase() != Phase.OPEN
                            || before.selectedResponse() != null
                            || ctx.data().getFuture().isDone()
                            || before.cancellation() != null
                                    && (!cancelled
                                            || before.cancellation().reason()
                                                    == CancelReason.DEADLINE_EXCEEDED)) {
                        return false;
                    }
                    boolean success =
                            completion == RESPONSE && (response == null || response.isSuccess());
                    if (success
                            || before.delivery() != null
                            || before.engine().decodeAccepted()
                            || before.preemption() != null
                            || before.placement() != null && !cancelled) {
                        return false;
                    }
                    publication = new Registration(responseCompletions, ctx);
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
                        return false;
                    }
                    if (!cancelled) {
                        var end =
                                ctx.beginClose(
                                        EndEvidence.externalFailure(
                                                before.route(), response, failure),
                                        System.currentTimeMillis());
                        if (end.outcome() == null) return false;
                    }
                    candidate =
                            new ResponseCandidate(
                                    ctx,
                                    cancelled ? ResponseKind.CANCELLATION : ResponseKind.TERMINAL,
                                    System.currentTimeMillis());
                    ctx.selectResponse(candidate);
                }
                try {
                    Throwable cleanup = Failures.run(null, () -> closeRequest(ctx, null));
                    if (cleanup != null) taskPool.recordFailure(cleanup);
                    return publishResponse(publication, candidate, response, failure, synchronous);
                } catch (RuntimeException | Error failed) {
                    retryPublication(
                            publication, candidate, response, failure, synchronous, failed);
                    throw failed;
                }
            } finally {
                if (publication != null) publication.close();
            }
        }
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
        Throwable failure = Failures.run(null, taskPool::awaitIdle);
        failure = Failures.run(failure, dispatcher::awaitStops);
        failure = Failures.run(failure, taskPool::close);
        failure = Failures.run(failure, responseCompletions::close);
        Failures.rethrow(failure, "request executors failed to close");
    }

    private PlacementResult<ContextResult.RouteDelivered, PlacementKey> commitEncoderRoute(
            RouteResult route) {
        var context = route.ctx();

        var operation = placementFor(route);
        if (operation == null) return PlacementResult.closed();
        if (!route.encoderEndpoint().commit(route, encoderRequestLimit()))
            return PlacementResult.blocked(
                    new PlacementKey(
                            RoleType.ENCODER,
                            route.encoderEndpoint().group(),
                            route.encoderEndpoint().ipPort()));
        if (!context.publishRoute(operation, route)) return PlacementResult.closed();
        if (!route.selectedEndpointsAvailable())
            return route.validateEndpoints(PlacementResult.closed());
        var claim =
                commitDelivery(
                        new Delivery.Member(route, null), Delivery.SendMode.ROUTE_DECISION, 0);
        if (claim == null) return PlacementResult.closed();
        long now = System.currentTimeMillis();
        return PlacementResult.success(
                new ContextResult.RouteDelivered(
                        claim, context.snapshot().inactivityExpiresAtMs(), now));
    }

    private boolean onRouteFailure(ContextResult.RouteFailure failure, DeliveryEvidence evidence) {
        var route = failure.route();
        var ctx = route.ctx();
        if (ctx.data().getRequestScheduler() != this
                || !repository.isCurrent(ctx)
                || !shouldRetryRouteFailure(failure)) return false;
        try (var work = taskPool.tryRegisterRequestWork(ctx)) {
            if (work == null || !ctx.failUnsentRoute(failure, evidence)) return false;
            withdrawFailedRoute(ctx);
            return true;
        }
    }

    private void withdrawFailedRoute(RequestContext ctx) {
        if (repository.isClosed()) return;
        var operation = ctx.beginRouteWithdrawal();
        if (operation == null) return;
        completePlacement(operation, PublicationResult.WITHDRAWN);
    }

    private static boolean needsFollowUpActions(Snapshot before, Snapshot after) {
        return after.preemption() != null
                || after.pendingEnd() != null
                || after.phase() != Phase.OPEN
                || after.canConfirmDelivery() && !after.delivery().acknowledged()
                || before.deadlines() != after.deadlines();
    }

    /**
     * Select one effect owner after recording facts; that owner also handles its failures.
     */
    private void advanceRequest(RequestContext ctx) {
        var state = ctx.snapshot();
        if (state.routeFailure() != null
                && state.placement() == null
                && state.preemption() == null
                && state.phase() == Phase.OPEN
                && state.cancellation() == null
                && state.pendingEnd() == null) {
            withdrawFailedRoute(ctx);
            state = ctx.snapshot();
        }
        Throwable failure = null;
        if (state.preemption() != null) {
            try {
                reconcilePreemption(ctx);
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
            if (current.phase() == Phase.OPEN
                    && expirationTimer.needsUpdate(ctx, current.deadlines())) {
                recover(
                        taskPool::recordFailure,
                        () -> expirationTimer.update(ctx, current.deadlines()));
            }
        } catch (RuntimeException | Error failed) {
            failure = Failures.append(failure, failed);
        }
        Failures.rethrow(failure, "request continuation failed");
    }

    /**
     * The exact reconciliation owns ledger retries and its post-commit capacity/resumption effects.
     */
    private void reconcilePreemption(RequestContext ctx) {
        ContextResult.Snapshot current;
        PreemptionRegistration attempt;
        DecodeEndpoint decode;
        boolean capacityChanged;
        EndEvidence terminal;
        boolean resumeDelivery;
        synchronized (ctx) {
            current = ctx.snapshot();
            if (current.preemption() == null) {
                return;
            }
            var claim = ctx.canReconcilePreemption(current.preemption().identity());
            if (!claim) {
                return;
            }
            attempt = current.preemption().identity();
            decode = current.route().decodeEndpoint();
            terminal = current.preemption().pendingTerminal();
            resumeDelivery =
                    terminal == null
                            && current.canConfirmDelivery()
                            && !current.delivery().acknowledged();
            capacityChanged =
                    decode != null && !(terminal != null && current.close().decodeSettled());
        }
        // Reconciliation is a two-phase handoff: the ledger mutation may block on Decode state,
        // while the Context lock is reserved only for the exact commit.
        var mutation =
                !capacityChanged
                        ? null
                        : recover(
                                taskPool::recordFailure,
                                () ->
                                        decode.updatePreemption(
                                                attempt.attemptToken(),
                                                terminal == null
                                                        ? DecodeResources.PreemptionUpdate.active(
                                                                attempt.reservation())
                                                        : terminal.kind() == EndKind.PRIORITY
                                                                ? DecodeResources.PreemptionUpdate
                                                                        .canceled(
                                                                                attempt
                                                                                        .reservation())
                                                                : DecodeResources.PreemptionUpdate
                                                                        .finished(
                                                                                attempt
                                                                                        .reservation())));
        boolean committed = !capacityChanged || mutation != null && mutation.status().applied();
        boolean recorded;
        synchronized (ctx) {
            recorded =
                    ctx.completePreemption(
                            attempt,
                            new PreemptionReport(
                                    null,
                                    committed,
                                    committed && terminal != null,
                                    false,
                                    terminal),
                            System.currentTimeMillis());
            if (!committed) {
                return;
            }
        }
        if (capacityChanged) {
            recover(
                    taskPool::recordFailure,
                    () -> decode.publishCapacity(DecodeEndpoint.CapacityEdge.BOTH));
        }
        if (recorded && resumeDelivery) {
            recover(
                    taskPool::recordFailure,
                    () ->
                            attempt.complete(
                                    new VictimResolution(
                                            ctx.data().getRequestId(),
                                            VictimResolution.Outcome.DELIVERY_RESUMED)));
        }
    }

    /**
     * Confirm delivery or start settlement; deadline synchronization follows the resulting state.
     */
    private void confirmDelivery(RequestContext ctx) {
        Registration publication = null;
        ResponseCandidate candidate = null;
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
                        publication =
                                Registration.reserve(
                                        responseCompletions, ctx, taskPool::recordFailure);
                    }
                    var acknowledged = ctx.confirmDelivery(System.currentTimeMillis());
                    if (publication != null && !acknowledged) return;
                    if (publication != null) {
                        ctx.data().discardGenerateInput();
                        candidate =
                                new ResponseCandidate(
                                        ctx, ResponseKind.ACK, System.currentTimeMillis());
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
                            taskPool::recordFailure,
                            () -> expirationTimer.update(ctx, ctx.snapshot().deadlines()));
                }
                return;
            }
            var claim = ctx.snapshot().delivery().identity();
            var response =
                    claim.route().successResponse(claim.kind() == Delivery.SendMode.BATCH_ENQUEUE);
            Failures.rethrow(
                    recover(
                            taskPool::recordFailure,
                            () -> expirationTimer.update(ctx, ctx.snapshot().deadlines())),
                    "delivery deadline installation failed");
            publishResponse(publication, candidate, response, null, false);
        } catch (RuntimeException | Error failed) {
            publicationFailed(ctx, publication, candidate, failed);
            throw failed;
        } finally {
            if (publication != null) {
                publication.close();
            }
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
                                && (state.delivery() == null || state.placement() != null);
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
                                : state.placement() != null
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

    private boolean publishResponse(
            Registration publication,
            ResponseCandidate candidate,
            Response response,
            Throwable failure,
            boolean synchronous) {
        var ctx = publication.owner;
        java.util.function.BooleanSupplier task =
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
                    var completion =
                            failure != null
                                    ? RequestFuture.Completion.FAILURE
                                    : candidate.kind() == ResponseKind.CANCELLATION
                                            ? RequestFuture.Completion.CANCELLATION
                                            : RESPONSE;
                    return ctx.selectResponse(candidate)
                            && publication.complete(completion, response, failure);
                };
        if (synchronous) return publication.completeNow(task);
        publication.submit(task);
        return true;
    }

    private void publicationFailed(
            RequestContext ctx,
            Registration publication,
            ResponseCandidate candidate,
            Throwable failed) {
        if (ctx.snapshot().phase() == Phase.OPEN) {
            recover(
                    taskPool::recordFailure,
                    () -> expirationTimer.update(ctx, ctx.snapshot().deadlines()));
        }
        if (publication != null) {
            retryPublication(publication, candidate, null, failed, false, failed);
        }
    }

    private void retryPublication(
            Registration publication,
            ResponseCandidate candidate,
            Response response,
            Throwable failure,
            boolean synchronous,
            Throwable primary) {
        if (publication.hasCompletionOwner() || candidate == null) return;
        Failures.run(
                primary,
                () -> {
                    try (var fallback = new Registration(responseCompletions, publication.owner)) {
                        publishResponse(fallback, candidate, response, failure, synchronous);
                    }
                });
    }

    /**
     * Hand off one complete close operation; rejected work retains its unfinished cleanup state.
     */
    private void submitClose(
            RequestContext ctx,
            RequestTaskPool.WorkRegistration work,
            Registration publication,
            ResponseCandidate candidate,
            Response response) {
        var state = ctx.snapshot();
        boolean local = state.outcome() == null || state.outcome().kind() != EndKind.RETIREMENT;
        local &=
                state.delivery() == null
                        || state.delivery().identity().kind() != Delivery.SendMode.BATCH_ENQUEUE;
        boolean inactivity =
                state.outcome() != null && state.outcome().kind() == EndKind.INACTIVITY;
        boolean inline = local && !inactivity;
        // Preserve synchronous queue-removal errors on ordinary callers. A timer handoff
        // removes the queue entry inside its counted job instead of blocking the timer.
        Throwable queueFailure = !inactivity ? Failures.run(null, () -> removeRequest(ctx)) : null;
        Registration queuedPublication = publication;
        if (publication != null && !local && !inactivity) {
            // A known remote outcome can respond while its independent physical cleanup waits.
            try {
                publishResponse(publication, candidate, response, null, false);
            } catch (RuntimeException | Error failure) {
                publicationFailed(ctx, publication, candidate, failure);
                queueFailure = Failures.append(queueFailure, failure);
            } finally {
                publication.close();
            }
            queuedPublication = null;
        }
        final Registration responseOwner = queuedPublication;
        Runnable close = () -> runClose(ctx, responseOwner, candidate, response);
        java.util.function.Consumer<Throwable> rejected =
                failure -> {
                    try {
                        publicationFailed(ctx, responseOwner, candidate, failure);
                    } finally {
                        if (responseOwner != null) responseOwner.close();
                    }
                };
        try {
            if (responseOwner == null) work.submitClose(close, inline, rejected);
            else work.submitCloseResponse(close, inline, rejected);
        } catch (RuntimeException | Error failure) {
            Failures.append(failure, queueFailure);
            throw failure;
        }
        if (queueFailure != null) taskPool.recordFailure(queueFailure);
        Failures.rethrow(queueFailure, "request queue removal failed");
    }

    private void runClose(
            RequestContext ctx,
            Registration publication,
            ResponseCandidate candidate,
            Response response) {
        try {
            var state = ctx.snapshot();
            Throwable failure =
                    state.outcome() != null && state.outcome().kind() == EndKind.INACTIVITY
                            ? Failures.run(null, () -> removeRequest(ctx))
                            : null;
            boolean releaseBeforeResponse =
                    (state.outcome() == null || state.outcome().kind() != EndKind.RETIREMENT)
                            && (state.delivery() == null
                                    || state.delivery().identity().kind()
                                            != Delivery.SendMode.BATCH_ENQUEUE);
            if (releaseBeforeResponse) failure = Failures.run(failure, () -> cleanupRequest(ctx));
            if (publication != null) {
                failure =
                        Failures.run(
                                failure,
                                () ->
                                        publishResponse(
                                                publication, candidate, response, null, false));
            }
            if (!releaseBeforeResponse) failure = Failures.run(failure, () -> cleanupRequest(ctx));
            if (failure != null) taskPool.recordFailure(failure);
            Failures.rethrow(failure, "request close failed");
        } catch (RuntimeException | Error failed) {
            publicationFailed(ctx, publication, candidate, failed);
            throw failed;
        } finally {
            if (publication != null) publication.close();
        }
    }

    private void cleanupRequest(RequestContext ctx) {
        Throwable failure = null;
        for (int pass = 0; pass < 3 && ctx.snapshot().phase() == Phase.ENDING; pass++) {
            Throwable problem = notifyRequestEnd(ctx);
            var reason = ctx.releaseReason();
            if (reason != null) {
                var route = ctx.snapshot().route();
                if (route == null) {
                    ctx.recordResourcesReleased(null, true, true, System.currentTimeMillis());
                } else {
                    problem = Failures.append(problem, releaseRouteResources(route, reason));
                }
            }
            failure = Failures.append(failure, problem);
            var after = ctx.snapshot();
            boolean notificationReady =
                    after.preemption() != null
                            && after.preemption().finished()
                            && !after.close().notificationsDone();
            if (problem == null && !notificationReady) {
                break;
            }
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

    private Throwable notifyRequestEnd(RequestContext ctx) {
        var deadlines = ctx.snapshot().deadlines();
        Throwable timerFailure =
                expirationTimer.needsUpdate(ctx, deadlines)
                        ? Failures.run(null, () -> expirationTimer.update(ctx, deadlines))
                        : null;
        var state = ctx.snapshot();
        if (state.phase() != Phase.ENDING) {
            return timerFailure;
        }
        boolean timersDone = !expirationTimer.needsUpdate(ctx, deadlines);
        Throwable failure = null;
        var delivery = state.delivery();
        if (!state.close().notificationsAttempted()
                && delivery != null
                && (delivery.abandonment() != null || state.cancellation() != null)) {
            failure = Failures.run(null, () -> dispatcher.stop(delivery.identity()));
            if (failure != null) {
                ctx.recordCloseNotifications(false, false, System.currentTimeMillis());
                return Failures.append(timerFailure, failure);
            }
        }
        var preemption = state.preemption();
        if (timersDone
                && !state.close().notificationsDone()
                && preemption != null
                && preemption.finished()) {
            failure =
                    Failures.run(
                            null,
                            () ->
                                    preemption
                                            .identity()
                                            .complete(
                                                    new VictimResolution(
                                                            ctx.data().getRequestId(),
                                                            VictimResolution.Outcome.REQUEST_END)));
        }
        ctx.recordCloseNotifications(
                true,
                failure == null && timersDone && (preemption == null || preemption.finished()),
                System.currentTimeMillis());
        return Failures.append(timerFailure, failure);
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

    public record PreemptionEvent(
            PreemptionCancelPhase phase,
            boolean release,
            DecodeEndpoint endpoint,
            DecodeResources.RequestReservation reservation,
            String detail,
            CleanupAcknowledgement cleanup) {
        public PreemptionEvent(
                PreemptionCancelPhase phase,
                boolean release,
                DecodeEndpoint endpoint,
                DecodeResources.RequestReservation reservation,
                String detail) {
            this(phase, release, endpoint, reservation, detail, null);
        }

        PreemptionRegistration.Reply progress(boolean applied) {
            return !applied
                    ? PreemptionRegistration.Reply.UNKNOWN
                    : phase == PreemptionCancelPhase.NOT_FOUND_STALE
                            ? PreemptionRegistration.Reply.NOT_FOUND
                            : PreemptionRegistration.Reply.WAITING;
        }
    }
}
