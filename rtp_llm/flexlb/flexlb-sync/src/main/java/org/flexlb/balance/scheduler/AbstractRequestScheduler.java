package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
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

import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import static com.google.common.base.Preconditions.checkState;
import static org.flexlb.balance.scheduler.ContextResult.Abandonment;
import static org.flexlb.balance.scheduler.ContextResult.Cancellation;
import static org.flexlb.balance.scheduler.ContextResult.CancellationFacts;
import static org.flexlb.balance.scheduler.ContextResult.CleanupAcknowledgement;
import static org.flexlb.balance.scheduler.ContextResult.CloseProgress;
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
 */
public abstract class AbstractRequestScheduler implements RequestScheduler {

    /**
     * Remove this exact request from the implementation's scheduling queues.
     */
    protected abstract void removeRequest(RequestContext request);

    protected final RequestRepository requests;
    protected final RequestTaskPool tasks;
    protected final ExpirationTimer expirationTimer;

    protected final DeliveryDispatcher dispatcher;
    protected final ResponseCompletionExecutor responseCompletions;
    protected final DeliveryMetricsReporter deliveryReporter;

    protected final TheoryCacheHitReporter cacheTrace;

    private final Object admissionLock = new Object();
    private boolean admitting = true;
    private long activeAdmissions;

    boolean registerAdmission() {
        synchronized (admissionLock) {
            if (!admitting || requests.isClosed()) {
                return false;
            }
            activeAdmissions++;
            return true;
        }
    }

    void finishAdmission() {
        synchronized (admissionLock) {
            checkState(activeAdmissions > 0L, "admission ownership underflow");
            if (--activeAdmissions == 0L) {
                admissionLock.notifyAll();
            }
        }
    }

    public void stopAdmissionsAndAwaitCompletion() {
        boolean interrupted = false;
        synchronized (admissionLock) {
            admitting = false;
            while (activeAdmissions != 0L) {
                try {
                    admissionLock.wait();
                } catch (InterruptedException ignored) {
                    interrupted = true;
                }
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    public void awaitDeliveryCleanup(boolean batchOnly) {
        var active = requests.snapshotActive();
        var pending =
                active.stream()
                        .map(context -> context.snapshot().delivery())
                        .filter(Objects::nonNull)
                        .map(ContextResult.DeliveryFacts::identity)
                        .filter(Objects::nonNull)
                        .filter(
                                delivery ->
                                        !batchOnly
                                                || delivery.kind()
                                                        == Delivery.SendMode.BATCH_ENQUEUE)
                        .map(delivery -> delivery.completion().toCompletableFuture())
                        .toArray(CompletableFuture[]::new);
        try {
            CompletableFuture.allOf(pending)
                    .get(DeliveryDispatcher.CLEANUP_TIMEOUT_MS + 1_000L, TimeUnit.MILLISECONDS);
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(
                    "Interrupted waiting for delivery cleanup", interrupted);
        } catch (ExecutionException | TimeoutException failure) {
            throw new IllegalStateException(
                    "Unsettled deliveries at shutdown: "
                            + active.stream()
                                    .map(context -> context.data().getRequestId())
                                    .toList(),
                    failure);
        }
    }

    protected AbstractRequestScheduler(
            RequestRepository requests,
            RequestTaskPool tasks,
            ExpirationTimer deadlines,
            DeliveryDispatcher dispatcher,
            ResponseCompletionExecutor responseCompletions,
            DeliveryMetricsReporter deliveryReporter,
            TheoryCacheHitReporter cacheTrace) {
        this.requests = Objects.requireNonNull(requests);
        this.tasks = Objects.requireNonNull(tasks);
        this.expirationTimer = Objects.requireNonNull(deadlines);

        this.dispatcher = Objects.requireNonNull(dispatcher);
        this.responseCompletions = Objects.requireNonNull(responseCompletions);
        this.deliveryReporter = Objects.requireNonNull(deliveryReporter);
        this.cacheTrace = Objects.requireNonNull(cacheTrace);
    }

    private Map<String, Object> describeWait(
            RequestContext request, CancelReason reason, String message) {
        if (reason != CancelReason.DEADLINE_EXCEEDED
                || request.data().schedulingTimeoutError() != StrategyErrorType.RESOURCE_EXHAUSTED)
            return null;
        var state = request.snapshot();
        if (state.delivery() != null) return Map.of("cause", message);
        return state.route() == null || state.route().encoderEndpoint() != null
                ? request.data().getSchedulingDiagnostics()
                : state.route().prefillEndpoint().getLatestQueueWaitSnapshot();
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
            if (requests.isStopping() || requests.isClosed()) {
                return CompletableFuture.completedFuture(
                        Response.buildErrorResponse(
                                StrategyErrorType.DISPATCH_FAILED,
                                "request scheduler is not accepting this request configuration"));
            }
            long expires = data.getRequestExpiresAtMs();
            if (expires <= 0L || System.currentTimeMillis() >= expires) {
                var rejected =
                        CompletableFuture.completedFuture(
                                Response.buildErrorResponse(
                                        data.schedulingTimeoutError(),
                                        "request scheduling deadline has expired before placement"));
                data.setFuture(rejected);
                return rejected;
            }
            var future =
                    new RequestFuture(
                            (kind, response, failure) ->
                                    completeExternal(ctx, kind, response, failure, true));
            var result = requests.register(ctx, this, future);
            if (result == RequestRepository.RegistrationResult.INPUT_REQUIRED) {
                return CompletableFuture.completedFuture(
                        Response.buildErrorResponse(
                                StrategyErrorType.GENERATE_INPUT_REQUIRED,
                                "missing serialized generate_input for batch dispatch"));
            }
            if (result != RequestRepository.RegistrationResult.REGISTERED) {
                return CompletableFuture.completedFuture(
                        Response.buildErrorResponse(
                                result == RequestRepository.RegistrationResult.CLOSED
                                        ? StrategyErrorType.DISPATCH_FAILED
                                        : StrategyErrorType.INVALID_REQUEST,
                                result == RequestRepository.RegistrationResult.DUPLICATE_ID
                                        ? "duplicate request_id: " + data.getRequestId()
                                        : "request registration rejected: " + result));
            }
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
            if (requests.isClosed()) {
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
        var ctx = requests.findActive(id, phase);
        if (ctx != null) {
            if (ctx.data().getRequestScheduler() != this) {
                return null;
            }
            var result = cancel(ctx, batchId, reason);
            if (result != null) {
                return result;
            }
        }
        var terminal = requests.findTerminal(id, phase);
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
        if (ctx == null || ctx.data().getRequestScheduler() != this || !requests.isCurrent(ctx)) {
            return null;
        }
        try (var accepted = tasks.tryRegisterRequestWork(ctx)) {
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
                                ? ctx.data().schedulingTimeoutError()
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
                tasks.recordFailure(failure);
            }
            Failures.rethrow(failure, "cancellation effects failed");
            var state = RequestState.from(ctx.snapshot());
            return state != null && state.matchesBatch(batchId) ? state : null;
        }
    }

    public void rejectRoute(RouteExecution route, Throwable cause) {
        var ctx = route.ctx();
        if (ctx.data().getRequestScheduler() != this || !requests.isCurrent(ctx)) {
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

    /**
     * Advance on the status caller; user callbacks and BATCH cleanup keep their own executors.
     */
    public void onWorkerRequestEvent(RequestContext ctx, WorkerRequestEvent evidence) {
        if (ctx == null || ctx.data().getRequestScheduler() != this) {
            return;
        }
        try (var accepted = tasks.tryRegisterRequestWork(ctx)) {
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
            tasks.recordFailure(failure);
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
        try (var accepted = tasks.tryRegisterRequestWork(ctx)) {
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
            tasks.recordFailure(failure);
            Logger.error("Endpoint retirement processing failed", failure);
        }
    }

    /**
     * All unsent route failures share the same withdrawal and cleanup, independent of the failing role.
     */
    public boolean onRouteFailure(
            RouteExecution route, ContextResult.RouteFailureReason reason, String detail) {
        return onRouteFailure(new ContextResult.RouteFailure(route, reason, detail), null);
    }

    private boolean onRouteFailure(ContextResult.RouteFailure failure, DeliveryEvidence evidence) {
        var route = failure.route();
        var ctx = route.ctx();
        if (ctx.data().getRequestScheduler() != this
                || !requests.isCurrent(ctx)
                || !shouldRetryRouteFailure(failure)) return false;
        try (var work = tasks.tryRegisterRequestWork(ctx)) {
            if (work == null || !ctx.failUnsentRoute(failure, evidence)) return false;
            withdrawFailedRoute(ctx);
            return true;
        }
    }

    private void withdrawFailedRoute(RequestContext ctx) {
        if (!registerAdmission()) return;
        var operation = ctx.beginRouteWithdrawal();
        if (operation == null) {
            finishAdmission();
            return;
        }
        operation
                .withdrawingRoute()
                .afterDeliveryPreparation(
                        preparationFailure -> {
                            if (preparationFailure == null)
                                AbstractRequestScheduler.completePlacement(
                                        operation, PublicationResult.WITHDRAWN);
                            else {
                                var cleanupFailure =
                                        Failures.append(
                                                preparationFailure,
                                                releaseRouteResources(
                                                        operation.withdrawingRoute(),
                                                        DecodeResources.ReleaseReason
                                                                .UNSENT_ROUTE_ROLLBACK));
                                tasks.recordFailure(cleanupFailure);
                                AbstractRequestScheduler.failPlacement(
                                        operation,
                                        buildErrorResponse(
                                                StrategyErrorType.DISPATCH_FAILED,
                                                "Delivery preparation cleanup failed"));
                            }
                        });
    }

    private static boolean needsFollowUpActions(Snapshot before, Snapshot after) {
        return after.preemption() != null
                || after.pendingEnd() != null
                || after.phase() != Phase.OPEN
                || after.canConfirmDelivery() && !after.delivery().acknowledged()
                || before.deadlines() != after.deadlines();
    }

    public void onDeadline(ExpirationTimer.Token exact, long nowMs) {
        var ctx = exact.owner();
        if (ctx.data().getRequestScheduler() != this || !requests.isCurrent(ctx)) {
            return;
        }
        try (var accepted = tasks.tryRegisterRequestWork(ctx)) {
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
                            ctx.data().schedulingTimeoutError(),
                            evidence == null ? detail : String.valueOf(evidence.get("cause")));
            if (evidence != null) ctx.data().setSchedulingDiagnostics(evidence);
        }
        return ctx.recordDeadline(token, nowMs, diagnosis);
    }

    public boolean isAdmissionOpen(RequestContext ctx) {
        if (requests.isClosed() || ctx == null || ctx.data().getRequestScheduler() != this) {
            return false;
        }
        synchronized (ctx) {
            var state = ctx.snapshot();
            return requests.isCurrent(ctx)
                    && state.phase() == Phase.OPEN
                    && state.cancellation() == null;
        }
    }

    public PlacementOperation claimPlacement(RequestContext ctx) {
        if (!registerAdmission()) {
            return null;
        }
        boolean transferred = false;
        try {
            if (ctx == null || ctx.data().getRequestScheduler() != this) {
                return null;
            }
            synchronized (ctx) {
                var claim = requests.isCurrent(ctx) ? ctx.beginPlacement(null) : null;
                if (claim == null) {
                    return null;
                }
                transferred = true;
                return claim;
            }
        } finally {
            if (!transferred) {
                finishAdmission();
            }
        }
    }

    /**
     * Attach the exact acquired route before resource publication; cancellation sees this identity.
     */
    public PlacementOperation beginRoute(RouteExecution route) {
        var ctx = route.ctx();
        synchronized (ctx) {
            var state = ctx.snapshot();
            if (requests.isClosed()
                    || !requests.isCurrent(ctx)
                    || ctx.data().getRequestScheduler() != this
                    || state.placement() == null) return null;
            var operation = state.placement().operation();
            return ctx.bindPlacement(operation, route, PublicationResult.PREPARED)
                    ? operation
                    : null;
        }
    }

    /**
     * Claim an unstarted request for preemption; finish or restore the returned handle.
     */
    public PlacementOperation beginPreemption(
            DecodeEndpoint endpoint,
            DecodeResources.RequestReservation victim,
            int incomingPriority) {
        if (!registerAdmission()) {
            return null;
        }
        PlacementOperation token = null;
        RequestContext context = requests.findActive(victim.requestId());
        try {
            if (context == null
                    || !context.data().getConfig().isQueue()
                    || context.data().getRequestScheduler() != this) {
                return null;
            }
            synchronized (context) {
                RouteExecution item = context.snapshot().route();
                // Context serializes the exact queue claim with delivery permit transfer.
                if (item == null
                        || item.priority() >= incomingPriority
                        || item.decodeEndpoint() != endpoint
                        || !Objects.equals(item.decodeReservation(), victim)
                        || !requests.isCurrent(context)
                        || context.data().getFuture().isDone()
                        || item.requestExpired(System.currentTimeMillis())
                        || !endpoint.ownership(victim).withdrawable()) {
                    return null;
                }
                token = context.beginPlacement(item);
                return token;
            }
        } finally {
            if (token == null) finishAdmission();
        }
    }

    /**
     * Reserve Decode capacity and insert the selected route into its Prefill queue.
     * The route's Prefill reservation records physical publication even if notification fails.
     */
    PlacementResult<RouteExecution, PlacementKey> enqueueRoute(
            RouteExecution route, QueueExecutionSettings settings) {
        var context = route.ctx();
        if (route.decodeReservation() == null
                && !reserveDecode(route, dispatcher.decodeAdmissionCapacity)) {
            return route.validateEndpoints(
                    PlacementResult.blocked(route.placementKey(RoleType.DECODE)));
        }
        context.data().setRouteSubmittedNanos(System.nanoTime());
        var operation = beginRoute(route);
        if (operation == null) return PlacementResult.closed();
        var selection = route.validateEndpoints(PlacementResult.success(route));
        if (selection.status() != PlacementResult.Status.SUCCESS) return selection;
        Throwable failure = null;
        try {
            route.prefillEndpoint().enqueueRoute(route, settings);
        } catch (Throwable problem) {
            failure = problem;
        }
        boolean published = route.prefillReservation() != null;
        boolean recorded = recordRoute(operation, route, published, failure);
        if (!recorded) return PlacementResult.closed();
        if (published) {
            try {
                deliveryReporter.reportLatency(
                        DeliveryMetricsReporter.Latency.ROUTE_SUBMIT,
                        route.prefill().getRole().name(),
                        route.prefillEndpoint().getIp(),
                        System.currentTimeMillis() - route.ctx().data().getStartTime());
            } catch (Throwable telemetryFailure) {
                Logger.warn("Failed to record route-submit telemetry", telemetryFailure);
            }
        }
        return published
                ? selection
                : route.validateEndpoints(
                        PlacementResult.blocked(route.placementKey(RoleType.PREFILL)));
    }

    /**
     * Record publication on the same admission identity, then wake scheduling outside Context.
     */
    static boolean recordRoute(
            PlacementOperation operation,
            RouteExecution route,
            boolean published,
            Throwable failure) {
        boolean recorded = false;
        try {
            recorded =
                    route.ctx()
                            .bindPlacement(
                                    operation,
                                    route,
                                    published
                                            ? PublicationResult.PUBLISHED
                                            : PublicationResult.NOT_PUBLISHED);
            if (published && recorded && route.prefillEndpoint() != null)
                route.prefillEndpoint().signal(PrefillEndpoint.Signal.INPUT);
        } catch (Throwable problem) {
            failure = Failures.append(failure, problem);
        }
        Failures.rethrow(failure, "route publication failed");
        return recorded;
    }

    private static boolean reserveDecode(
            RouteExecution route, DecodeResources.AdmissionCapacity capacity) {
        DecodeResources.RequestReservation reservation = null;
        if (route.decodeEndpoint() != null) {
            var request = route.ctx().data().getRequest();
            reservation =
                    route.decodeEndpoint()
                            .tryReserveQueuedRequest(
                                    route.requestId(),
                                    request.getHardKvTokens(),
                                    request.getExpectedKvTokens(),
                                    route.priority(),
                                    capacity)
                            .value();
            if (reservation == null) return false;
        }
        try {
            route.attachReservation(reservation);
            return true;
        } catch (RuntimeException | Error failure) {
            try {
                AbstractRequestScheduler.releaseDecode(
                        route.decodeEndpoint(),
                        reservation,
                        DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
            } catch (RuntimeException | Error cleanup) {
                Failures.append(failure, cleanup);
            }
            throw failure;
        }
    }

    private static void rollbackPreparation(RouteExecution route, PlacementOperation operation) {
        Failures.rethrow(
                AbstractRequestScheduler.releaseRouteResources(
                        route, DecodeResources.ReleaseReason.LOCAL_ROLLBACK),
                "route preparation rollback failed");
        if (operation != null) {
            operation.owner().bindPlacement(operation, route, PublicationResult.ROLLBACK_COMPLETED);
        }
    }

    /**
     * Publish one selected route and finish its placement before any response callback.
     * Only a capacity-blocked queued route retains its attempt for the scheduler to reclaim
     * capacity or roll back. No other result leaves placement cleanup to the caller.
     */
    public PlacementResult<RouteExecution, PlacementKey> publishPlacement(
            RouteExecution route, PlacementOperation admission, QueueExecutionSettings queue) {
        var placement = route.ctx().snapshot().placement();
        checkState(
                placement != null
                        && placement.operation() == admission
                        && admission.owner() == route.ctx(),
                "route requires its exact active admission");
        PlacementResult<RouteExecution, PlacementKey> result = null;
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
            if (result.status() == PlacementResult.Status.ENDPOINT_CHANGED
                    && route.ctx().snapshot().delivery() == null
                    && (route.prefillReservation() != null || route.encoderCommitted()))
                rollbackPreparation(route, admission);
        } catch (RuntimeException | Error problem) {
            failure = problem;
            throw problem;
        } finally {
            boolean waitingForCapacity =
                    failure == null
                            && queue != null
                            && result != null
                            && result.status() == PlacementResult.Status.BLOCKED;
            if (!waitingForCapacity) {
                Throwable cleanup =
                        Failures.run(
                                null,
                                () ->
                                        Failures.rethrow(
                                                finishPlacement(route, admission),
                                                "placement cleanup failed"));
                if (failure != null) Failures.append(failure, cleanup);
                else if (delivered != null && cleanup != null) tasks.recordFailure(cleanup);
                else Failures.rethrow(cleanup, "placement cleanup failed");
            }
        }
        if (delivered != null) onDeliveryResult(delivered);
        return result;
    }

    /**
     * End placement. Return resource cleanup failures; admission completion failures propagate.
     */
    public static Throwable finishPlacement(RouteExecution route, PlacementOperation admission) {
        Throwable failure =
                route == null || route.prefillReservation() != null || route.encoderCommitted()
                        ? null
                        : Failures.run(null, () -> rollbackPreparation(route, admission));
        boolean rollbackFailed = failure != null;
        if (admission != null) {
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

    PlacementResult<ContextResult.RouteDelivered, PlacementKey> commitDirectRoute(
            RouteExecution route) {
        var context = route.ctx();
        if (route.encoderEndpoint() != null) return commitEncoderRoute(route);
        if (!reserveDecode(route, null))
            return PlacementResult.blocked(route.placementKey(RoleType.DECODE));
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
            var operation = beginRoute(route);
            if (operation == null) return PlacementResult.closed();
            CapacityBoundary.Attempt<PrefillEndpoint.RequestRecord> reservation = null;
            Throwable publicationFailure = null;
            try {
                var data = context.data();
                data.initializeExecutionOrder(data.getEnqueueTime());
                reservation =
                        route.prefillEndpoint().reserveUnqueuedRoute(route, route.prefillWorkMs());
            } catch (Throwable problem) {
                publicationFailure = problem;
            }
            boolean published = reservation != null && reservation.accepted();
            if (!recordRoute(operation, route, published, publicationFailure))
                return PlacementResult.closed();
            if (!published) return PlacementResult.rejected(reservationFailure(reservation));
            var committed = route.prefillEndpoint().commitDirect(route, reservation.value());
            if (committed == null) {
                route.prefillEndpoint().rollback(reservation.value());
                route.clearPrefillReservation(reservation.value());
                return PlacementResult.rejected(Response.error(StrategyErrorType.DISPATCH_FAILED));
            }
            var preceding = committed.calculate();
            if (!route.selectedEndpointsAvailable())
                return route.validateEndpoints(PlacementResult.closed());
            var claim = member.claimDelivery(Delivery.SendMode.ROUTE_DECISION, 0L);
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
                if (cleanup != null) tasks.recordFailure(cleanup);
            } else Failures.rethrow(cleanup, "DIRECT preparation cleanup failed");
        }
    }

    private PlacementResult<ContextResult.RouteDelivered, PlacementKey> commitEncoderRoute(
            RouteExecution route) {
        var context = route.ctx();

        route.attachReservation(null);
        var operation = beginRoute(route);
        if (operation == null) return PlacementResult.closed();
        var config = context.data().getConfig();
        int limit =
                config.isQueue()
                        ? config.getDispatcher().getMaxInflightPerEncoderWorker()
                        : Integer.MAX_VALUE;
        boolean published = route.encoderEndpoint().commit(route, limit);
        if (published) route.markEncoderCommitted();
        if (!recordRoute(operation, route, published, null)) return PlacementResult.closed();
        if (!published)
            return PlacementResult.blocked(
                    new PlacementKey(
                            RoleType.ENCODER,
                            route.encoderEndpoint().group(),
                            route.encoderEndpoint().ipPort()));
        if (!route.selectedEndpointsAvailable())
            return route.validateEndpoints(PlacementResult.closed());
        var claim =
                new Delivery.Member(route, null).claimDelivery(Delivery.SendMode.ROUTE_DECISION, 0);
        if (claim == null) return PlacementResult.closed();
        long now = System.currentTimeMillis();
        return PlacementResult.success(
                new ContextResult.RouteDelivered(
                        claim, context.snapshot().inactivityExpiresAtMs(), now));
    }

    private static Response reservationFailure(CapacityBoundary.Attempt<?> result) {
        if (result == null) return Response.error(StrategyErrorType.REQUEST_CANCELLED);
        var boundary = result.boundary();
        return switch (boundary.status()) {
            case UNAVAILABLE ->
                    Response.buildErrorResponse(
                            StrategyErrorType.RESOURCE_EXHAUSTED,
                            boundary.resource().getDescription());
            case OWNERSHIP_LOST -> Response.error(StrategyErrorType.RESOURCE_EXHAUSTED);
            case FAILED -> Response.error(StrategyErrorType.DISPATCH_FAILED);
        };
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

    void completePlacement(PlacementOperation handle, PublicationResult result, Response failure) {
        var ctx = handle.owner();
        checkState(ctx.data().getRequestScheduler() == this, "foreign placement operation");
        if (!ctx.beginPlacementCompletion(handle)) return;
        var previous = handle.withdrawingRoute();
        var routeFailure = ctx.snapshot().routeFailure();
        var withdrawingDelivery = previous == null ? null : ctx.snapshot().delivery();
        Throwable problem = null;
        try {
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
            Snapshot before;
            Snapshot after;
            synchronized (ctx) {
                before = ctx.snapshot();
                after = ctx.completePlacement(handle, result, System.currentTimeMillis());
            }
            checkState(after != null, "placement completion lost its exact request ownership");
            if (result == PublicationResult.WITHDRAWN && withdrawingDelivery != null)
                withdrawingDelivery.identity().complete();
            if (previous != null && after.route() == previous && failure == null) {
                problem =
                        Failures.run(
                                problem,
                                () ->
                                        previous.prefillEndpoint()
                                                .signal(PrefillEndpoint.Signal.INPUT));
            }
            EndEvidence end = after.pendingEnd();
            if ((end == null
                            || failure != null
                                    && end.kind() == EndKind.SHUTDOWN
                                    && after.cancellation() == null)
                    && (failure != null
                            || after.route() != null
                                    && before.placement() != null
                                    && !before.placement().published())) {
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
                tasks.recordFailure(problem);
            }
        } finally {
            finishAdmission();
        }
        Failures.rethrow(problem, "placement completion failed");
    }

    /**
     * Validate exact request ownership and deadlines before Dispatcher commits resources.
     */
    ContextResult.DeliveryFacts prepareDelivery(
            Delivery.Member member, Delivery.SendMode kind, long batchId) {
        var ctx = member.owner();
        checkState(Thread.holdsLock(ctx), "delivery validation requires Context lock");
        if (!requests.isCurrent(ctx)
                || ctx.data().getRequestScheduler() != this
                || ctx.data().getFuture().isDone()) {
            return null;
        }
        long nowMs = System.currentTimeMillis();
        var before = ctx.snapshot();
        boolean inactive = nowMs >= before.inactivityExpiresAtMs();
        boolean expired = nowMs >= ctx.data().getRequestExpiresAtMs();
        if (before.preparedDelivery() && before.cancellation() == null && (inactive || expired)) {
            String detail =
                    inactive
                            ? "REQUEST_INACTIVE: no matching Engine request status before"
                                    + " inactivity timeout"
                            : "request scheduling deadline exceeded before delivery";
            var deadline =
                    inactive ? before.deadlines().inactivity() : before.deadlines().request();
            if (deadline != null) recordDeadline(ctx, deadline, nowMs, detail);
        }
        return ctx.prepareDelivery(member, kind, batchId, nowMs);
    }

    boolean tryStartSend(Delivery.Member claim) {
        RequestContext.SendResult started;
        Snapshot state;
        synchronized (claim.owner()) {
            started = claim.owner().beginSend(claim, System.currentTimeMillis());
            state = claim.owner().snapshot();
        }
        if (started == RequestContext.SendResult.STARTED) {
            return true;
        }
        if (started == RequestContext.SendResult.ENDPOINT_UNAVAILABLE) {
            onRouteFailure(
                    claim.route(),
                    ContextResult.RouteFailureReason.ENDPOINT_UNAVAILABLE,
                    "Selected Endpoint unavailable before sending");
            return false;
        }
        if (started == RequestContext.SendResult.CANCELLED) {
            CancelReason reason =
                    state.cancellation() != null
                            ? state.cancellation().reason()
                            : state.delivery().abandonment() != null
                                    ? state.delivery().abandonment()
                                    : state.phase() != Phase.OPEN
                                            ? CancelReason.SHUTDOWN
                                            : CancelReason.DEADLINE_EXCEEDED;
            onDeliveryResult(new Abandonment(claim, reason, System.currentTimeMillis()));
        }
        return false;
    }

    static void preparationFailed(RouteExecution route, String detail) {
        route.ctx()
                .data()
                .getRequestScheduler()
                .onDeliveryResult(
                        new PreparationFailure(null, route, detail, System.currentTimeMillis()));
    }

    /**
     * Record an exact delivery result, then settle or continue the request outside its lock.
     */
    public void onDeliveryResult(DeliveryEvidence evidence) {
        Objects.requireNonNull(evidence, "evidence");
        var claim = evidence.delivery();
        var ctx = claim == null ? ((PreparationFailure) evidence).route().ctx() : claim.owner();
        if (ctx.data().getRequestScheduler() != this || !requests.isCurrent(ctx)) {
            return;
        }
        var routeFailure = ctx.unsentDeliveryFailure(evidence);
        if (routeFailure != null && onRouteFailure(routeFailure, evidence)) return;
        try (var accepted = tasks.registerRequestWork(ctx)) {
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
                tasks.recordFailure(failed);
            }
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
                                        responseCompletions, ctx, tasks::recordFailure);
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
                    reconcilePreemption(ctx);
                    recover(
                            tasks::recordFailure,
                            () -> expirationTimer.update(ctx, ctx.snapshot().deadlines()));
                }
                return;
            }
            var claim = ctx.snapshot().delivery().identity();
            var response =
                    claim.route().successResponse(claim.kind() == Delivery.SendMode.BATCH_ENQUEUE);
            Failures.rethrow(
                    recover(
                            tasks::recordFailure,
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

    public Optional<PreemptionRegistration> tryClaim(
            DecodeResources.RequestReservation reservation, long attempt) {
        var ctx = requests.findActive(reservation.requestId());
        if (ctx == null || ctx.data().getRequestScheduler() != this) {
            return Optional.empty();
        }
        synchronized (ctx) {
            var route = ctx.snapshot().route();
            if (route == null) {
                return Optional.empty();
            }
            var prefill = route.prefill();
            var target = new CancelTarget(prefill.getServerIp(), prefill.getGrpcPort());
            var claim = ctx.beginPreemption(reservation, attempt, target);
            return Optional.ofNullable(claim);
        }
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

    public boolean onPreemption(PreemptionRegistration attempt, PreemptionEvent event) {
        var ctx = attempt.owner();
        try (var accepted = tasks.registerRequestWork(ctx)) {
            boolean applied = false, recorded = false, committed = false;
            Snapshot before;
            DecodeEndpoint endpoint = null;
            synchronized (ctx) {
                if (!requests.isCurrent(ctx) || ctx.data().getRequestScheduler() != this) {
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
                                tasks::recordFailure,
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
                        tasks::recordFailure,
                        () -> committedEndpoint.publishCapacity(DecodeEndpoint.CapacityEdge.BOTH));
            }
            if (recorded) {
                try {
                    advanceRequest(ctx);
                } catch (RuntimeException | Error failed) {
                    tasks.recordFailure(failed);
                }
            }
            return applied;
        }
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
                        tasks::recordFailure,
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
        boolean confirmationPending;
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
            confirmationPending = terminal == null && current.preemption().confirmationPending();
            capacityChanged =
                    decode != null && !(terminal != null && current.close().decodeSettled());
        }
        // Reconciliation is a two-phase handoff: the ledger mutation may block on Decode state,
        // while the Context lock is reserved only for the exact commit.
        var mutation =
                !capacityChanged
                        ? null
                        : recover(
                                tasks::recordFailure,
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
                    tasks::recordFailure,
                    () -> decode.publishCapacity(DecodeEndpoint.CapacityEdge.BOTH));
        }
        if (recorded && confirmationPending) {
            recover(
                    tasks::recordFailure,
                    () ->
                            attempt.complete(
                                    new VictimResolution(
                                            ctx.data().getRequestId(),
                                            VictimResolution.Outcome.DELIVERY_RESUMED)));
        }
    }

    /**
     * Own closing through completion or a recorded wait, including recovery after an effect fails.
     */
    void closeRequest(RequestContext ctx, EndEvidence evidence) {
        if (ctx.data().getRequestScheduler() != this || !requests.isCurrent(ctx)) {
            return;
        }
        RequestTaskPool.WorkRegistration execution = null;
        Registration publication = null;
        ResponseCandidate candidate = null;
        Response response = null;
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
                if (ending == EndKind.INACTIVITY) {
                    execution = tasks.registerRequestWork(ctx);
                }
                if (before.outcome() == null
                        && ctx.readyToClose(ending)
                        && before.selectedResponse() == null
                        && !ctx.data().getFuture().isDone()) {
                    publication =
                            Registration.reserve(responseCompletions, ctx, tasks::recordFailure);
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
            if (execution != null) {
                var selectedPublication = publication;
                var selectedCandidate = candidate;
                var selectedResponse = response;
                execution.submit(
                        () ->
                                runClose(
                                        ctx,
                                        selectedPublication,
                                        selectedCandidate,
                                        selectedResponse));
                return;
            }
        } catch (RuntimeException | Error failed) {
            if (execution == null || !execution.hasTaskOwner()) {
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
        runClose(ctx, publication, candidate, response);
    }

    /**
     * Publishes the selected response; physical effects run through the request's close lane.
     */
    private void runClose(
            RequestContext ctx,
            Registration publication,
            ResponseCandidate candidate,
            Response response) {
        try {
            Throwable failure = Failures.run(null, () -> scheduleClose(ctx));
            if (publication != null) {
                failure =
                        Failures.run(
                                failure,
                                () ->
                                        publishResponse(
                                                publication, candidate, response, null, false));
            }
            if (failure != null) tasks.recordFailure(failure);
            Failures.rethrow(failure, "request close failed");
        } catch (RuntimeException | Error failed) {
            publicationFailed(ctx, publication, candidate, failed);
            throw failed;
        } finally {
            if (publication != null) publication.close();
        }
    }

    private void publicationFailed(
            RequestContext ctx,
            Registration publication,
            ResponseCandidate candidate,
            Throwable failed) {
        if (ctx.snapshot().phase() == Phase.OPEN) {
            recover(
                    tasks::recordFailure,
                    () -> expirationTimer.update(ctx, ctx.snapshot().deadlines()));
        }
        if (publication != null) {
            retryPublication(publication, candidate, null, failed, false, failed);
        }
    }

    private Throwable notifyOwners(RequestContext ctx) {
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
                ctx.completeClose(
                        new CloseProgress(false, false, false, false), System.currentTimeMillis());
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
        ctx.completeClose(
                new CloseProgress(
                        true,
                        false,
                        false,
                        failure == null
                                && timersDone
                                && (preemption == null || preemption.finished())),
                System.currentTimeMillis());
        return Failures.append(timerFailure, failure);
    }

    void scheduleClose(RequestContext ctx) {
        if (Thread.holdsLock(ctx)) {
            throw new IllegalStateException(
                    "request closing requires the Context lock to be released");
        }
        var state = ctx.snapshot();
        if (state.phase() == Phase.OPEN
                && state.pendingEnd() == null
                && state.cancellation() == null) {
            return;
        }
        Throwable failure = Failures.run(null, () -> removeRequest(ctx));
        if (state.phase() == Phase.OPEN) {
            Failures.rethrow(failure, "request queue removal failed");
            return;
        }
        boolean inline =
                (state.outcome() == null || state.outcome().kind() != EndKind.RETIREMENT)
                        && (state.delivery() == null
                                || state.delivery().identity().kind()
                                        != Delivery.SendMode.BATCH_ENQUEUE);
        for (int attempt = 0; attempt < 3; attempt++) {
            // Closed admission is permanent. Only a task that was never accepted may be retried.
            try (var accepted = tasks.registerRequestWork(ctx)) {
                try {
                    accepted.submitClose(() -> finishClose(ctx), inline);
                    break;
                } catch (RuntimeException | Error problem) {
                    failure = Failures.append(failure, problem);
                    if (!accepted.canRetrySubmission()) break;
                }
            }
        }
        Failures.rethrow(failure, "close submission failed");
    }

    private void finishClose(RequestContext ctx) {
        Throwable failure = null;
        for (int pass = 0; pass < 3 && ctx.snapshot().phase() == Phase.ENDING; pass++) {
            Throwable problem = notifyOwners(ctx);
            var reason = ctx.releaseReason();
            if (reason != null) {
                var route = ctx.snapshot().route();
                if (route == null) {
                    ctx.completeClose(
                            new CloseProgress(false, true, true, false),
                            System.currentTimeMillis());
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
            failure = Failures.run(failure, () -> requests.archive(ctx, terminal));
            failure = Failures.run(failure, () -> expirationTimer.remove(ctx));
        }
        Failures.rethrow(failure, "request closing failed");
    }

    /**
     * One resource-release implementation for admission rollback, route withdrawal and request end.
     * Rollback checks every acquired resource; request end retains successful release progress.
     */
    static Throwable releaseRouteResources(
            RouteExecution route, DecodeResources.ReleaseReason reason) {
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
                        releaseDecode(route.decodeEndpoint(), route.decodeReservation(), reason);
            }
        } catch (Throwable problem) {
            failure = Failures.append(failure, problem);
        }
        synchronized (ctx) {
            if (ctx.snapshot().route() == route)
                ctx.completeClose(
                        new CloseProgress(false, prefillAndEncoderReleased, decodeReleased, false),
                        System.currentTimeMillis());
        }
        return failure;
    }

    static boolean releaseDecode(
            DecodeEndpoint endpoint,
            DecodeResources.RequestReservation reservation,
            DecodeResources.ReleaseReason reason) {
        if (endpoint == null || reservation == null) return true;
        var released = endpoint.release(reservation, reason).status();
        boolean complete =
                released == DecodeResources.ReservationReleaseResult.RELEASED
                        || released == DecodeResources.ReservationReleaseResult.STALE;
        checkState(
                complete
                        || reason != DecodeResources.ReleaseReason.LOCAL_ROLLBACK
                                && reason != DecodeResources.ReleaseReason.UNSENT_ROUTE_ROLLBACK,
                "route rollback retained Decode resources: %s",
                released);
        return complete;
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

    private static Response response(
            ContextResult.End outcome, RouteExecution route, boolean batched) {
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

    /**
     * Local submission failure uses the same arbitration as external Future completion.
     */
    public boolean publish(RequestContext ctx, Response response, boolean synchronous) {
        return completeExternal(
                ctx, RequestFuture.Completion.RESPONSE, response, null, synchronous);
    }

    final boolean completeExternal(
            RequestContext ctx,
            RequestFuture.Completion completion,
            Response response,
            Throwable failure,
            boolean synchronous) {
        if (Thread.holdsLock(ctx)) {
            throw new IllegalStateException(
                    "Future callbacks require the Context lock to be released");
        }
        if (ctx.data().getRequestScheduler() != this || !requests.isCurrent(ctx)) return false;
        try (var accepted = tasks.tryRegisterRequestWork(ctx)) {
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
                                    Cancellation.frontend(System.currentTimeMillis(), null))) {
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
                    if (cleanup != null) tasks.recordFailure(cleanup);
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

    public void closeOutstandingAndTerminalize() {
        Throwable failure = null;
        for (var ctx : requests.snapshotActive()) {
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
     * Starts a distinct execution for each successful multi-role selection.
     */
    protected PlacementResult<RouteExecution, PlacementKey> prepareSelectedRoute(
            RequestContext context, PlacementResult<RouteResult, PlacementKey> selected) {
        return RouteExecution.fromSelection(context, selected);
    }
}
