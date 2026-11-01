package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.balance.preemption.VictimResolution;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static org.flexlb.balance.scheduler.ContextResult.Abandonment;
import static org.flexlb.balance.scheduler.ContextResult.Cancellation;
import static org.flexlb.balance.scheduler.ContextResult.CancellationFacts;
import static org.flexlb.balance.scheduler.ContextResult.Claim;
import static org.flexlb.balance.scheduler.ContextResult.CleanupAcknowledgement;
import static org.flexlb.balance.scheduler.ContextResult.CloseProgress;
import static org.flexlb.balance.scheduler.ContextResult.DeliveryEvidence;
import static org.flexlb.balance.scheduler.ContextResult.DeliveryFact;
import static org.flexlb.balance.scheduler.ContextResult.EndEvidence;
import static org.flexlb.balance.scheduler.ContextResult.EndKind;
import static org.flexlb.balance.scheduler.ContextResult.EngineEvidence;
import static org.flexlb.balance.scheduler.ContextResult.EngineKind;
import static org.flexlb.balance.scheduler.ContextResult.Granted;
import static org.flexlb.balance.scheduler.ContextResult.Phase;
import static org.flexlb.balance.scheduler.ContextResult.PlacementToken;
import static org.flexlb.balance.scheduler.ContextResult.Prediction;
import static org.flexlb.balance.scheduler.ContextResult.PreemptionReport;
import static org.flexlb.balance.scheduler.ContextResult.PreparationFailure;
import static org.flexlb.balance.scheduler.ContextResult.PublicationResult;
import static org.flexlb.balance.scheduler.ContextResult.Rejected;
import static org.flexlb.balance.scheduler.ContextResult.Rejection;
import static org.flexlb.balance.scheduler.ContextResult.ResponseCandidate;
import static org.flexlb.balance.scheduler.ContextResult.ResponseKind;
import static org.flexlb.balance.scheduler.ContextResult.RouteDelivered;
import static org.flexlb.balance.scheduler.ContextResult.RoutePermit;
import static org.flexlb.balance.scheduler.ContextResult.SendState;
import static org.flexlb.balance.scheduler.ContextResult.Snapshot;
import static org.flexlb.balance.scheduler.ContextResult.Stale;
import static org.flexlb.balance.scheduler.ContextResult.TransportResult;
import static org.flexlb.balance.scheduler.ContextResult.Update;
import static org.flexlb.balance.scheduler.ContextResult.UpdateKind;
import static org.flexlb.balance.scheduler.RequestFuture.Completion.RESPONSE;
import static org.flexlb.dao.loadbalance.Response.buildErrorResponse;
import static org.flexlb.util.Failures.recover;

/**
 * Owns request flows. Context methods only maintain facts across actual external operation
 * boundaries.
 */
public class RequestLifecycle {
    /** Internal notifications after lifecycle state changes; callers use submit/cancel. */
    public interface Scheduling {
        void remove(RequestContext request);

        boolean reschedule(RequestRoute previousRoute);

        default Map<String, Object> describeWait(
                RequestContext request, CancelReason reason, String detail) {
            return null;
        }
    }

    private RequestScheduler owner;
    private Scheduling scheduling;
    final RequestRepository requests;
    final RequestTaskPool tasks;
    final ExpirationTimer expirationTimer;
    final RequestSignals signals;
    final DeliveryDispatcher dispatcher;
    final ResponseCompletionExecutor responseCompletions;
    final org.flexlb.service.monitor.DeliveryMetricsReporter deliveryReporter;
    final TaskTimer taskTimer;

    final DecodeResources.AdmissionCapacity decodeAdmissionCapacity;
    private final RecentCacheKeyTraceReporter cacheTrace;

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
                                                        == DeliveryClaimKind.BATCH_ENQUEUE)
                        .map(delivery -> signals.delivery(delivery).toCompletableFuture())
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

    public RequestLifecycle(
            RequestRepository requests,
            RequestTaskPool tasks,
            ExpirationTimer deadlines,
            RequestSignals signals,
            DeliveryDispatcher dispatcher,
            ResponseCompletionExecutor responseCompletions,
            org.flexlb.service.monitor.DeliveryMetricsReporter deliveryReporter,
            TaskTimer taskTimer,
            DecodeResources.AdmissionCapacity capacity,
            RecentCacheKeyTraceReporter cacheTrace) {
        this.requests = Objects.requireNonNull(requests);
        this.tasks = Objects.requireNonNull(tasks);
        this.expirationTimer = Objects.requireNonNull(deadlines);
        this.signals = Objects.requireNonNull(signals);
        this.dispatcher = Objects.requireNonNull(dispatcher);
        this.responseCompletions = Objects.requireNonNull(responseCompletions);
        this.deliveryReporter = Objects.requireNonNull(deliveryReporter);
        this.taskTimer = Objects.requireNonNull(taskTimer);
        this.decodeAdmissionCapacity = Objects.requireNonNull(capacity);
        this.cacheTrace = Objects.requireNonNull(cacheTrace);
    }

    public void bindScheduler(RequestScheduler owner, Scheduling scheduling) {
        checkState(this.owner == null, "request lifecycle already bound");
        this.owner = Objects.requireNonNull(owner);
        this.scheduling = Objects.requireNonNull(scheduling);
    }

    public CompletableFuture<Response> register(
            RequestContext ctx, Runnable onRegistered, StrategyErrorType expiredError) {
        if (ctx == null || ctx.data().getRequest() == null) {
            return CompletableFuture.completedFuture(
                    Response.error(StrategyErrorType.INVALID_REQUEST));
        }
        var data = ctx.data();
        var config = data.getConfig();
        if ((requests.isStopping() || requests.isClosed())) {
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
                                    expiredError,
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
        return future;
    }

    public final RequestState cancel(long id, long batchId, CancelReason reason) {
        Objects.requireNonNull(reason);
        var ctx = requests.findActive(id);
        if (ctx != null) {
            if (ctx.data().getScheduler() != this) {
                return null;
            }
            var result = cancel(ctx, batchId, reason);
            if (result != null) {
                return result;
            }
        }
        var terminal = requests.findTerminal(id);
        return terminal != null
                        && terminal.owner() == this
                        && terminal.state().matchesBatch(batchId)
                ? terminal.state()
                : null;
    }

    /** Cancellation freezes its first cause, then directs the owner of the real resource. */
    public final RequestState cancel(RequestContext ctx, long batchId, CancelReason reason) {
        if (ctx == null || ctx.data().getScheduler() != this || !requests.isCurrent(ctx)) {
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
                Map<String, Object> evidence = this.scheduling.describeWait(ctx, reason, message);
                var error =
                        evidence != null
                                ? StrategyErrorType.RESOURCE_EXHAUSTED
                                : reason == CancelReason.DEADLINE_EXCEEDED
                                        ? StrategyErrorType.BATCH_SLO_EXPIRED
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
                failure = Failures.run(null, () -> this.scheduling.remove(ctx));
                failure = Failures.run(failure, () -> advanceRequest(ctx, ctx.snapshot()));
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
            return requests.getRequestState(ctx.data().getRequestId(), batchId);
        }
    }

    public enum QueueKind {
        EXPIRED,
        REJECTED
    }

    public void onQueueEvent(RequestRoute route, QueueKind kind, Throwable cause) {
        var ctx = route.ctx();
        if (ctx.data().getScheduler() != this || !requests.isCurrent(ctx)) {
            return;
        }
        if (kind == QueueKind.EXPIRED) {
            cancel(ctx, 0L, CancelReason.DEADLINE_EXCEEDED);
            return;
        }
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

    /** Advance on the status caller; user callbacks and BATCH cleanup keep their own executors. */
    public void onEngineEvidence(RequestContext ctx, EngineEvidence evidence) {
        if (ctx == null || ctx.data().getScheduler() != this) {
            return;
        }
        try (var accepted = tasks.tryRegisterRequestWork(ctx)) {
            if (accepted == null) {
                return;
            }
            Update update = ctx.recordEngineEvidence(evidence);
            if (update.kind() != UpdateKind.APPLIED) {
                return;
            }
            var after = update.after();
            var delivery = after.delivery();
            boolean confirmation =
                    delivery != null
                            && !delivery.acknowledged()
                            && delivery.abandonment() == null
                            && (delivery.send() == SendState.DELIVERED
                                    || delivery.send() == SendState.UNKNOWN
                                            && after.engine().decodeAccepted());
            if (after.pendingEnd() != null && after.pendingEnd().kind() == EndKind.RETIREMENT) {
                closeRequest(ctx, after.pendingEnd());
                return;
            }
            if (after.preemption() == null
                    && after.pendingEnd() == null
                    && after.phase() == Phase.OPEN
                    && !confirmation
                    && update.before().deadlines() == after.deadlines()) {
                return;
            }
            accepted.runInline(() -> advanceRequest(ctx, ctx.snapshot()));
        } catch (Throwable failure) {
            tasks.recordFailure(failure);
            Logger.error("Engine evidence processing failed", failure);
        }
    }

    public void onDeadline(ExpirationTimer.Token exact, long nowMs) {
        var ctx = exact.owner();
        if (ctx.data().getScheduler() != this || !requests.isCurrent(ctx)) {
            return;
        }
        try (var accepted = tasks.tryRegisterRequestWork(ctx)) {
            if (accepted == null) {
                return;
            }
            Update update;
            synchronized (ctx) {
                var before = ctx.snapshot();
                CancellationFacts diagnosis = null;
                boolean inactivity =
                        exact.kind() == ExpirationTimer.Kind.INACTIVITY
                                && before.deadlines().inactivity() == exact
                                && nowMs >= before.inactivityExpiresAtMs();
                boolean scheduling =
                        exact.kind() == ExpirationTimer.Kind.REQUEST
                                && before.deadlines().request() == exact
                                && nowMs >= exact.atMs()
                                && (before.delivery() == null || before.placement() != null);
                if (before.phase() == Phase.OPEN
                        && before.cancellation() == null
                        && (inactivity || scheduling)) {
                    String detail =
                            inactivity
                                    ? "REQUEST_INACTIVE: no matching Engine request status before"
                                            + " inactivity timeout"
                                    : before.placement() != null
                                            ? "request scheduling deadline exceeded during"
                                                    + " admission"
                                            : CancelReason.DEADLINE_EXCEEDED.getMessage();
                    var evidence =
                            this.scheduling.describeWait(
                                    ctx, CancelReason.DEADLINE_EXCEEDED, detail);
                    diagnosis =
                            new CancellationFacts(
                                    CancelReason.DEADLINE_EXCEEDED,
                                    evidence == null
                                            ? StrategyErrorType.BATCH_SLO_EXPIRED
                                            : StrategyErrorType.RESOURCE_EXHAUSTED,
                                    evidence == null
                                            ? detail
                                            : String.valueOf(evidence.get("cause")));
                    if (evidence != null) {
                        ctx.data().setSchedulingDiagnostics(evidence);
                    }
                }
                update = ctx.recordDeadline(exact, nowMs, diagnosis);
            }
            if (update.kind() == UpdateKind.STALE) {
                return;
            }
            accepted.submit(
                    () -> {
                        var current = ctx.snapshot();
                        if (nowMs < exact.atMs()) {
                            expirationTimer.update(ctx, current.deadlines());
                            return;
                        }
                        Throwable failure = null;
                        if (update.before().cancellation() == null
                                && update.after().cancellation() != null) {
                            failure = Failures.run(null, () -> this.scheduling.remove(ctx));
                        }
                        failure = Failures.run(failure, () -> advanceRequest(ctx, ctx.snapshot()));
                        Failures.rethrow(failure, "deadline effects failed");
                    });
        }
    }

    public boolean isAdmissionOpen(long id, CompletableFuture<?> future) {
        var ctx = requests.findActive(id);
        if (requests.isClosed()
                || ctx == null
                || ctx.data().getScheduler() != this
                || ctx.data().getFuture() != future) {
            return false;
        }
        synchronized (ctx) {
            var state = ctx.snapshot();
            return requests.isCurrent(ctx)
                    && state.phase() == Phase.OPEN
                    && state.cancellation() == null;
        }
    }

    public AdmissionHandle claimAdmissionHandle(long id, CompletableFuture<?> future) {
        if (!registerAdmission()) {
            return null;
        }
        boolean transferred = false;
        try {
            var ctx = requests.findActive(id);
            if (ctx == null
                    || ctx.data().getScheduler() != this
                    || ctx.data().getFuture() != future) {
                return null;
            }
            synchronized (ctx) {
                var claim = requests.isCurrent(ctx) ? ctx.beginPlacement(null) : null;
                if (!(claim instanceof Granted<PlacementToken> granted)) {
                    return null;
                }
                transferred = true;
                return new AdmissionHandle(granted.value(), this);
            }
        } finally {
            if (!transferred) {
                finishAdmission();
            }
        }
    }

    public interface Publication {
        void publish();

        boolean published();
    }

    public PlacementResult.Status commitRoute(RequestRoute route, Publication publication) {
        var ctx = route.ctx();
        PlacementToken operation;
        synchronized (ctx) {
            var state = ctx.snapshot();
            if (requests.isClosed()
                    || !requests.isCurrent(ctx)
                    || ctx.data().getScheduler() != this
                    || state.placement() == null) {
                return PlacementResult.Status.CLOSED;
            }
            operation = state.placement().operation();
            var bound = ctx.bindPlacement(operation, route, PublicationResult.PREPARED);
            if (!(bound instanceof Granted<RoutePermit>)) {
                return PlacementResult.Status.CLOSED;
            }
        }
        Throwable failure = Failures.run(null, publication::publish);
        boolean published = publication.published();
        Claim<RoutePermit> reported;
        synchronized (ctx) {
            reported =
                    ctx.bindPlacement(
                            operation,
                            route,
                            published
                                    ? PublicationResult.PUBLISHED
                                    : PublicationResult.NOT_PUBLISHED);
        }
        if (published && !(reported instanceof Stale<?>)) {
            failure =
                    Failures.run(
                            failure,
                            () ->
                                    route.prefillEp()
                                            .signal(
                                                    org.flexlb.balance.endpoint.PrefillEndpoint
                                                            .Signal.INPUT));
        }
        Failures.rethrow(failure, "route publication failed");
        return reported instanceof Stale<?>
                ? PlacementResult.Status.CLOSED
                : published ? PlacementResult.Status.SUCCESS : PlacementResult.Status.BLOCKED;
    }

    void completePlacement(AdmissionHandle handle, PublicationResult result, Response failure) {
        var ctx = handle.owner();
        var previous = handle.withdrawingRoute();
        Throwable problem = null;
        try {
            if (result == PublicationResult.WITHDRAWN) {
                try {
                    checkState(previous != null, "withdrawal requires an exact old route");
                    checkState(
                            previous.prefillEp().removeQueued(previous),
                            "withdrawn route is no longer queued: %s",
                            ctx.data().getRequestId());
                } catch (Throwable detachFailure) {
                    problem = detachFailure;
                    result = PublicationResult.FAILED;
                    failure =
                            buildErrorResponse(
                                    StrategyErrorType.DISPATCH_FAILED,
                                    "queued route withdrawal failed");
                }
            }
            var update = ctx.completePlacement(handle.token, result, System.currentTimeMillis());
            if (update.kind() == UpdateKind.STALE) {
                handle.notifyPlacementFailure();
                return;
            }
            var after = update.after();
            if (previous != null && after.route() == previous && failure == null) {
                problem =
                        Failures.run(
                                problem,
                                () -> previous.prefillEp().signal(PrefillEndpoint.Signal.INPUT));
            }
            EndEvidence end = after.pendingEnd();
            if ((end == null
                            || failure != null
                                    && end.kind() == EndKind.SHUTDOWN
                                    && after.cancellation() == null)
                    && (failure != null
                            || after.route() != null
                                    && update.before().placement() != null
                                    && !update.before().placement().published())) {
                end = EndEvidence.publicationFailure(after.route(), failure);
            }
            if (end != null) {
                var actual = end;
                problem = Failures.run(problem, () -> closeRequest(ctx, actual));
            } else if (result == PublicationResult.WITHDRAWN
                    && after.phase() == Phase.OPEN
                    && !ctx.data().getFuture().isDone()) {
                boolean requeued = false;
                try {
                    requeued = this.scheduling.reschedule(previous);
                } catch (Throwable requeueFailure) {
                    problem = Failures.append(problem, requeueFailure);
                }
                if (requeued) {
                    problem =
                            Failures.run(
                                    problem,
                                    () -> expirationTimer.update(ctx, ctx.snapshot().deadlines()));
                } else {
                    var withdrawn = EndEvidence.withdrawnFailure("queued route requeue failed");
                    problem = Failures.run(problem, () -> closeRequest(ctx, withdrawn));
                }
            } else {
                problem = Failures.run(problem, () -> advanceRequest(ctx, ctx.snapshot()));
            }
            if (problem != null) {
                problem = Failures.run(problem, handle::notifyPlacementFailure);
            }
            if (problem != null) {
                tasks.recordFailure(problem);
            }
        } catch (Throwable completionFailure) {
            try {
                handle.notifyPlacementFailure();
            } catch (Throwable cleanupFailure) {
                completionFailure.addSuppressed(cleanupFailure);
            }
            throw completionFailure;
        } finally {
            finishAdmission();
        }
        Failures.rethrow(problem, "placement completion failed");
    }

    /**
     * Context validation, Decode ownership CAS and Context commit share one short critical section.
     */
    public DeliveryClaim claimDelivery(
            RequestRoute route,
            DeliveryClaimKind kind,
            long batchId,
            DecodeEndpoint.EngineDispatchPermit permit) {
        var ctx = route.ctx();
        DecodeResources.ResourceChange<DecodeResources.EngineDispatchPermitTransferStatus>
                transfer = null;
        DeliveryClaim transferred = null;
        try {
            synchronized (ctx) {
                if (!requests.isCurrent(ctx)
                        || ctx.data().getScheduler() != this
                        || ctx.data().getFuture().isDone()) {
                    return null;
                }
                long nowMs = System.currentTimeMillis();
                var before = ctx.snapshot();
                boolean inactive = nowMs >= before.inactivityExpiresAtMs();
                boolean expired = nowMs >= ctx.data().getRequestExpiresAtMs();
                if (before.preparedDelivery()
                        && before.cancellation() == null
                        && (inactive || expired)) {
                    String detail =
                            inactive
                                    ? "REQUEST_INACTIVE: no matching Engine request status before"
                                            + " inactivity timeout"
                                    : "request scheduling deadline exceeded before delivery";
                    var evidence =
                            this.scheduling.describeWait(
                                    ctx, CancelReason.DEADLINE_EXCEEDED, detail);
                    var diagnosis =
                            new CancellationFacts(
                                    CancelReason.DEADLINE_EXCEEDED,
                                    evidence == null
                                            ? StrategyErrorType.BATCH_SLO_EXPIRED
                                            : StrategyErrorType.RESOURCE_EXHAUSTED,
                                    evidence == null
                                            ? detail
                                            : String.valueOf(evidence.get("cause")));
                    var deadline =
                            inactive
                                    ? before.deadlines().inactivity()
                                    : before.deadlines().request();
                    if (deadline != null) {
                        ctx.recordDeadline(deadline, nowMs, diagnosis);
                    }
                    if (evidence != null) {
                        ctx.data().setSchedulingDiagnostics(evidence);
                    }
                }
                var claim = ctx.prepareDelivery(route, kind, batchId, nowMs);
                if (claim instanceof Granted<ContextResult.DeliveryFacts> granted) {
                    if (permit != null) {
                        checkArgument(
                                permit.belongsTo(route.decodeEp(), route.decodeReservation()),
                                "foreign Decode dispatch permit");
                        transfer = permit.dispatch();
                        if (transfer.status()
                                != DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED) {
                            throw new IllegalStateException("Decode dispatch ownership lost");
                        }
                    }
                    transferred = granted.value().identity();
                    ctx.publishDelivery(granted.value());
                    return granted.value().identity();
                }
            }
        } catch (RuntimeException | Error failure) {
            if (transferred != null) {
                onDeliveryFact(
                        new PreparationFailure(
                                transferred,
                                route,
                                failure.getMessage(),
                                System.currentTimeMillis()));
            }
            throw failure;
        } finally {
            if (transfer != null && transfer.capacityReleased()) {
                recover(
                        tasks::recordFailure,
                        () ->
                                route.decodeEp()
                                        .publishCapacity(DecodeEndpoint.CapacityEdge.DISPATCH));
            }
        }
        if (ctx.snapshot().pendingEnd() != null) {
            closeRequest(ctx, null);
        } else if (System.currentTimeMillis() >= ctx.data().getRequestExpiresAtMs()) {
            cancel(ctx, 0L, CancelReason.DEADLINE_EXCEEDED);
        }
        return null;
    }

    boolean tryStartSend(DeliveryClaim claim) {
        var started = claim.owner().beginSend(claim, System.currentTimeMillis());
        if (started instanceof Granted<?>) {
            return true;
        }
        if (started instanceof Rejected<DeliveryClaim> rejected
                && rejected.reason() == Rejection.CANCELLED) {
            var state = started.after();
            CancelReason reason =
                    state.cancellation() != null
                            ? state.cancellation().reason()
                            : state.delivery().abandonment() != null
                                    ? state.delivery().abandonment()
                                    : state.phase() != Phase.OPEN
                                            ? CancelReason.SHUTDOWN
                                            : CancelReason.DEADLINE_EXCEEDED;
            onDeliveryFact(new Abandonment(claim, reason, System.currentTimeMillis()));
        }
        return false;
    }

    static void preparationFailed(RequestRoute route, String detail) {
        route.ctx()
                .data()
                .getScheduler()
                .onDeliveryFact(
                        new PreparationFailure(null, route, detail, System.currentTimeMillis()));
    }

    /** Receives every immutable delivery or preemption fact at one business ingress. */
    public PreemptionRegistration.Reply onDeliveryFact(DeliveryFact fact) {
        Objects.requireNonNull(fact, "fact");
        if (fact instanceof DeliveryEvidence evidence) {
            applyDeliveryEvidence(evidence.delivery(), evidence);
            return null;
        }
        if (fact instanceof ContextResult.PreemptionFact preemption) {
            var event = preemption.event();
            return event.progress(onPreemption(preemption.attempt(), event));
        }
        throw new IllegalArgumentException("unsupported delivery fact: " + fact.getClass());
    }

    private void applyDeliveryEvidence(DeliveryClaim claim, DeliveryEvidence evidence) {
        var ctx = claim == null ? ((PreparationFailure) evidence).route().ctx() : claim.owner();
        if (ctx.data().getScheduler() != this || !requests.isCurrent(ctx)) {
            return;
        }
        try (var accepted = tasks.registerRequestWork(ctx)) {
            Update update;
            Snapshot before;
            synchronized (ctx) {
                before = ctx.snapshot();
                if ((evidence instanceof Prediction || evidence instanceof RouteDelivered)
                        && before.delivery() != null
                        && before.delivery().identity() == claim
                        && before.route() == claim.item()
                        && !before.engine().decodeAccepted()
                        && claim.item().decodeEp() != null
                        && claim.item()
                                .decodeEp()
                                .ownership(claim.item().decodeReservation())
                                .confirmed()) {
                    ctx.recordEngineEvidence(
                            new EngineEvidence(
                                    claim.item(),
                                    claim.item().decodeReservation(),
                                    RoleType.DECODE,
                                    EngineKind.ACTIVE,
                                    0L,
                                    null,
                                    evidence.nowMs()));
                }
                update = ctx.recordDeliveryResult(evidence);
            }
            if (update.kind() == UpdateKind.STALE || update.after() == before) {
                return;
            }
            try {
                if (evidence instanceof PreparationFailure
                        || evidence instanceof TransportResult result && result.result().failed()) {
                    closeRequest(ctx, null);
                } else {
                    advanceRequest(ctx, update.after());
                }
            } catch (RuntimeException | Error failed) {
                // The accepted operation's owner has already recovered its remaining effects.
                tasks.recordFailure(failed);
            }
        }
    }

    /** True transfers effects to confirmation/close; false leaves only deadline synchronization. */
    private boolean confirmDelivery(RequestContext ctx) {
        ResponsePublication publication = null;
        ResponseCandidate candidate = null;
        try {
            synchronized (ctx) {
                var state = ctx.snapshot();
                var delivery = state.delivery();
                if (state.phase() != Phase.OPEN
                        || state.pendingEnd() != null
                        || state.cancellation() != null
                        || delivery == null
                        || delivery.acknowledged()
                        || delivery.abandonment() != null
                        || !(delivery.send() == SendState.DELIVERED
                                || delivery.send() == SendState.UNKNOWN
                                        && state.engine().decodeAccepted())) {
                    return false;
                }
                if (System.currentTimeMillis() >= state.inactivityExpiresAtMs()) {
                    String detail =
                            "REQUEST_INACTIVE: no matching Engine request status before inactivity"
                                    + " timeout";
                    var evidence =
                            this.scheduling.describeWait(
                                    ctx, CancelReason.DEADLINE_EXCEEDED, detail);
                    var diagnosis =
                            new CancellationFacts(
                                    CancelReason.DEADLINE_EXCEEDED,
                                    evidence == null
                                            ? StrategyErrorType.BATCH_SLO_EXPIRED
                                            : StrategyErrorType.RESOURCE_EXHAUSTED,
                                    evidence == null
                                            ? detail
                                            : String.valueOf(evidence.get("cause")));
                    if (state.deadlines().inactivity() != null) {
                        ctx.recordDeadline(
                                state.deadlines().inactivity(),
                                System.currentTimeMillis(),
                                diagnosis);
                    }
                } else {
                    if (state.preemption() == null || state.preemption().finished()) {
                        publication =
                                ResponsePublication.reserve(
                                        responseCompletions, ctx, true, tasks::recordFailure);
                    }
                    var acknowledged =
                            ctx.selectResponse(
                                    new ResponseCandidate(
                                            ctx,
                                            ResponseKind.CONFIRMATION,
                                            null,
                                            System.currentTimeMillis()));
                    if (publication != null) {
                        if (!(acknowledged instanceof Granted<?>)) {
                            return false;
                        }
                        ctx.data().discardGenerateInput();
                        candidate =
                                new ResponseCandidate(
                                        ctx, ResponseKind.ACK, null, System.currentTimeMillis());
                        if (delivery.identity().kind() == DeliveryClaimKind.BATCH_ENQUEUE
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
                return true;
            }
            var claim = ctx.snapshot().delivery().identity();
            var response =
                    claim.item().successResponse(claim.kind() == DeliveryClaimKind.BATCH_ENQUEUE);
            Failures.rethrow(
                    recover(
                            tasks::recordFailure,
                            () -> expirationTimer.update(ctx, ctx.snapshot().deadlines())),
                    "delivery deadline installation failed");
            publication.publish(candidate, RESPONSE, response, null, false, deliveryReporter);
            return true;
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
            DecodeResources.ReservationHandle reservation, long attempt) {
        var ctx = requests.findActive(reservation.requestId());
        if (ctx == null || ctx.data().getScheduler() != this) {
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
            return claim instanceof Granted<PreemptionRegistration> granted
                    ? Optional.of(granted.value())
                    : Optional.empty();
        }
    }

    public record PreemptionEvent(
            PreemptionCancelPhase phase,
            boolean release,
            DecodeEndpoint endpoint,
            DecodeResources.ReservationHandle reservation,
            String detail,
            CleanupAcknowledgement cleanup) {
        public PreemptionEvent(
                PreemptionCancelPhase phase,
                boolean release,
                DecodeEndpoint endpoint,
                DecodeResources.ReservationHandle reservation,
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
                if (!requests.isCurrent(ctx) || ctx.data().getScheduler() != this) {
                    return false;
                }
                before = ctx.snapshot();
                if (before.preemption() != null
                        && before.preemption().identity() == attempt
                        && (event.endpoint() == null
                                || before.route().decodeEp() == event.endpoint()
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
                                || current.route().decodeEp() == endpoint
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
                    recorded |=
                            ctx.recordDeliveryResult(event.cleanup()).kind() == UpdateKind.APPLIED;
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
                    advanceRequest(ctx, ctx.snapshot());
                } catch (RuntimeException | Error failed) {
                    tasks.recordFailure(failed);
                }
            }
            return applied;
        }
    }

    /** Select one effect owner after recording facts; that owner also handles its failures. */
    private void advanceRequest(RequestContext ctx, Snapshot state) {
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
            if (state.pendingEnd() != null || state.phase() == Phase.ENDING) {
                closeRequest(ctx, null);
            } else if (state.phase() == Phase.OPEN) {
                if (!confirmDelivery(ctx)) {
                    var deadlines = ctx.snapshot().deadlines();
                    if (expirationTimer.needsUpdate(ctx, deadlines)) {
                        recover(tasks::recordFailure, () -> expirationTimer.update(ctx, deadlines));
                    }
                }
            }
        } catch (RuntimeException | Error failed) {
            failure = Failures.append(failure, failed);
        }
        Failures.rethrow(failure, "request continuation failed");
    }

    /** The exact reconciliation owns ledger retries and its post-commit capacity/resumption effects. */
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
            var claim = ctx.prepareReconciliation(current.preemption().identity());
            if (!(claim instanceof Granted<PreemptionRegistration> granted)) {
                return;
            }
            attempt = granted.value();
            decode = current.route().decodeEp();
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
                            signals.resolvePreemption(
                                    attempt,
                                    new VictimResolution(
                                            ctx.data().getRequestId(),
                                            VictimResolution.Outcome.DELIVERY_RESUMED)));
        }
    }

    /** Own closing through completion or a recorded wait, including recovery after an effect fails. */
    void closeRequest(RequestContext ctx, EndEvidence evidence) {
        if (ctx.data().getScheduler() != this || !requests.isCurrent(ctx)) {
            return;
        }
        boolean deferred = evidence != null && evidence.kind() == EndKind.INACTIVITY;
        try (var execution = deferred ? tasks.registerRequestWork(ctx) : null) {
            ResponsePublication publication = null;
            boolean handedOff = false;
            ResponseCandidate candidate = null;
            try {
                Response response = null;
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
                    if (before.outcome() == null
                            && ctx.readyToClose(ending)
                            && before.selectedResponse() == null
                            && !ctx.data().getFuture().isDone()) {
                        publication =
                                ResponsePublication.reserve(
                                        responseCompletions, ctx, false, tasks::recordFailure);
                    }
                    var after = ctx.beginClose(evidence, System.currentTimeMillis());
                    if (after.outcome() != null) {
                        var outcome = after.outcome();
                        if (publication != null) {
                            candidate =
                                    new ResponseCandidate(
                                            ctx,
                                            ResponseKind.TERMINAL,
                                            outcome,
                                            System.currentTimeMillis());
                            if (outcome.kind() == EndKind.DELIVERY_FAILURE) {
                                ctx.selectResponse(candidate);
                            }
                            boolean batched =
                                    after.delivery() != null
                                            && after.delivery().identity().kind()
                                                    == DeliveryClaimKind.BATCH_ENQUEUE;
                            response =
                                    ResponsePublication.response(outcome, before.route(), batched);
                        }
                    } else if (publication != null) {
                        publication.close();
                        publication = null;
                    }
                }
                if (deferred) {
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
                    handedOff = true;
                } else {
                    handedOff = true;
                    runClose(ctx, publication, candidate, response);
                }
            } catch (RuntimeException | Error failed) {
                if (!handedOff && (execution == null || !execution.hasTaskOwner())) {
                    publicationFailed(ctx, publication, candidate, failed);
                    Failures.run(failed, () -> scheduleClose(ctx));
                }
                throw failed;
            } finally {
                if (!handedOff
                        && (execution == null || !execution.hasTaskOwner())
                        && publication != null) {
                    publication.close();
                }
            }
        }
    }

    /** Publishes the selected response; physical effects run through the request's close lane. */
    private void runClose(
            RequestContext ctx,
            ResponsePublication publication,
            ResponseCandidate candidate,
            Response response) {
        try {
            Throwable failure = Failures.run(null, () -> scheduleClose(ctx));
            if (publication != null) {
                failure =
                        Failures.run(
                                failure,
                                () ->
                                        publication.publish(
                                                candidate,
                                                RESPONSE,
                                                response,
                                                null,
                                                false,
                                                deliveryReporter));
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
            ResponsePublication publication,
            ResponseCandidate candidate,
            Throwable failed) {
        if (ctx.snapshot().phase() == Phase.OPEN) {
            recover(
                    tasks::recordFailure,
                    () -> expirationTimer.update(ctx, ctx.snapshot().deadlines()));
        }
        if (publication != null) {
            publication.publishAfterFailure(
                    failed,
                    candidate,
                    RequestFuture.Completion.FAILURE,
                    null,
                    failed,
                    false,
                    deliveryReporter);
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
                        new CloseProgress(false, false, false, false, timersDone),
                        System.currentTimeMillis());
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
                                    signals.resolvePreemption(
                                            preemption.identity(),
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
                                && (preemption == null || preemption.finished()),
                        timersDone),
                System.currentTimeMillis());
        return Failures.append(timerFailure, failure);
    }

    void scheduleClose(RequestContext ctx) {
        if (Thread.holdsLock(ctx)) {
            throw new IllegalStateException(
                    "request closing requires the Context lock to be released");
        }
        var state = ctx.snapshot();
        if (state.phase() == Phase.OPEN) {
            return;
        }
        boolean inline =
                (state.outcome() == null || state.outcome().kind() != EndKind.RETIREMENT)
                        && (state.delivery() == null
                                || state.delivery().identity().kind()
                                        != DeliveryClaimKind.BATCH_ENQUEUE);
        Throwable failure = null;
        for (int retry = 0; retry < 3; retry++) {
            RequestTaskPool.WorkRegistration accepted = null;
            try {
                accepted = tasks.registerRequestWork(ctx);
                accepted.submitClose(() -> finishClose(ctx), inline);
                break;
            } catch (RuntimeException | Error failed) {
                failure = Failures.append(failure, failed);
                if (accepted != null && !accepted.canRetrySubmission()) {
                    break;
                }
            } finally {
                if (accepted != null) {
                    accepted.close();
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
            var state = ctx.snapshot();
            if (reason != null) {
                var settled =
                        releaseResources(
                                state.route(),
                                reason,
                                state.close().prefillSettled(),
                                state.close().decodeSettled());
                problem = Failures.append(problem, settled.failure());
                ctx.completeClose(
                        new CloseProgress(
                                false,
                                settled.prefillSettled(),
                                settled.decodeSettled(),
                                false,
                                false),
                        System.currentTimeMillis());
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
            failure = Failures.run(failure, () -> signals.completeDelivery(delivery.identity()));
        }
        if (after.terminalRecord() != null) {
            var terminal = after.terminalRecord();
            failure = Failures.run(failure, () -> requests.archive(ctx, terminal));
            failure = Failures.run(failure, () -> expirationTimer.remove(ctx));
        }
        Failures.rethrow(failure, "request closing failed");
    }

    record Settlement(boolean prefillSettled, boolean decodeSettled, Throwable failure) {}

    static Settlement releaseResources(
            RequestRoute route,
            DecodeResources.ReleaseReason reason,
            boolean prefill,
            boolean decode) {
        if (route == null) {
            return new Settlement(true, true, null);
        }
        Throwable failure = null;
        try {
            if (!prefill) {
                route.prefillEp().release(null, route);
                prefill = true;
            }
        } catch (Throwable problem) {
            failure = problem;
        }
        try {
            if (!decode) {
                if (route.decodeEp() == null || route.decodeReservation() == null) {
                    decode = true;
                } else if (reason != null) {
                    var released =
                            route.decodeEp().release(route.decodeReservation(), reason).status();
                    decode =
                            released == DecodeResources.ReservationReleaseResult.RELEASED
                                    || released == DecodeResources.ReservationReleaseResult.STALE;
                } else {
                    decode = !route.decodeEp().ownership(route.decodeReservation()).owned();
                }
            }
        } catch (Throwable problem) {
            failure = Failures.append(failure, problem);
        }
        return new Settlement(prefill, decode, failure);
    }

    /** Local submission failure uses the same arbitration as external Future completion. */
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
        if (ctx.data().getScheduler() != this || !requests.isCurrent(ctx)) return false;
        try (var accepted = tasks.tryRegisterRequestWork(ctx)) {
            if (accepted == null) return false;
            ResponsePublication publication = null;
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
                    publication = new ResponsePublication(responseCompletions, ctx, false);
                    if (cancelled) {
                        var reason = CancelReason.CLIENT_CANCELLED;
                        var diagnostics = scheduling.describeWait(ctx, reason, reason.getMessage());
                        if (before.cancellation() == null) {
                            String cause =
                                    diagnostics == null
                                            ? null
                                            : String.valueOf(diagnostics.get("cause"));
                            var cancel =
                                    ctx.requestCancel(
                                            Cancellation.frontend(
                                                    System.currentTimeMillis(), cause));
                            if (!cancel) return false;
                        }
                    } else {
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
                                    cancelled
                                            ? ResponseKind.CANCELLATION
                                            : completion == RequestFuture.Completion.FAILURE
                                                    ? ResponseKind.EXTERNAL_FAILURE
                                                    : ResponseKind.TERMINAL,
                                    cancelled ? null : ctx.snapshot().outcome(),
                                    System.currentTimeMillis());
                    ctx.selectResponse(candidate);
                }
                try {
                    Throwable cleanup = Failures.run(null, () -> scheduling.remove(ctx));
                    cleanup = Failures.run(cleanup, () -> closeRequest(ctx, null));
                    if (cleanup != null) tasks.recordFailure(cleanup);
                    return publication.publish(
                            candidate,
                            completion,
                            response,
                            failure,
                            synchronous,
                            deliveryReporter);
                } catch (RuntimeException | Error failed) {
                    publication.publishAfterFailure(
                            failed,
                            candidate,
                            completion,
                            response,
                            failure,
                            synchronous,
                            deliveryReporter);
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
            if (ctx.data().getScheduler() == this) {
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

    /** The algorithm owns timing; this component owns cancellation and settlement. */
    public RequestScheduler scheduler() {
        return owner;
    }
}
