package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.balance.preemption.VictimResolution;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient.CancelReply;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.telemetry.FlexlbTrace;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;

import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicLong;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static org.flexlb.balance.scheduler.ContextResult.Abandonment;
import static org.flexlb.balance.scheduler.ContextResult.CancelEvidence;
import static org.flexlb.balance.scheduler.ContextResult.CancelEvidenceKind;
import static org.flexlb.balance.scheduler.ContextResult.CancelPermit;
import static org.flexlb.balance.scheduler.ContextResult.Cancellation;
import static org.flexlb.balance.scheduler.ContextResult.CancellationFacts;
import static org.flexlb.balance.scheduler.ContextResult.Claim;
import static org.flexlb.balance.scheduler.ContextResult.ClaimKind;
import static org.flexlb.balance.scheduler.ContextResult.CleanupAcknowledgement;
import static org.flexlb.balance.scheduler.ContextResult.ClosePermit;
import static org.flexlb.balance.scheduler.ContextResult.CloseProgress;
import static org.flexlb.balance.scheduler.ContextResult.DeliveryEvidence;
import static org.flexlb.balance.scheduler.ContextResult.End;
import static org.flexlb.balance.scheduler.ContextResult.EndEvidence;
import static org.flexlb.balance.scheduler.ContextResult.EndKind;
import static org.flexlb.balance.scheduler.ContextResult.EndToken;
import static org.flexlb.balance.scheduler.ContextResult.EngineEvidence;
import static org.flexlb.balance.scheduler.ContextResult.EngineKind;
import static org.flexlb.balance.scheduler.ContextResult.Granted;
import static org.flexlb.balance.scheduler.ContextResult.Phase;
import static org.flexlb.balance.scheduler.ContextResult.PlacementToken;
import static org.flexlb.balance.scheduler.ContextResult.Prediction;
import static org.flexlb.balance.scheduler.ContextResult.PreemptionReport;
import static org.flexlb.balance.scheduler.ContextResult.PreparationFailure;
import static org.flexlb.balance.scheduler.ContextResult.ProgressKind;
import static org.flexlb.balance.scheduler.ContextResult.PublicationResult;
import static org.flexlb.balance.scheduler.ContextResult.ReconciliationPermit;
import static org.flexlb.balance.scheduler.ContextResult.Rejected;
import static org.flexlb.balance.scheduler.ContextResult.Rejection;
import static org.flexlb.balance.scheduler.ContextResult.ResponseCandidate;
import static org.flexlb.balance.scheduler.ContextResult.ResponseKind;
import static org.flexlb.balance.scheduler.ContextResult.ResponseSelection;
import static org.flexlb.balance.scheduler.ContextResult.RouteDelivered;
import static org.flexlb.balance.scheduler.ContextResult.RoutePermit;
import static org.flexlb.balance.scheduler.ContextResult.SendState;
import static org.flexlb.balance.scheduler.ContextResult.Snapshot;
import static org.flexlb.balance.scheduler.ContextResult.TransportResult;
import static org.flexlb.balance.scheduler.ContextResult.Update;
import static org.flexlb.balance.scheduler.ContextResult.UpdateKind;
import static org.flexlb.dao.loadbalance.Response.buildErrorResponse;

/**
 * Owns request flows. Context methods only maintain facts across actual external operation
 * boundaries.
 */
public abstract class AbstractRequestScheduler implements RequestScheduler {
    static final AtomicLong WORKER_SEQUENCE = new AtomicLong();
    protected final RequestRepository requests;
    protected final SchedulerRuntime runtime;
    protected final DecodeResources.AdmissionCapacity decodeAdmissionCapacity;
    protected final boolean queued;
    private final RecentCacheKeyTraceReporter cacheTrace;

    protected AbstractRequestScheduler(
            SchedulerRuntime runtime,
            FlexlbConfig config,
            DecodeResources.AdmissionCapacity capacity,
            RecentCacheKeyTraceReporter cacheTrace) {
        this.runtime = Objects.requireNonNull(runtime);
        requests = runtime.requests();
        queued = Objects.requireNonNull(config).isQueue();
        decodeAdmissionCapacity = Objects.requireNonNull(capacity);
        this.cacheTrace = Objects.requireNonNull(cacheTrace);
    }

    protected CompletableFuture<Response> register(
            RequestContext ctx, Runnable onRegistered, StrategyErrorType expiredError) {
        if (ctx == null || ctx.data().getRequest() == null) {
            return CompletableFuture.completedFuture(
                    Response.error(StrategyErrorType.INVALID_REQUEST));
        }
        var data = ctx.data();
        var config = data.getConfig();
        if (!runtime.isAccepting()) {
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
        runtime.deadlines().update(ctx, ctx.snapshot().deadlines());
        onRegistered.run();
        if (requests.isClosed()) {
            cancel(ctx, 0L, CancelReason.SHUTDOWN);
        } else if (System.currentTimeMillis() >= expires) {
            cancel(ctx, 0L, CancelReason.DEADLINE_EXCEEDED);
        }
        return future;
    }

    @Override
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
        try (var accepted = runtime.tryRegisterRequestWork(ctx)) {
            if (accepted == null) {
                return null;
            }
            Claim<CancelPermit> claim;
            synchronized (ctx) {
                var state = ctx.snapshot();
                if (state.phase() == Phase.CLOSED) {
                    return state.terminalRecord();
                }
                String message = reason.getMessage();
                Map<String, Object> evidence = cancellationDiagnostics(ctx, reason, message);
                var error =
                        evidence != null
                                ? StrategyErrorType.RESOURCE_EXHAUSTED
                                : reason == CancelReason.DEADLINE_EXCEEDED
                                        ? StrategyErrorType.BATCH_SLO_EXPIRED
                                        : StrategyErrorType.REQUEST_CANCELLED;
                claim =
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
                if (claim instanceof Granted<?> && evidence != null) {
                    ctx.data().setSchedulingDiagnostics(evidence);
                }
            }
            if (claim instanceof Granted<CancelPermit> granted) {
                var permission = granted.value();
                switch (permission.kind()) {
                    case GLOBAL_QUEUE, WAIT_PLACEMENT -> queueControl(ctx, null);
                    case WORKER_QUEUE -> {
                        if (!permission
                                .route()
                                .prefillEp()
                                .control(permission.route(), WorkerBatcher.Control.WAKE)) {
                            accepted.submit(
                                    () ->
                                            onQueueEvent(
                                                    permission.route(),
                                                    QueueKind.CONTROL,
                                                    null,
                                                    null));
                        }
                    }
                    case REMOTE -> {
                        closeRequest(
                                ctx,
                                EndEvidence.failure(
                                        permission.route(),
                                        EndKind.FRONTEND,
                                        StrategyErrorType.REQUEST_CANCELLED,
                                        reason.getMessage()),
                                null);
                        return requests.getRequestState(ctx.data().getRequestId(), batchId);
                    }
                }
            }
            var deadlines = ctx.snapshot().deadlines();
            if (runtime.deadlines().needsUpdate(ctx, deadlines)) {
                runtime.deadlines().update(ctx, deadlines);
            }
            return requests.getRequestState(ctx.data().getRequestId(), batchId);
        }
    }

    public enum QueueKind {
        CONTROL,
        EXPIRED,
        PREEMPTED,
        REJECTED
    }

    public void onQueueEvent(
            RequestRoute route, QueueKind kind, RequestRoute incoming, Throwable cause) {
        var ctx = route.ctx();
        if (ctx.data().getScheduler() != this || !requests.isCurrent(ctx)) {
            return;
        }
        if (kind == QueueKind.EXPIRED) {
            cancel(ctx, 0L, CancelReason.DEADLINE_EXCEEDED);
            return;
        }
        Throwable failure = null;
        if (kind == QueueKind.PREEMPTED) {
            failure =
                    Failures.run(
                            failure,
                            () ->
                                    runtime.requestReporter()
                                            .reportVictim(
                                                    route.priority(),
                                                    incoming.priority(),
                                                    "prefill_queued",
                                                    "prefill_inflight_requests"));
        }
        var error =
                kind == QueueKind.PREEMPTED
                        ? StrategyErrorType.PRIORITY_PREEMPTED
                        : StrategyErrorType.DISPATCH_FAILED;
        String detail =
                kind == QueueKind.PREEMPTED
                        ? "preempted by higher-priority request " + incoming.requestId()
                        : kind == QueueKind.REJECTED
                                ? "Worker scheduling queue rejected request: "
                                        + (cause == null
                                                ? "endpoint publication failed"
                                                : cause.getMessage())
                                : "queued request cancelled";
        if (kind == QueueKind.CONTROL) {
            synchronized (ctx) {
                var cancellation = ctx.snapshot().cancellation();
                if (cancellation == null || cancellation.operation() == null) {
                    return;
                }
                ctx.completeCancel(
                        cancellation.operation(),
                        new CancelEvidence(CancelEvidenceKind.LOCAL_WITHDRAWN, false, false),
                        System.currentTimeMillis());
            }
        }
        failure =
                Failures.run(
                        failure,
                        () ->
                                closeRequest(
                                        ctx,
                                        EndEvidence.failure(
                                                route,
                                                kind == QueueKind.CONTROL
                                                        ? EndKind.FRONTEND
                                                        : EndKind.FAILURE,
                                                error,
                                                detail),
                                        null));
        Failures.rethrow(failure, "queue event processing failed");
    }

    /** Match Engine identity and merge facts before submitting external follow-up work. */
    public void onEngineEvidence(RequestContext ctx, EngineEvidence evidence) {
        if (ctx == null || ctx.data().getScheduler() != this) {
            return;
        }
        try (var accepted = runtime.tryRegisterRequestWork(ctx)) {
            if (accepted == null) {
                return;
            }
            Update update = ctx.recordEngineEvidence(evidence);
            if (update.kind() == UpdateKind.STALE) {
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
                closeRequest(ctx, after.pendingEnd(), null);
                return;
            }
            if (after.preemption() == null
                    && after.pendingEnd() == null
                    && after.phase() == Phase.OPEN
                    && !confirmation
                    && update.before().deadlines() == after.deadlines()) {
                return;
            }
            accepted.submit(() -> advanceRequest(ctx, ctx.snapshot()));
        } catch (Throwable failure) {
            runtime.recordFailure(failure);
            Logger.error("Engine evidence processing failed", failure);
        }
    }

    public void onDeadline(ExpirationTimer.Token exact, long nowMs) {
        var ctx = exact.owner();
        if (ctx.data().getScheduler() != this || !requests.isCurrent(ctx)) {
            return;
        }
        try (var accepted = runtime.tryRegisterRequestWork(ctx)) {
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
                            cancellationDiagnostics(ctx, CancelReason.DEADLINE_EXCEEDED, detail);
                    diagnosis =
                            new CancellationFacts(
                                    CancelReason.DEADLINE_EXCEEDED,
                                    evidence == null
                                            ? StrategyErrorType.BATCH_SLO_EXPIRED
                                            : StrategyErrorType.RESOURCE_EXHAUSTED,
                                    evidence == null
                                            ? detail
                                            : String.valueOf(evidence.get("cause")),
                                    null);
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
                            runtime.deadlines().update(ctx, current.deadlines());
                            return;
                        }
                        if (exact.kind() == ExpirationTimer.Kind.REQUEST) {
                            cancel(ctx, 0L, CancelReason.DEADLINE_EXCEEDED);
                        } else {
                            advanceRequest(ctx, current);
                        }
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
        if (!runtime.registerAdmission()) {
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
                runtime.finishAdmission();
            }
        }
    }

    interface Publication {
        void publish();

        boolean published();
    }

    PlacementResult.Status commitRoute(RequestRoute route, Publication publication) {
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
        if (ctx.data().getWorkerEnqueueSequence() == 0L) {
            ctx.data()
                    .initializeWorkerQueue(
                            queued ? System.currentTimeMillis() : ctx.data().getEnqueueTime(),
                            WORKER_SEQUENCE.incrementAndGet());
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
        if (published && reported.kind() != ClaimKind.STALE) {
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
        return reported.kind() == ClaimKind.STALE
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
                            previous.prefillEp().control(previous, WorkerBatcher.Control.REMOVE),
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
                problem = Failures.run(problem, () -> closeRequest(ctx, actual, null));
            } else if (after.cancellation() != null && ctx.data().getFuture().isCancelled()) {
                var cancelled = after.cancellation();
                var frontend =
                        EndEvidence.failure(
                                after.route(),
                                EndKind.FRONTEND,
                                cancelled.errorType(),
                                cancelled.detail());
                problem = Failures.run(problem, () -> closeRequest(ctx, frontend, null));
            } else if (after.cancellation() != null) {
                problem =
                        Failures.run(problem, () -> cancel(ctx, 0L, after.cancellation().reason()));
            } else if (result == PublicationResult.WITHDRAWN
                    && after.phase() == Phase.OPEN
                    && !ctx.data().getFuture().isDone()) {
                boolean requeued = false;
                try {
                    requeued = queueControl(ctx, previous);
                } catch (Throwable requeueFailure) {
                    problem = Failures.append(problem, requeueFailure);
                }
                if (requeued) {
                    problem =
                            Failures.run(
                                    problem,
                                    () ->
                                            runtime.deadlines()
                                                    .update(ctx, ctx.snapshot().deadlines()));
                } else {
                    var withdrawn = EndEvidence.withdrawnFailure("queued route requeue failed");
                    problem = Failures.run(problem, () -> closeRequest(ctx, withdrawn, null));
                }
            } else {
                problem = Failures.run(problem, () -> advanceRequest(ctx, ctx.snapshot()));
            }
            if (problem != null) {
                runtime.recordFailure(problem);
            }
        } finally {
            runtime.finishAdmission();
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
        Claim<DeliveryClaim> claim;
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
                var evidence = cancellationDiagnostics(ctx, CancelReason.DEADLINE_EXCEEDED, detail);
                var diagnosis =
                        new CancellationFacts(
                                CancelReason.DEADLINE_EXCEEDED,
                                evidence == null
                                        ? StrategyErrorType.BATCH_SLO_EXPIRED
                                        : StrategyErrorType.RESOURCE_EXHAUSTED,
                                evidence == null ? detail : String.valueOf(evidence.get("cause")),
                                null);
                var deadline =
                        inactive ? before.deadlines().inactivity() : before.deadlines().request();
                if (deadline != null) {
                    ctx.recordDeadline(deadline, nowMs, diagnosis);
                }
                if (evidence != null) {
                    ctx.data().setSchedulingDiagnostics(evidence);
                }
            }
            claim = ctx.beginDelivery(route, kind, batchId, nowMs);
            if (claim instanceof Granted<DeliveryClaim> granted) {
                boolean transferred = false;
                try {
                    if (permit != null) {
                        checkArgument(
                                permit.belongsTo(route.decodeEp(), route.decodeReservation()),
                                "foreign Decode dispatch permit");
                        if (permit.dispatch().status()
                                != DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED) {
                            throw new IllegalStateException("Decode dispatch ownership lost");
                        }
                    }
                    transferred = true;
                    ctx.commitDelivery(granted.value(), true, System.currentTimeMillis());
                    return granted.value();
                } finally {
                    if (!transferred) {
                        ctx.commitDelivery(granted.value(), false, System.currentTimeMillis());
                    }
                }
            }
        }
        if (System.currentTimeMillis() >= ctx.data().getRequestExpiresAtMs()) {
            cancel(ctx, 0L, CancelReason.DEADLINE_EXCEEDED);
        } else {
            var pending = ctx.snapshot().pendingEnd();
            if (pending != null) {
                closeRequest(ctx, pending, null);
            }
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
            onDeliveryEvidence(claim, new Abandonment(claim, reason, System.currentTimeMillis()));
        }
        return false;
    }

    public void onDeliveryEvidence(DeliveryClaim claim, DeliveryEvidence evidence) {
        var ctx = claim == null ? ((PreparationFailure) evidence).route().ctx() : claim.owner();
        if (ctx.data().getScheduler() != this || !requests.isCurrent(ctx)) {
            return;
        }
        try (var accepted = runtime.registerRequestWork(ctx)) {
            if (evidence instanceof PreparationFailure
                    || evidence instanceof TransportResult result && result.result().failed()) {
                var route = claim == null ? ((PreparationFailure) evidence).route() : claim.item();
                closeRequest(ctx, EndEvidence.deliveryFailure(route, evidence), evidence);
                return;
            }
            Update update;
            synchronized (ctx) {
                var before = ctx.snapshot();
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
            if (update.kind() == UpdateKind.STALE) {
                return;
            }
            advanceRequest(ctx, update.after());
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
                        || !delivery.transferred()
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
                            cancellationDiagnostics(ctx, CancelReason.DEADLINE_EXCEEDED, detail);
                    var diagnosis =
                            new CancellationFacts(
                                    CancelReason.DEADLINE_EXCEEDED,
                                    evidence == null
                                            ? StrategyErrorType.BATCH_SLO_EXPIRED
                                            : StrategyErrorType.RESOURCE_EXHAUSTED,
                                    evidence == null
                                            ? detail
                                            : String.valueOf(evidence.get("cause")),
                                    null);
                    if (state.deadlines().inactivity() != null) {
                        ctx.recordDeadline(
                                state.deadlines().inactivity(),
                                System.currentTimeMillis(),
                                diagnosis);
                    }
                } else {
                    if (state.preemption() == null || state.preemption().finished()) {
                        publication = new ResponsePublication(runtime, ctx, true);
                    }
                    var acknowledged =
                            ctx.selectResponse(
                                    new ResponseCandidate(
                                            ctx,
                                            ResponseKind.CONFIRMATION,
                                            null,
                                            System.currentTimeMillis()));
                    if (publication != null) {
                        if (acknowledged.kind() != ClaimKind.GRANTED) {
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
                    closeRequest(ctx, state.pendingEnd(), null);
                } else {
                    reconcilePreemption(ctx);
                    runtime.deadlines().update(ctx, ctx.snapshot().deadlines());
                }
                return true;
            }
            var claim = ctx.snapshot().delivery().identity();
            var response =
                    claim.item().successResponse(claim.kind() == DeliveryClaimKind.BATCH_ENQUEUE);
            runtime.deadlines().update(ctx, ctx.snapshot().deadlines());
            publishSelected(
                    publication,
                    candidate,
                    RequestFuture.Completion.RESPONSE,
                    response,
                    null,
                    false);
            return true;
        } catch (RuntimeException | Error failed) {
            new CloseTask(ctx, null, publication, candidate, null).fail(failed, true);
            throw failed;
        } finally {
            if (publication != null) {
                publication.close();
            }
        }
    }

    public Optional<PreemptionRegistration> tryClaim(
            DecodeResources.ReservationHandle reservation, long attempt, String detail) {
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

        static PreemptionEvent receipt(
                PreemptionRegistration attempt,
                DecodeEndpoint endpoint,
                CancelReply reply,
                String detail,
                DeliveryClaim delivery) {
            var phase =
                    switch (reply) {
                        case ACCEPTED, REQUEST_FENCED -> PreemptionCancelPhase.CANCEL_REQUESTED;
                        case NOT_FOUND -> PreemptionCancelPhase.NOT_FOUND_STALE;
                        case FAILED, UNSUPPORTED -> PreemptionCancelPhase.CANCEL_UNKNOWN;
                        case REQUEST_CLEANED -> null;
                    };
            return new PreemptionEvent(
                    phase,
                    false,
                    reply == CancelReply.REQUEST_CLEANED ? endpoint : null,
                    reply == CancelReply.REQUEST_CLEANED ? attempt.reservation() : null,
                    reply == CancelReply.REQUEST_CLEANED ? detail : null,
                    delivery == null ? null : acknowledgement(delivery, reply));
        }

        static CleanupAcknowledgement acknowledgement(DeliveryClaim delivery, CancelReply reply) {
            return new CleanupAcknowledgement(
                    delivery,
                    switch (reply) {
                        case REQUEST_CLEANED -> CancelEvidenceKind.CLEANED;
                        case REQUEST_FENCED -> CancelEvidenceKind.FENCED;
                        case ACCEPTED -> CancelEvidenceKind.ACKNOWLEDGED;
                        case NOT_FOUND -> CancelEvidenceKind.NOT_FOUND;
                        case UNSUPPORTED -> CancelEvidenceKind.UNSUPPORTED;
                        case FAILED -> CancelEvidenceKind.UNKNOWN;
                    },
                    System.currentTimeMillis());
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
        boolean applied = false, recorded = false, committed = false;
        synchronized (ctx) {
            if (!requests.isCurrent(ctx) || ctx.data().getScheduler() != this) {
                return false;
            }
            var current = ctx.snapshot();
            if (current.preemption() != null
                    && current.preemption().identity() == attempt
                    && (event.endpoint() == null
                            || current.route().decodeEp() == event.endpoint()
                                    && Objects.equals(
                                            current.route().decodeReservation(),
                                            event.reservation()))) {
                committed =
                        event.endpoint() != null
                                && event.endpoint()
                                        .updatePreemption(
                                                attempt.attemptToken(),
                                                DecodeResources.PreemptionUpdate.fenced(
                                                        event.reservation()))
                                        .status()
                                        .applied();
                var terminal =
                        !committed
                                ? null
                                : new EndEvidence(
                                        current.route(),
                                        EndKind.PRIORITY,
                                        null,
                                        false,
                                        0L,
                                        StrategyErrorType.PRIORITY_PREEMPTED,
                                        event.detail(),
                                        false,
                                        true);
                var update =
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
                recorded = update.kind() == UpdateKind.APPLIED;
                applied = recorded && (event.endpoint() == null || committed);
            }
            // A stale attempt must not discard a current delivery's cleanup proof.
            if (event.cleanup() != null) {
                recorded |= ctx.recordDeliveryResult(event.cleanup()).kind() != UpdateKind.STALE;
            }
        }
        if (!recorded) {
            return false;
        }
        Throwable failure =
                !committed
                        ? null
                        : Failures.run(
                                null,
                                () ->
                                        event.endpoint()
                                                .publishCapacity(
                                                        org.flexlb.balance.endpoint.DecodeEndpoint
                                                                .CapacityEdge.BOTH));
        failure = Failures.run(failure, () -> advanceRequest(ctx, ctx.snapshot()));
        Failures.rethrow(failure, "preemption continuation failed");
        return applied;
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
        var end =
                state.cancellation() != null
                                && state.delivery() != null
                                && state.delivery().prefillProof()
                        ? EndEvidence.failure(
                                state.route(),
                                EndKind.FRONTEND,
                                state.cancellation().errorType(),
                                state.cancellation().detail())
                        : state.pendingEnd();
        try {
            if (end != null || state.phase() == Phase.ENDING) {
                closeRequest(ctx, end, null);
            } else if (state.phase() == Phase.OPEN) {
                if (state.preemption() != null && state.cancellation() != null) {
                    cancel(ctx, 0L, state.cancellation().reason());
                } else if (!confirmDelivery(ctx)) {
                    var deadlines = ctx.snapshot().deadlines();
                    if (runtime.deadlines().needsUpdate(ctx, deadlines)) {
                        runtime.deadlines().update(ctx, deadlines);
                    }
                }
            }
        } catch (RuntimeException | Error failed) {
            failure = Failures.append(failure, failed);
        }
        Failures.rethrow(failure, "request continuation failed");
    }

    /** Reconcile the real Decode ledger first, then commit only that exact attempt's state. */
    private void reconcilePreemption(RequestContext ctx) {
        ReconciliationPermit permit;
        DecodeEndpoint decode;
        boolean capacityChanged;
        synchronized (ctx) {
            var current = ctx.snapshot();
            if (current.preemption() == null) {
                return;
            }
            var claim = ctx.beginReconciliation(current.preemption().identity());
            if (!(claim instanceof Granted<ReconciliationPermit> granted)) {
                return;
            }
            permit = granted.value();
            decode = permit.route().decodeEp();
            var terminal = permit.terminal();
            boolean decodeAlreadyApplied = terminal != null && current.close().decodeSettled();
            capacityChanged = decode != null && !decodeAlreadyApplied;
            boolean committed;
            try {
                committed =
                        !capacityChanged
                                || decode.updatePreemption(
                                                permit.attempt().attemptToken(),
                                                terminal == null
                                                        ? DecodeResources.PreemptionUpdate.active(
                                                                permit.route().decodeReservation())
                                                        : terminal.kind() == EndKind.PRIORITY
                                                                ? DecodeResources.PreemptionUpdate
                                                                        .canceled(
                                                                                permit.route()
                                                                                        .decodeReservation())
                                                                : DecodeResources.PreemptionUpdate
                                                                        .finished(
                                                                                permit.route()
                                                                                        .decodeReservation()))
                                        .status()
                                        .applied();
            } catch (RuntimeException | Error failed) {
                Failures.run(
                        failed,
                        () ->
                                ctx.completePreemption(
                                        permit,
                                        new PreemptionReport(null, false, false, false, terminal),
                                        System.currentTimeMillis()));
                throw failed;
            }
            ctx.completePreemption(
                    permit,
                    new PreemptionReport(
                            null, committed, committed && terminal != null, false, terminal),
                    System.currentTimeMillis());
            if (!committed) {
                return;
            }
        }
        Throwable failure =
                capacityChanged
                        ? Failures.run(
                                null,
                                () ->
                                        decode.publishCapacity(
                                                org.flexlb.balance.endpoint.DecodeEndpoint
                                                        .CapacityEdge.BOTH))
                        : null;
        var resolution =
                permit.terminal() == null
                        ? VictimResolution.Outcome.DELIVERY_RESUMED
                        : VictimResolution.Outcome.REQUEST_END;
        if (permit.terminal() != null || permit.confirmationPending()) {
            failure =
                    Failures.run(
                            failure,
                            () ->
                                    runtime.signals()
                                            .resolvePreemption(
                                                    permit.attempt(),
                                                    new VictimResolution(
                                                            ctx.data().getRequestId(),
                                                            resolution)));
        }
        Failures.rethrow(failure, "preemption reconciliation failed");
    }

    /** Own closing through completion or a recorded wait, including recovery after an effect fails. */
    void closeRequest(RequestContext ctx, EndEvidence evidence, DeliveryEvidence failedDelivery) {
        if (ctx.data().getScheduler() != this || !requests.isCurrent(ctx)) {
            return;
        }
        boolean deferred = evidence != null && evidence.kind() == EndKind.INACTIVITY;
        try (var execution = deferred ? runtime.registerRequestWork(ctx) : null) {
            ResponsePublication publication = null;
            boolean handedOff = false;
            EndToken token = null;
            ResponseCandidate candidate = null;
            try {
                Response response = null;
                synchronized (ctx) {
                    if (failedDelivery != null
                            && ctx.recordDeliveryResult(failedDelivery).kind()
                                    == UpdateKind.STALE) {
                        return;
                    }
                    var before = ctx.snapshot();
                    if (before.phase() == Phase.CLOSED
                            || evidence == null && before.phase() != Phase.ENDING
                            || evidence != null
                                    && evidence.route() != null
                                    && before.route() != evidence.route()) {
                        return;
                    }
                    EndKind ending =
                            before.pendingEnd() != null
                                    ? before.pendingEnd().kind()
                                    : evidence != null ? evidence.kind() : before.outcome().kind();
                    boolean deliveryFailure =
                            ending == EndKind.DELIVERY_FAILURE && before.cancellation() == null;
                    boolean placementReady = before.placement() == null || deliveryFailure;
                    boolean preemptionReady =
                            before.preemption() == null
                                    || before.preemption().finished()
                                    || deliveryFailure
                                    || ending == EndKind.FRONTEND
                                    || ending == EndKind.INACTIVITY
                                    || ending == EndKind.RETIREMENT;
                    if (before.outcome() == null
                            && placementReady
                            && preemptionReady
                            && before.selectedResponse() == null
                            && !ctx.data().getFuture().isDone()) {
                        publication = new ResponsePublication(runtime, ctx, false);
                    }
                    var claim = ctx.beginClose(evidence, System.currentTimeMillis());
                    if (claim instanceof Granted<EndToken> granted) {
                        token = granted.value();
                        var outcome = granted.after().outcome();
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
                                    granted.after().delivery() != null
                                            && granted.after().delivery().identity().kind()
                                                    == DeliveryClaimKind.BATCH_ENQUEUE;
                            response = CloseTask.response(outcome, token.route(), batched);
                        }
                    } else if (publication != null) {
                        publication.close();
                        publication = null;
                    }
                }
                var task = new CloseTask(ctx, token, publication, candidate, response);
                if (deferred) {
                    execution.submit(task);
                    handedOff = true;
                } else {
                    handedOff = true;
                    task.run();
                }
            } catch (RuntimeException | Error failed) {
                if (!handedOff && (execution == null || !execution.hasTaskOwner())) {
                    new CloseTask(ctx, token, publication, candidate, null).fail(failed, false);
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

    /** Owns close effects and response-publication cleanup after terminal selection. */
    private final class CloseTask implements Runnable {
        private final RequestContext ctx;
        private final EndToken token;
        private final ResponsePublication publication;
        private final ResponseCandidate candidate;
        private final Response response;

        private CloseTask(
                RequestContext ctx,
                EndToken token,
                ResponsePublication publication,
                ResponseCandidate candidate,
                Response response) {
            this.ctx = ctx;
            this.token = token;
            this.publication = publication;
            this.candidate = candidate;
            this.response = response;
        }

        private static Response response(End outcome, RequestRoute route, boolean batched) {
            if (outcome.outcome().phase() == RequestState.Phase.COMPLETED && route != null) {
                return route.successResponse(batched);
            }
            return outcome.errorType() == StrategyErrorType.RESOURCE_EXHAUSTED
                    ? Response.error(
                            outcome.errorType(),
                            AdmissionRejectReason.RESOURCE_EXHAUSTED,
                            outcome.detail())
                    : Response.buildErrorResponse(outcome.errorType(), outcome.detail());
        }

        @Override
        public void run() {
            boolean reported = false;
            try {
                Throwable failure = notifyOwners();
                reported = true;
                failure = Failures.run(failure, () -> requestCleanup(ctx));
                if (publication != null) {
                    failure =
                            Failures.run(
                                    failure,
                                    () ->
                                            publishSelected(
                                                    publication,
                                                    candidate,
                                                    RequestFuture.Completion.RESPONSE,
                                                    response,
                                                    null,
                                                    false));
                }
                if (failure != null) {
                    runtime.recordFailure(failure);
                }
                Failures.rethrow(failure, "request close failed");
            } catch (RuntimeException | Error failed) {
                fail(failed, reported);
                throw failed;
            } finally {
                if (publication != null) {
                    publication.close();
                }
            }
        }

        private Throwable notifyOwners() {
            var deadlines = ctx.snapshot().deadlines();
            Throwable timerFailure = null;
            for (int retry = 0;
                    retry < 3 && runtime.deadlines().needsUpdate(ctx, deadlines);
                    retry++) {
                timerFailure =
                        Failures.run(
                                timerFailure, () -> runtime.deadlines().update(ctx, deadlines));
            }
            if (token == null) {
                // An existing or blocked close has no new terminal notifications to publish.
                return timerFailure;
            }
            boolean timersDone = !runtime.deadlines().needsUpdate(ctx, deadlines);
            Throwable failure = null;
            if (token.preemption() != null) {
                failure =
                        Failures.run(
                                failure,
                                () ->
                                        runtime.signals()
                                                .resolvePreemption(
                                                        token.preemption(),
                                                        new VictimResolution(
                                                                ctx.data().getRequestId(),
                                                                VictimResolution.Outcome
                                                                        .REQUEST_END)));
            }
            var delivery = ctx.snapshot().delivery();
            if (delivery != null
                    && (delivery.abandonment() != null || ctx.snapshot().cancellation() != null)) {
                failure = Failures.run(failure, () -> runtime.dispatcher.stop(delivery.identity()));
            }
            ctx.completeClose(
                    token,
                    new CloseProgress(
                            ProgressKind.NOTIFICATIONS,
                            false,
                            false,
                            failure == null && timersDone,
                            timersDone),
                    System.currentTimeMillis());
            return Failures.append(timerFailure, failure);
        }

        private boolean publishExternal(
                SchedulerRuntime.WorkRegistration accepted,
                RequestFuture.Completion completion,
                Throwable failure,
                boolean synchronous) {
            try {
                Throwable cleanup =
                        completion == RequestFuture.Completion.CANCELLATION
                                ? Failures.run(null, () -> queueControl(ctx, null))
                                : null;
                var after = ctx.snapshot();
                if (token != null) {
                    cleanup = Failures.append(cleanup, notifyOwners());
                    cleanup = Failures.run(cleanup, () -> requestCleanup(ctx));
                } else if (after.route() != null
                        && !after.route()
                                .prefillEp()
                                .control(after.route(), WorkerBatcher.Control.WAKE)) {
                    accepted.submit(
                            () -> onQueueEvent(after.route(), QueueKind.CONTROL, null, null));
                }
                if (cleanup != null) {
                    runtime.recordFailure(cleanup);
                }
                return publishSelected(
                        publication, candidate, completion, response, failure, synchronous);
            } catch (RuntimeException | Error failed) {
                if (!publication.hasCompletionOwner()) {
                    Failures.run(
                            failed,
                            () -> {
                                try (var fallback =
                                        new ResponsePublication(
                                                runtime, ctx, publication.delivery)) {
                                    publishSelected(
                                            fallback,
                                            candidate,
                                            completion,
                                            response,
                                            failure,
                                            synchronous);
                                }
                            });
                }
                throw failed;
            } finally {
                publication.close();
            }
        }

        private void fail(Throwable failed, boolean reported) {
            if (token == null) {
                Failures.run(
                        failed, () -> Failures.rethrow(notifyOwners(), "deadline recovery failed"));
            }
            if (token != null && !reported) {
                Failures.run(
                        failed,
                        () ->
                                ctx.completeClose(
                                        token,
                                        new CloseProgress(
                                                ProgressKind.NOTIFICATIONS,
                                                false,
                                                false,
                                                false,
                                                false),
                                        System.currentTimeMillis()));
            }
            if (publication != null && !publication.hasCompletionOwner() && candidate != null) {
                Failures.run(
                        failed,
                        () -> {
                            try (var fallback =
                                    new ResponsePublication(runtime, ctx, publication.delivery)) {
                                publishSelected(
                                        fallback,
                                        candidate,
                                        RequestFuture.Completion.FAILURE,
                                        null,
                                        failed,
                                        false);
                            }
                        });
            }
            if (token != null && !reported) {
                Failures.run(failed, () -> requestCleanup(ctx));
            }
        }
    }

    void requestCleanup(RequestContext ctx) {
        if (Thread.holdsLock(ctx)) {
            throw new IllegalStateException(
                    "resource cleanup requires the Context lock to be released");
        }
        var state = ctx.snapshot();
        if (state.phase() == Phase.OPEN
                || state.placement() != null
                || !state.deliveryReleasable()) {
            return;
        }
        boolean inline =
                (state.outcome() == null || state.outcome().kind() != EndKind.RETIREMENT)
                        && (state.delivery() == null
                                || state.delivery().identity().kind()
                                        != DeliveryClaimKind.BATCH_ENQUEUE);
        Throwable failure = null;
        for (int retry = 0; retry < 3; retry++) {
            SchedulerRuntime.WorkRegistration accepted = null;
            try {
                accepted = runtime.registerRequestWork(ctx);
                accepted.submitSettlement(() -> settle(ctx), inline);
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
        Failures.rethrow(failure, "settlement submission failed");
    }

    private void settle(RequestContext ctx) {
        var claim = ctx.beginSettlement();
        if (!(claim instanceof Granted<ClosePermit> granted)) {
            var terminal = claim.after().terminalRecord();
            if (terminal != null) {
                if (claim.after().delivery() != null) {
                    runtime.signals().completeDelivery(claim.after().delivery().identity());
                }
                requests.archive(ctx, terminal);
                runtime.deadlines().remove(ctx);
            }
            return;
        }
        var permit = granted.value();
        var settled = releaseResources(permit);
        Throwable failure = settled.failure();
        for (int retry = 1; retry < 3 && settled.failure() != null; retry++) {
            settled =
                    releaseResources(
                            new ClosePermit(
                                    ctx,
                                    permit.sequence(),
                                    permit.route(),
                                    permit.releaseReason(),
                                    settled.prefillSettled(),
                                    settled.decodeSettled(),
                                    permit.notificationsDone(),
                                    permit.timersDone()));
            failure = Failures.append(failure, settled.failure());
        }
        var update =
                ctx.completeClose(
                        permit,
                        new CloseProgress(
                                ProgressKind.RELEASE,
                                settled.prefillSettled(),
                                settled.decodeSettled(),
                                false,
                                false),
                        System.currentTimeMillis());
        var after = update.after();
        var delivery = after.delivery();
        if (delivery != null && after.close().prefillSettled() && after.close().decodeSettled()) {
            failure =
                    Failures.run(
                            failure, () -> runtime.signals().completeDelivery(delivery.identity()));
        }
        for (int attempt = 1;
                attempt < 3
                        && after.close().endOperation() != null
                        && after.close().notificationsAttempted()
                        && (!after.close().notificationsDone() || !after.close().timersDone());
                attempt++) {
            var remaining = after;
            Throwable notification =
                    remaining.close().timersDone()
                            ? null
                            : Failures.run(
                                    null,
                                    () ->
                                            runtime.deadlines()
                                                    .update(ctx, ctx.snapshot().deadlines()));
            boolean timersDone = remaining.close().timersDone() || notification == null;
            var preemption = remaining.preemption();
            if (!remaining.close().notificationsDone()
                    && preemption != null
                    && preemption.finished()) {
                var resolution =
                        new VictimResolution(
                                ctx.data().getRequestId(), VictimResolution.Outcome.REQUEST_END);
                notification =
                        Failures.run(
                                notification,
                                () ->
                                        runtime.signals()
                                                .resolvePreemption(
                                                        preemption.identity(), resolution));
            }
            failure = Failures.append(failure, notification);
            after =
                    ctx.completeClose(
                                    remaining.close().endOperation(),
                                    new CloseProgress(
                                            ProgressKind.NOTIFICATIONS,
                                            false,
                                            false,
                                            notification == null,
                                            timersDone),
                                    System.currentTimeMillis())
                            .after();
        }
        if (after.terminalRecord() != null) {
            var terminal = after.terminalRecord();
            failure = Failures.run(failure, () -> requests.archive(ctx, terminal));
            failure = Failures.run(failure, () -> runtime.deadlines().remove(ctx));
        }
        Failures.rethrow(failure, "request settlement failed");
    }

    record Settlement(boolean prefillSettled, boolean decodeSettled, Throwable failure) {}

    static Settlement releaseResources(ClosePermit permit) {
        var route = permit.route();
        if (route == null) {
            return new Settlement(true, true, null);
        }
        boolean prefill = permit.prefillSettled();
        boolean decode = permit.decodeSettled();
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
                } else if (permit.releaseReason() != null) {
                    var released =
                            route.decodeEp()
                                    .release(route.decodeReservation(), permit.releaseReason())
                                    .status();
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
    boolean publish(RequestContext ctx, Response response, boolean synchronous) {
        return completeExternal(
                ctx, RequestFuture.Completion.RESPONSE, response, null, synchronous);
    }

    private boolean publishSelected(
            ResponsePublication publication,
            ResponseCandidate candidate,
            RequestFuture.Completion completion,
            Response response,
            Throwable failure,
            boolean synchronous) {
        var ctx = publication.owner;
        var task =
                (java.util.function.Supplier<Boolean>)
                        () -> {
                            if (publication.delivery && ctx.data().getAckAtMs() > 0L) {
                                var delivery = ctx.snapshot().delivery();
                                try {
                                    runtime.deliveryReporter()
                                            .reportLatency(
                                                    DeliveryMetricsReporter.Latency.DISPATCH_ACK,
                                                    RoleType.PREFILL.name(),
                                                    delivery.identity().item().prefillEp().getIp(),
                                                    Math.max(
                                                            0L,
                                                            ctx.data().getAckAtMs()
                                                                    - delivery.startedAtMs()));
                                } catch (Throwable ignored) {
                                }
                            }
                            var selected = ctx.selectResponse(candidate);
                            return selected instanceof Granted<ResponseSelection>
                                    && publication.complete(completion, response, failure);
                        };
        if (synchronous) {
            return publication.completeNow(task);
        }
        publication.submit(task);
        return true;
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
        if (ctx.data().getScheduler() != this || !requests.isCurrent(ctx)) {
            return false;
        }
        try (var accepted = runtime.tryRegisterRequestWork(ctx)) {
            if (accepted == null) {
                return false;
            }
            ResponsePublication publication;
            ResponseCandidate candidate;
            EndToken notifications = null;
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
                        completion == RequestFuture.Completion.RESPONSE
                                && (response == null || response.isSuccess());
                if (success
                        || before.delivery() != null
                        || before.engine().decodeAccepted()
                        || before.preemption() != null
                        || before.placement() != null && !cancelled) {
                    return false;
                }
                publication = new ResponsePublication(runtime, ctx, false);
                try {
                    if (cancelled) {
                        String message = CancelReason.CLIENT_CANCELLED.getMessage();
                        var diagnostics =
                                cancellationDiagnostics(
                                        ctx, CancelReason.CLIENT_CANCELLED, message);
                        if (before.cancellation() == null) {
                            var cancel =
                                    ctx.requestCancel(
                                            Cancellation.frontend(
                                                    System.currentTimeMillis(),
                                                    diagnostics == null
                                                            ? null
                                                            : String.valueOf(
                                                                    diagnostics.get("cause"))));
                            if (!(cancel instanceof Granted<?>)) {
                                publication.close();
                                return false;
                            }
                        }
                        candidate =
                                new ResponseCandidate(
                                        ctx,
                                        ResponseKind.CANCELLATION,
                                        null,
                                        System.currentTimeMillis());
                    } else {
                        var end =
                                ctx.beginClose(
                                        EndEvidence.externalFailure(
                                                before.route(), response, failure),
                                        System.currentTimeMillis());
                        if (!(end instanceof Granted<EndToken> granted)) {
                            publication.close();
                            return false;
                        }
                        notifications = granted.value();
                        candidate =
                                new ResponseCandidate(
                                        ctx,
                                        completion == RequestFuture.Completion.FAILURE
                                                ? ResponseKind.EXTERNAL_FAILURE
                                                : ResponseKind.TERMINAL,
                                        granted.after().outcome(),
                                        System.currentTimeMillis());
                    }
                    ctx.selectResponse(candidate);
                } catch (RuntimeException | Error error) {
                    publication.close();
                    throw error;
                }
            }
            return new CloseTask(ctx, notifications, publication, candidate, response)
                    .publishExternal(accepted, completion, failure, synchronous);
        }
    }

    void closeOutstandingAndTerminalize() {
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
                                                        "request scheduler is shutting down"),
                                                null));
            }
        }
        Failures.rethrow(failure, "request shutdown finalization failed");
    }

    void closePlacement() {}

    protected boolean queueControl(RequestContext ctx, RequestRoute withdrawn) {
        if (withdrawn != null) {
            return false;
        }
        var state = ctx.snapshot();
        var cancellation = state.cancellation();
        if (cancellation == null || cancellation.operation() == null) {
            return false;
        }
        if (state.placement() == null && state.route() == null) {
            ctx.completeCancel(
                    cancellation.operation(),
                    new CancelEvidence(CancelEvidenceKind.LOCAL_WITHDRAWN, false, false),
                    System.currentTimeMillis());
        }
        closeRequest(
                ctx,
                EndEvidence.failure(
                        state.route(),
                        EndKind.FRONTEND,
                        cancellation.errorType(),
                        cancellation.detail()),
                null);
        return true;
    }

    protected Map<String, Object> cancellationDiagnostics(
            RequestContext ctx, CancelReason reason, String detail) {
        return null;
    }
}
