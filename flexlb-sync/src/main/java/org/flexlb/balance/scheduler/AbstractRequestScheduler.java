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
import static org.flexlb.balance.scheduler.ContextResult.CancelEvidenceKind;
import static org.flexlb.balance.scheduler.ContextResult.Cancellation;
import static org.flexlb.balance.scheduler.ContextResult.CancellationFacts;
import static org.flexlb.balance.scheduler.ContextResult.Claim;
import static org.flexlb.balance.scheduler.ContextResult.ClaimKind;
import static org.flexlb.balance.scheduler.ContextResult.CleanupAcknowledgement;
import static org.flexlb.balance.scheduler.ContextResult.CloseProgress;
import static org.flexlb.balance.scheduler.ContextResult.DeliveryEvidence;
import static org.flexlb.balance.scheduler.ContextResult.End;
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
            Update update;
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
                update =
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
                if (update.kind() == UpdateKind.APPLIED && evidence != null) {
                    ctx.data().setSchedulingDiagnostics(evidence);
                }
            }
            Throwable failure = null;
            if (update.kind() == UpdateKind.APPLIED) {
                failure = Failures.run(null, () -> queueControl(ctx, null));
                failure = Failures.run(failure, () -> closeRequest(ctx, null));
            }
            var deadlines = ctx.snapshot().deadlines();
            if (ctx.snapshot().phase() == Phase.OPEN
                    && runtime.deadlines().needsUpdate(ctx, deadlines)) {
                failure = Failures.run(failure, () -> runtime.deadlines().update(ctx, deadlines));
            }
            if (failure != null) {
                runtime.recordFailure(failure);
            }
            Failures.rethrow(failure, "cancellation effects failed");
            return requests.getRequestState(ctx.data().getRequestId(), batchId);
        }
    }

    public enum QueueKind {
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
        failure =
                Failures.run(
                        failure,
                        () ->
                                closeRequest(
                                        ctx,
                                        EndEvidence.failure(
                                                route, EndKind.FAILURE, error, detail)));
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
                            runtime.deadlines().update(ctx, current.deadlines());
                            return;
                        }
                        Throwable failure = null;
                        if (update.before().cancellation() == null
                                && update.after().cancellation() != null) {
                            failure = Failures.run(null, () -> queueControl(ctx, null));
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
                    problem = Failures.run(problem, () -> closeRequest(ctx, withdrawn));
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
                            cancellationDiagnostics(ctx, CancelReason.DEADLINE_EXCEEDED, detail);
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
                onDeliveryEvidence(
                        transferred,
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
                        runtime::recordFailure,
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
                runtime.recordFailure(failed);
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
                            cancellationDiagnostics(ctx, CancelReason.DEADLINE_EXCEEDED, detail);
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
                        publication = CloseTask.registerPublication(runtime, ctx, true);
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
                    closeRequest(ctx, state.pendingEnd());
                } else {
                    reconcilePreemption(ctx);
                    recover(
                            runtime::recordFailure,
                            () -> runtime.deadlines().update(ctx, ctx.snapshot().deadlines()));
                }
                return true;
            }
            var claim = ctx.snapshot().delivery().identity();
            var response =
                    claim.item().successResponse(claim.kind() == DeliveryClaimKind.BATCH_ENQUEUE);
            Failures.rethrow(
                    recover(
                            runtime::recordFailure,
                            () -> runtime.deadlines().update(ctx, ctx.snapshot().deadlines())),
                    "delivery deadline installation failed");
            publishSelected(publication, candidate, RESPONSE, response, null, false);
            return true;
        } catch (RuntimeException | Error failed) {
            new CloseTask(ctx, publication, candidate, null).fail(failed);
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
        try (var accepted = runtime.registerRequestWork(ctx)) {
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
                    var mutation =
                            event.endpoint() == null
                                    ? null
                                    : recover(
                                            runtime::recordFailure,
                                            () -> {
                                                return event.endpoint()
                                                        .updatePreemption(
                                                                attempt.attemptToken(),
                                                                DecodeResources.PreemptionUpdate
                                                                        .fenced(
                                                                                event
                                                                                        .reservation()));
                                            });
                    committed = mutation != null && mutation.status().applied();
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
                    recorded |=
                            ctx.recordDeliveryResult(event.cleanup()).kind() == UpdateKind.APPLIED;
                }
            }
            if (recorded) {
                if (committed) {
                    recover(
                            runtime::recordFailure,
                            () ->
                                    event.endpoint()
                                            .publishCapacity(DecodeEndpoint.CapacityEdge.BOTH));
                }
                try {
                    advanceRequest(ctx, ctx.snapshot());
                } catch (RuntimeException | Error failed) {
                    runtime.recordFailure(failed);
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
                    if (runtime.deadlines().needsUpdate(ctx, deadlines)) {
                        recover(
                                runtime::recordFailure,
                                () -> runtime.deadlines().update(ctx, deadlines));
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
            var terminal = current.preemption().pendingTerminal();
            capacityChanged =
                    decode != null && !(terminal != null && current.close().decodeSettled());
            var mutation =
                    !capacityChanged
                            ? null
                            : recover(
                                    runtime::recordFailure,
                                    () -> {
                                        return decode.updatePreemption(
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
                                                                                        .reservation()));
                                    });
            boolean committed = !capacityChanged || mutation != null && mutation.status().applied();
            ctx.completePreemption(
                    attempt,
                    new PreemptionReport(
                            null, committed, committed && terminal != null, false, terminal),
                    System.currentTimeMillis());
            if (!committed) {
                return;
            }
        }
        if (capacityChanged) {
            recover(
                    runtime::recordFailure,
                    () -> decode.publishCapacity(DecodeEndpoint.CapacityEdge.BOTH));
        }
        if (current.preemption().pendingTerminal() == null
                && current.preemption().confirmationPending()) {
            recover(
                    runtime::recordFailure,
                    () ->
                            runtime.signals()
                                    .resolvePreemption(
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
        try (var execution = deferred ? runtime.registerRequestWork(ctx) : null) {
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
                        publication = CloseTask.registerPublication(runtime, ctx, false);
                    }
                    var update = ctx.beginClose(evidence, System.currentTimeMillis());
                    if (update.after().outcome() != null) {
                        var outcome = update.after().outcome();
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
                                    update.after().delivery() != null
                                            && update.after().delivery().identity().kind()
                                                    == DeliveryClaimKind.BATCH_ENQUEUE;
                            response = CloseTask.response(outcome, before.route(), batched);
                        }
                    } else if (publication != null) {
                        publication.close();
                        publication = null;
                    }
                }
                var task = new CloseTask(ctx, publication, candidate, response);
                if (deferred) {
                    execution.submit(task);
                    handedOff = true;
                } else {
                    handedOff = true;
                    task.run();
                }
            } catch (RuntimeException | Error failed) {
                if (!handedOff && (execution == null || !execution.hasTaskOwner())) {
                    new CloseTask(ctx, publication, candidate, null).fail(failed);
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
    private final class CloseTask implements Runnable {
        private final RequestContext ctx;
        private final ResponsePublication publication;
        private final ResponseCandidate candidate;
        private final Response response;

        private CloseTask(
                RequestContext ctx,
                ResponsePublication publication,
                ResponseCandidate candidate,
                Response response) {
            this.ctx = ctx;
            this.publication = publication;
            this.candidate = candidate;
            this.response = response;
        }

        private static ResponsePublication registerPublication(
                SchedulerRuntime runtime, RequestContext ctx, boolean delivery) {
            var publication =
                    recover(
                            runtime::recordFailure,
                            () -> {
                                return new ResponsePublication(runtime, ctx, delivery);
                            });
            if (publication == null) {
                throw new IllegalStateException("frontend publication is closed");
            }
            return publication;
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
            try {
                Throwable failure = Failures.run(null, () -> scheduleClose(ctx));
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
                fail(failed);
                throw failed;
            } finally {
                if (publication != null) {
                    publication.close();
                }
            }
        }

        private static Throwable notifyOwners(AbstractRequestScheduler owner, RequestContext ctx) {
            var runtime = owner.runtime;
            var deadlines = ctx.snapshot().deadlines();
            Throwable timerFailure =
                    runtime.deadlines().needsUpdate(ctx, deadlines)
                            ? Failures.run(null, () -> runtime.deadlines().update(ctx, deadlines))
                            : null;
            var state = ctx.snapshot();
            if (state.phase() != Phase.ENDING) {
                return timerFailure;
            }
            boolean timersDone = !runtime.deadlines().needsUpdate(ctx, deadlines);
            Throwable failure = null;
            var delivery = state.delivery();
            if (!state.close().notificationsAttempted()
                    && delivery != null
                    && (delivery.abandonment() != null || state.cancellation() != null)) {
                failure = Failures.run(null, () -> runtime.dispatcher.stop(delivery.identity()));
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
                                        runtime.signals()
                                                .resolvePreemption(
                                                        preemption.identity(),
                                                        new VictimResolution(
                                                                ctx.data().getRequestId(),
                                                                VictimResolution.Outcome
                                                                        .REQUEST_END)));
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

        private boolean publishExternal(
                RequestFuture.Completion completion, Throwable failure, boolean synchronous) {
            try {
                Throwable cleanup = Failures.run(null, () -> queueControl(ctx, null));
                cleanup = Failures.run(cleanup, () -> closeRequest(ctx, null));
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

        private void fail(Throwable failed) {
            if (ctx.snapshot().phase() == Phase.OPEN) {
                recover(
                        runtime::recordFailure,
                        () -> runtime.deadlines().update(ctx, ctx.snapshot().deadlines()));
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
        }
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
            SchedulerRuntime.WorkRegistration accepted = null;
            try {
                accepted = runtime.registerRequestWork(ctx);
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
            Throwable problem = CloseTask.notifyOwners(this, ctx);
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
            failure =
                    Failures.run(
                            failure, () -> runtime.signals().completeDelivery(delivery.identity()));
        }
        if (after.terminalRecord() != null) {
            var terminal = after.terminalRecord();
            failure = Failures.run(failure, () -> requests.archive(ctx, terminal));
            failure = Failures.run(failure, () -> runtime.deadlines().remove(ctx));
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
                (java.util.function.BooleanSupplier)
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
                            return selected instanceof Granted<ResponseCandidate>
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
                            if (cancel.kind() != UpdateKind.APPLIED) {
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
                        if (end.after().outcome() == null) {
                            publication.close();
                            return false;
                        }
                        candidate =
                                new ResponseCandidate(
                                        ctx,
                                        completion == RequestFuture.Completion.FAILURE
                                                ? ResponseKind.EXTERNAL_FAILURE
                                                : ResponseKind.TERMINAL,
                                        end.after().outcome(),
                                        System.currentTimeMillis());
                    }
                    ctx.selectResponse(candidate);
                } catch (RuntimeException | Error error) {
                    publication.close();
                    throw error;
                }
            }
            return new CloseTask(ctx, publication, candidate, response)
                    .publishExternal(completion, failure, synchronous);
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
                                                        "request scheduler is shutting down")));
            }
        }
        Failures.rethrow(failure, "request shutdown finalization failed");
    }

    void closePlacement() {}

    protected boolean queueControl(RequestContext ctx, RequestRoute withdrawn) {
        return false;
    }

    protected Map<String, Object> cancellationDiagnostics(
            RequestContext ctx, CancelReason reason, String detail) {
        return null;
    }
}
