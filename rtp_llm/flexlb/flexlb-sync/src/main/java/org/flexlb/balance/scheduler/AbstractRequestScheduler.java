package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.balance.preemption.VictimResolution;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
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
import static org.flexlb.balance.scheduler.ContextResult.*;

/** Owns request flows. Context methods only maintain facts across actual external operation boundaries. */
public abstract class AbstractRequestScheduler implements RequestScheduler {
    static final AtomicLong WORKER_SEQUENCE = new AtomicLong();
    protected final RequestRepository requests;
    protected final SchedulerRuntime runtime;
    protected final DecodeResources.AdmissionCapacity decodeAdmissionCapacity;
    protected final boolean queued;
    private final RecentCacheKeyTraceReporter cacheTrace;

    protected AbstractRequestScheduler(SchedulerRuntime runtime, FlexlbConfig config,
                                      DecodeResources.AdmissionCapacity capacity,
                                      RecentCacheKeyTraceReporter cacheTrace) {
        this.runtime = Objects.requireNonNull(runtime);
        requests = runtime.requests();
        queued = Objects.requireNonNull(config).isQueue();
        decodeAdmissionCapacity = Objects.requireNonNull(capacity);
        this.cacheTrace = Objects.requireNonNull(cacheTrace);
    }

    protected CompletableFuture<Response> register(RequestContext ctx, Runnable onRegistered, StrategyErrorType expiredError) {
        if (ctx == null || ctx.data().getRequest() == null) {
            return CompletableFuture.completedFuture(Response.error(StrategyErrorType.INVALID_REQUEST));
        }
        var data = ctx.data();
        var config = data.getConfig();
        if (!runtime.isAccepting()) {
            return CompletableFuture.completedFuture(Response.buildErrorResponse(StrategyErrorType.DISPATCH_FAILED,
                    "request scheduler is not accepting this request configuration"));
        }
        if (config.getDispatcher().requiresGenerateInput() && !data.hasGenerateInput()) {
            return CompletableFuture.completedFuture(Response.buildErrorResponse(StrategyErrorType.INVALID_REQUEST,
                    "missing serialized generate_input for batch dispatch"));
        }
        long expires = data.getRequestExpiresAtMs();
        if (expires <= 0L || System.currentTimeMillis() >= expires) {
            var rejected = CompletableFuture.completedFuture(Response.buildErrorResponse(expiredError,
                    "request scheduling deadline has expired before placement"));
            data.setFuture(rejected);
            return rejected;
        }
        var future = new RequestFuture((kind, response, failure) -> completeExternal(ctx, kind, response, failure));
        var result = requests.register(ctx, this, future);
        if (result != RequestRepository.RegistrationResult.REGISTERED) {
            return CompletableFuture.completedFuture(Response.buildErrorResponse(
                    result == RequestRepository.RegistrationResult.CLOSED ? StrategyErrorType.DISPATCH_FAILED
                            : StrategyErrorType.INVALID_REQUEST, result == RequestRepository.RegistrationResult.DUPLICATE_ID
                            ? "duplicate request_id: " + data.getRequestId() : "request registration rejected: " + result));
        }
        data.setEnqueueTime(System.currentTimeMillis());
        FlexlbTrace.setScheduleAttribute(data.getTraceContext(), FlexlbTrace.SCHEDULE_MODE,
                config.getDispatcher().requiresGenerateInput() ? "BATCH" : config.getScheduler().getType().name());
        future.whenComplete((response, failure) -> {
            if (failure == null && response != null && response.isSuccess()) {
                try { cacheTrace.report(ctx); }
                catch (RuntimeException error) { Logger.warn("Route tracing failed", error); }
            }
        });
        runtime.deadlines().update(ctx, ctx.snapshot().deadlines());
        onRegistered.run();
        if (requests.isClosed()) { cancel(ctx, 0L, CancelReason.SHUTDOWN); }
        else if (System.currentTimeMillis() >= expires) { cancel(ctx, 0L, CancelReason.DEADLINE_EXCEEDED); }
        return future;
    }

    @Override public final RequestState cancel(long id, long batchId, CancelReason reason) {
        Objects.requireNonNull(reason);
        var ctx = requests.findActive(id);
        if (ctx != null) {
            if (ctx.data().getScheduler() != this) { return null; }
            var result = cancel(ctx, batchId, reason);
            if (result != null) { return result; }
        }
        var terminal = requests.findTerminal(id);
        return terminal != null && terminal.owner() == this && terminal.state().matchesBatch(batchId)
                ? terminal.state() : null;
    }

    /** Cancellation freezes its first cause, then directs the owner of the real resource. */
    public final RequestState cancel(RequestContext ctx, long batchId, CancelReason reason) {
        if (ctx == null || ctx.data().getScheduler() != this || !requests.isCurrent(ctx)) { return null; }
        try (var accepted = runtime.tryRegisterRequestWork(ctx)) {
            if (accepted == null) { return null; }
            Claim<CancelPermit> claim;
            synchronized (ctx) {
                var state = ctx.snapshot();
                if (state.phase() == Phase.CLOSED) { return state.terminalRecord(); }
                String message = reason.getMessage();
                Map<String, Object> evidence = cancellationDiagnostics(ctx, reason, message);
                var error = evidence != null ? StrategyErrorType.RESOURCE_EXHAUSTED
                        : reason == CancelReason.DEADLINE_EXCEEDED ? StrategyErrorType.BATCH_SLO_EXPIRED
                        : StrategyErrorType.REQUEST_CANCELLED;
                claim = ctx.requestCancel(new Cancellation(batchId, reason, false, System.currentTimeMillis(),
                        error, evidence == null ? message : String.valueOf(evidence.get("cause"))));
                if (claim instanceof Granted<?> && evidence != null) { ctx.data().setSchedulingDiagnostics(evidence); }
            }
            if (claim instanceof Granted<CancelPermit> granted) {
                var permission = granted.value();
                switch (permission.kind()) {
                    case GLOBAL_QUEUE, WAIT_PLACEMENT -> queueControl(ctx);
                    case WORKER_QUEUE -> {
                        if (!permission.route().prefillEp().control(permission.route(), WorkerBatcher.Control.WAKE)) {
                            accepted.submit(() -> onQueueEvent(permission.route(), QueueKind.CONTROL, null, null));
                        }
                    }
                    case REMOTE -> {
                        if (permission.delivery() != null) { runtime.startDeliveryCleanup(permission.delivery()); }
                        closeRequest(ctx, new EndEvidence(permission.route(), EndKind.FRONTEND, null,
                                false, 0L, StrategyErrorType.REQUEST_CANCELLED, reason.getMessage(), false, false), null);
                    }
                    case WAIT_PREEMPTION -> { }
                }
            }
            return requests.getRequestState(ctx.data().getRequestId(), batchId);
        }
    }

    public enum QueueKind { CONTROL, EXPIRED, PREEMPTED, REJECTED }
    public void onQueueEvent(RequestRoute route, QueueKind kind, RequestRoute incoming, Throwable cause) {
        var ctx = route.ctx();
        if (ctx.data().getScheduler() != this || !requests.isCurrent(ctx)) { return; }
        if (kind == QueueKind.EXPIRED) { cancel(ctx, 0L, CancelReason.DEADLINE_EXCEEDED); return; }
        Throwable failure = null;
        if (kind == QueueKind.PREEMPTED) {
            failure = Failures.run(failure, () -> runtime.requestReporter().reportVictim(
                    route.priority(), incoming.priority(), "prefill_queued", "prefill_inflight_requests"));
        }
        var error = kind == QueueKind.PREEMPTED ? StrategyErrorType.PRIORITY_PREEMPTED : StrategyErrorType.DISPATCH_FAILED;
        String detail = kind == QueueKind.PREEMPTED ? "preempted by higher-priority request " + incoming.requestId()
                : kind == QueueKind.REJECTED ? "Worker scheduling queue rejected request: "
                + (cause == null ? "endpoint publication failed" : cause.getMessage()) : "queued request cancelled";
        if (kind == QueueKind.CONTROL) {
            synchronized (ctx) {
                var cancellation = ctx.snapshot().cancellation();
                if (cancellation == null || cancellation.operation() == null) { return; }
                ctx.completeCancel(cancellation.operation(), new CancelEvidence(CancelEvidenceKind.LOCAL_WITHDRAWN,
                        false, false), System.currentTimeMillis());
            }
        }
        failure = Failures.run(failure, () -> closeRequest(ctx, new EndEvidence(route,
                kind == QueueKind.CONTROL ? EndKind.FRONTEND : EndKind.FAILURE,
                null, false, 0L, error, detail, false, false), null));
        Failures.rethrow(failure, "queue event processing failed");
    }

    /** Match Engine identity and merge facts before submitting external follow-up work. */
    public void onEngineEvidence(RequestContext ctx, EngineEvidence evidence) {
        if (ctx == null || ctx.data().getScheduler() != this) { return; }
        try (var accepted = runtime.tryRegisterRequestWork(ctx)) {
            if (accepted == null) { return; }
            Update update = ctx.recordEngineEvidence(evidence);
            if (update.kind() == UpdateKind.STALE) { return; }
            var after = update.after();
            var delivery = after.delivery();
            boolean confirmation = delivery != null && !delivery.acknowledged() && delivery.abandonment() == null
                    && (delivery.send() == SendState.DELIVERED || delivery.send() == SendState.UNKNOWN && after.engine().decodeAccepted());
            if (after.pendingEnd() != null && after.pendingEnd().kind() == EndKind.RETIREMENT) {
                closeRequest(ctx, after.pendingEnd(), null);
            }
            if (after.preemption() == null && after.pendingEnd() == null && after.phase() == Phase.OPEN
                    && !confirmation && update.before().deadlines() == after.deadlines()) { return; }
            accepted.submit(() -> {
                Throwable failure = Failures.run(null, () -> reconcilePreemption(ctx));
                var current = ctx.snapshot();
                if (current.pendingEnd() != null) {
                    failure = Failures.run(failure, () -> closeRequest(ctx, current.pendingEnd(), null));
                }
                failure = Failures.run(failure, () -> confirmDelivery(ctx));
                failure = Failures.run(failure, () -> runtime.deadlines().update(ctx, ctx.snapshot().deadlines()));
                if (ctx.snapshot().phase() == Phase.ENDING) { failure = Failures.run(failure, () -> requestCleanup(ctx)); }
                Failures.rethrow(failure, "Engine evidence continuation failed");
            });
        } catch (Throwable failure) {
            runtime.recordFailure(failure);
            Logger.error("Engine evidence processing failed", failure);
        }
    }

    public void onDeadline(ExpirationTimer.Token exact, long nowMs) {
        var ctx = exact.owner();
        if (ctx.data().getScheduler() != this || !requests.isCurrent(ctx)) { return; }
        try (var accepted = runtime.tryRegisterRequestWork(ctx)) {
            if (accepted == null) { return; }
            Update update;
            synchronized (ctx) {
                var before = ctx.snapshot();
                CancellationFacts diagnosis = null;
                boolean inactivity = exact.kind() == ExpirationTimer.Kind.INACTIVITY
                        && before.deadlines().inactivity() == exact && nowMs >= before.inactivityExpiresAtMs();
                boolean scheduling = exact.kind() == ExpirationTimer.Kind.REQUEST && before.deadlines().request() == exact
                        && nowMs >= exact.atMs() && (before.delivery() == null || before.placement() != null);
                if (before.phase() == Phase.OPEN && before.cancellation() == null && (inactivity || scheduling)) {
                    String detail = inactivity ? "REQUEST_INACTIVE: no matching Engine request status before inactivity timeout"
                            : before.placement() != null ? "request scheduling deadline exceeded during admission"
                            : CancelReason.DEADLINE_EXCEEDED.getMessage();
                    var evidence = cancellationDiagnostics(ctx, CancelReason.DEADLINE_EXCEEDED, detail);
                    diagnosis = new CancellationFacts(CancelReason.DEADLINE_EXCEEDED,
                            evidence == null ? StrategyErrorType.BATCH_SLO_EXPIRED : StrategyErrorType.RESOURCE_EXHAUSTED,
                            evidence == null ? detail : String.valueOf(evidence.get("cause")), null);
                    if (evidence != null) { ctx.data().setSchedulingDiagnostics(evidence); }
                }
                update = ctx.recordDeadline(exact, nowMs, diagnosis);
            }
            if (update.kind() == UpdateKind.STALE) { return; }
            accepted.submit(() -> {
                var current = ctx.snapshot();
                if (nowMs < exact.atMs()) {
                    runtime.deadlines().update(ctx, current.deadlines());
                    return;
                }
                if (exact.kind() == ExpirationTimer.Kind.REQUEST) { cancel(ctx, 0L, CancelReason.DEADLINE_EXCEEDED); }
                else if (current.pendingEnd() != null) { closeRequest(ctx, current.pendingEnd(), null); }
                if (ctx.snapshot().phase() == Phase.ENDING) { requestCleanup(ctx); }
                runtime.deadlines().update(ctx, ctx.snapshot().deadlines());
            });
        }
    }

    public boolean isAdmissionOpen(long id, CompletableFuture<?> future) {
        var ctx = requests.findActive(id);
        if (requests.isClosed() || ctx == null || ctx.data().getScheduler() != this || ctx.data().getFuture() != future) { return false; }
        synchronized (ctx) {
            var state = ctx.snapshot();
            return requests.isCurrent(ctx) && state.phase() == Phase.OPEN && state.cancellation() == null;
        }
    }

    public AdmissionHandle claimAdmissionHandle(long id, CompletableFuture<?> future) {
        if (!runtime.registerAdmission()) { return null; }
        boolean transferred = false;
        try {
            var ctx = requests.findActive(id);
            if (ctx == null || ctx.data().getScheduler() != this || ctx.data().getFuture() != future) { return null; }
            synchronized (ctx) {
                var claim = requests.isCurrent(ctx) ? ctx.beginPlacement(null) : null;
                if (!(claim instanceof Granted<PlacementToken> granted)) { return null; }
                transferred = true;
                return new AdmissionHandle(granted.value(), this);
            }
        } finally { if (!transferred) { runtime.finishAdmission(); } }
    }

    interface Publication { void publish(); boolean published(); }
    PlacementResult.Status commitRoute(RequestRoute route, Publication publication) {
        var ctx = route.ctx();
        PlacementToken operation;
        synchronized (ctx) {
            var state = ctx.snapshot();
            if (requests.isClosed() || !requests.isCurrent(ctx) || ctx.data().getScheduler() != this || state.placement() == null) { return PlacementResult.Status.CLOSED; }
            operation = state.placement().operation();
            var bound = ctx.bindPlacement(operation, route, PublicationResult.PREPARED);
            if (!(bound instanceof Granted<RoutePermit>)) { return PlacementResult.Status.CLOSED; }
        }
        if (ctx.data().getWorkerEnqueueSequence() == 0L) {
            ctx.data().initializeWorkerQueue(queued ? System.currentTimeMillis() : ctx.data().getEnqueueTime(),
                    WORKER_SEQUENCE.incrementAndGet());
        }
        Throwable failure = Failures.run(null, publication::publish);
        boolean published = publication.published();
        Claim<RoutePermit> reported;
        synchronized (ctx) {
            reported = ctx.bindPlacement(operation, route, published ? PublicationResult.PUBLISHED : PublicationResult.NOT_PUBLISHED);
        }
        if (published && reported.kind() != ClaimKind.STALE) { failure = Failures.run(failure, () -> route.prefillEp().signal(org.flexlb.balance.endpoint.PrefillEndpoint.Signal.INPUT)); }
        Failures.rethrow(failure, "route publication failed");
        return reported.kind() == ClaimKind.STALE ? PlacementResult.Status.CLOSED
                : published ? PlacementResult.Status.SUCCESS : PlacementResult.Status.BLOCKED;
    }

    void completePlacement(AdmissionHandle handle, Response failure) {
        var ctx = handle.owner();
        Throwable problem = null;
        try {
            var update = ctx.completePlacement(handle.token, failure == null ? PublicationResult.PUBLISHED
                    : PublicationResult.FAILED, System.currentTimeMillis());
            if (update.kind() == UpdateKind.STALE) { return; }
            var after = update.after();
            if (handle.withdrawingRoute() != null && after.route() == handle.withdrawingRoute()) {
                problem = Failures.run(problem, () -> after.route().prefillEp().signal(org.flexlb.balance.endpoint.PrefillEndpoint.Signal.INPUT));
            }
            EndEvidence end = after.pendingEnd();
            if ((end == null || failure != null && end.kind() == EndKind.SHUTDOWN && after.cancellation() == null)
                    && (failure != null || after.route() != null && update.before().placement() != null
                    && !update.before().placement().published())) {
                StrategyErrorType error = failure == null ? StrategyErrorType.DISPATCH_FAILED
                        : java.util.Arrays.stream(StrategyErrorType.values()).filter(value -> value.getErrorCode() == failure.getCode())
                        .findFirst().orElse(StrategyErrorType.DISPATCH_FAILED);
                end = new EndEvidence(after.route(), EndKind.FAILURE, null, false, 0L,
                        error, failure == null ? "route publication did not commit"
                        : failure.getErrorMessage(), false, false);
            }
            if (end != null) { var actual = end; problem = Failures.run(problem, () -> closeRequest(ctx, actual, null)); }
            else if (after.cancellation() != null && ctx.data().getFuture().isCancelled()) {
                var cancelled = after.cancellation();
                problem = Failures.run(problem, () -> closeRequest(ctx, new EndEvidence(after.route(), EndKind.FRONTEND,
                        null, false, 0L, cancelled.errorType(), cancelled.detail(), false, false), null));
            } else if (after.cancellation() != null) { problem = Failures.run(problem, () -> cancel(ctx, 0L, after.cancellation().reason())); }
            problem = Failures.run(problem, () -> runtime.deadlines().update(ctx, ctx.snapshot().deadlines()));
            if (ctx.snapshot().phase() == Phase.ENDING) { problem = Failures.run(problem, () -> requestCleanup(ctx)); }
            if (problem != null) { runtime.recordFailure(problem); }
        } finally { runtime.finishAdmission(); }
        Failures.rethrow(problem, "placement completion failed");
    }

    /** Context validation, Decode ownership CAS and Context commit share one short critical section. */
    public DeliveryClaim claimDelivery(RequestRoute route, DeliveryClaimKind kind, long batchId,
                                       DecodeEndpoint.EngineDispatchPermit permit) {
        var ctx = route.ctx();
        Claim<DeliveryClaim> claim;
        synchronized (ctx) {
            if (!requests.isCurrent(ctx) || ctx.data().getScheduler() != this || ctx.data().getFuture().isDone()) { return null; }
            long nowMs = System.currentTimeMillis();
            var before = ctx.snapshot();
            boolean inactive = nowMs >= before.inactivityExpiresAtMs();
            boolean expired = nowMs >= ctx.data().getRequestExpiresAtMs();
            if (before.preparedDelivery() && before.cancellation() == null && (inactive || expired)) {
                String detail = inactive ? "REQUEST_INACTIVE: no matching Engine request status before inactivity timeout"
                        : "request scheduling deadline exceeded before delivery";
                var evidence = cancellationDiagnostics(ctx, CancelReason.DEADLINE_EXCEEDED, detail);
                var diagnosis = new CancellationFacts(CancelReason.DEADLINE_EXCEEDED,
                        evidence == null ? StrategyErrorType.BATCH_SLO_EXPIRED : StrategyErrorType.RESOURCE_EXHAUSTED,
                        evidence == null ? detail : String.valueOf(evidence.get("cause")), null);
                var deadline = inactive ? before.deadlines().inactivity() : before.deadlines().request();
                if (deadline != null) { ctx.recordDeadline(deadline, nowMs, diagnosis); }
                if (evidence != null) { ctx.data().setSchedulingDiagnostics(evidence); }
            }
            claim = ctx.beginDelivery(route, kind, batchId, nowMs);
            if (claim instanceof Granted<DeliveryClaim> granted) {
                boolean transferred = false;
                try {
                    if (permit != null) {
                        checkArgument(permit.belongsTo(route.decodeEp(), route.decodeReservation()), "foreign Decode dispatch permit");
                        if (permit.dispatch().status() != DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED) {
                            throw new IllegalStateException("Decode dispatch ownership lost");
                        }
                    }
                    transferred = true;
                    ctx.commitDelivery(granted.value(), true, System.currentTimeMillis());
                    return granted.value();
                } finally {
                    if (!transferred) { ctx.commitDelivery(granted.value(), false, System.currentTimeMillis()); }
                }
            }
        }
        if (System.currentTimeMillis() >= ctx.data().getRequestExpiresAtMs()) { cancel(ctx, 0L, CancelReason.DEADLINE_EXCEEDED); }
        else {
            var pending = ctx.snapshot().pendingEnd();
            if (pending != null) { closeRequest(ctx, pending, null); }
        }
        return null;
    }

    boolean tryStartSend(DeliveryClaim claim) {
        var started = claim.owner().beginSend(claim, System.currentTimeMillis());
        if (started instanceof Granted<?>) { return true; }
        if (started instanceof Rejected<DeliveryClaim> rejected && rejected.reason() == Rejection.CANCELLED) {
            var state = started.after();
            CancelReason reason = state.cancellation() != null ? state.cancellation().reason()
                    : state.delivery().abandonment() != null ? state.delivery().abandonment()
                    : state.phase() != Phase.OPEN ? CancelReason.SHUTDOWN : CancelReason.DEADLINE_EXCEEDED;
            onDeliveryEvidence(claim, new Abandonment(claim, reason, System.currentTimeMillis()));
        }
        return false;
    }

    public void onDeliveryEvidence(DeliveryClaim claim, DeliveryEvidence evidence) {
        var ctx = claim == null ? ((PreparationFailure) evidence).route().ctx() : claim.owner();
        if (ctx.data().getScheduler() != this || !requests.isCurrent(ctx)) { return; }
        try (var accepted = runtime.registerRequestWork(ctx)) {
            if (evidence instanceof PreparationFailure || evidence instanceof TransportResult result && result.result().failed()) {
                var route = claim == null ? ((PreparationFailure) evidence).route() : claim.item();
                Throwable cause = evidence instanceof TransportResult result ? result.result().cause() : null;
                String message = cause == null ? "unknown delivery failure" : cause.getMessage();
                String detail = evidence instanceof PreparationFailure failed ? failed.detail()
                        : "Delivery failed: " + (message == null || message.isBlank() ? cause.getClass().getSimpleName() : message);
                try {
                    closeRequest(ctx, new EndEvidence(route, EndKind.DELIVERY_FAILURE, null, false, 0L,
                            StrategyErrorType.DISPATCH_FAILED, detail, false, false), evidence);
                } finally {
                    if (ctx.snapshot().phase() == Phase.ENDING) { requestCleanup(ctx); }
                }
                return;
            }
            Update update;
            synchronized (ctx) {
                var before = ctx.snapshot();
                if ((evidence instanceof Prediction || evidence instanceof RouteDelivered) && before.delivery() != null
                        && before.delivery().identity() == claim && before.route() == claim.item()
                        && !before.engine().decodeAccepted() && claim.item().decodeEp() != null
                        && claim.item().decodeEp().ownership(claim.item().decodeReservation()).confirmed()) {
                    ctx.recordEngineEvidence(new EngineEvidence(claim.item(), claim.item().decodeReservation(),
                            RoleType.DECODE, EngineKind.ACTIVE, 0L, null, evidence.nowMs()));
                }
                update = ctx.recordDeliveryResult(evidence);
            }
            if (update.kind() == UpdateKind.STALE) { return; }
            var after = update.after();
            var delivery = after.delivery();
            if (delivery != null && delivery.abandonment() != null
                    && (update.before().delivery() == null || update.before().delivery().abandonment() == null)) {
                runtime.startDeliveryCleanup(delivery.identity());
            }
            if (evidence instanceof CleanupAcknowledgement ack && after.cancellation() != null
                    && (ack.kind() == CancelEvidenceKind.CLEANED || ack.kind() == CancelEvidenceKind.FENCED)) {
                closeRequest(ctx, new EndEvidence(after.route(), EndKind.FRONTEND, null, false, 0L,
                        after.cancellation().errorType(), after.cancellation().detail(), false, false), null);
            }
            if (after.pendingEnd() != null && after.pendingEnd().kind() == EndKind.INACTIVITY) {
                closeRequest(ctx, after.pendingEnd(), null);
                runtime.deadlines().update(ctx, ctx.snapshot().deadlines());
                return;
            }
            if (after.pendingEnd() != null) { closeRequest(ctx, after.pendingEnd(), null); }
            confirmDelivery(ctx);
            runtime.deadlines().update(ctx, ctx.snapshot().deadlines());
            if (ctx.snapshot().phase() == Phase.ENDING) { requestCleanup(ctx); }

        }
    }

    /** ACK selection occurs only when its accepted response task actually runs. */
    private void confirmDelivery(RequestContext ctx) {
        ResponsePublication publication = null;
        ResponseCandidate candidate = null;
        try {
            synchronized (ctx) {
                var state = ctx.snapshot();
                var delivery = state.delivery();
                if (state.phase() != Phase.OPEN || state.pendingEnd() != null || state.cancellation() != null || delivery == null || !delivery.transferred()
                        || delivery.acknowledged() || delivery.abandonment() != null
                        || !(delivery.send() == SendState.DELIVERED || delivery.send() == SendState.UNKNOWN && state.engine().decodeAccepted())) { return; }
                if (System.currentTimeMillis() >= state.inactivityExpiresAtMs()) {
                    String detail = "REQUEST_INACTIVE: no matching Engine request status before inactivity timeout";
                    var evidence = cancellationDiagnostics(ctx, CancelReason.DEADLINE_EXCEEDED, detail);
                    var diagnosis = new CancellationFacts(CancelReason.DEADLINE_EXCEEDED,
                            evidence == null ? StrategyErrorType.BATCH_SLO_EXPIRED : StrategyErrorType.RESOURCE_EXHAUSTED,
                            evidence == null ? detail : String.valueOf(evidence.get("cause")), null);
                    if (state.deadlines().inactivity() != null) {
                        ctx.recordDeadline(state.deadlines().inactivity(), System.currentTimeMillis(), diagnosis);
                    }
                    publication = null;
                    candidate = null;
                } else if (state.preemption() != null && !state.preemption().finished()) {
                    ctx.selectResponse(new ResponseCandidate(ctx, ResponseKind.CONFIRMATION, null, System.currentTimeMillis()));
                    publication = null;
                    candidate = null;
                } else {
                    publication = new ResponsePublication(runtime, ctx, true);
                    var acknowledged = ctx.selectResponse(new ResponseCandidate(ctx, ResponseKind.CONFIRMATION, null, System.currentTimeMillis()));
                    if (acknowledged.kind() != ClaimKind.GRANTED) { publication.close(); return; }
                    candidate = new ResponseCandidate(ctx, ResponseKind.ACK, null, System.currentTimeMillis());
                    if (delivery.identity().kind() == DeliveryClaimKind.BATCH_ENQUEUE && delivery.send() == SendState.DELIVERED) {
                        ctx.data().acknowledge(System.currentTimeMillis(), System.nanoTime());
                    }
                }
            }
            if (publication == null) {
                var state = ctx.snapshot();
                if (state.pendingEnd() != null) { closeRequest(ctx, state.pendingEnd(), null); }
                else { reconcilePreemption(ctx); }
                return;
            }
            var facts = ctx.snapshot().delivery();
            var response = facts.identity().item().successResponse(facts.identity().kind() == DeliveryClaimKind.BATCH_ENQUEUE);
            runtime.deadlines().update(ctx, ctx.snapshot().deadlines());
            publishSelected(publication, candidate, RequestFuture.Completion.RESPONSE, response, null, false);
        } catch (RuntimeException | Error failed) {
            if (publication != null && !publication.hasCompletionOwner() && candidate != null) {
                var prepared = publication; var selected = candidate;
                Failures.run(failed, () -> {
                    try (var fallback = new ResponsePublication(runtime, ctx, prepared.delivery)) {
                        publishSelected(fallback, selected, RequestFuture.Completion.FAILURE, null, failed, false);
                    }
                });
            }
            throw failed;
        } finally { if (publication != null) { publication.close(); } }
    }

    public Optional<PreemptionRegistration> tryClaim(DecodeResources.ReservationHandle reservation, long attempt, String detail) {
        var ctx = requests.findActive(reservation.requestId());
        if (ctx == null || ctx.data().getScheduler() != this) { return Optional.empty(); }
        synchronized (ctx) {
            var route = ctx.snapshot().route();
            if (route == null) { return Optional.empty(); }
            var target = new CancelTarget(route.prefill().getServerIp(), route.prefill().getGrpcPort());
            var claim = ctx.beginPreemption(reservation, attempt, target);
            return claim instanceof Granted<PreemptionRegistration> granted ? Optional.of(granted.value()) : Optional.empty();
        }
    }

    public record PreemptionEvent(PreemptionCancelPhase phase, boolean release, DecodeEndpoint endpoint,
                                  DecodeResources.ReservationHandle reservation, String detail) { }
    public boolean onPreemption(PreemptionRegistration attempt, PreemptionEvent event) {
        var ctx = attempt.owner();
        Update update;
        synchronized (ctx) {
            if (!requests.isCurrent(ctx) || ctx.data().getScheduler() != this) { return false; }
            var current = ctx.snapshot();
            if (current.preemption() == null || current.preemption().identity() != attempt
                    || event.endpoint() != null && (current.route().decodeEp() != event.endpoint()
                    || !Objects.equals(current.route().decodeReservation(), event.reservation()))) { return false; }
            boolean committed = event.endpoint() != null && event.endpoint().updatePreemption(attempt.attemptToken(),
                    DecodeResources.PreemptionUpdate.fenced(event.reservation())).status().applied();
            if (event.endpoint() != null && !committed) { return false; }
            var terminal = event.endpoint() == null ? null : new EndEvidence(current.route(), EndKind.PRIORITY,
                    null, false, 0L, StrategyErrorType.PRIORITY_PREEMPTED, event.detail(), false, true);
            update = ctx.completePreemption(attempt, new PreemptionReport(event.phase(), committed,
                    event.endpoint() != null, event.release(), terminal), System.currentTimeMillis());
        }
        if (update.kind() != UpdateKind.APPLIED) { return false; }
        Throwable failure = event.endpoint() == null ? null : Failures.run(null, () -> event.endpoint().publishCapacity(org.flexlb.balance.endpoint.DecodeEndpoint.CapacityEdge.BOTH));
        failure = Failures.run(failure, () -> reconcilePreemption(ctx));
        var after = ctx.snapshot();
        if (after.pendingEnd() != null) { failure = Failures.run(failure, () -> closeRequest(ctx, after.pendingEnd(), null)); }
        else { failure = Failures.run(failure, () -> confirmDelivery(ctx)); }
        if (after.cancellation() != null) { failure = Failures.run(failure, () -> cancel(ctx, 0L, after.cancellation().reason())); }
        if (ctx.snapshot().phase() == Phase.ENDING) { failure = Failures.run(failure, () -> requestCleanup(ctx)); }
        Failures.rethrow(failure, "preemption continuation failed");
        return true;
    }

    /** Reconcile the real Decode ledger first, then commit only that exact attempt's state. */
    private void reconcilePreemption(RequestContext ctx) {
        ReconciliationPermit permit;
        DecodeEndpoint decode;
        boolean capacityChanged;
        synchronized (ctx) {
            var current = ctx.snapshot();
            if (current.preemption() == null) { return; }
            var claim = ctx.beginReconciliation(current.preemption().identity());
            if (!(claim instanceof Granted<ReconciliationPermit> granted)) { return; }
            permit = granted.value();
            decode = permit.route().decodeEp();
            var terminal = permit.terminal();
            boolean decodeAlreadyApplied = terminal != null && current.close().decodeSettled();
            capacityChanged = decode != null && !decodeAlreadyApplied;
            boolean committed;
            try {
                committed = !capacityChanged || decode.updatePreemption(permit.attempt().attemptToken(),
                        terminal == null ? DecodeResources.PreemptionUpdate.active(permit.route().decodeReservation())
                                : terminal.kind() == EndKind.PRIORITY ? DecodeResources.PreemptionUpdate.canceled(permit.route().decodeReservation())
                                : DecodeResources.PreemptionUpdate.finished(permit.route().decodeReservation())).status().applied();
            } catch (RuntimeException | Error failed) {
                Failures.run(failed, () -> ctx.completePreemption(permit,
                        new PreemptionReport(null, false, false, false, terminal), System.currentTimeMillis()));
                throw failed;
            }
            ctx.completePreemption(permit, new PreemptionReport(null, committed,
                    committed && terminal != null, false, terminal), System.currentTimeMillis());
            if (!committed) { return; }
        }
        Throwable failure = capacityChanged ? Failures.run(null, () -> decode.publishCapacity(org.flexlb.balance.endpoint.DecodeEndpoint.CapacityEdge.BOTH)) : null;
        var resolution = permit.terminal() == null ? VictimResolution.Outcome.DELIVERY_RESUMED : VictimResolution.Outcome.REQUEST_END;
        if (permit.terminal() != null || permit.confirmationPending()) {
            failure = Failures.run(failure, () -> runtime.signals().resolvePreemption(permit.attempt(),
                    new VictimResolution(ctx.data().getRequestId(), resolution)));
        }
        Failures.rethrow(failure, "preemption reconciliation failed");
    }

    /** Freeze a logical outcome and transfer unique notification ownership before physical release. */
    void closeRequest(RequestContext ctx, EndEvidence evidence, DeliveryEvidence failedDelivery) {
        if (ctx.data().getScheduler() != this || !requests.isCurrent(ctx)) { return; }
        boolean deferred = evidence != null && evidence.kind() == EndKind.INACTIVITY;
        try (var execution = deferred ? runtime.registerRequestWork(ctx) : null) {
            ResponsePublication publication = null;
            boolean handedOff = false;
            EndToken token = null;
            ResponseCandidate candidate = null;
            try {
                Response response = null;
                synchronized (ctx) {
                    if (failedDelivery != null && ctx.recordDeliveryResult(failedDelivery).kind() == UpdateKind.STALE) { return; }
                    var before = ctx.snapshot();
                    if (before.phase() == Phase.CLOSED || evidence == null || evidence.route() != null && before.route() != evidence.route()) { return; }
                    EndKind ending = before.pendingEnd() == null ? evidence.kind() : before.pendingEnd().kind();
                    boolean deliveryFailure = ending == EndKind.DELIVERY_FAILURE && before.cancellation() == null;
                    boolean placementReady = before.placement() == null || deliveryFailure;
                    boolean preemptionReady = before.preemption() == null || before.preemption().finished()
                            || deliveryFailure || ending == EndKind.INACTIVITY || ending == EndKind.RETIREMENT;
                    if (before.outcome() == null && placementReady && preemptionReady
                            && before.selectedResponse() == null && !ctx.data().getFuture().isDone()) {
                        publication = new ResponsePublication(runtime, ctx, false);
                    }
                    var claim = ctx.beginClose(evidence, System.currentTimeMillis());
                    if (!(claim instanceof Granted<EndToken> granted)) { if (publication != null) { publication.close(); } return; }
                    token = granted.value();
                    var outcome = granted.after().outcome();
                    if (publication != null) {
                        candidate = new ResponseCandidate(ctx, ResponseKind.TERMINAL, outcome, System.currentTimeMillis());
                        if (outcome.kind() == EndKind.DELIVERY_FAILURE) { ctx.selectResponse(candidate); }
                        response = outcome.outcome().phase() == RequestState.Phase.COMPLETED && token.route() != null
                                ? token.route().successResponse(granted.after().delivery() != null
                                && granted.after().delivery().identity().kind() == DeliveryClaimKind.BATCH_ENQUEUE)
                                : outcome.errorType() == StrategyErrorType.RESOURCE_EXHAUSTED
                                ? Response.error(outcome.errorType(), AdmissionRejectReason.RESOURCE_EXHAUSTED, outcome.detail())
                                : Response.buildErrorResponse(outcome.errorType(), outcome.detail());
                    }
                }
                var prepared = publication; var selected = candidate; var value = response; var endToken = token;
                Runnable effects = () -> {
                    boolean reported = false, cleanupStarted = false;
                    try {
                        Throwable failure = Failures.run(null, () -> runtime.deadlines().update(ctx, ctx.snapshot().deadlines()));
                        if (endToken.preemption() != null) {
                            failure = Failures.run(failure, () -> runtime.signals().resolvePreemption(endToken.preemption(),
                                    new VictimResolution(ctx.data().getRequestId(), VictimResolution.Outcome.REQUEST_END)));
                        }
                        var delivery = ctx.snapshot().delivery();
                        if (delivery != null && delivery.abandonment() != null) {
                            failure = Failures.run(failure, () -> runtime.startDeliveryCleanup(delivery.identity()));
                        }
                        ctx.completeClose(endToken, new CloseProgress(ProgressKind.NOTIFICATIONS, false, false, failure == null, failure == null),
                                System.currentTimeMillis());
                        reported = true;
                        cleanupStarted = true;
                        failure = Failures.run(failure, () -> requestCleanup(ctx));
                        if (prepared != null) {
                            failure = Failures.run(failure, () -> publishSelected(prepared, selected, RequestFuture.Completion.RESPONSE, value, null, false));
                        }
                        if (failure != null) { runtime.recordFailure(failure); }
                        Failures.rethrow(failure, "request close failed");
                    } catch (RuntimeException | Error failed) {
                        if (!reported) {
                            Failures.run(failed, () -> ctx.completeClose(endToken, new CloseProgress(ProgressKind.NOTIFICATIONS, false, false, false, false), System.currentTimeMillis()));
                        }
                        if (prepared != null && !prepared.hasCompletionOwner() && selected != null) {
                            Failures.run(failed, () -> {
                                try (var fallback = new ResponsePublication(runtime, ctx, prepared.delivery)) {
                                    publishSelected(fallback, selected, RequestFuture.Completion.FAILURE, null, failed, false);
                                }
                            });
                        }
                        if (!cleanupStarted) { Failures.run(failed, () -> requestCleanup(ctx)); }
                        throw failed;
                    } finally { if (prepared != null) { prepared.close(); } }
                };
                if (deferred) { execution.submit(effects); handedOff = true; }
                else { handedOff = true; effects.run(); }
            } catch (RuntimeException | Error failed) {
                if (!handedOff && (execution == null || !execution.hasTaskOwner())) {
                    if (token != null) {
                        var exact = token;
                        Failures.run(failed, () -> ctx.completeClose(exact, new CloseProgress(ProgressKind.NOTIFICATIONS, false, false, false, false), System.currentTimeMillis()));
                        Failures.run(failed, () -> requestCleanup(ctx));
                    }
                    if (publication != null && !publication.hasCompletionOwner() && candidate != null) {
                        var prepared = publication; var selected = candidate;
                        Failures.run(failed, () -> {
                            try (var fallback = new ResponsePublication(runtime, ctx, prepared.delivery)) {
                                publishSelected(fallback, selected, RequestFuture.Completion.FAILURE, null, failed, false);
                            }
                        });
                    }
                }
                throw failed;
            } finally {
                if (!handedOff && (execution == null || !execution.hasTaskOwner()) && publication != null) { publication.close(); }
            }
        }
    }

    void requestCleanup(RequestContext ctx) {
        if (Thread.holdsLock(ctx)) { throw new IllegalStateException("resource cleanup requires the Context lock to be released"); }
        var state = ctx.snapshot();
        if (state.phase() == Phase.OPEN || state.placement() != null || !state.deliveryReleasable()) { return; }
        try (var accepted = runtime.registerRequestWork(ctx)) {
            boolean inline = (state.outcome() == null || state.outcome().kind() != EndKind.RETIREMENT)
                    && (state.delivery() == null || state.delivery().identity().kind() != DeliveryClaimKind.BATCH_ENQUEUE);
            accepted.submitSettlement(() -> settle(ctx), inline);
        }
    }

    private void settle(RequestContext ctx) {
        var claim = ctx.beginSettlement();
        if (!(claim instanceof Granted<ClosePermit> granted)) {
            var terminal = claim.after().terminalRecord();
            if (terminal != null) {
                if (claim.after().delivery() != null) { runtime.signals().completeDelivery(claim.after().delivery().identity()); }
                requests.archive(ctx, terminal);
                runtime.deadlines().remove(ctx);
            }
            return;
        }
        var permit = granted.value();
        var settled = releaseResources(permit);
        Throwable failure = settled.failure();
        for (int retry = 1; retry < 3 && settled.failure() != null; retry++) {
            settled = releaseResources(new ClosePermit(ctx, permit.sequence(), permit.route(), permit.releaseReason(),
                    settled.prefillSettled(), settled.decodeSettled(), permit.notificationsDone(), permit.timersDone()));
            failure = Failures.append(failure, settled.failure());
        }
        var update = ctx.completeClose(permit, new CloseProgress(ProgressKind.RELEASE, settled.prefillSettled(),
                settled.decodeSettled(), false, false), System.currentTimeMillis());
        var after = update.after();
        if (after.delivery() != null && after.close().prefillSettled() && after.close().decodeSettled()) {
            failure = Failures.run(failure, () -> runtime.signals().completeDelivery(update.after().delivery().identity()));
        }
        for (int attempt = 1; attempt < 3 && after.close().endOperation() != null
                && after.close().notificationsAttempted()
                && (!after.close().notificationsDone() || !after.close().timersDone()); attempt++) {
            var remaining = after;
            Throwable notification = remaining.close().timersDone() ? null
                    : Failures.run(null, () -> runtime.deadlines().update(ctx, ctx.snapshot().deadlines()));
            boolean timersDone = remaining.close().timersDone() || notification == null;
            if (!remaining.close().notificationsDone()) {
                if (remaining.preemption() != null && remaining.preemption().finished()) {
                    notification = Failures.run(notification, () -> runtime.signals().resolvePreemption(remaining.preemption().identity(),
                            new VictimResolution(ctx.data().getRequestId(), VictimResolution.Outcome.REQUEST_END)));
                }
                if (remaining.delivery() != null && remaining.delivery().abandonment() != null) {
                    notification = Failures.run(notification, () -> runtime.startDeliveryCleanup(remaining.delivery().identity()));
                }
            }
            failure = Failures.append(failure, notification);
            after = ctx.completeClose(remaining.close().endOperation(), new CloseProgress(ProgressKind.NOTIFICATIONS,
                    false, false, notification == null, timersDone), System.currentTimeMillis()).after();
        }
        if (after.terminalRecord() != null) {
            var terminal = after.terminalRecord();
            failure = Failures.run(failure, () -> requests.archive(ctx, terminal));
            failure = Failures.run(failure, () -> runtime.deadlines().remove(ctx));
        }
        Failures.rethrow(failure, "request settlement failed");
    }

    record Settlement(boolean prefillSettled, boolean decodeSettled, Throwable failure) { }
    static Settlement releaseResources(ClosePermit permit) {
        var route = permit.route();
        if (route == null) { return new Settlement(true, true, null); }
        boolean prefill = permit.prefillSettled();
        boolean decode = permit.decodeSettled();
        Throwable failure = null;
        try { if (!prefill) { route.prefillEp().release(null, route); prefill = true; } }
        catch (Throwable problem) { failure = problem; }
        try {
            if (!decode) {
                if (route.decodeEp() == null || route.decodeReservation() == null) { decode = true; }
                else if (permit.releaseReason() != null) {
                    var released = route.decodeEp().release(route.decodeReservation(), permit.releaseReason()).status();
                    decode = released == DecodeResources.ReservationReleaseResult.RELEASED
                            || released == DecodeResources.ReservationReleaseResult.STALE;
                } else { decode = !route.decodeEp().ownership(route.decodeReservation()).owned(); }
            }
        } catch (Throwable problem) { failure = Failures.append(failure, problem); }
        return new Settlement(prefill, decode, failure);
    }

    /** Local submission failure uses the same arbitration as external Future completion. */
    boolean publish(RequestContext ctx, Response response, boolean synchronous) {
        return completeExternal(ctx, RequestFuture.Completion.RESPONSE, response, null, synchronous);
    }

    private boolean publishSelected(ResponsePublication publication, ResponseCandidate candidate,
                                    RequestFuture.Completion completion, Response response, Throwable failure,
                                    boolean synchronous) {
        var ctx = publication.owner;
        var task = (java.util.function.Supplier<Boolean>) () -> {
            if (publication.delivery && ctx.data().getAckAtMs() > 0L) {
                var delivery = ctx.snapshot().delivery();
                try { runtime.deliveryReporter().reportLatency(DeliveryMetricsReporter.Latency.DISPATCH_ACK,
                        RoleType.PREFILL.name(), delivery.identity().item().prefillEp().getIp(),
                        Math.max(0L, ctx.data().getAckAtMs() - delivery.startedAtMs())); }
                catch (Throwable ignored) { }
            }
            var selected = ctx.selectResponse(candidate);
            return selected instanceof Granted<ResponseSelection> && publication.complete(completion, response, failure);
        };
        if (synchronous) { return publication.completeNow(task); }
        publication.submit(task);
        return true;
    }

    final boolean completeExternal(RequestContext ctx, RequestFuture.Completion completion, Response response, Throwable failure) {
        return completeExternal(ctx, completion, response, failure, true);
    }

    private boolean completeExternal(RequestContext ctx, RequestFuture.Completion completion, Response response,
                                     Throwable failure, boolean synchronous) {
        if (Thread.holdsLock(ctx)) { throw new IllegalStateException("Future callbacks require the Context lock to be released"); }
        if (ctx.data().getScheduler() != this || !requests.isCurrent(ctx)) { return false; }
        try (var accepted = runtime.tryRegisterRequestWork(ctx)) {
            if (accepted == null) { return false; }
            ResponsePublication publication;
            ResponseCandidate candidate;
            EndToken notifications = null;
            synchronized (ctx) {
                var before = ctx.snapshot();
                boolean cancelled = completion == RequestFuture.Completion.CANCELLATION;
                if (before.phase() != Phase.OPEN || before.selectedResponse() != null || ctx.data().getFuture().isDone()
                        || before.cancellation() != null && (!cancelled
                        || before.cancellation().reason() == CancelReason.DEADLINE_EXCEEDED)) { return false; }
                boolean success = completion == RequestFuture.Completion.RESPONSE && (response == null || response.isSuccess());
                if (success && before.delivery() == null || before.delivery() != null || before.engine().decodeAccepted()
                        || before.preemption() != null || before.placement() != null && !cancelled) { return false; }
                publication = new ResponsePublication(runtime, ctx, false);
                try {
                    if (cancelled) {
                        String message = CancelReason.CLIENT_CANCELLED.getMessage();
                        Map<String, Object> diagnostics = cancellationDiagnostics(ctx, CancelReason.CLIENT_CANCELLED, message);
                        if (before.cancellation() == null) {
                            var cancel = ctx.requestCancel(new Cancellation(0L, CancelReason.CLIENT_CANCELLED, true,
                                    System.currentTimeMillis(), diagnostics == null ? StrategyErrorType.REQUEST_CANCELLED
                                    : StrategyErrorType.RESOURCE_EXHAUSTED,
                                    diagnostics == null ? message : String.valueOf(diagnostics.get("cause"))));
                            if (!(cancel instanceof Granted<?>)) { publication.close(); return false; }
                        }
                        candidate = new ResponseCandidate(ctx, ResponseKind.CANCELLATION, null, System.currentTimeMillis());
                    } else {
                        String detail = failure != null ? "external future failure: " + failure.getMessage()
                                : response != null && response.getErrorMessage() != null ? response.getErrorMessage() : "external future completion";
                        var end = ctx.beginClose(new EndEvidence(before.route(), EndKind.FRONTEND, null, success,
                                0L, StrategyErrorType.DISPATCH_FAILED,
                                detail, false, false), System.currentTimeMillis());
                        if (!(end instanceof Granted<EndToken> granted)) { publication.close(); return false; }
                        notifications = granted.value();
                        candidate = new ResponseCandidate(ctx, completion == RequestFuture.Completion.FAILURE
                                ? ResponseKind.EXTERNAL_FAILURE : ResponseKind.TERMINAL, granted.after().outcome(), System.currentTimeMillis());
                    }
                    ctx.selectResponse(candidate);
                } catch (RuntimeException | Error error) { publication.close(); throw error; }
            }
            try {
                Throwable cleanup = null;
                if (completion == RequestFuture.Completion.CANCELLATION) { cleanup = Failures.run(cleanup, () -> queueControl(ctx)); }
                var after = ctx.snapshot();
                if (notifications != null) {
                    Throwable timer = Failures.run(null, () -> runtime.deadlines().update(ctx, after.deadlines()));
                    cleanup = Failures.append(cleanup, timer);
                    ctx.completeClose(notifications, new CloseProgress(ProgressKind.NOTIFICATIONS, false, false,
                            timer == null, timer == null), System.currentTimeMillis());
                    cleanup = Failures.run(cleanup, () -> requestCleanup(ctx));
                } else if (after.route() != null && !after.route().prefillEp().control(after.route(), WorkerBatcher.Control.WAKE)) {
                    accepted.submit(() -> onQueueEvent(after.route(), QueueKind.CONTROL, null, null));
                }
                if (cleanup != null) { runtime.recordFailure(cleanup); }
                return publishSelected(publication, candidate, completion, response, failure, synchronous);
            } catch (RuntimeException | Error failed) {
                if (!publication.hasCompletionOwner()) {
                    Failures.run(failed, () -> {
                        try (var fallback = new ResponsePublication(runtime, ctx, publication.delivery)) {
                            publishSelected(fallback, candidate, completion, response, failure, synchronous);
                        }
                    });
                }
                throw failed;
            } finally { publication.close(); }
        }
    }

    void closeOutstandingAndTerminalize() {
        Throwable failure = null;
        for (var ctx : requests.snapshotActive()) {
            if (ctx.data().getScheduler() == this) {
                failure = Failures.run(failure, () -> closeRequest(ctx, new EndEvidence(ctx.snapshot().route(), EndKind.SHUTDOWN,
                        null, false, 0L, StrategyErrorType.DISPATCH_FAILED, "request scheduler is shutting down", false, false), null));
            }
        }
        Failures.rethrow(failure, "request shutdown finalization failed");
    }

    void closePlacement() { }
    protected void queueControl(RequestContext ctx) {
        var state = ctx.snapshot();
        var cancellation = state.cancellation();
        if (cancellation == null || cancellation.operation() == null) { return; }
        if (state.placement() == null && state.route() == null) {
            ctx.completeCancel(cancellation.operation(), new CancelEvidence(CancelEvidenceKind.LOCAL_WITHDRAWN,
                    false, false), System.currentTimeMillis());
        }
        closeRequest(ctx, new EndEvidence(state.route(), EndKind.FRONTEND, null, false, 0L,
                cancellation.errorType(), cancellation.detail(), false, false), null);
    }
    protected Map<String, Object> cancellationDiagnostics(RequestContext ctx, CancelReason reason, String detail) { return null; }
}
