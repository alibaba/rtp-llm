package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

import static org.flexlb.balance.scheduler.ContextResult.*;
import static com.google.common.math.LongMath.saturatedAdd;

/** Atomic request facts. Scheduler alone executes transport, publication, ledger and shutdown flows. */
public final class RequestContext {
    private static final AtomicLong INCARNATIONS = new AtomicLong();
    private static final long HANDOFF_GRACE_MS = 10_000L;
    private final RequestData data;
    private Identity key;
    private static final Snapshot EMPTY = new Snapshot(null, 0L, Phase.OPEN, null, null, null,
            new EngineFacts(false, 0L, false, 0L, false, 0L), null, null, null, null, null,
            new CloseFacts(null, null, true, true, false, false, false, false),
            new ExpirationTimer.Plan(0L, null, null, null), null, 0L, 0L, true, false, true, true, 0L);
    private Phase phase = Phase.OPEN;
    private RequestRoute route;
    private PlacementFacts placement;
    private DeliveryFacts delivery;
    private EngineFacts engine = EMPTY.engine();
    private PreemptionFacts preemption;
    private CancellationFacts cancellation;
    private End outcome;
    private EndEvidence pendingEnd;
    private ResponseSelection selected;
    private CloseFacts close = EMPTY.close();
    private ExpirationTimer.Plan deadlines = EMPTY.deadlines();
    private RequestState terminalRecord;
    private Snapshot state;
    private Update cachedNoChange;
    private long sequence;
    private long createdAtMs;
    private long lastWorkerAtMs;
    private long absoluteDeadline;
    private boolean requestDeadlineConsumed;
    private long inactivityMs;

    public RequestContext(FlexlbConfig config) {
        data = new RequestData(Objects.requireNonNull(config));
        state = new Snapshot(EMPTY.facts(), 0L, 0L, 0L);
    }

    public RequestData data() { return data; }

    public synchronized Update register(long requestId, long nowMs, long expiresAtMs, long inactivityTimeoutMs) {
        var before = state;
        if (key != null) { return update(before, before); }
        if (inactivityTimeoutMs <= 0L) { throw new IllegalArgumentException("positive inactivity timeout required"); }
        key = new Identity(requestId, INCARNATIONS.incrementAndGet());
        createdAtMs = lastWorkerAtMs = nowMs;
        absoluteDeadline = expiresAtMs;
        inactivityMs = inactivityTimeoutMs;
        var after = refresh(nowMs);
        return update(before, after);
    }

    public synchronized Snapshot snapshot() { return state; }

    public synchronized Claim<PlacementToken> beginPlacement(RequestRoute withdrawing) {
        var before = state;
        if (key == null) { return new Rejected<>(before, before, Rejection.NOT_REGISTERED); }
        if (phase != Phase.OPEN) { return new Rejected<>(before, before, Rejection.CLOSED); }
        if (cancellation != null) { return new Rejected<>(before, before, Rejection.CANCELLED); }
        if (placement != null) { return new Blocked<>(before, before, Blocker.PLACEMENT); }
        if (preemption != null) { return new Blocked<>(before, before, Blocker.PREEMPTION); }
        if (withdrawing == null ? route != null : route != withdrawing || delivery != null || engine.decodeAccepted()) {
            return new Rejected<>(before, before, Rejection.INVALID_STATE);
        }
        var token = new PlacementToken(this, ++sequence, withdrawing);
        placement = new PlacementFacts(token, withdrawing, withdrawing != null, null, null, false);
        return new Granted<>(before, refresh(before.updatedAtMs()), token);
    }

    public synchronized Claim<RoutePermit> bindPlacement(PlacementToken exact, RequestRoute acquired, PublicationResult result) {
        var before = state;
        Objects.requireNonNull(acquired);
        if (placement == null || placement.operation() != exact || exact.owner() != this) { return new Stale<>(before); }
        if (acquired.requestId() != key.requestId() || acquired.ctx() != this) { return new Stale<>(before); }
        if (result != PublicationResult.PREPARED) {
            if (route != acquired || placement.acquired() != acquired) { return new Stale<>(before); }
            placement = new PlacementFacts(exact, acquired, result == PublicationResult.PUBLISHED,
                    placement.pendingTerminal(), placement.pendingRetirement(), placement.inactivityExpired());
            if (result == PublicationResult.WITHDRAWN || result == PublicationResult.ROLLBACK_COMPLETED) {
                placement = new PlacementFacts(exact, null, false, placement.pendingTerminal(),
                        placement.pendingRetirement(), placement.inactivityExpired());
                route = null;
                close = new CloseFacts(close.endOperation(), close.releaseOperation(), true, true,
                        close.notificationsDone(), close.timersDone(), close.expired(), close.notificationsAttempted());
            }
            return new Granted<>(before, refresh(before.updatedAtMs()), new RoutePermit(this, exact, acquired,
                    result == PublicationResult.PUBLISHED && cancellation == null && phase == Phase.OPEN));
        }
        if (placement.acquired() != null && placement.acquired() != acquired) {
            return new Rejected<>(before, before, Rejection.INVALID_STATE);
        }
        placement = new PlacementFacts(exact, acquired, false, placement.pendingTerminal(),
                placement.pendingRetirement(), placement.inactivityExpired());
        route = acquired;
        close = new CloseFacts(close.endOperation(), close.releaseOperation(), false,
                acquired.decodeEp() == null || acquired.decodeReservation() == null,
                close.notificationsDone(), close.timersDone(), close.expired(), close.notificationsAttempted());
        boolean publishable = phase == Phase.OPEN && cancellation == null;
        var after = refresh(before.updatedAtMs());
        var permit = new RoutePermit(this, exact, acquired, publishable);
        return publishable ? new Granted<>(before, after, permit)
                : new Rejected<>(before, after, Rejection.CANCELLED);
    }

    public synchronized Update completePlacement(PlacementToken exact, PublicationResult result, long nowMs) {
        var before = state;
        if (placement == null || placement.operation() != exact) { return new Update(UpdateKind.STALE, before, before); }
        var retained = placement;
        placement = null;
        if (result == PublicationResult.WITHDRAWN || result == PublicationResult.ROLLBACK_COMPLETED) {
            route = null;
            delivery = null;
            close = new CloseFacts(close.endOperation(), close.releaseOperation(), true, true,
                        close.notificationsDone(), close.timersDone(), close.expired(), close.notificationsAttempted());
        }
        boolean rolledBack = route == null && (retained.acquired() == null
                || result == PublicationResult.ROLLBACK_COMPLETED || result == PublicationResult.WITHDRAWN)
                && close.prefillSettled() && close.decodeSettled();
        if (rolledBack && pendingEnd != null) {
            var end = pendingEnd;
            pendingEnd = new EndEvidence(null, end.kind(), end.role(), end.success(), end.errorCode(),
                    end.errorType(), end.detail(), end.prefillSettled(), end.decodeSettled());
        }
        boolean admissionFailure = result == PublicationResult.FAILED && outcome == null && cancellation == null;
        if (admissionFailure && pendingEnd != null && pendingEnd.kind() == EndKind.SHUTDOWN) { pendingEnd = null; }
        if (retained.pendingTerminal() != null
                && !(admissionFailure && retained.pendingTerminal().kind() == EndKind.SHUTDOWN)) {
            var end = retained.pendingTerminal();
            retainEnd(rolledBack ? new EndEvidence(null, end.kind(), end.role(), end.success(), end.errorCode(),
                    end.errorType(), end.detail(), end.prefillSettled(), end.decodeSettled()) : end, nowMs);
        }
        if (pendingEnd == null && outcome == null && retained.pendingRetirement() != null) {
            var retired = retained.pendingRetirement();
            retainEnd(new EndEvidence(rolledBack ? null : retired.route(), EndKind.RETIREMENT, retired.role(), false, 0L,
                    StrategyErrorType.DISPATCH_FAILED, retired.detail(), retired.role().supportsPrefill(),
                    retired.role() == RoleType.DECODE || retired.role() == RoleType.PDFUSION), nowMs);
        }
        if (outcome == null && retained.inactivityExpired()) {
            retainEnd(new EndEvidence(route, EndKind.INACTIVITY, null, false, 0L,
                    StrategyErrorType.BATCH_SLO_EXPIRED, "request inactive during placement", false, false), nowMs);
        }
        var after = refresh(nowMs);
        return update(before, after);
    }

    /** Prepare identity only. The endpoint CAS must succeed before commitDelivery exposes a handoff. */
    public synchronized Claim<DeliveryClaim> beginDelivery(RequestRoute exact, DeliveryClaimKind kind, long batchId, long nowMs) {
        var before = state;
        if (phase != Phase.OPEN || route == null || route != exact || delivery != null || cancellation != null || pendingEnd != null) {
            return new Rejected<>(before, before, Rejection.INVALID_STATE);
        }
        if (placement != null && (!placement.published() || placement.operation().withdrawing() != null
                || placement.pendingTerminal() != null || placement.pendingRetirement() != null)) {
            return new Blocked<>(before, before, Blocker.PLACEMENT);
        }
        if (preemption != null) { return new Blocked<>(before, before, Blocker.PREEMPTION); }
        if (absoluteDeadline > 0L && nowMs >= absoluteDeadline || nowMs - lastWorkerAtMs >= inactivityMs) {
            if (cancellation == null) {
                cancellation = new CancellationFacts(CancelReason.DEADLINE_EXCEEDED,
                        StrategyErrorType.BATCH_SLO_EXPIRED, "request expired before delivery", null);
            }
            retainEnd(new EndEvidence(route, EndKind.INACTIVITY, null, false, 0L,
                    StrategyErrorType.BATCH_SLO_EXPIRED, cancellation.detail(), false, false), nowMs);
            return new Rejected<>(before, refresh(nowMs), Rejection.CANCELLED);
        }
        var claim = new DeliveryClaim(this, ++sequence, exact, kind, batchId);
        delivery = new DeliveryFacts(claim, SendState.NOT_STARTED, false, null, false, false,
                exact.decodeEp() == null || exact.decodeReservation() == null, false, nowMs, 0L);
        return new Granted<>(before, refresh(nowMs), claim);
    }

    public synchronized Update commitDelivery(DeliveryClaim exact, boolean transferred, long nowMs) {
        var before = state;
        if (delivery == null || delivery.identity() != exact) { return new Update(UpdateKind.STALE, before, before); }
        if (delivery.transferred()) { return update(before, before); }
        if (!transferred) { delivery = null; }
        else {
            delivery = new DeliveryFacts(exact, delivery.send(), delivery.acknowledged(), delivery.abandonment(),
                    delivery.executionFinished(), delivery.prefillProof(), delivery.decodeProof(), true,
                    delivery.startedAtMs(), delivery.acknowledgedAtMs());
        }
        var after = refresh(nowMs);
        return update(before, after);
    }

    public synchronized Claim<DeliveryClaim> beginSend(DeliveryClaim exact, long nowMs) {
        var before = state;
        if (delivery == null || delivery.identity() != exact) { return new Stale<>(before); }
        if (!delivery.transferred() || delivery.send() != SendState.NOT_STARTED
                || exact.kind() != DeliveryClaimKind.BATCH_ENQUEUE) {
            return new Rejected<>(before, before, Rejection.INVALID_STATE);
        }
        if (phase != Phase.OPEN || cancellation != null || delivery.abandonment() != null
                || pendingEnd != null || placement != null && (placement.pendingTerminal() != null || placement.pendingRetirement() != null)
                || preemption != null && preemption.pendingTerminal() != null
                || absoluteDeadline > 0L && nowMs >= absoluteDeadline || nowMs - lastWorkerAtMs >= inactivityMs) {
            return new Rejected<>(before, before, Rejection.CANCELLED);
        }
        delivery = new DeliveryFacts(exact, SendState.SENDING, delivery.acknowledged(), null,
                delivery.executionFinished(), delivery.prefillProof(), delivery.decodeProof(), true,
                delivery.startedAtMs(), delivery.acknowledgedAtMs());
        return new Granted<>(before, refresh(nowMs), exact);
    }

    public synchronized Update recordDeliveryResult(DeliveryEvidence evidence) {
        var before = state;
        Objects.requireNonNull(evidence);
        var exact = evidence.delivery();
        if (evidence instanceof PreparationFailure failed && exact == null) {
            if (route != failed.route() || phase != Phase.OPEN || delivery != null) {
                return new Update(UpdateKind.STALE, before, before);
            }
            retainEnd(new EndEvidence(route, EndKind.DELIVERY_FAILURE, null, false, 0L,
                    StrategyErrorType.DISPATCH_FAILED, failed.detail(), false, false), failed.nowMs());
            var after = refresh(failed.nowMs());
            return update(before, after);
        }
        if (delivery == null || delivery.identity() != exact || phase == Phase.CLOSED) {
            return new Update(UpdateKind.STALE, before, before);
        }
        if ((evidence instanceof Prediction || evidence instanceof RouteDelivered) && phase != Phase.OPEN) {
            return new Update(UpdateKind.STALE, before, before);
        }
        if (evidence instanceof RouteDelivered && exact.kind() != DeliveryClaimKind.ROUTE_DECISION) {
            throw new IllegalArgumentException("route completion requires route delivery identity");
        }
        var previous = delivery;
        var send = delivery.send();
        var abandoned = delivery.abandonment();
        boolean p = delivery.prefillProof(), d = delivery.decodeProof();
        boolean executed = delivery.executionFinished();
        if (evidence instanceof TransportResult transport) {
            if (exact.kind() != DeliveryClaimKind.BATCH_ENQUEUE) { throw new IllegalArgumentException("transport result requires batch delivery identity"); }
            var actual = switch (transport.result().status()) {
                case NOT_SENT -> SendState.NOT_SENT;
                case PREFILL_REJECTED -> SendState.REJECTED;
                case DELIVERED -> SendState.DELIVERED;
                case UNCERTAIN -> SendState.UNKNOWN;
            };
            if (send != SendState.NOT_STARTED && send != SendState.SENDING) {
                if (send == actual) { return update(before, before); }
                throw new IllegalStateException("conflicting result after sender completion");
            }
            send = actual;
            if (transport.result().failed()) {
                boolean releaseAfterExit = exact.kind() == DeliveryClaimKind.ROUTE_DECISION || send == SendState.NOT_SENT
                        || (abandoned == null ? executed : p && d);
                if (!releaseAfterExit && abandoned == null) {
                    abandoned = cancellation == null ? CancelReason.CLIENT_CANCELLED : cancellation.reason();
                }
                String detail = transport.result().cause().getMessage();
                if (detail == null || detail.isBlank()) { detail = transport.result().cause().getClass().getSimpleName(); }
                retainEnd(new EndEvidence(route, EndKind.DELIVERY_FAILURE, null, false, 0L,
                        StrategyErrorType.DISPATCH_FAILED, "Delivery failed: " + detail,
                        false, false), evidence.nowMs());
            }
        } else if (evidence instanceof RouteDelivered) { send = SendState.DELIVERED; }
        else if (evidence instanceof Abandonment abandonment) {
            if (abandoned == null && !canRelease()) { abandoned = abandonment.reason(); }
            if (exact.kind() == DeliveryClaimKind.ROUTE_DECISION && send == SendState.NOT_STARTED) {
                send = SendState.ROUTE_CLOSED;
            }
        } else if (evidence instanceof CleanupAcknowledgement ack) {
            p |= ack.kind() == CancelEvidenceKind.CLEANED || ack.kind() == CancelEvidenceKind.FENCED;
            d |= ack.kind() == CancelEvidenceKind.CLEANED;
        } else if (evidence instanceof PreparationFailure failed) {
            if (send == SendState.SENDING) { throw new IllegalStateException("preparation failure crossed send boundary"); }
            if (send == SendState.NOT_STARTED) { send = SendState.NOT_SENT; }
            retainEnd(new EndEvidence(route, EndKind.DELIVERY_FAILURE, null, false, 0L,
                    StrategyErrorType.DISPATCH_FAILED, failed.detail(), false, false), failed.nowMs());
        }
        if (evidence instanceof Prediction || evidence instanceof RouteDelivered) {
            if (engine.predictionSequence() == exact.sequence()) {
                throw new IllegalStateException("delivery prediction already consumed");
            }
            long at = evidence instanceof Prediction predicted ? predicted.predictedAtMs()
                    : ((RouteDelivered) evidence).predictedAtMs();
            if (at < 0L) { throw new IllegalArgumentException("prediction timestamp must be non-negative"); }
            long visibility = engine.decodeAccepted() || engine.prefillObserved() && engine.prefillCompletedAtMs() == 0L
                    ? 0L : engine.prefillCompletedAtMs() > 0L ? saturatedAdd(engine.prefillCompletedAtMs(), HANDOFF_GRACE_MS) : at;
            engine = new EngineFacts(engine.prefillObserved(), engine.prefillCompletedAtMs(), engine.decodeAccepted(),
                    visibility, false, exact.sequence());
        }
        if (send != previous.send() || abandoned != previous.abandonment() || p != previous.prefillProof()
                || d != previous.decodeProof() || executed != previous.executionFinished()) {
            delivery = new DeliveryFacts(exact, send, previous.acknowledged(), abandoned, executed, p, d,
                    previous.transferred(), previous.startedAtMs(), previous.acknowledgedAtMs());
        }
        var after = refresh(evidence.nowMs());
        return update(before, after);
    }

    public synchronized Update recordEngineEvidence(EngineEvidence evidence) {
        var before = state;
        if (!matches(evidence.route(), evidence.reservation(), evidence.role()) || phase == Phase.CLOSED) {
            return new Update(UpdateKind.STALE, before, before);
        }
        boolean isDecode = evidence.role() == RoleType.DECODE;
        boolean terminal = evidence.kind() == EngineKind.COMPLETED || evidence.kind() == EngineKind.FAILED
                || evidence.kind() == EngineKind.PRIORITY_CANCELLED || evidence.kind() == EngineKind.RETIRED;
        boolean decodeAccepted = engine.decodeAccepted() || isDecode && evidence.kind() != EngineKind.RETIRED;
        boolean pObserved = engine.prefillObserved() || !isDecode && evidence.kind() == EngineKind.ACTIVE;
        long pCompleted = !isDecode && evidence.kind() == EngineKind.COMPLETED && engine.prefillCompletedAtMs() == 0L
                ? evidence.nowMs() : engine.prefillCompletedAtMs();
        long visibility = decodeAccepted || pObserved && pCompleted == 0L ? 0L
                : pCompleted > 0L && route.decodeEp() != null && engine.predictionSequence() != 0L
                        ? saturatedAdd(pCompleted, HANDOFF_GRACE_MS) : engine.visibilityAtMs();
        if (pObserved != engine.prefillObserved() || pCompleted != engine.prefillCompletedAtMs()
                || decodeAccepted != engine.decodeAccepted() || visibility != engine.visibilityAtMs()) {
            engine = new EngineFacts(pObserved, pCompleted, decodeAccepted, visibility, false, engine.predictionSequence());
        }
        if (evidence.kind() != EngineKind.RETIRED) { lastWorkerAtMs = Math.max(lastWorkerAtMs, evidence.nowMs()); }
        if (preemption != null && !preemption.finished() && !preemption.restorationObserved()
                && (isDecode ? evidence.kind() == EngineKind.ALLOCATED : evidence.kind() == EngineKind.ACTIVE)) {
            preemption = new PreemptionFacts(preemption.identity(), preemption.phase(), preemption.cancelAcknowledged(),
                    false, preemption.pendingTerminal(), preemption.confirmationPending(), true, preemption.reconciliation());
        }
        if (terminal && delivery != null) {
            var old = delivery;
            delivery = new DeliveryFacts(old.identity(), old.send(), old.acknowledged(), old.abandonment(),
                    old.executionFinished() || isDecode || evidence.role() == RoleType.PDFUSION,
                    old.prefillProof() || !isDecode, old.decodeProof() || isDecode,
                    old.transferred(), old.startedAtMs(), old.acknowledgedAtMs());
        }
        if (terminal) {
            close = new CloseFacts(close.endOperation(), close.releaseOperation(), close.prefillSettled() || !isDecode,
                    close.decodeSettled() || isDecode, close.notificationsDone(), close.timersDone(), close.expired(), close.notificationsAttempted());
            boolean ending = isDecode || evidence.role() == RoleType.PDFUSION
                    || evidence.kind() == EngineKind.FAILED || evidence.kind() == EngineKind.PRIORITY_CANCELLED
                    && preemption != null && !preemption.finished() && preemption.phase().acceptsPriorityTerminal();
            if (evidence.kind() == EngineKind.RETIRED && !isDecode) {
                ending = delivery == null && !engine.decodeAccepted() && preemption == null;
                if (placement != null && ending) {
                    placement = new PlacementFacts(placement.operation(), placement.acquired(), placement.published(),
                            placement.pendingTerminal(), evidence, placement.inactivityExpired());
                    ending = false;
                }
            }
            if (outcome == null && ending) {
                retainEnd(new EndEvidence(route, evidence.kind() == EngineKind.RETIRED ? EndKind.RETIREMENT
                        : evidence.kind() == EngineKind.PRIORITY_CANCELLED ? EndKind.PRIORITY : EndKind.WORKER,
                        evidence.role(), evidence.kind() == EngineKind.COMPLETED, evidence.errorCode(),
                        evidence.kind() == EngineKind.RETIRED ? StrategyErrorType.DISPATCH_FAILED
                                : evidence.kind() == EngineKind.PRIORITY_CANCELLED ? StrategyErrorType.PRIORITY_PREEMPTED
                                : StrategyErrorType.WORKER_EXECUTION_FAILED, evidence.detail(), !isDecode, isDecode), evidence.nowMs());
            }
        }
        var after = refresh(evidence.nowMs());
        return update(before, after);
    }

    public synchronized Claim<CancelPermit> requestCancel(Cancellation request) {
        var before = state;
        if (key == null) { return new Rejected<>(before, before, Rejection.NOT_REGISTERED); }
        Objects.requireNonNull(request.reason());
        long batchId = delivery == null ? 0L : delivery.identity().batchId();
        if (request.expectedBatchId() != 0L && request.expectedBatchId() != batchId) { return new Stale<>(before); }
        if (phase != Phase.OPEN) { return new Rejected<>(before, before, Rejection.CLOSED); }
        CancelKind currentKind = placement != null ? CancelKind.WAIT_PLACEMENT
                : delivery != null && delivery.transferred() ? CancelKind.REMOTE
                : preemption != null && !preemption.finished() ? CancelKind.WAIT_PREEMPTION
                : route == null ? CancelKind.GLOBAL_QUEUE : CancelKind.WORKER_QUEUE;
        if (cancellation != null && (cancellation.reason() != request.reason()
                || cancellation.operation() != null && cancellation.operation().kind() == currentKind)) {
            return new Rejected<>(before, before, Rejection.CANCELLED);
        }
        if (request.reason() == CancelReason.DEADLINE_EXCEEDED && delivery != null && cancellation == null) {
            return new Rejected<>(before, before, Rejection.INVALID_STATE);
        }
        boolean beforeSend = delivery == null || !delivery.transferred();
        if (request.frontend() && (selected != null || !beforeSend)) {
            return new Rejected<>(before, before, Rejection.ALREADY_SELECTED);
        }
        StrategyErrorType error = request.errorType() != null ? request.errorType()
                : request.reason() == CancelReason.DEADLINE_EXCEEDED
                        ? StrategyErrorType.BATCH_SLO_EXPIRED : StrategyErrorType.REQUEST_CANCELLED;
        if (cancellation == null) { cancellation = new CancellationFacts(request.reason(), error,
                request.detail() == null ? request.reason().getMessage() : request.detail(), null); }
        if (delivery != null && !canRelease()) {
            var old = delivery;
            var send = old.identity().kind() == DeliveryClaimKind.ROUTE_DECISION && old.send() == SendState.NOT_STARTED
                    ? SendState.ROUTE_CLOSED : old.send();
            delivery = new DeliveryFacts(old.identity(), send, old.acknowledged(), request.reason(), old.executionFinished(),
                    old.prefillProof(), old.decodeProof(), old.transferred(), old.startedAtMs(), old.acknowledgedAtMs());
        }
        CancelKind kind = currentKind;
        var permit = new CancelPermit(this, ++sequence, route, delivery == null ? null : delivery.identity(), kind, request.reason());
        cancellation = new CancellationFacts(cancellation.reason(), cancellation.errorType(), cancellation.detail(), permit);
        return new Granted<>(before, refresh(request.nowMs()), permit);
    }

    public synchronized Update completeCancel(CancelPermit exact, CancelEvidence evidence, long nowMs) {
        var before = state;
        if (exact.owner() != this || cancellation == null || cancellation.operation() != exact || cancellation.reason() != exact.reason()
                || exact.route() != route || exact.delivery() != (delivery == null ? null : delivery.identity())
                || phase == Phase.CLOSED) { return new Update(UpdateKind.STALE, before, before); }
        if (delivery != null && (evidence.kind() == CancelEvidenceKind.CLEANED || evidence.kind() == CancelEvidenceKind.FENCED)) {
            var old = delivery;
            delivery = new DeliveryFacts(old.identity(), old.send(), old.acknowledged(), old.abandonment(), old.executionFinished(),
                    old.prefillProof() || evidence.kind() == CancelEvidenceKind.CLEANED || evidence.kind() == CancelEvidenceKind.FENCED,
                    old.decodeProof() || evidence.kind() == CancelEvidenceKind.CLEANED, old.transferred(),
                    old.startedAtMs(), old.acknowledgedAtMs());
        }
        close = new CloseFacts(close.endOperation(), close.releaseOperation(), close.prefillSettled() || evidence.prefillSettled(),
                close.decodeSettled() || evidence.decodeSettled(), close.notificationsDone(), close.timersDone(), close.expired(), close.notificationsAttempted());
        if (evidence.kind() == CancelEvidenceKind.LOCAL_WITHDRAWN) {
            retainEnd(new EndEvidence(route, EndKind.FRONTEND, null, false, 0L, cancellation.errorType(),
                    cancellation.detail(), evidence.prefillSettled(), evidence.decodeSettled()), nowMs);
        }
        var after = refresh(nowMs);
        return update(before, after);
    }

    public synchronized Update recordDeadline(ExpirationTimer.Token exact, long nowMs, CancellationFacts diagnosis) {
        var before = state;
        var current = switch (exact.kind()) {
            case REQUEST -> deadlines.request();
            case VISIBILITY -> deadlines.visibility();
            case INACTIVITY -> deadlines.inactivity();
        };
        if (exact.owner() != this || current != exact || phase == Phase.CLOSED) {
            return new Update(UpdateKind.STALE, before, before);
        }
        if (nowMs < exact.atMs()) {
            deadlines = new ExpirationTimer.Plan(deadlines.revision() + 1L,
                    exact.kind() == ExpirationTimer.Kind.REQUEST ? null : deadlines.request(),
                    exact.kind() == ExpirationTimer.Kind.VISIBILITY ? null : deadlines.visibility(),
                    exact.kind() == ExpirationTimer.Kind.INACTIVITY ? null : deadlines.inactivity());
            return update(before, refresh(nowMs));
        }
        if (exact.kind() == ExpirationTimer.Kind.VISIBILITY) {
            engine = new EngineFacts(engine.prefillObserved(), engine.prefillCompletedAtMs(),
                    engine.decodeAccepted(), 0L, true, engine.predictionSequence());
        } else if (exact.kind() == ExpirationTimer.Kind.INACTIVITY) {
            deadlines = new ExpirationTimer.Plan(deadlines.revision() + 1L, deadlines.request(), deadlines.visibility(), null);
            long activity = Math.max(lastWorkerAtMs, delivery == null ? 0L : delivery.startedAtMs());
            if (nowMs - activity >= inactivityMs) {
                if (cancellation == null && outcome == null) {
                    cancellation = new CancellationFacts(CancelReason.DEADLINE_EXCEEDED,
                            diagnosis == null ? StrategyErrorType.BATCH_SLO_EXPIRED : diagnosis.errorType(),
                            diagnosis == null ? "request inactive" : diagnosis.detail(), null);
                }
                close = new CloseFacts(close.endOperation(), close.releaseOperation(), close.prefillSettled(),
                        close.decodeSettled(), close.notificationsDone(), close.timersDone(), true, close.notificationsAttempted());
                retainEnd(new EndEvidence(route, EndKind.INACTIVITY, null, false, 0L,
                        StrategyErrorType.BATCH_SLO_EXPIRED, "request inactive", false, false), nowMs);
            }
        } else {
            requestDeadlineConsumed = true;
            if (outcome == null && (delivery == null || placement != null)) {
                if (cancellation == null) {
                    cancellation = new CancellationFacts(CancelReason.DEADLINE_EXCEEDED,
                            diagnosis == null ? StrategyErrorType.BATCH_SLO_EXPIRED : diagnosis.errorType(),
                            diagnosis == null ? "request scheduling deadline exceeded" : diagnosis.detail(), null);
                }
                if (placement != null) {
                    placement = new PlacementFacts(placement.operation(), placement.acquired(), placement.published(),
                            placement.pendingTerminal(), placement.pendingRetirement(), true);
                } else if (delivery == null && !engine.decodeAccepted()) {
                    retainEnd(new EndEvidence(route, EndKind.INACTIVITY, null, false, 0L,
                            StrategyErrorType.BATCH_SLO_EXPIRED, "request scheduling deadline exceeded", false, false), nowMs);
                }
            }
        }
        var after = refresh(nowMs);
        return update(before, after);
    }

    public synchronized Claim<PreemptionRegistration> beginPreemption(DecodeResources.ReservationHandle exact,
                                                                       long attempt, CancelTarget target) {
        var before = state;
        if (phase != Phase.OPEN || route == null || cancellation != null || preemption != null
                || !Objects.equals(route.decodeReservation(), exact)) { return new Rejected<>(before, before, Rejection.INVALID_STATE); }
        if (placement != null) { return new Blocked<>(before, before, Blocker.PLACEMENT); }
        if (delivery != null && delivery.identity().kind() == DeliveryClaimKind.ROUTE_DECISION && !delivery.acknowledged()) {
            return new Rejected<>(before, before, Rejection.INVALID_STATE);
        }
        var registration = new PreemptionRegistration(this, attempt, exact, target);
        preemption = new PreemptionFacts(registration, PreemptionCancelPhase.CLAIMED, false, false, null, false, false, null);
        return new Granted<>(before, refresh(before.updatedAtMs()), registration);
    }

    public synchronized Claim<ReconciliationPermit> beginReconciliation(PreemptionRegistration exact) {
        var before = state;
        if (preemption == null || preemption.identity() != exact || phase == Phase.CLOSED) { return new Stale<>(before); }
        if (preemption.reconciliation() != null) { return new Blocked<>(before, before, Blocker.RECONCILIATION); }
        if (preemption.finished()) { return new Rejected<>(before, before, Rejection.INVALID_STATE); }
        var terminal = preemption.pendingTerminal();
        boolean authoritative = terminal != null && terminal.kind() == EndKind.WORKER
                && (terminal.role() == RoleType.DECODE || terminal.role() == RoleType.PDFUSION
                        || !terminal.success() || engine.decodeAccepted());
        boolean priorityProof = terminal != null && terminal.kind() == EndKind.PRIORITY
                && preemption.phase().acceptsPriorityTerminal();
        boolean observed = preemption.restorationObserved();
        boolean eligible = authoritative || priorityProof
                || preemption.phase() == PreemptionCancelPhase.NOT_FOUND_STALE
                        && (terminal == null ? preemption.confirmationPending() || observed : observed
                        || delivery != null && delivery.decodeProof());
        if (!eligible) {
            return new Blocked<>(before, before, Blocker.RELEASE_PROOF);
        }
        var permit = new ReconciliationPermit(this, exact, route, terminal, preemption.confirmationPending());
        preemption = new PreemptionFacts(exact, preemption.phase(), preemption.cancelAcknowledged(), false,
                terminal, preemption.confirmationPending(), preemption.restorationObserved(), permit);
        return new Granted<>(before, refresh(before.updatedAtMs()), permit);
    }

    public synchronized Update completePreemption(GrantValue operation, PreemptionReport report, long nowMs) {
        var before = state;
        var exact = operation instanceof ReconciliationPermit permit ? permit.attempt()
                : operation instanceof PreemptionRegistration registration ? registration : null;
        if (preemption == null || preemption.identity() != exact || phase == Phase.CLOSED
                || operation instanceof ReconciliationPermit && preemption.reconciliation() != operation
                || report.ledgerCommitted() && preemption.reconciliation() != null && preemption.reconciliation() != operation) {
            return new Update(UpdateKind.STALE, before, before);
        }
        var previous = preemption;
        if (previous.finished() && report.phase() != null) { return update(before, before); }
        var next = report.phase() == null ? previous.phase() : report.phase();
        if (report.phase() != null && next != previous.phase() && !previous.phase().canTransitionTo(next)) {
            return update(before, before);
        }
        boolean acknowledged = previous.cancelAcknowledged() || next == PreemptionCancelPhase.CANCEL_REQUESTED;
        var terminal = report.terminal() == null ? previous.pendingTerminal() : report.terminal();
        boolean finished = previous.finished() || report.ledgerCommitted() && terminal != null;
        preemption = new PreemptionFacts(exact, next, acknowledged, finished, terminal,
                previous.confirmationPending(), previous.restorationObserved(), null);
        if (report.ledgerCommitted()) {
            if (delivery != null && terminal != null) {
                var old = delivery;
                delivery = new DeliveryFacts(old.identity(), old.send(), old.acknowledged(), old.abandonment(),
                        old.executionFinished() || terminal.kind() == EndKind.WORKER,
                        old.prefillProof() || terminal.prefillSettled(), old.decodeProof() || report.decodeSettled(),
                        old.transferred(), old.startedAtMs(), old.acknowledgedAtMs());
            }
            close = new CloseFacts(close.endOperation(), close.releaseOperation(),
                    close.prefillSettled() || terminal != null && terminal.prefillSettled(),
                    close.decodeSettled() || report.decodeSettled() || terminal != null && terminal.decodeSettled(),
                    close.notificationsDone(), close.timersDone(), close.expired(), close.notificationsAttempted());
            if (terminal != null) { retainEnd(terminal, nowMs); }
            else { preemption = null; }
        } else if (report.releaseAttempt() && next.isLocallyReleasable()) {
            preemption = null;
            if (terminal != null) { retainEnd(terminal, nowMs); }
        }
        var after = refresh(nowMs);
        return update(before, after);
    }

    public synchronized Claim<ResponseSelection> selectResponse(ResponseCandidate candidate) {
        var before = state;
        if (candidate.owner() != this) { return new Stale<>(before); }
        if (key == null) { return new Rejected<>(before, before, Rejection.NOT_REGISTERED); }
        if (selected != null) {
            return selected.candidate() == candidate ? new Granted<>(before, before, selected)
                    : new Rejected<>(before, before, Rejection.ALREADY_SELECTED);
        }
        if (phase == Phase.CLOSED && candidate.kind() != ResponseKind.TERMINAL) {
            return new Rejected<>(before, before, Rejection.CLOSED);
        }
        boolean acknowledgement = candidate.kind() == ResponseKind.ACK || candidate.kind() == ResponseKind.CONFIRMATION;
        if (acknowledgement) {
            if (phase != Phase.OPEN || pendingEnd != null || cancellation != null || delivery == null
                    || delivery.abandonment() != null || delivery.send() != SendState.DELIVERED
                            && !(delivery.send() == SendState.UNKNOWN && engine.decodeAccepted())) {
                return new Rejected<>(before, before, Rejection.INVALID_STATE);
            }
            if (preemption != null && !preemption.finished() && !delivery.acknowledged()) {
                preemption = new PreemptionFacts(preemption.identity(), preemption.phase(), preemption.cancelAcknowledged(),
                        false, preemption.pendingTerminal(), true, preemption.restorationObserved(), preemption.reconciliation());
                return new Blocked<>(before, refresh(candidate.nowMs()), Blocker.PREEMPTION);
            }
            long activity = Math.max(lastWorkerAtMs, delivery.startedAtMs());
            if (candidate.nowMs() - activity >= inactivityMs) {
                if (cancellation == null) {
                    cancellation = new CancellationFacts(CancelReason.DEADLINE_EXCEEDED,
                            StrategyErrorType.BATCH_SLO_EXPIRED, "request inactive before confirmation", null);
                }
                retainEnd(new EndEvidence(route, EndKind.INACTIVITY, null, false, 0L,
                        StrategyErrorType.BATCH_SLO_EXPIRED, "request inactive before confirmation", false, false), candidate.nowMs());
                return new Rejected<>(before, refresh(candidate.nowMs()), Rejection.CANCELLED);
            }
            if (!delivery.acknowledged()) {
                var old = delivery;
                delivery = new DeliveryFacts(old.identity(), old.send(), true, old.abandonment(), old.executionFinished(),
                        old.prefillProof(), old.decodeProof(), old.transferred(), old.startedAtMs(), candidate.nowMs());
            }
        } else if (candidate.kind() == ResponseKind.TERMINAL && outcome == null
                || candidate.kind() == ResponseKind.CANCELLATION && cancellation == null) {
            return new Rejected<>(before, before, Rejection.NO_OUTCOME);
        }
        var choice = new ResponseSelection(this, candidate.kind(), acknowledgement ? null : outcome == null
                ? candidate.result() : outcome, route, delivery == null ? 0L : delivery.identity().batchId(), candidate.nowMs(), candidate);
        if (candidate.kind() != ResponseKind.CONFIRMATION) { selected = choice; }
        return new Granted<>(before, refresh(candidate.nowMs()), choice);
    }

    /** Publisher capacity is reserved by Scheduler before this outcome/notification claim. */
    public synchronized Claim<EndToken> beginClose(EndEvidence evidence, long nowMs) {
        var before = state;
        if (key == null) { return new Rejected<>(before, before, Rejection.NOT_REGISTERED); }
        if (phase == Phase.CLOSED) { return new Rejected<>(before, before, Rejection.CLOSED); }
        if (evidence != null && evidence.route() != null && evidence.route() != route) { return new Stale<>(before); }
        if (evidence != null) { retainEnd(evidence, nowMs); }
        boolean deliveryFailure = cancellation == null && pendingEnd != null && pendingEnd.kind() == EndKind.DELIVERY_FAILURE;
        if (placement != null && !deliveryFailure) {
            return new Blocked<>(before, refresh(nowMs), Blocker.PLACEMENT);
        }
        if (preemption != null && !preemption.finished() && !deliveryFailure
                && (pendingEnd == null || pendingEnd.kind() != EndKind.FRONTEND
                        && pendingEnd.kind() != EndKind.INACTIVITY && pendingEnd.kind() != EndKind.RETIREMENT)) {
            return new Blocked<>(before, refresh(nowMs), Blocker.PREEMPTION);
        }
        if (outcome == null && pendingEnd == null) { return new Rejected<>(before, before, Rejection.NO_OUTCOME); }
        if (close.endOperation() != null || close.notificationsDone()) {
            return new Blocked<>(before, before, Blocker.CLOSE_IN_PROGRESS);
        }
        if (outcome == null) {
            outcome = classify(pendingEnd);
            phase = Phase.ENDING;
            close = new CloseFacts(null, close.releaseOperation(), close.prefillSettled() || pendingEnd.prefillSettled()
                    || engine.prefillCompletedAtMs() > 0L, close.decodeSettled() || pendingEnd.decodeSettled(),
                    false, false, close.expired() || pendingEnd.kind() == EndKind.INACTIVITY, close.notificationsAttempted());
            if (delivery != null) {
                var old = delivery;
                boolean success = pendingEnd.kind() == EndKind.WORKER && pendingEnd.success() && cancellation == null;
                var abandoned = success || canRelease() ? old.abandonment()
                        : old.abandonment() != null ? old.abandonment()
                        : cancellation == null ? CancelReason.CLIENT_CANCELLED : cancellation.reason();
                var send = old.send();
                delivery = new DeliveryFacts(old.identity(), send, old.acknowledged(), abandoned,
                        old.executionFinished() || success, old.prefillProof(), old.decodeProof(), old.transferred(),
                        old.startedAtMs(), old.acknowledgedAtMs());
            }
            if (preemption != null && (close.expired() || pendingEnd.kind() == EndKind.RETIREMENT
                    && pendingEnd.decodeSettled())) {
                preemption = new PreemptionFacts(preemption.identity(), preemption.phase(), preemption.cancelAcknowledged(),
                        true, preemption.pendingTerminal(), preemption.confirmationPending(), preemption.restorationObserved(), preemption.reconciliation());
            }
            pendingEnd = null;
        }
        var exact = new EndToken(this, ++sequence, route, preemption == null || !preemption.finished() ? null : preemption.identity());
        close = new CloseFacts(exact, close.releaseOperation(), close.prefillSettled(), close.decodeSettled(),
                false, close.timersDone(), close.expired(), false);
        return new Granted<>(before, refresh(nowMs), exact);
    }

    public synchronized Claim<ClosePermit> beginSettlement() {
        var before = state;
        if (outcome == null || phase == Phase.CLOSED) { return new Rejected<>(before, before, Rejection.NO_OUTCOME); }
        if (placement != null) { return new Blocked<>(before, before, Blocker.PLACEMENT); }
        if (close.releaseOperation() != null) { return new Blocked<>(before, before, Blocker.CLOSE_IN_PROGRESS); }
        if (!canRelease()) {
            boolean sending = delivery != null && (delivery.send() == SendState.SENDING || delivery.send() == SendState.NOT_STARTED);
            return new Blocked<>(before, before, sending ? Blocker.SENDER : Blocker.RELEASE_PROOF);
        }
        var proof = delivery != null && delivery.send() == SendState.NOT_SENT ? DecodeResources.ReleaseReason.NOT_SENT
                : delivery != null && delivery.prefillProof() && delivery.decodeProof()
                        ? DecodeResources.ReleaseReason.REMOTE_CLEANUP
                        : close.expired() ? DecodeResources.ReleaseReason.EXPIRED : DecodeResources.ReleaseReason.COUNTERPART_FINISHED;
        var exact = new ClosePermit(this, ++sequence, route, proof, close.prefillSettled(), close.decodeSettled(),
                close.notificationsDone(), close.timersDone());
        close = new CloseFacts(close.endOperation(), exact, close.prefillSettled(), close.decodeSettled(),
                close.notificationsDone(), close.timersDone(), close.expired(), close.notificationsAttempted());
        return new Granted<>(before, refresh(before.updatedAtMs()), exact);
    }

    public synchronized Update completeClose(CloseOperation exact, CloseProgress progress, long nowMs) {
        var before = state;
        boolean notification = exact instanceof EndToken;
        if (exact.owner() != this || phase == Phase.CLOSED
                || (notification ? exact != close.endOperation() : exact != close.releaseOperation())) {
            return new Update(UpdateKind.STALE, before, before);
        }
        if (notification != (progress.kind() == ProgressKind.NOTIFICATIONS)) {
            throw new IllegalArgumentException("close progress belongs to another operation");
        }
        if (notification && delivery != null && delivery.identity().kind() == DeliveryClaimKind.ROUTE_DECISION
                && delivery.send() == SendState.NOT_STARTED) {
            var old = delivery;
            delivery = new DeliveryFacts(old.identity(), SendState.ROUTE_CLOSED, old.acknowledged(), old.abandonment(),
                    old.executionFinished(), old.prefillProof(), old.decodeProof(), old.transferred(),
                    old.startedAtMs(), old.acknowledgedAtMs());
        }
        boolean notified = close.notificationsDone() || progress.notificationsDone()
                && (preemption == null || preemption.finished());
        close = new CloseFacts(notification && notified ? null : close.endOperation(), notification ? close.releaseOperation() : null,
                close.prefillSettled() || progress.prefillSettled(), close.decodeSettled() || progress.decodeSettled(),
                notified, close.timersDone() || progress.timersDone(), close.expired(), close.notificationsAttempted() || notification);
        if (preemption != null && close.decodeSettled() && (close.expired()
                || delivery != null && delivery.abandonment() != null && delivery.prefillProof() && delivery.decodeProof())) {
            preemption = new PreemptionFacts(preemption.identity(), preemption.phase(), preemption.cancelAcknowledged(),
                    true, preemption.pendingTerminal(), preemption.confirmationPending(), preemption.restorationObserved(), preemption.reconciliation());
        }
        if (close.prefillSettled() && close.decodeSettled() && close.notificationsDone() && close.timersDone()
                && canRelease() && placement == null && close.releaseOperation() == null && close.endOperation() == null
                && (preemption == null || preemption.finished())) {
            terminalRecord = new RequestState(key.requestId(), outcome.outcome().phase(),
                    delivery == null ? DeliveryClaimKind.NONE : delivery.identity().kind(),
                    delivery == null ? 0L : delivery.identity().batchId(), createdAtMs, nowMs, outcome.detail());
            phase = Phase.CLOSED;
            route = null;
            preemption = null;
            pendingEnd = null;
        }
        var after = refresh(nowMs);
        return update(before, after);
    }

    private void retainEnd(EndEvidence evidence, long nowMs) {
        if (outcome != null) {
            close = new CloseFacts(close.endOperation(), close.releaseOperation(),
                    close.prefillSettled() || evidence.prefillSettled(), close.decodeSettled() || evidence.decodeSettled(),
                    close.notificationsDone(), close.timersDone(), close.expired() || evidence.kind() == EndKind.INACTIVITY, close.notificationsAttempted());
            return;
        }
        boolean deliveryFailure = cancellation == null && evidence.kind() == EndKind.DELIVERY_FAILURE;
        if (placement != null && !deliveryFailure) {
            var previous = placement.pendingTerminal();
            boolean better = previous == null || evidence.kind() == EndKind.WORKER && previous.kind() != EndKind.WORKER
                    || evidence.kind() == EndKind.RETIREMENT && previous.kind() != EndKind.WORKER;
            placement = new PlacementFacts(placement.operation(), placement.acquired(), placement.published(),
                    better ? evidence : previous, placement.pendingRetirement(),
                    placement.inactivityExpired() || evidence.kind() == EndKind.INACTIVITY);
            return;
        }
        if (preemption != null && !preemption.finished() && !deliveryFailure
                && evidence.kind() != EndKind.FRONTEND && evidence.kind() != EndKind.INACTIVITY && evidence.kind() != EndKind.RETIREMENT) {
            var previous = preemption.pendingTerminal();
            boolean better = previous == null || evidence.kind() == EndKind.WORKER && previous.kind() != EndKind.WORKER;
            preemption = new PreemptionFacts(preemption.identity(), preemption.phase(), preemption.cancelAcknowledged(),
                    false, better ? evidence : previous, preemption.confirmationPending(), preemption.restorationObserved(), preemption.reconciliation());
            return;
        }
        if (pendingEnd == null || evidence.kind() == EndKind.WORKER && pendingEnd.kind() != EndKind.WORKER) {
            pendingEnd = evidence;
        }
    }

    private End classify(EndEvidence evidence) {
        boolean cancelled = cancellation != null && (evidence.kind() != EndKind.FAILURE || delivery == null);
        if (cancelled || evidence.kind() == EndKind.INACTIVITY) {
            var cause = cancellation == null ? CancelReason.DEADLINE_EXCEEDED : cancellation.reason();
            String detail = cancellation == null ? evidence.detail() : cancellation.detail();
            return new End(evidence.kind(), TerminalOutcome.cancellation(cause, detail),
                    cancellation == null ? StrategyErrorType.BATCH_SLO_EXPIRED : cancellation.errorType(), detail);
        }
        if (evidence.kind() == EndKind.WORKER && evidence.success()
                || evidence.kind() == EndKind.FRONTEND && evidence.success()) {
            String detail = evidence.detail() == null || evidence.detail().isBlank() ? "request completed" : evidence.detail();
            return new End(evidence.kind(), TerminalOutcome.complete(detail), null, detail);
        }
        var error = evidence.errorType() != null ? evidence.errorType()
                : evidence.kind() == EndKind.PRIORITY ? StrategyErrorType.PRIORITY_PREEMPTED
                : evidence.kind() == EndKind.WORKER ? StrategyErrorType.WORKER_EXECUTION_FAILED : StrategyErrorType.DISPATCH_FAILED;
        String detail = evidence.detail() == null || evidence.detail().isBlank() ? evidence.kind() == EndKind.WORKER
                ? "worker error code " + evidence.errorCode() : evidence.kind() == EndKind.PRIORITY
                ? "priority victim canceled by worker" : "request ended" : evidence.detail();
        return new End(evidence.kind(), evidence.kind() == EndKind.PRIORITY
                ? TerminalOutcome.cancel(detail) : TerminalOutcome.fail(detail), error, detail);
    }

    private boolean canRelease() {
        if (delivery == null) { return true; }
        if (delivery.send() == SendState.NOT_STARTED || delivery.send() == SendState.SENDING) { return false; }
        if (delivery.identity().kind() == DeliveryClaimKind.ROUTE_DECISION || delivery.send() == SendState.NOT_SENT) { return true; }
        return delivery.abandonment() == null ? delivery.executionFinished() : delivery.prefillProof() && delivery.decodeProof();
    }

    private boolean matches(RequestRoute exact, DecodeResources.ReservationHandle reservation, RoleType role) {
        if (route == null || route != exact || exact.ctx() != this || role == null
                || role != RoleType.DECODE && !role.supportsPrefill()) { return false; }
        return role != RoleType.DECODE || Objects.equals(route.decodeReservation(), reservation);
    }

    private long inactivityExpiresAtMs() {
        long activeSince = Math.max(lastWorkerAtMs, delivery == null ? 0L : delivery.startedAtMs());
        return activeSince > Long.MAX_VALUE - inactivityMs ? Long.MAX_VALUE : activeSince + inactivityMs;
    }

    private void deriveDeadlines() {
        long requestAt = phase == Phase.OPEN && !requestDeadlineConsumed
                && (delivery == null || !delivery.acknowledged()) ? absoluteDeadline : 0L;
        long visibilityAt = phase == Phase.OPEN && outcome == null ? engine.visibilityAtMs() : 0L;
        boolean settled = phase == Phase.ENDING && close.prefillSettled() && close.decodeSettled()
                && close.notificationsDone() && close.timersDone() && canRelease() && placement == null && (preemption == null || preemption.finished());
        long inactivityAt = phase == Phase.CLOSED || close.expired() || settled ? 0L : inactivityExpiresAtMs();
        var oldRequest = deadlines.request();
        var oldVisibility = deadlines.visibility();
        var oldInactivity = deadlines.inactivity();
        if (oldInactivity != null && inactivityAt > 0L && oldInactivity.atMs() <= inactivityAt) {
            inactivityAt = oldInactivity.atMs(); // Existing wake-up will recheck the latest activity.
        }
        boolean sameRequest = requestAt <= 0L ? oldRequest == null : oldRequest != null && oldRequest.atMs() == requestAt;
        boolean sameVisibility = visibilityAt <= 0L ? oldVisibility == null : oldVisibility != null && oldVisibility.atMs() == visibilityAt;
        boolean sameInactivity = inactivityAt <= 0L ? oldInactivity == null : oldInactivity != null && oldInactivity.atMs() == inactivityAt;
        if (sameRequest && sameVisibility && sameInactivity) { return; }
        long revision = deadlines.revision() + 1L;
        var request = sameRequest ? oldRequest : requestAt <= 0L ? null
                : new ExpirationTimer.Token(this, ExpirationTimer.Kind.REQUEST, revision, requestAt);
        var visibility = sameVisibility ? oldVisibility : visibilityAt <= 0L ? null
                : new ExpirationTimer.Token(this, ExpirationTimer.Kind.VISIBILITY, revision, visibilityAt);
        var inactivity = sameInactivity ? oldInactivity : inactivityAt <= 0L ? null
                : new ExpirationTimer.Token(this, ExpirationTimer.Kind.INACTIVITY, revision, inactivityAt);
        deadlines = new ExpirationTimer.Plan(revision, request, visibility, inactivity);
    }

    private Update update(Snapshot before, Snapshot after) {
        if (after != before) { return new Update(UpdateKind.APPLIED, before, after); }
        if (cachedNoChange == null || cachedNoChange.before() != before) {
            cachedNoChange = new Update(UpdateKind.NO_CHANGE, before, before);
        }
        return cachedNoChange;
    }

    private Snapshot refresh(long nowMs) {
        if (key != null) { deriveDeadlines(); }
        if (Objects.equals(state.placement(), placement)) { placement = state.placement(); }
        if (Objects.equals(state.delivery(), delivery)) { delivery = state.delivery(); }
        if (Objects.equals(state.engine(), engine)) { engine = state.engine(); }
        if (Objects.equals(state.preemption(), preemption)) { preemption = state.preemption(); }
        if (Objects.equals(state.close(), close)) { close = state.close(); }
        if (Objects.equals(state.pendingEnd(), pendingEnd)) { pendingEnd = state.pendingEnd(); }
        boolean sameFacts = state.key() == key && state.createdAtMs() == createdAtMs && state.phase() == phase && state.route() == route
                && state.placement() == placement && state.delivery() == delivery && state.engine() == engine
                && state.preemption() == preemption && state.cancellation() == cancellation && state.outcome() == outcome
                && state.pendingEnd() == pendingEnd && state.selectedResponse() == selected && state.close() == close
                && state.deadlines() == deadlines && state.terminalRecord() == terminalRecord;
        if (sameFacts && state.lastWorkerAtMs() == lastWorkerAtMs) { return state; }
        var facts = sameFacts ? state.facts() : new SnapshotFacts(key, createdAtMs, phase, route, placement, delivery, engine, preemption,
                cancellation, outcome, pendingEnd, selected, close, deadlines, terminalRecord, canRelease(),
                phase == Phase.OPEN && route != null && delivery == null && cancellation == null && preemption == null
                        && pendingEnd == null && (placement == null || placement.published()
                        && placement.operation().withdrawing() == null && placement.pendingTerminal() == null
                        && placement.pendingRetirement() == null && !placement.inactivityExpired()),
                placement == null && !engine.decodeAccepted() && (delivery == null || !delivery.transferred()
                        || delivery.identity().kind() == DeliveryClaimKind.ROUTE_DECISION && !delivery.acknowledged()),
                phase == Phase.OPEN && selected == null && (cancellation == null
                        || cancellation.reason() != CancelReason.DEADLINE_EXCEEDED)
                        && !engine.decodeAccepted() && (delivery == null || !delivery.transferred()));
        state = new Snapshot(facts, lastWorkerAtMs, Math.max(state.updatedAtMs(), nowMs), inactivityExpiresAtMs());
        return state;
    }
}
