package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

import static org.flexlb.balance.scheduler.ContextResult.Abandonment;
import static org.flexlb.balance.scheduler.ContextResult.Blocked;
import static org.flexlb.balance.scheduler.ContextResult.Blocker;
import static org.flexlb.balance.scheduler.ContextResult.CancelEvidenceKind;
import static org.flexlb.balance.scheduler.ContextResult.Cancellation;
import static org.flexlb.balance.scheduler.ContextResult.CancellationFacts;
import static org.flexlb.balance.scheduler.ContextResult.Claim;
import static org.flexlb.balance.scheduler.ContextResult.CleanupAcknowledgement;
import static org.flexlb.balance.scheduler.ContextResult.CloseFacts;
import static org.flexlb.balance.scheduler.ContextResult.CloseProgress;
import static org.flexlb.balance.scheduler.ContextResult.DeliveryEvidence;
import static org.flexlb.balance.scheduler.ContextResult.DeliveryFacts;
import static org.flexlb.balance.scheduler.ContextResult.End;
import static org.flexlb.balance.scheduler.ContextResult.EndEvidence;
import static org.flexlb.balance.scheduler.ContextResult.EndKind;
import static org.flexlb.balance.scheduler.ContextResult.EngineEvidence;
import static org.flexlb.balance.scheduler.ContextResult.EngineFacts;
import static org.flexlb.balance.scheduler.ContextResult.EngineKind;
import static org.flexlb.balance.scheduler.ContextResult.Granted;
import static org.flexlb.balance.scheduler.ContextResult.Identity;
import static org.flexlb.balance.scheduler.ContextResult.Phase;
import static org.flexlb.balance.scheduler.ContextResult.PlacementFacts;
import static org.flexlb.balance.scheduler.ContextResult.PlacementToken;
import static org.flexlb.balance.scheduler.ContextResult.Prediction;
import static org.flexlb.balance.scheduler.ContextResult.PreemptionFacts;
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
import static org.flexlb.balance.scheduler.ContextResult.SnapshotFacts;
import static org.flexlb.balance.scheduler.ContextResult.Stale;
import static org.flexlb.balance.scheduler.ContextResult.TransportResult;
import static org.flexlb.balance.scheduler.ContextResult.Update;
import static org.flexlb.balance.scheduler.ContextResult.UpdateKind;

/**
 * Atomic request facts. Scheduler alone executes transport, publication, ledger and shutdown flows.
 */
public final class RequestContext {
    private static final AtomicLong INCARNATIONS = new AtomicLong();
    private final RequestData data;
    private Identity key;
    private static final SnapshotFacts EMPTY =
            new SnapshotFacts(
                    null,
                    0L,
                    Phase.OPEN,
                    null,
                    null,
                    null,
                    new EngineFacts(false, 0L, false, 0L, false, 0L),
                    null,
                    null,
                    null,
                    null,
                    null,
                    new CloseFacts(true, true, false, false, false, false),
                    new ExpirationTimer.Plan(0L, null, null, null),
                    null,
                    true,
                    false);
    private Phase phase = Phase.OPEN;
    private RequestRoute route;
    private PlacementFacts placement;
    private DeliveryFacts delivery;
    private EngineFacts engine = EMPTY.engine();
    private PreemptionFacts preemption;
    private CancellationFacts cancellation;
    private End outcome;
    private EndEvidence pendingEnd;
    private ResponseCandidate selected;
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
        state = new Snapshot(EMPTY, 0L, 0L, 0L);
    }

    public RequestData data() {
        return data;
    }

    public synchronized Update register(
            long requestId, long nowMs, long expiresAtMs, long inactivityTimeoutMs) {
        var before = state;
        if (key != null) {
            return update(before, before);
        }
        if (inactivityTimeoutMs <= 0L) {
            throw new IllegalArgumentException("positive inactivity timeout required");
        }
        key = new Identity(requestId, INCARNATIONS.incrementAndGet());
        createdAtMs = lastWorkerAtMs = nowMs;
        absoluteDeadline = expiresAtMs;
        inactivityMs = inactivityTimeoutMs;
        var after = refresh(nowMs);
        return update(before, after);
    }

    public synchronized Snapshot snapshot() {
        return state;
    }

    public synchronized Claim<PlacementToken> beginPlacement(RequestRoute withdrawing) {
        var before = state;
        if (key == null) {
            return new Rejected<>(before, before, Rejection.NOT_REGISTERED);
        }
        if (phase != Phase.OPEN) {
            return new Rejected<>(before, before, Rejection.CLOSED);
        }
        if (cancellation != null) {
            return new Rejected<>(before, before, Rejection.CANCELLED);
        }
        if (placement != null) {
            return new Blocked<>(before, before, Blocker.PLACEMENT);
        }
        if (preemption != null) {
            return new Blocked<>(before, before, Blocker.PREEMPTION);
        }
        if (withdrawing == null
                ? route != null
                : route != withdrawing || delivery != null || engine.decodeAccepted()) {
            return new Rejected<>(before, before, Rejection.INVALID_STATE);
        }
        var token = new PlacementToken(this, ++sequence, withdrawing);
        placement = new PlacementFacts(token, withdrawing, withdrawing != null, null, null, false);
        return new Granted<>(before, refresh(before.updatedAtMs()), token);
    }

    public synchronized Claim<RoutePermit> bindPlacement(
            PlacementToken exact, RequestRoute acquired, PublicationResult result) {
        var before = state;
        Objects.requireNonNull(acquired);
        if (placement == null || placement.operation() != exact || exact.owner() != this) {
            return new Stale<>(before);
        }
        if (acquired.requestId() != key.requestId() || acquired.ctx() != this) {
            return new Stale<>(before);
        }
        if (result != PublicationResult.PREPARED) {
            if (route != acquired || placement.acquired() != acquired) {
                return new Stale<>(before);
            }
            placement =
                    new PlacementFacts(
                            exact,
                            acquired,
                            result == PublicationResult.PUBLISHED,
                            placement.pendingTerminal(),
                            placement.pendingRetirement(),
                            placement.inactivityExpired());
            if (result == PublicationResult.WITHDRAWN
                    || result == PublicationResult.ROLLBACK_COMPLETED) {
                placement =
                        new PlacementFacts(
                                exact,
                                null,
                                false,
                                placement.pendingTerminal(),
                                placement.pendingRetirement(),
                                placement.inactivityExpired());
                route = null;
                close = close.withSettlement(true, true);
            }
            return new Granted<>(
                    before,
                    refresh(before.updatedAtMs()),
                    new RoutePermit(
                            this,
                            exact,
                            acquired,
                            result == PublicationResult.PUBLISHED
                                    && cancellation == null
                                    && phase == Phase.OPEN));
        }
        if (placement.acquired() != null && placement.acquired() != acquired) {
            return new Rejected<>(before, before, Rejection.INVALID_STATE);
        }
        placement =
                new PlacementFacts(
                        exact,
                        acquired,
                        false,
                        placement.pendingTerminal(),
                        placement.pendingRetirement(),
                        placement.inactivityExpired());
        route = acquired;
        close =
                close.withSettlement(
                        false, acquired.decodeEp() == null || acquired.decodeReservation() == null);
        boolean publishable = phase == Phase.OPEN && cancellation == null;
        var after = refresh(before.updatedAtMs());
        var permit = new RoutePermit(this, exact, acquired, publishable);
        return publishable
                ? new Granted<>(before, after, permit)
                : new Rejected<>(before, after, Rejection.CANCELLED);
    }

    public synchronized Update completePlacement(
            PlacementToken exact, PublicationResult result, long nowMs) {
        var before = state;
        if (placement == null || placement.operation() != exact) {
            return new Update(UpdateKind.STALE, before, before);
        }
        var retained = placement;
        placement = null;
        if (result == PublicationResult.WITHDRAWN
                || result == PublicationResult.ROLLBACK_COMPLETED) {
            route = null;
            delivery = null;
            close = close.withSettlement(true, true);
        }
        boolean rolledBack =
                route == null
                        && (retained.acquired() == null
                                || result == PublicationResult.ROLLBACK_COMPLETED
                                || result == PublicationResult.WITHDRAWN)
                        && close.prefillSettled()
                        && close.decodeSettled();
        if (rolledBack && pendingEnd != null) {
            pendingEnd = pendingEnd.withoutRoute();
        }
        boolean admissionFailure =
                result == PublicationResult.FAILED && outcome == null && cancellation == null;
        if (admissionFailure && pendingEnd != null && pendingEnd.kind() == EndKind.SHUTDOWN) {
            pendingEnd = null;
        }
        if (retained.pendingTerminal() != null
                && !(admissionFailure && retained.pendingTerminal().kind() == EndKind.SHUTDOWN)) {
            var end = retained.pendingTerminal();
            retainEnd(rolledBack ? end.withoutRoute() : end, nowMs);
        }
        if (pendingEnd == null && outcome == null && retained.pendingRetirement() != null) {
            var retired = retained.pendingRetirement();
            retainEnd(
                    new EndEvidence(
                            rolledBack ? null : retired.route(),
                            EndKind.RETIREMENT,
                            retired.role(),
                            false,
                            0L,
                            StrategyErrorType.DISPATCH_FAILED,
                            retired.detail(),
                            retired.role().supportsPrefill(),
                            retired.role() == RoleType.DECODE
                                    || retired.role() == RoleType.PDFUSION),
                    nowMs);
        }
        if (outcome == null && retained.inactivityExpired()) {
            retainEnd(
                    EndEvidence.failure(
                            route,
                            EndKind.INACTIVITY,
                            StrategyErrorType.BATCH_SLO_EXPIRED,
                            "request inactive during placement"),
                    nowMs);
        }
        var after = refresh(nowMs);
        return update(before, after);
    }

    /**
     * Prepare immutable facts locally; Scheduler holds this monitor through Decode transfer and publish.
     */
    public synchronized Claim<DeliveryFacts> prepareDelivery(
            RequestRoute exact, DeliveryClaimKind kind, long batchId, long nowMs) {
        var before = state;
        if (phase != Phase.OPEN
                || route == null
                || route != exact
                || delivery != null
                || cancellation != null
                || pendingEnd != null) {
            return new Rejected<>(before, before, Rejection.INVALID_STATE);
        }
        if (placement != null
                && (!placement.published()
                        || placement.operation().withdrawing() != null
                        || placement.pendingTerminal() != null
                        || placement.pendingRetirement() != null)) {
            return new Blocked<>(before, before, Blocker.PLACEMENT);
        }
        if (preemption != null) {
            return new Blocked<>(before, before, Blocker.PREEMPTION);
        }
        if (absoluteDeadline > 0L && nowMs >= absoluteDeadline
                || nowMs - lastWorkerAtMs >= inactivityMs) {
            if (cancellation == null) {
                cancellation =
                        new CancellationFacts(
                                CancelReason.DEADLINE_EXCEEDED,
                                StrategyErrorType.BATCH_SLO_EXPIRED,
                                "request expired before delivery");
            }
            retainEnd(
                    EndEvidence.failure(
                            route,
                            EndKind.INACTIVITY,
                            StrategyErrorType.BATCH_SLO_EXPIRED,
                            cancellation.detail()),
                    nowMs);
            return new Rejected<>(before, refresh(nowMs), Rejection.CANCELLED);
        }
        var claim = new DeliveryClaim(this, ++sequence, exact, kind, batchId);
        var prepared =
                new DeliveryFacts(
                        claim,
                        SendState.NOT_STARTED,
                        false,
                        null,
                        false,
                        false,
                        exact.decodeEp() == null || exact.decodeReservation() == null,
                        nowMs);
        return new Granted<>(before, before, prepared);
    }

    /** Scheduler retains this monitor from preparation through Decode transfer and publication. */
    synchronized void publishDelivery(DeliveryFacts prepared) {
        delivery = prepared;
        refresh(prepared.startedAtMs());
    }

    public synchronized Claim<DeliveryClaim> beginSend(DeliveryClaim exact, long nowMs) {
        var before = state;
        if (delivery == null || delivery.identity() != exact) {
            return new Stale<>(before);
        }
        if (delivery.send() != SendState.NOT_STARTED
                || exact.kind() != DeliveryClaimKind.BATCH_ENQUEUE) {
            return new Rejected<>(before, before, Rejection.INVALID_STATE);
        }
        if (phase != Phase.OPEN
                || cancellation != null
                || delivery.abandonment() != null
                || pendingEnd != null
                || placement != null
                        && (placement.pendingTerminal() != null
                                || placement.pendingRetirement() != null)
                || preemption != null && preemption.pendingTerminal() != null
                || absoluteDeadline > 0L && nowMs >= absoluteDeadline
                || nowMs - lastWorkerAtMs >= inactivityMs) {
            return new Rejected<>(before, before, Rejection.CANCELLED);
        }
        delivery = delivery.withSendState(SendState.SENDING);
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
            retainEnd(EndEvidence.deliveryFailure(route, failed), failed.nowMs());
            var after = refresh(failed.nowMs());
            return update(before, after);
        }
        if (delivery == null || delivery.identity() != exact || phase == Phase.CLOSED) {
            return new Update(UpdateKind.STALE, before, before);
        }
        if ((evidence instanceof Prediction || evidence instanceof RouteDelivered)
                && phase != Phase.OPEN) {
            return new Update(UpdateKind.STALE, before, before);
        }
        if (evidence instanceof RouteDelivered
                && exact.kind() != DeliveryClaimKind.ROUTE_DECISION) {
            throw new IllegalArgumentException("route completion requires route delivery identity");
        }
        var previous = delivery;
        var send = delivery.send();
        var abandoned = delivery.abandonment();
        boolean p = delivery.prefillProof(), d = delivery.decodeProof();
        if (evidence instanceof TransportResult transport) {
            if (exact.kind() != DeliveryClaimKind.BATCH_ENQUEUE) {
                throw new IllegalArgumentException(
                        "transport result requires batch delivery identity");
            }
            var actual =
                    switch (transport.result().status()) {
                        case NOT_SENT -> SendState.NOT_SENT;
                        case PREFILL_REJECTED -> SendState.REJECTED;
                        case DELIVERED -> SendState.DELIVERED;
                        case UNCERTAIN -> SendState.UNKNOWN;
                    };
            if (send != SendState.NOT_STARTED && send != SendState.SENDING) {
                if (send == actual) {
                    return update(before, before);
                }
                throw new IllegalStateException("conflicting result after sender completion");
            }
            send = actual;
            if (transport.result().failed()) {
                boolean releaseAfterExit =
                        send == SendState.NOT_SENT
                                || (abandoned == null ? delivery.executionFinished() : p && d);
                if (!releaseAfterExit && abandoned == null) {
                    abandoned =
                            cancellation == null
                                    ? CancelReason.CLIENT_CANCELLED
                                    : cancellation.reason();
                }
                retainEnd(EndEvidence.deliveryFailure(route, transport), evidence.nowMs());
            }
        } else if (evidence instanceof RouteDelivered) {
            send = SendState.DELIVERED;
        } else if (evidence instanceof Abandonment abandonment) {
            if (abandoned == null && !canRelease()) {
                abandoned = abandonment.reason();
            }
            if (exact.kind() == DeliveryClaimKind.ROUTE_DECISION && send == SendState.NOT_STARTED) {
                send = SendState.ROUTE_CLOSED;
            }
        } else if (evidence instanceof CleanupAcknowledgement ack) {
            p |=
                    ack.kind() == CancelEvidenceKind.CLEANED
                            || ack.kind() == CancelEvidenceKind.FENCED;
            d |= ack.kind() == CancelEvidenceKind.CLEANED;
        } else if (evidence instanceof PreparationFailure failed) {
            if (send == SendState.SENDING) {
                throw new IllegalStateException("preparation failure crossed send boundary");
            }
            if (send == SendState.NOT_STARTED) {
                send = SendState.NOT_SENT;
            }
            retainEnd(EndEvidence.deliveryFailure(route, failed), failed.nowMs());
        }
        if (evidence instanceof Prediction || evidence instanceof RouteDelivered) {
            long at =
                    evidence instanceof Prediction predicted
                            ? predicted.predictedAtMs()
                            : ((RouteDelivered) evidence).predictedAtMs();
            engine = engine.predict(exact.sequence(), at);
        }
        if (send != previous.send()
                || abandoned != previous.abandonment()
                || p != previous.prefillProof()
                || d != previous.decodeProof()) {
            delivery = previous.recordResult(send, abandoned, p, d);
        }
        var after = refresh(evidence.nowMs());
        return update(before, after);
    }

    public synchronized Update recordEngineEvidence(EngineEvidence evidence) {
        var before = state;
        if (!matches(evidence.route(), evidence.reservation(), evidence.role())
                || phase == Phase.CLOSED) {
            return new Update(UpdateKind.STALE, before, before);
        }
        boolean isDecode = evidence.role() == RoleType.DECODE;
        boolean terminal =
                evidence.kind() == EngineKind.COMPLETED
                        || evidence.kind() == EngineKind.FAILED
                        || evidence.kind() == EngineKind.PRIORITY_CANCELLED
                        || evidence.kind() == EngineKind.RETIRED;
        engine = engine.observe(evidence, route.decodeEp() != null);
        if (evidence.kind() != EngineKind.RETIRED) {
            lastWorkerAtMs = Math.max(lastWorkerAtMs, evidence.nowMs());
        }
        if (preemption != null
                && !preemption.finished()
                && !preemption.restorationObserved()
                && (isDecode
                        ? evidence.kind() == EngineKind.ALLOCATED
                        : evidence.kind() == EngineKind.ACTIVE)) {
            preemption = preemption.observeRestoration();
        }
        if (terminal && delivery != null) {
            delivery =
                    delivery.mergeReleaseProof(
                            isDecode || evidence.role() == RoleType.PDFUSION, !isDecode, isDecode);
        }
        if (terminal) {
            close = close.mergeSettlement(!isDecode, isDecode);
            boolean ending =
                    isDecode
                            || evidence.role() == RoleType.PDFUSION
                            || evidence.kind() == EngineKind.FAILED
                            || evidence.kind() == EngineKind.PRIORITY_CANCELLED
                                    && preemption != null
                                    && !preemption.finished()
                                    && preemption.phase().acceptsPriorityTerminal();
            if (evidence.kind() == EngineKind.RETIRED && !isDecode) {
                ending = delivery == null && !engine.decodeAccepted() && preemption == null;
                if (placement != null && ending) {
                    placement =
                            new PlacementFacts(
                                    placement.operation(),
                                    placement.acquired(),
                                    placement.published(),
                                    placement.pendingTerminal(),
                                    evidence,
                                    placement.inactivityExpired());
                    ending = false;
                }
            }
            if (outcome == null && ending) {
                retainEnd(
                        new EndEvidence(
                                route,
                                evidence.kind() == EngineKind.RETIRED
                                        ? EndKind.RETIREMENT
                                        : evidence.kind() == EngineKind.PRIORITY_CANCELLED
                                                ? EndKind.PRIORITY
                                                : EndKind.WORKER,
                                evidence.role(),
                                evidence.kind() == EngineKind.COMPLETED,
                                evidence.errorCode(),
                                evidence.kind() == EngineKind.RETIRED
                                        ? StrategyErrorType.DISPATCH_FAILED
                                        : evidence.kind() == EngineKind.PRIORITY_CANCELLED
                                                ? StrategyErrorType.PRIORITY_PREEMPTED
                                                : StrategyErrorType.WORKER_EXECUTION_FAILED,
                                evidence.detail(),
                                !isDecode,
                                isDecode),
                        evidence.nowMs());
            }
        }
        var after = refresh(evidence.nowMs());
        return update(before, after);
    }

    public synchronized Update requestCancel(Cancellation request) {
        var before = state;
        if (key == null) {
            return new Update(UpdateKind.STALE, before, before);
        }
        Objects.requireNonNull(request.reason());
        long batchId = delivery == null ? 0L : delivery.identity().batchId();
        if (request.expectedBatchId() != 0L && request.expectedBatchId() != batchId) {
            return new Update(UpdateKind.STALE, before, before);
        }
        if (phase != Phase.OPEN) {
            return update(before, before);
        }
        if (cancellation != null) {
            return update(before, before);
        }
        if (request.reason() == CancelReason.DEADLINE_EXCEEDED
                && delivery != null
                && cancellation == null) {
            return update(before, before);
        }
        boolean beforeSend = delivery == null;
        if (request.frontend() && (selected != null || !beforeSend)) {
            return update(before, before);
        }
        StrategyErrorType error =
                request.errorType() != null
                        ? request.errorType()
                        : request.reason() == CancelReason.DEADLINE_EXCEEDED
                                ? StrategyErrorType.BATCH_SLO_EXPIRED
                                : StrategyErrorType.REQUEST_CANCELLED;
        if (cancellation == null) {
            cancellation =
                    new CancellationFacts(
                            request.reason(),
                            error,
                            request.detail() == null
                                    ? request.reason().getMessage()
                                    : request.detail());
        }
        if (delivery != null && !canRelease()) {
            var old = delivery;
            var send =
                    old.identity().kind() == DeliveryClaimKind.ROUTE_DECISION
                                    && old.send() == SendState.NOT_STARTED
                            ? SendState.ROUTE_CLOSED
                            : old.send();
            delivery = old.abandon(send, request.reason());
        }
        retainEnd(
                EndEvidence.failure(
                        route, EndKind.FRONTEND, cancellation.errorType(), cancellation.detail()),
                request.nowMs());
        return update(before, refresh(request.nowMs()));
    }

    public synchronized Update recordDeadline(
            ExpirationTimer.Token exact, long nowMs, CancellationFacts diagnosis) {
        var before = state;
        var current =
                switch (exact.kind()) {
                    case REQUEST -> deadlines.request();
                    case VISIBILITY -> deadlines.visibility();
                    case INACTIVITY -> deadlines.inactivity();
                };
        if (exact.owner() != this || current != exact || phase == Phase.CLOSED) {
            return new Update(UpdateKind.STALE, before, before);
        }
        if (nowMs < exact.atMs()) {
            deadlines =
                    new ExpirationTimer.Plan(
                            deadlines.revision() + 1L,
                            exact.kind() == ExpirationTimer.Kind.REQUEST
                                    ? null
                                    : deadlines.request(),
                            exact.kind() == ExpirationTimer.Kind.VISIBILITY
                                    ? null
                                    : deadlines.visibility(),
                            exact.kind() == ExpirationTimer.Kind.INACTIVITY
                                    ? null
                                    : deadlines.inactivity());
            return update(before, refresh(nowMs));
        }
        if (exact.kind() == ExpirationTimer.Kind.VISIBILITY) {
            engine = engine.expireVisibility();
        } else if (exact.kind() == ExpirationTimer.Kind.INACTIVITY) {
            deadlines =
                    new ExpirationTimer.Plan(
                            deadlines.revision() + 1L,
                            deadlines.request(),
                            deadlines.visibility(),
                            null);
            long activity =
                    Math.max(lastWorkerAtMs, delivery == null ? 0L : delivery.startedAtMs());
            if (nowMs - activity >= inactivityMs) {
                if (cancellation == null && outcome == null) {
                    cancellation =
                            new CancellationFacts(
                                    CancelReason.DEADLINE_EXCEEDED,
                                    diagnosis == null
                                            ? StrategyErrorType.BATCH_SLO_EXPIRED
                                            : diagnosis.errorType(),
                                    diagnosis == null ? "request inactive" : diagnosis.detail());
                }
                close = close.markExpired();
                retainEnd(
                        EndEvidence.failure(
                                route,
                                EndKind.INACTIVITY,
                                StrategyErrorType.BATCH_SLO_EXPIRED,
                                "request inactive"),
                        nowMs);
            }
        } else {
            requestDeadlineConsumed = true;
            if (outcome == null && (delivery == null || placement != null)) {
                if (cancellation == null) {
                    cancellation =
                            new CancellationFacts(
                                    CancelReason.DEADLINE_EXCEEDED,
                                    diagnosis == null
                                            ? StrategyErrorType.BATCH_SLO_EXPIRED
                                            : diagnosis.errorType(),
                                    diagnosis == null
                                            ? "request scheduling deadline exceeded"
                                            : diagnosis.detail());
                }
                if (placement != null) {
                    placement =
                            new PlacementFacts(
                                    placement.operation(),
                                    placement.acquired(),
                                    placement.published(),
                                    placement.pendingTerminal(),
                                    placement.pendingRetirement(),
                                    true);
                } else if (delivery == null && !engine.decodeAccepted()) {
                    retainEnd(
                            EndEvidence.failure(
                                    route,
                                    EndKind.INACTIVITY,
                                    StrategyErrorType.BATCH_SLO_EXPIRED,
                                    "request scheduling deadline exceeded"),
                            nowMs);
                }
            }
        }
        var after = refresh(nowMs);
        return update(before, after);
    }

    public synchronized Claim<PreemptionRegistration> beginPreemption(
            DecodeResources.ReservationHandle exact, long attempt, CancelTarget target) {
        var before = state;
        if (phase != Phase.OPEN
                || route == null
                || cancellation != null
                || preemption != null
                || !Objects.equals(route.decodeReservation(), exact)) {
            return new Rejected<>(before, before, Rejection.INVALID_STATE);
        }
        if (placement != null) {
            return new Blocked<>(before, before, Blocker.PLACEMENT);
        }
        if (delivery != null
                && delivery.identity().kind() == DeliveryClaimKind.ROUTE_DECISION
                && !delivery.acknowledged()) {
            return new Rejected<>(before, before, Rejection.INVALID_STATE);
        }
        var registration = new PreemptionRegistration(this, attempt, exact, target);
        preemption =
                new PreemptionFacts(
                        registration,
                        PreemptionCancelPhase.CLAIMED,
                        false,
                        false,
                        null,
                        false,
                        false);
        return new Granted<>(before, refresh(before.updatedAtMs()), registration);
    }

    public synchronized Claim<PreemptionRegistration> prepareReconciliation(
            PreemptionRegistration exact) {
        var before = state;
        if (preemption == null || preemption.identity() != exact || phase == Phase.CLOSED) {
            return new Stale<>(before);
        }
        if (preemption.finished()) {
            return new Rejected<>(before, before, Rejection.INVALID_STATE);
        }
        var terminal = preemption.pendingTerminal();
        boolean authoritative =
                terminal != null
                        && terminal.kind() == EndKind.WORKER
                        && (terminal.role() == RoleType.DECODE
                                || terminal.role() == RoleType.PDFUSION
                                || !terminal.success()
                                || engine.decodeAccepted());
        boolean priorityProof =
                terminal != null
                        && terminal.kind() == EndKind.PRIORITY
                        && preemption.phase().acceptsPriorityTerminal();
        boolean observed = preemption.restorationObserved();
        boolean eligible =
                authoritative
                        || priorityProof
                        || preemption.phase() == PreemptionCancelPhase.NOT_FOUND_STALE
                                && (terminal == null
                                        ? preemption.confirmationPending() || observed
                                        : observed || delivery != null && delivery.decodeProof());
        if (!eligible) {
            return new Blocked<>(before, before, Blocker.RELEASE_PROOF);
        }
        return new Granted<>(before, before, exact);
    }

    public synchronized Update completePreemption(
            PreemptionRegistration exact, PreemptionReport report, long nowMs) {
        var before = state;
        if (preemption == null || preemption.identity() != exact || phase == Phase.CLOSED) {
            return new Update(UpdateKind.STALE, before, before);
        }
        var previous = preemption;
        if (previous.finished() && report.phase() != null) {
            return update(before, before);
        }
        var next = report.phase() == null ? previous.phase() : report.phase();
        if (report.phase() != null
                && next != previous.phase()
                && !previous.phase().canTransitionTo(next)) {
            return update(before, before);
        }
        var terminal = report.terminal() == null ? previous.pendingTerminal() : report.terminal();
        preemption = previous.recordReport(next, terminal, report.ledgerCommitted());
        if (report.ledgerCommitted()) {
            if (delivery != null && terminal != null) {
                delivery =
                        delivery.mergeReleaseProof(
                                terminal.kind() == EndKind.WORKER,
                                terminal.prefillSettled(),
                                report.decodeSettled());
            }
            close =
                    close.mergeSettlement(
                            terminal != null && terminal.prefillSettled(),
                            report.decodeSettled() || terminal != null && terminal.decodeSettled());
            if (terminal != null) {
                retainEnd(terminal, nowMs);
            } else {
                preemption = null;
            }
        } else if (report.releaseAttempt() && next.isLocallyReleasable()) {
            preemption = null;
            if (terminal != null) {
                retainEnd(terminal, nowMs);
            }
        }
        var after = refresh(nowMs);
        return update(before, after);
    }

    public synchronized Claim<ResponseCandidate> selectResponse(ResponseCandidate candidate) {
        var before = state;
        if (candidate.owner() != this) {
            return new Stale<>(before);
        }
        if (key == null) {
            return new Rejected<>(before, before, Rejection.NOT_REGISTERED);
        }
        if (selected != null) {
            return selected == candidate
                    ? new Granted<>(before, before, selected)
                    : new Rejected<>(before, before, Rejection.ALREADY_SELECTED);
        }
        if (phase == Phase.CLOSED && candidate.kind() != ResponseKind.TERMINAL) {
            return new Rejected<>(before, before, Rejection.CLOSED);
        }
        boolean acknowledgement =
                candidate.kind() == ResponseKind.ACK
                        || candidate.kind() == ResponseKind.CONFIRMATION;
        if (acknowledgement) {
            if (phase != Phase.OPEN
                    || pendingEnd != null
                    || cancellation != null
                    || delivery == null
                    || delivery.abandonment() != null
                    || delivery.send() != SendState.DELIVERED
                            && !(delivery.send() == SendState.UNKNOWN && engine.decodeAccepted())) {
                return new Rejected<>(before, before, Rejection.INVALID_STATE);
            }
            if (preemption != null && !preemption.finished() && !delivery.acknowledged()) {
                preemption = preemption.awaitConfirmation();
                return new Blocked<>(before, refresh(candidate.nowMs()), Blocker.PREEMPTION);
            }
            long activity = Math.max(lastWorkerAtMs, delivery.startedAtMs());
            if (candidate.nowMs() - activity >= inactivityMs) {
                if (cancellation == null) {
                    cancellation =
                            new CancellationFacts(
                                    CancelReason.DEADLINE_EXCEEDED,
                                    StrategyErrorType.BATCH_SLO_EXPIRED,
                                    "request inactive before confirmation");
                }
                retainEnd(
                        EndEvidence.failure(
                                route,
                                EndKind.INACTIVITY,
                                StrategyErrorType.BATCH_SLO_EXPIRED,
                                "request inactive before confirmation"),
                        candidate.nowMs());
                return new Rejected<>(before, refresh(candidate.nowMs()), Rejection.CANCELLED);
            }
            if (!delivery.acknowledged()) {
                delivery = delivery.acknowledge();
            }
        } else if (candidate.kind() == ResponseKind.TERMINAL && outcome == null
                || candidate.kind() == ResponseKind.CANCELLATION && cancellation == null) {
            return new Rejected<>(before, before, Rejection.NO_OUTCOME);
        }
        if (candidate.kind() != ResponseKind.CONFIRMATION) {
            selected = candidate;
        }
        return new Granted<>(before, refresh(candidate.nowMs()), candidate);
    }

    /** Publisher capacity is reserved by Scheduler before freezing the outcome. */
    public synchronized Update beginClose(EndEvidence evidence, long nowMs) {
        var before = state;
        if (key == null) {
            return new Update(UpdateKind.STALE, before, before);
        }
        if (phase == Phase.CLOSED) {
            return update(before, before);
        }
        if (evidence != null && evidence.route() != null && evidence.route() != route) {
            return new Update(UpdateKind.STALE, before, before);
        }
        if (evidence != null) {
            retainEnd(evidence, nowMs);
        }
        boolean deliveryFailure =
                cancellation == null
                        && pendingEnd != null
                        && pendingEnd.kind() == EndKind.DELIVERY_FAILURE;
        if (placement != null && !deliveryFailure) {
            return update(before, refresh(nowMs));
        }
        if (outcome == null
                && preemption != null
                && !preemption.finished()
                && !deliveryFailure
                && (pendingEnd == null
                        || pendingEnd.kind() != EndKind.FRONTEND
                                && pendingEnd.kind() != EndKind.INACTIVITY
                                && pendingEnd.kind() != EndKind.RETIREMENT)) {
            return update(before, refresh(nowMs));
        }
        if (outcome == null && pendingEnd == null) {
            return update(before, before);
        }
        if (outcome == null) {
            outcome = classify(pendingEnd);
            phase = Phase.ENDING;
            close = close.beginEnding(pendingEnd, engine.prefillCompletedAtMs() > 0L);
            if (delivery != null) {
                var old = delivery;
                boolean success =
                        pendingEnd.kind() == EndKind.WORKER
                                && pendingEnd.success()
                                && cancellation == null;
                var abandoned =
                        success || canRelease()
                                ? old.abandonment()
                                : old.abandonment() != null
                                        ? old.abandonment()
                                        : cancellation == null
                                                ? CancelReason.CLIENT_CANCELLED
                                                : cancellation.reason();
                delivery = old.recordEnd(abandoned, success);
            }
            if (preemption != null
                    && (close.expired()
                            || pendingEnd.kind() == EndKind.RETIREMENT
                                    && pendingEnd.decodeSettled())) {
                preemption = preemption.finish();
            }
            pendingEnd = null;
        }
        return update(before, refresh(nowMs));
    }

    public synchronized DecodeResources.ReleaseReason releaseReason() {
        if (outcome == null || phase == Phase.CLOSED || placement != null || !canRelease()) {
            return null;
        }
        return delivery != null && delivery.send() == SendState.NOT_SENT
                ? DecodeResources.ReleaseReason.NOT_SENT
                : delivery != null && delivery.prefillProof() && delivery.decodeProof()
                        ? DecodeResources.ReleaseReason.REMOTE_CLEANUP
                        : close.expired()
                                ? DecodeResources.ReleaseReason.EXPIRED
                                : DecodeResources.ReleaseReason.COUNTERPART_FINISHED;
    }

    public synchronized Update completeClose(CloseProgress progress, long nowMs) {
        var before = state;
        if (phase != Phase.ENDING) {
            return new Update(UpdateKind.STALE, before, before);
        }
        if (delivery != null
                && delivery.identity().kind() == DeliveryClaimKind.ROUTE_DECISION
                && delivery.send() == SendState.NOT_STARTED) {
            delivery = delivery.withSendState(SendState.ROUTE_CLOSED);
        }
        close = close.recordProgress(progress, preemption == null || preemption.finished());
        if (preemption != null
                && close.decodeSettled()
                && (close.expired()
                        || delivery == null
                        || delivery.send() == SendState.NOT_SENT
                        || delivery.abandonment() != null
                                && delivery.prefillProof()
                                && delivery.decodeProof())) {
            preemption = preemption.finish();
        }
        if (close.prefillSettled()
                && close.decodeSettled()
                && close.notificationsDone()
                && close.timersDone()
                && canRelease()
                && placement == null
                && (preemption == null || preemption.finished())) {
            terminalRecord =
                    new RequestState(
                            key.requestId(),
                            outcome.outcome().phase(),
                            delivery == null ? DeliveryClaimKind.NONE : delivery.identity().kind(),
                            delivery == null ? 0L : delivery.identity().batchId(),
                            createdAtMs,
                            nowMs,
                            outcome.detail());
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
            close = close.retainEnd(evidence);
            return;
        }
        boolean deliveryFailure =
                cancellation == null && evidence.kind() == EndKind.DELIVERY_FAILURE;
        if (placement != null && !deliveryFailure) {
            var previous = placement.pendingTerminal();
            boolean better =
                    previous == null
                            || evidence.kind() == EndKind.WORKER
                                    && previous.kind() != EndKind.WORKER
                            || evidence.kind() == EndKind.RETIREMENT
                                    && previous.kind() != EndKind.WORKER;
            placement =
                    new PlacementFacts(
                            placement.operation(),
                            placement.acquired(),
                            placement.published(),
                            better ? evidence : previous,
                            placement.pendingRetirement(),
                            placement.inactivityExpired() || evidence.kind() == EndKind.INACTIVITY);
            return;
        }
        if (preemption != null
                && !preemption.finished()
                && !deliveryFailure
                && evidence.kind() != EndKind.FRONTEND
                && evidence.kind() != EndKind.INACTIVITY
                && evidence.kind() != EndKind.RETIREMENT) {
            var previous = preemption.pendingTerminal();
            boolean better =
                    previous == null
                            || evidence.kind() == EndKind.WORKER
                                    && previous.kind() != EndKind.WORKER;
            preemption = preemption.retainTerminal(better ? evidence : previous);
            return;
        }
        if (pendingEnd == null
                || evidence.kind() == EndKind.WORKER && pendingEnd.kind() != EndKind.WORKER) {
            pendingEnd = evidence;
        }
    }

    private End classify(EndEvidence evidence) {
        boolean cancelled =
                cancellation != null && (evidence.kind() != EndKind.FAILURE || delivery == null);
        if (cancelled || evidence.kind() == EndKind.INACTIVITY) {
            var cause =
                    cancellation == null ? CancelReason.DEADLINE_EXCEEDED : cancellation.reason();
            String detail = cancellation == null ? evidence.detail() : cancellation.detail();
            return new End(
                    evidence.kind(),
                    TerminalOutcome.cancellation(cause, detail),
                    cancellation == null
                            ? StrategyErrorType.BATCH_SLO_EXPIRED
                            : cancellation.errorType(),
                    detail);
        }
        if (evidence.kind() == EndKind.WORKER && evidence.success()
                || evidence.kind() == EndKind.FRONTEND && evidence.success()) {
            String detail =
                    evidence.detail() == null || evidence.detail().isBlank()
                            ? "request completed"
                            : evidence.detail();
            return new End(evidence.kind(), TerminalOutcome.complete(detail), null, detail);
        }
        var error =
                evidence.errorType() != null
                        ? evidence.errorType()
                        : evidence.kind() == EndKind.PRIORITY
                                ? StrategyErrorType.PRIORITY_PREEMPTED
                                : evidence.kind() == EndKind.WORKER
                                        ? StrategyErrorType.WORKER_EXECUTION_FAILED
                                        : StrategyErrorType.DISPATCH_FAILED;
        String detail =
                evidence.detail() == null || evidence.detail().isBlank()
                        ? evidence.kind() == EndKind.WORKER
                                ? "worker error code " + evidence.errorCode()
                                : evidence.kind() == EndKind.PRIORITY
                                        ? "priority victim canceled by worker"
                                        : "request ended"
                        : evidence.detail();
        return new End(
                evidence.kind(),
                evidence.kind() == EndKind.PRIORITY
                        ? TerminalOutcome.cancel(detail)
                        : TerminalOutcome.fail(detail),
                error,
                detail);
    }

    private boolean canRelease() {
        if (delivery == null) {
            return true;
        }
        if (delivery.send() == SendState.NOT_STARTED || delivery.send() == SendState.SENDING) {
            return false;
        }
        if (delivery.identity().kind() == DeliveryClaimKind.ROUTE_DECISION
                || delivery.send() == SendState.NOT_SENT) {
            return true;
        }
        return delivery.abandonment() == null
                ? delivery.executionFinished()
                : delivery.prefillProof() && delivery.decodeProof();
    }

    private boolean matches(
            RequestRoute exact, DecodeResources.ReservationHandle reservation, RoleType role) {
        if (route == null
                || route != exact
                || exact.ctx() != this
                || role == null
                || role != RoleType.DECODE && !role.supportsPrefill()) {
            return false;
        }
        return role != RoleType.DECODE || Objects.equals(route.decodeReservation(), reservation);
    }

    private long inactivityExpiresAtMs() {
        long activeSince = Math.max(lastWorkerAtMs, delivery == null ? 0L : delivery.startedAtMs());
        return activeSince > Long.MAX_VALUE - inactivityMs
                ? Long.MAX_VALUE
                : activeSince + inactivityMs;
    }

    private void deriveDeadlines() {
        long requestAt =
                phase == Phase.OPEN
                                && !requestDeadlineConsumed
                                && (delivery == null || !delivery.acknowledged())
                        ? absoluteDeadline
                        : 0L;
        long visibilityAt = phase == Phase.OPEN && outcome == null ? engine.visibilityAtMs() : 0L;
        boolean settled =
                phase == Phase.ENDING
                        && close.prefillSettled()
                        && close.decodeSettled()
                        && close.notificationsDone()
                        && close.timersDone()
                        && canRelease()
                        && placement == null
                        && (preemption == null || preemption.finished());
        long inactivityAt =
                phase == Phase.CLOSED || close.expired() || settled ? 0L : inactivityExpiresAtMs();
        var oldRequest = deadlines.request();
        var oldVisibility = deadlines.visibility();
        var oldInactivity = deadlines.inactivity();
        if (oldInactivity != null && inactivityAt > 0L && oldInactivity.atMs() <= inactivityAt) {
            inactivityAt =
                    oldInactivity.atMs(); // Existing wake-up will recheck the latest activity.
        }
        boolean sameRequest =
                requestAt <= 0L
                        ? oldRequest == null
                        : oldRequest != null && oldRequest.atMs() == requestAt;
        boolean sameVisibility =
                visibilityAt <= 0L
                        ? oldVisibility == null
                        : oldVisibility != null && oldVisibility.atMs() == visibilityAt;
        boolean sameInactivity =
                inactivityAt <= 0L
                        ? oldInactivity == null
                        : oldInactivity != null && oldInactivity.atMs() == inactivityAt;
        if (sameRequest && sameVisibility && sameInactivity) {
            return;
        }
        long revision = deadlines.revision() + 1L;
        var request =
                sameRequest
                        ? oldRequest
                        : requestAt <= 0L
                                ? null
                                : new ExpirationTimer.Token(
                                        this, ExpirationTimer.Kind.REQUEST, revision, requestAt);
        var visibility =
                sameVisibility
                        ? oldVisibility
                        : visibilityAt <= 0L
                                ? null
                                : new ExpirationTimer.Token(
                                        this,
                                        ExpirationTimer.Kind.VISIBILITY,
                                        revision,
                                        visibilityAt);
        var inactivity =
                sameInactivity
                        ? oldInactivity
                        : inactivityAt <= 0L
                                ? null
                                : new ExpirationTimer.Token(
                                        this,
                                        ExpirationTimer.Kind.INACTIVITY,
                                        revision,
                                        inactivityAt);
        deadlines = new ExpirationTimer.Plan(revision, request, visibility, inactivity);
    }

    private Update update(Snapshot before, Snapshot after) {
        if (after != before) {
            return new Update(UpdateKind.APPLIED, before, after);
        }
        if (cachedNoChange == null || cachedNoChange.before() != before) {
            cachedNoChange = new Update(UpdateKind.NO_CHANGE, before, before);
        }
        return cachedNoChange;
    }

    private Snapshot refresh(long nowMs) {
        if (key != null) {
            deriveDeadlines();
        }
        if (Objects.equals(state.placement(), placement)) {
            placement = state.placement();
        }
        if (Objects.equals(state.delivery(), delivery)) {
            delivery = state.delivery();
        }
        if (Objects.equals(state.engine(), engine)) {
            engine = state.engine();
        }
        if (Objects.equals(state.preemption(), preemption)) {
            preemption = state.preemption();
        }
        if (Objects.equals(state.close(), close)) {
            close = state.close();
        }
        if (Objects.equals(state.pendingEnd(), pendingEnd)) {
            pendingEnd = state.pendingEnd();
        }
        boolean sameFacts =
                state.key() == key
                        && state.createdAtMs() == createdAtMs
                        && state.phase() == phase
                        && state.route() == route
                        && state.placement() == placement
                        && state.delivery() == delivery
                        && state.engine() == engine
                        && state.preemption() == preemption
                        && state.cancellation() == cancellation
                        && state.outcome() == outcome
                        && state.pendingEnd() == pendingEnd
                        && state.selectedResponse() == selected
                        && state.close() == close
                        && state.deadlines() == deadlines
                        && state.terminalRecord() == terminalRecord;
        if (sameFacts && state.lastWorkerAtMs() == lastWorkerAtMs) {
            return state;
        }
        var facts =
                sameFacts
                        ? state.facts()
                        : new SnapshotFacts(
                                key,
                                createdAtMs,
                                phase,
                                route,
                                placement,
                                delivery,
                                engine,
                                preemption,
                                cancellation,
                                outcome,
                                pendingEnd,
                                selected,
                                close,
                                deadlines,
                                terminalRecord,
                                canRelease(),
                                phase == Phase.OPEN
                                        && route != null
                                        && delivery == null
                                        && cancellation == null
                                        && preemption == null
                                        && pendingEnd == null
                                        && (placement == null
                                                || placement.published()
                                                        && placement.operation().withdrawing()
                                                                == null
                                                        && placement.pendingTerminal() == null
                                                        && placement.pendingRetirement() == null
                                                        && !placement.inactivityExpired()));
        state =
                new Snapshot(
                        facts,
                        lastWorkerAtMs,
                        Math.max(state.updatedAtMs(), nowMs),
                        inactivityExpiresAtMs());
        return state;
    }
}
