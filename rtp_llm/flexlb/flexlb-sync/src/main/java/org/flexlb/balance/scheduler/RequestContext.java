package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

import static org.flexlb.balance.scheduler.ContextResult.Cancellation;
import static org.flexlb.balance.scheduler.ContextResult.CancellationFacts;
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
import static org.flexlb.balance.scheduler.ContextResult.Identity;
import static org.flexlb.balance.scheduler.ContextResult.PlacementFacts;
import static org.flexlb.balance.scheduler.ContextResult.Prediction;
import static org.flexlb.balance.scheduler.ContextResult.PreemptionFacts;
import static org.flexlb.balance.scheduler.ContextResult.PreemptionReport;
import static org.flexlb.balance.scheduler.ContextResult.PreparationFailure;
import static org.flexlb.balance.scheduler.ContextResult.PublicationResult;
import static org.flexlb.balance.scheduler.ContextResult.ResponseCandidate;
import static org.flexlb.balance.scheduler.ContextResult.ResponseKind;
import static org.flexlb.balance.scheduler.ContextResult.RouteDelivered;
import static org.flexlb.balance.scheduler.ContextResult.SendState;
import static org.flexlb.balance.scheduler.ContextResult.Snapshot;
import static org.flexlb.balance.scheduler.ContextResult.TransportResult;

/** Atomic request facts; Lifecycle settles results and Dispatcher performs transport outside this lock. */
public final class RequestContext {
    private static final AtomicLong INCARNATIONS = new AtomicLong();
    private final RequestData data;
    private Identity key;
    private static final Snapshot EMPTY =
            new Snapshot(
                    null,
                    0L,
                    null,
                    null,
                    null,
                    new EngineFacts(false, 0L, false, 0L, false, null),
                    null,
                    null,
                    null,
                    null,
                    null,
                    new CloseFacts(true, true, false, false, false),
                    new ExpirationTimer.Plan(0L, null, null, null),
                    null,
                    0L,
                    0L,
                    0L);
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
    private long createdAtMs;
    private long lastWorkerAtMs;
    private long absoluteDeadline;
    private boolean requestDeadlineConsumed;
    private long inactivityMs;

    public RequestContext(FlexlbConfig config) {
        data = new RequestData(Objects.requireNonNull(config));
        state = EMPTY;
    }

    public RequestData data() {
        return data;
    }

    public synchronized void register(
            long requestId, long nowMs, long expiresAtMs, long inactivityTimeoutMs) {
        if (key != null) {
            return;
        }
        if (inactivityTimeoutMs <= 0L) {
            throw new IllegalArgumentException("positive inactivity timeout required");
        }
        key = new Identity(requestId, INCARNATIONS.incrementAndGet());
        createdAtMs = lastWorkerAtMs = nowMs;
        absoluteDeadline = expiresAtMs;
        inactivityMs = inactivityTimeoutMs;
        commitSnapshot(nowMs);
    }

    public synchronized Snapshot snapshot() {
        return state;
    }

    public synchronized AdmissionHandle beginPlacement(RequestRoute withdrawing) {
        var before = state;
        if (key == null
                || outcome != null
                || cancellation != null
                || pendingEnd != null
                || placement != null
                || preemption != null) {
            return null;
        }
        if (withdrawing == null
                ? route != null
                : route != withdrawing || delivery != null || engine.decodeAccepted()) {
            return null;
        }
        var token = new AdmissionHandle(this, withdrawing);
        placement = new PlacementFacts(token, withdrawing != null, null, false);
        commitSnapshot(before.updatedAtMs());
        return token;
    }

    public synchronized boolean bindPlacement(
            AdmissionHandle exact, RequestRoute acquired, PublicationResult result) {
        var before = state;
        Objects.requireNonNull(acquired);
        if (placement == null || placement.operation() != exact || exact.owner() != this) {
            return false;
        }
        if (result != PublicationResult.PREPARED) {
            if (route != acquired) {
                return false;
            }
            boolean withdrawn =
                    result == PublicationResult.WITHDRAWN
                            || result == PublicationResult.ROLLBACK_COMPLETED;
            placement = placement.withPublished(result == PublicationResult.PUBLISHED);
            if (withdrawn) {
                route = null;
                close = close.withSettlement(true, true);
            }
            commitSnapshot(before.updatedAtMs());
            return true;
        }
        if (acquired.requestId() != key.requestId()
                || acquired.ctx() != this
                || placement.published()
                || (route != null && route != acquired)) {
            return false;
        }
        placement = placement.withPublished(false);
        route = acquired;
        close =
                close.withSettlement(
                        false, acquired.decodeEp() == null || acquired.decodeReservation() == null);
        boolean publishable = outcome == null && cancellation == null;
        commitSnapshot(before.updatedAtMs());
        return publishable;
    }

    public synchronized Snapshot completePlacement(
            AdmissionHandle exact, PublicationResult result, long nowMs) {
        if (placement == null || placement.operation() != exact) {
            return null;
        }
        var retained = placement;
        placement = null;
        if (result == PublicationResult.WITHDRAWN
                || result == PublicationResult.ROLLBACK_COMPLETED) {
            route = null;
            delivery = null;
            close = close.withSettlement(true, true);
        }
        boolean rolledBack = route == null && close.prefillSettled() && close.decodeSettled();
        if (rolledBack && pendingEnd != null) {
            pendingEnd = pendingEnd.withRoute(null);
        }
        boolean admissionFailure =
                result == PublicationResult.FAILED && outcome == null && cancellation == null;
        if (admissionFailure && pendingEnd != null && pendingEnd.kind() == EndKind.SHUTDOWN) {
            pendingEnd = null;
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
                                    || retired.role() == RoleType.PDFUSION));
        }
        if (outcome == null && retained.inactivityExpired()) {
            retainEnd(
                    EndEvidence.failure(
                            route,
                            EndKind.INACTIVITY,
                            StrategyErrorType.BATCH_SLO_EXPIRED,
                            "request inactive during placement"));
        }
        return commitSnapshot(nowMs);
    }

    /**
     * Prepare immutable facts locally; Dispatcher holds this monitor through Decode transfer and publish.
     */
    public synchronized DeliveryFacts prepareDelivery(
            Delivery.Member claim, Delivery.SendMode kind, long batchId, long nowMs) {
        RequestRoute exact = claim.route();
        if (route != exact || !state.preparedDelivery()) {
            return null;
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
                            cancellation.detail()));
            commitSnapshot(nowMs);
            return null;
        }
        claim.initialize(kind, batchId);
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
        return prepared;
    }

    /** Dispatcher retains this monitor from preparation through Decode transfer and publication. */
    synchronized void publishDelivery(DeliveryFacts prepared) {
        delivery = prepared;
        commitSnapshot(prepared.startedAtMs());
    }

    public enum SendResult {
        STARTED,
        CANCELLED,
        UNAVAILABLE
    }

    public synchronized SendResult beginSend(Delivery.Member exact, long nowMs) {
        if (delivery == null || delivery.identity() != exact) {
            return SendResult.UNAVAILABLE;
        }
        if (delivery.send() != SendState.NOT_STARTED
                || exact.kind() != Delivery.SendMode.BATCH_ENQUEUE) {
            return SendResult.UNAVAILABLE;
        }
        if (outcome != null
                || cancellation != null
                || delivery.abandonment() != null
                || pendingEnd != null
                || placement != null && placement.pendingRetirement() != null
                || preemption != null && preemption.pendingTerminal() != null
                || absoluteDeadline > 0L && nowMs >= absoluteDeadline
                || nowMs - lastWorkerAtMs >= inactivityMs) {
            return SendResult.CANCELLED;
        }
        delivery = delivery.withSend(SendState.SENDING);
        commitSnapshot(nowMs);
        return SendResult.STARTED;
    }

    public synchronized Snapshot recordDeliveryResult(DeliveryEvidence evidence) {
        var before = state;
        Objects.requireNonNull(evidence);
        var exact = evidence.delivery();
        if (evidence instanceof PreparationFailure failed && exact == null) {
            if (route != failed.route() || outcome != null || delivery != null) {
                return null;
            }
            retainEnd(EndEvidence.deliveryFailure(route, failed));
            return commitSnapshot(failed.nowMs());
        }
        if (delivery == null || delivery.identity() != exact || terminalRecord != null) {
            return null;
        }
        if ((evidence instanceof Prediction || evidence instanceof RouteDelivered)
                && outcome != null) {
            return null;
        }
        var recorded =
                delivery.record(
                        evidence,
                        cancellation == null
                                ? CancelReason.CLIENT_CANCELLED
                                : cancellation.reason());
        if (evidence instanceof TransportResult && recorded == delivery) return before;
        delivery = recorded;
        if (evidence instanceof PreparationFailure
                || evidence instanceof TransportResult transport && transport.result().failed()) {
            retainEnd(EndEvidence.deliveryFailure(route, evidence));
        }
        if (evidence instanceof Prediction || evidence instanceof RouteDelivered) {
            long at =
                    evidence instanceof Prediction predicted
                            ? predicted.predictedAtMs()
                            : ((RouteDelivered) evidence).predictedAtMs();
            engine = engine.predict(exact, at);
        }
        return commitSnapshot(evidence.nowMs());
    }

    public synchronized Snapshot recordEngineEvidence(EngineEvidence evidence) {
        if (!matches(evidence.route(), evidence.reservation(), evidence.role())
                || terminalRecord != null) {
            return null;
        }
        boolean isDecode = evidence.role() == RoleType.DECODE;
        engine = engine.observe(evidence, route.decodeEp() != null);
        if (evidence.kind() != EngineKind.RETIRED)
            lastWorkerAtMs = Math.max(lastWorkerAtMs, evidence.nowMs());
        if (preemption != null
                && !preemption.finished()
                && !preemption.restorationObserved()
                && (isDecode
                        ? evidence.kind() == EngineKind.ALLOCATED
                        : evidence.kind() == EngineKind.ACTIVE)) {
            preemption = preemption.withRestorationObserved(true);
        }
        if (!evidence.terminal()) return commitSnapshot(evidence.nowMs());
        if (delivery != null) {
            delivery =
                    delivery.mergeReleaseProof(
                            isDecode || evidence.role() == RoleType.PDFUSION, !isDecode, isDecode);
        }
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
                placement = placement.withPendingRetirement(evidence);
                return commitSnapshot(evidence.nowMs());
            }
        }
        if (outcome == null && ending) retainEnd(evidence.end());
        return commitSnapshot(evidence.nowMs());
    }

    public synchronized boolean requestCancel(Cancellation request) {
        var before = state;
        if (key == null) {
            return false;
        }
        Objects.requireNonNull(request.reason());
        long batchId = delivery == null ? 0L : delivery.identity().batchId();
        if (request.expectedBatchId() != 0L && request.expectedBatchId() != batchId) {
            return false;
        }
        if (outcome != null) {
            return false;
        }
        if (cancellation != null) {
            return false;
        }
        if (request.reason() == CancelReason.DEADLINE_EXCEEDED && delivery != null) {
            return false;
        }
        boolean beforeSend = delivery == null;
        if (request.frontend() && (selected != null || !beforeSend)) {
            return false;
        }
        StrategyErrorType error =
                request.errorType() != null
                        ? request.errorType()
                        : request.reason() == CancelReason.DEADLINE_EXCEEDED
                                ? StrategyErrorType.BATCH_SLO_EXPIRED
                                : StrategyErrorType.REQUEST_CANCELLED;
        cancellation =
                new CancellationFacts(
                        request.reason(),
                        error,
                        request.detail() == null
                                ? request.reason().getMessage()
                                : request.detail());
        if (delivery != null && !DeliveryFacts.releasable(delivery)) {
            var old = delivery;
            var send =
                    old.identity().kind() == Delivery.SendMode.ROUTE_DECISION
                                    && old.send() == SendState.NOT_STARTED
                            ? SendState.ROUTE_CLOSED
                            : old.send();
            delivery = old.abandon(send, request.reason());
        }
        retainEnd(
                EndEvidence.failure(
                        route, EndKind.FRONTEND, cancellation.errorType(), cancellation.detail()));
        return commitSnapshot(request.nowMs()) != before;
    }

    public synchronized Snapshot recordDeadline(
            ExpirationTimer.Token exact, long nowMs, CancellationFacts diagnosis) {
        if (exact.owner() != this
                || deadlines.token(exact.kind()) != exact
                || terminalRecord != null) {
            return null;
        }
        if (nowMs < exact.atMs()) {
            deadlines = deadlines.without(exact.kind());
            return commitSnapshot(nowMs);
        }
        if (exact.kind() == ExpirationTimer.Kind.VISIBILITY) {
            engine = engine.expireVisibility();
            return commitSnapshot(nowMs);
        }
        boolean inactive = exact.kind() == ExpirationTimer.Kind.INACTIVITY;
        if (inactive) {
            deadlines = deadlines.without(exact.kind());
            long activity =
                    Math.max(lastWorkerAtMs, delivery == null ? 0L : delivery.startedAtMs());
            if (nowMs - activity < inactivityMs) return commitSnapshot(nowMs);
            close = close.withExpired(true);
        } else {
            requestDeadlineConsumed = true;
            if (outcome != null || delivery != null && placement == null)
                return commitSnapshot(nowMs);
        }
        String detail = inactive ? "request inactive" : "request scheduling deadline exceeded";
        if (cancellation == null && outcome == null) {
            cancellation =
                    new CancellationFacts(
                            CancelReason.DEADLINE_EXCEEDED,
                            diagnosis == null
                                    ? StrategyErrorType.BATCH_SLO_EXPIRED
                                    : diagnosis.errorType(),
                            diagnosis == null ? detail : diagnosis.detail());
        }
        if (!inactive && placement != null) {
            placement = placement.withInactivityExpired(true);
        } else if (inactive || delivery == null && !engine.decodeAccepted()) {
            retainEnd(
                    EndEvidence.failure(
                            route,
                            EndKind.INACTIVITY,
                            StrategyErrorType.BATCH_SLO_EXPIRED,
                            detail));
        }
        return commitSnapshot(nowMs);
    }

    public synchronized PreemptionRegistration beginPreemption(
            DecodeResources.ReservationHandle exact, long attempt, CancelTarget target) {
        var before = state;
        if (outcome != null
                || route == null
                || cancellation != null
                || preemption != null
                || !Objects.equals(route.decodeReservation(), exact)) {
            return null;
        }
        if (placement != null) {
            return null;
        }
        if (delivery != null
                && delivery.identity().kind() == Delivery.SendMode.ROUTE_DECISION
                && !delivery.acknowledged()) {
            return null;
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
        commitSnapshot(before.updatedAtMs());
        return registration;
    }

    public synchronized boolean canReconcilePreemption(PreemptionRegistration exact) {
        if (preemption == null || preemption.identity() != exact || terminalRecord != null) {
            return false;
        }
        if (preemption.finished()) {
            return false;
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
        return eligible;
    }

    public synchronized boolean completePreemption(
            PreemptionRegistration exact, PreemptionReport report, long nowMs) {
        var before = state;
        if (preemption == null || preemption.identity() != exact || terminalRecord != null) {
            return false;
        }
        var previous = preemption;
        if (previous.finished() && report.phase() != null) {
            return false;
        }
        var next = report.phase() == null ? previous.phase() : report.phase();
        if (report.phase() != null
                && next != previous.phase()
                && !previous.phase().canTransitionTo(next)) {
            return false;
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
                retainEnd(terminal);
            } else {
                preemption = null;
            }
        } else if (report.releaseAttempt() && next.isLocallyReleasable()) {
            preemption = null;
            if (terminal != null) {
                retainEnd(terminal);
            }
        }
        return commitSnapshot(nowMs) != before;
    }

    public synchronized boolean selectResponse(ResponseCandidate candidate) {
        if (candidate.owner() != this) {
            return false;
        }
        if (key == null) {
            return false;
        }
        if (selected != null) {
            return selected == candidate;
        }
        if (terminalRecord != null && candidate.kind() != ResponseKind.TERMINAL) {
            return false;
        }
        if (candidate.kind() == ResponseKind.ACK) {
            if (!confirmDelivery(candidate.nowMs())) return false;
        } else if (candidate.kind() == ResponseKind.TERMINAL && outcome == null
                || candidate.kind() == ResponseKind.CANCELLATION && cancellation == null) {
            return false;
        }
        selected = candidate;
        commitSnapshot(candidate.nowMs());
        return true;
    }

    /** A transport receipt does not select or publish the frontend response. */
    synchronized boolean confirmDelivery(long nowMs) {
        if (!state.canConfirmDelivery()) return false;
        if (preemption != null && !preemption.finished() && !delivery.acknowledged()) {
            preemption = preemption.withConfirmationPending(true);
            commitSnapshot(nowMs);
            return false;
        }
        long activity = Math.max(lastWorkerAtMs, delivery.startedAtMs());
        if (nowMs - activity >= inactivityMs) {
            cancellation =
                    new CancellationFacts(
                            CancelReason.DEADLINE_EXCEEDED,
                            StrategyErrorType.BATCH_SLO_EXPIRED,
                            "request inactive before confirmation");
            retainEnd(
                    EndEvidence.failure(
                            route,
                            EndKind.INACTIVITY,
                            StrategyErrorType.BATCH_SLO_EXPIRED,
                            "request inactive before confirmation"));
            commitSnapshot(nowMs);
            return false;
        }
        if (!delivery.acknowledged()) {
            delivery = delivery.withAcknowledged(true);
        }
        commitSnapshot(nowMs);
        return true;
    }

    /** Publisher capacity is reserved by Scheduler before freezing the outcome. */
    synchronized boolean readyToClose(EndKind ending) {
        return !waitsForPlacement(ending) && !waitsForPreemption(ending);
    }

    private boolean waitsForPlacement(EndKind ending) {
        return placement != null && !(ending == EndKind.DELIVERY_FAILURE && cancellation == null);
    }

    private boolean waitsForPreemption(EndKind ending) {
        return preemption != null
                && !preemption.finished()
                && !(ending == EndKind.DELIVERY_FAILURE && cancellation == null)
                && ending != EndKind.FRONTEND
                && ending != EndKind.INACTIVITY
                && ending != EndKind.RETIREMENT;
    }

    public synchronized Snapshot beginClose(EndEvidence evidence, long nowMs) {
        var before = state;
        if (key == null) {
            return before;
        }
        if (terminalRecord != null) {
            return before;
        }
        if (evidence != null && evidence.route() != null && evidence.route() != route) {
            return before;
        }
        if (evidence != null) {
            retainEnd(evidence);
        }
        EndKind ending = pendingEnd == null ? null : pendingEnd.kind();
        if (waitsForPlacement(ending) || outcome == null && waitsForPreemption(ending)) {
            return commitSnapshot(nowMs);
        }
        if (outcome == null && pendingEnd == null) {
            return before;
        }
        if (outcome == null) {
            outcome = classify(pendingEnd);
            close = close.beginEnding(pendingEnd, engine.prefillCompletedAtMs() > 0L);
            if (delivery != null) {
                var old = delivery;
                boolean success =
                        pendingEnd.kind() == EndKind.WORKER
                                && pendingEnd.success()
                                && cancellation == null;
                var abandoned =
                        success || DeliveryFacts.releasable(delivery)
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
                preemption = preemption.withFinished(true);
            }
            pendingEnd = null;
        }
        return commitSnapshot(nowMs);
    }

    public synchronized DecodeResources.ReleaseReason releaseReason() {
        if (outcome == null
                || terminalRecord != null
                || placement != null
                || !DeliveryFacts.releasable(delivery)) {
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

    public synchronized void completeClose(CloseProgress progress, long nowMs) {
        if ((outcome == null || terminalRecord != null)) {
            return;
        }
        if (delivery != null
                && delivery.identity().kind() == Delivery.SendMode.ROUTE_DECISION
                && delivery.send() == SendState.NOT_STARTED) {
            delivery = delivery.withSend(SendState.ROUTE_CLOSED);
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
            preemption = preemption.withFinished(true);
        }
        if (close.prefillSettled()
                && close.decodeSettled()
                && close.notificationsDone()
                && DeliveryFacts.releasable(delivery)
                && placement == null
                && (preemption == null || preemption.finished())) {
            terminalRecord =
                    new RequestState(
                            key.requestId(),
                            outcome.phase(),
                            delivery == null ? Delivery.SendMode.NONE : delivery.identity().kind(),
                            delivery == null ? 0L : delivery.identity().batchId(),
                            createdAtMs,
                            nowMs,
                            outcome.detail());
            route = null;
            preemption = null;
            pendingEnd = null;
        }
        commitSnapshot(nowMs);
    }

    private void retainEnd(EndEvidence evidence) {
        if (outcome != null) {
            close = close.retainEnd(evidence);
            return;
        }
        if (placement != null) {
            // A preparation failure can settle before placement returns; other ends wait for it.
            boolean immediateFailure =
                    evidence.kind() == EndKind.DELIVERY_FAILURE && cancellation == null;
            boolean previousFailure =
                    pendingEnd != null
                            && pendingEnd.kind() == EndKind.DELIVERY_FAILURE
                            && cancellation == null;
            if (immediateFailure
                    || !previousFailure
                            && (pendingEnd == null
                                    || evidence.kind() == EndKind.WORKER
                                            && pendingEnd.kind() != EndKind.WORKER
                                    || evidence.kind() == EndKind.RETIREMENT
                                            && pendingEnd.kind() != EndKind.WORKER)) {
                pendingEnd = evidence;
            }
            if (evidence.kind() == EndKind.INACTIVITY && !placement.inactivityExpired()) {
                placement = placement.withInactivityExpired(true);
            }
            return;
        }
        if (waitsForPreemption(evidence.kind())) {
            var previous = preemption.pendingTerminal();
            boolean better =
                    previous == null
                            || evidence.kind() == EndKind.WORKER
                                    && previous.kind() != EndKind.WORKER;
            preemption = preemption.withPendingTerminal(better ? evidence : previous);
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
                    cause == CancelReason.DEADLINE_EXCEEDED
                            ? RequestState.Phase.TIMED_OUT
                            : RequestState.Phase.CANCELLED,
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
            return new End(evidence.kind(), RequestState.Phase.COMPLETED, null, detail);
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
                        ? RequestState.Phase.CANCELLED
                        : RequestState.Phase.FAILED,
                error,
                detail);
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

    private void deriveDeadlines(long inactivityAtMs) {
        long requestAt =
                outcome == null
                                && !requestDeadlineConsumed
                                && (delivery == null || !delivery.acknowledged())
                        ? absoluteDeadline
                        : 0L;
        long visibilityAt = outcome == null ? engine.visibilityAtMs() : 0L;
        boolean settled =
                outcome != null
                        && terminalRecord == null
                        && close.prefillSettled()
                        && close.decodeSettled()
                        && close.notificationsDone()
                        && DeliveryFacts.releasable(delivery)
                        && placement == null
                        && (preemption == null || preemption.finished());
        long inactivityAt =
                terminalRecord != null || close.expired() || settled ? 0L : inactivityAtMs;
        deadlines = deadlines.schedule(this, requestAt, visibilityAt, inactivityAt);
    }

    private Snapshot commitSnapshot(long nowMs) {
        long inactivityAtMs =
                com.google.common.math.LongMath.saturatedAdd(
                        Math.max(lastWorkerAtMs, delivery == null ? 0L : delivery.startedAtMs()),
                        inactivityMs);
        if (key != null) {
            deriveDeadlines(inactivityAtMs);
        }

        // An unchanged observation preserves snapshot identity and does not allocate a wrapper.
        if (Objects.equals(key, state.key())
                && createdAtMs == state.createdAtMs()
                && route == state.route()
                && Objects.equals(placement, state.placement())
                && Objects.equals(delivery, state.delivery())
                && Objects.equals(engine, state.engine())
                && Objects.equals(preemption, state.preemption())
                && Objects.equals(cancellation, state.cancellation())
                && Objects.equals(outcome, state.outcome())
                && Objects.equals(pendingEnd, state.pendingEnd())
                && Objects.equals(selected, state.selectedResponse())
                && Objects.equals(close, state.close())
                && Objects.equals(deadlines, state.deadlines())
                && Objects.equals(terminalRecord, state.terminalRecord())
                && lastWorkerAtMs == state.lastWorkerAtMs()) {
            return state;
        }
        state =
                new Snapshot(
                        key,
                        createdAtMs,
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
                        lastWorkerAtMs,
                        Math.max(state.updatedAtMs(), nowMs),
                        inactivityAtMs);
        return state;
    }
}
