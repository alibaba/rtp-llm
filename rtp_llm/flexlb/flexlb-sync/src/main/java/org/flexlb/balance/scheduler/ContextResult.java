package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;

import java.util.Objects;

/** Immutable facts and exact permissions, without execution callbacks or infrastructure handles. */
public final class ContextResult {
    private ContextResult() { }
    public enum UpdateKind { APPLIED, NO_CHANGE, STALE }
    public enum ClaimKind { GRANTED, BLOCKED, REJECTED, STALE }
    public enum Phase { OPEN, ENDING, CLOSED }
    public enum Blocker { PLACEMENT, PREEMPTION, SENDER, RELEASE_PROOF, CLOSE_IN_PROGRESS, RECONCILIATION }
    public enum Rejection { NOT_REGISTERED, CLOSED, CANCELLED, INVALID_STATE, ALREADY_SELECTED, NO_OUTCOME }
    public enum SendState { NOT_STARTED, SENDING, NOT_SENT, REJECTED, DELIVERED, UNKNOWN, ROUTE_CLOSED }
    public enum PublicationResult { PREPARED, PUBLISHED, NOT_PUBLISHED, WITHDRAWN, ROLLBACK_COMPLETED, FAILED }
    public enum ResponseKind { CONFIRMATION, ACK, TERMINAL, CANCELLATION, EXTERNAL_SUCCESS, EXTERNAL_FAILURE }
    public enum CancelKind { GLOBAL_QUEUE, WORKER_QUEUE, REMOTE, WAIT_PLACEMENT, WAIT_PREEMPTION }
    public enum CancelEvidenceKind { LOCAL_WITHDRAWN, CLEANED, FENCED, ACKNOWLEDGED, NOT_FOUND, UNKNOWN, UNSUPPORTED }
    public enum EndKind { FAILURE, DELIVERY_FAILURE, WORKER, RETIREMENT, PRIORITY, INACTIVITY, SHUTDOWN, FRONTEND }
    public enum EngineKind { ACTIVE, ALLOCATED, COMPLETED, FAILED, PRIORITY_CANCELLED, RETIRED }
    public enum ProgressKind { NOTIFICATIONS, RELEASE }

    public record Identity(long requestId, long incarnation) { }
    public record End(EndKind kind, TerminalOutcome outcome, StrategyErrorType errorType, String detail) { }
    public record EndEvidence(RequestRoute route, EndKind kind, RoleType role, boolean success,
                              long errorCode, StrategyErrorType errorType, String detail,
                              boolean prefillSettled, boolean decodeSettled) { }
    public record EngineEvidence(RequestRoute route, DecodeResources.ReservationHandle reservation,
                                 RoleType role, EngineKind kind, long errorCode, String detail, long nowMs) { }
    public record Cancellation(long expectedBatchId, CancelReason reason, boolean frontend, long nowMs,
                               StrategyErrorType errorType, String detail) { }
    public record CancellationFacts(CancelReason reason, StrategyErrorType errorType, String detail, CancelPermit operation) { }
    public record CancelEvidence(CancelEvidenceKind kind, boolean prefillSettled, boolean decodeSettled) { }
    public record PreemptionReport(PreemptionCancelPhase phase, boolean ledgerCommitted,
                                   boolean decodeSettled, boolean releaseAttempt, EndEvidence terminal) { }
    public record CloseProgress(ProgressKind kind, boolean prefillSettled, boolean decodeSettled,
                                boolean notificationsDone, boolean timersDone) { }
    public record ResponseCandidate(RequestContext owner, ResponseKind kind, End result, long nowMs) { }

    public sealed interface DeliveryEvidence permits TransportResult, Prediction, RouteDelivered, Abandonment,
            CleanupAcknowledgement, PreparationFailure { DeliveryClaim delivery(); long nowMs(); }
    public record TransportResult(DeliveryClaim delivery, DeliveryResult result, long nowMs) implements DeliveryEvidence { }
    public record Prediction(DeliveryClaim delivery, long predictedAtMs, long nowMs) implements DeliveryEvidence { }
    public record RouteDelivered(DeliveryClaim delivery, long predictedAtMs, long nowMs) implements DeliveryEvidence { }
    public record Abandonment(DeliveryClaim delivery, CancelReason reason, long nowMs) implements DeliveryEvidence { }
    public record CleanupAcknowledgement(DeliveryClaim delivery, CancelEvidenceKind kind, long nowMs) implements DeliveryEvidence { }
    public record PreparationFailure(DeliveryClaim delivery, RequestRoute route, Throwable cause,
                                     String detail, long nowMs) implements DeliveryEvidence {
        public PreparationFailure(DeliveryClaim delivery, RequestRoute route, String detail, long nowMs) {
            this(delivery, route, null, detail, nowMs);
        }
    }

    public interface GrantValue { RequestContext owner(); }
    public sealed interface CloseOperation extends GrantValue permits EndToken, ClosePermit { }
    public record PlacementToken(RequestContext owner, long sequence, RequestRoute withdrawing) implements GrantValue { }
    public record RoutePermit(RequestContext owner, PlacementToken operation, RequestRoute route,
                              boolean publishable) implements GrantValue { }
    public record CancelPermit(RequestContext owner, long sequence, RequestRoute route,
                               DeliveryClaim delivery, CancelKind kind, CancelReason reason) implements GrantValue { }
    public record ReconciliationPermit(RequestContext owner, PreemptionRegistration attempt,
                                       RequestRoute route, EndEvidence terminal,
                                       boolean confirmationPending) implements GrantValue { }
    public record EndToken(RequestContext owner, long sequence, RequestRoute route,
                            PreemptionRegistration preemption) implements CloseOperation { }
    public record ClosePermit(RequestContext owner, long sequence, RequestRoute route,
                              DecodeResources.ReleaseReason releaseReason,
                              boolean prefillSettled, boolean decodeSettled,
                              boolean notificationsDone, boolean timersDone) implements CloseOperation { }
    public record ResponseSelection(RequestContext owner, ResponseKind kind, End result,
                                    RequestRoute route, long batchId, long selectedAtMs, ResponseCandidate candidate) implements GrantValue { }

    public record PlacementFacts(PlacementToken operation, RequestRoute acquired, boolean published,
                                 EndEvidence pendingTerminal, EngineEvidence pendingRetirement,
                                 boolean inactivityExpired) { }
    public record EngineFacts(boolean prefillObserved, long prefillCompletedAtMs, boolean decodeAccepted,
                              long visibilityAtMs, boolean visibilityOverdue, long predictionSequence) { }
    public record DeliveryFacts(DeliveryClaim identity, SendState send, boolean acknowledged, CancelReason abandonment,
                                boolean executionFinished, boolean prefillProof, boolean decodeProof,
                                boolean transferred, long startedAtMs, long acknowledgedAtMs) { }
    public record PreemptionFacts(PreemptionRegistration identity, PreemptionCancelPhase phase,
                                  boolean cancelAcknowledged, boolean finished,
                                  EndEvidence pendingTerminal, boolean confirmationPending, boolean restorationObserved,
                                  ReconciliationPermit reconciliation) { }
    public record CloseFacts(EndToken endOperation, ClosePermit releaseOperation,
                             boolean prefillSettled, boolean decodeSettled,
                             boolean notificationsDone, boolean timersDone, boolean expired, boolean notificationsAttempted) { }
    /** Control facts are shared by observations that change only activity timestamps. */
    public record SnapshotFacts(Identity key, long createdAtMs, Phase phase, RequestRoute route, PlacementFacts placement,
                               DeliveryFacts delivery, EngineFacts engine, PreemptionFacts preemption, CancellationFacts cancellation,
                               End outcome, EndEvidence pendingEnd, ResponseSelection selectedResponse, CloseFacts close,
                               ExpirationTimer.Plan deadlines, RequestState terminalRecord,
                               boolean deliveryReleasable, boolean preparedDelivery, boolean localEndEligible,
                               boolean frontendCancelEligible) { }
    public record Snapshot(@lombok.experimental.Delegate SnapshotFacts facts, long lastWorkerAtMs,
                           long updatedAtMs, long inactivityExpiresAtMs) {
        public Snapshot(Identity key, long createdAtMs, Phase phase, RequestRoute route, PlacementFacts placement,
                        DeliveryFacts delivery, EngineFacts engine, PreemptionFacts preemption, CancellationFacts cancellation,
                        End outcome, EndEvidence pendingEnd, ResponseSelection selectedResponse, CloseFacts close,
                        ExpirationTimer.Plan deadlines, RequestState terminalRecord, long lastWorkerAtMs, long updatedAtMs,
                        boolean deliveryReleasable, boolean preparedDelivery, boolean localEndEligible,
                        boolean frontendCancelEligible, long inactivityExpiresAtMs) {
            this(new SnapshotFacts(key, createdAtMs, phase, route, placement, delivery, engine, preemption, cancellation,
                    outcome, pendingEnd, selectedResponse, close, deadlines, terminalRecord, deliveryReleasable,
                    preparedDelivery, localEndEligible, frontendCancelEligible), lastWorkerAtMs, updatedAtMs, inactivityExpiresAtMs);
        }
    }


    public record Update(UpdateKind kind, Snapshot before, Snapshot after) {
        public Update {
            Objects.requireNonNull(kind); Objects.requireNonNull(before); Objects.requireNonNull(after);
            if (before.key() != null && before.key() != after.key()) { throw new IllegalArgumentException("request incarnation changed"); }
            if (kind != UpdateKind.APPLIED && before != after) {
                throw new IllegalArgumentException("unchanged updates must reuse their snapshot");
            }
        }
    }
    public sealed interface Claim<T extends GrantValue> permits Granted, Blocked, Rejected, Stale {
        ClaimKind kind(); Snapshot before(); Snapshot after();
    }
    public record Granted<T extends GrantValue>(Snapshot before, Snapshot after, T value) implements Claim<T> {
        public Granted { Objects.requireNonNull(before); Objects.requireNonNull(after); Objects.requireNonNull(value); }
        @Override public ClaimKind kind() { return ClaimKind.GRANTED; }
    }
    public record Blocked<T extends GrantValue>(Snapshot before, Snapshot after, Blocker blocker) implements Claim<T> {
        public Blocked { Objects.requireNonNull(blocker); }
        @Override public ClaimKind kind() { return ClaimKind.BLOCKED; }
    }
    public record Rejected<T extends GrantValue>(Snapshot before, Snapshot after, Rejection reason) implements Claim<T> {
        public Rejected { Objects.requireNonNull(reason); }
        @Override public ClaimKind kind() { return ClaimKind.REJECTED; }
    }
    public record Stale<T extends GrantValue>(Snapshot before) implements Claim<T> {
        @Override public ClaimKind kind() { return ClaimKind.STALE; }
        @Override public Snapshot after() { return before; }
    }
}
