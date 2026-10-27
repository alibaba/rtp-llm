package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;

import java.util.Objects;

import static com.google.common.math.LongMath.saturatedAdd;

/** Immutable facts and exact permissions, without execution callbacks or infrastructure handles. */
public final class ContextResult {
    private ContextResult() {}

    public enum UpdateKind {
        APPLIED,
        NO_CHANGE,
        STALE
    }

    public enum ClaimKind {
        GRANTED,
        BLOCKED,
        REJECTED,
        STALE
    }

    public enum Phase {
        OPEN,
        ENDING,
        CLOSED
    }

    public enum Blocker {
        PLACEMENT,
        PREEMPTION,
        RELEASE_PROOF
    }

    public enum Rejection {
        NOT_REGISTERED,
        CLOSED,
        CANCELLED,
        INVALID_STATE,
        ALREADY_SELECTED,
        NO_OUTCOME
    }

    public enum SendState {
        NOT_STARTED,
        SENDING,
        NOT_SENT,
        REJECTED,
        DELIVERED,
        UNKNOWN,
        ROUTE_CLOSED
    }

    public enum PublicationResult {
        PREPARED,
        RESTORED,
        PUBLISHED,
        NOT_PUBLISHED,
        WITHDRAWN,
        ROLLBACK_COMPLETED,
        FAILED
    }

    public enum ResponseKind {
        CONFIRMATION,
        ACK,
        TERMINAL,
        CANCELLATION,
        EXTERNAL_SUCCESS,
        EXTERNAL_FAILURE
    }

    public enum CancelEvidenceKind {
        CLEANED,
        FENCED,
        ACKNOWLEDGED,
        NOT_FOUND,
        UNKNOWN,
        UNSUPPORTED
    }

    public enum EndKind {
        FAILURE,
        DELIVERY_FAILURE,
        WORKER,
        RETIREMENT,
        PRIORITY,
        INACTIVITY,
        SHUTDOWN,
        FRONTEND
    }

    public enum EngineKind {
        ACTIVE,
        ALLOCATED,
        COMPLETED,
        FAILED,
        PRIORITY_CANCELLED,
        RETIRED
    }

    public record Identity(long requestId, long incarnation) {}

    public record End(
            EndKind kind, TerminalOutcome outcome, StrategyErrorType errorType, String detail) {}

    public record EndEvidence(
            RequestRoute route,
            EndKind kind,
            RoleType role,
            boolean success,
            long errorCode,
            StrategyErrorType errorType,
            String detail,
            boolean prefillSettled,
            boolean decodeSettled) {
        public static EndEvidence failure(
                RequestRoute route, EndKind kind, StrategyErrorType errorType, String detail) {
            return new EndEvidence(route, kind, null, false, 0L, errorType, detail, false, false);
        }

        public static EndEvidence publicationFailure(RequestRoute route, Response response) {
            var error = StrategyErrorType.DISPATCH_FAILED;
            if (response != null) {
                for (var candidate : StrategyErrorType.values()) {
                    if (candidate.getErrorCode() == response.getCode()) {
                        error = candidate;
                        break;
                    }
                }
            }
            return failure(
                    route,
                    EndKind.FAILURE,
                    error,
                    response == null
                            ? "route publication did not commit"
                            : response.getErrorMessage());
        }

        public static EndEvidence withdrawnFailure(String detail) {
            return new EndEvidence(
                    null,
                    EndKind.FAILURE,
                    null,
                    false,
                    0L,
                    StrategyErrorType.DISPATCH_FAILED,
                    detail,
                    true,
                    true);
        }

        public static EndEvidence externalFailure(
                RequestRoute route, Response response, Throwable error) {
            String detail =
                    error != null
                            ? "external future failure: " + error.getMessage()
                            : response != null && response.getErrorMessage() != null
                                    ? response.getErrorMessage()
                                    : "external future completion";
            return failure(route, EndKind.FRONTEND, StrategyErrorType.DISPATCH_FAILED, detail);
        }

        public static EndEvidence deliveryFailure(RequestRoute route, DeliveryEvidence evidence) {
            String detail;
            if (evidence instanceof PreparationFailure failed) {
                detail = failed.detail();
            } else {
                var cause = ((TransportResult) evidence).result().cause();
                String message = cause == null ? "unknown delivery failure" : cause.getMessage();
                detail =
                        "Delivery failed: "
                                + (message == null || message.isBlank()
                                        ? cause.getClass().getSimpleName()
                                        : message);
            }
            return failure(
                    route, EndKind.DELIVERY_FAILURE, StrategyErrorType.DISPATCH_FAILED, detail);
        }

        EndEvidence withoutRoute() {
            return new EndEvidence(
                    null,
                    kind,
                    role,
                    success,
                    errorCode,
                    errorType,
                    detail,
                    prefillSettled,
                    decodeSettled);
        }
    }

    public record EngineEvidence(
            RequestRoute route,
            DecodeResources.ReservationHandle reservation,
            RoleType role,
            EngineKind kind,
            long errorCode,
            String detail,
            long nowMs) {}

    public record Cancellation(
            long expectedBatchId,
            CancelReason reason,
            boolean frontend,
            long nowMs,
            StrategyErrorType errorType,
            String detail) {
        public static Cancellation frontend(long nowMs, String exhaustionCause) {
            return new Cancellation(
                    0L,
                    CancelReason.CLIENT_CANCELLED,
                    true,
                    nowMs,
                    exhaustionCause == null
                            ? StrategyErrorType.REQUEST_CANCELLED
                            : StrategyErrorType.RESOURCE_EXHAUSTED,
                    exhaustionCause == null
                            ? CancelReason.CLIENT_CANCELLED.getMessage()
                            : exhaustionCause);
        }
    }

    public record CancellationFacts(
            CancelReason reason, StrategyErrorType errorType, String detail) {}

    public record PreemptionReport(
            PreemptionCancelPhase phase,
            boolean ledgerCommitted,
            boolean decodeSettled,
            boolean releaseAttempt,
            EndEvidence terminal) {}

    public record CloseProgress(
            boolean notificationsAttempted,
            boolean prefillSettled,
            boolean decodeSettled,
            boolean notificationsDone,
            boolean timersDone) {}

    public record ResponseCandidate(RequestContext owner, ResponseKind kind, End result, long nowMs)
            implements GrantValue {}

    /** A delivery fact is already immutable; callers must not wrap it in an event object. */
    public interface DeliveryFact {}

    public sealed interface DeliveryEvidence extends DeliveryFact
            permits TransportResult,
                    Prediction,
                    RouteDelivered,
                    Abandonment,
                    CleanupAcknowledgement,
                    PreparationFailure {
        DeliveryClaim delivery();

        long nowMs();
    }

    public record TransportResult(DeliveryClaim delivery, DeliveryResult result, long nowMs)
            implements DeliveryEvidence {}

    public record Prediction(DeliveryClaim delivery, long predictedAtMs, long nowMs)
            implements DeliveryEvidence {}

    public record RouteDelivered(DeliveryClaim delivery, long predictedAtMs, long nowMs)
            implements DeliveryEvidence {}

    public record Abandonment(DeliveryClaim delivery, CancelReason reason, long nowMs)
            implements DeliveryEvidence {}

    public record CleanupAcknowledgement(
            DeliveryClaim delivery, CancelEvidenceKind kind, long nowMs)
            implements DeliveryEvidence {}

    public record PreparationFailure(
            DeliveryClaim delivery, RequestRoute route, String detail, long nowMs)
            implements DeliveryEvidence {}

    public record PreemptionFact(
            PreemptionRegistration attempt,
            PreemptionCancelPhase phase,
            boolean release,
            DecodeEndpoint endpoint,
            DecodeResources.ReservationHandle reservation,
            String detail,
            CleanupAcknowledgement cleanup)
            implements DeliveryFact {
        public PreemptionFact {
            Objects.requireNonNull(attempt, "attempt");
        }
    }

    public interface GrantValue {
        RequestContext owner();
    }

    public record PlacementToken(RequestContext owner, long sequence, RequestRoute withdrawing)
            implements GrantValue {}

    public record RoutePermit(
            RequestContext owner, PlacementToken operation, RequestRoute route, boolean publishable)
            implements GrantValue {}

    public record PlacementFacts(
            PlacementToken operation,
            RequestRoute acquired,
            boolean published,
            EndEvidence pendingTerminal,
            EngineEvidence pendingRetirement,
            boolean inactivityExpired) {}

    public record EngineFacts(
            boolean prefillObserved,
            long prefillCompletedAtMs,
            boolean decodeAccepted,
            long visibilityAtMs,
            boolean visibilityOverdue,
            long predictionSequence) {
        private static final long HANDOFF_GRACE_MS = 10_000L;

        EngineFacts predict(long sequence, long atMs) {
            if (predictionSequence == sequence) {
                throw new IllegalStateException("delivery prediction already consumed");
            }
            if (atMs < 0L) {
                throw new IllegalArgumentException("prediction timestamp must be non-negative");
            }
            return new EngineFacts(
                    prefillObserved,
                    prefillCompletedAtMs,
                    decodeAccepted,
                    visibility(decodeAccepted, prefillObserved, prefillCompletedAtMs, atMs, true),
                    false,
                    sequence);
        }

        EngineFacts observe(EngineEvidence evidence, boolean hasDecode) {
            boolean isDecode = evidence.role() == RoleType.DECODE;
            boolean accepted = decodeAccepted || isDecode && evidence.kind() != EngineKind.RETIRED;
            boolean observed = prefillObserved || !isDecode && evidence.kind() == EngineKind.ACTIVE;
            long completed =
                    !isDecode
                                    && evidence.kind() == EngineKind.COMPLETED
                                    && prefillCompletedAtMs == 0L
                            ? evidence.nowMs()
                            : prefillCompletedAtMs;
            long visibility =
                    visibility(
                            accepted,
                            observed,
                            completed,
                            visibilityAtMs,
                            hasDecode && predictionSequence != 0L);
            return accepted == decodeAccepted
                            && observed == prefillObserved
                            && completed == prefillCompletedAtMs
                            && visibility == visibilityAtMs
                    ? this
                    : new EngineFacts(
                            observed, completed, accepted, visibility, false, predictionSequence);
        }

        EngineFacts expireVisibility() {
            return new EngineFacts(
                    prefillObserved,
                    prefillCompletedAtMs,
                    decodeAccepted,
                    0L,
                    true,
                    predictionSequence);
        }

        private static long visibility(
                boolean accepted,
                boolean observed,
                long completed,
                long predicted,
                boolean handoff) {
            if (accepted || observed && completed == 0L) {
                return 0L;
            }
            return completed > 0L && handoff
                    ? saturatedAdd(completed, HANDOFF_GRACE_MS)
                    : predicted;
        }
    }

    public record DeliveryFacts(
            DeliveryClaim identity,
            SendState send,
            boolean acknowledged,
            CancelReason abandonment,
            boolean executionFinished,
            boolean prefillProof,
            boolean decodeProof,
            long startedAtMs)
            implements GrantValue {
        @Override
        public RequestContext owner() {
            return identity.owner();
        }

        DeliveryFacts withSendState(SendState next) {
            return recordResult(next, abandonment, prefillProof, decodeProof);
        }

        DeliveryFacts recordResult(
                SendState next, CancelReason abandoned, boolean prefillProof, boolean decodeProof) {
            return new DeliveryFacts(
                    identity,
                    next,
                    acknowledged,
                    abandoned,
                    executionFinished,
                    prefillProof,
                    decodeProof,
                    startedAtMs);
        }

        DeliveryFacts mergeReleaseProof(
                boolean finished, boolean prefillSettled, boolean decodeSettled) {
            return new DeliveryFacts(
                    identity,
                    send,
                    acknowledged,
                    abandonment,
                    executionFinished || finished,
                    prefillProof || prefillSettled,
                    decodeProof || decodeSettled,
                    startedAtMs);
        }

        DeliveryFacts abandon(SendState next, CancelReason reason) {
            return recordResult(next, reason, prefillProof, decodeProof);
        }

        DeliveryFacts acknowledge() {
            return new DeliveryFacts(
                    identity,
                    send,
                    true,
                    abandonment,
                    executionFinished,
                    prefillProof,
                    decodeProof,
                    startedAtMs);
        }

        DeliveryFacts recordEnd(CancelReason abandoned, boolean completed) {
            return new DeliveryFacts(
                    identity,
                    send,
                    acknowledged,
                    abandoned,
                    executionFinished || completed,
                    prefillProof,
                    decodeProof,
                    startedAtMs);
        }
    }

    public record PreemptionFacts(
            PreemptionRegistration identity,
            PreemptionCancelPhase phase,
            boolean cancelAcknowledged,
            boolean finished,
            EndEvidence pendingTerminal,
            boolean confirmationPending,
            boolean restorationObserved) {
        PreemptionFacts observeRestoration() {
            return new PreemptionFacts(
                    identity,
                    phase,
                    cancelAcknowledged,
                    finished,
                    pendingTerminal,
                    confirmationPending,
                    true);
        }

        PreemptionFacts recordReport(
                PreemptionCancelPhase next, EndEvidence terminal, boolean ledgerCommitted) {
            return new PreemptionFacts(
                    identity,
                    next,
                    cancelAcknowledged || next == PreemptionCancelPhase.CANCEL_REQUESTED,
                    finished || ledgerCommitted && terminal != null,
                    terminal,
                    confirmationPending,
                    restorationObserved);
        }

        PreemptionFacts awaitConfirmation() {
            return new PreemptionFacts(
                    identity,
                    phase,
                    cancelAcknowledged,
                    finished,
                    pendingTerminal,
                    true,
                    restorationObserved);
        }

        PreemptionFacts finish() {
            return new PreemptionFacts(
                    identity,
                    phase,
                    cancelAcknowledged,
                    true,
                    pendingTerminal,
                    confirmationPending,
                    restorationObserved);
        }

        PreemptionFacts retainTerminal(EndEvidence terminal) {
            return new PreemptionFacts(
                    identity,
                    phase,
                    cancelAcknowledged,
                    finished,
                    terminal,
                    confirmationPending,
                    restorationObserved);
        }
    }

    public record CloseFacts(
            boolean prefillSettled,
            boolean decodeSettled,
            boolean notificationsDone,
            boolean timersDone,
            boolean expired,
            boolean notificationsAttempted) {
        CloseFacts withSettlement(boolean prefillSettled, boolean decodeSettled) {
            return new CloseFacts(
                    prefillSettled,
                    decodeSettled,
                    notificationsDone,
                    timersDone,
                    expired,
                    notificationsAttempted);
        }

        CloseFacts mergeSettlement(boolean prefillSettled, boolean decodeSettled) {
            return withSettlement(
                    this.prefillSettled || prefillSettled, this.decodeSettled || decodeSettled);
        }

        CloseFacts markExpired() {
            return new CloseFacts(
                    prefillSettled,
                    decodeSettled,
                    notificationsDone,
                    timersDone,
                    true,
                    notificationsAttempted);
        }

        CloseFacts beginEnding(EndEvidence evidence, boolean prefillCompleted) {
            return new CloseFacts(
                    prefillSettled || evidence.prefillSettled() || prefillCompleted,
                    decodeSettled || evidence.decodeSettled(),
                    false,
                    false,
                    expired || evidence.kind() == EndKind.INACTIVITY,
                    notificationsAttempted);
        }

        CloseFacts recordProgress(CloseProgress progress, boolean preemptionFinished) {
            boolean notified =
                    notificationsDone || progress.notificationsDone() && preemptionFinished;
            return new CloseFacts(
                    prefillSettled || progress.prefillSettled(),
                    decodeSettled || progress.decodeSettled(),
                    notified,
                    timersDone || progress.timersDone(),
                    expired,
                    notificationsAttempted || progress.notificationsAttempted());
        }

        CloseFacts retainEnd(EndEvidence evidence) {
            return new CloseFacts(
                    prefillSettled || evidence.prefillSettled(),
                    decodeSettled || evidence.decodeSettled(),
                    notificationsDone,
                    timersDone,
                    expired || evidence.kind() == EndKind.INACTIVITY,
                    notificationsAttempted);
        }
    }

    /** Control facts are shared by observations that change only activity timestamps. */
    public record SnapshotFacts(
            Identity key,
            long createdAtMs,
            Phase phase,
            RequestRoute route,
            PlacementFacts placement,
            DeliveryFacts delivery,
            EngineFacts engine,
            PreemptionFacts preemption,
            CancellationFacts cancellation,
            End outcome,
            EndEvidence pendingEnd,
            ResponseCandidate selectedResponse,
            CloseFacts close,
            ExpirationTimer.Plan deadlines,
            RequestState terminalRecord,
            boolean deliveryReleasable,
            boolean preparedDelivery) {}

    public record Snapshot(
            @lombok.experimental.Delegate SnapshotFacts facts,
            long lastWorkerAtMs,
            long updatedAtMs,
            long inactivityExpiresAtMs) {}

    public record Update(UpdateKind kind, Snapshot before, Snapshot after) {
        public Update {
            Objects.requireNonNull(kind);
            Objects.requireNonNull(before);
            Objects.requireNonNull(after);
            if (before.key() != null && before.key() != after.key()) {
                throw new IllegalArgumentException("request incarnation changed");
            }
            if (kind != UpdateKind.APPLIED && before != after) {
                throw new IllegalArgumentException("unchanged updates must reuse their snapshot");
            }
        }
    }

    public sealed interface Claim<T extends GrantValue> permits Granted, Blocked, Rejected, Stale {
        ClaimKind kind();

        Snapshot before();

        Snapshot after();
    }

    public record Granted<T extends GrantValue>(Snapshot before, Snapshot after, T value)
            implements Claim<T> {
        public Granted {
            Objects.requireNonNull(before);
            Objects.requireNonNull(after);
            Objects.requireNonNull(value);
        }

        @Override
        public ClaimKind kind() {
            return ClaimKind.GRANTED;
        }
    }

    public record Blocked<T extends GrantValue>(Snapshot before, Snapshot after, Blocker blocker)
            implements Claim<T> {
        public Blocked {
            Objects.requireNonNull(blocker);
        }

        @Override
        public ClaimKind kind() {
            return ClaimKind.BLOCKED;
        }
    }

    public record Rejected<T extends GrantValue>(Snapshot before, Snapshot after, Rejection reason)
            implements Claim<T> {
        public Rejected {
            Objects.requireNonNull(reason);
        }

        @Override
        public ClaimKind kind() {
            return ClaimKind.REJECTED;
        }
    }

    public record Stale<T extends GrantValue>(Snapshot before) implements Claim<T> {
        @Override
        public ClaimKind kind() {
            return ClaimKind.STALE;
        }

        @Override
        public Snapshot after() {
            return before;
        }
    }
}
