package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;

import static com.google.common.math.LongMath.saturatedAdd;

/**
 * Immutable facts and exact permissions, without execution callbacks or infrastructure handles.
 */
public final class ContextResult {
    private ContextResult() {}

    public enum Phase {
        OPEN,
        ENDING,
        CLOSED
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

    public enum RouteFailureReason {
        ENDPOINT_UNAVAILABLE,
        PREPARATION_FAILED
    }

    /**
     * Failure of one exact route; cleanup must finish before scheduling another route.
     */
    public record RouteFailure(
            RouteExecution route,
            RouteFailureReason reason,
            org.flexlb.balance.endpoint.WorkerEndpoint failedEndpoint,
            String detail) {
        public RouteFailure(RouteExecution route, RouteFailureReason reason, String detail) {
            this(
                    route,
                    reason,
                    reason == RouteFailureReason.ENDPOINT_UNAVAILABLE
                            ? route.firstUnavailableEndpoint()
                            : null,
                    detail);
        }

        public RoleType failedRole() {
            return failedEndpoint == null ? null : failedEndpoint.getStatus().getRole();
        }

        public RouteFailure {
            java.util.Objects.requireNonNull(route, "route");
            java.util.Objects.requireNonNull(reason, "reason");
            detail = detail == null ? "Route admission failed" : detail;
        }
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
        ACK,
        TERMINAL,
        CANCELLATION
    }

    public enum CancelEvidenceKind {
        CLEANED,
        FENCED,
        ACKNOWLEDGED,
        NOT_FOUND,
        UNKNOWN
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

    /**
     * Observations about a request on a Worker; not mutually exclusive request states.
     */
    public enum WorkerRequestEventType {
        ACTIVE,
        ALLOCATED,
        COMPLETED,
        FAILED,
        PRIORITY_CANCELLED
    }

    public record Identity(
            String requestId,
            long originalNumericRequestId,
            org.flexlb.dao.route.RequestPhase phase,
            long incarnation) {}

    public record End(
            EndKind kind, RequestState.Phase phase, StrategyErrorType errorType, String detail) {
        public End {
            java.util.Objects.requireNonNull(phase, "phase");
            com.google.common.base.Preconditions.checkArgument(
                    phase.isTerminal(), "terminal phase required");
        }
    }

    public record EndEvidence(
            @lombok.With(lombok.AccessLevel.PACKAGE) RouteExecution route,
            EndKind kind,
            RoleType role,
            boolean success,
            long errorCode,
            StrategyErrorType errorType,
            String detail,
            boolean prefillSettled,
            boolean decodeSettled) {
        public static EndEvidence failure(
                RouteExecution route, EndKind kind, StrategyErrorType errorType, String detail) {
            return new EndEvidence(route, kind, null, false, 0L, errorType, detail, false, false);
        }

        public static EndEvidence publicationFailure(RouteExecution route, Response response) {
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
                RouteExecution route, Response response, Throwable error) {
            String detail =
                    error != null
                            ? "external future failure: " + error.getMessage()
                            : response != null && response.getErrorMessage() != null
                                    ? response.getErrorMessage()
                                    : "external future completion";
            return failure(route, EndKind.FRONTEND, StrategyErrorType.DISPATCH_FAILED, detail);
        }

        public static EndEvidence deliveryFailure(RouteExecution route, DeliveryEvidence evidence) {
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
    }

    public record WorkerRequestEvent(
            RouteExecution route,
            DecodeResources.RequestReservation reservation,
            RoleType role,
            WorkerRequestEventType type,
            long errorCode,
            String detail,
            long nowMs) {
        boolean terminal() {
            return type == WorkerRequestEventType.COMPLETED
                    || type == WorkerRequestEventType.FAILED
                    || type == WorkerRequestEventType.PRIORITY_CANCELLED;
        }

        EndEvidence end() {
            boolean decode = role == RoleType.DECODE;
            EndKind end =
                    type == WorkerRequestEventType.PRIORITY_CANCELLED
                            ? EndKind.PRIORITY
                            : EndKind.WORKER;
            StrategyErrorType error =
                    switch (end) {
                        case PRIORITY -> StrategyErrorType.PRIORITY_PREEMPTED;
                        default -> StrategyErrorType.WORKER_EXECUTION_FAILED;
                    };
            return new EndEvidence(
                    route,
                    end,
                    role,
                    type == WorkerRequestEventType.COMPLETED,
                    errorCode,
                    error,
                    detail,
                    !decode,
                    decode);
        }
    }

    /**
     * An exact Endpoint retirement retained while admission finishes or rolls back.
     */
    public record EndpointRetirement(
            RouteExecution route,
            DecodeResources.RequestReservation reservation,
            RoleType role,
            org.flexlb.balance.endpoint.WorkerEndpoint endpoint,
            String detail,
            long nowMs) {
        EndEvidence end() {
            return new EndEvidence(
                    route,
                    EndKind.RETIREMENT,
                    role,
                    false,
                    0L,
                    StrategyErrorType.DISPATCH_FAILED,
                    detail,
                    role != RoleType.DECODE,
                    role == RoleType.DECODE);
        }
    }

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

    /**
     * notificationsDone includes successful removal/update of deadline registrations.
     */
    public record CloseProgress(
            boolean notificationsAttempted,
            boolean prefillSettled,
            boolean decodeSettled,
            boolean notificationsDone) {}

    public record ResponseCandidate(RequestContext owner, ResponseKind kind, long nowMs) {}

    public sealed interface DeliveryEvidence
            permits TransportResult,
                    Prediction,
                    RouteDelivered,
                    Abandonment,
                    CleanupAcknowledgement,
                    PreparationFailure {
        Delivery.Member delivery();

        long nowMs();
    }

    public record TransportResult(Delivery.Member delivery, Delivery.Result result, long nowMs)
            implements DeliveryEvidence {}

    public record Prediction(Delivery.Member delivery, long predictedAtMs, long nowMs)
            implements DeliveryEvidence {}

    public record RouteDelivered(Delivery.Member delivery, long predictedAtMs, long nowMs)
            implements DeliveryEvidence {}

    public record Abandonment(Delivery.Member delivery, CancelReason reason, long nowMs)
            implements DeliveryEvidence {}

    public record CleanupAcknowledgement(
            Delivery.Member delivery, CancelEvidenceKind kind, long nowMs)
            implements DeliveryEvidence {}

    public record PreparationFailure(
            Delivery.Member delivery, RouteExecution route, String detail, long nowMs)
            implements DeliveryEvidence {}

    public record PlacementFacts(
            PlacementOperation operation,
            @lombok.With(lombok.AccessLevel.PACKAGE) boolean completing,
            @lombok.With(lombok.AccessLevel.PACKAGE) boolean published,
            @lombok.With(lombok.AccessLevel.PACKAGE) EndpointRetirement pendingRetirement,
            @lombok.With(lombok.AccessLevel.PACKAGE) boolean inactivityExpired) {}

    public record EngineFacts(
            boolean prefillObserved,
            long prefillCompletedAtMs,
            boolean decodeAccepted,
            long visibilityAtMs,
            boolean visibilityOverdue,
            Delivery.Member predictedDelivery) {
        private static final long HANDOFF_GRACE_MS = 10_000L;

        EngineFacts predict(Delivery.Member delivery, long atMs) {
            java.util.Objects.requireNonNull(delivery, "delivery");
            if (predictedDelivery == delivery) {
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
                    delivery);
        }

        EngineFacts observe(WorkerRequestEvent evidence, boolean hasDecode) {
            boolean isDecode = evidence.role() == RoleType.DECODE;
            boolean accepted = decodeAccepted || isDecode;
            boolean observed =
                    prefillObserved
                            || !isDecode && evidence.type() == WorkerRequestEventType.ACTIVE;
            long completed =
                    !isDecode
                                    && evidence.type() == WorkerRequestEventType.COMPLETED
                                    && prefillCompletedAtMs == 0L
                            ? evidence.nowMs()
                            : prefillCompletedAtMs;
            long visibility =
                    visibility(
                            accepted,
                            observed,
                            completed,
                            visibilityAtMs,
                            hasDecode && predictedDelivery != null);
            return accepted == decodeAccepted
                            && observed == prefillObserved
                            && completed == prefillCompletedAtMs
                            && visibility == visibilityAtMs
                    ? this
                    : new EngineFacts(
                            observed, completed, accepted, visibility, false, predictedDelivery);
        }

        EngineFacts expireVisibility() {
            return new EngineFacts(
                    prefillObserved,
                    prefillCompletedAtMs,
                    decodeAccepted,
                    0L,
                    true,
                    predictedDelivery);
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
            Delivery.Member identity,
            @lombok.With(lombok.AccessLevel.PACKAGE) SendState send,
            @lombok.With(lombok.AccessLevel.PACKAGE) boolean acknowledged,
            CancelReason abandonment,
            boolean executionFinished,
            boolean prefillProof,
            boolean decodeProof,
            long startedAtMs) {
        static boolean releasable(DeliveryFacts delivery) {
            if (delivery == null) {
                return true;
            }
            if (delivery.send() == SendState.NOT_STARTED || delivery.send() == SendState.SENDING) {
                return false;
            }
            if (delivery.identity().kind() == Delivery.SendMode.ROUTE_DECISION
                    || delivery.send() == SendState.NOT_SENT) {
                return true;
            }
            return delivery.abandonment() == null
                    ? delivery.executionFinished()
                    : delivery.prefillProof() && delivery.decodeProof();
        }

        public RequestContext owner() {
            return identity.owner();
        }

        DeliveryFacts record(DeliveryEvidence evidence, CancelReason reason) {
            return switch (evidence) {
                case TransportResult transport -> recordTransport(transport.result(), reason);
                case RouteDelivered ignored -> {
                    if (identity.kind() != Delivery.SendMode.ROUTE_DECISION) {
                        throw new IllegalArgumentException(
                                "route completion requires route delivery identity");
                    }
                    yield withSend(SendState.DELIVERED);
                }
                case Abandonment abandoned ->
                        abandon(
                                identity.kind() == Delivery.SendMode.ROUTE_DECISION
                                                && send == SendState.NOT_STARTED
                                        ? SendState.ROUTE_CLOSED
                                        : send,
                                abandonment == null && !releasable(this)
                                        ? abandoned.reason()
                                        : abandonment);
                case CleanupAcknowledgement ack ->
                        mergeReleaseProof(
                                false,
                                ack.kind() == CancelEvidenceKind.CLEANED
                                        || ack.kind() == CancelEvidenceKind.FENCED,
                                ack.kind() == CancelEvidenceKind.CLEANED);
                case PreparationFailure ignored -> {
                    if (send == SendState.SENDING) {
                        throw new IllegalStateException(
                                "preparation failure crossed send boundary");
                    }
                    yield send == SendState.NOT_STARTED ? withSend(SendState.NOT_SENT) : this;
                }
                case Prediction ignored -> this;
            };
        }

        /**
         * Exact sender result; cleanup evidence and cancellation remain separate facts.
         */
        DeliveryFacts recordTransport(Delivery.Result result, CancelReason reason) {
            if (identity.kind() != Delivery.SendMode.BATCH_ENQUEUE) {
                throw new IllegalArgumentException(
                        "transport result requires batch delivery identity");
            }
            var next =
                    switch (result.status()) {
                        case NOT_SENT -> SendState.NOT_SENT;
                        case PREFILL_REJECTED -> SendState.REJECTED;
                        case DELIVERED -> SendState.DELIVERED;
                        case UNCERTAIN -> SendState.UNKNOWN;
                    };
            if (send != SendState.NOT_STARTED && send != SendState.SENDING) {
                if (send == next) return this;
                throw new IllegalStateException("conflicting result after sender completion");
            }
            boolean released =
                    next == SendState.NOT_SENT
                            || (abandonment == null
                                    ? executionFinished
                                    : prefillProof && decodeProof);
            var abandoned =
                    result.failed() && !released && abandonment == null ? reason : abandonment;
            return recordResult(next, abandoned, prefillProof, decodeProof);
        }

        DeliveryFacts recordResult(
                SendState next, CancelReason abandoned, boolean prefillProof, boolean decodeProof) {
            if (send == next
                    && abandonment == abandoned
                    && this.prefillProof == prefillProof
                    && this.decodeProof == decodeProof) return this;
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
            if ((!finished || executionFinished)
                    && (!prefillSettled || prefillProof)
                    && (!decodeSettled || decodeProof)) return this;
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
            @lombok.With(lombok.AccessLevel.PACKAGE) boolean finished,
            @lombok.With(lombok.AccessLevel.PACKAGE) EndEvidence pendingTerminal,
            @lombok.With(lombok.AccessLevel.PACKAGE) boolean confirmationPending,
            @lombok.With(lombok.AccessLevel.PACKAGE) boolean restorationObserved) {

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
    }

    public record CloseFacts(
            boolean prefillSettled,
            boolean decodeSettled,
            boolean notificationsDone,
            @lombok.With(lombok.AccessLevel.PACKAGE) boolean expired,
            boolean notificationsAttempted) {
        CloseFacts withSettlement(boolean prefillSettled, boolean decodeSettled) {
            return new CloseFacts(
                    prefillSettled,
                    decodeSettled,
                    notificationsDone,
                    expired,
                    notificationsAttempted);
        }

        CloseFacts mergeSettlement(boolean prefillSettled, boolean decodeSettled) {
            return withSettlement(
                    this.prefillSettled || prefillSettled, this.decodeSettled || decodeSettled);
        }

        CloseFacts beginEnding(EndEvidence evidence, boolean prefillCompleted) {
            return new CloseFacts(
                    prefillSettled || evidence.prefillSettled() || prefillCompleted,
                    decodeSettled || evidence.decodeSettled(),
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
                    expired,
                    notificationsAttempted || progress.notificationsAttempted());
        }

        CloseFacts retainEnd(EndEvidence evidence) {
            return new CloseFacts(
                    prefillSettled || evidence.prefillSettled(),
                    decodeSettled || evidence.decodeSettled(),
                    notificationsDone,
                    expired || evidence.kind() == EndKind.INACTIVITY,
                    notificationsAttempted);
        }
    }

    /**
     * Immutable request facts and observation times captured in one transition.
     */
    public record Snapshot(
            Identity key,
            long createdAtMs,
            RouteExecution route,
            PlacementFacts placement,
            DeliveryFacts delivery,
            EngineFacts engine,
            PreemptionFacts preemption,
            CancellationFacts cancellation,
            End outcome,
            EndEvidence pendingEnd,
            RouteFailure routeFailure,
            ResponseCandidate selectedResponse,
            CloseFacts close,
            ExpirationTimer.Plan deadlines,
            RequestState terminalRecord,
            long lastWorkerAtMs,
            long updatedAtMs,
            long inactivityExpiresAtMs) {
        public boolean deliveryReleasable() {
            return DeliveryFacts.releasable(delivery);
        }

        /**
         * State eligibility only; Context still checks inactivity and preemption before confirming.
         */
        public boolean canConfirmDelivery() {
            return key != null
                    && phase() == Phase.OPEN
                    && selectedResponse == null
                    && pendingEnd == null
                    && cancellation == null
                    && delivery != null
                    && delivery.abandonment() == null
                    && (delivery.send() == SendState.DELIVERED
                            || delivery.send() == SendState.UNKNOWN && engine.decodeAccepted());
        }

        public boolean preparedDelivery() {
            if (outcome != null
                    || route == null
                    || delivery != null
                    || cancellation != null
                    || preemption != null
                    || routeFailure != null
                    || pendingEnd != null) {
                return false;
            }
            return placement == null
                    || placement.published()
                            && placement.operation().withdrawingRoute() == null
                            && placement.pendingRetirement() == null
                            && !placement.inactivityExpired();
        }

        public Phase phase() {
            return terminalRecord != null
                    ? Phase.CLOSED
                    : outcome != null ? Phase.ENDING : Phase.OPEN;
        }
    }
}
