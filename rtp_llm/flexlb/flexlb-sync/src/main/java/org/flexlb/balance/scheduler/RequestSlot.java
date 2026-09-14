package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.balance.preemption.VictimTerminal;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.ExpirationTimer.DecisionDeadline;
import org.flexlb.balance.scheduler.ExpirationTimer.InactivityDeadline;
import org.flexlb.balance.scheduler.ExpirationTimer.RequestDeadline;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.Logger;

import java.util.Objects;
import java.util.Optional;
import java.util.OptionalLong;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BiConsumer;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;

import static org.flexlb.balance.scheduler.RequestResponses.buildErrorResponse;
import static org.flexlb.balance.scheduler.RequestResponses.buildSuccessResponse;

/**
 * Canonical aggregate root for one exact request generation.
 *
 * <p>Event entry points acquire this monitor, select the request transition,
 * and execute detached cleanup/publication after unlocking. Internal state
 * reducers require the monitor; directory queries and timer installation use
 * the same monitor. Endpoint settlement transactions keep their existing lock
 * ordering. Transport and preemption callbacks carry exact capabilities.
 *
 * <p>Admission first cause, request terminal ownership, endpoint settlement,
 * and frontend publication are distinct decisions. A delivery ACK can select
 * a response before the request itself reaches its terminal state.
 */
public final class RequestSlot {

    private final ExpirationTimer expirationTimer;
    private final RequestTerminalCleanup terminalCleanup;
    private final Runnable admissionFinished;
    private static final Runnable NO_POST_LOCK_ACTION = () -> { };

    private final RequestCompletionPublisher completionPublisher;
    private final long requestId;
    private final long createdAtMs;
    private final RequestFuture future;
    private RequestState.Phase state = RequestState.Phase.QUEUED;
    private long updatedAtMs;
    private String detail = "queued";
    private DeliveryClaimKind deliveryClaimKind = DeliveryClaimKind.NONE;
    private long batchId;
    private long batchEnqueueStartedAtMs;
    /** Cleared exactly when the directory removes this request generation. */
    private boolean currentGeneration = true;

    private ScheduledRequest item;
    private StrategyErrorType deadlineErrorType =
            StrategyErrorType.BATCH_SLO_EXPIRED;

    /** Storage/cleanup ownership; distinct from the public request lifecycle. */
    private SlotPhase slotPhase = SlotPhase.ACTIVE;
    private EngineOwnership engineOwnership = EngineOwnership.DECODE_PENDING;
    private CancelReason cancellationReason;
    /** One frontend result may win before its unlocked future completion runs. */
    private PublicationKind publicationWinner;

    private boolean admissionOpen = true;
    private AdmissionMutation admissionMutation;
    private CancelReason pendingAdmissionCancelReason;
    private boolean pendingAdmissionInactivityExpired;
    private RequestDeadline requestDeadline;
    private DecisionDeadline decisionDeadline;
    private InactivityDeadline inactivityDeadline;
    private long inactivityTimeoutMs;
    private long lastWorkerStatusAtMs;
    private static final long DECODE_HANDOFF_GRACE_MS = 10_000L;
    private enum DecisionStage { UNDECIDED, WAITING_ENGINE, PREFILL_RUNNING, WAITING_DECODE, ACCEPTED }
    private DecisionStage decisionStage = DecisionStage.UNDECIDED;
    private boolean deliveryPredictionConsumed;
    private long prefillCompletedAtMs;
    private OptionalLong decisionExpiresAtMs = OptionalLong.empty();
    private boolean decisionExpired;

    private PreemptionRegistration preemption;

    RequestSlot(
            RequestCompletionPublisher completionPublisher,
            long requestId, ExpirationTimer expirationTimer,
            RequestTerminalCleanup terminalCleanup, Runnable admissionFinished) {
        this.expirationTimer = expirationTimer;
        this.terminalCleanup = terminalCleanup;
        this.admissionFinished = admissionFinished;
        this.completionPublisher = Objects.requireNonNull(
                completionPublisher, "completionPublisher");
        this.requestId = requestId;
        this.createdAtMs = System.currentTimeMillis();
        this.updatedAtMs = createdAtMs;
        this.lastWorkerStatusAtMs = createdAtMs;
        this.future = new RequestFuture(completionPublisher, this);
    }

    // Event entry points: each owns the complete per-request decision.

    public RequestState cancelRequest(long expectedBatchId,
                                       CancelReason reason) {
        Objects.requireNonNull(reason, "reason");
        TerminalAction localCompletion = null;
        RequestState result;
        synchronized (this) {
            if (!this.isCurrentGeneration()) {
                return null;
            }
            RequestState current = this.snapshot();
            if (!batchMatches(current, expectedBatchId)) {
                return null;
            }
            if (!this.ownsActiveGeneration() || current.state().isTerminal()) {
                return current;
            }
            if (this.hasCancellationFirstCause()
                    || (reason == CancelReason.DEADLINE_EXCEEDED
                        && current.deliveryClaimKind() != DeliveryClaimKind.NONE)) {
                return current;
            }
            String detail = cancelDetail(reason);
            if (this.deferCancellationDuringAdmission(reason, detail)) {
                return this.snapshot();
            }
            this.markCancellationRequested(reason, detail);
            CancelReason firstCause = this.requireCancellationFirstCause();
            ScheduledRequest item = this.activeItem();
            if (item == null || this.canClaimLocalTerminal()) {
                localCompletion = beginTerminalLocked(item != null, item != null,
                        TerminalOutcome.cancellation(firstCause, detail),
                        buildErrorResponse(this.cancellationErrorType(firstCause), detail));
            }
            result = this.snapshot();
        }
        terminalCleanup.submitTerminal(localCompletion);
        return result;
    }

    void expireInactiveRequest(long nowMs) {
        TerminalAction expiration;
        synchronized (this) {
            if (!this.isCurrentGeneration() || !this.ownsActiveGeneration()
                    || this.snapshot().state().isTerminal() || !this.requestInactive(nowMs)) {
                return;
            }
            String detail = "REQUEST_INACTIVE: no matching Engine request status before inactivity timeout";
            if (this.deferInactivityExpiryDuringAdmission(detail)) {
                return;
            }
            this.markCancellationRequested(CancelReason.DEADLINE_EXCEEDED, detail);
            expiration = beginExpiredRequestLocked(detail);
        }
        terminalCleanup.submitTerminal(expiration);
    }

    void observeEngineFact(Function<RequestSlot, EngineObservation> observation) {
        EngineObservation effects;
        Runnable work;
        synchronized (this) {
            if (!isCurrentGeneration()) { return; }
            effects = observation.apply(this);
            work = materializePostLockActionLocked(effects.transition(), null);
        }
        try { cancelDecisionDeadline(effects.obsoleteDeadline()); }
        finally {
            try { armDecisionDeadline(); }
            finally { runPostLock(work); }
        }
    }

    void onFailure(StrategyErrorType error, String detail) {
        Runnable work;
        synchronized (this) {
            work = reduceDeferredTerminalFactLocked(DeferredTerminal.failure(error, detail));
        }
        runPostLock(work);
    }

    void onPrefillRetired(PrefillEndpoint source, ScheduledRequest exact) {
        String detail = "Prefill endpoint generation retired: " + source.ipPort()
                + "#" + source.getStatus().getGenerationId();
        TerminalAction action;
        synchronized (this) {
            action = beginPrefillRetirementTerminal(source, exact,
                    TerminalOutcome.fail(detail), buildErrorResponse(StrategyErrorType.DISPATCH_FAILED, detail));
        }
        terminalCleanup.submitTerminal(action);
    }

    void onDecodeRetired(DecodeEndpoint source, DecodeEndpoint.ReservationHandle exact) {
        Runnable work;
        synchronized (this) {
            work = materializePostLockActionLocked(reduceDecodeGenerationRetired(source, exact,
                            "Decode endpoint generation retired: generation=" + exact.endpointGenerationId()), null);
        }
        runPostLock(work);
    }

    void onAdmissionCompleted(
            AdmissionMutation exact) {
        try {
            RequestSlot.AdmissionMutationCompletion completion;
            synchronized (this) {
                completion = this.completeAdmissionMutation(exact);
            }
            if (!completion.owned()) {
                return;
            }
            if (completion.pendingTerminal() != null) {
                Runnable work;
                synchronized (this) {
                    work = reduceDeferredTerminalFactLocked(completion.pendingTerminal());
                }
                runPostLock(work);
            } else if (completion.pendingRetirement() != null) {
                terminalCleanup.submitTerminal(completion.pendingRetirement());
            }
            // A retained delivery failure may have become stale after Engine acceptance.
            // Resume the already-selected cancellation even when that replay is a no-op.
            if (completion.cancellationToResume() != null) {
                resumeCancellationAfterAdmission(completion.cancellationToResume(), completion.inactivityExpired());
            }
        } finally {
            try {
                expirationTimer.attachInactivityDeadline(this);
            } finally {
                admissionFinished.run();
            }
        }
    }

    void onAdmissionFailed(
            AdmissionMutation exact,
            Response failure) {
        try {
            if (failure.isSuccess()) {
                throw new IllegalArgumentException(
                        "admission termination requires a failure response");
            }
            RequestSlot.AdmissionMutationCompletion completion;
            Runnable retainedWork = null;
            TerminalAction action = null;
            synchronized (this) {
                completion = this.claimAdmissionMutationTermination(exact);
                if (completion.pendingTerminal() != null) {
                    retainedWork = reduceDeferredTerminalFactLocked(completion.pendingTerminal());
                } else if (completion.pendingRetirement() == null) {
                    CancelReason pendingCancel =
                            completion.cancellationToResume();
                    Response terminalResponse = failure;
                    TerminalOutcome transition;
                    if (pendingCancel == null) {
                        String detail = failure.getErrorMessage() == null
                                ? "eviction admission failed"
                                : failure.getErrorMessage();
                        transition = TerminalOutcome.fail(detail);
                    } else {
                        String detail = cancelDetail(pendingCancel);
                        terminalResponse = buildErrorResponse(
                                this.cancellationErrorType(pendingCancel), detail);
                        transition = TerminalOutcome.cancellation(pendingCancel, detail);
                    }
                    action = beginTerminalLocked(false, false, transition, terminalResponse);
                    if (action == null) {
                        throw new IllegalStateException(
                                "failed to claim admission terminal for request "
                                        + this.requestId());
                    }
                }
            }
            if (retainedWork != null) {
                runPostLock(retainedWork);
            } else if (completion.pendingRetirement() != null) {
                terminalCleanup.submitTerminal(completion.pendingRetirement());
            } else {
                terminalCleanup.submitTerminal(action);
            }
            if (completion.cancellationToResume() != null) {
                resumeCancellationAfterAdmission(completion.cancellationToResume(), completion.inactivityExpired());
            }
        } finally {
            try {
                expirationTimer.attachInactivityDeadline(this);
            } finally {
                admissionFinished.run();
            }
        }
    }

    PlacementResult.Status commitRoute(ScheduledRequest exact, BooleanSupplier publication) {
        synchronized (this) {
            if (!tryBindItemForPublication(exact)) { return PlacementResult.Status.CLOSED; }
        }
        // Endpoint queue publication must not retain this monitor. Admission pins the binding.
        try {
            if (publication.getAsBoolean()) { return PlacementResult.Status.SUCCESS; }
        } catch (RuntimeException | Error failure) {
            try { synchronized (this) { rollbackItemPublication(exact); } }
            catch (RuntimeException | Error rollbackFailure) {
                if (rollbackFailure != failure) { failure.addSuppressed(rollbackFailure); }
            }
            throw failure;
        }
        synchronized (this) { rollbackItemPublication(exact); }
        return PlacementResult.Status.BLOCKED;
    }

    public <T> Optional<T> prepareIfOwned(ScheduledRequest exact, Supplier<T> preparation) {
        synchronized (this) {
            return ownsPreparedDelivery(exact) ? Optional.of(preparation.get()) : Optional.empty();
        }
    }

    DeliveryClaim claimDelivery(ScheduledRequest exact, DeliveryClaimKind kind,
                                long correlationId, BooleanSupplier handoff) {
        synchronized (this) {
            if (!ownsPreparedDelivery(exact)) { return null; }
            DeliveryClaim claim = new DeliveryClaim(this, exact, kind, correlationId);
            if (!handoff.getAsBoolean()) {
                throw new IllegalStateException("endpoint ownership lost for request " + requestId);
            }
            switch (kind) {
                case BATCH_ENQUEUE -> { startBatchEnqueue(correlationId); markBatchEnqueueStarted(); }
                case ROUTE_DECISION -> startRouteDecisionDelivery();
                case NONE -> throw new IllegalArgumentException("delivery claim kind cannot be NONE");
            }
            return claim;
        }
    }

    void onDeliveryStarted(DeliveryClaim claim, WorkSnapshot work, long unstartedMs) {
        Objects.requireNonNull(work, "precedingWork");
        if (unstartedMs < 0L) { throw new IllegalArgumentException("unstarted work must be non-negative"); }
        ScheduledRequest exact = claim.item;
        if (exact.decodeEp() != null && exact.decodeEp().isReservationAccepted(exact.decodeReservation())) {
            observeEngineFact(slot -> slot.observeDecodeFact(exact.decodeEp(),
                    DecodeEndpoint.WorkerStatusFact.accepted(exact.decodeReservation()), System.currentTimeMillis()));
        }
        synchronized (this) {
            if (!ownsDeliveryClaim(exact, claim.kind, claim.correlationId)) { return; }
            startDecisionTracking(work, unstartedMs, System.currentTimeMillis());
        }
        armDecisionDeadline();
    }

    void onDeliveryResult(
            DeliveryClaim claim,
            DeliveryResult completion) {
        DeliveryClaim exact = claim != null && claim.slot == this ? claim : null;
        if (exact == null) {
            throw new IllegalArgumentException(
                    "delivery claim was not created by this scheduler");
        }
        Runnable work = null;
        synchronized (exact.slot) {
            // WorkerStatus may settle this generation before the RPC callback arrives.
            // Its terminal proof wins; an old transport outcome cannot reopen ownership.
            if (!ownsDeliveryClaim(exact.item, exact.kind, exact.correlationId)) {
                return;
            }
            if (exact.completed) {
                throw new IllegalStateException(
                        "delivery claim is already completed: request_id="
                                + exact.item.requestId());
            }
            exact.completed = true;
            if (completion.status()
                    == DeliveryResult.Status.DELIVERED) {
                work = switch (exact.kind) {
                    case BATCH_ENQUEUE -> confirmBatchEnqueueLocked(exact.item);
                    case ROUTE_DECISION -> confirmRouteDecisionLocked(exact.item);
                    case NONE -> throw new IllegalStateException(
                            "delivery claim kind cannot be NONE");
                };
            } else if (completion.status()
                    == DeliveryResult.Status.FAILED) {
                String detail = "Delivery failed: "
                        + detailOf(completion.cause());
                if (exact.slot.decodeOwnsRequest()) {
                    work = materializePostLockActionLocked(exact.slot.reduceDeliveryConfirmed(
                                    exact.correlationId),
                            null);
                } else {
                    DecodeEndpoint decode = exact.item.decodeEp();
                    DecodeEndpoint.ReservationHandle reservation =
                            exact.item.decodeReservation();
                    DecodeEndpoint.DispatchRejectionSettlement settlement =
                            decode == null || reservation == null
                                    ? DecodeEndpoint.DispatchRejectionSettlement.RELEASED
                                    : decode.settleDefiniteDispatchRejection(
                                            reservation);
                    switch (settlement) {
                        case RELEASED -> work = reduceDeferredTerminalFactLocked(DeferredTerminal.deliveryRejected(detail));
                        case ENGINE_ACCEPTED -> work = materializePostLockActionLocked(exact.slot.reduceDeliveryConfirmed(
                                        exact.correlationId),
                                null);
                        case CONFLICT -> exact.slot.markAwaitingConfirmation(detail);
                        case STALE -> work = null;
                    }
                }
            } else if (exact.slot.decodeOwnsRequest()) {
                work = materializePostLockActionLocked(exact.slot.reduceDeliveryConfirmed(
                                exact.correlationId),
                        null);
            } else {
                String detail = switch (completion.status()) {
                    case TIMED_OUT ->
                            "Delivery timed out: "
                                    + detailOf(completion.cause());
                    case UNCERTAIN ->
                            "Delivery outcome uncertain: "
                                    + detailOf(completion.cause());
                    case DELIVERED ->
                            throw new IllegalStateException(
                                    "delivered outcome was already handled");
                    case FAILED ->
                            throw new IllegalStateException(
                                    "failed outcome was already handled");
                };
                exact.slot.markAwaitingConfirmation(detail);
            }
        }
        runPostLock(work);
    }

    void onPreparationFailed(ScheduledRequest exact, Throwable cause) {
        Runnable work;
        try {
            synchronized (this) {
                if (!ownsPreparedDelivery(exact)) { return; }
                work = reduceDeferredTerminalFactLocked(DeferredTerminal.deliveryFailure(
                        StrategyErrorType.DISPATCH_FAILED, "Delivery preparation failed: " + detailOf(cause)));
            }
            runPostLock(work);
        } catch (Throwable failure) {
            if (cause != null && cause != failure) { failure.addSuppressed(cause); }
            Logger.error("Prepared delivery failure reduction failed request_id={}", requestId, failure);
        }
    }

    boolean onPreemptionPhase(PreemptionRegistration claim, PreemptionCancelPhase next) {
        return next != null && onPreemption(claim, false, slot -> slot.applyPreemptionPhase(claim, next));
    }

    boolean onPreemptionReleased(PreemptionRegistration claim) {
        return onPreemption(claim, false, slot -> slot.applyPreemptionRelease(claim));
    }

    boolean onPreemptionTerminal(PreemptionRegistration claim, String detail) {
        return onPreemption(claim, true, slot -> slot.applyPreemptionTombstone(claim, detail));
    }

    void onDecisionExpired(DecisionExpiry expiry) {
        synchronized (this) {
            if (expiry.needsConfirmation() && expiry.item() != null && ownsActiveItem(expiry.item())) {
                markAwaitingConfirmation("decision lifetime expired; awaiting Engine confirmation");
            }
        }
    }

    TerminalAction onShutdown() {
        synchronized (this) {
            if (!isCurrentGeneration() || !canClaimLocalTerminal()) { return null; }
            ScheduledRequest active = activeItem();
            String message = "request scheduler is shutting down";
            return beginTerminalLocked(active != null, active != null,
                    TerminalOutcome.fail(message), buildErrorResponse(StrategyErrorType.DISPATCH_FAILED, message));
        }
    }

    boolean publishDecisionResponse(Response response) {
        PublicationPermit permit = publishExternalResponse(response);
        if (permit == null) { return false; }
        try { completionPublisher.submitTerminalResponse(permit, response); return true; }
        catch (RuntimeException | Error failure) { permit.abortClaimedPublication(); throw failure; }
    }

    RequestSlot.PublicationPermit publishExternalResponse(Response response) {
        String detail = response != null && response.getErrorMessage() != null
                ? response.getErrorMessage() : "external future completion";
        TerminalOutcome transition =
                response != null && !response.isSuccess()
                        ? TerminalOutcome.fail(detail)
                        : TerminalOutcome.complete(detail);
        TerminalAction action = claimExternalLocalTerminal(transition);
        return action == null ? null : terminalCleanup.finishTerminal(action);
    }

    RequestSlot.PublicationPermit publishExternalFailure(Throwable error) {
        Objects.requireNonNull(error, "error");
        String detail = "external future failure"
                + (error.getMessage() == null ? "" : ": " + error.getMessage());
        TerminalAction action = claimExternalLocalTerminal(TerminalOutcome.fail(detail));
        return action == null ? null : terminalCleanup.finishTerminal(action);
    }

    RequestSlot.PublicationPermit publishExternalCancellation() {
        String detail = cancelDetail(CancelReason.CLIENT_CANCELLED);
        TerminalAction action = claimExternalLocalTerminal(TerminalOutcome.cancel(detail));
        return action == null ? null : terminalCleanup.finishTerminal(action);
    }

    // State queries and transitions. Callers hold this monitor unless synchronized.

    long requestId() {
        return requestId;
    }

    RequestFuture future() {
        return future;
    }

    boolean ownsFuture(CompletableFuture<?> expected) {
        return future == expected;
    }

    long createdAtMs() {
        return createdAtMs;
    }

    void startBatchEnqueue(long assignedBatchId) {
        requireSlotLock("batch delivery claim");
        if (assignedBatchId <= 0L) {
            throw new IllegalArgumentException("batchId must be positive");
        }
        requireCompatibleDelivery(
                DeliveryClaimKind.BATCH_ENQUEUE, assignedBatchId);
        ensureTransitionAllowed(RequestState.Phase.DISPATCHING);
        if (deliveryClaimKind == DeliveryClaimKind.NONE) {
            deliveryClaimKind = DeliveryClaimKind.BATCH_ENQUEUE;
            batchId = assignedBatchId;
        }
        transition(RequestState.Phase.DISPATCHING,
                "batch enqueue started");
    }

    void startRouteDecisionDelivery() {
        requireSlotLock("route delivery claim");
        requireCompatibleDelivery(DeliveryClaimKind.ROUTE_DECISION, 0L);
        ensureTransitionAllowed(RequestState.Phase.DISPATCHING);
        if (deliveryClaimKind == DeliveryClaimKind.NONE) {
            deliveryClaimKind = DeliveryClaimKind.ROUTE_DECISION;
        }
        transition(RequestState.Phase.DISPATCHING,
                "route decision delivery started");
    }

    void markBatchEnqueueStarted() {
        requireSlotLock("batch enqueue timestamp");
        if (deliveryClaimKind != DeliveryClaimKind.BATCH_ENQUEUE) {
            throw new IllegalStateException(
                    "batch enqueue timestamp requires a batch delivery claim");
        }
        if (batchEnqueueStartedAtMs == 0L) {
            batchEnqueueStartedAtMs = System.currentTimeMillis();
            assertInvariant();
        }
    }

    synchronized long getBatchEnqueueStartedAtMs() {
        return batchEnqueueStartedAtMs;
    }

    RequestState markDeliveryConfirmed() {
        requireSlotLock("delivery confirmation");
        if (state.isTerminal()
                || state == RequestState.Phase.CANCEL_REQUESTED) {
            return snapshot();
        }
        String confirmationDetail = switch (deliveryClaimKind) {
            case BATCH_ENQUEUE -> "batch enqueue acknowledged";
            case ROUTE_DECISION -> "route decision delivered";
            case NONE -> throw new IllegalStateException(
                    "cannot confirm delivery without a delivery claim");
        };
        return transition(RequestState.Phase.ACKNOWLEDGED,
                confirmationDetail);
    }

    RequestState timeout(String message) {
        requireSlotLock("request timeout");
        return state.isTerminal()
                ? snapshot()
                : transition(RequestState.Phase.TIMED_OUT, message);
    }

    RequestState fail(String message) {
        requireSlotLock("request failure");
        return state.isTerminal()
                ? snapshot()
                : transition(RequestState.Phase.FAILED, message);
    }

    RequestState complete(String message) {
        requireSlotLock("request completion");
        return state.isTerminal()
                ? snapshot()
                : transition(RequestState.Phase.COMPLETED, message);
    }

    RequestState requestCancel(String message) {
        requireSlotLock("request cancellation");
        return state.isTerminal()
                ? snapshot()
                : transition(RequestState.Phase.CANCEL_REQUESTED, message);
    }

    RequestState cancel(String message) {
        requireSlotLock("request cancellation completion");
        if (state.isTerminal()) {
            return snapshot();
        }
        if (state != RequestState.Phase.CANCEL_REQUESTED) {
            transition(RequestState.Phase.CANCEL_REQUESTED, message);
        }
        return transition(RequestState.Phase.CANCELLED, message);
    }

    synchronized RequestState snapshot() {
        return new RequestState(
                requestId, state, deliveryClaimKind, batchId,
                createdAtMs, updatedAtMs, detail);
    }

    private void requireCompatibleDelivery(
            DeliveryClaimKind requestedKind,
            long requestedBatchId) {
        if (deliveryClaimKind == DeliveryClaimKind.NONE) {
            return;
        }
        if (deliveryClaimKind != requestedKind) {
            throw new IllegalStateException(
                    "request already has a " + deliveryClaimKind
                            + " delivery claim");
        }
        if (deliveryClaimKind == DeliveryClaimKind.BATCH_ENQUEUE
                && batchId != requestedBatchId) {
            throw new IllegalStateException(
                    "request already belongs to batch " + batchId);
        }
    }

    private void ensureTransitionAllowed(RequestState.Phase next) {
        if (!state.canTransitionTo(next)) {
            throw new IllegalStateException(
                    "invalid request lifecycle transition "
                            + state + " -> " + next);
        }
    }

    private RequestState transition(
            RequestState.Phase next,
            String message) {
        if (state == next) {
            return snapshot();
        }
        ensureTransitionAllowed(next);
        state = next;
        detail = message == null ? "" : message;
        updatedAtMs = System.currentTimeMillis();
        assertInvariant();
        return snapshot();
    }

    record EngineObservation(PreemptionReduction transition, DecisionDeadline obsoleteDeadline) {
        static final EngineObservation STALE = new EngineObservation(PreemptionReduction.STALE, null);
    }

    EngineObservation observePrefillFact(PrefillEndpoint source, RoleType role,
                                         PrefillState.WorkerStatusFact fact, long nowMs) {
        requireSlotLock("Prefill fact reduction");
        if (!ownsPrefillFact(source, fact.item())) { return EngineObservation.STALE; }
        lastWorkerStatusAtMs = Math.max(lastWorkerStatusAtMs, nowMs);
        PreemptionReduction transition = switch (fact.kind()) {
            case ACTIVE -> {
                observeDecisionPrefillActive();
                yield reducePrefillActive(source, fact.item());
            }
            case COMPLETED -> {
                observeDecisionPrefillCompleted(nowMs, role != RoleType.PDFUSION && item.decodeEp() != null);
                yield role == RoleType.PDFUSION
                        ? reduceWorkerTerminal(fact.item(), DeferredTerminal.worker(
                                WorkerTerminalSource.PREFILL_BACKED, true, fact.errorCode()))
                        : PreemptionReduction.NONE;
            }
            case FAILED -> reduceWorkerTerminal(fact.item(), DeferredTerminal.worker(
                    WorkerTerminalSource.PREFILL_BACKED, false, fact.errorCode()));
            case PRIORITY_CANCELED -> reducePriorityCanceled(source, fact.item());
        };
        reconcileDecisionEvidence();
        return new EngineObservation(transition, detachObsoleteDecisionDeadline());
    }

    EngineObservation observeDecodeFact(DecodeEndpoint source, DecodeEndpoint.WorkerStatusFact fact, long nowMs) {
        requireSlotLock("Decode fact reduction");
        if (!ownsDecodeFact(source, fact.reservation())) { return EngineObservation.STALE; }
        lastWorkerStatusAtMs = Math.max(lastWorkerStatusAtMs, nowMs);
        if (fact.kind() == DecodeEndpoint.WorkerStatusFact.Kind.TERMINAL) {
            advanceDecision(DecisionStage.ACCEPTED, OptionalLong.empty());
            markDecodeTerminalOwned();
            return new EngineObservation(reduceWorkerTerminal(item, DeferredTerminal.worker(
                    WorkerTerminalSource.DECODE_ENDPOINT_SETTLED, fact.errorCode() == 0L, fact.errorCode())),
                    detachObsoleteDecisionDeadline());
        }
        // Both a repeated ACTIVE observation and first ACCEPTED prove Decode ownership.
        DecodeAcceptance acceptance = markDecodeAccepted();
        return new EngineObservation(PreemptionReduction.NONE,
                acceptance.detachedDecisionDeadline());
    }

    StrategyErrorType timeoutErrorType() {
        requireSlotLock("deadline error lookup");
        return deadlineErrorType;
    }

    StrategyErrorType cancellationErrorType(CancelReason reason) {
        requireSlotLock("cancellation error lookup");
        return reason == CancelReason.DEADLINE_EXCEEDED
                ? deadlineErrorType : StrategyErrorType.REQUEST_CANCELLED;
    }

    void configureDeadlineError(StrategyErrorType errorType) {
        requireSlotLock("deadline error configuration");
        if (item != null || !admissionOpen || slotPhase != SlotPhase.ACTIVE) {
            throw new IllegalStateException(
                    "deadline error must be configured before admission");
        }
        deadlineErrorType = errorType;
        assertInvariant();
    }

    /**
     * Reserve this exact generation for queue publication.
     *
     * <p>The admission mutation is the logical pin that lets the endpoint
     * queue publish without retaining this monitor. Binding the canonical
     * {@code item} is itself the readiness proof: before binding there is no
     * exact item to deliver; after binding every queue-visible identity is
     * immediately claimable.
     */
    boolean tryBindItemForPublication(ScheduledRequest candidate) {
        requireSlotLock("request item publication begin");
        if (!ownsActiveGeneration()
                || !isOpen()
                || item != null
                || admissionMutation == null
                || candidate.requestId() != requestId
                || candidate.future() != future) {
            return false;
        }
        item = candidate;
        assertInvariant();
        return true;
    }

    /** Roll back only the exact binding whose queue publication did not commit. */
    void rollbackItemPublication(ScheduledRequest exact) {
        requireSlotLock("request item publication rollback");
        if (item != exact || admissionMutation == null) {
            throw new IllegalStateException(
                    "request item publication ownership changed for "
                            + requestId);
        }
        item = null;
        assertInvariant();
    }

    /** Exact ACTIVE item, or null when this generation no longer owns one. */
    ScheduledRequest activeItem() {
        requireSlotLock("active item lookup");
        return ownsActiveGeneration() ? item : null;
    }

    boolean ownsActiveGeneration() {
        requireSlotLock("active generation lookup");
        return isCurrentGeneration()
                && slotPhase == SlotPhase.ACTIVE
                && !state.isTerminal();
    }

    boolean ownsActiveItem(ScheduledRequest expected) {
        requireSlotLock("active item ownership lookup");
        return ownsActiveGeneration() && item == expected;
    }

    boolean ownsPrefillFact(PrefillEndpoint source, ScheduledRequest expected) {
        requireSlotLock("Prefill fact ownership lookup");
        return ownsActiveItem(expected) && expected.prefillEp() == source;
    }

    boolean ownsDecodeFact(
            DecodeEndpoint source,
            DecodeEndpoint.ReservationHandle reservation) {
        requireSlotLock("Decode fact ownership lookup");
        return ownsActiveGeneration()
                && item != null
                && item.decodeEp() == source
                && reservation.equals(item.decodeReservation());
    }

    ScheduledRequest activeItemForReservation(long reservationToken) {
        requireSlotLock("reservation item lookup");
        DecodeEndpoint.ReservationHandle reservation =
                item == null ? null : item.decodeReservation();
        return ownsActiveGeneration()
                && reservation != null
                && reservation.reservationToken() == reservationToken
                ? item : null;
    }

    boolean ownsDeliveryClaim(
            ScheduledRequest expected,
            DeliveryClaimKind kind,
            long expectedBatchId) {
        requireSlotLock("delivery claim lookup");
        return ownsActiveItem(expected)
                && deliveryClaimKind == kind
                && batchId == expectedBatchId
                && !state.isTerminal();
    }

    /**
     * Commit the unique delivery-confirmation edge and move every capability
     * needed by its unlocked publication out of the slot.
     */
    DeliveryConfirmation confirmDeliveryForPublication(
            ScheduledRequest expected,
            DeliveryClaimKind expectedKind,
            long expectedBatchId) {
        requireSlotLock("delivery confirmation");
        if (state != RequestState.Phase.DISPATCHING
                || !ownsDeliveryClaim(
                        expected, expectedKind, expectedBatchId)) {
            return null;
        }

        PublicationPermit permit = requirePublicationPermit(
                PublicationKind.DELIVERY);
        boolean transferred = false;
        try {
            long enqueueStartedAtMs = getBatchEnqueueStartedAtMs();
            markDeliveryConfirmed();
            if (state != RequestState.Phase.ACKNOWLEDGED) {
                throw new IllegalStateException(
                        "delivery confirmation did not acknowledge request "
                                + requestId);
            }

            RequestDeadline detachedRequestDeadline = requestDeadline;
            requestDeadline = null;
            DeliveryConfirmation result = new DeliveryConfirmation(
                    permit,
                    detachedRequestDeadline,
                    enqueueStartedAtMs);
            transferred = true;
            assertInvariant();
            return result;
        } finally {
            if (!transferred) {
                permit.abandonIfUnclaimed();
            }
        }
    }

    boolean decodeOwnsRequest() {
        requireSlotLock("Decode ownership lookup");
        return engineOwnership == EngineOwnership.DECODE_OWNED;
    }

    boolean canClaimLocalTerminal() {
        requireSlotLock("local terminal eligibility");
        return ownsActiveGeneration()
                && !future.isDone()
                && admissionMutation == null
                && preemption == null
                && engineOwnership == EngineOwnership.DECODE_PENDING
                && state != RequestState.Phase.ACKNOWLEDGED
                && !deliveryClaimKind.isClaimed();
    }

    /**
     * Whether the exact published queue item may prepare or commit delivery.
     * Binding the canonical item makes it delivery-ready before endpoint
     * publication, while the admission mutation defers cancellation and
     * terminal cleanup. Therefore every queue-visible identity is immediately
     * claimable without nesting the slot monitor and endpoint queue lock.
     */
    boolean canClaimDelivery() {
        requireSlotLock("delivery eligibility");
        return ownsActiveGeneration()
                && !future.isDone()
                && preemption == null
                && state != RequestState.Phase.ACKNOWLEDGED
                && !deliveryClaimKind.isClaimed();
    }

    boolean isOpen() {
        requireSlotLock("admission state lookup");
        return slotPhase == SlotPhase.ACTIVE
                && admissionOpen
                && !future.isDone()
                && !state.isTerminal();
    }

    boolean isLiveGeneration() {
        requireSlotLock("live generation lookup");
        return isCurrentGeneration() && slotPhase != SlotPhase.TOMBSTONE;
    }

    boolean isTombstone() {
        requireSlotLock("tombstone lookup");
        return slotPhase == SlotPhase.TOMBSTONE;
    }

    boolean isRemovableTombstone(long updatedBeforeMs) {
        requireSlotLock("tombstone retention lookup");
        return isCurrentGeneration()
                && slotPhase == SlotPhase.TOMBSTONE
                && state.isTerminal()
                && updatedAtMs < updatedBeforeMs
                && item == null;
    }

    // ==================== Admission mutation ====================

    AdmissionMutation tryBeginAdmissionMutation(
            BiConsumer<AdmissionMutation, Response> termination,
            Consumer<AdmissionMutation> completion) {
        requireSlotLock("admission mutation claim");
        if (!ownsActiveGeneration()
                || !isOpen()
                || item != null
                || admissionMutation != null
                || preemption != null) {
            return null;
        }
        AdmissionMutation exact = new AdmissionMutation(
                termination, completion);
        admissionMutation = exact;
        assertInvariant();
        return exact;
    }

    AdmissionMutationCompletion completeAdmissionMutation(
            AdmissionMutation exact) {
        requireSlotLock("admission mutation completion");
        if (admissionMutation == null || admissionMutation != exact) {
            return AdmissionMutationCompletion.NOT_OWNED;
        }
        admissionMutation = null;
        return finishAdmissionMutation();
    }

    AdmissionMutationCompletion claimAdmissionMutationTermination(
            AdmissionMutation exact) {
        requireSlotLock("admission mutation terminal claim");
        if (!ownsActiveGeneration()
                || admissionMutation == null
                || admissionMutation != exact) {
            throw new IllegalStateException(
                    "admission mutation no longer owns request " + requestId);
        }
        admissionMutation = null;
        return finishAdmissionMutation();
    }

    /** Resolve retained facts once for both ordinary close and terminal close. */
    private AdmissionMutationCompletion finishAdmissionMutation() {
        CancelReason cancellationToResume = pendingAdmissionCancelReason;
        pendingAdmissionCancelReason = null;
        boolean inactivityExpired = pendingAdmissionInactivityExpired;
        pendingAdmissionInactivityExpired = false;
        // A retained worker terminal may settle cancellation, but must
        // not erase the first cause already chosen during admission.
        cancellationToResume = promoteAdmissionCancellation(cancellationToResume);
        DeferredTerminal pendingTerminal = admissionPendingTerminal != null
                && admissionPendingTerminal.authoritativeWorker()
                        ? admissionPendingTerminal : null;
        TerminalAction pendingRetirement = pendingTerminal == null
                        ? beginPendingPrefillRetirement(
                                admissionPendingPrefillRetirement)
                        : null;
        if (pendingTerminal == null
                && pendingRetirement == null) {
            pendingTerminal = admissionPendingTerminal;
        }
        admissionPendingTerminal = null;
        admissionPendingPrefillRetirement = null;
        if (pendingRetirement != null || pendingTerminal != null && pendingTerminal.authoritativeWorker()) {
            cancellationToResume = null;
        }
        assertInvariant();
        return new AdmissionMutationCompletion(
                true, cancellationToResume, pendingTerminal,
                pendingRetirement, inactivityExpired);
    }

    boolean deferInactivityExpiryDuringAdmission(String detail) {
        requireSlotLock("admission inactivity expiry");
        if (!deferCancellationDuringAdmission(CancelReason.DEADLINE_EXCEEDED, detail)) {
            return false;
        }
        pendingAdmissionInactivityExpired = true;
        assertInvariant();
        return true;
    }

    boolean deferCancellationDuringAdmission(
            CancelReason reason,
            String detail) {
        requireSlotLock("admission cancellation");
        if (!ownsActiveGeneration() || admissionMutation == null) {
            return false;
        }
        admissionOpen = false;
        if (pendingAdmissionCancelReason == null) {
            pendingAdmissionCancelReason = reason;
            requestCancel(detail);
        }
        assertInvariant();
        return true;
    }

    /**
     * Atomically move the admission-scoped first cause into the canonical
     * cancellation owner before releasing the slot lock. The lifecycle was
     * already moved to {@code CANCEL_REQUESTED} when the cause was deferred;
     * this transfer prevents a later cancel from replacing it while the
     * admission event handler resumes cancellation effects outside the lock.
     */
    private CancelReason promoteAdmissionCancellation(CancelReason pending) {
        requireSlotLock("admission cancellation promotion");
        if (pending == null) {
            return null;
        }
        if (cancellationReason == null) {
            cancellationReason = pending;
        }
        return cancellationReason;
    }

    private DeferredTerminal admissionPendingTerminal;
    private PendingPrefillRetirement admissionPendingPrefillRetirement;

    private void retainAdmissionTerminal(DeferredTerminal candidate) {
            if (admissionPendingTerminal == null
                    || (!admissionPendingTerminal.endpointAlreadyRetired()
                        && (candidate.endpointAlreadyRetired()
                            || (!admissionPendingTerminal.authoritativeWorker()
                                && candidate.authoritativeWorker())))) {
                admissionPendingTerminal = candidate;
            }
    }

    private void retainAdmissionPrefillRetirement(
                PendingPrefillRetirement candidate) {
            if (admissionPendingPrefillRetirement == null) {
                admissionPendingPrefillRetirement = candidate;
            } else if (admissionPendingPrefillRetirement.source != candidate.source
                    || admissionPendingPrefillRetirement.item != candidate.item) {
                throw new IllegalStateException(
                        "admission mutation observed another Prefill generation"
                                + " for request " + requestId);
            }
    }

    /** Exact Prefill retirement retained only by its in-flight admission. */
    private record PendingPrefillRetirement(
            PrefillEndpoint source,
            ScheduledRequest item,
            TerminalOutcome transition,
            Response response) {
    }

    // ==================== Exact deadline capabilities ====================

    void configureInactivityTimeout(long timeoutMs) {
        requireSlotLock("request inactivity configuration");
        if (timeoutMs <= 0L) {
            throw new IllegalArgumentException("request inactivity timeout must be positive");
        }
        inactivityTimeoutMs = timeoutMs;
    }

    OptionalLong inactivityDeadlineAtMs() {
        requireSlotLock("request inactivity deadline planning");
        return ownsActiveGeneration() && !state.isTerminal() && inactivityDeadline == null
                && pendingAdmissionCancelReason == null && inactivityTimeoutMs > 0L
                ? OptionalLong.of(deadlineAfter(lastWorkerStatusAtMs, inactivityTimeoutMs))
                : OptionalLong.empty();
    }

    boolean installInactivityDeadline(InactivityDeadline exact) {
        requireSlotLock("request inactivity installation");
        if (inactivityDeadlineAtMs().isEmpty()) {
            return false;
        }
        inactivityDeadline = exact;
        return true;
    }

    boolean expireInactivityDeadline(InactivityDeadline exact) {
        requireSlotLock("request inactivity check");
        if (inactivityDeadline != exact || !ownsActiveGeneration()) {
            return false;
        }
        inactivityDeadline = null;
        return !state.isTerminal();
    }

    void startDecisionTracking(WorkSnapshot precedingWork, long unstartedWorkMs, long nowMs) {
        requireSlotLock("delivery prediction consumption");
        Objects.requireNonNull(precedingWork, "precedingWork");
        if (unstartedWorkMs < 0L) {
            throw new IllegalArgumentException("unstarted work must be non-negative");
        }
        if (deliveryPredictionConsumed) {
            throw new IllegalStateException("delivery prediction already consumed");
        }
        double lifetime = item.ctx().getConfig().getRequestLifecycle().getDecision().getLifetime();
        if (!Double.isFinite(lifetime) || lifetime < 1.0) {
            throw new IllegalArgumentException("invalid decision lifetime");
        }
        deliveryPredictionConsumed = true;
        switch (decisionStage) {
            case UNDECIDED -> {
                decisionStage = DecisionStage.WAITING_ENGINE;
                OptionalLong precedingMs = precedingWork.totalRemainingWorkMsAt(nowMs);
                // Unknown work cannot prove a request is lost; inactivity detection still applies.
                if (precedingMs.isPresent()) {
                    long remainingMs = addWork(precedingMs.getAsLong(), unstartedWorkMs);
                    double scaled = Math.ceil(remainingMs * lifetime);
                    long durationMs = scaled >= Long.MAX_VALUE ? Long.MAX_VALUE : (long) scaled;
                    decisionExpiresAtMs = OptionalLong.of(deadlineAfter(
                            nowMs, addWork(durationMs, DECODE_HANDOFF_GRACE_MS)));
                }
            }
            case WAITING_DECODE -> decisionExpiresAtMs = OptionalLong.of(
                    deadlineAfter(prefillCompletedAtMs, DECODE_HANDOFF_GRACE_MS));
            case PREFILL_RUNNING, ACCEPTED -> { }
            case WAITING_ENGINE -> throw new IllegalStateException("delivery already observed");
        }
    }

    private void observeDecisionPrefillActive() {
        if (decisionStage == DecisionStage.UNDECIDED || decisionStage == DecisionStage.WAITING_ENGINE) {
            advanceDecision(DecisionStage.PREFILL_RUNNING, OptionalLong.empty());
        }
    }

    private void observeDecisionPrefillCompleted(long observedAtMs, boolean separateDecode) {
        if (decisionStage == DecisionStage.ACCEPTED || decisionStage == DecisionStage.WAITING_DECODE) {
            return;
        }
        prefillCompletedAtMs = observedAtMs;
        advanceDecision(separateDecode ? DecisionStage.WAITING_DECODE : DecisionStage.ACCEPTED,
                separateDecode && deliveryPredictionConsumed
                        ? OptionalLong.of(deadlineAfter(observedAtMs, DECODE_HANDOFF_GRACE_MS))
                        : OptionalLong.empty());
    }

    private void advanceDecision(DecisionStage next, OptionalLong nextDeadline) {
        decisionStage = next;
        decisionExpiresAtMs = nextDeadline;
        decisionExpired = false;
    }

    private static long addWork(long precedingMs, long unstartedMs) {
        return precedingMs > Long.MAX_VALUE - unstartedMs ? Long.MAX_VALUE : precedingMs + unstartedMs;
    }

    private static long deadlineAfter(long startedAtMs, long durationMs) {
        if (startedAtMs < 0L || durationMs <= 0L) {
            throw new IllegalArgumentException("deadline requires a valid start and positive duration");
        }
        return startedAtMs > Long.MAX_VALUE - durationMs ? Long.MAX_VALUE : startedAtMs + durationMs;
    }

    boolean requestInactive(long nowMs) {
        requireSlotLock("request inactivity lookup");
        return inactivityTimeoutMs > 0L
                && nowMs >= deadlineAfter(lastWorkerStatusAtMs, inactivityTimeoutMs);
    }

    boolean needsDecisionConfirmation() {
        requireSlotLock("decision evidence lookup");
        return ownsActiveGeneration() && item != null
                && cancellationReason == null && pendingAdmissionCancelReason == null
                && (decisionExpired && (decisionStage == DecisionStage.WAITING_ENGINE
                    || decisionStage == DecisionStage.WAITING_DECODE));
    }

    /** Retain uncertainty as a diagnostic while normal ACK/status and TTL stay active. */
    boolean markAwaitingConfirmation(String message) {
        requireSlotLock("delivery confirmation wait");
        if (!ownsActiveGeneration() || cancellationReason != null
                || pendingAdmissionCancelReason != null
                || decisionStage == DecisionStage.PREFILL_RUNNING
                || decisionStage == DecisionStage.ACCEPTED) {
            return false;
        }
        detail = "SUSPECTED_LOST: " + Objects.requireNonNull(message, "message");
        updatedAtMs = System.currentTimeMillis();
        assertInvariant();
        return true;
    }

    /** Matching Engine evidence resolves the diagnostic suspicion. */
    void reconcileDecisionEvidence() {
        requireSlotLock("decision evidence reconciliation");
        if (!ownsActiveGeneration() || (decisionStage == DecisionStage.UNDECIDED || decisionStage == DecisionStage.WAITING_ENGINE)
                || needsDecisionConfirmation() || cancellationReason != null
                || pendingAdmissionCancelReason != null) {
            return;
        }
        if (detail.startsWith("SUSPECTED_LOST")) {
            detail = "Engine request observed; waiting for completion";
        }
    }

    boolean installRequestDeadline(RequestDeadline exact) {
        requireSlotLock("request deadline installation");
        if (!ownsActiveGeneration() || !isOpen()) {
            return false;
        }
        if (requestDeadline != null) {
            throw new IllegalStateException(
                    "request deadline already installed for " + requestId);
        }
        requestDeadline = exact;
        assertInvariant();
        return true;
    }

    boolean expireRequestDeadline(RequestDeadline exact) {
        requireSlotLock("request deadline expiry");
        if (requestDeadline != exact) {
            return false;
        }
        requestDeadline = null;
        if (!ownsActiveGeneration() || future.isDone() || !isOpen()) {
            assertInvariant();
            return false;
        }
        admissionOpen = false;
        if (admissionMutation != null) {
            if (pendingAdmissionCancelReason == null) {
                pendingAdmissionCancelReason = CancelReason.DEADLINE_EXCEEDED;
                requestCancel(
                        "request scheduling deadline exceeded during admission");
            }
            assertInvariant();
            return false;
        }
        assertInvariant();
        return true;
    }

    OptionalLong decisionDeadlineAtMs() {
        requireSlotLock("decision deadline planning");
        return ownsActiveGeneration() && decisionDeadline == null
                ? decisionExpiresAtMs : OptionalLong.empty();
    }

    boolean installDecisionDeadline(DecisionDeadline exact) {
        requireSlotLock("decision deadline installation");
        if (!ownsActiveGeneration() || decisionDeadline != null
                || !decisionExpiresAtMs.equals(OptionalLong.of(exact.deadlineAtMs()))) {
            return false;
        }
        decisionDeadline = exact;
        return true;
    }

    /** Detach an old phase's capability before arming the next phase. */
    DecisionDeadline detachObsoleteDecisionDeadline() {
        requireSlotLock("decision deadline reconciliation");
        if (decisionDeadline == null || decisionExpiresAtMs.equals(
                OptionalLong.of(decisionDeadline.deadlineAtMs()))) {
            return null;
        }
        return detachDecisionDeadline();
    }

    DecisionExpiry expireDecisionDeadline(DecisionDeadline exact) {
        requireSlotLock("decision deadline expiry");
        if (decisionDeadline != exact) {
            return null;
        }
        decisionDeadline = null;
        // An Engine fact may have changed the phase before the old timer fired.
        if (!decisionExpiresAtMs.equals(OptionalLong.of(exact.deadlineAtMs()))) {
            return null;
        }
        decisionExpired = true;
        decisionExpiresAtMs = OptionalLong.empty();
        boolean needsConfirmation = needsDecisionConfirmation();
        if (needsConfirmation) {
            markAwaitingConfirmation(decisionStage == DecisionStage.WAITING_ENGINE
                    ? "no Engine request evidence before visibility deadline"
                    : "Decode acceptance missing after Prefill completion");
        }
        assertInvariant();
        return new DecisionExpiry(item, needsConfirmation);
    }

    /** Atomically detach all timer-owned capabilities during timer close. */
    ExpirationTimer.DetachedDeadlines detachDeadlinesForTimerClose() {
        requireSlotLock("deadline detach for timer close");
        ExpirationTimer.DetachedDeadlines detached =
                new ExpirationTimer.DetachedDeadlines(
                        requestDeadline, decisionDeadline, inactivityDeadline);
        requestDeadline = null;
        decisionDeadline = null;
        inactivityDeadline = null;
        assertInvariant();
        return detached;
    }

    private DecisionDeadline detachDecisionDeadline() {
        DecisionDeadline deadline = decisionDeadline;
        decisionDeadline = null;
        return deadline;
    }

    // ==================== Cancellation first cause ====================

    boolean hasCancellationFirstCause() {
        requireSlotLock("cancellation first-cause lookup");
        return cancellationReason != null;
    }

    CancelReason requireCancellationFirstCause() {
        requireSlotLock("cancellation first-cause claim");
        if (cancellationReason == null) {
            throw new IllegalStateException(
                    "missing cancellation first cause for request " + requestId);
        }
        return cancellationReason;
    }

    RequestState markCancellationRequested(
            CancelReason reason,
            String detail) {
        requireSlotLock("cancellation claim");
        if (!ownsActiveGeneration()) {
            return snapshot();
        }
        if (cancellationReason == null) {
            cancellationReason = reason;
            admissionOpen = false;
            requestCancel(detail);
        }
        assertInvariant();
        return snapshot();
    }

    // ==================== Preemption sub-state machine ====================

    PreemptionRegistration tryInstallPreemption(
            long reservationToken,
            long attemptToken,
            String detail) {
        requireSlotLock("preemption installation");
        DecodeEndpoint.ReservationHandle reservation =
                item == null ? null : item.decodeReservation();
        if (!ownsActiveGeneration()
                || admissionMutation != null
                || preemption != null
                || cancellationReason != null
                || reservation == null
                || reservation.reservationToken() != reservationToken
                || (deliveryClaimKind
                        == DeliveryClaimKind.ROUTE_DECISION
                    && state
                        == RequestState.Phase.DISPATCHING)) {
            return null;
        }
        preemption = new PreemptionRegistration(
                this, requestId, attemptToken, detail);
        assertInvariant();
        return preemption;
    }

    /** Advance one exact coordinator-owned Cancel phase. */
    PreemptionReduction applyPreemptionPhase(
            PreemptionRegistration claim,
            PreemptionCancelPhase next) {
        requireSlotLock("preemption phase reduction");
        PreemptionRegistration exact = exactPreemption(claim);
        if (!ownsActiveGeneration()
                || exact == null
                || preemption != exact
                || (next == PreemptionCancelPhase.CANCEL_IN_FLIGHT
                    && cancellationReason != null)
                || !exact.advanceTo(next)) {
            return PreemptionReduction.STALE;
        }
        if (next == PreemptionCancelPhase.CANCEL_REQUESTED) {
            requestCancel(exact.detail());
        }
        assertInvariant();
        return switch (next) {
            case CLAIMED -> PreemptionReduction.STALE;
            case CANCEL_IN_FLIGHT, CANCEL_REQUESTED ->
                    PreemptionReduction.NONE;
            case NOT_FOUND_STALE -> materializePendingReplay(exact, false, exact);
            case CANCEL_UNKNOWN ->
                    materializePendingReplay(exact, true, exact);
        };
    }

    PreemptionReduction applyPreemptionRelease(
            PreemptionRegistration claim) {
        requireSlotLock("preemption release reduction");
        PreemptionRegistration exact = exactPreemption(claim);
        if (!ownsActiveGeneration()
                || exact == null
                || preemption != exact
                || !exact.isReleasable()) {
            return PreemptionReduction.STALE;
        }
        detachPreemptionOwner(exact);
        return materializePendingReplay(exact, false, exact);
    }

    PreemptionReduction applyPreemptionTombstone(
            PreemptionRegistration claim,
            String detail) {
        requireSlotLock("preemption tombstone reduction");
        PreemptionRegistration exact = exactPreemption(claim);
        if (!ownsActiveGeneration()
                || exact == null
                || !exact.canSettleTombstone()
                || !exact.settle()) {
            return PreemptionReduction.STALE;
        }
        DeferredTerminal terminal = DeferredTerminal.priority(detail);
        exact.retainTerminal(terminal);
        // DecodePreemptionCoordinator has already consumed the exact endpoint
        // claim before publishing TOMBSTONED. Reconciliation is therefore
        // neither required nor legal on this authoritative path.
        detachPreemptionOwner(exact);
        assertInvariant();
        return PreemptionReduction.replay(
                PendingReplay.terminal(terminal),
                exact);
    }

    /**
     * Reduce one exact transport/endpoint fact without exposing the mutable preemption
     * registration. The caller holds {@code synchronized (slot)} and only executes the returned,
     * already-selected effect.
     */
    PreemptionReduction reducePrefillActive(PrefillEndpoint source, ScheduledRequest expected) {
        requireSlotLock("Prefill activity reduction");
        PreemptionRegistration exact = preemption;
        if (!ownsPrefillFact(source, expected) || exact == null || !exact.isNotFound()) {
            return PreemptionReduction.STALE;
        }
        DecodeEndpoint decode = expected.decodeEp();
        if (decode == null
                || decode.reconcilePriorityVictimActive(
                        exact.attemptToken(), expected.decodeReservation())) {
            detachPreemptionOwner(exact);
        }
        return PreemptionReduction.NONE;
    }

    PreemptionReduction reduceWorkerTerminal(ScheduledRequest expected, DeferredTerminal terminal) {
        requireSlotLock("worker terminal reduction");
        if (!terminal.authoritativeWorker()) {
            throw new IllegalArgumentException(
                    "worker terminal requires authoritative observation");
        }
        if (!ownsActiveItem(expected)) {
            return PreemptionReduction.STALE;
        }
        if (admissionMutation != null) {
            retainAdmissionTerminal(terminal);
            assertInvariant();
            return PreemptionReduction.NONE;
        }
        PreemptionRegistration exact = preemptionOwner();
        if (exact == null) {
            return PreemptionReduction.replay(PendingReplay.terminal(terminal), null);
        }
        if (exact.isSettled()) {
            return PreemptionReduction.STALE;
        }
        exact.retainTerminal(terminal);
        if (!ownsActiveGeneration() || preemptionOwner() != exact || !exact.settle()) {
            return PreemptionReduction.STALE;
        }
        assertInvariant();
        return materializePendingReplay(exact, false, exact);
    }

    PreemptionReduction reduceDispatchRejected(
            DecodeEndpoint source,
            DecodeEndpoint.ReservationHandle reservation,
            ScheduledRequest expected,
            DeferredTerminal terminal) {
        requireSlotLock("dispatch rejection reduction");
        if (terminal.kind() != DeferredTerminal.Kind.DELIVERY_REJECTED) {
            throw new IllegalArgumentException(
                    "dispatch rejection requires delivery-rejected terminal");
        }
        if (!ownsActiveItem(expected) || !ownsDecodeFact(source, reservation)) {
            return PreemptionReduction.STALE;
        }
        if (admissionMutation != null) {
            retainAdmissionTerminal(terminal);
            assertInvariant();
            return PreemptionReduction.NONE;
        }
        PreemptionRegistration exact = preemptionOwner();
        PreemptionRegistration signal = null;
        if (exact != null) {
            exact.retainTerminal(terminal);
            if (exact.settle()) {
                signal = exact;
            }
            detachPreemptionOwner(exact);
        }
        assertInvariant();
        return PreemptionReduction.replay(PendingReplay.terminal(terminal), signal);
    }

    PreemptionReduction reduceOrdinaryTerminal(
            ScheduledRequest expected, DeferredTerminal terminal) {
        requireSlotLock("ordinary terminal reduction");
        if (!ownsActiveItem(expected)) {
            return PreemptionReduction.STALE;
        }
        if (terminal.authoritativeWorker()) {
            throw new IllegalArgumentException("authoritative worker fact requires WorkerTerminal");
        }
        if (admissionMutation != null) {
            retainAdmissionTerminal(terminal);
            assertInvariant();
            return PreemptionReduction.NONE;
        }

        PreemptionRegistration exact = preemptionOwner();
        if (engineOwnership == EngineOwnership.DECODE_OWNED && terminal.deliveryFailure()) {
            DeliveryClaimKind deliveryKind = deliveryClaimKind;
            long deliveryBatchId = batchId;
            if (exact == null) {
                DeliveryConfirmation confirmation =
                        confirmDeliveryForPublication(
                                expected, deliveryKind, deliveryBatchId);
                return confirmation == null
                        ? PreemptionReduction.STALE
                        : PreemptionReduction.replay(
                                PendingReplay.delivery(
                                        confirmation, expected, deliveryKind,
                                        deliveryBatchId),
                                null);
            }
            if (exact.isSettled()) {
                return PreemptionReduction.STALE;
            }
            exact.recordDeliveryConfirmation(deliveryBatchId);
            assertInvariant();
            return materializePendingReplay(exact, false, null);
        }

        if (exact == null) {
            return PreemptionReduction.replay(PendingReplay.terminal(terminal), null);
        }
        if (exact.isSettled()) {
            return PreemptionReduction.STALE;
        }
        exact.retainTerminal(terminal);
        assertInvariant();
        if (!exact.isNotFound() && !exact.isUnknown()) {
            return PreemptionReduction.NONE;
        }
        return materializePendingReplay(exact, exact.isUnknown(), exact);
    }

    PreemptionReduction reducePriorityCanceled(PrefillEndpoint source, ScheduledRequest expected) {
        requireSlotLock("priority cancellation reduction");
        PreemptionRegistration exact = ownsPrefillFact(source, expected) ? preemptionOwner() : null;
        DecodeEndpoint decode = expected.decodeEp();
        if (exact == null
                || exact.isSettled()
                || decode == null
                || expected.decodeReservation() == null
                || !decode.settlePriorityCanceled(
                        exact.attemptToken(), expected.decodeReservation())
                || !ownsActiveGeneration()
                || preemptionOwner() != exact
                || !exact.settle()) {
            return PreemptionReduction.STALE;
        }
        DeferredTerminal terminal = DeferredTerminal.priority("priority victim canceled by worker");
        exact.retainTerminal(terminal);
        detachPreemptionOwner(exact);
        assertInvariant();
        return PreemptionReduction.replay(PendingReplay.terminal(terminal), exact);
    }

    PreemptionReduction reduceDecodeGenerationRetired(
            DecodeEndpoint source, DecodeEndpoint.ReservationHandle reservation, String detail) {
        requireSlotLock("Decode generation retirement reduction");
        Objects.requireNonNull(detail, "detail");
        if (!ownsDecodeFact(source, reservation)) {
            return PreemptionReduction.STALE;
        }
        DeferredTerminal terminal = DeferredTerminal.decodeGenerationRetired(detail);
        if (admissionMutation != null) {
            retainAdmissionTerminal(terminal);
            assertInvariant();
            return PreemptionReduction.NONE;
        }

        PreemptionRegistration exact = preemptionOwner();
        PreemptionRegistration signal = null;
        if (exact != null) {
            exact.retainTerminal(terminal);
            exact.settle();
            signal = exact;
        }
        detachPreemptionOwner(exact);

        assertInvariant();
        return PreemptionReduction.replay(
                PendingReplay.terminal(terminal), signal);
    }

    PreemptionReduction reduceDeliveryConfirmed(long batchId) {
        requireSlotLock("delivery confirmation reduction");
        ScheduledRequest active = activeItem();
        DeliveryClaimKind deliveryKind = deliveryClaimKind;
        if (active == null || !ownsDeliveryClaim(active, deliveryKind, batchId)) {
            return PreemptionReduction.STALE;
        }
        if (cancellationReason != null || pendingAdmissionCancelReason != null) {
            return PreemptionReduction.NONE;
        }
        PreemptionRegistration exact = preemptionOwner();
        if (exact == null) {
            DeliveryConfirmation confirmation =
                    confirmDeliveryForPublication(active, deliveryKind, batchId);
            return confirmation == null
                    ? PreemptionReduction.STALE
                    : PreemptionReduction.replay(
                            PendingReplay.delivery(confirmation, active, deliveryKind, batchId),
                            null);
        }
        if (exact.isSettled()) {
            return PreemptionReduction.STALE;
        }
        exact.recordDeliveryConfirmation(batchId);
        assertInvariant();
        return materializePendingReplay(exact, false, null);
    }

    private PreemptionReduction materializePendingReplay(
            PreemptionRegistration exact,
            boolean transportUnknown,
            PreemptionRegistration signal) {
        DeferredTerminal terminal = exact.pendingTerminal();
        if (terminal != null
                && (!transportUnknown || terminal.authoritativeWorker())) {
            ScheduledRequest active = activeItem();
            DecodeEndpoint decode = active == null ? null : active.decodeEp();
            boolean ordinaryWon = terminalOwnsDecodeSettlement(
                    terminal,
                    decode,
                    exact.attemptToken(),
                    active == null ? null : active.decodeReservation());
            if (!ordinaryWon) {
                return PreemptionReduction.NONE;
            }
            detachPreemptionOwner(exact);
            return PreemptionReduction.replay(
                    PendingReplay.terminal(terminal),
                    signal);
        }
        if (transportUnknown || !exact.hasPendingDeliveryConfirmation()) {
            return PreemptionReduction.NONE;
        }

        ScheduledRequest active = activeItem();
        DecodeEndpoint decode = active == null ? null : active.decodeEp();
        boolean activeWon = decode == null
                || decode.reconcilePriorityVictimActive(
                        exact.attemptToken(),
                        active.decodeReservation());
        if (!activeWon) {
            return PreemptionReduction.NONE;
        }
        detachPreemptionOwner(exact);
        if (active == null) {
            return PreemptionReduction.STALE;
        }
        DeliveryConfirmation confirmation = confirmDeliveryForPublication(
                active,
                deliveryClaimKind,
                exact.pendingConfirmationBatchId());
        return confirmation == null
                ? PreemptionReduction.STALE
                : PreemptionReduction.replay(
                        PendingReplay.delivery(
                                confirmation,
                                active,
                                deliveryClaimKind,
                                exact.pendingConfirmationBatchId()),
                        signal);
    }

    /**
     * Resolve the unique owner of Decode priority settlement. A terminal
     * reduced by DecodeEndpoint is already post-commit and must never be sent
     * back into the endpoint claim state machine. Prefill-backed observations
     * still need the exact reconciliation transaction.
     */
    static boolean terminalOwnsDecodeSettlement(
            DeferredTerminal terminal,
            DecodeEndpoint decode,
            long attemptToken,
            DecodeEndpoint.ReservationHandle reservation) {
        if (decode == null || terminal.decodeSettlementCommitted()) {
            return true;
        }
        return decode.reconcilePriorityVictimFinished(
                attemptToken, reservation);
    }

    private boolean detachPreemptionOwner(PreemptionRegistration exact) {
        requireSlotLock("preemption detach");
        if (exact == null || preemption != exact) {
            return false;
        }
        preemption = null;
        assertInvariant();
        return true;
    }

    private PreemptionRegistration exactPreemption(
            PreemptionRegistration claim) {
        if (!(claim instanceof PreemptionRegistration exact)
                || exact.requestId() != requestId
                || preemptionOwner() != exact) {
            return null;
        }
        return exact;
    }

    private PreemptionRegistration preemptionOwner() {
        return preemption;
    }

    /** Authoritative Decode ownership ends the decision-confirmation watch. */
    DecodeAcceptance markDecodeAccepted() {
        requireSlotLock("Decode acceptance");
        if (!ownsActiveGeneration()) {
            return DecodeAcceptance.NONE;
        }
        engineOwnership = EngineOwnership.DECODE_OWNED;
        advanceDecision(DecisionStage.ACCEPTED, OptionalLong.empty());
        reconcileDecisionEvidence();
        DecisionDeadline detachedDeadline = detachDecisionDeadline();
        assertInvariant();
        return new DecodeAcceptance(detachedDeadline);
    }

    void markDecodeTerminalOwned() {
        requireSlotLock("Decode terminal ownership");
        if (ownsActiveGeneration()) {
            engineOwnership = EngineOwnership.DECODE_OWNED;
            assertInvariant();
        }
    }

    TerminalAction beginPrefillRetirementTerminal(
            PrefillEndpoint source,
            ScheduledRequest expected,
            TerminalOutcome transition,
            Response response) {
        requireSlotLock("Prefill retirement terminal claim");
        PendingPrefillRetirement pending = new PendingPrefillRetirement(
                source, expected, transition, response);
        if (!canTerminateFromPrefillRetirement(source, expected)) {
            return null;
        }
        if (admissionMutation != null) {
            retainAdmissionPrefillRetirement(pending);
            assertInvariant();
            return null;
        }
        return beginPendingPrefillRetirement(pending);
    }

    private boolean canTerminateFromPrefillRetirement(
            PrefillEndpoint source,
            ScheduledRequest expected) {
        return ownsPrefillFact(source, expected)
                && engineOwnership != EngineOwnership.DECODE_OWNED
                && preemption == null
                && !deliveryClaimKind.isClaimed();
    }

    private TerminalAction beginPendingPrefillRetirement(
            PendingPrefillRetirement pending) {
        if (pending == null
                || !canTerminateFromPrefillRetirement(
                        pending.source, pending.item)) {
            return null;
        }
        return beginTerminalizing(
                true,
                true,
                false,
                null,
                pending.transition,
                pending.response,
                true);
    }

    // ==================== Terminal ownership ====================

    TerminalAction beginTerminalizing(
            boolean removePrefillQueue,
            boolean releaseDecode,
            boolean releasePrefill,
            Runnable counterpartCleanup,
            TerminalOutcome transition,
            Response response) {
        return beginTerminalizing(
                removePrefillQueue,
                releaseDecode,
                releasePrefill,
                counterpartCleanup,
                transition,
                response,
                response != null);
    }

    /** Claim a locally reversible terminal specifically for public-future use. */
    TerminalAction beginExternalTerminalizing(
            TerminalOutcome transition) {
        requireSlotLock("external terminal claim");
        if (!canClaimLocalTerminal()) {
            return null;
        }
        boolean ownsItem = item != null;
        return beginTerminalizing(
                ownsItem,
                ownsItem,
                ownsItem,
                null,
                transition,
                null,
                true);
    }

    private TerminalAction beginTerminalizing(
            boolean removePrefillQueue,
            boolean releaseDecode,
            boolean releasePrefill,
            Runnable counterpartCleanup,
            TerminalOutcome transition,
            Response response,
            boolean requestPublication) {
        requireSlotLock("terminal claim");
        if (transition == null) {
            throw new IllegalStateException(
                    "terminal transition is required for request " + requestId);
        }
        if (!ownsActiveGeneration() || admissionMutation != null) {
            return null;
        }
        boolean publishable = requestPublication
                && publicationWinner == null
                && !future.isDone();
        PublicationPermit publication = publishable
                ? requirePublicationPermit(PublicationKind.TERMINAL) : null;
        boolean transferred = false;
        try {
            slotPhase = SlotPhase.TERMINALIZING;
            admissionOpen = false;
            if (publication != null) {
                publicationWinner = PublicationKind.TERMINAL;
            }

            PreemptionRegistration claimedPreemption = preemption;
            preemption = null;
            if (claimedPreemption != null) {
                claimedPreemption.settle();
            }

            RequestDeadline claimedRequestDeadline = requestDeadline;
            requestDeadline = null;
            DecisionDeadline detachedDecisionDeadline = detachDecisionDeadline();
            InactivityDeadline claimedInactivityDeadline = inactivityDeadline;
            inactivityDeadline = null;
            TerminalResources terminalResources =
                    claimedRequestDeadline == null && detachedDecisionDeadline == null && claimedInactivityDeadline == null
                            ? null
                            : new TerminalResources(
                                    claimedRequestDeadline, detachedDecisionDeadline, claimedInactivityDeadline);
            TerminalAction action = new TerminalAction(
                    this,
                    item,
                    claimedPreemption,
                    terminalResources,
                    removePrefillQueue,
                    releaseDecode,
                    releasePrefill,
                    counterpartCleanup,
                    "TERMINAL_RELEASE",
                    transition,
                    publishable ? response : null,
                    publication);
            transferred = true;
            assertInvariant();
            return action;
        } finally {
            if (!transferred && publication != null) {
                publication.abandonIfUnclaimed();
            }
        }
    }

    TombstoneResult finishTombstone(TerminalAction action) {
        requireSlotLock("terminal tombstone");
        if (!isCurrentGeneration()
                || slotPhase != SlotPhase.TERMINALIZING
                || action.slot() != this
                || item != action.item()) {
            if (action.publication() != null) {
                action.publication().abandonIfUnclaimed();
            }
            return new TombstoneResult(
                    null,
                    new IllegalStateException(
                            "terminal slot identity changed: request_id="
                                    + requestId),
                    null);
        }
        RequestState terminal;
        Throwable transitionFailure = null;
        try {
            terminal = applyTerminalOutcome(action.transition());
        } catch (Throwable failure) {
            transitionFailure = failure;
            terminal = fail("terminal projection failed");
        }
        if (!terminal.state().isTerminal()) {
            transitionFailure = appendFailure(
                    transitionFailure,
                    new IllegalStateException(
                            "terminal transition did not terminate request "
                                    + requestId));
            terminal = fail("terminal projection did not terminate");
        }

        item = null;
        preemption = null;
        cancellationReason = null;
        admissionMutation = null;
        pendingAdmissionCancelReason = null;
        pendingAdmissionInactivityExpired = false;
        requestDeadline = null;
        decisionDeadline = null;
        inactivityDeadline = null;
        slotPhase = SlotPhase.TOMBSTONE;
        assertInvariant();
        return new TombstoneResult(
                terminal,
                transitionFailure,
                action.publication());
    }

    private PublicationPermit requirePublicationPermit(
            PublicationKind kind) {
        PublicationPermit permit = completionPublisher.tryReservePublication(
                this, kind);
        if (permit == null || permit.slot() != this) {
            if (permit != null) {
                permit.abandonIfUnclaimed();
            }
            throw new IllegalStateException(
                    "frontend publication is closed for request " + requestId);
        }
        return permit;
    }

    /** Select the response under the slot lock; complete its future only after unlocking. */
    private boolean claimPublicationResult(PublicationKind kind) {
        requireSlotLock("frontend result selection");
        if (future.isDone()) {
            return false;
        }
        if (kind == PublicationKind.TERMINAL) {
            return publicationWinner == PublicationKind.TERMINAL;
        }
        if (publicationWinner != null || !ownsActiveGeneration()
                || state != RequestState.Phase.ACKNOWLEDGED
                || cancellationReason != null || pendingAdmissionCancelReason != null) {
            return false;
        }
        publicationWinner = PublicationKind.DELIVERY;
        return true;
    }

    private boolean isCurrentGeneration() {
        return currentGeneration;
    }

    void detachGeneration() {
        requireSlotLock("request generation detach");
        if (!currentGeneration) {
            throw new IllegalStateException(
                    "request generation already detached: " + requestId);
        }
        currentGeneration = false;
    }

    private void assertInvariant() {
        requireSlotLock("request slot invariant");
        invariantHolds();
    }

    /** Verify the aggregate at every mutation boundary. */
    private void invariantHolds() {
        if (admissionMutation != null && preemption != null) {
            throw new IllegalStateException(
                    "admission mutation overlaps preemption for " + requestId);
        }
        if ((pendingAdmissionCancelReason != null || pendingAdmissionInactivityExpired)
                && admissionMutation == null) {
            throw new IllegalStateException(
                    "pending admission cancellation has no mutation owner for "
                            + requestId);
        }
        if (slotPhase == SlotPhase.TERMINALIZING
                && (admissionOpen || admissionMutation != null)) {
            throw new IllegalStateException(
                    "terminalizing request still owns admission "
                            + requestId);
        }
        if (slotPhase != SlotPhase.TOMBSTONE) {
            return;
        }
        if (admissionOpen
                || item != null
                || cancellationReason != null
                || preemption != null
                || admissionMutation != null
                || pendingAdmissionCancelReason != null
                || pendingAdmissionInactivityExpired
                || requestDeadline != null
                || decisionDeadline != null || inactivityDeadline != null) {
            throw new IllegalStateException(
                    "tombstone retains request-owned state for " + requestId);
        }
        if (!state.isTerminal()) {
            throw new IllegalStateException(
                    "tombstone lifecycle is not terminal for " + requestId);
        }
    }

    private void requireSlotLock(String operation) {
        if (!Thread.holdsLock(this)) {
            throw new IllegalStateException(
                    operation + " requires slot lock for request " + requestId);
        }
    }

    private static Throwable appendFailure(
            Throwable first,
            Throwable next) {
        if (first == null) {
            return next;
        }
        if (first != next) {
            first.addSuppressed(next);
        }
        return first;
    }

    private void resumeCancellationAfterAdmission(
            CancelReason reason,
            boolean inactivityExpired) {
        TerminalAction localCompletion = null;
        synchronized (this) {
            if (!this.isCurrentGeneration() || !this.ownsActiveGeneration()) {
                return;
            }
            CancelReason firstCause = this.requireCancellationFirstCause();
            if (firstCause != reason) {
                throw new IllegalStateException(
                        "admission cancellation first cause changed for request " + this.requestId());
            }
            String detail = inactivityExpired
                    ? "REQUEST_INACTIVE: no matching Engine request status before inactivity timeout"
                    : cancelDetail(firstCause);
            ScheduledRequest item = this.activeItem();
            if (inactivityExpired || firstCause == CancelReason.DEADLINE_EXCEEDED
                    || this.requestInactive(System.currentTimeMillis())) {
                localCompletion = beginExpiredRequestLocked(detail);
            } else if (item == null || this.canClaimLocalTerminal()) {
                localCompletion = beginTerminalLocked(item != null, item != null,
                        TerminalOutcome.cancellation(firstCause, detail),
                        buildErrorResponse(this.cancellationErrorType(firstCause), detail));
            }
        }
        terminalCleanup.submitTerminal(localCompletion);
    }

    private Runnable materializePostLockActionLocked(
            RequestSlot.PreemptionReduction reduction,
            Runnable priorityCounterpartCleanup) {
        if (!Thread.holdsLock(this)) {
            throw new IllegalStateException(
                    "preemption reduction requires slot lock");
        }
        if (reduction.status()
                == RequestSlot.PreemptionReduction.Status.STALE) {
            return null;
        }
        if (reduction.status()
                == RequestSlot.PreemptionReduction.Status.NONE) {
            return NO_POST_LOCK_ACTION;
        }
        Runnable publication = materializeReplayLocked(reduction.replay(), priorityCounterpartCleanup);
        if (publication == null) {
            throw new IllegalStateException(
                    "accepted preemption replay produced no publication for request "
                            + this.requestId());
        }
        return replayPostLockAction(publication,
                reduction.signal());
    }

    private Runnable materializeReplayLocked(
            RequestSlot.PendingReplay replay,
            Runnable priorityCounterpartCleanup) {
        if (replay.terminal() != null) {
            DeferredTerminal exact = replay.terminal();
            if (exact.kind() == DeferredTerminal.Kind.PRIORITY) {
                return terminalPublication(
                        beginSettledPriorityTerminalLocked(exact.detail(),
                                priorityCounterpartCleanup));
            }
            return applyOrdinaryTerminalLocked(exact);
        }
        return deliveryPublication(replay.item(),
                replay.confirmation(),
                replay.kind(),
                replay.batchId());
    }

    private Runnable replayPostLockAction(
            Runnable publication,
            PreemptionRegistration terminalSignal) {
        return () -> {
            try {
                publication.run();
            } finally {
                if (terminalSignal != null) {
                    terminalSignal.signalTerminal(new VictimTerminal(this.requestId()));
                }
            }
        };
    }

    private TerminalAction beginSettledPriorityTerminalLocked(
            String detail,
            Runnable counterpartCleanup) {
        if (this.hasCancellationFirstCause()) {
            CancelReason firstCause = this.requireCancellationFirstCause();
            String cancellationDetail = cancelDetail(firstCause);
            return beginWorkerStatusTerminalLocked(counterpartCleanup,
                    TerminalOutcome.cancellation(firstCause, cancellationDetail),
                    buildErrorResponse(
                            this.cancellationErrorType(firstCause),
                            cancellationDetail));
        }
        return beginWorkerStatusTerminalLocked(counterpartCleanup,
                TerminalOutcome.cancel(detail),
                buildErrorResponse(
                        StrategyErrorType.PRIORITY_PREEMPTED, detail));
    }

    private TerminalAction beginExpiredRequestLocked( String detail) {
        ScheduledRequest item = this.activeItem();
        CancelReason firstCause = this.requireCancellationFirstCause();
        return beginTerminalLocked(true, false, false,
                item == null ? null : () -> RequestTerminalCleanup.expireEndpointAccounting(item),
                TerminalOutcome.cancellation(firstCause, detail),
                buildErrorResponse(this.cancellationErrorType(firstCause), detail));
    }

    private TerminalAction settleCancellationFromWorkerStatusLocked(
            String proof,
            WorkerTerminalSource source) {
        return settleCancellationAfterEndpointSettlementLocked(proof,
                workerStatusCounterpartCleanup(source));
    }

    private TerminalAction settleCancellationAfterEndpointSettlementLocked(
            String proof,
            Runnable counterpartCleanup) {
        CancelReason reason = this.requireCancellationFirstCause();
        String detail = cancelDetail(reason) + "; " + proof;
        return beginWorkerStatusTerminalLocked(counterpartCleanup,
                TerminalOutcome.cancellation(reason, detail),
                buildErrorResponse(
                        this.cancellationErrorType(reason), detail));
    }

    private static String cancelDetail(CancelReason reason) {
        return reason == CancelReason.DEADLINE_EXCEEDED
                ? "request deadline exceeded"
                : "request cancelled by client";
    }

    private Runnable reduceDeferredTerminalFactLocked(
            DeferredTerminal terminal) {
        ScheduledRequest item = this.activeItem();
        if (item == null) {
            return null;
        }
        RequestSlot.PreemptionReduction reduction;
        if (terminal.kind() == DeferredTerminal.Kind.DELIVERY_REJECTED
                && item.decodeEp() != null
                && item.decodeReservation() != null) {
            reduction = this.reduceDispatchRejected(
                    item.decodeEp(), item.decodeReservation(), item, terminal);
        } else {
            reduction = terminal.authoritativeWorker()
                    ? this.reduceWorkerTerminal(
                            item, terminal)
                    : this.reduceOrdinaryTerminal(
                            item, terminal);
        }
        return materializePostLockActionLocked(reduction, null);
    }

    private Runnable applyOrdinaryTerminalLocked(
            DeferredTerminal terminal) {
        if (terminal.endpointAlreadyRetired()) {
            return applyDecodeSettledTerminalLocked(terminal);
        }
        return switch (terminal.kind()) {
            case FAILURE -> applyFailureTerminalLocked(terminal, true);
            case TIMEOUT -> applyTimeoutTerminalLocked(terminal);
            case DELIVERY_FAILURE ->
                    applyFailureTerminalLocked(terminal, true);
            case DELIVERY_REJECTED ->
                    applyDecodeSettledTerminalLocked(terminal);
            case WORKER -> applyWorkerTerminalLocked(terminal);
            case PRIORITY ->
                    throw new IllegalStateException(
                            "priority terminal requires its typed reducer");
            case DECODE_GENERATION_RETIRED ->
                    throw new IllegalStateException(
                            "retired Decode generation was not marked retired");
        };
    }

    private Runnable applyTimeoutTerminalLocked(
            DeferredTerminal timeout) {
        return terminalPublication(beginTerminalLocked(true,
                true,
                TerminalOutcome.timeout(timeout.detail()),
                buildErrorResponse(
                        this.timeoutErrorType(), timeout.detail())));
    }

    private Runnable applyFailureTerminalLocked(
            DeferredTerminal failure,
            boolean releaseDecode) {
        return terminalPublication(beginTerminalLocked(true, releaseDecode,
                TerminalOutcome.fail(failure.detail()),
                buildErrorResponse(failure.errorType(), failure.detail())));
    }

    private Runnable applyDecodeSettledTerminalLocked(
            DeferredTerminal terminal) {
        String terminalDetail;
        if (terminal.kind() == DeferredTerminal.Kind.DELIVERY_REJECTED
                || terminal.kind()
                    == DeferredTerminal.Kind.DECODE_GENERATION_RETIRED) {
            terminalDetail = terminal.detail();
        } else {
            throw new IllegalArgumentException(
                    "Decode-settled reducer requires a Decode terminal");
        }
        String detail = terminalDetail == null
                ? "Decode endpoint generation retired"
                : terminalDetail;
        if (this.hasCancellationFirstCause()) {
            CancelReason firstCause = this.requireCancellationFirstCause();
            String cancellationDetail = cancelDetail(firstCause)
                    + "; " + detail;
            return terminalPublication(beginTerminalLocked(false,
                    true,
                    TerminalOutcome.cancellation(firstCause, cancellationDetail),
                    buildErrorResponse(
                            this.cancellationErrorType(firstCause),
                            cancellationDetail)));
        }
        return terminalPublication(beginTerminalLocked(false,
                true,
                TerminalOutcome.fail(detail),
                buildErrorResponse(
                        StrategyErrorType.DISPATCH_FAILED, detail)));
    }

    private Runnable applyWorkerTerminalLocked(
            DeferredTerminal terminal) {
        if (this.hasCancellationFirstCause()) {
            String proof = terminal.workerSource()
                    == WorkerTerminalSource.PREFILL_BACKED
                            ? "Prefill terminal observed after cancellation"
                            : "Decode terminal observed after cancellation";
            return terminalPublication(
                    settleCancellationFromWorkerStatusLocked(proof, terminal.workerSource()));
        }
        TerminalOutcome transition;
        Response response;
        if (terminal.workerSuccessful()) {
            transition = TerminalOutcome.complete("decode completed");
            ScheduledRequest item = this.activeItem();
            response = buildSuccessResponse(
                    item, this.snapshot().deliveryClaimKind());
        } else {
            String detail = "worker error code "
                    + terminal.workerErrorCode();
            transition = TerminalOutcome.fail(detail);
            response = buildErrorResponse(
                    StrategyErrorType.WORKER_EXECUTION_FAILED, detail);
        }
        return terminalPublication(beginWorkerStatusTerminalLocked(workerStatusCounterpartCleanup(terminal.workerSource()),
                transition,
                response));
    }

    private boolean ownsPreparedDelivery( ScheduledRequest item) {
        RequestState snapshot = this.snapshot();
        return this.ownsActiveItem(item)
                && this.isOpen()
                && this.canClaimDelivery()
                && snapshot.state() == RequestState.Phase.QUEUED
                && snapshot.deliveryClaimKind() == DeliveryClaimKind.NONE;
    }

    private Runnable confirmRouteDecisionLocked(
            ScheduledRequest item) {
        if (!this.ownsDeliveryClaim(
                item, DeliveryClaimKind.ROUTE_DECISION, 0L)) {
            return null;
        }
        return materializePostLockActionLocked(this.reduceDeliveryConfirmed(0L),
                null);
    }

    private Runnable confirmBatchEnqueueLocked(
            ScheduledRequest item) {
        RequestState current = this.snapshot();
        long batchId = current.batchId();
        if (!this.ownsDeliveryClaim(
                item, DeliveryClaimKind.BATCH_ENQUEUE, batchId)) {
            Logger.debug("Ignoring EnqueueBatch ACK without a batch claim request_id={}",
                    item.requestId());
            return null;
        }
        item.ctx().setAckAtMs(System.currentTimeMillis());
        item.ctx().setAckAtNanos(System.nanoTime());
        return materializePostLockActionLocked(this.reduceDeliveryConfirmed(batchId),
                null);
    }

    private Runnable deliveryPublication(
            ScheduledRequest item,
            RequestSlot.DeliveryConfirmation confirmation,
            DeliveryClaimKind deliveryKind,
            long batchId) {
        Response response = buildSuccessResponse(
                item, deliveryKind);
        return () -> completionPublisher.publishDelivery(
                this, item, response, confirmation, deliveryKind);
    }

    private TerminalAction beginTerminalLocked(
            boolean releaseDecode,
            boolean releasePrefill,
            TerminalOutcome transition,
            Response response) {
        return beginTerminalLocked(true, releaseDecode, releasePrefill, null,
                transition, response);
    }

    private TerminalAction beginTerminalLocked(
            boolean removePrefillQueue,
            boolean releaseDecode,
            boolean releasePrefill,
            Runnable counterpartCleanup,
            TerminalOutcome transition,
            Response response) {
        return this.beginTerminalizing(
                removePrefillQueue,
                releaseDecode,
                releasePrefill,
                counterpartCleanup,
                transition,
                response);
    }

    private TerminalAction beginWorkerStatusTerminalLocked(
            Runnable counterpartCleanup,
            TerminalOutcome transition,
            Response response) {
        return beginTerminalLocked(false, false, false, counterpartCleanup,
                transition, response);
    }

    private Runnable terminalPublication(TerminalAction action) {
        return action == null ? null : () -> terminalCleanup.submitTerminal(action);
    }

    void runPostLock(Runnable action) {
        if (action == null) {
            return;
        }
        action.run();
    }

    private void armDecisionDeadline() {
        java.util.OptionalLong deadline;
        synchronized (this) {
            deadline = this.decisionDeadlineAtMs();
        }
        if (deadline.isPresent()) {
            expirationTimer.registerDecisionDeadline(
                    this, deadline.getAsLong());
        }
    }

    void cancelDecisionDeadline(
            ExpirationTimer.DecisionDeadline cleanup) {
        if (cleanup == null) {
            return;
        }
        try {
            expirationTimer.release(cleanup);
        } catch (Throwable failure) {
            Logger.error("Admission cleanup isolated", failure);
        }
    }

    private static String detailOf(Throwable cause) {
        if (cause == null) {
            return "unknown delivery failure";
        }
        String message = cause.getMessage();
        return message == null || message.isBlank()
                ? cause.getClass().getSimpleName() : message;
    }

    private Runnable workerStatusCounterpartCleanup(
            WorkerTerminalSource source) {
        ScheduledRequest item = this.activeItem();
        if (item == null) {
            return null;
        }
        if (source == WorkerTerminalSource.PREFILL_BACKED) {
            DecodeEndpoint decode = item.decodeEp();
            DecodeEndpoint.ReservationHandle reservation =
                    item.decodeReservation();
            return decode == null || reservation == null
                    ? null
                    : () -> decode.releaseLocalShadowIfExact(reservation);
        }
        if (this.snapshot().deliveryClaimKind()
                != DeliveryClaimKind.ROUTE_DECISION) {
            return null;
        }
        return exactPrefillCounterpartCleanup(item);
    }

    private static Runnable exactPrefillCounterpartCleanup(ScheduledRequest item) {
        PrefillEndpoint prefill = item == null ? null : item.prefillEp();
        return prefill == null
                ? null : () -> prefill.releaseCommittedItem(item);
    }

    private TerminalAction claimExternalLocalTerminal(
            TerminalOutcome transition) {
        synchronized (this) {
            if (!this.isCurrentGeneration() || !this.canClaimLocalTerminal()) {
                return null;
            }
            return this.beginExternalTerminalizing(transition);
        }
    }

    /** Reduce one exact preemption event before executing detached effects. */
    private boolean onPreemption(PreemptionRegistration claim, boolean cleanCounterpart,
                         Function<RequestSlot, PreemptionReduction> reduction) {
        Runnable work;
        synchronized (this) {
            Runnable cleanup = cleanCounterpart ? exactPrefillCounterpartCleanup(activeItem()) : null;
            work = materializePostLockActionLocked(reduction.apply(this), cleanup);
        }
        runPostLock(work);
        return work != null;
    }

    private static boolean batchMatches(RequestState snapshot, long expected) {
        return snapshot != null && (expected == 0 || snapshot.batchId() == expected);
    }

    private RequestState applyTerminalOutcome(TerminalOutcome outcome) {
        return switch (outcome.phase()) {
            case COMPLETED -> complete(outcome.detail());
            case FAILED -> fail(outcome.detail());
            case CANCELLED -> cancel(outcome.detail());
            case TIMED_OUT -> timeout(outcome.detail());
            default -> throw new IllegalArgumentException("not a terminal outcome: " + outcome.phase());
        };
    }

    private enum SlotPhase {
        ACTIVE,
        TERMINALIZING,
        TOMBSTONE
    }

    private enum EngineOwnership {
        DECODE_PENDING,
        DECODE_OWNED
    }

    enum PublicationKind {
        DELIVERY,
        TERMINAL
    }

    /** Immutable replay already selected under the exact slot lock. */
    record PendingReplay(
            DeferredTerminal terminal,
            DeliveryConfirmation confirmation,
            ScheduledRequest item,
            DeliveryClaimKind kind,
            long batchId) {

        static PendingReplay terminal(DeferredTerminal terminal) {
            return new PendingReplay(
                    Objects.requireNonNull(terminal), null, null, null, 0L);
        }

        static PendingReplay delivery(
                DeliveryConfirmation confirmation,
                ScheduledRequest item,
                DeliveryClaimKind kind,
                long batchId) {
            return new PendingReplay(
                    null, confirmation, item, kind, batchId);
        }
    }

    /** The only effect exposed after a preemption ownership reduction. */
    record PreemptionReduction(
            Status status,
            PendingReplay replay,
            PreemptionRegistration signal) {

        static final PreemptionReduction STALE = new PreemptionReduction(Status.STALE, null, null);
        static final PreemptionReduction NONE = new PreemptionReduction(Status.NONE, null, null);

        PreemptionReduction {
            Objects.requireNonNull(status, "status");
            boolean replays = status == Status.REPLAY;
            if (replays != (replay != null) || (!replays && signal != null)) {
                throw new IllegalArgumentException(
                        "preemption reduction status requires its exact payload");
            }
        }

        static PreemptionReduction replay(PendingReplay replay, PreemptionRegistration signal) {
            return new PreemptionReduction(Status.REPLAY, replay, signal);
        }

        enum Status { STALE, NONE, REPLAY }
    }

    record AdmissionMutationCompletion(
            boolean owned,
            CancelReason cancellationToResume,
            DeferredTerminal pendingTerminal,
            TerminalAction pendingRetirement,
            boolean inactivityExpired) {
        private static final AdmissionMutationCompletion NOT_OWNED =
                new AdmissionMutationCompletion(
                        false, null, null, null, false);
    }

    record DecisionExpiry(
            ScheduledRequest item,
            boolean needsConfirmation) {
    }

    record DeliveryConfirmation(
            PublicationPermit publication,
            RequestDeadline requestDeadline,
            long batchEnqueueStartedAtMs) {
    }

    /** Exact terminal cleanup detached atomically at ACTIVE -> TERMINALIZING. */
    static final class TerminalResources {
        private final RequestDeadline requestDeadline;
        private final InactivityDeadline inactivityDeadline;
        private final DecisionDeadline detachedDecisionDeadline;
        private boolean released;

        private TerminalResources(
                RequestDeadline requestDeadline,
                DecisionDeadline detachedDecisionDeadline, InactivityDeadline inactivityDeadline) {
            this.requestDeadline = requestDeadline;
            this.inactivityDeadline = inactivityDeadline;
            this.detachedDecisionDeadline = detachedDecisionDeadline;
        }

        synchronized void release(ExpirationTimer timer) {
            if (released) {
                return;
            }
            released = true;
            Throwable failure = null;
            if (inactivityDeadline != null) {
                try { timer.cancel(inactivityDeadline); }
                catch (Throwable timerFailure) { failure = timerFailure; }
            }
            if (requestDeadline != null) {
                try {
                    timer.cancel(requestDeadline);
                } catch (Throwable timerFailure) {
                    failure = appendFailure(failure, timerFailure);
                }
            }
            if (detachedDecisionDeadline != null) {
                try {
                    timer.cancel(detachedDecisionDeadline);
                } catch (Throwable admissionFailure) {
                    failure = appendFailure(failure, admissionFailure);
                }
            }
            rethrowCleanup(failure);
        }
    }

    /**
     * Invocation-local proof that one exact lifecycle edge owns one frontend
     * publication. The capability is never stored in a slot or registry.
     */
    static final class PublicationPermit {
        private final RequestCompletionPublisher publisher;
        private final RequestSlot slot;
        private final RequestFuture future;
        private final PublicationKind kind;
        private final AtomicBoolean claimed = new AtomicBoolean();
        private final AtomicBoolean closed = new AtomicBoolean();

        PublicationPermit(
                RequestCompletionPublisher publisher,
                RequestSlot slot,
                PublicationKind kind) {
            this.publisher = Objects.requireNonNull(publisher, "publisher");
            this.slot = slot;
            this.future = slot.future;
            this.kind = kind;
        }

        RequestSlot slot() {
            return slot;
        }

        boolean ownedBy(RequestCompletionPublisher expected) {
            return publisher == expected;
        }

        void closePublication() {
            if (closed.compareAndSet(false, true)) {
                publisher.exitPublication();
            }
        }

        BooleanSupplier claimDeliveryResponse(Response response) {
            requireDelivery("delivery response");
            claim();
            return claimResult() ? () -> future.completeOwned(response) : () -> false;
        }

        BooleanSupplier claimTerminalResponse(Response response) {
            requireTerminal("external response");
            claim();
            return claimResult() ? () -> future.completeOwned(response) : () -> false;
        }

        BooleanSupplier claimFailure(Throwable failure) {
            requireTerminal("failure");
            claim();
            return claimResult() ? () -> future.completeExceptionallyOwned(failure) : () -> false;
        }

        BooleanSupplier claimCancellation(boolean mayInterruptIfRunning) {
            requireTerminal("cancellation");
            claim();
            return claimResult() ? () -> future.cancelOwned(mayInterruptIfRunning) : () -> false;
        }

        private boolean claimResult() {
            synchronized (slot) {
                return slot.claimPublicationResult(kind);
            }
        }

        /** Abandon a permit only when no other submitter consumed it. */
        void abandonIfUnclaimed() {
            if (claimed.compareAndSet(false, true)) {
                closePublication();
            }
        }

        /** Settle a claim whose publication could not enter its executor. */
        void abortClaimedPublication() {
            closePublication();
        }

        private void requireTerminal(String operation) {
            if (kind != PublicationKind.TERMINAL) {
                throw new IllegalStateException(
                        operation
                                + " publication requires a terminal permit");
            }
        }

        private void requireDelivery(String operation) {
            if (kind != PublicationKind.DELIVERY) {
                throw new IllegalStateException(
                        operation
                                + " publication requires a delivery permit");
            }
        }

        private void claim() {
            if (!claimed.compareAndSet(false, true)) {
                throw new IllegalStateException(
                        "publication permit already consumed for request "
                                + slot.requestId);
            }
        }
    }

    private static void rethrowCleanup(Throwable failure) {
        if (failure instanceof RuntimeException runtime) {
            throw runtime;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        if (failure != null) {
            throw new IllegalStateException(
                    "request slot cleanup failed", failure);
        }
    }
}

enum WorkerTerminalSource {
    PREFILL_BACKED(false),
    DECODE_ENDPOINT_SETTLED(true);

    private final boolean decodeSettlementCommitted;

    WorkerTerminalSource(boolean decodeSettlementCommitted) {
        this.decodeSettlementCommitted = decodeSettlementCommitted;
    }

    boolean decodeSettlementCommitted() {
        return decodeSettlementCommitted;
    }
}

/** Non-persistent decision produced by the RequestSlot acceptance transition. */
record DecodeAcceptance(
        ExpirationTimer.DecisionDeadline detachedDecisionDeadline) {
    static final DecodeAcceptance NONE =
            new DecodeAcceptance(null);
}

/** First ordinary terminal observed while priority Cancel owns the slot. */
record DeferredTerminal(
        Kind kind,
        StrategyErrorType errorType,
        String detail,
        WorkerTerminalSource workerSource,
        boolean workerSuccessful,
        long workerErrorCode) {

    enum Kind {
        FAILURE,
        TIMEOUT,
        DELIVERY_FAILURE,
        DELIVERY_REJECTED,
        WORKER,
        PRIORITY,
        DECODE_GENERATION_RETIRED
    }

    DeferredTerminal {
        Objects.requireNonNull(kind, "kind");
        boolean valid = switch (kind) {
            case FAILURE, DELIVERY_FAILURE ->
                    errorType != null && workerSource == null;
            case WORKER -> errorType == null && workerSource != null;
            case TIMEOUT, DELIVERY_REJECTED, PRIORITY,
                    DECODE_GENERATION_RETIRED ->
                    errorType == null && workerSource == null;
        };
        if (!valid) {
            throw new IllegalArgumentException(
                    "deferred terminal kind requires its exact payload");
        }
    }

    static DeferredTerminal failure(
            StrategyErrorType errorType, String detail) {
        return new DeferredTerminal(
                Kind.FAILURE, errorType, detail, null, false, 0L);
    }

    static DeferredTerminal timeout(String detail) {
        return new DeferredTerminal(
                Kind.TIMEOUT, null, detail, null, false, 0L);
    }

    static DeferredTerminal deliveryFailure(
            StrategyErrorType errorType, String detail) {
        return new DeferredTerminal(
                Kind.DELIVERY_FAILURE, errorType, detail, null, false, 0L);
    }

    static DeferredTerminal deliveryRejected(String detail) {
        return new DeferredTerminal(
                Kind.DELIVERY_REJECTED, null, detail, null, false, 0L);
    }

    static DeferredTerminal worker(
            WorkerTerminalSource source,
            boolean successful,
            long errorCode) {
        return new DeferredTerminal(
                Kind.WORKER, null, null,
                Objects.requireNonNull(source, "source"), successful, errorCode);
    }

    static DeferredTerminal priority(String detail) {
        return new DeferredTerminal(
                Kind.PRIORITY, null, detail, null, false, 0L);
    }

    static DeferredTerminal decodeGenerationRetired(String detail) {
        return new DeferredTerminal(
                Kind.DECODE_GENERATION_RETIRED, null, detail,
                null, false, 0L);
    }

    boolean authoritativeWorker() {
        return kind == Kind.WORKER
                || kind == Kind.DECODE_GENERATION_RETIRED;
    }

    boolean endpointAlreadyRetired() {
        return kind == Kind.DECODE_GENERATION_RETIRED;
    }

    boolean decodeSettlementCommitted() {
        return kind == Kind.WORKER
                && workerSource.decodeSettlementCommitted();
    }

    boolean deliveryFailure() {
        return kind == Kind.DELIVERY_FAILURE;
    }
}

/** One-shot capability moved out of an ACTIVE slot; never stored or retried. */
record TerminalAction(
        RequestSlot slot,
        ScheduledRequest item,
        PreemptionRegistration preemption,
        RequestSlot.TerminalResources terminalResources,
        boolean removePrefillQueue,
        boolean releaseDecode,
        boolean releasePrefill,
        Runnable counterpartCleanup,
        String queueReason,
        TerminalOutcome transition,
        Response response,
        RequestSlot.PublicationPermit publication) {
}

/** Non-persistent proof that a claimed terminal action reached its tombstone. */
record TombstoneResult(
        RequestState terminal,
        Throwable transitionFailure,
        RequestSlot.PublicationPermit publication) {
}

/** Stateless public-future adapter bound to one exact canonical slot. */
final class RequestFuture extends CompletableFuture<Response> {
    private final RequestCompletionPublisher publisher;
    private final RequestSlot slot;

    RequestFuture(
            RequestCompletionPublisher publisher,
            RequestSlot slot) {
        this.publisher = publisher;
        this.slot = slot;
    }

    @Override
    public boolean complete(Response response) {
        return publisher.publishResponse(slot, response);
    }

    @Override
    public boolean completeExceptionally(Throwable error) {
        return publisher.publishFailure(slot, error);
    }

    @Override
    public boolean cancel(boolean mayInterruptIfRunning) {
        return publisher.publishCancellation(slot, mayInterruptIfRunning);
    }

    boolean completeOwned(Response response) {
        return super.complete(response);
    }

    boolean completeExceptionallyOwned(Throwable error) {
        return super.completeExceptionally(error);
    }

    boolean cancelOwned(boolean mayInterruptIfRunning) {
        return super.cancel(mayInterruptIfRunning);
    }
}
