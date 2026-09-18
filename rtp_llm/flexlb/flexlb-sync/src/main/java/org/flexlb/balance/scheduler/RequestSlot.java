package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.balance.preemption.VictimTerminal;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.ExpirationTimer.DecisionDeadline;
import org.flexlb.balance.scheduler.ExpirationTimer.InactivityDeadline;
import org.flexlb.balance.scheduler.ExpirationTimer.RequestDeadline;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;

import java.util.Objects;
import java.util.OptionalLong;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BooleanSupplier;

import static org.flexlb.dao.loadbalance.Response.buildErrorResponse;
import static org.flexlb.dao.loadbalance.Response.buildSuccessResponse;

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

    /**
     * One-shot handle for one admission attempt on this slot.
     * Its identity prevents an earlier attempt from finishing a later one.
     */
    public static final class AdmissionHandle implements AutoCloseable {

        private final AtomicBoolean resolved = new AtomicBoolean();
        private final RequestSlot owner;

        AdmissionHandle(RequestSlot owner) {
            this.owner = Objects.requireNonNull(owner, "owner");
        }

        /** Transfer this exact admission attempt to canonical terminal ownership. */
        public void terminate(Response failure) {
            if (resolved.compareAndSet(false, true)) {
                owner.terminateAdmission(this, failure);
            }
        }

        /** Finish this admission attempt after success or without committed side effects. */
        @Override
        public void close() {
            if (resolved.compareAndSet(false, true)) {
                owner.finishAdmission(this);
            }
        }
    }

    /** Exact asynchronous delivery identity. All consumption is guarded by the owning slot. */
    public static final class DeliveryClaim {
        final RequestSlot slot;
        private final ScheduledRequest item;
        private final DeliveryClaimKind kind;
        private final long correlationId;
        private boolean completed;

        private DeliveryClaim(RequestSlot slot, ScheduledRequest item, DeliveryClaimKind kind, long correlationId) {
            this.slot = slot;
            this.item = item;
            this.kind = kind;
            this.correlationId = correlationId;
        }

        public void complete(DeliveryResult result) {
            slot.completeDelivery(this, result);
        }
    }

    private final BatchSchedulerReporter reporter;
    private final ExpirationTimer expirationTimer;
    private final RequestTerminalCleanup terminalCleanup;
    private final Runnable admissionFinished;

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

    /** Routing identity survives handoff and is used to reject late facts. */
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
    private AdmissionHandle admissionHandle;
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
            RequestTerminalCleanup terminalCleanup, Runnable admissionFinished, BatchSchedulerReporter reporter) {
        this.reporter = reporter;
        this.expirationTimer = expirationTimer;
        this.terminalCleanup = terminalCleanup;
        this.admissionFinished = admissionFinished;
        this.completionPublisher = Objects.requireNonNull(
                completionPublisher, "completionPublisher");
        this.requestId = requestId;
        this.createdAtMs = System.currentTimeMillis();
        this.updatedAtMs = createdAtMs;
        this.lastWorkerStatusAtMs = createdAtMs;
        this.future = new RequestFuture(this);
    }

    // Event entry points: each owns the complete per-request decision.

    /** Return the resulting request snapshot; accepting cancellation does not imply immediate cleanup. */
    public RequestState cancelRequest(long expectedBatchId, CancelReason reason) {
        Objects.requireNonNull(reason, "reason");
        TerminalAction action = null;
        RequestState result;
        synchronized (this) {
            if (!isCurrentGeneration() || !snapshot().matchesBatch(expectedBatchId)) { return null; }
            if (reason == CancelReason.DEADLINE_EXCEEDED && deliveryClaimKind != DeliveryClaimKind.NONE) {
                return snapshot();
            }
            if (recordCancellationLocked(reason, reason.getMessage())) {
                action = tryTerminateCancellationLocked();
            }
            result = snapshot();
        }
        terminalCleanup.submitTerminal(action);
        return result;
    }

    /** Record the first cancellation only. Resource ownership is unchanged. */
    private boolean recordCancellationLocked(CancelReason reason, String message) {
        requireSlotLock("record cancellation");
        Objects.requireNonNull(reason, "reason");
        if (!ownsActiveGeneration() || state.isTerminal()
                || cancellationReason != null || pendingAdmissionCancelReason != null) {
            return false;
        }
        if (admissionHandle != null) {
            pendingAdmissionCancelReason = reason;
        } else {
            cancellationReason = reason;
        }
        admissionOpen = false;
        transition(RequestState.Phase.CANCEL_REQUESTED, message);
        assertInvariant();
        return true;
    }

    /** Claim terminal ownership only when a recorded cancellation is locally reversible. */
    private TerminalAction tryTerminateCancellationLocked() {
        requireSlotLock("local cancellation termination");
        if (!ownsActiveGeneration() || admissionHandle != null || cancellationReason == null) { return null; }
        ScheduledRequest active = activeItem();
        if (active != null && !canClaimLocalTerminal()) { return null; }
        String message = cancellationReason.getMessage();
        return beginTerminalizing(
                TerminalOutcome.cancellation(cancellationReason, message),
                buildErrorResponse(cancellationErrorType(cancellationReason), message));
    }

    void expire(RequestDeadline exact) {
        TerminalAction action = null;
        synchronized (this) {
            if (requestDeadline != exact) { return; }
            requestDeadline = null;
            if (!ownsActiveGeneration() || future.isDone() || !isOpen()) {
                assertInvariant();
                return;
            }
            admissionOpen = false;
            if (admissionHandle != null) {
                recordCancellationLocked(CancelReason.DEADLINE_EXCEEDED,
                        "request scheduling deadline exceeded during admission");
            } else if (deliveryClaimKind == DeliveryClaimKind.NONE
                    && recordCancellationLocked(CancelReason.DEADLINE_EXCEEDED,
                            CancelReason.DEADLINE_EXCEEDED.getMessage())) {
                action = tryTerminateCancellationLocked();
            }
            assertInvariant();
        }
        terminalCleanup.submitTerminal(action);
    }

    void expire(InactivityDeadline exact, long nowMs) {
        TerminalAction action;
        synchronized (this) {
            if (!consumeInactivityDeadline(exact)) { return; }
            action = decideInactivityLocked(nowMs);
        }
        terminalCleanup.submitTerminal(action);
    }

    void expireInactiveRequest(long nowMs) {
        TerminalAction action;
        synchronized (this) { action = decideInactivityLocked(nowMs); }
        terminalCleanup.submitTerminal(action);
    }

    private TerminalAction decideInactivityLocked(long nowMs) {
        if (!isCurrentGeneration() || !ownsActiveGeneration() || state.isTerminal() || !requestInactive(nowMs)) {
            return null;
        }
        String message = "REQUEST_INACTIVE: no matching Engine request status before inactivity timeout";
        recordCancellationLocked(CancelReason.DEADLINE_EXCEEDED, message);
        if (admissionHandle != null) {
            pendingAdmissionInactivityExpired = true;
            assertInvariant();
            return null;
        }
        return beginExpiredRequestLocked(message);
    }

    void processPrefillStatus(PrefillEndpoint source, RoleType role, PrefillState.WorkerStatusFact fact) {
        EngineObservation observation;
        RequestEffect work;
        synchronized (this) {
            if (!isCurrentGeneration()) { return; }
            observation = applyPrefillStatusLocked(source, role, fact, System.currentTimeMillis());
            work = observation.transition();
        }
        executeEngineObservationEffects(observation, work);
    }

    void processDecodeStatus(DecodeEndpoint source, DecodeEndpoint.WorkerStatusFact fact) {
        EngineObservation observation;
        RequestEffect work;
        synchronized (this) {
            if (!isCurrentGeneration()) { return; }
            observation = applyDecodeStatusLocked(source, fact, System.currentTimeMillis());
            work = observation.transition();
        }
        executeEngineObservationEffects(observation, work);
    }

    private void executeEngineObservationEffects(EngineObservation observation, RequestEffect work) {
        try { cancelDecisionDeadline(observation.obsoleteDeadline()); }
        finally {
            try { armDecisionDeadline(); }
            finally { execute(work); }
        }
    }

    void recordSchedulingFailure(StrategyErrorType error, String detail) {
        RequestEffect work;
        synchronized (this) {
            work = reduceDeferredTerminalFactLocked(DeferredTerminal.failure(error, detail));
        }
        execute(work);
    }

    void recordPrefillRetirement(PrefillEndpoint source, ScheduledRequest exact) {
        String detail = "Prefill endpoint generation retired: " + source.ipPort()
                + "#" + source.getStatus().getGenerationId();
        TerminalAction action;
        synchronized (this) {
            action = beginPrefillRetirementTerminal(source, exact,
                    TerminalOutcome.fail(detail), buildErrorResponse(StrategyErrorType.DISPATCH_FAILED, detail));
        }
        terminalCleanup.submitTerminal(action);
    }

    void recordDecodeRetirement(DecodeEndpoint source, DecodeEndpoint.ReservationHandle exact) {
        RequestEffect work;
        synchronized (this) {
            work = reduceDecodeGenerationRetired(source, exact,
                            "Decode endpoint generation retired: generation=" + exact.endpointGenerationId());
        }
        execute(work);
    }

    /** End the exact admission transaction and process events retained while it was open. */
    void finishAdmission(AdmissionHandle exact) {
        try {
            RequestEffect effect;
            synchronized (this) {
                AdmissionHandleCompletion completion = completeAdmissionHandle(exact);
                if (!completion.owned()) { return; }
                effect = processAdmissionResultLocked(completion, null);
            }
            execute(effect);
        } finally {
            releaseAdmissionGate();
        }
    }

    /** End an aborted admission; retained cancellation or Worker proof determines the request outcome first. */
    void terminateAdmission(AdmissionHandle exact, Response failure) {
        try {
            if (failure.isSuccess()) {
                throw new IllegalArgumentException("admission termination requires a failure response");
            }
            RequestEffect effect;
            synchronized (this) {
                effect = processAdmissionResultLocked(claimAdmissionHandleTermination(exact), failure);
            }
            execute(effect);
        } finally {
            releaseAdmissionGate();
        }
    }

    private void releaseAdmissionGate() {
        try { expirationTimer.attachInactivityDeadline(this); }
        finally { admissionFinished.run(); }
    }

    /** Decide retained evidence once, then reconsider a cancellation that has no terminal proof yet. */
    private RequestEffect processAdmissionResultLocked(AdmissionHandleCompletion completion, Response failure) {
        requireSlotLock("admission result processing");
        RequestEffect effect = null;
        if (completion.pendingTerminal() != null) {
            effect = reduceDeferredTerminalFactLocked(completion.pendingTerminal());
        } else if (completion.pendingRetirement() != null) {
            effect = RequestEffect.terminal(completion.pendingRetirement(), null);
        } else if (failure != null) {
            CancelReason cancellation = completion.cancellationReason();
            String message = cancellation != null ? cancellation.getMessage()
                    : failure.getErrorMessage() == null ? "eviction admission failed" : failure.getErrorMessage();
            TerminalOutcome outcome = cancellation == null ? TerminalOutcome.fail(message)
                    : TerminalOutcome.cancellation(cancellation, message);
            Response response = cancellation == null ? failure
                    : buildErrorResponse(cancellationErrorType(cancellation), message);
            effect = RequestEffect.terminal(beginTerminalizing(outcome, response), null);
        }
        if (effect != null && effect.status() == RequestEffect.Status.READY) { return effect; }
        if (completion.cancellationReason() == null || !ownsActiveGeneration()) { return effect; }
        CancelReason firstCause = requireCancellationFirstCause();
        if (firstCause != completion.cancellationReason()) {
            throw new IllegalStateException("admission cancellation first cause changed for request " + requestId);
        }
        TerminalAction action;
        if (completion.inactivityExpired() || firstCause == CancelReason.DEADLINE_EXCEEDED
                || requestInactive(System.currentTimeMillis())) {
            String message = completion.inactivityExpired()
                    ? "REQUEST_INACTIVE: no matching Engine request status before inactivity timeout"
                    : firstCause.getMessage();
            action = beginExpiredRequestLocked(message);
        } else {
            action = tryTerminateCancellationLocked();
        }
        return action == null ? effect : RequestEffect.terminal(action, null);
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

    CapacityBoundary.Attempt<BatchDeliveryStrategy.BatchTransaction> prepareBatchDelivery(
            ScheduledRequest exact, BatchDeliveryStrategy strategy) {
        synchronized (this) {
            return ownsPreparedDelivery(exact) ? strategy.prepareAdmission(exact)
                    : CapacityBoundary.Attempt.rejected(CapacityBoundary.OWNERSHIP_LOST);
        }
    }

    CapacityBoundary.Attempt<ScheduledRequest> prepareBatchMember(
            ScheduledRequest exact, BatchDeliveryStrategy.BatchTransaction transaction) {
        synchronized (this) {
            return ownsPreparedDelivery(exact) ? transaction.append(exact)
                    : CapacityBoundary.Attempt.rejected(CapacityBoundary.OWNERSHIP_LOST);
        }
    }

    CapacityBoundary.Attempt<ScheduledRequest> prepareRouteMember(
            ScheduledRequest exact, RouteDeliveryStrategy.RouteTransaction transaction,
            PrefillTimePredictor.Evaluator evaluator) {
        synchronized (this) {
            return ownsPreparedDelivery(exact) ? transaction.append(exact, evaluator)
                    : CapacityBoundary.Attempt.rejected(CapacityBoundary.OWNERSHIP_LOST);
        }
    }

    DeliveryClaim claimBatchDelivery(ScheduledRequest exact, BatchDeliveryStrategy.BatchTransaction transaction) {
        synchronized (this) {
            if (transaction.batchId() <= 0L) { throw new IllegalArgumentException("batchId must be positive"); }
            if (!ownsPreparedDelivery(exact)) { return null; }
            ensureTransitionAllowed(RequestState.Phase.DISPATCHING);
            DeliveryClaim claim = new DeliveryClaim(this, exact, DeliveryClaimKind.BATCH_ENQUEUE, transaction.batchId());
            if (!transaction.transferToEndpoint(exact)) {
                throw new IllegalStateException("endpoint ownership lost for request " + requestId);
            }
            deliveryClaimKind = claim.kind;
            batchId = claim.correlationId;
            batchEnqueueStartedAtMs = System.currentTimeMillis();
            transition(RequestState.Phase.DISPATCHING, "batch enqueue started");
            return claim;
        }
    }

    DeliveryClaim claimRouteDelivery(ScheduledRequest exact, PrefillAdmissionResources.CommittedAdmissionOwner admission) {
        synchronized (this) {
            if (!ownsPreparedDelivery(exact)) { return null; }
            ensureTransitionAllowed(RequestState.Phase.DISPATCHING);
            DeliveryClaim claim = new DeliveryClaim(this, exact, DeliveryClaimKind.ROUTE_DECISION, 0L);
            if (!admission.transferToEndpoint(exact)) {
                throw new IllegalStateException("endpoint ownership lost for request " + requestId);
            }
            deliveryClaimKind = claim.kind;
            transition(RequestState.Phase.DISPATCHING, "route decision delivery started");
            return claim;
        }
    }

    void setDeliveryPrediction(DeliveryClaim claim, WorkSnapshot work, long predictedMs) {
        DecisionDeadline obsolete;
        synchronized (this) {
            if (!ownsDeliveryClaim(claim)) { return; }
            obsolete = updateDeliveryPredictionLocked(work, predictedMs, System.currentTimeMillis());
        }
        try { cancelDecisionDeadline(obsolete); }
        finally { armDecisionDeadline(); }
    }

    void publishRoute(DeliveryClaim claim, WorkSnapshot work, long predictedMs) {
        RequestEffect effect;
        DecisionDeadline obsolete;
        synchronized (this) {
            if (!ownsDeliveryClaim(claim)) { return; }
            if (claim.kind != DeliveryClaimKind.ROUTE_DECISION) {
                throw new IllegalArgumentException("route publication requires a route claim");
            }
            if (claim.completed) { throw new IllegalStateException("delivery result already consumed for request " + requestId); }
            obsolete = updateDeliveryPredictionLocked(work, predictedMs, System.currentTimeMillis());
            claim.completed = true;
            effect = acknowledgeDeliveryLocked(claim.correlationId, null);
        }
        try { cancelDecisionDeadline(obsolete); }
        finally {
            try { armDecisionDeadline(); }
            finally { execute(effect); }
        }
    }

    void completeDelivery(DeliveryClaim claim, DeliveryResult result) {
        Objects.requireNonNull(result, "delivery result");
        RequestEffect effect = null;
        synchronized (this) {
            if (!ownsDeliveryClaim(claim)) { return; }
            if (claim.completed) { throw new IllegalStateException("delivery result already consumed for request " + requestId); }
            claim.completed = true;
            if (result.status() == DeliveryResult.Status.DELIVERED) {
                if (claim.kind == DeliveryClaimKind.BATCH_ENQUEUE) {
                    claim.item.ctx().setAckAtMs(System.currentTimeMillis());
                    claim.item.ctx().setAckAtNanos(System.nanoTime());
                }
                effect = acknowledgeDeliveryLocked(claim.correlationId, null);
            } else if (decodeOwnsRequest()) {
                effect = acknowledgeDeliveryLocked(claim.correlationId, null);
            } else if (result.status() == DeliveryResult.Status.FAILED) {
                String message = "Delivery failed: " + detailOf(result.cause());
                DecodeEndpoint decode = claim.item.decodeEp();
                DecodeEndpoint.ReservationHandle reservation = claim.item.decodeReservation();
                DecodeEndpoint.DispatchRejectionSettlement settlement = decode == null || reservation == null
                        ? DecodeEndpoint.DispatchRejectionSettlement.RELEASED
                        : decode.settleDefiniteDispatchRejection(reservation);
                switch (settlement) {
                    case RELEASED -> effect = reduceDeferredTerminalFactLocked(DeferredTerminal.deliveryRejected(message));
                    case ENGINE_ACCEPTED -> effect = acknowledgeDeliveryLocked(claim.correlationId, null);
                    case CONFLICT -> markAwaitingConfirmation(message);
                    case STALE -> { }
                }
            } else {
                markAwaitingConfirmation((result.status() == DeliveryResult.Status.TIMED_OUT
                        ? "Delivery timed out: " : "Delivery outcome uncertain: ") + detailOf(result.cause()));
            }
        }
        execute(effect);
    }

    private boolean ownsDeliveryClaim(DeliveryClaim claim) {
        requireSlotLock("delivery identity");
        if (claim == null || claim.slot != this) { throw new IllegalArgumentException("foreign delivery claim"); }
        return ownsDeliveryClaim(claim.item, claim.kind, claim.correlationId);
    }

    void failDeliveryPreparation(ScheduledRequest exact, Throwable cause) {
        RequestEffect work;
        try {
            synchronized (this) {
                if (!ownsPreparedDelivery(exact)) { return; }
                work = reduceDeferredTerminalFactLocked(DeferredTerminal.deliveryFailure(
                        StrategyErrorType.DISPATCH_FAILED, "Delivery preparation failed: " + detailOf(cause)));
            }
            execute(work);
        } catch (Throwable failure) {
            if (cause != null && cause != failure) { failure.addSuppressed(cause); }
            Logger.error("Prepared delivery failure reduction failed request_id={}", requestId, failure);
        }
    }

    boolean updatePreemption(PreemptionRegistration claim, PreemptionCancelPhase next) {
        if (next == null) { return false; }
        RequestEffect work;
        synchronized (this) {
            work = applyPreemptionPhase(claim, next);
        }
        execute(work);
        return work != null && work.status() != RequestEffect.Status.STALE;
    }

    boolean releasePreemption(PreemptionRegistration claim) {
        RequestEffect work;
        synchronized (this) {
            work = applyPreemptionRelease(claim);
        }
        execute(work);
        return work != null && work.status() != RequestEffect.Status.STALE;
    }

    boolean completePreemption(PreemptionRegistration claim, String detail) {
        RequestEffect work;
        synchronized (this) {
            work = applyPreemptionCompleted(claim, detail);
        }
        execute(work);
        return work != null && work.status() != RequestEffect.Status.STALE;
    }

    TerminalAction prepareShutdown() {
        synchronized (this) {
            if (!isCurrentGeneration() || !canClaimLocalTerminal()) { return null; }
            String message = "request scheduler is shutting down";
            return beginTerminalizing(
                    TerminalOutcome.fail(message), buildErrorResponse(StrategyErrorType.DISPATCH_FAILED, message));
        }
    }

    boolean publishDecisionResponse(Response response) {
        PublicationPermit permit = finishExternalResponse(response);
        if (permit == null) { return false; }
        try { submitTerminalResponse(permit, response); return true; }
        catch (RuntimeException | Error failure) { permit.abortClaimedPublication(); throw failure; }
    }

    boolean completeExternalResponse(Response response) {
        requireOutsideSlotLock("external Future completion");
        PublicationPermit permit = finishExternalResponse(response);
        return permit != null && completionPublisher.publishNow(selectResponse(permit, response));
    }

    boolean completeExternalFailure(Throwable error) {
        requireOutsideSlotLock("external Future completion");
        PublicationPermit permit = finishExternalFailure(error);
        return permit != null && completionPublisher.publishNow(selectFailure(permit, error));
    }

    boolean cancelExternalFuture(boolean mayInterruptIfRunning) {
        requireOutsideSlotLock("external Future completion");
        PublicationPermit permit = finishExternalCancellation();
        return permit != null && completionPublisher.publishNow(selectCancellation(permit, mayInterruptIfRunning));
    }

    void submitTerminalResponse(PublicationPermit permit, Response response) {
        completionPublisher.submit(selectResponse(permit, response));
    }

    private PublicationPermit finishExternalResponse(Response response) {
        String detail = response != null && response.getErrorMessage() != null
                ? response.getErrorMessage() : "external future completion";
        TerminalOutcome transition =
                response != null && !response.isSuccess()
                        ? TerminalOutcome.fail(detail)
                        : TerminalOutcome.complete(detail);
        return finishExternalTerminal(transition);
    }

    private PublicationPermit finishExternalFailure(Throwable error) {
        Objects.requireNonNull(error, "error");
        String detail = "external future failure"
                + (error.getMessage() == null ? "" : ": " + error.getMessage());
        return finishExternalTerminal(TerminalOutcome.fail(detail));
    }

    private PublicationPermit finishExternalCancellation() {
        String detail = CancelReason.CLIENT_CANCELLED.getMessage();
        return finishExternalTerminal(TerminalOutcome.cancel(detail));
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

    synchronized RequestState snapshot() {
        return new RequestState(
                requestId, state, deliveryClaimKind, batchId,
                createdAtMs, updatedAtMs, detail);
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

    record EngineObservation(RequestEffect transition, DecisionDeadline obsoleteDeadline) {
        static final EngineObservation STALE = new EngineObservation(RequestEffect.STALE, null);
    }

    EngineObservation applyPrefillStatusLocked(PrefillEndpoint source, RoleType role,
                                         PrefillState.WorkerStatusFact fact, long nowMs) {
        requireSlotLock("Prefill fact reduction");
        if (!ownsPrefillFact(source, fact.item())) { return EngineObservation.STALE; }
        lastWorkerStatusAtMs = Math.max(lastWorkerStatusAtMs, nowMs);
        RequestEffect transition = switch (fact.kind()) {
            case ACTIVE -> {
                observeDecisionPrefillActive();
                yield reducePrefillActive(source, fact.item());
            }
            case COMPLETED -> {
                observeDecisionPrefillCompleted(nowMs, role != RoleType.PDFUSION && item.decodeEp() != null);
                yield role == RoleType.PDFUSION
                        ? reduceWorkerTerminal(fact.item(), DeferredTerminal.worker(
                                WorkerTerminalSource.PREFILL_ENDPOINT, true, fact.errorCode()))
                        : RequestEffect.NONE;
            }
            case FAILED -> reduceWorkerTerminal(fact.item(), DeferredTerminal.worker(
                    WorkerTerminalSource.PREFILL_ENDPOINT, false, fact.errorCode()));
            case PRIORITY_CANCELED -> reducePriorityCanceled(source, fact.item());
        };
        reconcileDecisionEvidence();
        return new EngineObservation(transition, detachObsoleteDecisionDeadline());
    }

    EngineObservation applyDecodeStatusLocked(DecodeEndpoint source, DecodeEndpoint.WorkerStatusFact fact, long nowMs) {
        requireSlotLock("Decode fact reduction");
        if (!ownsDecodeFact(source, fact.reservation())) { return EngineObservation.STALE; }
        lastWorkerStatusAtMs = Math.max(lastWorkerStatusAtMs, nowMs);
        if (fact.kind() == DecodeEndpoint.WorkerStatusFact.Kind.TERMINAL) {
            advanceDecision(DecisionStage.ACCEPTED, OptionalLong.empty());
            markDecodeTerminalOwned();
            return new EngineObservation(reduceWorkerTerminal(item, DeferredTerminal.worker(
                    WorkerTerminalSource.DECODE_ENDPOINT, fact.errorCode() == 0L, fact.errorCode())),
                    detachObsoleteDecisionDeadline());
        }
        // Both a repeated ACTIVE observation and first ACCEPTED prove Decode ownership.
        DecodeAcceptance acceptance = markDecodeAccepted();
        return new EngineObservation(RequestEffect.NONE,
                acceptance.detachedDecisionDeadline());
    }

    private StrategyErrorType timeoutErrorType() {
        requireSlotLock("deadline error lookup");
        return deadlineErrorType;
    }

    private StrategyErrorType cancellationErrorType(CancelReason reason) {
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
     * <p>The admission handle is the logical pin that lets the endpoint
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
                || admissionHandle == null
                || candidate.requestId() != requestId
                || candidate.future() != future) {
            return false;
        }
        item = candidate;
        assertInvariant();
        return true;
    }

    /** Roll back only the exact binding whose queue publication did not commit. */
    private void rollbackItemPublication(ScheduledRequest exact) {
        requireSlotLock("request item publication rollback");
        if (item != exact || admissionHandle == null) {
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

    private boolean ownsActiveGeneration() {
        requireSlotLock("active generation lookup");
        return isCurrentGeneration()
                && slotPhase == SlotPhase.ACTIVE
                && !state.isTerminal();
    }

    boolean ownsActiveItem(ScheduledRequest expected) {
        requireSlotLock("active item ownership lookup");
        return ownsActiveGeneration() && item == expected;
    }

    private boolean ownsPrefillFact(PrefillEndpoint source, ScheduledRequest expected) {
        requireSlotLock("Prefill fact ownership lookup");
        return ownsActiveItem(expected) && expected.prefillEp() == source;
    }

    private boolean ownsDecodeFact(
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

    private boolean ownsDeliveryClaim(
            ScheduledRequest expected,
            DeliveryClaimKind kind,
            long expectedBatchId) {
        requireSlotLock("delivery claim lookup");
        return ownsActiveItem(expected)
                && deliveryClaimKind == kind
                && batchId == expectedBatchId
                && !state.isTerminal();
    }

    boolean decodeOwnsRequest() {
        requireSlotLock("Decode ownership lookup");
        return engineOwnership == EngineOwnership.DECODE_OWNED;
    }

    private boolean canClaimLocalTerminal() {
        requireSlotLock("local terminal eligibility");
        return ownsActiveGeneration()
                && !future.isDone()
                && admissionHandle == null
                && preemption == null
                && engineOwnership == EngineOwnership.DECODE_PENDING
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
        return isCurrentGeneration() && slotPhase != SlotPhase.TERMINAL_RECORD;
    }

    boolean isTerminalRecord() {
        requireSlotLock("terminal record lookup");
        return slotPhase == SlotPhase.TERMINAL_RECORD;
    }

    boolean isRemovableTerminalRecord(long updatedBeforeMs) {
        requireSlotLock("terminal record retention lookup");
        return isCurrentGeneration()
                && slotPhase == SlotPhase.TERMINAL_RECORD
                && state.isTerminal()
                && updatedAtMs < updatedBeforeMs
                && item == null;
    }

    // ==================== Admission handle ====================

    AdmissionHandle tryBeginAdmissionHandle() {
        requireSlotLock("admission handle claim");
        if (!ownsActiveGeneration()
                || !isOpen()
                || item != null
                || admissionHandle != null
                || preemption != null) {
            return null;
        }
        AdmissionHandle exact = new AdmissionHandle(this);
        admissionHandle = exact;
        assertInvariant();
        return exact;
    }

    AdmissionHandleCompletion completeAdmissionHandle(
            AdmissionHandle exact) {
        requireSlotLock("admission handle completion");
        if (admissionHandle == null || admissionHandle != exact) {
            return AdmissionHandleCompletion.NOT_OWNED;
        }
        admissionHandle = null;
        return finishAdmissionHandle();
    }

    private AdmissionHandleCompletion claimAdmissionHandleTermination(
            AdmissionHandle exact) {
        requireSlotLock("admission handle terminal claim");
        if (!ownsActiveGeneration()
                || admissionHandle == null
                || admissionHandle != exact) {
            throw new IllegalStateException(
                    "admission handle no longer owns request " + requestId);
        }
        admissionHandle = null;
        return finishAdmissionHandle();
    }

    /** Resolve retained facts once for both ordinary close and terminal close. */
    private AdmissionHandleCompletion finishAdmissionHandle() {
        CancelReason cancellationReason = pendingAdmissionCancelReason;
        pendingAdmissionCancelReason = null;
        boolean inactivityExpired = pendingAdmissionInactivityExpired;
        pendingAdmissionInactivityExpired = false;
        // A retained worker terminal may settle cancellation, but must
        // not erase the first cause already chosen during admission.
        cancellationReason = promoteAdmissionCancellation(cancellationReason);
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
            cancellationReason = null;
        }
        assertInvariant();
        return new AdmissionHandleCompletion(
                true, cancellationReason, pendingTerminal,
                pendingRetirement, inactivityExpired);
    }

    /**
     * Atomically move the admission-scoped first cause into the canonical
     * cancellation owner before releasing the slot lock. The lifecycle was
     * already moved to {@code CANCEL_REQUESTED} when the cause was deferred;
     * this transfer keeps the first cause available to admission settlement
     * and subsequent request events.
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
                        "admission handle observed another Prefill generation"
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

    boolean consumeInactivityDeadline(InactivityDeadline exact) {
        requireSlotLock("request inactivity check");
        if (inactivityDeadline != exact || !ownsActiveGeneration()) {
            return false;
        }
        inactivityDeadline = null;
        return !state.isTerminal();
    }

    private DecisionDeadline updateDeliveryPredictionLocked(WorkSnapshot precedingWork, long unstartedWorkMs, long nowMs) {
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
        // Reconcile the exact reservation in this decision. Engine acceptance overrides the prediction.
        if (item.decodeEp() != null && item.decodeEp().isReservationAccepted(item.decodeReservation())) {
            return applyDecodeStatusLocked(item.decodeEp(),
                    DecodeEndpoint.WorkerStatusFact.accepted(item.decodeReservation()), nowMs).obsoleteDeadline();
        }
        return null;
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
    private void markAwaitingConfirmation(String message) {
        requireSlotLock("delivery confirmation wait");
        if (!ownsActiveGeneration() || cancellationReason != null
                || pendingAdmissionCancelReason != null
                || decisionStage == DecisionStage.PREFILL_RUNNING
                || decisionStage == DecisionStage.ACCEPTED) {
            return;
        }
        detail = "SUSPECTED_LOST: " + Objects.requireNonNull(message, "message");
        updatedAtMs = System.currentTimeMillis();
        assertInvariant();
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
    private DecisionDeadline detachObsoleteDecisionDeadline() {
        requireSlotLock("decision deadline reconciliation");
        if (decisionDeadline == null || decisionExpiresAtMs.equals(
                OptionalLong.of(decisionDeadline.deadlineAtMs()))) {
            return null;
        }
        return detachDecisionDeadline();
    }

    synchronized DecisionExpiry expire(DecisionDeadline exact) {
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

    // Request decisions: admission and preemption constrain which facts can settle.

    PreemptionRegistration tryInstallPreemption(
            long reservationToken,
            long attemptToken,
            String detail) {
        requireSlotLock("preemption installation");
        DecodeEndpoint.ReservationHandle reservation =
                item == null ? null : item.decodeReservation();
        if (!ownsActiveGeneration()
                || admissionHandle != null
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
    RequestEffect applyPreemptionPhase(
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
            return RequestEffect.STALE;
        }
        if (next == PreemptionCancelPhase.CANCEL_REQUESTED) {
            if (!state.isTerminal()) { transition(RequestState.Phase.CANCEL_REQUESTED, exact.detail()); }
        }
        assertInvariant();
        return switch (next) {
            case CLAIMED -> RequestEffect.STALE;
            case CANCEL_IN_FLIGHT, CANCEL_REQUESTED ->
                    RequestEffect.NONE;
            case NOT_FOUND_STALE -> processPendingEventsUnderPreemptionLocked(exact, false, exact);
            case CANCEL_UNKNOWN ->
                    processPendingEventsUnderPreemptionLocked(exact, true, exact);
        };
    }

    private RequestEffect applyPreemptionRelease(
            PreemptionRegistration claim) {
        requireSlotLock("preemption release reduction");
        PreemptionRegistration exact = exactPreemption(claim);
        if (!ownsActiveGeneration()
                || exact == null
                || preemption != exact
                || !exact.isReleasable()) {
            return RequestEffect.STALE;
        }
        detachPreemptionOwner(exact);
        return processPendingEventsUnderPreemptionLocked(exact, false, exact);
    }

    RequestEffect applyPreemptionCompleted(
            PreemptionRegistration claim,
            String detail) {
        requireSlotLock("preemption completion reduction");
        PreemptionRegistration exact = exactPreemption(claim);
        if (!ownsActiveGeneration()
                || exact == null
                || !exact.canCompletePreemption()
                || !exact.tryFinish()) {
            return RequestEffect.STALE;
        }
        DeferredTerminal terminal = DeferredTerminal.priority(detail);
        retainPreemptionTerminalLocked(exact, terminal);
        // DecodePreemptionCoordinator has already consumed the exact endpoint
        // claim before publishing REQUEST_FENCED. Reconciliation is therefore
        // neither required nor legal on this authoritative path.
        detachPreemptionOwner(exact);
        assertInvariant();
        return RequestEffect.terminal(beginPreemptedRequestTerminalLocked(terminal), exact);
    }

    /**
     * Reduce one exact transport/endpoint fact without exposing the mutable preemption
     * registration. The caller holds {@code synchronized (slot)} and only executes the returned,
     * already-selected effect.
     */
    private RequestEffect reducePrefillActive(PrefillEndpoint source, ScheduledRequest expected) {
        requireSlotLock("Prefill activity reduction");
        PreemptionRegistration exact = preemption;
        if (!ownsPrefillFact(source, expected) || exact == null || !exact.isNotFound()) {
            return RequestEffect.STALE;
        }
        DecodeEndpoint decode = expected.decodeEp();
        if (decode == null
                || decode.reconcilePriorityVictimActive(
                        exact.attemptToken(), expected.decodeReservation())) {
            detachPreemptionOwner(exact);
        }
        return RequestEffect.NONE;
    }

    RequestEffect reduceWorkerTerminal(ScheduledRequest expected, DeferredTerminal terminal) {
        requireSlotLock("worker terminal reduction");
        if (!terminal.authoritativeWorker()) {
            throw new IllegalArgumentException(
                    "worker terminal requires authoritative observation");
        }
        if (!ownsActiveItem(expected)) {
            return RequestEffect.STALE;
        }
        if (admissionHandle != null) {
            retainAdmissionTerminal(terminal);
            assertInvariant();
            return RequestEffect.NONE;
        }
        PreemptionRegistration exact = preemptionOwner();
        if (exact == null) {
            return terminalEffectLocked(terminal, null);
        }
        if (exact.isFinished()) {
            return RequestEffect.STALE;
        }
        retainPreemptionTerminalLocked(exact, terminal);
        if (!ownsActiveGeneration() || preemptionOwner() != exact || !exact.tryFinish()) {
            return RequestEffect.STALE;
        }
        assertInvariant();
        return processPendingEventsUnderPreemptionLocked(exact, false, exact);
    }

    private RequestEffect reduceDispatchRejected(
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
            return RequestEffect.STALE;
        }
        if (admissionHandle != null) {
            retainAdmissionTerminal(terminal);
            assertInvariant();
            return RequestEffect.NONE;
        }
        PreemptionRegistration exact = preemptionOwner();
        PreemptionRegistration signal = null;
        if (exact != null) {
            retainPreemptionTerminalLocked(exact, terminal);
            if (exact.tryFinish()) {
                signal = exact;
            }
            detachPreemptionOwner(exact);
        }
        assertInvariant();
        return terminalEffectLocked(terminal, signal);
    }

    private RequestEffect reduceOrdinaryTerminal(
            ScheduledRequest expected, DeferredTerminal terminal) {
        requireSlotLock("ordinary terminal reduction");
        if (!ownsActiveItem(expected)) {
            return RequestEffect.STALE;
        }
        if (terminal.authoritativeWorker()) {
            throw new IllegalArgumentException("authoritative worker fact requires WorkerTerminal");
        }
        if (admissionHandle != null) {
            retainAdmissionTerminal(terminal);
            assertInvariant();
            return RequestEffect.NONE;
        }

        PreemptionRegistration exact = preemptionOwner();
        if (engineOwnership == EngineOwnership.DECODE_OWNED && terminal.deliveryFailure()) {
            long deliveryBatchId = batchId;
            if (exact == null) {
                return acknowledgeDeliveryLocked(deliveryBatchId, null);
            }
            if (exact.isFinished()) {
                return RequestEffect.STALE;
            }
            exact.recordDeliveryConfirmation(deliveryBatchId);
            assertInvariant();
            return processPendingEventsUnderPreemptionLocked(exact, false, null);
        }

        if (exact == null) {
            return terminalEffectLocked(terminal, null);
        }
        if (exact.isFinished()) {
            return RequestEffect.STALE;
        }
        retainPreemptionTerminalLocked(exact, terminal);
        assertInvariant();
        if (!exact.isNotFound() && !exact.isUnknown()) {
            return RequestEffect.NONE;
        }
        return processPendingEventsUnderPreemptionLocked(exact, exact.isUnknown(), exact);
    }

    private RequestEffect reducePriorityCanceled(PrefillEndpoint source, ScheduledRequest expected) {
        requireSlotLock("priority cancellation reduction");
        PreemptionRegistration exact = ownsPrefillFact(source, expected) ? preemptionOwner() : null;
        DecodeEndpoint decode = expected.decodeEp();
        if (exact == null
                || exact.isFinished()
                || decode == null
                || expected.decodeReservation() == null
                || !decode.settlePriorityCanceled(
                        exact.attemptToken(), expected.decodeReservation())
                || !ownsActiveGeneration()
                || preemptionOwner() != exact
                || !exact.tryFinish()) {
            return RequestEffect.STALE;
        }
        DeferredTerminal terminal = DeferredTerminal.priority("priority victim canceled by worker");
        retainPreemptionTerminalLocked(exact, terminal);
        detachPreemptionOwner(exact);
        assertInvariant();
        return terminalEffectLocked(terminal, exact);
    }

    private RequestEffect reduceDecodeGenerationRetired(
            DecodeEndpoint source, DecodeEndpoint.ReservationHandle reservation, String detail) {
        requireSlotLock("Decode generation retirement reduction");
        Objects.requireNonNull(detail, "detail");
        if (!ownsDecodeFact(source, reservation)) {
            return RequestEffect.STALE;
        }
        DeferredTerminal terminal = DeferredTerminal.decodeGenerationRetired(detail);
        if (admissionHandle != null) {
            retainAdmissionTerminal(terminal);
            assertInvariant();
            return RequestEffect.NONE;
        }

        PreemptionRegistration exact = preemptionOwner();
        PreemptionRegistration signal = null;
        if (exact != null) {
            retainPreemptionTerminalLocked(exact, terminal);
            exact.tryFinish();
            signal = exact;
        }
        detachPreemptionOwner(exact);

        assertInvariant();
        return terminalEffectLocked(terminal, signal);
    }

    /** Shared confirmation decision for transport, Engine evidence and released preemption. */
    private RequestEffect acknowledgeDeliveryLocked(long expectedBatchId, PreemptionRegistration signal) {
        requireSlotLock("delivery acknowledgement");
        if (item == null || !ownsDeliveryClaim(item, deliveryClaimKind, expectedBatchId)) { return RequestEffect.STALE; }
        if (cancellationReason != null || pendingAdmissionCancelReason != null) { return RequestEffect.NONE; }
        PreemptionRegistration blocked = preemptionOwner();
        if (blocked != null) {
            if (blocked.isFinished()) { return RequestEffect.STALE; }
            blocked.recordDeliveryConfirmation(expectedBatchId);
            assertInvariant();
            return processPendingEventsUnderPreemptionLocked(blocked, false, null);
        }
        if (state != RequestState.Phase.DISPATCHING) { return RequestEffect.STALE; }
        PublicationPermit permit = requirePublicationPermit(PublicationKind.DELIVERY);
        try {
            Response response = buildSuccessResponse(item.routeResponse(), deliveryClaimKind == DeliveryClaimKind.BATCH_ENQUEUE);
            transition(RequestState.Phase.ACKNOWLEDGED, deliveryClaimKind == DeliveryClaimKind.BATCH_ENQUEUE
                    ? "batch enqueue acknowledged" : "route decision delivered");
            DeliveryPublication publication = new DeliveryPublication(item, response, permit,
                    requestDeadline, batchEnqueueStartedAtMs);
            requestDeadline = null;
            assertInvariant();
            return RequestEffect.delivery(publication, signal);
        } catch (RuntimeException | Error failure) {
            permit.abandonIfUnclaimed();
            throw failure;
        }
    }

    private void retainPreemptionTerminalLocked(PreemptionRegistration exact, DeferredTerminal candidate) {
        requireSlotLock("preemption terminal evidence");
        DeferredTerminal previous = exact.pendingTerminal();
        if (previous == null || !previous.authoritativeWorker() && candidate.authoritativeWorker()) {
            exact.storeTerminal(candidate);
        }
    }

    private RequestEffect processPendingEventsUnderPreemptionLocked(
            PreemptionRegistration exact,
            boolean transportUnknown,
            PreemptionRegistration signal) {
        DeferredTerminal terminal = exact.pendingTerminal();
        if (terminal != null
                && (!transportUnknown || terminal.authoritativeWorker())) {
            ScheduledRequest active = activeItem();
            DecodeEndpoint decode = active == null ? null : active.decodeEp();
            boolean decodeTerminalApplied = tryReconcileDecodeTerminal(
                    terminal,
                    decode,
                    exact.attemptToken(),
                    active == null ? null : active.decodeReservation());
            if (!decodeTerminalApplied) {
                return RequestEffect.NONE;
            }
            detachPreemptionOwner(exact);
            return terminalEffectLocked(terminal, signal);
        }
        if (transportUnknown || !exact.hasPendingDeliveryConfirmation()) {
            return RequestEffect.NONE;
        }

        ScheduledRequest active = activeItem();
        DecodeEndpoint decode = active == null ? null : active.decodeEp();
        boolean activeWon = decode == null
                || decode.reconcilePriorityVictimActive(
                        exact.attemptToken(),
                        active.decodeReservation());
        if (!activeWon) {
            return RequestEffect.NONE;
        }
        detachPreemptionOwner(exact);
        if (active == null) {
            return RequestEffect.STALE;
        }
        return acknowledgeDeliveryLocked(exact.pendingConfirmationBatchId(), signal);
    }

    /**
     * Apply a terminal observation to the exact Decode preemption claim when needed.
     * DecodeEndpoint emits its terminal fact after updating its ledger; do not apply it twice.
     * A Prefill observation still needs the Decode reconciliation transaction.
     */
    static boolean tryReconcileDecodeTerminal(
            DeferredTerminal terminal,
            DecodeEndpoint decode,
            long attemptToken,
            DecodeEndpoint.ReservationHandle reservation) {
        if (decode == null || terminal.decodeTerminalAlreadyApplied()) {
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

    private void markDecodeTerminalOwned() {
        requireSlotLock("Decode terminal ownership");
        if (ownsActiveGeneration()) {
            engineOwnership = EngineOwnership.DECODE_OWNED;
            assertInvariant();
        }
    }

    private TerminalAction beginPrefillRetirementTerminal(
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
        if (admissionHandle != null) {
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
        return beginTerminalizing(pending.transition, pending.response);
    }

    // ==================== Terminal ownership ====================

    /**
     * The terminal claim freezes the delivery stage until cleanup commits the terminal record.
     * Endpoint operations run without the slot monitor and validate their own exact ledger.
     */
    void releaseTerminalEndpoints(TerminalAction action) {
        requireOutsideSlotLock("terminal endpoint cleanup");
        DeliveryClaimKind delivery;
        synchronized (this) {
            if (slotPhase != SlotPhase.TERMINALIZING || action.slot() != this || action.item() != item) {
                throw new IllegalStateException("terminal cleanup does not own request " + requestId);
            }
            delivery = deliveryClaimKind;
        }
        ScheduledRequest exact = action.item();
        if (exact == null) { return; }
        DeferredTerminal event = action.event();
        Throwable failure = null;
        if (delivery == DeliveryClaimKind.NONE && exact.prefillEp() != null) {
            failure = RequestTerminalCleanup.runTerminalLeaf(failure,
                    () -> exact.prefillEp().removeQueued(exact, "TERMINAL_RELEASE"));
        }
        if (event != null && event.kind() == DeferredTerminal.Kind.INACTIVITY_EXPIRED) {
            // Expiry ends exact local tracking even when delivery is uncertain or Engine-owned.
            failure = RequestTerminalCleanup.runTerminalLeaf(failure,
                    exact.decodeEp() == null ? null
                            : () -> exact.decodeEp().expireReservationExact(exact.decodeReservation()));
            failure = RequestTerminalCleanup.runTerminalLeaf(failure,
                    exact.prefillEp() == null ? null : () -> exact.prefillEp().expireCommittedItem(exact));
        } else {
            // A projection may lag endpoint ownership. Never roll back an Engine/protocol owner.
            failure = RequestTerminalCleanup.runTerminalLeaf(failure,
                    exact.decodeEp() == null ? null
                            : () -> exact.decodeEp().releaseLocalShadowIfExact(exact.decodeReservation()));
            failure = RequestTerminalCleanup.runTerminalLeaf(failure,
                    () -> cleanupPrefillAfterRequestTerminal(exact, delivery, event));
        }
        rethrowCleanup(failure);
    }

    private void cleanupPrefillAfterRequestTerminal(ScheduledRequest exact, DeliveryClaimKind delivery, DeferredTerminal event) {
        if (exact.prefillEp() == null) { return; }
        if (event != null) {
            switch (event.kind()) {
                case WORKER -> {
                    // Prefill facts are post-settlement. Decode facts do not end a BATCH group.
                    if (event.workerSource() == WorkerTerminalSource.PREFILL_ENDPOINT
                            || delivery == DeliveryClaimKind.BATCH_ENQUEUE) { return; }
                }
                case DELIVERY_REJECTED, DECODE_GENERATION_RETIRED, PRIORITY -> {
                    // These exact settlements end the member, including one handed to a batch.
                    exact.prefillEp().releaseCommittedItem(exact);
                    return;
                }
                default -> { }
            }
        }
        if (delivery != DeliveryClaimKind.BATCH_ENQUEUE) {
            // Before handoff or after NON_BATCH delivery, accounting is still per-request.
            exact.prefillEp().releaseCommittedItem(exact);
        }
    }

    TerminalAction beginTerminalizing(TerminalOutcome transition, Response response) {
        return beginTerminalizing(null, transition, response, response != null);
    }

    /** Claim a locally reversible terminal specifically for public-future use. */
    private TerminalAction beginExternalTerminalizing(TerminalOutcome transition) {
        requireSlotLock("external terminal claim");
        return canClaimLocalTerminal() ? beginTerminalizing(null, transition, null, true) : null;
    }

    private TerminalAction beginTerminalizing(DeferredTerminal event, TerminalOutcome transition, Response response) {
        return beginTerminalizing(event, transition, response, response != null);
    }

    private TerminalAction beginTerminalizing(DeferredTerminal event, TerminalOutcome transition, Response response,
                                               boolean requestPublication) {
        requireSlotLock("terminal claim");
        if (transition == null) {
            throw new IllegalStateException(
                    "terminal transition is required for request " + requestId);
        }
        if (!ownsActiveGeneration() || admissionHandle != null) {
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
                claimedPreemption.tryFinish();
            }

            RequestDeadline claimedRequestDeadline = requestDeadline;
            requestDeadline = null;
            DecisionDeadline detachedDecisionDeadline = detachDecisionDeadline();
            InactivityDeadline claimedInactivityDeadline = inactivityDeadline;
            inactivityDeadline = null;
            TerminalResources terminalResources = new TerminalResources(
                    claimedRequestDeadline, detachedDecisionDeadline, claimedInactivityDeadline);
            TerminalAction action = new TerminalAction(
                    this, item, claimedPreemption, terminalResources, event, transition,
                    publishable ? response : null, publication);
            transferred = true;
            assertInvariant();
            return action;
        } finally {
            if (!transferred && publication != null) {
                publication.abandonIfUnclaimed();
            }
        }
    }

    TerminationResult finishTermination(TerminalAction action) {
        requireSlotLock("termination completion");
        if (!isCurrentGeneration()
                || slotPhase != SlotPhase.TERMINALIZING
                || action.slot() != this
                || item != action.item()) {
            if (action.publication() != null) {
                action.publication().abandonIfUnclaimed();
            }
            return new TerminationResult(
                    null,
                    new IllegalStateException(
                            "terminal slot identity changed: request_id="
                                    + requestId),
                    null);
        }
        RequestState terminal;
        Throwable transitionFailure = null;
        try {
            terminal = commitTerminalStateLocked(action.transition());
        } catch (Throwable failure) {
            transitionFailure = failure;
            terminal = commitTerminalStateLocked(TerminalOutcome.fail("terminal projection failed"));
        }
        if (!terminal.state().isTerminal()) {
            transitionFailure = appendFailure(
                    transitionFailure,
                    new IllegalStateException(
                            "terminal transition did not terminate request "
                                    + requestId));
            terminal = commitTerminalStateLocked(TerminalOutcome.fail("terminal projection did not terminate"));
        }

        item = null;
        preemption = null;
        cancellationReason = null;
        admissionHandle = null;
        pendingAdmissionCancelReason = null;
        pendingAdmissionInactivityExpired = false;
        requestDeadline = null;
        decisionDeadline = null;
        inactivityDeadline = null;
        slotPhase = SlotPhase.TERMINAL_RECORD;
        assertInvariant();
        return new TerminationResult(
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
        if (admissionHandle != null && preemption != null) {
            throw new IllegalStateException(
                    "admission handle overlaps preemption for " + requestId);
        }
        if ((pendingAdmissionCancelReason != null || pendingAdmissionInactivityExpired)
                && admissionHandle == null) {
            throw new IllegalStateException(
                    "pending admission cancellation has no admission handle for "
                            + requestId);
        }
        if (slotPhase == SlotPhase.TERMINALIZING
                && (admissionOpen || admissionHandle != null)) {
            throw new IllegalStateException(
                    "terminalizing request still owns admission "
                            + requestId);
        }
        if (slotPhase != SlotPhase.TERMINAL_RECORD) {
            return;
        }
        if (admissionOpen
                || item != null
                || cancellationReason != null
                || preemption != null
                || admissionHandle != null
                || pendingAdmissionCancelReason != null
                || pendingAdmissionInactivityExpired
                || requestDeadline != null
                || decisionDeadline != null || inactivityDeadline != null) {
            throw new IllegalStateException(
                    "terminal record retains request-owned state for " + requestId);
        }
        if (!state.isTerminal()) {
            throw new IllegalStateException(
                    "terminal record lifecycle is not terminal for " + requestId);
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

    private RequestEffect terminalEffectLocked(DeferredTerminal terminal, PreemptionRegistration signal) {
        TerminalAction action = terminal.kind() == DeferredTerminal.Kind.PRIORITY
                ? beginPreemptedRequestTerminalLocked(terminal)
                : decideTerminalLocked(terminal);
        return RequestEffect.terminal(action, signal);
    }

    private void execute(RequestEffect effect) {
        if (effect == null || effect.status() != RequestEffect.Status.READY) { return; }
        requireOutsideSlotLock("request effects");
        try {
            if (effect.terminal() != null) {
                terminalCleanup.submitTerminal(effect.terminal());
            } else {
                DeliveryPublication delivery = effect.delivery();
                publishDelivery(delivery);
            }
        } finally {
            if (effect.signal() != null) {
                effect.signal().signalTerminal(new VictimTerminal(requestId));
            }
        }
    }

    private TerminalAction beginPreemptedRequestTerminalLocked(DeferredTerminal terminal) {
        String detail = terminal.detail();
        if (this.hasCancellationFirstCause()) {
            CancelReason firstCause = this.requireCancellationFirstCause();
            String cancellationDetail = firstCause.getMessage();
            return beginTerminalizing(terminal,
                    TerminalOutcome.cancellation(firstCause, cancellationDetail),
                    buildErrorResponse(
                            this.cancellationErrorType(firstCause),
                            cancellationDetail));
        }
        return beginTerminalizing(terminal,
                TerminalOutcome.cancel(detail),
                buildErrorResponse(
                        StrategyErrorType.PRIORITY_PREEMPTED, detail));
    }

    private TerminalAction beginExpiredRequestLocked(String detail) {
        CancelReason firstCause = requireCancellationFirstCause();
        return beginTerminalizing(DeferredTerminal.inactivityExpired(detail),
                TerminalOutcome.cancellation(firstCause, detail),
                buildErrorResponse(cancellationErrorType(firstCause), detail));
    }

    private RequestEffect reduceDeferredTerminalFactLocked(
            DeferredTerminal terminal) {
        ScheduledRequest item = this.activeItem();
        if (item == null) {
            return null;
        }
        RequestSlot.RequestEffect reduction;
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
        return reduction;
    }

    private TerminalAction decideTerminalLocked(
            DeferredTerminal terminal) {
        if (terminal.endpointAlreadyRetired()) {
            return applyDecodeRejectionOrRetirementLocked(terminal);
        }
        return switch (terminal.kind()) {
            case FAILURE -> applyFailureTerminalLocked(terminal);
            case TIMEOUT -> applyTimeoutTerminalLocked(terminal);
            case DELIVERY_FAILURE ->
                    applyFailureTerminalLocked(terminal);
            case DELIVERY_REJECTED ->
                    applyDecodeRejectionOrRetirementLocked(terminal);
            case WORKER -> applyWorkerTerminalLocked(terminal);
            case PRIORITY ->
                    throw new IllegalStateException(
                            "priority terminal requires its typed reducer");
            case INACTIVITY_EXPIRED -> beginExpiredRequestLocked(terminal.detail());
            case DECODE_GENERATION_RETIRED ->
                    throw new IllegalStateException(
                            "retired Decode generation was not marked retired");
        };
    }

    private TerminalAction applyTimeoutTerminalLocked(
            DeferredTerminal timeout) {
        return beginTerminalizing(timeout,
                TerminalOutcome.timeout(timeout.detail()),
                buildErrorResponse(
                        this.timeoutErrorType(), timeout.detail()));
    }

    private TerminalAction applyFailureTerminalLocked(
            DeferredTerminal failure) {
        return beginTerminalizing(failure,
                TerminalOutcome.fail(failure.detail()),
                buildErrorResponse(failure.errorType(), failure.detail()));
    }

    private TerminalAction applyDecodeRejectionOrRetirementLocked(
            DeferredTerminal terminal) {
        String terminalDetail;
        if (terminal.kind() == DeferredTerminal.Kind.DELIVERY_REJECTED
                || terminal.kind()
                    == DeferredTerminal.Kind.DECODE_GENERATION_RETIRED) {
            terminalDetail = terminal.detail();
        } else {
            throw new IllegalArgumentException(
                    "Decode rejection/retirement reducer requires a rejection or retirement event");
        }
        String detail = terminalDetail == null
                ? "Decode endpoint generation retired"
                : terminalDetail;
        if (this.hasCancellationFirstCause()) {
            CancelReason firstCause = this.requireCancellationFirstCause();
            String cancellationDetail = firstCause.getMessage()
                    + "; " + detail;
            return beginTerminalizing(terminal,
                    TerminalOutcome.cancellation(firstCause, cancellationDetail),
                    buildErrorResponse(
                            this.cancellationErrorType(firstCause),
                            cancellationDetail));
        }
        return beginTerminalizing(terminal,
                TerminalOutcome.fail(detail),
                buildErrorResponse(
                        StrategyErrorType.DISPATCH_FAILED, detail));
    }

    private TerminalAction applyWorkerTerminalLocked(
            DeferredTerminal terminal) {
        if (this.hasCancellationFirstCause()) {
            String proof = terminal.workerSource()
                    == WorkerTerminalSource.PREFILL_ENDPOINT
                            ? "Prefill terminal observed after cancellation"
                            : "Decode terminal observed after cancellation";
            CancelReason firstCause = requireCancellationFirstCause();
            String message = firstCause.getMessage() + "; " + proof;
            return beginTerminalizing(terminal,
                    TerminalOutcome.cancellation(firstCause, message),
                    buildErrorResponse(cancellationErrorType(firstCause), message));
        }
        TerminalOutcome transition;
        Response response;
        if (terminal.workerSuccessful()) {
            transition = TerminalOutcome.complete("decode completed");
            ScheduledRequest item = this.activeItem();
            response = buildSuccessResponse(
                    item.routeResponse(), this.snapshot().deliveryClaimKind() == DeliveryClaimKind.BATCH_ENQUEUE);
        } else {
            String detail = "worker error code "
                    + terminal.workerErrorCode();
            transition = TerminalOutcome.fail(detail);
            response = buildErrorResponse(
                    StrategyErrorType.WORKER_EXECUTION_FAILED, detail);
        }
        return beginTerminalizing(terminal,
                transition,
                response);
    }

    /** Queue publication makes an exact item claimable even while admission still pins its resources. */
    private boolean ownsPreparedDelivery(ScheduledRequest exact) {
        requireSlotLock("delivery eligibility");
        return ownsActiveItem(exact) && isOpen() && preemption == null
                && state == RequestState.Phase.QUEUED && deliveryClaimKind == DeliveryClaimKind.NONE;
    }

    /** Execute detached work before selecting the response, preserving cancellation's publication race. */
    private void publishDelivery(DeliveryPublication delivery) {
        try {
            if (delivery.requestDeadline() != null) { expirationTimer.cancel(delivery.requestDeadline()); }
        } catch (Throwable failure) {
            Logger.error("Delivery deadline cancellation failed request_id={}", requestId, failure);
        }
        try {
            if (delivery.batchEnqueueStartedAtMs() > 0L && delivery.item().ctx().getAckAtMs() > 0L) {
                reporter.reportDispatchAckTimeMs(RoleType.PREFILL.name(),
                        delivery.item().prefillEp() == null ? "" : delivery.item().prefillEp().getIp(),
                        Math.max(0L, delivery.item().ctx().getAckAtMs() - delivery.batchEnqueueStartedAtMs()));
            }
        } catch (Throwable failure) {
            Logger.error("Delivery ACK reporting failed request_id={}", requestId, failure);
        }
        completionPublisher.submit(selectResponse(delivery.publication(), delivery.response()));
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

    private void cancelDecisionDeadline(
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

    private PublicationPermit finishExternalTerminal(TerminalOutcome transition) {
        TerminalAction action;
        synchronized (this) {
            if (!isCurrentGeneration() || !canClaimLocalTerminal()) { return null; }
            action = beginExternalTerminalizing(transition);
        }
        return action == null ? null : terminalCleanup.finishTerminal(action);
    }

    private void requireOutsideSlotLock(String operation) {
        if (Thread.holdsLock(this)) {
            throw new IllegalStateException(operation + " must run outside the RequestSlot lock");
        }
    }

    private RequestState commitTerminalStateLocked(TerminalOutcome outcome) {
        requireSlotLock("terminal state commitment");
        if (state.isTerminal()) { return snapshot(); }
        if (outcome.phase() == RequestState.Phase.CANCELLED && state != RequestState.Phase.CANCEL_REQUESTED) {
            transition(RequestState.Phase.CANCEL_REQUESTED, outcome.detail());
        }
        return transition(outcome.phase(), outcome.detail());
    }

    private enum SlotPhase {
        ACTIVE,
        TERMINALIZING,
        TERMINAL_RECORD
    }

    private enum EngineOwnership {
        DECODE_PENDING,
        DECODE_OWNED
    }

    enum PublicationKind {
        DELIVERY,
        TERMINAL
    }

    record DeliveryPublication(ScheduledRequest item, Response response, PublicationPermit publication,
            RequestDeadline requestDeadline, long batchEnqueueStartedAtMs) { }

    /** A completed request decision. Blocked facts never carry executable actions. */
    record RequestEffect(Status status, TerminalAction terminal, DeliveryPublication delivery,
            PreemptionRegistration signal) {
        static final RequestEffect STALE = new RequestEffect(Status.STALE, null, null, null);
        static final RequestEffect NONE = new RequestEffect(Status.NONE, null, null, null);

        RequestEffect {
            Objects.requireNonNull(status, "status");
            boolean hasAction = (terminal != null) != (delivery != null);
            if (status == Status.READY ? !hasAction : terminal != null || delivery != null || signal != null) {
                throw new IllegalArgumentException("request effect must contain exactly one selected action");
            }
        }

        static RequestEffect terminal(TerminalAction action, PreemptionRegistration signal) {
            return new RequestEffect(Status.READY, Objects.requireNonNull(action, "terminal action"), null, signal);
        }

        static RequestEffect delivery(DeliveryPublication delivery, PreemptionRegistration signal) {
            return new RequestEffect(Status.READY, null, Objects.requireNonNull(delivery, "delivery"), signal);
        }

        enum Status { STALE, NONE, READY }
    }

    record AdmissionHandleCompletion(
            boolean owned,
            CancelReason cancellationReason,
            DeferredTerminal pendingTerminal,
            TerminalAction pendingRetirement,
            boolean inactivityExpired) {
        private static final AdmissionHandleCompletion NOT_OWNED =
                new AdmissionHandleCompletion(
                        false, null, null, null, false);
    }

    record DecisionExpiry(
            ScheduledRequest item,
            boolean needsConfirmation) {
    }

    /** Timers detached atomically at ACTIVE -> TERMINALIZING. */
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
        private final PublicationKind kind;
        private final AtomicBoolean claimed = new AtomicBoolean();
        private final AtomicBoolean closed = new AtomicBoolean();

        PublicationPermit(
                RequestCompletionPublisher publisher,
                RequestSlot slot,
                PublicationKind kind) {
            this.publisher = Objects.requireNonNull(publisher, "publisher");
            this.slot = slot;
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

        private void claim() {
            if (!claimed.compareAndSet(false, true)) {
                throw new IllegalStateException(
                        "publication permit already consumed for request "
                                + slot.requestId);
            }
        }
    }

    SelectedPublication selectResponse(PublicationPermit permit, Response response) {
        return selectPublication(permit, ResponseCompletion.RESPONSE, response, null, false);
    }

    SelectedPublication selectFailure(PublicationPermit permit, Throwable failure) {
        return selectPublication(permit, ResponseCompletion.FAILURE, null, failure, false);
    }

    SelectedPublication selectCancellation(PublicationPermit permit, boolean mayInterruptIfRunning) {
        return selectPublication(permit, ResponseCompletion.CANCELLATION, null, null, mayInterruptIfRunning);
    }

    private SelectedPublication selectPublication(PublicationPermit permit, ResponseCompletion completion,
            Response response, Throwable failure, boolean mayInterruptIfRunning) {
        requireOutsideSlotLock("response selection");
        if (permit.slot != this || completion != ResponseCompletion.RESPONSE && permit.kind != PublicationKind.TERMINAL) {
            throw new IllegalArgumentException("incompatible publication permit");
        }
        permit.claim();
        try {
            synchronized (this) {
                return new SelectedPublication(permit, future, claimPublicationResult(permit.kind), completion,
                        response, failure, mayInterruptIfRunning);
            }
        } catch (RuntimeException | Error selectionFailure) {
            permit.closePublication();
            throw selectionFailure;
        }
    }

    private enum ResponseCompletion { RESPONSE, FAILURE, CANCELLATION }

    /** Immutable result of Slot arbitration. Execution never re-enters request decisions. */
    record SelectedPublication(PublicationPermit permit, RequestFuture future, boolean selected,
            ResponseCompletion completion, Response response, Throwable failure, boolean mayInterruptIfRunning) {
        boolean complete() {
            if (!selected) { return false; }
            return switch (completion) {
                case RESPONSE -> future.completeOwned(response);
                case FAILURE -> future.completeExceptionallyOwned(failure);
                case CANCELLATION -> future.cancelOwned(mayInterruptIfRunning);
            };
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

/** Endpoint that emitted the terminal fact; Decode facts follow its ledger update. */
enum WorkerTerminalSource {
    PREFILL_ENDPOINT(false),
    DECODE_ENDPOINT(true);

    private final boolean decodeTerminalAlreadyApplied;

    WorkerTerminalSource(boolean decodeTerminalAlreadyApplied) {
        this.decodeTerminalAlreadyApplied = decodeTerminalAlreadyApplied;
    }

    boolean decodeTerminalAlreadyApplied() {
        return decodeTerminalAlreadyApplied;
    }
}

/** Non-persistent decision produced by the RequestSlot acceptance transition. */
record DecodeAcceptance(
        ExpirationTimer.DecisionDeadline detachedDecisionDeadline) {
    static final DecodeAcceptance NONE =
            new DecodeAcceptance(null);
}

/** Terminal event, retained across admission/preemption and carried through lock-free cleanup. */
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
        DECODE_GENERATION_RETIRED,
        INACTIVITY_EXPIRED
    }

    DeferredTerminal {
        Objects.requireNonNull(kind, "kind");
        boolean valid = switch (kind) {
            case FAILURE, DELIVERY_FAILURE ->
                    errorType != null && workerSource == null;
            case WORKER -> errorType == null && workerSource != null;
            case TIMEOUT, INACTIVITY_EXPIRED, DELIVERY_REJECTED, PRIORITY,
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

    static DeferredTerminal inactivityExpired(String detail) {
        return new DeferredTerminal(Kind.INACTIVITY_EXPIRED, null, detail, null, false, 0L);
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

    boolean decodeTerminalAlreadyApplied() {
        return kind == Kind.WORKER
                && workerSource.decodeTerminalAlreadyApplied();
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
        DeferredTerminal event,
        TerminalOutcome transition,
        Response response,
        RequestSlot.PublicationPermit publication) {
}

/** Non-persistent proof that a claimed terminal action reached its terminal record. */
record TerminationResult(
        RequestState terminal,
        Throwable transitionFailure,
        RequestSlot.PublicationPermit publication) {
}

/** Stateless public-future adapter bound to one exact canonical slot. */
final class RequestFuture extends CompletableFuture<Response> {
    private final RequestSlot slot;

    RequestFuture(RequestSlot slot) {
        this.slot = slot;
    }

    @Override
    public boolean complete(Response response) {
        return slot.completeExternalResponse(response);
    }

    @Override
    public boolean completeExceptionally(Throwable error) {
        return slot.completeExternalFailure(error);
    }

    @Override
    public boolean cancel(boolean mayInterruptIfRunning) {
        return slot.cancelExternalFuture(mayInterruptIfRunning);
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
