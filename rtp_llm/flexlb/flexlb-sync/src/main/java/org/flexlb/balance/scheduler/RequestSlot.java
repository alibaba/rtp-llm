package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.CapacityBoundary;
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
import org.flexlb.balance.scheduler.RequestCompletionPublisher.PublicationPermit;
import org.flexlb.balance.scheduler.RequestCompletionPublisher.SelectedPublication;
import org.flexlb.balance.scheduler.RequestCompletionPublisher.ResponseCompletion;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;

import java.util.Objects;
import java.util.OptionalLong;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BooleanSupplier;

import static org.flexlb.dao.loadbalance.Response.buildErrorResponse;
import static org.flexlb.balance.scheduler.RequestTerminalCleanup.appendFailure;
import static org.flexlb.balance.scheduler.RequestTerminalCleanup.runStep;
import static org.flexlb.balance.scheduler.RequestTerminalCleanup.rethrowCleanup;
import static org.flexlb.dao.loadbalance.Response.buildSuccessResponse;

/**
 * One request's response decision and resource-tracking lifetime.
 *
 * <p>Read the entry points first: admission → dispatch → Worker evidence → cleanup.
 * Cancellation, deadlines and preemption enter the same lifecycle. A response may
 * finish before its resource obligations do; only the common close gate ends tracking.
 *
 * <p>Methods are grouped by function. Each group keeps its event entry, locked
 * decisions and outside-lock work together. Private decisions end in Locked;
 * component entry points acquire the monitor themselves.
 * Endpoint preparation/handoff stays atomic under this monitor. Ordinary cleanup,
 * metrics and Future completion execute after releasing it.
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
            Objects.requireNonNull(failure, "failure");
            if (failure.isSuccess()) { throw new IllegalArgumentException("admission termination requires a failure"); }
            if (resolved.compareAndSet(false, true)) {
                owner.finishAdmission(this, failure);
            }
        }

        /** Finish this admission attempt after success or without committed side effects. */
        @Override
        public void close() {
            if (resolved.compareAndSet(false, true)) {
                owner.finishAdmission(this, null);
            }
        }
    }

    /** Exact asynchronous delivery identity. All consumption is guarded by the owning slot. */
    public static final class DeliveryClaim {
        final RequestSlot slot;
        private final ScheduledRequest item;
        private final DeliveryClaimKind kind;
        private final long correlationId;
        /** Prevents repeated complete() calls; guarded by the owning slot. */
        private boolean completeCalled;

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

    /** Storage/cleanup ownership; distinct from the public request lifecycle. */
    private SlotPhase slotPhase = SlotPhase.ACTIVE;
    private EngineOwnership engineOwnership = EngineOwnership.DECODE_PENDING;
    private CancelReason cancellationReason;
    /** One frontend result may win before its unlocked future completion runs. */
    private PublicationKind publicationWinner;

    private boolean admissionOpen = true;
    private AdmissionHandle admissionHandle;
    /** Pins a queued route while Decode capacity is transferred to a higher priority owner. */
    private ScheduledRequest withdrawingRoute;
    private CancelReason pendingAdmissionCancelReason;
    private boolean pendingAdmissionInactivityExpired;
    private DeferredTerminal admissionPendingTerminal;
    private PendingPrefillRetirement admissionPendingPrefillRetirement;
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
    /** Resource cleanup only; request identity/result and preemption stay in their existing owners. */
    private CleanupProgress cleanupProgress;

    private static final class CleanupProgress {
        // PENDING blocks close before the first pass or while a requested follow-up has not started.
        enum Phase { PENDING, RUNNING, RUN_AGAIN, WAITING }

        final DeliveryResult.Status source;
        Phase phase = Phase.PENDING;
        boolean prefillSettled;
        boolean decodeSettled;
        /** Sticky decision: later Worker activity cannot revoke an already requested expiry. */
        boolean expired;

        CleanupProgress(ScheduledRequest item, DeliveryResult.Status source) {
            this.source = source;
            prefillSettled = item.prefillEp() == null;
            decodeSettled = item.decodeEp() == null || item.decodeReservation() == null;
        }

        boolean ready() {
            return phase == Phase.WAITING && prefillSettled && decodeSettled;
        }
    }

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
        this.future = new RequestFuture(this);
    }

    // ── 请求身份与状态：只读查询、生命周期条件和状态提交 ──

    long requestId() {
        return requestId;
    }

    RequestFuture future() {
        return future;
    }

    long createdAtMs() {
        return createdAtMs;
    }

    boolean ownsFuture(CompletableFuture<?> expected) {
        return future == expected;
    }

    synchronized RequestState snapshot() {
        return new RequestState(
                requestId, state, deliveryClaimKind, batchId,
                createdAtMs, updatedAtMs, detail);
    }

    /** Exact ACTIVE item, or null when this generation no longer owns one. */
    synchronized ScheduledRequest activeItem() {
        return ownsActiveGenerationLocked() ? item : null;
    }

    synchronized boolean ownsActiveItem(ScheduledRequest expected) {
        return ownsActiveGenerationLocked() && item == expected;
    }

    synchronized ScheduledRequest activeItemForReservation(long reservationToken) {
        DecodeEndpoint.ReservationHandle reservation =
                item == null ? null : item.decodeReservation();
        return ownsActiveGenerationLocked()
                && reservation != null
                && reservation.reservationToken() == reservationToken
                ? item : null;
    }

    synchronized boolean isOpen() {
        return slotPhase == SlotPhase.ACTIVE
                && admissionOpen
                && !future.isDone()
                && !state.isTerminal();
    }

    synchronized boolean isLiveGeneration() {
        return currentGeneration && slotPhase != SlotPhase.TERMINAL_RECORD;
    }

    synchronized boolean isRemovableTerminalRecord(long updatedBeforeMs) {
        return currentGeneration
                && slotPhase == SlotPhase.TERMINAL_RECORD
                && state.isTerminal()
                && updatedAtMs < updatedBeforeMs
                && item == null;
    }

    synchronized void detachGeneration() {
        if (!currentGeneration) {
            throw new IllegalStateException(
                    "request generation already detached: " + requestId);
        }
        currentGeneration = false;
    }

    private boolean ownsActiveGenerationLocked() {
        requireSlotLock("active generation lookup");
        return currentGeneration
                && slotPhase == SlotPhase.ACTIVE
                && !state.isTerminal();
    }

    private boolean ownsResourceTrackingLocked() {
        requireSlotLock("resource tracking lookup");
        return currentGeneration && slotPhase == SlotPhase.ACTIVE;
    }

    private boolean decodeOwnsRequestLocked() {
        return engineOwnership == EngineOwnership.DECODE_OWNED;
    }

    private boolean isTerminalRecordLocked() {
        return slotPhase == SlotPhase.TERMINAL_RECORD;
    }

    private void ensureTransitionAllowedLocked(RequestState.Phase next) {
        if (!state.canTransitionTo(next)) {
            throw new IllegalStateException(
                    "invalid request lifecycle transition "
                            + state + " -> " + next);
        }
    }

    private RequestState transitionLocked(
            RequestState.Phase next,
            String message) {
        if (state == next) {
            return snapshot();
        }
        ensureTransitionAllowedLocked(next);
        state = next;
        detail = message == null ? "" : message;
        updatedAtMs = System.currentTimeMillis();
        assertInvariantLocked();
        return snapshot();
    }

    private RequestState commitTerminalStateLocked(TerminalOutcome outcome) {
        requireSlotLock("terminal state commitment");
        if (state.isTerminal()) { return snapshot(); }
        if (outcome.phase() == RequestState.Phase.CANCELLED && state != RequestState.Phase.CANCEL_REQUESTED) {
            transitionLocked(RequestState.Phase.CANCEL_REQUESTED, outcome.detail());
        }
        return transitionLocked(outcome.phase(), outcome.detail());
    }

    /** Verify the aggregate at every mutation boundary. */
    private void assertInvariantLocked() {
        requireSlotLock("request slot invariant");
        if (withdrawingRoute != null && admissionHandle == null) {
            throw new IllegalStateException("route withdrawal has no admission handle for " + requestId);
        }
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

    // ── 准入：开始 → 发布调度结果 → 结束并结算暂存事实 ──

    synchronized AdmissionHandle tryBeginAdmissionHandle() {
        if (!ownsActiveGenerationLocked()
                || !isOpen()
                || item != null
                || admissionHandle != null
                || preemption != null) {
            return null;
        }
        AdmissionHandle exact = new AdmissionHandle(this);
        admissionHandle = exact;
        assertInvariantLocked();
        return exact;
    }

    synchronized AdmissionHandle tryBeginRouteWithdrawal(
            DecodeEndpoint endpoint, DecodeEndpoint.ReservationHandle reservation) {
        if (item == null || item.decodeEp() != endpoint
                || !Objects.equals(item.decodeReservation(), reservation)
                || !ownsPreparedDeliveryLocked(item) || admissionHandle != null
                || item.requestExpired(System.currentTimeMillis())) {
            return null;
        }
        admissionHandle = new AdmissionHandle(this);
        withdrawingRoute = item;
        return admissionHandle;
    }

    /** Called only after the exact Decode reservation has been atomically replaced. */
    void detachWithdrawnRoute(AdmissionHandle claim, ScheduledRequest exact) {
        // Never acquire the Prefill queue lock under the request monitor.
        if (!exact.prefillEp().removeQueued(exact, "DECODE_RESERVATION_YIELDED")) {
            throw new IllegalStateException("withdrawn route is no longer queued: " + requestId);
        }
        synchronized (this) {
            if (admissionHandle != claim || withdrawingRoute != exact || item != exact) {
                throw new IllegalStateException("route withdrawal lost its owner: " + requestId);
            }
            item = null;
            detail = "queued after Decode reservation withdrawal";
            updatedAtMs = System.currentTimeMillis();
        }
    }

    PlacementResult.Status commitRoute(ScheduledRequest exact, BooleanSupplier publication) {
        synchronized (this) {
            if (!tryBindItemForPublicationLocked(exact)) { return PlacementResult.Status.CLOSED; }
        }
        // Endpoint queue publication must not retain this monitor. Admission pins the binding.
        try {
            if (publication.getAsBoolean()) { return PlacementResult.Status.SUCCESS; }
        } catch (RuntimeException | Error failure) {
            try { synchronized (this) { rollbackItemPublicationLocked(exact); } }
            catch (RuntimeException | Error rollbackFailure) {
                if (rollbackFailure != failure) { failure.addSuppressed(rollbackFailure); }
            }
            throw failure;
        }
        synchronized (this) { rollbackItemPublicationLocked(exact); }
        return PlacementResult.Status.BLOCKED;
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
    private boolean tryBindItemForPublicationLocked(ScheduledRequest candidate) {
        requireSlotLock("request item publication begin");
        if (!ownsActiveGenerationLocked()
                || !isOpen()
                || item != null
                || admissionHandle == null
                || candidate.requestId() != requestId
                || candidate.future() != future) {
            return false;
        }
        item = candidate;
        assertInvariantLocked();
        return true;
    }

    /** Roll back only the exact binding whose queue publication did not commit. */
    private void rollbackItemPublicationLocked(ScheduledRequest exact) {
        requireSlotLock("request item publication rollback");
        if (item != exact || admissionHandle == null) {
            throw new IllegalStateException(
                    "request item publication ownership changed for "
                            + requestId);
        }
        item = null;
        assertInvariantLocked();
    }

    /** Close one admission and settle all facts retained while it owned the request. */
    private void finishAdmission(AdmissionHandle exact, Response failureResponse) {
        Throwable failure = null;
        try {
            RequestEffect effect = RequestEffect.STALE;
            boolean cleanupPending;
            synchronized (this) {
                if (admissionHandle == exact) {
                    admissionHandle = null;
                    withdrawingRoute = null;
                    effect = settleAdmissionLocked(failureResponse);
                }
                cleanupPending = cleanupProgress != null;
            }
            if (cleanupPending) {
                resumeCleanup();
            } else {
                execute(effect);
            }
        } catch (Throwable settlementFailure) {
            failure = settlementFailure;
        }
        // A failed settlement must not strand the expiry watch or the admission gate.
        failure = runStep(failure, () -> expirationTimer.attachInactivityDeadline(this));
        failure = runStep(failure, admissionFinished);
        rethrowCleanup(failure);
    }

    private RequestEffect settleAdmissionLocked(Response failure) {
        requireSlotLock("admission settlement");
        CancelReason pendingCancellation = pendingAdmissionCancelReason;
        boolean inactive = pendingAdmissionInactivityExpired;
        DeferredTerminal pending = admissionPendingTerminal;
        PendingPrefillRetirement retirement = admissionPendingPrefillRetirement;
        pendingAdmissionCancelReason = null;
        pendingAdmissionInactivityExpired = false;
        admissionPendingTerminal = null;
        admissionPendingPrefillRetirement = null;
        if (cancellationReason == null) { cancellationReason = pendingCancellation; }
        assertInvariantLocked();

        if (cleanupProgress != null) {
            cleanupProgress.expired |= inactive;
            if (pending != null && pending.decodeTerminalAlreadyApplied()) {
                cleanupProgress.decodeSettled = true;
                if (preemption != null) { preemption.tryFinish(); }
            }
            return RequestEffect.DEFERRED;
        }
        // Authoritative completion takes precedence over retirement and admission failure.
        if (pending != null && pending.authoritativeWorker()) {
            return processRequestEndLocked(item, pending);
        }
        TerminalAction retired = claimPrefillRetirementLocked(retirement);
        if (retired != null) { return RequestEffect.terminal(retired, null); }
        if (inactive && pendingCancellation != null) {
            return RequestEffect.terminal(decideRequestEndLocked(DeferredTerminal.inactivityExpired(
                    "REQUEST_INACTIVE: no matching Engine request status before inactivity timeout")), null);
        }

        RequestEffect effect = RequestEffect.APPLIED;
        if (pending != null) {
            effect = processRequestEndLocked(item, pending);
        } else if (failure != null) {
            String message = pendingCancellation != null ? pendingCancellation.getMessage()
                    : failure.getErrorMessage() == null ? "eviction admission failed" : failure.getErrorMessage();
            TerminalOutcome outcome = pendingCancellation == null ? TerminalOutcome.fail(message)
                    : TerminalOutcome.cancellation(pendingCancellation, message);
            Response response = pendingCancellation == null ? failure
                    : buildErrorResponse(cancellationErrorTypeLocked(pendingCancellation), message);
            effect = RequestEffect.terminal(claimTerminalActionLocked(null, outcome, response, true), null);
        }
        if (effect.terminal() != null || pendingCancellation == null || !ownsActiveGenerationLocked()) { return effect; }
        TerminalAction cancelled = pendingCancellation == CancelReason.DEADLINE_EXCEEDED
                || requestInactiveLocked(System.currentTimeMillis())
                ? decideRequestEndLocked(DeferredTerminal.inactivityExpired(pendingCancellation.getMessage()))
                : tryTerminateCancellationLocked();
        return cancelled == null ? effect : RequestEffect.terminal(cancelled, null);
    }

    private void retainAdmissionTerminalLocked(DeferredTerminal candidate) {
        DeferredTerminal previous = admissionPendingTerminal;
        if (previous == null || !previous.endpointAlreadyRetired()
                && (candidate.endpointAlreadyRetired() || !previous.authoritativeWorker() && candidate.authoritativeWorker())) {
            admissionPendingTerminal = candidate;
        }
    }

    private void retainAdmissionPrefillRetirementLocked(PendingPrefillRetirement candidate) {
        if (admissionPendingPrefillRetirement == null) {
            admissionPendingPrefillRetirement = candidate;
            return;
        }
        if (admissionPendingPrefillRetirement.source != candidate.source
                || admissionPendingPrefillRetirement.item != candidate.item) {
            throw new IllegalStateException("admission observed another Prefill generation for request " + requestId);
        }
    }

    // ── 派发：准备 → 交接 → 预测 → 接收结果 / 发布路由 ──

    /** Eligibility and preparation are one transaction; the operation must not publish or call user code. */
    synchronized <T> CapacityBoundary.Attempt<T> prepareDispatch(ScheduledRequest exact,
            java.util.function.Supplier<CapacityBoundary.Attempt<T>> prepare) {
        return ownsPreparedDeliveryLocked(exact) ? prepare.get()
                : CapacityBoundary.Attempt.rejected(CapacityBoundary.OWNERSHIP_LOST);
    }

    /** Atomically arbitrate expiry against the one endpoint-ownership handoff. */
    DeliveryClaim claimDelivery(ScheduledRequest exact, DeliveryClaimKind kind,
            long correlationId, BooleanSupplier transferToEndpoint) {
        TerminalAction expired;
        boolean cleanup;
        synchronized (this) {
            if (kind == DeliveryClaimKind.NONE || kind == DeliveryClaimKind.BATCH_ENQUEUE && correlationId <= 0L) {
                throw new IllegalArgumentException("invalid delivery identity");
            }
            if (!ownsPreparedDeliveryLocked(exact)) { return null; }
            // Read time under the same lock as the claim. Timer execution can
            // lag, so an unconsumed timer is not proof that this item is live.
            long nowMs = System.currentTimeMillis();
            if (requestInactiveLocked(nowMs)) {
                expired = decideInactivityLocked(nowMs);
            } else if (exact.requestExpired(nowMs)) {
                recordCancellationLocked(CancelReason.DEADLINE_EXCEEDED,
                        "request scheduling deadline exceeded before delivery");
                expired = tryTerminateCancellationLocked();
            } else {
                ensureTransitionAllowedLocked(RequestState.Phase.DISPATCHING);
                if (!transferToEndpoint.getAsBoolean()) {
                    throw new IllegalStateException("endpoint ownership lost for request " + requestId);
                }
                DeliveryClaim claim = new DeliveryClaim(this, exact, kind, correlationId);
                deliveryClaimKind = kind;
                batchId = correlationId;
                if (kind == DeliveryClaimKind.BATCH_ENQUEUE) { batchEnqueueStartedAtMs = nowMs; }
                transitionLocked(RequestState.Phase.DISPATCHING,
                        kind == DeliveryClaimKind.BATCH_ENQUEUE ? "batch enqueue started" : "route decision delivery started");
                return claim;
            }
            cleanup = cleanupProgress != null && cleanupProgress.expired;
        }
        // Endpoint cleanup and response callbacks must never run under Slot's lock.
        if (cleanup) { resumeCleanup(); }
        else { terminalCleanup.submitTerminal(expired); }
        return null;
    }

    /** Queue publication makes an exact item claimable even while admission still pins its resources. */
    private boolean ownsPreparedDeliveryLocked(ScheduledRequest exact) {
        requireSlotLock("delivery eligibility");
        return ownsActiveItem(exact) && isOpen() && preemption == null && withdrawingRoute == null
                && state == RequestState.Phase.QUEUED && deliveryClaimKind == DeliveryClaimKind.NONE;
    }

    private boolean ownsDeliveryClaimLocked(DeliveryClaim claim) {
        requireSlotLock("delivery identity");
        if (claim == null || claim.slot != this) { throw new IllegalArgumentException("foreign delivery claim"); }
        return ownsDeliveryClaimLocked(claim.item, claim.kind, claim.correlationId);
    }

    private boolean ownsDeliveryClaimLocked(
            ScheduledRequest expected,
            DeliveryClaimKind kind,
            long expectedBatchId) {
        requireSlotLock("delivery claim lookup");
        return ownsActiveItem(expected)
                && deliveryClaimKind == kind
                && batchId == expectedBatchId
                && !state.isTerminal();
    }

    void setDeliveryPrediction(DeliveryClaim claim, WorkSnapshot work, long predictedMs) {
        DecisionDeadline obsolete;
        synchronized (this) {
            if (!ownsDeliveryClaimLocked(claim)) { return; }
            obsolete = updateDeliveryPredictionLocked(work, predictedMs, System.currentTimeMillis());
        }
        expirationTimer.releaseDecisionDeadline(obsolete);
        expirationTimer.attachDecisionDeadline(this);
    }

    void publishRoute(DeliveryClaim claim, WorkSnapshot work, long predictedMs) {
        RequestEffect effect;
        DecisionDeadline obsolete;
        synchronized (this) {
            if (!ownsDeliveryClaimLocked(claim)) { return; }
            if (claim.kind != DeliveryClaimKind.ROUTE_DECISION) {
                throw new IllegalArgumentException("route publication requires a route claim");
            }
            obsolete = updateDeliveryPredictionLocked(work, predictedMs, System.currentTimeMillis());
            effect = acknowledgeDeliveryLocked(claim.correlationId, null);
        }
        executeEngineObservationEffects(new EngineObservation(effect, obsolete));
    }

    private void completeDelivery(DeliveryClaim claim, DeliveryResult result) {
        Objects.requireNonNull(result, "delivery result");
        SelectedPublication failureResponse = null;
        RequestEffect acknowledgement = RequestEffect.DEFERRED;
        boolean failed = result.failed();
        synchronized (this) {
            if (claim == null || claim.slot != this) {
                throw new IllegalArgumentException("foreign delivery claim");
            }
            if (claim.kind != DeliveryClaimKind.BATCH_ENQUEUE) {
                throw new IllegalArgumentException("complete() requires a batch delivery claim");
            }
            if (claim.completeCalled && ownsDeliveryClaimLocked(claim)) {
                throw new IllegalStateException("complete() already called for request " + requestId);
            }
            if (claim.completeCalled) { return; }
            claim.completeCalled = true;
            if (failed) {
                failureResponse = selectDeliveryFailureLocked(claim.item, result.status(),
                        "Delivery failed: " + detailOf(result.cause()));
            } else if (ownsDeliveryClaimLocked(claim)) {
                if (result.status() == DeliveryResult.Status.DELIVERED) {
                    claim.item.ctx().setAckAtMs(System.currentTimeMillis());
                    claim.item.ctx().setAckAtNanos(System.nanoTime());
                } else if (!decodeOwnsRequestLocked()) {
                    markAwaitingConfirmationLocked((result.status() == DeliveryResult.Status.TIMED_OUT
                            ? "Delivery timed out: " : "Delivery outcome uncertain: ") + detailOf(result.cause()));
                    return;
                }
                acknowledgement = acknowledgeDeliveryLocked(claim.correlationId, null);
            }
        }
        if (failed) {
            SelectedPublication response = failureResponse;
            Throwable failure = runStep(null,
                    response == null ? null : () -> completionPublisher.submit(response));
            failure = runStep(failure, () -> cleanUpRequest(claim.item, result.status()));
            rethrowCleanup(failure);
        } else {
            execute(acknowledgement);
        }
    }

    /** Shared confirmation decision for transport, Engine evidence and released preemption. */
    private RequestEffect acknowledgeDeliveryLocked(long expectedBatchId, PreemptionRegistration signal) {
        requireSlotLock("delivery acknowledgement");
        if (item == null || !ownsDeliveryClaimLocked(item, deliveryClaimKind, expectedBatchId)) { return RequestEffect.STALE; }
        if (cancellationReason != null || pendingAdmissionCancelReason != null) { return RequestEffect.DEFERRED; }
        PreemptionRegistration blocked = preemption;
        if (blocked != null) {
            if (blocked.isFinished()) { return RequestEffect.STALE; }
            blocked.recordDeliveryConfirmation(expectedBatchId);
            assertInvariantLocked();
            return processPendingEventsUnderPreemptionLocked(blocked, false, null);
        }
        if (state != RequestState.Phase.DISPATCHING) { return RequestEffect.STALE; }
        PublicationPermit permit = requirePublicationPermitLocked(PublicationKind.DELIVERY);
        try {
            Response response = buildSuccessResponse(item.routeResponse(), deliveryClaimKind == DeliveryClaimKind.BATCH_ENQUEUE);
            transitionLocked(RequestState.Phase.ACKNOWLEDGED, deliveryClaimKind == DeliveryClaimKind.BATCH_ENQUEUE
                    ? "batch enqueue acknowledged" : "route decision delivered");
            DeliveryPublication publication = new DeliveryPublication(item, response, permit,
                    requestDeadline, batchEnqueueStartedAtMs);
            requestDeadline = null;
            assertInvariantLocked();
            return RequestEffect.delivery(publication, signal);
        } catch (RuntimeException | Error failure) {
            permit.abandonIfUnclaimed();
            throw failure;
        }
    }

    void failDeliveryPreparation(ScheduledRequest exact, Throwable cause) {
        SelectedPublication response;
        synchronized (this) {
            if (!ownsPreparedDeliveryLocked(exact)) { return; }
            response = selectDeliveryFailureLocked(exact, DeliveryResult.Status.NOT_SENT,
                    "Delivery preparation failed: " + detailOf(cause));
        }
        Throwable failure = runStep(null,
                response == null ? null : () -> completionPublisher.submit(response));
        failure = runStep(failure,
                () -> cleanUpRequest(exact, DeliveryResult.Status.NOT_SENT));
        rethrowCleanup(failure);
    }

    private static String detailOf(Throwable cause) {
        if (cause == null) {
            return "unknown delivery failure";
        }
        String message = cause.getMessage();
        return message == null || message.isBlank()
                ? cause.getClass().getSimpleName() : message;
    }

    // ── 派发失败清理：选定响应 → 两侧结算 → 单调合并进度 ──

    /** Select the failure now; endpoint cleanup and protocol completion may follow later. */
    private SelectedPublication selectDeliveryFailureLocked(ScheduledRequest exact, DeliveryResult.Status source, String detail) {
        requireSlotLock("request failure");
        if (!ownsActiveItem(exact) || cleanupProgress != null) { return null; }
        CancelReason cancellation = cancellationReason != null ? cancellationReason : pendingAdmissionCancelReason;
        String message = cancellation == null ? detail : cancellation.getMessage() + "; " + detail;
        TerminalOutcome outcome = cancellation == null ? TerminalOutcome.fail(message)
                : TerminalOutcome.cancellation(cancellation, message);
        Response response = buildErrorResponse(cancellation == null ? StrategyErrorType.DISPATCH_FAILED
                : cancellationErrorTypeLocked(cancellation), message);
        PublicationPermit permit = publicationWinner == null && !future.isDone()
                ? requirePublicationPermitLocked(PublicationKind.TERMINAL) : null;
        cleanupProgress = new CleanupProgress(exact, source);
        admissionOpen = false;
        commitTerminalStateLocked(outcome);
        decisionExpiresAtMs = OptionalLong.empty();
        if (permit == null) { return null; }
        publicationWinner = PublicationKind.TERMINAL;
        permit.claim();
        return new SelectedPublication(permit, future, true, ResponseCompletion.RESPONSE, response, null, false);
    }

    /**
     * One failure path: settle each endpoint outside the slot lock, then close tracking.
     * Completion facts only move forward; an older "still owned" result cannot undo them.
     * Exact identities also let late callbacks settle resources after their slot is gone.
     */
    private void cleanUpRequest(ScheduledRequest exact, DeliveryResult.Status source) {
        requireOutsideSlotLock("request cleanup");
        CleanupProgress progress;
        boolean prefillDone;
        boolean decodeDone;
        boolean expired;
        RequestDeadline obsoleteRequest = null;
        DecisionDeadline obsoleteDecision = null;
        synchronized (this) {
            progress = item == exact ? cleanupProgress : null;
            if (progress != null && admissionHandle != null) { return; }
            if (progress != null && (progress.phase == CleanupProgress.Phase.RUNNING
                    || progress.phase == CleanupProgress.Phase.RUN_AGAIN)) {
                progress.phase = CleanupProgress.Phase.RUN_AGAIN;
                return;
            }
            if (progress != null) {
                progress.phase = CleanupProgress.Phase.RUNNING;
                obsoleteRequest = requestDeadline;
                requestDeadline = null;
                obsoleteDecision = detachDecisionDeadlineLocked();
            }
            prefillDone = progress != null && progress.prefillSettled;
            decodeDone = progress != null && progress.decodeSettled;
            expired = progress != null && progress.expired;
        }
        Throwable error = null;
        try {
            if (obsoleteRequest != null) { expirationTimer.cancel(obsoleteRequest); }
            expirationTimer.releaseDecisionDeadline(obsoleteDecision);
        } catch (Throwable problem) { error = problem; }
        try {
            if (!prefillDone) { terminalCleanup.settlePrefill(exact, expired); }
            prefillDone = true;
        } catch (Throwable problem) { error = appendFailure(error, problem); }
        try {
            if (!decodeDone) { decodeDone = terminalCleanup.settleDecode(exact, source, expired); }
        } catch (Throwable problem) { error = appendFailure(error, problem); }
        TerminalAction action = null;
        boolean resume = false;
        synchronized (this) {
            if (progress != null && cleanupProgress == progress) {
                resume = progress.phase == CleanupProgress.Phase.RUN_AGAIN;
                progress.phase = resume ? CleanupProgress.Phase.PENDING : CleanupProgress.Phase.WAITING;
                progress.prefillSettled |= prefillDone;
                progress.decodeSettled |= decodeDone;
                // A successful unsent rollback or local expiry also settles a local protocol owner.
                if (preemption != null && decodeDone && (expired || source == DeliveryResult.Status.NOT_SENT)) {
                    preemption.tryFinish();
                }
                action = tryCloseAfterCleanupLocked();
            }
        }
        try {
            if (resume) {
                cleanUpRequest(exact, source);
            } else {
                terminalCleanup.submitTerminal(action);
            }
        } catch (Throwable problem) {
            error = appendFailure(error, problem);
        }
        rethrowCleanup(error);
    }

    private void resumeCleanup() {
        ScheduledRequest exact;
        DeliveryResult.Status source;
        synchronized (this) {
            if (cleanupProgress == null) { return; }
            exact = item;
            source = cleanupProgress.source;
        }
        cleanUpRequest(exact, source);
    }

    /** Cleanup completion never reselects the response. */
    private TerminalAction tryCloseAfterCleanupLocked() {
        requireSlotLock("cleanup completion");
        if (cleanupProgress == null) { return null; }
        return claimTerminalActionLocked(null, new TerminalOutcome(state, detail), null, false);
    }

    // ── Worker 事实与 Endpoint 退出：接收、核验、推进 ──

    void processPrefillStatus(PrefillEndpoint source, RoleType role, PrefillState.WorkerStatusFact fact) {
        EngineObservation observation;
        RequestEffect work;
        boolean cleanupUnblocked;
        synchronized (this) {
            if (!currentGeneration) { return; }
            PreemptionRegistration previous = preemption;
            observation = applyPrefillStatusLocked(source, role, fact, System.currentTimeMillis());
            work = observation.transition();
            // A selected close action already owns cleanup; only a released protocol needs resuming.
            cleanupUnblocked = previous != null && preemption == null && cleanupProgress != null
                    && work.terminal() == null && work.delivery() == null;
        }
        if (cleanupUnblocked) {
            resumeCleanup();
        } else {
            executeEngineObservationEffects(observation);
        }
    }

    private EngineObservation applyPrefillStatusLocked(PrefillEndpoint source, RoleType role,
                                         PrefillState.WorkerStatusFact fact, long nowMs) {
        requireSlotLock("Prefill fact reduction");
        if (!ownsPrefillFactLocked(source, fact.item())) { return EngineObservation.STALE; }
        lastWorkerStatusAtMs = Math.max(lastWorkerStatusAtMs, nowMs);
        if (cleanupProgress != null) {
            if (fact.kind() == PrefillState.WorkerStatusFact.Kind.ACTIVE) {
                return new EngineObservation(applyPrefillActivityLocked(source, fact.item()), null);
            }
            cleanupProgress.prefillSettled = true;
            if (fact.kind() == PrefillState.WorkerStatusFact.Kind.PRIORITY_CANCELED
                    && preemption != null && !preemption.isFinished()) {
                return new EngineObservation(applyPriorityCancellationLocked(source, fact.item()), null);
            }
            return new EngineObservation(RequestEffect.terminal(tryCloseAfterCleanupLocked(), null), null);
        }
        RequestEffect transition = switch (fact.kind()) {
            case ACTIVE -> {
                if (decisionStage == DecisionStage.UNDECIDED || decisionStage == DecisionStage.WAITING_ENGINE) {
                    advanceDecisionLocked(DecisionStage.PREFILL_RUNNING, OptionalLong.empty());
                }
                yield applyPrefillActivityLocked(source, fact.item());
            }
            case COMPLETED -> {
                if (decisionStage != DecisionStage.ACCEPTED && decisionStage != DecisionStage.WAITING_DECODE) {
                    boolean separateDecode = role != RoleType.PDFUSION && item.decodeEp() != null;
                    prefillCompletedAtMs = nowMs;
                    advanceDecisionLocked(separateDecode ? DecisionStage.WAITING_DECODE : DecisionStage.ACCEPTED,
                            separateDecode && deliveryPredictionConsumed
                                    ? OptionalLong.of(deadlineAfter(nowMs, DECODE_HANDOFF_GRACE_MS))
                                    : OptionalLong.empty());
                }
                yield role == RoleType.PDFUSION
                        ? processRequestEndLocked(fact.item(), DeferredTerminal.worker(
                                WorkerTerminalSource.PREFILL_ENDPOINT, true, fact.errorCode()))
                        : RequestEffect.APPLIED;
            }
            case FAILED -> processRequestEndLocked(fact.item(), DeferredTerminal.worker(
                    WorkerTerminalSource.PREFILL_ENDPOINT, false, fact.errorCode()));
            case PRIORITY_CANCELED -> applyPriorityCancellationLocked(source, fact.item());
        };
        reconcileDecisionEvidenceLocked();
        return new EngineObservation(transition, detachObsoleteDecisionDeadlineLocked());
    }

    private boolean ownsPrefillFactLocked(PrefillEndpoint source, ScheduledRequest expected) {
        requireSlotLock("Prefill fact ownership lookup");
        return ownsResourceTrackingLocked() && item == expected && expected.prefillEp() == source;
    }

    void processDecodeStatus(DecodeEndpoint source, DecodeEndpoint.WorkerStatusFact fact) {
        EngineObservation observation;
        synchronized (this) {
            if (!currentGeneration) { return; }
            observation = applyDecodeStatusLocked(source, fact, System.currentTimeMillis());
        }
        executeEngineObservationEffects(observation);
    }

    private EngineObservation applyDecodeStatusLocked(DecodeEndpoint source, DecodeEndpoint.WorkerStatusFact fact, long nowMs) {
        requireSlotLock("Decode fact reduction");
        if (!ownsDecodeFactLocked(source, fact.reservation())) { return EngineObservation.STALE; }
        lastWorkerStatusAtMs = Math.max(lastWorkerStatusAtMs, nowMs);
        if (fact.kind() == DecodeEndpoint.WorkerStatusFact.Kind.TERMINAL) {
            if (cleanupProgress == null) {
                advanceDecisionLocked(DecisionStage.ACCEPTED, OptionalLong.empty());
                engineOwnership = EngineOwnership.DECODE_OWNED;
            }
            return new EngineObservation(processRequestEndLocked(item, DeferredTerminal.worker(
                    WorkerTerminalSource.DECODE_ENDPOINT, fact.errorCode() == 0L, fact.errorCode())),
                    detachObsoleteDecisionDeadlineLocked());
        }
        if (cleanupProgress != null) { return new EngineObservation(RequestEffect.APPLIED, null); }
        // Both a repeated ACTIVE observation and first ACCEPTED prove Decode ownership.
        DecodeAcceptance acceptance = markDecodeAcceptedLocked();
        return new EngineObservation(RequestEffect.APPLIED,
                acceptance.detachedDecisionDeadline());
    }

    private boolean ownsDecodeFactLocked(
            DecodeEndpoint source,
            DecodeEndpoint.ReservationHandle reservation) {
        requireSlotLock("Decode fact ownership lookup");
        return ownsResourceTrackingLocked()
                && item != null
                && item.decodeEp() == source
                && reservation.equals(item.decodeReservation());
    }

    /** Authoritative Decode ownership ends the decision-confirmation watch. */
    private DecodeAcceptance markDecodeAcceptedLocked() {
        requireSlotLock("Decode acceptance");
        if (!ownsActiveGenerationLocked()) {
            return DecodeAcceptance.NONE;
        }
        engineOwnership = EngineOwnership.DECODE_OWNED;
        advanceDecisionLocked(DecisionStage.ACCEPTED, OptionalLong.empty());
        reconcileDecisionEvidenceLocked();
        DecisionDeadline detachedDeadline = detachDecisionDeadlineLocked();
        assertInvariantLocked();
        return new DecodeAcceptance(detachedDeadline);
    }

    private void executeEngineObservationEffects(EngineObservation observation) {
        expirationTimer.releaseDecisionDeadline(observation.obsoleteDeadline());
        Throwable failure = runStep(null, () -> expirationTimer.attachDecisionDeadline(this));
        failure = runStep(failure, () -> execute(observation.transition()));
        rethrowCleanup(failure);
    }

    void recordPrefillRetirement(PrefillEndpoint source, ScheduledRequest exact) {
        String detail = "Prefill endpoint generation retired: " + source.ipPort()
                + "#" + source.getStatus().getGenerationId();
        TerminalAction action;
        synchronized (this) {
            if (cleanupProgress != null && ownsPrefillFactLocked(source, exact)) {
                cleanupProgress.prefillSettled = true;
                action = tryCloseAfterCleanupLocked();
            } else {
                action = claimPrefillRetirementLocked(new PendingPrefillRetirement(source, exact,
                        TerminalOutcome.fail(detail), buildErrorResponse(StrategyErrorType.DISPATCH_FAILED, detail)));
            }
        }
        terminalCleanup.submitTerminal(action);
    }

    private TerminalAction claimPrefillRetirementLocked(PendingPrefillRetirement pending) {
        requireSlotLock("Prefill retirement");
        if (pending == null || !ownsPrefillFactLocked(pending.source, pending.item)
                || engineOwnership == EngineOwnership.DECODE_OWNED
                || preemption != null || deliveryClaimKind.isClaimed()) {
            return null;
        }
        if (admissionHandle != null) {
            retainAdmissionPrefillRetirementLocked(pending);
            assertInvariantLocked();
            return null;
        }
        return claimTerminalActionLocked(null, pending.transition, pending.response, pending.response != null);
    }

    void recordDecodeRetirement(DecodeEndpoint source, DecodeEndpoint.ReservationHandle exact) {
        RequestEffect work;
        synchronized (this) {
            work = applyDecodeRetirementLocked(source, exact,
                            "Decode endpoint generation retired: generation=" + exact.endpointGenerationId());
        }
        execute(work);
    }

    private RequestEffect applyDecodeRetirementLocked(
            DecodeEndpoint source, DecodeEndpoint.ReservationHandle reservation, String detail) {
        requireSlotLock("Decode generation retirement reduction");
        Objects.requireNonNull(detail, "detail");
        if (!ownsDecodeFactLocked(source, reservation)) {
            return RequestEffect.STALE;
        }
        if (cleanupProgress != null) {
            cleanupProgress.decodeSettled = true;
            if (preemption != null) { preemption.tryFinish(); }
            return RequestEffect.terminal(tryCloseAfterCleanupLocked(), null);
        }
        DeferredTerminal terminal = DeferredTerminal.decodeGenerationRetired(detail);
        if (admissionHandle != null) {
            retainAdmissionTerminalLocked(terminal);
            assertInvariantLocked();
            return RequestEffect.DEFERRED;
        }

        PreemptionRegistration exact = preemption;
        PreemptionRegistration signal = null;
        if (exact != null) {
            retainPreemptionTerminalLocked(exact, terminal);
            exact.tryFinish();
            signal = exact;
        }
        detachPreemptionOwnerLocked(exact);

        assertInvariantLocked();
        return RequestEffect.terminal(decideRequestEndLocked(terminal), signal);
    }

    // ── 取消、调度失败与关闭：确定原因，再进入共同收尾 ──

    /** Return the resulting request snapshot; accepting cancellation does not imply immediate cleanup. */
    public RequestState cancelRequest(long expectedBatchId, CancelReason reason) {
        Objects.requireNonNull(reason, "reason");
        TerminalAction action = null;
        RequestState result;
        synchronized (this) {
            if (!currentGeneration || !snapshot().matchesBatch(expectedBatchId)) { return null; }
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
        if (!ownsActiveGenerationLocked() || state.isTerminal()
                || cancellationReason != null || pendingAdmissionCancelReason != null) {
            return false;
        }
        if (admissionHandle != null) {
            pendingAdmissionCancelReason = reason;
        } else {
            cancellationReason = reason;
        }
        admissionOpen = false;
        transitionLocked(RequestState.Phase.CANCEL_REQUESTED, message);
        assertInvariantLocked();
        return true;
    }

    /** Claim terminal ownership only when a recorded cancellation is locally reversible. */
    private TerminalAction tryTerminateCancellationLocked() {
        requireSlotLock("local cancellation termination");
        if (!ownsActiveGenerationLocked() || admissionHandle != null || cancellationReason == null) { return null; }
        ScheduledRequest active = activeItem();
        if (active != null && !canClaimLocalTerminalLocked()) { return null; }
        String message = cancellationReason.getMessage();
        return claimTerminalActionLocked(null, TerminalOutcome.cancellation(cancellationReason, message),
                buildErrorResponse(cancellationErrorTypeLocked(cancellationReason), message), true);
    }

    private boolean hasCancellationFirstCauseLocked() {
        return cancellationReason != null;
    }

    private CancelReason requireCancellationFirstCauseLocked() {
        if (cancellationReason == null) {
            throw new IllegalStateException(
                    "missing cancellation first cause for request " + requestId);
        }
        return cancellationReason;
    }

    private StrategyErrorType cancellationErrorTypeLocked(CancelReason reason) {
        requireSlotLock("cancellation error lookup");
        return reason == CancelReason.DEADLINE_EXCEEDED
                ? StrategyErrorType.BATCH_SLO_EXPIRED : StrategyErrorType.REQUEST_CANCELLED;
    }

    void recordSchedulingFailure(StrategyErrorType error, String detail) {
        RequestEffect work;
        synchronized (this) {
            work = processRequestEndLocked(activeItem(), DeferredTerminal.failure(error, detail));
        }
        execute(work);
    }

    TerminalAction claimShutdownAction() {
        synchronized (this) {
            if (!currentGeneration || !canClaimLocalTerminalLocked()) { return null; }
            String message = "request scheduler is shutting down";
            return claimTerminalActionLocked(null, TerminalOutcome.fail(message),
                    buildErrorResponse(StrategyErrorType.DISPATCH_FAILED, message), true);
        }
    }

    private boolean canClaimLocalTerminalLocked() {
        requireSlotLock("local terminal eligibility");
        return ownsActiveGenerationLocked()
                && !future.isDone()
                && admissionHandle == null
                && preemption == null
                && engineOwnership == EngineOwnership.DECODE_PENDING
                && state != RequestState.Phase.ACKNOWLEDGED
                && !deliveryClaimKind.isClaimed();
    }

    // ── 调度期限：安装与到期 ──

    synchronized boolean installRequestDeadline(RequestDeadline exact) {
        if (!ownsActiveGenerationLocked() || !isOpen()) {
            return false;
        }
        if (requestDeadline != null) {
            throw new IllegalStateException(
                    "request deadline already installed for " + requestId);
        }
        requestDeadline = exact;
        assertInvariantLocked();
        return true;
    }

    void onSchedulingDeadline(RequestDeadline exact) {
        TerminalAction action = null;
        synchronized (this) {
            if (requestDeadline != exact) { return; }
            requestDeadline = null;
            if (!ownsActiveGenerationLocked() || future.isDone() || !isOpen()) {
                assertInvariantLocked();
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
            assertInvariantLocked();
        }
        terminalCleanup.submitTerminal(action);
    }

    // ── 沉默期限：计划、安装、消费与显式检查 ──

    synchronized void configureInactivityTimeout(long timeoutMs) {
        if (timeoutMs <= 0L) {
            throw new IllegalArgumentException("request inactivity timeout must be positive");
        }
        inactivityTimeoutMs = timeoutMs;
    }

    synchronized OptionalLong inactivityDeadlineAtMs() {
        return ownsResourceTrackingLocked() && inactivityDeadline == null
                && (cleanupProgress == null || !cleanupProgress.expired)
                && pendingAdmissionCancelReason == null && inactivityTimeoutMs > 0L
                ? OptionalLong.of(inactivityExpiresAtMsLocked())
                : OptionalLong.empty();
    }

    synchronized boolean installInactivityDeadline(InactivityDeadline exact) {
        if (inactivityDeadlineAtMs().isEmpty()) {
            return false;
        }
        inactivityDeadline = exact;
        return true;
    }

    void onInactivityDeadline(InactivityDeadline exact, long nowMs) {
        TerminalAction action;
        boolean cleanup;
        synchronized (this) {
            if (!consumeInactivityDeadlineLocked(exact)) { return; }
            action = decideInactivityLocked(nowMs);
            cleanup = cleanupProgress != null && cleanupProgress.expired;
        }
        if (cleanup) {
            resumeCleanup();
        } else {
            terminalCleanup.submitTerminal(action);
        }
    }

    private boolean consumeInactivityDeadlineLocked(InactivityDeadline exact) {
        requireSlotLock("request inactivity check");
        if (inactivityDeadline != exact || !ownsResourceTrackingLocked()) {
            return false;
        }
        inactivityDeadline = null;
        return true;
    }

    void expireInactiveRequest(long nowMs) {
        TerminalAction action;
        boolean cleanup;
        synchronized (this) {
            action = decideInactivityLocked(nowMs);
            cleanup = cleanupProgress != null && cleanupProgress.expired;
        }
        if (cleanup) {
            resumeCleanup();
        } else {
            terminalCleanup.submitTerminal(action);
        }
    }

    private TerminalAction decideInactivityLocked(long nowMs) {
        if (cleanupProgress != null && ownsResourceTrackingLocked() && requestInactiveLocked(nowMs)) {
            cleanupProgress.expired = true;
            return null;
        }
        if (!currentGeneration || !ownsActiveGenerationLocked() || state.isTerminal() || !requestInactiveLocked(nowMs)) {
            return null;
        }
        String message = "REQUEST_INACTIVE: no matching Engine request status before inactivity timeout";
        recordCancellationLocked(CancelReason.DEADLINE_EXCEEDED, message);
        if (admissionHandle != null) {
            pendingAdmissionInactivityExpired = true;
            assertInvariantLocked();
            return null;
        }
        return decideRequestEndLocked(DeferredTerminal.inactivityExpired(message));
    }

    private boolean requestInactiveLocked(long nowMs) {
        return inactivityTimeoutMs > 0L
                && nowMs >= inactivityExpiresAtMsLocked();
    }

    /**
     * Before batch handoff, silence is measured from registration/status.
     * Handoff starts one bounded Engine-observation window; ACK does not renew
     * it. Afterwards only matching Worker facts extend it. Reuse the canonical
     * handoff timestamp rather than introducing another dispatch state/timer.
     */
    private long inactivityExpiresAtMsLocked() {
        long observedSince = lastWorkerStatusAtMs;
        if (deliveryClaimKind == DeliveryClaimKind.BATCH_ENQUEUE) {
            observedSince = Math.max(observedSince, batchEnqueueStartedAtMs);
        }
        return deadlineAfter(observedSince, inactivityTimeoutMs);
    }

    // ── 可见性期限：预测、Engine 证据、疑似丢失诊断 ──

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
        if (item.decodeEp() != null && item.decodeEp().isAcceptedByEngine(item.decodeReservation())) {
            return applyDecodeStatusLocked(item.decodeEp(),
                    DecodeEndpoint.WorkerStatusFact.accepted(item.decodeReservation()), nowMs).obsoleteDeadline();
        }
        return null;
    }

    private void advanceDecisionLocked(DecisionStage next, OptionalLong nextDeadline) {
        decisionStage = next;
        decisionExpiresAtMs = nextDeadline;
        decisionExpired = false;
    }

    synchronized OptionalLong decisionDeadlineAtMs() {
        return ownsActiveGenerationLocked() && decisionDeadline == null
                ? decisionExpiresAtMs : OptionalLong.empty();
    }

    synchronized boolean installDecisionDeadline(DecisionDeadline exact) {
        if (!ownsActiveGenerationLocked() || decisionDeadline != null
                || !decisionExpiresAtMs.equals(OptionalLong.of(exact.deadlineAtMs()))) {
            return false;
        }
        decisionDeadline = exact;
        return true;
    }

    synchronized DecisionExpiry onDecisionVisibilityDeadline(DecisionDeadline exact) {
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
        boolean needsConfirmation = needsDecisionConfirmationLocked();
        if (needsConfirmation) {
            markAwaitingConfirmationLocked(decisionStage == DecisionStage.WAITING_ENGINE
                    ? "no Engine request evidence before visibility deadline"
                    : "Decode acceptance missing after Prefill completion");
        }
        assertInvariantLocked();
        return new DecisionExpiry(item, needsConfirmation);
    }

    private boolean needsDecisionConfirmationLocked() {
        return ownsActiveGenerationLocked() && item != null
                && cancellationReason == null && pendingAdmissionCancelReason == null
                && (decisionExpired && (decisionStage == DecisionStage.WAITING_ENGINE
                    || decisionStage == DecisionStage.WAITING_DECODE));
    }

    /** Retain uncertainty as a diagnostic while normal ACK/status and TTL stay active. */
    private void markAwaitingConfirmationLocked(String message) {
        requireSlotLock("delivery confirmation wait");
        if (!ownsActiveGenerationLocked() || cancellationReason != null
                || pendingAdmissionCancelReason != null
                || decisionStage == DecisionStage.PREFILL_RUNNING
                || decisionStage == DecisionStage.ACCEPTED) {
            return;
        }
        detail = "SUSPECTED_LOST: " + Objects.requireNonNull(message, "message");
        updatedAtMs = System.currentTimeMillis();
        assertInvariantLocked();
    }

    /** Matching Engine evidence resolves the diagnostic suspicion. */
    private void reconcileDecisionEvidenceLocked() {
        requireSlotLock("decision evidence reconciliation");
        if (!ownsActiveGenerationLocked() || (decisionStage == DecisionStage.UNDECIDED || decisionStage == DecisionStage.WAITING_ENGINE)
                || needsDecisionConfirmationLocked() || cancellationReason != null
                || pendingAdmissionCancelReason != null) {
            return;
        }
        if (detail.startsWith("SUSPECTED_LOST")) {
            detail = "Engine request observed; waiting for completion";
        }
    }

    /** Detach an old phase's capability before arming the next phase. */
    private DecisionDeadline detachObsoleteDecisionDeadlineLocked() {
        requireSlotLock("decision deadline reconciliation");
        if (decisionDeadline == null || decisionExpiresAtMs.equals(
                OptionalLong.of(decisionDeadline.deadlineAtMs()))) {
            return null;
        }
        return detachDecisionDeadlineLocked();
    }

    private DecisionDeadline detachDecisionDeadlineLocked() {
        DecisionDeadline deadline = decisionDeadline;
        decisionDeadline = null;
        return deadline;
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

    // ── Timer 关闭：摘除本请求的全部定时器 ──

    /** Atomically detach all timer-owned capabilities during timer close. */
    synchronized ExpirationTimer.DetachedDeadlines detachDeadlinesForTimerClose() {
        ExpirationTimer.DetachedDeadlines detached =
                new ExpirationTimer.DetachedDeadlines(
                        requestDeadline, decisionDeadline, inactivityDeadline);
        requestDeadline = null;
        decisionDeadline = null;
        inactivityDeadline = null;
        assertInvariantLocked();
        return detached;
    }

    // ── 抢占协议：注册、进展、释放与完成 ──

    synchronized PreemptionRegistration tryInstallPreemption(
            long reservationToken,
            long attemptToken,
            String detail) {
        DecodeEndpoint.ReservationHandle reservation =
                item == null ? null : item.decodeReservation();
        if (!ownsActiveGenerationLocked()
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
        assertInvariantLocked();
        return preemption;
    }

    private PreemptionRegistration exactPreemptionLocked(
            PreemptionRegistration claim) {
        if (claim == null || claim.requestId() != requestId
                || preemption != claim) {
            return null;
        }
        return claim;
    }

    boolean updatePreemption(PreemptionRegistration claim, PreemptionCancelPhase next) {
        if (next == null) { return false; }
        RequestEffect work;
        synchronized (this) {
            work = applyPreemptionPhaseLocked(claim, next);
        }
        execute(work);
        return work.status() != RequestEffect.Status.STALE;
    }

    /** Advance one exact coordinator-owned Cancel phase. */
    private RequestEffect applyPreemptionPhaseLocked(
            PreemptionRegistration claim,
            PreemptionCancelPhase next) {
        requireSlotLock("preemption phase reduction");
        PreemptionRegistration exact = exactPreemptionLocked(claim);
        if (!ownsResourceTrackingLocked()
                || exact == null
                || preemption != exact
                || (next == PreemptionCancelPhase.CANCEL_IN_FLIGHT
                    && cancellationReason != null)
                || !exact.advanceTo(next)) {
            return RequestEffect.STALE;
        }
        if (next == PreemptionCancelPhase.CANCEL_REQUESTED) {
            if (!state.isTerminal()) { transitionLocked(RequestState.Phase.CANCEL_REQUESTED, exact.detail()); }
        }
        assertInvariantLocked();
        if (cleanupProgress != null) { return RequestEffect.terminal(tryCloseAfterCleanupLocked(), null); }
        return switch (next) {
            case CLAIMED -> RequestEffect.STALE;
            case CANCEL_IN_FLIGHT, CANCEL_REQUESTED ->
                    RequestEffect.APPLIED;
            case NOT_FOUND_STALE -> processPendingEventsUnderPreemptionLocked(exact, false, exact);
            case CANCEL_UNKNOWN ->
                    processPendingEventsUnderPreemptionLocked(exact, true, exact);
        };
    }

    boolean releasePreemption(PreemptionRegistration claim) {
        RequestEffect work;
        boolean cleanupPending;
        synchronized (this) {
            work = applyPreemptionReleaseLocked(claim);
            if (work.status() == RequestEffect.Status.STALE) {
                return false;
            }
            cleanupPending = cleanupProgress != null;
        }
        if (cleanupPending) {
            resumeCleanup();
        } else {
            execute(work);
        }
        return true;
    }

    private RequestEffect applyPreemptionReleaseLocked(
            PreemptionRegistration claim) {
        requireSlotLock("preemption release reduction");
        PreemptionRegistration exact = exactPreemptionLocked(claim);
        if (!ownsResourceTrackingLocked()
                || exact == null
                || preemption != exact
                || !exact.isReleasable()) {
            return RequestEffect.STALE;
        }
        detachPreemptionOwnerLocked(exact);
        if (cleanupProgress != null) { return RequestEffect.DEFERRED; }
        return processPendingEventsUnderPreemptionLocked(exact, false, exact);
    }

    boolean completePreemption(PreemptionRegistration claim, String detail) {
        RequestEffect work;
        synchronized (this) {
            work = applyPreemptionCompletedLocked(claim, detail);
        }
        execute(work);
        return work.status() != RequestEffect.Status.STALE;
    }

    private RequestEffect applyPreemptionCompletedLocked(
            PreemptionRegistration claim,
            String detail) {
        requireSlotLock("preemption completion reduction");
        PreemptionRegistration exact = exactPreemptionLocked(claim);
        if (!ownsResourceTrackingLocked()
                || exact == null
                || !exact.canCompletePreemption()
                || !exact.tryFinish()) {
            return RequestEffect.STALE;
        }
        if (cleanupProgress != null) {
            cleanupProgress.decodeSettled = true;
            return RequestEffect.terminal(tryCloseAfterCleanupLocked(), null);
        }
        DeferredTerminal terminal = DeferredTerminal.priority(detail);
        retainPreemptionTerminalLocked(exact, terminal);
        // DecodePreemptionCoordinator has already consumed the exact endpoint
        // claim before publishing REQUEST_FENCED. Reconciliation is therefore
        // neither required nor legal on this authoritative path.
        detachPreemptionOwnerLocked(exact);
        assertInvariantLocked();
        return RequestEffect.terminal(decideRequestEndLocked(terminal), exact);
    }

    /**
     * Reduce one exact transport/endpoint fact without exposing the mutable preemption
     * registration. The caller holds {@code synchronized (slot)} and only executes the returned,
     * already-selected effect.
     */
    private RequestEffect applyPrefillActivityLocked(PrefillEndpoint source, ScheduledRequest expected) {
        requireSlotLock("Prefill activity reduction");
        PreemptionRegistration exact = preemption;
        if (!ownsPrefillFactLocked(source, expected)) { return RequestEffect.STALE; }
        if (exact == null || !exact.isNotFound()) { return RequestEffect.APPLIED; }
        DecodeEndpoint decode = expected.decodeEp();
        if (decode != null && !decode.updatePreemption(
                exact.attemptToken(),
                DecodeEndpoint.PreemptionUpdate.active(expected.decodeReservation()))) {
            return RequestEffect.DEFERRED;
        }
        detachPreemptionOwnerLocked(exact);
        return RequestEffect.APPLIED;
    }

    private RequestEffect applyPriorityCancellationLocked(PrefillEndpoint source, ScheduledRequest expected) {
        requireSlotLock("priority cancellation reduction");
        PreemptionRegistration exact = ownsPrefillFactLocked(source, expected) ? preemption : null;
        DecodeEndpoint decode = expected.decodeEp();
        if (exact == null
                || exact.isFinished()
                || decode == null
                || expected.decodeReservation() == null
                || !decode.updatePreemption(exact.attemptToken(), DecodeEndpoint.PreemptionUpdate.canceled(expected.decodeReservation()))
                || !ownsResourceTrackingLocked()
                || preemption != exact
                || !exact.tryFinish()) {
            return RequestEffect.STALE;
        }
        if (cleanupProgress != null) {
            cleanupProgress.prefillSettled = true;
            cleanupProgress.decodeSettled = true;
            return RequestEffect.terminal(tryCloseAfterCleanupLocked(), null);
        }
        DeferredTerminal terminal = DeferredTerminal.priority("priority victim canceled by worker");
        retainPreemptionTerminalLocked(exact, terminal);
        detachPreemptionOwnerLocked(exact);
        assertInvariantLocked();
        return RequestEffect.terminal(decideRequestEndLocked(terminal), exact);
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
            // Decode emits completion after committing its ledger; Prefill still needs reconciliation.
            boolean decodeTerminalApplied = decode == null || terminal.decodeTerminalAlreadyApplied()
                    || decode.updatePreemption(exact.attemptToken(), DecodeEndpoint.PreemptionUpdate.finished(active.decodeReservation()));
            if (!decodeTerminalApplied) {
                return RequestEffect.DEFERRED;
            }
            exact.tryFinish();
            detachPreemptionOwnerLocked(exact);
            return RequestEffect.terminal(decideRequestEndLocked(terminal), signal);
        }
        if (transportUnknown || !exact.hasPendingDeliveryConfirmation()) {
            return RequestEffect.DEFERRED;
        }

        ScheduledRequest active = activeItem();
        DecodeEndpoint decode = active == null ? null : active.decodeEp();
        boolean activeWon = decode == null
                || decode.updatePreemption(exact.attemptToken(), DecodeEndpoint.PreemptionUpdate.active(active.decodeReservation()));
        if (!activeWon) {
            return RequestEffect.DEFERRED;
        }
        detachPreemptionOwnerLocked(exact);
        if (active == null) {
            return RequestEffect.STALE;
        }
        return acknowledgeDeliveryLocked(exact.pendingConfirmationBatchId(), signal);
    }

    private boolean detachPreemptionOwnerLocked(PreemptionRegistration exact) {
        requireSlotLock("preemption detach");
        if (exact == null || preemption != exact) {
            return false;
        }
        preemption = null;
        assertInvariantLocked();
        return true;
    }

    // ── 共同收尾：消费结束事实 → 认领清理责任 → 提交最终记录 ──

    /** Retain an end event while admission or preemption still owns the request. */
    private RequestEffect processRequestEndLocked(ScheduledRequest expected, DeferredTerminal event) {
        requireSlotLock("request end");
        boolean workerProof = event.authoritativeWorker();
        if (expected == null || !ownsResourceTrackingLocked() || item != expected
                || !workerProof && !ownsActiveItem(expected)) {
            return RequestEffect.STALE;
        }
        if (cleanupProgress != null) {
            if (event.decodeTerminalAlreadyApplied() || event.endpointAlreadyRetired()) {
                cleanupProgress.decodeSettled = true;
                if (preemption != null) { preemption.tryFinish(); }
            } else {
                cleanupProgress.prefillSettled = true;
            }
            return RequestEffect.terminal(tryCloseAfterCleanupLocked(), null);
        }
        if (admissionHandle != null) {
            retainAdmissionTerminalLocked(event);
            assertInvariantLocked();
            return RequestEffect.DEFERRED;
        }
        PreemptionRegistration exact = preemption;
        if (exact == null) { return RequestEffect.terminal(decideRequestEndLocked(event), null); }
        if (exact.isFinished()) { return RequestEffect.STALE; }
        retainPreemptionTerminalLocked(exact, event);
        if (workerProof) {
            if (!ownsActiveGenerationLocked() || preemption != exact) { return RequestEffect.STALE; }
        } else if (!exact.isNotFound() && !exact.isUnknown()) {
            return RequestEffect.DEFERRED;
        }
        assertInvariantLocked();
        return processPendingEventsUnderPreemptionLocked(exact, !workerProof && exact.isUnknown(), exact);
    }

    /** Map one end event to its request result; cleanup completion has its own entry. */
    private TerminalAction decideRequestEndLocked(DeferredTerminal event) {
        requireSlotLock("request completion");
        Objects.requireNonNull(event, "request end event");
        if (cleanupProgress != null) { return tryCloseAfterCleanupLocked(); }
        String message = event.detail();
        TerminalOutcome outcome;
        StrategyErrorType error;
        switch (event.kind()) {
            case FAILURE -> {
                outcome = TerminalOutcome.fail(message);
                error = event.errorType();
            }
            case TIMEOUT -> {
                outcome = TerminalOutcome.timeout(message);
                error = StrategyErrorType.BATCH_SLO_EXPIRED;
            }
            case INACTIVITY_EXPIRED -> {
                CancelReason cause = requireCancellationFirstCauseLocked();
                outcome = TerminalOutcome.cancellation(cause, message);
                error = cancellationErrorTypeLocked(cause);
            }
            case PRIORITY, DECODE_GENERATION_RETIRED, WORKER -> {
                if (event.kind() == DeferredTerminal.Kind.DECODE_GENERATION_RETIRED && message == null) {
                    message = "Decode endpoint generation retired";
                }
                if (hasCancellationFirstCauseLocked()) {
                    CancelReason cause = requireCancellationFirstCauseLocked();
                    if (event.kind() == DeferredTerminal.Kind.WORKER) {
                        message = event.workerSource() == WorkerTerminalSource.PREFILL_ENDPOINT
                                ? "Prefill terminal observed after cancellation"
                                : "Decode terminal observed after cancellation";
                    }
                    message = cause.getMessage()
                            + (event.kind() == DeferredTerminal.Kind.PRIORITY ? "" : "; " + message);
                    outcome = TerminalOutcome.cancellation(cause, message);
                    error = cancellationErrorTypeLocked(cause);
                } else if (event.kind() == DeferredTerminal.Kind.PRIORITY) {
                    outcome = TerminalOutcome.cancel(message);
                    error = StrategyErrorType.PRIORITY_PREEMPTED;
                } else if (event.kind() == DeferredTerminal.Kind.DECODE_GENERATION_RETIRED) {
                    outcome = TerminalOutcome.fail(message);
                    error = StrategyErrorType.DISPATCH_FAILED;
                } else if (event.workerSuccessful()) {
                    return claimTerminalActionLocked(event, TerminalOutcome.complete("decode completed"),
                            buildSuccessResponse(activeItem().routeResponse(),
                                    deliveryClaimKind == DeliveryClaimKind.BATCH_ENQUEUE), true);
                } else {
                    message = "worker error code " + event.workerErrorCode();
                    outcome = TerminalOutcome.fail(message);
                    error = StrategyErrorType.WORKER_EXECUTION_FAILED;
                }
            }
            default -> throw new IllegalStateException("unsupported request end: " + event.kind());
        }
        return claimTerminalActionLocked(event, outcome, buildErrorResponse(error, message), true);
    }

    /** The only close gate: no event may discard an outstanding cleanup obligation. */
    private TerminalAction claimTerminalActionLocked(DeferredTerminal event, TerminalOutcome transition, Response response,
            boolean requestPublication) {
        requireSlotLock("terminal claim");
        if (transition == null) {
            throw new IllegalStateException(
                    "terminal transition is required for request " + requestId);
        }
        if (!ownsResourceTrackingLocked() || admissionHandle != null
                || cleanupProgress != null && (!cleanupProgress.ready()
                    || preemption != null && !preemption.isFinished())) {
            return null;
        }
        boolean publishable = requestPublication
                && publicationWinner == null
                && !future.isDone();
        PublicationPermit publication = publishable
                ? requirePublicationPermitLocked(PublicationKind.TERMINAL) : null;
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
            DecisionDeadline detachedDecisionDeadline = detachDecisionDeadlineLocked();
            InactivityDeadline claimedInactivityDeadline = inactivityDeadline;
            inactivityDeadline = null;
            ExpirationTimer.DetachedRequestTimers terminalResources = new ExpirationTimer.DetachedRequestTimers(
                    claimedRequestDeadline, detachedDecisionDeadline, claimedInactivityDeadline);
            TerminalAction action = new TerminalAction(
                    this, item, deliveryClaimKind, cleanupProgress != null, claimedPreemption, terminalResources, event, transition,
                    publishable ? response : null, publication);
            transferred = true;
            assertInvariantLocked();
            return action;
        } finally {
            if (!transferred && publication != null) {
                publication.abandonIfUnclaimed();
            }
        }
    }

    synchronized void requireCleanupOwner(TerminalAction action) {
        if (slotPhase != SlotPhase.TERMINALIZING || action.slot() != this || action.item() != item) {
            throw new IllegalStateException("cleanup does not own request " + requestId);
        }
    }

    synchronized TerminationResult commitTerminalRecord(TerminalAction action) {
        if (!currentGeneration
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
        cleanupProgress = null;
        preemption = null;
        cancellationReason = null;
        admissionHandle = null;
        pendingAdmissionCancelReason = null;
        pendingAdmissionInactivityExpired = false;
        requestDeadline = null;
        decisionDeadline = null;
        inactivityDeadline = null;
        slotPhase = SlotPhase.TERMINAL_RECORD;
        assertInvariantLocked();
        return new TerminationResult(
                terminal,
                transitionFailure,
                action.publication());
    }

    private void execute(RequestEffect effect) {
        if (effect.terminal() == null && effect.delivery() == null) { return; }
        requireOutsideSlotLock("request effects");
        Runnable action = effect.terminal() != null
                ? () -> terminalCleanup.submitTerminal(effect.terminal())
                : () -> completionPublisher.submitDelivery(effect.delivery(), expirationTimer);
        Throwable failure = runStep(null, action);
        if (effect.signal() != null) {
            failure = runStep(failure,
                    () -> effect.signal().signalTerminal(new VictimTerminal(requestId)));
        }
        rethrowCleanup(failure);
    }

    // ── 响应：本地结束、结果仲裁与发布交接 ──

    boolean terminateLocallyAndPublishResponse(Response response) {
        PublicationPermit permit = terminateLocallyAndAcquirePublication(responseOutcome(response));
        if (permit == null) { return false; }
        completionPublisher.submit(selectPublication(permit, ResponseCompletion.RESPONSE, response, null, false));
        return true;
    }

    boolean completeExternal(ResponseCompletion completion, Response response, Throwable error, boolean interrupt) {
        requireOutsideSlotLock("external Future completion");
        TerminalOutcome outcome = switch (completion) {
            case RESPONSE -> responseOutcome(response);
            case FAILURE -> {
                Objects.requireNonNull(error, "error");
                yield TerminalOutcome.fail("external future failure"
                        + (error.getMessage() == null ? "" : ": " + error.getMessage()));
            }
            case CANCELLATION -> TerminalOutcome.cancel(CancelReason.CLIENT_CANCELLED.getMessage());
        };
        PublicationPermit permit = terminateLocallyAndAcquirePublication(outcome);
        return permit != null && completionPublisher.publishNow(
                selectPublication(permit, completion, response, error, interrupt));
    }

    private static TerminalOutcome responseOutcome(Response response) {
        String detail = response != null && response.getErrorMessage() != null
                ? response.getErrorMessage() : "external future completion";
        return response != null && !response.isSuccess() ? TerminalOutcome.fail(detail) : TerminalOutcome.complete(detail);
    }

    private PublicationPermit terminateLocallyAndAcquirePublication(TerminalOutcome transition) {
        TerminalAction action;
        synchronized (this) {
            if (!currentGeneration || !canClaimLocalTerminalLocked()) { return null; }
            action = claimTerminalActionLocked(null, transition, null, true);
        }
        return action == null ? null : terminalCleanup.finishTerminal(action);
    }

    private PublicationPermit requirePublicationPermitLocked(
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

    SelectedPublication selectPublication(PublicationPermit permit, ResponseCompletion completion,
            Response response, Throwable failure, boolean mayInterruptIfRunning) {
        requireOutsideSlotLock("response selection");
        if (permit.slot != this || completion != ResponseCompletion.RESPONSE && permit.kind != PublicationKind.TERMINAL) {
            throw new IllegalArgumentException("incompatible publication permit");
        }
        permit.claim();
        try {
            synchronized (this) {
                return new SelectedPublication(permit, future, claimPublicationResultLocked(permit.kind), completion,
                        response, failure, mayInterruptIfRunning);
            }
        } catch (RuntimeException | Error selectionFailure) {
            permit.closePublication();
            throw selectionFailure;
        }
    }

    /** Select the response under the slot lock; complete its future only after unlocking. */
    private boolean claimPublicationResultLocked(PublicationKind kind) {
        requireSlotLock("frontend result selection");
        if (future.isDone()) {
            return false;
        }
        if (kind == PublicationKind.TERMINAL) {
            return publicationWinner == PublicationKind.TERMINAL;
        }
        if (publicationWinner != null || !ownsActiveGenerationLocked()
                || state != RequestState.Phase.ACKNOWLEDGED
                || cancellationReason != null || pendingAdmissionCancelReason != null) {
            return false;
        }
        publicationWinner = PublicationKind.DELIVERY;
        return true;
    }

    void submitTerminalResponse(PublicationPermit permit, Response response) {
        completionPublisher.submit(selectPublication(permit, ResponseCompletion.RESPONSE, response, null, false));
    }

    // ── 锁边界断言 ──

    private void requireSlotLock(String operation) {
        if (!Thread.holdsLock(this)) {
            throw new IllegalStateException(
                    operation + " requires slot lock for request " + requestId);
        }
    }

    private void requireOutsideSlotLock(String operation) {
        if (Thread.holdsLock(this)) {
            throw new IllegalStateException(operation + " must run outside the RequestSlot lock");
        }
    }

    // ── 本类使用的数据类型 ──

    record EngineObservation(RequestEffect transition, DecisionDeadline obsoleteDeadline) {
        static final EngineObservation STALE = new EngineObservation(RequestEffect.STALE, null);
    }

    /** Exact Prefill retirement retained only by its in-flight admission. */
    private record PendingPrefillRetirement(
            PrefillEndpoint source,
            ScheduledRequest item,
            TerminalOutcome transition,
            Response response) {
    }

    /**
     * The terminal claim freezes the delivery stage until cleanup commits the terminal record.
     * Endpoint operations run without the slot monitor and validate their own exact ledger.
     */

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

    /** Event disposition and optional outside-lock work; APPLIED need not produce an action. */
    record RequestEffect(Status status, TerminalAction terminal, DeliveryPublication delivery,
            PreemptionRegistration signal) {
        static final RequestEffect STALE = new RequestEffect(Status.STALE, null, null, null);
        static final RequestEffect DEFERRED = new RequestEffect(Status.DEFERRED, null, null, null);
        static final RequestEffect APPLIED = new RequestEffect(Status.APPLIED, null, null, null);

        RequestEffect {
            Objects.requireNonNull(status, "status");
            if (terminal != null && delivery != null
                    || status != Status.APPLIED && (terminal != null || delivery != null || signal != null)) {
                throw new IllegalArgumentException("only applied events can produce one action");
            }
        }

        static RequestEffect terminal(TerminalAction action, PreemptionRegistration signal) {
            return action == null ? DEFERRED : new RequestEffect(Status.APPLIED, action, null, signal);
        }

        static RequestEffect delivery(DeliveryPublication delivery, PreemptionRegistration signal) {
            return new RequestEffect(Status.APPLIED, null, Objects.requireNonNull(delivery, "delivery"), signal);
        }

        enum Status { STALE, DEFERRED, APPLIED }
    }

    record DecisionExpiry(
            ScheduledRequest item,
            boolean needsConfirmation) {
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
        WORKER,
        PRIORITY,
        DECODE_GENERATION_RETIRED,
        INACTIVITY_EXPIRED
    }

    DeferredTerminal {
        Objects.requireNonNull(kind, "kind");
        boolean valid = switch (kind) {
            case FAILURE ->
                    errorType != null && workerSource == null;
            case WORKER -> errorType == null && workerSource != null;
            case TIMEOUT, INACTIVITY_EXPIRED, PRIORITY,
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
}

/** One-shot capability moved out of an ACTIVE slot; never stored or retried. */
record TerminalAction(
        RequestSlot slot,
        ScheduledRequest item,
        DeliveryClaimKind deliveryKind,
        boolean endpointsSettled,
        PreemptionRegistration preemption,
        ExpirationTimer.DetachedRequestTimers terminalResources,
        DeferredTerminal event,
        TerminalOutcome transition,
        Response response,
        RequestCompletionPublisher.PublicationPermit publication) {
}

/** Non-persistent proof that a claimed terminal action reached its terminal record. */
record TerminationResult(
        RequestState terminal,
        Throwable transitionFailure,
        RequestCompletionPublisher.PublicationPermit publication) {
}
