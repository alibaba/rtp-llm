package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.balance.scheduler.ExpirationTimer.AcceptanceDeadline;
import org.flexlb.balance.scheduler.ExpirationTimer.RequestDeadline;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;

import java.util.Objects;
import java.util.OptionalLong;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * Canonical aggregate root for one exact request generation.
 *
 * <p>Every mutable field is private and every method which observes or changes
 * lifecycle state requires {@code synchronized (slot)} unless its contract
 * explicitly says otherwise. Endpoint and transport code receives exact
 * capabilities, never a writable state field or a generic transition API.
 */
final class RequestSlot {

    private static final int OUTSTANDING_ADMISSION_CLOSED = -1;

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
    private boolean priorityAdmission;
    private StrategyErrorType deadlineErrorType =
            StrategyErrorType.BATCH_SLO_EXPIRED;

    /** Storage/cleanup ownership; distinct from the public request lifecycle. */
    private SlotPhase slotPhase = SlotPhase.ACTIVE;
    private EngineOwnership engineOwnership = EngineOwnership.DECODE_PENDING;
    private CancelReason cancellationReason;
    /** Last full WorkerStatus observation proving this generation is active. */
    private long lastWorkerStatusAtMs;

    private boolean admissionOpen = true;
    private AdmissionMutation admissionMutation;
    private CancelReason pendingAdmissionCancelReason;
    private AdmissionResources admissionResources;
    private RequestDeadline requestDeadline;
    private AcceptanceDeadline acceptanceDeadline;

    private AtomicInteger outstandingCounter;
    private boolean outstandingPermitReleaseRequested;
    private boolean outstandingPermitReleased;

    private PreemptionRegistration preemption;
    private EngineFenceRegistration engineFence;

    RequestSlot(
            RequestCompletionPublisher completionPublisher,
            long requestId) {
        this.completionPublisher = Objects.requireNonNull(
                completionPublisher, "completionPublisher");
        this.requestId = requestId;
        this.createdAtMs = System.currentTimeMillis();
        this.updatedAtMs = createdAtMs;
        this.future = new RequestFuture(completionPublisher, this);
        this.lastWorkerStatusAtMs = createdAtMs;
    }

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

    long lastWorkerStatusAtMs() {
        requireSlotLock("worker status activity lookup");
        return lastWorkerStatusAtMs;
    }

    void observeWorkerStatus(long observedAtMs) {
        requireSlotLock("worker status activity update");
        lastWorkerStatusAtMs = Math.max(lastWorkerStatusAtMs, observedAtMs);
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
    boolean tryBindItemForPublication(
            ScheduledRequest candidate,
            boolean priority) {
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
        priorityAdmission = priority;
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
        priorityAdmission = false;
        assertInvariant();
    }

    /** Roll back an unpublished item and its exact acceptance capability. */
    AdmissionCleanup rollbackAdmissionPublication(ScheduledRequest exact) {
        rollbackItemPublication(exact);
        AdmissionCleanup cleanup = detachAdmissionCleanup(true);
        assertInvariant();
        return cleanup;
    }

    /** Exact ACTIVE item, or null when this generation no longer owns one. */
    ScheduledRequest activeItem() {
        requireSlotLock("active item lookup");
        return ownsActiveGeneration() ? item : null;
    }

    boolean wasPriorityAdmission() {
        requireSlotLock("priority admission lookup");
        return priorityAdmission;
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
            boolean armAcceptanceDeadline = expected.decodeEp() != null;
            AdmissionCleanup detachedAdmission = armAcceptanceDeadline
                    ? null : detachAdmissionCleanup(true);
            DeliveryConfirmation result = new DeliveryConfirmation(
                    permit,
                    detachedRequestDeadline,
                    detachedAdmission,
                    armAcceptanceDeadline,
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
                && engineFence == null
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
                && engineFence == null
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
                || preemption != null
                || engineFence != null) {
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
        EngineFenceRegistration tombstonedFence =
                authoritativeTombstonedFence();
        DeferredTerminal pendingTerminal = tombstonedFence == null
                && admissionPendingTerminal != null
                && admissionPendingTerminal.authoritativeWorker()
                        ? admissionPendingTerminal : null;
        TerminalAction pendingRetirement = tombstonedFence == null
                && pendingTerminal == null
                        ? beginPendingPrefillRetirement(
                                admissionPendingPrefillRetirement)
                        : null;
        if (tombstonedFence == null
                && pendingTerminal == null
                && pendingRetirement == null) {
            pendingTerminal = admissionPendingTerminal;
        }
        if (tombstonedFence != null || pendingTerminal != null
                || pendingRetirement != null) {
            cancellationToResume = null;
        } else {
            cancellationToResume =
                    promoteAdmissionCancellation(cancellationToResume);
        }
        admissionPendingTerminal = null;
        admissionPendingPrefillRetirement = null;
        assertInvariant();
        return new AdmissionMutationCompletion(
                true, cancellationToResume, tombstonedFence, pendingTerminal,
                pendingRetirement);
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
     * coordinator resumes the cancellation effects outside the lock.
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
            Function<RequestSlot, RequestState> transition,
            Response response) {
    }

    // ==================== Exact deadline capabilities ====================

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

    /**
     * Return the delay needed by the semantic timer, or empty when acceptance
     * is already owned by Decode or no admission resource remains attached.
     */
    OptionalLong acceptanceDeadlineDelayMs() {
        requireSlotLock("acceptance deadline planning");
        if (!ownsActiveGeneration()
                || admissionResources == null
                || admissionResources.acceptanceTimeoutMs <= 0L
                || engineOwnership == EngineOwnership.DECODE_OWNED
                || acceptanceDeadline != null) {
            return OptionalLong.empty();
        }
        return OptionalLong.of(admissionResources.acceptanceTimeoutMs);
    }

    boolean installAcceptanceDeadline(AcceptanceDeadline exact) {
        requireSlotLock("acceptance deadline installation");
        if (!ownsActiveGeneration()
                || admissionResources == null
                || admissionResources.acceptanceTimeoutMs <= 0L
                || engineOwnership == EngineOwnership.DECODE_OWNED) {
            return false;
        }
        if (acceptanceDeadline != null) {
            throw new IllegalStateException(
                    "acceptance deadline already installed for " + requestId);
        }
        acceptanceDeadline = exact;
        assertInvariant();
        return true;
    }

    AcceptanceExpiry expireAcceptanceDeadline(AcceptanceDeadline exact) {
        requireSlotLock("acceptance deadline expiry");
        if (acceptanceDeadline != exact) {
            return null;
        }
        acceptanceDeadline = null;
        ScheduledRequest expected = item;
        boolean needsFence = ownsActiveGeneration()
                && expected != null
                && engineOwnership != EngineOwnership.DECODE_OWNED;
        AdmissionCleanup cleanup = detachAdmissionCleanup(false);
        assertInvariant();
        return new AcceptanceExpiry(expected, needsFence, cleanup);
    }

    /** Atomically detach both timer-owned capabilities during timer close. */
    ExpirationTimer.DetachedDeadlines detachDeadlinesForTimerClose() {
        requireSlotLock("deadline detach for timer close");
        ExpirationTimer.DetachedDeadlines detached =
                new ExpirationTimer.DetachedDeadlines(
                        requestDeadline, acceptanceDeadline);
        requestDeadline = null;
        acceptanceDeadline = null;
        assertInvariant();
        return detached;
    }

    /**
     * Consume the raw release callback at the boundary and immediately hide it
     * behind an exact, one-shot admission capability.
     *
     * @return cleanup to execute immediately when attachment lost a legal
     *         lifecycle race; null when the slot retained the capability
     */
    AdmissionCleanup bindAdmissionResources(
            Runnable releaseAction,
            long acceptanceTimeoutMs) {
        requireSlotLock("admission resource binding");
        if (acceptanceTimeoutMs < 0L) {
            throw new IllegalArgumentException(
                    "acceptanceTimeoutMs must be non-negative");
        }
        if (admissionResources != null) {
            // Ownership of releaseAction has not crossed into the slot.
            throw new IllegalStateException(
                    "admission resources already installed for " + requestId);
        }
        AdmissionResources exact = new AdmissionResources(
                releaseAction, acceptanceTimeoutMs);
        if (!ownsActiveGeneration() || !isOpen()) {
            return new AdmissionCleanup(exact, null);
        }
        admissionResources = exact;
        if (engineOwnership == EngineOwnership.DECODE_OWNED) {
            AdmissionCleanup cleanup = detachAdmissionCleanup(true);
            assertInvariant();
            return cleanup;
        }
        assertInvariant();
        return null;
    }

    private AdmissionCleanup detachAdmissionCleanup(
            boolean detachAcceptanceDeadline) {
        AdmissionResources resources = admissionResources;
        admissionResources = null;
        AcceptanceDeadline deadline = detachAcceptanceDeadline
                ? acceptanceDeadline : null;
        if (detachAcceptanceDeadline) {
            acceptanceDeadline = null;
        }
        return resources == null && deadline == null
                ? null : new AdmissionCleanup(resources, deadline);
    }

    // ==================== Outstanding request permit ====================

    boolean bindOutstandingPermit(AtomicInteger counter) {
        requireSlotLock("outstanding permit binding");
        if (outstandingCounter != null) {
            throw new IllegalStateException(
                    "outstanding permit already bound");
        }
        outstandingCounter = counter;
        if (outstandingPermitReleaseRequested || future.isDone()) {
            releaseOutstandingPermitLocked();
            assertInvariant();
            return false;
        }
        assertInvariant();
        return true;
    }

    void releaseOutstandingPermit() {
        synchronized (this) {
            outstandingPermitReleaseRequested = true;
            releaseOutstandingPermitLocked();
            assertInvariant();
        }
    }

    private void releaseOutstandingPermitLocked() {
        if (outstandingCounter == null || outstandingPermitReleased) {
            return;
        }
        outstandingPermitReleased = true;
        while (true) {
            int current = outstandingCounter.get();
            if (current == OUTSTANDING_ADMISSION_CLOSED) {
                assertInvariant();
                return;
            }
            if (current <= 0) {
                throw new IllegalStateException(
                        "outstanding request permit counter underflow");
            }
            if (outstandingCounter.compareAndSet(current, current - 1)) {
                assertInvariant();
                return;
            }
        }
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

    RequestState rememberCancellation(
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
                || engineFence != null
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
                requestId, attemptToken, detail);
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
            case NOT_FOUND_STALE -> {
                EngineFenceRegistration started = null;
                if (exact.postDeliveryFenceDetail() != null) {
                    try {
                        started = installEngineFence(
                                cancellationReason == null
                                        ? EngineFenceCause.DELIVERY_UNCERTAIN
                                        : EngineFenceCause.CANCELLATION,
                                exact.postDeliveryFenceDetail(),
                                exact,
                                cancellationReason != null,
                                true);
                    } catch (NotFoundFenceTransferLost legalRace) {
                        started = null;
                    }
                }
                yield started == null
                        ? materializePendingReplay(exact, false, exact)
                        : PreemptionReduction.startFence(
                                started, cancelTarget(item));
            }
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
        boolean discardable = cancellationReason == null
                && engineOwnership == EngineOwnership.DECODE_OWNED;
        if (exact.postDeliveryFenceDetail() == null || discardable) {
            detachPreemptionOwner(exact);
            return materializePendingReplay(exact, false, exact);
        }
        EngineFenceRegistration started = cancellationReason == null
                ? installEngineFence(
                        EngineFenceCause.DELIVERY_UNCERTAIN,
                        exact.postDeliveryFenceDetail(),
                        exact,
                        false,
                        false)
                : installEngineFence(
                        EngineFenceCause.CANCELLATION,
                        exact.postDeliveryFenceDetail(),
                        exact,
                        true,
                        false);
        return started == null
                ? PreemptionReduction.STALE
                : PreemptionReduction.startFence(
                        started, cancelTarget(item));
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
                exact,
                null);
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
        EngineFenceRegistration started = null;
        if (exact.postDeliveryFenceDetail() != null) {
            try {
                started =
                        installEngineFence(
                                cancellationReason == null
                                        ? EngineFenceCause.DELIVERY_UNCERTAIN
                                        : EngineFenceCause.CANCELLATION,
                                exact.postDeliveryFenceDetail(),
                                exact,
                                cancellationReason != null,
                                true);
            } catch (NotFoundFenceTransferLost legalRace) {
                started = null;
            }
        }
        if (started != null) {
            return PreemptionReduction.startFence(started, cancelTarget(item));
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
            return PreemptionReduction.replay(PendingReplay.terminal(terminal), null, null);
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
        return PreemptionReduction.replay(PendingReplay.terminal(terminal), signal, null);
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
        if (engineFence != null) {
            return PreemptionReduction.NONE;
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
                                null,
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
            return PreemptionReduction.replay(PendingReplay.terminal(terminal), null, null);
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
        return PreemptionReduction.replay(PendingReplay.terminal(terminal), exact, null);
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

        ExactPrefillOnlyCleanup prefillCleanup = null;
        EngineFenceRegistration retiredFence = engineFence;
        if (retiredFence != null) {
            retiredFence.close();
            engineFence = null;
            prefillCleanup = retiredFence.resources.detachAfterDecodeGenerationRetired();
        }
        assertInvariant();
        return PreemptionReduction.replay(
                PendingReplay.terminal(terminal), signal, prefillCleanup);
    }

    PreemptionReduction reduceDeliveryConfirmed(long batchId) {
        requireSlotLock("delivery confirmation reduction");
        ScheduledRequest active = activeItem();
        DeliveryClaimKind deliveryKind = deliveryClaimKind;
        if (active == null || !ownsDeliveryClaim(active, deliveryKind, batchId)) {
            return PreemptionReduction.STALE;
        }
        if (engineFence != null) {
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
                            null,
                            null);
        }
        if (exact.isSettled()) {
            return PreemptionReduction.STALE;
        }
        exact.recordDeliveryConfirmation(batchId);
        assertInvariant();
        return materializePendingReplay(exact, false, null);
    }

    /** Install or join the canonical delivery-uncertainty fence. */
    FenceReduction requestDeliveryFence(String detail) {
        return requestFence(EngineFenceCause.DELIVERY_UNCERTAIN, detail, false);
    }

    /** Install or join the canonical cancellation fence. */
    FenceReduction requestCancellationFence(String detail) {
        return requestFence(EngineFenceCause.CANCELLATION, detail, true);
    }

    private FenceReduction requestFence(
            EngineFenceCause cause, String detail, boolean allowDecodeOwned) {
        requireSlotLock("Engine fence request");
        if (preemption != null) {
            PreemptionRegistration exact = preemption;
            exact.requirePostDeliveryFence(detail);
            if (!exact.isNotFound()) {
                assertInvariant();
                return FenceReduction.NONE;
            }
            EngineFenceRegistration transferred;
            try {
                transferred = installEngineFence(cause, detail, exact, allowDecodeOwned, true);
            } catch (NotFoundFenceTransferLost legalRace) {
                return FenceReduction.STALE;
            }
            return transferred == null
                    ? FenceReduction.STALE
                    : FenceReduction.start(transferred, cancelTarget(item));
        }
        EngineFenceRegistration installed =
                installEngineFence(cause, detail, null, allowDecodeOwned, false);
        return installed == null
                ? FenceReduction.STALE
                : FenceReduction.start(installed, cancelTarget(item));
    }

    /** Reduce one event against the exact Engine-fence owner. */
    FenceReduction applyFenceUpdate(
            EngineFenceRegistration handle, FenceUpdate update) {
        requireSlotLock("Engine fence update reduction");
        EngineFenceRegistration exact = exactFence(handle);
        if (!ownsActiveGeneration() || exact == null || update == null) {
            return FenceReduction.STALE;
        }
        return switch (update) {
            case CANCEL_STARTED -> {
                if (!exact.beginCancel()) {
                    yield FenceReduction.STALE;
                }
                assertInvariant();
                yield FenceReduction.NONE;
            }
            case AWAIT_TERMINAL -> {
                if (!exact.awaitTerminal()) {
                    yield FenceReduction.STALE;
                }
                assertInvariant();
                yield FenceReduction.NONE;
            }
            case TOMBSTONED -> applyFenceTombstoned(exact);
        };
    }

    private FenceReduction applyFenceTombstoned(
            EngineFenceRegistration exact) {
        if (!exact.recordTombstoned() && !exact.isClosed()) {
            return FenceReduction.STALE;
        }
        assertInvariant();
        if (admissionMutation != null) {
            return FenceReduction.DEFERRED;
        }
        return FenceReduction.terminalProof(
                new FenceTerminalProof(
                        exact.detail,
                        exact.transferredPreemption,
                        exact.resources.decodeAuthoritativeTerminalProof(),
                        exact));
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
                    signal,
                    null);
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
                        signal,
                    null);
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
        if (preemption != null) {
            return preemption;
        }
        return engineFence == null
                ? null : engineFence.transferredPreemption;
    }

    // ==================== Engine fence sub-state machine ====================

    private EngineFenceRegistration installEngineFence(
            EngineFenceCause cause,
            String detail,
            PreemptionRegistration expectedPreemption,
            boolean allowDecodeOwned,
            boolean transferNotFoundDecodeClaim) {
        requireSlotLock("Engine fence installation");
        if (!canInstallEngineFence(
                expectedPreemption,
                allowDecodeOwned,
                transferNotFoundDecodeClaim)) {
            return null;
        }

        EngineFenceResources resources = transferNotFoundDecodeClaim
                ? acquireNotFoundFenceResources(expectedPreemption)
                : acquireFenceResources();
        boolean attached = false;
        try {
            // Resource acquisition may call re-entrant endpoint code.
            // Revalidate exact ownership before attaching the capability.
            if (!canInstallEngineFence(
                    expectedPreemption,
                    allowDecodeOwned,
                    transferNotFoundDecodeClaim)) {
                return null;
            }
            EngineFenceRegistration installed =
                    new EngineFenceRegistration(
                            cause,
                            detail == null ? cause.name() : detail,
                            expectedPreemption,
                            resources);
            engineFence = installed;
            if (expectedPreemption != null) {
                preemption = null;
            }
            attached = true;
            assertInvariant();
            return installed;
        } finally {
            if (!attached) {
                resources.release();
            }
        }
    }

    private boolean canInstallEngineFence(
            PreemptionRegistration expectedPreemption,
            boolean allowDecodeOwned,
            boolean requireNotFoundPreemption) {
        return ownsActiveGeneration()
                && admissionMutation == null
                && engineFence == null
                && preemption == expectedPreemption
                && (expectedPreemption == null
                    || expectedPreemption.isFenceTransferable())
                && (!requireNotFoundPreemption
                    || (expectedPreemption != null
                        && expectedPreemption.isNotFound()))
                && (allowDecodeOwned || cancellationReason == null)
                && (allowDecodeOwned
                    || engineOwnership != EngineOwnership.DECODE_OWNED);
    }

    /** Acquire every exact fence leaf only after aggregate ownership is valid. */
    private EngineFenceResources acquireFenceResources() {
        requireSlotLock("Engine fence resource acquisition");
        ScheduledRequest active = activeItem();
        if (active == null) {
            throw new IllegalStateException(
                    "Engine fence requires active item for request " + requestId);
        }
        PrefillEndpoint prefill = active.prefillEp();
        PrefillState.Protection protection = null;
        if (prefill != null) {
            protection = deliveryClaimKind
                            == DeliveryClaimKind.BATCH_ENQUEUE
                            && batchId > 0L
                    ? prefill.acquireBatchMemberProtection(
                            batchId, active)
                    : prefill.acquireEngineFenceProtection(active);
        }
        return EngineFenceResources.acquire(active, protection);
    }

    /**
     * Move the exact NOT_FOUND Decode claim into the newly acquired fence.
     * A losing endpoint race releases every fresh leaf before reporting stale.
     */
    private EngineFenceResources acquireNotFoundFenceResources(
            PreemptionRegistration exact) {
        EngineFenceResources resources = acquireFenceResources();
        ScheduledRequest active = activeItem();
        DecodeEndpoint decode = active == null ? null : active.decodeEp();
        boolean transferred = decode == null
                || (engineOwnership == EngineOwnership.DECODE_OWNED
                    ? decode.reconcilePriorityVictimActive(
                            exact.attemptToken(),
                            active.decodeReservation())
                    : decode.transferPriorityNotFoundClaimToEngineFence(
                            exact.attemptToken(),
                            active.requestId()));
        if (transferred) {
            return resources;
        }
        resources.release();
        throw NotFoundFenceTransferLost.INSTANCE;
    }

    private EngineFenceRegistration exactFence(
            EngineFenceRegistration handle) {
        return engineFence == handle ? handle : null;
    }

    private static CancelTarget cancelTarget(ScheduledRequest active) {
        ServerStatus prefill = active == null ? null : active.prefill();
        return prefill == null ? null
                : new CancelTarget(
                        prefill.getServerIp(), prefill.getGrpcPort());
    }

    private EngineFenceRegistration authoritativeTombstonedFence() {
        return engineFence != null && engineFence.isClosed()
                ? engineFence : null;
    }

    private EngineFenceRegistration closeEngineFence(
            EngineFenceRegistration exact) {
        if (exact == null || engineFence != exact) {
            return null;
        }
        if (!exact.isClosed() && !exact.close()) {
            return null;
        }
        engineFence = null;
        assertInvariant();
        return exact;
    }

    /**
     * Commit authoritative Decode ownership and atomically decide whether an
     * unsent fence can be released. Fences which crossed the Cancel entry
     * boundary remain the canonical owner until authoritative terminal proof.
     */
    DecodeAcceptance markDecodeAccepted() {
        requireSlotLock("Decode acceptance");
        if (!ownsActiveGeneration()) {
            return DecodeAcceptance.NONE;
        }
        engineOwnership = EngineOwnership.DECODE_OWNED;
        EngineFenceRegistration fence = engineFence;
        if (fence == null) {
            DecodeAcceptance accepted = new DecodeAcceptance(
                    null, detachAdmissionCleanup(true));
            assertInvariant();
            return accepted;
        }
        if (cancellationReason != null
                || fence.transferredPreemption != null
                || fence.cancelMayHaveBeenInstalled()) {
            assertInvariant();
            return DecodeAcceptance.NONE;
        }
        EngineFenceRegistration releasable = closeEngineFence(fence);
        DecodeAcceptance accepted = new DecodeAcceptance(
                releasable, detachAdmissionCleanup(true));
        assertInvariant();
        return accepted;
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
            Function<RequestSlot, RequestState> transition,
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
                && engineFence == null
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
            Function<RequestSlot, RequestState> transition,
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
            Function<RequestSlot, RequestState> transition) {
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
            Function<RequestSlot, RequestState> transition,
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
                && state != RequestState.Phase.ACKNOWLEDGED
                && !future.isDone();
        PublicationPermit publication = publishable
                ? requirePublicationPermit(PublicationKind.TERMINAL) : null;
        boolean transferred = false;
        try {
            slotPhase = SlotPhase.TERMINALIZING;
            admissionOpen = false;

            EngineFenceRegistration claimedFence = engineFence;
            if (claimedFence != null) {
                if (!claimedFence.isClosed() && !claimedFence.close()) {
                    throw new IllegalStateException(
                            "terminal claim found closed Engine fence");
                }
                engineFence = null;
            }

            RequestDeadline claimedRequestDeadline = requestDeadline;
            requestDeadline = null;
            AdmissionCleanup admissionCleanup = detachAdmissionCleanup(true);
            TerminalResources terminalResources =
                    claimedRequestDeadline == null && admissionCleanup == null
                            ? null
                            : new TerminalResources(
                                    claimedRequestDeadline, admissionCleanup);
            TerminalAction action = new TerminalAction(
                    this,
                    item,
                    claimedFence,
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
            terminal = action.transition().apply(this);
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
        priorityAdmission = false;
        preemption = null;
        engineFence = null;
        cancellationReason = null;
        admissionMutation = null;
        pendingAdmissionCancelReason = null;
        admissionResources = null;
        requestDeadline = null;
        acceptanceDeadline = null;
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
        if (preemption != null && engineFence != null) {
            throw new IllegalStateException(
                    "preemption and Engine fence cannot directly own request "
                            + requestId);
        }
        if (admissionMutation != null
                && (preemption != null || engineFence != null)) {
            throw new IllegalStateException(
                    "admission mutation overlaps preemption or Engine fence for "
                            + requestId);
        }
        // A CLOSED fence may remain attached only as the authoritative
        // TOMBSTONED proof between the transport callback and the total
        // terminal reducer. No separate boolean mirrors that ownership.
        if (pendingAdmissionCancelReason != null
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
        if (acceptanceDeadline != null && admissionResources == null) {
            throw new IllegalStateException(
                    "acceptance deadline has no admission resources for "
                            + requestId);
        }
        if (slotPhase != SlotPhase.TOMBSTONE) {
            return;
        }
        if (admissionOpen
                || item != null
                || priorityAdmission
                || cancellationReason != null
                || preemption != null
                || engineFence != null
                || admissionMutation != null
                || pendingAdmissionCancelReason != null
                || admissionResources != null
                || requestDeadline != null
                || acceptanceDeadline != null) {
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

    private enum EngineFenceCause {
        CANCELLATION,
        DELIVERY_UNCERTAIN
    }

    /** Finite protocol events accepted by one exact Engine fence. */
    enum FenceUpdate {
        CANCEL_STARTED,
        AWAIT_TERMINAL,
        TOMBSTONED
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
            EngineFenceRegistration fence,
            CancelTarget target,
            PendingReplay replay,
            PreemptionRegistration signal,
            ExactPrefillOnlyCleanup prefillOnlyCleanup) {

        static final PreemptionReduction STALE = simple(Status.STALE);
        static final PreemptionReduction NONE = simple(Status.NONE);

        PreemptionReduction {
            Objects.requireNonNull(status, "status");
            boolean startsFence = status == Status.START_FENCE;
            boolean replays = status == Status.REPLAY;
            if (startsFence != (fence != null && target != null)
                    || replays != (replay != null)
                    || (!replays && (signal != null
                        || prefillOnlyCleanup != null))) {
                throw new IllegalArgumentException(
                        "preemption reduction status requires its exact payload");
            }
        }

        static PreemptionReduction startFence(
                EngineFenceRegistration fence, CancelTarget target) {
            return new PreemptionReduction(Status.START_FENCE, fence, target,
                    null, null, null);
        }

        static PreemptionReduction replay(
                PendingReplay replay,
                PreemptionRegistration signal,
                ExactPrefillOnlyCleanup prefillOnlyCleanup) {
            return new PreemptionReduction(Status.REPLAY, null, null,
                    replay, signal, prefillOnlyCleanup);
        }

        private static PreemptionReduction simple(Status status) {
            return new PreemptionReduction(
                    status, null, null, null, null, null);
        }

        enum Status {
            STALE,
            NONE,
            START_FENCE,
            REPLAY
        }
    }

    /** Exact terminal leaves detached from a closed Engine fence. */
    record FenceTerminalProof(
            String detail,
            PreemptionRegistration transferred,
            DecodeEndpoint.AuthoritativeTerminalProof decodeProof,
            EngineFenceRegistration cleanup) {
    }

    /** The only effect exposed after an Engine-fence reduction. */
    record FenceReduction(
            Status status,
            EngineFenceRegistration fence,
            CancelTarget target,
            FenceTerminalProof proof) {

        static final FenceReduction STALE = simple(Status.STALE);
        static final FenceReduction DEFERRED = simple(Status.DEFERRED);
        static final FenceReduction NONE = simple(Status.NONE);

        FenceReduction {
            Objects.requireNonNull(status, "status");
            boolean starts = status == Status.START;
            boolean terminal = status == Status.TERMINAL_PROOF;
            if (starts != (fence != null && target != null)
                    || terminal != (proof != null)) {
                throw new IllegalArgumentException(
                        "fence reduction status requires its exact payload");
            }
        }

        static FenceReduction start(
                EngineFenceRegistration fence, CancelTarget target) {
            return new FenceReduction(
                    Status.START, fence, target, null);
        }

        static FenceReduction terminalProof(FenceTerminalProof proof) {
            return new FenceReduction(
                    Status.TERMINAL_PROOF, null, null, proof);
        }

        private static FenceReduction simple(Status status) {
            return new FenceReduction(status, null, null, null);
        }

        enum Status {
            STALE,
            DEFERRED,
            NONE,
            START,
            TERMINAL_PROOF
        }
    }

    record AdmissionMutationCompletion(
            boolean owned,
            CancelReason cancellationToResume,
            EngineFenceRegistration tombstonedFence,
            DeferredTerminal pendingTerminal,
            TerminalAction pendingRetirement) {
        private static final AdmissionMutationCompletion NOT_OWNED =
                new AdmissionMutationCompletion(
                        false, null, null, null, null);
    }

    record AcceptanceExpiry(
            ScheduledRequest item,
            boolean needsFence,
            AdmissionCleanup cleanup) {
    }

    record DeliveryConfirmation(
            PublicationPermit publication,
            RequestDeadline requestDeadline,
            AdmissionCleanup admissionCleanup,
            boolean armAcceptanceDeadline,
            long batchEnqueueStartedAtMs) {
    }

    /**
     * Exact one-shot wrapper for dispatcher admission resources. The raw
     * callback is never returned after this capability has been created.
     */
    private static final class AdmissionResources {
        private final Runnable releaseAction;
        private final long acceptanceTimeoutMs;
        private boolean released;

        private AdmissionResources(
                Runnable releaseAction,
                long acceptanceTimeoutMs) {
            this.releaseAction = releaseAction;
            this.acceptanceTimeoutMs = acceptanceTimeoutMs;
        }

        private synchronized void release() {
            if (released) {
                return;
            }
            released = true;
            releaseAction.run();
        }
    }

    /** Transferable exact cleanup; owns both admission and timer capabilities. */
    static final class AdmissionCleanup {
        private final AdmissionResources resources;
        private final AcceptanceDeadline acceptanceDeadline;
        private boolean released;

        private AdmissionCleanup(
                AdmissionResources resources,
                AcceptanceDeadline acceptanceDeadline) {
            this.resources = resources;
            this.acceptanceDeadline = acceptanceDeadline;
        }

        synchronized void release(ExpirationTimer timer) {
            if (released) {
                return;
            }
            released = true;
            Throwable failure = null;
            if (acceptanceDeadline != null) {
                try {
                    timer.cancel(acceptanceDeadline);
                } catch (Throwable timerFailure) {
                    failure = timerFailure;
                }
            }
            if (resources != null) {
                try {
                    resources.release();
                } catch (Throwable resourceFailure) {
                    failure = appendFailure(failure, resourceFailure);
                }
            }
            rethrowCleanup(failure);
        }
    }

    /** Exact terminal cleanup detached atomically at ACTIVE -> TERMINALIZING. */
    static final class TerminalResources {
        private final RequestDeadline requestDeadline;
        private final AdmissionCleanup admissionCleanup;
        private boolean released;

        private TerminalResources(
                RequestDeadline requestDeadline,
                AdmissionCleanup admissionCleanup) {
            this.requestDeadline = requestDeadline;
            this.admissionCleanup = admissionCleanup;
        }

        synchronized void release(ExpirationTimer timer) {
            if (released) {
                return;
            }
            released = true;
            Throwable failure = null;
            if (requestDeadline != null) {
                try {
                    timer.cancel(requestDeadline);
                } catch (Throwable timerFailure) {
                    failure = timerFailure;
                }
            }
            if (admissionCleanup != null) {
                try {
                    admissionCleanup.release(timer);
                } catch (Throwable admissionFailure) {
                    failure = appendFailure(failure, admissionFailure);
                }
            }
            rethrowCleanup(failure);
        }
    }

    /**
     * Request-scoped Engine Cancel owner. The transport result never creates a
     * retry state: every non-authoritative result waits for endpoint terminal
     * proof while retaining all exact resources.
     */
    static final class EngineFenceRegistration {
        private final String detail;
        private final PreemptionRegistration transferredPreemption;
        private final EngineFenceResources resources;
        private FencePhase phase = FencePhase.INSTALLED;

        private EngineFenceRegistration(
                EngineFenceCause cause,
                String detail,
                PreemptionRegistration transferredPreemption,
                EngineFenceResources resources) {
            this.detail = detail;
            this.transferredPreemption = transferredPreemption;
            this.resources = resources;
        }

        void release() {
            if (phase != FencePhase.CLOSED) {
                throw new IllegalStateException(
                        "Engine fence resources require authoritative close");
            }
            resources.release();
        }

        private boolean beginCancel() {
            if (phase != FencePhase.INSTALLED) {
                return false;
            }
            phase = FencePhase.CANCEL_IN_FLIGHT;
            return true;
        }

        private boolean awaitTerminal() {
            if (phase != FencePhase.CANCEL_IN_FLIGHT) {
                return false;
            }
            phase = FencePhase.AWAITING_TERMINAL;
            return true;
        }

        private boolean recordTombstoned() {
            if (phase != FencePhase.CANCEL_IN_FLIGHT) {
                return false;
            }
            phase = FencePhase.CLOSED;
            return true;
        }

        private boolean close() {
            if (phase == FencePhase.CLOSED) {
                return false;
            }
            phase = FencePhase.CLOSED;
            return true;
        }

        private boolean isClosed() {
            return phase == FencePhase.CLOSED;
        }

        private boolean cancelMayHaveBeenInstalled() {
            return phase != FencePhase.INSTALLED;
        }

        private enum FencePhase {
            INSTALLED,
            CANCEL_IN_FLIGHT,
            AWAITING_TERMINAL,
            CLOSED
        }
    }

    /** Invocation-local signal for a lost NOT_FOUND endpoint-transfer race. */
    private static final class NotFoundFenceTransferLost
            extends RuntimeException {
        private static final NotFoundFenceTransferLost INSTANCE =
                new NotFoundFenceTransferLost();

        private NotFoundFenceTransferLost() {
            super(null, null, false, false);
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
        private final AtomicBoolean terminalOutstandingResolved =
                new AtomicBoolean();

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
            return () -> future.completeOwned(response);
        }

        BooleanSupplier claimTerminalResponse(Response response) {
            requireTerminal("external response");
            claim();
            return terminalPublication(() -> future.completeOwned(response));
        }

        BooleanSupplier claimFailure(Throwable failure) {
            requireTerminal("failure");
            claim();
            return terminalPublication(
                    () -> future.completeExceptionallyOwned(failure));
        }

        BooleanSupplier claimCancellation(boolean mayInterruptIfRunning) {
            requireTerminal("cancellation");
            claim();
            return terminalPublication(
                    () -> future.cancelOwned(mayInterruptIfRunning));
        }

        /** Abandon a permit only when no other submitter consumed it. */
        void abandonIfUnclaimed() {
            if (claimed.compareAndSet(false, true)) {
                resolveTerminalOutstanding();
                closePublication();
            }
        }

        /** Settle a claim whose publication could not enter its executor. */
        void abortClaimedPublication() {
            resolveTerminalOutstanding();
            closePublication();
        }

        private BooleanSupplier terminalPublication(BooleanSupplier publication) {
            return () -> {
                try {
                    return publication.getAsBoolean();
                } finally {
                    resolveTerminalOutstanding();
                }
            };
        }

        private void resolveTerminalOutstanding() {
            if (kind == PublicationKind.TERMINAL
                    && terminalOutstandingResolved.compareAndSet(
                            false, true)) {
                slot.releaseOutstandingPermit();
            }
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

    /**
     * One exact resource bundle retained by an Engine fence. Acquisition and
     * release remain request-scoped and idempotent; no individual lease leaks
     * into the scheduler state machine.
     */
    private static final class EngineFenceResources {
        private final PrefillEndpoint prefill;
        private final PrefillState.Protection prefillProtection;
        private final DecodeEndpoint.EngineFenceLease decodeProtection;
        private boolean released;

        private EngineFenceResources(
                PrefillEndpoint prefill,
                PrefillState.Protection prefillProtection,
                DecodeEndpoint.EngineFenceLease decodeProtection) {
            this.prefill = prefill;
            this.prefillProtection = prefillProtection;
            this.decodeProtection = decodeProtection;
        }

        private static EngineFenceResources acquire(
                ScheduledRequest item,
                PrefillState.Protection prefillProtection) {
            PrefillEndpoint prefill = item.prefillEp();
            DecodeEndpoint decode = item.decodeEp();
            DecodeEndpoint.EngineFenceLease decodeProtection = null;
            try {
                DecodeEndpoint.ReservationHandle reservation =
                        item.decodeReservation();
                if (decode != null && reservation != null) {
                    decodeProtection =
                            decode.beginEngineFenceProtection(reservation);
                }
                return new EngineFenceResources(
                        prefill, prefillProtection, decodeProtection);
            } catch (RuntimeException | Error protectionFailure) {
                try {
                    if (decodeProtection != null) {
                        decodeProtection.close();
                    }
                } catch (RuntimeException | Error cleanupFailure) {
                    protectionFailure.addSuppressed(cleanupFailure);
                }
                try {
                    if (prefillProtection != null && prefill != null) {
                        prefill.releaseEngineFenceProtection(
                                prefillProtection);
                    }
                } catch (RuntimeException | Error cleanupFailure) {
                    protectionFailure.addSuppressed(cleanupFailure);
                }
                throw protectionFailure;
            }
        }

        private DecodeEndpoint.AuthoritativeTerminalProof
                decodeAuthoritativeTerminalProof() {
            return decodeProtection == null
                    ? null
                    : decodeProtection.authoritativeTerminalProof();
        }

        /**
         * Decode generation retirement already consumed the Decode leaf. Move
         * only the still-live Prefill protection into an exact cleanup and
         * permanently suppress Decode close on this bundle.
         */
        private synchronized ExactPrefillOnlyCleanup
                detachAfterDecodeGenerationRetired() {
            if (released) {
                return null;
            }
            released = true;
            return prefillProtection == null || prefill == null
                    ? null
                    : new ExactPrefillOnlyCleanup(
                            prefill, prefillProtection);
        }

        private synchronized void release() {
            if (released) {
                return;
            }
            released = true;
            Throwable failure = null;
            if (decodeProtection != null) {
                try {
                    decodeProtection.close();
                } catch (Throwable decodeFailure) {
                    failure = decodeFailure;
                }
            }
            if (prefillProtection != null && prefill != null) {
                try {
                    prefill.releaseEngineFenceProtection(
                            prefillProtection);
                } catch (Throwable prefillFailure) {
                    failure = appendFailure(failure, prefillFailure);
                }
            }
            rethrowCleanup(failure);
        }
    }

    /** One-shot Prefill leaf detached after authoritative Decode retirement. */
    static final class ExactPrefillOnlyCleanup {
        private final PrefillEndpoint prefill;
        private final PrefillState.Protection protection;
        private final AtomicBoolean released = new AtomicBoolean();

        private ExactPrefillOnlyCleanup(
                PrefillEndpoint prefill,
                PrefillState.Protection protection) {
            this.prefill = prefill;
            this.protection = protection;
        }

        void release() {
            if (released.compareAndSet(false, true)) {
                prefill.releaseEngineFenceProtection(protection);
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
        RequestSlot.EngineFenceRegistration releasableFence,
        RequestSlot.AdmissionCleanup admissionCleanup) {
    static final DecodeAcceptance NONE =
            new DecodeAcceptance(null, null);
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
        RequestSlot.EngineFenceRegistration fence,
        RequestSlot.TerminalResources terminalResources,
        boolean removePrefillQueue,
        boolean releaseDecode,
        boolean releasePrefill,
        Runnable counterpartCleanup,
        String queueReason,
        Function<RequestSlot, RequestState> transition,
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
