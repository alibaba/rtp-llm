package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillWorkLedger;
import org.flexlb.balance.admission.AdmissionMutation;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.preemption.PreemptionClaim;
import org.flexlb.balance.preemption.PreemptionLifecyclePort;
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
import java.util.function.BooleanSupplier;
import java.util.function.Function;

/**
 * Canonical aggregate root for one exact request generation.
 *
 * <p>Every mutable field is private and every method which observes or changes
 * lifecycle state requires {@code synchronized (slot)} unless its contract
 * explicitly says otherwise. Endpoint and transport code receives exact
 * capabilities, never a writable state field or a generic transition API.
 */
final class RequestSlot extends RequestLifecycle {

    private static final int OUTSTANDING_ADMISSION_CLOSED = -1;

    private final RequestLifecycleCoordinator owner;
    private final RequestCompletionPublisher completionPublisher;
    private final long requestId;
    private final RequestFuture future;

    private BatchItem item;
    private boolean priorityAdmission;
    private StrategyErrorType deadlineErrorType =
            StrategyErrorType.BATCH_SLO_EXPIRED;

    /** Storage/cleanup ownership; distinct from the public request lifecycle. */
    private SlotPhase slotPhase = SlotPhase.ACTIVE;
    private EngineOwnership engineOwnership = EngineOwnership.DECODE_PENDING;
    private CancelReason cancellationReason;

    private boolean admissionOpen = true;
    private SlotAdmissionMutation admissionMutation;
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
            RequestLifecycleCoordinator owner,
            RequestCompletionPublisher completionPublisher,
            long requestId) {
        super(requestId);
        this.owner = Objects.requireNonNull(owner, "owner");
        this.completionPublisher = Objects.requireNonNull(
                completionPublisher, "completionPublisher");
        this.requestId = requestId;
        this.future = new RequestFuture(completionPublisher, this);
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
        return snapshot().createdAtMs();
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
            BatchItem candidate,
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
    void rollbackItemPublication(BatchItem exact) {
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
    AdmissionCleanup rollbackAdmissionPublication(BatchItem exact) {
        rollbackItemPublication(exact);
        AdmissionCleanup cleanup = detachAdmissionCleanup(true);
        assertInvariant();
        return cleanup;
    }

    /** Exact ACTIVE item, or null when this generation no longer owns one. */
    BatchItem activeItem() {
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
                && !snapshot().state().isTerminal();
    }

    boolean ownsActiveItem(BatchItem expected) {
        requireSlotLock("active item ownership lookup");
        return ownsActiveGeneration() && item == expected;
    }

    boolean ownsPrefillFact(PrefillEndpoint source, BatchItem expected) {
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

    BatchItem activeItemForReservation(long reservationToken) {
        requireSlotLock("reservation item lookup");
        DecodeEndpoint.ReservationHandle reservation =
                item == null ? null : item.decodeReservation();
        return ownsActiveGeneration()
                && reservation != null
                && reservation.reservationToken() == reservationToken
                ? item : null;
    }

    boolean ownsDeliveryClaim(
            BatchItem expected,
            DeliveryClaimKind kind,
            long expectedBatchId) {
        requireSlotLock("delivery claim lookup");
        RequestLifecycleSnapshot current = snapshot();
        return ownsActiveItem(expected)
                && current.deliveryClaimKind() == kind
                && current.batchId() == expectedBatchId
                && !current.state().isTerminal();
    }

    /**
     * Commit the unique delivery-confirmation edge and move every capability
     * needed by its unlocked publication out of the slot.
     */
    DeliveryConfirmation confirmDeliveryForPublication(
            BatchItem expected,
            DeliveryClaimKind expectedKind,
            long expectedBatchId) {
        requireSlotLock("delivery confirmation");
        RequestLifecycleSnapshot current = snapshot();
        if (current.state() != RequestLifecycleState.DISPATCHING
                || !ownsDeliveryClaim(
                        expected, expectedKind, expectedBatchId)) {
            return null;
        }

        PublicationLease lease = requirePublicationLease();
        boolean transferred = false;
        try {
            long enqueueStartedAtMs = getBatchEnqueueStartedAtMs();
            RequestLifecycleSnapshot confirmed = markDeliveryConfirmed();
            if (confirmed.state() != RequestLifecycleState.ACKNOWLEDGED) {
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
                    new PublicationPermit(
                            this, PublicationKind.DELIVERY, lease),
                    detachedRequestDeadline,
                    detachedAdmission,
                    armAcceptanceDeadline,
                    enqueueStartedAtMs);
            transferred = true;
            assertInvariant();
            return result;
        } finally {
            if (!transferred) {
                lease.close();
            }
        }
    }

    boolean decodeOwnsRequest() {
        requireSlotLock("Decode ownership lookup");
        return engineOwnership == EngineOwnership.DECODE_OWNED;
    }

    boolean canClaimLocalTerminal() {
        requireSlotLock("local terminal eligibility");
        RequestLifecycleSnapshot current = snapshot();
        return ownsActiveGeneration()
                && !future.isDone()
                && admissionMutation == null
                && preemption == null
                && engineFence == null
                && engineOwnership == EngineOwnership.DECODE_PENDING
                && current.state() != RequestLifecycleState.ACKNOWLEDGED
                && !current.deliveryClaimKind().isClaimed();
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
        RequestLifecycleSnapshot current = snapshot();
        return ownsActiveGeneration()
                && !future.isDone()
                && preemption == null
                && engineFence == null
                && engineOwnership == EngineOwnership.DECODE_PENDING
                && current.state() != RequestLifecycleState.ACKNOWLEDGED
                && !current.deliveryClaimKind().isClaimed();
    }

    boolean isOpen() {
        requireSlotLock("admission state lookup");
        return slotPhase == SlotPhase.ACTIVE
                && admissionOpen
                && !future.isDone()
                && !snapshot().state().isTerminal();
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
        RequestLifecycleSnapshot current = snapshot();
        return isCurrentGeneration()
                && slotPhase == SlotPhase.TOMBSTONE
                && current.state().isTerminal()
                && current.updatedAtMs() < updatedBeforeMs
                && item == null;
    }

    // ==================== Admission mutation ====================

    AdmissionMutation tryBeginAdmissionMutation() {
        requireSlotLock("admission mutation claim");
        if (!ownsActiveGeneration()
                || !isOpen()
                || item != null
                || admissionMutation != null
                || preemption != null
                || engineFence != null) {
            return null;
        }
        SlotAdmissionMutation exact = new SlotAdmissionMutation();
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
        SlotAdmissionMutation owned = (SlotAdmissionMutation) exact;
        admissionMutation = null;
        return finishAdmissionMutation(owned);
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
        SlotAdmissionMutation owned = (SlotAdmissionMutation) exact;
        admissionMutation = null;
        return finishAdmissionMutation(owned);
    }

    /** Resolve retained facts once for both ordinary close and terminal close. */
    private AdmissionMutationCompletion finishAdmissionMutation(
            SlotAdmissionMutation owned) {
        CancelReason cancellationToResume = pendingAdmissionCancelReason;
        pendingAdmissionCancelReason = null;
        EngineFenceRegistration tombstonedFence =
                authoritativeTombstonedFence();
        DeferredTerminal pendingTerminal = tombstonedFence == null
                && owned.pendingTerminal != null
                && owned.pendingTerminal.authoritativeWorker()
                        ? owned.pendingTerminal : null;
        TerminalAction pendingRetirement = tombstonedFence == null
                && pendingTerminal == null
                        ? beginPendingPrefillRetirement(
                                owned.pendingPrefillRetirement)
                        : null;
        if (tombstonedFence == null
                && pendingTerminal == null
                && pendingRetirement == null) {
            pendingTerminal = owned.pendingTerminal;
        }
        if (tombstonedFence != null || pendingTerminal != null
                || pendingRetirement != null) {
            cancellationToResume = null;
        } else {
            cancellationToResume =
                    promoteAdmissionCancellation(cancellationToResume);
        }
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

    private final class SlotAdmissionMutation implements AdmissionMutation {
        private final AtomicBoolean resolved = new AtomicBoolean();
        private DeferredTerminal pendingTerminal;
        private PendingPrefillRetirement pendingPrefillRetirement;

        private void retainTerminal(DeferredTerminal candidate) {
            if (pendingTerminal == null
                    || (!pendingTerminal.endpointAlreadyRetired()
                        && (candidate.endpointAlreadyRetired()
                            || (!pendingTerminal.authoritativeWorker()
                                && candidate.authoritativeWorker())))) {
                pendingTerminal = candidate;
            }
        }

        private void retainPrefillRetirement(
                PendingPrefillRetirement candidate) {
            if (pendingPrefillRetirement == null) {
                pendingPrefillRetirement = candidate;
            } else if (pendingPrefillRetirement.source != candidate.source
                    || pendingPrefillRetirement.item != candidate.item) {
                throw new IllegalStateException(
                        "admission mutation observed another Prefill generation"
                                + " for request " + requestId);
            }
        }

        @Override
        public void terminate(Response failure) {
            if (!resolved.compareAndSet(false, true)) {
                return;
            }
            owner.terminateAdmissionMutation(
                    RequestSlot.this, this, failure);
        }

        @Override
        public void close() {
            if (!resolved.compareAndSet(false, true)) {
                return;
            }
            owner.completeAdmissionMutation(RequestSlot.this, this);
        }
    }

    /** Exact Prefill retirement retained only by its in-flight admission. */
    private record PendingPrefillRetirement(
            PrefillEndpoint source,
            BatchItem item,
            Function<RequestSlot, RequestLifecycleSnapshot> transition,
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

    RequestDeadlineExpiry expireRequestDeadline(RequestDeadline exact) {
        requireSlotLock("request deadline expiry");
        if (requestDeadline != exact) {
            return RequestDeadlineExpiry.IGNORED;
        }
        requestDeadline = null;
        if (!ownsActiveGeneration() || future.isDone() || !isOpen()) {
            assertInvariant();
            return RequestDeadlineExpiry.IGNORED;
        }
        admissionOpen = false;
        if (admissionMutation != null) {
            if (pendingAdmissionCancelReason == null) {
                pendingAdmissionCancelReason = CancelReason.DEADLINE_EXCEEDED;
                requestCancel(
                        "request scheduling deadline exceeded during admission");
            }
            assertInvariant();
            return RequestDeadlineExpiry.DEFERRED_BY_ADMISSION;
        }
        assertInvariant();
        return RequestDeadlineExpiry.CANCEL_REQUEST;
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
            return AcceptanceExpiry.IGNORED;
        }
        acceptanceDeadline = null;
        BatchItem expected = item;
        boolean needsFence = ownsActiveGeneration()
                && expected != null
                && engineOwnership != EngineOwnership.DECODE_OWNED;
        AdmissionCleanup cleanup = detachAdmissionCleanup(false);
        assertInvariant();
        return new AcceptanceExpiry(true, expected, needsFence, cleanup);
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
        assert cancellationReason != null : "missing cancellation first cause";
        return cancellationReason;
    }

    RequestLifecycleSnapshot rememberCancellation(
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
        RequestLifecycleSnapshot lifecycle = snapshot();
        DecodeEndpoint.ReservationHandle reservation =
                item == null ? null : item.decodeReservation();
        if (!ownsActiveGeneration()
                || admissionMutation != null
                || preemption != null
                || engineFence != null
                || cancellationReason != null
                || reservation == null
                || reservation.reservationToken() != reservationToken
                || (lifecycle.deliveryClaimKind()
                        == DeliveryClaimKind.ROUTE_DECISION
                    && lifecycle.state()
                        == RequestLifecycleState.DISPATCHING)) {
            return null;
        }
        preemption = new PreemptionRegistration(
                requestId, attemptToken, detail);
        assertInvariant();
        return preemption;
    }

    /** Reduce one coordinator-owned update against its exact claim. */
    PreemptionReduction applyPreemptionUpdate(
            PreemptionClaim claim,
            PreemptionLifecyclePort.Update update) {
        requireSlotLock("preemption update reduction");
        if (claim == null || update == null) {
            return PreemptionReduction.Stale.INSTANCE;
        }
        PreemptionRegistration exact = exactPreemption(claim);
        if (exact == null) {
            return PreemptionReduction.Stale.INSTANCE;
        }
        return switch (update) {
            case PreemptionLifecyclePort.Tombstoned tombstoned ->
                    applyPreemptionTombstoned(
                            exact, tombstoned.detail());
            case PreemptionLifecyclePort.Step step -> switch (step) {
                case RELEASE -> applyPreemptionRelease(exact);
                case CANCEL_STARTED -> applyPreemptionCancelStarted(exact);
                case CANCEL_ACCEPTED -> applyPreemptionCancelAccepted(exact);
                case CANCEL_NOT_FOUND -> applyPreemptionCancelNotFound(exact);
                case CANCEL_UNKNOWN -> applyPreemptionCancelUnknown(exact);
            };
        };
    }

    private PreemptionReduction applyPreemptionCancelStarted(
            PreemptionRegistration exact) {
        boolean accepted = ownsActiveGeneration()
                && preemption == exact
                && cancellationReason == null
                && exact.beginCancel();
        assertInvariant();
        return accepted
                ? PreemptionReduction.None.INSTANCE
                : PreemptionReduction.Stale.INSTANCE;
    }

    private PreemptionReduction applyPreemptionCancelAccepted(
            PreemptionRegistration exact) {
        if (!ownsActiveGeneration()
                || preemption != exact
                || !exact.acceptCancel()) {
            return PreemptionReduction.Stale.INSTANCE;
        }
        requestCancel(exact.detail());
        assertInvariant();
        return PreemptionReduction.None.INSTANCE;
    }

    private PreemptionReduction applyPreemptionCancelNotFound(
            PreemptionRegistration exact) {
        if (!ownsActiveGeneration()
                || preemption != exact
                || !exact.markNotFound()) {
            return PreemptionReduction.Stale.INSTANCE;
        }
        assertInvariant();
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
        return started == null
                ? materializePendingReplay(exact, false, exact)
                : new PreemptionReduction.StartFence(
                        started, cancelTarget(item));
    }

    private PreemptionReduction applyPreemptionCancelUnknown(
            PreemptionRegistration exact) {
        if (!ownsActiveGeneration()
                || preemption != exact
                || !exact.markUnknown()) {
            return PreemptionReduction.Stale.INSTANCE;
        }
        assertInvariant();
        return materializePendingReplay(exact, true, exact);
    }

    private PreemptionReduction applyPreemptionRelease(
            PreemptionRegistration exact) {
        if (!ownsActiveGeneration()
                || preemption != exact
                || !exact.isReleasable()) {
            return PreemptionReduction.Stale.INSTANCE;
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
                ? PreemptionReduction.Stale.INSTANCE
                : new PreemptionReduction.StartFence(
                        started, cancelTarget(item));
    }

    private PreemptionReduction applyPreemptionTombstoned(
            PreemptionRegistration exact,
            String detail) {
        if (!ownsActiveGeneration()
                || preemptionOwner() != exact
                || !exact.canSettleTombstone()
                || !exact.settle()) {
            return PreemptionReduction.Stale.INSTANCE;
        }
        DeferredTerminal terminal = new DeferredTerminal.Priority(detail);
        exact.retainTerminal(terminal);
        // DecodePreemptionCoordinator has already consumed the exact endpoint
        // claim before publishing TOMBSTONED. Reconciliation is therefore
        // neither required nor legal on this authoritative path.
        detachPreemptionOwner(exact);
        assertInvariant();
        return new PreemptionReduction.Replay(
                new ReplayPayload.Terminal(terminal),
                exact,
                null);
    }

    /**
     * Reduce one exact transport/endpoint fact without exposing the mutable
     * preemption registration. The caller holds {@code synchronized (slot)}
     * and only executes the returned, already-selected effect.
     */
    PreemptionReduction applyPreemptionFact(PreemptionFact fact) {
        requireSlotLock("preemption fact reduction");
        if (fact instanceof PreemptionFact.PrefillActive active) {
            PreemptionRegistration exact = preemption;
            if (!ownsPrefillFact(active.source(), active.expected())
                    || exact == null
                    || !exact.isNotFound()) {
                return PreemptionReduction.Stale.INSTANCE;
            }
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
            if (started != null) {
                return new PreemptionReduction.StartFence(
                        started, cancelTarget(item));
            }
            DecodeEndpoint decode = active.expected().decodeEp();
            if (decode == null
                    || decode.reconcilePriorityVictimActive(
                            exact.attemptToken(),
                            active.expected().decodeReservation())) {
                detachPreemptionOwner(exact);
            }
            return PreemptionReduction.None.INSTANCE;
        }
        if (fact instanceof PreemptionFact.WorkerTerminal worker) {
            if (!ownsActiveItem(worker.expected())) {
                return PreemptionReduction.Stale.INSTANCE;
            }
            if (admissionMutation instanceof SlotAdmissionMutation mutation) {
                mutation.retainTerminal(worker.terminal());
                assertInvariant();
                return PreemptionReduction.None.INSTANCE;
            }
            PreemptionRegistration exact = preemptionOwner();
            if (exact == null) {
                return new PreemptionReduction.Replay(
                        new ReplayPayload.Terminal(worker.terminal()),
                        null,
                        null);
            }
            if (exact.isSettled()) {
                return PreemptionReduction.Stale.INSTANCE;
            }
            exact.retainTerminal(worker.terminal());
            if (!ownsActiveGeneration()
                    || preemptionOwner() != exact
                    || !exact.settle()) {
                return PreemptionReduction.Stale.INSTANCE;
            }
            assertInvariant();
            return materializePendingReplay(exact, false, exact);
        }
        if (fact instanceof PreemptionFact.DispatchRejected rejected) {
            if (!ownsActiveItem(rejected.expected())
                    || !ownsDecodeFact(
                            rejected.source(), rejected.reservation())) {
                return PreemptionReduction.Stale.INSTANCE;
            }
            DeferredTerminal terminal = rejected.terminal();
            if (admissionMutation instanceof SlotAdmissionMutation mutation) {
                mutation.retainTerminal(terminal);
                assertInvariant();
                return PreemptionReduction.None.INSTANCE;
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
            return new PreemptionReduction.Replay(
                    new ReplayPayload.Terminal(terminal),
                    signal,
                    null);
        }
        if (fact instanceof PreemptionFact.OrdinaryTerminal ordinary) {
            if (!ownsActiveItem(ordinary.expected())) {
                return PreemptionReduction.Stale.INSTANCE;
            }
            DeferredTerminal terminal = ordinary.terminal();
            if (terminal.authoritativeWorker()) {
                throw new IllegalArgumentException(
                        "authoritative worker fact requires WorkerTerminal");
            }
            if (engineFence != null) {
                return PreemptionReduction.None.INSTANCE;
            }
            if (admissionMutation instanceof SlotAdmissionMutation mutation) {
                mutation.retainTerminal(terminal);
                assertInvariant();
                return PreemptionReduction.None.INSTANCE;
            }

            PreemptionRegistration exact = preemptionOwner();
            if (engineOwnership == EngineOwnership.DECODE_OWNED
                    && terminal.deliveryFailure()) {
                RequestLifecycleSnapshot lifecycle = snapshot();
                DeliveryClaimKind deliveryKind = lifecycle.deliveryClaimKind();
                long batchId = lifecycle.batchId();
                if (exact == null) {
                    DeliveryConfirmation confirmation =
                            confirmDeliveryForPublication(
                                    ordinary.expected(),
                                    deliveryKind,
                                    batchId);
                    return confirmation == null
                            ? PreemptionReduction.Stale.INSTANCE
                            : new PreemptionReduction.Replay(
                                    new ReplayPayload.Delivery(
                                            confirmation,
                                            ordinary.expected(),
                                            deliveryKind,
                                            batchId),
                                    null,
                                    null);
                }
                if (exact.isSettled()) {
                    return PreemptionReduction.Stale.INSTANCE;
                }
                exact.recordDeliveryConfirmation(batchId);
                assertInvariant();
                return materializePendingReplay(exact, false, null);
            }

            if (exact == null) {
                    return new PreemptionReduction.Replay(
                            new ReplayPayload.Terminal(terminal),
                            null,
                            null);
            }
            if (exact.isSettled()) {
                return PreemptionReduction.Stale.INSTANCE;
            }
            exact.retainTerminal(terminal);
            assertInvariant();
            if (!exact.isNotFound() && !exact.isUnknown()) {
                return PreemptionReduction.None.INSTANCE;
            }
            return materializePendingReplay(
                    exact, exact.isUnknown(), exact);
        }
        if (fact instanceof PreemptionFact.PriorityCanceled canceled) {
            PreemptionRegistration exact =
                    ownsPrefillFact(canceled.source(), canceled.expected())
                            ? preemptionOwner() : null;
            DecodeEndpoint decode = canceled.expected().decodeEp();
            if (exact == null
                    || exact.isSettled()
                    || decode == null
                    || canceled.expected().decodeReservation() == null
                    || !decode.settlePriorityCanceled(
                            exact.attemptToken(),
                            canceled.expected().decodeReservation())
                    || !ownsActiveGeneration()
                    || preemptionOwner() != exact
                    || !exact.settle()) {
                return PreemptionReduction.Stale.INSTANCE;
            }
            DeferredTerminal terminal = new DeferredTerminal.Priority(
                    "priority victim canceled by worker");
            exact.retainTerminal(terminal);
            detachPreemptionOwner(exact);
            assertInvariant();
            return new PreemptionReduction.Replay(
                    new ReplayPayload.Terminal(terminal),
                    exact,
                    null);
        }
        if (fact instanceof PreemptionFact.DecodeGenerationRetired retired) {
            if (!ownsDecodeFact(retired.source(), retired.reservation())) {
                return PreemptionReduction.Stale.INSTANCE;
            }
            DeferredTerminal terminal =
                    new DeferredTerminal.DecodeGenerationRetired(
                            retired.detail());
            if (admissionMutation instanceof SlotAdmissionMutation mutation) {
                mutation.retainTerminal(terminal);
                assertInvariant();
                return PreemptionReduction.None.INSTANCE;
            }

            PreemptionRegistration exact = preemptionOwner();
            PreemptionRegistration signal = null;
            if (exact != null) {
                exact.retainTerminal(terminal);
                exact.settle();
                signal = exact;
            }
            detachPreemptionOwner(exact);

            PrefillOnlyCleanup prefillCleanup = null;
            EngineFenceRegistration retiredFence = engineFence;
            if (retiredFence != null) {
                retiredFence.close();
                engineFence = null;
                prefillCleanup = retiredFence.resources
                        .detachAfterDecodeGenerationRetired();
            }
            assertInvariant();
            return new PreemptionReduction.Replay(
                    new ReplayPayload.Terminal(terminal),
                    signal,
                    prefillCleanup);
        }
        PreemptionFact.DeliveryConfirmed delivered =
                (PreemptionFact.DeliveryConfirmed) fact;
        BatchItem active = activeItem();
        RequestLifecycleSnapshot lifecycle = snapshot();
        DeliveryClaimKind deliveryKind = lifecycle.deliveryClaimKind();
        if (active == null
                || !ownsDeliveryClaim(
                        active, deliveryKind, delivered.batchId())) {
            return PreemptionReduction.Stale.INSTANCE;
        }
        if (engineFence != null) {
            return PreemptionReduction.None.INSTANCE;
        }
        PreemptionRegistration exact = preemptionOwner();
        if (exact == null) {
            DeliveryConfirmation confirmation =
                    confirmDeliveryForPublication(
                            active, deliveryKind, delivered.batchId());
            return confirmation == null
                    ? PreemptionReduction.Stale.INSTANCE
                    : new PreemptionReduction.Replay(
                            new ReplayPayload.Delivery(
                                    confirmation,
                                    active,
                                    deliveryKind,
                                    delivered.batchId()),
                            null,
                            null);
        }
        if (exact.isSettled()) {
            return PreemptionReduction.Stale.INSTANCE;
        }
        exact.recordDeliveryConfirmation(delivered.batchId());
        assertInvariant();
        return materializePendingReplay(exact, false, null);
    }

    /** Install or join the canonical delivery-uncertainty fence. */
    FenceReduction requestDeliveryFence(String detail) {
        return requestFence(
                EngineFenceCause.DELIVERY_UNCERTAIN, detail, false);
    }

    /** Install or join the canonical cancellation fence. */
    FenceReduction requestCancellationFence(String detail) {
        return requestFence(EngineFenceCause.CANCELLATION, detail, true);
    }

    private FenceReduction requestFence(
            EngineFenceCause cause,
            String detail,
            boolean allowDecodeOwned) {
        requireSlotLock("Engine fence request");
        if (preemption != null) {
            PreemptionRegistration exact = preemption;
            exact.requirePostDeliveryFence(detail);
            if (!exact.isNotFound()) {
                assertInvariant();
                return FenceReduction.None.INSTANCE;
            }
            EngineFenceRegistration transferred;
            try {
                transferred = installEngineFence(
                        cause,
                        detail,
                        exact,
                        allowDecodeOwned,
                        true);
            } catch (NotFoundFenceTransferLost legalRace) {
                return FenceReduction.Stale.INSTANCE;
            }
            return transferred == null
                    ? FenceReduction.Stale.INSTANCE
                    : new FenceReduction.Start(
                            transferred, cancelTarget(item));
        }
        EngineFenceRegistration installed = installEngineFence(
                cause,
                detail,
                null,
                allowDecodeOwned,
                false);
        return installed == null
                ? FenceReduction.Stale.INSTANCE
                : new FenceReduction.Start(installed, cancelTarget(item));
    }

    /** Reduce one event against the exact Engine-fence owner. */
    FenceReduction applyFenceUpdate(
            FenceHandle handle,
            FenceUpdate update) {
        requireSlotLock("Engine fence update reduction");
        EngineFenceRegistration exact = exactFence(handle);
        if (!ownsActiveGeneration() || exact == null || update == null) {
            return FenceReduction.Stale.INSTANCE;
        }
        return switch (update) {
            case CANCEL_STARTED -> {
                if (!exact.beginCancel()) {
                    yield FenceReduction.Stale.INSTANCE;
                }
                assertInvariant();
                yield FenceReduction.None.INSTANCE;
            }
            case AWAIT_TERMINAL -> {
                if (!exact.awaitTerminal()) {
                    yield FenceReduction.Stale.INSTANCE;
                }
                assertInvariant();
                yield FenceReduction.None.INSTANCE;
            }
            case TOMBSTONED -> applyFenceTombstoned(exact);
        };
    }

    private FenceReduction applyFenceTombstoned(
            EngineFenceRegistration exact) {
        if (!exact.recordTombstoned() && !exact.isClosed()) {
            return FenceReduction.Stale.INSTANCE;
        }
        assertInvariant();
        if (admissionMutation != null) {
            return FenceReduction.Deferred.INSTANCE;
        }
        return new FenceReduction.TerminalProof(
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
            BatchItem active = activeItem();
            DecodeEndpoint decode = active == null ? null : active.decodeEp();
            boolean ordinaryWon = terminalOwnsDecodeSettlement(
                    terminal,
                    decode,
                    exact.attemptToken(),
                    active == null ? null : active.decodeReservation());
            if (!ordinaryWon) {
                return PreemptionReduction.None.INSTANCE;
            }
            detachPreemptionOwner(exact);
            return new PreemptionReduction.Replay(
                    new ReplayPayload.Terminal(terminal),
                    signal,
                    null);
        }
        if (transportUnknown || !exact.hasPendingDeliveryConfirmation()) {
            return PreemptionReduction.None.INSTANCE;
        }

        BatchItem active = activeItem();
        DecodeEndpoint decode = active == null ? null : active.decodeEp();
        boolean activeWon = decode == null
                || decode.reconcilePriorityVictimActive(
                        exact.attemptToken(),
                        active.decodeReservation());
        if (!activeWon) {
            return PreemptionReduction.None.INSTANCE;
        }
        detachPreemptionOwner(exact);
        if (active == null) {
            return PreemptionReduction.Stale.INSTANCE;
        }
        RequestLifecycleSnapshot lifecycle = snapshot();
        DeliveryConfirmation confirmation = confirmDeliveryForPublication(
                active,
                lifecycle.deliveryClaimKind(),
                exact.pendingConfirmationBatchId());
        return confirmation == null
                ? PreemptionReduction.Stale.INSTANCE
                : new PreemptionReduction.Replay(
                        new ReplayPayload.Delivery(
                                confirmation,
                                active,
                                lifecycle.deliveryClaimKind(),
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
            PreemptionClaim claim) {
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
        BatchItem active = activeItem();
        assert active != null : "Engine fence requires active item";
        RequestLifecycleSnapshot lifecycle = snapshot();
        PrefillEndpoint prefill = active.prefillEp();
        PrefillWorkLedger.Protection protection = null;
        if (prefill != null) {
            protection = lifecycle.deliveryClaimKind()
                            == DeliveryClaimKind.BATCH_ENQUEUE
                            && lifecycle.batchId() > 0L
                    ? prefill.acquireBatchMemberProtection(
                            lifecycle.batchId(), active)
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
        BatchItem active = activeItem();
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

    private EngineFenceRegistration exactFence(FenceHandle handle) {
        if (!(handle instanceof EngineFenceRegistration exact)
                || engineFence != exact) {
            return null;
        }
        return exact;
    }

    private static CancelTarget cancelTarget(BatchItem active) {
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
            return DecodeAcceptance.STALE;
        }
        engineOwnership = EngineOwnership.DECODE_OWNED;
        EngineFenceRegistration fence = engineFence;
        if (fence == null) {
            DecodeAcceptance accepted = new DecodeAcceptance(
                    true, null, detachAdmissionCleanup(true));
            assertInvariant();
            return accepted;
        }
        if (cancellationReason != null
                || fence.transferredPreemption != null
                || fence.cancelMayHaveBeenInstalled()) {
            assertInvariant();
            return DecodeAcceptance.FENCE_RETAINED;
        }
        EngineFenceRegistration releasable = closeEngineFence(fence);
        DecodeAcceptance accepted = new DecodeAcceptance(
                true, releasable, detachAdmissionCleanup(true));
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
            BatchItem expected,
            Function<RequestSlot, RequestLifecycleSnapshot> transition,
            Response response) {
        requireSlotLock("Prefill retirement terminal claim");
        PendingPrefillRetirement pending = new PendingPrefillRetirement(
                source, expected, transition, response);
        if (!canTerminateFromPrefillRetirement(source, expected)) {
            return null;
        }
        if (admissionMutation instanceof SlotAdmissionMutation mutation) {
            mutation.retainPrefillRetirement(pending);
            assertInvariant();
            return null;
        }
        return beginPendingPrefillRetirement(pending);
    }

    private boolean canTerminateFromPrefillRetirement(
            PrefillEndpoint source,
            BatchItem expected) {
        return ownsPrefillFact(source, expected)
                && engineOwnership != EngineOwnership.DECODE_OWNED
                && engineFence == null
                && preemption == null
                && !snapshot().deliveryClaimKind().isClaimed();
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
            Function<RequestSlot, RequestLifecycleSnapshot> transition,
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
            Function<RequestSlot, RequestLifecycleSnapshot> transition) {
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
            Function<RequestSlot, RequestLifecycleSnapshot> transition,
            Response response,
            boolean requestPublication) {
        requireSlotLock("terminal claim");
        assert transition != null : "terminal transition is required";
        if (!ownsActiveGeneration() || admissionMutation != null) {
            return null;
        }
        RequestLifecycleSnapshot current = snapshot();
        boolean publishable = requestPublication
                && current.state() != RequestLifecycleState.ACKNOWLEDGED
                && !future.isDone();
        PublicationLease lease = publishable
                ? requirePublicationLease() : null;
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
                    lease);
            transferred = true;
            assertInvariant();
            return action;
        } finally {
            if (!transferred && lease != null) {
                lease.close();
            }
        }
    }

    TombstoneResult finishTombstone(TerminalAction action) {
        requireSlotLock("terminal tombstone");
        if (!isCurrentGeneration()
                || slotPhase != SlotPhase.TERMINALIZING
                || action.slot() != this
                || item != action.item()) {
            if (action.publicationLease() != null) {
                action.publicationLease().close();
            }
            return new TombstoneResult(
                    null,
                    new IllegalStateException(
                            "terminal slot identity changed: request_id="
                                    + requestId),
                    null);
        }
        RequestLifecycleSnapshot terminal;
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
                action.publicationLease() == null
                        ? null : new PublicationPermit(
                                this,
                                PublicationKind.TERMINAL,
                                action.publicationLease()));
    }

    private PublicationLease requirePublicationLease() {
        PublicationLease lease =
                completionPublisher.tryReservePublication(this);
        if (lease == null || lease.exactSlot() != this) {
            if (lease != null) {
                lease.close();
            }
            throw new IllegalStateException(
                    "frontend publication is closed for request " + requestId);
        }
        return lease;
    }

    private boolean isCurrentGeneration() {
        return owner.isCurrentSlot(this);
    }

    @Override
    void afterLifecycleMutation() {
        assertInvariant();
    }

    private void assertInvariant() {
        assert invariantHolds();
    }

    /** Expensive aggregate verification used by tests and debug JVMs only. */
    private boolean invariantHolds() {
        assert Thread.holdsLock(this) : "request slot invariant requires slot lock";
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
            return true;
        }
        RequestLifecycleSnapshot lifecycle = snapshot();
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
        if (!lifecycle.state().isTerminal()) {
            throw new IllegalStateException(
                    "tombstone lifecycle is not terminal for " + requestId);
        }
        return true;
    }

    private void requireSlotLock(String operation) {
        assert Thread.holdsLock(this) : operation + " requires slot lock";
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

    private enum PublicationKind {
        DELIVERY,
        TERMINAL
    }

    private enum EngineFenceCause {
        CANCELLATION,
        DELIVERY_UNCERTAIN
    }

    /** Opaque identity of one exact Engine-fence registration. */
    sealed interface FenceHandle permits EngineFenceRegistration {
    }

    /** Finite protocol events accepted by one exact Engine fence. */
    enum FenceUpdate {
        CANCEL_STARTED,
        AWAIT_TERMINAL,
        TOMBSTONED
    }

    /** One-shot cleanup detached from an exact, already-closed Engine fence. */
    sealed interface FenceCleanup permits EngineFenceRegistration {
        void release();
    }

    /** Prefill-only fence leaf after Decode generation cleanup already won. */
    sealed interface PrefillOnlyCleanup permits ExactPrefillOnlyCleanup {
        void release();
    }

    /** External facts which advance the exact preemption owner. */
    sealed interface PreemptionFact {
        record PrefillActive(PrefillEndpoint source, BatchItem expected)
                implements PreemptionFact {
        }

        record WorkerTerminal(
                BatchItem expected,
                DeferredTerminal terminal)
                implements PreemptionFact {
            public WorkerTerminal {
                if (!terminal.authoritativeWorker()) {
                    throw new IllegalArgumentException(
                            "WorkerTerminal requires an authoritative worker fact");
                }
            }
        }

        record OrdinaryTerminal(
                BatchItem expected,
                DeferredTerminal terminal)
                implements PreemptionFact {
            public OrdinaryTerminal {
                if (terminal.authoritativeWorker()) {
                    throw new IllegalArgumentException(
                            "authoritative worker fact requires WorkerTerminal");
                }
            }
        }

        record DispatchRejected(
                DecodeEndpoint source,
                DecodeEndpoint.ReservationHandle reservation,
                BatchItem expected,
                DeferredTerminal terminal)
                implements PreemptionFact {
            public DispatchRejected {
                if (!(terminal instanceof DeferredTerminal.DeliveryRejected)) {
                    throw new IllegalArgumentException(
                            "DispatchRejected requires a delivery rejection terminal");
                }
            }
        }

        record PriorityCanceled(
                PrefillEndpoint source,
                BatchItem expected,
                DeferredTerminal terminal)
                implements PreemptionFact {
        }

        record DeliveryConfirmed(long batchId)
                implements PreemptionFact {
        }

        record DecodeGenerationRetired(
                DecodeEndpoint source,
                DecodeEndpoint.ReservationHandle reservation,
                String detail)
                implements PreemptionFact {
        }
    }

    /** Immutable replay already selected under the exact slot lock. */
    sealed interface ReplayPayload {
        record Terminal(DeferredTerminal exact) implements ReplayPayload {
        }

        record Delivery(
                DeliveryConfirmation exact,
                BatchItem item,
                DeliveryClaimKind kind,
                long batchId)
                implements ReplayPayload {
        }
    }

    /** The only effects exposed after a preemption ownership reduction. */
    sealed interface PreemptionReduction {
        enum Stale implements PreemptionReduction {
            INSTANCE
        }

        enum None implements PreemptionReduction {
            INSTANCE
        }

        record StartFence(FenceHandle exact, CancelTarget target)
                implements PreemptionReduction {
        }

        record Replay(
                ReplayPayload payload,
                PreemptionRegistration signal,
                PrefillOnlyCleanup prefillOnlyCleanup)
                implements PreemptionReduction {
        }
    }

    /** Exact terminal leaves detached from a closed Engine fence. */
    record FenceTerminalProof(
            String detail,
            PreemptionRegistration transferred,
            DecodeEndpoint.AuthoritativeTerminalProof decodeProof,
            FenceCleanup cleanup) {
    }

    /** The only effects exposed after an Engine-fence reduction. */
    sealed interface FenceReduction {
        enum Stale implements FenceReduction {
            INSTANCE
        }

        enum Deferred implements FenceReduction {
            INSTANCE
        }

        enum None implements FenceReduction {
            INSTANCE
        }

        record Start(FenceHandle exact, CancelTarget target)
                implements FenceReduction {
        }

        record TerminalProof(FenceTerminalProof proof)
                implements FenceReduction {
        }

    }

    enum RequestDeadlineExpiry {
        IGNORED,
        DEFERRED_BY_ADMISSION,
        CANCEL_REQUEST
    }

    record AdmissionMutationCompletion(
            boolean owned,
            CancelReason cancellationToResume,
            FenceHandle tombstonedFence,
            DeferredTerminal pendingTerminal,
            TerminalAction pendingRetirement) {
        private static final AdmissionMutationCompletion NOT_OWNED =
                new AdmissionMutationCompletion(
                        false, null, null, null, null);
    }

    record AcceptanceExpiry(
            boolean consumed,
            BatchItem item,
            boolean needsFence,
            AdmissionCleanup cleanup) {
        private static final AcceptanceExpiry IGNORED =
                new AcceptanceExpiry(false, null, false, null);
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
    private static final class EngineFenceRegistration
            implements FenceHandle, FenceCleanup {
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

        @Override
        public void release() {
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
        private final RequestSlot slot;
        private final RequestFuture future;
        private final PublicationKind kind;
        private final PublicationLease lease;
        private final AtomicBoolean claimed = new AtomicBoolean();
        private final AtomicBoolean terminalOutstandingResolved =
                new AtomicBoolean();

        private PublicationPermit(
                RequestSlot slot,
                PublicationKind kind,
                PublicationLease lease) {
            this.slot = slot;
            this.future = slot.future;
            this.kind = kind;
            this.lease = lease;
            if (lease.exactSlot() != slot) {
                throw new IllegalArgumentException(
                        "publication lease belongs to another slot");
            }
        }

        RequestSlot slot() {
            return slot;
        }

        PublicationLease lease() {
            return lease;
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
                lease.close();
            }
        }

        /** Settle a claim whose publication could not enter its executor. */
        void abortClaimedPublication() {
            resolveTerminalOutstanding();
            lease.close();
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
        private final PrefillWorkLedger.Protection prefillProtection;
        private final DecodeEndpoint.EngineFenceLease decodeProtection;
        private boolean released;

        private EngineFenceResources(
                PrefillEndpoint prefill,
                PrefillWorkLedger.Protection prefillProtection,
                DecodeEndpoint.EngineFenceLease decodeProtection) {
            this.prefill = prefill;
            this.prefillProtection = prefillProtection;
            this.decodeProtection = decodeProtection;
        }

        private static EngineFenceResources acquire(
                BatchItem item,
                PrefillWorkLedger.Protection prefillProtection) {
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
        private synchronized PrefillOnlyCleanup
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
    private static final class ExactPrefillOnlyCleanup
            implements PrefillOnlyCleanup {
        private final PrefillEndpoint prefill;
        private final PrefillWorkLedger.Protection protection;
        private final AtomicBoolean released = new AtomicBoolean();

        private ExactPrefillOnlyCleanup(
                PrefillEndpoint prefill,
                PrefillWorkLedger.Protection protection) {
            this.prefill = prefill;
            this.protection = protection;
        }

        @Override
        public void release() {
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

/** Exact close-barrier capability acquired before a frontend lifecycle edge. */
interface PublicationLease extends AutoCloseable {
    RequestSlot exactSlot();

    @Override
    void close();
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
        boolean acceptedBeforeCancel,
        RequestSlot.FenceCleanup releasableFence,
        RequestSlot.AdmissionCleanup admissionCleanup) {
    static final DecodeAcceptance STALE =
            new DecodeAcceptance(false, null, null);
    static final DecodeAcceptance FENCE_RETAINED =
            new DecodeAcceptance(false, null, null);
}

record WorkerTerminalObservation(
        WorkerTerminalSource source,
        boolean successful,
        long errorCode) {
}

/** First ordinary terminal observed while priority Cancel owns the slot. */
sealed interface DeferredTerminal
        permits DeferredTerminal.Failure,
                DeferredTerminal.DeliveryFailure,
                DeferredTerminal.DeliveryRejected,
                DeferredTerminal.Worker,
                DeferredTerminal.Priority,
                DeferredTerminal.DecodeGenerationRetired {

    record Failure(StrategyErrorType errorType, String detail)
            implements DeferredTerminal {
        public Failure {
            Objects.requireNonNull(errorType, "errorType");
        }
    }

    record DeliveryFailure(StrategyErrorType errorType, String detail)
            implements DeferredTerminal {
        public DeliveryFailure {
            Objects.requireNonNull(errorType, "errorType");
        }
    }

    record DeliveryRejected(String detail) implements DeferredTerminal {
    }

    record Worker(WorkerTerminalObservation observation)
            implements DeferredTerminal {
        public Worker {
            Objects.requireNonNull(observation, "observation");
        }
    }

    record Priority(String detail) implements DeferredTerminal {
    }

    record DecodeGenerationRetired(String detail)
            implements DeferredTerminal {
    }

    default boolean authoritativeWorker() {
        return this instanceof Worker
                || this instanceof DecodeGenerationRetired;
    }

    default boolean endpointAlreadyRetired() {
        return this instanceof DecodeGenerationRetired;
    }

    default boolean decodeSettlementCommitted() {
        return this instanceof Worker worker
                && worker.observation().source()
                    .decodeSettlementCommitted();
    }

    default boolean deliveryFailure() {
        return this instanceof DeliveryFailure;
    }
}

/** One-shot capability moved out of an ACTIVE slot; never stored or retried. */
record TerminalAction(
        RequestSlot slot,
        BatchItem item,
        RequestSlot.FenceCleanup fence,
        RequestSlot.TerminalResources terminalResources,
        boolean removePrefillQueue,
        boolean releaseDecode,
        boolean releasePrefill,
        Runnable counterpartCleanup,
        String queueReason,
        Function<RequestSlot, RequestLifecycleSnapshot> transition,
        Response response,
        PublicationLease publicationLease) {
}

/** Non-persistent proof that a claimed terminal action reached its tombstone. */
record TombstoneResult(
        RequestLifecycleSnapshot terminal,
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
