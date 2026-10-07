package org.flexlb.balance.endpoint;

import org.flexlb.balance.endpoint.DecodeEndpoint.AdmissionCapacity;
import org.flexlb.balance.endpoint.DecodeEndpoint.CapacityRelease;
import org.flexlb.balance.endpoint.DecodeEndpoint.CapacityUsage;
import org.flexlb.balance.endpoint.DecodeEndpoint.DecodeRequestView;
import org.flexlb.balance.endpoint.DecodeEndpoint.DecodeRoutingView;
import org.flexlb.balance.endpoint.DecodeEndpoint.DispatchOutcome;
import org.flexlb.balance.endpoint.DecodeEndpoint.EngineDispatchPermitAcquireStatus;
import org.flexlb.balance.endpoint.DecodeEndpoint.EngineDispatchPermitTransferStatus;
import org.flexlb.balance.endpoint.DecodeEndpoint.LayeredAdmissionView;
import org.flexlb.balance.endpoint.DecodeEndpoint.PreemptionBeginResult;
import org.flexlb.balance.endpoint.DecodeEndpoint.PreemptionDecision;
import org.flexlb.balance.endpoint.DecodeEndpoint.PreemptionUpdate;
import org.flexlb.balance.endpoint.DecodeEndpoint.ReleaseReason;
import org.flexlb.balance.endpoint.DecodeEndpoint.ReservationHandle;
import org.flexlb.balance.endpoint.DecodeEndpoint.ReservationReleaseResult;
import org.flexlb.balance.endpoint.DecodeEndpoint.WorkerStatusFact;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.enums.TaskPhase;
import org.flexlb.util.PriorityNormalizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Predicate;

/** Resource ledger for one Decode generation. All mutations share admissionLock.
 * Endpoint owns lifecycle pins and publishes notifications only after these calls return.
 */
final class DecodeState {
    private final WorkerStatus status;

    DecodeState(WorkerStatus status) {
        this.status = java.util.Objects.requireNonNull(status, "status");
    }

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");
    private static final Comparator<ReservationHandle> RETIREMENT_ORDER =
            Comparator.comparingLong(ReservationHandle::endpointGenerationId)
                    .thenComparing(ReservationHandle::requestId)
                    .thenComparingLong(ReservationHandle::reservationToken);

    private final ConcurrentHashMap<String, DecodeRequestState> decodeRequests = new ConcurrentHashMap<>();
    private final AtomicLong inputKvReservedTotal = new AtomicLong(0);
    private final AtomicLong inputAndMaxOutputKvReservedTotal = new AtomicLong(0);
    /**
     * Engine-confirmed request ownership: KV_ALLOCATED plus RUNNING, including
     * synthetic fenced slots. This is deliberately not the Engine's physical
     * running concurrency; confirmed entries provide the phase split.
     */
    private volatile int confirmedEngineOwnedCount;

    /** Number of reservation-phase entries in {@link #decodeRequests}. */
    private final AtomicInteger reservedRequestCount = new AtomicInteger();

    private final Map<Long, EndpointPreemptionAttempt> preemptionAttempts = new HashMap<>();

    /** KV that Decode has reported free but the Prefill CANCELED fence has not settled yet. */
    private final AtomicLong priorityPreemptionHeldRequiredKv = new AtomicLong();
    /** KV capacity budget retained until the same cancellation fence settles; budget >= required KV. */
    private final AtomicLong priorityPreemptionHeldKvBudget = new AtomicLong();

    /**
     * O(1) mirror of reservations whose request is still waiting in a Prefill
     * queue. Such reservations protect KV, but do not consume Decode engine
     * concurrency until delivery obtains an {@link DispatchLease}.
     * Queue publication increments the counter; dispatch, release, and status
     * reconciliation decrement it under {@link #admissionLock}.
     */
    private final AtomicInteger queuedPhaseCount = new AtomicInteger(0);

    /** Prompt-only KV held by reservations which are still Prefill-queued. */
    private final AtomicLong queuedInputKvReservedTotal = new AtomicLong(0);

    /** Input plus max_new_tokens reserved for requests still waiting in the Prefill queue. */
    private final AtomicLong queuedInputAndMaxOutputKvReservedTotal = new AtomicLong(0);

    /**
     * Input KV already committed to acquired pre-delivery permits.
     * Permit identities carry the request generation and monotonic reservation
     * token, so a stale release cannot affect request-id reuse.
     */
    private final AtomicLong engineDispatchPermitRequiredKvReservedTotal = new AtomicLong();
    /** Input plus max_new_tokens already committed to acquired pre-delivery permits. */
    private final AtomicLong engineDispatchPermitKvBudgetReservedTotal = new AtomicLong();
    /** Mutated under admissionLock; volatile for the lock-free waiter predicate. */
    private volatile int activeEngineDispatchPermitCount;
    /** Guarded by {@link #admissionLock}; zero is never issued. */
    private long nextReservationToken = 1L;

    /**
     * Serializes reserve/release, dispatch-permit, calibration, and eviction
     * transactions. Reads stay lock-free.
     */
    private final ReentrantLock admissionLock = new ReentrantLock();

    /** Monotonic diagnostic generation for the layered admission projection. */
    private final AtomicLong admissionVersion = new AtomicLong();

    /**
     * Generation-local, non-authoritative cache for bulk routing traversal.
     *
     * <p>Both key components are required: local admission mutations advance
     * {@link #admissionVersion}, while a committed Engine observation is
     * replaced by holder identity. The cache owns no endpoint or registry-map
     * reference and therefore disappears with this endpoint generation.</p>
     */
    private volatile DecodeRoutingView routingViewCache;
    private volatile DecodeEndpoint.AdmissionSummary admissionSummaryCache;

    // Reservation: acquire and release exact ownership.

    ReservationHandle reserve(String requestId, long requiredKv, long kvBudget, int priority,
                              boolean queued, AdmissionCapacity capacity) {
        admissionLock.lock();
        try {
            if (!requestIdAvailableForReservationLocked(requestId)) { return null; }
            if (capacity != null && queuedPlacementIsFullLocked(requiredKv, kvBudget, capacity)) { return null; }
            ReservationHandle reservation = reserveLocked(requestId, requiredKv, kvBudget, priority);
            if (queued) { addQueuedPhaseLocked(requestId, shadowReservation(requestId)); }
            return reservation;
        } finally {
            admissionLock.unlock();
        }
    }

    private ReservationHandle reserveLocked(String requestId,
                                            long kvTokens,
                                            long kvBudgetTokens,
                                            int priority) {
        if (!requestIdAvailableForReservationLocked(requestId)) {
            throw new IllegalStateException(
                    "Decode request id is still owned by this endpoint generation: "
                            + requestId);
        }
        long reservationToken = nextReservationTokenLocked();
        DecodeRequestState newReservation =
                new DecodeRequestState(
                        kvTokens, kvBudgetTokens, priority, reservationToken);
        ReservationHandle handle = new ReservationHandle(
                status.getGenerationId(), requestId, reservationToken);
        if (decodeRequests.putIfAbsent(requestId, newReservation) != null) {
            throw new IllegalStateException(
                    "Decode reservation appeared while admissionLock was held: "
                            + requestId);
        }
        reservedRequestCount.incrementAndGet();
        inputKvReservedTotal.addAndGet(kvTokens);
        inputAndMaxOutputKvReservedTotal.addAndGet(kvBudgetTokens);
        admissionVersion.incrementAndGet();
        return handle;
    }

    private long nextReservationTokenLocked() {
        if (nextReservationToken <= 0L
                || nextReservationToken == Long.MAX_VALUE) {
            throw new IllegalStateException(
                    "Decode reservation token space exhausted");
        }
        return nextReservationToken++;
    }

    private boolean requestIdAvailableForReservationLocked(String requestId) {
        if (decodeRequests.containsKey(requestId)) {
            return false;
        }
        for (EndpointPreemptionAttempt attempt : preemptionAttempts.values()) {
            if (java.util.Objects.equals(attempt.incomingRequestId, requestId)) {
                return false;
            }
        }
        return true;
    }

    private boolean queuedPlacementIsFullLocked(
            long requiredKvTokens, long kvBudgetTokens, AdmissionCapacity capacity) {
        return !capacity.evaluate(routingViewLocked().placementUsage(), requiredKvTokens, kvBudgetTokens).fits();
    }

    boolean isAcceptedByEngine(ReservationHandle handle) {
        if (handle == null) {
            return false;
        }
        admissionLock.lock();
        try {
            DecodeRequestState state = requestState(handle.requestId());
            return handle.endpointGenerationId() == status.getGenerationId()
                    && state != null && (state.confirmed() || state.waitingForWorkerFinishedReport)
                    && state.reservationToken() == handle.reservationToken();
        } finally {
            admissionLock.unlock();
        }
    }

    boolean hasOwnedResources(ReservationHandle reservation) {
        if (reservation == null || reservation.endpointGenerationId() != status.getGenerationId()) { return false; }
        admissionLock.lock();
        try {
            DecodeRequestState current = requestState(reservation.requestId());
            return (isExactReservation(current, reservation)
                    && (current.countsTowardCapacity() || current.waitingForWorkerFinishedReport
                        || current.hasProtocolOwner()))
                    || hasExactIncomingAttemptLocked(reservation);
        } finally {
            admissionLock.unlock();
        }
    }

    ReservationHandle reservationHandle(String requestId) {
        admissionLock.lock();
        try {
            DecodeRequestState current = shadowReservation(requestId);
            if (current == null || current.reservationToken() <= 0L) {
                return null;
            }
            return new ReservationHandle(
                    status.getGenerationId(),
                    requestId,
                    current.reservationToken());
        } finally {
            admissionLock.unlock();
        }
    }

    ReservationReleaseResult release(ReservationHandle reservation, ReleaseReason reason) {
        java.util.Objects.requireNonNull(reason, "reason");
        if (reservation == null) {
            if (reason == ReleaseReason.LOCAL_ROLLBACK || reason == ReleaseReason.NOT_SENT) {
                throw new IllegalArgumentException("Decode reservation is required for release");
            }
            return ReservationReleaseResult.STALE;
        }
        if (reservation.endpointGenerationId() != status.getGenerationId()) { return ReservationReleaseResult.STALE; }
        admissionLock.lock();
        try {
            return switch (reason) {
                case NOT_SENT -> releaseUnsentRequestLocked(reservation);
                case LOCAL_ROLLBACK, COUNTERPART_FINISHED -> {
                    if (releaseLocalReservationLocked(reservation, reason)) { yield ReservationReleaseResult.RELEASED; }
                    yield hasOwnedResources(reservation) ? ReservationReleaseResult.STILL_OWNED : ReservationReleaseResult.STALE;
                }
                case EXPIRED -> expireRequestLocked(reservation)
                        ? ReservationReleaseResult.RELEASED : ReservationReleaseResult.STALE;
            };
        } finally {
            admissionLock.unlock();
        }
    }

    private boolean releaseLocalReservationLocked(ReservationHandle reservation, ReleaseReason reason) {
        String requestId = reservation.requestId();
        DecodeRequestState current = requestState(requestId);
        boolean exact = isExactReservation(current, reservation);
        if (exact && current.returnedPreemptionToken != 0L) {
            rollbackReturnedPreemptionLocked(current);
        }
        boolean protectedOwner = hasExactIncomingAttemptLocked(reservation)
                || exact && (current.confirmed() || current.waitingForWorkerFinishedReport
                        || current.engineLifecycleOwned || current.hasProtocolOwner());
        if (protectedOwner) {
            if (reason == ReleaseReason.LOCAL_ROLLBACK) {
                throw localReleaseInvariant(reservation, "exact ownership is held by Engine/protocol lifecycle");
            }
            return false;
        }
        if (!exact || !current.countsTowardCapacity()) { return false; }
        DispatchLease permit = current.dispatchPermit();
        if (permit != null && !isExactReservation(permit.reservation, reservation)) {
            if (reason == ReleaseReason.LOCAL_ROLLBACK) {
                throw localReleaseInvariant(reservation, "dispatch permit belongs to another reservation");
            }
            return false;
        }
        if (!removeShadowExactLocked(requestId, current)) {
            throw localReleaseInvariant(reservation, "validated shadow changed while admissionLock was held");
        }
        if (reason == ReleaseReason.COUNTERPART_FINISHED && !decodeRequests.containsKey(requestId)) {
            rememberSettledLocked(requestId, System.currentTimeMillis());
        }
        admissionVersion.incrementAndGet();
        return true;
    }

    private ReservationReleaseResult releaseUnsentRequestLocked(ReservationHandle reservation) {
        String requestId = reservation.requestId();
        DecodeRequestState current = requestState(requestId);
        boolean exact = isExactReservation(current, reservation);
        PreemptionClaim claim = exact ? current.preemptionClaim : null;
        if (exact && (current.confirmed() || current.waitingForWorkerFinishedReport
                || claim != null && claim.owner == ClaimOwner.ENGINE_CONFIRMED)) {
            return ReservationReleaseResult.ENGINE_ACCEPTED;
        }
        if (claim != null) {
            return settlePriorityClaimTerminalLocked(claim.attemptToken, reservation, claim)
                    ? ReservationReleaseResult.RELEASED : ReservationReleaseResult.CONFLICT;
        }
        if (hasExactIncomingAttemptLocked(reservation)) { return ReservationReleaseResult.CONFLICT; }
        if (!exact || !current.countsTowardCapacity()) { return ReservationReleaseResult.STALE; }
        if (settleAuthoritativeTerminalLocked(reservation, true, System.currentTimeMillis())) {
            admissionVersion.incrementAndGet();
        }
        return ReservationReleaseResult.RELEASED;
    }

    private boolean expireRequestLocked(ReservationHandle reservation) {
        String requestId = reservation.requestId();
        DecodeRequestState state = requestState(requestId);
        if (!isExactReservation(state, reservation)
                || (!state.countsTowardCapacity() && !state.waitingForWorkerFinishedReport && !state.hasProtocolOwner()
                    && !hasExactIncomingAttemptLocked(reservation))) {
            return false;
        }

        // An expired incoming request cannot retain an admission attempt.
        // Other victims keep ambiguous Engine ownership until their own
        // status or inactivity deadline settles it.
        java.util.Iterator<Map.Entry<Long, EndpointPreemptionAttempt>> attempts =
                preemptionAttempts.entrySet().iterator();
        while (attempts.hasNext()) {
            Map.Entry<Long, EndpointPreemptionAttempt> entry = attempts.next();
            EndpointPreemptionAttempt attempt = entry.getValue();
            if (!java.util.Objects.equals(attempt.incomingRequestId, requestId)
                    || attempt.incomingReservationToken
                            != reservation.reservationToken()) {
                continue;
            }
            attempts.remove();
            for (ReservationHandle victim : attempt.remainingVictims.values()) {
                PreemptionClaim victimClaim = exactPreemptionClaimLocked(
                        entry.getKey(), victim);
                if (victimClaim != null && victimClaim.phase.isLocallyReleasable()) {
                    releaseHeldKv(victimClaim);
                    removePreemptionClaimLocked(victim.requestId(), victimClaim);
                }
            }
        }

        PreemptionClaim claim = state.preemptionClaim;
        boolean confirmedSlot = state.confirmed()
                || claim != null && claim.owner == ClaimOwner.ENGINE_CONFIRMED;
        if (claim != null) {
            EndpointPreemptionAttempt attempt = preemptionAttempts.get(claim.attemptToken);
            if (attempt != null) {
                attempt.remainingVictims.remove(requestId, reservation);
            }
            releaseHeldKv(claim);
            removePreemptionClaimLocked(requestId, claim);
        }
        if (state.confirmed()) {
            removeConfirmedExactLocked(requestId, state);
        } else if (state.countsTowardCapacity()) {
            removeShadowExactLocked(requestId, state);
        }
        if (confirmedSlot) {
            confirmedEngineOwnedCount = Math.max(0, confirmedEngineOwnedCount - 1);
        }
        // Use a history-only entry, so no old exact token remains live.
        decodeRequests.remove(requestId, state);
        rememberSettledLocked(requestId, System.currentTimeMillis());
        admissionVersion.incrementAndGet();
        return true;
    }

    private boolean settleAuthoritativeTerminalLocked(ReservationHandle reservation,
                                                      boolean retainTerminalRecord,
                                                      long settledAtMs) {
        String requestId = reservation.requestId();
        DecodeRequestState state = requestState(requestId);
        boolean exactState = isExactReservation(state, reservation);

        PreemptionClaim claim = state == null ? null : state.preemptionClaim;
        if (claim != null && exactState && !claim.returnedInstruction) {
            throw terminalInvariant(reservation,
                    "priority claim must settle before generic terminal ownership");
        }
        if (hasExactIncomingAttemptLocked(reservation)) {
            throw terminalInvariant(reservation,
                    "priority attempt still owns the exact incoming reservation");
        }

        boolean changed = false;
        if (claim != null && exactState && claim.returnedInstruction) {
            return settlePriorityClaimTerminalLocked(
                    claim.attemptToken, reservation, claim);
        }
        DecodeRequestState request = state != null && state.countsTowardCapacity()
                ? state : null;
        DispatchLease dispatchPermit = request == null
                || request.confirmed() ? null : request.dispatchPermit();
        if (dispatchPermit != null
                && isExactReservation(
                        dispatchPermit.reservation, reservation)) {
            changed = removeEngineDispatchPermitLocked(requestId) || changed;
        }

        if (isExactReservation(request, reservation)) {
            if (request.confirmed()
                    && removeConfirmedExactLocked(requestId, request)) {
                confirmedEngineOwnedCount = Math.max(
                        0, confirmedEngineOwnedCount - 1);
                changed = true;
            } else if (!request.confirmed()
                    && removeShadowExactLocked(requestId, request)) {
                changed = true;
            }
        }
        if (exactState && state.waitingForWorkerFinishedReport) {
            decodeRequests.remove(requestId, state);
            changed = true;
        }
        if (!decodeRequests.containsKey(requestId) && retainTerminalRecord) {
            changed = rememberSettledLocked(requestId, settledAtMs) || changed;
        }

        if (hasExactOwnerLocked(reservation)) {
            throw terminalInvariant(reservation,
                    "exact accounting remains after authoritative settlement");
        }
        return changed;
    }

    private boolean settleUntrackedWorkerTerminalLocked(String requestId) {
        DecodeRequestState confirmed = confirmedRequest(requestId);
        if (confirmed == null || confirmed.reservationToken() > 0L
                || !removeConfirmedExactLocked(requestId, confirmed)) {
            return false;
        }
        confirmedEngineOwnedCount = Math.max(0, confirmedEngineOwnedCount - 1);
        return true;
    }

    private boolean removeShadowExactLocked(String requestId, DecodeRequestState expected) {
        if (expected == null || expected.confirmed()
                || requestState(requestId) != expected
                || !expected.countsTowardCapacity()) {
            return false;
        }
        reservedRequestCount.decrementAndGet();
        clearShadowAccountingLocked(requestId, expected);
        expected.stopCountingTowardCapacity();
        pruneRequestStateLocked(requestId, expected);
        return true;
    }

    private void clearShadowAccountingLocked(
            String requestId, DecodeRequestState reservation) {
        removeEngineDispatchPermitLocked(reservation);
        reservation.engineLifecycleOwned = false;
        removeQueuedPhaseLocked(requestId, reservation);
        inputKvReservedTotal.addAndGet(-reservation.kvTokens());
        inputAndMaxOutputKvReservedTotal.addAndGet(
                -reservation.kvBudgetTokens());
    }

    private boolean removeConfirmedExactLocked(String requestId, DecodeRequestState expected) {
        if (expected == null || !expected.confirmed()
                || requestState(requestId) != expected) {
            return false;
        }
        expected.stopCountingTowardCapacity();
        pruneRequestStateLocked(requestId, expected);
        return true;
    }

    private static IllegalStateException localReleaseInvariant(
            ReservationHandle reservation,
            String detail) {
        return new IllegalStateException(
                "Illegal Decode local release: requestId="
                        + reservation.requestId()
                        + ", reservationToken="
                        + reservation.reservationToken()
                        + ", detail=" + detail);
    }

    private static IllegalStateException terminalInvariant(
            ReservationHandle reservation,
            String detail) {
        return new IllegalStateException(
                "Illegal Decode authoritative settlement: requestId="
                        + reservation.requestId()
                        + ", reservationToken="
                        + reservation.reservationToken()
                        + ", detail=" + detail);
    }

    // Dispatch: queue membership, permit acquisition, handoff and rollback.

    boolean markQueued(
            ReservationHandle reservation) {
        if (reservation == null) {
            throw new IllegalArgumentException(
                    "Decode reservation is required for queued transition");
        }
        admissionLock.lock();
        try {
            if (reservation.endpointGenerationId()
                    != status.getGenerationId()) {
                return false;
            }
            String requestId = reservation.requestId();
            DecodeRequestState current = shadowReservation(requestId);
            if (!isExactReservation(current, reservation)) {
                return false;
            }
            if (current.queued()) {
                return true;
            }
            addQueuedPhaseLocked(requestId, current);
            // Re-queueing begins a new dispatch round. Invalidate any
            // pre-delivery lease before publishing that transition.
            removeEngineDispatchPermitLocked(requestId);
            admissionVersion.incrementAndGet();
        } finally {
            admissionLock.unlock();
        }
        return true;
    }

    private boolean addQueuedPhaseLocked(String requestId, DecodeRequestState reservation) {
        if (reservation == null || !reservation.markQueued()) {
            return false;
        }
        queuedPhaseCount.incrementAndGet();
        queuedInputKvReservedTotal.addAndGet(reservation.kvTokens());
        queuedInputAndMaxOutputKvReservedTotal.addAndGet(reservation.kvBudgetTokens());
        return true;
    }

    private boolean removeQueuedPhaseLocked(String requestId, DecodeRequestState reservation) {
        if (reservation == null) {
            throw new IllegalStateException(
                    "queued Decode reservation missing for request " + requestId);
        }
        if (!reservation.clearQueued()) {
            return false;
        }
        queuedPhaseCount.decrementAndGet();
        queuedInputKvReservedTotal.addAndGet(-reservation.kvTokens());
        queuedInputAndMaxOutputKvReservedTotal.addAndGet(-reservation.kvBudgetTokens());
        return true;
    }

    DispatchAcquisition acquireDispatchPermit(ReservationHandle handle, AdmissionCapacity capacity) {
        admissionLock.lock();
        try {
            DecodeRequestState reservation = requestState(handle.requestId());
            if (handle.endpointGenerationId() != status.getGenerationId()
                    || !isExactReservation(reservation, handle)
                    || !reservation.countsTowardCapacity() || reservation.preemptionClaim != null) {
                return new DispatchAcquisition(EngineDispatchPermitAcquireStatus.NOT_OWNED, null);
            }
            if (reservation.confirmed()) {
                return new DispatchAcquisition(
                        EngineDispatchPermitAcquireStatus.ALREADY_ACCEPTED,
                        new DispatchLease(handle.requestId(), reservation));
            }
            if (!reservation.queued()) {
                return new DispatchAcquisition(EngineDispatchPermitAcquireStatus.NOT_QUEUED, null);
            }
            if (reservation.dispatchPermit() != null) {
                return new DispatchAcquisition(EngineDispatchPermitAcquireStatus.ALREADY_ACQUIRED, null);
            }
            if (isEngineDispatchCapacityFullSnapshot(
                    reservation, capacity, status.committedWorkerStatus().fields())) {
                return new DispatchAcquisition(EngineDispatchPermitAcquireStatus.CAPACITY_FULL, null);
            }

            DispatchLease permit = installEngineDispatchPermitLocked(
                    handle.requestId(), reservation);
            return new DispatchAcquisition(
                    EngineDispatchPermitAcquireStatus.ACQUIRED, permit);
        } finally {
            admissionLock.unlock();
        }
    }

    private DispatchLease installEngineDispatchPermitLocked(
            String requestId,
            DecodeRequestState reservation) {
        DispatchLease permit = new DispatchLease(requestId, reservation);
        reservation.installDispatchPermit(permit);
        activeEngineDispatchPermitCount++;
        engineDispatchPermitRequiredKvReservedTotal.addAndGet(
                reservation.kvTokens());
        engineDispatchPermitKvBudgetReservedTotal.addAndGet(
                reservation.kvBudgetTokens());
        admissionVersion.incrementAndGet();
        return permit;
    }

    DispatchResult dispatch(DispatchLease permit, DispatchOutcome outcome) {
        java.util.Objects.requireNonNull(permit, "permit");
        java.util.Objects.requireNonNull(outcome, "outcome");
        if (outcome == DispatchOutcome.ENGINE_OWNED) { return dispatchToEngine(permit); }
        admissionLock.lock();
        try {
            if (permit.retiredByEndpoint) {
                return new DispatchResult(EngineDispatchPermitTransferStatus.TRANSFERRED, false);
            }
            if (!isCurrentEngineDispatchPermitLocked(permit)) {
                return new DispatchResult(EngineDispatchPermitTransferStatus.OWNERSHIP_LOST, false);
            }
            removeEngineDispatchPermitLocked(permit.requestId);
            admissionVersion.incrementAndGet();
            return new DispatchResult(EngineDispatchPermitTransferStatus.TRANSFERRED, true);
        } finally {
            admissionLock.unlock();
        }
    }

    private DispatchResult dispatchToEngine(DispatchLease permit) {
        EngineDispatchPermitTransferStatus transferStatus;
        boolean capacityIncreased;
        admissionLock.lock();
        try {
            // Engine status may consume the acquired permit before publication.
            // Only the same canonical reservation can satisfy this handoff.
            DecodeRequestState current = requestState(permit.requestId);
            if (current == permit.reservation && (current.confirmed() || current.waitingForWorkerFinishedReport)
                    && current.preemptionClaim == null) {
                return new DispatchResult(EngineDispatchPermitTransferStatus.TRANSFERRED, false);
            }
            int usageBefore = engineDispatchHardGateUsageLocked();
            if (!isCurrentEngineDispatchPermitLocked(permit)) {
                return new DispatchResult(EngineDispatchPermitTransferStatus.OWNERSHIP_LOST, false);
            }
            if (shadowReservation(permit.requestId) != permit.reservation
                    || !permit.reservation.queued()
                    || permit.reservation.preemptionClaim != null) {
                removeEngineDispatchPermitLocked(permit.requestId);
                admissionVersion.incrementAndGet();
                transferStatus = EngineDispatchPermitTransferStatus.OWNERSHIP_LOST;
            } else {
                removeEngineDispatchPermitLocked(permit.requestId);
                // The identity and queued membership were checked while holding the
                // same lock. Transfer only changes ownership; it never re-reads the cap.
                removeQueuedPhaseLocked(permit.requestId, permit.reservation);
                permit.reservation.engineLifecycleOwned = true;
                if (permit.reservation.returnedPreemptionToken != 0L) {
                    preemptionAttempts.remove(
                            permit.reservation.returnedPreemptionToken);
                    permit.reservation.returnedPreemptionToken = 0L;
                }
                admissionVersion.incrementAndGet();
                transferStatus = EngineDispatchPermitTransferStatus.TRANSFERRED;
            }
            capacityIncreased = engineDispatchHardGateUsageLocked() < usageBefore;
        } finally {
            admissionLock.unlock();
        }
        return new DispatchResult(transferStatus, capacityIncreased);
    }

    private boolean isCurrentEngineDispatchPermitLocked(DispatchLease permit) {
        return permit.reservation.dispatchPermit() == permit
                && shadowReservation(permit.requestId) == permit.reservation;
    }

    private boolean removeEngineDispatchPermitLocked(String requestId) {
        DecodeRequestState reservation = shadowReservation(requestId);
        return reservation != null
                && removeEngineDispatchPermitLocked(reservation);
    }

    private boolean removeEngineDispatchPermitLocked(
            DecodeRequestState reservation) {
        DispatchLease removed = reservation.clearDispatchPermit();
        if (removed == null) {
            return false;
        }
        engineDispatchPermitRequiredKvReservedTotal.addAndGet(
                -removed.reservation.kvTokens());
        engineDispatchPermitKvBudgetReservedTotal.addAndGet(
                -removed.reservation.kvBudgetTokens());
        if (--activeEngineDispatchPermitCount < 0) {
            throw new IllegalStateException("negative active Decode dispatch permit count");
        }
        return true;
    }

    boolean shouldRetryDispatch(
            String requestId,
            AdmissionCapacity capacity) {
        DecodeRequestState candidate = shadowReservation(requestId);
        if (candidate == null
                || !candidate.queued()
                || candidate.dispatchPermit() != null) {
            return true;
        }
        WorkerStatus.CommittedWorkerStatus committed =
                status.committedWorkerStatus();
        return !isEngineDispatchCapacityFullSnapshot(
                candidate,
                capacity,
                committed.fields());
    }

    private boolean isEngineDispatchCapacityFullSnapshot(
            DecodeRequestState candidate,
            AdmissionCapacity capacity,
            WorkerStatus.EngineObservation fields) {
        if (candidate.returnedPreemptionToken != 0L) {
            return false;
        }
        return !capacity.evaluate(dispatchCapacityUsage(fields), candidate.kvTokens(), candidate.kvBudgetTokens()).fits();
    }

    private int engineDispatchHardGateUsageLocked() {
        int engineFacingInflight = Math.max(0,
                reservedRequestCount.get() - queuedPhaseCount.get());
        return confirmedEngineOwnedCount + engineFacingInflight
                + activeEngineDispatchPermitCount;
    }

    private CapacityUsage dispatchCapacityUsage(WorkerStatus.EngineObservation fields) {
        long heldRequiredKv = priorityPreemptionHeldRequiredKv.get();
        long dispatchInputKv = saturatedAddNonNegative(
                saturatedAddNonNegative(Math.max(0L, inputKvReservedTotal.get() - queuedInputKvReservedTotal.get()),
                        engineDispatchPermitRequiredKvReservedTotal.get()), heldRequiredKv);
        return new CapacityUsage(
                getEngineLoad() + Math.max(0, activeEngineDispatchPermitCount),
                Math.max(0L, fields.totalKvCacheTokens()), Math.max(0L, fields.availableKvCacheTokens()),
                dispatchInputKv, engineFacingKvUsed(fields));
    }

    private long engineFacingKvUsed(
            WorkerStatus.EngineObservation fields) {
        long totalCap = fields.totalKvCacheTokens();
        long avail = fields.availableKvCacheTokens();
        long reportedUsed = totalCap > 0 ? Math.max(0, totalCap - avail) : 0;
        long localEngineFacing = Math.max(0L,
                inputAndMaxOutputKvReservedTotal.get() - queuedInputAndMaxOutputKvReservedTotal.get())
                + engineDispatchPermitKvBudgetReservedTotal.get();
        return saturatedAddNonNegative(
                saturatedAddNonNegative(reportedUsed, localEngineFacing),
                priorityPreemptionHeldKvBudget.get());
    }

    private int getEngineLoad() {
        int inflight = reservedRequestCount.get();
        int queued = queuedPhaseCount.get();
        if (queued < 0 || queued > inflight) {
            queued = Math.max(0, Math.min(queued, inflight));
        }
        return confirmedEngineOwnedCount + Math.max(0, inflight - queued);
    }

    record DispatchAcquisition(EngineDispatchPermitAcquireStatus status, DispatchLease permit) { }

    record DispatchResult(EngineDispatchPermitTransferStatus status, boolean capacityReleased) { }

    static final class DispatchLease {
        private final String requestId;
        private final DecodeRequestState reservation;
        private boolean retiredByEndpoint;

        private DispatchLease(String requestId, DecodeRequestState reservation) {
            this.requestId = requestId;
            this.reservation = reservation;
        }
    }

    // Preemption: local replacement and the complete remote-cancel resource transaction.

    boolean replaceQueuedRequests(
            List<ReservationHandle> victims,
            String incomingRequestId, long kvTokens, long kvBudgetTokens,
            int priority,
            AdmissionCapacity capacity) {
        admissionLock.lock();
        try {
            if (victims == null || victims.isEmpty()
                    || !requestIdAvailableForReservationLocked(
                            incomingRequestId)) {
                return false;
            }
            Set<String> uniqueVictims = new HashSet<>(victims.size());
            CapacityRelease released = CapacityRelease.NONE;
            for (ReservationHandle victim : victims) {
                if (victim == null
                        || victim.endpointGenerationId()
                                != status.getGenerationId()
                        || java.util.Objects.equals(victim.requestId(), incomingRequestId)
                        || !uniqueVictims.add(victim.requestId())) {
                    return false;
                }
                DecodeRequestState held = shadowReservation(victim.requestId());
                DispatchLease permit = held == null
                        ? null : held.dispatchPermit();
                if (!isExactReservation(held, victim)
                        || !held.queued()
                        || hasEngineLifecycleReservationExactLocked(victim)
                        || held.hasProtocolOwner()
                        || hasExactIncomingAttemptLocked(victim)
                        || permit != null) {
                    return false;
                }
                released = released.plus(held.capacityRelease());
            }
            if (projectedEvictionCapacityFitsLocked(
                    capacity, kvTokens, kvBudgetTokens, CapacityRelease.NONE)) {
                return false;
            }
            if (!projectedEvictionCapacityFitsLocked(
                    capacity, kvTokens, kvBudgetTokens, released)) {
                return false;
            }

            for (ReservationHandle victim : victims) {
                DecodeRequestState exact = shadowReservation(victim.requestId());
                if (!removeShadowExactLocked(victim.requestId(), exact)) {
                    throw localReleaseInvariant(
                            victim,
                            "validated victim changed while admissionLock was held");
                }
            }
            reserveLocked(
                    incomingRequestId,
                    kvTokens,
                    kvBudgetTokens,
                    priority);
            return true;
        } finally {
            admissionLock.unlock();
        }
    }

    private boolean projectedEvictionCapacityFitsLocked(
            AdmissionCapacity capacity, long requiredKvTokens, long kvBudgetTokens, CapacityRelease released) {
        return capacity.evaluate(routingViewLocked().placementUsage(), requiredKvTokens, kvBudgetTokens, released).fits();
    }

    PreemptionBeginResult beginPreemption(
            long attemptToken,
            List<ReservationHandle> victims,
            String incomingRequestId,
            long incomingKvTokens,
            long incomingKvBudgetTokens,
            int incomingPriority,
            AdmissionCapacity capacity) {
        admissionLock.lock();
        try {
            if (preemptionAttempts.containsKey(attemptToken)) {
                return PreemptionBeginResult.ATTEMPT_ALREADY_EXISTS;
            }
            if (!requestIdAvailableForReservationLocked(incomingRequestId)) {
                return PreemptionBeginResult.INCOMING_ALREADY_RESERVED;
            }

            Map<String, ClaimOwner> owners = new HashMap<>();
            Map<String, ReservationHandle> exactVictims = new HashMap<>();
            CapacityRelease released = CapacityRelease.NONE;
            for (ReservationHandle victim : victims) {
                if (victim == null
                        || victim.endpointGenerationId()
                                != status.getGenerationId()
                        || java.util.Objects.equals(victim.requestId(), incomingRequestId)
                        || owners.containsKey(victim.requestId())) {
                    return PreemptionBeginResult.VICTIM_GONE;
                }
                String victimId = victim.requestId();
                DecodeRequestState victimState = requestState(victimId);
                if (victimState != null && victimState.hasProtocolOwner()
                        || hasExactIncomingAttemptLocked(victim)) {
                    return PreemptionBeginResult.VICTIM_ALREADY_CLAIMED;
                }
                DecodeRequestState request = ownedRequest(victimId);
                if (!isExactReservation(request, victim)) {
                    return PreemptionBeginResult.VICTIM_GONE;
                }
                DispatchLease dispatchPermit = request.confirmed()
                        ? null : request.dispatchPermit();
                if (dispatchPermit != null) {
                    return PreemptionBeginResult.VICTIM_GONE;
                }
                if (!request.confirmed() && !request.queued()) {
                    if (request.priority() <= 0
                            || request.priority() >= incomingPriority) {
                        return PreemptionBeginResult.INVALID_PRIORITY;
                    }
                    owners.put(victimId, ClaimOwner.SHADOW_IN_FLIGHT);
                    exactVictims.put(victimId, victim);

                } else if (request.confirmed()
                        && request.phase().isEngineConfirmed()) {
                    if (request.priority() <= 0
                            || request.priority() >= incomingPriority) {
                        return PreemptionBeginResult.INVALID_PRIORITY;
                    }
                    owners.put(victimId, ClaimOwner.ENGINE_CONFIRMED);
                    exactVictims.put(victimId, victim);

                } else {
                    return PreemptionBeginResult.VICTIM_GONE;
                }
                released = released.plus(request.capacityRelease());
            }
            if (projectedEvictionCapacityFitsLocked(
                    capacity, incomingKvTokens, incomingKvBudgetTokens,
                    CapacityRelease.NONE)) {
                return PreemptionBeginResult.INFEASIBLE;
            }
            if (!projectedEvictionCapacityFitsLocked(
                    capacity, incomingKvTokens, incomingKvBudgetTokens, released)) {
                return PreemptionBeginResult.INFEASIBLE;
            }

            // Allocate every victim claim before installing any incoming or
            // protocol ownership. The endpoint lock keeps these exact states
            // stable through the subsequent allocation-free installation.
            Map<String, PreemptionClaim> preparedClaims = new HashMap<>();
            for (ReservationHandle victim : victims) {
                String victimId = victim.requestId();
                DecodeRequestState request = ownedRequest(victimId);
                long requiredKv = request.kvTokens();
                long kvBudget = request.confirmed()
                        ? requiredKv : request.kvBudgetTokens();
                preparedClaims.put(
                        victimId,
                        new PreemptionClaim(
                                attemptToken,
                                owners.get(victimId),
                                requiredKv,
                                kvBudget));
            }

            // Provisional incoming ownership closes the free-pool race while
            // Cancel runs.  It is not visible to the prefill queue yet.
            ReservationHandle incomingReservation = reserveLocked(
                    incomingRequestId, incomingKvTokens,
                    incomingKvBudgetTokens, incomingPriority);
            EndpointPreemptionAttempt preparedAttempt = null;
            try {
                preparedAttempt = new EndpointPreemptionAttempt(
                        incomingRequestId,
                        incomingReservation.reservationToken(),
                        exactVictims);
                EndpointPreemptionAttempt previous = preemptionAttempts.put(
                        attemptToken, preparedAttempt);
                if (previous != null) {
                    throw new IllegalStateException(
                            "priority attempt appeared while admissionLock was held");
                }
                for (Map.Entry<String, PreemptionClaim> claim
                        : preparedClaims.entrySet()) {
                    DecodeRequestState request = ownedRequest(claim.getKey());
                    if (request == null || request.preemptionClaim != null) {
                        throw new IllegalStateException(
                                "validated priority victim changed before claim installation");
                    }
                    request.preemptionClaim = claim.getValue();
                }
            } catch (RuntimeException | Error installationFailure) {
                if (preparedAttempt != null) {
                    preemptionAttempts.remove(attemptToken, preparedAttempt);
                }
                for (Map.Entry<String, PreemptionClaim> claim
                        : preparedClaims.entrySet()) {
                    removePreemptionClaimLocked(claim.getKey(), claim.getValue());
                }
                DecodeRequestState incoming =
                        shadowReservation(incomingRequestId);
                if (isExactReservation(incoming, incomingReservation)) {
                    removeShadowExactLocked(incomingRequestId, incoming);
                }
                throw installationFailure;
            }
            admissionVersion.incrementAndGet();
            return PreemptionBeginResult.SUCCESS;
        } finally {
            admissionLock.unlock();
        }
    }

    PreemptionBeginResult beginReturnedPreemption(
            long attemptToken,
            List<ReservationHandle> victims,
            String incomingRequestId,
            long incomingKvTokens,
            long incomingKvBudgetTokens,
            int incomingPriority,
            AdmissionCapacity capacity) {
        admissionLock.lock();
        try {
            PreemptionBeginResult result = beginPreemption(
                    attemptToken, victims, incomingRequestId,
                    incomingKvTokens, incomingKvBudgetTokens,
                    incomingPriority, capacity);
            if (result == PreemptionBeginResult.SUCCESS) {
                shadowReservation(incomingRequestId).returnedPreemptionToken = attemptToken;
                for (ReservationHandle victim : victims) {
                    preemptionClaim(victim.requestId()).returnedInstruction = true;
                }
            }
            return result;
        } finally {
            admissionLock.unlock();
        }
    }

    private void rollbackReturnedPreemptionLocked(DecodeRequestState incoming) {
        long token = incoming.returnedPreemptionToken;
        if (token == 0L) {
            return;
        }
        EndpointPreemptionAttempt attempt = preemptionAttempts.remove(token);
        incoming.returnedPreemptionToken = 0L;
        if (attempt == null) {
            return;
        }
        for (ReservationHandle victim : attempt.remainingVictims.values()) {
            PreemptionClaim claim = exactPreemptionClaimLocked(token, victim);
            if (claim == null) {
                continue;
            }
            releaseHeldKv(claim);
            removePreemptionClaimLocked(victim.requestId(), claim);
        }
        admissionVersion.incrementAndGet();
    }

    boolean updatePreemption(long attemptToken, PreemptionUpdate update) {
        java.util.Objects.requireNonNull(update, "update");
        admissionLock.lock();
        try {
            if (update.kind() == PreemptionUpdate.Kind.CANCEL_SENDING) {
                return startCancelLocked(attemptToken);
            }
            if (update.kind() == PreemptionUpdate.Kind.CANCEL_REPLY) {
                PreemptionClaim claim = preemptionClaim(update.requestId());
                if (claim == null || claim.attemptToken != attemptToken
                        || !claim.phase.canTransitionTo(update.phase())) { return false; }
                claim.phase = update.phase();
                admissionVersion.incrementAndGet();
                return true;
            }
            ReservationHandle victim = update.reservation();
            if (victim.endpointGenerationId() != status.getGenerationId()) { return false; }
            PreemptionClaim claim = exactPreemptionClaimLocked(attemptToken, victim);
            if (claim == null) { return false; }
            return switch (update.kind()) {
                case CANCELED -> claim.phase.acceptsPriorityTerminal()
                        && settlePriorityClaimTerminalLocked(attemptToken, victim, claim);
                case REQUEST_FENCED -> claim.phase.acceptsRequestFenced()
                        && settlePriorityClaimTerminalLocked(attemptToken, victim, claim);
                case ACTIVE, FINISHED -> observeVictimLocked(victim, claim, update.kind());
                default -> throw new IllegalStateException("Unhandled preemption update: " + update.kind());
            };
        } finally {
            admissionLock.unlock();
        }
    }

    private boolean startCancelLocked(long attemptToken) {
        EndpointPreemptionAttempt attempt = preemptionAttempts.get(attemptToken);
        if (attempt == null) {
            return false;
        }
        for (ReservationHandle victim
                : attempt.remainingVictims.values()) {
            String victimId = victim.requestId();
            PreemptionClaim claim = exactPreemptionClaimLocked(
                    attemptToken, victim);
            if (claim == null
                    || !claim.phase.canTransitionTo(
                            PreemptionCancelPhase.CANCEL_IN_FLIGHT)) {
                return false;
            }
        }
        for (ReservationHandle victim
                : attempt.remainingVictims.values()) {
            preemptionClaim(victim.requestId()).phase =
                    PreemptionCancelPhase.CANCEL_IN_FLIGHT;
        }
        admissionVersion.incrementAndGet();
        return true;
    }

    private boolean observeVictimLocked(ReservationHandle victim, PreemptionClaim claim,
                                        PreemptionUpdate.Kind evidence) {
        boolean finished = evidence == PreemptionUpdate.Kind.FINISHED;
        if (finished ? !claim.phase.requiresOrdinaryReconciliation()
                : claim.phase != PreemptionCancelPhase.NOT_FOUND_STALE) { return false; }
        releaseHeldKv(claim);
        removePreemptionClaimLocked(victim.requestId(), claim);
        if (finished) {
            settleAuthoritativeTerminalLocked(victim, false, System.currentTimeMillis());
        }
        admissionVersion.incrementAndGet();
        return true;
    }

    private boolean settlePriorityClaimTerminalLocked(long attemptToken,
                                                      ReservationHandle reservation,
                                                      PreemptionClaim claim) {
        String requestId = reservation.requestId();
        DecodeRequestState state = requestState(requestId);
        if (exactPreemptionClaimLocked(attemptToken, reservation) != claim) {
            return false;
        }

        DecodeRequestState request = state.countsTowardCapacity() ? state : null;
        DispatchLease dispatchPermit = request == null
                || request.confirmed() ? null : request.dispatchPermit();
        if ((request != null && !isExactReservation(request, reservation))
                || (dispatchPermit != null
                    && !isExactReservation(
                            dispatchPermit.reservation, reservation))
                || hasExactIncomingAttemptLocked(reservation)) {
            return false;
        }
        EndpointPreemptionAttempt attempt =
                preemptionAttempts.get(attemptToken);
        if (attempt != null
                && !reservation.equals(
                        attempt.remainingVictims.get(requestId))) {
            return false;
        }

        if (dispatchPermit != null) {
            removeEngineDispatchPermitLocked(requestId);
        }
        if (request != null && request.confirmed()) {
            removeConfirmedExactLocked(requestId, request);
        }
        if (claim.owner == ClaimOwner.ENGINE_CONFIRMED) {
            confirmedEngineOwnedCount = Math.max(0, confirmedEngineOwnedCount - 1);
        }
        if (request != null && !request.confirmed()) {
            removeShadowExactLocked(requestId, request);
        }
        releaseHeldKv(claim);
        removePreemptionClaimLocked(requestId, claim);
        if (attempt != null) {
            attempt.remainingVictims.remove(requestId, reservation);
        }
        rememberSettledLocked(requestId, System.currentTimeMillis());
        admissionVersion.incrementAndGet();
        return true;
    }

    boolean finishPreemption(long attemptToken, PreemptionDecision decision) {
        java.util.Objects.requireNonNull(decision, "decision");
        admissionLock.lock();
        try {
            EndpointPreemptionAttempt attempt = preemptionAttempts.get(attemptToken);
            if (attempt == null) { return false; }
            if (decision == PreemptionDecision.COMMIT) {
                DecodeRequestState incoming = shadowReservation(attempt.incomingRequestId);
                if (incoming == null || incoming.reservationToken() != attempt.incomingReservationToken
                        || !attempt.remainingVictims.isEmpty()) { return false; }
            } else {
                releaseLocked(attempt.incomingRequestId);
                for (ReservationHandle victim : attempt.remainingVictims.values()) {
                    PreemptionClaim claim = exactPreemptionClaimLocked(attemptToken, victim);
                    if (claim != null && claim.phase.isLocallyReleasable()) {
                        releaseHeldKv(claim);
                        removePreemptionClaimLocked(victim.requestId(), claim);
                    }
                }
            }
            preemptionAttempts.remove(attemptToken);
            admissionVersion.incrementAndGet();
            return true;
        } finally {
            admissionLock.unlock();
        }
    }

    private boolean releaseLocked(String requestId) {
        DecodeRequestState removed = shadowReservation(requestId);
        boolean changed = removeShadowExactLocked(requestId, removed);
        if (changed) {
            admissionVersion.incrementAndGet();
        }
        return changed;
    }

    private PreemptionClaim preemptionClaim(String requestId) {
        DecodeRequestState state = requestState(requestId);
        return state == null ? null : state.preemptionClaim;
    }

    private PreemptionClaim exactPreemptionClaimLocked(
            long attemptToken, ReservationHandle reservation) {
        DecodeRequestState state = requestState(reservation.requestId());
        PreemptionClaim claim = state == null ? null : state.preemptionClaim;
        return claim != null
                && claim.attemptToken == attemptToken
                && isExactReservation(state, reservation) ? claim : null;
    }

    private boolean removePreemptionClaimLocked(
            String requestId, PreemptionClaim expected) {
        DecodeRequestState state = requestState(requestId);
        if (state == null || state.preemptionClaim != expected) {
            return false;
        }
        state.preemptionClaim = null;
        pruneRequestStateLocked(requestId, state);
        return true;
    }

    private boolean hasExactIncomingAttemptLocked(
            ReservationHandle reservation) {
        for (EndpointPreemptionAttempt attempt : preemptionAttempts.values()) {
            if (java.util.Objects.equals(attempt.incomingRequestId, reservation.requestId())
                    && attempt.incomingReservationToken
                            == reservation.reservationToken()) {
                return true;
            }
        }
        return false;
    }

    private void holdReleasedKv(PreemptionClaim claim) {
        requirePriorityPreemptionHoldLock();
        if (claim.kvHeldAfterWorkerRelease) {
            return;
        }
        long currentRequiredKv = priorityPreemptionHeldRequiredKv.get();
        long currentKvBudget = priorityPreemptionHeldKvBudget.get();
        requirePriorityPreemptionHoldInvariant(currentRequiredKv, currentKvBudget);
        long nextRequiredKv;
        long nextKvBudget;
        try {
            nextRequiredKv = Math.addExact(currentRequiredKv, claim.requiredKvTokens);
            nextKvBudget = Math.addExact(
                    currentKvBudget, claim.kvBudgetTokens);
        } catch (ArithmeticException overflow) {
            throw new IllegalStateException(
                    "Priority preemption KV hold counter overflow", overflow);
        }
        requirePriorityPreemptionHoldInvariant(nextRequiredKv, nextKvBudget);

        // Preserve budget >= required even for lock-free readers between writes.
        priorityPreemptionHeldKvBudget.set(nextKvBudget);
        priorityPreemptionHeldRequiredKv.set(nextRequiredKv);
        claim.kvHeldAfterWorkerRelease = true;
    }

    private void releaseHeldKv(PreemptionClaim claim) {
        requirePriorityPreemptionHoldLock();
        if (!claim.kvHeldAfterWorkerRelease) {
            return;
        }
        long currentRequiredKv = priorityPreemptionHeldRequiredKv.get();
        long currentKvBudget = priorityPreemptionHeldKvBudget.get();
        requirePriorityPreemptionHoldInvariant(currentRequiredKv, currentKvBudget);
        if (currentRequiredKv < claim.requiredKvTokens
                || currentKvBudget < claim.kvBudgetTokens) {
            throw new IllegalStateException(
                    "Priority preemption KV hold counter underflow: required="
                            + currentRequiredKv + "-" + claim.requiredKvTokens
                            + ", budget=" + currentKvBudget + "-"
                            + claim.kvBudgetTokens);
        }
        long nextRequiredKv = currentRequiredKv - claim.requiredKvTokens;
        long nextKvBudget = currentKvBudget - claim.kvBudgetTokens;
        requirePriorityPreemptionHoldInvariant(nextRequiredKv, nextKvBudget);

        // Preserve budget >= required even for lock-free readers between writes.
        priorityPreemptionHeldRequiredKv.set(nextRequiredKv);
        priorityPreemptionHeldKvBudget.set(nextKvBudget);
        claim.kvHeldAfterWorkerRelease = false;
    }

    private void requirePriorityPreemptionHoldLock() {
        if (!admissionLock.isHeldByCurrentThread()) {
            throw new IllegalStateException(
                    "Priority preemption KV hold mutation requires admissionLock");
        }
    }

    private static void requirePriorityPreemptionHoldInvariant(
            long requiredKvTokens, long kvBudgetTokens) {
        if (requiredKvTokens < 0L || kvBudgetTokens < requiredKvTokens) {
            throw new IllegalStateException(
                    "Invalid priority preemption KV hold counters: required="
                            + requiredKvTokens + ", budget=" + kvBudgetTokens);
        }
    }

    private enum ClaimOwner {
        SHADOW_IN_FLIGHT,
        ENGINE_CONFIRMED
    }

    private static final class PreemptionClaim {
        private final long attemptToken;
        private ClaimOwner owner;
        private final long requiredKvTokens;
        private final long kvBudgetTokens;
        private PreemptionCancelPhase phase = PreemptionCancelPhase.CLAIMED;
        private boolean returnedInstruction;
        private boolean kvHeldAfterWorkerRelease;

        private PreemptionClaim(long attemptToken, ClaimOwner owner,
                                long requiredKvTokens, long kvBudgetTokens) {
            if (requiredKvTokens < 0L || kvBudgetTokens < requiredKvTokens) {
                throw new IllegalArgumentException(
                        "Priority preemption claim requires expected KV >= hard KV >= 0");
            }
            this.attemptToken = attemptToken;
            this.owner = owner;
            this.requiredKvTokens = requiredKvTokens;
            this.kvBudgetTokens = kvBudgetTokens;
        }
    }

    private static final class EndpointPreemptionAttempt {
        private final String incomingRequestId;
        private final long incomingReservationToken;
        private final Map<String, ReservationHandle> remainingVictims;

        private EndpointPreemptionAttempt(
                String incomingRequestId,
                long incomingReservationToken,
                Map<String, ReservationHandle> victims) {
            this.incomingRequestId = incomingRequestId;
            this.incomingReservationToken = incomingReservationToken;
            this.remainingVictims = new HashMap<>(victims);
        }
    }

    // Calibration: apply one observation and produce immutable request facts.

    CalibrationResult calibrate(WorkerStatus.PreparedStatus prepared) {
        WorkerStatus.StatusObservation observation = prepared.observation();
        if (observation.owner() != status) {
            throw new IllegalArgumentException("Status belongs to another Decode generation");
        }
        admissionLock.lock();
        try {
            DecodeRoutingView before = routingViewLocked();
            List<WorkerStatusFact> facts = doCalibrate(observation.engine(), observation.finishedTasks());
            status.publishPreparedStatus(prepared);
            return new CalibrationResult(List.copyOf(facts), placementCapacityImproved(before, routingViewLocked()));
        } finally {
            admissionLock.unlock();
        }
    }

    void initialize(WorkerStatus.StatusObservation observation) {
        admissionLock.lock();
        try {
            if (!doCalibrate(observation.engine(), observation.finishedTasks()).isEmpty()) {
                throw new IllegalStateException("Private Decode candidate produced locally-owned status facts");
            }
        } finally {
            admissionLock.unlock();
        }
    }

    List<WorkerStatusFact> observeHeartbeat(WorkerStatus.StatusObservation observation) {
        if (observation.owner() != status) {
            throw new IllegalArgumentException(
                    "Status observation belongs to another Decode generation");
        }
        List<WorkerStatusFact> facts = new ArrayList<>(
                observation.runningTasks().size());
        admissionLock.lock();
        try {
            for (WorkerStatus.TaskObservation task
                    : observation.runningTasks().values()) {
                ReservationHandle active = workerStatusHandleLocked(
                        task.requestId());
                if (active != null) {
                    facts.add(WorkerStatusFact.active(active));
                }
            }
        } finally {
            admissionLock.unlock();
        }
        return List.copyOf(facts);
    }

    private List<WorkerStatusFact> doCalibrate(WorkerStatus.EngineObservation engine,
                                               Map<String, WorkerStatus.TaskObservation> finishedTasks) {
        admissionVersion.incrementAndGet();
        List<WorkerStatusFact> facts = new ArrayList<>();

        // Build one authoritative Decode view. Claimed victims that merely
        // disappear are held synthetically. An explicit Decode finished task
        // is a separate authoritative terminal outcome: it settles the exact
        // claim without reclassifying that outcome as priority CANCELED.
        Set<String> requestIdsInWorkerActiveReport = new HashSet<>();
        Set<String> confirmedNow = new HashSet<>();
        Set<String> terminalNow = new HashSet<>();
        for (WorkerStatus.TaskObservation task : finishedTasks.values()) {
            terminalNow.add(task.requestId());
        }
        int actualConfirmed = 0;
        int retainedConfirmed = 0;
        long now = System.currentTimeMillis();
        for (WorkerStatus.TaskObservation task
                : engine.runningTaskList().values()) {
            TaskPhase phase = task.phase();
            String requestId = task.requestId();
            DecodeRequestState current = requestState(requestId);
            if (terminalNow.contains(requestId)
                    || current != null && current.settledAtMs != 0L) {
                continue;
            }
            // Membership, allocation/running evidence and terminal evidence are distinct.
            // Engine may report RECEIVED again after freeing blocks, before publishing finished.
            requestIdsInWorkerActiveReport.add(requestId);
            if (phase == TaskPhase.KV_ALLOCATED || phase == TaskPhase.RUNNING) {
                actualConfirmed++;
                DecodeRequestState removed = shadowReservation(requestId);
                if (removed != null) {
                    clearShadowAccountingLocked(requestId, removed);
                    reservedRequestCount.decrementAndGet();
                }
                PreemptionClaim claim = preemptionClaim(requestId);
                if (claim != null
                        && claim.phase == PreemptionCancelPhase.NOT_FOUND_STALE
                        && !preemptionAttempts.containsKey(claim.attemptToken)) {
                    releaseHeldKv(claim);
                    removePreemptionClaimLocked(requestId, claim);
                    claim = null;
                }
                if (claim != null) {
                    claim.owner = ClaimOwner.ENGINE_CONFIRMED;
                    releaseHeldKv(claim);
                }
                confirmedNow.add(requestId);
                trackConfirmed(task, phase, now);
                ReservationHandle accepted = workerStatusHandleLocked(requestId);
                if (accepted != null) {
                    facts.add(WorkerStatusFact.accepted(accepted));
                }
            } else {
                if (current != null && current.confirmed()) {
                    // Preserve the exact token and last confirmed resource phase. This is a
                    // conservative ownership slot, not a claim about current GPU execution.
                    current.refresh(current.phase(), now);
                    if (current.preemptionClaim == null) { retainedConfirmed++; }
                    // Claims are counted/held below when allocation evidence is absent.
                }
                ReservationHandle active = workerStatusHandleLocked(requestId);
                if (active != null) {
                    facts.add(WorkerStatusFact.active(active));
                }
            }
        }

        // Explicit terminal evidence settles the exact request identity and
        // notifies the scheduler independently of active-snapshot membership.
        for (WorkerStatus.TaskObservation task
                : finishedTasks.values()) {
            String requestId = task.requestId();
            DecodeRequestState current = requestState(requestId);
            if (current != null && current.settledAtMs != 0L) {
                continue;
            }
            ReservationHandle terminal = workerStatusHandleLocked(requestId);
            confirmedNow.remove(requestId);
            PreemptionClaim claim = preemptionClaim(requestId);
            if (claim != null) {
                if (terminal != null
                        && settlePriorityClaimTerminalLocked(
                                claim.attemptToken, terminal, claim)) {
                    facts.add(WorkerStatusFact.terminal(
                            terminal, task.errorCode()));
                } else {
                    logger.error(
                            "Decode terminal did not match its exact priority claim: "
                                    + "request_id={} generation={}",
                            requestId, status.getGenerationId());
                }
                continue;
            }
            if (terminal != null) {
                facts.add(WorkerStatusFact.terminal(
                        terminal, task.errorCode()));
            }
            if (terminal != null) {
                settleAuthoritativeTerminalLocked(terminal, false, now);
            } else {
                settleUntrackedWorkerTerminalLocked(requestId);
            }
        }

        int syntheticallyHeldSlots = 0;
        for (Map.Entry<String, DecodeRequestState> entry : decodeRequests.entrySet()) {
            PreemptionClaim claim = entry.getValue().preemptionClaim;
            if (claim == null) {
                continue;
            }
            if (claim.owner == ClaimOwner.ENGINE_CONFIRMED
                    && !confirmedNow.contains(entry.getKey())) {
                syntheticallyHeldSlots++;
                holdReleasedKv(claim);
            }
        }

        // Requests missing from the Worker's active list no longer count toward capacity.
        // Keep their request ID and reservation token until Finished arrives or the request expires.
        // Requests being preempted keep their capacity until that operation completes.
        java.util.Iterator<Map.Entry<String, DecodeRequestState>> trackedRequestsIterator =
                decodeRequests.entrySet().iterator();
        while (trackedRequestsIterator.hasNext()) {
            Map.Entry<String, DecodeRequestState> requestEntry = trackedRequestsIterator.next();
            DecodeRequestState trackedRequest = requestEntry.getValue();
            if (!trackedRequest.countsTowardCapacity() || !trackedRequest.confirmed()) {
                continue;
            }
            String requestId = requestEntry.getKey();
            if (requestIdsInWorkerActiveReport.contains(requestId)) {
                continue;
            }
            if (trackedRequest.preemptionClaim != null) {
                continue;
            }
            // A positive token identifies a request reserved by this FlexLB instance.
            if (trackedRequest.reservationToken() > 0L) {
                trackedRequest.stopCountingTowardCapacity();
                trackedRequest.waitingForWorkerFinishedReport = true;
            } else {
                trackedRequestsIterator.remove();
                rememberSettledLocked(requestId, now);
            }
        }
        this.confirmedEngineOwnedCount = actualConfirmed + retainedConfirmed + syntheticallyHeldSlots;

        return facts;
    }

    private void trackConfirmed(WorkerStatus.TaskObservation task, TaskPhase phase, long now) {
        DecodeTaskPhase layer = phase == TaskPhase.KV_ALLOCATED
                ? DecodeTaskPhase.ACCEPTED_NOT_RUNNING
                : DecodeTaskPhase.RUNNING;
        DecodeRequestState tracked = requestState(task.requestId());
        if (tracked == null) {
            decodeRequests.put(task.requestId(),
                    DecodeRequestState.untrackedConfirmed(
                            task.inputLength(), layer, now));
        } else if (!tracked.countsTowardCapacity()) {
            tracked.confirm(task.inputLength(), layer, now);
        } else if (tracked.confirmed()) {
            tracked.refresh(layer, now);
        } else {
            tracked.confirm(task.inputLength(), layer, now);
        }
    }

    private ReservationHandle workerStatusHandleLocked(String requestId) {
        DecodeRequestState state = requestState(requestId);
        long reservationToken = state == null ? 0L : state.reservationToken();
        if (reservationToken <= 0L) {
            return null;
        }
        return new ReservationHandle(
                status.getGenerationId(),
                requestId,
                reservationToken);
    }

    private static boolean placementCapacityImproved(
            DecodeRoutingView before,
            DecodeRoutingView after) {
        return after.engineLoad() < before.engineLoad()
                || after.realKvAvailable() > before.realKvAvailable()
                || after.realKvUsed() < before.realKvUsed();
    }

    record CalibrationResult(List<WorkerStatusFact> facts, boolean capacityImproved) { }

    // Generation cleanup: drain resources and expire orphan/history records.

    List<ReservationHandle> retire() {
        admissionLock.lock();
        try {
            for (DecodeRequestState reservation : decodeRequests.values()) {
                DispatchLease permit = reservation.dispatchPermit();
                if (permit != null) { permit.retiredByEndpoint = true; }
            }
            List<ReservationHandle> retired = retireGenerationOwnershipLocked();
            admissionVersion.incrementAndGet();
            return retired;
        } finally {
            admissionLock.unlock();
        }
    }

    private List<ReservationHandle> retireGenerationOwnershipLocked() {
        Set<ReservationHandle> owners = new HashSet<>();
        long generationId = status.getGenerationId();

        decodeRequests.forEach((requestId, reservation) ->
                addRetiredOwner(
                        owners, generationId, requestId,
                        reservation.reservationToken()));
        for (EndpointPreemptionAttempt attempt : preemptionAttempts.values()) {
            addRetiredOwner(
                    owners, generationId,
                    attempt.incomingRequestId,
                    attempt.incomingReservationToken);
            for (ReservationHandle victim : attempt.remainingVictims.values()) {
                if (victim.endpointGenerationId() != generationId) {
                    logger.error(
                            "Decode retirement ignored a priority victim from another generation: "
                                    + "request_id={} expected_generation={} actual_generation={}",
                            victim.requestId(), generationId,
                            victim.endpointGenerationId());
                    continue;
                }
                owners.add(victim);
            }
        }

        List<ReservationHandle> ordered = new ArrayList<>(owners);
        ordered.sort(RETIREMENT_ORDER);
        List<ReservationHandle> retiredReservations = List.copyOf(ordered);

        // Canonical retirement commit. The immutable owner list and every
        // exact ReservationHandle have been validated above this line.
        decodeRequests.values().forEach(DecodeRequestState::clearDispatchPermit);
        activeEngineDispatchPermitCount = 0;
        engineDispatchPermitRequiredKvReservedTotal.set(0L);
        engineDispatchPermitKvBudgetReservedTotal.set(0L);

        decodeRequests.clear();
        reservedRequestCount.set(0);
        inputKvReservedTotal.set(0L);
        inputAndMaxOutputKvReservedTotal.set(0L);
        confirmedEngineOwnedCount = 0;

        queuedPhaseCount.set(0);
        queuedInputKvReservedTotal.set(0L);
        queuedInputAndMaxOutputKvReservedTotal.set(0L);

        preemptionAttempts.clear();
        priorityPreemptionHeldRequiredKv.set(0L);
        priorityPreemptionHeldKvBudget.set(0L);

        return retiredReservations;
    }

    private static void addRetiredOwner(
            Set<ReservationHandle> owners,
            long generationId,
            String requestId,
            long reservationToken) {
        if (reservationToken > 0L) {
            owners.add(new ReservationHandle(
                    generationId, requestId, reservationToken));
        }
    }

    CleanupResult evictExpiredRequests(long ttlMs, Predicate<String> retainForSchedulerCleanup) {
        int evicted;
        boolean capacityChanged;
        admissionLock.lock();
        try {
            // Scheduler-owned requests expire through their exact local lease.
            // This pass only sweeps endpoint orphans.
            evicted = evictExpiredInflightLocked(
                    ttlMs, retainForSchedulerCleanup);
            long cutoff = System.currentTimeMillis() - ttlMs;
            int trackedPurged = 0;
            java.util.Iterator<Map.Entry<String, DecodeRequestState>> trackedEvictIt =
                    decodeRequests.entrySet().iterator();
            while (trackedEvictIt.hasNext()) {
                Map.Entry<String, DecodeRequestState> entry = trackedEvictIt.next();
                if (entry.getValue().confirmed()
                        && entry.getValue().lastSeenMs() < cutoff
                        && !retainForSchedulerCleanup.test(entry.getKey())
                        && entry.getValue().preemptionClaim == null) {
                    trackedEvictIt.remove();
                    trackedPurged++;
                }
            }
            if (trackedPurged > 0) {
                confirmedEngineOwnedCount = Math.max(
                        0, confirmedEngineOwnedCount - trackedPurged);
            }
            boolean expiredWaitingRequestsRemoved = decodeRequests.entrySet()
                    .removeIf(entry -> entry.getValue().waitingForWorkerFinishedReport
                            && entry.getValue().lastSeenMs() < cutoff
                            && !retainForSchedulerCleanup.test(entry.getKey()));
            boolean settledTerminalRecordsPurged = decodeRequests.entrySet()
                    .removeIf(entry -> !entry.getValue().countsTowardCapacity()
                            && !entry.getValue().hasProtocolOwner()
                            && entry.getValue().settledAtMs != 0L
                            && entry.getValue().settledAtMs < cutoff);
            if (evicted > 0 || trackedPurged > 0 || expiredWaitingRequestsRemoved || settledTerminalRecordsPurged) {
                admissionVersion.incrementAndGet();
            }
            capacityChanged = evicted > 0 || trackedPurged > 0
                    || settledTerminalRecordsPurged;
        } finally {
            admissionLock.unlock();
        }
        return new CleanupResult(evicted, capacityChanged);
    }

    private int evictExpiredInflightLocked(long ttlMs, Predicate<String> retainForSchedulerCleanup) {
        long nowMs = System.currentTimeMillis();
        int evicted = 0;
        for (Map.Entry<String, DecodeRequestState> entry
                : decodeRequests.entrySet()) {
            String requestId = entry.getKey();
            DecodeRequestState request = entry.getValue();
            if (!request.countsTowardCapacity()
                    || nowMs - request.createdAtMs() <= ttlMs
                    || retainForSchedulerCleanup.test(requestId)
                    || request.preemptionClaim != null
                    || request.confirmed()
                    || !removeShadowExactLocked(requestId, request)) {
                continue;
            }
            evicted++;
        }
        return evicted;
    }

    private boolean rememberSettledLocked(String requestId, long settledAtMs) {
        DecodeRequestState state = requestState(requestId);
        if (state == null) {
            state = new DecodeRequestState(
                    0L, 0L, DecodeRequestState.DEFAULT_PRIORITY, 0L);
            state.stopCountingTowardCapacity();
            decodeRequests.put(requestId, state);
        }
        if (state.settledAtMs != 0L) {
            return false;
        }
        state.settledAtMs = settledAtMs;
        return true;
    }

    record CleanupResult(int expiredReservations, boolean capacityReleased) { }

    // Read-only views: capture, cache and metrics.

    DecodeEndpoint.AdmissionSummary admissionSummary() {
        DecodeEndpoint.AdmissionSummary cached = admissionSummaryCache;
        if (isCurrentAdmissionSummary(cached)) { return cached; }
        admissionLock.lock();
        try {
            cached = admissionSummaryCache;
            if (isCurrentAdmissionSummary(cached)) { return cached; }
            DecodeRoutingView routing = routingViewLocked();
            long[] requests = new long[PriorityNormalizer.MAX_PRIORITY + 1];
            long[] requiredKv = new long[PriorityNormalizer.MAX_PRIORITY + 1];
            long[] kvBudget = new long[PriorityNormalizer.MAX_PRIORITY + 1];
            long[] engineRequests = new long[PriorityNormalizer.MAX_PRIORITY + 1];
            long[] engineRequiredKv = new long[PriorityNormalizer.MAX_PRIORITY + 1];
            long[] engineKvBudget = new long[PriorityNormalizer.MAX_PRIORITY + 1];
            for (DecodeRequestState task : decodeRequests.values()) {
                if (!task.countsTowardCapacity()) { continue; }
                int priority = task.priorityKnown() && PriorityNormalizer.isValid(task.priority())
                        ? task.priority() : 0;
                requests[priority]++;
                long requiredTokens = task.confirmed() ? task.kvTokens() : task.releasableKvTokens();
                long budgetTokens = task.confirmed() ? task.kvTokens() : task.kvBudgetTokens();
                requiredKv[priority] = saturatedAddNonNegative(requiredKv[priority], requiredTokens);
                kvBudget[priority] = saturatedAddNonNegative(kvBudget[priority], budgetTokens);
                if (task.confirmed() || !task.queued() || task.dispatchPermit() != null) {
                    engineRequests[priority]++;
                    engineRequiredKv[priority] = saturatedAddNonNegative(engineRequiredKv[priority], requiredTokens);
                    engineKvBudget[priority] = saturatedAddNonNegative(engineKvBudget[priority], budgetTokens);
                }
            }
            CapacityRelease[] placementOccupancy = new CapacityRelease[PriorityNormalizer.MAX_PRIORITY + 1];
            CapacityRelease[] engineOccupancy = new CapacityRelease[PriorityNormalizer.MAX_PRIORITY + 1];
            for (int priority = 0; priority < placementOccupancy.length; priority++) {
                placementOccupancy[priority] = requests[priority] == 0 ? CapacityRelease.NONE
                        : new CapacityRelease(requests[priority], requiredKv[priority], kvBudget[priority]);
                engineOccupancy[priority] = engineRequests[priority] == 0 ? CapacityRelease.NONE
                        : new CapacityRelease(engineRequests[priority], engineRequiredKv[priority], engineKvBudget[priority]);
            }
            admissionSummaryCache = new DecodeEndpoint.AdmissionSummary(routing, placementOccupancy, engineOccupancy);
            return admissionSummaryCache;
        } finally {
            admissionLock.unlock();
        }
    }

    private boolean isCurrentAdmissionSummary(DecodeEndpoint.AdmissionSummary summary) {
        return summary != null
                && summary.routing().admissionVersion() == admissionVersion.get()
                && summary.routing().workerStatus() == status.committedWorkerStatus()
                && summary.routing().topology() == status.topologySnapshot();
    }

    LayeredAdmissionView resourceSnapshot() {
        admissionLock.lock();
        try {
            Map<String, DecodeRequestView> reserved = new HashMap<>(
                    reservedRequestCount.get());
            List<DecodeRequestView> confirmed = new java.util.ArrayList<>(
                    Math.max(0, confirmedEngineOwnedCount));
            decodeRequests.forEach((requestId, task) -> {
                if (!task.countsTowardCapacity()) {
                    return;
                }
                boolean protectedRequest = task.hasProtocolOwner();
                if (task.confirmed()) {
                    confirmed.add(new DecodeRequestView(
                            requestId, task.priority(), task.kvTokens(),
                            task.kvTokens(), task.phase(), task.priorityKnown(),
                            task.reservationToken(), false, protectedRequest));
                } else {
                    reserved.put(requestId, new DecodeRequestView(
                            requestId, task.priority(), task.releasableKvTokens(),
                            task.kvBudgetTokens(),
                            task.queued()
                                    ? DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED
                                    : DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN,
                            true,
                            task.reservationToken(), task.queued(),
                            protectedRequest));
                }
            });
            return new LayeredAdmissionView(routingViewLocked(),
                    Map.copyOf(reserved), List.copyOf(confirmed),
                    queuedPhaseCount.get(), activeEngineDispatchPermitCount);
        } finally {
            admissionLock.unlock();
        }
    }

    DecodeRoutingView routingView() {
        admissionLock.lock();
        try {
            return routingViewLocked();
        } finally {
            admissionLock.unlock();
        }
    }

    private DecodeRoutingView routingViewLocked() {
        WorkerStatus status = this.status;
        return routingViewLocked(
                status.getIpPort(),
                status.topologySnapshot(),
                status.committedWorkerStatus(),
                admissionVersion.get());
    }

    private DecodeRoutingView routingViewLocked(
            String address,
            WorkerStatus.TopologySnapshot topology,
            WorkerStatus.CommittedWorkerStatus committed,
            long version) {
        WorkerStatus.EngineObservation fields = committed.fields();
        int inflight = reservedRequestCount.get();
        int queued = Math.max(0, Math.min(queuedPhaseCount.get(), inflight));
        int totalLoad = confirmedEngineOwnedCount + inflight;
        int engineLoad = confirmedEngineOwnedCount + Math.max(0, inflight - queued);
        long reportedUsed = fields.totalKvCacheTokens() > 0
                ? Math.max(0L, fields.totalKvCacheTokens()
                        - fields.availableKvCacheTokens())
                : 0L;
        long reservedInputKv = inputKvReservedTotal.get();
        long reservedInputAndMaxOutputKv = inputAndMaxOutputKvReservedTotal.get();
        long used = saturatedAddNonNegative(
                saturatedAddNonNegative(reportedUsed, reservedInputAndMaxOutputKv),
                priorityPreemptionHeldKvBudget.get());
        long heldRequiredKv = priorityPreemptionHeldRequiredKv.get();
        long placementInputKv = saturatedAddNonNegative(reservedInputKv, heldRequiredKv);
        CapacityUsage placementUsage = new CapacityUsage(totalLoad,
                Math.max(0L, fields.totalKvCacheTokens()), Math.max(0L, fields.availableKvCacheTokens()),
                placementInputKv, used);
        CapacityUsage dispatchUsage = dispatchCapacityUsage(fields);
        return new DecodeRoutingView(
                address,
                status.getGenerationId(),
                topology,
                committed,
                version,
                totalLoad,
                engineLoad,
                placementUsage,
                dispatchUsage,
                reservedInputKv,
                reservedInputAndMaxOutputKv);
    }

    DecodeRoutingView routingViewSnapshot(String address) {
        long version = admissionVersion.get();
        WorkerStatus status = this.status;
        WorkerStatus.TopologySnapshot topology = status.topologySnapshot();
        DecodeRoutingView cached = routingViewCache;
        if (routingViewMatches(cached, address, version, topology)) {
            return cached;
        }
        admissionLock.lock();
        try {
            WorkerStatus.CommittedWorkerStatus committed =
                    status.committedWorkerStatus();
            topology = status.topologySnapshot();
            version = admissionVersion.get();
            cached = routingViewCache;
            if (routingViewMatches(cached, address, version, topology)) {
                return cached;
            }
            DecodeRoutingView routing = routingViewLocked(
                    address, topology, committed, version);
            routingViewCache = routing;
            return routing;
        } finally {
            admissionLock.unlock();
        }
    }

    private static boolean routingViewMatches(
            DecodeRoutingView cached,
            String address,
            long version,
            WorkerStatus.TopologySnapshot topology) {
        return cached != null
                && cached.address().equals(address)
                && cached.admissionVersion() == version
                && cached.topology() == topology;
    }

    long placementVersion() {
        return admissionVersion.get();
    }

    long realKvAvailable() {
        WorkerStatus.EngineObservation fields =
                status.committedWorkerStatus().fields();
        return Math.max(0, fields.availableKvCacheTokens()
                - inputKvReservedTotal.get()
                - priorityPreemptionHeldRequiredKv.get());
    }

    int getInflightCount() {
        return reservedRequestCount.get();
    }

    int getTotalLoad() {
        return confirmedEngineOwnedCount + reservedRequestCount.get();
    }

    /**
     * Numeric admission metrics without allocating per-request views.
     */
    AdmissionStats admissionStats() {
        admissionLock.lock();
        try {
            int reserved = 0;
            int running = 0;
            int accepted = 0;
            for (DecodeRequestState task : decodeRequests.values()) {
                if (!task.countsTowardCapacity()) {
                    continue;
                }
                if (!task.confirmed()) {
                    reserved++;
                } else if (task.phase() == DecodeTaskPhase.RUNNING) {
                    running++;
                } else if (task.phase() == DecodeTaskPhase.ACCEPTED_NOT_RUNNING) {
                    accepted++;
                }
            }
            int inflight = reservedRequestCount.get();
            int queued = Math.max(0, Math.min(queuedPhaseCount.get(), inflight));
            return new AdmissionStats(reserved, inputKvReservedTotal.get(), running, accepted,
                    confirmedEngineOwnedCount + Math.max(0, inflight - queued));
        } finally {
            admissionLock.unlock();
        }
    }

    record AdmissionStats(int reserved, long inputKvTokens, int running, int accepted, int engineLoad) { }

    Stats stats() {
        return new Stats(Math.max(0, getInflightCount() - queuedPhaseCount.get()), getTotalLoad(), inputAndMaxOutputKvReservedTotal.get(),
                inputKvReservedTotal.get(), inflightMaxAgeMs(System.currentTimeMillis()));
    }

    private long inflightMaxAgeMs(long nowMs) {
        long oldest = Long.MAX_VALUE;
        for (DecodeRequestState request : decodeRequests.values()) {
            if (request.countsTowardCapacity() && !request.confirmed()) {
                oldest = Math.min(oldest, request.createdAtMs());
            }
        }
        return oldest == Long.MAX_VALUE
                ? 0L : Math.max(0L, nowMs - oldest);
    }

    record Stats(int inflight, int totalLoad, long inputAndMaxOutputKvTokens, long inputKvTokens, long oldestAgeMs) { }

    // Shared request identity and accounting values.

    private DecodeRequestState requestState(String requestId) {
        return decodeRequests.get(requestId);
    }

    private DecodeRequestState ownedRequest(String requestId) {
        DecodeRequestState state = requestState(requestId);
        return state != null && state.countsTowardCapacity() ? state : null;
    }

    private DecodeRequestState shadowReservation(String requestId) {
        DecodeRequestState state = requestState(requestId);
        return state != null
                && state.phase() == DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN
                ? state : null;
    }

    private DecodeRequestState confirmedRequest(String requestId) {
        DecodeRequestState state = requestState(requestId);
        return state != null && state.confirmed() ? state : null;
    }

    private static boolean isExactReservation(
            DecodeRequestState current,
            ReservationHandle reservation) {
        return current != null
                && current.reservationToken()
                        == reservation.reservationToken();
    }

    private boolean hasExactOwnerLocked(ReservationHandle reservation) {
        String requestId = reservation.requestId();
        DecodeRequestState state = requestState(requestId);
        return isExactReservation(state, reservation)
                || hasExactIncomingAttemptLocked(reservation);
    }

    private boolean hasEngineLifecycleReservationExactLocked(
            ReservationHandle reservation) {
        DecodeRequestState current = shadowReservation(
                reservation.requestId());
        return isExactReservation(current, reservation)
                && current.engineLifecycleOwned;
    }

    private void pruneRequestStateLocked(String requestId, DecodeRequestState state) {
        if (state != null && !state.countsTowardCapacity() && !state.waitingForWorkerFinishedReport
                && !state.hasProtocolOwner() && state.settledAtMs == 0L) {
            decodeRequests.remove(requestId, state);
        }
    }

    static long saturatedAddNonNegative(long left, long right) {
        if (left < 0 || right < 0) {
            throw new IllegalArgumentException("KV admission counters must be non-negative");
        }
        return left > Long.MAX_VALUE - right ? Long.MAX_VALUE : left + right;
    }

    private static final class DecodeRequestState {
        static final int DEFAULT_PRIORITY = 0;

        private long kvTokens;
        /**
         * Before WorkerStatus confirmation: input tokens plus max_new_tokens.
         * After confirmation: actual KV tokens reported by the engine.
         */
        private long kvBudgetTokens;
        private final long createdAtMs;
        private int priority;
        private final long reservationToken;
        private boolean priorityKnown = true;
        /**
         * Request phase used for capacity accounting; null excludes the request from capacity.
         */
        private volatile DecodeTaskPhase phase = DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN;
        private long lastSeenAtMs;
        private boolean queued;
        /** True after dispatch crosses the Master rollback boundary. */
        private boolean engineLifecycleOwned;
        private DispatchLease dispatchPermit;
        /** Incoming instruction owner until dispatch crosses the rollback boundary. */
        private volatile long returnedPreemptionToken;
        /** Current protocol owners for this exact request generation. */
        private PreemptionClaim preemptionClaim;
        /** Non-zero while stale WorkerStatus must not resurrect this request id. */
        private long settledAtMs;
        /**
         * True when a previously active request is missing from the Worker's active report
         * and its Finished report has not arrived. The request ID and reservation token remain
         * available for scheduler completion, but the request no longer counts toward capacity.
         * A RUNNING or KV_ALLOCATED report clears this flag; Finished, expiration, or endpoint
         * removal clears the retained request record.
         */
        private boolean waitingForWorkerFinishedReport;

        DecodeRequestState(long kvTokens, long kvBudgetTokens,
                        int priority, long reservationToken) {
            if (reservationToken < 0L) {
                throw new IllegalArgumentException(
                        "reservationToken must be non-negative");
            }
            this.kvTokens = kvTokens;
            this.kvBudgetTokens = kvBudgetTokens;
            this.createdAtMs = System.currentTimeMillis();
            this.priority = priority;
            this.reservationToken = reservationToken;
        }

        long kvTokens() { return kvTokens; }
        long kvBudgetTokens() { return kvBudgetTokens; }
        CapacityRelease capacityRelease() {
            return new CapacityRelease(1L, kvTokens, kvBudgetTokens);
        }
        long createdAtMs() { return createdAtMs; }
        int priority() { return priority; }
        long reservationToken() { return reservationToken; }
        boolean queued() { return queued; }
        boolean countsTowardCapacity() { return phase != null; }
        boolean confirmed() { return phase != null
                && phase != DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN; }
        DecodeTaskPhase phase() { return phase; }
        boolean priorityKnown() { return priorityKnown; }
        long lastSeenAtMs() { return lastSeenAtMs; }

        void confirm(long engineKvTokens, DecodeTaskPhase phase, long observedAtMs) {
            kvTokens = Math.max(0L, engineKvTokens);
            kvBudgetTokens = kvTokens;
            this.phase = java.util.Objects.requireNonNull(phase, "phase");
            lastSeenAtMs = observedAtMs;
            waitingForWorkerFinishedReport = false;
            queued = false;
            engineLifecycleOwned = false;
            dispatchPermit = null;
        }

        static DecodeRequestState untrackedConfirmed(
                long engineKvTokens,
                DecodeTaskPhase phase,
                long observedAtMs) {
            DecodeRequestState state = new DecodeRequestState(
                    0L, 0L, DEFAULT_PRIORITY, 0L);
            state.priorityKnown = false;
            state.confirm(engineKvTokens, phase, observedAtMs);
            return state;
        }

        void refresh(DecodeTaskPhase phase, long observedAtMs) {
            if (!confirmed()) {
                throw new IllegalStateException(
                        "cannot refresh an unconfirmed Decode request");
            }
            this.phase = java.util.Objects.requireNonNull(phase, "phase");
            lastSeenAtMs = observedAtMs;
        }

        long lastSeenMs() { return lastSeenAtMs; }

        boolean markQueued() {
            if (queued) {
                return false;
            }
            queued = true;
            return true;
        }

        boolean clearQueued() {
            if (!queued) {
                return false;
            }
            queued = false;
            return true;
        }

        DispatchLease dispatchPermit() { return dispatchPermit; }

        void installDispatchPermit(DispatchLease permit) {
            if (dispatchPermit != null) {
                throw new IllegalStateException(
                        "Decode reservation already owns a dispatch permit");
            }
            dispatchPermit = java.util.Objects.requireNonNull(permit, "permit");
        }

        DispatchLease clearDispatchPermit() {
            DispatchLease current = dispatchPermit;
            dispatchPermit = null;
            return current;
        }

        long releasableKvTokens() { return kvTokens; }

        boolean hasProtocolOwner() { return preemptionClaim != null; }

        /**
         * Excludes this request from capacity accounting while preserving its ID and reservation token.
         * Callers update the aggregate capacity counters under admissionLock.
         */
        void stopCountingTowardCapacity() {
            phase = null;
        }
    }
}
