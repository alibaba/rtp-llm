package org.flexlb.balance.endpoint;

import org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.balance.endpoint.DecodeResources.CapacityRelease;
import org.flexlb.balance.endpoint.DecodeResources.CapacityUsage;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRequestStatus;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRequestView;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRoutingView;
import org.flexlb.balance.endpoint.DecodeResources.DispatchOutcome;
import org.flexlb.balance.endpoint.DecodeResources.EngineDispatchPermitAcquireStatus;
import org.flexlb.balance.endpoint.DecodeResources.EngineDispatchPermitTransferStatus;
import org.flexlb.balance.endpoint.DecodeResources.PreemptionBeginResult;
import org.flexlb.balance.endpoint.DecodeResources.PreemptionUpdate;
import org.flexlb.balance.endpoint.DecodeResources.ReleaseReason;
import org.flexlb.balance.endpoint.DecodeResources.ReservationHandle;
import org.flexlb.balance.endpoint.DecodeResources.ReservationReleaseResult;
import org.flexlb.balance.endpoint.DecodeResources.ResourceSnapshot;
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
import java.util.concurrent.locks.ReentrantLock;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static org.flexlb.balance.endpoint.DecodeResources.CapacityRelease.saturatedAddNonNegative;

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
                    .thenComparingLong(ReservationHandle::requestId)
                    .thenComparingLong(ReservationHandle::reservationToken);

    private final ConcurrentHashMap<Long, DecodeRequestState> decodeRequests = new ConcurrentHashMap<>();
    private final Map<Long, Long> settledHistory = new HashMap<>();
    /** Resource counters are mutated under admissionLock; advisory reads stay lock-free. */
    private final ResourceUsage reservedUsage = new ResourceUsage();
    private final ResourceUsage queuedUsage = new ResourceUsage();
    private final ResourceUsage dispatchUsage = new ResourceUsage();
    /** Includes synthetic Engine-owned slots retained by a preemption claim. */
    private volatile int confirmedEngineOwnedCount;
    private final Map<Long, PreemptionClaim> victimClaims = new HashMap<>();
    private final Map<Long, EndpointPreemptionAttempt> preemptionAttempts = new HashMap<>();
    /** Decode has freed this KV, but the Prefill CANCELED fence is still outstanding. */
    private volatile long priorityPreemptionHeldKv;
    private volatile long priorityPreemptionHeldExpectedKv;

    private static final class ResourceUsage {
        private volatile int requests;
        private volatile long hardKv;
        private volatile long expectedKv;

        void add(DecodeRequestState request) {
            requests++;
            hardKv += request.kvTokens;
            expectedKv += request.expectedKvTokens;
        }

        void remove(DecodeRequestState request) {
            hardKv -= request.kvTokens;
            expectedKv -= request.expectedKvTokens;
            requests--;
        }

        void clear() {
            requests = 0;
            hardKv = 0L;
            expectedKv = 0L;
        }
    }

    /** Guarded by {@link #admissionLock}; zero is never issued. */
    private long nextReservationToken = 1L;

    /**
     * Serializes reserve/release, dispatch-permit, calibration, and eviction
     * transactions. Reads stay lock-free.
     */
    private final ReentrantLock admissionLock = new ReentrantLock();

    /** Monotonic diagnostic generation for the layered admission projection. */
    private volatile long admissionVersion;

    /**
     * Generation-local, non-authoritative cache for bulk routing traversal.
     *
     * <p>Both key components are required: local admission mutations advance
     * {@link #admissionVersion}, while a committed Engine observation is
     * replaced by holder identity. The cache owns no endpoint or registry-map
     * reference and therefore disappears with this endpoint generation.</p>
     */
    private volatile DecodeRoutingView routingViewCache;
    private volatile DecodeResources.AdmissionSummary admissionSummaryCache;

    // Reservation: acquire and release exact ownership.

    ReservationHandle tryReserveQueuedRequest(long requestId, long hardKv, long expectedKv,
                                              int priority, AdmissionCapacity capacity) {
        admissionLock.lock();
        try {
            if (!requestIdAvailableForReservationLocked(requestId)) { return null; }
            if (capacity != null && queuedPlacementIsFullLocked(hardKv, expectedKv, capacity)) { return null; }
            return createReservationLocked(requestId, hardKv, expectedKv, priority,
                    DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED);
        } finally {
            admissionLock.unlock();
        }
    }

    /** Creates local or queued ownership after the caller checks admission under admissionLock. */
    private ReservationHandle createReservationLocked(long requestId,
                                                      long kvTokens,
                                                      long expectedKvTokens,
                                                      int priority,
                                                      DecodeTaskPhase initialPhase) {
        long reservationToken = nextReservationTokenLocked();
        DecodeRequestState newReservation =
                new DecodeRequestState(
                        kvTokens, expectedKvTokens, priority, reservationToken);
        ReservationHandle handle = new ReservationHandle(
                status.getGenerationId(), requestId, reservationToken);
        decodeRequests.put(requestId, newReservation);
        reservedUsage.add(newReservation);
        admissionVersion++;
        if (initialPhase == DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED) {
            setQueuedLocked(newReservation, true);
        }
        return handle;
    }

    private long nextReservationTokenLocked() {
        checkState(nextReservationToken > 0L && nextReservationToken != Long.MAX_VALUE,
                "Decode reservation token space exhausted");
        return nextReservationToken++;
    }

    private boolean requestIdAvailableForReservationLocked(long requestId) {
        if (decodeRequests.containsKey(requestId) || victimClaims.containsKey(requestId) || settledHistory.containsKey(requestId)) {
            return false;
        }
        for (EndpointPreemptionAttempt attempt : preemptionAttempts.values()) {
            if (attempt.incoming.requestId() == requestId) {
                return false;
            }
        }
        return true;
    }

    private boolean queuedPlacementIsFullLocked(
            long hardKvTokens, long expectedKvTokens, AdmissionCapacity capacity) {
        return !capacity.evaluate(routingViewLocked().placementUsage(), hardKvTokens, expectedKvTokens, CapacityRelease.NONE).fits();
    }

    boolean isAcceptedByEngine(ReservationHandle handle) {
        if (handle == null) {
            return false;
        }
        admissionLock.lock();
        try {
            DecodeRequestState state = confirmedRequest(handle.requestId());
            return handle.endpointGenerationId() == status.getGenerationId()
                    && state != null && state.reservationToken == handle.reservationToken();
        } finally {
            admissionLock.unlock();
        }
    }

    boolean hasOwnedResources(ReservationHandle reservation) {
        if (reservation == null || reservation.endpointGenerationId() != status.getGenerationId()) { return false; }
        admissionLock.lock();
        try {
            DecodeRequestState current = decodeRequests.get(reservation.requestId());
            return isExactReservation(current, reservation)
                    || exactPreemptionClaimLockedForReservation(reservation) != null
                    || hasExactIncomingAttemptLocked(reservation);
        } finally {
            admissionLock.unlock();
        }
    }

    ReservationReleaseResult release(ReservationHandle reservation, ReleaseReason reason) {
        java.util.Objects.requireNonNull(reason, "reason");
        if (reservation == null) {
            checkArgument(reason != ReleaseReason.LOCAL_ROLLBACK && reason != ReleaseReason.NOT_SENT,
                    "Decode reservation is required for release");
            return ReservationReleaseResult.STALE;
        }
        if (reservation.endpointGenerationId() != status.getGenerationId()) { return ReservationReleaseResult.STALE; }
        admissionLock.lock();
        try {
            if (reason == ReleaseReason.EXPIRED || reason == ReleaseReason.REMOTE_CLEANUP) {
                return expireRequestLocked(reservation)
                        ? ReservationReleaseResult.RELEASED : ReservationReleaseResult.STALE;
            }
            long requestId = reservation.requestId();
            DecodeRequestState current = decodeRequests.get(requestId);
            boolean exact = isExactReservation(current, reservation);
            if (reason == ReleaseReason.NOT_SENT) {
                if (exact && current.confirmed()) { return ReservationReleaseResult.ENGINE_ACCEPTED; }
                if (exact && victimClaims.containsKey(requestId)) {
                    return settlePriorityClaimTerminalLocked(reservation, victimClaims.get(requestId))
                            ? ReservationReleaseResult.RELEASED : ReservationReleaseResult.CONFLICT;
                }
                if (hasExactIncomingAttemptLocked(reservation)) { return ReservationReleaseResult.CONFLICT; }
            } else if (hasExactIncomingAttemptLocked(reservation)
                    || exact && (current.confirmed() || current.phase == DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN || victimClaims.containsKey(reservation.requestId()))) {
                if (reason == ReleaseReason.LOCAL_ROLLBACK) {
                    throw localReleaseInvariant(reservation, "exact ownership is held by Engine/protocol lifecycle");
                }
                return ReservationReleaseResult.STILL_OWNED;
            }
            if (!exact) { return ReservationReleaseResult.STALE; }
            // Every surviving path owns an unconfirmed request without a protocol claim.
            // NOT_SENT proves rollback is safe even after local dispatch handoff.
            if (!removeRequestOwnershipLocked(requestId, current)) {
                throw localReleaseInvariant(reservation, "validated shadow changed while admissionLock was held");
            }
            if (reason != ReleaseReason.LOCAL_ROLLBACK) {
                rememberSettledLocked(requestId, System.currentTimeMillis());
            }
            admissionVersion++;
            return ReservationReleaseResult.RELEASED;
        } finally {
            admissionLock.unlock();
        }
    }

    private boolean expireRequestLocked(ReservationHandle reservation) {
        long requestId = reservation.requestId();
        DecodeRequestState state = decodeRequests.get(requestId);
        if (!isExactReservation(state, reservation)
                && exactPreemptionClaimLockedForReservation(reservation) == null) {
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
            if (!attempt.incoming.equals(reservation)) {
                continue;
            }
            attempts.remove();
            releaseLocalVictimClaimsLocked(entry.getKey(), attempt);
        }

        PreemptionClaim claim = victimClaims.get(requestId);
        if (claim != null) {
            EndpointPreemptionAttempt attempt = preemptionAttempts.get(claim.attemptToken);
            if (attempt != null) {
                attempt.remainingVictims.remove(reservation);
            }
            releasePreemptionClaimLocked(requestId, claim);
        }
        removeRequestOwnershipLocked(requestId, state);
        // Use a history-only entry, so no old exact token remains live.
        rememberSettledLocked(requestId, System.currentTimeMillis());
        admissionVersion++;
        return true;
    }

    private void settleAuthoritativeTerminalLocked(ReservationHandle reservation) {
        long requestId = reservation.requestId();
        DecodeRequestState state = decodeRequests.get(requestId);
        if (isExactReservation(state, reservation) && victimClaims.containsKey(requestId)) {
            throw terminalInvariant(reservation,
                    "priority claim must settle before generic terminal ownership");
        }
        if (hasExactIncomingAttemptLocked(reservation)) {
            throw terminalInvariant(reservation,
                    "priority attempt still owns the exact incoming reservation");
        }
        if (isExactReservation(state, reservation)) {
            removeRequestOwnershipLocked(requestId, state);
        }
    }

    private void settleUntrackedWorkerTerminalLocked(long requestId) {
        DecodeRequestState confirmed = confirmedRequest(requestId);
        if (confirmed != null && confirmed.reservationToken == 0L) {
            removeRequestOwnershipLocked(requestId, confirmed);
        }
    }

    /** Settle one exact owner; protocol claims and terminal history have separate lifetimes. */
    private boolean removeRequestOwnershipLocked(long requestId, DecodeRequestState expected) {
        if (expected == null || decodeRequests.get(requestId) != expected) {
            return false;
        }
        if (expected.confirmed()) {
            confirmedEngineOwnedCount = Math.max(0, confirmedEngineOwnedCount - 1);
        } else {
            clearShadowAccountingLocked(expected);
        }
        decodeRequests.remove(requestId, expected);
        return true;
    }

    private void clearShadowAccountingLocked(DecodeRequestState reservation) {
        removeEngineDispatchPermitLocked(reservation);
        setQueuedLocked(reservation, false);
        reservedUsage.remove(reservation);
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
        checkArgument(reservation != null, "Decode reservation is required for queued transition");
        admissionLock.lock();
        try {
            if (reservation.endpointGenerationId()
                    != status.getGenerationId()) {
                return false;
            }
            long requestId = reservation.requestId();
            DecodeRequestState current = shadowReservation(requestId);
            if (!isExactReservation(current, reservation)) {
                return false;
            }
            if (current.queued()) { return true; }
            if (current.phase == DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN) { return false; }
            setQueuedLocked(current, true);
            // Re-queueing begins a new dispatch round. Invalidate any
            // pre-delivery lease before publishing that transition.
            removeEngineDispatchPermitLocked(current);
            admissionVersion++;
        } finally {
            admissionLock.unlock();
        }
        return true;
    }

    private void setQueuedLocked(DecodeRequestState reservation, boolean queued) {
        if (reservation.queued() == queued) {
            return;
        }
        reservation.phase = queued ? DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED
                : DecodeTaskPhase.LOCAL_RESERVED;
        if (queued) {
            queuedUsage.add(reservation);
        } else {
            queuedUsage.remove(reservation);
        }
    }

    DispatchAcquisition acquireDispatchPermit(
            ReservationHandle handle,
            AdmissionCapacity capacity) {
        admissionLock.lock();
        try {
            DecodeRequestState reservation = decodeRequests.get(handle.requestId());
            if (handle.endpointGenerationId() != status.getGenerationId()
                    || !isExactReservation(reservation, handle)
                    || victimClaims.containsKey(handle.requestId())) {
                return new DispatchAcquisition(EngineDispatchPermitAcquireStatus.NOT_OWNED, null);
            }
            if (reservation.confirmed() || reservation.phase == DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN) {
                return new DispatchAcquisition(
                        EngineDispatchPermitAcquireStatus.ALREADY_ACCEPTED,
                        new DispatchLease(handle.requestId(), reservation));
            }
            if (!reservation.queued()) {
                return new DispatchAcquisition(EngineDispatchPermitAcquireStatus.NOT_QUEUED, null);
            }
            if (reservation.dispatchPermit != null) {
                return new DispatchAcquisition(EngineDispatchPermitAcquireStatus.ALREADY_ACQUIRED, null);
            }
            if (isEngineDispatchCapacityFullSnapshot(
                    reservation, capacity, status.committedWorkerStatus().fields())) {
                return new DispatchAcquisition(EngineDispatchPermitAcquireStatus.CAPACITY_FULL, null);
            }

            DispatchLease permit = new DispatchLease(handle.requestId(), reservation);
            reservation.dispatchPermit = permit;
            dispatchUsage.add(reservation);
            admissionVersion++;
            return new DispatchAcquisition(
                    EngineDispatchPermitAcquireStatus.ACQUIRED, permit);
        } finally {
            admissionLock.unlock();
        }
    }

    DispatchResult dispatch(DispatchLease permit, DispatchOutcome outcome) {
        java.util.Objects.requireNonNull(permit, "permit");
        java.util.Objects.requireNonNull(outcome, "outcome");
        admissionLock.lock();
        try {
            boolean engineOwned = outcome == DispatchOutcome.ENGINE_OWNED;
            DecodeRequestState current = decodeRequests.get(permit.requestId);
            // Engine status can consume the permit before the sender hands it off.
            if (engineOwned ? current == permit.reservation && (current.confirmed()
                    || current.phase == DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN)
                    && !victimClaims.containsKey(permit.requestId) : permit.retiredByEndpoint) {
                return new DispatchResult(EngineDispatchPermitTransferStatus.TRANSFERRED, false);
            }
            if (current != permit.reservation || current.dispatchPermit != permit) {
                return new DispatchResult(EngineDispatchPermitTransferStatus.OWNERSHIP_LOST, false);
            }
            int usageBefore = engineDispatchHardGateUsageLocked();
            boolean transferable = !engineOwned
                    || permit.reservation.queued() && !victimClaims.containsKey(permit.requestId);
            removeEngineDispatchPermitLocked(permit.reservation);
            if (engineOwned && transferable) {
                setQueuedLocked(permit.reservation, false);
                permit.reservation.phase = DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN;
            }
            admissionVersion++;
            return new DispatchResult(transferable ? EngineDispatchPermitTransferStatus.TRANSFERRED
                    : EngineDispatchPermitTransferStatus.OWNERSHIP_LOST,
                    !engineOwned || engineDispatchHardGateUsageLocked() < usageBefore);
        } finally {
            admissionLock.unlock();
        }
    }

    private void removeEngineDispatchPermitLocked(
            DecodeRequestState reservation) {
        DispatchLease removed = reservation.clearDispatchPermit();
        if (removed == null) {
            return;
        }
        dispatchUsage.remove(reservation);
        checkState(!(dispatchUsage.requests < 0), "negative active Decode dispatch permit count");
    }

    boolean shouldRetryDispatch(
            long requestId,
            AdmissionCapacity capacity) {
        DecodeRequestState candidate = shadowReservation(requestId);
        if (candidate == null
                || !candidate.queued()
                || candidate.dispatchPermit != null) {
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
        return !capacity.evaluate(dispatchCapacityUsage(fields), candidate.kvTokens, candidate.expectedKvTokens, CapacityRelease.NONE).fits();
    }

    private int engineDispatchHardGateUsageLocked() {
        return getEngineLoad() + dispatchUsage.requests;
    }

    private CapacityUsage dispatchCapacityUsage(WorkerStatus.EngineObservation fields) {
        long heldHard = priorityPreemptionHeldKv;
        long dispatchHard = saturatedAddNonNegative(
                saturatedAddNonNegative(Math.max(0L, reservedUsage.hardKv - queuedUsage.hardKv),
                        dispatchUsage.hardKv), heldHard);
        return new CapacityUsage(
                getEngineLoad() + Math.max(0, dispatchUsage.requests),
                Math.max(0L, fields.totalKvCacheTokens()), Math.max(0L, fields.availableKvCacheTokens()),
                dispatchHard, engineFacingKvUsed(fields));
    }

    private long engineFacingKvUsed(
            WorkerStatus.EngineObservation fields) {
        long reportedUsed = reportedKvUsed(fields);
        long localEngineFacing = Math.max(0L,
                reservedUsage.expectedKv - queuedUsage.expectedKv)
                + dispatchUsage.expectedKv;
        return saturatedAddNonNegative(
                saturatedAddNonNegative(reportedUsed, localEngineFacing),
                priorityPreemptionHeldExpectedKv);
    }

    private static long reportedKvUsed(WorkerStatus.EngineObservation fields) {
        return fields.totalKvCacheTokens() > 0
                ? Math.max(0L, fields.totalKvCacheTokens() - fields.availableKvCacheTokens()) : 0L;
    }

    private int getEngineLoad() {
        int inflight = reservedUsage.requests;
        int queued = queuedUsage.requests;
        if (queued < 0 || queued > inflight) {
            queued = Math.clamp(queued, 0, Math.max(0, inflight));
        }
        return confirmedEngineOwnedCount + Math.max(0, inflight - queued);
    }

    record DispatchAcquisition(EngineDispatchPermitAcquireStatus status, DispatchLease permit) { }

    record DispatchResult(EngineDispatchPermitTransferStatus status, boolean capacityReleased) { }

    static final class DispatchLease {
        private final long requestId;
        private final DecodeRequestState reservation;
        private boolean retiredByEndpoint;

        boolean matches(ReservationHandle exact) {
            return exact != null && requestId == exact.requestId()
                    && reservation.reservationToken == exact.reservationToken();
        }

        private DispatchLease(long requestId, DecodeRequestState reservation) {
            this.requestId = requestId;
            this.reservation = reservation;
        }
    }

    // Preemption: local replacement and the complete remote-cancel resource transaction.

    ReservationHandle replaceQueuedRequests(
            List<ReservationHandle> victims,
            long incomingRequestId, long kvTokens, long expectedKvTokens,
            int priority,
            AdmissionCapacity capacity) {
        admissionLock.lock();
        try {
            if (victims == null || victims.isEmpty()
                    || !requestIdAvailableForReservationLocked(
                            incomingRequestId)) {
                return null;
            }
            Set<Long> uniqueVictims = new HashSet<>(victims.size());
            CapacityRelease released = CapacityRelease.NONE;
            for (ReservationHandle victim : victims) {
                if (victim == null
                        || victim.endpointGenerationId()
                                != status.getGenerationId()
                        || victim.requestId() == incomingRequestId
                        || !uniqueVictims.add(victim.requestId())) {
                    return null;
                }
                DecodeRequestState held = shadowReservation(victim.requestId());
                DispatchLease permit = held == null
                        ? null : held.dispatchPermit;
                if (!isExactReservation(held, victim)
                        || !held.queued()
                        || victimClaims.containsKey(victim.requestId())
                        || hasExactIncomingAttemptLocked(victim)
                        || permit != null) {
                    return null;
                }
                released = released.plus(held.capacityRelease());
            }
            CapacityUsage usage = routingViewLocked().placementUsage();
            if (capacity.evaluate(usage, kvTokens, expectedKvTokens, CapacityRelease.NONE).fits()
                    || !capacity.evaluate(usage, kvTokens, expectedKvTokens, released).fits()) {
                return null;
            }

            for (ReservationHandle victim : victims) {
                DecodeRequestState exact = shadowReservation(victim.requestId());
                if (!removeRequestOwnershipLocked(victim.requestId(), exact)) {
                    throw localReleaseInvariant(
                            victim,
                            "validated victim changed while admissionLock was held");
                }
            }
            return createReservationLocked(incomingRequestId, kvTokens, expectedKvTokens, priority,
                    DecodeTaskPhase.LOCAL_RESERVED);
        } finally {
            admissionLock.unlock();
        }
    }

    PreemptionBeginResult beginPreemption(
            long attemptToken,
            List<ReservationHandle> victims,
            long incomingRequestId,
            long incomingKvTokens,
            long incomingExpectedKvTokens,
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

            Set<ReservationHandle> exactVictims = new HashSet<>();
            CapacityRelease released = CapacityRelease.NONE;
            for (ReservationHandle victim : victims) {
                if (victim == null
                        || victim.endpointGenerationId()
                                != status.getGenerationId()
                        || victim.requestId() == incomingRequestId
                        || !exactVictims.add(victim)) {
                    return PreemptionBeginResult.VICTIM_GONE;
                }
                long victimId = victim.requestId();
                DecodeRequestState request = decodeRequests.get(victimId);
                if (victimClaims.containsKey(victimId)
                        || hasExactIncomingAttemptLocked(victim)) {
                    return PreemptionBeginResult.VICTIM_ALREADY_CLAIMED;
                }
                if (!isExactReservation(request, victim)) {
                    return PreemptionBeginResult.VICTIM_GONE;
                }
                if (request.dispatchPermit != null || request.queued()) {
                    return PreemptionBeginResult.VICTIM_GONE;
                }
                if (request.priority <= 0 || request.priority >= incomingPriority) {
                    return PreemptionBeginResult.INVALID_PRIORITY;
                }
                released = released.plus(request.capacityRelease());
            }
            CapacityUsage usage = routingViewLocked().placementUsage();
            if (capacity.evaluate(usage, incomingKvTokens, incomingExpectedKvTokens, CapacityRelease.NONE).fits()
                    || !capacity.evaluate(usage, incomingKvTokens, incomingExpectedKvTokens, released).fits()) {
                return PreemptionBeginResult.INFEASIBLE;
            }

            // Allocate every victim claim before installing any incoming or
            // protocol ownership. The endpoint lock keeps these exact states
            // stable through the subsequent allocation-free installation.
            Map<Long, PreemptionClaim> preparedClaims = new HashMap<>();
            for (ReservationHandle victim : victims) {
                DecodeRequestState request = decodeRequests.get(victim.requestId());
                preparedClaims.put(
                        victim.requestId(),
                        new PreemptionClaim(
                                attemptToken, victim,
                                request.kvTokens,
                                request.expectedKvTokens));
            }

            // Provisional incoming ownership closes the free-pool race while
            // Cancel runs.  It is not visible to the prefill queue yet.
            ReservationHandle incomingReservation = createReservationLocked(
                    incomingRequestId, incomingKvTokens, incomingExpectedKvTokens, incomingPriority,
                    DecodeTaskPhase.LOCAL_RESERVED);
            EndpointPreemptionAttempt preparedAttempt = null;
            try {
                preparedAttempt = new EndpointPreemptionAttempt(incomingReservation, exactVictims);
                preemptionAttempts.put(attemptToken, preparedAttempt);
                for (Map.Entry<Long, PreemptionClaim> claim
                        : preparedClaims.entrySet()) {
                    victimClaims.put(claim.getKey(), claim.getValue());
                }
            } catch (RuntimeException | Error installationFailure) {
                if (preparedAttempt != null) {
                    preemptionAttempts.remove(attemptToken, preparedAttempt);
                }
                for (Map.Entry<Long, PreemptionClaim> claim
                        : preparedClaims.entrySet()) {
                    releasePreemptionClaimLocked(claim.getKey(), claim.getValue());
                }
                DecodeRequestState incoming =
                        shadowReservation(incomingRequestId);
                if (isExactReservation(incoming, incomingReservation)) {
                    removeRequestOwnershipLocked(incomingRequestId, incoming);
                }
                throw installationFailure;
            }
            admissionVersion++;
            return PreemptionBeginResult.SUCCESS;
        } finally {
            admissionLock.unlock();
        }
    }

    boolean updatePreemption(long attemptToken, PreemptionUpdate update) {
        java.util.Objects.requireNonNull(update, "update");
        admissionLock.lock();
        try {
            ReservationHandle victim = update.reservation();
            if (victim.endpointGenerationId() != status.getGenerationId()) { return false; }
            PreemptionClaim claim = exactPreemptionClaimLocked(attemptToken, victim);
            if (claim == null) { return false; }
            return switch (update.kind()) {
                case CANCELED, REQUEST_FENCED -> settlePriorityClaimTerminalLocked(victim, claim);
                case CANCEL_HANDED_OFF -> {
                    claim.cancellationHandedOff = true;
                    yield true;
                }
                case ACTIVE, FINISHED -> {
                    releasePreemptionClaimLocked(victim.requestId(), claim);
                    if (update.kind() == PreemptionUpdate.Kind.FINISHED) {
                        settleAuthoritativeTerminalLocked(victim);
                    }
                    admissionVersion++;
                    yield true;
                }
                default -> throw new IllegalStateException("Unhandled preemption update: " + update.kind());
            };
        } finally {
            admissionLock.unlock();
        }
    }

    /** Callers supply the exact claim verified under the same admission lock. */
    private boolean settlePriorityClaimTerminalLocked(
            ReservationHandle reservation,
            PreemptionClaim claim) {
        long requestId = reservation.requestId();
        DecodeRequestState state = decodeRequests.get(requestId);
        EndpointPreemptionAttempt attempt = preemptionAttempts.get(claim.attemptToken);
        if (attempt != null
                && !attempt.remainingVictims.contains(reservation)) {
            return false;
        }

        removeRequestOwnershipLocked(requestId, state);
        releasePreemptionClaimLocked(requestId, claim);
        if (attempt != null) {
            attempt.remainingVictims.remove(reservation);
        }
        rememberSettledLocked(requestId, System.currentTimeMillis());
        admissionVersion++;
        return true;
    }

    ReservationHandle finishPreemption(long attemptToken, boolean commit) {
        admissionLock.lock();
        try {
            EndpointPreemptionAttempt attempt = preemptionAttempts.get(attemptToken);
            if (attempt == null) { return null; }
            if (commit) {
                DecodeRequestState incoming = shadowReservation(attempt.incoming.requestId());
                if (!isExactReservation(incoming, attempt.incoming)
                        || !attempt.remainingVictims.isEmpty()) { return null; }
            } else {
                long incomingId = attempt.incoming.requestId();
                removeRequestOwnershipLocked(incomingId, shadowReservation(incomingId));
                releaseLocalVictimClaimsLocked(attemptToken, attempt);
            }
            preemptionAttempts.remove(attemptToken);
            admissionVersion++;
            return attempt.incoming;
        } finally {
            admissionLock.unlock();
        }
    }

    private void releaseLocalVictimClaimsLocked(long attemptToken, EndpointPreemptionAttempt attempt) {
        for (ReservationHandle victim : attempt.remainingVictims) {
            PreemptionClaim claim = exactPreemptionClaimLocked(attemptToken, victim);
            if (claim != null && !claim.cancellationHandedOff) {
                releasePreemptionClaimLocked(victim.requestId(), claim);
            }
        }
    }

    private PreemptionClaim exactPreemptionClaimLockedForReservation(ReservationHandle reservation) {
        PreemptionClaim claim = victimClaims.get(reservation.requestId());
        return claim != null && claim.reservation.equals(reservation) ? claim : null;
    }

    private PreemptionClaim exactPreemptionClaimLocked(long attemptToken, ReservationHandle reservation) {
        PreemptionClaim claim = victimClaims.get(reservation.requestId());
        return claim != null && claim.attemptToken == attemptToken
                && claim.reservation.equals(reservation) ? claim : null;
    }

    private void releasePreemptionClaimLocked(long requestId, PreemptionClaim expected) {
        if (victimClaims.get(requestId) != expected) { return; }
        setKvHeld(expected, false);
        victimClaims.remove(requestId, expected);
    }

    private boolean hasExactIncomingAttemptLocked(
            ReservationHandle reservation) {
        for (EndpointPreemptionAttempt attempt : preemptionAttempts.values()) {
            if (attempt.incoming.equals(reservation)) {
                return true;
            }
        }
        return false;
    }

    /** A claim changes its KV hold once; writes preserve expected >= hard for advisory readers. */
    private void setKvHeld(PreemptionClaim claim, boolean held) {
        checkState(admissionLock.isHeldByCurrentThread(),
                "Priority preemption KV hold mutation requires admissionLock");
        if (claim.kvHeldAfterWorkerRelease == held) { return; }
        long hard = priorityPreemptionHeldKv;
        long expected = priorityPreemptionHeldExpectedKv;
        requirePriorityPreemptionHoldInvariant(hard, expected);
        if (!held && (hard < claim.hardKvTokens || expected < claim.expectedKvTokens)) {
            throw new IllegalStateException("Priority preemption KV hold counter underflow: hard="
                    + hard + "-" + claim.hardKvTokens + ", expected=" + expected + "-" + claim.expectedKvTokens);
        }
        try {
            hard = Math.addExact(hard, held ? claim.hardKvTokens : -claim.hardKvTokens);
            expected = Math.addExact(expected, held ? claim.expectedKvTokens : -claim.expectedKvTokens);
        } catch (ArithmeticException overflow) {
            throw new IllegalStateException("Priority preemption KV hold counter overflow", overflow);
        }
        requirePriorityPreemptionHoldInvariant(hard, expected);
        if (held) {
            priorityPreemptionHeldExpectedKv = expected;
            priorityPreemptionHeldKv = hard;
        } else {
            priorityPreemptionHeldKv = hard;
            priorityPreemptionHeldExpectedKv = expected;
        }
        claim.kvHeldAfterWorkerRelease = held;
    }

    private static void requirePriorityPreemptionHoldInvariant(long hardKvTokens, long expectedKvTokens) {
        checkState(hardKvTokens >= 0L && expectedKvTokens >= hardKvTokens,
                "Invalid priority preemption KV hold counters: hard=%s, expected=%s", hardKvTokens, expectedKvTokens);
    }

    private static final class PreemptionClaim {
        private final long attemptToken;
        private final ReservationHandle reservation;
        private final long hardKvTokens;
        private final long expectedKvTokens;
        /** Outbound Cancel may have reached Worker; local rollback cannot discard this hold. */
        private boolean cancellationHandedOff;
        private boolean kvHeldAfterWorkerRelease;

        private PreemptionClaim(long attemptToken, ReservationHandle reservation,
                                long hardKvTokens, long expectedKvTokens) {
            checkArgument(hardKvTokens >= 0L && expectedKvTokens >= hardKvTokens,
                    "Priority preemption claim requires expected KV >= hard KV >= 0");
            this.attemptToken = attemptToken;
            this.reservation = reservation;
            this.hardKvTokens = hardKvTokens;
            this.expectedKvTokens = expectedKvTokens;
        }
    }

    /** Owns the exact incoming reservation and the locally prepared victim set. */
    private static final class EndpointPreemptionAttempt {
        private final ReservationHandle incoming;
        private final Set<ReservationHandle> remainingVictims;

        private EndpointPreemptionAttempt(ReservationHandle incoming, Set<ReservationHandle> remainingVictims) {
            this.incoming = incoming;
            this.remainingVictims = remainingVictims;
        }
    }

    // Calibration: apply one observation and produce request statuses.

    ReentrantLock ownershipLock() { return admissionLock; }

    List<DecodeRequestStatus> calibrateLocked(WorkerStatus.StatusObservation observation) {
        checkState(admissionLock.isHeldByCurrentThread(), "Decode calibration requires ownershipLock");
        checkArgument(observation.owner() == status, "Status belongs to another Decode generation");
        List<DecodeRequestStatus> requestStatuses = doCalibrate(observation.engine(), observation.finishedTasks());
        return List.copyOf(requestStatuses);
    }

    void initialize(WorkerStatus.StatusObservation observation) {
        checkArgument(observation.owner() == status, "Status belongs to another Decode generation");
        admissionLock.lock();
        try {
            checkState(doCalibrate(observation.engine(), observation.finishedTasks()).isEmpty(),
                    "Private Decode candidate produced locally-owned request statuses");
        } finally {
            admissionLock.unlock();
        }
    }

    /** Collect active statuses matched to exact local reservations for Scheduler notification. */
    List<DecodeRequestStatus> collectHeartbeatRequestStatuses(WorkerStatus.StatusObservation observation) {
        checkArgument(observation.owner() == status, "Status observation belongs to another Decode generation");
        List<DecodeRequestStatus> requestStatuses = new ArrayList<>(
                observation.runningTasks().size());
        admissionLock.lock();
        try {
            for (WorkerStatus.TaskObservation task
                    : observation.runningTasks().values()) {
                ReservationHandle active = workerStatusHandleLocked(
                        task.requestId(), decodeRequests.get(task.requestId()));
                if (active != null) {
                    requestStatuses.add(task.phase() == TaskPhase.KV_ALLOCATED || task.phase() == TaskPhase.RUNNING
                            ? DecodeRequestStatus.allocated(active) : DecodeRequestStatus.active(active));
                }
            }
        } finally {
            admissionLock.unlock();
        }
        return List.copyOf(requestStatuses);
    }

    private List<DecodeRequestStatus> doCalibrate(
            WorkerStatus.EngineObservation engine,
            Map<String, WorkerStatus.TaskObservation> finishedTasks) {
        admissionVersion++;
        List<DecodeRequestStatus> requestStatuses = new ArrayList<>();

        // Build one authoritative Decode view. Claimed victims that merely
        // disappear are held synthetically. An explicit Decode finished task
        // is a separate authoritative terminal outcome: it settles the exact
        // claim without reclassifying that outcome as priority CANCELED.
        Set<Long> presentNow = new HashSet<>();
        Set<Long> allocatedVictims = new HashSet<>();
        Set<Long> terminalNow = new HashSet<>();
        for (WorkerStatus.TaskObservation task : finishedTasks.values()) {
            terminalNow.add(task.requestId());
        }
        long now = System.currentTimeMillis();
        for (WorkerStatus.TaskObservation task
                : engine.runningTaskList().values()) {
            TaskPhase phase = task.phase();
            long requestId = task.requestId();
            DecodeRequestState current = decodeRequests.get(requestId);
            if (terminalNow.contains(requestId)
                    || settledHistory.containsKey(requestId)) {
                continue;
            }
            // Membership, allocation/running evidence and terminal evidence are distinct.
            // Engine may report RECEIVED again after freeing blocks, before publishing finished.
            presentNow.add(requestId);
            if (phase == TaskPhase.KV_ALLOCATED || phase == TaskPhase.RUNNING) {
                if (current != null && !current.confirmed()) {
                    clearShadowAccountingLocked(current);
                }
                PreemptionClaim claim = victimClaims.get(requestId);
                if (claim != null) {
                    setKvHeld(claim, false);
                    allocatedVictims.add(requestId);
                }
                DecodeTaskPhase allocatedPhase = phase == TaskPhase.KV_ALLOCATED
                        ? DecodeTaskPhase.ACCEPTED_NOT_RUNNING : DecodeTaskPhase.RUNNING;
                boolean discoveredByEngine = current == null;
                if (discoveredByEngine) {
                    current = new DecodeRequestState(0L, 0L, DecodeRequestState.DEFAULT_PRIORITY, 0L);
                }
                current.recordEngineAllocation(task.inputLength(), allocatedPhase, now);
                if (discoveredByEngine) {
                    decodeRequests.put(requestId, current);
                }
            } else if (current != null && !current.confirmed()) {
                removeEngineDispatchPermitLocked(current);
                setQueuedLocked(current, false);
                current.phase = DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN;
                current.observedAtMs = now;
            } else if (current != null) {
                // Preserve the exact token and last confirmed resource phase. This is a
                // conservative ownership slot, not a claim about current GPU execution.
                current.refresh(current.phase, now);
                // Claims are counted/held below when allocation evidence is absent.
            }
            ReservationHandle active = workerStatusHandleLocked(requestId, current);
            if (active != null) {
                requestStatuses.add(task.phase() == TaskPhase.KV_ALLOCATED || task.phase() == TaskPhase.RUNNING
                            ? DecodeRequestStatus.allocated(active) : DecodeRequestStatus.active(active));
            }
        }

        // Terminal proof must be captured before absent-task pruning removes
        // the exact DecodeRequestState identity. Endpoint settlement happens here;
        // the downstream scheduler receives only the immutable result.
        for (WorkerStatus.TaskObservation task
                : finishedTasks.values()) {
            long requestId = task.requestId();
            DecodeRequestState current = decodeRequests.get(requestId);
            if (settledHistory.containsKey(requestId)) {
                continue;
            }
            ReservationHandle terminal = workerStatusHandleLocked(requestId, current);
            PreemptionClaim claim = victimClaims.get(requestId);
            if (claim != null) {
                if (terminal != null
                        && settlePriorityClaimTerminalLocked(terminal, claim)) {
                    requestStatuses.add(DecodeRequestStatus.terminal(
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
                requestStatuses.add(DecodeRequestStatus.terminal(
                        terminal, task.errorCode()));
                settleAuthoritativeTerminalLocked(terminal);
            } else {
                settleUntrackedWorkerTerminalLocked(requestId);
            }
        }

        int confirmedCount = 0;
        // Confirmed phase survives missing/regressed Engine evidence while a claim owns it.
        // Ordinary owners are pruned only when absent from the FULL active snapshot.
        var confirmedIt = decodeRequests.entrySet().iterator();
        while (confirmedIt.hasNext()) {
            Map.Entry<Long, DecodeRequestState> entry = confirmedIt.next();
            DecodeRequestState request = entry.getValue();
            if (!request.confirmed()) { continue; }
            long requestId = entry.getKey();
            PreemptionClaim claim = victimClaims.get(requestId);
            if (claim != null) {
                if (!allocatedVictims.contains(requestId)) {
                    setKvHeld(claim, true);
                }
            } else if (!presentNow.contains(requestId)) {
                confirmedIt.remove();
                rememberSettledLocked(requestId, now);
                continue;
            }
            confirmedCount++;
        }
        this.confirmedEngineOwnedCount = confirmedCount;

        return requestStatuses;
    }

    private ReservationHandle workerStatusHandleLocked(long requestId, DecodeRequestState state) {
        if (state == null) {
            PreemptionClaim claim = victimClaims.get(requestId);
            return claim == null ? null : claim.reservation;
        }
        long reservationToken = state.reservationToken;
        if (reservationToken <= 0L) {
            return null;
        }
        return new ReservationHandle(
                status.getGenerationId(),
                requestId,
                reservationToken);
    }

    static boolean placementCapacityImproved(
            DecodeRoutingView before,
            DecodeRoutingView after) {
        return after.engineLoad() < before.engineLoad()
                || after.realKvAvailable() > before.realKvAvailable()
                || after.realKvUsed() < before.realKvUsed();
    }

    // Generation cleanup: drain resources and expire orphan/history records.

    List<ReservationHandle> retire() {
        admissionLock.lock();
        try {
            Set<ReservationHandle> owners = new HashSet<>();
            long generationId = status.getGenerationId();
            decodeRequests.forEach((requestId, reservation) -> {
                if (reservation.reservationToken > 0L) {
                    owners.add(new ReservationHandle(generationId, requestId, reservation.reservationToken));
                }
            });
            for (EndpointPreemptionAttempt attempt : preemptionAttempts.values()) {
                owners.add(attempt.incoming);
                for (ReservationHandle victim : attempt.remainingVictims) {
                    if (victim.endpointGenerationId() != generationId) {
                        logger.error(
                                "Decode retirement ignored a priority victim from another generation: "
                                        + "request_id={} expected_generation={} actual_generation={}",
                                victim.requestId(), generationId, victim.endpointGenerationId());
                        continue;
                    }
                    owners.add(victim);
                }
            }
            victimClaims.values().forEach(claim -> owners.add(claim.reservation));
            List<ReservationHandle> ordered = new ArrayList<>(owners);
            ordered.sort(RETIREMENT_ORDER);
            List<ReservationHandle> retired = List.copyOf(ordered);

            // Build the immutable owner snapshot before changing any permit or accounting.
            for (DecodeRequestState reservation : decodeRequests.values()) {
                DispatchLease permit = reservation.clearDispatchPermit();
                if (permit != null) { permit.retiredByEndpoint = true; }
            }
            decodeRequests.clear();
            settledHistory.clear();
            victimClaims.clear();
            dispatchUsage.clear();
            reservedUsage.clear();
            queuedUsage.clear();
            confirmedEngineOwnedCount = 0;
            preemptionAttempts.clear();
            priorityPreemptionHeldKv = 0L;
            priorityPreemptionHeldExpectedKv = 0L;
            admissionVersion++;
            return retired;
        } finally {
            admissionLock.unlock();
        }
    }

    record CleanupCandidate(long reservationToken, DecodeTaskPhase phase, long observedAtMs) { }

    Map<Long, CleanupCandidate> cleanupCandidates() {
        admissionLock.lock();
        try {
            Map<Long, CleanupCandidate> candidates = new HashMap<>();
            decodeRequests.forEach((id, request) -> candidates.put(id, new CleanupCandidate(request.reservationToken, request.phase, request.observedAtMs)));
            return candidates;
        } finally { admissionLock.unlock(); }
    }

    CleanupResult evictExpiredRequests(long ttlMs, Map<Long, CleanupCandidate> orphanCandidates) {
        admissionLock.lock();
        try {
            long nowMs = System.currentTimeMillis();
            long cutoff = nowMs - ttlMs;
            int expiredReservations = 0;
            boolean changed = settledHistory.values().removeIf(settledAt -> settledAt < cutoff);
            for (Map.Entry<Long, DecodeRequestState> entry : decodeRequests.entrySet()) {
                long requestId = entry.getKey();
                DecodeRequestState request = entry.getValue();
                CleanupCandidate candidate = orphanCandidates.get(requestId);
                if (victimClaims.containsKey(requestId) || candidate == null
                        || candidate.reservationToken != request.reservationToken
                        || candidate.phase != request.phase
                        || candidate.observedAtMs != request.observedAtMs) { continue; }
                boolean confirmed = request.confirmed();
                boolean expired = confirmed ? request.observedAtMs < cutoff
                        : nowMs - request.observedAtMs > ttlMs;
                if (!expired || !removeRequestOwnershipLocked(requestId, request)) { continue; }
                if (!confirmed) { expiredReservations++; }
                changed = true;
            }
            if (changed) { admissionVersion++; }
            return new CleanupResult(expiredReservations, changed);
        } finally {
            admissionLock.unlock();
        }
    }

    /** Called after the last resource/protocol owner has been removed. */
    private void rememberSettledLocked(long requestId, long settledAtMs) {
        settledHistory.put(requestId, settledAtMs);
    }

    record CleanupResult(int expiredReservations, boolean capacityReleased) { }

    // Read-only views: capture, cache and metrics.

    DecodeResources.AdmissionSummary admissionSummary() {
        DecodeResources.AdmissionSummary cached = admissionSummaryCache;
        if (isCurrentAdmissionSummary(cached)) { return cached; }
        admissionLock.lock();
        try {
            cached = admissionSummaryCache;
            if (isCurrentAdmissionSummary(cached)) { return cached; }
            DecodeRoutingView routing = routingViewLocked();
            long[] requests = new long[PriorityNormalizer.MAX_PRIORITY + 1];
            long[] hardKv = new long[PriorityNormalizer.MAX_PRIORITY + 1];
            long[] expectedKv = new long[PriorityNormalizer.MAX_PRIORITY + 1];
            long[] engineRequests = new long[PriorityNormalizer.MAX_PRIORITY + 1];
            long[] engineHardKv = new long[PriorityNormalizer.MAX_PRIORITY + 1];
            long[] engineExpectedKv = new long[PriorityNormalizer.MAX_PRIORITY + 1];
            for (DecodeRequestState task : decodeRequests.values()) {
                int priority = task.priorityKnown() && PriorityNormalizer.isValid(task.priority)
                        ? task.priority : 0;
                requests[priority]++;
                long hard = task.kvTokens;
                long expected = task.confirmed() ? task.kvTokens : task.expectedKvTokens;
                hardKv[priority] = saturatedAddNonNegative(hardKv[priority], hard);
                expectedKv[priority] = saturatedAddNonNegative(expectedKv[priority], expected);
                if (task.confirmed() || !task.queued() || task.dispatchPermit != null) {
                    engineRequests[priority]++;
                    engineHardKv[priority] = saturatedAddNonNegative(engineHardKv[priority], hard);
                    engineExpectedKv[priority] = saturatedAddNonNegative(engineExpectedKv[priority], expected);
                }
            }
            CapacityRelease[] placementOccupancy = new CapacityRelease[PriorityNormalizer.MAX_PRIORITY + 1];
            CapacityRelease[] engineOccupancy = new CapacityRelease[PriorityNormalizer.MAX_PRIORITY + 1];
            for (int priority = 0; priority < placementOccupancy.length; priority++) {
                placementOccupancy[priority] = requests[priority] == 0 ? CapacityRelease.NONE
                        : new CapacityRelease(requests[priority], hardKv[priority], expectedKv[priority]);
                engineOccupancy[priority] = engineRequests[priority] == 0 ? CapacityRelease.NONE
                        : new CapacityRelease(engineRequests[priority], engineHardKv[priority], engineExpectedKv[priority]);
            }
            admissionSummaryCache = new DecodeResources.AdmissionSummary(routing, placementOccupancy, engineOccupancy);
            return admissionSummaryCache;
        } finally {
            admissionLock.unlock();
        }
    }

    private boolean isCurrentAdmissionSummary(DecodeResources.AdmissionSummary summary) {
        return summary != null
                && summary.routing().admissionVersion() == admissionVersion
                && summary.routing().workerStatus() == status.committedWorkerStatus()
                && summary.routing().topology() == status.topologySnapshot();
    }

    ResourceSnapshot resourceSnapshot() {
        admissionLock.lock();
        try {
            Map<Long, DecodeRequestView> requests = new HashMap<>(reservedUsage.requests + Math.max(0, confirmedEngineOwnedCount));
            decodeRequests.forEach((requestId, task) -> requests.put(requestId, new DecodeRequestView(
                    requestId, task.priority, task.kvTokens, task.expectedKvTokens,
                    task.phase, task.priorityKnown(), task.reservationToken, victimClaims.containsKey(requestId))));
            return new ResourceSnapshot(routingViewLocked(), requests,
                    queuedUsage.requests, dispatchUsage.requests);
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
                admissionVersion);
    }

    private DecodeRoutingView routingViewLocked(
            String address,
            WorkerStatus.TopologySnapshot topology,
            WorkerStatus.CommittedWorkerStatus committed,
            long version) {
        WorkerStatus.EngineObservation fields = committed.fields();
        int inflight = reservedUsage.requests;
        int totalLoad = confirmedEngineOwnedCount + inflight;
        int engineLoad = getEngineLoad();
        long reportedUsed = reportedKvUsed(fields);
        long hardInflight = reservedUsage.hardKv;
        long expectedInflight = reservedUsage.expectedKv;
        long used = saturatedAddNonNegative(
                saturatedAddNonNegative(reportedUsed, expectedInflight),
                priorityPreemptionHeldExpectedKv);
        long heldHard = priorityPreemptionHeldKv;
        long placementHard = saturatedAddNonNegative(hardInflight, heldHard);
        CapacityUsage placementUsage = new CapacityUsage(totalLoad,
                Math.max(0L, fields.totalKvCacheTokens()), Math.max(0L, fields.availableKvCacheTokens()),
                placementHard, used);
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
                hardInflight);
    }

    DecodeRoutingView routingViewSnapshot(String address) {
        long version = admissionVersion;
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
            version = admissionVersion;
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
        return admissionVersion;
    }

    int getInflightCount() {
        return reservedUsage.requests;
    }

    int getTotalLoad() {
        return confirmedEngineOwnedCount + reservedUsage.requests;
    }

    Stats stats() {
        return new Stats(getInflightCount(), getTotalLoad(), reservedUsage.expectedKv,
                reservedUsage.hardKv, inflightMaxAgeMs(System.currentTimeMillis()));
    }

    private long inflightMaxAgeMs(long nowMs) {
        long oldest = Long.MAX_VALUE;
        for (DecodeRequestState request : decodeRequests.values()) {
            long observedAtMs = request.observedAtMs;
            // A concurrent confirmation changes the timestamp's meaning; check phase after reading it.
            if (!request.confirmed()) {
                oldest = Math.min(oldest, observedAtMs);
            }
        }
        return oldest == Long.MAX_VALUE
                ? 0L : Math.max(0L, nowMs - oldest);
    }

    record Stats(int inflight, int totalLoad, long expectedKv, long hardKv, long oldestAgeMs) { }

    // Shared request identity and accounting values.

    private DecodeRequestState shadowReservation(long requestId) {
        DecodeRequestState state = decodeRequests.get(requestId);
        DecodeTaskPhase phase = state == null ? null : state.phase;
        return phase == DecodeTaskPhase.LOCAL_RESERVED || phase == DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED
                || phase == DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN ? state : null;
    }

    private DecodeRequestState confirmedRequest(long requestId) {
        DecodeRequestState state = decodeRequests.get(requestId);
        return state != null && state.confirmed() ? state : null;
    }

    private static boolean isExactReservation(
            DecodeRequestState current,
            ReservationHandle reservation) {
        return current != null
                && current.reservationToken
                        == reservation.reservationToken();
    }

    private static final class DecodeRequestState {
        static final int DEFAULT_PRIORITY = 0;

        private long kvTokens;
        private long expectedKvTokens;
        private final int priority;
        private final long reservationToken;
        /** Resource ownership phase; claim-only and settled history live in separate tables. */
        private volatile DecodeTaskPhase phase = DecodeTaskPhase.LOCAL_RESERVED;
        /** Creation while reserved, latest Engine observation while confirmed, then settlement time. */
        private long observedAtMs;
        private DispatchLease dispatchPermit;

        DecodeRequestState(long kvTokens, long expectedKvTokens,
                        int priority, long reservationToken) {
            this.kvTokens = kvTokens;
            this.expectedKvTokens = expectedKvTokens;
            this.observedAtMs = System.currentTimeMillis();
            this.priority = priority;
            this.reservationToken = reservationToken;
        }

        CapacityRelease capacityRelease() {
            return new CapacityRelease(1L, kvTokens, expectedKvTokens);
        }
        boolean queued() { return phase == DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED; }
        boolean confirmed() {
            DecodeTaskPhase observed = phase;
            return observed.isEngineConfirmed();
        }
        boolean priorityKnown() { return reservationToken > 0L; }

        /** First allocation replaces local estimates; later reports only renew phase and activity. */
        void recordEngineAllocation(long engineKvTokens, DecodeTaskPhase phase, long observedAtMs) {
            if (!confirmed()) {
                kvTokens = Math.max(0L, engineKvTokens);
                expectedKvTokens = kvTokens;
                dispatchPermit = null;
            }
            refresh(phase, observedAtMs);
        }

        void refresh(DecodeTaskPhase phase, long observedAtMs) {
            this.phase = java.util.Objects.requireNonNull(phase, "phase");
            this.observedAtMs = observedAtMs;
        }

        DispatchLease clearDispatchPermit() {
            DispatchLease current = dispatchPermit;
            dispatchPermit = null;
            return current;
        }

    }
}
