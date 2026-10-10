package org.flexlb.balance.endpoint;

import org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.balance.endpoint.DecodeResources.Acquisition;
import org.flexlb.balance.endpoint.DecodeResources.ResourceChange;
import org.flexlb.balance.endpoint.DecodeResources.ReserveStatus;
import org.flexlb.balance.endpoint.DecodeResources.MutationStatus;
import org.flexlb.balance.endpoint.DecodeResources.Ownership;
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

/** Single-generation resource ledger. Every ownership mutation is serialized by admissionLock.
 * Request phase/permit contributions change together; a cancellation claim has a separate lifetime.
 */
final class DecodeState {
    private static final Logger logger = LoggerFactory.getLogger("syncLogger");
    private static final Comparator<ReservationHandle> RETIREMENT_ORDER =
            Comparator.comparingLong(ReservationHandle::endpointGenerationId)
                    .thenComparingLong(ReservationHandle::requestId)
                    .thenComparingLong(ReservationHandle::reservationToken);
    private final WorkerStatus status;
    private final ReentrantLock admissionLock = new ReentrantLock();
    private final ConcurrentHashMap<Long, DecodeRequestState> decodeRequests = new ConcurrentHashMap<>();
    private final Map<Long, Long> settledHistory = new HashMap<>();
    private final Map<Long, PreemptionClaim> victimClaims = new HashMap<>();
    private final Map<Long, EndpointPreemptionAttempt> preemptionAttempts = new HashMap<>();
    private final ResourceUsage reservedUsage = new ResourceUsage();
    private final ResourceUsage queuedUsage = new ResourceUsage();
    private final ResourceUsage dispatchUsage = new ResourceUsage();
    private volatile int confirmedEngineOwnedCount;
    private volatile long priorityPreemptionHeldKv;
    private volatile long priorityPreemptionHeldExpectedKv;
    private volatile long admissionVersion;
    private volatile DecodeRoutingView routingViewCache;
    private volatile DecodeResources.AdmissionSummary admissionSummaryCache;
    private long nextReservationToken = 1L;
    /** Guarded by admissionLock; a full observation publishes one revision after all rows settle. */
    private boolean calibrating;

    DecodeState(WorkerStatus status) {
        this.status = java.util.Objects.requireNonNull(status, "status");
    }

    Acquisition<ReserveStatus, ReservationHandle> tryReserveQueuedRequest(long requestId, long hardKv, long expectedKv,
                                                                         int priority, AdmissionCapacity capacity) {
        admissionLock.lock();
        try {
            if (!requestIdAvailable(requestId)) { return Acquisition.unavailable(ReserveStatus.ALREADY_OWNED); }
            if (capacity != null
                    && !capacity.evaluate(routingView(status.getIpPort()).placementUsage(), hardKv, expectedKv, CapacityRelease.NONE).fits()) {
                return Acquisition.unavailable(ReserveStatus.CAPACITY_FULL);
            }
            return createReservation(requestId, hardKv, expectedKv, priority, DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED);
        } finally { admissionLock.unlock(); }
    }

    private Acquisition<ReserveStatus, ReservationHandle> createReservation(long requestId, long hardKv, long expectedKv,
                                                                            int priority, DecodeTaskPhase phase) {
        checkState(nextReservationToken > 0L && nextReservationToken != Long.MAX_VALUE,
                "Decode reservation token space exhausted");
        DecodeRequestState request = new DecodeRequestState(requestId, hardKv, expectedKv, priority, nextReservationToken++);
        var acquired = new Acquisition<>(ReserveStatus.ACQUIRED, request.handle());
        decodeRequests.put(requestId, request);
        reservedUsage.adjust(1, hardKv, expectedKv);
        if (phase == DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED) { queuedUsage.adjust(1, hardKv, expectedKv); }
        request.phase = phase;
        admissionVersion++;
        return acquired;
    }

    private boolean requestIdAvailable(long requestId) {
        // Every installed incoming attempt retains its exact request row until commit/abort/expiry.
        return !decodeRequests.containsKey(requestId) && !victimClaims.containsKey(requestId)
                && !settledHistory.containsKey(requestId);
    }

    Ownership ownership(ReservationHandle exact) {
        if (exact == null || exact.endpointGenerationId() != status.getGenerationId()) { return Ownership.NONE; }
        admissionLock.lock();
        try {
            DecodeRequestState request = decodeRequests.get(exact.requestId());
            if (request == null || !request.matches(exact)) {
                PreemptionClaim claim = victimClaims.get(exact.requestId());
                return claim != null && claim.reservation.equals(exact) ? Ownership.HELD : Ownership.NONE;
            }
            if (request.confirmed()) { return Ownership.CONFIRMED; }
            return request.queued() && request.dispatchPermit == null && request.incomingAttemptToken == 0L
                    && !victimClaims.containsKey(exact.requestId()) ? Ownership.WITHDRAWABLE : Ownership.HELD;
        } finally { admissionLock.unlock(); }
    }

    ResourceChange<ReservationReleaseResult> release(ReservationHandle exact, ReleaseReason reason) {
        java.util.Objects.requireNonNull(reason, "reason");
        if (exact == null) {
            checkArgument(reason != ReleaseReason.LOCAL_ROLLBACK && reason != ReleaseReason.NOT_SENT,
                    "Decode reservation is required for release");
            return ResourceChange.of(ReservationReleaseResult.STALE, false);
        }
        if (exact.endpointGenerationId() != status.getGenerationId()) { return ResourceChange.of(ReservationReleaseResult.STALE, false); }
        admissionLock.lock();
        try {
            if (reason == ReleaseReason.EXPIRED || reason == ReleaseReason.REMOTE_CLEANUP) {
                boolean released = settle(exact, Settlement.EXPIRED);
                return ResourceChange.of(released ? ReservationReleaseResult.RELEASED : ReservationReleaseResult.STALE, released);
            }
            DecodeRequestState request = decodeRequests.get(exact.requestId());
            boolean matches = request != null && request.matches(exact);
            if (reason == ReleaseReason.NOT_SENT) {
                if (matches && request.confirmed()) { return ResourceChange.of(ReservationReleaseResult.ENGINE_ACCEPTED, false); }
                if (matches && victimClaims.containsKey(exact.requestId())) {
                    boolean released = settle(exact, Settlement.TERMINAL);
                    return ResourceChange.of(released ? ReservationReleaseResult.RELEASED : ReservationReleaseResult.CONFLICT, released);
                }
                if (matches && request.incomingAttemptToken != 0L) { return ResourceChange.of(ReservationReleaseResult.CONFLICT, false); }
            } else if (matches && (request.incomingAttemptToken != 0L || request.confirmed()
                    || request.phase == DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN || victimClaims.containsKey(exact.requestId()))) {
                checkState(reason != ReleaseReason.LOCAL_ROLLBACK,
                        "Illegal Decode local release: exact ownership is held by Engine/protocol lifecycle: %s", exact);
                return ResourceChange.of(ReservationReleaseResult.STILL_OWNED, false);
            }
            if (!matches) { return ResourceChange.of(ReservationReleaseResult.STALE, false); }
            boolean released = settle(exact, reason == ReleaseReason.LOCAL_ROLLBACK ? Settlement.ROLLBACK : Settlement.TERMINAL);
            return ResourceChange.of(released ? ReservationReleaseResult.RELEASED : ReservationReleaseResult.STALE, released);
        } finally { admissionLock.unlock(); }
    }

    private enum Settlement { ROLLBACK, TERMINAL, OBSERVED_TERMINAL, EXPIRED }

    /** Only validated evidence reaches this kernel; request and claim obligations settle together. */
    private boolean settle(ReservationHandle exact, Settlement proof) {
        DecodeRequestState request = decodeRequests.get(exact.requestId());
        boolean matches = request != null && request.matches(exact);
        PreemptionClaim claim = victimClaims.get(exact.requestId());
        if (claim != null && !claim.reservation.equals(exact)) { claim = null; }
        if (!matches && claim == null) { return false; }
        long version = admissionVersion;
        if (matches && request.incomingAttemptToken != 0L) {
            checkState(proof == Settlement.EXPIRED, "Decode terminal still owns its incoming attempt: %s", exact);
            finishPreemption(request.incomingAttemptToken, false);
        }
        if (claim != null) {
            EndpointPreemptionAttempt attempt = preemptionAttempts.get(claim.attemptToken);
            if (attempt != null) { attempt.remainingVictims.remove(exact); }
            claim.close();
        }
        if (matches && decodeRequests.remove(exact.requestId(), request)) { transition(request, null, null, 0L, 0L); }
        // Ordinary Engine completion may be rediscovered; cancellation claims retain their explicit fence.
        if (proof != Settlement.ROLLBACK && (proof != Settlement.OBSERVED_TERMINAL || claim != null)) {
            settledHistory.put(exact.requestId(), System.currentTimeMillis());
        }
        if (!calibrating && admissionVersion == version) { admissionVersion++; }
        return true;
    }

    /** Update derived counters once from old/new contributions, without allocating contribution objects. */
    private void transition(DecodeRequestState request, DecodeTaskPhase next, DispatchLease permit,
                            long engineKv, long observedAtMs) {
        boolean reserved = !request.confirmed();
        boolean queued = request.queued();
        boolean dispatching = request.dispatchPermit != null;
        boolean nextReserved = next != null && !next.isEngineConfirmed();
        boolean nextQueued = next == DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED;
        boolean nextDispatching = permit != null;
        long hard = request.kvTokens, expected = request.expectedKvTokens;
        reservedUsage.adjust((nextReserved ? 1 : 0) - (reserved ? 1 : 0),
                (nextReserved ? hard : 0L) - (reserved ? hard : 0L),
                (nextReserved ? expected : 0L) - (reserved ? expected : 0L));
        queuedUsage.adjust((nextQueued ? 1 : 0) - (queued ? 1 : 0),
                (nextQueued ? hard : 0L) - (queued ? hard : 0L),
                (nextQueued ? expected : 0L) - (queued ? expected : 0L));
        dispatchUsage.adjust((nextDispatching ? 1 : 0) - (dispatching ? 1 : 0),
                (nextDispatching ? hard : 0L) - (dispatching ? hard : 0L),
                (nextDispatching ? expected : 0L) - (dispatching ? expected : 0L));
        confirmedEngineOwnedCount += (next != null && next.isEngineConfirmed() ? 1 : 0) - (reserved ? 0 : 1);
        if (reserved && next != null && next.isEngineConfirmed()) {
            request.kvTokens = Math.max(0L, engineKv);
            request.expectedKvTokens = request.kvTokens;
        }
        request.dispatchPermit = permit;
        if (observedAtMs > 0L) { request.observedAtMs = observedAtMs; }
        // A removed row retains its last phase for advisory readers that already captured it.
        if (next != null) { request.phase = next; }
        if (!calibrating) { admissionVersion++; }
    }

    ResourceChange<MutationStatus> markQueued(ReservationHandle exact) {
        checkArgument(exact != null, "Decode reservation is required for queued transition");
        admissionLock.lock();
        try {
            DecodeRequestState request = decodeRequests.get(exact.requestId());
            if (exact.endpointGenerationId() != status.getGenerationId() || request == null
                    || !request.matches(exact) || request.confirmed()) { return ResourceChange.of(MutationStatus.STALE, false); }
            if (request.queued()) { return ResourceChange.of(MutationStatus.APPLIED, false); }
            if (request.phase == DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN) { return ResourceChange.of(MutationStatus.STALE, false); }
            transition(request, DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED, null, 0L, 0L);
            return ResourceChange.of(MutationStatus.APPLIED, true);
        } finally { admissionLock.unlock(); }
    }

    Acquisition<EngineDispatchPermitAcquireStatus, DispatchLease> acquireDispatchPermit(ReservationHandle exact, AdmissionCapacity capacity) {
        admissionLock.lock();
        try {
            DecodeRequestState request = decodeRequests.get(exact.requestId());
            if (exact.endpointGenerationId() != status.getGenerationId() || request == null
                    || !request.matches(exact) || victimClaims.containsKey(exact.requestId())) {
                return Acquisition.unavailable(EngineDispatchPermitAcquireStatus.NOT_OWNED);
            }
            if (request.confirmed() || request.phase == DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN) {
                return new Acquisition<>(EngineDispatchPermitAcquireStatus.ALREADY_ACCEPTED, new DispatchLease(request));
            }
            if (!request.queued()) { return Acquisition.unavailable(EngineDispatchPermitAcquireStatus.NOT_QUEUED); }
            if (request.dispatchPermit != null) { return Acquisition.unavailable(EngineDispatchPermitAcquireStatus.ALREADY_ACQUIRED); }
            if (!capacity.evaluate(usage(true, status.committedWorkerStatus().fields()), request.kvTokens,
                    request.expectedKvTokens, CapacityRelease.NONE).fits()) {
                return Acquisition.unavailable(EngineDispatchPermitAcquireStatus.CAPACITY_FULL);
            }
            DispatchLease permit = new DispatchLease(request);
            transition(request, request.phase, permit, 0L, 0L);
            return new Acquisition<>(EngineDispatchPermitAcquireStatus.ACQUIRED, permit);
        } finally { admissionLock.unlock(); }
    }

    ResourceChange<EngineDispatchPermitTransferStatus> dispatch(DispatchLease permit, DispatchOutcome outcome) {
        java.util.Objects.requireNonNull(permit, "permit");
        java.util.Objects.requireNonNull(outcome, "outcome");
        admissionLock.lock();
        try {
            DecodeRequestState current = decodeRequests.get(permit.reservation.requestId);
            boolean engineOwned = outcome == DispatchOutcome.ENGINE_OWNED;
            if (engineOwned ? current == permit.reservation && (current.confirmed()
                    || current.phase == DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN)
                    && !victimClaims.containsKey(current.requestId) : permit.retiredByEndpoint) {
                return ResourceChange.of(EngineDispatchPermitTransferStatus.TRANSFERRED, false);
            }
            if (current != permit.reservation || current.dispatchPermit != permit) {
                return ResourceChange.of(EngineDispatchPermitTransferStatus.OWNERSHIP_LOST, false);
            }
            int before = confirmedEngineOwnedCount + reservedUsage.requests - queuedUsage.requests + dispatchUsage.requests;
            boolean transferable = !engineOwned || current.queued() && !victimClaims.containsKey(current.requestId);
            transition(current, engineOwned && transferable ? DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN : current.phase,
                    null, 0L, 0L);
            int after = confirmedEngineOwnedCount + reservedUsage.requests - queuedUsage.requests + dispatchUsage.requests;
            return ResourceChange.of(transferable ? EngineDispatchPermitTransferStatus.TRANSFERRED
                    : EngineDispatchPermitTransferStatus.OWNERSHIP_LOST, !engineOwned || after < before);
        } finally { admissionLock.unlock(); }
    }

    boolean shouldRetryDispatch(ReservationHandle exact, AdmissionCapacity capacity) {
        DecodeRequestState request = decodeRequests.get(exact.requestId());
        return exact.endpointGenerationId() != status.getGenerationId() || request == null || !request.matches(exact)
                || !request.queued() || request.dispatchPermit != null
                || capacity.evaluate(usage(true, status.committedWorkerStatus().fields()), request.kvTokens,
                        request.expectedKvTokens, CapacityRelease.NONE).fits();
    }

    private CapacityUsage usage(boolean dispatch, WorkerStatus.EngineObservation fields) {
        long hard = dispatch ? Math.max(0L, reservedUsage.hardKv - queuedUsage.hardKv) + dispatchUsage.hardKv : reservedUsage.hardKv;
        long expected = dispatch ? Math.max(0L, reservedUsage.expectedKv - queuedUsage.expectedKv) + dispatchUsage.expectedKv : reservedUsage.expectedKv;
        long reported = fields.totalKvCacheTokens() > 0L
                ? Math.max(0L, fields.totalKvCacheTokens() - fields.availableKvCacheTokens()) : 0L;
        int count = confirmedEngineOwnedCount + reservedUsage.requests;
        if (dispatch) { count += Math.max(0, dispatchUsage.requests) - Math.clamp(queuedUsage.requests, 0, Math.max(0, reservedUsage.requests)); }
        return new CapacityUsage(count, Math.max(0L, fields.totalKvCacheTokens()), Math.max(0L, fields.availableKvCacheTokens()),
                saturatedAddNonNegative(hard, priorityPreemptionHeldKv),
                saturatedAddNonNegative(saturatedAddNonNegative(reported, expected), priorityPreemptionHeldExpectedKv));
    }

    ReservationHandle replaceQueuedRequests(List<ReservationHandle> victims, long incomingId,
                                             long hardKv, long expectedKv, int priority, AdmissionCapacity capacity) {
        admissionLock.lock();
        try {
            if (victims == null || victims.isEmpty() || !requestIdAvailable(incomingId)) { return null; }
            Set<Long> unique = new HashSet<>(victims.size());
            CapacityRelease released = CapacityRelease.NONE;
            for (ReservationHandle victim : victims) {
                if (victim == null || victim.requestId() == incomingId || !unique.add(victim.requestId())
                        || ownership(victim) != Ownership.WITHDRAWABLE) { return null; }
                released = released.plus(decodeRequests.get(victim.requestId()).capacityRelease());
            }
            CapacityUsage usage = routingView(status.getIpPort()).placementUsage();
            if (capacity.evaluate(usage, hardKv, expectedKv, CapacityRelease.NONE).fits()
                    || !capacity.evaluate(usage, hardKv, expectedKv, released).fits()) { return null; }
            for (ReservationHandle victim : victims) { settle(victim, Settlement.ROLLBACK); }
            return createReservation(incomingId, hardKv, expectedKv, priority, DecodeTaskPhase.LOCAL_RESERVED).handle();
        } finally { admissionLock.unlock(); }
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
            if (!requestIdAvailable(incomingRequestId)) {
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
                        || request != null && request.incomingAttemptToken != 0L) {
                    return PreemptionBeginResult.VICTIM_ALREADY_CLAIMED;
                }
                if (request == null || !request.matches(victim)) {
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
            CapacityUsage usage = routingView(status.getIpPort()).placementUsage();
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
            ReservationHandle incomingReservation = createReservation(
                    incomingRequestId, incomingKvTokens, incomingExpectedKvTokens, incomingPriority,
                    DecodeTaskPhase.LOCAL_RESERVED).handle();
            EndpointPreemptionAttempt preparedAttempt = null;
            try {
                preparedAttempt = new EndpointPreemptionAttempt(incomingReservation, exactVictims);
                preemptionAttempts.put(attemptToken, preparedAttempt);
                decodeRequests.get(incomingRequestId).incomingAttemptToken = attemptToken;
                for (Map.Entry<Long, PreemptionClaim> claim
                        : preparedClaims.entrySet()) {
                    victimClaims.put(claim.getKey(), claim.getValue());
                }
            } catch (RuntimeException | Error installationFailure) {
                if (preparedAttempt != null) {
                    preemptionAttempts.remove(attemptToken, preparedAttempt);
                    decodeRequests.get(incomingRequestId).incomingAttemptToken = 0L;
                }
                for (Map.Entry<Long, PreemptionClaim> claim
                        : preparedClaims.entrySet()) {
                    claim.getValue().close();
                }
                settle(incomingReservation, Settlement.ROLLBACK);
                throw installationFailure;
            }
            admissionVersion++;
            return PreemptionBeginResult.SUCCESS;
        } finally {
            admissionLock.unlock();
        }
    }


    ResourceChange<MutationStatus> updatePreemption(long token, PreemptionUpdate update) {
        java.util.Objects.requireNonNull(update, "update");
        admissionLock.lock();
        try {
            PreemptionClaim claim = victimClaims.get(update.reservation().requestId());
            if (claim == null || claim.attemptToken != token || !claim.reservation.equals(update.reservation())) { return ResourceChange.of(MutationStatus.STALE, false); }
            switch (update.kind()) {
                case CANCELED, REQUEST_FENCED -> {
                    boolean changed = settle(claim.reservation, Settlement.TERMINAL);
                    return ResourceChange.of(changed ? MutationStatus.APPLIED : MutationStatus.STALE, changed);
                }
                case CANCEL_HANDED_OFF -> claim.cancellationHandedOff = true;
                case ACTIVE, FINISHED -> {
                    long version = admissionVersion;
                    claim.close();
                    if (update.kind() == PreemptionUpdate.Kind.FINISHED) { settle(claim.reservation, Settlement.OBSERVED_TERMINAL); }
                    if (!calibrating && admissionVersion == version) { admissionVersion++; }
                }
                default -> throw new IllegalStateException("Unhandled preemption update: " + update.kind());
            }
            return ResourceChange.of(MutationStatus.APPLIED, update.releasesCapacity());
        } finally { admissionLock.unlock(); }
    }

    ReservationHandle finishPreemption(long token, boolean commit) {
        admissionLock.lock();
        try {
            EndpointPreemptionAttempt attempt = preemptionAttempts.get(token);
            if (attempt == null) { return null; }
            DecodeRequestState incoming = decodeRequests.get(attempt.incoming.requestId());
            if (commit && (incoming == null || !incoming.matches(attempt.incoming)
                    || incoming.confirmed() || !attempt.remainingVictims.isEmpty())) { return null; }
            long version = admissionVersion;
            preemptionAttempts.remove(token);
            if (incoming != null && incoming.matches(attempt.incoming)) { incoming.incomingAttemptToken = 0L; }
            if (!commit) {
                for (ReservationHandle victim : attempt.remainingVictims) {
                    PreemptionClaim claim = victimClaims.get(victim.requestId());
                    if (claim != null && claim.attemptToken == token && claim.reservation.equals(victim)
                            && !claim.cancellationHandedOff) { claim.close(); }
                }
                if (incoming != null && !incoming.confirmed()) { settle(attempt.incoming, Settlement.ROLLBACK); }
            }
            if (!calibrating && admissionVersion == version) { admissionVersion++; }
            return attempt.incoming;
        } finally { admissionLock.unlock(); }
    }

    ReentrantLock ownershipLock() { return admissionLock; }

    List<DecodeRequestStatus> calibrateLocked(WorkerStatus.StatusObservation observation) {
        checkState(admissionLock.isHeldByCurrentThread(), "Decode calibration requires ownershipLock");
        checkArgument(observation.owner() == status, "Status belongs to another Decode generation");
        checkState(!calibrating, "Decode calibration cannot reenter");
        calibrating = true;
        try {
            List<DecodeRequestStatus> evidence = new ArrayList<>();
            Set<Long> present = new HashSet<>(), allocated = new HashSet<>(), terminal = new HashSet<>();
            observation.finishedTasks().values().forEach(task -> terminal.add(task.requestId()));
            long now = System.currentTimeMillis();
            for (WorkerStatus.TaskObservation task : observation.engine().runningTaskList().values()) {
                long id = task.requestId();
                if (terminal.contains(id) || settledHistory.containsKey(id)) { continue; }
                present.add(id);
                boolean allocation = task.phase() == TaskPhase.KV_ALLOCATED || task.phase() == TaskPhase.RUNNING;
                if (allocation) { allocated.add(id); }
                observeActive(task, allocation, now);
                DecodeRequestState request = decodeRequests.get(id);
                if (request != null && request.reservationToken > 0L) {
                    evidence.add(allocation ? DecodeRequestStatus.allocated(request.handle()) : DecodeRequestStatus.active(request.handle()));
                }
            }
            // Capture exact terminal proof before absent-owner pruning can discard its identity.
            for (WorkerStatus.TaskObservation task : observation.finishedTasks().values()) {
                long id = task.requestId();
                if (settledHistory.containsKey(id)) { continue; }
                DecodeRequestState request = decodeRequests.get(id);
                PreemptionClaim claim = victimClaims.get(id);
                ReservationHandle exact = request != null && request.reservationToken > 0L ? request.handle()
                        : claim == null ? null : claim.reservation;
                if (exact != null) {
                    if (settle(exact, Settlement.OBSERVED_TERMINAL)) { evidence.add(DecodeRequestStatus.terminal(exact, task.errorCode())); }
                } else if (request != null && request.confirmed()) {
                    decodeRequests.remove(id, request);
                    transition(request, null, null, 0L, 0L);
                }
            }
            for (DecodeRequestState request : decodeRequests.values()) {
                if (!request.confirmed()) { continue; }
                PreemptionClaim claim = victimClaims.get(request.requestId);
                if (claim != null) { claim.hold(!allocated.contains(request.requestId)); }
                else if (!present.contains(request.requestId)) {
                    // FULL absence ends this Engine owner and invalidates its provisional admission.
                    // Abort keeps already-outbound victim claims until their independent terminal proof.
                    if (request.incomingAttemptToken != 0L) { finishPreemption(request.incomingAttemptToken, false); }
                    if (decodeRequests.remove(request.requestId, request)) { transition(request, null, null, 0L, 0L); }
                    settledHistory.put(request.requestId, now);
                }
            }
            return List.copyOf(evidence);
        } finally {
            calibrating = false;
            admissionVersion++;
        }
    }

    private void observeActive(WorkerStatus.TaskObservation task, boolean allocated, long now) {
        DecodeRequestState request = decodeRequests.get(task.requestId());
        if (allocated) {
            PreemptionClaim claim = victimClaims.get(task.requestId());
            if (claim != null) { claim.hold(false); }
            if (request == null) {
                request = new DecodeRequestState(task.requestId(), 0L, 0L, 0, 0L);
                decodeRequests.put(task.requestId(), request);
                reservedUsage.adjust(1, 0L, 0L);
            }
            transition(request, task.phase() == TaskPhase.RUNNING ? DecodeTaskPhase.RUNNING
                    : DecodeTaskPhase.ACCEPTED_NOT_RUNNING, null, task.inputLength(), now);
        } else if (request != null) {
            transition(request, request.confirmed() ? request.phase : DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN,
                    null, 0L, now);
        }
    }

    /** A private candidate has no local owners and therefore cannot emit request evidence. */
    void initialize(WorkerStatus.StatusObservation observation) {
        checkArgument(observation.owner() == status, "Status belongs to another Decode generation");
        admissionLock.lock();
        try {
            checkState(calibrateLocked(observation).isEmpty(),
                    "Private Decode candidate produced locally-owned request statuses");
        } finally { admissionLock.unlock(); }
    }

    List<DecodeRequestStatus> collectHeartbeatRequestStatuses(WorkerStatus.StatusObservation observation) {
        checkArgument(observation.owner() == status, "Status observation belongs to another Decode generation");
        List<DecodeRequestStatus> evidence = new ArrayList<>(observation.runningTasks().size());
        admissionLock.lock();
        try {
            for (WorkerStatus.TaskObservation task : observation.runningTasks().values()) {
                DecodeRequestState request = decodeRequests.get(task.requestId());
                PreemptionClaim claim = victimClaims.get(task.requestId());
                ReservationHandle exact = request != null && request.reservationToken > 0L ? request.handle()
                        : request == null && claim != null ? claim.reservation : null;
                if (exact != null) { evidence.add(task.phase() == TaskPhase.KV_ALLOCATED || task.phase() == TaskPhase.RUNNING
                        ? DecodeRequestStatus.allocated(exact) : DecodeRequestStatus.active(exact)); }
            }
        } finally { admissionLock.unlock(); }
        return List.copyOf(evidence);
    }

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
                DispatchLease permit = reservation.dispatchPermit;
                if (permit != null) { permit.retiredByEndpoint = true; reservation.dispatchPermit = null; }
            }
            decodeRequests.clear();
            settledHistory.clear();
            victimClaims.clear();
            dispatchUsage.reset();
            reservedUsage.reset();
            queuedUsage.reset();
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
                if (!expired) { continue; }
                // An orphan incoming row must also surrender its attempt and local victim claims.
                if (request.incomingAttemptToken != 0L) { finishPreemption(request.incomingAttemptToken, false); }
                if (decodeRequests.remove(requestId, request)) { transition(request, null, null, 0L, 0L); }
                if (!confirmed) { expiredReservations++; }
                changed = true;
            }
            if (changed) { admissionVersion++; }
            return new CleanupResult(expiredReservations, changed);
        } finally {
            admissionLock.unlock();
        }
    }


    record CleanupResult(int expiredReservations, boolean capacityReleased) { }

    DecodeRoutingView routingView(String address) {
        DecodeRoutingView cached = routingViewCache;
        if (cached != null && cached.address().equals(address) && cached.admissionVersion() == admissionVersion
                && cached.workerStatus() == status.committedWorkerStatus() && cached.topology() == status.topologySnapshot()) { return cached; }
        admissionLock.lock();
        try {
            WorkerStatus.CommittedWorkerStatus committed = status.committedWorkerStatus();
            WorkerStatus.TopologySnapshot topology = status.topologySnapshot();
            cached = routingViewCache;
            if (cached == null || !cached.address().equals(address) || cached.admissionVersion() != admissionVersion
                    || cached.workerStatus() != committed || cached.topology() != topology) {
                cached = new DecodeRoutingView(address, status.getGenerationId(), topology, committed,
                        admissionVersion, confirmedEngineOwnedCount + reservedUsage.requests,
                        confirmedEngineOwnedCount + reservedUsage.requests - Math.clamp(queuedUsage.requests, 0, Math.max(0, reservedUsage.requests)),
                        usage(false, committed.fields()), usage(true, committed.fields()), reservedUsage.hardKv);
                routingViewCache = cached;
            }
            return cached;
        } finally { admissionLock.unlock(); }
    }

    DecodeResources.AdmissionSummary admissionSummary() {
        DecodeResources.AdmissionSummary cached = admissionSummaryCache;
        if (cached != null && cached.routing() == routingView(status.getIpPort())) { return cached; }
        admissionLock.lock();
        try {
            cached = admissionSummaryCache;
            if (cached != null && cached.routing() == routingView(status.getIpPort())) { return cached; }
            DecodeRoutingView routing = routingView(status.getIpPort());
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

    ResourceSnapshot resourceSnapshot() {
        admissionLock.lock();
        try {
            Map<Long, DecodeRequestView> requests = new HashMap<>(reservedUsage.requests + Math.max(0, confirmedEngineOwnedCount));
            decodeRequests.forEach((requestId, task) -> requests.put(requestId, new DecodeRequestView(
                    requestId, task.priority, task.kvTokens, task.expectedKvTokens,
                    task.phase, task.priorityKnown(), task.reservationToken, victimClaims.containsKey(requestId))));
            return new ResourceSnapshot(routingView(status.getIpPort()), requests,
                    queuedUsage.requests, dispatchUsage.requests);
        } finally {
            admissionLock.unlock();
        }
    }


    long placementVersion() { return admissionVersion; }

    Stats stats() {
        long oldest = Long.MAX_VALUE;
        for (DecodeRequestState request : decodeRequests.values()) {
            long observed = request.observedAtMs;
            if (!request.confirmed()) { oldest = Math.min(oldest, observed); }
        }
        long age = oldest == Long.MAX_VALUE ? 0L : Math.max(0L, System.currentTimeMillis() - oldest);
        return new Stats(reservedUsage.requests, confirmedEngineOwnedCount + reservedUsage.requests,
                reservedUsage.expectedKv, reservedUsage.hardKv, age);
    }

    record Stats(int inflight, int totalLoad, long expectedKv, long hardKv, long oldestAgeMs) { }

    static final class DispatchLease {
        private final DecodeRequestState reservation;
        private boolean retiredByEndpoint;
        private DispatchLease(DecodeRequestState reservation) { this.reservation = reservation; }
        boolean matches(ReservationHandle exact) { return reservation.matches(exact); }
    }

    private static final class ResourceUsage {
        private volatile int requests;
        private volatile long hardKv;
        private volatile long expectedKv;
        void adjust(int count, long hard, long expected) {
            if (count != 0) { requests += count; }
            if (hard != 0L) { hardKv += hard; }
            if (expected != 0L) { expectedKv += expected; }
        }
        void reset() { requests = 0; hardKv = 0L; expectedKv = 0L; }
    }

    private final class DecodeRequestState {
        private final long requestId;
        private final long reservationToken;
        private final int priority;
        private long kvTokens;
        private long expectedKvTokens;
        private long observedAtMs = System.currentTimeMillis();
        private volatile DecodeTaskPhase phase = DecodeTaskPhase.LOCAL_RESERVED;
        private DispatchLease dispatchPermit;
        /** Non-owning reverse index, maintained atomically with preemptionAttempts under admissionLock. */
        private long incomingAttemptToken;
        DecodeRequestState(long requestId, long hard, long expected, int priority, long token) {
            this.requestId = requestId;
            this.reservationToken = token;
            this.kvTokens = hard;
            this.expectedKvTokens = expected;
            this.priority = priority;
        }
        ReservationHandle handle() { return new ReservationHandle(status.getGenerationId(), requestId, reservationToken); }
        boolean matches(ReservationHandle exact) { return exact != null && exact.endpointGenerationId() == status.getGenerationId()
                && exact.requestId() == requestId && exact.reservationToken() == reservationToken; }
        boolean queued() { return phase == DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED; }
        boolean confirmed() { return phase.isEngineConfirmed(); }
        boolean priorityKnown() { return reservationToken > 0L; }
        CapacityRelease capacityRelease() { return new CapacityRelease(1L, kvTokens, expectedKvTokens); }
    }

    private record EndpointPreemptionAttempt(ReservationHandle incoming, Set<ReservationHandle> remainingVictims) { }

    private final class PreemptionClaim {
        private final long attemptToken;
        private final ReservationHandle reservation;
        private final long hardKvTokens;
        private final long expectedKvTokens;
        private boolean cancellationHandedOff;
        private boolean kvHeldAfterWorkerRelease;
        PreemptionClaim(long token, ReservationHandle reservation, long hard, long expected) {
            checkArgument(hard >= 0L && expected >= hard, "Priority preemption claim requires expected KV >= hard KV >= 0");
            this.attemptToken = token;
            this.reservation = reservation;
            this.hardKvTokens = hard;
            this.expectedKvTokens = expected;
        }
        void close() {
            if (victimClaims.get(reservation.requestId()) != this) { return; }
            hold(false);
            victimClaims.remove(reservation.requestId(), this);
        }
        void hold(boolean held) {
            checkState(admissionLock.isHeldByCurrentThread(), "Priority preemption KV hold mutation requires admissionLock");
            if (kvHeldAfterWorkerRelease == held) { return; }
            long hard = priorityPreemptionHeldKv, expected = priorityPreemptionHeldExpectedKv;
            checkState(hard >= 0L && expected >= hard, "Invalid priority preemption KV hold counters");
            checkState(held || hard >= hardKvTokens && expected >= expectedKvTokens, "Priority preemption KV hold counter underflow");
            try {
                hard = Math.addExact(hard, held ? hardKvTokens : -hardKvTokens);
                expected = Math.addExact(expected, held ? expectedKvTokens : -expectedKvTokens);
            } catch (ArithmeticException overflow) { throw new IllegalStateException("Priority preemption KV hold counter overflow", overflow); }
            checkState(hard >= 0L && expected >= hard, "Invalid priority preemption KV hold counters");
            if (held) { priorityPreemptionHeldExpectedKv = expected; priorityPreemptionHeldKv = hard; }
            else { priorityPreemptionHeldKv = hard; priorityPreemptionHeldExpectedKv = expected; }
            kvHeldAfterWorkerRelease = held;
        }
    }
}
