package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.InflightEvictor;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.LongPredicate;

/**
 * Decode-side endpoint with Auto-TPM shadow admission accounting.
 *
 * <p><b>Layered view (Phase 5):</b> {@code inflightRequests} only ever holds
     * shadow entries; engine-confirmed requests are
 * folded into {@code confirmedRunningCount} by calibrate exactly as in
 * Phase 4 (accounting unchanged), and are additionally tracked per-request in
 * {@link #trackedConfirmed} split by phase — {@code KV_ALLOCATED} →
 * {@code ACCEPTED_NOT_RUNNING} layer, {@code RUNNING} → {@code RUNNING}
 * layer — for accepted-eviction planning and layered gauges.
 * {@code totalLoad = confirmedRunningCount + reserved inflight count}.
 *
 * <p><b>Known accepted cost:</b> the admission lock and version bump on every
 * reserve/release/calibrate stay active even when Auto-TPM is disabled; the
 * uncontended ReentrantLock + AtomicLong overhead is negligible and keeping it
 * unconditional avoids divergent code paths (task10 P2-9, no structural change).
 */
public class DecodeEndpoint extends WorkerEndpoint {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final ConcurrentHashMap<Long, RequestInflight> inflightRequests = new ConcurrentHashMap<>();
    private final AtomicLong inflightKvReservedTotal = new AtomicLong(0);
    private final AtomicLong inflightExpectedKvReservedTotal = new AtomicLong(0);
    private final AtomicLong reportedKvAvailable = new AtomicLong();
    private final AtomicLong reportedKvTotal = new AtomicLong();
    private volatile int confirmedRunningCount;
    private final InflightEvictor<Long, RequestInflight> requestEvictor;

    /**
     * Layered registry of engine-confirmed requests (Phase 5): requestId →
     * accepted/running membership. Rebuilt against every calibrate report;
     * carries no shadow accounting — confirmed KV is engine-reported and the
     * slot count stays in {@code confirmedRunningCount}, so this registry is
     * pure metadata for eviction planning, cancel dedup and layered gauges.
     */
    private final ConcurrentHashMap<Long, ConfirmedTask> trackedConfirmed = new ConcurrentHashMap<>();

    /**
     * Token-fenced priority-preemption ownership.  Victim accounting remains
     * in its original layer until a typed Prefill WorkerStatus CANCELED event
     * is settled; an ACCEPTED Cancel response only advances the claim state.
     * All access is under {@link #admissionLock}.
     */
    private final Map<Long, PreemptionClaim> preemptionClaims = new HashMap<>();
    private final Map<Long, EndpointPreemptionAttempt> preemptionAttempts = new HashMap<>();

    /** KV that Decode has reported free but the Prefill CANCELED fence has not settled yet. */
    private final AtomicLong priorityPreemptionHeldKv = new AtomicLong();

    /**
     * Request-scoped ownership retained by a generic scheduler EngineFence.
     *
     * <p>All map and entry mutations are serialized by {@link #admissionLock}.
     * A protection initially pins an existing shadow or confirmed owner. If a
     * confirmed request temporarily disappears from WorkerStatus, ownership is
     * transferred to a synthetic slot plus local KV hold until the exact fence
     * clears or an ordinary authoritative terminal arrives. The two atomic KV
     * totals are read lock-free by routing; they do not own entry lifecycle.
     */
    private final Map<Long, EngineFenceProtection> engineFenceProtections = new HashMap<>();
    private final AtomicLong engineFenceHeldKv = new AtomicLong();
    private final AtomicLong engineFenceHeldExpectedKv = new AtomicLong();
    /** Number of generic synthetic slots; guarded by {@link #admissionLock}. */
    private int engineFenceHeldSlotCount;

    /**
     * Request-id fence against stale WorkerStatus resurrecting any
     * authoritatively settled request: requestId -> settlement time.
     *
     * <p>Only generations settled through a priority/generic EngineFence or an
     * explicit Engine TOMBSTONED proof publish into this registry. Ordinary
     * WorkerStatus completion needs no retained fence: status calibration is
     * serialized, and retaining every completion would make this map grow with
     * request throughput rather than with the small number of ambiguous
     * generations. The endpoint inflight TTL bounds retained fence lifetime;
     * request ids must not be reused inside that reconciliation window because
     * WorkerStatus does not carry a dispatch generation.
     */
    private final Map<Long, Long> settledTombstones = new HashMap<>();

    /**
     * Reserved entries whose request is still sitting in a prefill queue —
     * committed by the scheduler but not yet dispatched to the engine (N2,
     * plan-commit redesign). These reservations keep protecting KV against
     * oversell, but must not count against the decode concurrency limit:
     * counting them produced the shadow-saturation 8400 storm (root cause C —
     * queued reservations saturating {@code getTotalLoad()} while the engine
     * sat idle). Queue schedulers mark before queue publication
     * ({@code markQueuedPhase}) and unmark through an acquired
     * {@link EngineDispatchPermit}; release/calibrate prune it
     * alongside {@code inflightRequests}. DIRECT paths never mark, so their
     * accounting is unchanged.
     */
    private final java.util.Set<Long> queuedPhase = ConcurrentHashMap.newKeySet();

    /**
     * O(1) mirror of {@code |queuedPhase ∩ inflightRequests|} (PR-C):
     * incremented when a reservation is marked queued and decremented when
     * it is dispatched / released / calibrated out, so {@link #getEngineLoad}
     * avoids the per-call O(n) scan of the former full-scan formula. Read lock-free;
     * written under {@link #admissionLock}.
     */
    private final AtomicInteger queuedPhaseCount = new AtomicInteger(0);

    /** Prompt-only KV held by reservations which are still Prefill-queued. */
    private final AtomicLong queuedHardKvReservedTotal = new AtomicLong(0);

    /** Expected KV held by reservations which are still Prefill-queued. */
    private final AtomicLong queuedExpectedKvReservedTotal = new AtomicLong(0);

    /**
     * Decode slots reserved before a queued request is irreversibly exposed to
     * its Prefill/Decode delivery path. The request remains in
     * {@link #queuedPhase}, so {@link #getEngineLoad()} continues to describe
     * only engine-facing work. Capacity acquisition instead uses
     * {@code getEngineLoad() + activeEngineDispatchPermitCount} under
     * {@link #admissionLock}.
     *
     * <p>Each entry carries both the immutable reservation identity and a
     * monotonic token. The pair fences request-id reuse while the slot is
     * reserved. A successful commit removes the lease permanently: committed
     * Decode ownership is never rolled back by a delivery token.
     */
    private final Map<Long, EngineDispatchPermitLease> engineDispatchPermits =
            new ConcurrentHashMap<>();
    /** Hard prompt KV already committed to acquired pre-delivery permits. */
    private final AtomicLong engineDispatchPermitHardKvReservedTotal = new AtomicLong();
    /** Expected KV already committed to acquired pre-delivery permits. */
    private final AtomicLong engineDispatchPermitExpectedKvReservedTotal = new AtomicLong();
    /** Permit tokens invalidated specifically by this endpoint generation retiring. */
    private final Set<Long> retiredEngineDispatchPermitTokens = new HashSet<>();
    /** Mutated under admissionLock; volatile for the lock-free waiter predicate. */
    private volatile int activeEngineDispatchPermitCount;
    /** Closed under admissionLock and read lock-free by blocked capacity waiters. */
    private volatile boolean acceptingEngineDispatchPermits = true;
    /** Guarded by {@link #admissionLock}; zero is never issued. */
    private long nextEngineDispatchPermitToken = 1L;
    /**
     * Prefill workers currently routing to this Decode endpoint. Listeners are
     * invoked only after dropping {@link #admissionLock}.
     */
    private final Set<Runnable> engineDispatchCapacityListeners =
            ConcurrentHashMap.newKeySet();

    /**
     * Serializes admission-state mutations (reserve / release / dispatch
     * permit / calibrate / expired eviction) so that
     * {@link #tryReleaseVictimsAndReserveIncoming} can validate-then-apply
     * atomically against {@link #admissionVersion} (design doc 11.5/17.2).
     * Reads stay lock-free.
     */
    private final ReentrantLock admissionLock = new ReentrantLock();

    /**
     * Monotonic admission version bumped on every mutation of the local
     * admission state (reserve / release / dispatch permit / calibrate /
     * expired eviction).
     * Captured in Auto-TPM cluster snapshots (after this scheduler's own
     * reserve) and re-checked at plan commit time to detect interference.
     */
    private final AtomicLong admissionVersion = new AtomicLong();

    public DecodeEndpoint(WorkerStatus status) {
        super(status);
        this.reportedKvAvailable.set(status.getAvailableKvCacheTokens().get());
        this.reportedKvTotal.set(status.getTotalKvCacheTokens().get());
        this.requestEvictor = InflightEvictor.withKeyCallback(
                inflightRequests, (requestId, req) -> {
                    removeQueuedPhaseLocked(requestId, req);
                    inflightKvReservedTotal.addAndGet(-req.kvTokens());
                    inflightExpectedKvReservedTotal.addAndGet(-req.expectedKvTokens());
                });
    }

    /** Stop this exact endpoint generation from admitting new engine dispatches. */
    @Override
    public void close() {
        boolean capacityStateChanged = false;
        admissionLock.lock();
        try {
            if (!acceptingEngineDispatchPermits) {
                return;
            }
            acceptingEngineDispatchPermits = false;
            for (EngineDispatchPermitLease lease : engineDispatchPermits.values()) {
                retiredEngineDispatchPermitTokens.add(lease.token());
            }
            if (!engineDispatchPermits.isEmpty()) {
                engineDispatchPermits.clear();
                activeEngineDispatchPermitCount = 0;
                engineDispatchPermitHardKvReservedTotal.set(0);
                engineDispatchPermitExpectedKvReservedTotal.set(0);
            }
            admissionVersion.incrementAndGet();
            capacityStateChanged = true;
        } finally {
            admissionLock.unlock();
        }
        if (capacityStateChanged) {
            notifyEngineDispatchCapacityListeners();
        }
    }

    /** A blocked worker treats retirement as a wakeup and retries for a typed failure. */
    public boolean isRetired() {
        return !acceptingEngineDispatchPermits;
    }

    public void reserve(long requestId, long kvTokens, long expectedKvTokens) {
        reserve(requestId, kvTokens, expectedKvTokens,
                RequestInflight.DEFAULT_PRIORITY);
    }

    /**
     * Shadow-reserve decode capacity for a request, carrying its Auto-TPM
     * priority so the reservation can later be ranked as a decode eviction
     * candidate (design doc 10.1).
     */
    public void reserve(long requestId, long kvTokens, long expectedKvTokens,
                        int priority) {
        admissionLock.lock();
        try {
            reserveLocked(requestId, kvTokens, expectedKvTokens, priority);
        } finally {
            admissionLock.unlock();
        }
    }

    /** Caller holds admissionLock. */
    private void reserveLocked(long requestId,
                               long kvTokens,
                               long expectedKvTokens,
                               int priority) {
        // A re-reserve starts a new request-id generation. Invalidate any
        // pre-delivery capacity owned by the previous immutable reservation.
        removeEngineDispatchPermitLocked(requestId);
        RequestInflight previous = inflightRequests.get(requestId);
        removeQueuedPhaseLocked(requestId, previous);

        RequestInflight newReservation =
                new RequestInflight(kvTokens, expectedKvTokens, priority);
        previous = inflightRequests.put(requestId, newReservation);
        if (previous != null) {
            inflightKvReservedTotal.addAndGet(-previous.kvTokens());
            inflightExpectedKvReservedTotal.addAndGet(-previous.expectedKvTokens());
        }
        inflightKvReservedTotal.addAndGet(kvTokens);
        inflightExpectedKvReservedTotal.addAndGet(expectedKvTokens);
        admissionVersion.incrementAndGet();
    }

    /**
     * Atomically create a reservation in the Master-queued phase. This closes
     * the otherwise observable reserve-then-mark gap while concurrent
     * dispatch callbacks evaluate engine-facing capacity.
     */
    public void reserveQueued(long requestId, long kvTokens,
                              long expectedKvTokens, int priority) {
        admissionLock.lock();
        try {
            reserveLocked(requestId, kvTokens, expectedKvTokens, priority);
            addQueuedPhaseLocked(requestId, inflightRequests.get(requestId));
        } finally {
            admissionLock.unlock();
        }
    }

    public void release(long requestId) {
        boolean capacityChanged;
        admissionLock.lock();
        try {
            capacityChanged = releaseLocked(requestId);
        } finally {
            admissionLock.unlock();
        }
        if (capacityChanged) {
            notifyEngineDispatchCapacityListeners();
        }
    }

    /** Remove every local owner for one request. Caller holds admissionLock. */
    private boolean releaseLocked(long requestId) {
        boolean changed = removeEngineDispatchPermitLocked(requestId);
        changed = clearEngineFenceProtectionLocked(requestId) || changed;
        RequestInflight removed = inflightRequests.remove(requestId);
        changed = removeQueuedPhaseLocked(requestId, removed) || changed;
        if (removed != null) {
            inflightKvReservedTotal.addAndGet(-removed.kvTokens());
            inflightExpectedKvReservedTotal.addAndGet(-removed.expectedKvTokens());
            changed = true;
        }
        if (changed) {
            admissionVersion.incrementAndGet();
        }
        return changed;
    }

    /** Register a stable worker wakeup for Decode dispatch-capacity changes. */
    public void addEngineDispatchCapacityListener(Runnable listener) {
        if (listener != null) {
            engineDispatchCapacityListeners.add(listener);
        }
    }

    /** Remove a worker wakeup when its Prefill batcher shuts down. */
    public void removeEngineDispatchCapacityListener(Runnable listener) {
        if (listener != null) {
            engineDispatchCapacityListeners.remove(listener);
        }
    }

    private void notifyEngineDispatchCapacityListeners() {
        for (Runnable listener : engineDispatchCapacityListeners) {
            try {
                listener.run();
            } catch (Throwable listenerFailure) {
                logger.warn("Decode capacity listener failed", listenerFailure);
            }
        }
    }

    /**
     * Release one shadow reservation only when it is still the exact object
     * observed by a prior admission snapshot.
     *
     * <p>This is the orphan-cleanup counterpart of {@link #reservedView()}.
     * Request IDs can be reused after scheduler ownership is retired, so a
     * cleanup pass must not delete a newer reservation which replaced its
     * snapshot. Generic Engine-fence and priority ownership also take
     * precedence over orphan cleanup.</p>
     *
     * @return {@code true} only when the observed reservation was removed
     */
    public boolean releaseReservationIfCurrent(
            long requestId, RequestInflight expectedReservation) {
        if (expectedReservation == null) {
            return false;
        }
        boolean released = false;
        boolean capacityChanged = false;
        admissionLock.lock();
        try {
            if (inflightRequests.get(requestId) != expectedReservation
                    || engineFenceProtections.containsKey(requestId)
                    || preemptionClaims.containsKey(requestId)) {
                return false;
            }
            removeEngineDispatchPermitLocked(requestId);
            inflightRequests.remove(requestId);
            removeQueuedPhaseLocked(requestId, expectedReservation);
            inflightKvReservedTotal.addAndGet(-expectedReservation.kvTokens());
            inflightExpectedKvReservedTotal.addAndGet(
                    -expectedReservation.expectedKvTokens());
            admissionVersion.incrementAndGet();
            released = true;
            capacityChanged = true;
        } finally {
            admissionLock.unlock();
        }
        if (capacityChanged) {
            notifyEngineDispatchCapacityListeners();
        }
        return released;
    }

    /**
     * Retain the request's current Decode accounting while an EngineFence
     * reconciles ambiguous delivery ownership.
     *
     * <p>The protection is request-scoped and idempotent. It may attach to a
     * shadow reservation, an engine-confirmed entry, or an existing priority
     * claim. A missing request is not materialized: returning {@code false}
     * avoids a stale protection surviving request-id reuse. Callers may hold a
     * scheduler request-entry monitor; this method only acquires
     * {@link #admissionLock} and never calls back into scheduler/batcher code.
     *
     * @return {@code true} when live Decode accounting was protected (including
     *         an already-protected request), otherwise {@code false}
     */
    public boolean beginEngineFenceProtection(long requestId) {
        admissionLock.lock();
        try {
            if (engineFenceProtections.containsKey(requestId)) {
                return true;
            }
            RequestInflight shadow = inflightRequests.get(requestId);
            ConfirmedTask confirmed = trackedConfirmed.get(requestId);
            PreemptionClaim priorityClaim = preemptionClaims.get(requestId);
            boolean priorityOwnsAccounting = isPriorityAccountingOwner(priorityClaim);
            if (shadow == null && confirmed == null && !priorityOwnsAccounting) {
                return false;
            }

            long hardKvTokens = shadow != null ? shadow.kvTokens()
                    : confirmed != null ? confirmed.kvTokens()
                    : priorityClaim.hardKvTokens;
            long expectedKvTokens = shadow != null ? shadow.expectedKvTokens()
                    : priorityOwnsAccounting ? priorityClaim.expectedKvTokens
                    : hardKvTokens;
            boolean confirmedOwner = confirmed != null
                    || (priorityOwnsAccounting
                        && priorityClaim.owner == ClaimOwner.ENGINE_CONFIRMED);
            engineFenceProtections.put(requestId,
                    new EngineFenceProtection(hardKvTokens, expectedKvTokens,
                            confirmedOwner));
            admissionVersion.incrementAndGet();
            return true;
        } finally {
            admissionLock.unlock();
        }
    }

    /**
     * Release only the generic EngineFence owner. An overlapping token-fenced
     * priority owner remains charged independently; WorkerStatus or the exact
     * priority settlement path decides its lifetime.
     *
     * @return {@code true} only when a live generic protection was removed
     */
    public boolean endEngineFenceProtection(long requestId) {
        boolean removed = false;
        boolean capacityChanged = false;
        admissionLock.lock();
        try {
            if (!clearEngineFenceProtectionLocked(requestId)) {
                return false;
            }
            admissionVersion.incrementAndGet();
            removed = true;
            capacityChanged = true;
        } finally {
            admissionLock.unlock();
        }
        if (capacityChanged) {
            notifyEngineDispatchCapacityListeners();
        }
        return removed;
    }

    /**
     * Atomically settle an Engine {@code TOMBSTONED} proof for one request.
     *
     * <p>TOMBSTONED authoritatively proves that the request is absent and that
     * the engine fenced a late enqueue. Under {@link #admissionLock}, this
     * method removes shadow and confirmed ownership, queued membership, and
     * generic EngineFence synthetic slot/KV accounting, then installs a
     * bounded settled tombstone so a delayed WorkerStatus sample cannot
     * recreate the request. Repeated calls are idempotent.
     *
     * <p>An overlapping token-fenced priority owner remains independently
     * token-scoped; callers settle that exact owner before invoking this
     * request-scoped operation.
     *
     * @return {@code true} when this call removed accounting or installed the
     *         tombstone for the first time, otherwise {@code false}
     */
    public boolean settleTombstonedRequest(long requestId) {
        boolean changed;
        boolean capacityChanged;
        admissionLock.lock();
        try {
            changed = settleRequestAccountingLocked(
                    requestId, System.currentTimeMillis(), true);
            if (changed) {
                admissionVersion.incrementAndGet();
            }
            capacityChanged = changed;
        } finally {
            admissionLock.unlock();
        }
        if (capacityChanged) {
            notifyEngineDispatchCapacityListeners();
        }
        return changed;
    }

    // ==================== Auto-TPM decode reserved-only eviction ====================

    /** Result of {@link #tryReleaseVictimsAndReserveIncoming}. */
    public enum ReleaseReserveResult {
        /** All victims released and the incoming request reserved. */
        SUCCESS,
        /** Admission version moved since the plan snapshot; nothing applied. */
        VERSION_MISMATCH,
        /** A victim is no longer a reserved entry; nothing applied. */
        VICTIM_GONE
    }

    /**
     * Atomic decode eviction commit (design doc 11.5/17.2): under the single
     * endpoint admission lock — validate the version, validate every victim is
     * still Master-queued entries, then release all victims
     * and reserve the incoming request. Validate-first guarantees
     * all-or-nothing; on any validation failure nothing is applied.
     *
     * <p>This method only reverses the shadow accounting. Driving each victim
     * to its {@code PRIORITY_PREEMPTED} terminal state (future completion,
     * tombstone) remains the caller's job via
     * {@code InflightRegistrar.finishPreempted*}, whose own decode release is
     * a harmless no-op afterwards ({@link #release} is idempotent).
     */
    public ReleaseReserveResult tryReleaseVictimsAndReserveIncoming(
            List<Long> victimIds,
            long incomingRequestId, long kvTokens, long expectedKvTokens,
            int priority,
            long expectedAdmissionVersion) {
        ReleaseReserveResult result;
        boolean capacityChanged = false;
        admissionLock.lock();
        try {
            if (admissionVersion.get() != expectedAdmissionVersion) {
                return ReleaseReserveResult.VERSION_MISMATCH;
            }
            for (Long victimId : victimIds) {
                RequestInflight victim = inflightRequests.get(victimId);
                if (victim == null || !queuedPhase.contains(victimId)
                        || preemptionClaims.containsKey(victimId)
                        || engineFenceProtections.containsKey(victimId)) {
                    return ReleaseReserveResult.VICTIM_GONE;
                }
            }
            for (Long victimId : victimIds) {
                releaseLocked(victimId);
            }
            reserveLocked(incomingRequestId, kvTokens, expectedKvTokens, priority);
            capacityChanged = true;
            result = ReleaseReserveResult.SUCCESS;
        } finally {
            admissionLock.unlock();
        }
        if (capacityChanged) {
            notifyEngineDispatchCapacityListeners();
        }
        return result;
    }

    /**
     * CAS-style conditional release (redesign N3 §3.4): under the admission
     * lock, release the reservation only if it is still held as a
     * Master-queued shadow entry. Returns {@code false} —
     * touching nothing — when the reservation is gone (already released) or
     * has been folded into the confirmed layer (the engine owns the request;
     * contract 5.3 forbids terminal operations on dispatched requests).
     */
    public boolean releaseIfHeld(long requestId) {
        boolean released = false;
        boolean capacityChanged = false;
        admissionLock.lock();
        try {
            RequestInflight held = inflightRequests.get(requestId);
            if (held == null || !queuedPhase.contains(requestId)
                    || preemptionClaims.containsKey(requestId)
                    || engineFenceProtections.containsKey(requestId)) {
                return false;
            }
            released = releaseLocked(requestId);
            capacityChanged = released;
        } finally {
            admissionLock.unlock();
        }
        if (capacityChanged) {
            notifyEngineDispatchCapacityListeners();
        }
        return released;
    }

    /** Result of {@link #tryReleaseVictimsIfHeldAndReserveIncoming}. */
    public record PresenceEvictionOutcome(boolean success, List<Long> freedVictimIds) {
    }

    /**
     * Presence-guarded decode eviction commit: under the admission
     * lock, conditionally release every victim still holding a
     * Master-queued reservation ({@link #releaseIfHeld}).
     * All victims freed → reserve the incoming request and succeed. Any
     * victim already gone (dispatched / settled) → the freed releases are
     * NOT rolled back — their host requests are driven terminal by the
     * caller — and the commit reports a replan without reserving the
     * incoming. No admission-version check: unrelated reserve / release /
     * calibrate activity no longer aborts the commit.
     */
    public PresenceEvictionOutcome tryReleaseVictimsIfHeldAndReserveIncoming(
            List<Long> victimIds,
            long incomingRequestId, long kvTokens, long expectedKvTokens,
            int priority) {
        PresenceEvictionOutcome outcome;
        boolean capacityChanged;
        admissionLock.lock();
        try {
            List<Long> freed = new ArrayList<>(victimIds.size());
            for (Long victimId : victimIds) {
                RequestInflight held = inflightRequests.get(victimId);
                if (held != null && queuedPhase.contains(victimId)
                        && !preemptionClaims.containsKey(victimId)
                        && !engineFenceProtections.containsKey(victimId)
                        && releaseLocked(victimId)) {
                    freed.add(victimId);
                }
            }
            if (freed.size() < victimIds.size()) {
                outcome = new PresenceEvictionOutcome(false, List.copyOf(freed));
            } else {
                reserveLocked(incomingRequestId, kvTokens, expectedKvTokens, priority);
                outcome = new PresenceEvictionOutcome(true, List.copyOf(freed));
            }
            capacityChanged = !freed.isEmpty();
        } finally {
            admissionLock.unlock();
        }
        if (capacityChanged) {
            notifyEngineDispatchCapacityListeners();
        }
        return outcome;
    }

    /**
     * Point-in-time copy of the reserved (shadow) entries keyed by requestId,
     * for eviction planning snapshots. Confirmed (accepted/running) requests
     * never appear here — calibrate removes them (design doc 10.1).
     *
     * <p>Taken under {@link #admissionLock} so the returned map is a consistent
     * point w.r.t. concurrent mutations.
     */
    public Map<Long, RequestInflight> reservedView() {
        admissionLock.lock();
        try {
            return Map.copyOf(inflightRequests);
        } finally {
            admissionLock.unlock();
        }
    }

    /**
     * Return the exact live reservation for one request without copying the
     * endpoint-wide reservation map. The immutable value identity is used by
     * conditional handoff/cleanup operations to fence request-id reuse.
     */
    public RequestInflight reservationFor(long requestId) {
        return inflightRequests.get(requestId);
    }

    /**
     * Consistent (admissionVersion, reserved, confirmed layers) triple
     * captured atomically under {@link #admissionLock} (Phase 5). Calibrate
     * mutates the layered registry under the same lock and bumps the version,
     * so an unchanged version at commit time covers the layered view too.
     */
    public LayeredAdmissionView layeredAdmissionView() {
        admissionLock.lock();
        try {
            List<ConfirmedTaskView> confirmed = new java.util.ArrayList<>(trackedConfirmed.size());
            trackedConfirmed.forEach((requestId, task) ->
                    confirmed.add(new ConfirmedTaskView(requestId, task.priority(),
                            task.kvTokens(), task.phase(), task.priorityKnown(),
                            preemptionClaims.containsKey(requestId)
                                    || engineFenceProtections.containsKey(requestId))));
            Set<Long> claimed = Set.copyOf(preemptionClaims.keySet());
            if (!engineFenceProtections.isEmpty()) {
                Set<Long> combined = new HashSet<>(claimed);
                combined.addAll(engineFenceProtections.keySet());
                claimed = Set.copyOf(combined);
            }
            return new LayeredAdmissionView(admissionVersion.get(),
                    Map.copyOf(inflightRequests), List.copyOf(confirmed),
                    java.util.Set.copyOf(queuedPhase),
                    claimed);
        } finally {
            admissionLock.unlock();
        }
    }

    /** Atomic (admissionVersion, reserved, confirmed, queued) tuple — see {@link #layeredAdmissionView()}. */
    public record LayeredAdmissionView(long admissionVersion,
                                       Map<Long, RequestInflight> reserved,
                                       List<ConfirmedTaskView> confirmed,
                                       java.util.Set<Long> queued,
                                       Set<Long> claimed) {
    }

    /** Immutable point-in-time view of one layered-registry entry. */
    public record ConfirmedTaskView(long requestId,
                                    int priority,
                                    long kvTokens,
                                    DecodeTaskPhase phase,
                                    boolean priorityKnown,
                                    boolean claimedForPreemption) {
    }

    // ==================== Priority-preemption transaction ====================

    public enum PreemptionBeginResult {
        SUCCESS,
        VERSION_MISMATCH,
        VICTIM_GONE,
        VICTIM_ALREADY_CLAIMED,
        INVALID_PRIORITY,
        INCOMING_ALREADY_RESERVED
    }

    /**
     * Atomically claim Engine-visible victims and reserve the incoming demand
     * as provisional capacity.  Victim accounting is intentionally untouched:
     * Cancel ACCEPTED is only an intent acknowledgement.
     */
    public PreemptionBeginResult beginPriorityPreemption(
            long attemptToken,
            List<Long> victimIds,
            long incomingRequestId,
            long incomingKvTokens,
            long incomingExpectedKvTokens,
            int incomingPriority,
            long expectedAdmissionVersion,
            boolean requireVersionMatch) {
        admissionLock.lock();
        try {
            if (attemptToken <= 0 || victimIds == null || victimIds.isEmpty()) {
                throw new IllegalArgumentException("attempt token and victims are required");
            }
            if (requireVersionMatch && admissionVersion.get() != expectedAdmissionVersion) {
                return PreemptionBeginResult.VERSION_MISMATCH;
            }
            if (inflightRequests.containsKey(incomingRequestId)
                    || trackedConfirmed.containsKey(incomingRequestId)
                    || engineFenceProtections.containsKey(incomingRequestId)) {
                return PreemptionBeginResult.INCOMING_ALREADY_RESERVED;
            }

            Map<Long, ClaimOwner> owners = new HashMap<>();
            for (Long victimId : victimIds) {
                if (preemptionClaims.containsKey(victimId)
                        || engineFenceProtections.containsKey(victimId)) {
                    return PreemptionBeginResult.VICTIM_ALREADY_CLAIMED;
                }
                RequestInflight shadow = inflightRequests.get(victimId);
                ConfirmedTask confirmed = trackedConfirmed.get(victimId);
                if (shadow != null && !queuedPhase.contains(victimId)) {
                    if (shadow.priority() <= 0 || shadow.priority() >= incomingPriority) {
                        return PreemptionBeginResult.INVALID_PRIORITY;
                    }
                    owners.put(victimId, ClaimOwner.SHADOW_IN_FLIGHT);
                } else if (confirmed != null && confirmed.phase().isEngineConfirmed()) {
                    if (confirmed.priority() <= 0 || confirmed.priority() >= incomingPriority) {
                        return PreemptionBeginResult.INVALID_PRIORITY;
                    }
                    owners.put(victimId, ClaimOwner.ENGINE_CONFIRMED);
                } else {
                    return PreemptionBeginResult.VICTIM_GONE;
                }
            }

            // Provisional incoming ownership closes the free-pool race while
            // Cancel runs.  It is not visible to the prefill queue yet.
            reserveLocked(incomingRequestId, incomingKvTokens, incomingExpectedKvTokens,
                    incomingPriority);
            for (Map.Entry<Long, ClaimOwner> entry : owners.entrySet()) {
                // Normally impossible because an acquired permit keeps its
                // reservation queued and therefore ineligible above. Keep the
                // cleanup local to the ownership transition as a hard invariant.
                removeEngineDispatchPermitLocked(entry.getKey());
                RequestInflight shadow = inflightRequests.get(entry.getKey());
                ConfirmedTask confirmed = trackedConfirmed.get(entry.getKey());
                long hardKv = shadow != null ? shadow.kvTokens()
                        : confirmed != null ? confirmed.kvTokens() : 0;
                long expectedKv = shadow != null ? shadow.expectedKvTokens() : hardKv;
                preemptionClaims.put(entry.getKey(),
                        new PreemptionClaim(attemptToken, entry.getValue(), hardKv, expectedKv));
            }
            preemptionAttempts.put(attemptToken,
                    new EndpointPreemptionAttempt(incomingRequestId, Set.copyOf(victimIds)));
            admissionVersion.incrementAndGet();
            return PreemptionBeginResult.SUCCESS;
        } finally {
            admissionLock.unlock();
        }
    }

    /** CLAIMED -> CANCEL_IN_FLIGHT; must complete before any outbound RPC. */
    public boolean markPriorityCancelInFlight(long attemptToken) {
        admissionLock.lock();
        try {
            EndpointPreemptionAttempt attempt = preemptionAttempts.get(attemptToken);
            if (attempt == null) {
                return false;
            }
            for (Long victimId : attempt.victimIds) {
                PreemptionClaim claim = preemptionClaims.get(victimId);
                if (claim == null || claim.attemptToken != attemptToken
                        || claim.state != ClaimState.CLAIMED) {
                    return false;
                }
            }
            for (Long victimId : attempt.victimIds) {
                preemptionClaims.get(victimId).state = ClaimState.CANCEL_IN_FLIGHT;
            }
            admissionVersion.incrementAndGet();
            return true;
        } finally {
            admissionLock.unlock();
        }
    }

    /** Cancel ACCEPTED: retain every byte/slot and only advance control state. */
    public boolean markPriorityCancelAccepted(long attemptToken, long requestId) {
        return transitionClaim(attemptToken, requestId,
                ClaimState.CANCEL_IN_FLIGHT, ClaimState.CANCEL_REQUESTED);
    }

    public boolean markPriorityCancelNotFound(long attemptToken, long requestId) {
        return transitionClaim(attemptToken, requestId,
                ClaimState.CANCEL_IN_FLIGHT, ClaimState.NOT_FOUND_STALE);
    }

    public boolean markPriorityCancelUnknown(long attemptToken, long requestId) {
        admissionLock.lock();
        try {
            PreemptionClaim claim = preemptionClaims.get(requestId);
            if (claim == null || claim.attemptToken != attemptToken
                    || (claim.state != ClaimState.CANCEL_IN_FLIGHT
                        && claim.state != ClaimState.CANCEL_REQUESTED)) {
                return false;
            }
            claim.state = ClaimState.CANCEL_UNKNOWN;
            admissionVersion.incrementAndGet();
            return true;
        } finally {
            admissionLock.unlock();
        }
    }

    /**
     * Typed Prefill CANCELED settlement.  This is the sole transition that
     * deletes victim accounting after an accepted or transport-unknown
     * Cancel; duplicate observations are a token-fenced no-op.
     */
    public boolean settlePriorityCanceled(long attemptToken, long requestId) {
        boolean settled = false;
        boolean capacityChanged = false;
        admissionLock.lock();
        try {
            PreemptionClaim claim = preemptionClaims.get(requestId);
            if (claim == null || claim.attemptToken != attemptToken
                    || (claim.state != ClaimState.CANCEL_REQUESTED
                        && claim.state != ClaimState.CANCEL_UNKNOWN)) {
                return false;
            }
            settled = settlePriorityClaimLocked(attemptToken, requestId, claim);
            capacityChanged = settled;
        } finally {
            admissionLock.unlock();
        }
        if (capacityChanged) {
            notifyEngineDispatchCapacityListeners();
        }
        return settled;
    }

    /**
     * Settle an engine {@code TOMBSTONED} acknowledgement.
     *
     * <p>TOMBSTONED is stronger than NOT_FOUND: the addressed request was
     * absent and the engine atomically installed a late-enqueue fence. It is
     * therefore an authoritative terminal proof and may release the same
     * accounting as typed CANCELED without waiting for WorkerStatus.</p>
     */
    public boolean settlePriorityTombstoned(long attemptToken, long requestId) {
        boolean settled = false;
        boolean capacityChanged = false;
        admissionLock.lock();
        try {
            PreemptionClaim claim = preemptionClaims.get(requestId);
            if (claim == null || claim.attemptToken != attemptToken
                    || (claim.state != ClaimState.CANCEL_IN_FLIGHT
                        && claim.state != ClaimState.NOT_FOUND_STALE
                        && claim.state != ClaimState.CANCEL_UNKNOWN)) {
                return false;
            }
            settled = settlePriorityClaimLocked(attemptToken, requestId, claim);
            capacityChanged = settled;
        } finally {
            admissionLock.unlock();
        }
        if (capacityChanged) {
            notifyEngineDispatchCapacityListeners();
        }
        return settled;
    }

    /**
     * Transfer a NOT_FOUND priority claim to the scheduler's post-delivery
     * Engine fence without releasing any victim accounting.
     *
     * <p>The original attempt token becomes the exact endpoint-side fence
     * generation. In particular, {@code kvHeldAfterWorkerRelease} remains
     * charged when Decode has already stopped reporting an ENGINE_CONFIRMED
     * victim; dropping that synthetic hold here would oversell KV before the
     * new Cancel observes TOMBSTONED or another authoritative terminal.</p>
     */
    public boolean transferPriorityNotFoundClaimToEngineFence(
            long attemptToken,
            long requestId) {
        admissionLock.lock();
        try {
            PreemptionClaim claim = preemptionClaims.get(requestId);
            if (claim == null || claim.attemptToken != attemptToken
                    || claim.state != ClaimState.NOT_FOUND_STALE) {
                return false;
            }
            claim.state = ClaimState.ENGINE_FENCE;
            admissionVersion.incrementAndGet();
            return true;
        } finally {
            admissionLock.unlock();
        }
    }

    /** Authoritative terminal settlement for an exact transferred fence generation. */
    public boolean settleEngineFenceClaim(long attemptToken, long requestId) {
        boolean settled = false;
        boolean capacityChanged = false;
        admissionLock.lock();
        try {
            PreemptionClaim claim = preemptionClaims.get(requestId);
            if (claim == null || claim.attemptToken != attemptToken
                        || claim.state != ClaimState.ENGINE_FENCE) {
                return false;
            }
            settled = settlePriorityClaimLocked(attemptToken, requestId, claim);
            capacityChanged = settled;
        } finally {
            admissionLock.unlock();
        }
        if (capacityChanged) {
            notifyEngineDispatchCapacityListeners();
        }
        return settled;
    }

    /**
     * A fresh Decode-active observation before the transferred fence invokes
     * Cancel proves ordinary ownership. Drop only the control/synthetic hold;
     * the live confirmed or shadow accounting remains in its original layer.
     */
    public boolean releaseEngineFenceClaimActive(long attemptToken, long requestId) {
        boolean released = false;
        admissionLock.lock();
        try {
            PreemptionClaim claim = preemptionClaims.get(requestId);
            if (claim == null || claim.attemptToken != attemptToken
                    || claim.state != ClaimState.ENGINE_FENCE) {
                return false;
            }
            releaseHeldKv(claim);
            preemptionClaims.remove(requestId);
            admissionVersion.incrementAndGet();
            released = true;
        } finally {
            admissionLock.unlock();
        }
        if (released) {
            notifyEngineDispatchCapacityListeners();
        }
        return released;
    }

    /** Called with {@link #admissionLock} held. */
    private boolean settlePriorityClaimLocked(long attemptToken,
                                              long requestId,
                                              PreemptionClaim claim) {
        boolean genericFenceRetainsAccounting =
                transferPriorityAccountingToEngineFenceLocked(requestId, claim);
        if (claim.owner == ClaimOwner.SHADOW_IN_FLIGHT) {
            removeEngineDispatchPermitLocked(requestId);
            RequestInflight removed = inflightRequests.remove(requestId);
            if (removed != null) {
                inflightKvReservedTotal.addAndGet(-removed.kvTokens());
                inflightExpectedKvReservedTotal.addAndGet(-removed.expectedKvTokens());
            }
            removeQueuedPhaseLocked(requestId, removed);
        } else {
            trackedConfirmed.remove(requestId);
            if (!genericFenceRetainsAccounting) {
                confirmedRunningCount = Math.max(0, confirmedRunningCount - 1);
            }
        }
        releaseHeldKv(claim);
        claim.state = ClaimState.CANCELED_SETTLED;
        rememberSettledLocked(requestId, System.currentTimeMillis());
        // A proof can arrive after the incoming attempt already timed out and
        // released its provisional reservation. In that case no commit/abort
        // pass remains to discard the claim.
        if (!preemptionAttempts.containsKey(attemptToken)) {
            preemptionClaims.remove(requestId);
        }
        admissionVersion.incrementAndGet();
        return true;
    }

    /** Mark the full incoming reservation committed after every victim settles. */
    public boolean commitPriorityPreemption(long attemptToken) {
        admissionLock.lock();
        try {
            EndpointPreemptionAttempt attempt = preemptionAttempts.get(attemptToken);
            if (attempt == null || !inflightRequests.containsKey(attempt.incomingRequestId)) {
                return false;
            }
            for (Long victimId : attempt.victimIds) {
                PreemptionClaim claim = preemptionClaims.get(victimId);
                if (claim == null || claim.attemptToken != attemptToken
                        || claim.state != ClaimState.CANCELED_SETTLED) {
                    return false;
                }
            }
            for (Long victimId : attempt.victimIds) {
                preemptionClaims.remove(victimId);
            }
            preemptionAttempts.remove(attemptToken);
            admissionVersion.incrementAndGet();
            return true;
        } finally {
            admissionLock.unlock();
        }
    }

    /**
     * Abort the incoming attempt.  Provisional incoming ownership is released;
     * successfully canceled victims become ordinary free capacity, while
     * NOT_FOUND/unknown claims retain their accounting and reconciliation
     * fence.
     */
    public void abortPriorityPreemption(long attemptToken) {
        boolean capacityChanged = false;
        admissionLock.lock();
        try {
            EndpointPreemptionAttempt attempt = preemptionAttempts.remove(attemptToken);
            if (attempt == null) {
                return;
            }
            releaseLocked(attempt.incomingRequestId);
            for (Long victimId : attempt.victimIds) {
                PreemptionClaim claim = preemptionClaims.get(victimId);
                if (claim == null || claim.attemptToken != attemptToken) {
                    continue;
                }
                if (claim.state == ClaimState.CLAIMED
                        || claim.state == ClaimState.CANCEL_IN_FLIGHT
                        || claim.state == ClaimState.CANCELED_SETTLED) {
                    preemptionClaims.remove(victimId);
                }
            }
            admissionVersion.incrementAndGet();
            capacityChanged = true;
        } finally {
            admissionLock.unlock();
        }
        if (capacityChanged) {
            notifyEngineDispatchCapacityListeners();
        }
    }

    /** Fresh active status is the only path that reopens a NOT_FOUND_STALE victim. */
    public boolean reconcilePriorityVictimActive(long requestId) {
        boolean reconciled = false;
        admissionLock.lock();
        try {
            PreemptionClaim claim = preemptionClaims.get(requestId);
            if (claim == null || claim.state != ClaimState.NOT_FOUND_STALE) {
                return false;
            }
            releaseHeldKv(claim);
            preemptionClaims.remove(requestId);
            admissionVersion.incrementAndGet();
            reconciled = true;
        } finally {
            admissionLock.unlock();
        }
        if (reconciled) {
            notifyEngineDispatchCapacityListeners();
        }
        return reconciled;
    }

    /**
     * Reconcile a one-shot ordinary Decode terminal after Cancel NOT_FOUND or
     * a transport-unknown ACK. Neither outcome is typed priority completion,
     * so the ordinary terminal resumes the pre-existing completion path.
     */
    public boolean reconcilePriorityVictimFinished(long requestId) {
        boolean reconciled = false;
        boolean capacityChanged = false;
        admissionLock.lock();
        try {
            PreemptionClaim claim = preemptionClaims.get(requestId);
            if (claim == null || (claim.state != ClaimState.NOT_FOUND_STALE
                        && claim.state != ClaimState.CANCEL_UNKNOWN)) {
                return false;
            }
            // An ordinary finished sample is authoritative. Unlike an
            // ambiguous priority ACK, it must clear the generic fence rather
            // than transfer accounting into it.
            releaseHeldKv(claim);
            preemptionClaims.remove(requestId);
            settleRequestAccountingLocked(requestId, System.currentTimeMillis(), false);
            admissionVersion.incrementAndGet();
            reconciled = true;
            capacityChanged = true;
        } finally {
            admissionLock.unlock();
        }
        if (capacityChanged) {
            notifyEngineDispatchCapacityListeners();
        }
        return reconciled;
    }

    private boolean transitionClaim(long attemptToken, long requestId,
                                    ClaimState expected, ClaimState next) {
        admissionLock.lock();
        try {
            PreemptionClaim claim = preemptionClaims.get(requestId);
            if (claim == null || claim.attemptToken != attemptToken || claim.state != expected) {
                return false;
            }
            claim.state = next;
            admissionVersion.incrementAndGet();
            return true;
        } finally {
            admissionLock.unlock();
        }
    }

    /** Accepted-not-running layer size (Phase 5 gauge). */
    public int getAcceptedLayerCount() {
        int count = 0;
        for (ConfirmedTask task : trackedConfirmed.values()) {
            if (task.phase() == DecodeTaskPhase.ACCEPTED_NOT_RUNNING) {
                count++;
            }
        }
        return count;
    }

    /** True running layer size (Phase 5 gauge). */
    public int getRunningLayerCount() {
        int count = 0;
        for (ConfirmedTask task : trackedConfirmed.values()) {
            if (task.phase() == DecodeTaskPhase.RUNNING) {
                count++;
            }
        }
        return count;
    }

    /** Current admission version for Auto-TPM optimistic plan validation. */
    public long admissionVersion() {
        return admissionVersion.get();
    }

    @Override
    public void onWorkerStatusUpdate(WorkerStatus ws, WorkerStatusResponse resp) {
        super.onWorkerStatusUpdate(ws, resp);
        calibrate(resp.getRunningTaskInfo(), resp.getFinishedTaskInfo());
    }

    /**
     * Full calibration against worker status report.
     */
    private void calibrate(Map<String, TaskInfo> runningTaskInfo, Map<String, TaskInfo> finishedTaskInfo) {
        admissionLock.lock();
        try {
            doCalibrate(runningTaskInfo, finishedTaskInfo);
        } finally {
            admissionLock.unlock();
        }
        // KV availability and request ownership can change without changing
        // the aggregate concurrency count. Publish every authoritative Decode
        // status transition so a blocked lock-free waiter re-evaluates both gates.
        notifyEngineDispatchCapacityListeners();
    }

    private void doCalibrate(Map<String, TaskInfo> runningTaskInfo, Map<String, TaskInfo> finishedTaskInfo) {
        this.reportedKvAvailable.set(status.getAvailableKvCacheTokens().get());
        this.reportedKvTotal.set(status.getTotalKvCacheTokens().get());
        admissionVersion.incrementAndGet();

        // Build one authoritative Decode view.  Claimed victims that disappear
        // are held synthetically until the original Prefill publishes typed
        // CANCELED; generic Decode absence/finished must not release them.
        Set<Long> confirmedNow = new HashSet<>();
        int actualConfirmed = 0;
        long now = System.currentTimeMillis();
        if (runningTaskInfo != null) {
            for (TaskInfo task : runningTaskInfo.values()) {
                TaskPhase phase = task.getPhase();
                long requestId = task.getRequestId();
                if ((phase == TaskPhase.KV_ALLOCATED || phase == TaskPhase.RUNNING)
                        && !settledTombstones.containsKey(requestId)) {
                    actualConfirmed++;
                    RequestInflight removed = inflightRequests.remove(requestId);
                    removeEngineDispatchPermitLocked(requestId);
                    if (removed != null) {
                        removeQueuedPhaseLocked(requestId, removed);
                        inflightKvReservedTotal.addAndGet(-removed.kvTokens());
                        inflightExpectedKvReservedTotal.addAndGet(-removed.expectedKvTokens());
                    }
                    PreemptionClaim claim = preemptionClaims.get(requestId);
                    if (claim != null
                            && claim.state == ClaimState.NOT_FOUND_STALE
                            && !preemptionAttempts.containsKey(claim.attemptToken)) {
                        releaseHeldKv(claim);
                        preemptionClaims.remove(requestId);
                        claim = null;
                    }
                    if (claim != null) {
                        claim.owner = ClaimOwner.ENGINE_CONFIRMED;
                        releaseHeldKv(claim);
                    }
                    confirmedNow.add(requestId);
                    trackConfirmed(task, phase, removed, now);
                    observeEngineFenceConfirmedLocked(requestId, removed);
                }
            }
        }

        int syntheticallyHeldSlots = 0;
        for (Map.Entry<Long, PreemptionClaim> entry : preemptionClaims.entrySet()) {
            PreemptionClaim claim = entry.getValue();
            if (claim.owner == ClaimOwner.ENGINE_CONFIRMED
                    && claim.state != ClaimState.CANCELED_SETTLED
                    && !confirmedNow.contains(entry.getKey())) {
                syntheticallyHeldSlots++;
                holdReleasedKv(claim);
            }
        }

        // Preserve one union accounting owner for a confirmed request that
        // temporarily vanished. Priority claims take precedence while live;
        // the generic fence takes over only when no priority owner exists.
        // This replaces the prior removeIf traversal, so the generic protection
        // adds no extra per-status collection or wrapper allocation.
        java.util.Iterator<Map.Entry<Long, ConfirmedTask>> confirmedIt =
                trackedConfirmed.entrySet().iterator();
        while (confirmedIt.hasNext()) {
            Map.Entry<Long, ConfirmedTask> entry = confirmedIt.next();
            long requestId = entry.getKey();
            if (confirmedNow.contains(requestId)) {
                continue;
            }
            EngineFenceProtection protection = engineFenceProtections.get(requestId);
            PreemptionClaim priorityClaim = preemptionClaims.get(requestId);
            if (isPriorityAccountingOwner(priorityClaim)) {
                if (protection != null) {
                    releaseEngineFenceSyntheticHoldLocked(protection, true);
                }
                continue;
            }
            if (protection != null) {
                protection.confirmedOwner = true;
                ensureEngineFenceSyntheticHoldLocked(protection, true);
                continue;
            }
            confirmedIt.remove();
        }
        this.confirmedRunningCount = actualConfirmed + syntheticallyHeldSlots
                + engineFenceHeldSlotCount;

        if (finishedTaskInfo != null) {
            for (TaskInfo task : finishedTaskInfo.values()) {
                long requestId = task.getRequestId();
                if (settledTombstones.containsKey(requestId)
                        || preemptionClaims.containsKey(requestId)) {
                    continue;
                }
                confirmedNow.remove(requestId);
                settleRequestAccountingLocked(requestId, now, false);
            }
        }

    }

    /**
     * Register / refresh one engine-confirmed task in the layered registry:
     * {@code KV_ALLOCATED} → accepted layer, {@code RUNNING} → running layer.
     * Priority is inherited from the shadow entry removed this round; when
     * the WorkerStatus report precedes the reserve (or the shadow entry
     * expired) it is unknown here and falls back to the default. KV is approximated by
     * {@code TaskInfo.inputLength} — the engine does not report per-request
     * KV usage, so 0 stays 0.
     */
    private void trackConfirmed(TaskInfo task, TaskPhase phase, RequestInflight removed, long now) {
        DecodeTaskPhase layer = phase == TaskPhase.KV_ALLOCATED
                ? DecodeTaskPhase.ACCEPTED_NOT_RUNNING
                : DecodeTaskPhase.RUNNING;
        ConfirmedTask tracked = trackedConfirmed.get(task.getRequestId());
        boolean priorityKnown = removed != null;
        if (tracked == null || (!tracked.priorityKnown() && priorityKnown)) {
            int priority = removed != null ? removed.priority() : RequestInflight.DEFAULT_PRIORITY;
            long kvTokens = Math.max(0, task.getInputLength());
            trackedConfirmed.put(task.getRequestId(),
                    new ConfirmedTask(priority, kvTokens, layer, now, priorityKnown));
        } else {
            tracked.refresh(layer, now);
        }
    }

    /**
     * Remove every ordinary Decode accounting layer for one authoritative
     * terminal while {@link #admissionLock} is held.
     *
     * <p>A missing-confirmed generic fence deliberately retains the metadata
     * entry and represents its one slot synthetically. Removing both layers
     * therefore decrements {@link #confirmedRunningCount} exactly once: via
     * the synthetic release when present, otherwise via the confirmed entry.
     */
    private boolean settleRequestAccountingLocked(long requestId,
                                                   long settledAtMs,
                                                   boolean explicitTombstoneProof) {
        boolean changed = removeEngineDispatchPermitLocked(requestId);

        EngineFenceProtection protection = engineFenceProtections.remove(requestId);
        boolean syntheticSlotRemoved = protection != null && protection.syntheticHeld;
        if (protection != null) {
            releaseEngineFenceSyntheticHoldLocked(protection);
            changed = true;
        }

        ConfirmedTask confirmed = trackedConfirmed.remove(requestId);
        if (confirmed != null) {
            if (!syntheticSlotRemoved) {
                confirmedRunningCount = Math.max(0, confirmedRunningCount - 1);
            }
            changed = true;
        }

        RequestInflight shadow = inflightRequests.remove(requestId);
        if (shadow != null) {
            inflightKvReservedTotal.addAndGet(-shadow.kvTokens());
            inflightExpectedKvReservedTotal.addAndGet(-shadow.expectedKvTokens());
            changed = true;
        }
        if (removeQueuedPhaseLocked(requestId, shadow)) {
            changed = true;
        }
        // A generic EngineFence and an explicit TOMBSTONED acknowledgement both
        // close an ambiguous generation permanently. Keep their small retained
        // set fenced from a delayed active WorkerStatus sample. A plain finished
        // sample, however, is already ordered by status calibration and must not
        // allocate one boxed HashMap entry per completed request for the full TTL.
        if (explicitTombstoneProof || protection != null) {
            changed = rememberSettledLocked(requestId, settledAtMs) || changed;
        }
        return changed;
    }

    /** Publish one non-refreshing terminal fence while admissionLock is held. */
    private boolean rememberSettledLocked(long requestId, long settledAtMs) {
        return settledTombstones.putIfAbsent(requestId, settledAtMs) == null;
    }

    /** Package-private resource-retention probe for deterministic tests. */
    int settledTombstoneCountForTest() {
        admissionLock.lock();
        try {
            return settledTombstones.size();
        } finally {
            admissionLock.unlock();
        }
    }

    /** Called with {@link #admissionLock} held after a fresh active observation. */
    private void observeEngineFenceConfirmedLocked(long requestId, RequestInflight removed) {
        EngineFenceProtection protection = engineFenceProtections.get(requestId);
        if (protection == null) {
            return;
        }
        ConfirmedTask confirmed = trackedConfirmed.get(requestId);
        if (removed != null) {
            retainEngineFenceDemandLocked(
                    protection, removed.kvTokens(), removed.expectedKvTokens());
        }
        if (confirmed != null) {
            retainEngineFenceDemandLocked(
                    protection, confirmed.kvTokens(), confirmed.kvTokens());
        }
        protection.confirmedOwner = true;
        // The fresh engine report owns both slot and KV again. Any prior
        // generic synthetic owner must disappear before the aggregate count is
        // recomputed for this status round.
        releaseEngineFenceSyntheticHoldLocked(protection, true);
    }

    /**
     * Install the generic synthetic owner without replacing an existing slot
     * count. When {@code slotAlreadyCounted} is false this is a shadow-to-
     * synthetic transfer, so move the slot into confirmedRunningCount before
     * the shadow map entry is removed.
     */
    private void ensureEngineFenceSyntheticHoldLocked(
            EngineFenceProtection protection,
            boolean slotAlreadyCounted) {
        if (protection.syntheticHeld) {
            return;
        }
        engineFenceHeldKv.addAndGet(protection.hardKvTokens);
        engineFenceHeldExpectedKv.addAndGet(protection.expectedKvTokens);
        engineFenceHeldSlotCount++;
        protection.syntheticHeld = true;
        if (!slotAlreadyCounted) {
            confirmedRunningCount++;
        }
    }

    /** Grow a conservative demand estimate without corrupting a live hold. */
    private void retainEngineFenceDemandLocked(
            EngineFenceProtection protection,
            long hardKvTokens,
            long expectedKvTokens) {
        long previousHardKv = protection.hardKvTokens;
        long previousExpectedKv = protection.expectedKvTokens;
        protection.retainAtLeast(hardKvTokens, expectedKvTokens);
        if (protection.syntheticHeld) {
            engineFenceHeldKv.addAndGet(protection.hardKvTokens - previousHardKv);
            engineFenceHeldExpectedKv.addAndGet(
                    protection.expectedKvTokens - previousExpectedKv);
        }
    }

    /** Called with {@link #admissionLock} held. */
    private void releaseEngineFenceSyntheticHoldLocked(EngineFenceProtection protection) {
        releaseEngineFenceSyntheticHoldLocked(protection, false);
    }

    /**
     * Drop a synthetic hold. Calibration passes {@code preservePublishedSlot}
     * when a fresh engine/priority owner replaces the slot later in the same
     * critical section; keeping the old volatile count published until the
     * final aggregate assignment prevents lock-free routing from observing a
     * transient zero-load gap.
     */
    private void releaseEngineFenceSyntheticHoldLocked(
            EngineFenceProtection protection,
            boolean preservePublishedSlot) {
        if (!protection.syntheticHeld) {
            return;
        }
        engineFenceHeldKv.addAndGet(-protection.hardKvTokens);
        engineFenceHeldExpectedKv.addAndGet(-protection.expectedKvTokens);
        engineFenceHeldSlotCount--;
        if (!preservePublishedSlot) {
            confirmedRunningCount = Math.max(0, confirmedRunningCount - 1);
        }
        protection.syntheticHeld = false;
    }

    /** Called with {@link #admissionLock} held. */
    private boolean clearEngineFenceProtectionLocked(long requestId) {
        EngineFenceProtection protection = engineFenceProtections.remove(requestId);
        if (protection == null) {
            return false;
        }
        releaseEngineFenceSyntheticHoldLocked(protection);
        return true;
    }

    /**
     * Move the effective hold from a priority claim to an overlapping generic
     * fence. Both owners remain logically independent, but the union contributes
     * exactly one slot/KV hold after this admission-lock critical section.
     */
    private boolean transferPriorityAccountingToEngineFenceLocked(
            long requestId,
            PreemptionClaim claim) {
        EngineFenceProtection protection = engineFenceProtections.get(requestId);
        if (protection == null) {
            return false;
        }
        protection.confirmedOwner = true;
        retainEngineFenceDemandLocked(
                protection, claim.hardKvTokens, claim.expectedKvTokens);
        ensureEngineFenceSyntheticHoldLocked(
                protection, claim.owner == ClaimOwner.ENGINE_CONFIRMED);
        return true;
    }

    private static boolean isPriorityAccountingOwner(PreemptionClaim claim) {
        return claim != null && claim.state != ClaimState.CANCELED_SETTLED;
    }

    private void holdReleasedKv(PreemptionClaim claim) {
        if (!claim.kvHeldAfterWorkerRelease) {
            priorityPreemptionHeldKv.addAndGet(claim.hardKvTokens);
            claim.kvHeldAfterWorkerRelease = true;
        }
    }

    private void releaseHeldKv(PreemptionClaim claim) {
        if (claim.kvHeldAfterWorkerRelease) {
            priorityPreemptionHeldKv.addAndGet(-claim.hardKvTokens);
            claim.kvHeldAfterWorkerRelease = false;
        }
    }

    // ==================== KV Cache 三视图 ====================

    /**
     * Local inflight KV reservation (conservative estimate) not yet confirmed by the engine.
     * Sums {@code expectedKvTokens} (seqLen + maxNewTokens) to account for generation-phase
     * KV growth. Used for scoring / load balancing.
     * Backed by {@code inflightExpectedKvReservedTotal} counter — O(1) incremental maintenance.
     */
    private long inflightKvReserved() {
        return inflightExpectedKvReservedTotal.get();
    }

    /**
     * Local inflight KV reservation (hard demand) not yet confirmed by the engine.
     * Sums {@code kvTokens} (seqLen only) — the minimum KV needed for the prompt itself.
     * Used for hard-capacity filtering to ensure the prompt fits.
     * Backed by {@code inflightKvReservedTotal} counter — O(1) incremental maintenance.
     */
    public long inflightHardKvReserved() {
        return inflightKvReservedTotal.get();
    }

    /**
     * Local inflight expected KV reservation total (seqLen + maxNewTokens per
     * entry) — exposed for the Auto-TPM decode admission snapshot (10.2).
     */
    public long inflightExpectedKvReserved() {
        return inflightExpectedKvReservedTotal.get();
    }

    /**
     * Real KV used: engine-reported used (total - available), local inflight
     * reservations, and expected demand retained by a synthetic EngineFence owner.
     */
    public long realKvUsed() {
        long totalCap = status.getTotalKvCacheTokens().get();
        long avail = status.getAvailableKvCacheTokens().get();
        long reportedUsed = totalCap > 0 ? Math.max(0, totalCap - avail) : 0;
        return reportedUsed + inflightKvReserved() + engineFenceHeldExpectedKv.get();
    }

    /**
     * KV demand which may reach the engine now. Reservations parked in a
     * Prefill queue are soft placement hints: charging all of them against the
     * hard availability gate makes a long scheduler queue report every Decode
     * worker unavailable even though none of that work has been dispatched.
     */
    public long engineFacingKvUsed() {
        long totalCap = reportedKvTotal.get();
        long avail = reportedKvAvailable.get();
        long reportedUsed = totalCap > 0 ? Math.max(0, totalCap - avail) : 0;
        long localEngineFacing = Math.max(0L,
                inflightKvReserved() - queuedExpectedKvReservedTotal.get())
                + engineDispatchPermitExpectedKvReservedTotal.get();
        return saturatedAddNonNegative(
                saturatedAddNonNegative(reportedUsed, localEngineFacing),
                engineFenceHeldExpectedKv.get());
    }

    /**
     * Real KV available: engine-reported available minus local shadow,
     * priority-fence, and generic EngineFence hard reservations.
     *
     * <p>Uses {@link #inflightHardKvReserved()} (prompt-only KV) rather than
     * {@link #inflightKvReserved()} (expected KV with generation) so that the
     * hard-capacity filter only checks whether the prompt itself fits, without
     * being overly aggressive due to other inflight requests' expected growth.
     *
     * <p><b>Approximate:</b> reads {@code reportedKvAvailable} and
     * computes {@code inflightHardKvReserved()} non-atomically — the returned value may reflect a
     * slightly inconsistent snapshot. This is acceptable for scheduling decisions.
     */
    public long realKvAvailable() {
        return Math.max(0, reportedKvAvailable.get()
                - inflightHardKvReserved()
                - priorityPreemptionHeldKv.get()
                - engineFenceHeldKv.get());
    }

    /** Hard prompt KV available to the next engine dispatch, excluding soft queued holds. */
    public long engineFacingKvAvailable() {
        long localEngineFacing = Math.max(0L,
                inflightHardKvReserved() - queuedHardKvReservedTotal.get())
                + engineDispatchPermitHardKvReservedTotal.get();
        return Math.max(0, reportedKvAvailable.get()
                - localEngineFacing
                - priorityPreemptionHeldKv.get()
                - engineFenceHeldKv.get());
    }

    // ==================== Metrics ====================

    /**
     * Report per-worker decode inflight metrics via the given reporter.
     * Called periodically by {@link org.flexlb.balance.scheduler.PriorityScheduler}.
     */
    public void reportBatchMetrics(BatchSchedulerReporter reporter) {
        reporter.reportInflightRequestCount(RoleType.DECODE.name(), getIp(), getInflightCount());
        reporter.reportDecodeTotalLoad(getIp(), getTotalLoad());
        reporter.reportDecodeInflightKvReserved(getIp(), inflightKvReserved());
        reporter.reportDecodeInflightHardKvReserved(getIp(), inflightHardKvReserved());
        reporter.reportInflightMaxAgeMs(RoleType.DECODE.name(), getIp(),
                InflightEvictor.maxAgeMs(inflightRequests, System.currentTimeMillis()));
    }

    /**
     * Real KV total capacity reported by the engine.
     */
    public long realKvTotal() {
        return status.getTotalKvCacheTokens().get();
    }

    public int getInflightCount() {
        return inflightRequests.size();
    }

    /**
     * Evict inflight requests older than {@code ttlMs}.
     * Called periodically by the scheduler to clean up stale decode entries.
     * Also purges layered-registry entries not refreshed by any calibrate for
     * {@code ttlMs} (safety net for endpoints that stopped reporting; a live
     * endpoint refreshes its confirmed entries on every WorkerStatus round).
     *
     * @return number of entries evicted
     */
    public int evictExpiredRequests(long ttlMs) {
        return evictExpiredRequests(ttlMs, ignored -> false);
    }

    /** Evict only endpoint orphans which have no live scheduler generation. */
    public int evictExpiredRequests(long ttlMs,
                                    LongPredicate schedulerOwnsRequest) {
        int evicted;
        boolean capacityChanged;
        admissionLock.lock();
        try {
            // A priority claim or generic EngineFence is a stronger accounting
            // owner than age-only cleanup. In particular, an ambiguous
            // ENGINE_MAY_HAVE_SEEN shadow remains charged until reconciliation.
            evicted = requestEvictor.evictExpired(
                    ttlMs, requestId -> !schedulerOwnsRequest.test(requestId)
                            && !preemptionClaims.containsKey(requestId)
                            && !engineFenceProtections.containsKey(requestId));
            int permitsRemoved = 0;
            // InflightEvictor owns only the reservation map/counters. Reconcile
            // its token-fenced pre-delivery owners under the same admission lock.
            for (Long requestId : List.copyOf(engineDispatchPermits.keySet())) {
                EngineDispatchPermitLease permit = engineDispatchPermits.get(requestId);
                if (permit != null
                        && inflightRequests.get(requestId) != permit.reservation()
                        && removeEngineDispatchPermitLocked(requestId)) {
                    permitsRemoved++;
                }
            }
            long cutoff = System.currentTimeMillis() - ttlMs;
            int trackedPurged = 0;
            java.util.Iterator<Map.Entry<Long, ConfirmedTask>> trackedEvictIt =
                    trackedConfirmed.entrySet().iterator();
            while (trackedEvictIt.hasNext()) {
                Map.Entry<Long, ConfirmedTask> entry = trackedEvictIt.next();
                if (entry.getValue().lastSeenMs() < cutoff
                        && !schedulerOwnsRequest.test(entry.getKey())
                        && !preemptionClaims.containsKey(entry.getKey())
                        && !engineFenceProtections.containsKey(entry.getKey())) {
                    trackedEvictIt.remove();
                    trackedPurged++;
                }
            }
            if (trackedPurged > 0) {
                confirmedRunningCount = Math.max(
                        0, confirmedRunningCount - trackedPurged);
            }
            boolean settledTombstonesPurged = settledTombstones.entrySet()
                    .removeIf(entry -> entry.getValue() < cutoff);
            if (evicted > 0 || permitsRemoved > 0
                    || trackedPurged > 0 || settledTombstonesPurged) {
                admissionVersion.incrementAndGet();
            }
            capacityChanged = evicted > 0 || permitsRemoved > 0 || trackedPurged > 0;
        } finally {
            admissionLock.unlock();
        }
        if (capacityChanged) {
            notifyEngineDispatchCapacityListeners();
        }
        return evicted;
    }

    public int getTotalLoad() {
        return confirmedRunningCount + inflightRequests.size();
    }

    /**
     * Engine-facing load (N2): confirmed running/accepted requests plus
     * reserved entries that are <b>not</b> parked in a prefill queue. Queued
     * reservations remain in the full placement/priority view, but they must
     * not close the decode concurrency gate while the engine is idle (root
     * cause C of the 8400 storm). {@link #getTotalLoad()} keeps the full shadow
     * view for observability and eviction planning.
     *
     * <p>O(1) formula (PR-C): {@code confirmedRunningCount
     * + max(0, inflightRequests.size() − queuedPhaseCount)}. The
     * {@link #queuedPhaseCount} AtomicInteger is maintained incrementally at
     * every queuedPhase mutation point ({@link #markQueuedPhase},
     * {@link EngineDispatchPermit#transferToEngineLifecycle()}, {@link #reserve},
     * {@link #release},
     * {@link #doCalibrate}, {@link #evictExpiredRequests}) so the hot gate
     * path no longer scans the queued set.
     *
     * <p><b>Drift self-healing:</b> the counter is read lock-free and may
     * transiently drift outside {@code [0, inflight]} under torn updates;
     * the read side clamps to that range and emits a drift metric so the
     * gate never sees an out-of-range load.
     */
    public int getEngineLoad() {
        int inflight = inflightRequests.size();
        int queued = queuedPhaseCount.get();
        // Drift self-healing: queued should stay within [0, inflight].
        if (queued < 0 || queued > inflight) {
            logger.warn("Decode queuedPhaseCount drift: count={}, inflight={}, confirmed={}, "
                    + "clamping to [{}, {}]", queued, inflight, confirmedRunningCount, 0, inflight);
            queued = Math.max(0, Math.min(queued, inflight));
        }
        return confirmedRunningCount + Math.max(0, inflight - queued);
    }

    /**
     * Mark a reserved request as committed into a prefill queue (N2). Called
     * by queue schedulers before queue publication; no-op when the id holds no
     * reservation (DIRECT paths never call this).
     */
    public void markQueuedPhase(long requestId) {
        boolean capacityChanged = false;
        admissionLock.lock();
        try {
            RequestInflight reservation = inflightRequests.get(requestId);
            if (addQueuedPhaseLocked(requestId, reservation)) {
                // Re-queueing begins a new dispatch round. Invalidate any
                // pre-delivery lease before publishing that transition.
                removeEngineDispatchPermitLocked(requestId);
                admissionVersion.incrementAndGet();
                capacityChanged = true;
            }
        } finally {
            admissionLock.unlock();
        }
        if (capacityChanged) {
            notifyEngineDispatchCapacityListeners();
        }
    }

    private boolean addQueuedPhaseLocked(long requestId, RequestInflight reservation) {
        if (reservation == null || !queuedPhase.add(requestId)) {
            return false;
        }
        queuedPhaseCount.incrementAndGet();
        queuedHardKvReservedTotal.addAndGet(reservation.kvTokens());
        queuedExpectedKvReservedTotal.addAndGet(reservation.expectedKvTokens());
        return true;
    }

    private boolean removeQueuedPhaseLocked(long requestId, RequestInflight reservation) {
        if (!queuedPhase.remove(requestId)) {
            return false;
        }
        queuedPhaseCount.decrementAndGet();
        if (reservation == null) {
            throw new IllegalStateException(
                    "queued Decode reservation missing for request " + requestId);
        }
        queuedHardKvReservedTotal.addAndGet(-reservation.kvTokens());
        queuedExpectedKvReservedTotal.addAndGet(-reservation.expectedKvTokens());
        return true;
    }

    /** Outcome of acquiring a pre-delivery Decode slot. */
    public enum EngineDispatchPermitAcquireStatus {
        /** A token-fenced permit now owns one Decode hard-gate slot. */
        ACQUIRED,
        /** Concurrency or Decode KV has no unreserved hard capacity. */
        CAPACITY_FULL,
        /** The request no longer owns a live shadow reservation. */
        NOT_OWNED,
        /** The reservation is already engine-facing rather than Prefill-queued. */
        NOT_QUEUED,
        /** Another pre-delivery attempt already owns this request's permit. */
        ALREADY_ACQUIRED,
        /** This exact Decode endpoint generation no longer accepts delivery. */
        ENDPOINT_RETIRED
    }

    /** Outcome of transferring an acquired permit to engine lifecycle ownership. */
    public enum EngineDispatchPermitTransferStatus {
        TRANSFERRED,
        OWNERSHIP_LOST,
        ENDPOINT_RETIRED
    }

    /** Explicit result of {@link #acquireEngineDispatchPermit(long, long)}. */
    public record EngineDispatchPermitAcquisition(
            EngineDispatchPermitAcquireStatus status,
            EngineDispatchPermit permit) {

        public EngineDispatchPermitAcquisition {
            if (status == null) {
                throw new IllegalArgumentException("permit acquisition status is required");
            }
            if ((status == EngineDispatchPermitAcquireStatus.ACQUIRED) != (permit != null)) {
                throw new IllegalArgumentException(
                        "only an ACQUIRED result may carry an engine dispatch permit");
            }
        }
    }

    /**
     * Token-fenced reservation of one Decode concurrency slot.
     *
     * <p>{@link #transferToEngineLifecycle()} atomically converts the still-current queued
     * reservation into engine-facing ownership without checking capacity again.
     * {@link #release()} gives up only this token's temporary hard-gate slot;
     * the reservation stays queued. Both operations are idempotent with respect
     * to endpoint state, and an old request-id generation cannot affect a newer
     * permit.
     */
    public static final class EngineDispatchPermit {

        private enum Resolution {
            ACQUIRED,
            ENGINE_LIFECYCLE_OWNED,
            RELEASED,
            INVALIDATED,
            ENDPOINT_RETIRED
        }

        private final DecodeEndpoint endpoint;
        private final long requestId;
        private final long token;
        private final RequestInflight reservation;
        private Resolution resolution = Resolution.ACQUIRED;

        private EngineDispatchPermit(DecodeEndpoint endpoint,
                                     long requestId,
                                     long token,
                                     RequestInflight reservation) {
            this.endpoint = endpoint;
            this.requestId = requestId;
            this.token = token;
            this.reservation = reservation;
        }

        public long requestId() {
            return requestId;
        }

        /**
         * Transfer this acquired slot without a second capacity check.
         *
         * @return a typed distinction between transfer, prior request-owner
         *         loss, and endpoint-generation retirement
         */
        public synchronized EngineDispatchPermitTransferStatus
                transferToEngineLifecycle() {
            if (resolution == Resolution.ENGINE_LIFECYCLE_OWNED) {
                return EngineDispatchPermitTransferStatus.TRANSFERRED;
            }
            if (resolution == Resolution.ENDPOINT_RETIRED) {
                return EngineDispatchPermitTransferStatus.ENDPOINT_RETIRED;
            }
            if (resolution != Resolution.ACQUIRED) {
                return EngineDispatchPermitTransferStatus.OWNERSHIP_LOST;
            }
            EngineDispatchPermitTransferStatus transfer =
                    endpoint.transferEngineDispatchPermitToLifecycle(this);
            if (transfer == EngineDispatchPermitTransferStatus.TRANSFERRED) {
                resolution = Resolution.ENGINE_LIFECYCLE_OWNED;
            } else if (transfer == EngineDispatchPermitTransferStatus.ENDPOINT_RETIRED) {
                resolution = Resolution.ENDPOINT_RETIRED;
            } else {
                resolution = Resolution.INVALIDATED;
            }
            return transfer;
        }

        /**
         * Release this acquired slot while leaving the same reservation queued.
         *
         * @return {@code true} only when this invocation removed the live permit;
         *         repeated or stale releases return {@code false}
         */
        public synchronized boolean release() {
            if (resolution != Resolution.ACQUIRED) {
                return false;
            }
            if (endpoint.releaseEngineDispatchPermit(this)) {
                resolution = Resolution.RELEASED;
                return true;
            }
            resolution = Resolution.INVALIDATED;
            return false;
        }

    }

    /** Immutable endpoint-side fence for one active public permit object. */
    private record EngineDispatchPermitLease(
            long token,
            RequestInflight reservation) {
    }

    /**
     * Acquire one pre-delivery Decode slot while the reservation remains queued.
     * Decode concurrency and KV are validated and occupied under the same
     * admission lock, so concurrent acquisitions cannot oversell either gate.
     */
    public EngineDispatchPermitAcquisition acquireEngineDispatchPermit(
            long requestId, long concurrencyLimit) {
        return acquireEngineDispatchPermit(requestId, concurrencyLimit, -1L);
    }

    public EngineDispatchPermitAcquisition acquireEngineDispatchPermit(
            long requestId,
            long concurrencyLimit,
            long maxKvUsagePercent) {
        admissionLock.lock();
        try {
            if (!acceptingEngineDispatchPermits) {
                return rejectedEngineDispatchPermit(
                        EngineDispatchPermitAcquireStatus.ENDPOINT_RETIRED);
            }
            RequestInflight reservation = inflightRequests.get(requestId);
            if (reservation == null || preemptionClaims.containsKey(requestId)) {
                return rejectedEngineDispatchPermit(
                        EngineDispatchPermitAcquireStatus.NOT_OWNED);
            }
            if (!queuedPhase.contains(requestId)) {
                return rejectedEngineDispatchPermit(
                        EngineDispatchPermitAcquireStatus.NOT_QUEUED);
            }
            if (engineDispatchPermits.containsKey(requestId)) {
                return rejectedEngineDispatchPermit(
                        EngineDispatchPermitAcquireStatus.ALREADY_ACQUIRED);
            }
            if (isEngineDispatchCapacityFullLocked(
                    reservation, concurrencyLimit, maxKvUsagePercent)) {
                return rejectedEngineDispatchPermit(
                        EngineDispatchPermitAcquireStatus.CAPACITY_FULL);
            }

            long token = nextEngineDispatchPermitTokenLocked();
            EngineDispatchPermit permit = new EngineDispatchPermit(
                    this, requestId, token, reservation);
            engineDispatchPermits.put(
                    requestId, new EngineDispatchPermitLease(token, reservation));
            activeEngineDispatchPermitCount++;
            engineDispatchPermitHardKvReservedTotal.addAndGet(
                    reservation.kvTokens());
            engineDispatchPermitExpectedKvReservedTotal.addAndGet(
                    reservation.expectedKvTokens());
            admissionVersion.incrementAndGet();
            return new EngineDispatchPermitAcquisition(
                    EngineDispatchPermitAcquireStatus.ACQUIRED, permit);
        } finally {
            admissionLock.unlock();
        }
    }

    private static EngineDispatchPermitAcquisition rejectedEngineDispatchPermit(
            EngineDispatchPermitAcquireStatus status) {
        return new EngineDispatchPermitAcquisition(status, null);
    }

    private EngineDispatchPermitTransferStatus transferEngineDispatchPermitToLifecycle(
            EngineDispatchPermit permit) {
        EngineDispatchPermitTransferStatus transferStatus;
        boolean capacityIncreased;
        admissionLock.lock();
        try {
            int usageBefore = engineDispatchHardGateUsageLocked();
            if (retiredEngineDispatchPermitTokens.remove(permit.token)) {
                return EngineDispatchPermitTransferStatus.ENDPOINT_RETIRED;
            }
            if (!isCurrentEngineDispatchPermitLocked(permit)) {
                return EngineDispatchPermitTransferStatus.OWNERSHIP_LOST;
            }
            if (inflightRequests.get(permit.requestId) != permit.reservation
                    || !queuedPhase.contains(permit.requestId)
                    || preemptionClaims.containsKey(permit.requestId)) {
                removeEngineDispatchPermitLocked(permit.requestId);
                admissionVersion.incrementAndGet();
                transferStatus = EngineDispatchPermitTransferStatus.OWNERSHIP_LOST;
            } else {
                removeEngineDispatchPermitLocked(permit.requestId);
                // The identity and queued membership were checked while holding the
                // same lock. Transfer only changes ownership; it never re-reads the cap.
                removeQueuedPhaseLocked(permit.requestId, permit.reservation);
                admissionVersion.incrementAndGet();
                transferStatus = EngineDispatchPermitTransferStatus.TRANSFERRED;
            }
            capacityIncreased = engineDispatchHardGateUsageLocked() < usageBefore;
        } finally {
            admissionLock.unlock();
        }
        if (capacityIncreased) {
            notifyEngineDispatchCapacityListeners();
        }
        return transferStatus;
    }

    private boolean releaseEngineDispatchPermit(EngineDispatchPermit permit) {
        boolean released = false;
        admissionLock.lock();
        try {
            if (retiredEngineDispatchPermitTokens.remove(permit.token)) {
                return true;
            }
            if (!isCurrentEngineDispatchPermitLocked(permit)) {
                return false;
            }
            removeEngineDispatchPermitLocked(permit.requestId);
            admissionVersion.incrementAndGet();
            released = true;
        } finally {
            admissionLock.unlock();
        }
        if (released) {
            notifyEngineDispatchCapacityListeners();
        }
        return true;
    }

    private boolean isCurrentEngineDispatchPermitLocked(EngineDispatchPermit permit) {
        EngineDispatchPermitLease lease = engineDispatchPermits.get(permit.requestId);
        return lease != null
                && lease.token() == permit.token
                && lease.reservation() == permit.reservation;
    }

    private boolean removeEngineDispatchPermitLocked(long requestId) {
        EngineDispatchPermitLease removed = engineDispatchPermits.remove(requestId);
        if (removed == null) {
            return false;
        }
        engineDispatchPermitHardKvReservedTotal.addAndGet(
                -removed.reservation().kvTokens());
        engineDispatchPermitExpectedKvReservedTotal.addAndGet(
                -removed.reservation().expectedKvTokens());
        decrementActiveEngineDispatchPermitCountLocked();
        return true;
    }

    /** Caller holds admissionLock. */
    private void decrementActiveEngineDispatchPermitCountLocked() {
        activeEngineDispatchPermitCount--;
        if (activeEngineDispatchPermitCount < 0) {
            throw new IllegalStateException("negative active Decode dispatch permit count");
        }
    }

    private long nextEngineDispatchPermitTokenLocked() {
        if (nextEngineDispatchPermitToken <= 0L
                || nextEngineDispatchPermitToken == Long.MAX_VALUE) {
            throw new IllegalStateException("Decode dispatch permit token space exhausted");
        }
        return nextEngineDispatchPermitToken++;
    }

    private int engineDispatchHardGateUsageLocked() {
        int engineFacingInflight = Math.max(0,
                inflightRequests.size() - queuedPhaseCount.get());
        return confirmedRunningCount + engineFacingInflight
                + activeEngineDispatchPermitCount;
    }

    /** Caller holds admissionLock; this is the authoritative pre-admission gate. */
    private boolean isEngineDispatchCapacityFullLocked(
            RequestInflight candidate,
            long concurrencyLimit,
            long maxKvUsagePercent) {
        return isEngineDispatchCapacityFullSnapshot(
                candidate, concurrencyLimit, maxKvUsagePercent);
    }

    /**
     * Common O(1) gate math used by authoritative acquisition and the live
     * waiter hint. Acquired permits are already included in both the slot and
     * KV counters; {@code candidate} is still a queued soft reservation and is
     * projected exactly once here.
     */
    private boolean isEngineDispatchCapacityFullSnapshot(
            RequestInflight candidate,
            long concurrencyLimit,
            long maxKvUsagePercent) {
        if (concurrencyLimit > 0
                && getEngineLoad() + Math.max(0, activeEngineDispatchPermitCount)
                >= concurrencyLimit) {
            return true;
        }

        long totalKv = reportedKvTotal.get();
        if (maxKvUsagePercent < 0 || totalKv <= 0) {
            return false;
        }
        if (candidate.kvTokens() > engineFacingKvAvailable()) {
            return true;
        }
        if (maxKvUsagePercent == 0) {
            return false;
        }

        long projectedExpectedKv = saturatedAddNonNegative(
                engineFacingKvUsed(), candidate.expectedKvTokens());
        return (double) projectedExpectedKv * 100.0
                > (double) maxKvUsagePercent * (double) totalKv;
    }

    /** Saturating addition for non-negative admission counters. */
    private static long saturatedAddNonNegative(long left, long right) {
        if (left < 0 || right < 0) {
            throw new IllegalArgumentException("KV admission counters must be non-negative");
        }
        return left > Long.MAX_VALUE - right ? Long.MAX_VALUE : left + right;
    }

    /**
     * Lock-free live predicate for a WorkerBatcher capacity wait.
     *
     * <p>The queue condition invokes this method while holding its own lock, so
     * this method must never acquire {@link #admissionLock}. A torn snapshot can
     * only cause an extra authoritative acquisition attempt; every capacity or
     * ownership transition publishes a listener wake after its atomic/volatile
     * counters are updated. Returning {@code true} for retirement or ownership
     * loss lets the worker resume and obtain the exact typed acquisition result.
     */
    public boolean isEngineDispatchPermitAvailable(
            long requestId,
            long concurrencyLimit,
            long maxKvUsagePercent) {
        if (!acceptingEngineDispatchPermits) {
            return true;
        }
        RequestInflight candidate = inflightRequests.get(requestId);
        if (candidate == null
                || !queuedPhase.contains(requestId)
                || engineDispatchPermits.containsKey(requestId)) {
            return true;
        }
        return !isEngineDispatchCapacityFullSnapshot(
                candidate, concurrencyLimit, maxKvUsagePercent);
    }

    /** Live wait predicate only; permit acquisition remains the authoritative gate. */
    public boolean hasEngineDispatchCapacity(long concurrencyLimit) {
        if (concurrencyLimit <= 0) {
            return true;
        }
        return getEngineLoad() + Math.max(0, activeEngineDispatchPermitCount)
                < concurrencyLimit;
    }

    /**
     * Engine-confirmed (KV-allocated / running) request count from the last
     * calibration — the merged accepted + running total used by the totalLoad
     * accounting. The Phase 5 per-layer split is exposed separately via
     * {@link #getAcceptedLayerCount()} / {@link #getRunningLayerCount()}.
     */
    public int getConfirmedRunningCount() {
        return confirmedRunningCount;
    }

    /** Whether the latest Decode WorkerStatus still owns this request. */
    public boolean isConfirmedTracked(long requestId) {
        return trackedConfirmed.containsKey(requestId);
    }

    @Override
    public long getLoadMetric() {
        return getTotalLoad();
    }

    /**
     * Mutable layered-registry entry for one engine-confirmed request
     * (Phase 5). Identity fields (priority / KV estimate) are fixed
     * at first sight; {@code phase} and {@code lastSeenMs} are volatile and
     * only mutated under {@link #admissionLock} by calibration.
     */
    static final class ConfirmedTask {

        private final int priority;
        private final long kvTokens;
        private final boolean priorityKnown;
        private volatile DecodeTaskPhase phase;
        private volatile long lastSeenMs;

        ConfirmedTask(int priority, long kvTokens,
                      DecodeTaskPhase layer, long now, boolean priorityKnown) {
            this.priority = priority;
            this.kvTokens = kvTokens;
            this.priorityKnown = priorityKnown;
            this.phase = layer;
            this.lastSeenMs = now;
        }

        int priority() { return priority; }
        long kvTokens() { return kvTokens; }
        boolean priorityKnown() { return priorityKnown; }
        DecodeTaskPhase phase() { return phase; }
        long lastSeenMs() { return lastSeenMs; }

        /** Refresh layer membership and liveness on every calibrate round. */
        void refresh(DecodeTaskPhase layer, long now) {
            this.phase = layer;
            this.lastSeenMs = now;
        }
    }

    /**
     * Mutable generic-fence owner. Every field is accessed only while holding
     * {@link #admissionLock}; routing reads only the aggregate atomics/volatile
     * counters, never these entries.
     */
    private static final class EngineFenceProtection {
        private long hardKvTokens;
        private long expectedKvTokens;
        private boolean confirmedOwner;
        private boolean syntheticHeld;

        private EngineFenceProtection(long hardKvTokens,
                                      long expectedKvTokens,
                                      boolean confirmedOwner) {
            this.hardKvTokens = Math.max(0, hardKvTokens);
            this.expectedKvTokens = Math.max(this.hardKvTokens, expectedKvTokens);
            this.confirmedOwner = confirmedOwner;
        }

        private void retainAtLeast(long hardKvTokens, long expectedKvTokens) {
            this.hardKvTokens = Math.max(this.hardKvTokens, Math.max(0, hardKvTokens));
            this.expectedKvTokens = Math.max(
                    this.expectedKvTokens,
                    Math.max(this.hardKvTokens, expectedKvTokens));
        }
    }

    private enum ClaimOwner {
        SHADOW_IN_FLIGHT,
        ENGINE_CONFIRMED
    }

    private enum ClaimState {
        CLAIMED,
        CANCEL_IN_FLIGHT,
        CANCEL_REQUESTED,
        CANCELED_SETTLED,
        NOT_FOUND_STALE,
        CANCEL_UNKNOWN,
        /** Ownership transferred from a completed attempt to EngineFence. */
        ENGINE_FENCE
    }

    private static final class PreemptionClaim {
        private final long attemptToken;
        private ClaimOwner owner;
        private final long hardKvTokens;
        private final long expectedKvTokens;
        private ClaimState state = ClaimState.CLAIMED;
        private boolean kvHeldAfterWorkerRelease;

        private PreemptionClaim(long attemptToken, ClaimOwner owner,
                                long hardKvTokens, long expectedKvTokens) {
            this.attemptToken = attemptToken;
            this.owner = owner;
            this.hardKvTokens = hardKvTokens;
            this.expectedKvTokens = expectedKvTokens;
        }
    }

    private static final class EndpointPreemptionAttempt {
        private final long incomingRequestId;
        private final Set<Long> victimIds;

        private EndpointPreemptionAttempt(long incomingRequestId, Set<Long> victimIds) {
            this.incomingRequestId = incomingRequestId;
            this.victimIds = victimIds;
        }
    }

}
