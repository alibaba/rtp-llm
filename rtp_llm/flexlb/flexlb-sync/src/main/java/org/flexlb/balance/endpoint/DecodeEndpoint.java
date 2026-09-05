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
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Predicate;

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

    private final ConcurrentHashMap<String, RequestInflight> inflightRequests = new ConcurrentHashMap<>();
    private final AtomicLong inflightKvReservedTotal = new AtomicLong(0);
    private final AtomicLong inflightExpectedKvReservedTotal = new AtomicLong(0);
    private final AtomicLong reportedKvAvailable = new AtomicLong();
    private volatile int confirmedRunningCount;
    private final InflightEvictor<String, RequestInflight> requestEvictor;

    /**
     * Layered registry of engine-confirmed requests (Phase 5): requestId →
     * accepted/running membership. Rebuilt against every calibrate report;
     * carries no shadow accounting — confirmed KV is engine-reported and the
     * slot count stays in {@code confirmedRunningCount}, so this registry is
     * pure metadata for eviction planning, cancel dedup and layered gauges.
     */
    private final ConcurrentHashMap<String, ConfirmedTask> trackedConfirmed = new ConcurrentHashMap<>();

    /**
     * Token-fenced priority-preemption ownership.  Victim accounting remains
     * in its original layer until a typed Prefill WorkerStatus CANCELED event
     * is settled; an ACCEPTED Cancel response only advances the claim state.
     * All access is under {@link #admissionLock}.
     */
    private final Map<String, PreemptionClaim> preemptionClaims = new HashMap<>();
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
    private final Map<String, EngineFenceProtection> engineFenceProtections = new HashMap<>();
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
    private final Map<String, Long> settledTombstones = new HashMap<>();

    /**
     * Reserved entries whose request is still sitting in a prefill queue —
     * committed by the scheduler but not yet dispatched to the engine (N2,
     * plan-commit redesign). These reservations keep protecting KV against
     * oversell, but must not count against the decode concurrency limit:
     * counting them produced the shadow-saturation 8400 storm (root cause C —
     * queued reservations saturating {@code getTotalLoad()} while the engine
     * sat idle). Marked at plan commit ({@code markQueuedPhase}), unmarked at
     * batch dispatch ({@code tryMarkEngineMayHaveSeen}); release/calibrate prune
     * it alongside {@code inflightRequests}. Legacy/DIRECT paths never mark,
     * so their accounting is unchanged.
     */
    private final Set<String> queuedPhase = ConcurrentHashMap.newKeySet();

    /**
     * O(1) mirror of {@code |queuedPhase ∩ inflightRequests|} (PR-C):
     * incremented when a reservation is marked queued and decremented when
     * it is dispatched / released / calibrated out, so {@link #getEngineLoad}
     * avoids the per-call O(n) scan of the legacy formula. Read lock-free;
     * written under {@link #admissionLock}.
     */
    private final AtomicInteger queuedPhaseCount = new AtomicInteger(0);

    /**
     * Serializes admission-state mutations (reserve / release / calibrate /
     * expired eviction) so that {@link #tryReleaseVictimsAndReserveIncoming}
     * can validate-then-apply atomically against {@link #admissionVersion}
     * (design doc 11.5/17.2). Reads stay lock-free.
     */
    private final ReentrantLock admissionLock = new ReentrantLock();

    /**
     * Monotonic admission version bumped on every mutation of the local
     * admission state (reserve / release / calibrate / expired eviction).
     * Captured in Auto-TPM cluster snapshots (after this scheduler's own
     * reserve) and re-checked at plan commit time to detect interference.
     */
    private final AtomicLong admissionVersion = new AtomicLong();

    public DecodeEndpoint(WorkerStatus status) {
        super(status);
        this.requestEvictor = new InflightEvictor<>(inflightRequests, req -> {
            inflightKvReservedTotal.addAndGet(-req.kvTokens());
            inflightExpectedKvReservedTotal.addAndGet(-req.expectedKvTokens());
        });
    }

    public void reserve(String requestId, long kvTokens, long expectedKvTokens) {
        reserve(requestId, kvTokens, expectedKvTokens,
                RequestInflight.DEFAULT_PRIORITY);
    }

    /**
     * Shadow-reserve decode capacity for a request, carrying its Auto-TPM
     * priority so the reservation can later be ranked as a decode eviction
     * candidate (design doc 10.1).
     */
    public void reserve(String requestId, long kvTokens, long expectedKvTokens,
                        int priority) {
        admissionLock.lock();
        try {
            RequestInflight newRi = new RequestInflight(kvTokens, expectedKvTokens, priority);
            // A (re-)reserve puts the request back into the pre-queue state.
            if (queuedPhase.remove(requestId)) {
                queuedPhaseCount.decrementAndGet();
            }
            RequestInflight prev = inflightRequests.putIfAbsent(requestId, newRi);
            if (prev != null) {
                // requestId already exists — subtract the old kvTokens before overwriting,
                // otherwise the old value is silently lost and the counter stays inflated.
                inflightKvReservedTotal.addAndGet(-prev.kvTokens());
                inflightExpectedKvReservedTotal.addAndGet(-prev.expectedKvTokens());
                inflightRequests.put(requestId, newRi);
            }
            inflightKvReservedTotal.addAndGet(kvTokens);
            inflightExpectedKvReservedTotal.addAndGet(expectedKvTokens);
            admissionVersion.incrementAndGet();
        } finally {
            admissionLock.unlock();
        }
    }

    public void release(String requestId) {
        admissionLock.lock();
        try {
            boolean protectionRemoved = clearEngineFenceProtectionLocked(requestId);
            RequestInflight removed = inflightRequests.remove(requestId);
            if (queuedPhase.remove(requestId)) {
                queuedPhaseCount.decrementAndGet();
            }
            if (removed != null) {
                inflightKvReservedTotal.addAndGet(-removed.kvTokens());
                inflightExpectedKvReservedTotal.addAndGet(-removed.expectedKvTokens());
            }
            if (removed != null || protectionRemoved) {
                admissionVersion.incrementAndGet();
            }
        } finally {
            admissionLock.unlock();
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
            String requestId, RequestInflight expectedReservation) {
        if (expectedReservation == null) {
            return false;
        }
        admissionLock.lock();
        try {
            if (inflightRequests.get(requestId) != expectedReservation
                    || engineFenceProtections.containsKey(requestId)
                    || preemptionClaims.containsKey(requestId)) {
                return false;
            }
            inflightRequests.remove(requestId);
            if (queuedPhase.remove(requestId)) {
                queuedPhaseCount.decrementAndGet();
            }
            inflightKvReservedTotal.addAndGet(-expectedReservation.kvTokens());
            inflightExpectedKvReservedTotal.addAndGet(
                    -expectedReservation.expectedKvTokens());
            admissionVersion.incrementAndGet();
            return true;
        } finally {
            admissionLock.unlock();
        }
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
    public boolean beginEngineFenceProtection(String requestId) {
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
    public boolean endEngineFenceProtection(String requestId) {
        admissionLock.lock();
        try {
            if (!clearEngineFenceProtectionLocked(requestId)) {
                return false;
            }
            admissionVersion.incrementAndGet();
            return true;
        } finally {
            admissionLock.unlock();
        }
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
    public boolean settleTombstonedRequest(String requestId) {
        admissionLock.lock();
        try {
            boolean changed = settleRequestAccountingLocked(
                    requestId, System.currentTimeMillis(), true);
            if (changed) {
                admissionVersion.incrementAndGet();
            }
            return changed;
        } finally {
            admissionLock.unlock();
        }
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
            List<String> victimIds,
            String incomingRequestId, long kvTokens, long expectedKvTokens,
            int priority,
            long expectedAdmissionVersion) {
        admissionLock.lock();
        try {
            if (admissionVersion.get() != expectedAdmissionVersion) {
                return ReleaseReserveResult.VERSION_MISMATCH;
            }
            for (String victimId : victimIds) {
                RequestInflight victim = inflightRequests.get(victimId);
                if (victim == null || !queuedPhase.contains(victimId)
                        || preemptionClaims.containsKey(victimId)
                        || engineFenceProtections.containsKey(victimId)) {
                    return ReleaseReserveResult.VICTIM_GONE;
                }
            }
            for (String victimId : victimIds) {
                release(victimId);
            }
            reserve(incomingRequestId, kvTokens, expectedKvTokens, priority);
            return ReleaseReserveResult.SUCCESS;
        } finally {
            admissionLock.unlock();
        }
    }

    /**
     * CAS-style conditional release (redesign N3 §3.4): under the admission
     * lock, release the reservation only if it is still held as a
     * Master-queued shadow entry. Returns {@code false} —
     * touching nothing — when the reservation is gone (already released) or
     * has been folded into the confirmed layer (the engine owns the request;
     * contract 5.3 forbids terminal operations on dispatched requests).
     */
    public boolean releaseIfHeld(String requestId) {
        admissionLock.lock();
        try {
            RequestInflight held = inflightRequests.get(requestId);
            if (held == null || !queuedPhase.contains(requestId)
                    || preemptionClaims.containsKey(requestId)
                    || engineFenceProtections.containsKey(requestId)) {
                return false;
            }
            release(requestId);
            return true;
        } finally {
            admissionLock.unlock();
        }
    }

    /** Result of {@link #tryReleaseVictimsIfHeldAndReserveIncoming}. */
    public record PresenceEvictionOutcome(boolean success, List<String> freedVictimIds) {
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
            List<String> victimIds,
            String incomingRequestId, long kvTokens, long expectedKvTokens,
            int priority) {
        admissionLock.lock();
        try {
            List<String> freed = new ArrayList<>(victimIds.size());
            for (String victimId : victimIds) {
                if (releaseIfHeld(victimId)) {
                    freed.add(victimId);
                }
            }
            if (freed.size() < victimIds.size()) {
                return new PresenceEvictionOutcome(false, List.copyOf(freed));
            }
            reserve(incomingRequestId, kvTokens, expectedKvTokens, priority);
            return new PresenceEvictionOutcome(true, List.copyOf(freed));
        } finally {
            admissionLock.unlock();
        }
    }

    /**
     * Point-in-time copy of the reserved (shadow) entries keyed by requestId,
     * for eviction planning snapshots. Confirmed (accepted/running) requests
     * never appear here — calibrate removes them (design doc 10.1).
     *
     * <p>Taken under {@link #admissionLock} so the returned map is a consistent
     * point w.r.t. concurrent mutations.
     */
    public Map<String, RequestInflight> reservedView() {
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
    public RequestInflight reservationFor(String requestId) {
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
            List<ConfirmedTaskView> confirmed = new ArrayList<>(trackedConfirmed.size());
            trackedConfirmed.forEach((requestId, task) ->
                    confirmed.add(new ConfirmedTaskView(requestId, task.priority(),
                            task.kvTokens(), task.phase(), task.priorityKnown(),
                            preemptionClaims.containsKey(requestId)
                                    || engineFenceProtections.containsKey(requestId))));
            Set<String> claimed = Set.copyOf(preemptionClaims.keySet());
            if (!engineFenceProtections.isEmpty()) {
                Set<String> combined = new HashSet<>(claimed);
                combined.addAll(engineFenceProtections.keySet());
                claimed = Set.copyOf(combined);
            }
            return new LayeredAdmissionView(admissionVersion.get(),
                    Map.copyOf(inflightRequests), List.copyOf(confirmed),
                    Set.copyOf(queuedPhase),
                    claimed);
        } finally {
            admissionLock.unlock();
        }
    }

    /** Atomic (admissionVersion, reserved, confirmed, queued) tuple — see {@link #layeredAdmissionView()}. */
    public record LayeredAdmissionView(long admissionVersion,
                                       Map<String, RequestInflight> reserved,
                                       List<ConfirmedTaskView> confirmed,
                                       Set<String> queued,
                                       Set<String> claimed) {
    }

    /** Immutable point-in-time view of one layered-registry entry. */
    public record ConfirmedTaskView(String requestId,
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
            List<String> victimIds,
            String incomingRequestId,
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

            Map<String, ClaimOwner> owners = new HashMap<>();
            for (String victimId : victimIds) {
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
            reserve(incomingRequestId, incomingKvTokens, incomingExpectedKvTokens,
                    incomingPriority);
            for (Map.Entry<String, ClaimOwner> entry : owners.entrySet()) {
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
            for (String victimId : attempt.victimIds) {
                PreemptionClaim claim = preemptionClaims.get(victimId);
                if (claim == null || claim.attemptToken != attemptToken
                        || claim.state != ClaimState.CLAIMED) {
                    return false;
                }
            }
            for (String victimId : attempt.victimIds) {
                preemptionClaims.get(victimId).state = ClaimState.CANCEL_IN_FLIGHT;
            }
            admissionVersion.incrementAndGet();
            return true;
        } finally {
            admissionLock.unlock();
        }
    }

    /** Cancel ACCEPTED: retain every byte/slot and only advance control state. */
    public boolean markPriorityCancelAccepted(long attemptToken, String requestId) {
        return transitionClaim(attemptToken, requestId,
                ClaimState.CANCEL_IN_FLIGHT, ClaimState.CANCEL_REQUESTED);
    }

    public boolean markPriorityCancelNotFound(long attemptToken, String requestId) {
        return transitionClaim(attemptToken, requestId,
                ClaimState.CANCEL_IN_FLIGHT, ClaimState.NOT_FOUND_STALE);
    }

    public boolean markPriorityCancelUnknown(long attemptToken, String requestId) {
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
    public boolean settlePriorityCanceled(long attemptToken, String requestId) {
        admissionLock.lock();
        try {
            PreemptionClaim claim = preemptionClaims.get(requestId);
            if (claim == null || claim.attemptToken != attemptToken
                    || (claim.state != ClaimState.CANCEL_REQUESTED
                        && claim.state != ClaimState.CANCEL_UNKNOWN)) {
                return false;
            }
            return settlePriorityClaimLocked(attemptToken, requestId, claim);
        } finally {
            admissionLock.unlock();
        }
    }

    /**
     * Settle an engine {@code TOMBSTONED} acknowledgement.
     *
     * <p>TOMBSTONED is stronger than NOT_FOUND: the addressed request was
     * absent and the engine atomically installed a late-enqueue fence. It is
     * therefore an authoritative terminal proof and may release the same
     * accounting as typed CANCELED without waiting for WorkerStatus.</p>
     */
    public boolean settlePriorityTombstoned(long attemptToken, String requestId) {
        admissionLock.lock();
        try {
            PreemptionClaim claim = preemptionClaims.get(requestId);
            if (claim == null || claim.attemptToken != attemptToken
                    || (claim.state != ClaimState.CANCEL_IN_FLIGHT
                        && claim.state != ClaimState.NOT_FOUND_STALE
                        && claim.state != ClaimState.CANCEL_UNKNOWN)) {
                return false;
            }
            return settlePriorityClaimLocked(attemptToken, requestId, claim);
        } finally {
            admissionLock.unlock();
        }
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
            String requestId) {
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
    public boolean settleEngineFenceClaim(long attemptToken, String requestId) {
        admissionLock.lock();
        try {
            PreemptionClaim claim = preemptionClaims.get(requestId);
            if (claim == null || claim.attemptToken != attemptToken
                    || claim.state != ClaimState.ENGINE_FENCE) {
                return false;
            }
            return settlePriorityClaimLocked(attemptToken, requestId, claim);
        } finally {
            admissionLock.unlock();
        }
    }

    /**
     * A fresh Decode-active observation before the transferred fence invokes
     * Cancel proves ordinary ownership. Drop only the control/synthetic hold;
     * the live confirmed or shadow accounting remains in its original layer.
     */
    public boolean releaseEngineFenceClaimActive(long attemptToken, String requestId) {
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
            return true;
        } finally {
            admissionLock.unlock();
        }
    }

    /** Called with {@link #admissionLock} held. */
    private boolean settlePriorityClaimLocked(long attemptToken,
                                              String requestId,
                                              PreemptionClaim claim) {
        boolean genericFenceRetainsAccounting =
                transferPriorityAccountingToEngineFenceLocked(requestId, claim);
        if (claim.owner == ClaimOwner.SHADOW_IN_FLIGHT) {
            RequestInflight removed = inflightRequests.remove(requestId);
            if (removed != null) {
                inflightKvReservedTotal.addAndGet(-removed.kvTokens());
                inflightExpectedKvReservedTotal.addAndGet(-removed.expectedKvTokens());
            }
            if (queuedPhase.remove(requestId)) {
                queuedPhaseCount.decrementAndGet();
            }
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
            for (String victimId : attempt.victimIds) {
                PreemptionClaim claim = preemptionClaims.get(victimId);
                if (claim == null || claim.attemptToken != attemptToken
                        || claim.state != ClaimState.CANCELED_SETTLED) {
                    return false;
                }
            }
            for (String victimId : attempt.victimIds) {
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
        admissionLock.lock();
        try {
            EndpointPreemptionAttempt attempt = preemptionAttempts.remove(attemptToken);
            if (attempt == null) {
                return;
            }
            release(attempt.incomingRequestId);
            for (String victimId : attempt.victimIds) {
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
        } finally {
            admissionLock.unlock();
        }
    }

    /** Fresh active status is the only path that reopens a NOT_FOUND_STALE victim. */
    public boolean reconcilePriorityVictimActive(String requestId) {
        admissionLock.lock();
        try {
            PreemptionClaim claim = preemptionClaims.get(requestId);
            if (claim == null || claim.state != ClaimState.NOT_FOUND_STALE) {
                return false;
            }
            releaseHeldKv(claim);
            preemptionClaims.remove(requestId);
            admissionVersion.incrementAndGet();
            return true;
        } finally {
            admissionLock.unlock();
        }
    }

    /**
     * Reconcile a one-shot ordinary Decode terminal after Cancel NOT_FOUND or
     * a transport-unknown ACK. Neither outcome is typed priority completion,
     * so the ordinary terminal resumes the pre-existing completion path.
     */
    public boolean reconcilePriorityVictimFinished(String requestId) {
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
            return true;
        } finally {
            admissionLock.unlock();
        }
    }

    private boolean transitionClaim(long attemptToken, String requestId,
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
    }

    private void doCalibrate(Map<String, TaskInfo> runningTaskInfo, Map<String, TaskInfo> finishedTaskInfo) {
        this.reportedKvAvailable.set(status.getAvailableKvCacheTokens().get());
        admissionVersion.incrementAndGet();

        // Build one authoritative Decode view.  Claimed victims that disappear
        // are held synthetically until the original Prefill publishes typed
        // CANCELED; generic Decode absence/finished must not release them.
        Set<String> confirmedNow = new HashSet<>();
        int actualConfirmed = 0;
        long now = System.currentTimeMillis();
        if (runningTaskInfo != null) {
            for (TaskInfo task : runningTaskInfo.values()) {
                TaskPhase phase = task.getPhase();
                String requestId = task.getRequestId();
                if ((phase == TaskPhase.KV_ALLOCATED || phase == TaskPhase.RUNNING)
                        && !settledTombstones.containsKey(requestId)) {
                    actualConfirmed++;
                    RequestInflight removed = inflightRequests.remove(requestId);
                    if (removed != null) {
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
        for (Map.Entry<String, PreemptionClaim> entry : preemptionClaims.entrySet()) {
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
        Iterator<Map.Entry<String, ConfirmedTask>> confirmedIt =
                trackedConfirmed.entrySet().iterator();
        while (confirmedIt.hasNext()) {
            Map.Entry<String, ConfirmedTask> entry = confirmedIt.next();
            String requestId = entry.getKey();
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
                String requestId = task.getRequestId();
                if (settledTombstones.containsKey(requestId)
                        || preemptionClaims.containsKey(requestId)) {
                    continue;
                }
                confirmedNow.remove(requestId);
                settleRequestAccountingLocked(requestId, now, false);
            }
        }

        // N2: keep the queued-phase set consistent with the reserved entries
        // (calibrate removes confirmed/finished entries directly, bypassing
        // release()). Drop stale queued ids one-by-one so the O(1) counter
        // stays in sync (PR-C).
        Iterator<String> queuedIt = queuedPhase.iterator();
        while (queuedIt.hasNext()) {
            String requestId = queuedIt.next();
            if (!inflightRequests.containsKey(requestId)) {
                queuedIt.remove();
                queuedPhaseCount.decrementAndGet();
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
    private boolean settleRequestAccountingLocked(String requestId,
                                                   long settledAtMs,
                                                   boolean explicitTombstoneProof) {
        boolean changed = false;

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
        if (queuedPhase.remove(requestId)) {
            queuedPhaseCount.decrementAndGet();
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
    private boolean rememberSettledLocked(String requestId, long settledAtMs) {
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
    private void observeEngineFenceConfirmedLocked(String requestId, RequestInflight removed) {
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
    private boolean clearEngineFenceProtectionLocked(String requestId) {
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
            String requestId,
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

    // ==================== Metrics ====================

    /**
     * Report per-worker decode inflight metrics via the given reporter.
     * Called periodically by {@link org.flexlb.balance.scheduler.PriorityScheduler}.
     */
    public void reportBatchMetrics(BatchSchedulerReporter reporter) {
        reporter.reportInflightRequestCount(RoleType.DECODE.name(), getStatus().getIpIndex(), getInflightCount());
        reporter.reportDecodeTotalLoad(getStatus().getIpIndex(), getTotalLoad());
        reporter.reportDecodeInflightKvReserved(getStatus().getIpIndex(), inflightKvReserved());
        reporter.reportDecodeInflightHardKvReserved(getStatus().getIpIndex(), inflightHardKvReserved());
        reporter.reportInflightMaxAgeMs(RoleType.DECODE.name(), getStatus().getIpIndex(),
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
                                    Predicate<String> schedulerOwnsRequest) {
        admissionLock.lock();
        try {
            // A priority claim or generic EngineFence is a stronger accounting
            // owner than age-only cleanup. In particular, an ambiguous
            // ENGINE_MAY_HAVE_SEEN shadow remains charged until reconciliation.
            int evicted = requestEvictor.evictExpired(
                    ttlMs, requestId -> !schedulerOwnsRequest.test(requestId)
                            && !preemptionClaims.containsKey(requestId)
                            && !engineFenceProtections.containsKey(requestId));
            // Drop stale queued ids one-by-one so the O(1) counter stays in
            // sync (PR-C) — evictExpired may have removed inflight entries.
            Iterator<String> queuedEvictIt = queuedPhase.iterator();
            while (queuedEvictIt.hasNext()) {
                String requestId = queuedEvictIt.next();
                if (!inflightRequests.containsKey(requestId)) {
                    queuedEvictIt.remove();
                    queuedPhaseCount.decrementAndGet();
                }
            }
            long cutoff = System.currentTimeMillis() - ttlMs;
            int trackedPurged = 0;
            Iterator<Map.Entry<String, ConfirmedTask>> trackedEvictIt =
                    trackedConfirmed.entrySet().iterator();
            while (trackedEvictIt.hasNext()) {
                Map.Entry<String, ConfirmedTask> entry = trackedEvictIt.next();
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
            if (evicted > 0 || trackedPurged > 0 || settledTombstonesPurged) {
                admissionVersion.incrementAndGet();
            }
            return evicted;
        } finally {
            admissionLock.unlock();
        }
    }

    public int getTotalLoad() {
        return confirmedRunningCount + inflightRequests.size();
    }

    /**
     * Engine-facing load (N2): confirmed running/accepted requests plus
     * reserved entries that are <b>not</b> parked in a prefill queue. Queued
     * reservations only guard KV against oversell — they must not close the
     * decode concurrency gate while the engine is idle (root cause C of the
     * 8400 storm). {@link #getTotalLoad()} keeps the full shadow view for
     * observability and eviction planning.
     *
     * <p>O(1) formula (PR-C): {@code confirmedRunningCount
     * + max(0, inflightRequests.size() − queuedPhaseCount)}. The
     * {@link #queuedPhaseCount} AtomicInteger is maintained incrementally at
     * every queuedPhase mutation point ({@link #markQueuedPhase},
     * {@link #tryMarkEngineMayHaveSeen}, {@link #reserve}, {@link #release},
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
     * by the priority scheduler at plan-commit time; no-op when the id holds
     * no reservation (legacy paths never call this).
     */
    public void markQueuedPhase(String requestId) {
        admissionLock.lock();
        try {
            if (inflightRequests.containsKey(requestId)) {
                if (queuedPhase.add(requestId)) {
                    queuedPhaseCount.incrementAndGet();
                }
            }
        } finally {
            admissionLock.unlock();
        }
    }

    /** Outcome of an engine-dispatch ownership claim. */
    public enum DispatchClaimResult {
        /** The request owns a reservation and may be exposed to the engine. */
        CLAIMED,
        /** The reservation is still queued, but no engine-facing slot is free yet. */
        CAPACITY_FULL,
        /** Release, timeout, or preemption already owns this reservation. */
        NOT_OWNED
    }

    /**
     * Atomically move one queued reservation into the engine-facing layer,
     * subject to the configured Decode concurrency limit.
     *
     * <p>The availability check performed while routing cannot be the final
     * slot fence: queued reservations intentionally do not count in
     * {@link #getEngineLoad()}, and one Prefill batch can therefore contain
     * many requests for the same Decode endpoint. Checking and clearing the
     * queued bit under the same {@link #admissionLock} prevents that batch
     * from moving the endpoint from (for example) load 4 to load 20 when the
     * configured limit is 5.
     *
     * <p>Legacy/non-queued reservations are already engine-facing. They are
     * returned as {@link DispatchClaimResult#CLAIMED} without consuming a
     * second slot, preserving the existing legacy path and making retries
     * idempotent.
     */
    public DispatchClaimResult tryClaimEngineDispatch(String requestId,
                                                       long concurrencyLimit) {
        admissionLock.lock();
        try {
            if (!inflightRequests.containsKey(requestId)
                    || preemptionClaims.containsKey(requestId)) {
                return DispatchClaimResult.NOT_OWNED;
            }
            if (!queuedPhase.contains(requestId)) {
                return DispatchClaimResult.CLAIMED;
            }

            int engineFacingInflight = Math.max(0,
                    inflightRequests.size() - queuedPhaseCount.get());
            int engineLoad = confirmedRunningCount + engineFacingInflight;
            if (concurrencyLimit > 0 && engineLoad >= concurrencyLimit) {
                return DispatchClaimResult.CAPACITY_FULL;
            }

            // queuedPhase membership was checked under this same lock. The
            // remove must therefore succeed unless the invariant is broken.
            queuedPhase.remove(requestId);
            queuedPhaseCount.decrementAndGet();
            admissionVersion.incrementAndGet();
            return DispatchClaimResult.CLAIMED;
        } finally {
            admissionLock.unlock();
        }
    }

    /**
     * Claim dispatch ownership before the scheduler exposes a request to the
     * engine. Under the same admission lock used by local victim release, a
     * queued reservation moves to ENGINE_MAY_HAVE_SEEN; whichever side wins
     * excludes the other. A legacy/non-queued reservation is already in that
     * phase and remains dispatchable. Missing, released, or preemption-claimed
     * reservations return {@code false} and must not be sent.
     */
    public boolean tryMarkEngineMayHaveSeen(String requestId) {
        return tryClaimEngineDispatch(requestId, 0) == DispatchClaimResult.CLAIMED;
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
    public boolean isConfirmedTracked(String requestId) {
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
        private final String incomingRequestId;
        private final Set<String> victimIds;

        private EndpointPreemptionAttempt(String incomingRequestId, Set<String> victimIds) {
            this.incomingRequestId = incomingRequestId;
            this.victimIds = victimIds;
        }
    }

}
