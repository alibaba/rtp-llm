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
     * Request-id fence against a stale WorkerStatus resurrecting a settled
     * victim: requestId -> settlement time. It shares the endpoint inflight
     * TTL cleanup, so the fence is long enough for delayed status deltas but
     * bounded by the same retention policy as scheduler terminal tombstones.
     */
    private final Map<Long, Long> priorityCanceledTombstones = new HashMap<>();

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
    private final java.util.Set<Long> queuedPhase = ConcurrentHashMap.newKeySet();

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

    public void reserve(long requestId, long kvTokens, long expectedKvTokens) {
        reserve(requestId, kvTokens, expectedKvTokens,
                RequestInflight.DEFAULT_PRIORITY, 0);
    }

    /**
     * Shadow-reserve decode capacity for a request, carrying its Auto-TPM
     * priority and admission deadline so the reservation can later be ranked
     * as a decode eviction candidate (design doc 10.1).
     */
    public void reserve(long requestId, long kvTokens, long expectedKvTokens,
                        int priority, long deadlineMs) {
        admissionLock.lock();
        try {
            RequestInflight newRi = new RequestInflight(kvTokens, expectedKvTokens, priority, deadlineMs);
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

    public void release(long requestId) {
        admissionLock.lock();
        try {
            RequestInflight removed = inflightRequests.remove(requestId);
            if (queuedPhase.remove(requestId)) {
                queuedPhaseCount.decrementAndGet();
            }
            if (removed != null) {
                inflightKvReservedTotal.addAndGet(-removed.kvTokens());
                inflightExpectedKvReservedTotal.addAndGet(-removed.expectedKvTokens());
                admissionVersion.incrementAndGet();
            }
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
            List<Long> victimIds,
            long incomingRequestId, long kvTokens, long expectedKvTokens,
            int priority, long deadlineMs,
            long expectedAdmissionVersion) {
        admissionLock.lock();
        try {
            if (admissionVersion.get() != expectedAdmissionVersion) {
                return ReleaseReserveResult.VERSION_MISMATCH;
            }
            for (Long victimId : victimIds) {
                RequestInflight victim = inflightRequests.get(victimId);
                if (victim == null || !queuedPhase.contains(victimId)
                        || preemptionClaims.containsKey(victimId)) {
                    return ReleaseReserveResult.VICTIM_GONE;
                }
            }
            for (Long victimId : victimIds) {
                release(victimId);
            }
            reserve(incomingRequestId, kvTokens, expectedKvTokens, priority, deadlineMs);
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
    public boolean releaseIfHeld(long requestId) {
        admissionLock.lock();
        try {
            RequestInflight held = inflightRequests.get(requestId);
            if (held == null || !queuedPhase.contains(requestId)
                    || preemptionClaims.containsKey(requestId)) {
                return false;
            }
            release(requestId);
            return true;
        } finally {
            admissionLock.unlock();
        }
    }

    /** Result of {@link #tryReleaseVictimsIfHeldAndReserveIncoming}. */
    public record PresenceEvictionOutcome(boolean success, List<Long> freedVictimIds) {
    }

    /**
     * Presence-guarded decode eviction commit (redesign N3 §3.4,
     * {@code autoTpmVictimGuardMode=victim_presence}): under the admission
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
            int priority, long deadlineMs) {
        admissionLock.lock();
        try {
            List<Long> freed = new ArrayList<>(victimIds.size());
            for (Long victimId : victimIds) {
                if (releaseIfHeld(victimId)) {
                    freed.add(victimId);
                }
            }
            if (freed.size() < victimIds.size()) {
                return new PresenceEvictionOutcome(false, List.copyOf(freed));
            }
            reserve(incomingRequestId, kvTokens, expectedKvTokens, priority, deadlineMs);
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
    public Map<Long, RequestInflight> reservedView() {
        admissionLock.lock();
        try {
            return Map.copyOf(inflightRequests);
        } finally {
            admissionLock.unlock();
        }
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
                    confirmed.add(new ConfirmedTaskView(requestId, task.priority(), task.deadlineMs(),
                            task.kvTokens(), task.phase(), task.priorityKnown(),
                            preemptionClaims.containsKey(requestId))));
            return new LayeredAdmissionView(admissionVersion.get(),
                    Map.copyOf(inflightRequests), List.copyOf(confirmed),
                    java.util.Set.copyOf(queuedPhase),
                    Set.copyOf(preemptionClaims.keySet()));
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
                                    long deadlineMs,
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
            long incomingDeadlineMs,
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
                    || trackedConfirmed.containsKey(incomingRequestId)) {
                return PreemptionBeginResult.INCOMING_ALREADY_RESERVED;
            }

            Map<Long, ClaimOwner> owners = new HashMap<>();
            for (Long victimId : victimIds) {
                if (preemptionClaims.containsKey(victimId)) {
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
                    incomingPriority, incomingDeadlineMs);
            for (Map.Entry<Long, ClaimOwner> entry : owners.entrySet()) {
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
     * deletes the victim accounting; duplicate observations are
     * a token-fenced no-op.
     */
    public boolean settlePriorityCanceled(long attemptToken, long requestId) {
        admissionLock.lock();
        try {
            PreemptionClaim claim = preemptionClaims.get(requestId);
            if (claim == null || claim.attemptToken != attemptToken
                    || (claim.state != ClaimState.CANCEL_REQUESTED
                        && claim.state != ClaimState.CANCEL_UNKNOWN)) {
                return false;
            }
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
                confirmedRunningCount = Math.max(0, confirmedRunningCount - 1);
            }
            if (claim.kvHeldAfterWorkerRelease) {
                priorityPreemptionHeldKv.addAndGet(-claim.hardKvTokens);
                claim.kvHeldAfterWorkerRelease = false;
            }
            claim.state = ClaimState.CANCELED_SETTLED;
            priorityCanceledTombstones.put(requestId, System.currentTimeMillis());
            // A completion can arrive after the incoming attempt has already
            // timed out and released its provisional reservation.  In that
            // case there is no later commit/abort pass to discard the claim;
            // the typed CANCELED observation is still authoritative and must
            // finish the victim without leaving a permanent accounting fence.
            if (!preemptionAttempts.containsKey(attemptToken)) {
                preemptionClaims.remove(requestId);
            }
            admissionVersion.incrementAndGet();
            return true;
        } finally {
            admissionLock.unlock();
        }
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
        admissionLock.lock();
        try {
            EndpointPreemptionAttempt attempt = preemptionAttempts.remove(attemptToken);
            if (attempt == null) {
                return;
            }
            release(attempt.incomingRequestId);
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
        } finally {
            admissionLock.unlock();
        }
    }

    /** Fresh active status is the only path that reopens a NOT_FOUND_STALE victim. */
    public boolean reconcilePriorityVictimActive(long requestId) {
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
    public boolean reconcilePriorityVictimFinished(long requestId) {
        admissionLock.lock();
        try {
            PreemptionClaim claim = preemptionClaims.get(requestId);
            if (claim == null || (claim.state != ClaimState.NOT_FOUND_STALE
                    && claim.state != ClaimState.CANCEL_UNKNOWN)) {
                return false;
            }
            if (claim.owner == ClaimOwner.ENGINE_CONFIRMED) {
                trackedConfirmed.remove(requestId);
                confirmedRunningCount = Math.max(0, confirmedRunningCount - 1);
                releaseHeldKv(claim);
            } else {
                RequestInflight removed = inflightRequests.remove(requestId);
                if (removed != null) {
                    inflightKvReservedTotal.addAndGet(-removed.kvTokens());
                    inflightExpectedKvReservedTotal.addAndGet(-removed.expectedKvTokens());
                }
                if (queuedPhase.remove(requestId)) {
                    queuedPhaseCount.decrementAndGet();
                }
            }
            preemptionClaims.remove(requestId);
            admissionVersion.incrementAndGet();
            return true;
        } finally {
            admissionLock.unlock();
        }
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
    }

    private void doCalibrate(Map<String, TaskInfo> runningTaskInfo, Map<String, TaskInfo> finishedTaskInfo) {
        this.reportedKvAvailable.set(status.getAvailableKvCacheTokens().get());
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
                        && !priorityCanceledTombstones.containsKey(requestId)) {
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
        this.confirmedRunningCount = actualConfirmed + syntheticallyHeldSlots;

        // Keep claimed entries discoverable even if Decode reports them
        // finished first.  Unclaimed entries follow ordinary calibration.
        trackedConfirmed.entrySet().removeIf(entry ->
                !confirmedNow.contains(entry.getKey())
                        && !preemptionClaims.containsKey(entry.getKey()));

        if (finishedTaskInfo != null) {
            for (TaskInfo task : finishedTaskInfo.values()) {
                long requestId = task.getRequestId();
                if (priorityCanceledTombstones.containsKey(requestId)
                        || preemptionClaims.containsKey(requestId)) {
                    continue;
                }
                trackedConfirmed.remove(requestId);
                RequestInflight removed = inflightRequests.remove(requestId);
                if (removed != null) {
                    inflightKvReservedTotal.addAndGet(-removed.kvTokens());
                    inflightExpectedKvReservedTotal.addAndGet(-removed.expectedKvTokens());
                }
            }
        }

        // N2: keep the queued-phase set consistent with the reserved entries
        // (calibrate removes confirmed/finished entries directly, bypassing
        // release()). Drop stale queued ids one-by-one so the O(1) counter
        // stays in sync (PR-C).
        java.util.Iterator<Long> queuedIt = queuedPhase.iterator();
        while (queuedIt.hasNext()) {
            Long requestId = queuedIt.next();
            if (!inflightRequests.containsKey(requestId)) {
                queuedIt.remove();
                queuedPhaseCount.decrementAndGet();
            }
        }
    }

    /**
     * Register / refresh one engine-confirmed task in the layered registry:
     * {@code KV_ALLOCATED} → accepted layer, {@code RUNNING} → running layer.
     * Priority/deadline are inherited from the shadow entry removed this
     * round; when the WorkerStatus report precedes the reserve (or the shadow
     * entry expired) they are unknown here and fall back to the defaults
     * (priority 50, no deadline). KV is approximated by
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
            long deadlineMs = removed != null ? removed.deadlineMs() : 0;
            long kvTokens = Math.max(0, task.getInputLength());
            trackedConfirmed.put(task.getRequestId(),
                    new ConfirmedTask(priority, deadlineMs, kvTokens, layer, now, priorityKnown));
        } else {
            tracked.refresh(layer, now);
        }
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
     * Real KV used: engine-reported used (total - available) + local inflight reservations.
     */
    public long realKvUsed() {
        long totalCap = status.getTotalKvCacheTokens().get();
        long avail = status.getAvailableKvCacheTokens().get();
        long reportedUsed = totalCap > 0 ? Math.max(0, totalCap - avail) : 0;
        return reportedUsed + inflightKvReserved();
    }

    /**
     * Real KV available: engine-reported available - local inflight hard reservations.
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
                - inflightHardKvReserved() - priorityPreemptionHeldKv.get());
    }

    // ==================== Metrics ====================

    /**
     * Report per-worker decode inflight metrics via the given reporter.
     * Called periodically by {@link org.flexlb.balance.scheduler.FlexlbBatchScheduler}.
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
        return evictExpiredRequests(ttlMs, 0, requestId -> false);
    }

    /**
     * TTL eviction with a hard age cap: entries older than
     * {@code hardMaxAgeMs} are force-evicted even when a priority-preemption
     * claim exempts them from the regular TTL pass — a claim whose Prefill
     * CANCELED settlement never arrives (zombie cancel overlay in the engine
     * report) must not pin decode accounting forever. Entries still owned by
     * the batch scheduler are skipped ({@code schedulerOwnsRequest}) so the
     * scheduler's own lifecycle/fence handling stays authoritative.
     * {@code hardMaxAgeMs <= 0} disables the cap.
     */
    public int evictExpiredRequests(long ttlMs, long hardMaxAgeMs,
                                    LongPredicate schedulerOwnsRequest) {
        admissionLock.lock();
        try {
            // A priority claim is a stronger accounting owner than generic
            // TTL cleanup. In particular, an ENGINE_MAY_HAVE_SEEN shadow must
            // remain charged until typed Prefill CANCELED or explicit
            // NOT_FOUND/UNKNOWN reconciliation settles the claim.
            int evicted = requestEvictor.evictExpired(
                    ttlMs, requestId -> !preemptionClaims.containsKey(requestId));
            // Hard age cap pass: whatever survived above (claim-exempted) but
            // exceeds the cap is force-released with full counter cleanup.
            if (hardMaxAgeMs > 0) {
                long nowMs = System.currentTimeMillis();
                for (Map.Entry<Long, RequestInflight> entry : inflightRequests.entrySet()) {
                    long requestId = entry.getKey();
                    long ageMs = nowMs - entry.getValue().createdAtMs();
                    if (ageMs <= hardMaxAgeMs || schedulerOwnsRequest.test(requestId)) {
                        continue;
                    }
                    RequestInflight removed = inflightRequests.remove(requestId);
                    if (removed == null) {
                        continue;
                    }
                    inflightKvReservedTotal.addAndGet(-removed.kvTokens());
                    inflightExpectedKvReservedTotal.addAndGet(-removed.expectedKvTokens());
                    // Settle the zombie claim: return any KV still held behind
                    // the never-arriving CANCELED fence, then drop the claim.
                    PreemptionClaim claim = preemptionClaims.remove(requestId);
                    if (claim != null) {
                        releaseHeldKv(claim);
                    }
                    evicted++;
                    logger.warn("event=inflight_hard_age_eviction role=DECODE endpoint={} "
                                    + "request_id={} age_ms={} hard_max_age_ms={} created_at_ms={} "
                                    + "kv_tokens={} expected_kv_tokens={} priority={} deadline_ms={} "
                                    + "phase={} preemption_claim={} queued_phase={}",
                            getIp(), requestId, ageMs, hardMaxAgeMs, removed.createdAtMs(),
                            removed.kvTokens(), removed.expectedKvTokens(), removed.priority(),
                            removed.deadlineMs(), removed.phase(), claim != null,
                            queuedPhase.contains(requestId));
                }
            }
            // Drop stale queued ids one-by-one so the O(1) counter stays in
            // sync (PR-C) — evictExpired may have removed inflight entries.
            java.util.Iterator<Long> queuedEvictIt = queuedPhase.iterator();
            while (queuedEvictIt.hasNext()) {
                Long requestId = queuedEvictIt.next();
                if (!inflightRequests.containsKey(requestId)) {
                    queuedEvictIt.remove();
                    queuedPhaseCount.decrementAndGet();
                }
            }
            long cutoff = System.currentTimeMillis() - ttlMs;
            boolean trackedPurged = trackedConfirmed.values()
                    .removeIf(task -> task.lastSeenMs() < cutoff);
            boolean canceledTombstonesPurged = priorityCanceledTombstones.entrySet()
                    .removeIf(entry -> entry.getValue() < cutoff);
            if (evicted > 0 || trackedPurged || canceledTombstonesPurged) {
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
    public void markQueuedPhase(long requestId) {
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
    public DispatchClaimResult tryClaimEngineDispatch(long requestId,
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
    public boolean tryMarkEngineMayHaveSeen(long requestId) {
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
    public boolean isConfirmedTracked(long requestId) {
        return trackedConfirmed.containsKey(requestId);
    }

    /**
     * Whether the engine has confirmed this request in the layered registry
     * (KV_ALLOCATED / RUNNING) — the decode-side visibility check for the
     * scheduler's post-ACK inflight audit (F1). Mirrors
     * {@link #isConfirmedTracked} under the audit's engine-confirmed
     * vocabulary.
     */
    public boolean isEngineConfirmed(long requestId) {
        return isConfirmedTracked(requestId);
    }

    @Override
    public long getLoadMetric() {
        return getTotalLoad();
    }

    /**
     * Mutable layered-registry entry for one engine-confirmed request
     * (Phase 5). Identity fields (priority / deadline / KV estimate) are fixed
     * at first sight; {@code phase} and {@code lastSeenMs} are volatile and
     * only mutated under {@link #admissionLock} by calibration.
     */
    static final class ConfirmedTask {

        private final int priority;
        private final long deadlineMs;
        private final long kvTokens;
        private final boolean priorityKnown;
        private volatile DecodeTaskPhase phase;
        private volatile long lastSeenMs;

        ConfirmedTask(int priority, long deadlineMs, long kvTokens,
                      DecodeTaskPhase layer, long now, boolean priorityKnown) {
            this.priority = priority;
            this.deadlineMs = deadlineMs;
            this.kvTokens = kvTokens;
            this.priorityKnown = priorityKnown;
            this.phase = layer;
            this.lastSeenMs = now;
        }

        int priority() { return priority; }
        long deadlineMs() { return deadlineMs; }
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
        CANCEL_UNKNOWN
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
