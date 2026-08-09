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
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Decode-side endpoint with Auto-TPM shadow admission accounting.
 *
 * <p><b>Layered view (Phase 5):</b> {@code inflightRequests} only ever holds
 * {@code RESERVED_NOT_ACCEPTED} shadow entries; engine-confirmed requests are
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
     * Reserved entries whose request is still sitting in a prefill queue —
     * committed by the scheduler but not yet dispatched to the engine (N2,
     * plan-commit redesign). These reservations keep protecting KV against
     * oversell, but must not count against the decode concurrency limit:
     * counting them produced the shadow-saturation 8400 storm (root cause C —
     * queued reservations saturating {@code getTotalLoad()} while the engine
     * sat idle). Marked at plan commit ({@code markQueuedPhase}), unmarked at
     * batch dispatch ({@code markDispatchedPhase}); release/calibrate prune
     * it alongside {@code inflightRequests}. Legacy/DIRECT paths never mark,
     * so their accounting is unchanged.
     */
    private final java.util.Set<Long> queuedPhase = ConcurrentHashMap.newKeySet();

    /**
     * O(1) mirror of {@code |queuedPhase ∩ inflightRequests|} (PR-C):
     * incremented when a reservation is marked queued and decremented when
     * it is dispatched / released / calibrated out, so {@link #getEngineLoad}
     * avoids the per-call O(n) scan of the legacy formula. Read lock-free;
     * written under {@link #admissionLock} (except {@link #markDispatchedPhase}
     * which relies on the atomic add/remove return value).
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
     * still a {@code RESERVED_NOT_ACCEPTED} entry, then release all victims
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
                if (victim == null || victim.phase() != DecodeTaskPhase.RESERVED_NOT_ACCEPTED) {
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
     * {@code RESERVED_NOT_ACCEPTED} shadow entry. Returns {@code false} —
     * touching nothing — when the reservation is gone (already released) or
     * has been folded into the confirmed layer (the engine owns the request;
     * contract 5.3 forbids terminal operations on dispatched requests).
     */
    public boolean releaseIfHeld(long requestId) {
        admissionLock.lock();
        try {
            RequestInflight held = inflightRequests.get(requestId);
            if (held == null || held.phase() != DecodeTaskPhase.RESERVED_NOT_ACCEPTED) {
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
     * {@code RESERVED_NOT_ACCEPTED} reservation ({@link #releaseIfHeld}).
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
                            task.kvTokens(), task.phase(), task.cancelRequested())));
            return new LayeredAdmissionView(admissionVersion.get(),
                    Map.copyOf(inflightRequests), List.copyOf(confirmed),
                    java.util.Set.copyOf(queuedPhase));
        } finally {
            admissionLock.unlock();
        }
    }

    /** Atomic (admissionVersion, reserved, confirmed, queued) tuple — see {@link #layeredAdmissionView()}. */
    public record LayeredAdmissionView(long admissionVersion,
                                       Map<Long, RequestInflight> reserved,
                                       List<ConfirmedTaskView> confirmed,
                                       java.util.Set<Long> queued) {
    }

    /** Immutable point-in-time view of one layered-registry entry. */
    public record ConfirmedTaskView(long requestId,
                                    int priority,
                                    long deadlineMs,
                                    long kvTokens,
                                    DecodeTaskPhase phase,
                                    boolean cancelRequested) {
    }

    /**
     * Atomic begin of an accepted-eviction commit (Phase 5, design doc
     * 11.5/17.2): under the admission lock — validate the version, validate
     * every reserved victim is still {@code RESERVED_NOT_ACCEPTED} and every
     * accepted victim is still an {@code ACCEPTED_NOT_RUNNING} layered entry
     * without a pending cancel, then release the reserved victims and mark
     * the accepted victims {@code CANCEL_REQUESTED} (dedup against repeated
     * cancels). Validate-first: on any failure nothing is applied.
     *
     * <p>The incoming request is deliberately NOT reserved here — cancel is
     * only an intent injection and the release must be confirmed by a later
     * WorkerStatus report before the incoming may take the freed capacity
     * (iron rule 4: a cancel timeout never assumes the resources are free).
     */
    public ReleaseReserveResult tryBeginAcceptedEviction(List<Long> reservedVictimIds,
                                                         List<Long> acceptedVictimIds,
                                                         long expectedAdmissionVersion) {
        admissionLock.lock();
        try {
            if (admissionVersion.get() != expectedAdmissionVersion) {
                return ReleaseReserveResult.VERSION_MISMATCH;
            }
            return beginAcceptedEvictionValidated(reservedVictimIds, acceptedVictimIds);
        } finally {
            admissionLock.unlock();
        }
    }

    /**
     * Presence-guarded variant of {@link #tryBeginAcceptedEviction} (redesign
     * N3 §3.4, {@code autoTpmVictimGuardMode=victim_presence}): the same
     * all-or-nothing victim validation — every reserved victim still
     * {@code RESERVED_NOT_ACCEPTED}, every accepted victim still an
     * {@code ACCEPTED_NOT_RUNNING} layered entry without a pending cancel —
     * but without the admission-version check, so unrelated admission-state
     * mutations no longer abort the commit. The cancel-wait-confirm flow
     * (iron rule 4) is unchanged.
     */
    public ReleaseReserveResult tryBeginAcceptedEvictionPresent(List<Long> reservedVictimIds,
                                                                List<Long> acceptedVictimIds) {
        admissionLock.lock();
        try {
            return beginAcceptedEvictionValidated(reservedVictimIds, acceptedVictimIds);
        } finally {
            admissionLock.unlock();
        }
    }

    /** Validate-first accepted-eviction begin; caller holds {@link #admissionLock}. */
    private ReleaseReserveResult beginAcceptedEvictionValidated(List<Long> reservedVictimIds,
                                                                List<Long> acceptedVictimIds) {
        for (Long victimId : reservedVictimIds) {
            RequestInflight victim = inflightRequests.get(victimId);
            if (victim == null || victim.phase() != DecodeTaskPhase.RESERVED_NOT_ACCEPTED) {
                return ReleaseReserveResult.VICTIM_GONE;
            }
        }
        for (Long victimId : acceptedVictimIds) {
            ConfirmedTask victim = trackedConfirmed.get(victimId);
            if (victim == null || victim.phase() != DecodeTaskPhase.ACCEPTED_NOT_RUNNING
                    || victim.cancelRequested()) {
                return ReleaseReserveResult.VICTIM_GONE;
            }
        }
        for (Long victimId : reservedVictimIds) {
            release(victimId);
        }
        for (Long victimId : acceptedVictimIds) {
            trackedConfirmed.get(victimId).markCancelRequested();
        }
        admissionVersion.incrementAndGet();
        return ReleaseReserveResult.SUCCESS;
    }

    /**
     * Whether the request is still present in the confirmed (accepted or
     * running) layered registry. Turning {@code false} is the release
     * confirmation the accepted-eviction wait window polls for — calibrate
     * drops the entry when the next WorkerStatus report no longer lists the
     * request as confirmed (or lists it as finished).
     */
    public boolean isConfirmedTracked(long requestId) {
        return trackedConfirmed.containsKey(requestId);
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

        // Phase 1: process running requests — KV_ALLOCATED or RUNNING means the engine
        // has taken ownership, so we can release our inflight reservation.
        //
        // Two-pass to avoid transient undercount: if we remove from inflightRequests before
        // updating confirmedRunningCount, a task transitioning from inflight to confirmed
        // is briefly counted in neither, which could allow oversubscription. By updating
        // the count first and removing second, the transient window overcounts (conservative).
        int kvAllocatedRequests = 0;
        if (runningTaskInfo != null) {
            // First pass: count and update confirmedRunningCount
            for (TaskInfo task : runningTaskInfo.values()) {
                TaskPhase phase = task.getPhase();
                if (phase == TaskPhase.KV_ALLOCATED || phase == TaskPhase.RUNNING) {
                    kvAllocatedRequests++;
                }
            }
        }
        this.confirmedRunningCount = kvAllocatedRequests;

        // Second pass: remove confirmed tasks from inflightRequests, and sync
        // the layered registry (Phase 5). The shadow accounting transfer is
        // byte-for-byte the Phase 4 behavior; the layered registry is a pure
        // metadata addition on top of it.
        java.util.Set<Long> confirmedNow = new java.util.HashSet<>();
        long now = System.currentTimeMillis();
        if (runningTaskInfo != null) {
            for (TaskInfo task : runningTaskInfo.values()) {
                TaskPhase phase = task.getPhase();
                if (phase == TaskPhase.KV_ALLOCATED || phase == TaskPhase.RUNNING) {
                    RequestInflight removed = inflightRequests.remove(task.getRequestId());
                    if (removed != null) {
                        inflightKvReservedTotal.addAndGet(-removed.kvTokens());
                        inflightExpectedKvReservedTotal.addAndGet(-removed.expectedKvTokens());
                    }
                    confirmedNow.add(task.getRequestId());
                    trackConfirmed(task, phase, removed, now);
                }
            }
        }
        // Confirmed entries no longer reported (finished, or regressed out of
        // the confirmed phases) leave the layered registry — this is also the
        // release-confirmation signal for the accepted-eviction wait window.
        trackedConfirmed.keySet().retainAll(confirmedNow);

        // Phase 2: process finished non-success requests
        if (finishedTaskInfo != null) {
            for (TaskInfo task : finishedTaskInfo.values()) {
                if (task.getErrorCode() != 0) {
                    trackedConfirmed.remove(task.getRequestId());
                    RequestInflight removed = inflightRequests.remove(task.getRequestId());
                    if (removed != null) {
                        inflightKvReservedTotal.addAndGet(-removed.kvTokens());
                        inflightExpectedKvReservedTotal.addAndGet(-removed.expectedKvTokens());
                    } else {
                        logger.debug("Decode calibrate: finished failed request reqId={} not in inflight, error={}",
                                task.getRequestId(), task.getErrorMessage());
                    }
                }
            }

            // Phase 3: process finished success requests
            for (TaskInfo task : finishedTaskInfo.values()) {
                if (task.getErrorCode() == 0) {
                    trackedConfirmed.remove(task.getRequestId());
                    RequestInflight removed = inflightRequests.remove(task.getRequestId());
                    if (removed != null) {
                        inflightKvReservedTotal.addAndGet(-removed.kvTokens());
                        inflightExpectedKvReservedTotal.addAndGet(-removed.expectedKvTokens());
                    }
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
        if (tracked == null) {
            int priority = removed != null ? removed.priority() : RequestInflight.DEFAULT_PRIORITY;
            long deadlineMs = removed != null ? removed.deadlineMs() : 0;
            long kvTokens = Math.max(0, task.getInputLength());
            trackedConfirmed.put(task.getRequestId(),
                    new ConfirmedTask(priority, deadlineMs, kvTokens, layer, now));
        } else {
            tracked.refresh(layer, now);
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
        return Math.max(0, reportedKvAvailable.get() - inflightHardKvReserved());
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
        admissionLock.lock();
        try {
            int evicted = requestEvictor.evictExpired(ttlMs);
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
            if (evicted > 0 || trackedPurged) {
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
     * {@link #markDispatchedPhase}, {@link #reserve}, {@link #release},
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

    /**
     * Mark a queued request as dispatched to the engine (N2): from this point
     * its reservation counts against the engine concurrency again, until
     * calibrate confirms it. Idempotent.
     */
    public void markDispatchedPhase(long requestId) {
        if (queuedPhase.remove(requestId)) {
            queuedPhaseCount.decrementAndGet();
        }
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

    @Override
    public long getLoadMetric() {
        return getTotalLoad();
    }

    /**
     * Mutable layered-registry entry for one engine-confirmed request
     * (Phase 5). Identity fields (priority / deadline / KV estimate) are fixed
     * at first sight; {@code phase}, {@code cancelRequested} and
     * {@code lastSeenMs} are volatile and only mutated under
     * {@link #admissionLock} by calibrate / accepted-eviction begin.
     */
    static final class ConfirmedTask {

        private final int priority;
        private final long deadlineMs;
        private final long kvTokens;
        private volatile DecodeTaskPhase phase;
        private volatile boolean cancelRequested;
        private volatile long lastSeenMs;

        ConfirmedTask(int priority, long deadlineMs, long kvTokens,
                      DecodeTaskPhase layer, long now) {
            this.priority = priority;
            this.deadlineMs = deadlineMs;
            this.kvTokens = kvTokens;
            this.phase = layer;
            this.lastSeenMs = now;
        }

        int priority() { return priority; }
        long deadlineMs() { return deadlineMs; }
        long kvTokens() { return kvTokens; }
        DecodeTaskPhase phase() { return phase; }
        boolean cancelRequested() { return cancelRequested; }
        long lastSeenMs() { return lastSeenMs; }

        void markCancelRequested() {
            this.cancelRequested = true;
        }

        /** Refresh layer membership and liveness on every calibrate round. */
        void refresh(DecodeTaskPhase layer, long now) {
            this.phase = layer;
            this.lastSeenMs = now;
        }
    }

}
