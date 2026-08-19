package org.flexlb.balance.scheduler;

import org.flexlb.balance.scheduler.priority.PrefillQueueSnapshot;
import org.flexlb.balance.scheduler.priority.QueuedRequestSnapshot;

import java.util.ArrayList;
import java.util.List;

/**
 * Auto-TPM facade over one {@link WorkerBatcher} queue (design doc 8.2).
 *
 * <p>One instance per prefill endpoint, created by the batcher itself. It does
 * not own any state: every operation delegates to the batcher/context so the
 * queue stays the single source of truth and every mutation goes through the
 * shared queue lock and bumps {@code queueVersion} (optimistic plan
 * validation — "version unchanged ⇒ queue content unchanged").
 *
 * <p>Read operations ({@link #snapshot()}, {@link #estimateWaitMs}) capture a
 * consistent view under the queue lock; version-checked mutations
 * ({@link #tryOffer}, {@link #tryRemove}, {@link #tryReplaceVictimsWithIncoming})
 * are atomic against the version captured by a prior snapshot.
 */
public final class PrefillQueueManager {

    private final WorkerBatcher batcher;
    private final BatcherContext ctx;

    PrefillQueueManager(WorkerBatcher batcher, BatcherContext ctx) {
        this.batcher = batcher;
        this.ctx = ctx;
    }

    /** Current queue version (monotonic, bumped on every mutation). */
    public long queueVersion() {
        return batcher.queueVersion();
    }

    /** Current queue depth. */
    public int queueSize() {
        return batcher.queueSize();
    }

    /**
     * Consistent point-in-time view of the queue for eviction planning:
     * version + per-item {@link QueuedRequestSnapshot} in queue order.
     * The hard capacity reuses {@code flexlbBatchQueueMaxSize} (0 = unbounded).
     *
     * <p>task61 M2: only the membership copy and the version capture run under
     * the queue lock; the O(n log n) sort runs on the thread-confined copy
     * outside it, so the submit path's {@code offer()} never blocks behind a
     * snapshot sort. Correctness: the copy and the version are captured
     * atomically under the same lock hold, every queue mutation bumps the
     * version under that lock, and item sort fields are frozen once enqueued
     * ({@code sortKey} is set before the item enters the queue), so the
     * "version unchanged ⇒ queue content unchanged" invariant and the output
     * order are both preserved bit-for-bit.
     */
    public PrefillQueueSnapshot snapshot() {
        List<BatchItem> queued;
        long version;
        ctx.queueLock().lock();
        try {
            // Only live queue members are actionable eviction victims. A
            // staged callback member remains capacity-charged, but cannot be
            // removed by the versioned queue mutation APIs.
            queued = ctx.copiedItems();
            version = batcher.queueVersion();
        } finally {
            ctx.queueLock().unlock();
        }
        queued.sort(ctx.queueOrder());
        List<QueuedRequestSnapshot> items = new ArrayList<>(queued.size());
        for (BatchItem item : queued) {
            items.add(new QueuedRequestSnapshot(
                    item.requestId(), item.priority(), item.deadlineMs(),
                    item.enqueuedAtMs(), item.seqLen(), item.hitCache(),
                    QueuedRequestSnapshot.PREFILL_QUEUED));
        }
        return new PrefillQueueSnapshot(ctx.key(), version,
                ctx.cfg().getFlexlbBatchQueueMaxSize(), items);
    }

    /**
     * Measured queue congestion for an incoming request: the age of the
     * queue head — how long the oldest-next request has been waiting
     * ({@code now - head.enqueuedAtMs()}, 0 for an empty queue).
     *
     * <p>Design semantics (measured replaces predicted): the legacy chain
     * counted items ordered ahead of the probe ({@code ordersBefore} +
     * itemsAhead) and folded them into {@code batchCyclesAhead ×
     * dispatch-interval EMA + partial first cycle}. That chain needs a
     * per-cycle cost model that is cold-start blind — right after a master
     * restart the interval EMA falls back to the batching window and the
     * estimate collapses exactly when routing decisions matter most. The
     * head age is a direct measurement: a slow engine's head waits long,
     * so its age is large — no conversion factor to calibrate, no
     * cold-start distortion, no per-dispatch bookkeeping (the dispatch
     * interval EMA was removed together with this rewrite).
     *
     * <p>Priority-blind by design: the estimate answers "how congested is
     * this queue", not "when will this priority be served" — a P70 probe
     * sees the same congestion as a P30 probe. That is intentionally
     * conservative for routing: the 8/17 slow-engine attractor showed a
     * priority-aware jump estimate keeps looking cheap for high-priority
     * probes while the target engine drowns; a probe must never price
     * jumping into a congested queue below the queue's measured drain
     * latency.
     *
     * <p>O(1): one {@code peek()} and one subtraction under the queue lock,
     * no iteration over the queue. Clamped to {@code 1L << 40} outside the
     * lock so a pathological {@code enqueuedAtMs} cannot overflow the
     * downstream long score arithmetic.
     *
     * <p>The {@code priority}/{@code deadlineMs}/{@code requestId}
     * parameters are retained for call-site stability
     * (CostBasedPrefillStrategy → {@code PrefillEndpoint.batcherEstimatedWaitMs}
     * → this method) and are no longer part of the estimate.
     */
    public long estimateWaitMs(int priority, long deadlineMs, long requestId) {
        long now = ctx.now();
        long queueAgeMs;
        ctx.queueLock().lock();
        try {
            BatchItem head = ctx.peek();
            queueAgeMs = head != null ? Math.max(0, now - head.enqueuedAtMs()) : 0L;
        } finally {
            ctx.queueLock().unlock();
        }
        return Math.min(queueAgeMs, 1L << 40);
    }

    /**
     * Version-checked enqueue: applies only if the queue version still equals
     * {@code expectedVersion} (used by commit paths built on a snapshot).
     */
    public boolean tryOffer(BatchItem item, long expectedVersion) {
        return batcher.tryOfferAtVersion(item, expectedVersion);
    }

    /**
     * Version-checked removal of the given requests.
     *
     * @return removed items, or {@code null} on version mismatch
     */
    public List<BatchItem> tryRemove(List<Long> requestIds, long expectedVersion, String reason) {
        return batcher.tryRemoveAtVersion(requestIds, expectedVersion, reason);
    }

    /**
     * Version-agnostic idempotent removal (PR-D §2.7): removes the given
     * request from the queue without a version precondition. Used by
     * {@code AdmissionLease.close()} for deadline-timeout cleanup where the
     * snapshot version is long stale. No-op when the item is not queued
     * (already dispatched / evicted / removed).
     */
    public void tryRemove(long requestId, String reason) {
        batcher.tryRemoveNoVersion(List.of(requestId), reason);
    }

    /**
     * Atomic victim replacement (design doc 17.2): under the queue lock,
     * validate the version, remove all victims, enqueue the incoming item.
     */
    public ReplaceOutcome tryReplaceVictimsWithIncoming(List<Long> victimIds,
                                                        BatchItem incoming,
                                                        long expectedVersion) {
        return batcher.tryReplaceVictimsWithIncoming(victimIds, incoming, expectedVersion);
    }

    /**
     * Atomic victim replacement with victim-level presence guard (redesign N3
     * §3.4, {@code autoTpmVictimGuardMode=victim_presence}): no version
     * check — any missing victim aborts with a zero-side-effect
     * {@code VICTIM_GONE} carrying the missing ids.
     */
    public ReplaceOutcome tryReplaceVictimsPresent(List<Long> victimIds, BatchItem incoming) {
        return batcher.tryReplaceVictimsPresent(victimIds, incoming);
    }

    // ==================== Replace outcome ====================

    /**
     * Result of {@link #tryReplaceVictimsWithIncoming} /
     * {@link #tryReplaceVictimsPresent}. {@code removed} holds the victims
     * actually taken out of the queue — non-empty on success and on the
     * (defensively handled) partial failure, where victims are never
     * re-inserted (design doc 9.5) and must be driven to a terminal state by
     * the caller. {@code VICTIM_GONE} (presence guard only) is a
     * zero-side-effect abort carrying the victims no longer queued.
     */
    public static final class ReplaceOutcome {

        public enum Status { SUCCESS, VERSION_MISMATCH, PARTIAL_FAILURE, VICTIM_GONE }

        private final Status status;
        private final List<BatchItem> removed;
        private final List<Long> missingVictimIds;

        private ReplaceOutcome(Status status, List<BatchItem> removed, List<Long> missingVictimIds) {
            this.status = status;
            this.removed = List.copyOf(removed);
            this.missingVictimIds = List.copyOf(missingVictimIds);
        }

        static ReplaceOutcome success(List<BatchItem> removed) {
            return new ReplaceOutcome(Status.SUCCESS, removed, List.of());
        }

        static ReplaceOutcome versionMismatch() {
            return new ReplaceOutcome(Status.VERSION_MISMATCH, List.of(), List.of());
        }

        static ReplaceOutcome partialFailure(List<BatchItem> removed) {
            return new ReplaceOutcome(Status.PARTIAL_FAILURE, removed, List.of());
        }

        static ReplaceOutcome victimGone(List<Long> missingVictimIds) {
            return new ReplaceOutcome(Status.VICTIM_GONE, List.of(), missingVictimIds);
        }

        public Status status() {
            return status;
        }

        public boolean isSuccess() {
            return status == Status.SUCCESS;
        }

        public boolean isVersionMismatch() {
            return status == Status.VERSION_MISMATCH;
        }

        public boolean isPartialFailure() {
            return status == Status.PARTIAL_FAILURE;
        }

        public boolean isVictimGone() {
            return status == Status.VICTIM_GONE;
        }

        /** Victims removed from the queue (eviction order). */
        public List<BatchItem> removed() {
            return removed;
        }

        /** Victims no longer queued (presence-guard abort only). */
        public List<Long> missingVictimIds() {
            return missingVictimIds;
        }
    }
}
