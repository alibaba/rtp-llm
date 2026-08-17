package org.flexlb.balance.scheduler;

import org.flexlb.balance.scheduler.priority.PrefillQueueSnapshot;
import org.flexlb.balance.scheduler.priority.QueuedRequestSnapshot;
import org.flexlb.util.Prioritized;
import org.flexlb.util.PriorityOrdering;

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
     * <p>With {@code flexlbSnapshotSortOutsideLockEnabled} (task61 M2, default
     * on) only the membership copy and the version capture happen under the
     * queue lock; the O(n log n) sort runs on the thread-confined copy outside
     * it. Correctness: the copy and the version are captured atomically under
     * the same lock hold, sorting is a pure function of the copy, so the
     * "version unchanged ⇒ queue content unchanged" invariant and the output
     * order are both preserved bit-for-bit.
     */
    public PrefillQueueSnapshot snapshot() {
        if (ctx.cfg().isFlexlbSnapshotSortOutsideLockEnabled()) {
            List<BatchItem> queued;
            long version;
            ctx.queueLock().lock();
            try {
                queued = ctx.copiedItems();
                version = batcher.queueVersion();
            } finally {
                ctx.queueLock().unlock();
            }
            queued.sort(ctx.queueOrder());
            return new PrefillQueueSnapshot(ctx.key(), version,
                    ctx.cfg().getFlexlbBatchQueueMaxSize(), toItemSnapshots(queued));
        }
        ctx.queueLock().lock();
        try {
            // Only live queue members are actionable eviction victims. A
            // staged callback member remains capacity-charged, but cannot be
            // removed by the versioned queue mutation APIs.
            List<BatchItem> queued = ctx.sortedItems();
            return new PrefillQueueSnapshot(ctx.key(), batcher.queueVersion(),
                    ctx.cfg().getFlexlbBatchQueueMaxSize(), toItemSnapshots(queued));
        } finally {
            ctx.queueLock().unlock();
        }
    }

    private static List<QueuedRequestSnapshot> toItemSnapshots(List<BatchItem> queued) {
        List<QueuedRequestSnapshot> items = new ArrayList<>(queued.size());
        for (BatchItem item : queued) {
            items.add(new QueuedRequestSnapshot(
                    item.requestId(), item.priority(), item.deadlineMs(),
                    item.enqueuedAtMs(), item.seqLen(), item.hitCache(),
                    QueuedRequestSnapshot.PREFILL_QUEUED));
        }
        return items;
    }

    /**
     * Estimated queue wait for an incoming request (design doc 8.4):
     * {@code itemsAhead → batchCyclesAhead → estimatedWaitMs}, using the
     * dispatch-interval sliding average as the per-cycle cost and the head's
     * remaining window as the partial first cycle.
     *
     * <p>na130_4 depth term: the jump estimate above only counts items
     * ordered ahead of the probe (jump-in semantics), so a high-priority
     * request facing a queue already pinned at its cap reports a near-zero
     * wait and the slow engine keeps winning the routing score. When
     * {@code flexlbQueueDepthPenaltyEnabled} (default on), the estimate
     * additionally computes {@code depthWait = (queueSize / maxBatchSize) ×
     * intervalEMA × flexlbQueueDepthPenaltyFactor} — the drain horizon of the
     * full queue — and returns {@code max(jumpWait, depthWait)}, so a probe
     * can never look cheaper than the queue it jumps into. With the gate off
     * the legacy jump-only value is returned unchanged.
     *
     * <p>task61 L2: the previous implementation sorted the whole queue under
     * the lock only to take the ordered head and count the items ahead. The
     * count is order-independent, and the head is {@code queue.peek()} — the
     * backing {@link java.util.concurrent.PriorityBlockingQueue} is ordered
     * by the same comparator as {@code sortedItems()}, which on the Auto-TPM
     * path (the only caller of this estimate) is a total order thanks to the
     * {@code requestId} tie-break, so the heap root is exactly the sorted
     * head. Iterating the raw queue with the primitive comparison removes the
     * per-call O(n log n) sort and the per-item probe allocation (JFR:
     * PriorityOrdering sort CPU + ordersBefore 11.69% allocation) from the
     * queue-lock critical section that the submit path enqueues behind.
     */
    public long estimateWaitMs(int priority, long deadlineMs, long requestId) {
        long now = ctx.now();
        int itemsAhead = 0;
        int queueSize = 0;
        BatchItem head;
        ctx.queueLock().lock();
        try {
            head = ctx.peek();
            for (BatchItem item : ctx.queueItems()) {
                queueSize++;
                if (ordersBefore(item, priority, deadlineMs, now, requestId)) {
                    itemsAhead++;
                }
            }
        } finally {
            ctx.queueLock().unlock();
        }
        int maxBatchSize = Math.max(1, ctx.cfg().getFlexlbBatchSizeMax());
        long intervalMs = ctx.avgDispatchIntervalMs();
        long batchCyclesAhead = itemsAhead / maxBatchSize;
        long headRemainingWindowMs = 0;
        if (head != null) {
            long headWaitedMs = Math.max(0, now - head.enqueuedAtMs());
            headRemainingWindowMs = Math.max(0, intervalMs - headWaitedMs);
        }
        long jumpWaitMs = batchCyclesAhead * intervalMs + headRemainingWindowMs;
        if (!ctx.cfg().isFlexlbQueueDepthPenaltyEnabled()) {
            return jumpWaitMs;
        }
        // Depth term: full-queue drain horizon, independent of the probe's
        // priority. depthCycles is bounded by the queue capacity. A
        // misconfigured factor (NaN/Infinite or <= 0) is treated as the
        // default 1.0, and the product is clamped to 1<<40 ms (~13 days) so
        // an extreme factor can never wrap negative in the score sum.
        long depthCycles = queueSize / maxBatchSize;
        double factor = ctx.cfg().getFlexlbQueueDepthPenaltyFactor();
        if (!Double.isFinite(factor) || factor <= 0) {
            factor = 1.0;
        }
        long depthWaitMs = Math.min((long) (depthCycles * (double) intervalMs * factor), 1L << 40);
        return Math.max(jumpWaitMs, depthWaitMs);
    }

    /**
     * Whether a queued item is ordered before an incoming probe under
     * {@link WorkerBatcher#AUTO_TPM_QUEUE_ORDER} (priority desc → enqueue-seq
     * asc → requestId asc). The probe's enqueue-seq is its would-be arrival
     * time ({@code now}) — it has not been enqueued yet.
     *
     * <p>Delegates the priority + enqueue-seq comparison to
     * {@link PriorityOrdering#compareStrict} (task61 L1: the primitive form
     * of {@code STRICT}, replacing the temporary {@link Prioritized} probe
     * allocated per item — JFR allocation hotspot 11.69%), then breaks
     * residual ties by {@code requestId}. The {@code deadlineMs} parameter is
     * retained for call-site stability but is no longer part of the ordering
     * rule (PR-B removed the deadline key).
     */
    private static boolean ordersBefore(BatchItem item, int priority, long deadlineMs,
                                        long arrivalMs, long requestId) {
        int cmp = PriorityOrdering.compareStrict(item.priority(), item.enqueueSeq(),
                priority, arrivalMs);
        if (cmp != 0) {
            return cmp < 0;
        }
        return item.requestId() < requestId;
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
