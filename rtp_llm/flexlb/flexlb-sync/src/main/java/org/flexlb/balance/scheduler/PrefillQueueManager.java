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
     */
    public PrefillQueueSnapshot snapshot() {
        ctx.queueLock().lock();
        try {
            List<BatchItem> queued = ctx.sortedItems();
            List<QueuedRequestSnapshot> items = new ArrayList<>(queued.size());
            for (BatchItem item : queued) {
                items.add(new QueuedRequestSnapshot(
                        item.requestId(), item.priority(), item.deadlineMs(),
                        item.enqueuedAtMs(), item.seqLen(), item.hitCache(),
                        QueuedRequestSnapshot.PREFILL_QUEUED));
            }
            return new PrefillQueueSnapshot(ctx.key(), batcher.queueVersion(),
                    ctx.cfg().getFlexlbBatchQueueMaxSize(), items);
        } finally {
            ctx.queueLock().unlock();
        }
    }

    /**
     * Estimated queue wait for an incoming request (design doc 8.4):
     * {@code itemsAhead → batchCyclesAhead → estimatedWaitMs}, using the
     * dispatch-interval sliding average as the per-cycle cost and the head's
     * remaining window as the partial first cycle.
     */
    public long estimateWaitMs(int priority, long deadlineMs, long requestId) {
        long now = ctx.now();
        int itemsAhead = 0;
        BatchItem head = null;
        ctx.queueLock().lock();
        try {
            for (BatchItem item : ctx.sortedItems()) {
                if (head == null) {
                    head = item;
                }
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
        return batchCyclesAhead * intervalMs + headRemainingWindowMs;
    }

    /**
     * Whether a queued item is ordered before an incoming probe under
     * {@link WorkerBatcher#AUTO_TPM_QUEUE_ORDER} (priority desc → enqueue-seq
     * asc → requestId asc). The probe's enqueue-seq is its would-be arrival
     * time ({@code now}) — it has not been enqueued yet.
     *
     * <p>Delegates the priority + enqueue-seq comparison to
     * {@link PriorityOrdering#STRICT} via a temporary {@link Prioritized}
     * view of the probe, then breaks residual ties by {@code requestId}.
     * The {@code deadlineMs} parameter is retained for call-site stability
     * but is no longer part of the ordering rule (PR-B removed the deadline
     * key).
     */
    private static boolean ordersBefore(BatchItem item, int priority, long deadlineMs,
                                        long arrivalMs, long requestId) {
        Prioritized probe = new Prioritized() {
            @Override public int priority() { return priority; }
            @Override public long enqueueSeq() { return arrivalMs; }
        };
        int cmp = PriorityOrdering.STRICT.compare(item, probe);
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
