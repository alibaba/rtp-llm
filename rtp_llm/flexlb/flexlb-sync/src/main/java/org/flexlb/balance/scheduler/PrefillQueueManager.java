package org.flexlb.balance.scheduler;

import org.flexlb.balance.scheduler.priority.PrefillQueueSnapshot;
import org.flexlb.balance.scheduler.priority.QueuedRequestSnapshot;

import java.util.ArrayList;
import java.util.List;

/**
 * Priority scheduling facade over one {@link WorkerBatcher} queue.
 *
 * <p>One instance per prefill endpoint, created by the batcher itself. It does
 * not own any state: every operation delegates to the batcher/context so the
 * active queue stays the single source of truth and every mutation goes through
 * the shared queue lock.
 *
 * <p>Snapshots capture removable identities under the queue lock. The hot wait
 * probe uses a versioned primitive priority histogram and therefore retains no
 * request object or future. Victim replacement validates the selected victims
 * under the queue lock rather than invalidating a plan for unrelated mutations.
 */
public final class PrefillQueueManager {

    private final WorkerBatcher batcher;
    private final BatcherContext ctx;

    PrefillQueueManager(WorkerBatcher batcher, BatcherContext ctx) {
        this.batcher = batcher;
        this.ctx = ctx;
    }

    /** Current queue mutation generation (monotonic, bumped on every mutation). */
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
     * The hard capacity comes from the QUEUE scheduler's capacity policy.
     *
     * <p>Only the membership copy and the version capture run under the
     * queue lock; the O(n log n) sort runs on the thread-confined copy
     * outside it, so the submit path's {@code offer()} and the batcher
     * decision cycle never block behind a snapshot sort. Correctness: the
     * copy and the version are captured atomically under the same lock
     * hold, every queue mutation bumps the version under that lock, and
     * item sort fields are frozen once the item is constructed
     * ({@code enqueueSequence} is assigned at construction, before the
     * item can enter the queue), so the "version unchanged ⇒ queue content
     * unchanged" invariant and the output order are both preserved
     * bit-for-bit. A stale snapshot is harmless downstream: victim
     * replacement validates the selected victims under the queue lock
     * ({@link #tryReplaceVictimsPresent}).
     */
    public PrefillQueueSnapshot snapshot() {
        List<BatchItem> queued;
        long version;
        ctx.queueLock().lock();
        try {
            // Only live queue members are actionable eviction victims. A
            // callback-owned member remains capacity-charged, but is no
            // longer an actionable queue victim.
            queued = ctx.copiedItems();
            version = batcher.queueVersion();
        } finally {
            ctx.queueLock().unlock();
        }
        queued.sort(ctx.queueOrder());
        List<QueuedRequestSnapshot> items = new ArrayList<>(queued.size());
        for (BatchItem item : queued) {
            items.add(new QueuedRequestSnapshot(
                    item.requestId(), item.priority(), item.enqueuedAtMs(),
                    item.seqLen(), item.hitCache(),
                    QueuedRequestSnapshot.PREFILL_QUEUED));
        }
        return new PrefillQueueSnapshot(ctx.key(), version,
                ctx.maxQueueCapacity(), items);
    }

    /**
     * Estimated additional fixed-window collection delay for an incoming
     * request. Endpoint ledgers account for engine execution and inflight work;
     * this method deliberately does not reinterpret decision timestamps as a
     * request-completion rate.
     */
    public long estimateWaitMs(int priority, long requestId) {
        long now = ctx.now();
        return ctx.estimateIncomingWaitMs(priority, now, requestId);
    }

    /**
     * Idempotently remove a request during cancellation or expiration. No-op
     * when the item is already delivered, evicted, or removed.
     */
    public void tryRemove(long requestId, String reason) {
        batcher.tryRemove(List.of(requestId), reason);
    }

    /**
     * Atomic victim replacement with a victim-level presence guard: no version
     * check — any missing victim aborts with a zero-side-effect
     * {@code VICTIM_GONE} carrying the missing ids.
     */
    public ReplaceOutcome tryReplaceVictimsPresent(List<Long> victimIds, BatchItem incoming) {
        return batcher.tryReplaceVictimsPresent(victimIds, incoming);
    }

    // ==================== Replace outcome ====================

    /**
     * Result of {@link #tryReplaceVictimsPresent}. {@code removed} holds the victims
     * actually taken out of the queue — non-empty on success and on the
     * (defensively handled) partial failure, where victims are never
     * re-inserted (design doc 9.5) and must be driven to a terminal state by
     * the caller. {@code VICTIM_GONE} (presence guard only) is a
     * zero-side-effect abort carrying the victims no longer queued.
     */
    public static final class ReplaceOutcome {

        public enum Status { SUCCESS, PARTIAL_FAILURE, VICTIM_GONE }

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
