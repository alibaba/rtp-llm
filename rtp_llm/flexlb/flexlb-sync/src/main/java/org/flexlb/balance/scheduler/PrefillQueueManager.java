package org.flexlb.balance.scheduler;

import org.flexlb.balance.scheduler.priority.PrefillQueueSnapshot;
import org.flexlb.balance.scheduler.priority.QueuedRequestSnapshot;
import org.flexlb.util.PriorityOrdering;

import java.util.ArrayList;
import java.util.List;

/**
 * Priority scheduling facade over one {@link WorkerBatcher} queue.
 *
 * <p>One instance per prefill endpoint, created by the batcher itself. It does
 * not own any state: every operation delegates to the batcher/context so the
 * active queue plus ready-delivery backlog stay the single source of truth
 * and every mutation goes through the shared queue lock.
 *
 * <p>Read operations ({@link #snapshot()}, {@link #estimateWaitMs}) capture a
 * consistent view under the queue lock. Victim replacement validates the
 * selected victims under that same lock rather than invalidating a plan for
 * unrelated queue mutations.
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
     * The hard capacity comes from the active scheduler/dispatcher variants.
     */
    public PrefillQueueSnapshot snapshot() {
        ctx.queueLock().lock();
        try {
            // Only live queue members are actionable eviction victims. A
            // staged callback member remains capacity-charged, but is no
            // longer an actionable queue victim.
            List<BatchItem> queued = ctx.sortedQueuedItems();
            List<QueuedRequestSnapshot> items = new ArrayList<>(queued.size());
            for (BatchItem item : queued) {
                items.add(new QueuedRequestSnapshot(
                        item.requestId(), item.priority(), item.enqueuedAtMs(),
                        item.seqLen(), item.hitCache(),
                        QueuedRequestSnapshot.PREFILL_QUEUED));
            }
            return new PrefillQueueSnapshot(ctx.key(), batcher.queueVersion(),
                    ctx.maxQueueCapacity(), items);
        } finally {
            ctx.queueLock().unlock();
        }
    }

    /**
     * Estimated queue wait for an incoming request (design doc 8.4):
     * active {@code itemsAhead → batchCyclesAhead}, plus a request-accounted
     * drain estimate for route decisions whose logical group is already
     * ready. The decision-interval sliding average is the per-cycle cost and
     * the active head's remaining window is the partial first cycle.
     */
    public long estimateWaitMs(int priority, long requestId) {
        long now = ctx.now();
        int activeItemsAhead = 0;
        int readyItemsAhead = 0;
        BatchItem head = null;
        ctx.queueLock().lock();
        try {
            for (BatchItem item : ctx.sortedQueuedItems()) {
                // A decided route request is drained before the active queue,
                // regardless of a later arrival's priority. It is also
                // request-accounted: batchSizeMax must not collapse several
                // ready requests into one estimated cycle when the route cap
                // permits only individual deliveries.
                if (item.readyDeliveryReason() != null) {
                    readyItemsAhead++;
                    continue;
                }
                if (head == null) {
                    head = item;
                }
                if (ordersBefore(item, priority, now, requestId)) {
                    activeItemsAhead++;
                }
            }
        } finally {
            ctx.queueLock().unlock();
        }
        int maxBatchSize = ctx.maxDecisionRequests();
        long intervalMs = ctx.avgDecisionIntervalMs();
        long batchCyclesAhead = activeItemsAhead / maxBatchSize;
        long headRemainingWindowMs = 0;
        if (head != null) {
            long headWaitedMs = Math.max(0, now - head.enqueuedAtMs());
            headRemainingWindowMs = Math.max(0, intervalMs - headWaitedMs);
        }
        // There is no trustworthy completion-rate signal for frontend-owned
        // requests, so charge one interval per ready member. This is a
        // deliberate conservative bound; it avoids the severe cap=1
        // under-estimate while leaving active logical-batch timing unchanged.
        long readyDrainWaitMs = saturatedMultiply(readyItemsAhead, intervalMs);
        long activeWaitMs = saturatedMultiply(batchCyclesAhead, intervalMs);
        return saturatedAdd(readyDrainWaitMs,
                saturatedAdd(activeWaitMs, headRemainingWindowMs));
    }

    private static long saturatedMultiply(long count, long intervalMs) {
        if (count <= 0 || intervalMs <= 0) {
            return 0;
        }
        return count > Long.MAX_VALUE / intervalMs
                ? Long.MAX_VALUE : count * intervalMs;
    }

    private static long saturatedAdd(long left, long right) {
        return left > Long.MAX_VALUE - right ? Long.MAX_VALUE : left + right;
    }

    /**
     * Whether a queued item is ordered before an incoming probe under
     * {@link WorkerBatcher#PRIORITY_QUEUE_ORDER} (priority desc → enqueue-seq
     * asc → requestId asc). The probe's enqueue-seq is its would-be arrival
     * time ({@code now}) — it has not been enqueued yet.
     *
     * <p>Delegates the priority + enqueue-seq + request-id comparison to the
     * allocation-free primitive overload in {@link PriorityOrdering}.
     */
    private static boolean ordersBefore(BatchItem item, int priority,
                                        long arrivalMs, long requestId) {
        return PriorityOrdering.comesBefore(
                item, item.requestId(), priority, arrivalMs, requestId);
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
