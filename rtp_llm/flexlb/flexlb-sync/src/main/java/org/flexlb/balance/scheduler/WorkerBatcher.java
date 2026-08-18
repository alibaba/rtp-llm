package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;
import org.flexlb.util.PriorityOrdering;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CancellationException;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Per-worker scheduling queue that owns request grouping and delivery staging,
 * delegating grouping decisions to a pluggable {@link BatcherAlgorithm}.
 *
 * <p>One instance per Prefill worker. Requests are submitted via
 * {@link #offer(BatchItem)} and grouped by the configured algorithm. A group
 * may be delivered through EnqueueBatch or as individual route decisions.
 *
 * <p>The context-owned active queue and ready-delivery backlog together are
 * the single source of truth for pending requests. In Auto-TPM mode both are
 * ordered by the explicit priority comparator
 * ({@link #AUTO_TPM_QUEUE_ORDER}, design doc 8.1); in legacy mode the
 * algorithm-computed {@link BatchItem#sortKey()} order is preserved
 * byte-for-byte. All mutations go through {@link BatcherContext} (or the
 * lock-holding methods here) and bump {@code queueVersion} for optimistic
 * plan validation.
 */
public class WorkerBatcher {

    private static final long DELIVERY_CAPACITY_RETRY_NANOS =
            TimeUnit.MILLISECONDS.toNanos(1);

    /**
     * Auto-TPM queue order (PR-B unification): delegates to
     * {@link PriorityOrdering#STRICT} (priority desc → enqueue-seq asc for
     * same-priority FIFO) with {@code requestId} as the final deterministic
     * tie-break.
     *
     * <p>The previous third key — coarse admission deadline — has been
     * removed. It was a weak signal that conflicted with FIFO fairness under
     * bursty arrivals; same-priority FIFO is now the sole tie-break.
     *
     * <p>{@link #LEGACY_QUEUE_ORDER} is unchanged.
     */
    public static final Comparator<BatchItem> AUTO_TPM_QUEUE_ORDER =
            (left, right) -> PriorityOrdering.compareWithRequestId(
                    left.priority(), left.enqueueSeq(), left.requestId(),
                    right.priority(), right.enqueueSeq(), right.requestId());

    /** Legacy order: algorithm-computed sort key (unchanged behavior). */
    public static final Comparator<BatchItem> LEGACY_QUEUE_ORDER =
            Comparator.comparingLong(BatchItem::sortKey);

    private final String key;
    private final FlexlbConfig cfg;
    private final DecisionGroupHandler decisionHandler;
    private final boolean autoTpm;
    private final PriorityBlockingQueue<BatchItem> queue;
    private final AtomicInteger queueDepth = new AtomicInteger();
    /**
     * Monotonic queue version bumped on every queue mutation (enqueue,
     * remove, delivery, drain). Captured in Auto-TPM cluster snapshots and
     * re-checked at plan commit time to detect concurrent queue mutations
     * (optimistic concurrency control).
     */
    private final AtomicLong queueVersion = new AtomicLong();
    /**
     * Guards queue mutations so the Auto-TPM atomic victim-replace can rely
     * on "version unchanged ⇒ queue content unchanged".
     *
     * <p>Known accepted cost: the lock and the version bump stay active even
     * when Auto-TPM is disabled; the uncontended ReentrantLock + AtomicLong
     * overhead per queue mutation is negligible and keeping them
     * unconditional avoids divergent code paths (task10 P2-9, no structural
     * change).
     */
    private final ReentrantLock queueLock = new ReentrantLock();
    /**
     * One per-worker condition for both new queue work and route-slot release.
     * Every predicate transition and signal is serialized by queueLock, so a
     * release cannot race between the cap re-check and await.
     */
    private final Condition stateChanged = queueLock.newCondition();
    private final Thread workerThread;
    private volatile boolean stopped;
    private volatile boolean waitingForSignal;
    private final BatcherAlgorithm algorithm;
    private final BatcherContext ctx;
    private final PrefillQueueManager queueManager;
    private volatile long deliveryRetryNotBeforeNanos;

    public WorkerBatcher(String key, PrefillEndpoint prefillEp, FlexlbConfig cfg,
                         DecisionGroupHandler decisionHandler,
                         BatchSchedulerReporter reporter) {
        this.key = key;
        this.cfg = cfg;
        this.decisionHandler = decisionHandler;
        this.autoTpm = cfg.isAutoTpmEnabled();
        Comparator<BatchItem> queueOrder = autoTpm ? AUTO_TPM_QUEUE_ORDER : LEGACY_QUEUE_ORDER;
        this.queue = new PriorityBlockingQueue<>(11, queueOrder);
        this.algorithm = createAlgorithm(cfg);
        this.ctx = new BatcherContext(
                key, prefillEp, cfg, decisionHandler, queue, queueDepth, queueVersion, queueLock,
                queueOrder, reporter);
        this.queueManager = new PrefillQueueManager(this, ctx);
        this.workerThread = new Thread(this::runLoop, "flexlb-batcher-" + key);
        this.workerThread.setDaemon(true);
        this.workerThread.setUncaughtExceptionHandler((t, e) ->
                Logger.error("WorkerBatcher[{}] thread died unexpectedly", key, e));
    }

    private static BatcherAlgorithm createAlgorithm(FlexlbConfig config) {
        String algoName = config.getFlexlbBatchAlgorithm();
        if ("fixed_window".equalsIgnoreCase(algoName)) {
            return new FixedWindowBatcherAlgorithm();
        }
        // Fallback: slo_budget for any unrecognized value
        return new SloBudgetBatcherAlgorithm();
    }

    public void start() {
        workerThread.start();
    }

    public void offer(BatchItem item) {
        if (stopped) {
            decisionHandler.onOfferFailure(item,
                    new IllegalStateException("FlexLB worker scheduling queue stopped"));
            return;
        }
        int maxSize = cfg.getFlexlbBatchQueueMaxSize();
        if (!reserveQueueSlot(maxSize)) {
            decisionHandler.onOfferFailure(item,
                    new IllegalStateException(
                            "FlexLB worker scheduling queue full, maxSize=" + maxSize));
            return;
        }
        if (!enqueue(item)) {
            decisionHandler.onOfferFailure(item,
                    new IllegalStateException("FlexLB worker scheduling queue stopped"));
        }
    }

    /**
     * Auto-TPM variant of {@link #offer(BatchItem)} that reports failure via
     * return value instead of the {@link DecisionGroupHandler#onOfferFailure}
     * callback, letting the caller (PlanCommitter) roll back its decode
     * reservation and decide on retry.
     *
     * @return true when the item was enqueued; false when the worker queue is
     *         stopped or the queue is full (item not enqueued)
     */
    public boolean tryOffer(BatchItem item) {
        if (stopped) {
            return false;
        }
        if (!reserveQueueSlot(cfg.getFlexlbBatchQueueMaxSize())) {
            return false;
        }
        return enqueue(item);
    }

    private boolean enqueue(BatchItem item) {
        try {
            long sortKey = algorithm.computeSortKey(ctx, item);
            item.setSortKey(sortKey);
            algorithm.onOffer(ctx, item, System.currentTimeMillis());
            queueLock.lock();
            try {
                // Linearize the final stopped check with shutdown/drain. An
                // offer that reserved capacity just before shutdown must not
                // enqueue after the drain has completed.
                if (stopped) {
                    queueDepth.decrementAndGet();
                    return false;
                }
                queue.add(item);
                queueVersion.incrementAndGet();
                if (autoTpm) {
                    stateChanged.signal();
                }
                return true;
            } finally {
                queueLock.unlock();
            }
        } catch (RuntimeException | Error e) {
            queueDepth.decrementAndGet();
            throw e;
        }
    }

    public int queueSize() {
        return queueDepth.get();
    }

    /**
     * Snapshot of the current queue depth bucketed by normalized Auto-TPM
     * priority ({@code 0} for legacy items without a budget — same convention
     * as the batch wait-time-by-priority series). The active queue uses its
     * weakly-consistent iterator; the usually small ready-delivery backlog is
     * copied under {@link #queueLock} so a staged route decision remains
     * visible without racing removal or delivery. Only priorities present in
     * either queue appear in the result — the same empty-bucket behavior as
     * the batch wait-time-by-priority series.
     */
    public Map<Integer, Integer> queueSizeByPriority() {
        Map<Integer, Integer> sizeByPriority = new HashMap<>();
        for (BatchItem item : queue) {
            sizeByPriority.merge(item.priority(), 1, Integer::sum);
        }
        ctx.addReadyQueueSizeByPriority(sizeByPriority);
        return sizeByPriority;
    }

    /** Current active/ready queue version for Auto-TPM optimistic plan validation. */
    public long queueVersion() {
        return queueVersion.get();
    }

    /**
     * Estimated time a new request would wait in the queue before delivery.
     * Delegates to the algorithm-specific {@link BatcherAlgorithm#queueWaitMs}.
     */
    public long queueWaitMs() {
        return algorithm.queueWaitMs(ctx);
    }

    /** Auto-TPM queue facade (snapshot / estimateWait / atomic replace). */
    public PrefillQueueManager queueManager() {
        return queueManager;
    }

    /** Resolve a staged item as delivered or terminal and release its queue slot. */
    boolean completePendingDelivery(BatchItem item) {
        return ctx.completePendingDelivery(item);
    }

    /** Claim callback ownership, atomically fenced against batcher shutdown. */
    BatcherContext.PendingClaimResult claimPendingDelivery(BatchItem item) {
        // Do not consult WorkerBatcher.stopped outside queueLock. shutdown()
        // publishes that flag immediately before ctx.stopAndDrainTo(); an
        // out-of-lock early return in that window would leave the item STAGED
        // while the callback incorrectly assumes shutdown already drained it.
        // BatcherContext is the authoritative atomic claim-vs-drain fence.
        return ctx.claimPendingDelivery(item);
    }

    /**
     * Restore a temporarily capacity-blocked item without recomputing its
     * ordering metadata. Returns false when it was not pending or shutdown
     * already owns the batcher.
     */
    BatcherContext.PendingRestoreResult restorePendingDelivery(BatchItem item) {
        BatcherContext.PendingRestoreResult result = ctx.restorePendingDelivery(item);
        if (result == BatcherContext.PendingRestoreResult.RESTORED) {
            long retryAt = System.nanoTime() + DELIVERY_CAPACITY_RETRY_NANOS;
            deliveryRetryNotBeforeNanos = Math.max(deliveryRetryNotBeforeNanos, retryAt);
        }
        return result;
    }

    int pendingDeliveryCount() {
        return ctx.pendingDeliveryCount();
    }

    public void shutdown() {
        List<BatchItem> remaining = new ArrayList<>();
        queueLock.lock();
        try {
            stopped = true;
            // Reentrant on the same per-worker lock: drain and wake publish as
            // one state transition to both empty and capacity waiters.
            ctx.stopAndDrainTo(remaining);
            stateChanged.signalAll();
        } finally {
            queueLock.unlock();
        }
        workerThread.interrupt();
        algorithm.onShutdown(ctx);
        for (BatchItem item : remaining) {
            try {
                decisionHandler.onOfferFailure(item,
                        new CancellationException(
                                "FlexLB worker scheduling queue stopped: " + key));
            } catch (Throwable callbackFailure) {
                Logger.error("WorkerBatcher[{}] shutdown callback failed request_id={}",
                        key, item.requestId(), callbackFailure);
            }
        }
    }

    // ==================== Auto-TPM queue operations (called via PrefillQueueManager) ====================

    /**
     * Offer only if the queue version still matches — used by commit paths
     * that must not apply against a mutated queue.
     */
    boolean tryOfferAtVersion(BatchItem item, long expectedVersion) {
        return offerAtVersion(item, expectedVersion) == OfferAtVersionResult.SUCCESS;
    }

    /** Outcome of {@link #offerAtVersion(BatchItem, long)}. */
    public enum OfferAtVersionResult {
        /** Version matched and the item was enqueued. */
        SUCCESS,
        /** Queue version moved since the plan snapshot; nothing applied. */
        VERSION_MISMATCH,
        /** Version matched but the offer failed (batcher stopped / queue full). */
        OFFER_FAILED
    }

    /**
     * Version-checked offer executed atomically under {@link #queueLock}:
     * the version re-check and the enqueue happen in one critical section, so
     * no concurrent queue mutation can slip between them (task10 P1-4).
     * Distinguishes a stale version (retryable conflict) from a capacity
     * failure so the caller can map them to different outcomes.
     */
    public OfferAtVersionResult offerAtVersion(BatchItem item, long expectedVersion) {
        if (stopped) {
            return OfferAtVersionResult.OFFER_FAILED;
        }
        queueLock.lock();
        try {
            if (queueVersion.get() != expectedVersion) {
                return OfferAtVersionResult.VERSION_MISMATCH;
            }
            // Re-entrant: tryOffer -> enqueue re-acquires queueLock safely.
            return tryOffer(item) ? OfferAtVersionResult.SUCCESS : OfferAtVersionResult.OFFER_FAILED;
        } finally {
            queueLock.unlock();
        }
    }

    /**
     * Remove the given requests only if the queue version still matches.
     *
     * @return removed items, or {@code null} on version mismatch
     */
    List<BatchItem> tryRemoveAtVersion(List<Long> requestIds, long expectedVersion, String reason) {
        queueLock.lock();
        try {
            if (queueVersion.get() != expectedVersion) {
                return null;
            }
            List<BatchItem> removed = new ArrayList<>(requestIds.size());
            for (long requestId : requestIds) {
                BatchItem item = findQueued(requestId);
                if (item != null && ctx.remove(item)) {
                    removed.add(item);
                }
            }
            if (!removed.isEmpty()) {
                Logger.debug("[auto-tpm] queue remove: worker={} reason={} removed={}",
                        key, reason, removed.size());
            }
            return removed;
        } finally {
            queueLock.unlock();
        }
    }

    /**
     * Atomic victim replacement (design doc 17.2): under the single-endpoint
     * queue lock, validate the version, remove all victims, then enqueue the
     * incoming item into the freed slots. Any mutation since the plan snapshot
     * bumps the version, so a matching version guarantees every victim is
     * still queued — the partial-failure branch is defensive only.
     */
    PrefillQueueManager.ReplaceOutcome tryReplaceVictimsWithIncoming(
            List<Long> victimIds, BatchItem incoming, long expectedVersion) {
        queueLock.lock();
        try {
            if (stopped || queueVersion.get() != expectedVersion) {
                return PrefillQueueManager.ReplaceOutcome.versionMismatch();
            }
            List<BatchItem> removed = new ArrayList<>(victimIds.size());
            for (long victimId : victimIds) {
                BatchItem victim = findQueued(victimId);
                if (victim == null || !ctx.remove(victim)) {
                    // Unreachable under the lock discipline; victims already
                    // removed stay out (no re-insert, design doc 9.5).
                    return PrefillQueueManager.ReplaceOutcome.partialFailure(removed);
                }
                removed.add(victim);
            }
            if (!tryOffer(incoming)) {
                return PrefillQueueManager.ReplaceOutcome.partialFailure(removed);
            }
            return PrefillQueueManager.ReplaceOutcome.success(removed);
        } catch (RuntimeException | Error e) {
            Logger.error("WorkerBatcher[{}] victim replace failed", key, e);
            return PrefillQueueManager.ReplaceOutcome.partialFailure(List.of());
        } finally {
            queueLock.unlock();
        }
    }

    /**
     * Version-agnostic idempotent removal (PR-D §2.7): same as
     * {@link #tryRemoveAtVersion} but skips the version check. Used by
     * {@code AdmissionLease.close()} for deadline-timeout cleanup.
     */
    List<BatchItem> tryRemoveNoVersion(List<Long> requestIds, String reason) {
        queueLock.lock();
        try {
            List<BatchItem> removed = new ArrayList<>(requestIds.size());
            for (long requestId : requestIds) {
                BatchItem item = findQueued(requestId);
                if (item != null && ctx.remove(item)) {
                    removed.add(item);
                }
            }
            if (!removed.isEmpty()) {
                Logger.debug("[auto-tpm] queue remove (version-agnostic): worker={} reason={} removed={}",
                        key, reason, removed.size());
            }
            return removed;
        } finally {
            queueLock.unlock();
        }
    }

    private BatchItem findQueued(long requestId) {
        return ctx.findQueued(requestId);
    }

    /**
     * Atomic victim replacement with victim-level presence guard (redesign N3
     * §3.4, {@code autoTpmVictimGuardMode=victim_presence}): under the queue
     * lock, verify every victim is still queued — any missing victim aborts
     * with a zero-side-effect {@code VICTIM_GONE} (nothing removed, incoming
     * not enqueued) — then remove all victims and enqueue the incoming item.
     *
     * <p>"Still queued" is a sufficient guard: a BatchItem leaves the queue
     * only via delivery / eviction / drop, all under {@link #queueLock} and
     * all removing it from the queue — so an in-lock {@code findQueued} hit
     * proves the victim has not been delivered. Unrelated queue mutations
     * (which the legacy whole-queue version guard treated as conflicts) no
     * longer abort the commit.
     */
    PrefillQueueManager.ReplaceOutcome tryReplaceVictimsPresent(
            List<Long> victimIds, BatchItem incoming) {
        queueLock.lock();
        try {
            if (stopped) {
                // Shutdown: zero-side-effect abort (caller replans / fails fast).
                return PrefillQueueManager.ReplaceOutcome.victimGone(List.copyOf(victimIds));
            }
            List<BatchItem> present = new ArrayList<>(victimIds.size());
            List<Long> missing = new ArrayList<>();
            for (long victimId : victimIds) {
                BatchItem victim = findQueued(victimId);
                if (victim == null) {
                    missing.add(victimId);
                } else {
                    present.add(victim);
                }
            }
            if (!missing.isEmpty()) {
                return PrefillQueueManager.ReplaceOutcome.victimGone(missing);
            }
            List<BatchItem> removed = new ArrayList<>(present.size());
            for (BatchItem victim : present) {
                if (!ctx.remove(victim)) {
                    // Unreachable under the lock discipline; victims already
                    // removed stay out (no re-insert, design doc 9.5).
                    return PrefillQueueManager.ReplaceOutcome.partialFailure(removed);
                }
                removed.add(victim);
            }
            if (!tryOffer(incoming)) {
                return PrefillQueueManager.ReplaceOutcome.partialFailure(removed);
            }
            return PrefillQueueManager.ReplaceOutcome.success(removed);
        } catch (RuntimeException | Error e) {
            Logger.error("WorkerBatcher[{}] presence-guarded victim replace failed", key, e);
            return PrefillQueueManager.ReplaceOutcome.partialFailure(List.of());
        } finally {
            queueLock.unlock();
        }
    }

    // ==================== Internal: Run loop ====================

    private void runLoop() {
        while (!stopped && !Thread.currentThread().isInterrupted()) {
            try {
                waitForNonEmpty();
                waitForDeliveryRetry();
                BatcherContext.ReadyDeliveryResult readyDelivery =
                        ctx.deliverReadyRequests();
                if (readyDelivery == BatcherContext.ReadyDeliveryResult.CAPACITY_BLOCKED) {
                    if (ctx.isActiveEmpty()) {
                        awaitDeliveryCapacityOrActiveWork();
                        continue;
                    }
                    // A live config transition can leave BATCH_ENQUEUE work behind
                    // a route backlog. Ready route work remains preferred when
                    // capacity exists, but a full route cap must not HOL-block
                    // undecided active work.
                }
                if (readyDelivery == BatcherContext.ReadyDeliveryResult.DELIVERED) {
                    continue;
                }
                algorithm.processQueue(ctx);
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                return;
            } catch (Throwable t) {
                Logger.error("WorkerBatcher[{}] loop failed", key, t);
            }
        }
    }

    private void waitForNonEmpty() throws InterruptedException {
        if (autoTpm) {
            queueLock.lockInterruptibly();
            try {
                while (!stopped && !ctx.hasProcessableWork()) {
                    awaitStateChange();
                }
            } finally {
                queueLock.unlock();
            }
            return;
        }
        BatchItem item = queue.take();
        queue.put(item);
    }

    /**
     * Wait only while ready work is the sole work and the request cap is
     * still full. Enqueue and slot release both signal under queueLock; the
     * in-lock predicate re-check closes every missed-wakeup window.
     */
    private void awaitDeliveryCapacityOrActiveWork() throws InterruptedException {
        queueLock.lockInterruptibly();
        try {
            while (!stopped
                    && ctx.isActiveEmpty()
                    && ctx.readyDeliveryCount() > 0
                    && ctx.availableDeliverySlots() == 0) {
                awaitStateChange();
            }
        } finally {
            queueLock.unlock();
        }
    }

    private void awaitStateChange() throws InterruptedException {
        waitingForSignal = true;
        try {
            stateChanged.await();
        } finally {
            waitingForSignal = false;
        }
    }

    /** Called after Prefill releases a request-mode slot, outside its stripe. */
    public void signalDeliveryCapacityAvailable() {
        if (!autoTpm || ctx.readyDeliveryCount() == 0) {
            return;
        }
        queueLock.lock();
        try {
            stateChanged.signal();
        } finally {
            queueLock.unlock();
        }
    }

    /** Package-private deterministic wait-state probe for scheduler tests. */
    boolean isWaitingForSignal() {
        return waitingForSignal;
    }

    private void waitForDeliveryRetry() throws InterruptedException {
        long remaining = deliveryRetryNotBeforeNanos - System.nanoTime();
        if (remaining > 0) {
            TimeUnit.NANOSECONDS.sleep(remaining);
        }
    }

    private boolean reserveQueueSlot(int maxSize) {
        if (maxSize <= 0) {
            queueDepth.incrementAndGet();
            return true;
        }
        while (true) {
            int current = queueDepth.get();
            if (current >= maxSize) {
                return false;
            }
            if (queueDepth.compareAndSet(current, current + 1)) {
                return true;
            }
        }
    }
}
