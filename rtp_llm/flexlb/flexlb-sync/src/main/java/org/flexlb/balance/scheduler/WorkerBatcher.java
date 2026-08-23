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
 * delegating grouping decisions to a {@link BatcherAlgorithm}.
 *
 * <p>One instance per Prefill worker. Requests are submitted via
 * {@link #offer(BatchItem)} and grouped by the configured decision policy. A
 * group may independently be delivered through
 * EnqueueBatch or as individual route decisions.
 *
 * <p>The context-owned active queue and ready-delivery backlog together are
 * the single source of truth for pending requests. PRIORITY ordering uses
 * the explicit priority comparator ({@link #PRIORITY_QUEUE_ORDER}); FIFO uses
 * a unique monotonic enqueue sequence. All mutations go through
 * {@link BatcherContext} (or the
 * lock-holding methods here). A monotonic mutation generation lets snapshots
 * and diagnostics identify queue-state changes.
 */
public class WorkerBatcher {

    /**
     * A failed hard delivery claim or callback restore is retried after one
     * small, explicit delay. The fixed upper bound prevents a tight retry loop
     * without making capacity-release latency unbounded in the absence of a
     * Decode-side wakeup.
     */
    private static final long DELIVERY_RETRY_BACKOFF_NANOS =
            TimeUnit.MILLISECONDS.toNanos(10);

    /**
     * PRIORITY queue order: delegates to
     * {@link PriorityOrdering#STRICT} (priority desc → enqueue-seq asc for
     * same-priority FIFO) with {@code requestId} as the final deterministic
     * tie-break.
     *
     * <p>{@link #FIFO_QUEUE_ORDER} preserves enqueue order.
     */
    public static final Comparator<BatchItem> PRIORITY_QUEUE_ORDER =
            (left, right) -> PriorityOrdering.compareWithRequestId(
                    left.priority(), left.enqueueSeq(), left.requestId(),
                    right.priority(), right.enqueueSeq(), right.requestId());

    /** FIFO order: unique monotonic enqueue sequence. */
    public static final Comparator<BatchItem> FIFO_QUEUE_ORDER =
            Comparator.comparingLong(BatchItem::enqueueSeq);

    private final String key;
    private final FlexlbConfig cfg;
    private final DecisionGroupHandler decisionHandler;
    private final boolean priorityOrdering;
    private final PriorityBlockingQueue<BatchItem> queue;
    private final AtomicInteger queueDepth = new AtomicInteger();
    /**
     * Monotonic queue mutation generation, bumped on enqueue, removal,
     * delivery and drain. It is exposed in diagnostic snapshots.
     */
    private final AtomicLong queueVersion = new AtomicLong();
    /**
     * Guards queue mutations and atomic victim replacement.
     *
     * <p>The lock and generation stay active for both FIFO and PRIORITY so
     * every ordering mode has the same mutation guarantees.
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
    private volatile long readyRetryNotBeforeNanos;
    private volatile long activeRetryNotBeforeNanos;
    private volatile BatchItem activeRetryItem;

    public WorkerBatcher(String key, PrefillEndpoint prefillEp, FlexlbConfig cfg,
                         DecisionGroupHandler decisionHandler,
                         BatchSchedulerReporter reporter) {
        this.key = key;
        this.cfg = cfg;
        this.decisionHandler = decisionHandler;
        this.priorityOrdering = cfg.isPriorityOrdering();
        Comparator<BatchItem> queueOrder = priorityOrdering
                ? PRIORITY_QUEUE_ORDER : FIFO_QUEUE_ORDER;
        this.queue = new PriorityBlockingQueue<>(11, queueOrder);
        this.algorithm = cfg.isSingleDecision()
                ? new SingleRequestBatcherAlgorithm()
                : new FixedWindowBatcherAlgorithm();
        this.ctx = new BatcherContext(
                key, prefillEp, cfg, decisionHandler, queue, queueDepth, queueVersion, queueLock,
                queueOrder, reporter);
        this.queueManager = new PrefillQueueManager(this, ctx);
        this.workerThread = new Thread(this::runLoop, "flexlb-batcher-" + key);
        this.workerThread.setDaemon(true);
        this.workerThread.setUncaughtExceptionHandler((t, e) ->
                Logger.error("WorkerBatcher[{}] thread died unexpectedly", key, e));
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
        int maxSize = ctx.maxQueueCapacity();
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
     * priority scheduling variant of {@link #offer(BatchItem)} that reports failure via
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
        if (!reserveQueueSlot(ctx.maxQueueCapacity())) {
            return false;
        }
        return enqueue(item);
    }

    private boolean enqueue(BatchItem item) {
        try {
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
                stateChanged.signal();
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
     * Snapshot of the current queue depth bucketed by normalized priority scheduling
     * priority. The active queue uses its
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

    /** Current active/ready queue mutation generation. */
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

    /** priority scheduling queue facade (snapshot / estimateWait / atomic replace). */
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
        return ctx.restorePendingDelivery(item);
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

    // ==================== priority scheduling queue operations ====================

    /** Idempotently remove queued requests during cancellation or expiration. */
    List<BatchItem> tryRemove(List<Long> requestIds, String reason) {
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
                // External cancellation/expiration may replace a
                // capacity-blocked route head with processable batch work.
                // Publish that predicate transition under the same lock as
                // the removal so the worker cannot miss the wakeup.
                stateChanged.signal();
                Logger.debug("[priority-scheduler] queue remove: worker={} reason={} removed={}",
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
     * Atomic victim replacement with a victim-level presence guard: under the queue
     * lock, verify every victim is still queued — any missing victim aborts
     * with a zero-side-effect {@code VICTIM_GONE} (nothing removed, incoming
     * not enqueued) — then remove all victims and enqueue the incoming item.
     *
     * <p>"Still queued" is a sufficient guard: a BatchItem leaves the queue
     * only via delivery / eviction / drop, all under {@link #queueLock} and
     * all removing it from the queue — so an in-lock {@code findQueued} hit
     * proves the victim has not been delivered. Unrelated queue mutations do
     * not abort the commit.
     */
    PrefillQueueManager.ReplaceOutcome tryReplaceVictimsPresent(
            List<Long> victimIds, BatchItem incoming) {
        List<BatchItem> removed = new ArrayList<>(victimIds.size());
        boolean queueChanged = false;
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
            for (BatchItem victim : present) {
                if (!ctx.remove(victim)) {
                    // Unreachable under the lock discipline; victims already
                    // removed stay out (no re-insert, design doc 9.5).
                    return PrefillQueueManager.ReplaceOutcome.partialFailure(removed);
                }
                removed.add(victim);
                queueChanged = true;
            }
            if (!tryOffer(incoming)) {
                return PrefillQueueManager.ReplaceOutcome.partialFailure(removed);
            }
            return PrefillQueueManager.ReplaceOutcome.success(removed);
        } catch (RuntimeException | Error e) {
            Logger.error("WorkerBatcher[{}] presence-guarded victim replace failed", key, e);
            return PrefillQueueManager.ReplaceOutcome.partialFailure(removed);
        } finally {
            if (queueChanged) {
                // A partial replacement may remove the route head without a
                // successful incoming offer. Wake the condition waiter so a
                // newly exposed active head is not hidden until the removed
                // request's old expiration deadline.
                stateChanged.signal();
            }
            queueLock.unlock();
        }
    }

    // ==================== Internal: Run loop ====================

    private void runLoop() {
        while (!stopped && !Thread.currentThread().isInterrupted()) {
            try {
                waitForNonEmpty();
                BatcherContext.ReadyDeliveryResult readyDelivery =
                        BatcherContext.ReadyDeliveryResult.EMPTY;
                if (!readyRetryDeferred()) {
                    readyDelivery = ctx.deliverReadyRequests();
                    schedulePendingDeliveryRetries();
                }
                if (readyDelivery == BatcherContext.ReadyDeliveryResult.CAPACITY_BLOCKED) {
                    if (ctx.isActiveEmpty()) {
                        if (readyRetryDeferred()) {
                            awaitReadyRetryOrActiveWork();
                        } else {
                            awaitDeliveryCapacityOrActiveWork();
                        }
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
                if (readyRetryDeferred() && ctx.isActiveEmpty()) {
                    awaitReadyRetryOrActiveWork();
                    continue;
                }
                if (awaitActiveRetryIfDeferred()) {
                    continue;
                }
                if (awaitActiveRouteCapacityIfBlocked()) {
                    // A slot or queue-state transition ended the wait. Restart
                    // at the top so an older ready-delivery backlog retains
                    // priority over active work newly able to stage.
                    continue;
                }
                algorithm.processQueue(ctx);
                schedulePendingDeliveryRetries();
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                return;
            } catch (Throwable t) {
                // A callback failure restores still-STAGED ownership before it
                // escapes. Consume that restore signal here so the same
                // callback cannot immediately fail again in a tight loop.
                schedulePendingDeliveryRetries();
                Logger.error("WorkerBatcher[{}] loop failed", key, t);
            }
        }
    }

    private void waitForNonEmpty() throws InterruptedException {
        queueLock.lockInterruptibly();
        try {
            while (!stopped && !ctx.hasProcessableWork()) {
                awaitStateChange();
            }
        } finally {
            queueLock.unlock();
        }
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

    /**
     * Park active route-decision work while request-mode capacity is full.
     *
     * <p>The ordered head is the only member that can start the next logical
     * decision. A BATCH_ENQUEUE head must therefore remain processable even if
     * route work exists behind it. Slot release and active-queue mutations
     * signal the same condition under {@link #queueLock}, closing the
     * check-to-wait race. The route head's absolute expiration bounds each
     * wait so deadline cleanup still runs without a capacity poll.
     *
     * @return true when a signal made the route head processable and the run
     *         loop should restart to preserve ready-backlog priority; false
     *         when no wait was needed or an expiration deadline was reached
     */
    private boolean awaitActiveRouteCapacityIfBlocked() throws InterruptedException {
        queueLock.lockInterruptibly();
        try {
            boolean waited = false;
            while (!stopped
                    && ctx.activeHeadUsesRouteDelivery()
                    && ctx.availableDeliverySlots() == 0) {
                BatchItem head = ctx.peek();
                if (head == null
                        || !BatchShape.empty().add(head)
                        .fitsCompute(ctx.batchTokenCapacity())) {
                    // The algorithm owns permanent oversize rejection. Do
                    // not let unrelated request-slot pressure hide it.
                    return false;
                }
                long expiresAtMs = ctx.activeRouteHeadExpirationAtMs();
                long nowMs = System.currentTimeMillis();
                if (expiresAtMs <= nowMs) {
                    return false;
                }
                waited = true;
                if (expiresAtMs == Long.MAX_VALUE) {
                    awaitStateChange();
                } else {
                    awaitStateChangeUntil(expiresAtMs - nowMs);
                }
            }
            return waited;
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

    private void awaitStateChangeUntil(long remainingMs) throws InterruptedException {
        waitingForSignal = true;
        try {
            stateChanged.awaitNanos(TimeUnit.MILLISECONDS.toNanos(
                    Math.max(1L, remainingMs)));
        } finally {
            waitingForSignal = false;
        }
    }

    /** Called after Prefill releases a request-mode slot, outside its stripe. */
    public void signalDeliveryCapacityAvailable() {
        queueLock.lock();
        try {
            if (ctx.readyDeliveryCount() > 0
                    || ctx.activeHeadUsesRouteDelivery()) {
                stateChanged.signal();
            }
        } finally {
            queueLock.unlock();
        }
    }

    /** Package-private deterministic wait-state probe for scheduler tests. */
    boolean isWaitingForSignal() {
        return waitingForSignal;
    }

    private void schedulePendingDeliveryRetries() {
        BatcherContext.DeliveryRetrySignal signal = ctx.consumeDeliveryRetrySignal();
        if (signal == null) {
            return;
        }
        long retryAt = System.nanoTime() + DELIVERY_RETRY_BACKOFF_NANOS;
        if (signal.ready()) {
            readyRetryNotBeforeNanos = Math.max(readyRetryNotBeforeNanos, retryAt);
        }
        if (signal.active()) {
            activeRetryItem = signal.activeRetryItem();
            activeRetryNotBeforeNanos = Math.max(activeRetryNotBeforeNanos, retryAt);
        }
    }

    private boolean readyRetryDeferred() {
        long retryAt = readyRetryNotBeforeNanos;
        if (retryAt == 0L) {
            return false;
        }
        if (retryAt - System.nanoTime() > 0L) {
            return true;
        }
        readyRetryNotBeforeNanos = 0L;
        return false;
    }

    /**
     * Wait for a ready-backlog retry only while there is no unrelated active
     * work. A new offer/removal signals the same condition and is processed
     * immediately rather than sitting behind the failed ready delivery.
     */
    private void awaitReadyRetryOrActiveWork() throws InterruptedException {
        queueLock.lockInterruptibly();
        try {
            while (!stopped && ctx.isActiveEmpty() && ctx.readyDeliveryCount() > 0) {
                long remaining = readyRetryNotBeforeNanos - System.nanoTime();
                if (remaining <= 0L) {
                    readyRetryNotBeforeNanos = 0L;
                    return;
                }
                awaitStateChangeNanos(remaining);
            }
        } finally {
            queueLock.unlock();
        }
    }

    /**
     * Delay only the restored active head. Cancellation or a newly-arrived
     * higher-priority head changes the identity predicate and ends the wait,
     * so the bounded retry cannot starve independently processable work.
     */
    private boolean awaitActiveRetryIfDeferred() throws InterruptedException {
        BatchItem deferredItem = activeRetryItem;
        long retryAt = activeRetryNotBeforeNanos;
        if (deferredItem == null || retryAt - System.nanoTime() <= 0L) {
            activeRetryItem = null;
            activeRetryNotBeforeNanos = 0L;
            return false;
        }

        queueLock.lockInterruptibly();
        try {
            boolean waited = false;
            while (!stopped && ctx.peek() == deferredItem) {
                long remaining = activeRetryNotBeforeNanos - System.nanoTime();
                if (remaining <= 0L) {
                    activeRetryItem = null;
                    activeRetryNotBeforeNanos = 0L;
                    break;
                }
                waited = true;
                awaitStateChangeNanos(remaining);
            }
            if (ctx.peek() != deferredItem) {
                activeRetryItem = null;
                activeRetryNotBeforeNanos = 0L;
            }
            return waited;
        } finally {
            queueLock.unlock();
        }
    }

    private void awaitStateChangeNanos(long remainingNanos) throws InterruptedException {
        waitingForSignal = true;
        try {
            stateChanged.awaitNanos(Math.max(1L, remainingNanos));
        } finally {
            waitingForSignal = false;
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
