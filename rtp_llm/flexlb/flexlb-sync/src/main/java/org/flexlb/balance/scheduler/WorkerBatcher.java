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
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Per-worker scheduling queue that owns request grouping and hard-capacity admission,
 * delegating grouping decisions to a {@link BatcherAlgorithm}.
 *
 * <p>One instance per Prefill worker. Requests are submitted via
 * {@link #offer(BatchItem)} and grouped by the configured decision policy. A
 * group may independently be delivered through
 * EnqueueBatch or as individual route decisions.
 *
 * <p>The context-owned active queue is the single source of truth for queued
 * requests. PRIORITY ordering uses
 * the explicit priority comparator ({@link #PRIORITY_QUEUE_ORDER}); FIFO uses
 * a unique monotonic enqueue sequence. All mutations go through
 * {@link BatcherContext} (or the
 * lock-holding methods here). A monotonic mutation generation lets snapshots
 * and diagnostics identify queue-state changes.
 */
public class WorkerBatcher {

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
    private final DecisionGroupHandler decisionHandler;
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
     * One per-worker condition for new queue work and endpoint-capacity release.
     * Every predicate transition and signal is serialized by queueLock, so a
     * release cannot race between the cap re-check and await.
     */
    private final Condition stateChanged = queueLock.newCondition();
    private final Thread workerThread;
    private final AtomicBoolean terminationStarted = new AtomicBoolean();
    private volatile boolean stopped;
    private volatile boolean waitingForSignal;
    private final BatcherAlgorithm algorithm;
    private final BatcherContext ctx;
    private final PrefillQueueManager queueManager;
    private final Runnable capacityAvailableSignal =
            this::signalDeliveryCapacityAvailable;
    /** Exact resource event source for the currently blocked active head. */
    private DeliveryCapacityAdmission.CapacityAvailability
            subscribedCapacityAvailability;
    /** Exact active head for which this worker is waiting on a capacity event. */
    private BatcherCycleResult.CapacityBlocked capacityBlockedHead;

    public WorkerBatcher(String key, PrefillEndpoint prefillEp, FlexlbConfig config,
                         DecisionGroupHandler decisionHandler,
                         DeliveryCapacityAdmission capacityAdmission,
                         BatchSchedulerReporter reporter) {
        this.key = key;
        this.decisionHandler = decisionHandler;
        Comparator<BatchItem> queueOrder = config.isPriorityOrdering()
                ? PRIORITY_QUEUE_ORDER : FIFO_QUEUE_ORDER;
        this.queue = new PriorityBlockingQueue<>(11, queueOrder);
        this.algorithm = config.isSingleDecision()
                ? new SingleRequestBatcherAlgorithm()
                : new FixedWindowBatcherAlgorithm();
        this.ctx = new BatcherContext(
                key, prefillEp, config, decisionHandler, capacityAdmission,
                queue, queueDepth, queueVersion, queueLock, queueOrder, reporter);
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
     * Snapshot of the current active queue depth bucketed by normalized
     * scheduling priority. Only priorities present in the queue appear in the
     * result, matching the empty-bucket behavior of wait-time metrics.
     */
    public Map<Integer, Integer> queueSizeByPriority() {
        Map<Integer, Integer> sizeByPriority = new HashMap<>();
        for (BatchItem item : queue) {
            sizeByPriority.merge(item.priority(), 1, Integer::sum);
        }
        return sizeByPriority;
    }

    /** Current active queue mutation generation. */
    public long queueVersion() {
        return queueVersion.get();
    }

    /** Current generation of worker-status and prediction-model inputs. */
    public long schedulingInputVersion() {
        return ctx.schedulingInputVersionValue();
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

    int callbackOwnedRequestCount() {
        return ctx.callbackOwnedRequestCount();
    }

    public void shutdown() {
        stopAndDrain(
                "FlexLB worker scheduling queue stopped: " + key,
                null,
                true);
    }

    private void stopAfterUnexpectedLoopFailure(Throwable loopFailure) {
        Logger.error("WorkerBatcher[{}] stopped after an unexpected loop failure",
                key, loopFailure);
        stopAndDrain(
                "FlexLB worker scheduling queue failed: " + key,
                loopFailure,
                false);
    }

    private void stopAndDrain(
            String failureMessage,
            Throwable failureCause,
            boolean interruptWorker) {
        stopped = true;
        if (!terminationStarted.compareAndSet(false, true)) {
            if (interruptWorker && Thread.currentThread() != workerThread) {
                workerThread.interrupt();
            }
            return;
        }
        List<BatchItem> remaining = new ArrayList<>();
        ctx.stopAndDrainTo(remaining);
        queueLock.lock();
        try {
            unsubscribeFromBlockedCapacity();
            capacityBlockedHead = null;
            stateChanged.signalAll();
        } finally {
            queueLock.unlock();
        }
        if (interruptWorker && Thread.currentThread() != workerThread) {
            workerThread.interrupt();
        }
        try {
            algorithm.onShutdown(ctx);
        } catch (Throwable shutdownFailure) {
            Logger.error("WorkerBatcher[{}] algorithm shutdown failed", key, shutdownFailure);
        }
        for (BatchItem item : remaining) {
            try {
                decisionHandler.onOfferFailure(item,
                        failureCause == null
                                ? new CancellationException(failureMessage)
                                : new IllegalStateException(
                                        failureMessage, failureCause));
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
                runOneCycle();
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                return;
            } catch (Throwable t) {
                // Every expected delivery failure is terminalized inside the
                // typed cycle. An escaping Throwable is therefore an invariant
                // failure; retrying the same ACTIVE state can only spin.
                stopAfterUnexpectedLoopFailure(t);
                return;
            }
        }
    }

    private void runOneCycle() throws InterruptedException {
        waitForNonEmpty();
        if (stopped) {
            return;
        }

        BatcherCycleResult result = algorithm.processQueue(ctx);
        if (result instanceof BatcherCycleResult.CapacityBlocked blocked) {
            awaitBlockedHeadCapacity(blocked);
        } else if (result
                instanceof BatcherCycleResult.AwaitingSchedulingChange waiting) {
            awaitSchedulingChange(waiting);
        }
    }

    private void waitForNonEmpty() throws InterruptedException {
        queueLock.lockInterruptibly();
        try {
            while (!stopped && !ctx.hasProcessableWork()) {
                capacityBlockedHead = null;
                unsubscribeFromBlockedCapacity();
                awaitStateChange();
            }
        } finally {
            queueLock.unlock();
        }
    }

    /**
     * Subscribe to the exact resource which rejected the active head.
     * Caller holds {@link #queueLock}.
     */
    private void subscribeToBlockedCapacity(
            BatcherCycleResult.CapacityBlocked blocked) {
        DeliveryCapacityAdmission.CapacityAvailability nextSource =
                blocked.unavailable().availability();
        if (nextSource != subscribedCapacityAvailability) {
            unsubscribeFromBlockedCapacity();
            subscribedCapacityAvailability = nextSource;
            nextSource.addListener(capacityAvailableSignal);
        }
    }

    /** Caller holds queueLock. */
    private void unsubscribeFromBlockedCapacity() {
        DeliveryCapacityAdmission.CapacityAvailability source =
                subscribedCapacityAvailability;
        if (source == null) {
            return;
        }
        subscribedCapacityAvailability = null;
        source.removeListener(capacityAvailableSignal);
    }

    /**
     * Wait while the exact active head is blocked by the exact rejecting
     * resource. Offers wake the condition: a new higher-priority head exits the
     * wait immediately, while a tail offer leaves the predicate true. Capacity
     * is checked after listener installation while queueLock is held, closing
     * the release-before-await race without polling.
     */
    private void awaitBlockedHeadCapacity(
            BatcherCycleResult.CapacityBlocked blocked)
            throws InterruptedException {
        queueLock.lockInterruptibly();
        try {
            if (stopped) {
                return;
            }
            capacityBlockedHead = blocked;
            subscribeToBlockedCapacity(blocked);
            while (!stopped
                    && ctx.peek() == blocked.item()
                    && !blocked.item().ctx().requestExpired(System.currentTimeMillis())
                    && !blocked.unavailable().availability().isAvailable()) {
                long expiresAtMs = blocked.item().ctx().getRequestExpiresAtMs();
                long nowMs = System.currentTimeMillis();
                if (expiresAtMs <= nowMs) {
                    return;
                }
                if (expiresAtMs == Long.MAX_VALUE) {
                    awaitStateChange();
                } else {
                    awaitStateChangeUntil(expiresAtMs - nowMs);
                }
            }
        } finally {
            capacityBlockedHead = null;
            unsubscribeFromBlockedCapacity();
            queueLock.unlock();
        }
    }

    /**
     * Wait for the exact queue/input generation captured by the algorithm or
     * for its deadline. Every predicate and signal is serialized by queueLock.
     */
    private void awaitSchedulingChange(
            BatcherCycleResult.AwaitingSchedulingChange waiting)
            throws InterruptedException {
        queueLock.lockInterruptibly();
        try {
            while (!stopped
                    && ctx.peek() == waiting.head()
                    && queueVersion.get() == waiting.queueVersion()
                    && ctx.schedulingInputVersionValue()
                    == waiting.schedulingInputVersion()) {
                long nowMs = System.currentTimeMillis();
                if (waiting.wakeAtMs() <= nowMs) {
                    return;
                }
                if (waiting.wakeAtMs() == Long.MAX_VALUE) {
                    awaitStateChange();
                } else {
                    awaitStateChangeUntil(waiting.wakeAtMs() - nowMs);
                }
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

    private void awaitStateChangeUntil(long remainingMs) throws InterruptedException {
        waitingForSignal = true;
        try {
            stateChanged.awaitNanos(TimeUnit.MILLISECONDS.toNanos(
                    Math.max(1L, remainingMs)));
        } finally {
            waitingForSignal = false;
        }
    }

    /** Called after Prefill or Decode capacity changes, outside endpoint locks. */
    public void signalDeliveryCapacityAvailable() {
        queueLock.lock();
        try {
            if (capacityBlockedHead != null) {
                stateChanged.signal();
            }
        } finally {
            queueLock.unlock();
        }
    }

    /** Wake decisions whose advisory worker-status or predictor input changed. */
    public void signalSchedulingInputsChanged() {
        queueLock.lock();
        try {
            ctx.incrementSchedulingInputVersion();
            stateChanged.signal();
        } finally {
            queueLock.unlock();
        }
    }

    /** Package-private deterministic wait-state probe for scheduler tests. */
    boolean isWaitingForSignal() {
        return waitingForSignal;
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
