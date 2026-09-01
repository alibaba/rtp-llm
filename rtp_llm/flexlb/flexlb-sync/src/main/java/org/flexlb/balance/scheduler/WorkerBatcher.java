package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.balance.endpoint.EndpointEventSink;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.projection.QueueSnapshot.AdmissionBlock;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.util.Logger;
import org.flexlb.util.PriorityOrdering;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CancellationException;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Per-worker scheduling queue that owns request grouping and hard-capacity admission,
 * delegating grouping decisions to a {@link GroupPolicy}.
 *
 * <p>One instance per Prefill worker. Exact requests enter through the neutral
 * this concrete runtime and are grouped by the configured
 * decision policy. A
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
public final class WorkerBatcher {

    /** Immutable queue state materialized under this runtime's queue lock. */
    public record QueueSnapshot(
            String endpointId,
            long queueVersion,
            int queueCapacity,
            List<ScheduledRequest> items) {
        public QueueSnapshot {
            if (queueVersion < 0L) {
                throw new IllegalArgumentException(
                        "queueVersion must be non-negative");
            }
            if (queueCapacity < 0) {
                throw new IllegalArgumentException(
                        "queueCapacity must be non-negative");
            }
            items = List.copyOf(items);
        }
    }

    public enum QueueReplacementStatus {
        SUCCESS,
        CONFLICT,
        DECLINED
    }

    /** Result of one atomic exact-victim replacement. */
    public record QueueReplacement(QueueReplacementStatus status) {
    }

    private enum RuntimeState {
        NEW,
        STARTING,
        RUNNING,
        STOPPING,
        STOPPED
    }

    /**
     * PRIORITY queue order: delegates to
     * {@link PriorityOrdering#STRICT} (priority desc → enqueue-seq asc for
     * same-priority FIFO) with {@code requestId} as the final deterministic
     * tie-break.
     *
     * <p>{@link #FIFO_QUEUE_ORDER} preserves enqueue order.
     */
    public static final Comparator<ScheduledRequest> PRIORITY_QUEUE_ORDER =
            (left, right) -> PriorityOrdering.compareWithRequestId(
                    left.priority(), left.enqueueSeq(), left.requestId(),
                    right.priority(), right.enqueueSeq(), right.requestId());

    /** FIFO order: unique monotonic enqueue sequence. */
    public static final Comparator<ScheduledRequest> FIFO_QUEUE_ORDER =
            Comparator.comparingLong(ScheduledRequest::enqueueSeq);

    private static final Comparator<GroupPlanner.Item> PRIORITY_PROJECTION_ORDER =
            (left, right) -> PriorityOrdering.compareWithRequestId(
                    left.priority(), left.enqueueSeq(), left.requestId(),
                    right.priority(), right.enqueueSeq(), right.requestId());
    private static final Comparator<GroupPlanner.Item> FIFO_PROJECTION_ORDER =
            Comparator.comparingLong(GroupPlanner.Item::enqueueSeq)
                    .thenComparingLong(GroupPlanner.Item::requestId);

    private final String key;
    private final PrefillEndpoint prefillEndpoint;
    private final EndpointEventSink deliveryLifecycle;
    private final PriorityBlockingQueue<ScheduledRequest> queue;
    private final long maximumPendingRequests;
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
    /** Canonical request ownership guarded exclusively by {@link #queueLock}. */
    private final PrefillState prefillState;
    /**
     * One per-worker condition for new queue work and endpoint-capacity release.
     * Every predicate transition and signal is serialized by queueLock, so a
     * release cannot race between the cap re-check and await.
     */
    private final Condition stateChanged = queueLock.newCondition();
    private final Thread workerThread;
    /** Constructed before this endpoint generation can begin retirement. */
    private final CancellationException normalStopFailure;
    /** Fixed diagnostic for an impossible exact stop acknowledgement loss. */
    private final IllegalStateException stopAcknowledgementFailure;
    private boolean terminationStarted;
    private boolean terminationFinished;
    private Thread terminationOwner;
    private volatile RuntimeState runtimeState = RuntimeState.NEW;
    private Throwable stopFailure;
    private volatile boolean stopped;
    private final GroupPolicy groupPolicy;
    private final BatcherContext ctx;
    private final Runnable capacityAvailableSignal =
            this::signalDeliveryCapacityAvailable;
    /** Exact resource event source for the currently blocked active head. */
    private CapacityBoundary.Availability
            subscribedCapacityAvailability;
    /** Exact active head for which this worker is waiting on a capacity event. */
    private BatcherCycleResult capacityBlockedHead;

    public WorkerBatcher(
            String key,
            PrefillEndpoint prefillEp,
            FlexlbConfig config,
            DeliveryStrategy deliveryStrategy,
            EndpointEventSink deliveryLifecycle) {
        this.key = key;
        this.prefillEndpoint = prefillEp;
        boolean queueScheduling = config.isQueue();
        boolean priorityOrdering = config.isPriorityOrdering();
        this.deliveryLifecycle = deliveryLifecycle;
        this.maximumPendingRequests = config.getRouter().getRoles()
                .getPrefill().getAvailability().getMaxPendingRequests();
        Comparator<ScheduledRequest> queueOrder = priorityOrdering
                ? PRIORITY_QUEUE_ORDER : FIFO_QUEUE_ORDER;
        Comparator<GroupPlanner.Item> projectionOrder =
                priorityOrdering
                        ? PRIORITY_PROJECTION_ORDER : FIFO_PROJECTION_ORDER;
        this.queue = new PriorityBlockingQueue<>(11, queueOrder);
        this.prefillState = new PrefillState(
                queueLock, queue, capacityAvailableSignal);
        this.groupPolicy = config.isSingleDecision()
                ? new SingleRequestGroupPolicy()
                : new FixedWindowGroupPolicy();
        this.ctx = new BatcherContext(
                key, prefillEp, config, deliveryLifecycle,
                queue, queueVersion, queueLock, queueOrder,
                projectionOrder, queueScheduling, deliveryStrategy,
                prefillState);
        this.normalStopFailure = new CancellationException(
                "FlexLB worker scheduling queue stopped: " + key);
        this.stopAcknowledgementFailure = new IllegalStateException(
                "FlexLB worker stop callback lost its exact retained owner: "
                        + key);
        this.workerThread = new Thread(this::runLoop, "flexlb-batcher-" + key);
        this.workerThread.setDaemon(true);
        this.workerThread.setUncaughtExceptionHandler((t, e) ->
                Logger.error("WorkerBatcher[{}] thread died unexpectedly", key, e));
    }

    public synchronized void start() {
        if (runtimeState != RuntimeState.NEW) {
            throw new IllegalStateException(
                    "Prefill generation runtime cannot start from "
                            + runtimeState);
        }
        runtimeState = RuntimeState.STARTING;
        try {
            workerThread.start();
            runtimeState = RuntimeState.RUNNING;
        } catch (RuntimeException | Error startFailure) {
            Throwable cleanupFailure = stopAndDrain(
                    startFailure,
                    false);
            runtimeState = RuntimeState.STOPPED;
            notifyAll();
            addSuppressedNoFail(startFailure, cleanupFailure);
            throw startFailure;
        }
    }

    public boolean offer(ScheduledRequest exactItem) {
        ScheduledRequest item = exactItem;
        requireExactEndpoint(item, "incoming item");
        if (runtimeState != RuntimeState.RUNNING || stopped) {
            return false;
        }
        return enqueue(item, ctx.maxQueueCapacity()) == OfferResult.OFFERED;
    }

    public boolean offerForPlacement(ScheduledRequest exactItem) {
        ScheduledRequest item = exactItem;
        requireExactEndpoint(item, "incoming item");
        if (runtimeState != RuntimeState.RUNNING || stopped) {
            return false;
        }
        queueLock.lock();
        try {
            if (maximumPendingRequests > 0L
                    && prefillState.pendingRequestCount()
                            >= maximumPendingRequests) {
                return false;
            }
            return enqueueUnderLock(item, ctx.maxQueueCapacity())
                    == OfferResult.OFFERED;
        } finally {
            queueLock.unlock();
        }
    }

    private enum OfferResult {
        OFFERED,
        FULL,
        STOPPED
    }

    private OfferResult enqueue(ScheduledRequest item, int maximumQueueSize) {
        queueLock.lock();
        try {
            return enqueueUnderLock(item, maximumQueueSize);
        } finally {
            queueLock.unlock();
        }
    }

    /** Caller holds {@link #queueLock}. */
    private OfferResult enqueueUnderLock(
            ScheduledRequest item, int maximumQueueSize) {
        if (stopped) {
            return OfferResult.STOPPED;
        }
        if (maximumQueueSize > 0 && queue.size() >= maximumQueueSize) {
            return OfferResult.FULL;
        }
        if (!ctx.publishActiveIndexUnderLock(item)) {
            return OfferResult.STOPPED;
        }
        stateChanged.signal();
        return OfferResult.OFFERED;
    }

    public int queueSize() {
        return queue.size();
    }

    /**
     * Snapshot of the current active queue depth bucketed by normalized
     * scheduling priority. Only priorities present in the queue appear in the
     * result, matching the tagged queue-depth metric behavior.
     */
    public Map<Integer, Integer> queueSizeByPriority() {
        Map<Integer, Integer> sizeByPriority = new HashMap<>();
        for (ScheduledRequest item : queue) {
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

    public PrefillState ownedState() {
        return prefillState;
    }

    public RouteProjection.Inputs captureRouteProjectionInputs() {
        return ctx.captureRouteProjectionInputs(this::admissionBlockUnderLock);
    }

    /** Immutable delivery semantics used by a pure route projection. */
    public RouteProjection.DeliveryProjection deliveryProjection() {
        return ctx.deliveryProjection();
    }

    /**
     * Capture only an admission rejection whose worker wait predicate still
     * holds. The availability read is the already-subscribed wait predicate;
     * it neither previews nor reserves capacity.
     *
     * <p>Caller holds {@link #queueLock}.
     */
    private AdmissionBlock admissionBlockUnderLock() {
        if (!queueLock.isHeldByCurrentThread()) {
            throw new IllegalStateException(
                    "capacity block snapshot requires queueLock");
        }
        BatcherCycleResult blocked = capacityBlockedHead;
        if (blocked == null
                || ctx.peek() != blocked.item()
                || blocked.item().requestExpired(ctx.now())
                || blocked.unavailable().availability().isAvailable()) {
            return null;
        }
        return new AdmissionBlock(
                blocked.item().requestId(),
                blocked.item().enqueueSeq(),
                blocked.unavailable().projectionSemantics());
    }

    public Throwable stopAndAwait() {
        if (Thread.currentThread() == workerThread) {
            throw new IllegalStateException(
                    "Prefill runtime cannot await its own worker thread");
        }
        Throwable failure = stopAndDrain(
                normalStopFailure,
                true);
        boolean interrupted = false;
        while (true) {
            try {
                workerThread.join();
                break;
            } catch (InterruptedException interruption) {
                interrupted = true;
            }
        }
        synchronized (this) {
            runtimeState = RuntimeState.STOPPED;
            notifyAll();
            failure = stopFailure;
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
        return failure;
    }

    private void stopAfterUnexpectedLoopFailure(Throwable loopFailure) {
        try {
            Logger.error("WorkerBatcher[{}] stopped after an unexpected loop failure",
                    key, loopFailure);
        } catch (Throwable ignoredLoggingFailure) {
            // The exact stop transaction must still run.
        }
        Throwable cleanupFailure = stopAndDrain(
                loopFailure,
                false);
        if (cleanupFailure != null) {
            try {
                Logger.error("WorkerBatcher[{}] failure cleanup exposed invariants",
                        key, cleanupFailure);
            } catch (Throwable ignoredLoggingFailure) {
                // Cleanup is already complete.
            }
        }
    }

    private Throwable stopAndDrain(
            Throwable terminalFailure,
            boolean interruptWorker) {
        boolean interrupted = false;
        synchronized (this) {
            stopped = true;
            if (terminationStarted) {
                if (!terminationFinished
                        && terminationOwner == Thread.currentThread()) {
                    throw new IllegalStateException(
                            "Prefill runtime cannot await its active stop transaction");
                }
                while (!terminationFinished) {
                    try {
                        wait();
                    } catch (InterruptedException interruption) {
                        interrupted = true;
                    }
                }
                Throwable completedFailure = stopFailure;
                if (interrupted) {
                    Thread.currentThread().interrupt();
                }
                return completedFailure;
            }
            terminationStarted = true;
            terminationOwner = Thread.currentThread();
            runtimeState = runtimeState == RuntimeState.NEW
                    ? RuntimeState.STOPPED : RuntimeState.STOPPING;
        }

        Throwable cleanupFailure = null;
        try {
            boolean queueLocked = false;
            try {
                queueLock.lock();
                queueLocked = true;
                ctx.stopAcceptingUnderLock();
                try {
                    unsubscribeFromBlockedCapacity();
                } catch (Throwable subscriptionFailure) {
                    cleanupFailure = appendCleanupFailure(
                            cleanupFailure, subscriptionFailure);
                }
                capacityBlockedHead = null;
                stateChanged.signalAll();
            } catch (Throwable wakeFailure) {
                cleanupFailure = appendCleanupFailure(
                        cleanupFailure, wakeFailure);
            } finally {
                if (queueLocked) {
                    try {
                        queueLock.unlock();
                    } catch (Throwable unlockFailure) {
                        cleanupFailure = appendCleanupFailure(
                                cleanupFailure, unlockFailure);
                    }
                }
            }

            if (interruptWorker && Thread.currentThread() != workerThread) {
                try {
                    workerThread.interrupt();
                } catch (Throwable interruptFailure) {
                    cleanupFailure = appendCleanupFailure(
                            cleanupFailure, interruptFailure);
                }
            }

            while (true) {
                ScheduledRequest item;
                try {
                    item = detachNextStoppedItem();
                } catch (Throwable claimFailure) {
                    cleanupFailure = appendCleanupFailure(
                            cleanupFailure, claimFailure);
                    break;
                }
                if (item == null) {
                    break;
                }
                try {
                    deliveryLifecycle.onQueueOfferFailure(
                            item, terminalFailure);
                } catch (Throwable callbackFailure) {
                    try {
                        Logger.error("WorkerBatcher[{}] shutdown callback failed request_id={}",
                                key, item.requestId(), callbackFailure);
                    } catch (Throwable ignoredLoggingFailure) {
                        // Cleanup ownership cannot depend on diagnostics.
                    }
                    cleanupFailure = appendCleanupFailure(
                            cleanupFailure, callbackFailure);
                    // Do not acknowledge: generation retirement still owns
                    // this exact pending identity and will project it again.
                    continue;
                }
                try {
                    if (!acknowledgeStoppedItem(item)) {
                        cleanupFailure = appendCleanupFailure(
                                cleanupFailure,
                                stopAcknowledgementFailure);
                    }
                } catch (Throwable acknowledgementFailure) {
                    // The terminal fact was already delivered. If the exact
                    // entry remains, generation retirement converges it as a
                    // stale/idempotent replay.
                    cleanupFailure = appendCleanupFailure(
                            cleanupFailure, acknowledgementFailure);
                }
            }
        } catch (Throwable unexpectedCleanupFailure) {
            cleanupFailure = appendCleanupFailure(
                    cleanupFailure, unexpectedCleanupFailure);
        } finally {
            synchronized (this) {
                stopFailure = cleanupFailure;
                terminationFinished = true;
                terminationOwner = null;
                notifyAll();
            }
        }
        return cleanupFailure;
    }

    /**
     * Detach one exact ACTIVE queue index after the stop gate is closed while
     * retaining its canonical entry. The callback result decides whether that
     * entry is acknowledged or carried into generation retirement.
     */
    private ScheduledRequest detachNextStoppedItem() {
        queueLock.lock();
        try {
            ScheduledRequest item = ctx.detachNextStopTerminalUnderLock();
            stateChanged.signalAll();
            return item;
        } finally {
            queueLock.unlock();
        }
    }

    /** Acknowledge only the retained owner whose terminal callback returned. */
    private boolean acknowledgeStoppedItem(ScheduledRequest item) {
        queueLock.lock();
        try {
            return ctx.acknowledgeStopTerminalUnderLock(item);
        } finally {
            queueLock.unlock();
        }
    }

    private static Throwable appendCleanupFailure(
            Throwable first,
            Throwable next) {
        if (first == null) {
            return next;
        }
        // Preserve cleanup totality under allocation failure. The first causal
        // failure is sufficient; later leaves were still all attempted.
        return first;
    }

    private static void addSuppressedNoFail(
            Throwable primary,
            Throwable leaf) {
        if (primary == null || leaf == null || primary == leaf) {
            return;
        }
        try {
            primary.addSuppressed(leaf);
        } catch (Throwable ignoredAggregationFailure) {
            // The start transaction already drained every acquired leaf.
        }
    }

    // ==================== priority scheduling queue operations ====================

    public boolean removeQueued(
            ScheduledRequest exactItem,
            String reason) {
        ScheduledRequest item = exactItem;
        requireExactEndpoint(item, "queued item");
        boolean removed;
        queueLock.lock();
        try {
            removed = runtimeState == RuntimeState.RUNNING
                    && !stopped
                    && ctx.removeUnderLock(item);
            if (removed) {
                stateChanged.signal();
            }
        } finally {
            queueLock.unlock();
        }
        if (removed) {
            prefillEndpoint.signalPlacementCapacityChanged();
            try {
                Logger.debug(
                        "[request-scheduler] exact queue remove: worker={} reason={} request_id={}",
                        key, reason, item.requestId());
            } catch (Throwable ignoredLoggingFailure) {
                // Diagnostics cannot turn a committed exact removal into failure.
            }
        }
        return removed;
    }

    private void requireExactEndpoint(
            ScheduledRequest item, String operation) {
        if (item.prefillEp() != prefillEndpoint) {
            throw new IllegalArgumentException(
                    operation + " belongs to another Prefill generation");
        }
    }

    public QueueReplacement replaceQueued(
            List<ScheduledRequest> exactVictims,
            ScheduledRequest incoming) {
        QueueReplacement committed = new QueueReplacement(
                QueueReplacementStatus.SUCCESS);
        queueLock.lock();
        try {
            if (runtimeState != RuntimeState.RUNNING || stopped) {
                return new QueueReplacement(
                        QueueReplacementStatus.DECLINED);
            }
            int maximumQueueSize = ctx.maxQueueCapacity();
            int victimsRequiredNow = maximumQueueSize <= 0
                    ? 0
                    : Math.max(0, queue.size() + 1 - maximumQueueSize);
            if (victimsRequiredNow == 0
                    || exactVictims.size() != victimsRequiredNow) {
                return new QueueReplacement(
                        QueueReplacementStatus.DECLINED);
            }
            int postSwapSize = queue.size() - exactVictims.size() + 1;
            if (postSwapSize < 0
                    || (maximumQueueSize > 0
                    && postSwapSize > maximumQueueSize)) {
                return new QueueReplacement(
                        QueueReplacementStatus.DECLINED);
            }
            PrefillState.ActiveReplaceStatus replacement =
                    prefillState.replaceActiveExact(exactVictims, incoming);
            if (replacement
                    == PrefillState.ActiveReplaceStatus.CONFLICT) {
                return new QueueReplacement(
                        QueueReplacementStatus.CONFLICT);
            }
            queueVersion.incrementAndGet();
            stateChanged.signal();
            return committed;
        } finally {
            queueLock.unlock();
        }
    }

    public QueueSnapshot captureQueueSnapshot() {
        queueLock.lock();
        try {
            return new QueueSnapshot(
                    key,
                    queueVersion.get(),
                    ctx.maxQueueCapacity(),
                    ctx.activeItemsInSchedulingOrder());
        } finally {
            queueLock.unlock();
        }
    }

    // ==================== Internal: Run loop ====================

    private void runLoop() {
        try {
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
        } finally {
            synchronized (this) {
                runtimeState = RuntimeState.STOPPED;
                notifyAll();
            }
        }
    }

    private void runOneCycle() throws InterruptedException {
        waitForNonEmpty();
        if (stopped) {
            return;
        }

        BatcherCycleResult result = groupPolicy.processQueue(ctx);
        if (result.status() == BatcherCycleResult.Status.ADMITTED
                || result == BatcherCycleResult.QUEUE_CHANGED) {
            // Only a committed removal creates a new queue seat. Advisory
            // waits and delivery-capacity misses must not feed placement back
            // into itself.
            prefillEndpoint.signalPlacementCapacityChanged();
        }
        if (result.status() == BatcherCycleResult.Status.CAPACITY_BLOCKED) {
            awaitBlockedHeadCapacity(result);
        } else if (result.status()
                == BatcherCycleResult.Status.AWAITING_SCHEDULING_CHANGE) {
            awaitSchedulingChange(result);
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
            BatcherCycleResult blocked) {
        CapacityBoundary.Availability nextSource =
                blocked.unavailable().availability();
        if (nextSource != subscribedCapacityAvailability) {
            unsubscribeFromBlockedCapacity();
            subscribedCapacityAvailability = nextSource;
            nextSource.addListener(capacityAvailableSignal);
        }
    }

    /** Caller holds queueLock. */
    private void unsubscribeFromBlockedCapacity() {
        CapacityBoundary.Availability source =
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
            BatcherCycleResult blocked)
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
                    && !blocked.item().requestExpired(System.currentTimeMillis())
                    && !blocked.unavailable().availability().isAvailable()) {
                long expiresAtMs = blocked.item().expiresAtMs();
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
     * Wait for the exact queue/input generation captured by the policy or
     * for its deadline. Every predicate and signal is serialized by queueLock.
     */
    private void awaitSchedulingChange(
            BatcherCycleResult waiting)
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
        stateChanged.await();
    }

    private void awaitStateChangeUntil(long remainingMs) throws InterruptedException {
        stateChanged.awaitNanos(TimeUnit.MILLISECONDS.toNanos(
                Math.max(1L, remainingMs)));
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

}
