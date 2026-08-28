package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.InvalidPrefillPredictionException;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.QueueSnapshot.AdmissionBlock;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.util.Logger;
import org.flexlb.util.PriorityOrdering;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.concurrent.CancellationException;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Supplier;

/**
 * Per-worker scheduling queue that owns queue state, grouping, hard-capacity
 * admission and delivery handoff.
 *
 * <p>One instance belongs to each Prefill worker. Exact requests enter this
 * concrete runtime and are grouped by the configured decision mode. A group
 * may independently be delivered through
 * EnqueueBatch or as individual route decisions.
 *
 * <p>The active queue is the single source of truth for queued requests.
 * PRIORITY ordering uses
 * the explicit priority comparator ({@link #PRIORITY_QUEUE_ORDER}); FIFO uses
 * a unique monotonic enqueue sequence. A monotonic mutation generation lets
 * snapshots and diagnostics identify queue-state changes.
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

    private enum RuntimeState {
        NEW,
        STARTING,
        RUNNING,
        STOPPING,
        STOPPED
    }

    /** Exact predicate captured by one scheduling cycle. */
    private record BatcherCycleResult(
            boolean placementCapacityChanged,
            ScheduledRequest request,
            CapacityBoundary unavailable,
            long queueVersion,
            long schedulingInputVersion,
            long wakeAtMs) {

        private static final BatcherCycleResult NO_ACTION = simple(false);
        private static final BatcherCycleResult CAPACITY_CHANGED = simple(true);

        private BatcherCycleResult {
            boolean waiting = request != null;
            if ((placementCapacityChanged && waiting)
                    || (!waiting && unavailable != null)) {
                throw new IllegalArgumentException(
                        "worker cycle requires either a capacity change or an exact wait predicate");
            }
        }

        private static BatcherCycleResult capacityBlocked(
                ScheduledRequest item,
                CapacityBoundary unavailable) {
            return new BatcherCycleResult(false,
                    Objects.requireNonNull(item, "item"),
                    Objects.requireNonNull(unavailable, "unavailable"),
                    0L, 0L, 0L);
        }

        private static BatcherCycleResult awaitingSchedulingChange(
                ScheduledRequest head,
                long queueVersion,
                long schedulingInputVersion,
                long wakeAtMs) {
            return new BatcherCycleResult(false,
                    Objects.requireNonNull(head, "head"), null,
                    queueVersion, schedulingInputVersion, wakeAtMs);
        }

        private static BatcherCycleResult simple(boolean capacityChanged) {
            return new BatcherCycleResult(
                    capacityChanged, null, null, 0L, 0L, 0L);
        }

        private boolean capacityBlocked() {
            return unavailable != null;
        }

        private boolean awaitingSchedulingChange() {
            return request != null && unavailable == null;
        }
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
    private static final GroupPlanner.ItemAccess<ScheduledRequest>
            PLANNER_ITEM_ACCESS = new GroupPlanner.ItemAccess<>() {
                @Override
                public long enqueuedAtMs(ScheduledRequest item) {
                    return item.enqueuedAtMs();
                }

                @Override
                public long seqLen(ScheduledRequest item) {
                    return item.seqLen();
                }
            };
    private static final String SINGLE_DECISION_REASON = "single_request";

    private final String key;
    private final PrefillEndpoint prefillEndpoint;
    private final EndpointEventProjector endpointEvents;
    private final FlexlbConfig config;
    private final DecisionPolicyConfig fixedWindowDecision;
    private final boolean singleDecision;
    private final boolean queueScheduling;
    private final DeliveryStrategy deliveryStrategy;
    private final PriorityBlockingQueue<ScheduledRequest> queue;
    private final Comparator<ScheduledRequest> queueOrder;
    private final Comparator<GroupPlanner.Item> projectionOrder;
    private final long maximumPendingRequests;
    /**
     * Monotonic queue mutation generation, bumped on enqueue, removal,
     * delivery and drain. It is exposed in diagnostic snapshots.
     */
    private final AtomicLong queueVersion = new AtomicLong();
    /** Worker-status and predictor generation used by optimistic decisions. */
    private final AtomicLong schedulingInputVersion = new AtomicLong();
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
            EndpointEventProjector endpointEvents) {
        this.key = key;
        this.prefillEndpoint = prefillEp;
        this.config = config;
        this.queueScheduling = config.isQueue();
        this.singleDecision = config.isSingleDecision();
        DecisionPolicyConfig resolvedDecision = queueScheduling
                ? config.decisionPolicy() : null;
        this.fixedWindowDecision = resolvedDecision != null
                && resolvedDecision.getType()
                == DecisionPolicyConfig.Type.FIXED_WINDOW
                ? resolvedDecision : null;
        boolean priorityOrdering = config.isPriorityOrdering();
        this.endpointEvents = Objects.requireNonNull(
                endpointEvents, "endpointEvents");
        this.deliveryStrategy = Objects.requireNonNull(
                deliveryStrategy, "deliveryStrategy");
        this.maximumPendingRequests = config.getRouter().getRoles()
                .getPrefill().getAvailability().getMaxPendingRequests();
        this.queueOrder = priorityOrdering
                ? PRIORITY_QUEUE_ORDER : FIFO_QUEUE_ORDER;
        this.projectionOrder =
                priorityOrdering
                        ? PRIORITY_PROJECTION_ORDER : FIFO_PROJECTION_ORDER;
        this.queue = new PriorityBlockingQueue<>(11, queueOrder);
        this.prefillState = new PrefillState(
                queueLock, queue, capacityAvailableSignal);
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
        return enqueue(item, maxQueueCapacity());
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
            return enqueueUnderLock(item, maxQueueCapacity());
        } finally {
            queueLock.unlock();
        }
    }

    private boolean enqueue(ScheduledRequest item, int maximumQueueSize) {
        queueLock.lock();
        try {
            return enqueueUnderLock(item, maximumQueueSize);
        } finally {
            queueLock.unlock();
        }
    }

    /** Caller holds {@link #queueLock}. */
    private boolean enqueueUnderLock(
            ScheduledRequest item, int maximumQueueSize) {
        if (stopped) {
            return false;
        }
        if (maximumQueueSize > 0 && queue.size() >= maximumQueueSize) {
            return false;
        }
        if (!publishActiveIndexUnderLock(item)) {
            return false;
        }
        stateChanged.signal();
        return true;
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

    public PrefillState ownedState() {
        return prefillState;
    }

    public RouteProjection.Inputs captureRouteProjectionInputs() {
        return captureRouteProjectionInputs(this::admissionBlockUnderLock);
    }

    /** Immutable delivery semantics used by a pure route projection. */
    public RouteProjection.DeliveryProjection deliveryProjection() {
        return deliveryStrategy.projectionPolicy();
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
                || queue.peek() != blocked.request()
                || blocked.request().requestExpired(now())
                || blocked.unavailable().availability().isAvailable()) {
            return null;
        }
        return new AdmissionBlock(
                blocked.request().requestId(),
                blocked.request().enqueueSeq(),
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
                stopAcceptingUnderLock();
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
                    endpointEvents.onQueueOfferFailure(
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
            ScheduledRequest item = detachNextStopTerminalUnderLock();
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
            return acknowledgeStopTerminalUnderLock(item);
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
                    && removeUnderLock(item);
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

    public QueueReplacementStatus replaceQueued(
            List<ScheduledRequest> exactVictims,
            ScheduledRequest incoming) {
        queueLock.lock();
        try {
            if (runtimeState != RuntimeState.RUNNING || stopped) {
                return QueueReplacementStatus.DECLINED;
            }
            int maximumQueueSize = maxQueueCapacity();
            int victimsRequiredNow = maximumQueueSize <= 0
                    ? 0
                    : Math.max(0, queue.size() + 1 - maximumQueueSize);
            if (victimsRequiredNow == 0
                    || exactVictims.size() != victimsRequiredNow) {
                return QueueReplacementStatus.DECLINED;
            }
            int postSwapSize = queue.size() - exactVictims.size() + 1;
            if (postSwapSize < 0
                    || (maximumQueueSize > 0
                    && postSwapSize > maximumQueueSize)) {
                return QueueReplacementStatus.DECLINED;
            }
            if (!prefillState.replaceActiveExact(exactVictims, incoming)) {
                return QueueReplacementStatus.CONFLICT;
            }
            queueVersion.incrementAndGet();
            stateChanged.signal();
            return QueueReplacementStatus.SUCCESS;
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
                    maxQueueCapacity(),
                    activeItemsInSchedulingOrder());
        } finally {
            queueLock.unlock();
        }
    }

    // ==================== Queue ownership and projection ====================

    private int maxQueueCapacity() {
        return config.queueScheduler().getCapacity()
                .getMaxWaitingRequestsPerPrefillWorker();
    }

    private int maxDecisionRequests() {
        return fixedWindowDecision == null
                ? 1 : Math.max(1, fixedWindowDecision.getMaxRequests());
    }

    private long collectionWindowMs() {
        return fixedWindowDecision == null
                ? 0L : Math.max(0L,
                fixedWindowDecision.getMaxCollectionWaitMs());
    }

    private long predictedExecutionBudgetMs() {
        if (fixedWindowDecision == null) {
            return 0L;
        }
        Long configured = fixedWindowDecision.getMaxPredictedExecutionMs();
        return configured == null ? 0L : configured;
    }

    private static long now() {
        return System.currentTimeMillis();
    }

    /** Caller holds {@link #queueLock}. */
    private boolean publishActiveIndexUnderLock(ScheduledRequest item) {
        boolean published = prefillState.enqueueActiveUnderLock(item);
        if (published) {
            queueVersion.incrementAndGet();
        }
        return published;
    }

    /** Caller holds {@link #queueLock}. */
    private boolean removeTerminalActiveUnderLock(ScheduledRequest item) {
        return prefillState.terminalizeActiveUnderLock(item);
    }

    /** Caller holds {@link #queueLock}. */
    private void stopAcceptingUnderLock() {
        if (!queueLock.isHeldByCurrentThread()) {
            throw new IllegalStateException(
                    "queue mutation requires queueLock");
        }
        stopped = true;
    }

    /** Caller holds {@link #queueLock}. */
    private ScheduledRequest detachNextStopTerminalUnderLock() {
        ScheduledRequest item =
                prefillState.detachNextActiveForStopUnderLock();
        if (item != null) {
            queueVersion.incrementAndGet();
        }
        return item;
    }

    /** Caller holds {@link #queueLock}. */
    private boolean acknowledgeStopTerminalUnderLock(
            ScheduledRequest item) {
        return prefillState.acknowledgeStopTerminalUnderLock(item);
    }

    /** Caller holds {@link #queueLock}. */
    private boolean removeUnderLock(ScheduledRequest item) {
        boolean removed = removeTerminalActiveUnderLock(item);
        if (removed) {
            queueVersion.incrementAndGet();
        }
        return removed;
    }

    private List<ScheduledRequest> activeItemsInSchedulingOrder() {
        List<ScheduledRequest> candidates = new ArrayList<>(queue);
        candidates.sort(queueOrder);
        return candidates;
    }

    private ActiveQueueSnapshot snapshotActiveQueueHead() {
        queueLock.lock();
        try {
            ScheduledRequest head = queue.peek();
            return new ActiveQueueSnapshot(
                    queueVersion.get(), schedulingInputVersion.get(),
                    head == null ? List.of() : List.of(head));
        } finally {
            queueLock.unlock();
        }
    }

    private ActiveQueueSnapshot snapshotActiveQueue() {
        long version;
        long inputVersion;
        List<ScheduledRequest> items;
        queueLock.lock();
        try {
            version = queueVersion.get();
            inputVersion = schedulingInputVersion.get();
            if (queue.isEmpty()) {
                return new ActiveQueueSnapshot(
                        version, inputVersion, List.of());
            }
            items = new ArrayList<>(queue);
        } finally {
            queueLock.unlock();
        }
        items.sort(queueOrder);
        return new ActiveQueueSnapshot(version, inputVersion, items);
    }

    private record ActiveQueueSnapshot(
            long queueVersion,
            long schedulingInputVersion,
            List<ScheduledRequest> items) {
        ScheduledRequest head() {
            return items.isEmpty() ? null : items.get(0);
        }
    }

    private RouteProjection.Inputs captureRouteProjectionInputs(
            Supplier<AdmissionBlock> admissionBlockSnapshot) {
        queueLock.lock();
        try {
            BatchCapacitySnapshot capacity = batchCapacitySnapshot();
            PrefillState.Snapshot ownership =
                    prefillState.snapshotUnderLock(queueOrder);
            List<GroupPlanner.Item> items = ownership.activeItems().stream()
                    .map(WorkerBatcher::projectionItem)
                    .toList();
            org.flexlb.balance.projection.QueueSnapshot queueSnapshot =
                    new org.flexlb.balance.projection.QueueSnapshot(
                            ownership.capturedAtMs(),
                            queueScheduling,
                            projectionOrder,
                            new GroupPlanner.Constraints(
                                    maxDecisionRequests(),
                                    capacity.batchTokenCapacity(),
                                    capacity.batchKvCapacity(),
                                    predictedExecutionBudgetMs(),
                                    collectionWindowMs()),
                            items,
                            items.isEmpty()
                                    ? null : admissionBlockSnapshot.get());
            return new RouteProjection.Inputs(
                    queueSnapshot,
                    ownership.committedWork(),
                    ownership.pendingRequestCount());
        } finally {
            queueLock.unlock();
        }
    }

    private static GroupPlanner.Item projectionItem(ScheduledRequest item) {
        return new GroupPlanner.Item(
                item.requestId(), item.priority(), item.enqueueSeq(),
                item.enqueuedAtMs(), item.expiresAtMs(), item.seqLen(),
                item.hitCache());
    }

    private BatchCapacitySnapshot batchCapacitySnapshot() {
        long capacity = positiveOrUnlimited(
                config.getInternalRuntime()
                        .getFallbackBatchTokenCapacity());
        WorkerStatus status = prefillEndpoint != null
                ? prefillEndpoint.getStatus() : null;
        if (status == null) {
            return new BatchCapacitySnapshot(capacity, Long.MAX_VALUE);
        }
        WorkerStatus.EngineObservation engineStatus =
                status.committedEngineObservation();
        long engineCapacity = engineStatus.maxBatchTokensSize();
        if (engineCapacity <= 0) {
            engineCapacity = engineStatus.maxSeqLen();
        }
        long batchTokenCapacity = Math.min(
                capacity, positiveOrUnlimited(engineCapacity));
        long total = engineStatus.totalKvCacheTokens();
        if (total <= 0) {
            return new BatchCapacitySnapshot(
                    batchTokenCapacity, Long.MAX_VALUE);
        }
        long available = Math.max(
                0, engineStatus.availableKvCacheTokens());
        return new BatchCapacitySnapshot(
                batchTokenCapacity, Math.min(total, available));
    }

    private record BatchCapacitySnapshot(
            long batchTokenCapacity,
            long batchKvCapacity) {
    }

    private static long positiveOrUnlimited(long value) {
        return value > 0 ? value : Long.MAX_VALUE;
    }

    // ==================== Delivery ownership ====================

    private BatcherCycleResult admitAndDeliverCapacityFeasiblePrefix(
            List<ScheduledRequest> candidates,
            String decisionReason,
            PrefillTimePredictor.Evaluator evaluator,
            OptionalLong plannedCommittedPredictionMs) {
        if (candidates.isEmpty() || !selectionStillOwned(candidates)) {
            return BatcherCycleResult.NO_ACTION;
        }
        try (DeliveryStrategy.Transaction transaction =
                deliveryStrategy.prepare(
                        candidates, evaluator,
                        plannedCommittedPredictionMs)) {
            if (transaction.items().isEmpty()) {
                return commitBoundary(
                        transaction.blockedItem(),
                        transaction.blockedResult());
            }
            BatcherCycleResult admitted = commitPreparedSelection(
                    transaction, decisionReason);
            return admitted == null
                    ? BatcherCycleResult.NO_ACTION : admitted;
        }
    }

    private double projectGroupDurationMs(
            List<ScheduledRequest> items,
            PrefillTimePredictor.Evaluator evaluator) {
        return deliveryStrategy.projectGroupDurationMs(items, evaluator);
    }

    private void handoff(
            DeliveryStrategy.Transaction transaction,
            String decisionReason,
            int remainingQueueDepth) {
        Throwable deliveryFailure = null;
        try {
            transaction.handoff(decisionReason, remainingQueueDepth);
        } catch (Throwable failure) {
            deliveryFailure = failure;
        }
        Throwable unresolved = deliveryFailure != null
                ? deliveryFailure
                : new IllegalStateException(
                        "delivery returned without resolving owner");
        try {
            transaction.abort(unresolved);
        } catch (Throwable cleanupFailure) {
            if (deliveryFailure == null) {
                deliveryFailure = cleanupFailure;
            } else if (deliveryFailure != cleanupFailure) {
                deliveryFailure.addSuppressed(cleanupFailure);
            }
        }
        if (deliveryFailure != null) {
            Logger.error("WorkerBatcher[{}] committed delivery failed",
                    key, deliveryFailure);
        }
    }

    private BatcherCycleResult runPredictionBound(
            ScheduledRequest exactHead,
            Supplier<BatcherCycleResult> operation) {
        try {
            return operation.get();
        } catch (InvalidPrefillPredictionException failure) {
            return commitBoundary(
                    exactHead, CapacityBoundary.failed(failure));
        }
    }

    private boolean selectionStillOwned(
            List<ScheduledRequest> candidates) {
        queueLock.lock();
        try {
            if (stopped) {
                return false;
            }
            long nowMs = now();
            for (ScheduledRequest item : candidates) {
                if (!containsIdentity(item)
                        || item.requestExpired(nowMs)) {
                    return false;
                }
            }
            return true;
        } finally {
            queueLock.unlock();
        }
    }

    private BatcherCycleResult commitPreparedSelection(
            DeliveryStrategy.Transaction transaction,
            String decisionReason) {
        List<ScheduledRequest> items = transaction.items();
        if (items.isEmpty()) {
            throw new IllegalStateException(
                    "prepared selection commit requires a non-empty selection");
        }
        String committedReason = null;
        int remainingQueueDepth = 0;
        boolean removedTerminalBoundary = false;
        Throwable postCommitFailure = null;
        queueLock.lock();
        try {
            if (stopped) {
                return null;
            }
            long nowMs = now();
            for (ScheduledRequest item : items) {
                if (!containsIdentity(item)
                        || item.requestExpired(nowMs)) {
                    return null;
                }
            }
            transaction.commitUnderLock();
            try {
                removedTerminalBoundary = removeSelectionBoundaryUnderLock(
                        transaction.blockedItem(),
                        transaction.blockedResult(), nowMs);
                queueVersion.incrementAndGet();
                committedReason = transaction.blockedResult() != null
                        && transaction.blockedResult().unavailable()
                        ? "delivery_capacity_prefix" : decisionReason;
                remainingQueueDepth = queue.size();
            } catch (Throwable failure) {
                postCommitFailure = failure;
            }
        } finally {
            queueLock.unlock();
        }

        if (postCommitFailure != null) {
            Throwable failure = postCommitFailure;
            try {
                notifyTerminalAdmissionFailure(
                        removedTerminalBoundary,
                        transaction.blockedItem(),
                        transaction.blockedResult());
            } catch (Throwable notificationFailure) {
                failure = appendFailure(failure, notificationFailure);
            }
            try {
                transaction.abort(failure);
            } catch (Throwable ownerFailure) {
                failure = appendFailure(failure, ownerFailure);
            }
            throw propagateCommitFailure(failure);
        }
        notifyTerminalAdmissionFailure(
                removedTerminalBoundary,
                transaction.blockedItem(),
                transaction.blockedResult());
        handoff(transaction, Objects.requireNonNull(committedReason),
                remainingQueueDepth);
        return BatcherCycleResult.CAPACITY_CHANGED;
    }

    private BatcherCycleResult commitBoundary(
            ScheduledRequest blockedItem,
            CapacityBoundary blockedResult) {
        BatcherCycleResult result = BatcherCycleResult.NO_ACTION;
        boolean removedTerminalBoundary = false;
        queueLock.lock();
        try {
            if (stopped) {
                return result;
            }
            long nowMs = now();
            if (blockedResult != null && blockedResult.unavailable()) {
                result = resolveEmptyCapacityUnderLock(
                        blockedItem, blockedResult, nowMs);
            } else {
                removedTerminalBoundary = removeSelectionBoundaryUnderLock(
                        blockedItem, blockedResult, nowMs);
                if (removedTerminalBoundary) {
                    queueVersion.incrementAndGet();
                    result = BatcherCycleResult.CAPACITY_CHANGED;
                }
            }
        } finally {
            queueLock.unlock();
        }
        notifyTerminalAdmissionFailure(
                removedTerminalBoundary, blockedItem, blockedResult);
        return result;
    }

    /** Caller holds {@link #queueLock}. */
    private BatcherCycleResult resolveEmptyCapacityUnderLock(
            ScheduledRequest item,
            CapacityBoundary unavailable,
            long nowMs) {
        if (queue.peek() != item
                || item.requestExpired(nowMs)
                || !containsIdentity(item)) {
            return BatcherCycleResult.NO_ACTION;
        }
        return BatcherCycleResult.capacityBlocked(item, unavailable);
    }

    /** Caller holds {@link #queueLock}. */
    private boolean removeSelectionBoundaryUnderLock(
            ScheduledRequest blockedItem,
            CapacityBoundary blockedResult,
            long nowMs) {
        if (blockedItem == null
                || blockedResult.unavailable()
                || blockedResult == CapacityBoundary.OWNERSHIP_LOST) {
            return false;
        }
        if (!containsIdentity(blockedItem)
                || blockedItem.requestExpired(nowMs)
                || !removeTerminalActiveUnderLock(blockedItem)) {
            return false;
        }
        return true;
    }

    private void notifyTerminalAdmissionFailure(
            boolean removed,
            ScheduledRequest item,
            CapacityBoundary boundary) {
        if (removed && boundary.status() == CapacityBoundary.Status.FAILED) {
            notifyAdmissionFailure(item, boundary.cause());
        }
    }

    private boolean containsIdentity(ScheduledRequest expected) {
        for (ScheduledRequest item : queue) {
            if (item == expected) {
                return true;
            }
        }
        return false;
    }

    private void notifyAdmissionFailure(
            ScheduledRequest item,
            Throwable cause) {
        try {
            endpointEvents.onPreparedDeliveryFailure(item, cause);
        } catch (Throwable callbackFailure) {
            Logger.error("WorkerBatcher[{}] delivery-failure callback failed "
                            + "request_id={}",
                    key, item.requestId(), callbackFailure);
        }
    }

    private void dropHead(ScheduledRequest head) {
        boolean removed;
        queueLock.lock();
        try {
            removed = removeUnderLock(head);
        } finally {
            queueLock.unlock();
        }
        if (!removed) {
            return;
        }
        try {
            endpointEvents.onQueuedItemExpired(head);
        } catch (Throwable callbackFailure) {
            Logger.error("WorkerBatcher[{}] ACTIVE terminal callback failed "
                            + "request_id={} reason={}",
                    key, head.requestId(), "request_expired",
                    callbackFailure);
        }
    }

    private static Throwable appendFailure(
            Throwable first,
            Throwable next) {
        if (first == null) {
            return next;
        }
        if (first != next) {
            first.addSuppressed(next);
        }
        return first;
    }

    private static RuntimeException propagateCommitFailure(
            Throwable failure) {
        if (failure instanceof RuntimeException runtimeFailure) {
            return runtimeFailure;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        return new IllegalStateException(
                "delivery selection failed after ownership commit", failure);
    }

    // ==================== Group decisions ====================

    private BatcherCycleResult processQueue() {
        return singleDecision
                ? processSingleRequest()
                : processFixedWindow();
    }

    private BatcherCycleResult processSingleRequest() {
        ActiveQueueSnapshot snapshot = snapshotActiveQueueHead();
        ScheduledRequest head = snapshot.head();
        if (head == null) {
            return BatcherCycleResult.NO_ACTION;
        }

        long nowMs = now();
        if (head.requestExpired(nowMs)) {
            Logger.debug("flexlb_single_drop request_id={} "
                            + "reason=request_expired expires_at_ms={} now_ms={}",
                    head.requestId(), head.expiresAtMs(), nowMs);
            dropHead(head);
            return BatcherCycleResult.CAPACITY_CHANGED;
        }

        GroupPlanner.Shape shape =
                GroupPlanner.Shape.empty().add(head.seqLen());
        BatchCapacitySnapshot capacity = batchCapacitySnapshot();
        if (!shape.fitsKv(capacity.batchKvCapacity())) {
            return awaitPrefillKvCapacity(
                    head, snapshot.queueVersion(),
                    snapshot.schedulingInputVersion());
        }

        // Worker status and delivery ledgers advance independently. Re-read
        // the advisory gate immediately before hard-capacity admission.
        capacity = batchCapacitySnapshot();
        if (!shape.fitsKv(capacity.batchKvCapacity())) {
            return awaitPrefillKvCapacity(
                    head, snapshot.queueVersion(),
                    snapshot.schedulingInputVersion());
        }

        return runPredictionBound(head, () -> {
            PrefillTimePredictor predictor =
                    prefillEndpoint.getPredictor();
            PrefillTimePredictor.Evaluator evaluator =
                    predictor.evaluator();
            return admitAndDeliverCapacityFeasiblePrefix(
                    List.of(head), SINGLE_DECISION_REASON,
                    evaluator, OptionalLong.empty());
        });
    }

    private BatcherCycleResult processFixedWindow() {
        if (queue.isEmpty()) {
            return BatcherCycleResult.NO_ACTION;
        }

        // Cheap advisory gates avoid sorting while the worker cannot dispatch.
        ScheduledRequest observedHead;
        int observedSize;
        long observedOldestEnqueuedAtMs = Long.MAX_VALUE;
        long observedQueueVersion;
        long observedSchedulingInputVersion;
        queueLock.lock();
        try {
            observedHead = queue.peek();
            observedSize = queue.size();
            observedQueueVersion = queueVersion.get();
            observedSchedulingInputVersion = schedulingInputVersion.get();
            for (ScheduledRequest item : queue) {
                observedOldestEnqueuedAtMs = Math.min(
                        observedOldestEnqueuedAtMs, item.enqueuedAtMs());
            }
        } finally {
            queueLock.unlock();
        }
        if (observedHead == null) {
            return BatcherCycleResult.NO_ACTION;
        }

        long nowMs = now();
        long fixedWaitMs = collectionWindowMs();
        int batchMaxCount = maxDecisionRequests();
        long predictThresholdMs = predictedExecutionBudgetMs();
        BatchCapacitySnapshot capacity = batchCapacitySnapshot();
        long batchMaxTokens = capacity.batchTokenCapacity();

        if (observedHead.requestExpired(nowMs)) {
            Logger.debug("flexlb_batch_drop request_id={} "
                            + "reason=request_expired expires_at_ms={} now_ms={}",
                    observedHead.requestId(),
                    observedHead.expiresAtMs(), nowMs);
            dropHead(observedHead);
            return BatcherCycleResult.CAPACITY_CHANGED;
        }

        boolean fullCandidate =
                observedSize >= batchMaxCount;
        if (predictThresholdMs <= 0 && !fullCandidate
                && !GroupPlanner.windowElapsed(
                observedOldestEnqueuedAtMs, nowMs, fixedWaitMs)) {
            return awaitCollectionWindow(
                    observedHead,
                    observedQueueVersion,
                    observedSchedulingInputVersion,
                    observedOldestEnqueuedAtMs, fixedWaitMs);
        }

        long batchKvTokens = capacity.batchKvCapacity();
        if (!GroupPlanner.Shape.empty().add(observedHead.seqLen())
                .fitsKv(batchKvTokens)) {
            return awaitPrefillKvCapacity(
                    observedHead,
                    observedQueueVersion,
                    observedSchedulingInputVersion);
        }

        // The ordered snapshot is the selection linearization point;
        // prediction intentionally runs after releasing queueLock.
        ActiveQueueSnapshot snapshot = snapshotActiveQueue();
        ScheduledRequest head = snapshot.head();
        if (head == null) {
            return BatcherCycleResult.NO_ACTION;
        }

        nowMs = now();
        fixedWaitMs = collectionWindowMs();
        batchMaxCount = maxDecisionRequests();
        predictThresholdMs = predictedExecutionBudgetMs();
        capacity = batchCapacitySnapshot();
        batchMaxTokens = capacity.batchTokenCapacity();
        if (head.requestExpired(nowMs)) {
            Logger.debug("flexlb_batch_drop request_id={} "
                            + "reason=request_expired expires_at_ms={} now_ms={}",
                    head.requestId(), head.expiresAtMs(), nowMs);
            dropHead(head);
            return BatcherCycleResult.CAPACITY_CHANGED;
        }
        ScheduledRequest expiredMember = firstExpiredMember(
                snapshot.items(), batchMaxCount, nowMs);
        if (expiredMember != null) {
            Logger.debug("flexlb_batch_drop request_id={} "
                            + "reason=request_expired expires_at_ms={} now_ms={}",
                    expiredMember.requestId(),
                    expiredMember.expiresAtMs(), nowMs);
            dropHead(expiredMember);
            return BatcherCycleResult.CAPACITY_CHANGED;
        }
        GroupPlanner.Shape headShape =
                GroupPlanner.Shape.empty().add(head.seqLen());
        batchKvTokens = capacity.batchKvCapacity();
        if (!headShape.fitsKv(batchKvTokens)) {
            return awaitPrefillKvCapacity(
                    head, snapshot.queueVersion(),
                    snapshot.schedulingInputVersion());
        }

        long exactBatchMaxTokens = batchMaxTokens;
        long exactBatchKvTokens = batchKvTokens;
        int exactBatchMaxCount = batchMaxCount;
        long exactPredictThresholdMs = predictThresholdMs;
        long exactFixedWaitMs = fixedWaitMs;
        return runPredictionBound(head, () -> {
            PrefillTimePredictor predictor =
                    prefillEndpoint.getPredictor();
            PrefillTimePredictor.Evaluator evaluator = predictor == null
                    ? null : predictor.evaluator();
            PrefillTimePredictor.Evaluator planningEvaluator =
                    exactPredictThresholdMs > 0 ? evaluator : null;
            GroupPlanner.Constraints plannerConstraints =
                    new GroupPlanner.Constraints(
                            exactBatchMaxCount, exactBatchMaxTokens,
                            exactBatchKvTokens, exactPredictThresholdMs,
                            exactFixedWaitMs);
            GroupPlanner.Selection<ScheduledRequest> selection =
                    GroupPlanner.select(
                            snapshot.items(), PLANNER_ITEM_ACCESS,
                            plannerConstraints,
                            planningEvaluator == null ? null : items ->
                                    projectGroupDurationMs(
                                            items, planningEvaluator));
            GroupPlanner.Plan<ScheduledRequest> plan =
                    GroupPlanner.evaluateReadiness(
                            selection, plannerConstraints, now());
            if (plan.ready()) {
                return admitDecisionGroup(
                        plan.items(), plan.shape(), plan.reason(),
                        evaluator, committedPrediction(plan));
            }
            return awaitCollectionWindow(
                    head, snapshot.queueVersion(),
                    snapshot.schedulingInputVersion(),
                    plan.windowOpenedAtMs(), exactFixedWaitMs);
        });
    }

    private static BatcherCycleResult awaitCollectionWindow(
            ScheduledRequest head,
            long queueVersion,
            long schedulingInputVersion,
            long windowOpenedAtMs,
            long collectionWindowMs) {
        long collectionDeadline = GroupPlanner.collectionDeadlineMs(
                windowOpenedAtMs, collectionWindowMs);
        return BatcherCycleResult.awaitingSchedulingChange(
                head, queueVersion, schedulingInputVersion,
                Math.min(collectionDeadline, head.expiresAtMs()));
    }

    private static BatcherCycleResult awaitPrefillKvCapacity(
            ScheduledRequest head,
            long queueVersion,
            long schedulingInputVersion) {
        return BatcherCycleResult.awaitingSchedulingChange(
                head, queueVersion, schedulingInputVersion,
                head.expiresAtMs());
    }

    private static ScheduledRequest firstExpiredMember(
            List<ScheduledRequest> orderedItems,
            int maxCount,
            long nowMs) {
        int inspected = Math.min(
                Math.max(1, maxCount), orderedItems.size());
        for (int index = 0; index < inspected; index++) {
            ScheduledRequest item = orderedItems.get(index);
            if (item.requestExpired(nowMs)) {
                return item;
            }
        }
        return null;
    }

    private BatcherCycleResult admitDecisionGroup(
            List<ScheduledRequest> picked,
            GroupPlanner.Shape shape,
            String reason,
            PrefillTimePredictor.Evaluator evaluator,
            OptionalLong plannedCommittedPredictionMs) {
        BatchCapacitySnapshot capacity = batchCapacitySnapshot();
        if ((picked.size() > 1
                && !shape.fitsCompute(capacity.batchTokenCapacity()))
                || !shape.fitsKv(capacity.batchKvCapacity())) {
            return BatcherCycleResult.NO_ACTION;
        }
        return admitAndDeliverCapacityFeasiblePrefix(
                picked, reason, evaluator,
                plannedCommittedPredictionMs);
    }

    private static OptionalLong committedPrediction(
            GroupPlanner.Plan<ScheduledRequest> plan) {
        if (plan.selectedPredictionMs().isEmpty()) {
            return OptionalLong.empty();
        }
        return OptionalLong.of(
                PrefillPredictionBoundary.committedDecisionGroupMs(
                        plan.selectedPredictionMs().getAsDouble()));
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

        BatcherCycleResult result = processQueue();
        if (result.placementCapacityChanged()) {
            // Only a committed removal creates a new queue seat. Advisory
            // waits and delivery-capacity misses must not feed placement back
            // into itself.
            prefillEndpoint.signalPlacementCapacityChanged();
        }
        if (result.capacityBlocked()) {
            awaitBlockedHeadCapacity(result);
        } else if (result.awaitingSchedulingChange()) {
            awaitSchedulingChange(result);
        }
    }

    private void waitForNonEmpty() throws InterruptedException {
        queueLock.lockInterruptibly();
        try {
            while (!stopped && queue.isEmpty()) {
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
                    && queue.peek() == blocked.request()
                    && !blocked.request().requestExpired(System.currentTimeMillis())
                    && !blocked.unavailable().availability().isAvailable()) {
                long expiresAtMs = blocked.request().expiresAtMs();
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
                    && queue.peek() == waiting.request()
                    && queueVersion.get() == waiting.queueVersion()
                    && schedulingInputVersion.get()
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
            schedulingInputVersion.incrementAndGet();
            stateChanged.signal();
        } finally {
            queueLock.unlock();
        }
    }

}
