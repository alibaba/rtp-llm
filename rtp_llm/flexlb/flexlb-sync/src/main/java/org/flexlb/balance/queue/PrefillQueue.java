package org.flexlb.balance.queue;

import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.endpoint.PrefillActiveIndex;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.estimate.PrefillTtftEstimate;
import org.flexlb.balance.estimate.PrefillTtftEstimator;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.queue.dispatch.FixedWindowPicker;
import org.flexlb.balance.queue.dispatch.GroupCommitter;
import org.flexlb.balance.queue.dispatch.GroupingLimits;
import org.flexlb.balance.queue.dispatch.QueuePicker;
import org.flexlb.balance.queue.dispatch.SingleRequestPicker;
import org.flexlb.balance.queue.enqueue.PrefillQueueEnqueuer;
import org.flexlb.balance.queue.loop.PrefillQueueLifecycle;
import org.flexlb.balance.queue.loop.PrefillQueueLoop;
import org.flexlb.balance.queue.loop.QueueWaitRecorder;
import org.flexlb.balance.scheduler.EndpointEventHandler;
import org.flexlb.balance.scheduler.QueueCapacity;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.enums.PrefillQueueEnqueueResult;
import org.flexlb.util.Logger;
import org.flexlb.util.PriorityNormalizer;
import org.flexlb.util.PriorityOrdering;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
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
public final class PrefillQueue {

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

    private static final Comparator<GroupPlanner.Item> PRIORITY_ESTIMATE_ORDER =
            (left, right) -> PriorityOrdering.compareWithRequestId(
                    left.priority(), left.enqueueSeq(), left.requestId(),
                    right.priority(), right.enqueueSeq(), right.requestId());
    private static final Comparator<GroupPlanner.Item> FIFO_ESTIMATE_ORDER =
            Comparator.comparingLong(GroupPlanner.Item::enqueueSeq)
                    .thenComparing(GroupPlanner.Item::requestId);
    /** Initial queue index allocation; unrelated to admission limits. */
    private static final int INITIAL_QUEUE_ALLOCATION = 16;

    private final String key;
    private final PrefillEndpoint prefillEndpoint;
    private final PrefillTtftEstimator ttftEstimator;
    private final QueueState queueState;
    private final LoadViewCache loadViewCache;
    /** Canonical request ownership guarded exclusively by the queue lock. */
    private final PrefillState prefillState;
    private final PrefillQueueLifecycle lifecycle;
    private final PrefillQueueEnqueuer enqueuer;
    private final CapacityBlockedHead capacityBlockedHead;
    private final QueueWaitRecorder queueWaitRecorder;
    private final QueuedRequestRemover queuedRequestRemover;
    private final PrefillQueueLoop loop;

    public PrefillQueue(
            String key,
            PrefillEndpoint prefillEp,
            FlexlbConfig config,
            DeliveryStrategy deliveryStrategy,
            PrefillTtftEstimator ttftEstimator,
            EndpointEventHandler endpointEvents,
            QueueCapacity queueCapacity) {
        this(key, prefillEp, () -> config, deliveryStrategy, ttftEstimator, endpointEvents, queueCapacity);
    }

    public PrefillQueue(String key,
                         PrefillEndpoint prefillEp,
                         Supplier<FlexlbConfig> configSupplier,
                         DeliveryStrategy deliveryStrategy,
                         PrefillTtftEstimator ttftEstimator,
                         EndpointEventHandler endpointEvents,
                         QueueCapacity queueCapacity) {
        FlexlbConfig config = configSupplier.get();
        this.key = key;
        this.prefillEndpoint = prefillEp;
        GroupingLimits groupingLimits = new GroupingLimits(config, configSupplier);
        boolean priorityOrdering = config.isPriorityOrdering();
        this.ttftEstimator = ttftEstimator;
        Comparator<ScheduledRequest> queueOrder = priorityOrdering
                ? PRIORITY_QUEUE_ORDER : FIFO_QUEUE_ORDER;
        Comparator<GroupPlanner.Item> estimateOrder = priorityOrdering ? PRIORITY_ESTIMATE_ORDER : FIFO_ESTIMATE_ORDER;
        this.queueState = new QueueState(PrefillActiveIndex.ordered(INITIAL_QUEUE_ALLOCATION, queueOrder));
        this.capacityBlockedHead = new CapacityBlockedHead(queueState, prefillEp);
        this.prefillState = new PrefillState(queueState.reentrantLock(),
                                             queueState.activeIndex(),
                                             capacityBlockedHead.capacityAvailableSignal());
        this.queueWaitRecorder = new QueueWaitRecorder(key, queueState, prefillState);
        this.queuedRequestRemover = new QueuedRequestRemover(queueState,
                                                             prefillState,
                                                             prefillEp,
                                                             queueCapacity);
        GroupCommitter groupCommitter = new GroupCommitter(key,
                                                 queueState,
                                                 prefillEndpoint,
                                                 queueCapacity,
                                                 endpointEvents,
                                                 deliveryStrategy,
                                                 capacityBlockedHead,
                                                 queuedRequestRemover);
        QueuePicker queuePicker = config.isSingleDecision()
                ? new SingleRequestPicker(queueState, prefillEp, config, groupCommitter)
                : new FixedWindowPicker(queueState,
                                        prefillEp,
                                        config,
                                        groupingLimits,
                                        ttftEstimator,
                                        groupCommitter);
        this.loadViewCache = new LoadViewCache(queueState,
                                               prefillState,
                                               prefillEp,
                                               config,
                                               groupingLimits,
                                               estimateOrder,
                                               capacityBlockedHead::admissionBlockUnderLock);
        this.lifecycle = new PrefillQueueLifecycle(key,
                                                   queueState,
                                                   prefillState,
                                                   prefillEp,
                                                   queueCapacity,
                                                   endpointEvents,
                                                   capacityBlockedHead,
                                                   this::runLoop);
        this.enqueuer = new PrefillQueueEnqueuer(queueState,
                                                 prefillState,
                                                 prefillEp,
                                                 queueCapacity,
                                                 endpointEvents,
                                                 config);
        this.loop = new PrefillQueueLoop(queueState,
                                         prefillEp,
                                         capacityBlockedHead,
                                         queueWaitRecorder,
                                         lifecycle,
                                         queuePicker);
    }

    public void start() {
        lifecycle.start();
    }

    /**
     * 停止调度线程，把还在排队的请求逐个交还给请求调度器，并等调度线程退出。
     * 返回清理过程中遇到的第一个异常，没有则返回 null。
     */
    public Throwable stopAndAwait() {
        return lifecycle.stopAndAwait();
    }

    public PrefillQueueEnqueueResult offerWithResult(ScheduledRequest exactItem) {
        ScheduledRequest item = exactItem;
        requireExactEndpoint(item, "incoming item");
        if (!lifecycle.isRunning() || queueState.isStopped()) {
            return PrefillQueueEnqueueResult.REJECTED;
        }
        return enqueuer.enqueue(item);
    }

    public int queueSize() {
        queueState.lock();
        try {
            return queueState.activeIndex().size();
        } finally {
            queueState.unlock();
        }
    }

    public long oldestSubQueueWaitMs() {
        queueState.lock();
        try {
            long oldest = Long.MAX_VALUE;
            for (ScheduledRequest item : queueState.activeIndex()) {
                long enteredAt = item.subQueueEnteredAtNanos();
                if (enteredAt != 0L) {
                    oldest = Math.min(oldest, enteredAt);
                }
            }
            return oldest == Long.MAX_VALUE ? 0L
                    : TimeUnit.NANOSECONDS.toMillis(Math.max(0L, System.nanoTime() - oldest));
        } finally {
            queueState.unlock();
        }
    }

    /**
     * Snapshot of the current active queue depth bucketed by normalized
     * scheduling priority. Only priorities present in the queue appear in the
     * result, matching the tagged queue-depth metric behavior.
     */
    public Map<Integer, Integer> queueSizeByPriority() {
        Map<Integer, Integer> sizeByPriority = new HashMap<>();
        queueState.lock();
        try {
            for (int priority = 0; priority <= PriorityNormalizer.MAX_PRIORITY; priority++) {
                int count = queueState.activeIndex().size(priority);
                if (count > 0) { sizeByPriority.put(priority, count); }
            }
        } finally {
            queueState.unlock();
        }
        return sizeByPriority;
    }

    public PrefillState ownedState() {
        return prefillState;
    }

    /** Immutable delivery semantics used by a pure route estimate. */
    public PrefillTtftEstimate.TtftEstimationRule ttftEstimationRule() {
        return ttftEstimator.ttftEstimationRule();
    }

    // ==================== endpoint queue operations ====================

    public boolean removeQueued(
            ScheduledRequest exactItem,
            String reason) {
        ScheduledRequest item = exactItem;
        requireExactEndpoint(item, "queued item");
        boolean removed;
        queueState.lock();
        try {
            removed = lifecycle.isRunning()
                    && !queueState.isStopped()
                    && queuedRequestRemover.removeUnderLock(item);
            if (removed) {
                queueState.signalChange();
            }
        } finally {
            queueState.unlock();
        }
        if (removed) {
            capacityBlockedHead.signalCapacityAvailable();
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

    public Map<String, Object> waitDiagnostics() {
        return queueWaitRecorder.diagnostics();
    }

    public QueuedRequestsSnapshot captureQueueSnapshot() {
        queueState.lock();
        try {
            return new QueuedRequestsSnapshot(
                    key,
                    queueState.queueVersion(),
                    activeItemsInSchedulingOrder());
        } finally {
            queueState.unlock();
        }
    }

    // ==================== Queue ownership and load view ====================

    private List<ScheduledRequest> activeItemsInSchedulingOrder() {
        return queueState.activeIndex().capture().items();
    }

    public PrefillTtftEstimate.LoadView captureLoadView() {
        return loadViewCache.capture();
    }

    // ==================== Internal: Run loop ====================

    private void runLoop() {
        loop.run();
    }

    /** Wake decisions whose advisory worker-status or predictor input changed. */
    public void signalSchedulingInputsChanged() {
        queueState.lock();
        try {
            queueState.incrementSchedulingInputVersionUnderLock();
            queueState.signalChange();
        } finally {
            queueState.unlock();
        }
    }

}
