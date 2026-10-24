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
 * 一台 Prefill 节点的调度队列：保存排队请求，按配置把请求凑组，检查投递容量后交付给 Prefill。
 * <p>
 * 每台 Prefill 节点一个实例。请求入队后按配置的模式（单请求或 FIXED_WINDOW）选出来，
 * 一组请求可以通过 EnqueueBatch 一起交付，也可以逐个按单请求选路交付。
 * <p>
 * 排队请求只保存在 PrefillState 的排队索引里。PRIORITY 模式按 {@link #PRIORITY_QUEUE_ORDER} 排序，
 * FIFO 模式按入队序号排序。队列每次变化都会推进队列版本，快照和诊断信息用它判断队列有没有变。
 */
public final class PrefillQueue {

    /**
     * PRIORITY 模式的排队顺序：按 {@link PriorityOrdering#STRICT} 比较，优先级高的在前，
     * 同优先级按入队序号先来先出，最后按 {@code requestId} 比较，保证顺序固定。
     * <p>
     * {@link #FIFO_QUEUE_ORDER} 只按入队顺序。
     */
    public static final Comparator<ScheduledRequest> PRIORITY_QUEUE_ORDER =
            (left, right) -> PriorityOrdering.compareWithRequestId(
                    left.priority(), left.enqueueSeq(), left.requestId(),
                    right.priority(), right.enqueueSeq(), right.requestId());

    /**
     * FIFO 模式的排队顺序：按入队序号，序号唯一且递增。
     */
    public static final Comparator<ScheduledRequest> FIFO_QUEUE_ORDER =
            Comparator.comparingLong(ScheduledRequest::enqueueSeq);

    private static final Comparator<GroupPlanner.Item> PRIORITY_ESTIMATE_ORDER =
            (left, right) -> PriorityOrdering.compareWithRequestId(
                    left.priority(), left.enqueueSeq(), left.requestId(),
                    right.priority(), right.enqueueSeq(), right.requestId());
    private static final Comparator<GroupPlanner.Item> FIFO_ESTIMATE_ORDER =
            Comparator.comparingLong(GroupPlanner.Item::enqueueSeq)
                    .thenComparing(GroupPlanner.Item::requestId);
    /**
     * 排队索引的初始容量，和排队上限无关。
     */
    private static final int INITIAL_QUEUE_ALLOCATION = 16;

    private final String key;
    private final PrefillEndpoint prefillEndpoint;
    private final PrefillTtftEstimator ttftEstimator;
    private final QueueState queueState;
    private final LoadViewCache loadViewCache;
    /**
     * 这台 Prefill 上的请求记录和资源预留，只在持有队列锁时读写。
     */
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
     * 按归一化后的调度优先级统计当前排队请求数。
     * 只返回队列里有请求的优先级，和按优先级打标的排队数监控保持一致。
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

    /**
     * 选路估算 TTFT 时用的投递规则，创建后不会变化。
     */
    public PrefillTtftEstimate.TtftEstimationRule ttftEstimationRule() {
        return ttftEstimator.ttftEstimationRule();
    }

    // ==================== 对外的队列操作 ====================

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
                // 请求已经移出队列，打日志失败也不能让这次移除变成失败。
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

    // ==================== 排队快照和负载视图 ====================

    private List<ScheduledRequest> activeItemsInSchedulingOrder() {
        return queueState.activeIndex().capture().items();
    }

    public PrefillTtftEstimate.LoadView captureLoadView() {
        return loadViewCache.capture();
    }

    // ==================== 调度线程 ====================

    private void runLoop() {
        loop.run();
    }

    /**
     * Prefill 状态或耗时预测器的输入变了：推进调度输入版本，唤醒在等调度输入变化的调度线程。
     */
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
