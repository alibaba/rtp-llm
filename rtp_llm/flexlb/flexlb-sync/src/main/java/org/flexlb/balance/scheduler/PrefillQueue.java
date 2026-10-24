package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.endpoint.PrefillActiveIndex;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.estimate.PrefillTtftEstimate;
import org.flexlb.balance.estimate.PrefillTtftEstimator;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.queue.BatchCapacityView;
import org.flexlb.balance.queue.GroupingLimits;
import org.flexlb.balance.queue.LoadViewCache;
import org.flexlb.balance.queue.QueueState;
import org.flexlb.balance.queue.QueuedRequestsSnapshot;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.enums.PrefillQueueEnqueueResult;
import org.flexlb.util.Logger;
import org.flexlb.util.PriorityNormalizer;
import org.flexlb.util.PriorityOrdering;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
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
    /** Initial queue index allocation; unrelated to admission limits. */
    private static final int INITIAL_QUEUE_ALLOCATION = 16;

    private final String key;
    private final PrefillEndpoint prefillEndpoint;
    private final QueueCapacity queueCapacity;
    private final FlexlbConfig config;
    private final GroupingLimits groupingLimits;
    private final boolean singleDecision;
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
    private final GroupCommitter groupCommitter;
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
        this.queueCapacity = queueCapacity;
        this.config = config;
        this.singleDecision = config.isSingleDecision();
        this.groupingLimits = new GroupingLimits(config, configSupplier);
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
        this.groupCommitter = new GroupCommitter(key,
                                                 queueState,
                                                 prefillEndpoint,
                                                 queueCapacity,
                                                 endpointEvents,
                                                 deliveryStrategy,
                                                 capacityBlockedHead,
                                                 queuedRequestRemover);
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
                                         this::processQueue);
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

    private static long now() {
        return System.currentTimeMillis();
    }

    private List<ScheduledRequest> activeItemsInSchedulingOrder() {
        return queueState.activeIndex().capture().items();
    }

    private ActiveQueueSnapshot snapshotActiveQueueHead() {
        queueState.lock();
        try {
            ScheduledRequest head = queueState.activeIndex().peek();
            return new ActiveQueueSnapshot(
                    queueState.queueVersion(), queueState.schedulingInputVersion(),
                    head == null ? List.of() : List.of(head));
        } finally {
            queueState.unlock();
        }
    }

    private ActiveQueueSnapshot snapshotActiveQueue() {
        long version;
        long inputVersion;
        List<ScheduledRequest> items;
        queueState.lock();
        try {
            version = queueState.queueVersion();
            inputVersion = queueState.schedulingInputVersion();
            if (queueState.activeIndex().isEmpty()) {
                return new ActiveQueueSnapshot(
                        version, inputVersion, List.of());
            }
            items = queueState.activeIndex().capture().items();
        } finally {
            queueState.unlock();
        }
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

    public PrefillTtftEstimate.LoadView captureLoadView() {
        return loadViewCache.capture();
    }

    // ==================== Group decisions ====================

    private ProcessQueueResult processQueue() {
        return singleDecision
                ? processSingleRequest()
                : processFixedWindow();
    }

    private ProcessQueueResult processSingleRequest() {
        ActiveQueueSnapshot snapshot = snapshotActiveQueueHead();
        ScheduledRequest head = snapshot.head();
        if (head == null) {
            return ProcessQueueResult.NO_ACTION;
        }

        long nowMs = now();
        if (head.requestExpired(nowMs)) {
            Logger.debug("flexlb_single_drop request_id={} "
                            + "reason=request_expired expires_at_ms={} now_ms={}",
                    head.requestId(), head.expiresAtMs(), nowMs);
            groupCommitter.dropHead(head);
            return ProcessQueueResult.CAPACITY_CHANGED;
        }

        GroupPlanner.Shape shape =
                GroupPlanner.Shape.empty().add(head.seqLen());
        BatchCapacityView capacity = BatchCapacityView.of(prefillEndpoint, config);
        if (!shape.fitsKv(capacity.kvCapacityFor(shape))) {
            return awaitPrefillKvCapacity(
                    head, snapshot.queueVersion(),
                    snapshot.schedulingInputVersion());
        }

        // Worker status and delivery ledgers advance independently. Re-read
        // the advisory gate immediately before hard-capacity admission.
        capacity = BatchCapacityView.of(prefillEndpoint, config);
        if (!shape.fitsKv(capacity.kvCapacityFor(shape))) {
            return awaitPrefillKvCapacity(
                    head, snapshot.queueVersion(),
                    snapshot.schedulingInputVersion());
        }

        return groupCommitter.runPredictionBound(head, () -> {
            PrefillTimePredictor predictor =
                    prefillEndpoint.getPredictor();
            PrefillTimePredictor.Evaluator evaluator =
                    predictor.evaluator();
            return groupCommitter.admitAndDeliverCapacityFeasiblePrefix(
                    List.of(head), SINGLE_DECISION_REASON,
                    evaluator, OptionalLong.empty());
        });
    }

    private ProcessQueueResult processFixedWindow() {
        if (queueState.activeIndex().isEmpty()) {
            return ProcessQueueResult.NO_ACTION;
        }

        // Cheap advisory gates avoid sorting while the worker cannot dispatch.
        ScheduledRequest observedHead;
        int observedSize;
        long observedOldestEnqueuedAtMs = Long.MAX_VALUE;
        long observedQueueVersion;
        long observedSchedulingInputVersion;
        queueState.lock();
        try {
            observedHead = queueState.activeIndex().peek();
            observedSize = queueState.activeIndex().size();
            observedQueueVersion = queueState.queueVersion();
            observedSchedulingInputVersion = queueState.schedulingInputVersion();
            for (ScheduledRequest item : queueState.activeIndex()) {
                observedOldestEnqueuedAtMs = Math.min(
                        observedOldestEnqueuedAtMs, item.enqueuedAtMs());
            }
        } finally {
            queueState.unlock();
        }
        if (observedHead == null) {
            return ProcessQueueResult.NO_ACTION;
        }

        long nowMs = now();
        long fixedWaitMs = groupingLimits.collectionWindowMs();
        BatchCapacityView capacity = BatchCapacityView.of(prefillEndpoint, config);
        int batchMaxCount = groupingLimits.maxRequests();
        long predictThresholdMs = groupingLimits.predictedExecutionBudgetMs();
        long batchMaxTokens = capacity.batchTokenCapacity();

        if (observedHead.requestExpired(nowMs)) {
            Logger.debug("flexlb_batch_drop request_id={} "
                            + "reason=request_expired expires_at_ms={} now_ms={}",
                    observedHead.requestId(),
                    observedHead.expiresAtMs(), nowMs);
            groupCommitter.dropHead(observedHead);
            return ProcessQueueResult.CAPACITY_CHANGED;
        }

        boolean fullCandidate = observedSize >= batchMaxCount;
        if (predictThresholdMs <= 0 && !fullCandidate
                && !GroupPlanner.windowElapsed(
                observedOldestEnqueuedAtMs, nowMs, fixedWaitMs)) {
            return awaitCollectionWindow(
                    observedHead,
                    observedQueueVersion,
                    observedSchedulingInputVersion,
                    observedOldestEnqueuedAtMs, fixedWaitMs);
        }

        GroupPlanner.Shape observedHeadShape = GroupPlanner.Shape.empty().add(observedHead.seqLen());
        long batchKvTokens = capacity.kvCapacityFor(observedHeadShape);
        if (!observedHeadShape.fitsKv(batchKvTokens)) {
            return awaitPrefillKvCapacity(
                    observedHead,
                    observedQueueVersion,
                    observedSchedulingInputVersion);
        }

        // The ordered snapshot is the selection linearization point;
        // prediction intentionally runs after releasing the queue lock.
        ActiveQueueSnapshot snapshot = snapshotActiveQueue();
        ScheduledRequest head = snapshot.head();
        if (head == null) {
            return ProcessQueueResult.NO_ACTION;
        }

        nowMs = now();
        capacity = BatchCapacityView.of(prefillEndpoint, config);
        batchMaxCount = groupingLimits.maxRequests();
        predictThresholdMs = groupingLimits.predictedExecutionBudgetMs();
        batchMaxTokens = capacity.batchTokenCapacity();
        if (head.requestExpired(nowMs)) {
            Logger.debug("flexlb_batch_drop request_id={} "
                            + "reason=request_expired expires_at_ms={} now_ms={}",
                    head.requestId(), head.expiresAtMs(), nowMs);
            groupCommitter.dropHead(head);
            return ProcessQueueResult.CAPACITY_CHANGED;
        }
        ScheduledRequest expiredMember = firstExpiredMember(
                snapshot.items(), batchMaxCount, nowMs);
        if (expiredMember != null) {
            Logger.debug("flexlb_batch_drop request_id={} "
                            + "reason=request_expired expires_at_ms={} now_ms={}",
                    expiredMember.requestId(),
                    expiredMember.expiresAtMs(), nowMs);
            groupCommitter.dropHead(expiredMember);
            return ProcessQueueResult.CAPACITY_CHANGED;
        }
        GroupPlanner.Shape headShape =
                GroupPlanner.Shape.empty().add(head.seqLen());
        batchKvTokens = capacity.kvCapacityFor(headShape);
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
        return groupCommitter.runPredictionBound(head, () -> {
            PrefillTimePredictor.Evaluator evaluator = prefillEndpoint.getPredictor().evaluator();
            PrefillTimePredictor.Evaluator planningEvaluator =
                    exactPredictThresholdMs > 0 ? evaluator : null;
            GroupPlanner.Constraints plannerConstraints =
                    new GroupPlanner.Constraints(
                            exactBatchMaxCount, exactBatchMaxTokens,
                            exactBatchKvTokens, exactPredictThresholdMs,
                            exactFixedWaitMs);
            GroupPlanner.Selection<ScheduledRequest> selection =
                    GroupPlanner.selectWithPrediction(
                            snapshot.items(), PLANNER_ITEM_ACCESS,
                            plannerConstraints,
                            planningEvaluator == null ? null :
                                    ttftEstimator.newGroupPredictor(planningEvaluator));
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

    private static ProcessQueueResult awaitCollectionWindow(
            ScheduledRequest head,
            long queueVersion,
            long schedulingInputVersion,
            long windowOpenedAtMs,
            long collectionWindowMs) {
        long collectionDeadline = GroupPlanner.collectionDeadlineMs(
                windowOpenedAtMs, collectionWindowMs);
        return ProcessQueueResult.awaitingSchedulingChange(
                head, queueVersion, schedulingInputVersion,
                Math.min(collectionDeadline, head.expiresAtMs()), "Prefill collection window");
    }

    private static ProcessQueueResult awaitPrefillKvCapacity(
            ScheduledRequest head,
            long queueVersion,
            long schedulingInputVersion) {
        return ProcessQueueResult.awaitingSchedulingChange(
                head, queueVersion, schedulingInputVersion,
                head.expiresAtMs(), "Prefill KV capacity exhausted");
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

    private ProcessQueueResult admitDecisionGroup(List<ScheduledRequest> picked,
                                                  GroupPlanner.Shape shape,
                                                  String reason,
                                                  PrefillTimePredictor.Evaluator evaluator,
                                                  OptionalLong plannedCommittedPredictionMs) {
        BatchCapacityView capacity = BatchCapacityView.of(prefillEndpoint, config);
        if ((picked.size() > 1 && !shape.fitsCompute(capacity.batchTokenCapacity()))
                || !shape.fitsKv(capacity.kvCapacityFor(shape))) {
            return ProcessQueueResult.NO_ACTION;
        }
        return groupCommitter.admitAndDeliverCapacityFeasiblePrefix(
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
