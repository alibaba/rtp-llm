package org.flexlb.balance.queue.dispatch;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.estimate.PrefillTtftEstimator;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.queue.ProcessQueueResult;
import org.flexlb.balance.queue.QueueState;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.util.Logger;

import java.util.List;
import java.util.OptionalLong;

/**
 * FIXED_WINDOW 模式：队首最多等一个凑组窗口，把排在后面的请求按数量、token、KV 容量和预测耗时上限凑成一组再交付。
 */
public final class FixedWindowPicker implements QueuePicker {

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

    private final QueueState queueState;
    private final PrefillEndpoint prefillEndpoint;
    private final FlexlbConfig config;
    private final GroupingLimits groupingLimits;
    private final PrefillTtftEstimator ttftEstimator;
    private final GroupCommitter groupCommitter;

    public FixedWindowPicker(QueueState queueState,
                      PrefillEndpoint prefillEndpoint,
                      FlexlbConfig config,
                      GroupingLimits groupingLimits,
                      PrefillTtftEstimator ttftEstimator,
                      GroupCommitter groupCommitter) {
        this.queueState = queueState;
        this.prefillEndpoint = prefillEndpoint;
        this.config = config;
        this.groupingLimits = groupingLimits;
        this.ttftEstimator = ttftEstimator;
        this.groupCommitter = groupCommitter;
    }

    @Override
    public ProcessQueueResult processQueue() {
        if (queueState.activeIndex().isEmpty()) {
            return ProcessQueueResult.NO_ACTION;
        }

        // 先做几项开销很小的检查，Prefill 当前发不出去时就不必复制并排序整个队列。
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
            return ProcessQueueResult.awaitingPrefillKvCapacity(
                    observedHead,
                    observedQueueVersion,
                    observedSchedulingInputVersion);
        }

        // 选请求以这份排好序的快照为准；耗时预测放在释放队列锁之后做，避免长时间占着锁。
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
            return ProcessQueueResult.awaitingPrefillKvCapacity(
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

    private static long now() {
        return System.currentTimeMillis();
    }
}
