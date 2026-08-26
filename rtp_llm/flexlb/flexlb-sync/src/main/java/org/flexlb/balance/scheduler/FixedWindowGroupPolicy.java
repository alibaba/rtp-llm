package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryMetadata;
import org.flexlb.balance.planner.GroupPlanner;

import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.util.Logger;

import java.util.List;
import java.util.Objects;
import java.util.OptionalLong;

/**
 * Fixed-window decision policy. It proposes the largest ordered homogeneous
 * group allowed by the live shape bounds. Capacity admission then makes the
 * largest reservable prefix the final decision.
 *
 * <h3>Algorithm</h3>
 * <ol>
 *   <li>Request expiration: drop the head at its absolute expiration. This runs
 *       before backpressure to ensure stale requests are cleared even when
 *       the engine is under sustained backpressure.</li>
 *   <li>Oversized request rejection: if the head request's seqLen exceeds
 *       the worker token capacity, it can never be picked by any group,
 *       so it is dropped immediately instead of waiting for the deadline.</li>
 *   <li>Build the largest homogeneous prefix that stays inside every bound on
 *       group growth: the decision policy's largest group, the worker
 *       compute and KV shapes, and — when configured — a predicted execution
 *       time within the active prediction boundary. The head is a mandatory
 *       member, so a single request already at that boundary forms the whole
 *       group.</li>
 *   <li>Propose the group as soon as it reaches the predicted execution budget,
 *       growth would exceed that budget, or the group reaches its maximum
 *       request count. Compute and KV pressure are transient, so they stop
 *       growth without dispatching.</li>
 *   <li>Otherwise propose it once the collection window has run out, so an
 *       incomplete group never waits indefinitely for requests that have not
 *       arrived. The window covers the picked group, whose members are the ones
 *       paying for the wait; queue age itself is bounded by the absolute
 *       request expiration.</li>
 *   <li>Reserve hard delivery capacity in order. A non-empty reservable prefix
 *       becomes the final decision; its suffix remains active. If the head
 *       cannot reserve, wait for that exact resource-change event.</li>
 *   <li>Otherwise wait for a queue/status generation change or the exact
 *       collection/expiration deadline.</li>
 * </ol>
 *
 * <h3>Resource-shape filtering</h3>
 * When picking items, requests whose padded compute shape would exceed the
 * worker/fallback token capacity, or whose combined sequence length would
 * exceed the latest worker-reported KV budget, remain in the queue for a later
 * decision.
 * <p>
 * However, a request whose own seqLen already exceeds
 * that capacity can never be picked at all. Such
 * oversized requests are rejected immediately when they reach the head of
 * the queue (see step 0.5 below), rather than waiting for the queue
 * request expiration.
 *
 */
final class FixedWindowGroupPolicy implements GroupPolicy {

    private static final GroupPlanner.ItemAccess<BatchItem> PLANNER_ITEM_ACCESS =
            new GroupPlanner.ItemAccess<>() {
                @Override
                public long enqueuedAtMs(BatchItem item) {
                    return item.enqueuedAtMs();
                }

                @Override
                public long seqLen(BatchItem item) {
                    return item.seqLen();
                }
            };

    @Override
    public BatcherCycleResult processQueue(BatcherContext ctx) {
        if (ctx.isActiveEmpty()) {
            return BatcherCycleResult.Outcome.NO_ACTION;
        }

        // Cheap pre-gates avoid copying/sorting the active queue while the
        // worker cannot dispatch. They are advisory only: the authoritative
        // gates below run again against one stable ordered snapshot.
        BatcherContext.ActiveQueueState observedState = ctx.activeQueueState();
        BatchItem observedHead = observedState.head();
        if (observedHead == null) {
            return BatcherCycleResult.Outcome.NO_ACTION;
        }

        long nowMs = ctx.now();
        long fixedWaitMs = ctx.collectionWindowMs();
        int batchMaxCount = ctx.maxDecisionRequests();
        long predictThresholdMs = ctx.predictedExecutionBudgetMs();
        BatcherContext.BatchCapacitySnapshot capacity =
                ctx.batchCapacitySnapshot();
        long batchMaxTokens = capacity.batchTokenCapacity();

        // The caller's absolute request expiry is the only queue-age limit.
        // It is created once at admission and is never reset by retries.
        if (observedHead.requestExpired(nowMs)) {
            Logger.debug("flexlb_batch_drop request_id={} reason=request_expired "
                            + "expires_at_ms={} now_ms={}",
                    observedHead.requestId(), observedHead.expiresAtMs(), nowMs);
            ctx.dropHead(observedHead);
            return BatcherCycleResult.Outcome.QUEUE_CHANGED;
        }

        // The Engine admits a group only when padded context tokens are strictly
        // below max_batch_tokens_size. Reject an impossible head explicitly so
        // it cannot block the FIFO queue or cause an entire group to fast-fail.
        if (!GroupPlanner.Shape.empty().add(observedHead.seqLen())
                .fitsCompute(batchMaxTokens)) {
            ctx.rejectForBatchTokenCapacity(observedHead, batchMaxTokens);
            return BatcherCycleResult.Outcome.QUEUE_CHANGED;
        }

        boolean fullCandidate = observedState.activeSize() >= batchMaxCount;
        if (predictThresholdMs <= 0 && !fullCandidate
                && !GroupPlanner.windowElapsed(
                observedState.oldestEnqueuedAtMs(), nowMs, fixedWaitMs)) {
            // Preserve the disabled-threshold fast path: do not sort or shape
            // a partial queue until the count/window policy can fire.
            return awaitCollectionWindow(
                    ctx, observedHead,
                    observedState.queueVersion(),
                    observedState.schedulingInputVersion(),
                    observedState.oldestEnqueuedAtMs(), fixedWaitMs);
        }

        long batchKvTokens = capacity.batchKvCapacity();
        if (!GroupPlanner.Shape.empty().add(observedHead.seqLen())
                .fitsKv(batchKvTokens)) {
            // Dynamic KV pressure is a wait condition, not a rejection. Gate
            // it before prediction so an infeasible singleton cannot turn the
            // worker loop into a tight predict/release-fail spin.
            return awaitPrefillKvCapacity(
                    ctx, observedHead,
                    observedState.queueVersion(),
                    observedState.schedulingInputVersion());
        }

        // Capture ordered identities under queueLock. This is the selection
        // linearization point: a later insertion belongs to the next decision,
        // while removal of a member that would enter the admitted prefix makes
        // capacity admission reject that prefix. Prediction runs unlocked.
        BatcherContext.ActiveQueueSnapshot snapshot = ctx.snapshotActiveQueue();
        BatchItem head = snapshot.head();
        if (head == null) {
            return BatcherCycleResult.Outcome.NO_ACTION;
        }

        // Live config/resources may have changed since the advisory pre-gate.
        // Re-evaluate every hard gate against the stable snapshot head before
        // prediction or release.
        nowMs = ctx.now();
        fixedWaitMs = ctx.collectionWindowMs();
        batchMaxCount = ctx.maxDecisionRequests();
        predictThresholdMs = ctx.predictedExecutionBudgetMs();
        capacity = ctx.batchCapacitySnapshot();
        batchMaxTokens = capacity.batchTokenCapacity();
        if (head.requestExpired(nowMs)) {
            Logger.debug("flexlb_batch_drop request_id={} reason=request_expired "
                            + "expires_at_ms={} now_ms={}",
                    head.requestId(), head.expiresAtMs(), nowMs);
            ctx.dropHead(head);
            return BatcherCycleResult.Outcome.QUEUE_CHANGED;
        }
        BatchItem expiredMember = firstExpiredMember(
                snapshot.items(), batchMaxCount, nowMs);
        if (expiredMember != null) {
            Logger.debug("flexlb_batch_drop request_id={} reason=request_expired "
                            + "expires_at_ms={} now_ms={}",
                    expiredMember.requestId(),
                    expiredMember.expiresAtMs(), nowMs);
            ctx.dropHead(expiredMember);
            return BatcherCycleResult.Outcome.QUEUE_CHANGED;
        }
        GroupPlanner.Shape headShape =
                GroupPlanner.Shape.empty().add(head.seqLen());
        if (!headShape.fitsCompute(batchMaxTokens)) {
            ctx.rejectForBatchTokenCapacity(head, batchMaxTokens);
            return BatcherCycleResult.Outcome.QUEUE_CHANGED;
        }

        batchKvTokens = capacity.batchKvCapacity();
        if (!headShape.fitsKv(batchKvTokens)) {
            return awaitPrefillKvCapacity(
                    ctx, head,
                    snapshot.queueVersion(),
                    snapshot.schedulingInputVersion());
        }
        long exactBatchMaxTokens = batchMaxTokens;
        long exactBatchKvTokens = batchKvTokens;
        int exactBatchMaxCount = batchMaxCount;
        long exactPredictThresholdMs = predictThresholdMs;
        long exactFixedWaitMs = fixedWaitMs;
        return ctx.runPredictionBound(head, () -> {
            PrefillTimePredictor predictor = ctx.prefillEp().getPredictor();
            PrefillTimePredictor.Evaluator evaluator = predictor == null
                    ? null
                    : Objects.requireNonNull(
                            predictor.evaluator(), "predictor evaluator");
            PrefillTimePredictor.Evaluator planningEvaluator =
                    exactPredictThresholdMs > 0 ? evaluator : null;

            // Grow the group entirely from the stable snapshot, keeping it
            // inside every count, resource and predicted-time bound. Never
            // seed from one queue state and fill from another.
            GroupPlanner.Constraints plannerConstraints =
                    new GroupPlanner.Constraints(
                            exactBatchMaxCount, exactBatchMaxTokens,
                            exactBatchKvTokens, exactPredictThresholdMs,
                            exactFixedWaitMs);
            GroupPlanner.Selection<BatchItem> selection =
                    GroupPlanner.select(
                            snapshot.items(), PLANNER_ITEM_ACCESS,
                            plannerConstraints,
                            planningEvaluator == null ? null : items ->
                                    ctx.projectGroupDurationMs(
                                            items, planningEvaluator));
            // Prediction is deliberately outside queueLock and may be slower
            // than the remaining collection window. Use a fresh clock read so
            // a window that elapsed during prediction releases in this pass.
            long predictionCompletedAtMs = ctx.now();

            GroupPlanner.Plan<BatchItem> plan =
                    GroupPlanner.evaluateReadiness(
                            selection, plannerConstraints,
                            predictionCompletedAtMs);
            if (plan.ready()) {
                return admitDecisionGroup(
                        ctx, plan.items(), plan.shape(), plan.reason(),
                        evaluator, committedPrediction(plan));
            }

            return awaitCollectionWindow(
                    ctx, head,
                    snapshot.queueVersion(),
                    snapshot.schedulingInputVersion(),
                    plan.windowOpenedAtMs(), exactFixedWaitMs);
        });
    }

    private static BatcherCycleResult awaitCollectionWindow(
            BatcherContext ctx,
            BatchItem head,
            long queueVersion,
            long schedulingInputVersion,
            long windowOpenedAtMs,
            long collectionWindowMs) {
        long collectionDeadline = GroupPlanner.collectionDeadlineMs(
                windowOpenedAtMs, collectionWindowMs);
        long wakeAtMs = Math.min(
                collectionDeadline, head.expiresAtMs());
        return ctx.awaitingSchedulingChange(
                head, queueVersion, schedulingInputVersion, wakeAtMs,
                BatcherCycleResult.SchedulingWaitReason.COLLECTION_WINDOW);
    }

    private static BatcherCycleResult awaitPrefillKvCapacity(
            BatcherContext ctx,
            BatchItem head,
            long queueVersion,
            long schedulingInputVersion) {
        return ctx.awaitingSchedulingChange(
                head, queueVersion, schedulingInputVersion,
                head.expiresAtMs(),
                BatcherCycleResult.SchedulingWaitReason.PREFILL_KV_CAPACITY);
    }

    // ==================== Internal helpers ====================

    private static BatchItem firstExpiredMember(List<BatchItem> orderedItems,
                                                int maxCount,
                                                long nowMs) {
        int inspected = Math.min(Math.max(1, maxCount), orderedItems.size());
        for (int index = 0; index < inspected; index++) {
            BatchItem item = orderedItems.get(index);
            if (item.requestExpired(nowMs)) {
                return item;
            }
        }
        return null;
    }

    private static BatcherCycleResult admitDecisionGroup(
            BatcherContext ctx,
            List<BatchItem> picked,
            GroupPlanner.Shape shape,
            String reason,
            PrefillTimePredictor.Evaluator evaluator,
            OptionalLong plannedCommittedPredictionMs) {
        // WorkerStatus resource counters can change while prediction is
        // running. Re-check the exact shape immediately before capacity
        // admission; a later tick rebuilds the candidates against new values.
        BatcherContext.BatchCapacitySnapshot capacity =
                ctx.batchCapacitySnapshot();
        if (!shape.fitsCompute(capacity.batchTokenCapacity())
                || !shape.fitsKv(capacity.batchKvCapacity())) {
            return BatcherCycleResult.Outcome.NO_ACTION;
        }
        int queueBefore = ctx.size();
        // Prediction runs outside queueLock. An exact selected identity may be
        // removed before the delivery strategy performs its authoritative
        // ownership check, so this advisory depth must remain representable
        // even for an already-invalidated selection. A committed selection is
        // revalidated under the queue lock, where its cycle result receives
        // the canonical post-removal depth.
        int queueDepth = Math.max(0, queueBefore - picked.size());
        DeliveryMetadata metadata =
                new DeliveryMetadata(reason, queueDepth);
        return ctx.admitAndDeliverCapacityFeasiblePrefix(
                picked, metadata, evaluator,
                plannedCommittedPredictionMs);
    }

    private static OptionalLong committedPrediction(
            GroupPlanner.Plan<BatchItem> plan) {
        if (plan.selectedPredictionMs().isEmpty()) {
            return OptionalLong.empty();
        }
        return OptionalLong.of(PrefillPredictionBoundary.committedDecisionGroupMs(
                plan.selectedPredictionMs().getAsDouble()));
    }

}
