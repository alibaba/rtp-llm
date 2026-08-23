package org.flexlb.balance.scheduler;

import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Fixed-window decision policy. It grows the largest group the live bounds
 * allow and releases it to the independently configured delivery policy.
 *
 * <h3>Algorithm</h3>
 * <ol>
 *   <li>Request expiration: drop the head at its absolute expiration. This runs
 *       before backpressure to ensure stale requests are cleared even when
 *       the engine is under sustained backpressure.</li>
 *   <li>Oversized request rejection: if the head request's seqLen exceeds
 *       the worker token capacity, it can never be picked by any group,
 *       so it is dropped immediately instead of waiting for the deadline.</li>
 *   <li>Engine backpressure: if inflight batches ≥ max, park briefly.</li>
 *   <li>Build the largest homogeneous prefix that stays inside every bound on
 *       group growth: the decision policy's largest group, the worker
 *       compute and KV shapes, and — when configured — a predicted execution
     *       time below the active prediction boundary. The head is a mandatory
     *       member, so a single request already at that boundary forms the whole
     *       group.</li>
 *   <li>Dispatch as soon as growth stopped at a bound that cannot relax: the
 *       predicted execution budget or the largest decision group. Compute
 *       and KV pressure are transient, so they stop growth without dispatching.</li>
 *   <li>Otherwise dispatch once the collection window has run out, so an
 *       incomplete group never waits indefinitely for requests that have not
 *       arrived. The window covers the picked group, whose members are the ones
 *       paying for the wait; queue age itself is bounded by the absolute
 *       request expiration.</li>
 *   <li>Otherwise park briefly and retry.</li>
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
public class FixedWindowBatcherAlgorithm implements BatcherAlgorithm {

    @Override
    public long queueWaitMs(BatcherContext ctx) {
        return ctx.estimateFifoWaitMs();
    }

    @Override
    public void processQueue(BatcherContext ctx) throws InterruptedException {
        if (ctx.isActiveEmpty()) {
            return;
        }

        // Cheap pre-gates avoid copying/sorting the active queue while the
        // worker cannot dispatch. They are advisory only: the authoritative
        // gates below run again against one stable ordered snapshot.
        BatchItem observedHead = ctx.peek();
        if (observedHead == null) {
            return;
        }

        long nowMs = ctx.now();
        long fixedWaitMs = ctx.collectionWindowMs();
        int batchMaxCount = ctx.maxDecisionRequests();
        long predictThresholdMs = ctx.predictedExecutionBudgetMs();
        boolean strictPredictionLimit = ctx.usesStrictPredictedExecutionLimit();
        long batchMaxTokens = ctx.batchTokenCapacity();

        // The caller's absolute request expiry is the only queue-age limit.
        // It is created once at admission and is never reset by retries.
        if (observedHead.ctx().requestExpired(nowMs)) {
            Logger.debug("flexlb_batch_drop request_id={} reason=request_expired "
                            + "expires_at_ms={} now_ms={}",
                    observedHead.requestId(), observedHead.ctx().getRequestExpiresAtMs(), nowMs);
            ctx.dropHead(observedHead);
            return;
        }

        // The Engine admits a group only when padded context tokens are strictly
        // below max_batch_tokens_size. Reject an impossible head explicitly so
        // it cannot block the FIFO queue or cause an entire group to fast-fail.
        if (!BatchShape.empty().add(observedHead).fitsCompute(batchMaxTokens)) {
            ctx.rejectForBatchTokenCapacity(observedHead, batchMaxTokens);
            return;
        }

        // Engine backpressure: park if the prefill worker already has too many
        // batches inflight, to prevent overloading the engine.
        if (deliveryCapacityBlocked(ctx, observedHead)) {
            TimeUnit.MILLISECONDS.sleep(1);
            return;
        }

        boolean fullCandidate = ctx.activeSize() >= batchMaxCount;
        if (predictThresholdMs <= 0 && !fullCandidate
                && !windowElapsed(ctx.oldestActiveEnqueuedAtMs(), nowMs, fixedWaitMs)) {
            // Preserve the disabled-threshold fast path: do not sort or shape
            // a partial queue until the legacy full/timeout policy can fire.
            TimeUnit.MILLISECONDS.sleep(1);
            return;
        }

        long batchKvTokens = ctx.batchKvCapacity();
        if (!BatchShape.empty().add(observedHead).fitsKv(batchKvTokens)) {
            // Dynamic KV pressure is a wait condition, not a rejection. Gate
            // it before prediction so an infeasible singleton cannot turn the
            // worker loop into a tight predict/release-fail spin.
            TimeUnit.MILLISECONDS.sleep(1);
            return;
        }

        // Capture version + ordered identities together under queueLock. This
        // is the decision's linearization point: a later offer belongs to the
        // next decision, while removal of any picked member makes atomic stage
        // reject the whole group. Prediction runs without holding queueLock.
        BatcherContext.ActiveQueueSnapshot snapshot = ctx.snapshotActiveQueue(batchMaxCount);
        BatchItem head = snapshot.head();
        if (head == null) {
            return;
        }

        // Live config/resources may have changed since the advisory pre-gate.
        // Re-evaluate every hard gate against the stable snapshot head before
        // prediction or release.
        nowMs = ctx.now();
        fixedWaitMs = ctx.collectionWindowMs();
        batchMaxCount = ctx.maxDecisionRequests();
        predictThresholdMs = ctx.predictedExecutionBudgetMs();
        strictPredictionLimit = ctx.usesStrictPredictedExecutionLimit();
        batchMaxTokens = ctx.batchTokenCapacity();
        if (head.ctx().requestExpired(nowMs)) {
            Logger.debug("flexlb_batch_drop request_id={} reason=request_expired "
                            + "expires_at_ms={} now_ms={}",
                    head.requestId(), head.ctx().getRequestExpiresAtMs(), nowMs);
            ctx.dropHead(head);
            return;
        }
        BatchItem expiredMember = firstExpiredMember(
                snapshot.items(), batchMaxCount, nowMs);
        if (expiredMember != null) {
            Logger.debug("flexlb_batch_drop request_id={} reason=request_expired "
                            + "expires_at_ms={} now_ms={}",
                    expiredMember.requestId(),
                    expiredMember.ctx().getRequestExpiresAtMs(), nowMs);
            ctx.dropHead(expiredMember);
            return;
        }
        BatchShape headShape = BatchShape.empty().add(head);
        if (!headShape.fitsCompute(batchMaxTokens)) {
            ctx.rejectForBatchTokenCapacity(head, batchMaxTokens);
            return;
        }
        if (deliveryCapacityBlocked(ctx, head)) {
            TimeUnit.MILLISECONDS.sleep(1);
            return;
        }

        PrefillTimePredictor predictor = predictThresholdMs > 0
                ? ctx.prefillEp().getPredictor() : null;
        batchKvTokens = ctx.batchKvCapacity();
        if (!headShape.fitsKv(batchKvTokens)) {
            TimeUnit.MILLISECONDS.sleep(1);
            return;
        }
        long predictorGeneration = predictor == null ? 0L : predictor.generation();

        // Grow the group entirely from the stable snapshot, keeping it inside
        // every count, resource and predicted-time bound. Never seed from one
        // queue state and fill from another.
        FixedPick pick = pickWithinCapacity(
                snapshot, batchMaxCount, batchMaxTokens, batchKvTokens,
                predictor, predictThresholdMs, strictPredictionLimit);
        List<BatchItem> candidates = pick.items();
        if (predictor != null && predictor.generation() != predictorGeneration) {
            // Do not publish a decision made against a model revision that was
            // replaced while the formula was being evaluated. The next worker
            // tick retries against the new immutable parameters.
            return;
        }
        // Prediction is deliberately outside queueLock and may be slower than
        // the remaining collection window. Use a fresh clock read so a window
        // that elapsed during prediction releases in this pass instead of
        // paying for a second full prediction cycle.
        nowMs = ctx.now();

        // Dispatch once growth stopped at a bound that will not relax on its
        // own — the predicted execution boundary, or the feasible group itself
        // reaching batchMaxCount — or once the collection window has run
        // out and there is nothing left to wait for. A live configuration
        // transition may leave the other delivery mode behind the head; it must
        // neither join this group nor make it appear full. Transient compute/KV
        // pressure can also limit growth, and until the window elapses the
        // queue-order behavior is to wait rather than bypass members.
        String reason = null;
        if (pick.predictedTimeCapped()) {
            reason = "predicted_execution_cap";
        } else if (candidates.size() >= batchMaxCount) {
            reason = "batch_full";
        } else if (windowElapsed(pick.windowOpenedAtMs(), nowMs, fixedWaitMs)) {
            reason = "fixed_window_timeout";
        }
        if (reason != null) {
            if (!releaseDecisionGroup(ctx, candidates, pick.shape(), reason,
                    predictor, predictorGeneration, pick.queueVersion())) {
                TimeUnit.MILLISECONDS.sleep(1);
            }
            return;
        }

        TimeUnit.MILLISECONDS.sleep(1);
    }

    /**
     * Whether the collection window opened by {@code windowOpenedAtMs} has run
     * out. The window covers a group rather than any single member, so it is
     * anchored on that group's longest-waiting member. Under PRIORITY ordering a
     * later arrival sorts ahead of that member but joins the same group, so it
     * cannot reopen the window.
     *
     * <p>{@link Long#MAX_VALUE} means there is no member to wait for.
     */
    private static boolean windowElapsed(long windowOpenedAtMs,
                                         long nowMs,
                                         long fixedWaitMs) {
        return windowOpenedAtMs != Long.MAX_VALUE
                && nowMs >= windowOpenedAtMs
                && nowMs - windowOpenedAtMs >= fixedWaitMs;
    }

    private static boolean deliveryCapacityBlocked(BatcherContext ctx,
                                                    BatchItem head) {
        if (head.deliveryMode() == DeliveryMode.ROUTE_DECISION) {
            return ctx.availableDeliverySlots() == 0;
        }
        int maxInflightBatches = ctx.maxInflightBatches();
        return maxInflightBatches > 0
                && ctx.prefillEp().getInflightBatchCount()
                >= maxInflightBatches;
    }

    // ==================== Internal helpers ====================

    /**
     * Greedily pick up to {@code maxCount} items in the current queue order
     * while keeping the batch inside the Engine's compute and KV resource shape
     * and, when a predictor is supplied, below {@code predictThresholdMs} of
     * predicted execution time.
     *
     * <p>The queue head remains the mandatory first member while shaping, so a
     * request whose own predicted execution already reaches the active boundary
     * forms the whole batch. Temporary KV pressure prevents adding more members. The
     * complete picked shape is revalidated immediately before staging, so even a
     * singleton waits rather than being published against a newly insufficient
     * budget.
     *
     * <p>Each candidate is measured against the group it would join, which makes
     * the bound well defined for a predictor that is not monotone in batch size.
     */
    private static FixedPick pickWithinCapacity(
            BatcherContext.ActiveQueueSnapshot snapshot,
            int maxCount,
            long batchMaxTokens,
            long batchKvTokens,
            PrefillTimePredictor predictor,
            long predictThresholdMs,
            boolean strictPredictionLimit) {
        List<BatchItem> orderedItems = snapshot.items();
        BatchItem head = orderedItems.get(0);
        List<BatchItem> picked = new ArrayList<>(
                Math.min(maxCount, orderedItems.size()));
        picked.add(head);
        BatchShape shape = BatchShape.empty().add(head);
        long windowOpenedAtMs = head.enqueuedAtMs();
        int modePrefixSize = 1;
        boolean capacityOpen = true;
        boolean predictedTimeCapped = predictor != null
                && predictionBoundaryReached(
                predictor.predictBatchMs(picked), predictThresholdMs,
                strictPredictionLimit);
        for (int index = 1; index < orderedItems.size() && !predictedTimeCapped; index++) {
            BatchItem item = orderedItems.get(index);
            if (item.deliveryMode() != head.deliveryMode()) {
                break;
            }
            if (modePrefixSize >= maxCount) {
                break;
            }
            modePrefixSize++;
            if (!capacityOpen) {
                continue;
            }
            BatchShape candidate = shape.add(item);
            if (!candidate.fitsCompute(batchMaxTokens)) {
                capacityOpen = false;
                continue;
            }
            if (!candidate.fitsKv(batchKvTokens)) {
                capacityOpen = false;
                continue;
            }
            picked.add(item);
            if (predictor != null) {
                double predictedMs = predictor.predictBatchMs(picked);
                if (predictionBoundaryReached(
                        predictedMs, predictThresholdMs, strictPredictionLimit)) {
                    predictedTimeCapped = true;
                    // The head is indivisible, but every additional member that
                    // reaches the active boundary stays queued for the next
                    // group. Legacy earlyDispatch uses >=; the explicit strict
                    // maximum uses >, so equality is allowed only there.
                    picked.remove(picked.size() - 1);
                    break;
                }
            }
            shape = candidate;
            windowOpenedAtMs = Math.min(windowOpenedAtMs, item.enqueuedAtMs());
        }
        return new FixedPick(picked, shape,
                snapshot.queueVersion(), windowOpenedAtMs, predictedTimeCapped);
    }

    private static boolean predictionBoundaryReached(double predictedMs,
                                                      long thresholdMs,
                                                      boolean strictLimit) {
        return strictLimit ? predictedMs > thresholdMs : predictedMs >= thresholdMs;
    }

    private static BatchItem firstExpiredMember(List<BatchItem> orderedItems,
                                                int maxCount,
                                                long nowMs) {
        int inspected = Math.min(Math.max(1, maxCount), orderedItems.size());
        for (int index = 0; index < inspected; index++) {
            BatchItem item = orderedItems.get(index);
            if (item.ctx().requestExpired(nowMs)) {
                return item;
            }
        }
        return null;
    }

    private record FixedPick(List<BatchItem> items,
                             BatchShape shape,
                             long queueVersion,
                             long windowOpenedAtMs,
                             boolean predictedTimeCapped) {
    }

    private static boolean releaseDecisionGroup(BatcherContext ctx,
                                                List<BatchItem> picked,
                                                BatchShape shape,
                                                String reason,
                                                PrefillTimePredictor predictor,
                                                long expectedPredictorGeneration,
                                                long expectedQueueVersion) {
        BatchItem head = picked.get(0);
        // WorkerStatus resource counters can change independently of the
        // queue version while prediction is running. Re-check the exact shape
        // immediately before the version-atomic stage; a later tick will
        // rebuild P against the new capacities.
        if (!shape.fitsCompute(ctx.batchTokenCapacity())
                || !shape.fitsKv(ctx.batchKvCapacity())) {
            return false;
        }
        if (deliveryCapacityBlocked(ctx, head)) {
            return false;
        }
        // LearningPredictor publishes immutable weights under a generation.
        // Recheck identity and generation after all resource gates and as
        // close as possible to the version-atomic stage. A revision that lands
        // after prediction must never publish metadata from the older model.
        if (predictor != null
                && (ctx.prefillEp().getPredictor() != predictor
                || predictor.generation() != expectedPredictorGeneration)) {
            return false;
        }
        long waitMs = ctx.now() - head.enqueuedAtMs();
        int queueBefore = ctx.size();
        int queueDepth = queueBefore - picked.size();
        DecisionGroupMetadata metadata =
                new DecisionGroupMetadata(reason, queueDepth);
        if (!ctx.stageDecisionGroupIfVersion(
                picked, metadata, expectedQueueVersion,
                predictor, expectedPredictorGeneration)) {
            return false;
        }

        if (head.deliveryMode() == DeliveryMode.BATCH_ENQUEUE) {
            reportBatchDecision(ctx, picked, reason, waitMs, queueBefore);
        } else {
            Logger.debug("flexlb_route_decision reason={} picked_size={} "
                            + "wait_ms={} queue_before={} worker={} head_req_id={}",
                    reason, picked.size(), waitMs, queueBefore,
                    ctx.key(), head.requestId());
        }
        return true;
    }

    private static void reportBatchDecision(BatcherContext ctx,
                                            List<BatchItem> picked,
                                            String reason,
                                            long waitMs,
                                            int queueBefore) {
        BatchItem head = picked.get(0);
        ctx.reporter().reportDispatchReason(RoleType.PREFILL.name(), ctx.prefillEp().getIp(), reason);
        ctx.reporter().reportBatchSize(RoleType.PREFILL.name(), ctx.prefillEp().getIp(), reason, picked.size());

        // Compute batch-aggregated cache hit ratio
        long totalSeqLen = 0;
        long totalHitCache = 0;
        for (BatchItem item : picked) {
            totalSeqLen += item.seqLen();
            totalHitCache += item.hitCache();
        }
        ctx.reporter().reportBatchCacheHitMetrics(RoleType.PREFILL.name(), ctx.prefillEp().getIp(), totalHitCache, totalSeqLen);
        ctx.reporter().reportBatchTotalTokens(RoleType.PREFILL.name(), ctx.prefillEp().getIp(), reason, totalSeqLen);

        Logger.debug("flexlb_batch_decision reason={} picked_size={} "
                        + "wait_ms={} queue_before={} worker={} head_req_id={}",
                reason, picked.size(), waitMs, queueBefore, ctx.key(), head.requestId());
    }
}
