package org.flexlb.balance.scheduler;

import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Decision algorithm behind every worker queue. It grows the largest group the
 * live bounds allow and releases it to the queue's configured delivery, which
 * is the only thing the delivery mode changes. A dispatcher that permits one
 * request per group therefore decides each arrival on its own, and whatever it
 * cannot decide yet stays queued.
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
 *       group growth: the dispatcher's largest decision group, the worker
 *       compute and KV shapes, and — when configured — a predicted execution
 *       time below the dispatcher's budget. The head is a mandatory member, so
 *       a single request already over that budget forms the whole group.</li>
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
        long fixedWaitMs = ctx.collectionWindowMs();
        int batchMaxCount = ctx.maxDecisionRequests();

        // The charged-depth read is lock-free and covers the overwhelmingly
        // common empty-queue routing path. Fall back to the physical active
        // queue only when another request is already charged.
        if (ctx.isActiveEmpty()) {
            if (batchMaxCount <= 1) {
                return 0;
            }
            return fixedWaitMs;
        }

        // batchMaxCount == 1：每个请求独立成 batch，立即 dispatch
        if (batchMaxCount <= 1) {
            return 0;
        }

        long windowOpenedAtMs = ctx.oldestActiveEnqueuedAtMs();
        if (windowOpenedAtMs == Long.MAX_VALUE) {
            // 竞态：isActiveEmpty() 和读取窗口锚点之间队列被清空
            return fixedWaitMs;
        }

        long elapsedMs = ctx.now() - windowOpenedAtMs;
        int queueSize = ctx.activeSize();

        // 新请求恰好填满一个 batch（当前 batch 或前置 dispatch 后的最后一个 batch）
        // 前面的满 batch 通过 step 2 (batch_full) 连续 dispatch，之间无 sleep 延迟
        // 新请求所在 batch 立即触发 batch_full → 等待 ≈ 0
        if (queueSize % batchMaxCount == batchMaxCount - 1) {
            return 0;
        }

        // 队列深度 < batchMaxCount 且窗口已超时 → fixed_window_timeout 立即触发
        if (queueSize < batchMaxCount && elapsedMs >= fixedWaitMs) {
            return 0;
        }

        // 队列深度 < batchMaxCount 且窗口未超时 → 等窗口剩余时间
        if (queueSize < batchMaxCount) {
            return Math.max(0, fixedWaitMs - elapsedMs);
        }

        // 队列深度 >= batchMaxCount，新请求不填满最后一个 batch
        // 前置 dispatch 后存在 partial batch，剩余 head 的 enqueuedAtMs 未知
        // O(1) 约束下无法精确计算窗口剩余时间，保守返回 fixedWaitMs（上界估计）
        return fixedWaitMs;
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
        batchMaxTokens = ctx.batchTokenCapacity();
        if (head.ctx().requestExpired(nowMs)) {
            Logger.debug("flexlb_batch_drop request_id={} reason=request_expired "
                            + "expires_at_ms={} now_ms={}",
                    head.requestId(), head.ctx().getRequestExpiresAtMs(), nowMs);
            ctx.dropHead(head);
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
                predictor, predictThresholdMs);
        List<BatchItem> candidates = pick.items();
        if (predictor != null && predictor.generation() != predictorGeneration) {
            // Do not publish a decision made against a model revision that was
            // replaced while the formula was being evaluated. The next worker
            // tick retries against the new immutable parameters.
            return;
        }

        // Dispatch once growth stopped at a bound that will not relax on its
        // own — the predicted execution budget, or a homogeneous head-mode
        // prefix reaching batchMaxCount — or once the collection window has run
        // out and there is nothing left to wait for. A live configuration
        // transition may leave the other delivery mode behind the head; it must
        // neither join this group nor make it appear full. Transient compute/KV
        // pressure can also limit growth, and until the window elapses the
        // queue-order behavior is to wait rather than bypass members.
        String reason = null;
        if (pick.predictedTimeCapped()) {
            reason = "predicted_execution_cap";
        } else if (pick.modePrefixSize() >= batchMaxCount) {
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
                && nowMs - windowOpenedAtMs >= fixedWaitMs;
    }

    private static boolean deliveryCapacityBlocked(BatcherContext ctx,
                                                    BatchItem head) {
        if (head.deliveryMode() != DeliveryMode.BATCH_ENQUEUE) {
            return false;
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
     * request whose own predicted execution already exceeds the budget forms the
     * whole batch. Temporary KV pressure prevents adding more members. The
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
            long predictThresholdMs) {
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
                && predictor.predictBatchMs(picked) >= predictThresholdMs;
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
            if (predictor != null
                    && predictor.predictBatchMs(picked) >= predictThresholdMs) {
                picked.remove(picked.size() - 1);
                predictedTimeCapped = true;
                break;
            }
            shape = candidate;
            windowOpenedAtMs = Math.min(windowOpenedAtMs, item.enqueuedAtMs());
        }
        return new FixedPick(picked, modePrefixSize, shape,
                snapshot.queueVersion(), windowOpenedAtMs, predictedTimeCapped);
    }

    private record FixedPick(List<BatchItem> items,
                             int modePrefixSize,
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
        if (head.deliveryMode() == DeliveryMode.BATCH_ENQUEUE) {
            int maxInflightBatches = ctx.maxInflightBatches();
            if (maxInflightBatches > 0
                    && ctx.prefillEp().getInflightBatchCount()
                    >= maxInflightBatches) {
                return false;
            }
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
