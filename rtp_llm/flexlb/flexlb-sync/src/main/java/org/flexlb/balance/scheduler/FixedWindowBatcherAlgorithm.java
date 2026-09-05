package org.flexlb.balance.scheduler;

import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.config.BatchDispatcherConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Fixed-window batching algorithm with batch-full early dispatch, optional
 * predictor-based early dispatch, request-expiration drop, and resource-shape filtering.
 *
 * <h3>Algorithm</h3>
 * <ol>
 *   <li>Request expiration: drop the head at its absolute expiration. This runs
 *       before backpressure to ensure stale requests are cleared even when
 *       the engine is under sustained backpressure.</li>
 *   <li>Oversized request rejection: if the head request's seqLen exceeds
 *       the worker token capacity, it can never be picked by any batch,
 *       so it is dropped immediately instead of waiting for the deadline.</li>
 *   <li>Engine backpressure: if inflight batches ≥ max, park briefly.</li>
 *   <li>Batch full: if queue size reaches {@code dispatcher.maxRequests}, dispatch
 *       immediately without waiting for the window to expire.</li>
 *   <li>Fixed window timeout: if the head request has waited
 *       {@code dispatcher.maxCollectionWaitMs} or longer, dispatch whatever has
 *       accumulated (up to batch size limit).</li>
 *   <li>Predictor-based early dispatch when configured
 *       and the predictor estimates the accumulated batch will take at least
 *       that long, dispatch immediately.</li>
 *   <li>Otherwise park briefly and retry.</li>
 * </ol>
 *
 * <h3>Resource-shape filtering</h3>
 * When picking items for a batch, requests whose padded compute shape would
 * exceed the worker/fallback token capacity, or whose combined sequence length
 * would exceed the latest worker-reported KV budget, remain in the queue for
 * a later batch.
 * <p>
 * However, a request whose own seqLen already exceeds
 * that capacity can never be picked by any batch. Such
 * oversized requests are rejected immediately when they reach the head of
 * the queue (see step 0.5 below), rather than waiting for the queue
 * request expiration.
 *
 */
public class FixedWindowBatcherAlgorithm implements BatcherAlgorithm {

    @Override
    public long queueWaitMs(BatcherContext ctx) {
        BatchDispatcherConfig batch = ctx.cfg().batchDispatcher();
        long fixedWaitMs = batch.getMaxCollectionWaitMs();
        int batchMaxCount = Math.max(1, batch.getMaxRequests());

        // The charged-depth read is lock-free and covers the overwhelmingly
        // common empty-queue routing path. Fall back to the physical active
        // queue only when another request is already charged.
        if (ctx.isEmpty() || ctx.isActiveEmpty()) {
            if (batchMaxCount <= 1) {
                return 0;
            }
            return fixedWaitMs;
        }

        long now = ctx.now();
        BatchItem head = ctx.peek();
        if (head == null) {
            // 竞态：isEmpty() 和 peek() 之间队列被清空
            return fixedWaitMs;
        }

        // batchMaxCount == 1：每个请求独立成 batch，立即 dispatch
        if (batchMaxCount <= 1) {
            return 0;
        }

        long elapsedMs = now - head.enqueuedAtMs();
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

        BatchItem head = ctx.peek();
        if (head == null) {
            return;
        }

        long nowMs = ctx.now();
        long elapsedMs = nowMs - head.enqueuedAtMs();
        BatchDispatcherConfig batch = ctx.cfg().batchDispatcher();
        long fixedWaitMs = batch.getMaxCollectionWaitMs();
        int batchMaxCount = Math.max(1, batch.getMaxRequests());
        Long configuredPredictThresholdMs = batch.getEarlyDispatchPredictedExecutionMs();
        long predictThresholdMs = configuredPredictThresholdMs == null
                ? 0 : configuredPredictThresholdMs;
        long batchMaxTokens = ctx.batchTokenCapacity();

        // The caller's absolute request expiry is the only queue-age limit.
        // It is created once at admission and is never reset by retries.
        if (head.ctx().requestExpired(nowMs)) {
            Logger.debug("flexlb_batch_drop request_id={} reason=request_expired "
                            + "expires_at_ms={} now_ms={}",
                    head.requestId(), head.ctx().getRequestExpiresAtMs(), nowMs);
            ctx.dropHead(head);
            return;
        }

        // The Engine admits a group only when padded context tokens are strictly
        // below max_batch_tokens_size. Reject an impossible head explicitly so
        // it cannot block the FIFO queue or cause an entire group to fast-fail.
        if (!BatchShape.empty().add(head).fitsCompute(batchMaxTokens)) {
            ctx.rejectForBatchTokenCapacity(head, batchMaxTokens);
            return;
        }

        // 1. Engine backpressure: park if the prefill worker already has too
        //    many batches inflight, to prevent overloading the engine.
        if (head.deliveryMode() == DeliveryMode.BATCH_ENQUEUE) {
            Integer configuredMaxInflight = batch.getMaxInflightBatchesPerPrefillWorker();
            int maxInflightBatches = configuredMaxInflight == null
                    ? 0 : configuredMaxInflight;
            if (maxInflightBatches > 0
                    && ctx.prefillEp().getInflightBatchCount() >= maxInflightBatches) {
                TimeUnit.MILLISECONDS.sleep(1);
                return;
            }
        }

        FixedPick pick = null;

        // 2. A homogeneous head-mode prefix reaches batchMaxCount → dispatch
        //    immediately. A live configuration transition may leave the other
        //    delivery mode behind the head; it must neither join this group nor
        //    make the head's logical group appear full.
        if (ctx.activeSize() >= batchMaxCount) {
            pick = pickWithinCapacity(
                    ctx, head, batchMaxCount, batchMaxTokens, ctx.batchKvCapacity());
            if (pick.modePrefixSize() >= batchMaxCount) {
                releaseDecisionGroup(ctx, pick.items(), "batch_full");
                return;
            }
        }

        // 3. Queue size < batchMaxCount → check window timeout
        if (elapsedMs >= fixedWaitMs) {
            if (pick == null) {
                pick = pickWithinCapacity(
                        ctx, head, batchMaxCount, batchMaxTokens, ctx.batchKvCapacity());
            }
            releaseDecisionGroup(ctx, pick.items(), "fixed_window_timeout");
            return;
        }

        // 4. Predictor-based early dispatch
        if (predictThresholdMs > 0) {
            PrefillTimePredictor predictor = ctx.prefillEp().getPredictor();
            if (pick == null) {
                pick = pickWithinCapacity(
                        ctx, head, batchMaxCount, batchMaxTokens, ctx.batchKvCapacity());
            }
            List<BatchItem> candidates = pick.items();
            if (!candidates.isEmpty() && predictor.predictBatchMs(candidates) >= predictThresholdMs) {
                releaseDecisionGroup(ctx, candidates, "predict_threshold");
                return;
            }
        }

        // 5. Park
        TimeUnit.MILLISECONDS.sleep(1);
    }

    // ==================== Internal helpers ====================

    /**
     * Greedily pick up to {@code maxCount} items in FIFO order while keeping
     * the batch inside the Engine's compute and KV resource shape.
     *
     * <p>The FIFO head is never rejected on dynamic KV availability: temporary
     * KV pressure only prevents adding more members to this batch. The Engine
     * remains the final admission authority for the singleton request.
     */
    private static FixedPick pickWithinCapacity(BatcherContext ctx,
                                                BatchItem head,
                                                int maxCount,
                                                long batchMaxTokens,
                                                long batchKvTokens) {
        List<BatchItem> picked = new ArrayList<>();
        picked.add(head);
        BatchShape shape = BatchShape.empty().add(head);
        int modePrefixSize = 1;
        boolean capacityOpen = true;
        for (BatchItem item : ctx.sortedItems()) {
            if (item == head) {
                continue;
            }
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
            if (!picked.isEmpty() && !candidate.fitsKv(batchKvTokens)) {
                capacityOpen = false;
                continue;
            }
            picked.add(item);
            shape = candidate;
        }
        return new FixedPick(picked, modePrefixSize);
    }

    private record FixedPick(List<BatchItem> items, int modePrefixSize) {
    }

    private static void releaseDecisionGroup(BatcherContext ctx, List<BatchItem> picked, String reason) {
        BatchItem head = picked.get(0);
        long waitMs = ctx.now() - head.enqueuedAtMs();

        if (head.deliveryMode() == DeliveryMode.BATCH_ENQUEUE) {
            reportBatchDecision(ctx, picked, reason, waitMs);
        } else {
            Logger.debug("flexlb_route_decision reason={} picked_size={} "
                            + "wait_ms={} queue_before={} worker={} head_req_id={}",
                    reason, picked.size(), waitMs, ctx.size(), ctx.key(), head.requestId());
        }

        ctx.stageDecisionGroup(picked,
                new DecisionGroupMetadata(reason, ctx.size() - picked.size()));
    }

    private static void reportBatchDecision(BatcherContext ctx,
                                            List<BatchItem> picked,
                                            String reason,
                                            long waitMs) {
        BatchItem head = picked.get(0);
        ctx.reporter().reportDispatchReason(RoleType.PREFILL.name(), ctx.prefillEp().getStatus().getIpIndex(), reason);
        ctx.reporter().reportBatchSize(RoleType.PREFILL.name(), ctx.prefillEp().getStatus().getIpIndex(), reason, picked.size());

        // Compute batch-aggregated cache hit ratio
        long totalSeqLen = 0;
        long totalHitCache = 0;
        for (BatchItem item : picked) {
            totalSeqLen += item.seqLen();
            totalHitCache += item.hitCache();
        }
        ctx.reporter().reportBatchCacheHitMetrics(RoleType.PREFILL.name(), ctx.prefillEp().getStatus().getIpIndex(), totalHitCache, totalSeqLen);
        ctx.reporter().reportBatchTotalTokens(RoleType.PREFILL.name(), ctx.prefillEp().getStatus().getIpIndex(), reason, totalSeqLen);

        Logger.debug("flexlb_batch_decision reason={} picked_size={} "
                        + "wait_ms={} queue_before={} worker={} head_req_id={}",
                reason, picked.size(), waitMs, ctx.size(), ctx.key(), head.requestId());
    }
}
