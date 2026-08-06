package org.flexlb.balance.scheduler;

import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Fixed-window batching algorithm with batch-full early dispatch, optional
 * predictor-based early dispatch, queue deadline drop, and resource-shape filtering.
 *
 * <h3>Algorithm</h3>
 * <ol>
 *   <li>Queue deadline: if the head request has waited longer than
 *       {@code flexlbBatchEnqueueDeadlineMs}, drop it as expired. This runs
 *       before backpressure to ensure stale requests are cleared even when
 *       the engine is under sustained backpressure.</li>
 *   <li>Oversized request rejection: if the head request's seqLen exceeds
 *       {@code flexlbBatchMaxCapacity}, it can never be picked by any batch,
 *       so it is dropped immediately instead of waiting for the deadline.</li>
 *   <li>Engine backpressure: if inflight batches ≥ max, park briefly.</li>
 *   <li>Batch full: if queue size ≥ {@code flexlbBatchSizeMax}, dispatch
 *       immediately without waiting for the window to expire.</li>
 *   <li>Fixed window timeout: if the head request has waited
 *       {@code flexlbBatchFixedWaitMs} or longer, dispatch whatever has
 *       accumulated (up to batch size limit).</li>
 *   <li>Predictor-based early dispatch: if {@code flexlbBatchPredictThresholdMs > 0}
 *       and the predictor estimates the accumulated batch will take at least
 *       that long, dispatch immediately.</li>
 *   <li>Otherwise park briefly and retry.</li>
 * </ol>
 *
 * <h3>Resource-shape filtering</h3>
 * When picking items for a batch, requests whose padded compute shape would
 * exceed {@code flexlbBatchMaxCapacity}, or whose combined sequence length
 * would exceed the latest worker-reported KV budget, remain in the queue for
 * a later batch.
 * <p>
 * However, a request whose own seqLen already exceeds
 * {@code flexlbBatchMaxCapacity} can never be picked by any batch. Such
 * oversized requests are rejected immediately when they reach the head of
 * the queue (see step 0.5 below), rather than waiting for the queue
 * deadline to expire.
 *
 * <h3>Key differences from {@link SloBudgetBatcherAlgorithm}</h3>
 * <ul>
 *   <li>No SLO deadline tracking — the sort key is only used for FIFO ordering.</li>
 *   <li>No EMA arrival rate estimation.</li>
 *   <li>Uses FIFO selection subject to the Engine-reported aggregate token
 *       capacity; it does not use SLO incremental-cost admission.</li>
 *   <li>Deadline is a simple max-wait threshold, not an SLO-deadline
 *       computed from predicted prefill time.</li>
 * </ul>
 */
public class FixedWindowBatcherAlgorithm implements BatcherAlgorithm {

    @Override
    public long computeSortKey(BatcherContext ctx, BatchItem item) {
        // FIFO: arrival timestamp as sort key
        return item.enqueuedAtMs();
    }

    @Override
    public long queueWaitMs(BatcherContext ctx) {
        long now = ctx.now();
        long fixedWaitMs = ctx.cfg().getFlexlbBatchFixedWaitMs();
        int batchMaxCount = Math.max(1, ctx.cfg().getFlexlbBatchSizeMax());

        // 空队列 — 新请求启动新的 batch 周期
        if (ctx.isEmpty()) {
            if (batchMaxCount <= 1) {
                return 0;
            }
            return fixedWaitMs;
        }

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
        int queueSize = ctx.size();

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
        if (ctx.isEmpty()) {
            return;
        }

        BatchItem head = ctx.peek();
        if (head == null) {
            return;
        }

        long elapsedMs = ctx.now() - head.enqueuedAtMs();
        long fixedWaitMs = ctx.cfg().getFlexlbBatchFixedWaitMs();
        int batchMaxCount = Math.max(1, ctx.cfg().getFlexlbBatchSizeMax());
        long predictThresholdMs = ctx.cfg().getFlexlbBatchPredictThresholdMs();
        long batchMaxTokens = ctx.batchTokenCapacity();

        // The Engine admits a group only when padded context tokens are strictly
        // below max_batch_tokens_size. Reject an impossible head explicitly so
        // it cannot block the FIFO queue or cause an entire group to fast-fail.
        if (!BatchShape.empty().add(head).fitsCompute(batchMaxTokens)) {
            ctx.rejectForBatchTokenCapacity(head, batchMaxTokens);
            return;
        }

        // 0. Queue deadline: drop the head request if it has waited longer
        //     than the enqueue deadline. This runs BEFORE backpressure to
        //     ensure stale requests are cleared even when the engine is
        //     under sustained backpressure — otherwise the deadline check
        //     would never execute and expired requests would accumulate.
        //     Auto-TPM never drops prioritized requests silently (design doc
        //     8.3): entry rejection/rescue/eviction cover them; deadline
        //     rescue is a later phase. No-priority (legacy) requests never
        //     enter any priority mechanism, so the drop stays active for them
        //     regardless of the global switch (P0-2 protection-vacuum fix).
        long queueDeadlineMs = ctx.cfg().getFlexlbBatchEnqueueDeadlineMs();
        if (queueDeadlineMs > 0 && elapsedMs > queueDeadlineMs
                && !(ctx.cfg().isAutoTpmEnabled() && head.hasPriority())) {
            Logger.warn("flexlb_batch_drop request_id={} reason=queue_deadline_exceeded "
                            + "elapsed_ms={} deadline_ms={}",
                    head.requestId(), elapsedMs, queueDeadlineMs);
            ctx.dropHead(head);
            return;
        }

        // 1. Engine backpressure: park if the prefill worker already has too
        //    many batches inflight, to prevent overloading the engine.
        int maxInflightBatches = ctx.cfg().getFlexlbBatchFixedMaxInflightBatches();
        if (maxInflightBatches > 0 && ctx.prefillEp().getInflightBatchCount() >= maxInflightBatches) {
            TimeUnit.MILLISECONDS.sleep(1);
            return;
        }

        // 2. Queue size >= batchMaxCount → dispatch immediately (batch full)
        if (ctx.size() >= batchMaxCount) {
            List<BatchItem> picked = pickWithinCapacity(
                    ctx, batchMaxCount, batchMaxTokens, ctx.batchKvCapacity());
            if (!picked.isEmpty()) {
                dispatch(ctx, picked, "batch_full");
            }
            return;
        }

        // 3. Queue size < batchMaxCount → check window timeout
        if (elapsedMs >= fixedWaitMs) {
            List<BatchItem> picked = pickWithinCapacity(
                    ctx, batchMaxCount, batchMaxTokens, ctx.batchKvCapacity());
            if (!picked.isEmpty()) {
                dispatch(ctx, picked, "fixed_window_timeout");
            }
            return;
        }

        // 4. Predictor-based early dispatch
        if (predictThresholdMs > 0) {
            PrefillTimePredictor predictor = ctx.prefillEp().getPredictor();
            List<BatchItem> candidates = pickWithinCapacity(
                    ctx, batchMaxCount, batchMaxTokens, ctx.batchKvCapacity());
            if (!candidates.isEmpty() && predictor.predictBatchMs(candidates) >= predictThresholdMs) {
                dispatch(ctx, candidates, "predict_threshold");
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
    private static List<BatchItem> pickWithinCapacity(BatcherContext ctx,
                                                       int maxCount,
                                                       long batchMaxTokens,
                                                       long batchKvTokens) {
        List<BatchItem> picked = new ArrayList<>();
        BatchShape shape = BatchShape.empty();
        for (BatchItem item : ctx.sortedItems()) {
            if (picked.size() >= maxCount) {
                break;
            }
            BatchShape candidate = shape.add(item);
            if (!candidate.fitsCompute(batchMaxTokens)) {
                break;
            }
            if (!picked.isEmpty() && !candidate.fitsKv(batchKvTokens)) {
                break;
            }
            picked.add(item);
            shape = candidate;
        }
        return picked;
    }

    private static void dispatch(BatcherContext ctx, List<BatchItem> picked, String reason) {
        BatchItem head = picked.get(0);
        long waitMs = ctx.now() - head.enqueuedAtMs();

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
                reason, picked.size(), waitMs, ctx.size(), ctx.key(), head.requestId());

        ctx.dispatch(picked,
                new DispatchMeta(reason, ctx.size() - picked.size()));
    }
}
