package org.flexlb.balance.scheduler;

import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Fixed-window batching algorithm with batch-full early dispatch and optional
 * predictor-based early dispatch.
 *
 * <h3>Algorithm</h3>
 * <ol>
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
 * <h3>Key differences from {@link SloBudgetBatcherAlgorithm}</h3>
 * <ul>
 *   <li>No SLO deadline tracking — does not read {@code BatchItem.deadlineMs()}.</li>
 *   <li>No EMA arrival rate estimation.</li>
 *   <li>Uses FIFO selection subject to the Engine-reported aggregate token
 *       capacity; it does not use SLO incremental-cost admission.</li>
 *   <li>No inflight-batch backpressure check.</li>
 * </ul>
 */
@Component
public class FixedWindowBatcherAlgorithm implements BatcherAlgorithm {

    @Override
    public long computeSortKey(BatcherContext ctx, BatchItem item) {
        // FIFO: arrival timestamp as sort key; no SLO deadline tracking
        return item.enqueuedAtMs();
    }

    @Override
    public long headWaitMs(BatcherContext ctx) {
        BatchItem head = ctx.peek();
        if (head == null) {
            return 0;
        }
        long elapsedMs = ctx.now() - head.enqueuedAtMs();
        return Math.max(0, ctx.cfg().getFlexlbBatchFixedWaitMs() - elapsedMs);
    }

    @Override
    public long queueWaitMs(BatcherContext ctx) {
        if (!ctx.isEmpty()) {
            return headWaitMs(ctx);
        }
        return ctx.cfg().getFlexlbBatchFixedWaitMs();
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

        // The Engine admits a group only when total context tokens are strictly
        // below max_batch_tokens_size. Reject an impossible head explicitly so
        // it cannot block the FIFO queue or cause an entire group to fast-fail.
        if (!BatcherContext.fitsBatchTokenCapacity(0, head.seqLen(), batchMaxTokens)) {
            ctx.rejectForBatchTokenCapacity(head, batchMaxTokens);
            return;
        }

        // 0. Engine backpressure: park if the prefill worker already has too
        //    many batches inflight, to prevent overloading the engine.
        //    Default 0 disables this gate — the batcher always dispatches.
        int maxInflightBatches = ctx.cfg().getFlexlbBatchFixedMaxInflightBatches();
        if (maxInflightBatches > 0 && ctx.prefillEp().getInflightBatchCount() >= maxInflightBatches) {
            TimeUnit.MILLISECONDS.sleep(1);
            return;
        }

        // 1. Queue size >= batchMaxCount → dispatch immediately (batch full)
        if (ctx.size() >= batchMaxCount) {
            List<BatchItem> picked = pickWithinCapacity(ctx, batchMaxCount, batchMaxTokens);
            dispatch(ctx, picked, "batch_full");
            return;
        }

        // 2. Queue size < batchMaxCount → check window timeout
        if (elapsedMs >= fixedWaitMs) {
            List<BatchItem> picked = pickWithinCapacity(ctx, batchMaxCount, batchMaxTokens);
            if (!picked.isEmpty()) {
                dispatch(ctx, picked, "fixed_window_timeout");
            }
            return;
        }

        // 3. Predictor-based early dispatch
        if (predictThresholdMs > 0) {
            PrefillTimePredictor predictor = ctx.prefillEp().getPredictor();
            List<BatchItem> candidates = pickWithinCapacity(ctx, batchMaxCount, batchMaxTokens);
            if (!candidates.isEmpty() && predictor.predictBatchMs(candidates) >= predictThresholdMs) {
                dispatch(ctx, candidates, "predict_threshold");
                return;
            }
        }

        // 4. Park
        TimeUnit.MILLISECONDS.sleep(1);
    }

    // ==================== Internal helpers ====================

    /**
     * Greedily pick up to {@code maxCount} items in FIFO order while keeping
     * the aggregate request sequence length strictly below the Engine limit.
     */
    private static List<BatchItem> pickWithinCapacity(BatcherContext ctx, int maxCount, long batchMaxTokens) {
        List<BatchItem> picked = new ArrayList<>();
        long totalTokens = 0;
        for (BatchItem item : ctx.sortedItems()) {
            if (picked.size() >= maxCount) {
                break;
            }
            if (!BatcherContext.fitsBatchTokenCapacity(totalTokens, item.seqLen(), batchMaxTokens)) {
                continue;
            }
            picked.add(item);
            totalTokens += item.seqLen();
        }
        return picked;
    }

    private static void dispatch(BatcherContext ctx, List<BatchItem> picked, String reason) {
        BatchItem head = picked.get(0);
        long waitMs = ctx.now() - head.enqueuedAtMs();

        ctx.reporter().reportDispatchReason(RoleType.PREFILL.name(), ctx.prefillEp().getIp(), ctx.prefillEp().ipPort(), reason);
        ctx.reporter().reportBatchSize(RoleType.PREFILL.name(), ctx.prefillEp().getIp(), ctx.prefillEp().ipPort(), reason, picked.size());

        // Compute batch-aggregated cache hit ratio
        long totalSeqLen = 0;
        long totalHitCache = 0;
        for (BatchItem item : picked) {
            totalSeqLen += item.seqLen();
            totalHitCache += item.hitCache();
        }
        ctx.reporter().reportBatchCacheHitMetrics(RoleType.PREFILL.name(), ctx.prefillEp().getIp(), ctx.prefillEp().ipPort(), totalHitCache, totalSeqLen);
        ctx.reporter().reportBatchTotalTokens(RoleType.PREFILL.name(), ctx.prefillEp().getIp(), ctx.prefillEp().ipPort(), reason, totalSeqLen);

        Logger.debug("flexlb_batch_decision reason={} picked_size={} "
                        + "wait_ms={} queue_before={} worker={} head_req_id={}",
                reason, picked.size(), waitMs, ctx.size(), ctx.key(), head.requestId());

        ctx.dispatch(picked,
                new DispatchMeta(reason, 1.0, ctx.size() - picked.size()));
    }
}
