package org.flexlb.balance.scheduler;

import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Fixed-window batching algorithm with optional predictor-based early dispatch.
 *
 * <h3>Algorithm</h3>
 * <ol>
 *   <li>If the head request has waited {@code flexlbBatchFixedWaitMs} or longer,
 *       dispatch whatever has accumulated (up to batch size / capacity limits).</li>
 *   <li>Otherwise, if {@code flexlbBatchPredictThresholdMs > 0} and the
 *       predictor estimates the accumulated batch will take at least that long,
 *       dispatch immediately ("early dispatch").</li>
 *   <li>Otherwise park briefly and retry.</li>
 * </ol>
 *
 * <h3>Key differences from {@link SloBudgetBatcherAlgorithm}</h3>
 * <ul>
 *   <li>No SLO deadline tracking — does not read {@code BatchItem.deadlineMs()}.</li>
 *   <li>No EMA arrival rate estimation.</li>
 *   <li>No request dropping — oversized requests are skipped, never expired.</li>
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
        long batchMaxTokens = ctx.cfg().getFlexlbBatchMaxCapacity();
        long predictThresholdMs = ctx.cfg().getFlexlbBatchPredictThresholdMs();

        // 0. Engine backpressure: park if the prefill worker already has too
        //    many batches inflight, to prevent overloading the engine.
        //    Default 0 disables this gate — the batcher always dispatches.
        int maxInflightBatches = ctx.cfg().getFlexlbBatchFixedMaxInflightBatches();
        if (maxInflightBatches > 0 && ctx.prefillEp().getInflightBatchCount() >= maxInflightBatches) {
            TimeUnit.MILLISECONDS.sleep(1);
            return;
        }

        // 1. Fixed window timeout → must dispatch
        if (elapsedMs >= fixedWaitMs) {
            List<BatchItem> picked = pickUpTo(ctx, batchMaxCount, batchMaxTokens);
            if (picked.isEmpty() && !ctx.isEmpty()) {
                // All items exceed maxTokens — force-dispatch the head to avoid busy-wait
                BatchItem forced = ctx.peek();
                if (forced != null) {
                    picked = List.of(forced);
                }
            }
            if (!picked.isEmpty()) {
                dispatch(ctx, picked, "fixed_window_timeout");
            }
            return;
        }

        // 2. Predictor-based early dispatch
        if (predictThresholdMs > 0) {
            PrefillTimePredictor predictor = ctx.prefillEp().getPredictor();
            List<BatchItem> candidates = pickUpTo(ctx, batchMaxCount, batchMaxTokens);
            if (!candidates.isEmpty() && predictor.predictBatchMs(candidates) >= predictThresholdMs) {
                dispatch(ctx, candidates, "predict_threshold");
                return;
            }
        }

        // 3. Park
        TimeUnit.MILLISECONDS.sleep(1);
    }

    // ==================== Internal helpers ====================

    /**
     * Pick up to {@code maxCount} items from the queue, respecting
     * {@code maxTokens} (total token) limit. Items that would exceed
     * the capacity are skipped, not dropped.
     */
    private static List<BatchItem> pickUpTo(BatcherContext ctx, int maxCount, long maxTokens) {
        List<BatchItem> picked = new ArrayList<>();
        long sumTokens = 0;

        for (BatchItem item : ctx.sortedItems()) {
            if (picked.size() >= maxCount) {
                break;
            }
            long nextTokens = sumTokens + item.seqLen();
            if (nextTokens > maxTokens) {
                continue;  // skip, don't drop
            }
            picked.add(item);
            sumTokens = nextTokens;
        }
        return picked;
    }

    private static void dispatch(BatcherContext ctx, List<BatchItem> picked, String reason) {
        BatchItem head = picked.get(0);
        long waitMs = ctx.now() - head.enqueuedAtMs();

        ctx.reporter().reportDispatchReason("prefill", ctx.prefillEp().getIp(), reason);
        ctx.reporter().reportBatchSize("prefill", ctx.prefillEp().getIp(), reason, picked.size());

        // Compute batch-aggregated cache hit ratio
        long totalSeqLen = 0;
        long totalHitCache = 0;
        for (BatchItem item : picked) {
            totalSeqLen += item.seqLen();
            totalHitCache += item.hitCache();
        }
        ctx.reporter().reportBatchCacheHitMetrics("prefill", ctx.prefillEp().getIp(), totalHitCache, totalSeqLen);

        Logger.info("flexlb_batch_decision reason={} picked_size={} "
                        + "wait_ms={} queue_before={} worker={} head_req_id={}",
                reason, picked.size(), waitMs, ctx.size(), ctx.key(), head.requestId());

        ctx.dispatch(picked,
                new DispatchMeta(reason, 1.0, ctx.cfg().getFlexlbBatchMaxCapacity(), ctx.size() - picked.size()));
    }
}
