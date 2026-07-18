package org.flexlb.balance.scheduler;

import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Fixed-window batching algorithm with batch-full early dispatch, optional
 * predictor-based early dispatch, queue deadline drop, and token-cap filtering.
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
 * <h3>Token-cap filtering</h3>
 * When picking items for a batch, requests whose cumulative seqLen would
 * exceed {@code flexlbBatchMaxCapacity} are skipped (not included in the
 * current batch) but remain in the queue for subsequent batches.
 * <p>
 * However, a request whose own seqLen already exceeds
 * {@code flexlbBatchMaxCapacity} can never be picked by any batch. Such
 * oversized requests are rejected immediately when they reach the head of
 * the queue (see step 0.5 below), rather than waiting for the queue
 * deadline to expire.
 *
 * <h3>Key differences from {@link SloBudgetBatcherAlgorithm}</h3>
 * <ul>
 *   <li>No EMA arrival rate estimation.</li>
 *   <li>Deadline is a simple max-wait threshold, not an SLO-deadline
 *       computed from predicted prefill time.</li>
 * </ul>
 */
@Component
public class FixedWindowBatcherAlgorithm implements BatcherAlgorithm {

    @Override
    public long computeSortKey(BatcherContext ctx, BatchItem item) {
        // FIFO: arrival timestamp as sort key
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

        // 0. Queue deadline: drop the head request if it has waited longer
        //     than the enqueue deadline. This runs BEFORE backpressure to
        //     ensure stale requests are cleared even when the engine is
        //     under sustained backpressure — otherwise the deadline check
        //     would never execute and expired requests would accumulate.
        long queueDeadlineMs = ctx.cfg().getFlexlbBatchEnqueueDeadlineMs();
        if (queueDeadlineMs > 0 && elapsedMs > queueDeadlineMs) {
            Logger.warn("flexlb_batch_drop request_id={} reason=queue_deadline_exceeded "
                            + "elapsed_ms={} deadline_ms={}",
                    head.requestId(), elapsedMs, queueDeadlineMs);
            ctx.dropHead(head);
            return;
        }

        // 0.5. Reject oversized requests immediately instead of waiting for deadline.
        //     A request whose seqLen exceeds batchMaxTokens can never be picked by
        //     pickFirstN, so it would otherwise sit in the queue for 5s (deadline)
        //     before being dropped with a misleading BATCH_SLO_EXPIRED error.
        long batchMaxTokens = ctx.cfg().getFlexlbBatchMaxCapacity();
        if (batchMaxTokens > 0 && head.seqLen() > batchMaxTokens) {
            Logger.warn("flexlb_batch_drop request_id={} reason=request_oversized "
                            + "seq_len={} batch_max_tokens={}",
                    head.requestId(), head.seqLen(), batchMaxTokens);
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
            List<BatchItem> picked = pickFirstN(ctx, batchMaxCount);
            if (!picked.isEmpty()) {
                dispatch(ctx, picked, "batch_full");
            }
            return;
        }

        // 3. Queue size < batchMaxCount → check window timeout
        if (elapsedMs >= fixedWaitMs) {
            List<BatchItem> picked = pickFirstN(ctx, batchMaxCount);
            if (!picked.isEmpty()) {
                dispatch(ctx, picked, "fixed_window_timeout");
            }
            return;
        }

        // 4. Predictor-based early dispatch
        if (predictThresholdMs > 0) {
            PrefillTimePredictor predictor = ctx.prefillEp().getPredictor();
            List<BatchItem> candidates = pickFirstN(ctx, batchMaxCount);
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
     * Pick the first {@code maxCount} items from the queue in sorted order.
     * Requests whose cumulative seqLen would exceed {@code flexlbBatchMaxCapacity}
     * are skipped — they remain in the queue for subsequent batches.
     */
    private static List<BatchItem> pickFirstN(BatcherContext ctx, int maxCount) {
        long sumTokens = 0;
        long batchMaxTokens = ctx.cfg().getFlexlbBatchMaxCapacity();
        List<BatchItem> picked = new ArrayList<>();
        for (BatchItem item : ctx.sortedItems()) {
            if (picked.size() >= maxCount) {
                break;
            }
            long nextTokens = sumTokens + item.seqLen();
            if (batchMaxTokens > 0 && nextTokens > batchMaxTokens) {
                continue;
            }
            sumTokens = nextTokens;
            picked.add(item);
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
                new DispatchMeta(reason, 1.0, ctx.cfg().getFlexlbBatchMaxCapacity(), ctx.size() - picked.size()));
    }
}
