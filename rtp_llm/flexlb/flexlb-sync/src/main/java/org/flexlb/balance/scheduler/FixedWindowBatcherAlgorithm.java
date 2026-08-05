package org.flexlb.balance.scheduler;

import org.flexlb.balance.strategy.PrefillTimePredictor;

import java.util.ArrayList;
import java.util.List;

/**
 * Fixed-window batching algorithm with batch-full early dispatch, optional
 * predictor-based early dispatch, queue deadline drop, and resource-shape filtering.
 *
 * <p>The algorithm is a <b>pure decision function</b>: {@link #decide} only
 * reads batcher state through {@link BatcherReadView} and returns a
 * {@link BatchDecision} (or {@code null} for park / backpressure). All side
 * effects — queue mutation, dispatch, metric reporting, logging, parking —
 * are executed by {@link WorkerBatcher}.
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
 */
public class FixedWindowBatcherAlgorithm {

    /**
     * Compute the sort key used to order items in the per-worker
     * priority queue. Called by {@link WorkerBatcher#offer} before the
     * item is enqueued; the result is stored via
     * {@link BatchItem#setSortKey(long)}.
     */
    public long computeSortKey(BatcherReadView view, BatchItem item) {
        // FIFO: arrival timestamp as sort key
        return item.enqueuedAtMs();
    }

    /**
     * Estimated time a new request would wait before its batch is dispatched.
     */
    public long queueWaitMs(BatcherReadView view) {
        long now = view.now();
        long fixedWaitMs = view.cfg().getFlexlbBatchFixedWaitMs();
        int batchMaxCount = Math.max(1, view.cfg().getFlexlbBatchSizeMax());

        // 空队列 — 新请求启动新的 batch 周期
        if (view.isEmpty()) {
            if (batchMaxCount <= 1) {
                return 0;
            }
            return fixedWaitMs;
        }

        BatchItem head = view.peek();
        if (head == null) {
            // 竞态：isEmpty() 和 peek() 之间队列被清空
            return fixedWaitMs;
        }

        // batchMaxCount == 1：每个请求独立成 batch，立即 dispatch
        if (batchMaxCount <= 1) {
            return 0;
        }

        long elapsedMs = now - head.enqueuedAtMs();
        int queueSize = view.size();

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

    /**
     * Core decision method. Called by {@link WorkerBatcher#runLoop()} each
     * iteration when the queue is non-empty. Pure function of the read view:
     * no queue mutation, no dispatch, no metrics, no logging, no sleeping.
     *
     * @return the decision for this cycle, or {@code null} when there is no
     *         action to take (park / engine backpressure) and the caller
     *         should park briefly and re-invoke
     */
    public BatchDecision decide(BatcherReadView view) {
        if (view.isEmpty()) {
            return null;
        }

        BatchItem head = view.peek();
        if (head == null) {
            return null;
        }

        long elapsedMs = view.now() - head.enqueuedAtMs();
        long fixedWaitMs = view.cfg().getFlexlbBatchFixedWaitMs();
        int batchMaxCount = Math.max(1, view.cfg().getFlexlbBatchSizeMax());
        long predictThresholdMs = view.cfg().getFlexlbBatchPredictThresholdMs();
        long batchMaxTokens = view.batchTokenCapacity();

        // The Engine admits a group only when padded context tokens are strictly
        // below max_batch_tokens_size. Reject an impossible head explicitly so
        // it cannot block the FIFO queue or cause an entire group to fast-fail.
        if (!BatchShape.empty().add(head).fitsCompute(batchMaxTokens)) {
            return new BatchDecision.Drop(head,
                    BatchDecision.DropCause.EXCEEDS_BATCH_TOKEN_CAPACITY,
                    "seq_len=" + head.seqLen() + " capacity=" + batchMaxTokens);
        }

        // 0. Queue deadline: drop the head request if it has waited longer
        //     than the enqueue deadline. This runs BEFORE backpressure to
        //     ensure stale requests are cleared even when the engine is
        //     under sustained backpressure — otherwise the deadline check
        //     would never execute and expired requests would accumulate.
        long queueDeadlineMs = view.cfg().getFlexlbBatchEnqueueDeadlineMs();
        if (queueDeadlineMs > 0 && elapsedMs > queueDeadlineMs) {
            return new BatchDecision.Drop(head,
                    BatchDecision.DropCause.QUEUE_DEADLINE_EXCEEDED,
                    "elapsed_ms=" + elapsedMs + " deadline_ms=" + queueDeadlineMs);
        }

        // 1. Engine backpressure: park if the prefill worker already has too
        //    many batches inflight, to prevent overloading the engine.
        int maxInflightBatches = view.cfg().getFlexlbBatchFixedMaxInflightBatches();
        if (maxInflightBatches > 0 && view.currentInflightCount() >= maxInflightBatches) {
            return null;
        }

        // 2. Queue size >= batchMaxCount → dispatch immediately (batch full)
        int queueSizeBefore = view.size();
        if (queueSizeBefore >= batchMaxCount) {
            List<BatchItem> picked = pickWithinCapacity(
                    view, batchMaxCount, batchMaxTokens, view.batchKvCapacity());
            if (!picked.isEmpty()) {
                return new BatchDecision.Dispatch(picked, "batch_full", elapsedMs, queueSizeBefore);
            }
            return null;
        }

        // 3. Queue size < batchMaxCount → check window timeout
        if (elapsedMs >= fixedWaitMs) {
            List<BatchItem> picked = pickWithinCapacity(
                    view, batchMaxCount, batchMaxTokens, view.batchKvCapacity());
            if (!picked.isEmpty()) {
                return new BatchDecision.Dispatch(picked, "fixed_window_timeout", elapsedMs, queueSizeBefore);
            }
            return null;
        }

        // 4. Predictor-based early dispatch
        if (predictThresholdMs > 0) {
            PrefillTimePredictor predictor = view.predictor();
            List<BatchItem> candidates = pickWithinCapacity(
                    view, batchMaxCount, batchMaxTokens, view.batchKvCapacity());
            if (!candidates.isEmpty() && predictor.predictBatchMs(candidates) >= predictThresholdMs) {
                return new BatchDecision.Dispatch(candidates, "predict_threshold", elapsedMs, queueSizeBefore);
            }
        }

        // 5. Park
        return null;
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
    private static List<BatchItem> pickWithinCapacity(BatcherReadView view,
                                                       int maxCount,
                                                       long batchMaxTokens,
                                                       long batchKvTokens) {
        List<BatchItem> picked = new ArrayList<>();
        BatchShape shape = BatchShape.empty();
        for (BatchItem item : view.sortedItems()) {
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
}
