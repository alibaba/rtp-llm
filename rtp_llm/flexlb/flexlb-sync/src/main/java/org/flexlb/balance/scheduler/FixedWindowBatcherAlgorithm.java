package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * Fixed-window batching algorithm with batch-full early dispatch, optional
 * predictor-based early dispatch, queue deadline drop, and resource-shape filtering.
 *
 * <p>The algorithm owns its {@link LinkedBlockingQueue} container and is
 * responsible for all queue operations: {@link #offer} enqueues, and
 * {@link #decide} removes picked / dropped items from the queue before
 * returning. The enclosing {@link WorkerBatcher} is a thin shell that
 * handles thread coordination and side-effect execution (metric reporting,
 * dispatch to the engine, settlement) but never touches the queue.
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
 * the queue, rather than waiting for the queue deadline to expire.
 */
public class FixedWindowBatcherAlgorithm implements BatcherAlgorithm {

    private final FlexlbConfig cfg;
    private final PrefillEndpoint prefillEp;
    private final LinkedBlockingQueue<BatchItem> queue = new LinkedBlockingQueue<>();

    public FixedWindowBatcherAlgorithm(FlexlbConfig cfg, PrefillEndpoint prefillEp) {
        this.cfg = cfg;
        this.prefillEp = prefillEp;
    }

    // ==================== BatcherAlgorithm ====================

    @Override
    public void offer(BatchItem item) {
        queue.add(item);
    }

    /**
     * Core decision method. Removes picked / dropped items from the
     * internal queue before returning.
     *
     * @return the decision for this cycle, or {@code null} when there is no
     *         action to take (park / engine backpressure)
     */
    @Override
    public BatchDecision decide() {
        BatchItem head = queue.peek();
        if (head == null) {
            return null;
        }

        long now = System.currentTimeMillis();
        long elapsedMs = now - head.enqueuedAtMs();
        long fixedWaitMs = cfg.getFlexlbBatchFixedWaitMs();
        int batchMaxCount = Math.max(1, cfg.getFlexlbBatchSizeMax());
        long predictThresholdMs = cfg.getFlexlbBatchPredictThresholdMs();
        long batchMaxTokens = batchTokenCapacity();

        // 0. Oversized head: can never be picked by any batch
        if (!BatchShape.empty().add(head).fitsCompute(batchMaxTokens)) {
            queue.remove(head);
            return new BatchDecision.Drop(head,
                    BatchDecision.DropCause.EXCEEDS_BATCH_TOKEN_CAPACITY,
                    "seq_len=" + head.seqLen() + " capacity=" + batchMaxTokens);
        }

        // 1. Queue deadline: drop expired head before backpressure check
        long queueDeadlineMs = cfg.getFlexlbBatchEnqueueDeadlineMs();
        if (queueDeadlineMs > 0 && elapsedMs > queueDeadlineMs) {
            queue.remove(head);
            return new BatchDecision.Drop(head,
                    BatchDecision.DropCause.QUEUE_DEADLINE_EXCEEDED,
                    "elapsed_ms=" + elapsedMs + " deadline_ms=" + queueDeadlineMs);
        }

        // 2. Engine backpressure: park if too many inflight batches
        int maxInflightBatches = cfg.getFlexlbBatchFixedMaxInflightBatches();
        if (maxInflightBatches > 0 && currentInflightCount() >= maxInflightBatches) {
            return null;
        }

        // 3. Queue size >= batchMaxCount → dispatch immediately (batch full)
        int queueSizeBefore = queue.size();
        if (queueSizeBefore >= batchMaxCount) {
            List<BatchItem> picked = pickWithinCapacity(batchMaxCount, batchMaxTokens, batchKvCapacity());
            if (!picked.isEmpty()) {
                picked.forEach(queue::remove);
                return new BatchDecision.Dispatch(picked, "batch_full", elapsedMs, queueSizeBefore);
            }
            return null;
        }

        // 4. Queue size < batchMaxCount → check window timeout
        if (elapsedMs >= fixedWaitMs) {
            List<BatchItem> picked = pickWithinCapacity(batchMaxCount, batchMaxTokens, batchKvCapacity());
            if (!picked.isEmpty()) {
                picked.forEach(queue::remove);
                return new BatchDecision.Dispatch(picked, "fixed_window_timeout", elapsedMs, queueSizeBefore);
            }
            return null;
        }

        // 5. Predictor-based early dispatch
        if (predictThresholdMs > 0) {
            List<BatchItem> candidates = pickWithinCapacity(batchMaxCount, batchMaxTokens, batchKvCapacity());
            if (!candidates.isEmpty() && predictor().predictBatchMs(candidates) >= predictThresholdMs) {
                candidates.forEach(queue::remove);
                return new BatchDecision.Dispatch(candidates, "predict_threshold", elapsedMs, queueSizeBefore);
            }
        }

        // 6. Park
        return null;
    }

    @Override
    public long queueWaitMs() {
        BatchItem head = queue.peek();
        if (head == null) {
            return 0;
        }
        long fixedWaitMs = cfg.getFlexlbBatchFixedWaitMs();
        long elapsed = System.currentTimeMillis() - head.enqueuedAtMs();
        return Math.max(0, fixedWaitMs - elapsed);
    }

    @Override
    public int size() {
        return queue.size();
    }

    @Override
    public void shutdown() {
        queue.clear();
    }

    /**
     * Drain all remaining items into the destination list. Used by
     * {@link WorkerBatcher#shutdown()} to settle queued items on close.
     */
    void drainTo(List<BatchItem> dst) {
        queue.drainTo(dst);
    }

    // ==================== Internal: capacity queries ====================

    /**
     * Effective strict padded-token limit for one FlexLB batch.
     *
     * <p>The Engine's FIFO scheduler rejects a group when its padded context
     * shape ({@code maxSeqLen * batchSize}) is greater than or equal to
     * {@code max_batch_tokens_size}. Prefer that exact worker-reported
     * limit; {@code max_seq_len} is a conservative fallback for workers
     * that have not populated the newer field yet. The FlexLB setting
     * remains an operator-controlled upper bound.
     */
    private long batchTokenCapacity() {
        long capacity = positiveOrUnlimited(cfg.getFlexlbBatchMaxCapacity());
        WorkerStatus status = prefillEp != null ? prefillEp.getStatus() : null;
        if (status == null) {
            return capacity;
        }
        long engineCapacity = status.getMaxBatchTokensSize();
        if (engineCapacity <= 0) {
            engineCapacity = status.getMaxSeqLen();
        }
        return Math.min(capacity, positiveOrUnlimited(engineCapacity));
    }

    /**
     * Latest worker-reported KV budget. A zero total means the worker has not
     * published KV capacity yet, so batching remains compute-bound only.
     */
    private long batchKvCapacity() {
        WorkerStatus status = prefillEp != null ? prefillEp.getStatus() : null;
        long total = status == null ? 0 : status.getTotalKvCacheTokens().get();
        if (total <= 0) {
            return Long.MAX_VALUE;
        }
        long available = Math.max(0, status.getAvailableKvCacheTokens().get());
        return Math.min(total, available);
    }

    /** Current inflight batch count on the prefill worker, for backpressure. */
    private int currentInflightCount() {
        return prefillEp.prefillInflightCount() + prefillEp.prefillEngineWorkCount();
    }

    /** Prefill-time predictor for predictor-based early dispatch. */
    private PrefillTimePredictor predictor() {
        return prefillEp.getPredictor();
    }

    private static long positiveOrUnlimited(long value) {
        return value > 0 ? value : Long.MAX_VALUE;
    }

    // ==================== Internal: picking ====================

    /**
     * Greedily pick up to {@code maxCount} items in FIFO order while keeping
     * the batch inside the Engine's compute and KV resource shape.
     *
     * <p>The FIFO head is never rejected on dynamic KV availability: temporary
     * KV pressure only prevents adding more members to this batch. The Engine
     * remains the final admission authority for the singleton request.
     */
    private List<BatchItem> pickWithinCapacity(int maxCount,
                                                long batchMaxTokens,
                                                long batchKvTokens) {
        List<BatchItem> picked = new ArrayList<>();
        BatchShape shape = BatchShape.empty();
        for (BatchItem item : new ArrayList<>(queue)) {
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
