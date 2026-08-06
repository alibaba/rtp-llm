package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Priority-aware batching algorithm with queue yield semantics.
 *
 * <p>Sorting comparator: priority descending → deadline ascending →
 * enqueuedAtMs ascending → requestId ascending. Items with the same
 * priority maintain strict FIFO order.
 *
 * <p>Yield behavior in {@link #pickWithinCapacity}: when the head item
 * (highest priority) has SLO risk (elapsed > sloMs * 0.5), items whose
 * priority is strictly less than the head's priority are skipped for this
 * round but remain in the queue for future rounds.
 *
 * <p>Uses a synchronized {@link ArrayList} with insertion sort — lock
 * granularity is per-worker so contention is low.
 */
public class PriorityYieldBatcherAlgorithm implements BatcherAlgorithm {

    private final FlexlbConfig cfg;
    private final PrefillEndpoint prefillEp;
    private final List<BatchItem> queue = new ArrayList<>();
    private final Object lock = new Object();

    /** Comparator: priority DESC → deadline ASC → enqueuedAtMs ASC → requestId ASC. */
    private static final Comparator<BatchItem> COMPARATOR =
            Comparator.comparingInt(BatchItem::priority).reversed()
                    .thenComparingLong(PriorityYieldBatcherAlgorithm::deadline)
                    .thenComparingLong(BatchItem::enqueuedAtMs)
                    .thenComparingLong(BatchItem::requestId);

    public PriorityYieldBatcherAlgorithm(FlexlbConfig cfg, PrefillEndpoint prefillEp) {
        this.cfg = cfg;
        this.prefillEp = prefillEp;
    }

    @Override
    public void offer(BatchItem item) {
        synchronized (lock) {
            // Insertion sort to maintain sorted order
            int idx = insertionIndex(item);
            queue.add(idx, item);
        }
    }

    @Override
    public BatchDecision decide() {
        synchronized (lock) {
            if (queue.isEmpty()) {
                return null;
            }

            BatchItem head = queue.get(0);
            long now = System.currentTimeMillis();
            long elapsedMs = now - head.enqueuedAtMs();
            long fixedWaitMs = cfg.getFlexlbBatchFixedWaitMs();
            int batchMaxCount = Math.max(1, cfg.getFlexlbBatchSizeMax());
            long predictThresholdMs = cfg.getFlexlbBatchPredictThresholdMs();
            long batchMaxTokens = batchTokenCapacity();

            // 0. Oversized head: can never be picked by any batch
            if (!BatchShape.empty().add(head).fitsCompute(batchMaxTokens)) {
                queue.remove(0);
                return new BatchDecision.Drop(head,
                        BatchDecision.DropCause.EXCEEDS_BATCH_TOKEN_CAPACITY,
                        "seq_len=" + head.seqLen() + " capacity=" + batchMaxTokens);
            }

            // 1. Queue deadline: drop expired head before backpressure check
            long queueDeadlineMs = cfg.getFlexlbBatchEnqueueDeadlineMs();
            if (queueDeadlineMs > 0 && elapsedMs > queueDeadlineMs) {
                queue.remove(0);
                return new BatchDecision.Drop(head,
                        BatchDecision.DropCause.QUEUE_DEADLINE_EXCEEDED,
                        "elapsed_ms=" + elapsedMs + " deadline_ms=" + queueDeadlineMs);
            }

            // 2. Engine backpressure: park if too many inflight batches
            int maxInflightBatches = cfg.getFlexlbBatchFixedMaxInflightBatches();
            if (maxInflightBatches > 0 && currentInflightCount() >= maxInflightBatches) {
                return null;
            }

            // Determine yield condition: head has SLO risk
            boolean yieldActive = false;
            int headPriority = head.priority();
            long sloMs = cfg.resolveSloMs(head.seqLen());
            if (sloMs > 0 && elapsedMs > sloMs / 2) {
                yieldActive = true;
            }

            // 3. Queue size >= batchMaxCount → dispatch immediately (batch full)
            int queueSizeBefore = queue.size();
            if (queueSizeBefore >= batchMaxCount) {
                List<BatchItem> picked = pickWithinCapacity(
                        batchMaxCount, batchMaxTokens, batchKvCapacity(), yieldActive, headPriority);
                if (!picked.isEmpty()) {
                    queue.removeAll(picked);
                    return new BatchDecision.Dispatch(picked, "batch_full", elapsedMs, queueSizeBefore);
                }
                return null;
            }

            // 4. Queue size < batchMaxCount → check window timeout
            if (elapsedMs >= fixedWaitMs) {
                List<BatchItem> picked = pickWithinCapacity(
                        batchMaxCount, batchMaxTokens, batchKvCapacity(), yieldActive, headPriority);
                if (!picked.isEmpty()) {
                    queue.removeAll(picked);
                    return new BatchDecision.Dispatch(picked, "fixed_window_timeout", elapsedMs, queueSizeBefore);
                }
                return null;
            }

            // 5. Predictor-based early dispatch
            if (predictThresholdMs > 0) {
                List<BatchItem> candidates = pickWithinCapacity(
                        batchMaxCount, batchMaxTokens, batchKvCapacity(), yieldActive, headPriority);
                if (!candidates.isEmpty() && predictor().predictBatchMs(candidates) >= predictThresholdMs) {
                    queue.removeAll(candidates);
                    return new BatchDecision.Dispatch(candidates, "predict_threshold", elapsedMs, queueSizeBefore);
                }
            }

            // 6. Park
            return null;
        }
    }

    @Override
    public long queueWaitMs() {
        synchronized (lock) {
            if (queue.isEmpty()) {
                return 0;
            }
            long fixedWaitMs = cfg.getFlexlbBatchFixedWaitMs();
            long elapsed = System.currentTimeMillis() - queue.get(0).enqueuedAtMs();
            return Math.max(0, fixedWaitMs - elapsed);
        }
    }

    @Override
    public int size() {
        synchronized (lock) {
            return queue.size();
        }
    }

    @Override
    public void shutdown() {
        synchronized (lock) {
            queue.clear();
        }
    }

    @Override
    public void drainTo(List<BatchItem> dst) {
        synchronized (lock) {
            dst.addAll(queue);
            queue.clear();
        }
    }

    // ==================== Internal: picking with yield ====================

    /**
     * Greedily pick items in sorted order while respecting compute/KV capacity.
     * When yield is active, items with priority strictly less than headPriority
     * are skipped (but remain in queue for future rounds).
     */
    private List<BatchItem> pickWithinCapacity(int maxCount,
                                               long batchMaxTokens,
                                               long batchKvTokens,
                                               boolean yieldActive,
                                               int headPriority) {
        List<BatchItem> picked = new ArrayList<>();
        BatchShape shape = BatchShape.empty();
        for (BatchItem item : queue) {
            if (picked.size() >= maxCount) {
                break;
            }
            // Yield: skip lower-priority items when head has SLO risk
            if (yieldActive && item.priority() < headPriority) {
                continue;
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

    // ==================== Internal: utilities ====================

    private int insertionIndex(BatchItem item) {
        // Binary search for insertion point
        int lo = 0, hi = queue.size();
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (COMPARATOR.compare(item, queue.get(mid)) < 0) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }

    private static long deadline(BatchItem item) {
        // deadline = enqueuedAtMs + sloMs; approximate with 0 if unknown
        return item.enqueuedAtMs();
    }

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

    private long batchKvCapacity() {
        WorkerStatus status = prefillEp != null ? prefillEp.getStatus() : null;
        long total = status == null ? 0 : status.getTotalKvCacheTokens().get();
        if (total <= 0) {
            return Long.MAX_VALUE;
        }
        long available = Math.max(0, status.getAvailableKvCacheTokens().get());
        return Math.min(total, available);
    }

    private int currentInflightCount() {
        return prefillEp.prefillInflightCount() + prefillEp.prefillEngineTaskCount();
    }

    private PrefillTimePredictor predictor() {
        return prefillEp.getPredictor();
    }

    private static long positiveOrUnlimited(long value) {
        return value > 0 ? value : Long.MAX_VALUE;
    }
}
