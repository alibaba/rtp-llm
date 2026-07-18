package org.flexlb.balance.strategy;

import org.flexlb.balance.scheduler.BatchItem;
import java.util.List;

/**
 * Prefill-time predictor contract.
 *
 * <p>Two evaluation modes:
 * <ul>
 *   <li>{@link #estimateMs(long, long)} — single request prediction</li>
 *   <li>{@link #predictBatchMs(List)} — batch prediction with aggregation</li>
 * </ul>
 *
 * <p>An optional {@link #learn(List, long, long)} callback is invoked on each batch
 * completion to feed back the actual-vs-predicted timing.
 */
public interface PrefillTimePredictor {

    /**
     * Estimate prefill time for a single request from raw token counts.
     *
     * @param totalTokens input length
     * @param hitTokens   cache-hit token count (0 ≤ hitTokens ≤ totalTokens)
     * @return predicted time in milliseconds
     */
    long estimateMs(long totalTokens, long hitTokens);

    /**
     * Estimate prefill time for a batch of requests.
     *
     * @param items batch items (may be empty)
     * @return predicted time in milliseconds (0 for an empty batch)
     */
    double predictBatchMs(List<BatchItem> items);

    /**
     * Estimate prefill time for a batch of requests without consulting or
     * populating any internal cache.
     *
     * <p>Useful in trial-and-error loops (e.g. batcher algorithm candidate
     * evaluation) where each candidate batch differs and cache lookups are
     * pure overhead.
     *
     * @param items batch items (may be empty)
     * @return predicted time in milliseconds (0 for an empty batch)
     */
    double predictBatchMsUncached(List<BatchItem> items);

    /**
     * Estimate prefill time for a batch consisting of existing queue items
     * plus a new request that hasn't been enqueued yet.
     *
     * <p>When {@code existingItems} is empty the new request starts a fresh
     * batch and the result is equivalent to {@link #estimateMs(long, long)}.
     * Otherwise the items plus the new request form a hypothetical batch whose
     * aggregated statistics feed the batch-level predictor.
     *
     * @param existingItems items already in the queue that would be in the same batch
     * @param newSeqLen     sequence length of the new request
     * @param newCacheHit   cache-hit tokens of the new request
     * @return estimated prefill time in milliseconds
     */
    double predictBatchMs(List<BatchItem> existingItems, long newSeqLen, long newCacheHit);

    /**
     * Learn from a completed batch's actual execution time.
     *
     * @param items       the batch requests (contains seqLen, hitCache, etc.)
     * @param predictedMs the formula-predicted execution time for the batch
     * @param actualMs    the engine-reported actual execution time
     */
    void learn(List<BatchItem> items, long predictedMs, long actualMs);
}
