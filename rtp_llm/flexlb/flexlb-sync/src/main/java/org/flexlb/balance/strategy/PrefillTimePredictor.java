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
     * Learn from a completed batch's actual execution time.
     *
     * @param items       the batch requests (contains seqLen, hitCache, etc.)
     * @param predictedMs the formula-predicted execution time for the batch
     * @param actualMs    the engine-reported actual execution time
     */
    void learn(List<BatchItem> items, long predictedMs, long actualMs);
}
