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
     * DP-aware batch prediction: predict each dpRank sub-group separately and
     * return the maximum.  All DP ranks execute prefill in parallel, so the
     * batch completion time is governed by the slowest rank (bucket effect).
     *
     * <p>Default implementation degrades to flat prediction for backward
     * compatibility.  Concrete predictors that can benefit from per-DP
     * isolation should override this method.
     *
     * @param itemsByDp list of per-DP item groups (may contain empty sub-lists)
     * @return predicted time in milliseconds (0 if all groups are empty)
     */
    default double predictBatchMsByDp(List<List<BatchItem>> itemsByDp) {
        List<BatchItem> flat = itemsByDp.stream().flatMap(List::stream).toList();
        return predictBatchMs(flat);
    }

    /**
     * Learn from a completed batch's actual execution time.
     *
     * @param items       the batch requests (contains seqLen, hitCache, etc.)
     * @param predictedMs the formula-predicted execution time for the batch
     * @param actualMs    the engine-reported actual execution time
     */
    void learn(List<BatchItem> items, long predictedMs, long actualMs);
}
