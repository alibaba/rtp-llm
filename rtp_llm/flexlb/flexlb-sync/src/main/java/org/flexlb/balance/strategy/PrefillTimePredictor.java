package org.flexlb.balance.strategy;

import java.util.List;

import org.flexlb.balance.scheduler.BatchItem;

/**
 * Prefill-time predictor contract.
 *
 * <p>Two evaluation modes:
 * <ul>
 *   <li>{@link #estimateMs(long, long)} — single request prediction</li>
 *   <li>{@link #predictBatchMs(List)} — batch prediction with aggregation</li>
 * </ul>
 *
 * <p>{@link #learn(PrefillBatchFeatures, long, long)} is invoked on each
 * eligible batch completion to feed back the actual-vs-predicted timing.
 */
public interface PrefillTimePredictor {

    /**
     * Monotonic version of the prediction model.
     *
     * <p>Immutable predictors keep the default value. Online-learning
     * predictors increment it after publishing new parameters so scheduling
     * decisions made against an older model are discarded.
     */
    default long generation() {
        return 0L;
    }

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

    enum LearningResult {
        MODEL_UNCHANGED,
        MODEL_UPDATED
    }

    /**
     * Payload-free learning entry point used by long-lived inflight accounting.
     * The result is the publication boundary for scheduling: callers must wake
     * decisions exactly when this call publishes a new model generation.
     */
    LearningResult learn(
            PrefillBatchFeatures features, long predictedMs, long actualMs);
}
