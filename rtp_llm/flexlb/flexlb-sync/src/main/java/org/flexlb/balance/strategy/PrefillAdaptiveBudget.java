package org.flexlb.balance.strategy;

import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.config.FlexlbConfig;

/** Derive token budgets from the existing execution predictor and the deployment's wait SLO. */
public final class PrefillAdaptiveBudget {
    private PrefillAdaptiveBudget() { }

    public static long serviceWindowMs(FlexlbConfig config) {
        long wait = config.isQueue() ? config.queueScheduler().getQueueTimeoutMs() : 60_000;
        if (config.isQueue() && config.queueScheduler().getPrefillQueueShedding() != null) {
            wait = Math.min(wait, config.queueScheduler().getPrefillQueueShedding().getExpectedWaitThresholdMs());
        }
        return Math.clamp(wait / 15, 1, 2_000);
    }

    /** Bound incremental compute work, excluding the predictor's fixed per-execution overhead.
     * A fixed cost above the service window must not collapse the collection queue to one token.
     */
    public static long tokensFor(PrefillTimePredictor.Evaluator evaluator, long budgetMs) {
        if (budgetMs <= 0) { throw new IllegalArgumentException("positive service window required"); }
        long targetMs = com.google.common.math.LongMath.saturatedAdd(budgetMs,
                PrefillPredictionBoundary.predictSingleRequestMs(evaluator, 0, 0));
        long low = 1, high = 1;
        while (high < (1L << 30)
                && PrefillPredictionBoundary.predictSingleRequestMs(evaluator, high, 0) <= targetMs) {
            low = high;
            high *= 2;
        }
        while (high - low > 1) {
            long mid = low + (high - low) / 2;
            if (PrefillPredictionBoundary.predictSingleRequestMs(evaluator, mid, 0) <= targetMs) { low = mid; }
            else { high = mid; }
        }
        return low;
    }
}
