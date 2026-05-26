package org.flexlb.balance.dp;

/**
 * Abstract prefill-time estimator used by {@link SloBudgetBatcher} to decide
 * whether a candidate batch fits the head request's SLO budget.
 *
 * <p>Contract: the function MUST be monotonically non-decreasing in
 * {@code totalComputeTokens} when {@code totalHitCacheTokens} is held fixed.
 * The batcher does not assume linearity; non-linear predictors plug in as long
 * as the monotonicity invariant holds.
 */
public interface PrefillTimePredictor {

    /**
     * Estimate the prefill step time (ms) for a batch whose summed input length
     * is {@code totalComputeTokens} and whose summed prefix cache hit is
     * {@code totalHitCacheTokens}.
     */
    long estimateMs(long totalComputeTokens, long totalHitCacheTokens);
}
