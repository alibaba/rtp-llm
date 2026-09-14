package org.flexlb.balance.strategy.batch;

import java.util.List;

/**
 * Immutable candidates and cache-affinity policy for one request in a batch plan.
 *
 * <p>The candidate order is preserved because a plan stores one request-local
 * candidate index for each input request.</p>
 */
public record BatchPlanningRequest(List<BatchCandidate> candidates, long maxExtraTtftMs,
                                   double minPrefixHitPercent, long seqLen,
                                   boolean affinity) {

    public BatchPlanningRequest {
        candidates = List.copyOf(candidates);
        if (maxExtraTtftMs < 0L || minPrefixHitPercent < 0.0 || seqLen < 0L) {
            throw new IllegalArgumentException("request policy values must be non-negative");
        }
    }
}
