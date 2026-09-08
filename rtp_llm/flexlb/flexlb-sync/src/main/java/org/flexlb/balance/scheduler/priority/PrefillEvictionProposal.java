package org.flexlb.balance.scheduler.priority;

import java.util.Comparator;
import java.util.List;

/**
 * Pure planning result for one prefill-queue eviction on one endpoint.
 * Produced by {@link EvictionPlanner#planPrefillQueue}; carries no live
 * endpoint reference and has zero side effects.
 *
 * @param endpointId          prefill endpoint key ("ip:httpPort")
 * @param queueVersion        queue version the plan was built against
 * @param victims             selected victims in eviction order
 * @param rawCost             sum of f(victim.priority)
 * @param cacheHitTokens      incoming request's cache-hit tokens on this endpoint
 * @param boundedCacheBenefit cache benefit after the anti-inversion bound
 * @param netCost             rawCost - boundedCacheBenefit
 * @param cost                structured {@link PlanCost} for 7.2 comparison
 */
public record PrefillEvictionProposal(
        String endpointId,
        long queueVersion,
        List<QueuedRequestSnapshot> victims,
        long rawCost,
        long cacheHitTokens,
        long boundedCacheBenefit,
        long netCost,
        PlanCost cost) {

    /**
     * Cross-endpoint plan preference (smaller = better): exact priority harm
     * profile → netCost asc → rawCost asc → victimCount asc →
     * cacheHitTokens desc → maxVictimDeadline desc → endpointId asc.
     * The profile is deliberately first so cache benefit and saturated scalar
     * diagnostics cannot reverse an absolute priority decision.
    */
    public static final Comparator<PrefillEvictionProposal> ORDER = Comparator
            .comparing((PrefillEvictionProposal p) -> p.cost().priorityHarmProfile())
            .thenComparingLong(PrefillEvictionProposal::netCost)
            .thenComparingLong(PrefillEvictionProposal::rawCost)
            .thenComparingInt(p -> p.victims().size())
            .thenComparing(PrefillEvictionProposal::cacheHitTokens, Comparator.reverseOrder())
            .thenComparing(p -> p.cost().latestVictimDeadline(), Comparator.reverseOrder())
            .thenComparing(PrefillEvictionProposal::endpointId);
}
