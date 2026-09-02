package org.flexlb.balance.eviction;

import java.util.Comparator;

/**
 * Structured cost of one eviction plan (design doc 7.1).
 *
 * @param priorityHarmProfile exact 1-100 priority harm; the first and absolute
 *                            comparison dimension
 * @param minVictimPriority    lowest priority among the plan's victims
 * @param priorityCost         saturated scalar cost retained for metrics and
 *                             diagnostics; it does not decide priority safety
 * @param victimCount          number of victims
 * @param deterministicTieBreak stable tie-break (smallest victim request id)
 */
public record PlanCost(
        PriorityHarmProfile priorityHarmProfile,
        int minVictimPriority,
        long priorityCost,
        int victimCount,
        long deterministicTieBreak) {

    /**
     * Plan preference (smaller = better): exact priority harm profile →
     * victimCount asc → deterministic tie-break asc.
     * The saturated scalar cost is omitted:
     * equal profiles already have equal exact weighted priority harm, while
     * unequal profiles must never be reordered by an overflowing scalar.
     */
    public static final Comparator<PlanCost> ORDER = Comparator
            .comparing(PlanCost::priorityHarmProfile)
            .thenComparingInt(PlanCost::victimCount)
            .thenComparingLong(PlanCost::deterministicTieBreak);
}
