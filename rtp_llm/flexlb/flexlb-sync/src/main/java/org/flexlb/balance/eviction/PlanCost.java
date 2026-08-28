package org.flexlb.balance.eviction;

import java.util.Comparator;

/**
 * Structured cost of one eviction plan (design doc 7.1).
 *
 * @param priorityHarmProfile exact 1-100 priority harm; the first and absolute
 *                            comparison dimension
 * @param victimCount          number of victims
 * @param deterministicTieBreak stable tie-break (smallest victim request id)
 */
public record PlanCost(
        PriorityHarmProfile priorityHarmProfile,
        int victimCount,
        long deterministicTieBreak) {

    /**
     * Plan preference (smaller = better): exact priority harm profile →
     * victimCount asc → deterministic tie-break asc.
     * Scalar diagnostic cost stays on the proposal and never participates in
     * priority-safety ordering.
     */
    public static final Comparator<PlanCost> ORDER = Comparator
            .comparing(PlanCost::priorityHarmProfile)
            .thenComparingInt(PlanCost::victimCount)
            .thenComparingLong(PlanCost::deterministicTieBreak);
}
