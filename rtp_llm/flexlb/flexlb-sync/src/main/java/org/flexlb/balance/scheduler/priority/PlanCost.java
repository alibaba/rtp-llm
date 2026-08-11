package org.flexlb.balance.scheduler.priority;

import java.util.Comparator;

/**
 * Structured cost of one eviction plan (design doc 7.1).
 *
 * @param minVictimPriority    lowest priority among the plan's victims
 * @param priorityCost         net priority cost: sum of f(victim.priority)
 *                             minus the bounded cache benefit
 * @param victimCount          number of victims
 * @param cacheBenefit         bounded cache benefit already subtracted from
 *                             {@link #priorityCost}
 * @param latestVictimDeadline latest deadline among victims (epoch ms) —
 *                             later deadline = more slack = cheaper to evict
 * @param deterministicTieBreak stable tie-break (smallest victim request id)
 */
public record PlanCost(
        int minVictimPriority,
        long priorityCost,
        int victimCount,
        long cacheBenefit,
        long latestVictimDeadline,
        long deterministicTieBreak) {

    /**
     * Design doc 7.2 plan preference (smaller = better): priorityCost asc →
     * victimCount asc → cacheBenefit desc → latestVictimDeadline desc →
     * deterministic tie-break asc.
     */
    public static final Comparator<PlanCost> ORDER = Comparator
            .comparingLong(PlanCost::priorityCost)
            .thenComparingInt(PlanCost::victimCount)
            .thenComparing(PlanCost::cacheBenefit, Comparator.reverseOrder())
            .thenComparing(PlanCost::latestVictimDeadline, Comparator.reverseOrder())
            .thenComparingLong(PlanCost::deterministicTieBreak);
}
