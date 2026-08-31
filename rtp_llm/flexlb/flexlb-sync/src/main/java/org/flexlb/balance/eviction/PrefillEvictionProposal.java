package org.flexlb.balance.eviction;

import org.flexlb.balance.scheduler.ScheduledRequest;

import java.util.Comparator;
import java.util.List;

/**
 * Pure planning result for one prefill-queue eviction on one endpoint.
 * Produced by {@link EvictionPlanner#planPrefillQueue}; carries no live
 * endpoint reference and has zero side effects.
 *
 * @param endpointId          prefill endpoint key ("ip:httpPort")
 * @param victims             selected victims in eviction order
 * @param rawCost             sum of f(victim.priority)
 * @param cost                structured {@link PlanCost} for 7.2 comparison
 */
public record PrefillEvictionProposal(
        String endpointId,
        List<ScheduledRequest> victims,
        long rawCost,
        PlanCost cost) {

    public PrefillEvictionProposal {
        victims = List.copyOf(victims);
        if (victims.isEmpty()) {
            throw new IllegalArgumentException(
                    "Prefill eviction proposal requires exact victims");
        }
    }

    /**
     * Cross-endpoint plan preference (smaller = better): structured cost →
     * endpointId asc. The structured cost keeps priority harm absolute, then
     * minimizes victim count and applies a deterministic request-id tie-break.
    */
    public static final Comparator<PrefillEvictionProposal> ORDER = Comparator
            .comparing(PrefillEvictionProposal::cost, PlanCost.ORDER)
            .thenComparing(PrefillEvictionProposal::endpointId);
}
