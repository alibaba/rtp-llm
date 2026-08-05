package org.flexlb.balance.autotpm;

import org.flexlb.balance.scheduler.QueueSnapshot;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Pure function: {@link QueueSnapshot} + incoming request → {@link PrefillEvictionPlan}.
 *
 * <p>Stateless and side-effect free — performs no queue mutation, no CAS.
 * The caller ({@code PlanCommitter}) is responsible for committing the plan.
 *
 * <h2>Hard rule</h2>
 * A victim is eligible only when {@code victim.priority < incoming.priority}.
 * Same-priority requests are never evicted. This is enforced by the candidate
 * filter before any sorting or selection.
 *
 * <h2>Sort order (first = most preferred to evict)</h2>
 * <ol>
 *   <li>Priority ascending (lowest priority evicted first — minimizes the
 *       maximum victim priority)</li>
 *   <li>Deadline descending (later deadline first — more slack, less harmful
 *       to evict among the same priority)</li>
 *   <li>Request ID descending (later arrival proxy — stable tie-break)</li>
 * </ol>
 *
 * <p>{@link QueueSnapshot.ItemSummary} does not carry an arrival timestamp,
 * so request ID (a snowflake, monotonically-ish increasing with arrival) is
 * used as the arrival tie-breaker.
 */
public final class EvictionPlanner {

    /**
     * Generate an eviction plan for an incoming request that cannot fit.
     *
     * @param snapshot         current queue snapshot (version + items)
     * @param incomingPriority priority of the incoming request (30-70)
     * @param incomingSeqLen   sequence length of the incoming request (unused in Phase 3)
     * @param maxVictims       max victims per decision (from config)
     * @return eviction plan, or empty plan if no eligible victims
     */
    public PrefillEvictionPlan plan(QueueSnapshot snapshot,
                                    int incomingPriority,
                                    long incomingSeqLen,
                                    int maxVictims) {
        if (snapshot == null) {
            return emptyPlan(0L);
        }

        // 1. Filter candidates: priority STRICTLY LESS than incoming (hard rule)
        List<QueueSnapshot.ItemSummary> candidates = snapshot.items().stream()
                .filter(item -> item.priority() < incomingPriority)
                .collect(Collectors.toCollection(ArrayList::new));

        if (candidates.isEmpty()) {
            return emptyPlan(snapshot.version());
        }

        // 2. Sort by eviction preference (first = most preferred to evict)
        candidates.sort(Comparator
                .comparingInt(QueueSnapshot.ItemSummary::priority)
                .thenComparing(Comparator
                        .comparingLong(QueueSnapshot.ItemSummary::deadlineMs)
                        .reversed())
                .thenComparing(Comparator
                        .comparingLong(QueueSnapshot.ItemSummary::requestId)
                        .reversed()));

        // 3. Select up to maxVictims
        int selectCount = Math.min(Math.max(0, maxVictims), candidates.size());
        List<QueueSnapshot.ItemSummary> selected = candidates.subList(0, selectCount);

        // 4. Compute structured cost
        long priorityCost = 0L;
        int stageCost = 0;
        long tieBreak = 0L;
        for (QueueSnapshot.ItemSummary v : selected) {
            priorityCost += PriorityCostFunction.f(v.priority());
            // Phase 3: all prefill-queue items are NOT_ACCEPTED (not dispatched)
            stageCost += PriorityCostFunction.g(PriorityCostFunction.VictimStage.NOT_ACCEPTED);
            tieBreak += v.requestId();
        }
        // resourceCost = 0 (no KV info in Phase 3 summary);
        // cachePenalty = 0 (cache match is a tie-break only, computed elsewhere later)
        PlanCost cost = new PlanCost(priorityCost, stageCost, 0L,
                selected.size(), 0.0, tieBreak);

        List<Long> victimIds = selected.stream()
                .mapToLong(QueueSnapshot.ItemSummary::requestId)
                .boxed()
                .collect(Collectors.toCollection(ArrayList::new));

        return new PrefillEvictionPlan(victimIds, cost, snapshot.version());
    }

    private static PrefillEvictionPlan emptyPlan(long version) {
        return new PrefillEvictionPlan(Collections.emptyList(),
                new PlanCost(0L, 0, 0L, 0, 0.0, 0L), version);
    }
}
