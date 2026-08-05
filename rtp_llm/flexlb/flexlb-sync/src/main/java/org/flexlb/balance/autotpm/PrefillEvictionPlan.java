package org.flexlb.balance.autotpm;

import java.util.Collections;
import java.util.List;

/**
 * Immutable output of {@link EvictionPlanner#plan}.
 *
 * <p>Pure data — no side effects. Carries the victim request IDs (sorted by
 * eviction preference: lowest priority first, then later deadline, then later
 * arrival), the structured {@link PlanCost}, and the queue version at which
 * the snapshot was taken (used as the CAS expected version by the committer).
 */
public final class PrefillEvictionPlan {

    private final List<Long> victimRequestIds;
    private final PlanCost cost;
    private final long snapshotVersion;

    public PrefillEvictionPlan(List<Long> victimRequestIds, PlanCost cost, long snapshotVersion) {
        this.victimRequestIds = victimRequestIds == null
                ? Collections.emptyList()
                : Collections.unmodifiableList(victimRequestIds);
        this.cost = cost;
        this.snapshotVersion = snapshotVersion;
    }

    /** Victim request IDs in eviction order (first = most preferred to evict). */
    public List<Long> victimRequestIds() {
        return victimRequestIds;
    }

    public PlanCost cost() {
        return cost;
    }

    /** Queue version at snapshot time; pass to the committer as expectedVersion. */
    public long snapshotVersion() {
        return snapshotVersion;
    }

    public boolean isEmpty() {
        return victimRequestIds.isEmpty();
    }

    public int victimCount() {
        return victimRequestIds.size();
    }
}
