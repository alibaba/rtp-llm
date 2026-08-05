package org.flexlb.balance.autotpm;

import java.util.Collections;
import java.util.List;

/**
 * Immutable output of {@link DecodeEvictionPlanner}.
 *
 * <p>Pure data — no side effects. Carries the victim reservations selected
 * for eviction, the total slots and KV tokens freed, and the structured
 * {@link PlanCost}. The committer ({@link DecodeEvictionCommitter}) consumes
 * this to release victims from the {@link DecodeAdmissionTracker}.
 *
 * <p>Unlike {@link PrefillEvictionPlan}, there is no snapshot version because
 * the decode tracker uses per-reservation CAS (remove-if-present) rather than
 * a global version CAS. Victims that were already released (e.g. completed
 * between planning and committing) are simply skipped during commit.
 */
public final class DecodeEvictionPlan {

    private final List<DecodeReservation> victims;
    private final PlanCost cost;
    private final int slotsFreed;
    private final long kvFreed;
    private final String endpointKey;

    public DecodeEvictionPlan(List<DecodeReservation> victims, PlanCost cost,
                              int slotsFreed, long kvFreed, String endpointKey) {
        this.victims = victims == null
                ? Collections.emptyList()
                : Collections.unmodifiableList(victims);
        this.cost = cost;
        this.slotsFreed = slotsFreed;
        this.kvFreed = kvFreed;
        this.endpointKey = endpointKey;
    }

    /** Victims in eviction order (first = most preferred to evict). */
    public List<DecodeReservation> victims() {
        return victims;
    }

    /** Convenience: victim request IDs in eviction order. */
    public List<Long> victimRequestIds() {
        return victims.stream()
                .map(DecodeReservation::requestId)
                .collect(java.util.stream.Collectors.toList());
    }

    public PlanCost cost() {
        return cost;
    }

    /** Total decode slots freed by evicting these victims. */
    public int slotsFreed() {
        return slotsFreed;
    }

    /** Total KV cache tokens freed by evicting these victims. */
    public long kvFreed() {
        return kvFreed;
    }

    /** Endpoint key (ip:port) the victims belong to. */
    public String endpointKey() {
        return endpointKey;
    }

    public boolean isEmpty() {
        return victims.isEmpty();
    }

    public int victimCount() {
        return victims.size();
    }

    /**
     * @param neededSlots slots required by the incoming request
     * @return {@code true} if this plan frees enough slots
     */
    public boolean satisfiesSlots(int neededSlots) {
        return slotsFreed >= neededSlots;
    }

    /**
     * @param neededKv KV tokens required by the incoming request
     * @return {@code true} if this plan frees enough KV
     */
    public boolean satisfiesKv(long neededKv) {
        return kvFreed >= neededKv;
    }

    @Override
    public String toString() {
        return "DecodeEvictionPlan{ep=" + endpointKey
                + ", victims=" + victims.size()
                + ", slotsFreed=" + slotsFreed
                + ", kvFreed=" + kvFreed
                + ", cost=" + cost + "}";
    }
}
