package org.flexlb.balance.autotpm;

/**
 * Structured plan cost with lexicographic comparison.
 *
 * <p>Lower cost = better plan (preferred). Each field is compared in order;
 * the first non-zero comparison decides. This guarantees that the plan
 * minimizing the most significant dimension wins, with later fields acting
 * only as tie-breakers.
 *
 * <p>Comparison order:
 * <ol>
 *   <li>{@code priorityCost} — Σ f(victim.priority), minimize the maximum
 *       victim priority (evict least-important first)</li>
 *   <li>{@code stageCost} — Σ g(victim.stage), prefer evicting earlier-stage
 *       items (NOT_ACCEPTED before ACCEPTED_NOT_RUNNING before RUNNING)</li>
 *   <li>{@code resourceCost} — Σ k(victim.resource), prefer evicting items
 *       that release more resources (KV cache tokens)</li>
 *   <li>{@code victimCount} — fewer victims preferred</li>
 *   <li>{@code cachePenalty} — cache-hit benefit loss, used as tie-break only
 *       for prefill-queue scenarios (lower penalty = better)</li>
 *   <li>{@code tieBreak} — arrival-based stable tie-breaker</li>
 * </ol>
 */
public final class PlanCost implements Comparable<PlanCost> {

    private final long priorityCost;
    private final int stageCost;
    private final long resourceCost;
    private final int victimCount;
    private final double cachePenalty;
    private final long tieBreak;

    public PlanCost(long priorityCost, int stageCost, long resourceCost,
                    int victimCount, double cachePenalty, long tieBreak) {
        this.priorityCost = priorityCost;
        this.stageCost = stageCost;
        this.resourceCost = resourceCost;
        this.victimCount = victimCount;
        this.cachePenalty = cachePenalty;
        this.tieBreak = tieBreak;
    }

    public long priorityCost() { return priorityCost; }
    public int stageCost() { return stageCost; }
    public long resourceCost() { return resourceCost; }
    public int victimCount() { return victimCount; }
    public double cachePenalty() { return cachePenalty; }
    public long tieBreak() { return tieBreak; }

    @Override
    public int compareTo(PlanCost other) {
        int c = Long.compare(this.priorityCost, other.priorityCost);
        if (c != 0) return c;
        c = Integer.compare(this.stageCost, other.stageCost);
        if (c != 0) return c;
        c = Long.compare(this.resourceCost, other.resourceCost);
        if (c != 0) return c;
        c = Integer.compare(this.victimCount, other.victimCount);
        if (c != 0) return c;
        c = Double.compare(this.cachePenalty, other.cachePenalty);
        if (c != 0) return c;
        return Long.compare(this.tieBreak, other.tieBreak);
    }

    @Override
    public String toString() {
        return "PlanCost{pri=" + priorityCost + ", stage=" + stageCost
                + ", res=" + resourceCost + ", count=" + victimCount
                + ", cache=" + cachePenalty + ", tie=" + tieBreak + "}";
    }
}
