package org.flexlb.balance.autotpm;

/**
 * Cost function for priority-based eviction planning.
 *
 * <p>The priority cost of evicting a victim with priority {@code p} is
 * {@code f(p) = B^rank(p)} where {@code rank(P30)=0, rank(P40)=1, ...,
 * rank(P70)=4}. This makes evicting a single P70 (rank 4, cost B^4)
 * strictly more expensive than evicting any number of lower-priority
 * victims whose combined rank is less than 4 — guaranteeing the hard
 * rule "victim.priority &lt; incoming.priority" is reflected in the cost.
 *
 * <p>The stage cost {@code g(stage)} penalizes evicting items that have
 * progressed further in the lifecycle: a NOT_ACCEPTED (queued, not
 * dispatched) item is cheapest to evict, a RUNNING item most expensive.
 *
 * <p>Base {@code B=512} satisfies the constraint
 * {@code B > maxVictimsPerDecision * max(g) * max(k_ratio)}:
 * with {@code maxVictims=8}, {@code max(g)=32}, {@code max(k_ratio)=1},
 * the right-hand side is {@code 8 * 32 = 256 < 512}.
 */
public final class PriorityCostFunction {

    /** Base of the exponential priority cost. */
    static final int B = 512;

    private PriorityCostFunction() {
    }

    /**
     * Priority cost: {@code B^rank(priority)}.
     *
     * @param priority victim priority (30/40/50/60/70)
     * @return exponential cost; P30=1, P40=512, P50=512^2, ...
     */
    public static long f(int priority) {
        int rank = rank(priority);
        long result = 1;
        for (int i = 0; i < rank; i++) {
            result *= B;
        }
        return result;
    }

    /**
     * Stage cost. NOT_ACCEPTED is cheapest (queued, not dispatched);
     * RUNNING is most expensive.
     *
     * @param stage victim lifecycle stage
     * @return stage cost weight
     */
    public static int g(VictimStage stage) {
        switch (stage) {
            case NOT_ACCEPTED:
                return 1;
            case ACCEPTED_NOT_RUNNING:
                return 4;
            case RUNNING:
                return 32;
            default:
                return 1;
        }
    }

    /**
     * Map a priority value to its rank: P30=0, P40=1, P50=2, P60=3, P70=4.
     * Unknown priorities are clamped to the nearest rank.
     *
     * @param priority victim priority (30-70)
     * @return rank 0..4
     */
    public static int rank(int priority) {
        if (priority <= 30) {
            return 0;
        }
        if (priority >= 70) {
            return 4;
        }
        return (priority - 30) / 10;
    }

    /**
     * Lifecycle stage of a potential eviction victim.
     */
    public enum VictimStage {
        /** Queued in the batcher, not yet dispatched to the engine. */
        NOT_ACCEPTED,
        /** Accepted by the engine but not yet running (enqueued on engine side). */
        ACCEPTED_NOT_RUNNING,
        /** Currently executing on the engine. */
        RUNNING
    }
}
