package org.flexlb.balance.scheduler.priority;

import org.flexlb.enums.DecodeTaskPhase;

/**
 * Priority cost primitives for eviction planning (design doc 7.3).
 *
 * <p>Priorities 30/40/50/60/70 map to ranks 0..4 and the single-value cost is
 * {@code f(priority) = 1024^rank}, so one victim of the next-higher priority
 * always costs more than any feasible set of victims of lower priorities
 * (a plan has at most {@code EvictionPlanner.MAX_VICTIMS_PER_PLAN << 1024} victims).
 *
 * <p>Cache benefit is bounded by {@code MIN_ADJACENT_GAP / 2} so no benefit
 * term can ever reverse a cross-priority cost comparison — the absolute
 * priority boundary (design doc 3.3) is preserved arithmetically.
 */
public final class PriorityCostFunction {

    /** Smallest cost gap between adjacent ranks: f(40) - f(30) = 1024 - 1. */
    public static final long MIN_ADJACENT_GAP = 1023L;

    // h(case) cross-type weights (design doc 7.6). A combined slot+KV plan
    // sums its already-weighted parts and is never multiplied again.
    /** h(DECODE_SLOT_FULL): frees concurrency, may affect D admission. */
    public static final long H_DECODE_SLOT_FULL = 4L;
    /** h(DECODE_KV_FULL): KV-sensitive, released KV needs confirmation. */
    public static final long H_DECODE_KV_FULL = 8L;

    private PriorityCostFunction() {
    }

    /** Rank 0..4 of a normalized priority (30..70), clamped for safety. */
    public static int rank(int priority) {
        return Math.max(0, Math.min(4, (priority - 30) / 10));
    }

    /** Single-value victim cost: 1024^rank. */
    public static long f(int priority) {
        long cost = 1;
        int rank = rank(priority);
        for (int i = 0; i < rank; i++) {
            cost *= 1024;
        }
        return cost;
    }

    /**
     * Stage multiplier g(stage) (design doc 11.4/12.5): deeper stages are
     * exponentially more expensive to evict. Phase 4 only ever uses
     * {@code RESERVED_NOT_ACCEPTED}; the other two values are defined for the
     * Phase 5 preemption interface.
     */
    public static long g(DecodeTaskPhase stage) {
        return switch (stage) {
            case RESERVED_NOT_ACCEPTED -> 1L;
            case ACCEPTED_NOT_RUNNING -> 4L;
            case RUNNING -> 32L;
        };
    }

    /** KV bucket of a reservation: ceil(kvTokens / 1024) (design doc 12.3). */
    public static long kvBucket(long kvTokens) {
        return (Math.max(0, kvTokens) + 1023) / 1024;
    }

    /**
     * Length waste factor for KV eviction (design doc 12.5):
     * {@code max(1, sqrt(kvBucket))}, rounded to long — large releases sort
     * first but must not dramatically shrink cross-priority costs.
     */
    public static long lengthWasteCost(long kvTokens) {
        return Math.max(1L, Math.round(Math.sqrt((double) kvBucket(kvTokens))));
    }

    /**
     * Cache benefit bounded so it can never reverse the priority boundary:
     * {@code min(cacheHitTokens, benefitCap, MIN_ADJACENT_GAP / 2)}.
     * The operator cap ({@code autoTpmPlanCacheHitBenefitCap}) defaults to 0,
     * which disables cache benefit entirely.
     */
    public static long boundedCacheBenefit(long cacheHitTokens, long benefitCap) {
        long benefit = Math.min(Math.max(0, cacheHitTokens), Math.max(0, benefitCap));
        return Math.min(benefit, MIN_ADJACENT_GAP / 2);
    }
}
