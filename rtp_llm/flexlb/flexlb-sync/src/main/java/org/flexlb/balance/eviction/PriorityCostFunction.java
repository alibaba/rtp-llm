package org.flexlb.balance.eviction;

import org.flexlb.enums.DecodeTaskPhase;

/**
 * Priority cost primitives for eviction planning (design doc 7.3).
 *
 * <p>The scalar {@code f(priority) = 1024^rank} is retained for metrics and
 * diagnostics. It is not used to enforce absolute priority ordering because
 * an unbounded victim set can exceed any fixed-radix scalar. Plan selection
 * uses {@link PriorityHarmProfile} instead.
 *
 */
public final class PriorityCostFunction {

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

    /** Add non-negative diagnostic costs, saturating instead of wrapping. */
    public static long saturatedAdd(long left, long right) {
        if (left < 0 || right < 0) {
            throw new IllegalArgumentException("cost operands must be non-negative");
        }
        return left > Long.MAX_VALUE - right ? Long.MAX_VALUE : left + right;
    }

    /** Multiply non-negative diagnostic costs, saturating instead of wrapping. */
    public static long saturatedMultiply(long left, long right) {
        if (left < 0 || right < 0) {
            throw new IllegalArgumentException("cost operands must be non-negative");
        }
        if (left == 0 || right == 0) {
            return 0;
        }
        return left > Long.MAX_VALUE / right ? Long.MAX_VALUE : left * right;
    }

    /**
     * Stage multiplier g(stage) (design doc 11.4/12.5): deeper stages are
     * exponentially more expensive to evict.
     */
    public static long g(DecodeTaskPhase stage) {
        return switch (stage) {
            case MASTER_QUEUED_NOT_DISPATCHED -> 1L;
            case ENGINE_MAY_HAVE_SEEN -> 4L;
            case ACCEPTED_NOT_RUNNING -> 16L;
            case RUNNING -> 64L;
        };
    }

    /** KV bucket of a reservation: ceil(kvTokens / 1024) (design doc 12.3). */
    public static long kvBucket(long kvTokens) {
        long nonNegativeTokens = Math.max(0, kvTokens);
        return nonNegativeTokens == 0 ? 0 : 1 + (nonNegativeTokens - 1) / 1024;
    }

    /**
     * Length waste factor for KV eviction (design doc 12.5):
     * {@code max(1, sqrt(kvBucket))}, rounded to long — large releases sort
     * first but must not dramatically shrink cross-priority costs.
     */
    public static long lengthWasteCost(long kvTokens) {
        return Math.max(1L, Math.round(Math.sqrt((double) kvBucket(kvTokens))));
    }

}
