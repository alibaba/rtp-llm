package org.flexlb.balance.eviction;

import com.google.common.math.LongMath;
import org.flexlb.enums.DecodeTaskPhase;

/** Case, stage and length costs. Scalar priority cost is diagnostic;
 * {@link PriorityHarmProfile} enforces exact priority ordering. */
public final class PriorityCostFunction {

    private static final int MIN_PRIORITY_RANK = 0;
    private static final int MAX_PRIORITY_RANK = 4;
    private static final int PRIORITY_RANK_BASE = 30;
    private static final int PRIORITY_POINTS_PER_RANK = 10;
    private static final long COST_RADIX = 1_024L;
    private static final long KV_TOKENS_PER_COST_BUCKET = 1_024L;
    private static final long MASTER_QUEUED_STAGE_WEIGHT = 1L;
    private static final long ENGINE_MAY_HAVE_SEEN_STAGE_WEIGHT = 4L;
    private static final long ACCEPTED_STAGE_WEIGHT = 16L;
    private static final long RUNNING_STAGE_WEIGHT = 64L;

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
        return Math.clamp((priority - PRIORITY_RANK_BASE) / PRIORITY_POINTS_PER_RANK,
                MIN_PRIORITY_RANK, MAX_PRIORITY_RANK);
    }

    /** Single-value victim cost: 1024^rank. */
    public static long f(int priority) {
        return LongMath.pow(COST_RADIX, rank(priority));
    }

    /**
     * Stage multiplier g(stage) (design doc 11.4/12.5): deeper stages are
     * exponentially more expensive to evict.
     */
    public static long g(DecodeTaskPhase stage) {
        return switch (stage) {
            case LOCAL_RESERVED, MASTER_QUEUED_NOT_DISPATCHED -> MASTER_QUEUED_STAGE_WEIGHT;
            case ENGINE_MAY_HAVE_SEEN -> ENGINE_MAY_HAVE_SEEN_STAGE_WEIGHT;
            case ACCEPTED_NOT_RUNNING -> ACCEPTED_STAGE_WEIGHT;
            case RUNNING -> RUNNING_STAGE_WEIGHT;
        };
    }

    /** KV bucket of a reservation: ceil(kvTokens / 1024) (design doc 12.3). */
    public static long kvBucket(long kvTokens) {
        return Math.ceilDiv(Math.max(0L, kvTokens), KV_TOKENS_PER_COST_BUCKET);
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
