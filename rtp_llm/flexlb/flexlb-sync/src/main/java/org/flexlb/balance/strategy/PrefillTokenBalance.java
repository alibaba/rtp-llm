package org.flexlb.balance.strategy;

/** Shared production/simulation ranking. Inputs have already passed generation and hard safety filters. */
public final class PrefillTokenBalance {
    private PrefillTokenBalance() { }

    public static boolean cacheFirst(long maxHit, long inputTokens, double minHitPercent) {
        return inputTokens > 0 && maxHit > 0 && 100.0 * maxHit / inputTokens > minHitPercent;
    }

    /** Cold requests ignore cache affinity; hot requests must use a cache leader. */
    public static boolean eligible(long hit, long maxHit, boolean cacheFirst) {
        return !cacheFirst || hit == maxHit;
    }

    public static boolean hasCapacity(long pendingTokens, long limit) {
        return pendingTokens < limit;
    }
}
