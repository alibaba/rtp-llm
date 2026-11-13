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

    /** A single oversized request may start when idle; it must not starve forever.
     * Planned work is included by selectors and omitted by the serialized admission owner.
     */
    public static boolean withinColdWindow(PrefillTokenWorkload.Snapshot work, long computeTokens, Long limit) {
        return limit == null || work.score() + computeTokens <= Math.max(limit, computeTokens);
    }

    public static boolean hasCapacity(long pendingTokens, long limit) {
        return pendingTokens < limit;
    }
}
