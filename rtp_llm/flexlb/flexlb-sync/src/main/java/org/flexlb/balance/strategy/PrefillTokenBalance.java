package org.flexlb.balance.strategy;

import org.flexlb.config.RoutingConfig;

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

    /** True when a leader's hotspot release line no longer pins this hit rate. */
    public static boolean hotspotReleases(double releaseLine, long hitTokens, long inputTokens) {
        return releaseLine > 0.0 && inputTokens > 0
                && hitTokens * RoutingConfig.PERCENTAGE_SCALE / inputTokens < releaseLine;
    }
}
