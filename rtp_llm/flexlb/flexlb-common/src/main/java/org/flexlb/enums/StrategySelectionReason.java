package org.flexlb.enums;

/**
 * Selection reasons emitted by shortest-TTFT based strategies.
 */
public enum StrategySelectionReason {
    /**
     * The shortest-TTFT candidate was selected.
     */
    SHORTEST_TTFT,

    /**
     * Cache affinity was rejected because its cache leader exceeds the outstanding uncached-work threshold.
     */
    SHORTEST_TTFT_OUTSTANDING_GUARD,

    /**
     * Every candidate exceeds the outstanding uncached-work threshold, so shortest-TTFT falls back to all workers.
     */
    SHORTEST_TTFT_OUTSTANDING_GUARD_FALLBACK,

    /**
     * Cache affinity was rejected because its cache leader does not meet the minimum cache-hit rate.
     */
    SHORTEST_TTFT_LOW_CACHE_HIT,

    /**
     * A concurrent selection prevented the preferred shortest-TTFT worker from being selected.
     */
    SHORTEST_TTFT_FALLBACK,

    /**
     * CacheAffinityFirst selected the cache leader within its additional-work tolerance.
     */
    CACHE_LEADER,

    /**
     * A concurrent selection prevented the preferred cache leader from being selected, but another eligible cache-affinity worker was selected.
     */
    CACHE_AFFINITY_FALLBACK
}
