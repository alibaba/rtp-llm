package org.flexlb.cache.domain;

/**
 * Reports the local cache source for new lookups separately from a single failed query's fallback.
 */
public record CacheMatchStatus(
        boolean kvcmEnabled,
        boolean localStandbyEnabled,
        Source configuredMode,
        boolean autoSwitchEnabled,
        Source effectiveSource,
        Source queryFailureFallbackSource,
        boolean kvcmLeaderKnown,
        boolean kvcmStandbyRecommended,
        int consecutiveQueryFailures,
        int consecutiveHeartbeatFailures,
        int recoverySuccesses,
        long localStandbyEntries,
        long localStandbyMaximumEntries) {

    /**
     * Cache lookup sources supported by this FlexLB process.
     */
    public enum Source {
        LOCAL_SYNC,
        KVCM,
        LOCAL_STANDBY
    }
}
