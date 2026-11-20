package org.flexlb.cache;

import org.flexlb.cache.kvcm.KvcmHealthState;
import org.flexlb.config.cache.CacheMatchMode;

public record CacheMatchStatus(
        boolean kvcmEnabled,
        boolean localStandbyEnabled,
        CacheMatchMode configuredMode,
        boolean autoSwitchEnabled,
        CacheMatchSource effectiveSource,
        KvcmHealthState kvcmHealthState,
        int consecutiveQueryFailures,
        int consecutiveHeartbeatFailures,
        int consecutiveHeartbeatSuccesses,
        long lastHeartbeatSuccessTimeMs,
        long lastHeartbeatFailureTimeMs,
        long lastSwitchTimeMs,
        String lastSwitchReason,
        long localStandbyEntries,
        long localStandbyMaximumEntries) {}
