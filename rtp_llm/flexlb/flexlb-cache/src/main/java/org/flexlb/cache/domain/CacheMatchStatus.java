package org.flexlb.cache.domain;

import org.flexlb.config.CacheMatchMode;
import org.flexlb.dao.kvcm.KvcmHealthState;

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
        long localStandbyMaximumEntries) {
}
