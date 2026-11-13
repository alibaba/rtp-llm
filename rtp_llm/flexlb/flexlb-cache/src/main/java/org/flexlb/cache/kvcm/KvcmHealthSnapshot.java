package org.flexlb.cache.kvcm;

/**
 * Freezes KVCM query and heartbeat health counters used for source failover decisions.
 */
public record KvcmHealthSnapshot(
        KvcmHealthState state,
        int consecutiveHeartbeatFailures,
        int consecutiveHeartbeatSuccesses,
        int consecutiveQueryFailures,
        long lastHeartbeatSuccessTimeMs,
        long lastHeartbeatFailureTimeMs,
        String lastStateChangeReason) {

    public boolean isHealthy() {
        return state == KvcmHealthState.HEALTHY;
    }
}
