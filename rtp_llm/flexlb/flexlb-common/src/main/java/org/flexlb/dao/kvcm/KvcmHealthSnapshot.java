package org.flexlb.dao.kvcm;

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
