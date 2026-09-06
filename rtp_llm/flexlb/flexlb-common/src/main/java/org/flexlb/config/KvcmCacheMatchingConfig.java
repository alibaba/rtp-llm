package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public final class KvcmCacheMatchingConfig implements CacheMatchingConfig {

    public static final long DEFAULT_REQUEST_TIMEOUT_MS = 500L;
    public static final long DEFAULT_LEADER_REFRESH_INTERVAL_MS = 10_000L;
    public static final int DEFAULT_HEARTBEAT_FAILURE_THRESHOLD = 3;
    public static final int DEFAULT_QUERY_FAILURE_THRESHOLD = 10;
    public static final int DEFAULT_MAX_QUERY_RETRY_COUNT = 1;
    public static final int DEFAULT_RECOVERY_SUCCESS_THRESHOLD = 3;
    public static final int DEFAULT_P2P_HOST_COUNT = 0;

    private long requestTimeoutMs = DEFAULT_REQUEST_TIMEOUT_MS;
    private long leaderRefreshIntervalMs = DEFAULT_LEADER_REFRESH_INTERVAL_MS;
    private int heartbeatFailureThreshold = DEFAULT_HEARTBEAT_FAILURE_THRESHOLD;
    private int queryFailureThreshold = DEFAULT_QUERY_FAILURE_THRESHOLD;
    private int maxQueryRetryCount = DEFAULT_MAX_QUERY_RETRY_COUNT;
    private int recoverySuccessThreshold = DEFAULT_RECOVERY_SUCCESS_THRESHOLD;
    private int p2pHostCount = DEFAULT_P2P_HOST_COUNT;
    private LocalStandbyConfig localStandby = new LocalStandbyConfig();
}
