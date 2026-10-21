package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

import java.util.List;

@Getter
@Setter
public final class KvcmCacheMatchingConfig implements CacheMatchingConfig {

    public static final long DEFAULT_REQUEST_TIMEOUT_MS = 500L;
    public static final long DEFAULT_LEADER_REFRESH_INTERVAL_MS = 10_000L;
    public static final int DEFAULT_HEARTBEAT_FAILURE_THRESHOLD = 3;
    public static final int DEFAULT_QUERY_FAILURE_THRESHOLD = 10;
    public static final int DEFAULT_MAX_QUERY_RETRY_COUNT = 1;
    public static final int DEFAULT_RECOVERY_SUCCESS_THRESHOLD = 3;
    public static final int DEFAULT_GLOBAL_KVS_HOST_COUNT = 3;

    /**
     * Explicit query semantics for engines that do not report their cache group mode.
     * When absent, the client obtains the query type from worker status.
     */
    private CacheQueryType queryType;

    public enum CacheQueryType {
        QT_PREFIX_MATCH,
        QT_PREFIX_MATCH_WITH_MAMBA
    }

    /**
     * Total cache-query budget, including retries; also bounds each cluster-info RPC.
     */
    private long requestTimeoutMs = DEFAULT_REQUEST_TIMEOUT_MS;
    /**
     * Fixed when the KVCM client starts; a runtime update requires a restart.
     */
    private long leaderRefreshIntervalMs = DEFAULT_LEADER_REFRESH_INTERVAL_MS;
    private int heartbeatFailureThreshold = DEFAULT_HEARTBEAT_FAILURE_THRESHOLD;
    private int queryFailureThreshold = DEFAULT_QUERY_FAILURE_THRESHOLD;
    private int maxQueryRetryCount = DEFAULT_MAX_QUERY_RETRY_COUNT;
    private int recoverySuccessThreshold = DEFAULT_RECOVERY_SUCCESS_THRESHOLD;
    /**
     * Storage media to match; empty means all media.
     */
    private List<String> medium = List.of();
    /**
     * Logical engines with the longest local match to compute remote hits for.
     * Zero requests local matches only.
     */
    private int globalKvsHostCount = DEFAULT_GLOBAL_KVS_HOST_COUNT;
    /**
     * Enables P2P matches in the combined global prefix hit count.
     */
    private boolean enableP2p;
    private LocalStandbyConfig localStandby = new LocalStandbyConfig();

    public void setMedium(List<String> medium) {
        this.medium = medium == null ? List.of() : List.copyOf(medium);
    }
}
