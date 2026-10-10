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
    public static final int DEFAULT_TOP_K_HOST_COUNT = 3;

    /**
     * Additional cache sources supported by the KVCM host-match query.
     */
    public enum BackendType {
        ST_TAIRMEMPOOL,
        ST_EVENT_REPORT_L2
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
    private int topKHostCount = DEFAULT_TOP_K_HOST_COUNT;
    /**
     * Additional sources to combine with local caches; an empty list is sent unchanged.
     */
    private List<BackendType> backendTypes = List.of();
    private LocalStandbyConfig localStandby = new LocalStandbyConfig();

    public void setMedium(List<String> medium) {
        this.medium = medium == null ? List.of() : List.copyOf(medium);
    }

    public void setBackendTypes(List<BackendType> backendTypes) {
        this.backendTypes = backendTypes == null ? List.of() : List.copyOf(backendTypes);
    }
}
