package org.flexlb.dao.route;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

/**
 * KVCM cache match service configuration.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class KvcmConfig {

    private static final String GRPC_PROTOCOL = "grpc";

    public static final long DEFAULT_REQUEST_TIMEOUT_MS = 500L;
    public static final long DEFAULT_LEADER_REFRESH_INTERVAL_MS = 10_000L;
    public static final int DEFAULT_BOOTSTRAP_PORT = 6381;
    public static final int DEFAULT_HEARTBEAT_FAILURE_THRESHOLD = 3;
    public static final int DEFAULT_QUERY_FAILURE_THRESHOLD = 10;
    public static final int DEFAULT_MAX_QUERY_RETRY_COUNT = 1;
    public static final int DEFAULT_RECOVERY_SUCCESS_THRESHOLD = 3;

    @JsonProperty("enabled")
    private boolean enabled;

    @JsonProperty("address")
    private String address;

    @JsonProperty("namespace")
    private String namespace;

    /**
     * MetaService gRPC port used only to bootstrap leader discovery through GetClusterInfo.
     */
    @JsonProperty("port")
    private int port = DEFAULT_BOOTSTRAP_PORT;

    @JsonProperty("discovery")
    private DiscoveryConfig discovery;

    @JsonProperty("request_timeout_ms")
    private long requestTimeoutMs = DEFAULT_REQUEST_TIMEOUT_MS;

    @JsonProperty("leader_refresh_interval_ms")
    private long leaderRefreshIntervalMs = DEFAULT_LEADER_REFRESH_INTERVAL_MS;

    @JsonProperty("heartbeat_failure_threshold")
    private int heartbeatFailureThreshold = DEFAULT_HEARTBEAT_FAILURE_THRESHOLD;

    @JsonProperty("query_failure_threshold")
    private int queryFailureThreshold = DEFAULT_QUERY_FAILURE_THRESHOLD;

    @JsonProperty("max_query_retry_count")
    private int maxQueryRetryCount = DEFAULT_MAX_QUERY_RETRY_COUNT;

    @JsonProperty("recovery_success_threshold")
    private int recoverySuccessThreshold = DEFAULT_RECOVERY_SUCCESS_THRESHOLD;

    @JsonProperty("local_standby")
    private LocalStandbyConfig localStandby = new LocalStandbyConfig();

    public Endpoint toEndpoint() {
        Endpoint endpoint = new Endpoint();
        endpoint.setAddress(address);
        endpoint.setProtocol(GRPC_PROTOCOL);
        endpoint.setDiscovery(discovery);
        return endpoint;
    }
}
