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
    public static final long DEFAULT_LEADER_REFRESH_INTERVAL_MS = 5000L;
    public static final int DEFAULT_BOOTSTRAP_PORT = 6381;

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

    public Endpoint toEndpoint() {
        Endpoint endpoint = new Endpoint();
        endpoint.setAddress(address);
        endpoint.setProtocol(GRPC_PROTOCOL);
        endpoint.setDiscovery(discovery);
        return endpoint;
    }
}
