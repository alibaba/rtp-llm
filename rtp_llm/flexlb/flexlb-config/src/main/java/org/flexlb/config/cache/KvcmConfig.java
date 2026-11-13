package org.flexlb.config.cache;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import org.flexlb.util.discovery.DiscoveryEndpoint;
import org.flexlb.util.discovery.DiscoverySettings;

/**
 * KVCM cache match service configuration.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class KvcmConfig {

    private static final String GRPC_PROTOCOL = "grpc";

    public static final int DEFAULT_BOOTSTRAP_PORT = 6381;

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
    private DiscoverySettings discovery;

    public DiscoveryEndpoint toEndpoint() {
        DiscoveryEndpoint endpoint = new DiscoveryEndpoint();
        endpoint.setAddress(address);
        endpoint.setProtocol(GRPC_PROTOCOL);
        endpoint.setDiscovery(discovery);
        return endpoint;
    }
}
